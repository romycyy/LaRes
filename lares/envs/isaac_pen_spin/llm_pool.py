"""Concurrent LLM generation for multi-block pen-spin studies (R-049 ``two_channel``, R-054 ``self_trace``).

Until 2026-11-03 these studies ran every (block, arm) cell's generation in sequence. In R-054, generation accounted
for 4.73 h of the 5.32 h run, and Isaac evaluation for 0.52 h. The cells are independent: each has its own
seeds, manifests, prompt, ``LLMCallLedger`` and log directory, and no cell's prompt depends on another cell's
result. Here a cell's generation (the LLM calls, the validation subprocess and in-process construction) runs on a
worker thread. Isaac evaluation stays on the calling thread, one process at a time, in the old block and arm order.
Every evaluation is still a fresh process on one manifest (D-034, D-035), so the scores do not depend on the order.

Generation is front-loaded, so an interruption could otherwise lose calls that were already paid for.
:func:`save_generation` writes each cell's slots (source, validation, initial state) to ``generated.json`` before
any of its episodes run. On a rerun, :func:`load_generation` rebuilds those policies rather than calling the LLM
again. This is allowed because nothing has been evaluated yet. An arm that has started Isaac (its ``isaac``
directory exists) is still never rerun.

``llm_workers`` in the study block sets the pool size (default :data:`DEFAULT_LLM_WORKERS`). ``1`` runs each job
on the calling thread when its result is asked for, which is the old serial behaviour.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import Future, ThreadPoolExecutor

DEFAULT_LLM_WORKERS = 16
GENERATED = "generated.json"


class GenerationPool:
    """Runs keyed generation jobs on worker threads; :meth:`result` blocks for one job's output."""

    def __init__(self, workers: int = DEFAULT_LLM_WORKERS):
        self.workers = max(1, int(workers))
        self._executor = (ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="llm-gen")
                          if self.workers > 1 else None)
        self._jobs: dict = {}

    def submit(self, key, fn, *args, **kwargs) -> None:
        if key in self._jobs:
            raise ValueError(f"generation job {key!r} was already submitted")
        if self._executor is None:
            self._jobs[key] = (fn, args, kwargs)
        else:
            self._jobs[key] = self._executor.submit(fn, *args, **kwargs)

    def result(self, key):
        job = self._jobs.pop(key)
        if isinstance(job, Future):
            return job.result()
        fn, args, kwargs = job
        return fn(*args, **kwargs)

    def pending(self) -> list:
        return list(self._jobs)

    def close(self, cancel: bool = False) -> None:
        if self._executor is not None:
            # Running calls cannot be interrupted; with cancel=True the queued ones never start.
            self._executor.shutdown(wait=True, cancel_futures=cancel)

    def __enter__(self) -> "GenerationPool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(cancel=exc_type is not None)


def pool_for(block: dict) -> GenerationPool:
    return GenerationPool(int(block.get("llm_workers", DEFAULT_LLM_WORKERS)))


def save_generation(adir: str, g: dict) -> None:
    """Write a cell's generated slots before any of its episodes run (states as ``generated_slot<k>.pt``)."""
    import torch

    slots = []
    for s in g["slots"]:
        row = {k: v for k, v in s.items() if k != "policy"}
        if s["valid"]:
            state = f"generated_slot{s['slot']}.pt"
            torch.save(s["policy"].state_dict(), os.path.join(adir, state))
            row.update(state=state, fixed_action_scale=float(s["policy"].fixed_action_scale))
        slots.append(row)
    out = {k: v for k, v in g.items() if k != "slots"}
    out["slots"] = slots
    tmp = os.path.join(adir, GENERATED + ".tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2, default=str)
    os.replace(tmp, os.path.join(adir, GENERATED))


def load_generation(adir: str, env_id: str) -> dict | None:
    """The saved generation of a cell, policies rebuilt from source and state; ``None`` if there is none."""
    path = os.path.join(adir, GENERATED)
    if not os.path.exists(path):
        return None
    import torch

    from lares.envs.isaac_pen_spin.dense_fit import build_policy

    g = json.load(open(path))
    for s in g["slots"]:
        s["policy"] = None
        if s["valid"]:
            state = torch.load(os.path.join(adir, s["state"]), weights_only=True)
            s["policy"] = build_policy(s["code"], state, env_id, s["fixed_action_scale"])
        s.pop("state", None)
        s.pop("fixed_action_scale", None)
    return g


def generate_once(adir: str, env_id: str, generate) -> dict:
    """A cell's generation: the saved one if it exists, else ``generate()``, saved before it is returned."""
    g = load_generation(adir, env_id)
    if g is None:
        g = generate()
        save_generation(adir, g)
    return g
