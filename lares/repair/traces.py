"""Expert-only demonstration traces for plateau-triggered repair (R-037; the source D-027, D-028, D-030 and D-032 used).

The repair slot shows the LLM two successful scripted-expert episodes, subsampled every fourth step, and asks it to
rewrite a failing controller whole. This module is the part of that which needs no LLM:

* :func:`trace_rule` says how a task's traces are cut. ``assembly-v2`` keeps the established post-lift extraction (rows
  from the first step the nut is 3 cm above its start), with the wording D-032 used. Any other task with a scripted
  expert in ``EXPERT_POLICY_MAP`` and ``tcp``/``gripper``/``obj``/``goal`` in its observation schema gets whole-episode
  rows: no lift event is assumed for a task that may never lift anything. Anything else raises
  :class:`UnsupportedTraceTask` rather than producing an empty block.
* :func:`collect_expert_traces` runs the expert alone on a demonstration manifest through the shared runner, so its
  episodes are counted like every other rollout.
* :func:`qualify` applies the gate (at least ``min_successes`` of the episodes succeed) and :func:`select_traces` takes
  the first ``n`` successful episodes in manifest order, so the choice is deterministic.
* :func:`repair_request` writes the request text; for ``assembly-v2`` it is byte-identical to R-036's.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from lares.utils.hashing import sha256_text

TRACE_EVERY = 4
TRACE_EPISODES = 2
REQUIRED_FIELDS = ("tcp", "gripper", "obj", "goal")
EXTRACTION_POST_LIFT = "post_lift"
EXTRACTION_WHOLE_EPISODE = "whole_episode"

REQUEST_TEMPLATE = """Below is a controller for this task. It does not reliably {goal_phrase}. A scripted
expert solves this task; it succeeded on {successes} of {episodes} demonstration episodes.

Rewrite the controller so that it succeeds on its own; no expert is available. You may change any part of it. Keep
the class name GeneratedPolicy, read observations only through self.obs_field, and declare every parameter with
get_param_ranges.

```python
{parent}
```"""


class UnsupportedTraceTask(ValueError):
    """The task has no scripted expert or no schema fields to read a trace through."""


@dataclass(frozen=True)
class TraceRule:
    extraction: str
    object_word: str
    goal_phrase: str
    header: str
    lift_m: float = 0.0

    def to_dict(self) -> dict:
        return {"extraction": self.extraction, "object_word": self.object_word, "goal_phrase": self.goal_phrase,
                "lift_m": self.lift_m, "every": TRACE_EVERY, "episodes_shown": TRACE_EPISODES}


_RULES = {
    "assembly-v2": TraceRule(EXTRACTION_POST_LIFT, "nut", "seat the nut on the peg",
                             "Successful expert episodes after the expert lifted the nut", lift_m=0.03),
}


def trace_rule(task: str, schema=None) -> TraceRule:
    """How ``task``'s traces are cut; raises :class:`UnsupportedTraceTask` when they cannot be."""
    from lares.core.obs_schema import get_obs_schema
    from lares.core.training_pipeline import EXPERT_POLICY_MAP

    if task not in EXPERT_POLICY_MAP:
        raise UnsupportedTraceTask(f"{task}: no scripted expert in EXPERT_POLICY_MAP, so no demonstrations")
    try:
        schema = schema if schema is not None else get_obs_schema(task)
    except Exception as exc:
        raise UnsupportedTraceTask(f"{task}: no observation schema ({exc})") from exc
    missing = [n for n in REQUIRED_FIELDS if n not in {f.name for f in schema.fields}]
    if missing:
        raise UnsupportedTraceTask(f"{task}: the observation schema lacks {missing}; a trace cannot be read")
    return _RULES.get(task) or TraceRule(EXTRACTION_WHOLE_EPISODE, "obj", "complete the task",
                                         "Successful expert episodes")


def _field(obs, schema, name):
    f = schema.field(name)
    return obs[f.start:f.stop]


def collect_expert_traces(task, manifest, env, schema, actor=None, max_steps=150) -> dict:
    """The expert alone on every case of ``manifest``; rows cut by :func:`trace_rule`."""
    from lares.eval import ExpertActor
    from lares.eval.runner import run_episode

    rule = trace_rule(task, schema)
    actor = actor if actor is not None else ExpertActor(task)
    episodes = []
    for case in manifest.episodes:
        rows, state = [], {"z0": None, "on": rule.extraction == EXTRACTION_WHOLE_EPISODE}

        def observe(step, _rows=rows, _s=state):
            if step.get("action") is None:
                return
            o = np.asarray(step["obs"], dtype=np.float64)
            obj = _field(o, schema, "obj")
            if _s["z0"] is None:
                _s["z0"] = float(obj[2])
            if rule.extraction == EXTRACTION_POST_LIFT:
                _s["on"] = _s["on"] or float(obj[2]) - _s["z0"] >= rule.lift_m
            if _s["on"]:
                _rows.append({"t": int(step["timestep"]), "tcp": _field(o, schema, "tcp").round(4).tolist(),
                              "gripper": round(float(_field(o, schema, "gripper")[0]), 3), "obj": obj.round(4).tolist(),
                              "goal": _field(o, schema, "goal").round(4).tolist(),
                              "action": np.asarray(step["action"]).round(3).tolist()})

        ep = run_episode(actor, env, case, max_steps=max_steps, schema=schema, observer=observe)
        episodes.append({"case": case.case_id, "success": float(ep.success), "rows": rows})
    return {"task": task, "manifest_id": manifest.manifest_id, "rule": rule.to_dict(), "episodes": episodes,
            "successes": int(sum(e["success"] for e in episodes))}


def qualify(traces: dict, min_successes: int) -> dict:
    n = len(traces["episodes"])
    return {"episodes": n, "successes": traces["successes"], "min_successes": int(min_successes),
            "qualified": traces["successes"] >= int(min_successes)}


def select_traces(traces: dict, n: int = TRACE_EPISODES) -> list:
    """The first ``n`` successful episodes with rows, in manifest order."""
    return [e for e in traces["episodes"] if e["success"] and e["rows"]][:n]


def traces_sha256(selected: list) -> str:
    return sha256_text(json.dumps(selected, sort_keys=True))


def trace_block(task: str, selected: list, every: int = TRACE_EVERY) -> str:
    rule = trace_rule(task)
    lines = ["", f"{rule.header} (every {every}th step; positions in metres; the action is",
             "the executed action: x, y, z end-effector displacement and the gripper command):"]
    for k, e in enumerate(selected):
        lines.append(f"Episode {k + 1}:")
        lines += [f"  t={r['t']} tcp={r['tcp']} gripper={r['gripper']} {rule.object_word}={r['obj']} goal={r['goal']} "
                  f"action={r['action']}" for r in e["rows"][::every]]
    return "\n".join(lines)


def repair_request(task: str, parent_source: str, successes: int, episodes: int, selected: list) -> str:
    rule = trace_rule(task)
    head = (REQUEST_TEMPLATE.replace("{goal_phrase}", rule.goal_phrase).replace("{successes}", str(successes))
            .replace("{episodes}", str(episodes)).replace("{parent}", parent_source.strip()))
    return head + "\n" + trace_block(task, selected)


__all__ = [
    "EXTRACTION_POST_LIFT", "EXTRACTION_WHOLE_EPISODE", "TRACE_EPISODES", "TRACE_EVERY", "TraceRule",
    "UnsupportedTraceTask", "collect_expert_traces", "qualify", "repair_request", "select_traces", "trace_block",
    "trace_rule", "traces_sha256",
]
