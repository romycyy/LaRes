"""What the LLM block studies (R-046 ``gait_repair``, R-049 ``two_channel``, R-054 ``self_trace``, R-084
``coordinate_repair``) share: the one-slot generator behind every pen-spin LLM arm, the saved-record stand-in for an
incumbent, and the resume guards. Each study keeps its own ``run_arm``: they record different fields and score
different roles, so only the identical pieces live here.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

from lares.envs.isaac_pen_spin import cases


def make_slot_generator(adir: str, env_id: str, client, model: str, reasoning_effort: str, n_slots: int,
                        max_calls_per_slot: int, *, obs_dim: int = cases.OBS_DIM, prompts: dict | None = None,
                        slot_metadata=None, max_total_calls: int | None = None, extra_source_check=None,
                        action_dim: int = cases.NUM_ACTIONS):
    """A ``PlateauSlotGenerator`` for one arm directory, plus its ``LLMCallLedger``.

    Ledgers calls in ``<adir>/llm_calls.jsonl`` and transcribes them to ``<adir>/llm_transcript.log``. ``prompts`` defaults to the
    task's prompt assets; ``slot_metadata(k) -> dict`` is the lane metadata an exploration in slot ``k`` records
    (R-049). ``max_total_calls`` defaults to ``n_slots * max_calls_per_slot``. ``extra_source_check(code) -> [defects]``
    is an opt-in source rule (R-092's module contract); a defect is a validation failure the slot's correction sees.
    ``action_dim`` is the policy's output width (R-094's contact-rule arm emits 15 requests, not 20 joint targets).
    """
    from lares.core.policy_generation import PlateauSlotGenerator
    from lares.core.training_pipeline import load_policy_prompt_assets
    from lares.search.generation import LLMCallLedger

    os.makedirs(adir, exist_ok=True)
    if prompts is None:
        prompts = load_policy_prompt_assets(env_id)
    if max_total_calls is None:
        max_total_calls = int(n_slots) * int(max_calls_per_slot)
    ledger = LLMCallLedger(adir)
    gen = PlateauSlotGenerator(client, SimpleNamespace(model=model, reasoning_effort=reasoning_effort),
                               os.path.join(adir, "gen_0"), int(obs_dim), int(action_dim), env_id, prompts,
                               int(n_slots), ledger, llm_transcript_path=os.path.join(adir, "llm_transcript.log"),
                               max_total_calls=int(max_total_calls), max_repair_per_candidate=1,
                               slot_metadata=slot_metadata, extra_source_check=extra_source_check)
    return gen, ledger


def attempts_record(attempts: list) -> dict:
    """The per-slot record every study keeps of a generator's attempts: ids, validity, and the final policy."""
    final = attempts[-1]
    return {"attempts": [{"attempt_id": c.attempt_id, "valid": c.valid, "validation": c.validation,
                          "llm_call_ids": c.llm_call_ids} for c in attempts],
            "valid": bool(final.valid), "policy": final.policy if final.valid else None,
            "code": final.code if final.valid else None}


def parent_like(record: dict, manifest, returns: list[float], actor_name: str):
    """A saved per-case record rebuilt as a ``ManifestResult`` (paired by case id), the incumbent a challenger meets."""
    from lares.eval.runner import ManifestResult

    if record["manifest_id"] != manifest.manifest_id:
        raise ValueError("the saved record is for another manifest")
    res = ManifestResult(actor_name=actor_name, manifest_id=manifest.manifest_id, action_mode="deterministic",
                         schema_id=manifest.environment.env_id)
    for case, ok, ret in zip(manifest.episodes, record["per_case"], returns):
        res.episodes.append(SimpleNamespace(case_id=case.case_id, task_hash=case.task_hash, success=bool(ok),
                                            episode_return=float(ret), signed_goal_progress=None))
    return res


def arm_needs_generation(bdir: str, arm: str) -> bool:
    """False once the arm is finished; refuses an arm interrupted after spending episodes (never rerun)."""
    adir = os.path.join(bdir, arm)
    if os.path.exists(os.path.join(adir, "arm.json")):
        return False
    if os.path.exists(os.path.join(adir, "isaac")):
        raise SystemExit(f"{adir} was interrupted after spending episodes; an interrupted arm is not rerun")
    return True


def started(bdir: str, name: str) -> None:
    """Mark an evaluation as started; a second start of the same name means it was interrupted and stops the run."""
    marker = os.path.join(bdir, f"{name}.started")
    if os.path.exists(marker):
        raise SystemExit(f"{marker}: this evaluation was interrupted earlier; its cases are never rerun")
    os.makedirs(bdir, exist_ok=True)
    open(marker, "w").close()


def screen_best(actors: dict, manifest, stream) -> tuple[dict, int]:
    """Screen every candidate actor on ``manifest``; the best by the checkpoint tie rule (success, then tail hit)."""
    from lares.envs.isaac_pen_spin.gait_repair import public, score
    from lares.fitting.checkpoints import RolloutScore, is_better_checkpoint

    screens, best, best_k = {}, None, None
    for k, actor in actors.items():
        s = score(actor, manifest, stream)
        screens[str(k)] = public(s)
        r = RolloutScore(s["success"], s["tail_hit_fraction"])
        if is_better_checkpoint(r, best):
            best, best_k = r, k
    return screens, best_k


__all__ = ["arm_needs_generation", "attempts_record", "make_slot_generator", "parent_like", "screen_best", "started"]
