"""R-049: does requiring separate retention and spin feedback improve cold single-step exploration on Stages 1 and 2?

Driven by ``scripts/run_full_evolution.py --config config/pen_spin_plateau_repair.yaml --study two_channel_cold``.
Approved with advisor call_00098. Eight blocks (four Stage-1 then four Stage-2 pool seeds), each one frozen 80-case
manifest split by index into roles (``ROLES``). Per block (each step skipped if its output exists; an interrupted step
that spent episodes is never rerun):

1. ``null``: the neutral pose (every joint at its default) on all four roles, 80 episodes. It is the scored null
   incumbent: an arm keeps it unless its candidate beats it under the strict paired rule on the promotion cases.
2. ``baseline`` and ``two_channel``: four single-phase exploration slots each (one call plus at most one linked
   correction per slot) from the plain task prompt; the two arms' prompts differ only in ``TWO_CHANNEL_REQUIREMENT``,
   appended to the task description of the treatment. No demonstrations, no fitting, no parameter search. Every valid
   candidate is screened on the 10 screen cases; the best by the checkpoint tie rule is scored on the probe (10,
   recorded only), then promotion (30) against the null incumbent, then fresh (30). At most 110 episodes per arm.
3. ``summary.json``: primary two_channel minus baseline on fresh final incumbents, paired over the eight blocks;
   stage-stratified means, cold-search base rates (candidates with any screen success; blocks promoting) and plateau
   eligibility (blocks where every candidate of an arm scored zero on the screen and nothing promoted).
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin.gait_repair import public, score

LABEL = "two_channel_cold"
ROLES = {"screen": (0, 10), "promotion": (10, 40), "probe": (40, 50), "fresh": (50, 80)}
BLOCK_CASES = 80
ARMS = ("baseline", "two_channel")
PRIMARY = ("two_channel", "baseline")
SLOTS = 4

TWO_CHANNEL_REQUIREMENT = """
Structural requirement for this controller: it must contain two separately named feedback components whose outputs
are combined into the action.
1. A retention component: its output depends on the pen's measured position relative to in_hand_pos (it may also use
   the pen's velocity or orientation), and it acts to keep the pen in the hand.
2. A spin component: its output depends on the measured spin rate about the pen's long axis and on the hand's own
   joint state (hand_dof_pos or hand_dof_vel), and it acts to sustain the roll.
Compute each component in its own clearly named block of code with its own parameters; each must change the action
when its inputs change. No particular control law is required beyond this structure."""


# ---------------------------------------------------------------------------------------------------------------------
#  Manifests
# ---------------------------------------------------------------------------------------------------------------------


def block_stages(ccfg: dict) -> list[tuple[int, int]]:
    """``[(stage, seed)]`` in block order: the Stage-1 seeds, then the Stage-2 seeds."""
    out = [(1, int(s)) for s in ccfg["stage1_seeds"]] + [(2, int(s)) for s in ccfg["stage2_seeds"]]
    if [s for _, s in out] != [int(s) for s in ccfg["block_seeds"]]:
        raise ValueError("block_seeds must list the Stage-1 seeds then the Stage-2 seeds")
    return out


def manifest_file(stage: int, seed: int) -> str:
    return cases.manifest_path(stage, LABEL, seed)


def freeze_manifests(ccfg: dict) -> dict:
    from lares.eval.manifest import EvaluationManifest

    built, hashes = {}, {}
    for b, (stage, seed) in enumerate(block_stages(ccfg)):
        m = cases.build_manifest(stage, seed, BLOCK_CASES, label=LABEL)
        path = manifest_file(stage, seed)
        if os.path.exists(path):
            if EvaluationManifest.load(path).to_dict() != m.to_dict():
                raise ValueError(f"{path} exists and differs from the manifest this study builds")
        else:
            m.save(path)
        for c in m.episodes:
            if c.task_hash in hashes:
                raise ValueError(f"block{b} shares placement {c.task_hash} with {hashes[c.task_hash]}")
            hashes[c.task_hash] = f"block{b}"
        built[f"block{b}"] = path
    return built


def block_roles(stage: int, seed: int) -> dict:
    from lares.eval.manifest import EvaluationManifest

    m = EvaluationManifest.load(manifest_file(stage, seed))
    if len(m) != BLOCK_CASES or cases.stage_of(m) != stage:
        raise ValueError(f"block manifest for stage {stage} seed {seed} is not an {BLOCK_CASES}-case stage-{stage} one")
    return {role: EvaluationManifest(manifest_id=f"{m.manifest_id}:{role}", environment=m.environment, pool=m.pool,
                                     split=m.split, episodes=list(m.episodes[a:b]))
            for role, (a, b) in ROLES.items()}


# ---------------------------------------------------------------------------------------------------------------------
#  Steps
# ---------------------------------------------------------------------------------------------------------------------


def _parent_like(record: dict, manifest, returns: list[float]):
    """A saved per-case record rebuilt as a ManifestResult (paired by case id)."""
    from lares.eval.runner import ManifestResult

    if record["manifest_id"] != manifest.manifest_id:
        raise ValueError("the saved record is for another manifest")
    res = ManifestResult(actor_name="neutral_pose", manifest_id=manifest.manifest_id, action_mode="deterministic",
                         schema_id=manifest.environment.env_id)
    for case, ok, ret in zip(manifest.episodes, record["per_case"], returns):
        res.episodes.append(SimpleNamespace(case_id=case.case_id, task_hash=case.task_hash, success=bool(ok),
                                            episode_return=float(ret), signed_goal_progress=None))
    return res


def run_null(roles: dict, bdir: str, stream_factory, stage: int) -> dict:
    from lares.envs.isaac_pen_spin.pipeline_backend import NeutralActor

    path = os.path.join(bdir, "null.json")
    if os.path.exists(path):
        return json.load(open(path))
    started = os.path.join(bdir, "null.started")
    if os.path.exists(started):
        raise SystemExit(f"{started}: the null incumbent's evaluation was interrupted; it is never rerun")
    os.makedirs(bdir, exist_ok=True)
    open(started, "w").close()
    stream = stream_factory(stage, "null", os.path.join(bdir, "null"))
    actor = NeutralActor()
    out = {}
    for role in ROLES:
        s = score(actor, roles[role], stream)
        if role == "promotion":
            out["promotion_returns"] = [float(e.episode_return) for e in s["_result"].episodes]
        out[role] = public(s)
    out.update(isaac_processes=stream.calls, isaac_episodes=stream.episodes)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    return out


def generate_slots(arm: str, env_id: str, adir: str, client, model: str, reasoning_effort: str,
                   max_calls_per_slot: int = 2) -> dict:
    """Four single-phase exploration slots from one prompt; the treatment's task description carries the requirement."""
    import pickle

    from lares.core.policy_generation import PlateauSlotGenerator
    from lares.core.training_pipeline import load_policy_prompt_assets
    from lares.search.generation import LLMCallLedger
    from lares.utils.hashing import sha256_text

    os.makedirs(adir, exist_ok=True)
    data_pkl = os.path.join(adir, "data.pkl")
    with open(data_pkl, "wb") as f:
        pickle.dump([{"obs": np.zeros(157)}], f)
    prompts = dict(load_policy_prompt_assets(env_id))
    if arm == "two_channel":
        prompts["task_description"] = prompts["task_description"] + "\n" + TWO_CHANNEL_REQUIREMENT
    elif arm != "baseline":
        raise ValueError(f"unknown arm {arm!r}")
    specs = {k: SimpleNamespace(slot=k, lane_metadata=lambda k=k: {"study": LABEL, "arm": arm, "slot": k})
             for k in range(SLOTS)}
    plan = SimpleNamespace(n=SLOTS, slot=lambda k: specs[k])
    ledger = LLMCallLedger(adir)
    gen = PlateauSlotGenerator(client, SimpleNamespace(model=model, reasoning_effort=reasoning_effort),
                               os.path.join(adir, "gen_0"), 157, cases.NUM_ACTIONS, env_id, prompts, plan, data_pkl,
                               ledger, llm_transcript_path=os.path.join(adir, "llm_transcript.log"),
                               max_total_calls=SLOTS * int(max_calls_per_slot), max_repair_per_candidate=1)
    with open(os.path.join(adir, "user_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(gen.user_content)
    slots = []
    for k in range(SLOTS):
        attempts = gen.exploration(k)
        final = attempts[-1]
        slots.append({"slot": k, "attempts": [{"attempt_id": c.attempt_id, "valid": c.valid,
                                               "validation": c.validation, "llm_call_ids": c.llm_call_ids}
                                              for c in attempts],
                      "valid": bool(final.valid), "policy": final.policy if final.valid else None,
                      "code": final.code if final.valid else None})
    return {"slots": slots, "llm_calls": len(ledger), "llm_tokens": ledger.totals()["llm_total_tokens"],
            "prompt_sha256": sha256_text(gen.user_content)}


def run_arm(arm: str, ccfg: dict, stage: int, roles: dict, null: dict, bdir: str, stream_factory, client) -> dict:
    from lares.eval.promotion import compare_to_incumbent
    from lares.eval.runner import PolicyActor
    from lares.fitting.checkpoints import RolloutScore, is_better_checkpoint

    adir = os.path.join(bdir, arm)
    path = os.path.join(adir, "arm.json")
    if os.path.exists(path):
        return json.load(open(path))
    if os.path.exists(os.path.join(adir, "isaac")):
        raise SystemExit(f"{adir} was interrupted after spending episodes; an interrupted arm is not rerun")
    env_id = cases.STAGES[stage].task_id
    g = generate_slots(arm, env_id, adir, client, ccfg["model"], ccfg["reasoning_effort"],
                       int(ccfg.get("max_calls_per_slot", 2)))
    out = {"arm": arm, "stage": stage, "llm_calls": g["llm_calls"], "llm_tokens": g["llm_tokens"],
           "prompt_sha256": g["prompt_sha256"],
           "slots": [{k: v for k, v in s.items() if k != "policy"} for s in g["slots"]]}
    valid = [(s["slot"], s["policy"]) for s in g["slots"] if s["valid"]]
    null_fresh = null["fresh"]
    out["valid_candidates"] = len(valid)
    if not valid:
        out.update(promoted=False, selected_slot=None, isaac_episodes=0, screens={},
                   fresh_incumbent={"success": null_fresh["success"], "per_case": null_fresh["per_case"],
                                    "candidate": "null", "rule": "no valid candidate: the null incumbent"})
        with open(path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        return out
    stream = stream_factory(stage, arm, os.path.join(adir, "isaac"))
    actors = {k: PolicyActor(p, deterministic=True, name=f"{arm}:slot{k}") for k, p in valid}
    screens, best, best_k = {}, None, None
    for k, _ in valid:
        s = score(actors[k], roles["screen"], stream)
        screens[str(k)] = public(s)
        r = RolloutScore(s["success"], s["tail_hit_fraction"])
        if is_better_checkpoint(r, best):
            best, best_k = r, k
    actor = actors[best_k]
    probe = score(actor, roles["probe"], stream)
    promo = score(actor, roles["promotion"], stream)
    incumbent = _parent_like(null["promotion"], roles["promotion"], null["promotion_returns"])
    decision = compare_to_incumbent(promo["_result"], incumbent, f"{arm}:slot{best_k}", "neutral_pose",
                                    excluded_manifest_ids=[roles["screen"].manifest_id, roles["probe"].manifest_id])
    fresh = score(actor, roles["fresh"], stream)
    promoted = bool(decision.promoted)
    out.update(screens=screens, selected_slot=best_k, probe=public(probe), promotion=public(promo),
               promotion_decision=decision.to_dict(), promoted=promoted, fresh_selected=public(fresh),
               fresh_incumbent={"success": fresh["success"] if promoted else null_fresh["success"],
                                "per_case": fresh["per_case"] if promoted else null_fresh["per_case"],
                                "candidate": f"slot{best_k}" if promoted else "null",
                                "rule": "the arm's candidate if it promoted against the null incumbent, else that "
                                        "incumbent (the neutral pose)"},
               isaac_processes=stream.calls, isaac_episodes=stream.episodes,
               selected_code=dict(valid)[best_k]._generated_source)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    return out


# ---------------------------------------------------------------------------------------------------------------------
#  Study
# ---------------------------------------------------------------------------------------------------------------------


def _arm_stats(a: dict) -> dict:
    screens = a.get("screens", {})
    return {"valid_candidates": a.get("valid_candidates", 0),
            "candidates_with_screen_success": sum(1 for s in screens.values() if s["success"] > 0),
            "plateau": bool(not a.get("promoted") and all(s["success"] == 0 for s in screens.values())),
            "promoted": bool(a.get("promoted"))}


def summarise(blocks: list[dict]) -> dict:
    from lares.eval.stats import t_summary

    out = {"blocks": blocks, "n_blocks": len(blocks)}
    for a, b in (PRIMARY,):
        for key in ("fresh_incumbent", "fresh_selected"):
            d = [blk["arms"][a][key]["success"] - blk["arms"][b][key]["success"] for blk in blocks
                 if blk["arms"][a].get(key) is not None and blk["arms"][b].get(key) is not None]
            out[f"{a}_minus_{b}_{key}"] = {"per_block": d, **(t_summary(d) if len(d) >= 2 else {"n": len(d)})}
    out["paired_success_difference"] = out[f"{PRIMARY[0]}_minus_{PRIMARY[1]}_fresh_incumbent"].get("mean")
    by_stage = {}
    for stage in sorted({blk["stage"] for blk in blocks}):
        sb = [blk for blk in blocks if blk["stage"] == stage]
        row = {"blocks": len(sb), "null_fresh": float(np.mean([blk["null"]["fresh"]["success"] for blk in sb]))}
        for arm in ARMS:
            st = [_arm_stats(blk["arms"][arm]) for blk in sb]
            row[arm] = {"fresh_incumbent": float(np.mean([blk["arms"][arm]["fresh_incumbent"]["success"] for blk in sb])),
                        "promoted_blocks": sum(s["promoted"] for s in st),
                        "plateau_blocks": sum(s["plateau"] for s in st),
                        "candidates_with_screen_success": sum(s["candidates_with_screen_success"] for s in st),
                        "valid_candidates": sum(s["valid_candidates"] for s in st)}
        d = [blk["arms"]["two_channel"]["fresh_incumbent"]["success"] - blk["arms"]["baseline"]["fresh_incumbent"]["success"]
             for blk in sb]
        row["two_channel_minus_baseline"] = {"per_block": d, "mean": float(np.mean(d))}
        by_stage[f"stage{stage}"] = row
    out["by_stage"] = by_stage
    return out


def run_study(ccfg: dict, stream_factory, client) -> dict:
    sdir = ccfg["study_dir"]
    os.makedirs(sdir, exist_ok=True)
    freeze_manifests(ccfg)
    blocks = []
    for b, (stage, seed) in enumerate(block_stages(ccfg)):
        roles = block_roles(stage, seed)
        bdir = os.path.join(sdir, f"block{b}")
        null = run_null(roles, bdir, stream_factory, stage)
        row = {"block": b, "stage": stage, "seed": seed,
               "null": {k: null[k] for k in ROLES}}
        row["arms"] = {arm: run_arm(arm, ccfg, stage, roles, null, bdir, stream_factory, client) for arm in ARMS}
        blocks.append(row)
        json.dump(summarise(blocks), open(os.path.join(sdir, "summary.partial.json"), "w"), indent=2, default=str)
    summary = summarise(blocks)
    with open(os.path.join(sdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


__all__ = ["ARMS", "LABEL", "ROLES", "TWO_CHANNEL_REQUIREMENT", "block_roles", "block_stages", "freeze_manifests",
           "generate_slots", "run_study", "summarise"]
