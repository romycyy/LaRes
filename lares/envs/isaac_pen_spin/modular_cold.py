"""R-092 (M1): does an LLM-chosen, contract-validated modular decomposition improve cold Stage-3 generation after a
matched rollout search?

Driven by ``scripts/run_full_evolution.py --config config/pen_spin_plateau_repair.yaml --study modular_cold``.
Approved 2026-11-20 with advisor call_00185. Eight blocks (pool seeds 4710-4717), each one frozen 80-case Stage-3
manifest split by index into roles (``ROLES``). Unlike R-049 (``two_channel``, D-044) this is Stage 3, both arms are
searched, the decomposition is the LLM's own (no role is hinted) and it is enforced by an AST contract
(``lares.core.module_contract``). Per block (each step skipped if its output exists; an interrupted step that spent
episodes is never rerun):

1. ``null``: the neutral pose on all four roles, 80 episodes. It is the scored null incumbent.
2. ``monolithic`` and ``modular``: four single-phase cold slots each (one call plus at most one linked correction per
   slot) from the plain task prompt. The arms' prompts differ only in ``MODULAR_REQUIREMENT``, appended to the task
   description of the modular arm, whose slots must also pass ``check_module_contract``: a contract defect is a
   validation failure that gets the slot's one correction, and a slot still invalid runs no episode. No parent code,
   no demonstrations, no fitting.
   * screen: every valid slot on the 10 screen cases; the best by the checkpoint tie rule goes on.
   * search: ``rollout_search.independent_candidates`` of that slot (the unchanged candidate plus 23 draws, seeded by
     the block seed), each candidate one fresh process on the 10 probe cases; the winner by success, then tail-hit
     fraction, then the earlier candidate. Its state is written (``search.json``, ``selected_state.pt``) before
     anything below runs.
   * promotion (30) against the null incumbent under the strict paired rule, then fresh (30). ``fresh_incumbent`` is
     the winner's fresh result if it promoted, else the null's cached fresh scores. Nothing is selected on fresh.
   At most 40 + 240 + 30 + 30 = 340 episodes per arm, so 80 + 680 per block and 6080 over eight blocks.
3. ``summary.json``: primary modular minus monolithic on fresh final incumbents (native sustained success), paired
   over the blocks. Pre-declared secondaries, per arm and as block contrasts: the searched winner's fresh tail-hit
   fraction and success, candidates with any screen success, slots valid within one correction, search candidates
   with any probe success, the winner's fresh drops and failure labels, promoted and plateau blocks, LLM calls and
   tokens, Isaac processes and episodes. Tail-hit fraction never stands in for success.

Every (block, arm) generation is queued at the start and runs concurrently (``llm_pool``, ``llm_workers``); its slots
are saved to ``generated.json`` before any of its episodes run. Isaac steps run one process at a time, in the order
above.
"""

from __future__ import annotations

import json
import os

import numpy as np

from lares.core.module_contract import MODULAR_REQUIREMENT, check_module_contract
from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin.block_study import (arm_needs_generation, attempts_record, make_slot_generator,
                                                   parent_like, screen_best, started)
from lares.envs.isaac_pen_spin.gait_repair import public, score

LABEL = "modular_cold"
STAGE = 3
ROLES = {"screen": (0, 10), "promotion": (10, 40), "probe": (40, 50), "fresh": (50, 80)}
BLOCK_CASES = 80
ARMS = ("monolithic", "modular")
PRIMARY = ("modular", "monolithic")
SLOTS = 4
SEARCH_CANDIDATES = 24
#: 4 screens of 10, 24 probes of 10, promotion 30, fresh 30.
MAX_ARM_EPISODES = SLOTS * 10 + SEARCH_CANDIDATES * 10 + 30 + 30
SEARCH_RULE = "higher probe success, then higher tail-hit fraction, then the earlier candidate (the original first)"


# ---------------------------------------------------------------------------------------------------------------------
#  Manifests
# ---------------------------------------------------------------------------------------------------------------------


def manifest_file(seed: int) -> str:
    return cases.manifest_path(STAGE, LABEL, seed)


def freeze_manifests(ccfg: dict) -> dict:
    return cases.freeze_manifest_set(
        (manifest_file(int(seed)), cases.build_manifest(STAGE, int(seed), BLOCK_CASES, label=LABEL), f"block{b}")
        for b, seed in enumerate(ccfg["block_seeds"]))


def block_roles(seed: int) -> dict:
    from lares.eval.manifest import EvaluationManifest

    return cases.role_manifests(EvaluationManifest.load(manifest_file(seed)), ROLES, expected_cases=BLOCK_CASES,
                                stage=STAGE)


# ---------------------------------------------------------------------------------------------------------------------
#  Steps
# ---------------------------------------------------------------------------------------------------------------------


def run_null(roles: dict, bdir: str, stream_factory) -> dict:
    from lares.envs.isaac_pen_spin.pipeline_backend import NeutralActor

    path = os.path.join(bdir, "null.json")
    if os.path.exists(path):
        return json.load(open(path))
    started(bdir, "null")
    stream = stream_factory(STAGE, "null", os.path.join(bdir, "null"))
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


def arm_prompts(arm: str, env_id: str) -> dict:
    """The arm's prompt assets: the task's own, with ``MODULAR_REQUIREMENT`` appended for the modular arm only."""
    from lares.core.training_pipeline import load_policy_prompt_assets

    prompts = dict(load_policy_prompt_assets(env_id))
    if arm == "modular":
        prompts["task_description"] = prompts["task_description"] + "\n" + MODULAR_REQUIREMENT
    elif arm != "monolithic":
        raise ValueError(f"unknown arm {arm!r}")
    return prompts


def generate_slots(arm: str, env_id: str, adir: str, client, model: str, reasoning_effort: str,
                   max_calls_per_slot: int = 2) -> dict:
    """Four single-phase cold slots from one prompt; the modular arm's also pass the module contract."""
    from lares.utils.hashing import sha256_text

    gen, ledger = make_slot_generator(
        adir, env_id, client, model, reasoning_effort, SLOTS, int(max_calls_per_slot), prompts=arm_prompts(arm, env_id),
        slot_metadata=lambda k: {"study": LABEL, "arm": arm, "slot": k},
        extra_source_check=check_module_contract if arm == "modular" else None)
    with open(os.path.join(adir, "user_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(gen.user_content)
    slots = [{"slot": k, **attempts_record(gen.exploration(k))} for k in range(SLOTS)]
    return {"slots": slots, "llm_calls": len(ledger), "llm_tokens": ledger.totals()["llm_total_tokens"],
            "prompt_sha256": sha256_text(gen.user_content)}


def generate_arm(arm: str, ccfg: dict, bdir: str, client) -> dict:
    """The arm's four slots; no episode. Safe on a worker thread (``llm_pool``)."""
    from lares.envs.isaac_pen_spin import llm_pool

    adir = os.path.join(bdir, arm)
    env_id = cases.STAGES[STAGE].task_id
    os.makedirs(adir, exist_ok=True)
    return llm_pool.generate_once(adir, env_id, lambda: generate_slots(
        arm, env_id, adir, client, ccfg["model"], ccfg["reasoning_effort"], int(ccfg.get("max_calls_per_slot", 2))))


def state_record(policy) -> dict:
    """``{"sha256", "parameters"}`` of a policy's state: the snapshot a search trace names a candidate by."""
    from lares.utils.hashing import sha256_text

    params = {k: v.detach().cpu().tolist() for k, v in policy.state_dict().items()}
    return {"sha256": sha256_text(json.dumps(params, sort_keys=True)), "parameters": params}


def probe_search(policy, seed: int, label: str, probe, stream) -> dict:
    """The 24-candidate search of ``policy``: one fresh process per candidate on ``probe``, winner by ``SEARCH_RULE``.

    Returns ``{"policy", "selected_index", "trace", ...}``; each trace row has the candidate's kind, probe success,
    tail-hit fraction, per-case outcomes, failure labels and the hash of its parameter snapshot.
    """
    from lares.eval.runner import PolicyActor
    from lares.fitting.checkpoints import RolloutScore, is_better_checkpoint
    from lares.search import rollout_search as rs

    candidates = rs.independent_candidates(policy, int(seed))
    if len(candidates) != SEARCH_CANDIDATES or candidates[0][0] != "original":
        raise ValueError("the rollout search is the unchanged candidate plus 23 draws")
    best, best_i, trace = None, None, []
    for i, (kind, candidate) in enumerate(candidates):
        s = score(PolicyActor(candidate, deterministic=True, name=f"{label}:search{i}_{kind}"), probe, stream)
        trace.append({"index": i, "kind": kind, "success": s["success"], "tail_hit_fraction": s["tail_hit_fraction"],
                      "per_case": s["per_case"], "failure_labels": s["failure_labels"],
                      "state_sha256": state_record(candidate)["sha256"]})
        r = RolloutScore(s["success"], s["tail_hit_fraction"])
        if is_better_checkpoint(r, best):
            best, best_i = r, i
    return {"policy": candidates[best_i][1], "selected_index": best_i, "selected_kind": candidates[best_i][0],
            "sampler": rs.SAMPLER, "candidates": len(candidates), "seed": int(seed), "rule": SEARCH_RULE,
            "probe_manifest_id": probe.manifest_id, "trace": trace,
            "candidates_with_probe_success": sum(1 for t in trace if t["success"] > 0)}


def run_arm(arm: str, ccfg: dict, seed: int, roles: dict, null: dict, bdir: str, stream_factory, client,
            generated: dict | None = None) -> dict:
    import torch

    from lares.eval.promotion import compare_to_incumbent
    from lares.eval.runner import PolicyActor

    adir = os.path.join(bdir, arm)
    path = os.path.join(adir, "arm.json")
    if not arm_needs_generation(bdir, arm):
        return json.load(open(path))
    g = generated or generate_arm(arm, ccfg, bdir, client)
    os.makedirs(adir, exist_ok=True)
    out = {"arm": arm, "stage": STAGE, "seed": int(seed), "llm_calls": g["llm_calls"], "llm_tokens": g["llm_tokens"],
           "prompt_sha256": g["prompt_sha256"],
           "slots": [{k: v for k, v in s.items() if k != "policy"} for s in g["slots"]]}
    valid = [(s["slot"], s["policy"]) for s in g["slots"] if s["valid"]]
    null_fresh = null["fresh"]
    out["valid_candidates"] = len(valid)
    if not valid:
        out.update(promoted=False, selected_slot=None, isaac_processes=0, isaac_episodes=0, screens={},
                   fresh_incumbent={"success": null_fresh["success"], "per_case": null_fresh["per_case"],
                                    "candidate": "null", "rule": "no valid candidate: the null incumbent"})
        with open(path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        return out
    stream = stream_factory(STAGE, arm, os.path.join(adir, "isaac"))
    actors = {k: PolicyActor(p, deterministic=True, name=f"{arm}:slot{k}") for k, p in valid}
    screens, best_k = screen_best(actors, roles["screen"], stream)
    found = probe_search(dict(valid)[best_k], seed, f"{arm}:slot{best_k}", roles["probe"], stream)
    selected = found.pop("policy")
    # The selected state is frozen on disk before any promotion or fresh episode runs.
    state = state_record(selected)
    torch.save(selected.state_dict(), os.path.join(adir, "selected_state.pt"))
    search = {**found, "slot": best_k, "selected_state": {"file": "selected_state.pt", **state}}
    with open(os.path.join(adir, "search.json"), "w") as f:
        json.dump(search, f, indent=2, default=str)
    name = f"{arm}:slot{best_k}:search{found['selected_index']}"
    actor = PolicyActor(selected, deterministic=True, name=name)
    promo = score(actor, roles["promotion"], stream)
    incumbent = parent_like(null["promotion"], roles["promotion"], null["promotion_returns"], "neutral_pose")
    decision = compare_to_incumbent(promo["_result"], incumbent, name, "neutral_pose",
                                    excluded_manifest_ids=[roles["screen"].manifest_id, roles["probe"].manifest_id])
    fresh = score(actor, roles["fresh"], stream)
    promoted = bool(decision.promoted)
    want = 10 * len(valid) + 10 * SEARCH_CANDIDATES + 60
    if stream.episodes != want or stream.episodes > MAX_ARM_EPISODES:
        raise RuntimeError(f"{arm}: {stream.episodes} episodes ran, the protocol's count is {want} "
                           f"(at most {MAX_ARM_EPISODES})")
    out.update(screens=screens, selected_slot=best_k, search=search, promotion=public(promo),
               promotion_decision=decision.to_dict(), promoted=promoted, fresh_selected=public(fresh),
               fresh_incumbent={"success": fresh["success"] if promoted else null_fresh["success"],
                                "per_case": fresh["per_case"] if promoted else null_fresh["per_case"],
                                "candidate": name if promoted else "null",
                                "rule": "the arm's searched winner if it promoted against the null incumbent, else "
                                        "that incumbent (the neutral pose)"},
               isaac_processes=stream.calls, isaac_episodes=stream.episodes,
               selected_code=selected._generated_source)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    return out


# ---------------------------------------------------------------------------------------------------------------------
#  Study
# ---------------------------------------------------------------------------------------------------------------------


def _arm_stats(a: dict) -> dict:
    """One arm of one block reduced to the pre-declared secondaries (``None`` where the arm ran no candidate)."""
    screens, fresh, search = a.get("screens", {}), a.get("fresh_selected"), a.get("search")
    labels = fresh["failure_labels"] if fresh else []
    attempts = [s["attempts"] for s in a["slots"]]
    return {"slots": len(a["slots"]), "valid_slots": a.get("valid_candidates", 0),
            "first_attempt_valid_slots": sum(1 for at in attempts if at and at[0]["valid"]),
            "module_contract_defect_slots": sum(1 for at in attempts
                                                if any((x["validation"] or {}).get("source_check") for x in at)),
            "candidates_with_screen_success": sum(1 for s in screens.values() if s["success"] > 0),
            "search_candidates_with_probe_success": search["candidates_with_probe_success"] if search else 0,
            "search_selected_index": search["selected_index"] if search else None,
            "plateau": bool(not a.get("promoted") and all(s["success"] == 0 for s in screens.values())
                            and not (search and search["candidates_with_probe_success"])),
            "promoted": bool(a.get("promoted")),
            "fresh_selected_success": fresh["success"] if fresh else None,
            "fresh_selected_tail_hit_fraction": fresh["tail_hit_fraction"] if fresh else None,
            "fresh_selected_drops": labels.count("dropped") if fresh else None,
            "fresh_selected_failure_labels": {k: labels.count(k) for k in sorted({x for x in labels if x})},
            "fresh_incumbent_success": a["fresh_incumbent"]["success"],
            "llm_calls": a["llm_calls"], "llm_tokens": a["llm_tokens"],
            "isaac_processes": a.get("isaac_processes", 0), "isaac_episodes": a.get("isaac_episodes", 0)}


#: Per-block quantities contrasted modular minus monolithic (a block is skipped where either arm has no value).
CONTRASTS = ("fresh_incumbent_success", "fresh_selected_success", "fresh_selected_tail_hit_fraction",
             "fresh_selected_drops", "candidates_with_screen_success", "valid_slots",
             "search_candidates_with_probe_success")


def summarise(blocks: list[dict]) -> dict:
    from lares.eval.stats import block_contrast

    stats = [{arm: _arm_stats(blk["arms"][arm]) for arm in ARMS} for blk in blocks]
    a, b = PRIMARY
    out = {"study": LABEL, "blocks": blocks, "n_blocks": len(blocks), "primary": f"{a}_minus_{b}_fresh_incumbent",
           "success_rule": "native sustained success; tail-hit fraction is a secondary and never replaces it"}
    for key in CONTRASTS:
        out[f"{a}_minus_{b}_{key}"] = block_contrast(
            s[a][key] - s[b][key] for s in stats if s[a][key] is not None and s[b][key] is not None)
    out[f"{a}_minus_{b}_fresh_incumbent"] = out.pop(f"{a}_minus_{b}_fresh_incumbent_success")
    out[f"{a}_minus_{b}_fresh_selected"] = out.pop(f"{a}_minus_{b}_fresh_selected_success")
    out["paired_success_difference"] = out[f"{a}_minus_{b}_fresh_incumbent"].get("mean")
    out["null_fresh"] = float(np.mean([blk["null"]["fresh"]["success"] for blk in blocks])) if blocks else None
    arms = {}
    for arm in ARMS:
        st = [s[arm] for s in stats]

        def mean(key):
            vals = [s[key] for s in st if s[key] is not None]
            return float(np.mean(vals)) if vals else None

        total = lambda key: int(sum(s[key] or 0 for s in st))  # noqa: E731
        labels: dict = {}
        for s in st:
            for k, n in s["fresh_selected_failure_labels"].items():
                labels[k] = labels.get(k, 0) + n
        slots = total("slots")
        arms[arm] = {
            "fresh_incumbent": mean("fresh_incumbent_success"),
            "fresh_selected": mean("fresh_selected_success"),
            "fresh_selected_tail_hit_fraction": mean("fresh_selected_tail_hit_fraction"),
            "blocks_with_a_searched_winner": sum(1 for s in st if s["fresh_selected_success"] is not None),
            "candidates_with_screen_success": total("candidates_with_screen_success"),
            "slots": slots, "valid_slots": total("valid_slots"),
            "slots_valid_within_one_correction": total("valid_slots") / slots if slots else None,
            "first_attempt_valid_slots": total("first_attempt_valid_slots"),
            "module_contract_defect_slots": total("module_contract_defect_slots"),
            "search_candidates_with_probe_success": total("search_candidates_with_probe_success"),
            "fresh_selected_drops": total("fresh_selected_drops"),
            "fresh_selected_failure_labels": labels,
            "promoted_blocks": sum(s["promoted"] for s in st), "plateau_blocks": sum(s["plateau"] for s in st),
            "llm_calls": total("llm_calls"), "llm_tokens": total("llm_tokens"),
            "isaac_processes": total("isaac_processes"), "isaac_episodes": total("isaac_episodes")}
    out["arms"] = arms
    out["slots_valid_within_one_correction"] = {arm: arms[arm]["slots_valid_within_one_correction"] for arm in ARMS}
    null_eps = sum(blk["null"].get("isaac_episodes", 0) for blk in blocks)
    out["isaac_episodes"] = null_eps + sum(arms[arm]["isaac_episodes"] for arm in ARMS)
    out["isaac_processes"] = (sum(blk["null"].get("isaac_processes", 0) for blk in blocks)
                              + sum(arms[arm]["isaac_processes"] for arm in ARMS))
    out["llm_calls"] = sum(arms[arm]["llm_calls"] for arm in ARMS)
    out["llm_tokens"] = sum(arms[arm]["llm_tokens"] for arm in ARMS)
    return out


def run_study(ccfg: dict, stream_factory, client) -> dict:
    from lares.envs.isaac_pen_spin import llm_pool

    sdir = ccfg["study_dir"]
    os.makedirs(sdir, exist_ok=True)
    freeze_manifests(ccfg)
    seeds = [int(s) for s in ccfg["block_seeds"]]
    blocks = []
    with llm_pool.pool_for(ccfg) as pool:
        # Cold prompts depend on no episode, so every (block, arm) generation is queued before any Isaac work.
        for b in range(len(seeds)):
            bdir = os.path.join(sdir, f"block{b}")
            for arm in ARMS:
                if arm_needs_generation(bdir, arm):
                    pool.submit((b, arm), generate_arm, arm, ccfg, bdir, client)
        for b, seed in enumerate(seeds):
            roles = block_roles(seed)
            bdir = os.path.join(sdir, f"block{b}")
            null = run_null(roles, bdir, stream_factory)
            row = {"block": b, "stage": STAGE, "seed": seed,
                   "null": {**{k: null[k] for k in ROLES}, "isaac_processes": null["isaac_processes"],
                            "isaac_episodes": null["isaac_episodes"]}}
            row["arms"] = {arm: run_arm(arm, ccfg, seed, roles, null, bdir, stream_factory, client,
                                        generated=pool.result((b, arm)) if (b, arm) in pool.pending() else None)
                           for arm in ARMS}
            blocks.append(row)
            json.dump(summarise(blocks), open(os.path.join(sdir, "summary.partial.json"), "w"), indent=2, default=str)
    summary = summarise(blocks)
    with open(os.path.join(sdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


__all__ = ["ARMS", "LABEL", "MAX_ARM_EPISODES", "MODULAR_REQUIREMENT", "PRIMARY", "ROLES", "SEARCH_CANDIDATES", "SLOTS",
           "STAGE", "arm_needs_generation", "arm_prompts", "block_roles", "freeze_manifests", "generate_slots",
           "probe_search", "run_arm", "run_study", "summarise"]
