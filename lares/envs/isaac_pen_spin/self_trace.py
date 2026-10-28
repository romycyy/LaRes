"""R-054: does repair evidence drawn from the learner's own successes beat expert demonstrations?

Driven by ``scripts/run_full_evolution.py --config config/pen_spin_plateau_repair.yaml --study self_trace_source``.
Approved with advisor call_00110. The common parent is the frozen R-051 learner (the R-046 block-3 promoted
controller, ``config/pen_spin/r046_block3_incumbent/``). The two arms differ only in which successful episodes the
repair request shows:

* ``self``: two sustained successes of the parent itself, the first two in manifest order of the archived R-051 donor
  run (seed 4310, read only).
* ``expert``: two sustained successes of the D-036 PPO expert, the two traced episodes of the archived R-046 block-1
  expert collection (seed 4191, read only).

Each source shows 16 uniform keyframes per episode (32 rows), rendered with the same fields and rounding
(``lares.repair.pen_spin_traces``). The packets are built once by ``build_evidence`` and committed under
``config/pen_spin/r054_evidence/`` before any LLM call; the run checks their hashes. Parent code, its parameter values,
the block's failure evidence, the instructions and the candidate budget are identical across arms.

Eight blocks, one frozen 80-case Stage-3 manifest each (seeds 4500-4507), split by index into ``ROLES``. Per block
(each step skipped if its output exists; an interrupted step that spent episodes is never rerun):

1. ``parent``: the parent on the screen (10) and promotion (30) cases. Its screen outcome is the failure evidence
   both arms see.
2. ``self`` and ``expert``: four repair slots (one call plus at most one linked correction each), evaluated at their
   declared initial parameters (no fitting, no search). Every valid candidate is screened; the best by the
   checkpoint tie rule is scored on the promotion cases and replaces the parent only under the strict paired rule.
3. ``fresh``: each arm's final incumbent on the 40 fresh cases (the parent's fresh score is shared when an arm keeps it).

Primary: self minus expert fresh final-incumbent success, paired over the eight blocks (blocks are the units).
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin.gait_repair import public, score

LABEL = "self_trace_source"
STAGE = 3
ROLES = {"screen": (0, 10), "promotion": (10, 40), "fresh": (40, 80)}
BLOCK_CASES = 80
ARMS = ("self", "expert")
PRIMARY = ("self", "expert")
SLOTS = 4
TRACE_EPISODES = 2

SOURCE_TEXT = {
    "self": ("Below are successful episodes of THIS SAME controller, run with the parameter values above on other "
             "cases: its own states and its own executed actions when it succeeded."),
    "expert": ("Below are successful episodes of an expert policy (a closed-loop neural-network controller trained "
               "with reinforcement learning) on other cases: its states and its executed actions when it succeeded."),
}
FIELDS = ("Per kept step: t; STATE pen_offset = pen centre minus in_hand_pos (m), pen_axis = the pen's long axis in "
          "the world frame, pen_angvel = pen angular velocity (rad/s, world frame, NOT scaled by 0.2), roll = signed "
          "spin rate about the pen's own axis (rad/s), hand = hand_dof_pos exactly as the policy observes it (24 joint "
          "positions normalised to [-1, 1], not radians; hand joint order); ACTION = the 20 executed actions (action "
          "order).")

REQUEST_TEMPLATE = """Below is a controller for this task. It does not reliably {goal_phrase}. On this block's
10 screening episodes it succeeded on {successes} of 10 (failure labels: {failures}). It was evaluated with these
parameter values, which differ from the initial values its code declares: {parameters}

Rewrite the controller so that it succeeds more reliably. You may change any part of it. Your controller is evaluated
exactly at the initial parameter values its code declares, with no fitting or search. Keep the class name
GeneratedPolicy, read observations only through self.obs_field, and declare every parameter with get_param_ranges.

```python
{parent}
```

{source_text} {fields} {keyframes} steps per episode are shown.
{packet}"""


# ---------------------------------------------------------------------------------------------------------------------
#  Evidence (built once from archived runs, committed, hash-checked at run time)
# ---------------------------------------------------------------------------------------------------------------------


def _strip(rows: list[dict]) -> list[dict]:
    keep = ("t", "pen_offset", "pen_axis", "pen_angvel", "roll", "hand", "action")
    return [{k: r[k] for k in keep} for r in rows]


def self_episodes(donor_json: str) -> list[dict]:
    """The first ``TRACE_EPISODES`` sustained successes of the archived learner run, in manifest order."""
    from lares.repair import pen_spin_traces as pst

    rows = json.load(open(donor_json))["episodes"]
    z = np.load(donor_json.replace(".json", ".npz"))
    out = []
    for i, r in enumerate(rows):
        if r["sustained"]:
            out.append({"case": r["case_id"], "length": int(r["length"]),
                        "rows": pst.episode_rows(z["obs"][i], z["actions"][i], int(r["length"]))})
        if len(out) == TRACE_EPISODES:
            break
    return out


def expert_episodes(expert_json: str) -> list[dict]:
    e = json.load(open(expert_json))
    return [{"case": t["case"], "length": int(t["length"]), "rows": _strip(t["rows"])}
            for t in e["traced"] if t["success"]][:TRACE_EPISODES]


def packet(episodes: list[dict]) -> tuple[str, list]:
    from lares.repair import pen_spin_traces as pst

    lines, kept = [], []
    for k, e in enumerate(episodes):
        idx = pst.uniform_keyframes(len(e["rows"]))
        kept.append(idx)
        lines.append(f"Episode {k + 1} ({len(e['rows'])} steps):")
        lines += [pst.render_row(e["rows"][t]) for t in idx]
    return "\n".join(lines), kept


def build_evidence(scfg: dict) -> dict:
    """Write (or verify) the two frozen packets; refuses if either source has too few successful episodes."""
    from lares.utils.hashing import sha256_text

    ev = scfg["evidence"]
    sources = {"self": self_episodes(ev["self_source"]), "expert": expert_episodes(ev["expert_source"])}
    out = {}
    for arm, eps in sources.items():
        if len(eps) < TRACE_EPISODES:
            raise SystemExit(f"{arm}: only {len(eps)} archived successes; the predeclared evidence budget is "
                             f"{TRACE_EPISODES} episodes, so the comparison is not run")
        text, kept = packet(eps)
        rec = {"arm": arm, "source": ev[f"{arm}_source"], "cases": [e["case"] for e in eps],
               "lengths": [e["length"] for e in eps], "kept_steps": kept, "rows_shown": sum(len(k) for k in kept),
               "packet": text, "packet_sha256": sha256_text(text)}
        path = os.path.join(ev["dir"], f"{arm}.json")
        if os.path.exists(path):
            if json.load(open(path))["packet_sha256"] != rec["packet_sha256"]:
                raise ValueError(f"{path} exists and differs from the packet the archived source gives")
        else:
            os.makedirs(ev["dir"], exist_ok=True)
            with open(path, "w") as f:
                json.dump(rec, f, indent=1)
        out[arm] = {"path": path, "packet_sha256": rec["packet_sha256"], "cases": rec["cases"],
                    "rows_shown": rec["rows_shown"]}
    return out


def load_evidence(scfg: dict) -> dict:
    from lares.utils.hashing import sha256_text

    ev = scfg["evidence"]
    out = {}
    for arm in ARMS:
        rec = json.load(open(os.path.join(ev["dir"], f"{arm}.json")))
        if sha256_text(rec["packet"]) != rec["packet_sha256"] or rec["packet_sha256"] != ev[f"{arm}_sha256"]:
            raise SystemExit(f"{arm} evidence packet does not match its frozen hash")
        out[arm] = rec
    if out["self"]["rows_shown"] != out["expert"]["rows_shown"]:
        raise SystemExit("the two evidence packets show different numbers of rows")
    return out


# ---------------------------------------------------------------------------------------------------------------------
#  Manifests
# ---------------------------------------------------------------------------------------------------------------------


def manifest_file(seed: int) -> str:
    return cases.manifest_path(STAGE, LABEL, seed)


def freeze_manifests(scfg: dict) -> dict:
    from lares.eval.manifest import EvaluationManifest

    built, hashes = {}, {}
    for b, seed in enumerate(int(s) for s in scfg["block_seeds"]):
        m = cases.build_manifest(STAGE, seed, BLOCK_CASES, label=LABEL)
        path = manifest_file(seed)
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


def block_roles(seed: int) -> dict:
    from lares.eval.manifest import EvaluationManifest

    m = EvaluationManifest.load(manifest_file(seed))
    if len(m) != BLOCK_CASES or cases.stage_of(m) != STAGE:
        raise ValueError(f"block manifest for seed {seed} is not an {BLOCK_CASES}-case stage-{STAGE} one")
    return {role: EvaluationManifest(manifest_id=f"{m.manifest_id}:{role}", environment=m.environment, pool=m.pool,
                                     split=m.split, episodes=list(m.episodes[a:b]))
            for role, (a, b) in ROLES.items()}


# ---------------------------------------------------------------------------------------------------------------------
#  Parent, request, generation
# ---------------------------------------------------------------------------------------------------------------------


def load_parent(scfg: dict):
    import torch

    from lares.envs.isaac_pen_spin.dense_fit import build_policy
    from lares.envs.isaac_pen_spin.learner_recovery import check_learner

    lc = check_learner(scfg)
    source = open(lc["source"], encoding="utf-8").read()
    return build_policy(source, torch.load(lc["state"], weights_only=True), cases.STAGES[STAGE].task_id,
                        float(lc["fixed_action_scale"]))


def parent_parameters(policy) -> dict:
    return {k: (float(v) if v.numel() == 1 else v.detach().cpu().numpy().tolist())
            for k, v in policy.named_parameters()}


def _fmt_params(params: dict) -> str:
    def fmt(v):
        return f"{v:.4g}" if isinstance(v, float) else "[" + ", ".join(f"{x:.4g}" for x in v) + "]"

    return ", ".join(f"{k}={fmt(v)}" for k, v in sorted(params.items()))


def repair_request(arm: str, parent_source: str, params: dict, screen: dict, evidence: dict) -> tuple[str, dict]:
    from collections import Counter

    from lares.repair import pen_spin_traces as pst
    from lares.utils.hashing import sha256_text

    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}")
    failures = Counter(lab for ok, lab in zip(screen["per_case"], screen["failure_labels"]) if not ok)
    fail_text = ", ".join(f"{k} x{v}" for k, v in sorted(failures.items())) or "none"
    text = REQUEST_TEMPLATE.format(goal_phrase=pst.GOAL_PHRASE, successes=int(round(sum(screen["per_case"]))),
                                   failures=fail_text, parameters=_fmt_params(params), parent=parent_source.strip(),
                                   source_text=SOURCE_TEXT[arm], fields=FIELDS, keyframes=pst.KEYFRAMES,
                                   packet=evidence[arm]["packet"])
    return text, {"arm": arm, "packet_sha256": evidence[arm]["packet_sha256"], "request_sha256": sha256_text(text),
                  "request_chars": len(text), "trace_cases": evidence[arm]["cases"]}


def generate(arm: str, request: str, packet: dict, adir: str, client, model: str, reasoning_effort: str,
             max_calls_per_slot: int = 2) -> dict:
    """``SLOTS`` independent repairs of the parent from one request (each one call plus at most one correction)."""
    import pickle

    from lares.core.policy_generation import PlateauSlotGenerator
    from lares.core.training_pipeline import load_policy_prompt_assets
    from lares.search.generation import LLMCallLedger

    os.makedirs(adir, exist_ok=True)
    data_pkl = os.path.join(adir, "data.pkl")
    with open(data_pkl, "wb") as f:
        pickle.dump([{"obs": np.zeros(157)}], f)
    env_id = cases.STAGES[STAGE].task_id
    ledger = LLMCallLedger(adir)
    gen = PlateauSlotGenerator(client, SimpleNamespace(model=model, reasoning_effort=reasoning_effort),
                               os.path.join(adir, "gen_0"), 157, cases.NUM_ACTIONS, env_id,
                               load_policy_prompt_assets(env_id), SimpleNamespace(n=SLOTS), data_pkl, ledger,
                               llm_transcript_path=os.path.join(adir, "llm_transcript.log"),
                               max_total_calls=SLOTS * int(max_calls_per_slot), max_repair_per_candidate=1)
    slots = []
    for k in range(SLOTS):
        attempts = gen.trace_repair(k, {"candidate_id": "r051_learner"}, request,
                                    {"lane": "self_trace_source", "arm": arm, "slot": k, "parent": "r051_learner",
                                     **packet})
        final = attempts[-1]
        slots.append({"slot": k, "attempts": [{"attempt_id": c.attempt_id, "valid": c.valid,
                                               "validation": c.validation, "llm_call_ids": c.llm_call_ids}
                                              for c in attempts],
                      "valid": bool(final.valid), "policy": final.policy if final.valid else None,
                      "code": final.code if final.valid else None})
    return {"slots": slots, "llm_calls": len(ledger), "llm_tokens": ledger.totals()["llm_total_tokens"]}


# ---------------------------------------------------------------------------------------------------------------------
#  Steps
# ---------------------------------------------------------------------------------------------------------------------


def _started(bdir: str, name: str) -> None:
    started = os.path.join(bdir, f"{name}.started")
    if os.path.exists(started):
        raise SystemExit(f"{started}: this evaluation was interrupted earlier; its cases are never rerun")
    os.makedirs(bdir, exist_ok=True)
    open(started, "w").close()


def run_parent(parent, roles: dict, bdir: str, stream_factory) -> dict:
    from lares.eval.runner import PolicyActor

    path = os.path.join(bdir, "parent.json")
    if os.path.exists(path):
        return json.load(open(path))
    _started(bdir, "parent")
    stream = stream_factory("parent", os.path.join(bdir, "parent"))
    actor = PolicyActor(parent, deterministic=True, name="r051_learner")
    out = {}
    for role in ("screen", "promotion"):
        s = score(actor, roles[role], stream)
        if role == "promotion":
            out["promotion_returns"] = [float(e.episode_return) for e in s["_result"].episodes]
        out[role] = public(s)
    out.update(isaac_processes=stream.calls, isaac_episodes=stream.episodes)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    return out


def _parent_like(record: dict, manifest, returns: list[float]):
    from lares.eval.runner import ManifestResult

    if record["manifest_id"] != manifest.manifest_id:
        raise ValueError("the saved record is for another manifest")
    res = ManifestResult(actor_name="r051_learner", manifest_id=manifest.manifest_id, action_mode="deterministic",
                         schema_id=manifest.environment.env_id)
    for case, ok, ret in zip(manifest.episodes, record["per_case"], returns):
        res.episodes.append(SimpleNamespace(case_id=case.case_id, task_hash=case.task_hash, success=bool(ok),
                                            episode_return=float(ret), signed_goal_progress=None))
    return res


def run_arm(arm: str, scfg: dict, roles: dict, parent_rec: dict, parent_source: str, params: dict, evidence: dict,
            bdir: str, stream_factory, client) -> dict:
    """Generate, screen, promote against the parent. Fresh scoring is a separate step (``run_fresh``)."""
    from lares.eval.promotion import compare_to_incumbent
    from lares.eval.runner import PolicyActor
    from lares.fitting.checkpoints import RolloutScore, is_better_checkpoint

    adir = os.path.join(bdir, arm)
    path = os.path.join(adir, "arm.json")
    if os.path.exists(path):
        return json.load(open(path))
    if os.path.exists(os.path.join(adir, "isaac")):
        raise SystemExit(f"{adir} was interrupted after spending episodes; an interrupted arm is not rerun")
    request, pk = repair_request(arm, parent_source, params, parent_rec["screen"], evidence)
    os.makedirs(adir, exist_ok=True)
    with open(os.path.join(adir, "request.txt"), "w", encoding="utf-8") as f:
        f.write(request)
    g = generate(arm, request, pk, adir, client, scfg["model"], scfg["reasoning_effort"],
                 int(scfg.get("max_calls_per_slot", 2)))
    out = {"arm": arm, "packet": pk, "llm_calls": g["llm_calls"], "llm_tokens": g["llm_tokens"],
           "slots": [{k: v for k, v in s.items() if k != "policy"} for s in g["slots"]]}
    valid = [(s["slot"], s["policy"]) for s in g["slots"] if s["valid"]]
    out["valid_candidates"] = len(valid)
    if not valid:
        out.update(promoted=False, selected_slot=None, screens={}, isaac_episodes=0)
        with open(path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        return out
    stream = stream_factory(arm, os.path.join(adir, "isaac"))
    actors = {k: PolicyActor(p, deterministic=True, name=f"{arm}:slot{k}") for k, p in valid}
    screens, best, best_k = {}, None, None
    for k, _ in valid:
        s = score(actors[k], roles["screen"], stream)
        screens[str(k)] = public(s)
        r = RolloutScore(s["success"], s["tail_hit_fraction"])
        if is_better_checkpoint(r, best):
            best, best_k = r, k
    promo = score(actors[best_k], roles["promotion"], stream)
    incumbent = _parent_like(parent_rec["promotion"], roles["promotion"], parent_rec["promotion_returns"])
    decision = compare_to_incumbent(promo["_result"], incumbent, f"{arm}:slot{best_k}", "r051_learner",
                                    excluded_manifest_ids=[roles["screen"].manifest_id])
    chosen = dict(valid)[best_k]
    import torch

    torch.save(chosen.state_dict(), os.path.join(adir, "selected_state.pt"))
    with open(os.path.join(adir, "selected_source.py"), "w", encoding="utf-8") as f:
        f.write(chosen._generated_source)
    out.update(screens=screens, selected_slot=best_k, promotion=public(promo), promotion_decision=decision.to_dict(),
               promoted=bool(decision.promoted), isaac_processes=stream.calls, isaac_episodes=stream.episodes)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    return out


def _load_selected(adir: str):
    import torch

    from lares.envs.isaac_pen_spin.dense_fit import build_policy

    source = open(os.path.join(adir, "selected_source.py"), encoding="utf-8").read()
    return build_policy(source, torch.load(os.path.join(adir, "selected_state.pt"), weights_only=True),
                        cases.STAGES[STAGE].task_id, 0.1)


def run_fresh(arms: dict, parent, roles: dict, bdir: str, stream_factory) -> dict:
    """Each arm's final incumbent on the fresh cases; the parent is scored once if any arm keeps it."""
    from lares.eval.runner import PolicyActor

    path = os.path.join(bdir, "fresh.json")
    if os.path.exists(path):
        return json.load(open(path))
    _started(bdir, "fresh")
    stream = stream_factory("fresh", os.path.join(bdir, "fresh"))
    out, parent_fresh = {}, None
    for arm in ARMS:
        if arms[arm].get("promoted"):
            p = _load_selected(os.path.join(bdir, arm))
            s = public(score(PolicyActor(p, deterministic=True, name=f"{arm}:slot{arms[arm]['selected_slot']}"),
                             roles["fresh"], stream))
            out[arm] = {**s, "candidate": f"slot{arms[arm]['selected_slot']}"}
        else:
            if parent_fresh is None:
                parent_fresh = public(score(PolicyActor(parent, deterministic=True, name="r051_learner"),
                                            roles["fresh"], stream))
            out[arm] = {**parent_fresh, "candidate": "parent"}
    out.update(isaac_processes=stream.calls, isaac_episodes=stream.episodes)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    return out


# ---------------------------------------------------------------------------------------------------------------------
#  Study
# ---------------------------------------------------------------------------------------------------------------------


def summarise(blocks: list[dict]) -> dict:
    from lares.eval.stats import t_summary, win_loss_tie

    out = {"blocks": blocks, "n_blocks": len(blocks)}
    a, b = PRIMARY
    d = [blk["fresh"][a]["success"] - blk["fresh"][b]["success"] for blk in blocks]
    out[f"{a}_minus_{b}_fresh_incumbent"] = {"per_block": d, **(t_summary(d) if len(d) >= 2 else {"n": len(d)}),
                                             **win_loss_tie(d)}
    out["paired_success_difference"] = out[f"{a}_minus_{b}_fresh_incumbent"].get("mean")
    per_arm = {}
    for arm in ARMS:
        rows = [blk["arms"][arm] for blk in blocks]
        per_arm[arm] = {
            "promoted_blocks": sum(bool(r.get("promoted")) for r in rows),
            "valid_candidates": sum(r.get("valid_candidates", 0) for r in rows),
            "candidates_beating_parent_screen": sum(
                sum(1 for s in r.get("screens", {}).values() if s["success"] > blk["parent"]["screen"]["success"])
                for r, blk in zip(rows, blocks)),
            "mean_fresh_incumbent": float(np.mean([blk["fresh"][arm]["success"] for blk in blocks])),
            "mean_best_screen": float(np.mean([max([s["success"] for s in r.get("screens", {}).values()] or [0.0])
                                               for r in rows])),
            "llm_calls": sum(r.get("llm_calls", 0) for r in rows),
            "llm_tokens": sum(r.get("llm_tokens", 0) for r in rows)}
    out["per_arm"] = per_arm
    out["parent"] = {"mean_screen": float(np.mean([blk["parent"]["screen"]["success"] for blk in blocks])),
                     "mean_promotion": float(np.mean([blk["parent"]["promotion"]["success"] for blk in blocks]))}
    return out


def run_study(scfg: dict, stream_factory, client) -> dict:
    sdir = scfg["study_dir"]
    os.makedirs(sdir, exist_ok=True)
    freeze_manifests(scfg)
    evidence = load_evidence(scfg)
    parent = load_parent(scfg)
    parent_source = parent._generated_source
    params = parent_parameters(parent)
    blocks = []
    for b, seed in enumerate(int(s) for s in scfg["block_seeds"]):
        roles = block_roles(seed)
        bdir = os.path.join(sdir, f"block{b}")
        prec = run_parent(parent, roles, bdir, stream_factory)
        arms = {arm: run_arm(arm, scfg, roles, prec, parent_source, params, evidence, bdir, stream_factory, client)
                for arm in ARMS}
        fresh = run_fresh(arms, parent, roles, bdir, stream_factory)
        blocks.append({"block": b, "seed": seed, "parent": {k: prec[k] for k in ("screen", "promotion")},
                       "arms": arms, "fresh": {arm: fresh[arm] for arm in ARMS}})
        json.dump(summarise(blocks), open(os.path.join(sdir, "summary.partial.json"), "w"), indent=2, default=str)
    summary = summarise(blocks)
    summary["evidence"] = {arm: {k: evidence[arm][k] for k in ("source", "cases", "rows_shown", "packet_sha256")}
                           for arm in ARMS}
    with open(os.path.join(sdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


__all__ = ["ARMS", "LABEL", "ROLES", "SOURCE_TEXT", "block_roles", "build_evidence", "freeze_manifests",
           "load_evidence", "repair_request", "run_study", "summarise"]
