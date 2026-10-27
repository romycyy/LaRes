"""R-042: the keyframe-repair study on Stage-3 pen spinning, driven by ``scripts/run_full_evolution.py --study``.

Order (each step skipped if its output exists, so an interrupted study resumes without rerunning a case):

1. ``freeze``: every manifest from its pool seed (probe 4, development 40 = screen 0-9 + promotion 10-39, fresh 30
   per block; one 10-case demonstration manifest; one 30-case parity manifest). Run before the commit; no Isaac.
2. ``parity``: the D-036 expert through the new pipeline backend and through ``protocols.run_cell`` (the evaluator
   R-040 used), separate processes, on the parity manifest's first 4, first 10 and all 30 cases: zero sustained
   flips and at most ``floor(batch / 10)`` diverged environments per batch, or the study aborts before any LLM call.
3. Per block: the prefix run (slot 0 and the trigger, ``gen_0/prefix_state.json``). A block whose trigger does not
   fire has no plateau parent and is reported as missing. Otherwise: the shared demonstrations (collected once, at
   the first eligible block; fewer than 9/10 sustained successes stop the study as inconclusive, never rerun), then
   the three continuation arms in order ``no_trace``, ``uniform``, ``event``, each in a copy of the prefix run.
4. Fresh evaluation of each arm's final incumbent on the block's 30 fresh cases, one process each; a null incumbent
   scores 0/30 by definition and runs no episode.
5. ``summary.json``: per-block outcomes and the block-paired t summaries.
"""

from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from lares.envs.isaac_pen_spin import cases

STAGE = 3
ARMS = ("no_trace", "uniform", "event")
PRIMARY = ("event", "uniform")
SECONDARY = (("uniform", "no_trace"), ("event", "no_trace"))
PARITY_BATCHES = (4, 10, 30)
MIN_ELIGIBLE_BLOCKS = 3


# ---------------------------------------------------------------------------------------------------------------------
#  Manifests
# ---------------------------------------------------------------------------------------------------------------------


def manifest_specs(ps: dict) -> list[tuple[str, int, int, str]]:
    """``(label, pool_seed, cases, role)`` for every manifest the study names."""
    out = [("demo", int(ps["demonstration_seed"]), int(ps["demonstration_cases"]), "demonstration"),
           ("parity", int(ps["parity_seed"]), int(ps["parity_cases"]), "parity")]
    for b, blk in enumerate(ps["blocks"]):
        out += [("probe", int(blk["probe_seed"]), int(ps["probe_cases"]), f"block{b}:probe"),
                ("development", int(blk["development_seed"]), int(ps["development_cases"]), f"block{b}:development"),
                ("fresh", int(blk["fresh_seed"]), int(ps["fresh_cases"]), f"block{b}:fresh")]
    return out


def manifest_file(label: str, seed: int) -> str:
    return cases.manifest_path(STAGE, label, seed)


def freeze_manifests(ps: dict) -> dict:
    """Build (or verify) every manifest; refuse a placement shared between any two."""
    from lares.eval.manifest import EvaluationManifest

    built, hashes = {}, {}
    for label, seed, n, role in manifest_specs(ps):
        m = cases.build_manifest(STAGE, seed, n, label=label)
        path = manifest_file(label, seed)
        if os.path.exists(path):
            if EvaluationManifest.load(path).to_dict() != m.to_dict():
                raise ValueError(f"{path} exists and differs from the manifest this study builds")
        else:
            m.save(path)
        for c in m.episodes:
            if c.task_hash in hashes:
                raise ValueError(f"{role} shares placement {c.task_hash} with {hashes[c.task_hash]}")
            hashes[c.task_hash] = role
        built[role] = path
    return built


def load(label: str, seed: int):
    from lares.eval.manifest import EvaluationManifest

    return EvaluationManifest.load(manifest_file(label, seed))


# ---------------------------------------------------------------------------------------------------------------------
#  Parity gate
# ---------------------------------------------------------------------------------------------------------------------


def expert_actor(ps: dict):
    from lares.envs.isaac_pen_spin.pipeline_backend import IsaacExpertActor

    e = ps["expert"]
    return IsaacExpertActor(e["checkpoint"], e["checkpoint_sha256"], int(e["training_seed"]))


def compare_cells(a: dict, b: dict, batch: int) -> dict:
    """Outcome flips and diverged environments between two traces of the same cases."""
    flips = sum(bool(x["sustained"]) != bool(y["sustained"]) for x, y in zip(a["rows"], b["rows"]))
    diverged = set()
    for key in ("spin_rate", "center_dist"):
        x, y = np.nan_to_num(a[key], nan=-1e9), np.nan_to_num(b[key], nan=-1e9)
        diverged |= {int(i) for i in np.nonzero((np.abs(x - y) > 0).any(axis=0))[0]}
    diverged |= {i for i, (x, y) in enumerate(zip(a["rows"], b["rows"])) if x["length"] != y["length"]}
    allowed = batch // 10
    return {"batch": batch, "sustained_outcome_flips": int(flips), "diverged_envs": len(diverged),
            "max_diverged_envs": allowed, "passes": bool(flips == 0 and len(diverged) <= allowed),
            "successes": [int(sum(r["sustained"] for r in a["rows"])), int(sum(r["sustained"] for r in b["rows"]))]}


def run_cell_reference(ps: dict, manifest_path: str, out: str, config_path: str) -> dict:
    """The expert through ``protocols.run_cell`` (R-040's evaluator), its own Isaac process."""
    if not os.path.exists(out):
        cmd = (f"source {ps['isaac_env']} >/dev/null 2>&1 && python -u scripts/isaac/train_pen_spin_expert.py "
               f"--config {config_path} --phase cell --cell-manifest {manifest_path} "
               f"--checkpoint {ps['expert']['checkpoint']} --out {out}")
        subprocess.run(["bash", "-c", cmd], check=True)
    data = np.load(out.replace(".json", ".npz"))
    return {"rows": json.load(open(out))["episodes"], "spin_rate": data["spin_rate"], "center_dist": data["center_dist"]}


def parity_gate(ps: dict, study_dir: str, stream_factory, expert_config: str) -> dict:
    out_path = os.path.join(study_dir, "parity.json")
    if os.path.exists(out_path):
        return json.load(open(out_path))
    full = load("parity", int(ps["parity_seed"]))
    pdir = os.path.join(study_dir, "parity")
    os.makedirs(pdir, exist_ok=True)
    stream = stream_factory("parity", pdir)
    report = {}
    for n in PARITY_BATCHES:
        m = full.head(n)
        mpath = os.path.join(pdir, f"parity_head{n}.yaml")
        m.save(mpath)
        ref = run_cell_reference(ps, mpath, os.path.join(pdir, f"run_cell_head{n}.json"), expert_config)
        got = stream.collect_traces(expert_actor(ps), m)
        data = np.load(got["trace_path"])
        new = {"rows": got["episodes"], "spin_rate": data["spin_rate"], "center_dist": data["center_dist"]}
        report[f"batch{n}"] = compare_cells(new, ref, n)
    report["passes"] = all(v["passes"] for v in report.values() if isinstance(v, dict))
    report["episodes"] = 2 * sum(PARITY_BATCHES)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    return report


# ---------------------------------------------------------------------------------------------------------------------
#  Demonstrations
# ---------------------------------------------------------------------------------------------------------------------


def demonstrations(ps: dict, study_dir: str, stream_factory, min_successes: int, traces_shown: int) -> dict:
    """The expert alone on the 10-case demonstration manifest, once; the qualification gate; the traces shown."""
    from lares.repair import pen_spin_traces as pst

    path = os.path.join(study_dir, "demonstrations.json")
    if os.path.exists(path):
        return json.load(open(path))
    started = os.path.join(study_dir, "demonstrations.started")
    if os.path.exists(started):
        raise SystemExit("the demonstration collection was interrupted earlier; the study stops inconclusive "
                         "(R-042: demonstration cases are never rerun)")
    open(started, "w").close()
    m = load("demo", int(ps["demonstration_seed"]))
    collected = stream_factory("demonstrations", os.path.join(study_dir, "demonstrations")).collect_traces(
        expert_actor(ps), m)
    demos = pst.demonstrations_from_traces(m, collected)
    selected = pst.select_traces(demos, traces_shown)
    out = {"manifest_id": m.manifest_id, "episodes": len(demos["episodes"]), "successes": demos["successes"],
           "min_successes": int(min_successes), "qualified": demos["successes"] >= int(min_successes),
           "case_ids": [e["case"] for e in demos["episodes"]], "traces_case_ids": [e["case"] for e in selected],
           "traces_sha256": pst.traces_sha256(selected) if selected else None,
           "demonstrations_sha256": pst.traces_sha256(demos["episodes"]), "trace_path": demos["trace_path"],
           "selected": selected}
    with open(path, "w") as f:
        json.dump(out, f)
    return out


# ---------------------------------------------------------------------------------------------------------------------
#  Fresh evaluation and summary
# ---------------------------------------------------------------------------------------------------------------------


def final_incumbent(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "final_incumbent.json")) as f:
        return json.load(f)


def fresh_evaluation(run_dir: str, manifest, stream_factory, env_id: str) -> dict:
    """The arm's final incumbent on the block's fresh cases; a null incumbent scores 0 without an episode."""
    path = os.path.join(run_dir, "fresh_evaluation.json")
    if os.path.exists(path):
        return json.load(open(path))
    inc = final_incumbent(run_dir)
    if inc["candidate_id"] is None:
        out = {"candidate_id": None, "successes": 0, "cases": len(manifest), "success": 0.0, "episodes": 0,
               "rule": "null incumbent: 0 by the pipeline's NULL_INCUMBENT_ID definition, no episode run",
               "per_case": [0.0] * len(manifest)}
    else:
        from lares.eval.runner import PolicyActor, evaluate_manifest

        policy = rebuild_incumbent(run_dir, inc, env_id)
        res = evaluate_manifest(PolicyActor(policy, deterministic=True, name=inc["candidate_id"]), manifest,
                                stream_factory("fresh", os.path.join(run_dir, "isaac_fresh")))
        per = [float(e.success) for e in res.episodes]
        out = {"candidate_id": inc["candidate_id"], "successes": int(sum(per)), "cases": len(per),
               "success": float(np.mean(per)), "episodes": len(per), "per_case": per,
               "tail_hit_fraction": float(np.mean([e.signed_goal_progress for e in res.episodes]))}
    out["manifest_id"] = manifest.manifest_id
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return out


def rebuild_incumbent(run_dir: str, inc: dict, env_id: str):
    import torch

    from lares.core.obs_schema import active_schema, get_obs_schema
    from lares.core.policy_generation import GENERATED_IMPORTS

    source = open(os.path.join(run_dir, inc["code_path"]), encoding="utf-8").read()
    ns: dict = {}
    exec(GENERATED_IMPORTS + source, ns)
    with active_schema(get_obs_schema(env_id)):
        policy = ns["GeneratedPolicy"](157, cases.NUM_ACTIONS)
    policy.fixed_action_scale = float(inc["fixed_action_scale"])
    policy.load_state_dict(torch.load(os.path.join(run_dir, inc["state_path"]), weights_only=True))
    policy._generated_source = source
    policy.eval()
    return policy


def summarise(blocks: list[dict]) -> dict:
    from lares.eval.stats import t_summary

    eligible = [b for b in blocks if b.get("eligible")]
    out = {"blocks": blocks, "eligible_blocks": len(eligible), "min_eligible_blocks": MIN_ELIGIBLE_BLOCKS}
    for a, b in (PRIMARY,) + SECONDARY:
        d = [blk["fresh"][a]["success"] - blk["fresh"][b]["success"] for blk in eligible]
        out[f"{a}_minus_{b}"] = {"per_block": d, **(t_summary(d) if len(d) >= 2 else {"n": len(d)})}
    out["paired_success_difference"] = out[f"{PRIMARY[0]}_minus_{PRIMARY[1]}"].get("mean")
    out["conclusive"] = len(eligible) >= MIN_ELIGIBLE_BLOCKS
    return out


__all__ = ["ARMS", "PARITY_BATCHES", "compare_cells", "demonstrations", "freeze_manifests", "fresh_evaluation",
           "manifest_specs", "parity_gate", "summarise"]
