#!/usr/bin/env python
"""R-005: does a bounded LLM mutation beat its own parent when both are fitted the same way?

The two-lane run (``logs/two_lane_search``) fitted each fittable-lane child but never its parent
under the same conditions, so "the child is good" and "the parent fitted is just as good" were
never told apart. This refits every child (from its saved source, at its declared initial values)
and every unique parent (the seed-bank source the child's lineage names) with identical
demonstrations, objective, budget, checkpoints, probe cases and fitting seed. Each checkpoint is
selected by probe rollouts (development 40-49), never by loss. Every selected policy is then scored
on one fresh MT1 manifest from an unused pool seed.

The primary statistic is the paired t interval over placements: for each fresh case, the mean over
the six child-minus-parent success differences. The six pairs share placements, so the 300 episode
differences are not independent; the 50 placement averages are. Per-pair differences are descriptive.

Secondary (R-001, merged): the two-lane winner (``performance_incumbent`` in ``incumbents.json``,
here refitted with the R-005 fitting seed like every child) against the baseline run's best
candidate (the one sent to the promotion test, scored at its saved state) on the same cases.

Seeds are checked against the seed ledger and every saved manifest, then reserved with this
out-dir as the artifact. ``--reserve-only`` does just that, so the reservation can be committed
from a machine without MetaWorld before the run. A rerun into the same out-dir reuses its manifest.

Usage (from the project root)::

    python scripts/compare_child_vs_parent.py --run logs/two_lane_search \\
        --baseline logs/baseline_gpt61_sol_high --pool-seed 2610 --fit-seed 2611 \\
        --out-dir logs/child_vs_parent_2610
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402

from lares.eval.stats import t_interval, win_loss_tie  # noqa: E402
from lares.utils.hashing import sha256_file  # noqa: E402
from lares.utils.seeds import SEED_LEDGER, read_ledger, reserve_seeds, seed_conflicts  # noqa: E402

LANE_FITTABLE = "fittable"
FIT_KEYS = ("objective", "sampling", "budget_gradient_steps", "batch_size", "lr", "clip_grad_norm",
            "checkpoint_steps", "fixed_action_scale", "val_fraction", "split_seed")
DEMO_BUFFER = "logs/baseline_lock/demo_push-v2.pkl"


def rel(path: str) -> str:
    return os.path.relpath(os.path.abspath(path), _PROJECT_ROOT)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _records(run_dir: str) -> list:
    out = []
    for path in sorted(glob.glob(os.path.join(run_dir, "experiments", "*.json"))):
        with open(path, encoding="utf-8") as f:
            out.append(json.load(f))
    if not out:
        raise SystemExit(f"no experiment records under {run_dir}")
    return out


# ---------------------------------------------------------------------------
#  What is compared: read from the saved records, never re-derived
# ---------------------------------------------------------------------------


def discover_pairs(run_dir: str) -> list:
    """One entry per evaluated fittable-lane child: its source, its parent's source, its fitting."""
    pairs = []
    for r in _records(run_dir):
        lineage = r.get("lineage") or {}
        if lineage.get("lane") != LANE_FITTABLE or not r.get("fitting") or not r["code"].get("source_path"):
            continue
        reference = lineage["lane_metadata"]["reference"]
        parent_path = os.path.join(_PROJECT_ROOT, reference["source_path"])
        if sha256_file(parent_path) != reference["source_sha256"]:
            raise SystemExit(f"{r['experiment_id']}: parent {reference['source_path']} changed since the run")
        child_path = os.path.join(run_dir, r["code"]["source_path"])
        if sha256_file(child_path) != r["code"]["source_sha256"]:
            raise SystemExit(f"{r['experiment_id']}: child source changed since the run")
        pairs.append({
            "child_id": r["experiment_id"], "candidate_id": r["candidate_id"],
            "child_source_path": child_path, "child_source_sha256": r["code"]["source_sha256"],
            "child_initial_state": os.path.join(run_dir, r["fitting"]["initial_state"]["path"]),
            "parent_id": reference["structure_id"], "parent_source_path": parent_path,
            "parent_source_sha256": reference["source_sha256"],
            "scope": (lineage["lane_metadata"].get("mutation_scope") or {}).get("id"),
            "objective_config": r["fitting"]["objective_config"],
            "probe_manifest_id": r["fitting"]["probe_manifest_id"],
            "run_selected_step": r["fitting"]["selected_step"],
        })
    if not pairs:
        raise SystemExit(f"no evaluated fittable-lane children in {run_dir}")
    return pairs


def common_fitting(pairs: list) -> dict:
    """The fitting settings every child used; refuses a run whose children were fitted differently."""
    settings = {k: pairs[0]["objective_config"][k] for k in FIT_KEYS}
    for p in pairs[1:]:
        other = {k: p["objective_config"][k] for k in FIT_KEYS}
        if other != settings:
            raise SystemExit(f"{p['child_id']} was fitted differently: {other} vs {settings}")
    probes = {p["probe_manifest_id"] for p in pairs}
    if len(probes) != 1:
        raise SystemExit(f"children used different probe manifests: {probes}")
    return {**settings, "probe_manifest_id": probes.pop()}


def performance_incumbent(run_dir: str) -> str:
    with open(os.path.join(run_dir, "incumbents.json"), encoding="utf-8") as f:
        return json.load(f)["performance_incumbent"]["attempt_id"]


def baseline_best(baseline_dir: str) -> dict:
    """The baseline candidate its own run sent to the promotion test: its best screened challenger."""
    tested = [r for r in _records(baseline_dir)
              if ((r.get("evaluation") or {}).get("screening") or {}).get("promotion_tested")]
    if len(tested) != 1:
        raise SystemExit(f"expected one promotion-tested baseline candidate, found {[r['experiment_id'] for r in tested]}")
    r = tested[0]
    return {"id": r["experiment_id"], "source_path": os.path.join(baseline_dir, r["code"]["source_path"]),
            "state_path": os.path.join(baseline_dir, r["fitting"]["selected_state"]["path"]),
            "selected_step": r["fitting"]["selected_step"]}


# ---------------------------------------------------------------------------
#  Statistics: placement-level pairing
# ---------------------------------------------------------------------------


def placement_differences(per_case: dict, pairs: list) -> dict:
    """``{case_id: mean over pairs of child success - parent success}``.

    ``per_case`` maps a policy id to ``{case_id: success}``; every policy must cover the same cases.
    """
    cases = sorted(per_case[pairs[0]["child_id"]])
    for pid in {p["child_id"] for p in pairs} | {p["parent_id"] for p in pairs}:
        if sorted(per_case[pid]) != cases:
            raise ValueError(f"{pid} was not scored on the same cases")
    return {c: float(np.mean([float(per_case[p["child_id"]][c]) - float(per_case[p["parent_id"]][c])
                              for p in pairs])) for c in cases}


def paired(per_case: dict, a: str, b: str) -> dict:
    diffs = [float(per_case[a][c]) - float(per_case[b][c]) for c in sorted(per_case[b])]
    return {"a": a, "b": b, "n": len(diffs), **t_interval(diffs), **win_loss_tie(diffs)}


def summarise(per_case: dict, pairs: list, winner: str, baseline_id: str) -> dict:
    by_case = placement_differences(per_case, pairs)
    primary = t_interval(list(by_case.values()))
    success = {pid: float(np.mean([float(v) for v in cases.values()])) for pid, cases in per_case.items()}
    return {
        "paired_success_difference": primary["mean"],
        "primary": {"statistic": "mean over placements of the mean child-minus-parent success over pairs",
                    "n_placements": len(by_case), "n_pairs": len(pairs), **primary},
        "per_pair": [{"child": p["child_id"], "parent": p["parent_id"], "scope": p["scope"],
                      "child_success": success[p["child_id"]], "parent_success": success[p["parent_id"]],
                      **paired(per_case, p["child_id"], p["parent_id"])} for p in pairs],
        "mean_child_success": float(np.mean([success[p["child_id"]] for p in pairs])),
        "mean_parent_success": float(np.mean([success[p["parent_id"]] for p in pairs])),
        "secondary_r001": paired(per_case, winner, baseline_id) if baseline_id in per_case else None,
        "success_by_policy": success,
        "placement_differences": by_case,
    }


# ---------------------------------------------------------------------------
#  Seeds and manifest
# ---------------------------------------------------------------------------


def ensure_reserved(pool_seed: int, fit_seed: int, out_dir: str, ledger_path: str = SEED_LEDGER) -> str:
    """Reserve both seeds for this out-dir, or confirm they already are. Refuses any other use."""
    artifact = rel(out_dir)
    mine = {(int(e["seed"]), e["namespace"]) for e in read_ledger(ledger_path) if e.get("artifact") == artifact}
    wanted = {(pool_seed, "mt1_pool"), (fit_seed, "fitting")}
    if mine == wanted:
        return "already reserved for this out-dir"
    if mine:
        raise SystemExit(f"the ledger holds a partial reservation for {artifact}: {sorted(mine)}")
    conflicts = seed_conflicts([pool_seed], [fit_seed], ledger_path=ledger_path)
    if conflicts:
        raise SystemExit("seeds already used:\n  " + "\n  ".join(conflicts))
    stamp = now()
    reserve_seeds([
        {"seed": pool_seed, "namespace": "mt1_pool", "use": "R-005 child vs fitted parent fresh manifest",
         "artifact": artifact, "recorded_at": stamp, "backfilled": False},
        {"seed": fit_seed, "namespace": "fitting", "use": "R-005 child and parent refits",
         "artifact": artifact, "recorded_at": stamp, "backfilled": False},
    ], ledger_path=ledger_path)
    return "reserved"


def fresh_manifest(env_name: str, pool_seed: int, episodes: int, horizon: int, out_dir: str):
    from lares.eval.manifest import SPLIT_DEVELOPMENT, EvaluationManifest, assert_disjoint, build_manifest

    path = os.path.join(out_dir, f"{env_name}_confirmation_s{pool_seed}.yaml")
    manifest_id = f"{env_name}:development_confirmation:s{pool_seed}:n{episodes}"
    if os.path.exists(path):
        manifest = EvaluationManifest.load(path)
        if manifest.manifest_id != manifest_id:
            raise SystemExit(f"{rel(path)} holds {manifest.manifest_id}, not {manifest_id}")
        return manifest, path
    manifest, _ = build_manifest(env_name, SPLIT_DEVELOPMENT, [pool_seed], horizon=horizon,
                                 num_episodes=episodes, manifest_id=manifest_id)
    known = {}
    for pattern in ("config/manifests/*.yaml", "config/**/*confirmation*.yaml", "logs/**/*confirmation*.yaml"):
        for p in glob.glob(os.path.join(_PROJECT_ROOT, pattern), recursive=True):
            m = EvaluationManifest.load(p)
            known.setdefault(m.manifest_id, m)
    assert_disjoint(list(known.values()) + [manifest])
    manifest.save(path)
    return manifest, path


# ---------------------------------------------------------------------------
#  One job: fit (or not), then score on the fresh manifest
# ---------------------------------------------------------------------------


def run_job(job) -> dict:
    import torch

    torch.set_num_threads(1)
    from lares.core.obs_schema import get_obs_schema
    from lares.core.training_pipeline import DemoBuffer, ensure_mujoco_headless_gl
    from lares.eval import EvaluationManifest, make_manifest_env
    from lares.eval.runner import PolicyActor, evaluate_manifest
    from lares.fitting.checkpoints import RolloutScore
    from lares.fitting.optimizers import fit_policy_with_objective, prepare_data, state_parameters_sha256
    from lares.fitting.structure_bank import instantiate_source

    ensure_mujoco_headless_gl()
    started = time.time()
    schema = get_obs_schema(job["env_name"])
    fit_cfg, horizon = job["fitting"], int(job["horizon"])
    with open(job["source_path"], encoding="utf-8") as f:
        source = f.read()
    policy = instantiate_source(source, 39, 4, schema)
    policy.fixed_action_scale = float(fit_cfg["fixed_action_scale"])
    counters = {"probe_episodes": 0, "evaluation_episodes": 0, "bc_gradient_steps": 0}
    out = {"policy_id": job["policy_id"], "role": job["role"], "source_path": rel(job["source_path"]),
           "source_sha256": sha256_file(job["source_path"])}

    if job.get("check_state"):
        saved = torch.load(job["check_state"], weights_only=True)
        out["initial_state_matches_run"] = state_parameters_sha256(saved) == state_parameters_sha256(policy.state_dict())
        if not out["initial_state_matches_run"]:
            raise RuntimeError(f"{job['policy_id']}: source does not rebuild the run's initial state")
    if job.get("load_state"):
        policy.load_state_dict(torch.load(job["load_state"], weights_only=True))
        out["state_path"] = rel(job["load_state"])

    if job["fit"]:
        development = EvaluationManifest.load(job["development_manifest"])
        probe_manifest = development.tail(10)
        if probe_manifest.manifest_id != fit_cfg["probe_manifest_id"]:
            raise RuntimeError(f"probe manifest {probe_manifest.manifest_id} is not the run's {fit_cfg['probe_manifest_id']}")
        probe_env = make_manifest_env(probe_manifest, development.resolve_pool())

        def probe(pol):
            result = evaluate_manifest(PolicyActor(pol, deterministic=True, name="probe"), probe_manifest, probe_env,
                                       max_steps=horizon)
            counters["probe_episodes"] += len(result.episodes)
            progress = [e.signed_goal_progress for e in result.episodes if e.signed_goal_progress is not None]
            return RolloutScore(result.success_rate, float(np.mean(progress)) if progress else 0.0)

        data = prepare_data(DemoBuffer.load(job["demo_buffer"]), val_fraction=float(fit_cfg["val_fraction"]),
                            split_seed=int(fit_cfg["split_seed"]))
        fit = fit_policy_with_objective(
            policy, data, fit_cfg["objective"], budget=int(fit_cfg["budget_gradient_steps"]),
            batch_size=min(int(fit_cfg["batch_size"]), int(data.train_obs.shape[0])), lr=float(fit_cfg["lr"]),
            clip_grad_norm=float(fit_cfg["clip_grad_norm"]), seed=int(job["fit_seed"]), sampling=fit_cfg["sampling"],
            rollout_probe=probe, checkpoint_steps=[int(s) for s in fit_cfg["checkpoint_steps"]],
            structure_id=job["policy_id"],
        )
        policy.load_state_dict(fit.selected_state)
        counters["bc_gradient_steps"] = int(fit.bc_gradient_steps)
        state_path = os.path.join(job["out_dir"], "states", f"{job['policy_id']}.pt")
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        torch.save(fit.selected_state, state_path)
        out.update(selected_step=int(fit.selected_step), fitting_rejected=int(fit.selected_step) == 0,
                   checkpoint_history=fit.checkpoint_history, objective_config=fit.objective_config,
                   state_path=rel(state_path))

    manifest = EvaluationManifest.load(job["manifest_path"])
    env = make_manifest_env(manifest, manifest.resolve_pool())
    result = evaluate_manifest(PolicyActor(policy, deterministic=True, name=job["policy_id"]), manifest, env,
                               max_steps=horizon)
    counters["evaluation_episodes"] += len(result.episodes)
    labels = {}
    for e in result.episodes:
        labels[e.failure_label] = labels.get(e.failure_label, 0) + 1
    out.update(
        parameters_sha256=state_parameters_sha256(policy.state_dict()), manifest_id=manifest.manifest_id,
        success_rate=result.success_rate, mean_return=result.mean_reward, failure_labels=labels,
        per_case={e.case_id: {"success": bool(e.success), "return": e.episode_return,
                              "signed_goal_progress": e.signed_goal_progress} for e in result.episodes},
        resources=counters, wall_time_seconds=time.time() - started,
    )
    path = os.path.join(job["out_dir"], "policies", f"{job['policy_id']}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    return out


def build_jobs(pairs, fitting, baseline, args, manifest_path) -> list:
    common = {"env_name": args.env_name, "horizon": args.horizon, "fitting": fitting, "fit_seed": args.fit_seed,
              "out_dir": os.path.abspath(args.out_dir), "manifest_path": os.path.abspath(manifest_path),
              "demo_buffer": os.path.join(_PROJECT_ROOT, args.demo_buffer),
              "development_manifest": os.path.join(_PROJECT_ROOT, "config", "manifests", f"{args.env_name}_development.yaml")}
    jobs = [{**common, "policy_id": p["child_id"], "role": "child", "fit": True,
             "source_path": p["child_source_path"], "check_state": p["child_initial_state"]} for p in pairs]
    parents = {}
    for p in pairs:
        parents.setdefault(p["parent_id"], p["parent_source_path"])
    jobs += [{**common, "policy_id": pid, "role": "parent", "fit": True, "source_path": path}
             for pid, path in parents.items()]
    if baseline:
        jobs.append({**common, "policy_id": f"baseline:{baseline['id']}", "role": "baseline", "fit": False,
                     "source_path": baseline["source_path"], "load_state": baseline["state_path"]})
    return jobs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True, help="two-lane run directory")
    p.add_argument("--baseline", default="", help="zero-shot baseline run directory (R-001 secondary)")
    p.add_argument("--pool-seed", type=int, required=True)
    p.add_argument("--fit-seed", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--env-name", default="push-v2")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--horizon", type=int, default=150)
    p.add_argument("--demo-buffer", default=DEMO_BUFFER)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--reserve-only", action="store_true", help="check and reserve the seeds, then stop")
    args = p.parse_args(argv)

    pairs = discover_pairs(args.run)
    fitting = common_fitting(pairs)
    winner = performance_incumbent(args.run)
    if winner not in {q["child_id"] for q in pairs}:
        raise SystemExit(f"the two-lane winner {winner} is not a fittable-lane child")
    baseline = baseline_best(args.baseline) if args.baseline else None
    print(f"  {len(pairs)} children, {len({q['parent_id'] for q in pairs})} unique parents; fitting {fitting}")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"  seeds {args.pool_seed} (pool), {args.fit_seed} (fitting): "
          f"{ensure_reserved(args.pool_seed, args.fit_seed, args.out_dir)}")
    if args.reserve_only:
        return 0

    import multiprocessing as mp

    from lares.core.training_pipeline import ensure_mujoco_headless_gl

    ensure_mujoco_headless_gl()
    manifest, manifest_path = fresh_manifest(args.env_name, args.pool_seed, args.episodes, args.horizon, args.out_dir)
    print(f"  manifest {manifest.manifest_id}: {len(manifest)} cases -> {rel(manifest_path)}")
    demo_sha = sha256_file(os.path.join(_PROJECT_ROOT, args.demo_buffer))
    jobs = build_jobs(pairs, fitting, baseline, args, manifest_path)
    with open(os.path.join(args.out_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump({"started_at": now(), "argv": sys.argv, "pairs": [{k: v for k, v in q.items() if k != "objective_config"}
                   for q in pairs], "fitting": fitting, "fit_seed": args.fit_seed, "demo_buffer": args.demo_buffer,
                   "demo_buffer_sha256": demo_sha, "manifest_id": manifest.manifest_id,
                   "manifest_sha256": sha256_file(manifest_path), "two_lane_winner": winner, "baseline": baseline},
                  f, indent=2, default=str)

    results = {}
    with mp.get_context("spawn").Pool(processes=min(int(args.workers), len(jobs))) as pool:
        for done in pool.imap_unordered(run_job, jobs):
            results[done["policy_id"]] = done
            step = f"step {done['selected_step']}" if "selected_step" in done else "saved state"
            print(f"  {done['policy_id']:<36} {done['role']:<8} {step:<12} success {done['success_rate']:.2f}"
                  f"  ({done['wall_time_seconds']:.0f} s)", flush=True)

    per_case = {pid: {c: v["success"] for c, v in r["per_case"].items()} for pid, r in results.items()}
    baseline_id = f"baseline:{baseline['id']}" if baseline else ""
    summary = summarise(per_case, pairs, winner, baseline_id)
    summary.update(
        manifest_id=manifest.manifest_id, fit_seed=args.fit_seed, pool_seed=args.pool_seed, two_lane_winner=winner,
        baseline=baseline_id or None, selected_steps={pid: r.get("selected_step") for pid, r in results.items()},
        resources={k: sum(int(r["resources"][k]) for r in results.values())
                   for k in ("probe_episodes", "evaluation_episodes", "bc_gradient_steps")},
    )
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    pr = summary["primary"]
    print(f"\n  child - parent, placement-level over {pr['n_placements']} cases x {pr['n_pairs']} pairs: "
          f"{pr['mean']:+.3f} t95 [{pr['low']:+.3f}, {pr['high']:+.3f}]")
    for row in summary["per_pair"]:
        print(f"    {row['child']:<16} {row['child_success']:.2f}  vs {row['parent']:<32} {row['parent_success']:.2f}"
              f"  {row['mean']:+.3f}  w/l/t {row['wins']}/{row['losses']}/{row['ties']}")
    if summary["secondary_r001"]:
        s = summary["secondary_r001"]
        print(f"  R-001 {s['a']} - {s['b']}: {s['mean']:+.3f} t95 [{s['low']:+.3f}, {s['high']:+.3f}]")
    print(f"  resources {summary['resources']}\n  written: {rel(os.path.join(args.out_dir, 'summary.json'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
