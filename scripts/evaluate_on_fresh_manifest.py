#!/usr/bin/env python
"""Score saved policies, paired, on a fresh development-confirmation manifest.

Every committed split has been looked at: development screens and expands
candidates, its tail probes checkpoints, and the final-test split is locked
until Phase 7. A number reported on cases a search selected on is optimistic.
This builds a manifest from an MT1 pool seed that no committed manifest uses,
checks it is disjoint by placement hash from train, development and final-test,
and scores each named policy on it in the same case order.

The manifest's split is ``development``: it is never the final test. Once a seed
has been used here it has been looked at too, so record it and pick a new one for
the next confirmation (``debug_spec.md`` Stage E).

Usage (from the project root)::

    python scripts/evaluate_on_fresh_manifest.py --pool-seed 2100 --out-dir <dir> \\
        --policy new=logs/run/gen_2/candidates/x/policy_source.py:logs/run/gen_2/candidates/x/selected_state.pt \\
        --policy old=logs/old/best_policy_code.py:logs/old/best_policy.pt \\
        --expert --reference old
"""

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
import torch  # noqa: E402

from lares.core.obs_schema import get_obs_schema  # noqa: E402
from lares.core.training_pipeline import ensure_mujoco_headless_gl  # noqa: E402
from lares.eval import EvaluationManifest, ExpertActor, evaluate_manifest, make_manifest_env  # noqa: E402
from lares.eval.manifest import SPLIT_DEVELOPMENT, assert_disjoint, build_manifest  # noqa: E402
from lares.eval.runner import PolicyActor  # noqa: E402
from lares.fitting.structure_bank import instantiate_source  # noqa: E402
from lares.eval.stats import t_bounds as t_interval  # noqa: E402
from lares.search.generation import env_identity  # noqa: E402
from lares.utils.hashing import sha256_file  # noqa: E402


def load_policy(code_path, state_path, env_id, obs_dim=39, action_dim=4):
    with open(code_path, "r", encoding="utf-8") as f:
        source = f.read()
    policy = instantiate_source(source, obs_dim, action_dim, get_obs_schema(env_id))
    policy.load_state_dict(torch.load(state_path, weights_only=True))
    policy.eval()
    return policy


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env-name", default="push-v2")
    p.add_argument("--pool-seed", type=int, required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--horizon", type=int, default=150)
    p.add_argument("--policy", action="append", default=[], help="label=code_path:state_path (repeatable)")
    p.add_argument("--expert", action="store_true", help="Also score the scripted expert as a reference.")
    p.add_argument("--reference", default="", help="Label every other policy is paired against.")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--frozen", default="",
                   help="a manifest frozen before the runs (R-037); the rebuilt manifest must match it case for case")
    args = p.parse_args()

    ensure_mujoco_headless_gl()
    committed = [EvaluationManifest.load(path) for path in sorted(
        glob.glob(os.path.join(_PROJECT_ROOT, "config", "manifests", f"{args.env_name}_*.yaml")))]
    used_seeds = {int(s) for m in committed for s in m.pool.seeds}
    if args.pool_seed in used_seeds:
        raise SystemExit(f"pool seed {args.pool_seed} is used by a committed manifest ({sorted(used_seeds)})")
    manifest, pool = build_manifest(
        args.env_name, SPLIT_DEVELOPMENT, [args.pool_seed], horizon=args.horizon, num_episodes=args.episodes,
        manifest_id=f"{args.env_name}:development_confirmation:s{args.pool_seed}:n{args.episodes}",
    )
    assert_disjoint(committed + [manifest])
    if args.frozen:
        frozen = EvaluationManifest.load(args.frozen)
        if (frozen.manifest_id, [c.to_dict() for c in frozen.episodes]) != (
                manifest.manifest_id, [c.to_dict() for c in manifest.episodes]):
            raise SystemExit(f"the rebuilt manifest differs from the frozen {args.frozen}; refusing to score")
        print(f"  matches the frozen manifest {args.frozen}")
    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = manifest.save(os.path.join(args.out_dir, f"{args.env_name}_confirmation_s{args.pool_seed}.yaml"))
    print(f"  manifest {manifest.manifest_id}: {len(manifest)} cases, disjoint from {[m.manifest_id for m in committed]}")

    actors = []
    for spec in args.policy:
        label, paths = spec.split("=", 1)
        code_path, state_path = paths.split(":", 1)
        actors.append((label, PolicyActor(load_policy(code_path, state_path, args.env_name), deterministic=True, name=label),
                       {"code": code_path, "code_sha256": sha256_file(code_path),
                        "state": state_path, "state_sha256": sha256_file(state_path)}))
    if args.expert:
        actors.append(("expert", ExpertActor(args.env_name), {}))

    results = {}
    for label, actor, provenance in actors:
        env = make_manifest_env(manifest, pool)
        started = time.time()
        result = evaluate_manifest(actor, manifest, env, max_steps=args.horizon)
        env.close() if hasattr(env, "close") else None
        progress = [e.signed_goal_progress for e in result.episodes if e.signed_goal_progress is not None]
        labels = {}
        for e in result.episodes:
            labels[e.failure_label] = labels.get(e.failure_label, 0) + 1
        results[label] = {
            "provenance": provenance,
            "success_rate": result.success_rate,
            "success_se": result.success_se,
            "mean_return": result.mean_reward,
            "mean_signed_goal_progress": float(np.mean(progress)) if progress else None,
            "failure_labels": labels,
            "wall_time_seconds": time.time() - started,
            "per_case": {e.case_id: {"success": e.success, "return": e.episode_return,
                                     "signed_goal_progress": e.signed_goal_progress} for e in result.episodes},
        }
        print(f"  {label:<34} success {result.success_rate:.3f} +- {result.success_se:.3f}   "
              f"return {result.mean_reward:7.1f}   progress {results[label]['mean_signed_goal_progress'] or 0:+.4f}   {labels}")

    comparisons = {}
    if args.reference:
        ref = results[args.reference]["per_case"]
        for label, data in results.items():
            if label == args.reference:
                continue
            diffs = [data["per_case"][c]["success"] - ref[c]["success"] for c in ref]
            prog = [data["per_case"][c]["signed_goal_progress"] - ref[c]["signed_goal_progress"] for c in ref
                    if data["per_case"][c]["signed_goal_progress"] is not None and ref[c]["signed_goal_progress"] is not None]
            comparisons[label] = {
                "n": len(diffs),
                "success_mean_difference": float(np.mean(diffs)),
                "success_t95": t_interval(diffs),
                "wins": sum(d > 0 for d in diffs), "losses": sum(d < 0 for d in diffs), "ties": sum(d == 0 for d in diffs),
                "progress_mean_difference": float(np.mean(prog)) if prog else None,
                "progress_t95": t_interval(prog),
            }
            c = comparisons[label]
            ci = c["success_t95"]
            print(f"  {label} - {args.reference}: success {c['success_mean_difference']:+.3f} "
                  f"t95 [{ci[0]:+.3f}, {ci[1]:+.3f}]  w/l/t {c['wins']}/{c['losses']}/{c['ties']}   "
                  f"progress {c['progress_mean_difference'] or 0:+.4f}")

    out = os.path.join(args.out_dir, f"confirmation_s{args.pool_seed}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"manifest_id": manifest.manifest_id, "manifest_path": manifest_path,
                   "environment": env_identity(args.env_name), "pool_seed": args.pool_seed,
                   "committed_seeds_excluded": sorted(used_seeds), "reference": args.reference,
                   "results": results, "paired_vs_reference": comparisons}, f, indent=2, default=str)
    print(f"\n  written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
