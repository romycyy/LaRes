#!/usr/bin/env python
"""Phase 0: lock the evaluation contract and reproduce the baselines (``spec.md`` AC-0).

Builds the train / development / final-test manifests, then re-scores the
scripted expert, the saved incumbent and a deterministic smoke policy on the
development manifest.  Every check AC-0 lists is executed and reported, so the
exit status says whether the measurement contract actually holds.

Usage (from the project root)::

    python scripts/lock_baseline.py --build-manifests
    python scripts/lock_baseline.py                     # reuse committed manifests
    python scripts/lock_baseline.py --episodes 30
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from lares.core.symbolic_policy import SymbolicPolicy  # noqa: E402
from lares.core.training_pipeline import (  # noqa: E402
    DemoBuffer,
    ensure_mujoco_headless_gl,
    generate_dataset,
)
from lares.eval import (  # noqa: E402
    CHECKPOINT_FITTED,
    ConstantActor,
    EvaluationManifest,
    ExpertActor,
    PolicyActor,
    assert_disjoint,
    build_manifest,
    build_report,
    episode_table,
    evaluate_manifest,
    make_manifest_env,
    paired_difference,
)
from lares.eval.manifest import (  # noqa: E402
    SPLIT_DEVELOPMENT,
    SPLIT_FINAL_TEST,
    SPLIT_TRAIN,
)

MANIFEST_DIR = os.path.join(_PROJECT_ROOT, "config", "manifests")
DEFAULT_ENV = "push-v2"
DEFAULT_HORIZON = 150
DEFAULT_EPISODE_LENGTH = 200

#: Disjoint MT1 pool seeds per split. Each seed yields 50 placements.
#: Splitting by seed rather than by slicing one pool keeps provenance to a
#: single line and lets a split grow by adding a seed.
POOL_SEEDS = {
    SPLIT_TRAIN: (1000, 1001, 1002),
    SPLIT_DEVELOPMENT: (2000,),
    SPLIT_FINAL_TEST: (3000, 3001),
}
SPLIT_SIZES = {SPLIT_TRAIN: 150, SPLIT_DEVELOPMENT: 50, SPLIT_FINAL_TEST: 100}

INCUMBENT_CODE = os.path.join(_PROJECT_ROOT, "logs", "evolution", "best_policy_code.py")
INCUMBENT_WEIGHTS = os.path.join(_PROJECT_ROOT, "logs", "evolution", "best_policy.pt")


# ---------------------------------------------------------------------------
#  Reporting helpers
# ---------------------------------------------------------------------------


class Checks:
    """Records AC-0 pass/fail so the run reports a verdict rather than a log."""

    def __init__(self):
        self.rows = []

    def record(self, name, ok, detail=""):
        self.rows.append({"check": name, "ok": bool(ok), "detail": str(detail)})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
        return ok

    @property
    def failed(self):
        return [r for r in self.rows if not r["ok"]]


def separator(title):
    print(f"\n{'=' * 68}\n  {title}\n{'=' * 68}")


def _git(*args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=_PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def version_lock():
    """Everything that has to be pinned for a number to mean the same thing twice."""
    import importlib

    versions = {}
    for mod in ("torch", "numpy", "metaworld", "mujoco", "gymnasium", "gym", "openai"):
        try:
            from importlib.metadata import version as _v

            versions[mod] = _v(mod)
        except Exception:
            try:
                versions[mod] = str(
                    getattr(importlib.import_module(mod), "__version__", "unknown")
                )
            except Exception:
                versions[mod] = "missing"
    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty_files": len([ln for ln in _git("status", "--porcelain").splitlines() if ln]),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "mujoco_gl": os.environ.get("MUJOCO_GL", ""),
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }


# ---------------------------------------------------------------------------
#  Manifests
# ---------------------------------------------------------------------------


def build_all_manifests(env_id, horizon, out_dir):
    """Materialise the three splits and prove they share no placement."""
    manifests, pools = {}, {}
    for split, seeds in POOL_SEEDS.items():
        manifest, pool = build_manifest(
            env_id,
            split,
            seeds,
            horizon=horizon,
            num_episodes=SPLIT_SIZES[split],
        )
        manifests[split] = manifest
        pools[split] = pool
        path = os.path.join(out_dir, f"{env_id}_{split}.yaml")
        manifest.save(path)
        print(f"  {split:12s} {len(manifest):4d} cases  seeds={list(seeds)}  -> {path}")
    assert_disjoint(list(manifests.values()))
    return manifests, pools


def load_all_manifests(env_id, manifest_dir):
    manifests, pools = {}, {}
    for split in POOL_SEEDS:
        path = os.path.join(manifest_dir, f"{env_id}_{split}.yaml")
        manifest = EvaluationManifest.load(path)
        manifests[split] = manifest
        pools[split] = manifest.resolve_pool()
        print(f"  {split:12s} {len(manifest):4d} cases  {manifest.manifest_id}")
    return manifests, pools


# ---------------------------------------------------------------------------
#  Baseline actors
# ---------------------------------------------------------------------------


def load_incumbent(obs_dim, action_dim):
    """Load the saved evolution winner: its source plus its fitted weights."""
    if not (os.path.isfile(INCUMBENT_CODE) and os.path.isfile(INCUMBENT_WEIGHTS)):
        return None, "no saved incumbent under logs/evolution/"
    src = open(INCUMBENT_CODE, "r", encoding="utf-8").read()
    ns = {"torch": torch, "nn": nn, "np": np, "SymbolicPolicy": SymbolicPolicy}
    exec(src, ns)
    policy = ns["GeneratedPolicy"](obs_dim, action_dim)
    policy.load_state_dict(torch.load(INCUMBENT_WEIGHTS, map_location="cpu"))
    policy.validate()
    return policy, ""


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env-name", default=DEFAULT_ENV)
    p.add_argument("--build-manifests", action="store_true",
                   help="Regenerate and overwrite the committed manifests.")
    p.add_argument("--manifest-dir", default=MANIFEST_DIR)
    p.add_argument("--episodes", type=int, default=30,
                   help="Development episodes used for the baseline reproduction.")
    p.add_argument("--max-steps", type=int, default=DEFAULT_HORIZON)
    p.add_argument("--out-dir", default=os.path.join(_PROJECT_ROOT, "logs", "baseline_lock"))
    p.add_argument("--collect-demos", type=int, default=0,
                   help="Collect this many expert episodes from the train manifest into a "
                        "demo buffer with episode ids, so BC can split by trajectory.")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_mujoco_headless_gl()
    os.makedirs(args.out_dir, exist_ok=True)
    checks = Checks()
    report = {"version_lock": version_lock(), "args": vars(args)}

    separator("Version lock")
    for k, v in report["version_lock"].items():
        print(f"  {k:16s} {v}")
    checks.record(
        "AC-0.1 environment, seeds and versions recorded",
        report["version_lock"]["git_commit"] != "unknown",
    )

    separator("Manifests")
    if args.build_manifests:
        manifests, pools = build_all_manifests(args.env_name, args.max_steps, args.manifest_dir)
    else:
        manifests, pools = load_all_manifests(args.env_name, args.manifest_dir)
    try:
        assert_disjoint(list(manifests.values()))
        checks.record("AC-0.6 train / development / final-test are non-overlapping", True)
    except ValueError as exc:
        checks.record("AC-0.6 train / development / final-test are non-overlapping", False, str(exc))
    report["manifests"] = {
        s: {"manifest_id": m.manifest_id, "num_episodes": len(m), "pool_seeds": list(m.pool.seeds)}
        for s, m in manifests.items()
    }

    dev_full = manifests[SPLIT_DEVELOPMENT]
    dev = dev_full.head(min(args.episodes, len(dev_full)))
    train = manifests[SPLIT_TRAIN]

    # Separate env per stream. Never share: that is the property under test.
    dev_env = make_manifest_env(dev_full, pools[SPLIT_DEVELOPMENT], DEFAULT_EPISODE_LENGTH)
    probe_env = make_manifest_env(dev_full, pools[SPLIT_DEVELOPMENT], DEFAULT_EPISODE_LENGTH)
    collect_env = make_manifest_env(train, pools[SPLIT_TRAIN], DEFAULT_EPISODE_LENGTH)

    obs_dim = int(np.prod(dev_env.observation_space.shape))
    action_dim = int(np.prod(dev_env.action_space.shape))

    separator(f"Baseline reproduction on {dev.manifest_id}")
    actors = [ExpertActor(args.env_name), ConstantActor(np.zeros(action_dim), name="zero_action")]
    incumbent, why = load_incumbent(obs_dim, action_dim)
    if incumbent is not None:
        actors.append(PolicyActor(incumbent, deterministic=True, name="saved_incumbent"))
    else:
        print(f"  NOTE: incumbent not evaluated ({why})")

    results, timings = {}, {}
    for actor in actors:
        t0 = time.time()
        res = evaluate_manifest(actor, dev, dev_env, max_steps=args.max_steps)
        timings[actor.name] = time.time() - t0
        results[actor.name] = res
        s = res.summary()
        print(
            f"  {actor.name:18s} reward={s['mean_reward']:9.2f}+-{s['reward_se']:6.2f}  "
            f"success={s['success_rate']:.3f}+-{s['success_se']:.3f}  "
            f"obj_to_target={s['mean_terminal_obj_to_target']:.4f}"
        )
    report["baselines"] = {k: v.to_dict() for k, v in results.items()}
    report["wall_time_seconds"] = timings

    # A structured report per baseline: what the incumbent actually does wrong is
    # the finding, and a success rate of zero does not carry it.
    for name, res in results.items():
        rep = build_report(res, candidate_id=name, checkpoint=CHECKPOINT_FITTED,
                           action_dim=action_dim)
        rep.save(os.path.join(args.out_dir, f"report_{name.replace(':', '_')}.json"))
        if rep.rollout.failure_labels:
            print(f"  {name:18s} failure labels: {rep.rollout.failure_labels}")
        if rep.rollout.signed_goal_progress:
            print(f"  {'':18s} goal progress:  "
                  f"mean={rep.rollout.signed_goal_progress['mean']:+.4f}  "
                  f"lateral drift={rep.rollout.lateral_drift['mean']:.4f}")
    incumbent_report = None
    if "saved_incumbent" in results:
        incumbent_report = build_report(
            results["saved_incumbent"], "saved_incumbent", CHECKPOINT_FITTED,
            action_dim=action_dim,
        )
        print("\n" + episode_table(incumbent_report, limit=8))

    checks.record(
        "AC-0.7 scripted expert reproduces (success >= 0.9)",
        results["expert:" + args.env_name].success_rate >= 0.9,
        f"success={results['expert:' + args.env_name].success_rate:.3f}",
    )

    separator("Reproducibility")
    rerun = {}
    for name, res in results.items():
        actor = next(a for a in actors if a.name == name)
        again = evaluate_manifest(actor, dev, dev_env, max_steps=args.max_steps)
        identical = all(
            a.episode_return == b.episode_return and a.success == b.success
            for a, b in zip(res.episodes, again.episodes)
        )
        rerun[name] = identical
        checks.record(f"AC-0.3 {name} rerun is bit-identical", identical)
    report["rerun_identical"] = rerun

    # AC-0.4 / AC-0.5: doing other work between two evaluations must not move a case.
    baseline = results[actors[0].name]
    collect_cases = train.episodes[:3]
    generate_dataset(collect_env, args.env_name, collect_cases, max_steps=20)
    _ = evaluate_manifest(
        ConstantActor(np.ones(action_dim), name="probe"), dev.head(3), probe_env,
        max_steps=args.max_steps,
    )
    after = evaluate_manifest(actors[0], dev, dev_env, max_steps=args.max_steps)
    unshifted = all(
        a.episode_return == b.episode_return and a.task_id == b.task_id
        for a, b in zip(baseline.episodes, after.episodes)
    )
    checks.record(
        "AC-0.4/0.5 collection and debug rollouts do not shift evaluation cases", unshifted
    )
    report["cases_unshifted_after_other_work"] = unshifted

    if incumbent is not None:
        separator("Paired comparison")
        for metric in ("success", "reward"):
            diff = paired_difference(
                results["saved_incumbent"], results["expert:" + args.env_name], metric
            )
            print(
                f"  incumbent minus expert, {metric:7s}: "
                f"{diff['mean_difference']:+.3f}  "
                f"95% CI [{diff['ci95_low']:+.3f}, {diff['ci95_high']:+.3f}]  "
                f"n={diff['n_paired']}"
            )
            report.setdefault("paired", {})[metric] = diff

    if args.collect_demos > 0:
        separator("Expert demonstrations")
        cases = train.episodes[: args.collect_demos]
        buffer, demo_stats = generate_dataset(
            collect_env, args.env_name, cases, max_steps=args.max_steps
        )
        demo_path = os.path.join(args.out_dir, f"demo_{args.env_name}.pkl")
        buffer.save(demo_path)
        actions = np.asarray(buffer.actions)
        endpoint = {
            "any_axis_exactly_at_endpoint": float((np.abs(actions) >= 1.0).any(axis=1).mean()),
            "any_axis_above_0.99": float((np.abs(actions) > 0.99).any(axis=1).mean()),
            "per_axis_exactly_at_endpoint": [
                float(x) for x in (np.abs(actions) >= 1.0).mean(axis=0)
            ],
        }
        report["demonstrations"] = {
            **demo_stats,
            "path": demo_path,
            "episodes": len(cases),
            "endpoint_statistics": endpoint,
        }
        print(f"  {len(buffer)} transitions over {len(cases)} training placements -> {demo_path}")
        print(f"  actions exactly at an endpoint: "
              f"{endpoint['any_axis_exactly_at_endpoint']:.4f} of transitions")

    out_path = os.path.join(args.out_dir, f"baseline_lock_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    separator("Verdict")
    report["checks"] = checks.rows
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  {len(checks.rows) - len(checks.failed)}/{len(checks.rows)} AC-0 checks passed")
    print(f"  report: {out_path}")

    dev_env.close()
    probe_env.close()
    collect_env.close()
    sys.exit(1 if checks.failed else 0)


if __name__ == "__main__":
    main()
