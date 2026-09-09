#!/usr/bin/env python
"""Phase 3: compare fitting methods on frozen structures (``spec.md`` AC-3).

Holds the controller structure fixed, gives every method the same
objective-evaluation budget, scores each fitted result on the development
manifest under both checkpoint rules, and applies the selection rule declared in
``lares/fitting/benchmark.py`` before the comparison runs.

Usage (from the project root)::

    python scripts/benchmark_fitting.py                     # full bank
    python scripts/benchmark_fitting.py --family simple     # complexity ablation
    python scripts/benchmark_fitting.py --budget 500 --no-rollouts   # quick pass
"""

import argparse
import json
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from lares.core.obs_schema import get_obs_schema  # noqa: E402
from lares.core.training_pipeline import DemoBuffer, ensure_mujoco_headless_gl  # noqa: E402
from lares.eval import EvaluationManifest, make_manifest_env  # noqa: E402
from lares.fitting import METHODS, load_bank, prepare_data, summarise_bank  # noqa: E402
from lares.fitting.benchmark import (  # noqa: E402
    BenchmarkConfig,
    aggregate,
    format_paired,
    format_table,
    paired_method_comparison,
    run_benchmark,
    select_default,
)

DEFAULT_BUFFER = os.path.join(_PROJECT_ROOT, "logs", "baseline_lock", "demo_push-v2.pkl")
DEFAULT_MANIFEST = os.path.join(
    _PROJECT_ROOT, "config", "manifests", "push-v2_development.yaml"
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env-name", default="push-v2")
    p.add_argument("--buffer", default=DEFAULT_BUFFER)
    p.add_argument("--manifest", default=DEFAULT_MANIFEST)
    p.add_argument("--out-dir", default=os.path.join(_PROJECT_ROOT, "logs", "fitting_benchmark"))
    p.add_argument("--budget", type=int, default=2000,
                   help="Objective evaluations per method. 2000 reproduces the live BC path.")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--methods", nargs="+", default=list(METHODS))
    p.add_argument("--family", choices=["simple", "complex"], default=None,
                   help="Restrict the bank, for the parameter-count ablation (E3).")
    p.add_argument("--structures", nargs="+", default=None)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--eval-max-steps", type=int, default=150)
    p.add_argument("--no-rollouts", action="store_true",
                   help="Skip development rollouts; compare on loss only.")
    return p.parse_args()


def separator(title):
    print(f"\n{'=' * 78}\n  {title}\n{'=' * 78}")


def main():
    args = parse_args()
    ensure_mujoco_headless_gl()
    os.makedirs(args.out_dir, exist_ok=True)
    schema = get_obs_schema(args.env_name)

    separator("Structure bank")
    structures = load_bank(env_id=args.env_name, family=args.family)
    if args.structures:
        wanted = set(args.structures)
        structures = [s for s in structures if s.structure_id in wanted]
    if not structures:
        print("  no structures selected")
        return 1
    print(summarise_bank(structures))

    separator("Fitting data")
    if not os.path.isfile(args.buffer):
        print(f"  missing demo buffer: {args.buffer}")
        print("  build one with: python scripts/lock_baseline.py --collect-demos 150")
        return 1
    buffer = DemoBuffer.load(args.buffer)
    data = prepare_data(buffer, split_seed=0)
    print(f"  {len(buffer)} transitions; split {data.split_info}")
    if not data.has_validation:
        print("  WARNING: no validation split, so the best-validation checkpoint rule "
              "falls back to the final parameters.")

    dev_manifest = dev_env = None
    if not args.no_rollouts:
        separator("Development manifest")
        full = EvaluationManifest.load(args.manifest)
        dev_manifest = full.head(min(args.eval_episodes, len(full)))
        dev_env = make_manifest_env(full, full.resolve_pool(), 200)
        print(f"  {dev_manifest.manifest_id} ({len(dev_manifest)} cases)")

    config = BenchmarkConfig(
        env_name=args.env_name,
        methods=tuple(args.methods),
        budget=args.budget,
        batch_size=args.batch_size,
        seeds=tuple(args.seeds),
        eval_episodes=args.eval_episodes,
        eval_max_steps=args.eval_max_steps,
        val_every=max(1, args.budget // 20),
    )

    separator("Matched-budget comparison")
    print(f"  {len(structures)} structures x {len(config.methods)} methods x "
          f"{len(config.seeds)} seeds, budget {config.budget} objective evaluations")
    t0 = time.time()

    def progress(done, total, structure_id, method, seed):
        print(f"  [{done:3d}/{total}] {structure_id:<20} {method:<24} seed={seed} "
              f"({time.time() - t0:.0f}s elapsed)")

    payload = run_benchmark(
        structures, data, schema, config,
        dev_manifest=dev_manifest, dev_env=dev_env, progress=progress,
    )

    separator("Results by method and checkpoint rule")
    by_method = aggregate(payload["rows"], by=("method", "checkpoint_rule"))
    print(format_table(by_method))

    separator("Results by parameter family (spec.md E3)")
    by_family = aggregate(payload["rows"], by=("family", "method"))
    print(format_table(by_family, columns=[
        ("family", "family", "<12", None),
        ("method", "method", "<24", None),
        ("n", "n", ">4", None),
        ("train_loss", "train", ">9", 5),
        ("validation_loss", "val", ">9", 5),
        ("success_rate", "success", ">9", 3),
        ("signed_goal_progress", "progress", ">10", 4),
        ("wall_time_seconds", "seconds", ">9", 2),
    ]))

    separator("Paired against the reproduced baseline, by structure")
    paired = paired_method_comparison(payload["rows"], reference="adam_baseline")
    payload["paired_vs_baseline"] = paired
    print("  development success")
    print("  " + format_paired(paired, "success_rate").replace("\n", "\n  "))
    print("\n  validation loss")
    print("  " + format_paired(paired, "validation_loss").replace("\n", "\n  "))

    separator("Does a lower imitation loss mean a better controller?")
    lvs = payload["loss_versus_success"]
    if "unavailable" in lvs:
        print(f"  unavailable: {lvs['unavailable']}")
    else:
        print(f"  n={lvs['n']}  Pearson r={lvs['pearson_r']:+.3f} (p={lvs['pearson_p']:.4f})"
              f"  Spearman r={lvs['spearman_r']:+.3f} (p={lvs['spearman_p']:.4f})")
        print("  A negative correlation is the expected sign: lower loss, higher success.")

    separator("Selection")
    selection = select_default(by_method)
    payload["aggregate_by_method"] = by_method
    payload["aggregate_by_family"] = by_family
    payload["selection"] = selection
    for key, value in selection["rule"].items():
        print(f"  {key:<14} {value}")
    print(f"\n  selected: {selection['selected']}")
    print(f"  because:  {selection['reason']}")

    dead = {
        sid: info["dead_parameters"]
        for sid, info in payload["sensitivity"].items()
        if info["dead_parameters"]
    }
    if dead:
        separator("Parameters the executed action does not respond to")
        for sid, names in sorted(dead.items()):
            print(f"  {sid:<20} {names}")

    out_path = os.path.join(
        args.out_dir, f"fitting_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n  report: {out_path}")
    print(f"  total wall time: {payload['wall_time_seconds']:.1f}s")

    if dev_env is not None:
        dev_env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
