#!/usr/bin/env python
"""Visual failure analysis over a candidate's failed episodes (FR-11, AC-2).

Records media for the worst failed development episodes under the fixed
selection rule, optionally asks a vision model about them, merges whatever comes
back with the simulator's own measurements, and writes the block a repair prompt
would receive.

With no model this still does useful work: it captures and saves the selected
frames, reports which moments the episode could not supply, and produces the
numeric-diagnostics-only side of E10 through the identical code path.

The final-test split is refused outright. Final-test media must never return to
the search loop.

Usage (from the project root)::

    python scripts/analyse_failures.py --structures simple_standoff
    python scripts/analyse_failures.py --structures simple_standoff --model Qwen/...
"""

import argparse
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from lares.core.obs_schema import get_obs_schema  # noqa: E402
from lares.core.training_pipeline import DemoBuffer, ensure_mujoco_headless_gl  # noqa: E402
from lares.eval import EvaluationManifest, make_manifest_env  # noqa: E402
from lares.eval.runner import PolicyActor, evaluate_manifest  # noqa: E402
from lares.fitting import DETERMINISTIC_MSE, load_bank, prepare_data  # noqa: E402
from lares.fitting.optimizers import fit_with_objective  # noqa: E402
from lares.vision import (  # noqa: E402
    QwenAnalyst,
    ScriptedAnalyst,
    analyse_failures,
    evidence_block,
)

DEFAULT_BUFFER = os.path.join(_PROJECT_ROOT, "logs", "baseline_lock", "demo_push-v2.pkl")
MANIFESTS = os.path.join(_PROJECT_ROOT, "config", "manifests")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env-name", default="push-v2")
    p.add_argument("--buffer", default=DEFAULT_BUFFER)
    p.add_argument("--structures", nargs="+", default=["simple_standoff"])
    p.add_argument("--budget", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--episodes-to-analyse", type=int, default=3)
    p.add_argument("--model", default="",
                   help="Qwen checkpoint; without one no model is called and only "
                        "the media and the numbers are produced")
    p.add_argument("--quantization", default="fp16")
    p.add_argument("--out-dir", default=os.path.join(_PROJECT_ROOT, "logs", "vision"))
    return p.parse_args()


def separator(title):
    print(f"\n{'=' * 80}\n  {title}\n{'=' * 80}")


def main():
    args = parse_args()
    ensure_mujoco_headless_gl()
    schema = get_obs_schema(args.env_name)

    if not os.path.isfile(args.buffer):
        print(f"missing demo buffer: {args.buffer}")
        return 1
    data = prepare_data(DemoBuffer.load(args.buffer), split_seed=0)

    full = EvaluationManifest.load(
        os.path.join(MANIFESTS, f"{args.env_name}_development.yaml")
    )
    manifest = full.head(min(args.eval_episodes, len(full)))
    env = make_manifest_env(full, full.resolve_pool(), 200)

    analyst = (
        QwenAnalyst(model_id=args.model, quantization=args.quantization)
        if args.model else ScriptedAnalyst([])
    )

    bank = {s.structure_id: s for s in load_bank(env_id=args.env_name)}
    for structure_id in args.structures:
        structure = bank[structure_id]
        separator(f"{structure_id}")
        fitted = fit_with_objective(
            structure, data, DETERMINISTIC_MSE, budget=args.budget,
            batch_size=256, seed=args.seed, schema=schema,
        )
        policy = structure.build(schema)
        policy.load_state_dict(fitted.final_state)
        actor = PolicyActor(policy, deterministic=True, name=structure_id)
        result = evaluate_manifest(actor, manifest, env, max_steps=args.max_steps)
        print(f"  success {result.success_rate:.3f} over {len(result.episodes)} episodes")

        media_dir = os.path.join(args.out_dir, structure_id, "media")
        analyses, merged, run = analyse_failures(
            actor, env, manifest, result, analyst, schema,
            candidate_id=structure_id, max_steps=args.max_steps,
            limit=args.episodes_to_analyse, media_dir=media_dir,
        )
        print(f"  {run.episodes_analysed} episode(s) analysed, "
              f"{run.frames_shown} frames selected, "
              f"{run.analyses_returned} analyses returned, "
              f"{run.analyses_rejected} rejected, {run.conflicts} conflicts")
        for note in run.unavailable:
            print(f"    unavailable: {note}")
        for reason in run.rejection_reasons:
            print(f"    rejected: {reason}")

        out = os.path.join(args.out_dir, structure_id)
        os.makedirs(out, exist_ok=True)
        for analysis in analyses:
            analysis.save(os.path.join(out, "analyses"))
        with open(os.path.join(out, "run.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"run": run.to_dict(), "merged": [m.to_dict() for m in merged]},
                f, indent=2, default=str,
            )
        block = evidence_block(merged, analyses)
        with open(os.path.join(out, "evidence.md"), "w", encoding="utf-8") as f:
            f.write(block)
        print()
        print(block)
        print(f"\n  written: {out}")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
