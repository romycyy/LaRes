#!/usr/bin/env python
"""Phase 5 comparisons: objectives and data coverage (``spec.md`` E7, E8, AC-5).

Three questions, all at matched fitting budgets on the frozen structure bank:

E7  Does learning a distribution beat the deterministic regression baseline?
    Each objective is scored under *its own* execution contract, and separately
    under the deterministic contract, because scoring a censored model with a
    tanh sampler would measure neither.

E8  Does labelling the states the learner actually visits beat a fixed expert
    buffer? Expert queries and environment steps are reported, since an
    aggregation round is not free.

    Alongside it, uniform against phase-balanced minibatches. The expert spends
    75% of its steps in the push phase, so uniform sampling gives the descend and
    hover phases few gradients however much they matter.

Usage (from the project root)::

    python scripts/compare_objectives.py
    python scripts/compare_objectives.py --budget 500 --family simple
    python scripts/compare_objectives.py --skip-dagger
"""

import argparse
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
from lares.core.training_pipeline import DemoBuffer, ensure_mujoco_headless_gl  # noqa: E402
from lares.eval import EvaluationManifest, make_manifest_env  # noqa: E402
from lares.eval.manifest import SPLIT_TRAIN  # noqa: E402
from lares.eval.report import CHECKPOINT_FITTED, build_report  # noqa: E402
from lares.eval.runner import PolicyActor, evaluate_manifest  # noqa: E402
from lares.fitting import (  # noqa: E402
    DETERMINISTIC_MSE,
    EXEC_DETERMINISTIC_TANH,
    OBJECTIVES,
    expert_phase_labels,
    load_bank,
    phase_fractions,
    prepare_data,
    spec,
)
from lares.fitting.optimizers import (  # noqa: E402
    SAMPLING_PHASE_BALANCED,
    SAMPLING_UNIFORM,
    fit_with_objective,
)

DEFAULT_BUFFER = os.path.join(_PROJECT_ROOT, "logs", "baseline_lock", "demo_push-v2.pkl")
MANIFESTS = os.path.join(_PROJECT_ROOT, "config", "manifests")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env-name", default="push-v2")
    p.add_argument("--buffer", default=DEFAULT_BUFFER)
    p.add_argument("--budget", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--family", choices=["simple", "complex"], default="simple")
    p.add_argument("--structures", nargs="+", default=None)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--eval-max-steps", type=int, default=150)
    p.add_argument("--dagger-episodes", type=int, default=20)
    p.add_argument("--skip-dagger", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out-dir", default=os.path.join(_PROJECT_ROOT, "logs", "objective_comparison")
    )
    return p.parse_args()


def separator(title):
    print(f"\n{'=' * 84}\n  {title}\n{'=' * 84}")


def score(policy, structure, dev_manifest, dev_env, execution, max_steps, label):
    """Roll out under one execution contract and report the outcome."""
    deterministic = execution == EXEC_DETERMINISTIC_TANH
    actor = PolicyActor(policy, deterministic=deterministic, name=label)
    result = evaluate_manifest(actor, dev_manifest, dev_env, max_steps=max_steps)
    report = build_report(result, label, CHECKPOINT_FITTED, action_dim=structure.action_dim)
    progress = report.rollout.signed_goal_progress
    return {
        "execution": execution,
        "action_mode": result.action_mode,
        "success_rate": report.rollout.success_rate,
        "mean_return": report.rollout.mean_return,
        "signed_goal_progress": progress["mean"] if progress else None,
        "failure_labels": report.rollout.failure_labels,
    }


def main():
    args = parse_args()
    ensure_mujoco_headless_gl()
    os.makedirs(args.out_dir, exist_ok=True)
    schema = get_obs_schema(args.env_name)

    structures = load_bank(env_id=args.env_name, family=args.family)
    if args.structures:
        wanted = set(args.structures)
        structures = [s for s in structures if s.structure_id in wanted]
    if not structures:
        print("no structures selected")
        return 1

    if not os.path.isfile(args.buffer):
        print(f"missing demo buffer: {args.buffer}")
        print("build one with: python scripts/lock_baseline.py --collect-demos 150")
        return 1
    buffer = DemoBuffer.load(args.buffer)
    data = prepare_data(buffer, split_seed=0)
    train_labels = expert_phase_labels(data.train_obs.numpy(), schema)

    dev_full = EvaluationManifest.load(os.path.join(MANIFESTS, f"{args.env_name}_development.yaml"))
    dev_manifest = dev_full.head(min(args.eval_episodes, len(dev_full)))
    dev_env = make_manifest_env(dev_full, dev_full.resolve_pool(), 200)

    payload = {
        "config": vars(args),
        "structures": [s.to_dict() for s in structures],
        "split": data.split_info,
        "phase_fractions": phase_fractions(train_labels),
        "objective_specs": {name: spec(name).describe() for name in OBJECTIVES},
    }

    separator("Expert phase balance in the training split")
    for name, fraction in payload["phase_fractions"].items():
        print(f"  {name:<10} {fraction:.3f}")

    # ---------------- E7: objectives ----------------
    separator(f"E7  Objectives, {args.budget} gradient steps each, {len(structures)} structures")
    e7 = []
    for structure in structures:
        for objective in OBJECTIVES:
            t0 = time.time()
            result = fit_with_objective(
                structure, data, objective, budget=args.budget,
                batch_size=args.batch_size, seed=args.seed, schema=schema,
            )
            policy = structure.build(schema)
            policy.load_state_dict(result.final_state)
            own = score(
                policy, structure, dev_manifest, dev_env, spec(objective).execution,
                args.eval_max_steps, f"{structure.structure_id}:{objective}",
            )
            deterministic = own
            if spec(objective).execution != EXEC_DETERMINISTIC_TANH:
                deterministic = score(
                    policy, structure, dev_manifest, dev_env, EXEC_DETERMINISTIC_TANH,
                    args.eval_max_steps, f"{structure.structure_id}:{objective}:det",
                )
            e7.append(
                {
                    "structure_id": structure.structure_id,
                    "objective": objective,
                    "execution": spec(objective).execution,
                    "is_likelihood": spec(objective).is_likelihood,
                    "excludes_censored": spec(objective).excludes_censored,
                    "frozen_parameters": result.notes.get("frozen_parameters", []),
                    "mse_train": result.train_loss_final,
                    "mse_validation": result.validation_loss_final,
                    "wall_time_seconds": time.time() - t0,
                    "own_contract": own,
                    "deterministic": deterministic,
                }
            )
            print(
                f"  {structure.structure_id:<20}{objective:<24}"
                f"own {own['success_rate']:.3f}  det {deterministic['success_rate']:.3f}  "
                f"mse {result.validation_loss_final:.5f}"
            )
    payload["e7"] = e7

    separator("E7 summary (mean over structures)")
    header = (
        f"{'objective':<24}{'likelihood':>11}{'own exec':>10}{'determin.':>11}"
        f"{'mse val':>10}{'frozen':>8}"
    )
    print(header)
    print("-" * len(header))
    e7_summary = {}
    for objective in OBJECTIVES:
        rows = [r for r in e7 if r["objective"] == objective]
        entry = {
            "own_success": float(np.mean([r["own_contract"]["success_rate"] for r in rows])),
            "deterministic_success": float(
                np.mean([r["deterministic"]["success_rate"] for r in rows])
            ),
            "mse_validation": float(np.mean([r["mse_validation"] for r in rows])),
            "frozen_parameters": rows[0]["frozen_parameters"] if rows else [],
            "is_likelihood": spec(objective).is_likelihood,
        }
        e7_summary[objective] = entry
        print(
            f"{objective:<24}{str(entry['is_likelihood']):>11}"
            f"{entry['own_success']:>10.3f}{entry['deterministic_success']:>11.3f}"
            f"{entry['mse_validation']:>10.5f}{len(entry['frozen_parameters']):>8}"
        )
    payload["e7_summary"] = e7_summary
    print("\n  'own exec' scores each objective under the execution it assumes.")
    print("  'determin.' scores the same parameters under tanh(mean), the fitness the")
    print("  evolution loop uses. Reporting both keeps a likelihood from being credited")
    print("  with a result its own sampler did not produce.")

    # ---------------- sampling ----------------
    separator(f"Uniform against phase-balanced minibatches, objective {DETERMINISTIC_MSE}")
    sampling_rows = []
    for structure in structures:
        for sampling in (SAMPLING_UNIFORM, SAMPLING_PHASE_BALANCED):
            result = fit_with_objective(
                structure, data, DETERMINISTIC_MSE, budget=args.budget,
                batch_size=args.batch_size, seed=args.seed, schema=schema,
                sampling=sampling, phase_labels=train_labels,
            )
            policy = structure.build(schema)
            policy.load_state_dict(result.final_state)
            outcome = score(
                policy, structure, dev_manifest, dev_env, EXEC_DETERMINISTIC_TANH,
                args.eval_max_steps, f"{structure.structure_id}:{sampling}",
            )
            sampling_rows.append(
                {
                    "structure_id": structure.structure_id,
                    "sampling": sampling,
                    "mse_validation": result.validation_loss_final,
                    **{k: v for k, v in outcome.items() if k != "failure_labels"},
                }
            )
            print(f"  {structure.structure_id:<20}{sampling:<18}"
                  f"success {outcome['success_rate']:.3f}  mse {result.validation_loss_final:.5f}")
    payload["sampling"] = sampling_rows
    for sampling in (SAMPLING_UNIFORM, SAMPLING_PHASE_BALANCED):
        rows = [r for r in sampling_rows if r["sampling"] == sampling]
        print(f"\n  {sampling:<16} mean success {np.mean([r['success_rate'] for r in rows]):.3f}, "
              f"mean validation loss {np.mean([r['mse_validation'] for r in rows]):.5f}")

    # ---------------- E8: learner-state imitation ----------------
    if not args.skip_dagger:
        separator("E8  Fixed buffer against learner-state aggregation")
        from lares.fitting.dagger import aggregate_learner_states, summarise

        train_manifest = EvaluationManifest.load(
            os.path.join(MANIFESTS, f"{args.env_name}_train.yaml")
        )
        train_pool = train_manifest.resolve_pool()
        collect_env = make_manifest_env(train_manifest, train_pool, 200)
        dagger_cases = train_manifest.episodes[: args.dagger_episodes]

        e8 = []
        for structure in structures:
            base = fit_with_objective(
                structure, data, DETERMINISTIC_MSE, budget=args.budget,
                batch_size=args.batch_size, seed=args.seed, schema=schema,
            )
            learner = structure.build(schema)
            learner.load_state_dict(base.final_state)
            fixed = score(
                learner, structure, dev_manifest, dev_env, EXEC_DETERMINISTIC_TANH,
                args.eval_max_steps, f"{structure.structure_id}:fixed",
            )

            aggregated, stats = aggregate_learner_states(
                learner, collect_env, dagger_cases, args.env_name, schema,
                buffer=DemoBuffer.load(args.buffer), max_steps=args.eval_max_steps,
            )
            print(f"\n  {structure.structure_id}: aggregation")
            print(summarise(stats))

            dagger_data = prepare_data(aggregated, split_seed=0)
            dagger_labels = expert_phase_labels(dagger_data.train_obs.numpy(), schema)
            retrained = fit_with_objective(
                structure, dagger_data, DETERMINISTIC_MSE, budget=args.budget,
                batch_size=args.batch_size, seed=args.seed, schema=schema,
            )
            policy = structure.build(schema)
            policy.load_state_dict(retrained.final_state)
            after = score(
                policy, structure, dev_manifest, dev_env, EXEC_DETERMINISTIC_TANH,
                args.eval_max_steps, f"{structure.structure_id}:dagger",
            )
            e8.append(
                {
                    "structure_id": structure.structure_id,
                    "fixed_buffer": fixed,
                    "learner_state": after,
                    "aggregation": stats.to_dict(),
                    "phase_fractions_after": phase_fractions(dagger_labels),
                    "budget_matched": args.budget,
                }
            )
            print(f"  fixed buffer   success {fixed['success_rate']:.3f}")
            print(f"  learner state  success {after['success_rate']:.3f}")
        payload["e8"] = e8
        collect_env.close()

        separator("E8 summary")
        fixed_mean = float(np.mean([r["fixed_buffer"]["success_rate"] for r in e8]))
        dagger_mean = float(np.mean([r["learner_state"]["success_rate"] for r in e8]))
        queries = sum(r["aggregation"]["expert_queries"] for r in e8)
        steps = sum(r["aggregation"]["environment_steps"] for r in e8)
        print(f"  fixed buffer   mean success {fixed_mean:.3f}")
        print(f"  learner state  mean success {dagger_mean:.3f}")
        print(f"  cost of aggregation: {queries} expert queries, {steps} environment steps")
        payload["e8_summary"] = {
            "fixed_success": fixed_mean,
            "learner_state_success": dagger_mean,
            "expert_queries": queries,
            "environment_steps": steps,
        }

    dev_env.close()
    path = os.path.join(
        args.out_dir, f"objective_comparison_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n  report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
