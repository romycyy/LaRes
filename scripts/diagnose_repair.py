#!/usr/bin/env python
"""Phase 6: diagnose a policy's failures by controlled substitution (FR-9, AC-6).

Fits a structure, finds its recurrent failures on the development screen, forms
at least two competing explanations for each, and tests every one by replacing a
single component with the scripted expert or by forcing a phase gate.

Every rollout here is expert-assisted or gate-forced. The results are diagnoses,
never policy scores, and are labelled so that nothing downstream can rank them as
fitness.

Usage (from the project root)::

    python scripts/diagnose_repair.py
    python scripts/diagnose_repair.py --structures simple_standoff --budget 2000
"""

import argparse
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from lares.core.obs_schema import get_obs_schema  # noqa: E402
from lares.core.training_pipeline import DemoBuffer, ensure_mujoco_headless_gl  # noqa: E402
from lares.eval import EvaluationManifest, make_manifest_env  # noqa: E402
from lares.fitting import DETERMINISTIC_MSE, load_bank, prepare_data  # noqa: E402
from lares.fitting.optimizers import fit_with_objective  # noqa: E402
from lares.repair import (  # noqa: E402
    records_from_study,
    run_repair,
    run_study,
    save,
    summarise,
)

DEFAULT_BUFFER = os.path.join(_PROJECT_ROOT, "logs", "baseline_lock", "demo_push-v2.pkl")
MANIFESTS = os.path.join(_PROJECT_ROOT, "config", "manifests")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env-name", default="push-v2")
    p.add_argument("--buffer", default=DEFAULT_BUFFER)
    p.add_argument("--family", choices=["simple", "complex"], default="simple")
    p.add_argument("--structures", nargs="+", default=None)
    p.add_argument("--budget", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--eval-max-steps", type=int, default=150)
    p.add_argument("--max-clusters", type=int, default=2)
    p.add_argument("--repair-operator", choices=["search", "refit"], default="search")
    p.add_argument("--repair-budget", type=int, default=400,
                   help="Adam steps for the refit operator")
    p.add_argument("--repair-episodes", type=int, default=200,
                   help="rollout episodes for the search operator")
    p.add_argument("--held-out", type=int, default=20,
                   help="development cases reserved for scoring the repair, taken from the end")
    p.add_argument("--no-repair", action="store_true")
    p.add_argument("--out-dir", default=os.path.join(_PROJECT_ROOT, "logs", "repair_studies"))
    return p.parse_args()


def separator(title):
    print(f"\n{'=' * 80}\n  {title}\n{'=' * 80}")


def describe_repair(record) -> str:
    """The repair rendered for a terminal."""
    if not record.get("attempted"):
        return f"Repair: not attempted. {record['reason']}"
    repair = record["repair"]
    decision = record["decision"]
    focused = decision.get("focused_check", {})
    protected = decision.get("protected_check", {})
    lines = [
        f"Repair ({record['operator']}) of {repair['suspected_component']} "
        f"from {record['explanation']} "
        f"({record['cluster']}, {record['focused_cases']} cases)",
        f"  parameters   {repair['parameters']}",
        f"  scope        axes {repair['axes']}, phases {repair['phases'] or 'all'}",
        f"  moved        {({k: round(v, 5) for k, v in repair['changed'].items()})}",
    ]
    if focused.get("measurable"):
        lines.append(
            f"  focused      {focused['metric']} {focused['before']:+.4f} -> "
            f"{focused['after']:+.4f} ({focused['delta']:+.4f}, wanted "
            f"{focused['predicted_direction']})"
        )
    lines.append(
        f"  protected    {protected.get('n_cases', 0)} cases on {protected.get('basis')}, "
        f"success drop {protected.get('success_drop', float('nan')):+.4f}"
    )
    secondary = protected.get("secondary")
    if secondary and secondary.get("measurable"):
        lines.append(
            f"               {secondary['metric']} drop {secondary['drop']:+.4f} "
            f"(tolerance {secondary['tolerance']})"
        )
    lines.append(
        f"  success      {record['success_before']:.3f} -> {record['success_after']:.3f}"
    )
    held = record.get("held_out")
    if held:
        lines.append(
            f"  held out     {held['n_cases']} cases: {held['success_before']:.3f} -> "
            f"{held['success_after']:.3f}  (scored after the decision)"
        )
    lines.append(f"  => {'ACCEPTED' if decision['accepted'] else 'REJECTED'}: {decision['reason']}")
    return "\n".join(lines)


def main():
    args = parse_args()
    ensure_mujoco_headless_gl()
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
    data = prepare_data(DemoBuffer.load(args.buffer), split_seed=0)

    full = EvaluationManifest.load(os.path.join(MANIFESTS, f"{args.env_name}_development.yaml"))
    manifest = full.head(min(args.eval_episodes, len(full)))
    env = make_manifest_env(full, full.resolve_pool(), 200)

    total_rollouts = 0
    total_interventions = 0
    for structure in structures:
        separator(f"{structure.structure_id}  ({structure.num_parameters} parameters)")
        result = fit_with_objective(
            structure, data, DETERMINISTIC_MSE, budget=args.budget,
            batch_size=args.batch_size, seed=args.seed, schema=schema,
        )
        policy = structure.build(schema)
        policy.load_state_dict(result.final_state)

        def progress(label, explanation, success):
            print(f"    {label:<26}{explanation:<26} success {success:.3f}")

        payload = run_study(
            policy, structure, manifest, env, args.env_name, schema,
            max_steps=args.eval_max_steps, max_clusters=args.max_clusters,
            progress=progress,
        )
        payload["fitting"] = {
            "objective": DETERMINISTIC_MSE,
            "budget": args.budget,
            "validation_loss": result.validation_loss_final,
        }
        print()
        print(summarise(payload))

        if not args.no_repair:
            held_out = full.tail(args.held_out) if args.held_out else None
            if held_out is not None:
                overlap = ({c.case_id for c in manifest.episodes}
                           & {c.case_id for c in held_out.episodes})
                if overlap:
                    raise SystemExit(
                        f"held-out slice overlaps the diagnostic cases on {sorted(overlap)}; "
                        f"reduce --held-out or --eval-episodes"
                    )
            record, _ = run_repair(
                policy, structure, payload, data, schema, manifest, env,
                held_out=held_out, operator=args.repair_operator,
                budget=args.repair_budget, search_episodes=args.repair_episodes,
                seed=args.seed, max_steps=args.eval_max_steps,
            )
            payload["repair"] = record
            payload["rollout_episodes"] += record.get("rollout_episodes", 0)
            print()
            print(describe_repair(record))

        total_rollouts += payload["rollout_episodes"]
        total_interventions += payload["intervention_episodes"]
        path = save(payload, args.out_dir)
        records = records_from_study(payload, structure)
        for record in records:
            record.save(os.path.join(args.out_dir, "experiments"))
        print(f"  {len(records)} experiment records written; "
              f"{sum(1 for r in records if r.intervention)} marked as interventions "
              f"and excluded from fitness ranking")
        print(f"\n  study: {path}")

    separator("Diagnostic cost")
    print(f"  {total_rollouts} rollout episodes across {len(structures)} structures")
    print(f"  {total_interventions} of them were expert-assisted or gate-forced and are "
          f"excluded from fitness ranking.")
    print(f"  The remaining {total_rollouts - total_interventions} ran the policy unaided: "
          f"the control, the repair search, and the validation.")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
