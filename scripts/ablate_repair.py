#!/usr/bin/env python
"""AC-6 / E9: is the repair loop worth its budget? (``spec.md`` Phase 6, step 6)

Five arms start from the identical fitted policy and are given the identical
budget of development-screen rollout episodes. They differ only in what they
spend it on:

``full``             a control run, two controlled substitutions to decide between
                     the competing explanations, then a bounded search over the
                     parameters of the component the evidence supported.
``diagnostics_only`` a control run, then the same bounded search aimed at the
                     first plausible explanation, which is never tested.
``evaluation_only``  no diagnostics: a component picked at random, searched
                     against generic task progress.
``global_search``    no localisation: every parameter searched at once, same
                     budget. The control for "does naming the component help?".
``optimizer_only``   no search over structure at all: a longer fit whose
                     checkpoint is chosen by rollout probes.

Everything is scored afterwards on held-out development cases that no arm saw.
The screen and the held-out slice are disjoint by construction.

Usage (from the project root)::

    python scripts/ablate_repair.py
    python scripts/ablate_repair.py --structures simple_standoff --seeds 0 --budget 100
"""

import argparse
import copy
import json
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402

from lares.core.obs_schema import get_obs_schema  # noqa: E402
from lares.core.training_pipeline import DemoBuffer, ensure_mujoco_headless_gl  # noqa: E402
from lares.eval import EvaluationManifest, make_manifest_env  # noqa: E402
from lares.eval.report import CHECKPOINT_FITTED, build_report  # noqa: E402
from lares.eval.runner import PolicyActor, evaluate_manifest, paired_difference  # noqa: E402
from lares.fitting import DETERMINISTIC_MSE, load_bank, prepare_data  # noqa: E402
from lares.fitting.optimizers import fit_with_objective  # noqa: E402
from lares.repair import (  # noqa: E402
    COMPONENT_SCOPES,
    BoundedRepair,
    InterventionActor,
    competing_explanations,
    episode_metric_mean,
    find_clusters,
    phase_subset,
    rank_explanations,
    search_repair,
    select_parameters,
)
from lares.repair.hypotheses import InterventionOutcome  # noqa: E402

ARMS = ("full", "diagnostics_only", "evaluation_only", "global_search", "optimizer_only")

#: What an arm searches for when no diagnosis named a focused metric.
GENERIC_METRIC = "signed_goal_progress"
GENERIC_DIRECTION = "increase"

DEFAULT_BUFFER = os.path.join(_PROJECT_ROOT, "logs", "baseline_lock", "demo_push-v2.pkl")
MANIFESTS = os.path.join(_PROJECT_ROOT, "config", "manifests")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env-name", default="push-v2")
    p.add_argument("--buffer", default=DEFAULT_BUFFER)
    p.add_argument("--structures", nargs="+",
                   default=["simple_reach_push", "simple_damped_push", "simple_standoff"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--budget", type=int, default=200,
                   help="development-screen rollout episodes each arm may spend")
    p.add_argument("--fit-budget", type=int, default=2000)
    p.add_argument("--screen", type=int, default=10)
    p.add_argument("--held-out", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--population", type=int, default=5)
    p.add_argument("--out-dir", default=os.path.join(_PROJECT_ROOT, "logs", "repair_ablation"))
    p.add_argument("--reference-arm", default="full",
                   help="the arm every other is paired against")
    p.add_argument("--summarise", default=None,
                   help="re-print the comparison from a saved run instead of running anything")
    return p.parse_args()


def separator(title):
    print(f"\n{'=' * 84}\n  {title}\n{'=' * 84}")


def scoped_repair(policy, explanation, data, schema):
    """A bounded repair aimed at an explanation's component."""
    from lares.repair.repair import propose_repair

    return propose_repair(policy, explanation, data, schema)


def all_parameter_repair(policy):
    """The unlocalised control: every declared parameter is in scope."""
    return BoundedRepair(
        explanation="global_search",
        suspected_component="every parameter",
        parameters=sorted(policy.get_param_ranges()),
        axes=(0, 1, 2, 3),
        phases=(),
        focused_metric=GENERIC_METRIC,
        predicted_direction=GENERIC_DIRECTION,
        note="no component was named; the whole parameter vector is in scope",
    )


def random_component_repair(policy, data, schema, rng):
    """The diagnosis-free control: a component chosen without evidence."""
    component = sorted(COMPONENT_SCOPES)[int(rng.integers(0, len(COMPONENT_SCOPES)))]
    scope = COMPONENT_SCOPES[component]
    obs = phase_subset(data.train_obs, tuple(scope["phases"]), schema)[0][:512]
    return BoundedRepair(
        explanation="random_component",
        suspected_component=component,
        parameters=select_parameters(policy, obs, tuple(scope["axes"])),
        axes=tuple(scope["axes"]),
        phases=tuple(scope["phases"]),
        focused_metric=GENERIC_METRIC,
        predicted_direction=GENERIC_DIRECTION,
        note="component drawn at random; no diagnostic evidence was used",
    )


def diagnose(policy, structure, screen, env, max_steps):
    """The control run every diagnostic arm pays for."""
    result = evaluate_manifest(
        PolicyActor(policy, deterministic=True, name="control"), screen, env,
        max_steps=max_steps,
    )
    report = build_report(result, f"{structure.structure_id}:control", CHECKPOINT_FITTED,
                          action_dim=structure.action_dim)
    return result, report, find_clusters(report)


def run_arm(arm, structure, policy, data, schema, screen, env, env_name, args, seed):
    """Spend one arm's budget and return the policy it produced."""
    rng = np.random.default_rng(seed)
    cost = {"screen_episodes": 0, "adam_steps": 0, "expert_assisted_episodes": 0}
    notes = {"arm": arm}
    working = copy.deepcopy(policy)

    if arm == "optimizer_only":
        probes = max(1, args.budget // len(screen))
        every = max(1, args.fit_budget // probes)

        def probe(candidate):
            result = evaluate_manifest(
                PolicyActor(candidate, deterministic=True, name="probe"), screen, env,
                max_steps=args.max_steps,
            )
            cost["screen_episodes"] += len(screen)
            progress = episode_metric_mean(result, "signed_goal_progress")
            return result.success_rate + 0.01 * float(progress or 0.0)

        fitted = fit_with_objective(
            structure, data, DETERMINISTIC_MSE, budget=args.fit_budget,
            batch_size=256, seed=seed, schema=schema,
            rollout_probe=probe, rollout_every=every,
        )
        cost["adam_steps"] = args.fit_budget
        working = structure.build(schema)
        working.load_state_dict(fitted.best_rollout_state or fitted.final_state)
        notes["checkpoint"] = "best_rollout" if fitted.best_rollout_state else "final"
        notes["probe_score"] = fitted.best_rollout_score
        notes["probes"] = probes
        return working, cost, notes

    repair = None
    focused_cases = None
    if arm in ("full", "diagnostics_only"):
        control, report, clusters = diagnose(working, structure, screen, env, args.max_steps)
        cost["screen_episodes"] += len(screen)
        if not clusters:
            notes["skipped"] = "no recurrent failure cluster to repair"
            return working, cost, notes
        cluster = clusters[0]
        focused_cases = cluster.case_ids
        explanations = competing_explanations(cluster)
        notes["cluster"] = cluster.label
        notes["competing"] = [e.name for e in explanations]

        if arm == "full":
            outcomes = []
            for explanation in explanations:
                actor = InterventionActor(working, explanation.substitution, env_name, schema,
                                          name=f"probe:{explanation.name}")
                result = evaluate_manifest(actor, screen, env, max_steps=args.max_steps)
                cost["screen_episodes"] += len(screen)
                cost["expert_assisted_episodes"] += len(screen)
                before = episode_metric_mean(control, explanation.focused_metric, cluster.case_ids)
                after = episode_metric_mean(result, explanation.focused_metric, cluster.case_ids)
                outcomes.append(
                    InterventionOutcome(
                        explanation=explanation.name,
                        substitution=explanation.substitution.to_dict(),
                        focused_metric=explanation.focused_metric,
                        control_value=before, intervened_value=after,
                        check=explanation.check(before, after),
                        control_success=control.success_rate,
                        intervened_success=result.success_rate,
                        n_cases=len(cluster.case_ids),
                    )
                )
            ranked = rank_explanations(outcomes)
            best = ranked[0]
            chosen = next(e for e in explanations if e.name == best.explanation)
            notes["supported"] = best.check.get("supported", False)
            notes["chosen"] = chosen.name
            notes["rejected"] = [o.explanation for o in ranked[1:]]
        else:
            chosen = explanations[0]
            notes["chosen"] = chosen.name
            notes["untested"] = [e.name for e in explanations[1:]]
        repair = scoped_repair(working, chosen, data, schema)
    elif arm == "evaluation_only":
        repair = random_component_repair(working, data, schema, rng)
        notes["chosen"] = repair.suspected_component
    elif arm == "global_search":
        repair = all_parameter_repair(working)
    else:
        raise ValueError(f"unknown arm {arm!r}")

    remaining = args.budget - cost["screen_episodes"]
    if repair.is_empty or remaining < len(screen):
        notes["skipped"] = "no parameter in scope" if repair.is_empty else "budget spent on diagnosis"
        notes["repair"] = repair.to_dict()
        return working, cost, notes

    working, trace = search_repair(
        working, repair, screen, env, focused_case_ids=focused_cases,
        budget_episodes=remaining, population=args.population, seed=seed,
        max_steps=args.max_steps,
    )
    cost["screen_episodes"] += repair.cost.get("rollout_episodes", 0)
    notes["repair"] = repair.to_dict()
    notes["search_evaluations"] = repair.cost.get("evaluations", 0)
    notes["trace_final"] = trace[-1] if trace else None
    return working, cost, notes


def main():
    args = parse_args()
    if args.summarise:
        with open(args.summarise, encoding="utf-8") as f:
            saved = json.load(f)
        args.arms = saved["arms"]
        separator(f"{os.path.basename(args.summarise)}  "
                  f"({saved['budget_screen_episodes']} screen episodes per arm, "
                  f"held out on {saved['held_out_manifest']})")
        print(summarise(saved["rows"], args))
        print(render_paired(saved["rows"], reference=args.reference_arm))
        return 0

    ensure_mujoco_headless_gl()
    schema = get_obs_schema(args.env_name)

    if not os.path.isfile(args.buffer):
        print(f"missing demo buffer: {args.buffer}")
        return 1
    data = prepare_data(DemoBuffer.load(args.buffer), split_seed=0)

    full = EvaluationManifest.load(os.path.join(MANIFESTS, f"{args.env_name}_development.yaml"))
    screen, held_out = full.head(args.screen), full.tail(args.held_out)
    overlap = ({c.case_id for c in screen.episodes} & {c.case_id for c in held_out.episodes})
    if overlap:
        raise SystemExit(f"screen and held-out slices overlap on {sorted(overlap)}")
    env = make_manifest_env(full, full.resolve_pool(), 200)

    bank = {s.structure_id: s for s in load_bank(env_id=args.env_name)}
    missing = [s for s in args.structures if s not in bank]
    if missing:
        raise SystemExit(f"not in the structure bank: {missing}")

    started = time.time()
    rows = []
    for structure_id in args.structures:
        structure = bank[structure_id]
        for seed in args.seeds:
            separator(f"{structure_id}  seed {seed}")
            fitted = fit_with_objective(
                structure, data, DETERMINISTIC_MSE, budget=args.fit_budget,
                batch_size=256, seed=seed, schema=schema,
            )
            start_policy = structure.build(schema)
            start_policy.load_state_dict(fitted.final_state)
            baseline = evaluate_manifest(
                PolicyActor(start_policy, deterministic=True, name="start"), held_out, env,
                max_steps=args.max_steps,
            )
            print(f"  start policy: held-out success {baseline.success_rate:.3f}")

            for arm in args.arms:
                policy, cost, notes = run_arm(
                    arm, structure, start_policy, data, schema, screen, env,
                    args.env_name, args, seed,
                )
                scored = evaluate_manifest(
                    PolicyActor(policy, deterministic=True, name=arm), held_out, env,
                    max_steps=args.max_steps,
                )
                row = {
                    "structure_id": structure_id,
                    "seed": seed,
                    "arm": arm,
                    "held_out_success": scored.success_rate,
                    "held_out_progress": episode_metric_mean(scored, "signed_goal_progress"),
                    "baseline_success": baseline.success_rate,
                    "paired_vs_start": paired_difference(scored, baseline, "success"),
                    "cost": cost,
                    "notes": notes,
                }
                rows.append(row)
                delta = row["paired_vs_start"]
                print(
                    f"    {arm:<18} held-out {scored.success_rate:.3f} "
                    f"(start {baseline.success_rate:.3f}, paired {delta['mean_difference']:+.3f} "
                    f"[{delta['ci95_low']:+.3f}, {delta['ci95_high']:+.3f}])  "
                    f"screen episodes {cost['screen_episodes']}"
                )

    separator("Arm comparison on held-out development cases")
    print(summarise(rows, args))
    print(render_paired(rows, reference=args.reference_arm))
    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"ablation_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "arms": list(args.arms),
                "structures": list(args.structures),
                "seeds": list(args.seeds),
                "budget_screen_episodes": args.budget,
                "screen_manifest": screen.manifest_id,
                "held_out_manifest": held_out.manifest_id,
                "wall_time_seconds": time.time() - started,
                "rows": rows,
            },
            f, indent=2, default=str,
        )
    print(f"\n  written: {path}")
    env.close()
    return 0


def paired_arms(rows, reference: str) -> list:
    """Each arm minus ``reference`` on the replicates they share.

    Arms run on the same structures and seeds from the same start policy, so the
    comparison is paired and the difference is what should carry an interval.
    Comparing five separately computed means is the inference section 12 rule 5
    forbids, and with six replicates it would be the whole result.
    """
    by_arm = {}
    for row in rows:
        by_arm.setdefault(row["arm"], {})[(row["structure_id"], row["seed"])] = row
    base = by_arm.get(reference, {})
    out = []
    for arm, scored in sorted(by_arm.items()):
        if arm == reference:
            continue
        shared = sorted(set(base) & set(scored))
        if not shared:
            continue
        diffs = [scored[k]["held_out_success"] - base[k]["held_out_success"] for k in shared]
        mean = float(np.mean(diffs))
        se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
        out.append({
            "arm": arm,
            "reference": reference,
            "n_replicates": len(diffs),
            "mean_difference": mean,
            "se": se,
            "ci95_low": mean - 1.96 * se,
            "ci95_high": mean + 1.96 * se,
            "wins": sum(1 for d in diffs if d > 0),
            "losses": sum(1 for d in diffs if d < 0),
            "ties": sum(1 for d in diffs if d == 0),
        })
    return out


def render_paired(rows, reference: str) -> str:
    comparisons = paired_arms(rows, reference)
    if not comparisons:
        return ""
    header = f"{'arm':<18}{'difference':>12}{'95% interval':>22}{'w/l/t':>10}"
    lines = ["", f"Paired against {reference}, on the replicates both arms ran", header, "-" * 62]
    for c in comparisons:
        interval = f"[{c['ci95_low']:+.3f}, {c['ci95_high']:+.3f}]"
        record = f"{c['wins']}/{c['losses']}/{c['ties']}"
        lines.append(
            f"{c['arm']:<18}{c['mean_difference']:>+12.3f}{interval:>22}{record:>10}"
        )
    lines.append("")
    lines.append(
        f"The interval spans {comparisons[0]['n_replicates']} independent replicates, not "
        f"episodes within one run. It is wide because six runs is a small number."
    )
    return "\n".join(lines)


def summarise(rows, args) -> str:
    """Per-arm means over replicates, with the cost each arm actually spent."""
    lines = [
        f"{'arm':<18}{'held-out':>10}{'vs start':>10}{'progress':>10}"
        f"{'episodes':>10}{'adam':>8}{'expert':>8}  replicates",
        "-" * 84,
    ]
    for arm in args.arms:
        arm_rows = [r for r in rows if r["arm"] == arm]
        if not arm_rows:
            continue
        success = float(np.mean([r["held_out_success"] for r in arm_rows]))
        start = float(np.mean([r["baseline_success"] for r in arm_rows]))
        progress = float(np.mean([r["held_out_progress"] or 0.0 for r in arm_rows]))
        episodes = float(np.mean([r["cost"]["screen_episodes"] for r in arm_rows]))
        adam = float(np.mean([r["cost"]["adam_steps"] for r in arm_rows]))
        expert = float(np.mean([r["cost"]["expert_assisted_episodes"] for r in arm_rows]))
        lines.append(
            f"{arm:<18}{success:>10.3f}{success - start:>+10.3f}{progress:>10.4f}"
            f"{episodes:>10.0f}{adam:>8.0f}{expert:>8.0f}  {len(arm_rows)}"
        )
    lines.append("")
    lines.append(
        "Budget is matched on development-screen episodes. The optimizer-only arm "
        "additionally spends Adam steps the search arms do not, which favours it. "
        "Held-out cases took no part in any arm's decisions."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
