"""Matched-budget comparison of fitting methods (``spec.md`` Phase 3, AC-3).

Structures are held fixed, every method gets the same objective-evaluation
budget, and each fitted result is scored on the development manifest under both
checkpoint rules. Wall time and transitions processed are reported alongside,
since equal evaluation counts are not equal compute.

The selection rule is declared here, in code, before any comparison runs.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from lares.fitting.optimizers import (
    CHECKPOINT_BEST_VALIDATION,
    CHECKPOINT_FINAL,
    METHODS,
    fit,
)

from lares.fitting.sensitivity import action_sensitivity, dead_parameters, gradient_sensitivity

#: The optimizer comparison runs no rollout probe during fitting, so the
#: best-rollout rule would fall back to the final parameters and add a row that
#: only duplicates one already there. A caller that probes rollouts passes it in.
LOSS_BASED_CHECKPOINT_RULES = (CHECKPOINT_FINAL, CHECKPOINT_BEST_VALIDATION)

#: Predeclared selection rule (AC-3). Fixed before the comparison is run and not
#: revised afterwards: the default is the configuration with the highest mean
#: development success across the frozen bank, ties broken by mean signed goal
#: progress and then by wall time. Development data only; the final-test manifest
#: is never consulted here.
SELECTION_RULE = {
    "primary": "mean development success rate across the structure bank",
    "tie_break_1": "mean signed goal progress",
    "tie_break_2": "lower mean wall time",
    "data": "development split only",
}


@dataclass
class BenchmarkConfig:
    env_name: str = "push-v2"
    methods: tuple = METHODS
    checkpoint_rules: tuple = LOSS_BASED_CHECKPOINT_RULES
    budget: int = 2000
    batch_size: int = 256
    seeds: tuple = (0,)
    val_fraction: float = 0.2
    split_seed: int = 0
    eval_episodes: int = 10
    eval_max_steps: int = 150
    restarts: int = 4
    de_popsize: int = 12
    val_every: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("methods", "checkpoint_rules", "seeds"):
            d[key] = list(getattr(self, key))
        return d


@dataclass
class BenchmarkRow:
    """One structure, one method, one seed, one checkpoint rule."""

    structure_id: str
    family: str
    num_parameters: int
    method: str
    seed: int
    checkpoint_rule: str
    train_loss: float
    validation_loss: float
    objective_evaluations: int
    transition_evaluations: int
    wall_time_seconds: float
    converged_at: int | None
    success_rate: float | None = None
    mean_return: float | None = None
    signed_goal_progress: float | None = None
    failure_labels: dict | None = None
    bound_activity_fraction: float | None = None
    dead_parameters: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _loss(policy, obs, actions) -> float:
    """Fitting loss of one parameter set on one split."""
    from lares.fitting.optimizers import bc_objective

    if obs is None:
        return float("nan")
    import torch

    with torch.no_grad():
        return float(bc_objective(policy, obs, actions))


def loss_versus_success(rows) -> dict:
    """Does a better imitation fit mean a better controller?

    The whole inner loop assumes it does: candidates are fitted on imitation loss
    and then judged on task success. Measuring the relationship directly is
    cheap, and a weak or negative one changes what the fitting phase is for.
    """
    pairs = [
        (r["validation_loss"], r["success_rate"])
        for r in rows
        if r.get("success_rate") is not None
        and r.get("validation_loss") is not None
        and not np.isnan(r["validation_loss"])
    ]
    if len(pairs) < 3:
        return {"unavailable": "fewer than three scored rows"}
    losses = np.asarray([p[0] for p in pairs], dtype=float)
    successes = np.asarray([p[1] for p in pairs], dtype=float)
    out = {"n": len(pairs)}
    if losses.std() < 1e-12 or successes.std() < 1e-12:
        out["unavailable"] = "no variation in one of the two series"
        return out
    try:
        from scipy.stats import pearsonr, spearmanr

        pr, pp = pearsonr(losses, successes)
        sr, sp = spearmanr(losses, successes)
        out.update(
            {
                "pearson_r": float(pr),
                "pearson_p": float(pp),
                "spearman_r": float(sr),
                "spearman_p": float(sp),
            }
        )
    except Exception as exc:  # pragma: no cover - scipy absent
        out["unavailable"] = f"scipy unavailable: {exc!r}"
    return out


def _bound_fraction(bound_activity) -> float | None:
    """Share of parameters resting on a declared bound after fitting."""
    if not bound_activity:
        return None
    values = [
        max(v.get("at_lower", 0.0), v.get("at_upper", 0.0))
        for v in bound_activity.values()
        if isinstance(v, dict)
    ]
    return float(np.mean(values)) if values else None


def run_benchmark(
    structures,
    data,
    schema,
    config: BenchmarkConfig,
    dev_manifest=None,
    dev_env=None,
    progress=None,
) -> dict:
    """Fit every structure with every method and score the result.

    ``dev_manifest`` and ``dev_env`` are optional: without them the comparison is
    loss-only and the rollout columns stay ``None`` rather than being filled with
    a stand-in.
    """
    from lares.core.training_pipeline import evaluate_policy
    from lares.eval.report import build_report

    rows: list[BenchmarkRow] = []
    sensitivity: dict[str, dict] = {}
    started = time.time()
    total = len(structures) * len(config.methods) * len(config.seeds)
    done = 0

    for structure in structures:
        policy = structure.build(schema)
        obs = data.train_obs[: config.batch_size]
        actions = data.train_actions[: config.batch_size]
        action_stats = action_sensitivity(policy, obs)
        sensitivity[structure.structure_id] = {
            "action": action_stats,
            "gradient": gradient_sensitivity(policy, obs, actions),
            "dead_parameters": dead_parameters(action_stats),
        }

        for method in config.methods:
            for seed in config.seeds:
                kwargs = dict(
                    budget=config.budget,
                    batch_size=config.batch_size,
                    seed=seed,
                    schema=schema,
                )
                if method == "multistart_adam":
                    kwargs["restarts"] = config.restarts
                if method == "differential_evolution":
                    kwargs["popsize"] = config.de_popsize
                else:
                    kwargs["val_every"] = config.val_every
                result = fit(method, structure, data, **kwargs)
                done += 1
                if progress is not None:
                    progress(done, total, structure.structure_id, method, seed)

                for rule in config.checkpoint_rules:
                    # Score the parameters the rule actually selects, rather than
                    # reusing the end-of-run scalars: otherwise the two rules
                    # differ only in a label.
                    selected = structure.build(schema)
                    selected.load_state_dict(result.state_for(rule))
                    row = BenchmarkRow(
                        structure_id=structure.structure_id,
                        family=structure.family,
                        num_parameters=structure.num_parameters,
                        method=method,
                        seed=seed,
                        checkpoint_rule=rule,
                        train_loss=_loss(selected, data.train_obs, data.train_actions),
                        validation_loss=_loss(selected, data.val_obs, data.val_actions),
                        objective_evaluations=result.objective_evaluations,
                        transition_evaluations=result.transition_evaluations,
                        wall_time_seconds=result.wall_time_seconds,
                        converged_at=result.converged_at,
                        bound_activity_fraction=_bound_fraction(result.bound_activity),
                        dead_parameters=sensitivity[structure.structure_id]["dead_parameters"],
                    )
                    if dev_manifest is not None and dev_env is not None:
                        manifest_result = evaluate_policy(
                            selected, dev_env, dev_manifest, max_steps=config.eval_max_steps
                        )
                        report = build_report(
                            manifest_result,
                            candidate_id=f"{structure.structure_id}:{method}:{rule}",
                            checkpoint="fitted",
                            action_dim=structure.action_dim,
                        )
                        row.success_rate = report.rollout.success_rate
                        row.mean_return = report.rollout.mean_return
                        row.signed_goal_progress = (
                            report.rollout.signed_goal_progress["mean"]
                            if report.rollout.signed_goal_progress
                            else None
                        )
                        row.failure_labels = report.rollout.failure_labels
                    rows.append(row)

    return {
        "config": config.to_dict(),
        "selection_rule": SELECTION_RULE,
        "split": data.split_info,
        "structures": [s.to_dict() for s in structures],
        "rows": [r.to_dict() for r in rows],
        "sensitivity": sensitivity,
        "wall_time_seconds": time.time() - started,
        "rollouts_scored": dev_manifest is not None and dev_env is not None,
        "loss_versus_success": loss_versus_success([r.to_dict() for r in rows]),
    }


# ---------------------------------------------------------------------------
#  Aggregation and selection
# ---------------------------------------------------------------------------


def _mean(values):
    clean = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.mean(clean)) if clean else None


def aggregate(rows, by=("method", "checkpoint_rule")) -> list[dict]:
    """Group rows and average the comparison columns."""
    groups: dict[tuple, list] = {}
    for row in rows:
        key = tuple(row[k] for k in by)
        groups.setdefault(key, []).append(row)
    out = []
    for key, members in sorted(groups.items()):
        entry = dict(zip(by, key))
        entry.update(
            {
                "n": len(members),
                "train_loss": _mean([m["train_loss"] for m in members]),
                "validation_loss": _mean([m["validation_loss"] for m in members]),
                "success_rate": _mean([m["success_rate"] for m in members]),
                "mean_return": _mean([m["mean_return"] for m in members]),
                "signed_goal_progress": _mean([m["signed_goal_progress"] for m in members]),
                "wall_time_seconds": _mean([m["wall_time_seconds"] for m in members]),
                "objective_evaluations": _mean([m["objective_evaluations"] for m in members]),
                "bound_activity_fraction": _mean(
                    [m["bound_activity_fraction"] for m in members]
                ),
                "converged_at": _mean([m["converged_at"] for m in members]),
            }
        )
        out.append(entry)
    return out


def select_default(aggregated) -> dict:
    """Apply the predeclared rule and say which configuration it picks."""
    if not aggregated:
        return {"selected": None, "reason": "no results"}

    def key(entry):
        success = entry["success_rate"]
        progress = entry["signed_goal_progress"]
        wall = entry["wall_time_seconds"]
        return (
            -(success if success is not None else -1e9),
            -(progress if progress is not None else -1e9),
            wall if wall is not None else 1e9,
        )

    ranked = sorted(aggregated, key=key)
    best = ranked[0]
    return {
        "selected": {"method": best["method"], "checkpoint_rule": best["checkpoint_rule"]},
        "rule": SELECTION_RULE,
        "ranking": ranked,
        "reason": (
            f"highest mean development success ({best['success_rate']}), "
            f"goal progress {best['signed_goal_progress']}, "
            f"wall time {best['wall_time_seconds']}"
        ),
    }


def paired_method_comparison(rows, reference: str, rule: str = CHECKPOINT_FINAL) -> list[dict]:
    """Per-structure differences against a reference method, with an interval.

    The structures are common to every method, so the structure is the paired
    unit. Comparing two column means instead would throw that pairing away and
    make small differences look more certain than they are (``spec.md``
    section 12, rule 5).
    """
    selected = [r for r in rows if r["checkpoint_rule"] == rule]
    by_method: dict[str, dict[str, dict]] = {}
    for r in selected:
        by_method.setdefault(r["method"], {})[r["structure_id"]] = r
    if reference not in by_method:
        return []

    out = []
    base = by_method[reference]
    for method, entries in sorted(by_method.items()):
        if method == reference:
            continue
        shared = sorted(set(entries) & set(base))
        if not shared:
            continue
        row = {"method": method, "reference": reference, "n_structures": len(shared)}
        for metric in ("success_rate", "signed_goal_progress", "validation_loss"):
            diffs = [
                entries[s][metric] - base[s][metric]
                for s in shared
                if entries[s].get(metric) is not None and base[s].get(metric) is not None
            ]
            if not diffs:
                row[metric] = None
                continue
            arr = np.asarray(diffs, dtype=float)
            mean = float(arr.mean())
            se = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
            row[metric] = {
                "mean_difference": mean,
                "se": se,
                "ci95_low": mean - 1.96 * se,
                "ci95_high": mean + 1.96 * se,
                "n": int(arr.size),
                "excludes_zero": bool((mean - 1.96 * se) * (mean + 1.96 * se) > 0),
            }
        out.append(row)
    return out


def format_paired(comparisons, metric: str = "success_rate") -> str:
    header = (
        f"{'method vs reference':<32}{'n':>4}{'difference':>13}{'95% CI':>26}  verdict"
    )
    lines = [header, "-" * len(header)]
    for entry in comparisons:
        stats = entry.get(metric)
        if stats is None:
            lines.append(f"{entry['method']:<32}{entry['n_structures']:>4}{'n/a':>13}")
            continue
        verdict = "interval excludes zero" if stats["excludes_zero"] else "not distinguishable"
        lines.append(
            f"{entry['method'] + ' - ' + entry['reference']:<32}{stats['n']:>4}"
            f"{stats['mean_difference']:>+13.4f}"
            f"{'[' + format(stats['ci95_low'], '+.4f') + ', ' + format(stats['ci95_high'], '+.4f') + ']':>26}"
            f"  {verdict}"
        )
    return "\n".join(lines)


def format_table(aggregated, columns=None) -> str:
    """Comparison table for the terminal and the handoff document."""
    columns = columns or [
        ("method", "method", "<24", None),
        ("checkpoint_rule", "checkpoint", "<17", None),
        ("n", "n", ">4", None),
        ("train_loss", "train", ">9", 5),
        ("validation_loss", "val", ">9", 5),
        ("success_rate", "success", ">9", 3),
        ("signed_goal_progress", "progress", ">10", 4),
        ("bound_activity_fraction", "at bound", ">10", 3),
        ("wall_time_seconds", "seconds", ">9", 2),
    ]
    header = "".join(f"{label:{width}}" for _, label, width, _ in columns)
    lines = [header, "-" * len(header)]
    for entry in aggregated:
        cells = []
        for key, _, width, places in columns:
            value = entry.get(key)
            if value is None:
                cells.append(f"{'n/a':{width}}")
            elif places is None:
                cells.append(f"{value:{width}}")
            else:
                cells.append(f"{value:{width}.{places}f}")
        lines.append("".join(cells))
    return "\n".join(lines)
