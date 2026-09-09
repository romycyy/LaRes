"""The ``EvaluationReport`` contract (``spec.md`` 7.3, FR-3, AC-2).

One report per candidate per checkpoint. It carries the validity gates the
candidate passed, how its numerical fitting went, what its rollouts did, and the
full per-episode table. Aggregate scores are never the whole record: a candidate
that never reaches the object and one that reaches it and pushes the wrong way
both score zero, and only the per-episode geometry tells them apart.

Anything that could not be measured is listed in ``unavailable_fields`` and left
as ``None``. A field is never silently zero-filled.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

from lares.eval.diagnostics import aggregate_by_axis, aggregate_gate_statistics
from lares.eval.runner import ManifestResult

CHECKPOINT_ZERO_SHOT = "zero_shot"
CHECKPOINT_INTERMEDIATE = "intermediate"
CHECKPOINT_FITTED = "fitted"
CHECKPOINTS = (CHECKPOINT_ZERO_SHOT, CHECKPOINT_INTERMEDIATE, CHECKPOINT_FITTED)


def _distribution(values: Sequence, label: str) -> dict | None:
    """Mean and quartiles for one measurement, or ``None`` if never measured."""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return None
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


@dataclass
class ValiditySection:
    """Which pre-rollout gates the candidate passed (``spec.md`` 7.3 ``validity``)."""

    compile_ok: bool | None = None
    shape_ok: bool | None = None
    finite_ok: bool | None = None
    batch_independence_ok: bool | None = None
    range_coverage_ok: bool | None = None
    input_sensitivity_ok: bool | None = None
    no_raw_obs_indexing_ok: bool | None = None
    errors: list = field(default_factory=list)

    @classmethod
    def from_validation_report(cls, report) -> "ValiditySection":
        from lares.core import policy_validator as pv

        cats = report.categories
        return cls(
            compile_ok=pv.FORWARD_RAISED not in cats,
            shape_ok=pv.WRONG_SHAPE not in cats,
            finite_ok=not (cats & {pv.NON_FINITE, pv.NON_POSITIVE_STD}),
            batch_independence_ok=pv.BATCH_COUPLING not in cats,
            range_coverage_ok=not (
                cats
                & {
                    pv.UNDECLARED_PARAMETER,
                    pv.UNKNOWN_RANGE_ENTRY,
                    pv.INVALID_RANGE,
                    pv.INIT_OUT_OF_RANGE,
                    pv.RANGES_NOT_A_DICT,
                }
            ),
            input_sensitivity_ok=pv.INSENSITIVE_INPUT not in cats,
            no_raw_obs_indexing_ok=pv.RAW_OBS_INDEXING not in cats,
            errors=[{"category": e.category, "message": e.message} for e in report.errors],
        )


@dataclass
class FittingSection:
    """Numerical-fitting evidence (``spec.md`` 7.3 ``fitting``).

    Distinguishes a structure that cannot express the expert from one whose
    constants were badly tuned. Training loss alone cannot: both plateau.
    """

    train_loss_by_axis: dict | None = None
    validation_loss_by_axis: dict | None = None
    loss_by_phase: dict | None = None
    gradient_norms: dict | None = None
    bound_activity: dict | None = None
    train_loss_start: float | None = None
    train_loss_end: float | None = None
    validation_loss_end: float | None = None
    num_steps: int | None = None
    split: dict | None = None

    @classmethod
    def from_bc_stats(cls, stats: dict | None) -> "FittingSection":
        if not stats:
            return cls()
        return cls(
            train_loss_by_axis=stats.get("train_loss_by_axis"),
            validation_loss_by_axis=stats.get("validation_loss_by_axis"),
            loss_by_phase=stats.get("loss_by_phase"),
            gradient_norms=stats.get("gradient_norms"),
            bound_activity=stats.get("bound_activity"),
            train_loss_start=stats.get("train_loss_start"),
            train_loss_end=stats.get("final_loss"),
            validation_loss_end=stats.get("validation_loss_end"),
            num_steps=stats.get("num_steps"),
            split=stats.get("split"),
        )


@dataclass
class RolloutSection:
    """Aggregated rollout outcomes and diagnostics (``spec.md`` 7.3 ``rollout``).

    Success is the primary outcome. Return and the distance measures are
    secondary and exist to rank candidates that all score zero.
    """

    success_rate: float = 0.0
    success_se: float = 0.0
    mean_return: float = 0.0
    return_se: float = 0.0
    num_episodes: int = 0
    final_goal_distance: dict | None = None
    initial_goal_distance: dict | None = None
    signed_goal_progress: dict | None = None
    lateral_drift: dict | None = None
    min_tcp_object_distance: dict | None = None
    object_displacement: dict | None = None
    action_saturation_by_axis: list | None = None
    action_variation_by_axis: list | None = None
    proximity_rate: dict | None = None
    phase_occupancy: dict | None = None
    phase_transitions: dict | None = None
    gate_statistics: dict | None = None
    timeout_rate: float = 0.0
    failure_labels: dict | None = None


@dataclass
class EvaluationReport:
    """One candidate, one checkpoint, one manifest, one action mode."""

    candidate_id: str
    manifest_id: str
    checkpoint: str
    action_mode: str
    validity: ValiditySection = field(default_factory=ValiditySection)
    fitting: FittingSection = field(default_factory=FittingSection)
    rollout: RolloutSection = field(default_factory=RolloutSection)
    visual_analysis_ids: list = field(default_factory=list)
    episodes: list = field(default_factory=list)
    unavailable_fields: list = field(default_factory=list)
    measurement_notes: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.checkpoint not in CHECKPOINTS:
            raise ValueError(
                f"checkpoint must be one of {CHECKPOINTS}, got {self.checkpoint!r}. "
                f"Zero-shot, intermediate and fitted results are never conflated."
            )

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: str) -> dict:
        """Return the saved report as a plain dict, for aggregation scripts."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def headline(self) -> str:
        r = self.rollout
        parts = [
            f"{self.candidate_id} @{self.checkpoint}/{self.action_mode}",
            f"success={r.success_rate:.3f}+-{r.success_se:.3f}",
            f"return={r.mean_return:.1f}",
        ]
        if r.final_goal_distance:
            parts.append(f"final_dist={r.final_goal_distance['mean']:.3f}")
        if r.failure_labels:
            top = max(r.failure_labels.items(), key=lambda kv: kv[1])
            parts.append(f"most_common={top[0]}({top[1]})")
        return "  ".join(parts)


def build_report(
    result: ManifestResult,
    candidate_id: str,
    checkpoint: str,
    validation_report=None,
    bc_stats: dict | None = None,
    action_dim: int = 4,
    visual_analysis_ids: Sequence[str] | None = None,
) -> EvaluationReport:
    """Assemble the report for one evaluated checkpoint.

    Fields with nothing behind them are recorded in ``unavailable_fields`` rather
    than defaulted, so a reader can tell "measured and zero" from "not measured".
    """
    episodes = result.episodes
    unavailable: list[str] = []

    def dist(attr):
        out = _distribution([getattr(e, attr) for e in episodes], attr)
        if out is None:
            unavailable.append(f"rollout.{attr}")
        return out

    gate_stats = aggregate_gate_statistics([e.gate_statistics for e in episodes])
    if gate_stats is None:
        unavailable.extend(
            ["rollout.gate_statistics", "rollout.phase_occupancy", "rollout.phase_transitions"]
        )
        phase_occupancy = phase_transitions = None
    else:
        phase_occupancy = {k: v["occupancy_above_half"] for k, v in gate_stats.items()}
        phase_transitions = {k: v["mean_transitions"] for k, v in gate_stats.items()}

    saturation = aggregate_by_axis(
        [e.action_saturation_by_axis for e in episodes], action_dim
    )
    variation = aggregate_by_axis([e.action_variation_by_axis for e in episodes], action_dim)
    if saturation is None:
        unavailable.append("rollout.action_saturation_by_axis")
    if variation is None:
        unavailable.append("rollout.action_variation_by_axis")

    labels = [e.failure_label for e in episodes if e.failure_label is not None]
    if not labels:
        unavailable.append("rollout.failure_labels")
        label_counts = None
    else:
        label_counts = {name: labels.count(name) for name in sorted(set(labels))}

    rollout = RolloutSection(
        success_rate=result.success_rate,
        success_se=result.success_se,
        mean_return=result.mean_reward,
        return_se=result.reward_se,
        num_episodes=len(episodes),
        final_goal_distance=dist("final_goal_distance"),
        initial_goal_distance=dist("initial_goal_distance"),
        signed_goal_progress=dist("signed_goal_progress"),
        lateral_drift=dist("lateral_drift"),
        min_tcp_object_distance=dist("min_tcp_object_distance"),
        object_displacement=dist("object_displacement"),
        action_saturation_by_axis=saturation,
        action_variation_by_axis=variation,
        proximity_rate=_distribution([e.near_object_rate for e in episodes], "proximity"),
        phase_occupancy=phase_occupancy,
        phase_transitions=phase_transitions,
        gate_statistics=gate_stats,
        timeout_rate=(
            float(np.mean([float(e.timed_out) for e in episodes])) if episodes else 0.0
        ),
        failure_labels=label_counts,
    )

    validity = (
        ValiditySection.from_validation_report(validation_report)
        if validation_report is not None
        else ValiditySection()
    )
    if validation_report is None:
        unavailable.append("validity")

    fitting = FittingSection.from_bc_stats(bc_stats)
    if not bc_stats:
        unavailable.append("fitting")
    else:
        for name in (
            "validation_loss_by_axis",
            "loss_by_phase",
            "gradient_norms",
            "bound_activity",
        ):
            if getattr(fitting, name) is None:
                unavailable.append(f"fitting.{name}")

    from lares.eval.diagnostics import MOVED_THRESHOLD, REACH_RADIUS, SATURATION_THRESHOLD

    notes = {
        "near_object": (
            "proximity, not verified contact: MetaWorld sets the flag when the "
            "end-effector is within 0.03 m of the object"
        ),
        "timeout_rate": (
            "MetaWorld does not terminate early, so every episode reaches the horizon "
            "and this rate is 1.0 by construction"
        ),
        "failure_label": (
            f"heuristic from geometry; reach radius {REACH_RADIUS} m, moved threshold "
            f"{MOVED_THRESHOLD} m, saturation threshold {SATURATION_THRESHOLD}"
        ),
        "schema": result.schema_id or "none: geometry fields are unavailable",
    }

    return EvaluationReport(
        candidate_id=candidate_id,
        manifest_id=result.manifest_id,
        checkpoint=checkpoint,
        action_mode=result.action_mode,
        validity=validity,
        fitting=fitting,
        rollout=rollout,
        visual_analysis_ids=list(visual_analysis_ids or []),
        episodes=[e.to_dict() for e in episodes],
        unavailable_fields=sorted(set(unavailable)),
        measurement_notes=notes,
    )


def episode_table(report: EvaluationReport, limit: int | None = None) -> str:
    """Compact per-episode table for prompts and terminal output."""
    header = (
        f"{'case':<20}{'succ':>5}{'return':>9}{'final_d':>9}"
        f"{'progress':>10}{'min_tcp_d':>11}  label"
    )
    lines = [header, "-" * len(header)]
    rows = report.episodes if limit is None else report.episodes[:limit]
    for e in rows:
        def fmt(key, width, places=3):
            v = e.get(key)
            return f"{v:>{width}.{places}f}" if isinstance(v, (int, float)) else f"{'n/a':>{width}}"

        lines.append(
            f"{e['case_id']:<20}{e['success']:>5.0f}{e['episode_return']:>9.1f}"
            f"{fmt('final_goal_distance', 9)}{fmt('signed_goal_progress', 10)}"
            f"{fmt('min_tcp_object_distance', 11)}  {e.get('failure_label') or 'n/a'}"
        )
    if limit is not None and len(report.episodes) > limit:
        lines.append(f"... {len(report.episodes) - limit} more episodes")
    return "\n".join(lines)
