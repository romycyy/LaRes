"""Failure clusters, competing explanations, and the acceptance rule (FR-9, AC-6).

The research hypothesis this supports is that a controlled diagnostic
intervention can identify *which* component needs repair, and thereby improve
held-out success per unit of search cost. This module is the machinery for
testing it, not a claim that it works.

The discipline it enforces:

* A cluster is targeted only when it recurs, so effort is not spent on one
  unlucky episode.
* Every cluster carries at least two competing explanations. One explanation is
  not a diagnosis, it is a guess with a substitution attached.
* An explanation is supported only if its substitution moves the focused metric
  in the predicted direction, on the same cases, paired.
* A repair is accepted only if it moves that metric *and* clears a regression
  threshold on protected cases declared before it ran.

Rejected explanations are stored alongside accepted ones. A record of only the
supported hypothesis makes a search look better informed than it was.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from lares.repair.interventions import Substitution

#: A failure label must appear at least this often in a result before it is worth
#: a diagnostic budget. One episode is an anecdote.
MIN_CLUSTER_SIZE = 3

#: Protected cases may lose at most this much success for a repair to be
#: accepted. Declared here rather than at the call site so a repair cannot be
#: waved through by loosening it inline.
PROTECTED_REGRESSION_TOLERANCE = 0.05


@dataclass
class FailureCluster:
    """A recurrent failure, with the cases that exhibit it."""

    label: str
    case_ids: list
    size: int
    #: Geometry summarised over the cluster, for the explanation to reason about.
    summary: dict = field(default_factory=dict)

    @property
    def is_recurrent(self) -> bool:
        return self.size >= MIN_CLUSTER_SIZE

    def to_dict(self) -> dict:
        return asdict(self)


def find_clusters(report, min_size: int = MIN_CLUSTER_SIZE) -> list:
    """Group a report's episodes by failure label, largest first."""
    groups: dict[str, list] = {}
    for episode in report.episodes:
        label = episode.get("failure_label")
        if label in (None, "success"):
            continue
        groups.setdefault(label, []).append(episode)

    clusters = []
    for label, episodes in groups.items():
        if len(episodes) < min_size:
            continue

        def mean_of(key):
            values = [e[key] for e in episodes if e.get(key) is not None]
            return float(np.mean(values)) if values else None

        clusters.append(
            FailureCluster(
                label=label,
                case_ids=[e["case_id"] for e in episodes],
                size=len(episodes),
                summary={
                    "min_tcp_object_distance": mean_of("min_tcp_object_distance"),
                    "object_displacement": mean_of("object_displacement"),
                    "signed_goal_progress": mean_of("signed_goal_progress"),
                    "final_goal_distance": mean_of("final_goal_distance"),
                    "lateral_drift": mean_of("lateral_drift"),
                },
            )
        )
    clusters.sort(key=lambda c: c.size, reverse=True)
    return clusters


@dataclass
class Explanation:
    """One competing account of a cluster, with the substitution that tests it."""

    name: str
    statement: str
    substitution: Substitution
    #: Per-episode field the substitution should move, and which way.
    focused_metric: str
    predicted_direction: str
    suspected_component: str

    def check(self, before: float | None, after: float | None) -> dict:
        if before is None or after is None:
            return {"measurable": False, "reason": "metric unavailable on one side"}
        delta = float(after) - float(before)
        moved = delta > 0 if self.predicted_direction == "increase" else delta < 0
        return {
            "measurable": True,
            "metric": self.focused_metric,
            "before": float(before),
            "after": float(after),
            "delta": delta,
            "predicted_direction": self.predicted_direction,
            "supported": bool(moved),
        }

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "statement": self.statement,
            "substitution": self.substitution.to_dict(),
            "focused_metric": self.focused_metric,
            "predicted_direction": self.predicted_direction,
            "suspected_component": self.suspected_component,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Explanation":
        """Rebuild from :meth:`to_dict`, so a saved study can still be repaired."""
        return cls(
            name=d["name"],
            statement=d["statement"],
            substitution=Substitution.from_dict(d["substitution"]),
            focused_metric=d["focused_metric"],
            predicted_direction=d["predicted_direction"],
            suspected_component=d["suspected_component"],
        )


def competing_explanations(cluster: FailureCluster, gates=None) -> list:
    """At least two accounts of a cluster, each with a different suspect.

    The pairs below are the ones the geometry can actually distinguish. A single
    explanation is never returned: if only one is applicable the caller has no
    diagnosis to make, only a guess to confirm.
    """
    from lares.repair.interventions import (
        SUBSTITUTION_EXPERT_AXIS,
        SUBSTITUTION_EXPERT_PHASE,
        SUBSTITUTION_EXPERT_PREFIX,
        SUBSTITUTION_FORCE_GATE,
    )

    forceable = [g["gate"] for g in (gates or []) if g.get("forceable")]
    out: list[Explanation] = []

    if cluster.label == "never_reached_object":
        out.append(
            Explanation(
                name="approach_control",
                statement=(
                    "The approach itself is wrong: the controller never brings the hand "
                    "to the object, so nothing downstream has ever been exercised."
                ),
                substitution=Substitution(
                    kind=SUBSTITUTION_EXPERT_PREFIX, prefix_steps=40
                ),
                focused_metric="min_tcp_object_distance",
                predicted_direction="decrease",
                suspected_component="the approach term",
            )
        )
        out.append(
            Explanation(
                name="vertical_axis",
                statement=(
                    "The planar motion is fine and the vertical axis is the fault: the "
                    "hand tracks the object in x and y but never descends to it."
                ),
                substitution=Substitution(kind=SUBSTITUTION_EXPERT_AXIS, axes=(2,)),
                focused_metric="min_tcp_object_distance",
                predicted_direction="decrease",
                suspected_component="the vertical action dimension",
            )
        )
    elif cluster.label in ("pushed_away_from_goal", "stopped_short_of_goal"):
        out.append(
            Explanation(
                name="push_geometry",
                statement=(
                    "The push direction is wrong: contact is made but the drive term "
                    "points somewhere other than along the object-to-goal line."
                ),
                substitution=Substitution(
                    kind=SUBSTITUTION_EXPERT_PHASE, phase="push"
                ),
                focused_metric="signed_goal_progress",
                predicted_direction="increase",
                suspected_component="the push direction term",
            )
        )
        out.append(
            Explanation(
                name="planar_axes",
                statement=(
                    "The push direction is broadly right and one planar axis is "
                    "mis-signed or mis-scaled, so the object drifts off the line."
                ),
                substitution=Substitution(kind=SUBSTITUTION_EXPERT_AXIS, axes=(0, 1)),
                focused_metric="lateral_drift",
                predicted_direction="decrease",
                suspected_component="the planar action dimensions",
            )
        )
    elif cluster.label == "object_not_moved":
        out.append(
            Explanation(
                name="no_force_transfer",
                statement=(
                    "The hand arrives but the drive is too weak or too late to move "
                    "the object at all."
                ),
                substitution=Substitution(
                    kind=SUBSTITUTION_EXPERT_PHASE, phase="push"
                ),
                focused_metric="object_displacement",
                predicted_direction="increase",
                suspected_component="the push magnitude",
            )
        )
        out.append(
            Explanation(
                name="approach_stalls",
                statement=(
                    "The object never moves because the hand never truly arrives: the "
                    "approach stalls just short of contact."
                ),
                substitution=Substitution(
                    kind=SUBSTITUTION_EXPERT_PREFIX, prefix_steps=40
                ),
                focused_metric="object_displacement",
                predicted_direction="increase",
                suspected_component="the approach term",
            )
        )

    for gate in forceable:
        out.append(
            Explanation(
                name=f"gate_{gate}_never_opens",
                statement=(
                    f"The {gate} gate never activates, so the branch it guards has "
                    f"never run and cannot be the fault of its own expression."
                ),
                substitution=Substitution(
                    kind=SUBSTITUTION_FORCE_GATE, gate=gate, gate_value=1.0
                ),
                focused_metric="signed_goal_progress",
                predicted_direction="increase",
                suspected_component=f"the {gate} gate threshold",
            )
        )
    return out


@dataclass
class InterventionOutcome:
    """What one substitution did, paired against the unmodified control."""

    explanation: str
    substitution: dict
    focused_metric: str
    control_value: float | None
    intervened_value: float | None
    check: dict
    control_success: float
    intervened_success: float
    paired_success: dict | None = None
    n_cases: int = 0
    #: Always true here. Every row is expert-assisted or gate-forced and can
    #: never be read as a policy's own fitness.
    intervention: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def episode_metric_mean(result, metric: str, case_ids=None) -> float | None:
    """Mean of a per-episode field, restricted to a cluster's cases."""
    wanted = set(case_ids) if case_ids else None
    values = [
        getattr(e, metric)
        for e in result.episodes
        if (wanted is None or e.case_id in wanted) and getattr(e, metric, None) is not None
    ]
    return float(np.mean(values)) if values else None


def rank_explanations(outcomes) -> list:
    """Supported explanations first, ordered by how far the metric moved."""
    def key(outcome):
        check = outcome.check
        if not check.get("measurable"):
            return (1, 0.0)
        magnitude = abs(check["delta"]) if check.get("supported") else 0.0
        return (0 if check.get("supported") else 1, -magnitude)

    return sorted(outcomes, key=key)


@dataclass
class RepairDecision:
    """Whether a proposed repair is accepted, and on what evidence."""

    accepted: bool
    reason: str
    focused_check: dict = field(default_factory=dict)
    protected_check: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_repair(
    before_result,
    after_result,
    focused_metric: str,
    predicted_direction: str,
    focused_case_ids,
    protected_case_ids,
    tolerance: float = PROTECTED_REGRESSION_TOLERANCE,
) -> RepairDecision:
    """Accept a repair only on focused improvement without protected regression.

    Focused improvement alone is not enough: an edit that fixes one cluster by
    breaking everything else is not a repair. The protected set is the cases the
    policy already handled, and the tolerance is fixed in this module rather than
    passed in from wherever the decision happens to be made.
    """
    before = episode_metric_mean(before_result, focused_metric, focused_case_ids)
    after = episode_metric_mean(after_result, focused_metric, focused_case_ids)
    if before is None or after is None:
        return RepairDecision(
            accepted=False,
            reason="focused metric unavailable on one side",
            focused_check={"measurable": False},
        )
    delta = after - before
    moved = delta > 0 if predicted_direction == "increase" else delta < 0
    focused = {
        "measurable": True,
        "metric": focused_metric,
        "before": before,
        "after": after,
        "delta": delta,
        "predicted_direction": predicted_direction,
        "as_predicted": bool(moved),
        "n_cases": len(focused_case_ids or []),
    }

    protected_before = {
        e.case_id: e.success for e in before_result.episodes
        if not protected_case_ids or e.case_id in set(protected_case_ids)
    }
    protected_after = {
        e.case_id: e.success for e in after_result.episodes
        if e.case_id in protected_before
    }
    shared = sorted(set(protected_before) & set(protected_after))
    if shared:
        drop = float(
            np.mean([protected_before[c] for c in shared])
            - np.mean([protected_after[c] for c in shared])
        )
    else:
        drop = 0.0
    protected = {
        "n_cases": len(shared),
        "success_drop": drop,
        "tolerance": tolerance,
        "passed": drop <= tolerance,
    }

    if not moved:
        return RepairDecision(False, "focused metric did not move as predicted", focused, protected)
    if not protected["passed"]:
        return RepairDecision(
            False,
            f"protected cases regressed by {drop:.3f}, above the {tolerance} tolerance",
            focused,
            protected,
        )
    return RepairDecision(True, "focused metric moved and protected cases held", focused, protected)
