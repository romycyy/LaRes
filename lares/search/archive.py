"""Behaviour-based candidate archive (``spec.md`` FR-10, AC-4).

Keeping only the best-scoring candidate throws away the two things a search most
needs later: the simplest thing that worked, and something that fails
differently. The archive holds three slots.

``robust``
    Highest development success, ties broken by goal progress.

``simplest``
    Fewest parameters among candidates that are competitive with the robust
    slot, so complexity has to earn itself.

``distinct``
    The candidate whose *behaviour* is furthest from the robust one. Distance is
    measured on the rollout failure profile and geometry, not on source text:
    two programs that read differently but fail identically are not diverse, and
    two that read alike but fail differently are.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from lares.eval.diagnostics import FAILURE_LABELS

SLOT_ROBUST = "robust"
SLOT_SIMPLEST = "simplest"
SLOT_DISTINCT = "distinct"
SLOTS = (SLOT_ROBUST, SLOT_SIMPLEST, SLOT_DISTINCT)

#: A candidate counts as competitive with the leader if its success is within
#: this much. Declared here rather than inline so the simplicity slot cannot be
#: quietly loosened to admit a favourite.
COMPETITIVE_SUCCESS_MARGIN = 0.05


def behavior_descriptor(report) -> np.ndarray:
    """A fixed-length behavioural signature of one evaluated candidate.

    The failure-label distribution plus a few normalised geometry statistics.
    Two candidates that reach the object and stall look alike here; one that
    never approaches and one that overshoots do not.
    """
    labels = report.rollout.failure_labels or {}
    total = sum(labels.values()) or 1
    profile = [labels.get(name, 0) / total for name in FAILURE_LABELS]

    def stat(dist, scale):
        if not dist:
            return 0.0
        return float(np.clip(dist["mean"] / scale, -2.0, 2.0))

    geometry = [
        stat(report.rollout.final_goal_distance, 0.3),
        stat(report.rollout.signed_goal_progress, 0.2),
        stat(report.rollout.lateral_drift, 0.1),
        stat(report.rollout.min_tcp_object_distance, 0.2),
        stat(report.rollout.object_displacement, 0.2),
    ]
    saturation = report.rollout.action_saturation_by_axis or []
    geometry.append(float(np.mean(saturation)) if saturation else 0.0)
    return np.asarray(profile + geometry, dtype=float)


def behavioral_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two behavioural signatures."""
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


@dataclass
class ArchiveEntry:
    candidate_id: str
    generation: int
    success_rate: float
    signed_goal_progress: float | None
    num_parameters: int
    source_hash: str
    descriptor: list = field(default_factory=list)
    failure_labels: dict | None = None
    slots: list = field(default_factory=list)
    #: Intervention results are recorded but never occupy a slot.
    intervention: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class Archive:
    """Retains a robust, a simple and a behaviourally distinct candidate."""

    def __init__(self, competitive_margin: float = COMPETITIVE_SUCCESS_MARGIN):
        self.competitive_margin = float(competitive_margin)
        self.entries: list[ArchiveEntry] = []

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, candidate_id, generation, report, num_parameters, source_hash,
            intervention: bool = False) -> ArchiveEntry:
        """Record one evaluated candidate. Every candidate is kept, not just winners."""
        progress = report.rollout.signed_goal_progress
        entry = ArchiveEntry(
            candidate_id=candidate_id,
            generation=int(generation),
            success_rate=float(report.rollout.success_rate),
            signed_goal_progress=float(progress["mean"]) if progress else None,
            num_parameters=int(num_parameters),
            source_hash=str(source_hash),
            descriptor=[float(x) for x in behavior_descriptor(report)],
            failure_labels=report.rollout.failure_labels,
            intervention=bool(intervention),
        )
        self.entries.append(entry)
        self._recompute_slots()
        return entry

    # -- slot assignment ---------------------------------------------------

    @property
    def rankable(self) -> list:
        """Entries eligible for a slot. Intervention results are excluded."""
        return [e for e in self.entries if not e.intervention]

    def _recompute_slots(self) -> None:
        for entry in self.entries:
            entry.slots = []
        pool = self.rankable
        if not pool:
            return

        robust = max(
            pool,
            key=lambda e: (e.success_rate, e.signed_goal_progress or -math.inf),
        )
        robust.slots.append(SLOT_ROBUST)

        competitive = [
            e for e in pool if e.success_rate >= robust.success_rate - self.competitive_margin
        ]
        simplest = min(competitive, key=lambda e: (e.num_parameters, -e.success_rate))
        if SLOT_SIMPLEST not in simplest.slots:
            simplest.slots.append(SLOT_SIMPLEST)

        others = [e for e in pool if e is not robust]
        if others:
            reference = np.asarray(robust.descriptor)
            distinct = max(
                others, key=lambda e: behavioral_distance(reference, e.descriptor)
            )
            if behavioral_distance(reference, distinct.descriptor) > 1e-9:
                distinct.slots.append(SLOT_DISTINCT)

    def slot(self, name: str) -> ArchiveEntry | None:
        for entry in self.entries:
            if name in entry.slots:
                return entry
        return None

    def occupied_slots(self) -> dict:
        return {name: self.slot(name) for name in SLOTS}

    def to_dict(self) -> dict:
        return {
            "competitive_margin": self.competitive_margin,
            "num_entries": len(self.entries),
            "slots": {
                name: (entry.candidate_id if entry else None)
                for name, entry in self.occupied_slots().items()
            },
            "entries": [e.to_dict() for e in self.entries],
        }

    def summary(self) -> str:
        lines = [
            f"{'slot':<10}{'candidate':<18}{'success':>9}{'progress':>10}{'params':>8}  failures",
            "-" * 78,
        ]
        for name in SLOTS:
            entry = self.slot(name)
            if entry is None:
                lines.append(f"{name:<10}{'unoccupied':<18}")
                continue
            progress = (
                f"{entry.signed_goal_progress:>10.4f}"
                if entry.signed_goal_progress is not None
                else f"{'n/a':>10}"
            )
            labels = entry.failure_labels or {}
            top = max(labels.items(), key=lambda kv: kv[1])[0] if labels else "n/a"
            lines.append(
                f"{name:<10}{entry.candidate_id:<18}{entry.success_rate:>9.3f}{progress}"
                f"{entry.num_parameters:>8}  {top}"
            )
        excluded = [e for e in self.entries if e.intervention]
        if excluded:
            lines.append(
                f"\n{len(excluded)} intervention result(s) recorded and excluded from "
                f"every slot: {[e.candidate_id for e in excluded]}"
            )
        return "\n".join(lines)
