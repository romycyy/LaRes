"""Staged development budget and promotion rules (``spec.md`` section 12, FR-4).

A candidate is first screened on a small fixed subset, then, only if it clears a
rule declared *before* the generation ran, re-scored on the larger subset. The
rule is a stored object rather than an inline condition so the record shows what
threshold a promotion was judged against, not just that it happened.

The screening subset is a prefix of the expanded one, so a candidate promoted on
ten cases is re-scored on those same ten plus twenty more. Screening and
expansion therefore stay paired and no candidate is judged on a fresh draw.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

#: Reasons a candidate did or did not advance past screening. Stable strings:
#: experiment records key on them.
PROMOTED_ON_SUCCESS = "success_above_threshold"
PROMOTED_ON_PROGRESS = "top_k_by_goal_progress"
REJECTED_INVALID = "failed_validity_gates"
REJECTED_BELOW_RULE = "below_screening_rule"


@dataclass
class PromotionPolicy:
    """The development-budget policy, fixed before a generation is evaluated.

    ``spec.md`` section 9 sets ten screening cases and thirty expanded cases as
    starting values, to be changed only by a documented protocol revision.
    """

    screen_episodes: int = 10
    expanded_episodes: int = 30
    #: A candidate clearing this success rate on the screen always advances.
    screen_min_success: float = 1e-9
    #: When nothing clears on success, this many candidates advance on the best
    #: secondary evidence instead. Without it a generation where every candidate
    #: scores zero would advance nobody and the search would learn nothing.
    fallback_top_k: int = 2
    #: Secondary ranking used by the fallback, in order of preference.
    fallback_metric: str = "signed_goal_progress"
    tie_break: str = "mean_return"

    def to_dict(self) -> dict:
        return asdict(self)

    # -- ranking ----------------------------------------------------------

    @staticmethod
    def _progress(report) -> float:
        """Mean signed goal progress, falling back to return when unavailable."""
        dist = report.rollout.signed_goal_progress
        if dist is not None:
            return float(dist["mean"])
        return float(report.rollout.mean_return)

    def screen(self, entries) -> list[dict]:
        """Decide which screened candidates earn the expanded evaluation.

        Args:
            entries: sequence of ``{"candidate_id", "report", "valid"}`` dicts.

        Returns:
            One decision dict per entry, in the input order, each carrying
            ``promoted`` and the ``reason`` string that justified it.
        """
        decisions = []
        for e in entries:
            decisions.append(
                {
                    "candidate_id": e["candidate_id"],
                    "success_rate": e["report"].rollout.success_rate,
                    "progress": self._progress(e["report"]),
                    "mean_return": e["report"].rollout.mean_return,
                    "valid": bool(e.get("valid", True)),
                    "promoted": False,
                    "reason": None,
                }
            )

        for d in decisions:
            if not d["valid"]:
                d["reason"] = REJECTED_INVALID
            elif d["success_rate"] >= self.screen_min_success:
                d["promoted"] = True
                d["reason"] = PROMOTED_ON_SUCCESS

        if not any(d["promoted"] for d in decisions) and self.fallback_top_k > 0:
            eligible = [d for d in decisions if d["valid"]]
            eligible.sort(key=lambda d: (d["progress"], d["mean_return"]), reverse=True)
            for d in eligible[: self.fallback_top_k]:
                d["promoted"] = True
                d["reason"] = PROMOTED_ON_PROGRESS

        for d in decisions:
            if not d["promoted"] and d["reason"] is None:
                d["reason"] = REJECTED_BELOW_RULE
        return decisions


@dataclass
class PromotionDecision:
    """Whether a contender replaces the incumbent, and on what evidence."""

    candidate_id: str
    incumbent_id: str | None
    promoted: bool
    reason: str
    paired_success: dict | None = None
    paired_return: dict | None = None
    contender: dict = field(default_factory=dict)
    incumbent: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def compare_to_incumbent(
    contender_result,
    incumbent_result,
    contender_id: str,
    incumbent_id: str | None,
) -> PromotionDecision:
    """Paired comparison of a contender against the current best.

    Uncertainty is estimated for the paired difference itself. Two separately
    computed intervals overlapping, or not overlapping, says nothing about the
    difference (``spec.md`` section 12, rule 5).

    Promotion is decided on the success difference, with mean return breaking a
    tie only when the success difference is exactly zero.
    """
    from lares.eval.runner import paired_difference

    if incumbent_result is None:
        return PromotionDecision(
            candidate_id=contender_id,
            incumbent_id=None,
            promoted=True,
            reason="no_incumbent",
            contender={
                "success_rate": contender_result.success_rate,
                "mean_return": contender_result.mean_reward,
            },
        )

    success = paired_difference(contender_result, incumbent_result, "success")
    returns = paired_difference(contender_result, incumbent_result, "reward")

    if success["mean_difference"] > 0:
        promoted, reason = True, "higher_paired_success"
    elif success["mean_difference"] < 0:
        promoted, reason = False, "lower_paired_success"
    elif returns["mean_difference"] > 0:
        promoted, reason = True, "tied_success_higher_paired_return"
    else:
        promoted, reason = False, "tied_success_no_better_return"

    return PromotionDecision(
        candidate_id=contender_id,
        incumbent_id=incumbent_id,
        promoted=promoted,
        reason=reason,
        paired_success=success,
        paired_return=returns,
        contender={
            "success_rate": contender_result.success_rate,
            "mean_return": contender_result.mean_reward,
        },
        incumbent={
            "success_rate": incumbent_result.success_rate,
            "mean_return": incumbent_result.mean_reward,
        },
    )


def summarise_decisions(decisions) -> str:
    """One line per screened candidate, for terminal output and the record."""
    lines = [
        f"{'candidate':<16}{'success':>9}{'progress':>10}{'return':>10}  decision",
        "-" * 62,
    ]
    for d in decisions:
        mark = "promote" if d["promoted"] else "hold"
        lines.append(
            f"{d['candidate_id']:<16}{d['success_rate']:>9.3f}{d['progress']:>10.4f}"
            f"{d['mean_return']:>10.1f}  {mark} ({d['reason']})"
        )
    return "\n".join(lines)
