"""Staged development budget and promotion rules (``spec.md`` section 12, FR-4).

A candidate is first screened on a small fixed subset. Screening only decides which
single challenger earns a promotion test; it never replaces the incumbent.

Promotion requires real evidence (decided 2026-09-14, after the debug-cycle run
promoted twice on differences held-out data did not support). A challenger
replaces the incumbent only when, on a *promotion manifest* used for neither
screening nor checkpoint selection:

1. the paired success difference against the incumbent is strictly positive, and
2. the lower bound of its paired t interval is above zero, at 95%, or at the
   Holm-adjusted level when several challengers are tested in one round.

Tied success never promotes. Return and goal progress may rank candidates or admit
them to the archive, but cannot replace the incumbent. Before any promotion the
incumbent is the documented 0.00-success baseline, so the first challenger must
show success whose interval excludes zero.
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


#: Reasons attached to a promotion test. Stable strings: records key on them.
PROMOTED_INTERVAL_ABOVE_ZERO = "paired_success_interval_above_zero"
REJECTED_TIED_SUCCESS = "tied_paired_success"
REJECTED_LOWER_SUCCESS = "lower_paired_success"
REJECTED_INTERVAL_INCLUDES_ZERO = "paired_success_interval_includes_zero"
REJECTED_TOO_FEW_CASES = "too_few_paired_cases"
REJECTED_HOLM_STOPPED = "holm_stopped_at_an_earlier_failure"
REJECTED_NOT_BEST_PASSING = "passed_but_not_the_best_passing_challenger"
NOT_BEST_SCREENED_CHALLENGER = "not_the_best_screened_challenger"

#: The incumbent before any promotion. The saved incumbent reproduces at 0.00
#: success (``RESEARCH_HANDOFF.md``), so a first challenger is paired against zero.
NULL_INCUMBENT_ID = "null_incumbent_zero_success"
PROMOTION_ALPHA = 0.05
PROMOTION_RULE = (
    "replace the incumbent only if, on a promotion manifest not used for screening or checkpoint "
    "selection, the paired success difference is strictly positive and the lower bound of its paired "
    "t interval (95%, Holm-adjusted across challengers tested together) is above zero; tied success never "
    "promotes; return and progress never replace the incumbent"
)


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
    evidence_manifest_id: str | None = None
    excluded_manifest_ids: list = field(default_factory=list)
    interval_level: float = 1.0 - PROMOTION_ALPHA
    holm: dict | None = None
    rule: str = PROMOTION_RULE

    def to_dict(self) -> dict:
        return asdict(self)


def paired_success_test(contender_result, incumbent_result=None, level: float = 0.95) -> dict:
    """Paired t test of contender minus incumbent success over shared cases.

    With no incumbent, the difference is against zero success on every case.
    """
    from scipy import stats

    if incumbent_result is None:
        diffs = [float(e.success) for e in contender_result.episodes]
    else:
        by_case = {e.case_id: float(e.success) for e in incumbent_result.episodes}
        diffs = [float(e.success) - by_case[e.case_id] for e in contender_result.episodes if e.case_id in by_case]
        if not diffs:
            raise ValueError("results share no cases; they are not paired")
    arr = np.asarray(diffs, dtype=float)
    n = int(arr.size)
    mean = float(arr.mean()) if n else 0.0
    out = {
        "metric": "success",
        "method": "paired t",
        "n_paired": n,
        "mean_difference": mean,
        "wins": int((arr > 0).sum()),
        "losses": int((arr < 0).sum()),
        "ties": int((arr == 0).sum()),
        "level": float(level),
    }
    if n < 2:
        out.update(se=None, ci_low=None, ci_high=None, ci95_low=None, ci95_high=None, p_value=1.0)
        return out
    se = float(arr.std(ddof=1) / np.sqrt(n))
    if se == 0.0:
        p_value = 0.0 if mean != 0.0 else 1.0
        low = high = low95 = high95 = mean
    else:
        p_value = float(2.0 * stats.t.sf(abs(mean / se), n - 1))
        half = float(stats.t.ppf(0.5 + level / 2.0, n - 1) * se)
        half95 = float(stats.t.ppf(0.975, n - 1) * se)
        low, high, low95, high95 = mean - half, mean + half, mean - half95, mean + half95
    out.update(se=se, ci_low=low, ci_high=high, ci95_low=low95, ci95_high=high95, p_value=p_value)
    return out


def compare_to_incumbent(
    contender_result,
    incumbent_result,
    contender_id: str,
    incumbent_id: str | None,
    excluded_manifest_ids=(),
    level: float = 1.0 - PROMOTION_ALPHA,
) -> PromotionDecision:
    """Paired promotion test of one contender against the current incumbent.

    Uncertainty is estimated for the paired difference itself, with a
    small-sample t interval. The evidence manifest must be the incumbent's and
    must not be one used for screening or checkpoint selection: evidence reused
    from those decisions is refused rather than weighed.
    """
    from lares.eval.runner import paired_difference

    evidence = contender_result.manifest_id
    excluded = [m for m in excluded_manifest_ids if m]
    if evidence in excluded:
        raise ValueError(
            f"promotion evidence comes from {evidence!r}, which was used for screening or checkpoint "
            f"selection; promotion needs a manifest that took no part in those decisions"
        )
    if incumbent_result is not None and incumbent_result.manifest_id != evidence:
        raise ValueError(
            f"contender scored on {evidence!r} but the incumbent on {incumbent_result.manifest_id!r}; "
            f"promotion is decided on paired cases only"
        )

    success = paired_success_test(contender_result, incumbent_result, level)
    returns = paired_difference(contender_result, incumbent_result, "reward") if incumbent_result else None
    if success["n_paired"] < 2:
        promoted, reason = False, REJECTED_TOO_FEW_CASES
    elif success["mean_difference"] == 0.0:
        promoted, reason = False, REJECTED_TIED_SUCCESS
    elif success["mean_difference"] < 0.0:
        promoted, reason = False, REJECTED_LOWER_SUCCESS
    elif success["ci_low"] is None or success["ci_low"] <= 0.0:
        promoted, reason = False, REJECTED_INTERVAL_INCLUDES_ZERO
    else:
        promoted, reason = True, PROMOTED_INTERVAL_ABOVE_ZERO

    return PromotionDecision(
        candidate_id=contender_id,
        incumbent_id=incumbent_id if incumbent_result is not None else NULL_INCUMBENT_ID,
        promoted=promoted,
        reason=reason,
        paired_success=success,
        paired_return=returns,
        contender={"success_rate": contender_result.success_rate, "mean_return": contender_result.mean_reward},
        incumbent=(
            {"success_rate": incumbent_result.success_rate, "mean_return": incumbent_result.mean_reward}
            if incumbent_result is not None
            else {"success_rate": 0.0, "mean_return": None}
        ),
        evidence_manifest_id=evidence,
        excluded_manifest_ids=excluded,
        interval_level=float(level),
    )


def holm_promotion(
    contenders,
    incumbent_result,
    incumbent_id: str | None,
    excluded_manifest_ids=(),
    alpha: float = PROMOTION_ALPHA,
) -> list:
    """Promotion tests for several challengers at once, with Holm's step-down correction.

    ``contenders`` is ``[(candidate_id, result), ...]``. Tests are ordered by
    p-value; the i-th (from 1) is judged at interval level ``1 - alpha / (m - i + 1)``,
    and once one fails every later one fails too. At most one challenger is
    promoted: the passing one with the largest paired difference. With a single
    contender this is exactly :func:`compare_to_incumbent` at 95%.

    Returns decisions in the input order.
    """
    contenders = list(contenders)
    m = len(contenders)
    if m == 0:
        return []
    raw = {cid: paired_success_test(res, incumbent_result, 1.0 - alpha)["p_value"] for cid, res in contenders}
    order = sorted(range(m), key=lambda i: (raw[contenders[i][0]], i))
    decisions = [None] * m
    still_testing = True
    for rank, index in enumerate(order, start=1):
        cid, res = contenders[index]
        level = 1.0 - alpha / (m - rank + 1)
        decision = compare_to_incumbent(res, incumbent_result, cid, incumbent_id, excluded_manifest_ids, level)
        decision.holm = {"m": m, "rank": rank, "alpha": alpha, "level": level, "raw_p_value": raw[cid]}
        if not still_testing and decision.promoted:
            decision.promoted, decision.reason = False, REJECTED_HOLM_STOPPED
        elif not decision.promoted:
            still_testing = False
        decisions[index] = decision
    passing = [d for d in decisions if d.promoted]
    if len(passing) > 1:
        best = max(passing, key=lambda d: (d.paired_success["mean_difference"], d.paired_success["ci_low"]))
        for d in passing:
            if d is not best:
                d.promoted, d.reason = False, REJECTED_NOT_BEST_PASSING
    return decisions


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
