"""Evidence handed to the generator between generations (``spec.md`` FR-5, AC-4).

What the loop used to send was two floats: a success rate and a mean return.
A policy that reads the puck's orientation as its position scores about the same
as one with a merely bad control law, so those two numbers cannot distinguish a
structural mistake from a tuning one.

What it sends now, in order of usefulness:

1. Full code for two *informative* parents, chosen to differ rather than to both
   be good, so the model can attribute an outcome to a difference.
2. A per-candidate diagnostic table for the whole population, ranked.
3. The failure-label distribution and the geometry that produced it.
4. Parameters the executed action cannot respond to, which are dead weight the
   model can remove.
5. Errors from candidates that never compiled, so the same mistake is not made
   twice.

Phase 3 measured that validation loss does not predict development success on
this task, correlating at +0.19 in the wrong direction. Loss is therefore
reported as context and explicitly marked as a poor predictor rather than
presented as a quality signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Reported alongside the fitting loss so the generator is not misled by it.
LOSS_CAVEAT = (
    "Note: across 144 fitted configurations on this task, behavioural-cloning "
    "validation loss correlated with development success at Pearson +0.19 "
    "(p=0.023). That is the wrong sign: a better fit to the expert did not give a "
    "better controller. Treat the loss as context, not as evidence of quality."
)


@dataclass
class CandidateEvidence:
    """One candidate's contribution to the feedback block."""

    candidate_id: str
    code: str
    report: object
    zero_shot_report: object = None
    num_parameters: int = 0
    dead_parameters: list = field(default_factory=list)
    error: str = ""

    @property
    def success(self) -> float:
        return self.report.rollout.success_rate if self.report else 0.0

    @property
    def progress(self) -> float | None:
        dist = self.report.rollout.signed_goal_progress if self.report else None
        return dist["mean"] if dist else None

    @property
    def dominant_failure(self) -> str:
        labels = (self.report.rollout.failure_labels if self.report else None) or {}
        if not labels:
            return "unavailable"
        return max(labels.items(), key=lambda kv: kv[1])[0]


def choose_informative_parents(evidence, limit: int = 2) -> list:
    """Pick parents that differ, not merely parents that scored well.

    The best two candidates of a generation are often near-identical, and two
    near-identical parents support no inference about what mattered. This takes
    the leader, then the candidate whose behaviour is furthest from it.
    """
    scored = [e for e in evidence if e.report is not None]
    if not scored:
        return []
    ranked = sorted(scored, key=lambda e: (e.success, e.progress or -1e9), reverse=True)
    if len(ranked) <= limit:
        return ranked[:limit]

    from lares.search.archive import behavior_descriptor, behavioral_distance

    leader = ranked[0]
    reference = behavior_descriptor(leader.report)
    rest = sorted(
        ranked[1:],
        key=lambda e: behavioral_distance(reference, behavior_descriptor(e.report)),
        reverse=True,
    )
    return [leader] + rest[: limit - 1]


def _distribution_line(label, dist, scale=""):
    if not dist:
        return f"  {label:<26} unavailable"
    return (
        f"  {label:<26} mean {dist['mean']:+.4f}   p25 {dist['p25']:+.4f}   "
        f"p75 {dist['p75']:+.4f}{scale}"
    )


def diagnostics_block(evidence: CandidateEvidence) -> str:
    """The numeric evidence for one candidate."""
    report = evidence.report
    if report is None:
        return f"  no rollout: {evidence.error or 'candidate never reached evaluation'}"
    rollout = report.rollout
    lines = [
        f"  success {rollout.success_rate:.3f} +- {rollout.success_se:.3f}   "
        f"mean return {rollout.mean_return:.1f}   over {rollout.num_episodes} cases",
        f"  failure labels: {rollout.failure_labels or 'unavailable'}",
        _distribution_line("final goal distance (m)", rollout.final_goal_distance),
        _distribution_line("signed goal progress (m)", rollout.signed_goal_progress),
        _distribution_line("lateral drift (m)", rollout.lateral_drift),
        _distribution_line("closest approach (m)", rollout.min_tcp_object_distance),
        _distribution_line("object displacement (m)", rollout.object_displacement),
    ]
    if rollout.action_saturation_by_axis:
        lines.append(
            "  action saturation by axis: "
            + ", ".join(f"{x:.2f}" for x in rollout.action_saturation_by_axis)
        )
    if rollout.phase_occupancy:
        lines.append("  gate occupancy above 0.5:")
        for name, value in sorted(rollout.phase_occupancy.items()):
            transitions = (rollout.phase_transitions or {}).get(name)
            gates = (rollout.gate_statistics or {}).get(name, {})
            crossed = gates.get("episodes_with_crossing")
            measured = gates.get("episodes_measured")
            detail = f"crossed in {crossed}/{measured} episodes" if measured else ""
            lines.append(
                f"    {name:<20} occupancy {value:.3f}   transitions {transitions}   {detail}"
            )
    else:
        lines.append("  gate telemetry: not exposed by this policy")
    if evidence.dead_parameters:
        lines.append(
            f"  parameters the executed action does NOT respond to: "
            f"{evidence.dead_parameters}. These are dead weight; remove them or "
            f"connect them to the action."
        )
    if evidence.zero_shot_report is not None:
        zero = evidence.zero_shot_report.rollout
        lines.append(
            f"  before fitting: success {zero.success_rate:.3f}, "
            f"return {zero.mean_return:.1f}. "
            + (
                "Fitting made it worse, so the structure was not the problem."
                if zero.success_rate > rollout.success_rate
                else "Fitting did not make it worse."
            )
        )
    if report.fitting and report.fitting.validation_loss_end is not None:
        lines.append(
            f"  fitting: train {report.fitting.train_loss_end:.5f} -> "
            f"validation {report.fitting.validation_loss_end:.5f}"
        )
    return "\n".join(lines)


def population_table(evidence) -> str:
    """Every candidate, ranked, one line each."""
    ranked = sorted(
        evidence, key=lambda e: (e.success, e.progress or -1e9), reverse=True
    )
    header = (
        f"{'candidate':<18}{'success':>9}{'progress':>10}{'params':>8}"
        f"{'dead':>6}  dominant failure"
    )
    lines = [header, "-" * len(header)]
    for e in ranked:
        if e.report is None:
            lines.append(f"{e.candidate_id:<18}{'invalid':>9}{'':>10}{'':>8}{'':>6}  {e.error[:40]}")
            continue
        progress = f"{e.progress:>10.4f}" if e.progress is not None else f"{'n/a':>10}"
        lines.append(
            f"{e.candidate_id:<18}{e.success:>9.3f}{progress}{e.num_parameters:>8}"
            f"{len(e.dead_parameters):>6}  {e.dominant_failure}"
        )
    return "\n".join(lines)


def build_feedback(
    evidence,
    generation: int,
    manifest_id: str,
    errors=None,
    history=None,
    parent_limit: int = 2,
    archive=None,
) -> str:
    """Assemble the whole evidence block for the next generation's prompt."""
    parents = choose_informative_parents(evidence, limit=parent_limit)
    parts = [
        f"## Evidence from generation {generation}",
        f"All candidates were scored on the identical ordered case list "
        f"`{manifest_id}`, so differences between them are differences in the "
        f"policy and not in which placements they drew.",
        "",
        "### Population",
        population_table(evidence),
        "",
    ]

    if parents:
        parts.append("### The two most informative candidates, in full")
        parts.append(
            "These two are shown together because they differ behaviourally, not "
            "because both scored well. Compare them to work out which structural "
            "difference produced the difference in outcome."
        )
        for i, parent in enumerate(parents):
            parts.append("")
            parts.append(f"#### Parent {i + 1}: {parent.candidate_id}")
            parts.append(diagnostics_block(parent))
            parts.append("")
            parts.append("```python")
            parts.append(parent.code.strip())
            parts.append("```")
        parts.append("")

    others = [e for e in evidence if e not in parents and e.report is not None]
    if others:
        parts.append("### Diagnostics for the remaining candidates")
        for e in others:
            parts.append(f"\n#### {e.candidate_id}")
            parts.append(diagnostics_block(e))
        parts.append("")

    invalid = [e for e in evidence if e.report is None]
    if invalid or errors:
        parts.append("### Candidates that never reached evaluation")
        parts.append(
            "Do not repeat these mistakes. Each one cost a slot in the population."
        )
        for e in invalid:
            parts.append(f"  {e.candidate_id}: {e.error}")
        for err in errors or []:
            parts.append(f"  {err}")
        parts.append("")

    if archive is not None and len(archive):
        parts.append("### Archive")
        parts.append(
            "Retained across all generations so far: the most robust, the simplest "
            "competitive, and the most behaviourally distinct candidate."
        )
        parts.append(archive.summary())
        parts.append("")

    if history:
        parts.append("### History")
        for line in history:
            parts.append(f"  {line}")
        parts.append("")

    parts.append("### How to read this")
    parts.append(LOSS_CAVEAT)
    parts.append(
        "Success is the outcome that matters. When every candidate scores zero, "
        "signed goal progress is the discriminating measure: positive means the "
        "object ended closer to the goal than it started, negative means the "
        "controller pushed it away. `never_reached_object` and "
        "`pushed_away_from_goal` are different faults and need different repairs."
    )
    return "\n".join(parts)
