"""Merging visual evidence with the numbers (``spec.md`` FR-11, AC-2).

Two rules the spec is explicit about, made mechanical here:

* Visual evidence never decides anything on its own. Every analysis must
  recommend numeric checks, and :func:`merge_analysis` runs them against the
  episode record so a repair prompt carries the measurement beside the claim.
* When the model and the simulator disagree, the conflict is marked. Nothing in
  this module picks a winner. A merge that silently preferred one source would
  hide exactly the case the spec asks to be surfaced.

The conflict test is deliberately narrow. It compares the stage the model says
it saw against the failure label the geometry computed, through a declared
mapping, and it compares any check the model recommended against the value that
check actually has. Free text is not parsed: a claim that cannot be checked is
reported as uncheckable rather than guessed at.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

#: What each numerical failure label implies the video should show. The model is
#: asked for a stage, not a label, so the mapping is one-to-many and a stage
#: outside the list is a disagreement rather than an error.
STAGE_FOR_LABEL = {
    "never_reached_object": ("approach", "hover", "descend", "reach"),
    "object_not_moved": ("contact", "descend", "push", "stall"),
    "pushed_away_from_goal": ("push", "contact"),
    "stopped_short_of_goal": ("push", "contact", "stall"),
    "success": ("push", "goal", "success"),
}

AGREE = "agree"
CONFLICT = "conflict"
UNCHECKABLE = "uncheckable"


@dataclass
class MergedEvidence:
    """One episode's visual analysis, checked against its numbers."""

    case_id: str
    candidate_id: str
    analysis_id: str
    numerical_label: str | None = None
    observed_stage: str = ""
    stage_verdict: str = UNCHECKABLE
    checks: list = field(default_factory=list)
    conflicts: list = field(default_factory=list)
    #: Set when the analysis is the only source for something. Always false:
    #: the field exists so a reader can see the guarantee, not infer it.
    visual_evidence_decided_anything: bool = False

    @property
    def has_conflict(self) -> bool:
        return bool(self.conflicts)

    def to_dict(self) -> dict:
        return asdict(self)


def _episode_value(episode, metric):
    value = getattr(episode, metric, None)
    if value is None and isinstance(episode, dict):
        value = episode.get(metric)
    return value


def merge_analysis(analysis, episode) -> MergedEvidence:
    """Run the recommended checks and mark every disagreement.

    ``episode`` is an :class:`~lares.eval.runner.EpisodeRecord` or a per-episode
    row from a report. Metrics it does not carry are reported as uncheckable,
    never as zero.
    """
    label = _episode_value(episode, "failure_label")
    merged = MergedEvidence(
        case_id=analysis.case_id,
        candidate_id=analysis.candidate_id,
        analysis_id=analysis.analysis_id,
        numerical_label=label,
        observed_stage=analysis.observed_stage,
    )

    expected = STAGE_FOR_LABEL.get(label) if label else None
    if not analysis.observed_stage or expected is None:
        merged.stage_verdict = UNCHECKABLE
    elif analysis.observed_stage.strip().lower() in expected:
        merged.stage_verdict = AGREE
    else:
        merged.stage_verdict = CONFLICT
        merged.conflicts.append({
            "kind": "stage",
            "visual": analysis.observed_stage,
            "numerical_label": label,
            "expected_stages": list(expected),
            "note": "the model and the geometry describe different phases; neither "
                    "is preferred here",
        })

    for metric in analysis.recommended_numeric_checks:
        value = _episode_value(episode, metric)
        merged.checks.append({
            "metric": metric,
            "value": value,
            "status": UNCHECKABLE if value is None else "measured",
        })

    return merged


def merge_all(analyses, episodes) -> list:
    """Merge each analysis with the episode it describes."""
    by_case = {}
    for episode in episodes:
        case_id = _episode_value(episode, "case_id")
        by_case[case_id] = episode
    merged = []
    for analysis in analyses:
        episode = by_case.get(analysis.case_id)
        if episode is None:
            continue
        merged.append(merge_analysis(analysis, episode))
    return merged


def repair_prompt_block(merged, analyses) -> str:
    """The block a repair prompt receives: claims, measurements, conflicts.

    The measurement sits next to the claim so the correction model cannot read
    the visual account without the number that bears on it.
    """
    by_id = {a.analysis_id: a for a in analyses}
    lines = ["## Visual analysis", ""]
    if not merged:
        return "## Visual analysis\n\nNone available for these episodes."
    conflicts = [m for m in merged if m.has_conflict]
    lines.append(
        f"{len(merged)} failed episode(s) analysed, {len(conflicts)} where the model "
        f"and the simulator disagree."
    )
    lines.append(
        "Visual claims are supporting evidence. Nothing below has been confirmed by "
        "a rollout, and none of it can promote, reject or repair a candidate on its own."
    )
    for m in merged:
        analysis = by_id.get(m.analysis_id)
        lines.append("")
        lines.append(f"### {m.case_id}  (numerical label: {m.numerical_label})")
        if analysis is not None:
            lines.append(f"model saw stage {m.observed_stage!r}: {analysis.failure_summary}")
            for claim in analysis.evidence:
                lines.append(
                    f"  - {claim.statement}  [{claim.frame_or_clip_id}, t={claim.timestep}]"
                )
            if analysis.uncertainties:
                lines.append(f"  model is unsure about: {'; '.join(analysis.uncertainties)}")
        for check in m.checks:
            value = check["value"]
            rendered = "not measured" if value is None else f"{value}"
            lines.append(f"  check {check['metric']}: {rendered}")
        for conflict in m.conflicts:
            lines.append(
                f"  CONFLICT ({conflict['kind']}): the model reports "
                f"{conflict['visual']!r} while the geometry labels this "
                f"{conflict['numerical_label']!r}. Both are shown; neither is resolved."
            )
    return "\n".join(lines)
