"""Running visual analysis over a candidate's failed episodes (FR-11).

The one entry point the rest of the pipeline needs. It selects failed
development episodes, records media for each under the fixed rule, asks the
analyst, validates what comes back, merges it with the numbers, and returns the
block a repair prompt would receive together with what it cost.

An analyst that returns nothing is the numeric-diagnostics-only arm of E10, and
it runs this identical code path, so the two sides of that comparison differ in
the model and nothing else.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from lares.vision.analyst import AnalysisRequest
from lares.vision.media import (
    MEDIA_SELECTION_RULE,
    record_episode_media,
    save_media,
    select_media,
)
from lares.vision.merge import merge_all, repair_prompt_block


@dataclass
class AnalysisRun:
    """What one pass of visual analysis produced and what it cost."""

    candidate_id: str
    model_id: str
    media_selection_rule: str = MEDIA_SELECTION_RULE
    episodes_analysed: int = 0
    episodes_rerun: int = 0
    frames_shown: int = 0
    analyses_returned: int = 0
    analyses_rejected: int = 0
    rejection_reasons: list = field(default_factory=list)
    conflicts: int = 0
    latency_seconds: float = 0.0
    unavailable: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def failed_cases(result, limit: int = 3) -> list:
    """The failed episodes worth spending frames on, worst first.

    Ordered by how little progress they made, so a fixed budget is spent on the
    episodes that went most wrong rather than on whichever ran first.
    """
    failures = [e for e in result.episodes if e.success <= 0.0]
    failures.sort(key=lambda e: (e.signed_goal_progress if e.signed_goal_progress
                                 is not None else 0.0))
    return failures[:limit]


def analyse_failures(
    actor,
    env,
    manifest,
    result,
    analyst,
    schema,
    candidate_id: str,
    max_steps: int = 150,
    limit: int = 3,
    media_dir: str = "",
):
    """Analyse a candidate's worst failures. Returns ``(analyses, merged, run)``.

    Episodes are re-run to capture frames, because the evaluation rollout does
    not render. That cost is counted in ``episodes_rerun`` and is the price of
    keeping rendering out of the measurement path.
    """
    started = time.time()
    run = AnalysisRun(candidate_id=candidate_id, model_id=getattr(analyst, "model_id", "none"))
    chosen = failed_cases(result, limit=limit)
    by_case = {c.case_id: c for c in manifest.episodes}

    analyses, episodes = [], []
    for record in chosen:
        case = by_case.get(record.case_id)
        if case is None:
            continue
        _, media = record_episode_media(
            actor, env, case, manifest.split, schema, max_steps=max_steps
        )
        run.episodes_rerun += 1
        items, unavailable = select_media(media, prefix=f"{record.case_id}")
        run.unavailable.extend(f"{record.case_id}: {u}" for u in unavailable)
        if not items:
            continue
        if media_dir:
            items = save_media(media, items, media_dir)
        frames = [media.frames[i.start_timestep] for i in items
                  if i.start_timestep < len(media.frames)]
        run.episodes_analysed += 1
        run.frames_shown += len(items)

        analysis = analyst.analyse(AnalysisRequest(
            candidate_id=candidate_id,
            case_id=record.case_id,
            media=items,
            frames=frames,
            metadata={
                "success": record.success,
                "length": record.length,
                "timed_out": record.timed_out,
            },
        ))
        if analysis is None:
            continue
        try:
            analysis.validate()
        except ValueError as exc:
            # A model that cites a frame it was never shown is telling us
            # something. It is recorded and dropped, never repaired into shape.
            run.analyses_rejected += 1
            run.rejection_reasons.append(f"{record.case_id}: {exc}")
            continue
        run.analyses_returned += 1
        analyses.append(analysis)
        episodes.append(record)

    merged = merge_all(analyses, episodes)
    run.conflicts = sum(1 for m in merged if m.has_conflict)
    run.latency_seconds = time.time() - started
    return analyses, merged, run


def evidence_block(merged, analyses) -> str:
    """The block for a repair prompt, or a plain statement that there is none."""
    return repair_prompt_block(merged, analyses)
