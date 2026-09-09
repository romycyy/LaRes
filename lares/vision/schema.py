"""The ``VisualFailureAnalysis`` contract (``spec.md`` 7.4, FR-11, AC-2).

A vision model will describe things it did not see. The contract is what makes
that checkable rather than a matter of trust: every claim must name a frame that
was actually shown and a timestep inside that frame's span, and every follow-up
it recommends must name a quantity the pipeline measures. An analysis that
cannot be wrong teaches nothing, which is the same rule
:class:`~lares.search.schemas.MutationProposal` applies to predictions.

Nothing here promotes or rejects a candidate. The analysis is evidence for a
repair prompt, and :mod:`lares.vision.merge` is where it meets the numbers.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field

#: Bumped whenever the prompt changes, so an analysis can be tied to the words
#: that produced it. Recorded on every analysis.
PROMPT_VERSION = "v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass
class MediaItem:
    """One frame or short clip that was actually shown to the model."""

    frame_or_clip_id: str
    start_timestep: int
    end_timestep: int
    #: Why the selection rule chose this one. Part of the audit trail.
    reason: str = ""
    path: str = ""

    def covers(self, timestep: int) -> bool:
        return self.start_timestep <= int(timestep) <= self.end_timestep

    def validate(self):
        _require(self.frame_or_clip_id, "a media item needs an id")
        _require(
            self.end_timestep >= self.start_timestep,
            f"{self.frame_or_clip_id}: end_timestep {self.end_timestep} precedes "
            f"start_timestep {self.start_timestep}",
        )
        return True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Evidence:
    """One claim, tied to the frame and timestep it was read from."""

    statement: str
    frame_or_clip_id: str
    timestep: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VisualFailureAnalysis:
    """What the vision model said, in a form that can be checked."""

    analysis_id: str
    candidate_id: str
    case_id: str
    model_id: str
    prompt_version: str
    media: list = field(default_factory=list)
    observed_stage: str = ""
    failure_summary: str = ""
    evidence: list = field(default_factory=list)
    candidate_hypotheses: list = field(default_factory=list)
    uncertainties: list = field(default_factory=list)
    recommended_numeric_checks: list = field(default_factory=list)
    #: Model and runtime provenance the spec asks to be kept for audit.
    runtime: dict = field(default_factory=dict)

    def validate(self, measurable=None):
        for name in ("analysis_id", "candidate_id", "case_id", "model_id", "prompt_version"):
            _require(getattr(self, name), f"an analysis needs {name}")
        _require(self.media, "an analysis must name the media it was shown")
        _require(self.failure_summary, "an analysis must say what it thinks went wrong")
        _require(self.evidence, "an analysis with no cited evidence is an assertion")

        by_id = {}
        for item in self.media:
            item.validate()
            _require(
                item.frame_or_clip_id not in by_id,
                f"duplicate media id {item.frame_or_clip_id}",
            )
            by_id[item.frame_or_clip_id] = item

        for claim in self.evidence:
            _require(claim.statement, "every piece of evidence needs a statement")
            _require(
                claim.frame_or_clip_id in by_id,
                f"evidence cites {claim.frame_or_clip_id!r}, which was never shown to the "
                f"model; it names {sorted(by_id)}",
            )
            _require(
                by_id[claim.frame_or_clip_id].covers(claim.timestep),
                f"evidence cites timestep {claim.timestep} on "
                f"{claim.frame_or_clip_id!r}, which spans "
                f"{by_id[claim.frame_or_clip_id].start_timestep} to "
                f"{by_id[claim.frame_or_clip_id].end_timestep}",
            )

        _require(
            self.recommended_numeric_checks,
            "an analysis must recommend at least one numeric check; visual evidence "
            "cannot promote, reject or repair a candidate on its own",
        )
        known = set(measurable) if measurable is not None else _measurable()
        unknown = [c for c in self.recommended_numeric_checks if c not in known]
        _require(
            not unknown,
            f"recommended checks name quantities the pipeline does not measure: "
            f"{unknown}. A check that cannot be run cannot confirm or refute the "
            f"analysis.",
        )
        return True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["media"] = [m.to_dict() for m in self.media]
        d["evidence"] = [e.to_dict() for e in self.evidence]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "VisualFailureAnalysis":
        known = set(cls.__dataclass_fields__)
        payload = {k: v for k, v in d.items() if k in known}
        payload["media"] = [MediaItem(**m) for m in d.get("media", [])]
        payload["evidence"] = [Evidence(**e) for e in d.get("evidence", [])]
        return cls(**payload)

    def save(self, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.analysis_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
        return path


def _measurable() -> set:
    from lares.search.schemas import measurable_metrics

    return measurable_metrics()


def parse_analysis(text: str, candidate_id: str, case_id: str, model_id: str,
                   media, runtime=None) -> VisualFailureAnalysis:
    """Turn a model's JSON reply into an analysis, filling the fields we own.

    The model is not trusted with identity fields: the candidate, the case, the
    model id, the prompt version and the media list are supplied by the caller,
    because those are facts about the request rather than opinions about the
    video.
    """
    payload = json.loads(text)
    analysis_id = hashlib.sha256(
        f"{candidate_id}:{case_id}:{model_id}:{text}".encode("utf-8")
    ).hexdigest()[:16]
    return VisualFailureAnalysis(
        analysis_id=analysis_id,
        candidate_id=candidate_id,
        case_id=case_id,
        model_id=model_id,
        prompt_version=PROMPT_VERSION,
        media=list(media),
        observed_stage=str(payload.get("observed_stage", "")),
        failure_summary=str(payload.get("failure_summary", "")),
        evidence=[
            Evidence(
                statement=str(e.get("statement", "")),
                frame_or_clip_id=str(e.get("frame_or_clip_id", "")),
                timestep=int(e.get("timestep", -1)),
            )
            for e in payload.get("evidence", [])
        ],
        candidate_hypotheses=[str(h) for h in payload.get("candidate_hypotheses", [])],
        uncertainties=[str(u) for u in payload.get("uncertainties", [])],
        recommended_numeric_checks=[
            str(c) for c in payload.get("recommended_numeric_checks", [])
        ],
        runtime=dict(runtime or {}),
    )
