"""Qwen VLM-assisted failure analysis (``spec.md`` FR-11, AC-2).

Optional throughout. The vision model is one source of evidence among several,
it is never the only source for a decision, and the comparison that decides
whether it earns its place in the final method is E10.
"""

from lares.vision.analyst import (
    SYSTEM_PROMPT,
    AnalysisRequest,
    QwenAnalyst,
    ScriptedAnalyst,
    VisionAnalyst,
)
from lares.vision.media import (
    GATE_THRESHOLD,
    MEDIA_SELECTION_RULE,
    MOMENTS,
    EpisodeMedia,
    EpisodeRecorder,
    FinalTestMediaRefused,
    record_episode_media,
    save_media,
    select_media,
)
from lares.vision.pipeline import (
    AnalysisRun,
    analyse_failures,
    evidence_block,
    failed_cases,
)
from lares.vision.merge import (
    AGREE,
    CONFLICT,
    STAGE_FOR_LABEL,
    UNCHECKABLE,
    MergedEvidence,
    merge_all,
    merge_analysis,
    repair_prompt_block,
)
from lares.vision.schema import (
    PROMPT_VERSION,
    Evidence,
    MediaItem,
    VisualFailureAnalysis,
    parse_analysis,
)

__all__ = [
    "AGREE",
    "AnalysisRun",
    "analyse_failures",
    "evidence_block",
    "failed_cases",
    "AnalysisRequest",
    "CONFLICT",
    "EpisodeMedia",
    "EpisodeRecorder",
    "Evidence",
    "FinalTestMediaRefused",
    "GATE_THRESHOLD",
    "MEDIA_SELECTION_RULE",
    "MOMENTS",
    "MediaItem",
    "MergedEvidence",
    "PROMPT_VERSION",
    "QwenAnalyst",
    "STAGE_FOR_LABEL",
    "SYSTEM_PROMPT",
    "ScriptedAnalyst",
    "UNCHECKABLE",
    "VisionAnalyst",
    "VisualFailureAnalysis",
    "merge_all",
    "merge_analysis",
    "parse_analysis",
    "record_episode_media",
    "repair_prompt_block",
    "save_media",
    "select_media",
]
