"""The vision model behind an interface (``spec.md`` FR-11).

The model is the part of FR-11 that costs money and disk, and the part least
worth coupling to. Everything else in :mod:`lares.vision` works against this
interface, so the contract, the media rule, the merge and the conflict marking
are all testable without downloading anything.

:class:`ScriptedAnalyst` is that test double. :class:`QwenAnalyst` is the real
one; it refuses to guess at a model when ``transformers`` is absent or no model
has been chosen, and says what decision is outstanding.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from lares.vision.media import MEDIA_SELECTION_RULE
from lares.vision.schema import PROMPT_VERSION, parse_analysis

SYSTEM_PROMPT = """You are looking at a small number of frames from one failed
robot manipulation episode. The robot must push a puck to a goal position.

You are given the frames and nothing else. Do not assume what happened between
them. Every claim you make must name the frame you read it from and the timestep
of that frame.

Reply with JSON only, in this shape:

{
  "observed_stage": "one of: approach, hover, descend, contact, push, stall, goal",
  "failure_summary": "one sentence on what appears to have gone wrong",
  "evidence": [
    {"statement": "...", "frame_or_clip_id": "...", "timestep": 0}
  ],
  "candidate_hypotheses": ["..."],
  "uncertainties": ["what you cannot tell from these frames"],
  "recommended_numeric_checks": ["..."]
}

Recommend only checks from this list: {measurable}

Your analysis is supporting evidence. It will be compared against the simulator's
own measurements, and where you disagree with them the disagreement is recorded
rather than resolved in your favour. Saying you cannot tell is more useful than
guessing."""


@dataclass
class AnalysisRequest:
    """One episode's worth of media and metadata, and nothing more."""

    candidate_id: str
    case_id: str
    media: list = field(default_factory=list)
    frames: list = field(default_factory=list)
    #: Episode metadata the spec allows: basic, not the diagnosis itself.
    metadata: dict = field(default_factory=dict)


class VisionAnalyst:
    """What the rest of the pipeline needs from a vision model."""

    model_id = "none"
    quantization = "none"

    def analyse(self, request: AnalysisRequest):
        raise NotImplementedError

    def prompt(self, request: AnalysisRequest) -> str:
        from lares.search.schemas import measurable_metrics

        text = SYSTEM_PROMPT.replace(
            "{measurable}", ", ".join(sorted(measurable_metrics()))
        )
        if request.media:
            lines = [f"- image {i + 1}: frame_or_clip_id \"{m.frame_or_clip_id}\", timestep {m.start_timestep} "
                     f"({m.reason})" for i, m in enumerate(request.media)]
            text += ("\n\nThe images above are, in order (cite these ids and timesteps exactly):\n"
                     + "\n".join(lines))
        return text


class ScriptedAnalyst(VisionAnalyst):
    """Returns a prepared reply. The test double, and the E10 control.

    Used with an empty reply it is also the "numeric diagnostics only" arm of
    E10, so both sides of that comparison run the identical code path.
    """

    model_id = "scripted"

    def __init__(self, replies, model_id="scripted"):
        self.replies = list(replies)
        self.model_id = model_id
        self.calls = 0

    def analyse(self, request: AnalysisRequest):
        if not self.replies:
            return None
        text = self.replies[self.calls % len(self.replies)]
        self.calls += 1
        started = time.time()
        return parse_analysis(
            text, request.candidate_id, request.case_id, self.model_id, request.media,
            runtime={
                "quantization": self.quantization,
                "prompt_version": PROMPT_VERSION,
                "media_selection_rule": MEDIA_SELECTION_RULE,
                "latency_seconds": time.time() - started,
                "frames_shown": len(request.media),
                "scripted": True,
            },
        )


class QwenAnalyst(VisionAnalyst):
    """A small Qwen vision model, once one has been chosen.

    Two decisions are outstanding and neither is safe to make silently: which
    Qwen checkpoint, and which quantization. The box is a Turing T4, so fp16 and
    not bf16. Rather than pick a default that would then be frozen into a
    comparison, this raises and says so.
    """

    #: ``nf4`` quantizes the language model's linear layers to 4 bits (bitsandbytes) with fp16 compute and
    #: leaves the vision tower and the output head in fp16. Chosen for R-006 (Qwen2.5-VL-7B on a 16 GB T4).
    QUANTIZATIONS = ("fp16", "nf4")

    def __init__(self, model_id: str = "", quantization: str = "fp16", device: str = "cuda", revision: str = "",
                 max_pixels: int | None = None, max_new_tokens: int = 512):
        if not model_id:
            raise ValueError(
                "QwenAnalyst needs an explicit model id. The checkpoint and "
                "quantization are open research decisions (RESEARCH_HANDOFF.md, "
                "question 2) and picking one here would freeze it into the E10 "
                "comparison without anyone choosing it."
            )
        if quantization not in self.QUANTIZATIONS:
            raise ValueError(f"quantization {quantization!r} is not one of {self.QUANTIZATIONS}")
        self.model_id = model_id
        self.quantization = quantization
        self.device = device
        #: An immutable hub revision; without one the checkpoint can change under a frozen comparison.
        self.revision = revision or None
        self.max_pixels = max_pixels
        self.max_new_tokens = int(max_new_tokens)
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers is not installed in .venv-metaworld. FR-11 needs it; "
                "installing it and pulling a checkpoint is a deliberate, sizeable "
                "step rather than an incidental one."
            ) from exc
        import torch

        processor_kwargs = {"max_pixels": int(self.max_pixels)} if self.max_pixels else {}
        self._processor = AutoProcessor.from_pretrained(self.model_id, revision=self.revision, **processor_kwargs)
        model_kwargs = {"torch_dtype": torch.float16, "device_map": self.device, "revision": self.revision}
        if self.quantization == "nf4":
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                llm_int8_skip_modules=["visual", "lm_head"],
            )
        self._model = AutoModelForVision2Seq.from_pretrained(self.model_id, **model_kwargs)
        self._model.eval()

    def analyse(self, request: AnalysisRequest):
        self._load()
        started = time.time()
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": frame} for frame in request.frames]
            + [{"type": "text", "text": self.prompt(request)}],
        }]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        from PIL import Image

        images = [frame if isinstance(frame, Image.Image) else Image.fromarray(frame) for frame in request.frames]
        inputs = self._processor(
            text=[text], images=images, return_tensors="pt"
        ).to(self.device)
        # Greedy decoding, so a frozen comparison does not also sample the analyst.
        generated = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        reply = self._processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]
        return parse_analysis(
            reply, request.candidate_id, request.case_id, self.model_id, request.media,
            runtime={
                "quantization": self.quantization,
                "revision": self.revision,
                "max_pixels": self.max_pixels,
                "device": self.device,
                "prompt_version": PROMPT_VERSION,
                "media_selection_rule": MEDIA_SELECTION_RULE,
                "latency_seconds": time.time() - started,
                "frames_shown": len(request.media),
                "generated_tokens": int(generated.shape[1] - inputs["input_ids"].shape[1]),
            },
        )
