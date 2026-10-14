"""Generation-time contracts: identity, lineage, feedback and LLM provenance.

``debug_spec.md`` section 4. The live generator used to return three parallel
lists, which dropped everything a later reader needs to explain a candidate:
which idea it implemented, which parents it edited, which feedback it was shown,
which LLM calls produced it and what they cost. This module holds the typed
replacements. It imports nothing from the training pipeline or the generator, so
both can depend on it without a cycle.

Four contracts:

``FeedbackBundle``
    The evidence text shown to the generator, hashed on disk, with the ids of
    every policy whose evidence it contained. Semantic parents must come from
    that list.

``GeneratedCandidate``
    One attempt at one population slot. The slot, the attempt and the ids are
    allocated by code *before* any LLM or validation call, so a failure still has
    an identity and a correction links to the attempt it corrects.

``LLMCallRecord`` / ``LLMCallLedger``
    One entry per real LLM call, with the prompt and response written to files
    and content-hashed. A shared ideation call is one entry, so run totals count
    it once however many candidates it served.

Structured parsing
    Ideas and proposals are parsed strictly. A missing, duplicate or malformed
    entry is an error with a reason, never padded into shape.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

from lares.search.schemas import (  # noqa: F401  (lineage constants re-exported)
    EXPLORATION_LABEL,
    FITTABLE_LABEL,
    LANE_EXPLORATION,
    LANE_FITTABLE,
    LINEAGE_COMPLETE,
    LINEAGE_FAILED_VALIDATION,
    LINEAGE_LEGACY_MISSING,
    LINEAGE_PARTIAL_RECOVERY,
    LINEAGE_STATUSES,
    PRE_IDEA_STAGES,
    MODE_CROSSOVER,
    MODE_EXPLORATION,
    MODE_FITTABLE_MUTATION,
    MODE_PARAMETER_ONLY,
    MODE_REPAIR,
    MODE_SIMPLIFICATION,
    MUTATION_MODES,
    MutationProposal,
    PolicyIdea,
    SchemaError,
)

# ---------------------------------------------------------------------------
#  Environment identity
# ---------------------------------------------------------------------------

#: The rule ``lares.utils.metaworld_env`` and ``lares.eval.manifest.to_v3`` apply.
ENV_ID_MAPPING_RULE = "metaworld>=3.0: replace '-v2' (or '-v1') with '-v3' before looking up ALL_V3_ENVIRONMENTS"
ENV_ID_MAPPING_VERSION = "v2-to-v3/1"


def resolve_registry_env_id(env_id: str) -> str:
    """The id MetaWorld 3 registers for a project-facing ``-v2`` task name."""
    return str(env_id).replace("-v2", "-v3").replace("-v1", "-v3")


def env_identity(env_id: str) -> dict:
    """The environment naming block every new manifest and run record stores."""
    return {
        "canonical_env_id": str(env_id),
        "resolved_registry_env_id": resolve_registry_env_id(env_id),
        "env_id_mapping_rule": ENV_ID_MAPPING_RULE,
        "env_id_mapping_version": ENV_ID_MAPPING_VERSION,
    }


# ---------------------------------------------------------------------------
#  Hashing and run-relative paths
# ---------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def run_relative(path: str, run_root: str) -> str:
    """``path`` relative to the run root, refusing anything outside it.

    A record that points outside its run cannot be moved or archived with it, so
    the reference would silently break.
    """
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(run_root))
    if rel == ".." or rel.startswith(".." + os.sep) or os.path.isabs(rel):
        raise ValueError(f"{path!r} is outside the run root {run_root!r}")
    return rel


def write_artifact(run_root: str, rel_path: str, data: bytes | str) -> dict:
    """Write one artifact under the run root and return its hashed reference."""
    abs_path = os.path.join(run_root, rel_path)
    os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
    payload = data.encode("utf-8") if isinstance(data, str) else data
    with open(abs_path, "wb") as f:
        f.write(payload)
    return {"path": run_relative(abs_path, run_root), "sha256": sha256_bytes(payload)}


def verify_artifact(run_root: str, ref: dict | None) -> str | None:
    """``None`` if the reference resolves and its hash matches, else the reason."""
    if not ref or not ref.get("path") or not ref.get("sha256"):
        return "reference is missing a path or a hash"
    if os.path.isabs(ref["path"]):
        return f"{ref['path']!r} is absolute, not run-relative"
    abs_path = os.path.join(run_root, ref["path"])
    if not os.path.isfile(abs_path):
        return f"{ref['path']!r} does not exist under the run root"
    actual = sha256_file(abs_path)
    if actual != ref["sha256"]:
        return f"{ref['path']!r} hash {actual[:12]} does not match recorded {ref['sha256'][:12]}"
    return None


# ---------------------------------------------------------------------------
#  Feedback bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeedbackBundle:
    """The evidence one generation was shown, frozen and hashed.

    ``context_parent_ids`` lists every policy whose evidence appeared in the
    text. It is not a claim that each one was used by each proposal; semantic
    parents are declared per proposal and must be members of this list.
    """

    feedback_id: str
    source_generation: int
    manifest_id: str
    context_parent_ids: list
    text_path: str
    text_sha256: str

    def validate(self, run_root: str | None = None) -> bool:
        _require(self.feedback_id, "feedback bundle needs an id")
        _require(isinstance(self.source_generation, int), "source_generation must be an int")
        _require(self.manifest_id, "feedback bundle must name the manifest its evidence came from")
        _require(self.text_path and self.text_sha256, "feedback bundle must reference hashed text")
        _require(
            len(set(self.context_parent_ids)) == len(self.context_parent_ids),
            "context_parent_ids contains duplicates",
        )
        if run_root is not None:
            problem = verify_artifact(run_root, {"path": self.text_path, "sha256": self.text_sha256})
            _require(problem is None, f"feedback text does not verify: {problem}")
        return True

    def to_dict(self) -> dict:
        d = asdict(self)
        d["context_parent_ids"] = list(self.context_parent_ids)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FeedbackBundle":
        return cls(
            feedback_id=str(d["feedback_id"]),
            source_generation=int(d["source_generation"]),
            manifest_id=str(d["manifest_id"]),
            context_parent_ids=[str(x) for x in d.get("context_parent_ids", [])],
            text_path=str(d["text_path"]),
            text_sha256=str(d["text_sha256"]),
        )

    def read_text(self, run_root: str) -> str:
        with open(os.path.join(run_root, self.text_path), "r", encoding="utf-8") as f:
            return f.read()


def write_feedback_bundle(
    run_root: str,
    source_generation: int,
    manifest_id: str,
    context_parent_ids: Sequence[str],
    text: str,
) -> FeedbackBundle:
    """Write ``gen_<N>/feedback.md`` plus its bundle descriptor and return the bundle."""
    ids = []
    for pid in context_parent_ids:
        if pid not in ids:
            ids.append(str(pid))
    ref = write_artifact(run_root, os.path.join(f"gen_{source_generation}", "feedback.md"), text)
    bundle = FeedbackBundle(
        feedback_id=f"fb_gen{source_generation}_{ref['sha256'][:12]}",
        source_generation=int(source_generation),
        manifest_id=str(manifest_id),
        context_parent_ids=ids,
        text_path=ref["path"],
        text_sha256=ref["sha256"],
    )
    bundle.validate(run_root)
    write_artifact(
        run_root,
        os.path.join(f"gen_{source_generation}", "feedback_bundle.json"),
        json.dumps(bundle.to_dict(), indent=2),
    )
    return bundle


# ---------------------------------------------------------------------------
#  LLM-call ledger
# ---------------------------------------------------------------------------

LABEL_IDEATION = "ideation"
LABEL_GENERATION = "generation"
LABEL_CORRECTION = "correction"
LABEL_REPAIR = "repair"
# Research-advisor calls (lares.research.advisor) share this ledger format and live in their own ledger.
LABEL_RESEARCH_PROPOSE = "research_propose"
LABEL_RESEARCH_ANALYSE = "research_analyse"
LABEL_RESEARCH_REVIEW = "research_review"
RESEARCH_LABELS = (LABEL_RESEARCH_PROPOSE, LABEL_RESEARCH_ANALYSE, LABEL_RESEARCH_REVIEW)
LLM_LABELS = (LABEL_IDEATION, LABEL_GENERATION, LABEL_CORRECTION, LABEL_REPAIR) + RESEARCH_LABELS


@dataclass
class LLMCallRecord:
    """One actual LLM call. Prompt and response live in hashed files."""

    call_id: str
    label: str
    model: str
    temperature: float
    decoding: dict
    prompt_path: str
    prompt_sha256: str
    response_path: str
    response_sha256: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    wall_time_seconds: float = 0.0
    #: What the call was for inside its label, e.g. ``proposal`` or ``fill``.
    purpose: str = ""
    generation: int | None = None
    #: Attempt ids this call served. More than one means the call was shared, and
    #: its cost belongs to the run, not to any single candidate.
    attempt_ids: list = field(default_factory=list)
    timestamp: str = ""

    def validate(self) -> bool:
        _require(self.call_id, "LLM call needs an id")
        _require(self.label in LLM_LABELS, f"label must be one of {LLM_LABELS}, got {self.label!r}")
        _require(self.model, f"{self.call_id}: LLM call must name its model")
        for name in ("prompt_path", "prompt_sha256", "response_path", "response_sha256"):
            _require(getattr(self, name), f"{self.call_id}: missing {name}")
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = getattr(self, name)
            _require(isinstance(value, int) and value >= 0, f"{self.call_id}: {name} must be a non-negative int")
        return True

    @property
    def shared(self) -> bool:
        return len(self.attempt_ids) != 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LLMCallRecord":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    def reference(self) -> dict:
        """The subset a candidate record embeds, so prompt evidence travels with it."""
        return {
            "call_id": self.call_id,
            "label": self.label,
            "purpose": self.purpose,
            "prompt_sha256": self.prompt_sha256,
            "response_sha256": self.response_sha256,
        }


def _as_int(value) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


class LLMCallLedger:
    """Append-only run-level ledger of LLM calls, ``<run_root>/llm_calls.jsonl``."""

    FILENAME = "llm_calls.jsonl"
    CALLS_DIR = "llm_calls"

    def __init__(self, run_root: str):
        self.run_root = os.path.abspath(run_root)
        os.makedirs(self.run_root, exist_ok=True)
        self.path = os.path.join(self.run_root, self.FILENAME)
        self._records: dict[str, LLMCallRecord] = {}
        for record in self.load(self.path):
            self._records[record.call_id] = record

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, call_id) -> bool:
        return call_id in self._records

    def get(self, call_id: str) -> LLMCallRecord:
        return self._records[call_id]

    def records(self) -> list:
        return list(self._records.values())

    def record(
        self,
        label: str,
        model: str,
        temperature: float,
        messages,
        responses: Sequence[str],
        usage: dict | None,
        wall_time_seconds: float,
        decoding: dict | None = None,
        purpose: str = "",
        generation: int | None = None,
        attempt_ids: Sequence[str] = (),
    ) -> LLMCallRecord:
        if label not in LLM_LABELS:
            raise ValueError(f"label must be one of {LLM_LABELS}, got {label!r}")
        seq = len(self._records) + 1
        call_id = f"call_{seq:05d}"
        while call_id in self._records:
            seq += 1
            call_id = f"call_{seq:05d}"
        prompt = write_artifact(
            self.run_root,
            os.path.join(self.CALLS_DIR, f"{call_id}.prompt.json"),
            json.dumps(messages, ensure_ascii=False, indent=2),
        )
        response = write_artifact(
            self.run_root,
            os.path.join(self.CALLS_DIR, f"{call_id}.response.json"),
            json.dumps(list(responses), ensure_ascii=False, indent=2),
        )
        usage = usage or {}
        record = LLMCallRecord(
            call_id=call_id,
            label=label,
            model=str(model),
            temperature=float(temperature),
            decoding=dict(decoding or {}),
            prompt_path=prompt["path"],
            prompt_sha256=prompt["sha256"],
            response_path=response["path"],
            response_sha256=response["sha256"],
            prompt_tokens=_as_int(usage.get("prompt_tokens")),
            completion_tokens=_as_int(usage.get("completion_tokens")),
            total_tokens=_as_int(usage.get("total_tokens")),
            wall_time_seconds=float(wall_time_seconds),
            purpose=str(purpose),
            generation=generation,
            attempt_ids=[str(a) for a in attempt_ids],
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        record.validate()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict()) + "\n")
        self._records[call_id] = record
        return record

    def totals(self, call_ids: Iterable[str] | None = None) -> dict:
        """Usage over *unique* call ids; every call when ``call_ids`` is ``None``."""
        if call_ids is None:
            chosen = list(self._records.values())
            unknown = []
        else:
            unique = list(dict.fromkeys(call_ids))
            chosen = [self._records[c] for c in unique if c in self._records]
            unknown = [c for c in unique if c not in self._records]
        return {
            "llm_calls": len(chosen),
            "llm_prompt_tokens": sum(r.prompt_tokens for r in chosen),
            "llm_completion_tokens": sum(r.completion_tokens for r in chosen),
            "llm_total_tokens": sum(r.total_tokens for r in chosen),
            "llm_wall_time_seconds": float(sum(r.wall_time_seconds for r in chosen)),
            "unknown_call_ids": unknown,
        }

    def verify(self) -> dict:
        """``{call_id: reason}`` for every call whose prompt or response does not verify."""
        problems = {}
        for record in self._records.values():
            for kind in ("prompt", "response"):
                reason = verify_artifact(
                    self.run_root,
                    {"path": getattr(record, f"{kind}_path"), "sha256": getattr(record, f"{kind}_sha256")},
                )
                if reason:
                    problems.setdefault(record.call_id, []).append(f"{kind}: {reason}")
        return problems

    @staticmethod
    def load(path: str) -> list:
        if not os.path.isfile(path):
            return []
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(LLMCallRecord.from_dict(json.loads(line)))
        return out


# ---------------------------------------------------------------------------
#  Candidate identity
# ---------------------------------------------------------------------------

#: Recommended complexity budget for the next controlled run (``debug_spec.md``
#: Stage A item 10). Larger policies need an explicit justification.
PARAMETER_BUDGET = (6, 10)
MAX_PHASES_WITHOUT_JUSTIFICATION = 4


def candidate_id_for(generation: int, slot: int) -> str:
    return f"gen{int(generation)}_cand{int(slot)}"


def attempt_id_for(candidate_id: str, attempt: int) -> str:
    return f"{candidate_id}_a{int(attempt)}"


class CandidateAllocator:
    """Hands out slot and attempt ids before anything that can fail is called."""

    def __init__(self, generation: int):
        self.generation = int(generation)
        self._slots: list[str] = []
        self._attempts: dict[str, int] = {}

    def allocate_slots(self, n: int) -> list:
        out = []
        for _ in range(int(n)):
            slot = len(self._slots)
            cid = candidate_id_for(self.generation, slot)
            self._slots.append(cid)
            out.append((cid, slot))
        return out

    def next_attempt(self, candidate_id: str) -> int:
        if candidate_id not in self._slots:
            raise KeyError(f"{candidate_id!r} was never allocated a slot")
        attempt = self._attempts.get(candidate_id, -1) + 1
        self._attempts[candidate_id] = attempt
        return attempt

    @property
    def slot_ids(self) -> list:
        return list(self._slots)


@dataclass
class GeneratedCandidate:
    """One attempt at one slot. Failures keep the metadata allocated for them."""

    candidate_id: str
    generation: int
    slot: int
    attempt: int
    policy: object | None = None
    code: str | None = None
    response: str | None = None
    idea: PolicyIdea | None = None
    proposal: MutationProposal | None = None
    feedback_id: str | None = None
    context_parent_ids: list = field(default_factory=list)
    llm_call_ids: list = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    #: Set on a correction: the attempt whose failure it answers.
    previous_attempt_id: str | None = None
    #: How the code was attached to the idea (batched fence index, per-idea call).
    mapping: dict = field(default_factory=dict)
    #: ``fittable`` or ``exploration`` in a two-lane search; empty otherwise.
    lane: str = ""
    #: Allocated with the slot, before any call: the label, and for the fittable lane the
    #: seed-bank reference, the predeclared mutation scope and (once parsed) the mutation.
    lane_metadata: dict = field(default_factory=dict)

    @property
    def attempt_id(self) -> str:
        return attempt_id_for(self.candidate_id, self.attempt)

    @property
    def valid(self) -> bool:
        return bool(self.validation.get("passed")) and self.policy is not None

    @property
    def idea_id(self) -> str:
        return self.idea.idea_id if self.idea is not None else ""

    @property
    def proposal_id(self) -> str:
        return self.proposal.proposal_id if self.proposal is not None else ""

    @property
    def parent_ids(self) -> list:
        """Semantic parents, declared by the proposal. Never the context list."""
        return list(self.proposal.parent_ids) if self.proposal is not None else []

    @property
    def mode(self) -> str:
        return self.proposal.mode if self.proposal is not None else ""

    def lineage_problems(self, feedback: FeedbackBundle | None = None) -> list:
        return check_lineage(self, feedback)

    def lineage_status(self, feedback: FeedbackBundle | None = None) -> str:
        if not self.valid:
            return LINEAGE_FAILED_VALIDATION
        return LINEAGE_COMPLETE if not self.lineage_problems(feedback) else LINEAGE_FAILED_VALIDATION

    def lineage_dict(self, feedback: FeedbackBundle | None = None) -> dict:
        return {
            "status": self.lineage_status(feedback),
            "problems": self.lineage_problems(feedback),
            "generation": self.generation,
            "slot": self.slot,
            "attempt": self.attempt,
            "attempt_id": self.attempt_id,
            "previous_attempt_id": self.previous_attempt_id,
            "feedback_id": self.feedback_id,
            "context_parent_ids": list(self.context_parent_ids),
            "parent_ids": self.parent_ids,
            "mode": self.mode,
            "idea": self.idea.to_dict() if self.idea is not None else None,
            "proposal": self.proposal.to_dict() if self.proposal is not None else None,
            "mapping": dict(self.mapping),
            "lane": self.lane,
            "lane_metadata": dict(self.lane_metadata),
        }


# ---------------------------------------------------------------------------
#  Lineage rules
# ---------------------------------------------------------------------------


def _require(condition, message):
    if not condition:
        raise SchemaError(message)


def check_semantic_parents(proposal: MutationProposal, context_parent_ids: Sequence[str]) -> list:
    """Parent-count and membership rules for one proposal (``debug_spec.md`` 4.2)."""
    problems = []
    parents = list(proposal.parent_ids)
    if len(set(parents)) != len(parents):
        problems.append(f"{proposal.proposal_id}: duplicate parent ids {parents}")
    if proposal.mode in (MODE_REPAIR, MODE_SIMPLIFICATION) and len(parents) != 1:
        problems.append(
            f"{proposal.proposal_id}: a {proposal.mode} proposal edits exactly one parent, got {len(parents)}"
        )
    if proposal.mode == MODE_FITTABLE_MUTATION and len(parents) != 1:
        problems.append(
            f"{proposal.proposal_id}: a fittable mutation edits exactly one seed-bank reference, got {len(parents)}"
        )
    if proposal.mode == MODE_CROSSOVER and len(parents) < 2:
        problems.append(f"{proposal.proposal_id}: a crossover needs at least two parents, got {len(parents)}")
    if proposal.mode == MODE_EXPLORATION and not parents and not proposal.no_parent_rationale.strip():
        problems.append(
            f"{proposal.proposal_id}: an exploration with no parent must say why it is not "
            f"using one (no_parent_rationale)"
        )
    if proposal.mode == MODE_PARAMETER_ONLY:
        problems.append(
            f"{proposal.proposal_id}: parameter_only edits are the numerical optimizer's job "
            f"and are not accepted from the generator"
        )
    context = set(context_parent_ids)
    outside = [p for p in parents if p not in context]
    if outside:
        problems.append(
            f"{proposal.proposal_id}: parents {outside} were not in the feedback context "
            f"{sorted(context)}; a proposal may only build on evidence it was shown"
        )
    return problems


def check_complexity(idea: PolicyIdea, budget: tuple = PARAMETER_BUDGET) -> list:
    """Declared parameter count and phase count against the declared budget."""
    problems = []
    justification = (idea.complexity_justification or "").strip()
    n_params = len(idea.parameters)
    if n_params > int(budget[1]) and not justification:
        problems.append(
            f"idea {idea.idea_id!r} declares {n_params} parameters, above the budget of "
            f"{budget[1]}; give a complexity_justification or remove parameters"
        )
    if len(idea.phases) > MAX_PHASES_WITHOUT_JUSTIFICATION and not justification:
        problems.append(
            f"idea {idea.idea_id!r} declares {len(idea.phases)} phases, above "
            f"{MAX_PHASES_WITHOUT_JUSTIFICATION}; give a complexity_justification"
        )
    return problems


def check_lineage(candidate: GeneratedCandidate, feedback: FeedbackBundle | None = None) -> list:
    """Every lineage rule a candidate has to satisfy, as a list of problems."""
    problems = []
    if candidate.idea is None:
        problems.append(f"{candidate.attempt_id}: no valid PolicyIdea")
    else:
        try:
            candidate.idea.validate()
        except SchemaError as exc:
            problems.append(f"{candidate.attempt_id}: idea invalid: {exc}")

    if candidate.lane == LANE_FITTABLE:
        return problems + _fittable_lineage_problems(candidate)
    if candidate.lane == LANE_EXPLORATION:
        meta = candidate.lane_metadata or {}
        if meta.get("label") != EXPLORATION_LABEL or meta.get("seeded") is not False:
            problems.append(f"{candidate.attempt_id}: an exploration must be labelled {EXPLORATION_LABEL!r}, unseeded")
        if candidate.proposal is not None:
            problems.append(f"{candidate.attempt_id}: an unseeded exploration has no proposal and no parent")
        return problems

    if candidate.generation == 0:
        if candidate.proposal is not None:
            problems.append(f"{candidate.attempt_id}: generation 0 has no semantic parent, so no proposal")
        return problems

    if candidate.proposal is None:
        problems.append(f"{candidate.attempt_id}: generation {candidate.generation} requires a MutationProposal")
        return problems
    try:
        candidate.proposal.validate()
    except SchemaError as exc:
        problems.append(f"{candidate.attempt_id}: proposal invalid: {exc}")
    if not candidate.feedback_id:
        problems.append(f"{candidate.attempt_id}: generation {candidate.generation} must reference its feedback bundle")
    if feedback is not None:
        if candidate.feedback_id != feedback.feedback_id:
            problems.append(
                f"{candidate.attempt_id}: feedback id {candidate.feedback_id!r} does not match "
                f"the bundle shown, {feedback.feedback_id!r}"
            )
        if list(candidate.context_parent_ids) != list(feedback.context_parent_ids):
            problems.append(f"{candidate.attempt_id}: context parents differ from the feedback bundle")
    problems.extend(check_semantic_parents(candidate.proposal, candidate.context_parent_ids))
    return problems


def _fittable_lineage_problems(candidate: GeneratedCandidate) -> list:
    """A fittable-lane candidate: one proposal editing its slot's reference, inside its scope."""
    who = candidate.attempt_id
    meta = candidate.lane_metadata or {}
    reference = (meta.get("reference") or {}).get("structure_id")
    problems = []
    if meta.get("label") != FITTABLE_LABEL or meta.get("seeded") is not True:
        problems.append(f"{who}: a fittable-lane candidate must be labelled {FITTABLE_LABEL!r}, seeded")
    if not reference or not (meta.get("reference") or {}).get("source_sha256"):
        problems.append(f"{who}: no seed-bank reference with a source hash")
    if not (meta.get("mutation_scope") or {}).get("id"):
        problems.append(f"{who}: no predeclared mutation scope")
    if candidate.proposal is None:
        problems.append(f"{who}: a fittable-lane candidate requires its mutation proposal")
        return problems
    try:
        candidate.proposal.validate()
    except SchemaError as exc:
        problems.append(f"{who}: proposal invalid: {exc}")
    if candidate.proposal.mode != MODE_FITTABLE_MUTATION:
        problems.append(f"{who}: mode must be {MODE_FITTABLE_MUTATION!r}, got {candidate.proposal.mode!r}")
    if list(candidate.proposal.parent_ids) != [reference]:
        problems.append(f"{who}: parents {candidate.proposal.parent_ids} are not the slot reference [{reference!r}]")
    mutation = meta.get("mutation") or {}
    if not str(mutation.get("structural_change", "")).strip():
        problems.append(f"{who}: the mutation does not state its one structural change")
    problems.extend(check_semantic_parents(candidate.proposal, candidate.context_parent_ids))
    return problems


# ---------------------------------------------------------------------------
#  Strict structured parsing
# ---------------------------------------------------------------------------


class StructuredOutputError(SchemaError):
    """Structured model output was missing, malformed or the wrong size."""

    def __init__(self, message: str, problems: Sequence[str] = ()):
        super().__init__(message)
        self.problems = list(problems) or [message]


def extract_json_object(text: str) -> dict:
    """The one JSON object in ``text``. A fenced block is unwrapped; nothing is guessed."""
    if text is None:
        raise StructuredOutputError("the response was empty")
    t = text.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", t, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        t = fence.group(1).strip()
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start < 0 or end <= start:
            raise StructuredOutputError("the response contains no JSON object")
        try:
            data = json.loads(t[start : end + 1])
        except json.JSONDecodeError as exc:
            raise StructuredOutputError(f"the response is not valid JSON: {exc}")
    if not isinstance(data, dict):
        raise StructuredOutputError(f"expected a JSON object, got {type(data).__name__}")
    return data


_IDEA_KEYS = ("name", "summary", "named_observations", "phases", "parameters", "zero_shot_behavior")
_PHASE_KEYS = ("name", "activation", "action_intent")
_PARAM_KEYS = ("name", "role", "type", "units", "initial_value", "range")
_PROPOSAL_KEYS = (
    "parent_ids",
    "mode",
    "observation",
    "hypothesis",
    "suspected_component",
    "intervention",
    "predicted_metric",
    "predicted_direction",
    "protected_behaviors",
)


def _missing_keys(d, keys, where) -> list:
    if not isinstance(d, dict):
        return [f"{where} is {type(d).__name__}, not an object"]
    return [f"{where} is missing {k!r}" for k in keys if k not in d]


def _parse_idea(d: dict, idea_id: str, where: str, schema=None, budget=PARAMETER_BUDGET):
    problems = _missing_keys(d, _IDEA_KEYS, where)
    if problems:
        return None, problems
    for i, phase in enumerate(d.get("phases") or []):
        problems.extend(_missing_keys(phase, _PHASE_KEYS, f"{where}.phases[{i}]"))
    for i, param in enumerate(d.get("parameters") or []):
        problems.extend(_missing_keys(param, _PARAM_KEYS, f"{where}.parameters[{i}]"))
    if problems:
        return None, problems
    try:
        idea = PolicyIdea.from_dict({**d, "idea_id": idea_id})
        idea.validate(schema)
    except (SchemaError, TypeError, ValueError) as exc:
        return None, [f"{where}: {exc}"]
    complexity = check_complexity(idea, budget)
    if complexity:
        return None, [f"{where}: {p}" for p in complexity]
    return idea, []


def _duplicate_key(idea: PolicyIdea) -> tuple:
    norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())  # noqa: E731
    return norm(idea.name), norm(idea.summary)


@dataclass
class SlotParse:
    """One slot's parse: the objects when valid, the reasons when not."""

    slot: int
    idea: PolicyIdea | None = None
    proposal: MutationProposal | None = None
    problems: list = field(default_factory=list)
    #: A fittable mutation's declaration: structural change and parameter changes.
    mutation: dict | None = None

    @property
    def ok(self) -> bool:
        return not self.problems and self.idea is not None


def _mark_duplicates(parsed: list) -> None:
    seen = {}
    for sp in parsed:
        if sp.idea is None:
            continue
        name, summary = _duplicate_key(sp.idea)
        for other_slot, (other_name, other_summary) in seen.items():
            if name == other_name or summary == other_summary:
                sp.problems.append(
                    f"slot {sp.slot}: duplicates slot {other_slot} ({sp.idea.name!r}); "
                    f"each idea must be a distinct design"
                )
                break
        else:
            seen[sp.slot] = (name, summary)


def parse_ideas_payload(
    text: str,
    expected_n: int,
    generation: int,
    schema=None,
    budget: tuple = PARAMETER_BUDGET,
    slot_offset: int = 0,
) -> list:
    """Strictly parse ``{"ideas": [<PolicyIdea>, ...]}`` into per-slot results.

    A response that is not JSON or has the wrong number of ideas raises
    :class:`StructuredOutputError`: positional mapping onto slots would otherwise
    be a guess. An individual malformed or duplicate idea fails only its slot.
    Ids are assigned here, never taken from the model.
    """
    data = extract_json_object(text)
    ideas = data.get("ideas")
    if not isinstance(ideas, list):
        raise StructuredOutputError('expected {"ideas": [...]} with a list of idea objects')
    if len(ideas) != int(expected_n):
        raise StructuredOutputError(
            f"expected exactly {expected_n} ideas, got {len(ideas)}; nothing is padded or dropped"
        )
    out = []
    for i, raw in enumerate(ideas):
        slot = int(slot_offset) + i
        idea, problems = _parse_idea(raw, f"gen{generation}_idea{slot}", f"ideas[{i}]", schema, budget)
        out.append(SlotParse(slot=slot, idea=idea, problems=problems))
    _mark_duplicates(out)
    return out


def parse_proposals_payload(
    text: str,
    expected_n: int,
    generation: int,
    context_parent_ids: Sequence[str],
    schema=None,
    metrics: set | None = None,
    budget: tuple = PARAMETER_BUDGET,
) -> list:
    """Strictly parse ``{"proposals": [{"proposal": ..., "idea": ...}, ...]}``.

    Every proposal must satisfy the mode's parent-count rule and draw its
    parents from ``context_parent_ids``. Proposal and idea ids are assigned here.
    """
    data = extract_json_object(text)
    items = data.get("proposals")
    if not isinstance(items, list):
        raise StructuredOutputError('expected {"proposals": [...]} with a list of proposal objects')
    if len(items) != int(expected_n):
        raise StructuredOutputError(
            f"expected exactly {expected_n} proposals, got {len(items)}; nothing is padded or dropped"
        )
    out = []
    for i, item in enumerate(items):
        where = f"proposals[{i}]"
        sp = SlotParse(slot=i)
        if not isinstance(item, dict) or "proposal" not in item or "idea" not in item:
            sp.problems.append(f'{where} must be an object with "proposal" and "idea"')
            out.append(sp)
            continue
        raw = item["proposal"]
        missing = _missing_keys(raw, _PROPOSAL_KEYS, f"{where}.proposal")
        if missing:
            sp.problems.extend(missing)
        else:
            mode = str(raw.get("mode", ""))
            if mode == MODE_FITTABLE_MUTATION:
                sp.problems.append(
                    f"{where}.proposal: {MODE_FITTABLE_MUTATION!r} is produced only by the fittable lane's "
                    f"seeded request, never by an ordinary proposal"
                )
            elif mode not in MUTATION_MODES:
                sp.problems.append(f"{where}.proposal: mode must be one of {MUTATION_MODES}, got {mode!r}")
            else:
                proposal = MutationProposal.from_dict({**raw, "proposal_id": f"gen{generation}_prop{i}"})
                try:
                    proposal.validate(metrics)
                except SchemaError as exc:
                    sp.problems.append(f"{where}.proposal: {exc}")
                sp.problems.extend(f"{where}: {p}" for p in check_semantic_parents(proposal, context_parent_ids))
                sp.proposal = proposal
        idea, problems = _parse_idea(item["idea"], f"gen{generation}_idea{i}", f"{where}.idea", schema, budget)
        sp.idea = idea
        sp.problems.extend(problems)
        out.append(sp)
    _mark_duplicates(out)
    return out


_MUTATION_ENTRY_KEYS = ("slot", "parent_id", "mutation_scope", "structural_change", "parameter_changes", "proposal", "idea")
_MUTATION_PROPOSAL_KEYS = tuple(k for k in _PROPOSAL_KEYS if k not in ("parent_ids", "mode"))


def parse_fittable_mutations_payload(
    text: str,
    slots: Sequence[dict],
    generation: int,
    schema=None,
    metrics: set | None = None,
    budget: tuple = PARAMETER_BUDGET,
    declaration_check=None,
) -> list:
    """Strictly parse ``{"mutations": [...]}`` for the fittable lane's predeclared slots.

    ``slots`` is ``[{"slot", "reference", "scope"}, ...]`` in slot order; the allocation is
    fixed by configuration, so the model must echo each slot's reference and scope and may not
    choose another. A wrong count raises; a bad entry fails only its slot. Proposal and idea ids,
    the mode and the parent are set here, never taken from the model. ``declaration_check(slot,
    declaration, idea)`` returns further problems (the scope and bounded-change rules).
    """
    data = extract_json_object(text)
    items = data.get("mutations")
    if not isinstance(items, list):
        raise StructuredOutputError('expected {"mutations": [...]} with a list of mutation objects')
    if len(items) != len(slots):
        raise StructuredOutputError(
            f"expected exactly {len(slots)} mutations, one per fittable slot, got {len(items)}; "
            f"nothing is padded or dropped"
        )
    out = []
    for i, (item, slot) in enumerate(zip(items, slots)):
        where = f"mutations[{i}]"
        sp = SlotParse(slot=int(slot["slot"]))
        missing = _missing_keys(item, _MUTATION_ENTRY_KEYS, where)
        if missing:
            sp.problems.extend(missing)
            out.append(sp)
            continue
        if item.get("slot") != slot["slot"]:
            sp.problems.append(f"{where}: slot {item.get('slot')!r} is not the predeclared slot {slot['slot']}")
        if item.get("parent_id") != slot["reference"]:
            sp.problems.append(
                f"{where}: parent {item.get('parent_id')!r} is not this slot's reference {slot['reference']!r}; "
                f"the allocation is fixed"
            )
        if item.get("mutation_scope") != slot["scope"]:
            sp.problems.append(f"{where}: scope {item.get('mutation_scope')!r} is not this slot's scope {slot['scope']!r}")
        raw = item["proposal"]
        missing = _missing_keys(raw, _MUTATION_PROPOSAL_KEYS, f"{where}.proposal")
        if missing:
            sp.problems.extend(missing)
        else:
            proposal = MutationProposal.from_dict({
                **raw, "proposal_id": f"gen{generation}_prop{slot['slot']}",
                "parent_ids": [slot["reference"]], "mode": MODE_FITTABLE_MUTATION, "no_parent_rationale": "",
            })
            try:
                proposal.validate(metrics)
            except SchemaError as exc:
                sp.problems.append(f"{where}.proposal: {exc}")
            sp.proposal = proposal
        changes = item["parameter_changes"]
        declaration = {"structural_change": str(item.get("structural_change") or "").strip()}
        if not declaration["structural_change"]:
            sp.problems.append(f"{where}: structural_change must state the one structural change")
        if not isinstance(changes, dict) or any(not isinstance(changes.get(k, []), list) for k in ("adds", "removes", "changes")):
            sp.problems.append(f'{where}: parameter_changes must be {{"adds": [...], "removes": [...], "changes": [...]}}')
            changes = {}
        for key in ("adds", "removes", "changes"):
            declaration[key] = [str(x) for x in (changes.get(key) or [])]
        sp.mutation = declaration
        idea, problems = _parse_idea(item["idea"], f"gen{generation}_idea{slot['slot']}", f"{where}.idea", schema, budget)
        sp.idea = idea
        sp.problems.extend(problems)
        if declaration_check is not None:
            sp.problems.extend(f"{where}: {p}" for p in declaration_check(slot, declaration, idea))
        out.append(sp)
    _mark_duplicates(out)
    return out
