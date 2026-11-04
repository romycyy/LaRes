"""Research-idea ledger: the contract one level above ``ExperimentRecord``.

An ``ExperimentRecord`` says what one candidate policy did. A ``ResearchIdea``
says what we expected a *method change* to do, how we agreed to test it, what
happened, and what both the advisor model and the implementer concluded. The
rules mirror the candidate layer:

- a prediction must name a metric the reports measure (``measurable_metrics``),
  with a direction and a threshold, so it can be wrong;
- a protocol is declared before any result exists and names its script, config,
  seeds and manifests; a protocol that touches the final-test split is refused;
- every idea passes through ``proposed -> approved -> running -> analysed ->
  decided`` (or ``rejected`` / ``deferred``), and nothing runs before it is
  approved. Since 2026-10-26 approval is *triage*: the advisor model and the
  implementer must reach the same position (approve / reject / defer) over one or
  more recorded rounds; the user sets the research direction instead of ruling
  on each idea, and only a negotiation that exhausts its rounds is ``escalated``;
- a decision is written only when the advisor's verdict and the implementer's
  verdict agree, again over recorded rounds; otherwise the idea is ``escalated``.

Files are JSON under ``research/ideas/<id>.json`` and ``research/decisions.jsonl``;
every triage round is also appended to ``research/triage.jsonl`` and every raw
proposal (parsed or not) to ``research/proposals.jsonl`` for later inspection.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Sequence

from lares.utils.clock import now_iso
from lares.utils.jsonl import append_jsonl, read_jsonl

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESEARCH_ROOT = os.path.join(_PROJECT_ROOT, "research")
IDEAS_DIR = "ideas"
DECISIONS_FILE = "decisions.jsonl"
STATE_FILE = "STATE.md"
BRIEFINGS_DIR = "briefings"
CONSTRAINTS_FILE = "constraints.json"
DIRECTION_FILE = "direction.json"
TRIAGE_LOG = "triage.jsonl"
PROPOSALS_LOG = "proposals.jsonl"
DEFAULT_MAX_ROUNDS = 5

STATUS_PROPOSED = "proposed"
STATUS_APPROVED = "approved"
STATUS_RUNNING = "running"
STATUS_ANALYSED = "analysed"
STATUS_DECIDED = "decided"
STATUS_ESCALATED = "escalated"
STATUS_REJECTED = "rejected"
STATUS_DEFERRED = "deferred"
STATUSES = (
    STATUS_PROPOSED,
    STATUS_APPROVED,
    STATUS_RUNNING,
    STATUS_ANALYSED,
    STATUS_DECIDED,
    STATUS_ESCALATED,
    STATUS_REJECTED,
    STATUS_DEFERRED,
)
# Legal transitions. ``escalated`` is reached from triage (rounds exhausted before approval) or from
# the verdict stage (rounds exhausted before a decision); the user's ruling moves it on.
TRANSITIONS = {
    STATUS_PROPOSED: {STATUS_APPROVED, STATUS_REJECTED, STATUS_DEFERRED, STATUS_ESCALATED},
    STATUS_DEFERRED: {STATUS_PROPOSED, STATUS_APPROVED, STATUS_REJECTED},
    STATUS_APPROVED: {STATUS_RUNNING, STATUS_REJECTED},
    STATUS_RUNNING: {STATUS_ANALYSED, STATUS_REJECTED},
    STATUS_ANALYSED: {STATUS_DECIDED, STATUS_ESCALATED},
    STATUS_ESCALATED: {STATUS_ANALYSED, STATUS_DECIDED, STATUS_REJECTED, STATUS_APPROVED, STATUS_DEFERRED},
    STATUS_DECIDED: set(),
    STATUS_REJECTED: set(),
}

# Triage: what the advisor and the implementer each say about a proposed idea.
TRIAGE_APPROVE = "approve"
TRIAGE_REVISE = "revise"
TRIAGE_DEFER = "defer"
TRIAGE_REJECT = "reject"
TRIAGE_POSITIONS = (TRIAGE_APPROVE, TRIAGE_REVISE, TRIAGE_DEFER, TRIAGE_REJECT)
TRIAGE_FINAL_STATUS = {TRIAGE_APPROVE: STATUS_APPROVED, TRIAGE_DEFER: STATUS_DEFERRED, TRIAGE_REJECT: STATUS_REJECTED}
STAGE_TRIAGE = "triage"
STAGE_VERDICT = "verdict"

VERDICT_SUPPORTED = "supported"
VERDICT_REFUTED = "refuted"
VERDICT_INCONCLUSIVE = "inconclusive"
VERDICTS = (VERDICT_SUPPORTED, VERDICT_REFUTED, VERDICT_INCONCLUSIVE)

DIRECTIONS = ("increase", "decrease")
SOURCE_ADVISOR = "advisor"
SOURCE_IMPLEMENTER = "implementer"
SOURCE_USER = "user"

ID_PATTERN = re.compile(r"^R-\d{3,}$")
FINAL_TEST_MARKERS = ("final_test", "final-test", "finaltest")


class ResearchSchemaError(ValueError):
    """A research object did not satisfy its contract."""


def _require(condition, message):
    if not condition:
        raise ResearchSchemaError(message)


def now() -> str:
    return now_iso()


def _measurable() -> set[str]:
    """The metric names a prediction may use; imported lazily so the ledger loads without torch."""
    try:
        from lares.search.schemas import measurable_metrics

        return set(measurable_metrics())
    except Exception:  # pragma: no cover - only when the eval stack is not importable
        return set()


# Metric names the research layer adds on top of the per-candidate reports: aggregates a
# protocol script writes to its results file. Kept short and explicit so a prediction cannot
# name a quantity nothing produces.
RESEARCH_METRICS = {
    "promotion_success_rate",      # success on the promotion / held-out manifest
    "screen_success_rate",         # success on the screening cases
    "paired_success_difference",   # paired difference against a named control
    "paired_interval_lower",       # lower bound of the paired t interval
    "slots_valid_within_one_correction",
    "llm_total_tokens",
    "llm_calls",
    "simulator_steps",
    "wall_time_seconds",
}


def allowed_metrics() -> set[str]:
    return _measurable() | RESEARCH_METRICS


@dataclass
class Prediction:
    """What the idea claims will happen. Must be able to fail."""

    metric: str
    direction: str
    threshold: float
    against: str = ""          # the control the metric is compared with (idea id, run dir, "baseline")
    rationale: str = ""

    def validate(self, metrics: set[str] | None = None):
        metrics = allowed_metrics() if metrics is None else metrics
        _require(self.metric, "prediction must name a metric")
        _require(
            self.metric in metrics,
            f"prediction names {self.metric!r}, which nothing measures; allowed: {sorted(metrics)}",
        )
        _require(self.direction in DIRECTIONS, f"direction must be one of {DIRECTIONS}, got {self.direction!r}")
        _require(isinstance(self.threshold, (int, float)), "threshold must be a number")

    def check(self, before, after) -> dict:
        """Did the measured change match? ``None`` values leave the outcome ``None``."""
        if before is None or after is None:
            return {"metric": self.metric, "before": before, "after": after, "matched": None}
        delta = float(after) - float(before)
        matched = delta >= self.threshold if self.direction == "increase" else -delta >= self.threshold
        return {"metric": self.metric, "before": before, "after": after, "delta": delta, "matched": bool(matched)}


@dataclass
class Protocol:
    """How the idea is tested. Declared before any result exists."""

    script: str                                 # e.g. scripts/stage_f.py
    config: str = ""                            # e.g. config/stage_f/stage_f.yaml
    command: str = ""                           # the exact command, from the project root
    seeds: list = field(default_factory=list)   # seeds this protocol will spend (must be unspent)
    manifests: list = field(default_factory=list)
    episodes: int | None = None                 # simulator episodes the protocol will spend
    llm_calls: int | None = None                # LLM calls it will spend
    code_changes: list = field(default_factory=list)  # files the implementer expects to change
    needs_gpu: bool = True
    needs_llm: bool = False

    def validate(self):
        _require(self.script, "protocol must name a script")
        text = json.dumps(asdict(self)).lower()
        for marker in FINAL_TEST_MARKERS:
            _require(marker not in text, f"protocol mentions {marker!r}: the final-test split is locked until Phase 7")
        _require(all(isinstance(s, int) for s in self.seeds), "seeds must be integers")


@dataclass
class Verdict:
    source: str
    verdict: str
    confidence: float
    reasoning: str = ""
    cited_artifacts: list = field(default_factory=list)
    next_experiment: str = ""
    recorded_at: str = field(default_factory=now)
    call_id: str = ""

    def validate(self):
        _require(self.source in (SOURCE_ADVISOR, SOURCE_IMPLEMENTER, SOURCE_USER), f"bad verdict source {self.source!r}")
        _require(self.verdict in VERDICTS, f"verdict must be one of {VERDICTS}, got {self.verdict!r}")
        _require(0.0 <= float(self.confidence) <= 1.0, "confidence must be in [0, 1]")
        if self.source == SOURCE_ADVISOR:
            _require(self.cited_artifacts, "an advisor verdict must cite at least one artifact path")


@dataclass
class ResearchIdea:
    id: str
    title: str
    hypothesis: str
    prediction: Prediction
    protocol: Protocol
    status: str = STATUS_PROPOSED
    source: str = SOURCE_ADVISOR
    falsified_by: str = ""                      # what observation would refute the hypothesis
    cost_estimate: dict = field(default_factory=dict)
    open_question_refs: list = field(default_factory=list)   # RESEARCH_HANDOFF question numbers
    briefing_sha256: str = ""                   # the packet the advisor saw when it proposed this
    proposal_call_id: str = ""
    approval: dict = field(default_factory=dict)      # {by, at, note}
    run: dict = field(default_factory=dict)           # {started_at, git_sha, finished_at, log_dir}
    result: dict = field(default_factory=dict)        # {metrics: {name: value}, controls: {...}, artifacts: [...], notes}
    prediction_outcome: dict = field(default_factory=dict)
    verdicts: list = field(default_factory=list)      # Verdict dicts
    decision_id: str = ""
    history: list = field(default_factory=list)       # [{at, from, to, note}]
    created_at: str = field(default_factory=now)
    direction: str = ""                               # the user's research direction when proposed
    triage: dict = field(default_factory=dict)        # {rounds: [...], outcome, agreed_round, escalated}
    escalation: dict = field(default_factory=dict)    # {stage: triage|verdict, at, reason}
    revisions: list = field(default_factory=list)     # [{at, round, patch, note}]

    # -- validation ---------------------------------------------------------------
    def validate(self, metrics: set[str] | None = None):
        _require(ID_PATTERN.match(self.id or ""), f"idea id must look like R-001, got {self.id!r}")
        _require(self.title.strip(), f"{self.id}: title is empty")
        _require(self.hypothesis.strip(), f"{self.id}: hypothesis is empty")
        _require(self.falsified_by.strip(), f"{self.id}: must state what observation would refute it")
        _require(self.status in STATUSES, f"{self.id}: unknown status {self.status!r}")
        self.prediction.validate(metrics)
        self.protocol.validate()
        for v in self.verdicts:
            Verdict(**v).validate()
        if self.status in (STATUS_RUNNING, STATUS_ANALYSED, STATUS_DECIDED):
            _require(self.approval.get("by"), f"{self.id}: is {self.status} but was never approved")
        if self.status in (STATUS_ANALYSED, STATUS_DECIDED):
            _require(self.result.get("metrics"), f"{self.id}: is {self.status} but has no result metrics")
        if self.status == STATUS_ESCALATED:
            stage = self.escalation.get("stage")
            _require(stage in (STAGE_TRIAGE, STAGE_VERDICT), f"{self.id}: escalated without a stage")
            if stage == STAGE_VERDICT:
                _require(self.approval.get("by") and self.result.get("metrics"),
                         f"{self.id}: escalated at the verdict stage needs an approval and a result")
            else:
                _require(self.triage.get("rounds"), f"{self.id}: escalated at triage without any triage round")
        if self.status == STATUS_DECIDED:
            _require(self.decision_id, f"{self.id}: decided without a decision id")
        for r in self.triage.get("rounds", []):
            adv = r.get("advisor") or {}
            imp = r.get("implementer") or {}
            _require(adv.get("decision") in TRIAGE_POSITIONS, f"{self.id}: triage round {r.get('round')} has no advisor decision")
            if imp:
                _require(imp.get("position") in TRIAGE_POSITIONS, f"{self.id}: triage round {r.get('round')} bad implementer position")

    # -- transitions --------------------------------------------------------------
    def transition(self, to: str, note: str = ""):
        _require(to in TRANSITIONS.get(self.status, set()), f"{self.id}: cannot go from {self.status} to {to}")
        self.history.append({"at": now(), "from": self.status, "to": to, "note": note})
        self.status = to

    def verdict_from(self, source: str) -> Verdict | None:
        found = [Verdict(**v) for v in self.verdicts if v.get("source") == source]
        return found[-1] if found else None

    def add_verdict(self, verdict: Verdict):
        verdict.validate()
        self.verdicts.append(asdict(verdict))

    # -- io -----------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ResearchIdea":
        d = dict(d)
        d["prediction"] = Prediction(**d["prediction"])
        d["protocol"] = Protocol(**d["protocol"])
        return cls(**d)

    def path(self, root: str = RESEARCH_ROOT) -> str:
        return os.path.join(root, IDEAS_DIR, f"{self.id}.json")

    def save(self, root: str = RESEARCH_ROOT) -> str:
        self.validate()
        path = self.path(root)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            f.write("\n")
        return path

    @classmethod
    def load(cls, path: str) -> "ResearchIdea":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


@dataclass
class Decision:
    id: str
    idea_id: str
    outcome: str                 # supported | refuted | inconclusive
    agreed: bool
    advisor_verdict: dict
    implementer_verdict: dict
    user_ruling: dict | None = None
    summary: str = ""
    next_step: str = ""
    recorded_at: str = field(default_factory=now)

    def validate(self):
        _require(self.outcome in VERDICTS, f"decision outcome must be one of {VERDICTS}")
        _require(self.summary.strip(), "a decision needs a one-paragraph summary for the handoff")
        if not self.agreed:
            _require(self.user_ruling, "a decision the two parties did not agree on needs the user's ruling")

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
#  Ledger access
# ---------------------------------------------------------------------------


def ideas_dir(root: str = RESEARCH_ROOT) -> str:
    return os.path.join(root, IDEAS_DIR)


def load_ideas(root: str = RESEARCH_ROOT) -> list:
    directory = ideas_dir(root)
    if not os.path.isdir(directory):
        return []
    ideas = []
    for name in sorted(os.listdir(directory)):
        if name.endswith(".json"):
            ideas.append(ResearchIdea.load(os.path.join(directory, name)))
    return ideas


def find_idea(idea_id: str, root: str = RESEARCH_ROOT) -> ResearchIdea:
    path = os.path.join(ideas_dir(root), f"{idea_id}.json")
    _require(os.path.isfile(path), f"no idea {idea_id!r} under {ideas_dir(root)}")
    return ResearchIdea.load(path)


def next_idea_id(root: str = RESEARCH_ROOT) -> str:
    numbers = [int(i.id.split("-", 1)[1]) for i in load_ideas(root)]
    return f"R-{(max(numbers) + 1 if numbers else 1):03d}"


def load_decisions(root: str = RESEARCH_ROOT) -> list:
    return read_jsonl(os.path.join(root, DECISIONS_FILE))


def append_decision(decision: Decision, root: str = RESEARCH_ROOT) -> str:
    decision.validate()
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, DECISIONS_FILE)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(decision.to_dict(), ensure_ascii=False) + "\n")
    return path


def next_decision_id(root: str = RESEARCH_ROOT) -> str:
    return f"D-{len(load_decisions(root)) + 1:03d}"


def verdict_rounds(idea: ResearchIdea) -> int:
    """How many times the advisor has stated a verdict (initial analysis plus reviews)."""
    return sum(1 for v in idea.verdicts if v.get("source") == SOURCE_ADVISOR)


def decide(idea: ResearchIdea, root: str = RESEARCH_ROOT, user_ruling: dict | None = None,
           summary: str = "", next_step: str = "", max_rounds: int = DEFAULT_MAX_ROUNDS) -> Decision | None:
    """Compare the latest two verdicts. Agreement writes a decision.

    Disagreement returns ``None`` and leaves the idea ``analysed`` while fewer than
    ``max_rounds`` advisor verdicts exist, so the implementer can restate (``verdict``)
    and ask the advisor to reconsider (``decide --review``) again. Once the rounds are
    exhausted the idea is ``escalated`` and only ``user_ruling``
    (``{"outcome": ..., "note": ...}``) decides it; the disagreement is kept in the entry.
    """
    advisor = idea.verdict_from(SOURCE_ADVISOR)
    implementer = idea.verdict_from(SOURCE_IMPLEMENTER)
    _require(advisor is not None, f"{idea.id}: no advisor verdict yet (run analyse)")
    _require(implementer is not None, f"{idea.id}: no implementer verdict yet (run verdict)")
    agreed = advisor.verdict == implementer.verdict
    if not agreed and not user_ruling:
        rounds = verdict_rounds(idea)
        if rounds >= max_rounds and idea.status != STATUS_ESCALATED:
            idea.escalation = {"stage": STAGE_VERDICT, "at": now(),
                               "reason": f"{rounds} advisor verdicts without agreement "
                                         f"(advisor {advisor.verdict}, implementer {implementer.verdict})"}
            idea.transition(STATUS_ESCALATED, idea.escalation["reason"])
        idea.save(root)
        return None
    outcome = user_ruling["outcome"] if user_ruling else advisor.verdict
    decision = Decision(
        id=next_decision_id(root),
        idea_id=idea.id,
        outcome=outcome,
        agreed=agreed,
        advisor_verdict=asdict(advisor),
        implementer_verdict=asdict(implementer),
        user_ruling=user_ruling,
        summary=summary or implementer.reasoning or advisor.reasoning,
        next_step=next_step or implementer.next_experiment or advisor.next_experiment,
    )
    append_decision(decision, root)
    idea.decision_id = decision.id
    idea.transition(STATUS_DECIDED, f"{decision.id}: {outcome}" + ("" if agreed else " (user ruling)"))
    idea.save(root)
    return decision


def load_constraints(root: str = RESEARCH_ROOT) -> list:
    """Standing facts every advisor prompt carries, each with its source. Hand-maintained."""
    path = os.path.join(root, CONSTRAINTS_FILE)
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for entry in data:
        _require(entry.get("fact") and entry.get("source"), "every constraint needs a fact and a source")
    return data


# ---------------------------------------------------------------------------
#  Research direction (the user's input to the loop)
# ---------------------------------------------------------------------------


def load_direction(root: str = RESEARCH_ROOT) -> dict:
    path = os.path.join(root, DIRECTION_FILE)
    if not os.path.isfile(path):
        return {"text": "", "set_at": "", "history": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_direction(text: str, root: str = RESEARCH_ROOT, by: str = SOURCE_USER) -> dict:
    """Record the user's current research direction; earlier directions stay in ``history``."""
    _require(text and text.strip(), "a direction must not be empty")
    current = load_direction(root)
    if current.get("text"):
        current.setdefault("history", []).append(
            {"text": current["text"], "set_at": current.get("set_at", ""), "replaced_at": now()})
    current.update({"text": text.strip(), "set_at": now(), "by": by})
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, DIRECTION_FILE), "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return current


# ---------------------------------------------------------------------------
#  Append-only inspection logs
# ---------------------------------------------------------------------------


def _append_jsonl(root: str, name: str, entry: dict) -> str:
    return append_jsonl(os.path.join(root, name), [entry], ensure_ascii=False)


def log_proposal(root: str, call_id: str, briefing_sha256: str, direction: str, index: int,
                 raw: dict, idea_id: str = "", parse_error: str = "") -> str:
    """Every proposal the advisor made, parsed into an idea or not, with the call that produced it."""
    return _append_jsonl(root, PROPOSALS_LOG, {
        "at": now(), "call_id": call_id, "briefing_sha256": briefing_sha256, "direction": direction,
        "index": index, "idea_id": idea_id, "parse_error": parse_error, "raw": raw,
    })


def load_jsonl(root: str, name: str) -> list:
    return read_jsonl(os.path.join(root, name))


# ---------------------------------------------------------------------------
#  Triage: advisor and implementer must agree before anything is spent
# ---------------------------------------------------------------------------


def triage_rounds(idea: ResearchIdea) -> list:
    return idea.triage.setdefault("rounds", [])


def open_triage_round(idea: ResearchIdea) -> dict | None:
    """The latest round if the implementer has not answered it yet."""
    rounds = triage_rounds(idea)
    if rounds and not rounds[-1].get("implementer"):
        return rounds[-1]
    return None


def record_triage_advice(idea: ResearchIdea, advice: dict, call_id: str, briefing_sha256: str = "") -> dict:
    """Open a new round with the advisor's decision. Refuses while a round awaits the implementer."""
    _require(idea.status in (STATUS_PROPOSED, STATUS_DEFERRED), f"{idea.id} is {idea.status}; only a proposed or deferred idea is triaged")
    _require(open_triage_round(idea) is None, f"{idea.id}: round {len(triage_rounds(idea))} still awaits the implementer's position")
    decision = str(advice.get("decision", "")).strip()
    _require(decision in TRIAGE_POSITIONS, f"{idea.id}: advisor decision must be one of {TRIAGE_POSITIONS}, got {decision!r}")
    if idea.status == STATUS_DEFERRED:
        idea.transition(STATUS_PROPOSED, "re-triaged")
    rounds = triage_rounds(idea)
    entry = {
        "round": len(rounds) + 1,
        "advisor": {
            "decision": decision,
            "reasoning": str(advice.get("reasoning", "")),
            "suggested_changes": list(advice.get("suggested_changes", []) or []),
            "priority": advice.get("priority"),
            "call_id": call_id,
            "briefing_sha256": briefing_sha256,
            "at": now(),
        },
        "implementer": None,
        "outcome": None,
    }
    rounds.append(entry)
    return entry


def record_triage_position(idea: ResearchIdea, position: str, reasoning: str, root: str = RESEARCH_ROOT,
                           changes_applied: Sequence[str] = (), max_rounds: int = DEFAULT_MAX_ROUNDS) -> dict:
    """Close the open round with the implementer's position and resolve it.

    Returns ``{"outcome": agreed|disagreed|escalated, "position": ..., "status": ...}``.
    Agreement on approve/defer/reject moves the idea to that status (approval is recorded as
    advisor+implementer with both call ids). Agreement on *revise* keeps the idea proposed:
    the implementer applies the changes (``revise``) and the next round re-triages the revised
    idea. Disagreement leaves the idea proposed for another round until ``max_rounds``, then
    escalates to the user.
    """
    _require(position in TRIAGE_POSITIONS, f"position must be one of {TRIAGE_POSITIONS}, got {position!r}")
    _require(reasoning and reasoning.strip(), "the implementer's position needs its reasoning")
    rnd = open_triage_round(idea)
    _require(rnd is not None, f"{idea.id}: no open triage round (run triage first)")
    rnd["implementer"] = {"position": position, "reasoning": reasoning.strip(),
                          "changes_applied": list(changes_applied), "at": now()}
    advisor_decision = rnd["advisor"]["decision"]
    agreed = advisor_decision == position
    if agreed:
        rnd["outcome"] = "agreed"
        if position in TRIAGE_FINAL_STATUS:
            idea.triage["outcome"] = position
            idea.triage["agreed_round"] = rnd["round"]
            if position == TRIAGE_APPROVE:
                idea.approval = {"by": f"{SOURCE_ADVISOR}+{SOURCE_IMPLEMENTER}", "at": now(),
                                 "advisor_call_id": rnd["advisor"]["call_id"], "round": rnd["round"],
                                 "note": rnd["advisor"]["reasoning"][:300]}
            idea.transition(TRIAGE_FINAL_STATUS[position], f"triage round {rnd['round']}: both say {position}")
        else:  # both say revise: stays proposed, awaiting the revision and the next round
            idea.triage["outcome"] = None
        outcome = "agreed"
    elif len(triage_rounds(idea)) >= max_rounds:
        rnd["outcome"] = "escalated"
        idea.triage["escalated"] = True
        idea.escalation = {"stage": STAGE_TRIAGE, "at": now(),
                           "reason": f"{len(triage_rounds(idea))} rounds without agreement "
                                     f"(advisor {advisor_decision}, implementer {position})"}
        idea.transition(STATUS_ESCALATED, idea.escalation["reason"])
        outcome = "escalated"
    else:
        rnd["outcome"] = "disagreed"
        outcome = "disagreed"
    idea.save(root)
    _append_jsonl(root, TRIAGE_LOG, {
        "at": now(), "idea_id": idea.id, "title": idea.title, "round": rnd["round"], "direction": idea.direction,
        "advisor": rnd["advisor"], "implementer": rnd["implementer"], "outcome": outcome, "status": idea.status,
    })
    return {"outcome": outcome, "position": position, "status": idea.status, "round": rnd["round"]}


def apply_revision(idea: ResearchIdea, patch: dict, note: str = "", root: str = RESEARCH_ROOT,
                   metrics: set[str] | None = None) -> ResearchIdea:
    """Merge a patch into a proposed idea's hypothesis, prediction, protocol or cost and record it.

    Only the fields an advisor may ask to change are patchable; status, approval, result and the
    logs are not. The revised idea is validated before it is saved.
    """
    _require(idea.status in (STATUS_PROPOSED, STATUS_DEFERRED), f"{idea.id} is {idea.status}; only a proposed idea can be revised")
    allowed = {"title", "hypothesis", "falsified_by", "prediction", "protocol", "cost_estimate", "open_question_refs"}
    unknown = set(patch) - allowed
    _require(not unknown, f"revision touches fields that are not revisable: {sorted(unknown)}")
    for key, value in patch.items():
        if key == "prediction":
            idea.prediction = Prediction(**{**asdict(idea.prediction), **value})
        elif key == "protocol":
            idea.protocol = Protocol(**{**asdict(idea.protocol), **value})
        else:
            setattr(idea, key, value)
    idea.validate(metrics)
    idea.revisions.append({"at": now(), "round": len(triage_rounds(idea)), "patch": patch, "note": note})
    idea.save(root)
    return idea


def rule_triage(idea: ResearchIdea, outcome: str, note: str, root: str = RESEARCH_ROOT) -> ResearchIdea:
    """The user's override for an idea stuck in triage (escalated, proposed or deferred)."""
    _require(outcome in TRIAGE_FINAL_STATUS, f"outcome must be one of {sorted(TRIAGE_FINAL_STATUS)}")
    _require(note and note.strip(), "a ruling needs a note")
    _require(idea.status in (STATUS_PROPOSED, STATUS_DEFERRED, STATUS_ESCALATED),
             f"{idea.id} is {idea.status}; a ruling applies to a proposed, deferred or escalated idea")
    if idea.status == STATUS_ESCALATED:
        _require(idea.escalation.get("stage") == STAGE_TRIAGE, f"{idea.id} is escalated at the verdict stage; use decide --ruling")
    idea.triage["outcome"] = outcome
    idea.triage["user_ruling"] = {"outcome": outcome, "note": note, "at": now()}
    if outcome == TRIAGE_APPROVE:
        idea.approval = {"by": SOURCE_USER, "at": now(), "note": note}
    idea.transition(TRIAGE_FINAL_STATUS[outcome], f"user ruling: {note}")
    idea.save(root)
    _append_jsonl(root, TRIAGE_LOG, {"at": now(), "idea_id": idea.id, "title": idea.title, "round": None,
                                     "user_ruling": idea.triage["user_ruling"], "outcome": "user_ruling",
                                     "status": idea.status})
    return idea
