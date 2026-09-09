"""Structured contracts for search (``spec.md`` 7.2, 7.5, 7.6; FR-5, FR-10, AC-4).

Three schemas, each enforcing a different discipline:

``PolicyIdea``
    A design hypothesis specified tightly enough to be *initialisable*. Every
    parameter declares a role, a type, units, an initial value and a range, and
    the idea states what the untrained policy should already do. Free text
    cannot be checked against the code it produced; this can.

``MutationProposal``
    An edit stated as a testable hypothesis. A proposal is rejected before any
    code is generated if its predicted effect names nothing the reports measure,
    because an unmeasurable prediction cannot be wrong and so teaches nothing.

``ExperimentRecord``
    One record per attempted candidate, including the ones that never compiled.
    Keeping only the winners is how a search loses the evidence that would have
    explained why it was not improving.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

# ---------------------------------------------------------------------------
#  Vocabulary
# ---------------------------------------------------------------------------

PARAM_STRUCTURAL = "structural"
PARAM_GAIN = "gain"
PARAMETER_TYPES = (PARAM_STRUCTURAL, PARAM_GAIN)

MODE_REPAIR = "repair"
MODE_SIMPLIFICATION = "simplification"
MODE_CROSSOVER = "crossover"
MODE_EXPLORATION = "exploration"
MODE_PARAMETER_ONLY = "parameter_only"
MUTATION_MODES = (
    MODE_REPAIR,
    MODE_SIMPLIFICATION,
    MODE_CROSSOVER,
    MODE_EXPLORATION,
    MODE_PARAMETER_ONLY,
)

#: Modes that must name one bounded component and one predicted change. An
#: exploration is allowed to be broad; everything else is a stated hypothesis.
BOUNDED_MODES = (MODE_REPAIR, MODE_SIMPLIFICATION, MODE_PARAMETER_ONLY)

STATUS_GENERATED = "generated"
STATUS_INVALID = "invalid"
STATUS_FITTED = "fitted"
STATUS_EVALUATED = "evaluated"
STATUS_PROMOTED = "promoted"
STATUS_REJECTED = "rejected"
STATUSES = (
    STATUS_GENERATED,
    STATUS_INVALID,
    STATUS_FITTED,
    STATUS_EVALUATED,
    STATUS_PROMOTED,
    STATUS_REJECTED,
)


def measurable_metrics() -> set[str]:
    """Metric names a proposal may predict a change in.

    Drawn from the report sections rather than hard-coded, so a prediction can
    only reference something the pipeline actually measures.
    """
    from dataclasses import fields

    from lares.eval.report import FittingSection, RolloutSection

    names = {f.name for f in fields(RolloutSection)}
    names |= {f"fitting.{f.name}" for f in fields(FittingSection)}
    names |= {
        # Per-episode columns a focused validation can key on.
        "failure_label",
        "final_goal_distance",
        "signed_goal_progress",
        "lateral_drift",
        "min_tcp_object_distance",
        "object_displacement",
        "action_saturation_by_axis",
        "gate_statistics",
    }
    return names


class SchemaError(ValueError):
    """A structured object did not satisfy its contract."""


def _require(condition, message):
    if not condition:
        raise SchemaError(message)


def source_hash(source: str) -> str:
    """Stable digest used to tie an evaluation back to exact code."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
#  PolicyIdea
# ---------------------------------------------------------------------------


@dataclass
class IdeaParameter:
    name: str
    role: str
    type: str
    units: str
    initial_value: float
    range: tuple

    def validate(self):
        _require(self.name, "parameter needs a name")
        _require(self.role, f"parameter {self.name!r} needs a role")
        _require(
            self.type in PARAMETER_TYPES,
            f"parameter {self.name!r} type must be one of {PARAMETER_TYPES}, got {self.type!r}",
        )
        _require(
            isinstance(self.range, (tuple, list)) and len(self.range) == 2,
            f"parameter {self.name!r} needs a (lo, hi) range",
        )
        lo, hi = float(self.range[0]), float(self.range[1])
        _require(lo < hi, f"parameter {self.name!r} has lo={lo} >= hi={hi}")
        _require(
            lo <= float(self.initial_value) <= hi,
            f"parameter {self.name!r} initialises at {self.initial_value}, "
            f"outside its own range [{lo}, {hi}]",
        )
        return True


@dataclass
class IdeaPhase:
    name: str
    activation: str
    action_intent: str

    def validate(self):
        _require(self.name, "phase needs a name")
        _require(self.activation, f"phase {self.name!r} needs an activation condition")
        _require(self.action_intent, f"phase {self.name!r} needs an action intent")
        return True


@dataclass
class PolicyIdea:
    """A design hypothesis concrete enough to instantiate before any training."""

    idea_id: str
    name: str
    summary: str
    named_observations: list = field(default_factory=list)
    phases: list = field(default_factory=list)
    parameters: list = field(default_factory=list)
    zero_shot_behavior: str = ""
    expected_failure: str = ""
    protected_behaviors: list = field(default_factory=list)

    def validate(self, schema=None):
        _require(self.idea_id, "idea needs an id")
        _require(self.name, "idea needs a name")
        _require(len(self.summary) >= 20, "idea summary is too short to implement from")
        _require(self.phases, f"idea {self.idea_id!r} declares no phases")
        _require(self.parameters, f"idea {self.idea_id!r} declares no parameters")
        _require(
            self.zero_shot_behavior,
            f"idea {self.idea_id!r} must say what the untrained policy should do; "
            f"the zero-shot checkpoint is evaluated and an unstated expectation "
            f"cannot be checked against it",
        )
        for phase in self.phases:
            phase.validate()
        seen = set()
        for parameter in self.parameters:
            parameter.validate()
            _require(
                parameter.name not in seen, f"duplicate parameter {parameter.name!r}"
            )
            seen.add(parameter.name)
        if schema is not None:
            unknown = [n for n in self.named_observations if n not in schema.names]
            _require(
                not unknown,
                f"idea {self.idea_id!r} names observation fields that do not exist: "
                f"{unknown}. Available: {sorted(schema.names)}",
            )
        return True

    def check_against_policy(self, policy) -> list:
        """Differences between what the idea declared and what the code registered."""
        declared = {p.name for p in self.parameters}
        registered = {name for name, _ in policy.named_parameters()}
        problems = []
        for name in sorted(registered - declared):
            problems.append(f"parameter {name!r} exists in the code but not in the idea")
        for name in sorted(declared - registered):
            problems.append(f"idea declares {name!r}, which the code does not register")
        return problems

    @property
    def complexity(self) -> dict:
        """Measured, not assumed (``spec.md`` 7.2)."""
        return {
            "num_parameters": len(self.parameters),
            "num_phases": len(self.phases),
            "num_structural": sum(1 for p in self.parameters if p.type == PARAM_STRUCTURAL),
            "num_gains": sum(1 for p in self.parameters if p.type == PARAM_GAIN),
        }

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PolicyIdea":
        return cls(
            idea_id=str(d.get("idea_id", "")),
            name=str(d.get("name", "")),
            summary=str(d.get("summary", "")),
            named_observations=[str(x) for x in d.get("named_observations", [])],
            phases=[
                IdeaPhase(
                    name=str(p.get("name", "")),
                    activation=str(p.get("activation", "")),
                    action_intent=str(p.get("action_intent", "")),
                )
                for p in d.get("phases", [])
            ],
            parameters=[
                IdeaParameter(
                    name=str(p.get("name", "")),
                    role=str(p.get("role", "")),
                    type=str(p.get("type", PARAM_GAIN)),
                    units=str(p.get("units", "")),
                    initial_value=float(p.get("initial_value", 0.0)),
                    range=tuple(p.get("range", (0.0, 1.0))),
                )
                for p in d.get("parameters", [])
            ],
            zero_shot_behavior=str(d.get("zero_shot_behavior", "")),
            expected_failure=str(d.get("expected_failure", "")),
            protected_behaviors=[str(x) for x in d.get("protected_behaviors", [])],
        )

    def implementation_brief(self) -> str:
        """The idea rendered for the implementation prompt."""
        lines = [f"### {self.name} ({self.idea_id})", self.summary, "", "Phases:"]
        for phase in self.phases:
            lines.append(
                f"  - {phase.name}: active when {phase.activation}; "
                f"action {phase.action_intent}"
            )
        lines.append("")
        lines.append("Parameters (declare every one of these, and nothing else):")
        for p in self.parameters:
            lines.append(
                f"  - {p.name} ({p.type}, {p.units}): {p.role}; "
                f"init {p.initial_value}, range {tuple(p.range)}"
            )
        lines.append("")
        lines.append(f"Observation fields used: {', '.join(self.named_observations)}")
        lines.append(f"Untrained behaviour should be: {self.zero_shot_behavior}")
        if self.expected_failure:
            lines.append(f"Expected failure mode: {self.expected_failure}")
        if self.protected_behaviors:
            lines.append(f"Must not lose: {'; '.join(self.protected_behaviors)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
#  MutationProposal
# ---------------------------------------------------------------------------


@dataclass
class MutationProposal:
    """An edit stated as a hypothesis, with a prediction that can be wrong."""

    proposal_id: str
    parent_ids: list
    mode: str
    observation: str
    hypothesis: str
    suspected_component: str
    intervention: str
    predicted_metric: str = ""
    predicted_direction: str = ""
    protected_behaviors: list = field(default_factory=list)
    focused_cases: list = field(default_factory=list)

    DIRECTIONS = ("increase", "decrease")

    def validate(self, metrics: set | None = None):
        metrics = metrics if metrics is not None else measurable_metrics()
        _require(self.proposal_id, "proposal needs an id")
        _require(
            self.mode in MUTATION_MODES,
            f"mode must be one of {MUTATION_MODES}, got {self.mode!r}",
        )
        _require(self.observation, "proposal must name the observation that prompted it")
        if self.mode == MODE_EXPLORATION:
            # Exploration is allowed to be broad; that is what distinguishes it.
            return True
        _require(self.hypothesis, f"{self.mode} proposals need one primary hypothesis")
        _require(
            self.suspected_component,
            f"{self.mode} proposals must name the component they suspect",
        )
        _require(self.intervention, f"{self.mode} proposals must describe the edit")
        _require(
            self.predicted_metric in metrics,
            f"predicted metric {self.predicted_metric!r} is not measured by the "
            f"evaluation report, so the prediction could never be wrong. "
            f"Pick one of: {sorted(metrics)}",
        )
        _require(
            self.predicted_direction in self.DIRECTIONS,
            f"predicted direction must be one of {self.DIRECTIONS}, "
            f"got {self.predicted_direction!r}",
        )
        _require(
            self.protected_behaviors,
            f"{self.mode} proposals must name at least one behaviour the edit must "
            f"not lose, so a regression is detectable",
        )
        if self.mode in (MODE_REPAIR, MODE_PARAMETER_ONLY):
            _require(
                len(self.parent_ids) == 1,
                f"a {self.mode} edits one parent; got {len(self.parent_ids)}",
            )
        if self.mode == MODE_CROSSOVER:
            _require(
                len(self.parent_ids) >= 2,
                "a crossover needs at least two parents",
            )
        return True

    @property
    def bypasses_llm(self) -> bool:
        """Parameter-only search is the numerical optimizer's job (FR-5)."""
        return self.mode == MODE_PARAMETER_ONLY

    def check_outcome(self, before, after) -> dict:
        """Did the predicted metric move in the predicted direction?"""
        if before is None or after is None:
            return {"measurable": False, "reason": "metric unavailable on one side"}
        delta = float(after) - float(before)
        moved = delta > 0 if self.predicted_direction == "increase" else delta < 0
        return {
            "measurable": True,
            "metric": self.predicted_metric,
            "before": float(before),
            "after": float(after),
            "delta": delta,
            "predicted_direction": self.predicted_direction,
            "as_predicted": bool(moved),
        }

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MutationProposal":
        return cls(
            proposal_id=str(d.get("proposal_id", "")),
            parent_ids=[str(x) for x in d.get("parent_ids", [])],
            mode=str(d.get("mode", MODE_EXPLORATION)),
            observation=str(d.get("observation", "")),
            hypothesis=str(d.get("hypothesis", "")),
            suspected_component=str(d.get("suspected_component", "")),
            intervention=str(d.get("intervention", "")),
            predicted_metric=str(d.get("predicted_metric", "")),
            predicted_direction=str(d.get("predicted_direction", "")),
            protected_behaviors=[str(x) for x in d.get("protected_behaviors", [])],
            focused_cases=[str(x) for x in d.get("focused_cases", [])],
        )


# ---------------------------------------------------------------------------
#  ExperimentRecord
# ---------------------------------------------------------------------------


@dataclass
class ExperimentRecord:
    """Everything needed to reconstruct and re-evaluate one attempted candidate.

    Written for invalid and rejected candidates too. A record that only exists
    for winners cannot explain a search that is not improving.
    """

    experiment_id: str
    candidate_id: str
    generation: int
    status: str
    parent_ids: list = field(default_factory=list)
    idea_id: str = ""
    proposal_id: str = ""
    code: dict = field(default_factory=dict)
    llm: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    fitting: dict = field(default_factory=dict)
    evaluation: dict = field(default_factory=dict)
    resources: dict = field(default_factory=dict)
    timestamps: dict = field(default_factory=dict)
    #: True when the result used expert substitution or a forced gate. Such a
    #: result describes an intervention, not a policy, and must never enter
    #: ordinary fitness ranking (``spec.md`` FR-10).
    intervention: bool = False
    intervention_detail: str = ""
    final_disposition: str = ""

    def validate(self):
        _require(self.experiment_id, "record needs an experiment id")
        _require(self.candidate_id, "record needs a candidate id")
        _require(
            self.status in STATUSES,
            f"status must be one of {STATUSES}, got {self.status!r}",
        )
        _require(isinstance(self.generation, int), "generation must be an integer")
        if self.status != STATUS_GENERATED:
            _require(
                self.code.get("source_hash"),
                f"{self.candidate_id}: a record past generation must carry a source hash, "
                f"otherwise its evaluation cannot be tied to exact code",
            )
        if self.status in (STATUS_EVALUATED, STATUS_PROMOTED):
            _require(
                self.evaluation.get("manifest_ids"),
                f"{self.candidate_id}: an evaluated record must name the manifests it ran",
            )
            _require(
                self.evaluation.get("action_mode"),
                f"{self.candidate_id}: an evaluated record must name its action mode",
            )
        if self.intervention:
            _require(
                self.intervention_detail,
                "an intervention record must say what was substituted",
            )
        return True

    @property
    def eligible_for_ranking(self) -> bool:
        """Intervention results are evidence, not fitness."""
        return not self.intervention

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentRecord":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.experiment_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: str) -> "ExperimentRecord":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def load_records(directory: str) -> list:
    """Every record in a directory, sorted by id."""
    if not os.path.isdir(directory):
        return []
    out = []
    for name in sorted(os.listdir(directory)):
        if name.endswith(".json"):
            out.append(ExperimentRecord.load(os.path.join(directory, name)))
    return out
