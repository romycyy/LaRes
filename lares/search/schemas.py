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

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from lares.utils.hashing import short_hash_text

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
#: A bounded, reference-seeded edit in the fittable lane (decided 2026-09-14). Only the lane's own
#: seeded request produces it: one seed-bank parent, one declared structural change, and a
#: prediction about a fitting diagnostic.
MODE_FITTABLE_MUTATION = "fittable_mutation"
MUTATION_MODES = (
    MODE_REPAIR,
    MODE_SIMPLIFICATION,
    MODE_CROSSOVER,
    MODE_EXPLORATION,
    MODE_PARAMETER_ONLY,
    MODE_FITTABLE_MUTATION,
)

#: Modes that must name one bounded component and one predicted change. An
#: exploration is allowed to be broad; everything else is a stated hypothesis.
BOUNDED_MODES = (MODE_REPAIR, MODE_SIMPLIFICATION, MODE_PARAMETER_ONLY)

#: Search lanes (decided 2026-09-14). The fittable lane holds reference-seeded bounded mutations and
#: is the only lane that may run rollout-selected fitting; the exploration lane holds unseeded,
#: zero-shot designs that never consume fitting or probe resources.
LANE_FITTABLE = "fittable"
LANE_EXPLORATION = "exploration"
LANES = (LANE_FITTABLE, LANE_EXPLORATION)
FITTABLE_LABEL = "reference_seeded_bounded_mutation"
EXPLORATION_LABEL = "unseeded_exploration"
#: How a selected step of 0 is recorded: the probe preferred the untrained parameters.
FITTING_REJECTED = "fitting_rejected"

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


def fitting_metrics() -> set[str]:
    """The fitting diagnostics a fittable-lane mutation may predict an improvement in."""
    return {m for m in measurable_metrics() if m.startswith("fitting.")}


class SchemaError(ValueError):
    """A structured object did not satisfy its contract."""


def _require(condition, message):
    if not condition:
        raise SchemaError(message)


def source_hash(source: str) -> str:
    """Stable digest used to tie an evaluation back to exact code."""
    return short_hash_text(source)


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
    #: Required when the idea exceeds the declared parameter or phase budget.
    complexity_justification: str = ""

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
            complexity_justification=str(d.get("complexity_justification", "") or ""),
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
    #: Why an exploration names no parent. Required when ``parent_ids`` is empty.
    no_parent_rationale: str = ""

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
        if self.mode == MODE_FITTABLE_MUTATION:
            _require(
                self.predicted_metric.startswith("fitting."),
                f"a fittable mutation predicts which fitting diagnostic improves; "
                f"{self.predicted_metric!r} is not one of {sorted(fitting_metrics())}",
            )
        if self.mode in (MODE_REPAIR, MODE_SIMPLIFICATION, MODE_PARAMETER_ONLY, MODE_FITTABLE_MUTATION):
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
            no_parent_rationale=str(d.get("no_parent_rationale", "") or ""),
        )


# ---------------------------------------------------------------------------
#  ExperimentRecord
# ---------------------------------------------------------------------------


#: Where a record came from. Generated candidates carry the full lineage and
#: provenance contract (``debug_spec.md`` Stage C); a repair study and a legacy
#: record predate or sit outside that loop and keep the older, weaker rules.
ORIGIN_GENERATED = "generated"
ORIGIN_REPAIR_STUDY = "repair_study"
ORIGIN_LEGACY = "legacy"
ORIGINS = (ORIGIN_GENERATED, ORIGIN_REPAIR_STUDY, ORIGIN_LEGACY)

#: Lineage completeness, distinguished in every summary (``debug_spec.md`` 6).
LINEAGE_COMPLETE = "complete"
LINEAGE_PARTIAL_RECOVERY = "partial_recovery"
LINEAGE_LEGACY_MISSING = "legacy_missing"
LINEAGE_FAILED_VALIDATION = "failed_validation"
LINEAGE_STATUSES = (
    LINEAGE_COMPLETE,
    LINEAGE_PARTIAL_RECOVERY,
    LINEAGE_LEGACY_MISSING,
    LINEAGE_FAILED_VALIDATION,
)

#: Failure stages reached before an idea exists, so an empty idea id is expected.
PRE_IDEA_STAGES = ("ideation", "proposal", "budget_exhausted")

#: Statuses that mean the candidate was fitted and rolled out.
EVALUATED_STATUSES = (STATUS_EVALUATED, STATUS_PROMOTED, STATUS_REJECTED)

RECONSTRUCTION_VERIFIED = "verified"
RECONSTRUCTION_FAILED = "failed"


def _ref_ok(ref) -> bool:
    return isinstance(ref, dict) and bool(ref.get("path")) and bool(ref.get("sha256"))


@dataclass
class ExperimentRecord:
    """Everything needed to reconstruct and re-evaluate one attempted candidate.

    Written for invalid and rejected candidates too. A record that only exists
    for winners cannot explain a search that is not improving.

    A generated record is one *attempt*: a slot that failed validation and was
    corrected leaves two records, linked through ``lineage.previous_attempt_id``.
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
    origin: str = ORIGIN_GENERATED
    #: Slot, attempt, feedback, context and semantic parents, idea and proposal.
    lineage: dict = field(default_factory=dict)
    #: Ids into the run-level LLM-call ledger, shared calls included.
    llm_call_ids: list = field(default_factory=list)
    #: Paths and hashes that rebuild the policy and the check that they did.
    reconstruction: dict = field(default_factory=dict)
    #: Incremented each time material evidence changes; old versions are kept.
    record_version: int = 1

    def validate(self):
        _require(self.experiment_id, "record needs an experiment id")
        _require(self.candidate_id, "record needs a candidate id")
        _require(
            self.status in STATUSES,
            f"status must be one of {STATUSES}, got {self.status!r}",
        )
        _require(isinstance(self.generation, int), "generation must be an integer")
        _require(self.origin in ORIGINS, f"origin must be one of {ORIGINS}, got {self.origin!r}")
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
        if self.origin == ORIGIN_GENERATED:
            self._validate_generated()
        return True

    # -- the generated-candidate contract -----------------------------------

    def _validate_generated(self):
        from lares.search.resources import FORBIDDEN_KEYS, resource_problems

        who = self.experiment_id
        lineage = self.lineage or {}
        _require(lineage, f"{who}: a generated record must carry lineage, even when it failed")
        for key in ("slot", "attempt"):
            value = lineage.get(key)
            _require(
                isinstance(value, int) and value >= 0,
                f"{who}: lineage.{key} must be a non-negative integer allocated before generation",
            )
        _require(lineage.get("attempt_id"), f"{who}: lineage needs an attempt id")
        _require(lineage.get("generation") == self.generation, f"{who}: lineage generation disagrees")
        _require(
            lineage.get("status") in LINEAGE_STATUSES,
            f"{who}: lineage.status must be one of {LINEAGE_STATUSES}",
        )
        _require(
            isinstance(self.validation.get("passed"), bool),
            f"{who}: a generated record must carry a validation result",
        )
        _require(self.code.get("source_hash"), f"{who}: a generated record needs a code hash")
        _require(
            self.llm_call_ids,
            f"{who}: a generated record must reference the LLM calls that produced it",
        )
        calls = self.llm.get("calls")
        _require(isinstance(calls, list) and calls, f"{who}: llm.calls must embed the call references")
        _require(
            {c.get("call_id") for c in calls} == set(self.llm_call_ids),
            f"{who}: llm.calls and llm_call_ids disagree",
        )
        for call in calls:
            _require(
                call.get("prompt_sha256") and call.get("response_sha256"),
                f"{who}: call {call.get('call_id')} is missing its prompt or response hash",
            )
        for key in FORBIDDEN_KEYS:
            _require(key not in self.resources, f"{who}: resources.{key} is forbidden; name the stream")

        stage = self.validation.get("stage", "")
        if stage not in PRE_IDEA_STAGES:
            _require(
                self.idea_id and lineage.get("idea"),
                f"{who}: a record past ideation must name the idea it implemented",
            )
        lane = lineage.get("lane") or ""
        _require(not lane or lane in LANES, f"{who}: lineage.lane must be one of {LANES}, got {lane!r}")
        if lane == LANE_FITTABLE:
            self._validate_fittable_lineage(lineage, stage)
        elif lane == LANE_EXPLORATION:
            self._validate_exploration_lineage(lineage)
        elif self.generation == 0:
            _require(
                not self.parent_ids and not self.proposal_id,
                f"{who}: generation 0 has no semantic parent and no proposal",
            )
        elif stage not in PRE_IDEA_STAGES:
            self._validate_proposal_lineage(lineage)

        if self.status == STATUS_INVALID:
            _require(not self.validation["passed"], f"{who}: an invalid record cannot have passed validation")
            _require(
                lineage["status"] == LINEAGE_FAILED_VALIDATION,
                f"{who}: an invalid record's lineage status is {LINEAGE_FAILED_VALIDATION}",
            )
        if self.status in EVALUATED_STATUSES:
            self._validate_evaluated(resource_problems)
            if lane:
                self._validate_lane_fitting(lane)

    def _validate_fittable_lineage(self, lineage, stage):
        """A reference-seeded mutation: its slot's reference and scope, then a bounded proposal."""
        who = self.experiment_id
        meta = lineage.get("lane_metadata") or {}
        reference = meta.get("reference") or {}
        scope = meta.get("mutation_scope") or {}
        _require(meta.get("label") == FITTABLE_LABEL and meta.get("seeded") is True,
                 f"{who}: a fittable-lane record must be labelled {FITTABLE_LABEL!r} and seeded")
        _require(reference.get("structure_id") and reference.get("source_sha256"),
                 f"{who}: a fittable-lane record must name its seed-bank reference and its source hash")
        _require(scope.get("id"), f"{who}: a fittable-lane record must carry its predeclared mutation scope")
        if stage in PRE_IDEA_STAGES:
            return
        proposal = lineage.get("proposal") or {}
        _require(self.proposal_id and proposal, f"{who}: a fittable mutation needs its proposal")
        _require(proposal.get("mode") == MODE_FITTABLE_MUTATION,
                 f"{who}: a fittable-lane proposal has mode {MODE_FITTABLE_MUTATION!r}")
        _require(list(self.parent_ids) == [reference["structure_id"]] == list(proposal.get("parent_ids", [])),
                 f"{who}: a fittable mutation has exactly one parent, its slot's reference "
                 f"{reference['structure_id']!r}; got {self.parent_ids}")
        _require(reference["structure_id"] in (lineage.get("context_parent_ids") or []),
                 f"{who}: the reference is not in the seed-bank context")
        _require(str(proposal.get("predicted_metric", "")).startswith("fitting."),
                 f"{who}: a fittable mutation must predict a fitting diagnostic")
        mutation = meta.get("mutation") or {}
        _require(str(mutation.get("structural_change", "")).strip(),
                 f"{who}: a fittable mutation must state its one structural change")
        for key in ("adds", "removes", "changes"):
            _require(isinstance(mutation.get(key), list), f"{who}: mutation.{key} must be a list")

    def _validate_exploration_lineage(self, lineage):
        who = self.experiment_id
        meta = lineage.get("lane_metadata") or {}
        _require(meta.get("label") == EXPLORATION_LABEL and meta.get("seeded") is False,
                 f"{who}: an exploration-lane record must be labelled {EXPLORATION_LABEL!r} and unseeded")
        _require(not self.parent_ids and not self.proposal_id and not lineage.get("proposal"),
                 f"{who}: an unseeded exploration has no parent and no proposal")

    def _validate_lane_fitting(self, lane):
        who = self.experiment_id
        step = self.fitting.get("selected_step")
        if lane == LANE_EXPLORATION:
            for key in ("bc_gradient_steps", "bc_objective_evaluations", "probe_episodes", "probe_environment_steps"):
                _require(int(self.resources.get(key, 0) or 0) == 0,
                         f"{who}: an exploration consumed {key}; that lane never fits or probes")
            _require(float((self.resources.get("wall_time_seconds") or {}).get("fitting", 0.0)) == 0.0,
                     f"{who}: an exploration spent fitting wall time")
            _require(step == 0 and self.fitting.get("checkpoint_history") == [],
                     f"{who}: an exploration is evaluated at its initial values only")
        else:
            _require(isinstance(self.fitting.get("fitting_rejected"), bool)
                     and self.fitting["fitting_rejected"] == (step == 0),
                     f"{who}: fitting.fitting_rejected must record whether step 0 was selected")
            steps = [h.get("step") for h in self.fitting.get("checkpoint_history") or []]
            _require(0 in steps, f"{who}: the fittable lane must probe step 0 so fitting can be rejected")

    def _validate_proposal_lineage(self, lineage):
        who = self.experiment_id
        proposal = lineage.get("proposal") or {}
        _require(self.proposal_id and proposal, f"{who}: generation {self.generation} needs a proposal")
        _require(lineage.get("feedback_id"), f"{who}: generation {self.generation} needs a feedback id")
        context = lineage.get("context_parent_ids")
        _require(isinstance(context, list), f"{who}: lineage.context_parent_ids must be a list")
        parents = list(self.parent_ids)
        _require(
            parents == list(proposal.get("parent_ids", [])),
            f"{who}: record parent_ids disagree with the proposal's",
        )
        outside = [p for p in parents if p not in context]
        _require(not outside, f"{who}: parents {outside} were not in the feedback context")
        mode = proposal.get("mode")
        if mode in (MODE_REPAIR, MODE_SIMPLIFICATION):
            _require(len(parents) == 1, f"{who}: a {mode} has exactly one parent")
        if mode == MODE_CROSSOVER:
            _require(len(parents) >= 2, f"{who}: a crossover has at least two parents")
        if mode == MODE_EXPLORATION and not parents:
            _require(
                str(proposal.get("no_parent_rationale", "")).strip(),
                f"{who}: a parentless exploration must say why",
            )

    def _validate_evaluated(self, resource_problems):
        who = self.experiment_id
        _require(self.validation.get("passed") is True, f"{who}: only a validated candidate is evaluated")
        _require(
            self.lineage.get("status") == LINEAGE_COMPLETE,
            f"{who}: an evaluated generated candidate must have complete lineage",
        )
        _require(self.evaluation.get("manifest_ids"), f"{who}: evaluated record must name its manifests")
        _require(self.evaluation.get("action_mode"), f"{who}: evaluated record must name its action mode")
        environment = self.evaluation.get("environment") or {}
        _require(
            environment.get("canonical_env_id") and environment.get("resolved_registry_env_id"),
            f"{who}: evaluated record must store canonical and resolved environment ids",
        )
        reports = self.evaluation.get("reports")
        _require(
            isinstance(reports, dict) and reports and all(_ref_ok(r) for r in reports.values()),
            f"{who}: evaluated record must reference its hashed evaluation reports",
        )
        for key in ("objective", "action_mode"):
            _require(self.fitting.get(key), f"{who}: fitting.{key} is required")
        _require(isinstance(self.fitting.get("objective_config"), dict), f"{who}: fitting.objective_config is required")
        _require(self.fitting.get("fixed_action_scale") is not None, f"{who}: fitting.fixed_action_scale is required")
        _require(isinstance(self.fitting.get("selected_step"), int), f"{who}: fitting.selected_step is required")
        for key in ("initial_state", "final_state", "selected_state"):
            _require(_ref_ok(self.fitting.get(key)), f"{who}: fitting.{key} must be a hashed reference")
        _require(isinstance(self.fitting.get("checkpoint_history"), list), f"{who}: fitting.checkpoint_history is required")
        for key in ("final_diagnostics", "selected_diagnostics"):
            _require(isinstance(self.fitting.get(key), dict), f"{who}: fitting.{key} is required")
        problems = resource_problems(self.resources)
        _require(not problems, f"{who}: {'; '.join(problems)}")
        recon = self.reconstruction or {}
        for key in ("source", "selected_state", "probe_batch", "expected_actions"):
            _require(_ref_ok(recon.get(key)), f"{who}: reconstruction.{key} must be a hashed reference")
        _require(
            recon.get("status") in (RECONSTRUCTION_VERIFIED, RECONSTRUCTION_FAILED),
            f"{who}: reconstruction.status must say whether the rebuild matched",
        )

    @property
    def eligible_for_ranking(self) -> bool:
        """Intervention results are evidence, not fitness."""
        return not self.intervention

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentRecord":
        known = {f for f in cls.__dataclass_fields__}
        data = {k: v for k, v in d.items() if k in known}
        # Records written before ``origin`` existed are legacy, not generated:
        # defaulting them to the generated contract would misdescribe them.
        data.setdefault("origin", ORIGIN_LEGACY)
        return cls(**data)

    def save(self, directory: str) -> str:
        """Write the record, keeping any materially different earlier version.

        Rewriting identical content is a no-op. Different content moves the
        previous file to ``history/<id>.v<N>.json`` before the new version is
        written, so evidence is superseded visibly rather than overwritten.
        """
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.experiment_id}.json")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                previous = json.load(f)
            prev_version = int(previous.get("record_version", 1) or 1)
            current = json.loads(json.dumps({**self.to_dict(), "record_version": prev_version}, default=str))
            if json.dumps(previous, sort_keys=True) == json.dumps(current, sort_keys=True):
                self.record_version = prev_version
                return path
            history = os.path.join(directory, "history")
            os.makedirs(history, exist_ok=True)
            hist_path = os.path.join(history, f"{self.experiment_id}.v{prev_version}.json")
            if os.path.exists(hist_path):
                raise FileExistsError(
                    f"{hist_path} already exists; refusing to overwrite a superseded record version"
                )
            with open(hist_path, "w", encoding="utf-8") as f:
                json.dump(previous, f, indent=2)
            self.record_version = prev_version + 1
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
