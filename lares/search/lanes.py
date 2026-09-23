"""The two-lane search: plan, allocation, mutation contract, incumbents and resources.

Decided 2026-09-14, after the fittable-bank qualification study. One generation of exactly nine
candidates in two lanes:

``fittable``
    Six reference-seeded bounded mutations. The allocation is fixed by configuration, never by
    ranking references: one slot for each of the four qualified reach-push references and two for
    ``simple_damped_push``, with two different predeclared scopes (gate/control and geometry).
    Only this lane runs rollout-selected fitting, with checkpoints {0, 500, 1000} chosen on the
    probe manifest. Step 0 stays selectable and is recorded as ``fitting_rejected``.

``exploration``
    Three unseeded, zero-shot designs, labelled ``unseeded_exploration``. They never touch the
    demonstration data, the fitter or the probe.

Both lanes are screened on the same screening manifest and challenge on the same promotion
manifest; checkpoint selection, screening and promotion use three disjoint case sets. Two
incumbents are kept apart: ``performance_incumbent`` may come from either lane, while
``fittable_incumbent`` needs a fittable-lane candidate whose selected step is above 0 and whose
fitted state beats its own step-0 state on the promotion manifest under the strict paired rule
(:mod:`lares.eval.promotion`), and which also beats the current fittable incumbent under the same
rule. Tied success never promotes either.

This module holds the parts that need no LLM and no simulator, so every rule is testable alone.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import yaml

from lares.search.schemas import (
    EXPLORATION_LABEL,
    FITTABLE_LABEL,
    FITTING_REJECTED,
    LANE_EXPLORATION,
    LANE_FITTABLE,
    LANES,
    fitting_metrics,
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: The decided shape. A plan with any other count is refused rather than adapted.
TOTAL_CANDIDATES = 9
FITTABLE_SLOTS = 6
EXPLORATION_SLOTS = 3
FITTABLE_CHECKPOINTS = (0, 500, 1000)
#: Where each decision's data comes from. Stored with the plan and checked on load.
DATA_ROLES = {
    "checkpoint_selection_data": "probe",
    "candidate_ranking_data": "screen",
    "promotion_data": "separate_confirmation",
}

ROLE_FITTABLE_SEED = "fittable_seed"
ROLE_NON_FITTABLE_CONTROL = "non_fittable_control"
ROLE_RETIRED = "retired"
STRUCTURE_ROLES = (ROLE_FITTABLE_SEED, ROLE_NON_FITTABLE_CONTROL, ROLE_RETIRED)

#: The fixed 4:2 split: one slot per reach-push seed, two for this reference, one per scope.
DAMPED_PUSH_REFERENCE = "simple_damped_push"
DAMPED_PUSH_SCOPES = ("gate_control", "geometry")
REACH_PUSH_FAMILY = "reach_push"

ALL = "*"

PERFORMANCE_INCUMBENT = "performance_incumbent"
FITTABLE_INCUMBENT = "fittable_incumbent"
NOT_FITTABLE_LANE = "not_a_fittable_lane_candidate"
FITTABLE_GAIN_REJECTED = "fitted_state_not_better_than_its_own_step_0"
FITTABLE_INCUMBENT_NOT_BEATEN = "did_not_beat_the_current_fittable_incumbent"
FITTABLE_PROMOTED = "fitted_gain_and_incumbent_intervals_above_zero"
NOT_BEST_FITTABLE_CHALLENGER = "not_the_best_screened_fittable_challenger"


class LanePlanError(ValueError):
    """A two-lane plan that does not match the decided shape."""


def _rel(path: str) -> str:
    return os.path.relpath(path, _PROJECT_ROOT)


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_PROJECT_ROOT, path)


def _sha256_file(path: str) -> str:
    from lares.search.generation import sha256_file

    return sha256_file(path)


# ---------------------------------------------------------------------------
#  Structure roles
# ---------------------------------------------------------------------------


def load_structure_roles(path: str) -> dict:
    """``{structure_id: {"role", "family", "source_path", ...}}``, every role checked."""
    with open(_abs(path), "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    structures = raw.get("structures") or {}
    for sid, entry in structures.items():
        if entry.get("role") not in STRUCTURE_ROLES:
            raise LanePlanError(f"{sid}: role must be one of {STRUCTURE_ROLES}, got {entry.get('role')!r}")
        if not os.path.isfile(_abs(entry.get("source_path", ""))):
            raise LanePlanError(f"{sid}: source {entry.get('source_path')!r} does not exist")
    return {"seed_bank": raw.get("seed_bank"), "structures": structures}


def active_parent_ids(roles: dict) -> list:
    """Structures a fittable-lane slot may name as its parent, in file order."""
    return [sid for sid, e in roles["structures"].items() if e["role"] == ROLE_FITTABLE_SEED]


def control_ids(roles: dict) -> list:
    return [sid for sid, e in roles["structures"].items() if e["role"] == ROLE_NON_FITTABLE_CONTROL]


# ---------------------------------------------------------------------------
#  Plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MutationScope:
    """Which parent parameters a mutation may change or remove. ``*`` means any."""

    scope_id: str
    description: str
    may_change: tuple = (ALL,)
    may_remove: tuple = (ALL,)
    allow_adds: bool = True

    def permits(self, kind: str, name: str) -> bool:
        allowed = self.may_change if kind == "changes" else self.may_remove
        return ALL in allowed or name in allowed

    def to_dict(self) -> dict:
        return {
            "id": self.scope_id,
            "description": self.description,
            "may_change": list(self.may_change),
            "may_remove": list(self.may_remove),
            "allow_adds": self.allow_adds,
        }


@dataclass(frozen=True)
class ReferenceStructure:
    structure_id: str
    family: str
    source_path: str
    source_sha256: str
    source: str = field(repr=False, default="")

    def to_dict(self) -> dict:
        return {"structure_id": self.structure_id, "family": self.family,
                "source_path": self.source_path, "source_sha256": self.source_sha256}


@dataclass(frozen=True)
class SlotSpec:
    """One of the nine slots, fixed before any call."""

    slot: int
    lane: str
    reference: ReferenceStructure | None = None
    scope: MutationScope | None = None

    def lane_metadata(self, seed_bank_sha256: str = "", mutation: dict | None = None) -> dict:
        if self.lane == LANE_EXPLORATION:
            return {"label": EXPLORATION_LABEL, "seeded": False, "fitting": "disabled"}
        meta = {
            "label": FITTABLE_LABEL,
            "seeded": True,
            "reference": {**self.reference.to_dict(), "seed_bank_sha256": seed_bank_sha256},
            "mutation_scope": self.scope.to_dict(),
        }
        if mutation is not None:
            meta["mutation"] = dict(mutation)
        return meta

    def as_parse_slot(self) -> dict:
        return {"slot": self.slot, "reference": self.reference.structure_id, "scope": self.scope.scope_id}


@dataclass
class TwoLanePlan:
    slots: list
    fittable_budget: int
    checkpoint_steps: tuple
    max_parameter_changes: int
    min_parent_parameters_kept: float
    seed_bank_path: str
    seed_bank_sha256: str
    roles_path: str
    roles_sha256: str
    controls: list = field(default_factory=list)
    _parent_cache: dict = field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        return len(self.slots)

    @property
    def fittable_slots(self) -> list:
        return [s for s in self.slots if s.lane == LANE_FITTABLE]

    @property
    def exploration_slots(self) -> list:
        return [s for s in self.slots if s.lane == LANE_EXPLORATION]

    def slot(self, index: int) -> SlotSpec:
        return self.slots[int(index)]

    def reference_ids(self) -> list:
        """The seed-bank context a fittable mutation's parent must come from, sorted."""
        return sorted({s.reference.structure_id for s in self.fittable_slots})

    def parent_sources(self) -> dict:
        return {s.reference.structure_id: s.reference.source for s in self.fittable_slots}

    # -- mutation contract ---------------------------------------------------

    def parent_policy(self, structure_id: str, schema=None):
        from lares.core.obs_schema import get_obs_schema
        from lares.fitting.structure_bank import instantiate_source

        ref = next(s.reference for s in self.fittable_slots if s.reference.structure_id == structure_id)
        return instantiate_source(ref.source, 39, 4, schema=schema if schema is not None else get_obs_schema("push-v2"))

    def parent_parameters(self, structure_id: str, schema=None) -> dict:
        """``{name: (values, range)}`` of the parent's non-scale parameters, cached."""
        if structure_id not in self._parent_cache:
            policy = self.parent_policy(structure_id, schema)
            self._parent_cache[structure_id] = parameter_table(policy, exclude=scale_parameter_names(policy))
        return self._parent_cache[structure_id]

    def declaration_problems(self, slot: dict, declaration: dict, idea, schema=None) -> list:
        """Parse-time check of one declared mutation against its slot, before any code exists."""
        spec = self.slot(slot["slot"])
        parent = self.parent_parameters(spec.reference.structure_id, schema)
        return check_mutation_declaration(
            declaration, set(parent), spec.scope, self.max_parameter_changes, self.min_parent_parameters_kept,
            idea_parameters=[p.name for p in idea.parameters] if idea is not None else None,
        )

    def child_problems(self, slot_index: int, child_policy, declaration: dict, schema=None) -> list:
        """Code-level check: the implemented policy differs from its parent exactly as declared."""
        spec = self.slot(slot_index)
        return check_bounded_mutation(
            self.parent_parameters(spec.reference.structure_id, schema), child_policy, declaration, spec.scope,
            self.max_parameter_changes, self.min_parent_parameters_kept,
        )

    def to_dict(self) -> dict:
        return {
            "candidates": self.n,
            "reference_seeded_bounded_mutations": len(self.fittable_slots),
            "open_explorations": len(self.exploration_slots),
            "fittable_budget": self.fittable_budget,
            "checkpoint_steps": list(self.checkpoint_steps),
            "max_parameter_changes": self.max_parameter_changes,
            "min_parent_parameters_kept": self.min_parent_parameters_kept,
            "seed_bank": {"path": self.seed_bank_path, "sha256": self.seed_bank_sha256},
            "structure_roles": {"path": self.roles_path, "sha256": self.roles_sha256},
            "non_fittable_controls": list(self.controls),
            "allocation": "fixed by configuration (4 reach-push references x 1, simple_damped_push x 2); never ranked",
            **DATA_ROLES,
            "slots": [
                {"slot": s.slot, "lane": s.lane,
                 "reference": s.reference.to_dict() if s.reference else None,
                 "scope": s.scope.to_dict() if s.scope else None}
                for s in self.slots
            ],
        }


def _scope_from(scope_id: str, raw: dict) -> MutationScope:
    def names(value):
        if value in (None, ALL):
            return (ALL,)
        return tuple(str(v) for v in value)

    return MutationScope(
        scope_id=scope_id,
        description=str(raw.get("description", "")),
        may_change=names(raw.get("may_change", ALL)),
        may_remove=names(raw.get("may_remove", ALL)),
        allow_adds=bool(raw.get("allow_adds", True)),
    )


def check_fixed_allocation(slots, roles: dict) -> list:
    """The 4:2 rule, checked against the roles file rather than trusted from the config."""
    problems = []
    seeds = active_parent_ids(roles)
    reach_push = [sid for sid in seeds if roles["structures"][sid].get("family") == REACH_PUSH_FAMILY]
    fittable = [s for s in slots if s.lane == LANE_FITTABLE]
    counts = {}
    for s in fittable:
        counts.setdefault(s.reference.structure_id, []).append(s.scope.scope_id)
    if len(reach_push) != 4:
        problems.append(f"expected four reach-push seeds in the roles file, found {reach_push}")
    for sid in reach_push:
        if len(counts.get(sid, [])) != 1:
            problems.append(f"reach-push reference {sid} must have exactly one slot, has {len(counts.get(sid, []))}")
    damped = counts.get(DAMPED_PUSH_REFERENCE, [])
    if sorted(damped) != sorted(DAMPED_PUSH_SCOPES):
        problems.append(
            f"{DAMPED_PUSH_REFERENCE} must have exactly two slots with scopes {list(DAMPED_PUSH_SCOPES)}, got {damped}"
        )
    extra = sorted(set(counts) - set(reach_push) - {DAMPED_PUSH_REFERENCE})
    if extra:
        problems.append(f"slots name references outside the 4:2 allocation: {extra}")
    return problems


def load_two_lane_plan(block: dict) -> TwoLanePlan:
    """Build and check the plan from the ``two_lane_search`` config block.

    Refused when: the counts are not exactly 9 = 6 + 3; there is more than one generation; a slot
    names a control, a retired structure or anything outside the seed bank; the allocation is not
    the fixed 4:2 split; the damped-push slots do not carry the two predeclared scopes; the fittable
    checkpoints are not {0, 500, 1000}; or the data roles differ from the decided ones. Seed-bank
    scores are never read: the allocation cannot depend on ranking.
    """
    problems = []
    if int(block.get("generations", 0)) != 1:
        problems.append(f"generations must be 1, got {block.get('generations')!r}")
    counts = (int(block.get("candidates", -1)), int(block.get("reference_seeded_bounded_mutations", -1)),
              int(block.get("open_explorations", -1)))
    if counts != (TOTAL_CANDIDATES, FITTABLE_SLOTS, EXPLORATION_SLOTS):
        problems.append(
            f"candidates/reference_seeded_bounded_mutations/open_explorations must be "
            f"{TOTAL_CANDIDATES}/{FITTABLE_SLOTS}/{EXPLORATION_SLOTS}, got {'/'.join(map(str, counts))}"
        )
    for key, expected in DATA_ROLES.items():
        if block.get(key) != expected:
            problems.append(f"{key} must be {expected!r}, got {block.get(key)!r}")
    if str(block.get("open_exploration_fitting", "")).lower() != "disabled":
        problems.append("open_exploration_fitting must be 'disabled'")

    lane = block.get("fittable_lane") or {}
    steps = tuple(int(s) for s in lane.get("checkpoints", ()))
    if steps != FITTABLE_CHECKPOINTS:
        problems.append(f"fittable_lane.checkpoints must be {list(FITTABLE_CHECKPOINTS)}, got {list(steps)}")
    budget = int(lane.get("fitting_budget", 0))
    if steps and budget != max(steps):
        problems.append(f"fittable_lane.fitting_budget must equal the last checkpoint ({max(steps)}), got {budget}")

    roles_path = block.get("structure_roles", "")
    roles = load_structure_roles(roles_path)
    seed_bank_path = block.get("seed_bank") or roles.get("seed_bank")
    with open(_abs(seed_bank_path), "r", encoding="utf-8") as f:
        members = (json.load(f).get("members") or {})
    scopes = {sid: _scope_from(sid, raw) for sid, raw in (lane.get("scopes") or {}).items()}

    slots = []
    for index, raw in enumerate(lane.get("slots") or []):
        sid, scope_id = raw.get("reference"), raw.get("scope")
        entry = roles["structures"].get(sid)
        if entry is None:
            problems.append(f"slot {index}: {sid!r} is not in the structure roles file")
            continue
        if entry["role"] != ROLE_FITTABLE_SEED:
            problems.append(f"slot {index}: {sid} is a {entry['role']}, never a fittable-lane parent")
            continue
        if sid not in members:
            problems.append(f"slot {index}: {sid} is not a member of the fittable seed bank")
            continue
        if members[sid].get("source_path") != entry["source_path"]:
            problems.append(f"slot {index}: {sid} source differs between the seed bank and the roles file")
        if scope_id not in scopes:
            problems.append(f"slot {index}: scope {scope_id!r} is not declared under fittable_lane.scopes")
            continue
        path = _abs(entry["source_path"])
        with open(path, "r", encoding="utf-8") as f:
            source = f.read()
        reference = ReferenceStructure(sid, entry.get("family", ""), entry["source_path"], _sha256_file(path), source)
        slots.append(SlotSpec(slot=index, lane=LANE_FITTABLE, reference=reference, scope=scopes[scope_id]))
    if len(slots) != FITTABLE_SLOTS and not problems:
        problems.append(f"fittable_lane.slots must declare {FITTABLE_SLOTS} slots, got {len(slots)}")
    problems.extend(check_fixed_allocation(slots, roles))
    for i in range(counts[2] if counts[2] > 0 else 0):
        slots.append(SlotSpec(slot=len(slots), lane=LANE_EXPLORATION))
    if problems:
        raise LanePlanError("two-lane plan refused:\n  " + "\n  ".join(problems))
    return TwoLanePlan(
        slots=slots,
        fittable_budget=budget,
        checkpoint_steps=steps,
        max_parameter_changes=int(lane.get("max_parameter_changes", 3)),
        min_parent_parameters_kept=float(lane.get("min_parent_parameters_kept", 0.5)),
        seed_bank_path=_rel(_abs(seed_bank_path)),
        seed_bank_sha256=_sha256_file(_abs(seed_bank_path)),
        roles_path=_rel(_abs(roles_path)),
        roles_sha256=_sha256_file(_abs(roles_path)),
        controls=control_ids(roles),
    )


def slots_prompt_block(plan: TwoLanePlan, schema=None) -> str:
    """The fixed slot table the fittable-lane request shows, with each parent's source."""
    lines = ["## Fixed slots"]
    for spec in plan.fittable_slots:
        ref, scope = spec.reference, spec.scope
        params = sorted(plan.parent_parameters(ref.structure_id, schema))
        lines += [
            "",
            f"### Slot {spec.slot}: parent {ref.structure_id}, scope {scope.scope_id}",
            f"Scope: {scope.description}",
            f"May change: {', '.join(scope.may_change)}; may remove: {', '.join(scope.may_remove)}; "
            f"may add parameters: {'yes' if scope.allow_adds else 'no'}",
            f"Parent parameters (action scale excluded): {', '.join(params)}",
            "```python",
            ref.source.strip(),
            "```",
        ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Bounded-mutation contract
# ---------------------------------------------------------------------------


def scale_parameter_names(policy) -> list:
    """Parameters that move the scale but not the executed action, found by measurement."""
    import torch

    from lares.fitting.objectives import find_scale_parameters

    generator = torch.Generator().manual_seed(0)
    probe = torch.rand(64, 39, generator=generator) * 0.6 - 0.3
    return find_scale_parameters(policy, probe)


def parameter_table(policy, exclude=()) -> dict:
    ranges = policy.get_param_ranges()
    return {
        name: (tuple(round(float(v), 12) for v in p.detach().reshape(-1).tolist()),
               tuple(float(x) for x in ranges.get(name, (None, None))))
        for name, p in policy.named_parameters()
        if name not in set(exclude)
    }


def _declared_sets(declaration: dict) -> tuple:
    return (set(declaration.get("adds") or []), set(declaration.get("removes") or []),
            set(declaration.get("changes") or []))


def check_mutation_declaration(declaration: dict, parent_names: set, scope: MutationScope,
                               max_changes: int, min_kept: float, idea_parameters=None) -> list:
    """The declared change on its own: scope, size, and that the idea lists what it implies."""
    problems = []
    adds, removes, changes = _declared_sets(declaration)
    if not str(declaration.get("structural_change", "")).strip():
        problems.append("no structural_change stated")
    if adds & parent_names:
        problems.append(f"adds {sorted(adds & parent_names)} already exist in the parent; list them under changes")
    for kind, names in (("removes", removes), ("changes", changes)):
        unknown = names - parent_names
        if unknown:
            problems.append(f"{kind} {sorted(unknown)} are not parameters of the parent")
        outside = sorted(n for n in names & parent_names if not scope.permits(kind, n))
        if outside:
            problems.append(f"{kind} {outside} fall outside scope {scope.scope_id!r}")
    if adds and not scope.allow_adds:
        problems.append(f"scope {scope.scope_id!r} does not allow new parameters")
    if removes & changes:
        problems.append(f"{sorted(removes & changes)} are both removed and changed")
    total = len(adds) + len(removes) + len(changes)
    if total > int(max_changes):
        problems.append(f"{total} parameter changes, above the bound of {max_changes}; that is not one bounded change")
    kept = len(parent_names - removes - changes)
    if parent_names and kept / len(parent_names) < float(min_kept):
        problems.append(
            f"keeps {kept} of {len(parent_names)} parent parameters unchanged, below {min_kept:.0%}: a wholesale "
            f"rewrite, which only the exploration lane may propose"
        )
    if idea_parameters is not None:
        expected = (parent_names - removes) | adds
        if set(idea_parameters) != expected:
            problems.append(
                f"the idea declares {sorted(idea_parameters)}, but the parent minus removes plus adds is {sorted(expected)}"
            )
    return problems


def check_bounded_mutation(parent_table: dict, child_policy, declaration: dict, scope: MutationScope,
                           max_changes: int, min_kept: float) -> list:
    """The implemented child against its parent: every difference declared, nothing else moved.

    ``parent_table`` excludes the parent's measured scale parameters. The child may declare no
    scale parameter at all, since a trainable action scale is forbidden.
    """
    problems = check_mutation_declaration(declaration, set(parent_table), scope, max_changes, min_kept)
    child_scale = scale_parameter_names(child_policy)
    if child_scale:
        problems.append(f"the child declares trainable action-scale parameters {child_scale}")
    child = parameter_table(child_policy, exclude=child_scale)
    adds, removes, changes = _declared_sets(declaration)
    actual_adds = set(child) - set(parent_table)
    actual_removes = set(parent_table) - set(child)
    actual_changes = {n for n in set(parent_table) & set(child) if parent_table[n] != child[n]}
    for kind, declared, actual in (("adds", adds, actual_adds), ("removes", removes, actual_removes),
                                   ("changes", changes, actual_changes)):
        if declared != actual:
            problems.append(f"{kind}: the code does {sorted(actual)}, the mutation declared {sorted(declared)}")
    return problems


# ---------------------------------------------------------------------------
#  Incumbents
# ---------------------------------------------------------------------------


def fittable_eligibility(lane: str, selected_step: int) -> tuple:
    """``(eligible, reason)`` for the fittable incumbent, before any promotion evidence."""
    if lane != LANE_FITTABLE:
        return False, NOT_FITTABLE_LANE
    if int(selected_step) <= 0:
        return False, FITTING_REJECTED
    return True, "eligible"


def decide_fittable_incumbent(candidate_id: str, lane: str, selected_step: int, fitted_result, step0_result,
                              incumbent_result, incumbent_id, excluded_manifest_ids=()) -> dict:
    """Whether a fittable-lane candidate replaces the fittable incumbent.

    Two paired tests on the promotion manifest, both under the strict rule of
    :func:`lares.eval.promotion.compare_to_incumbent` (difference strictly positive, interval lower
    bound above zero, ties never promote): the fitted state against the same structure's step-0
    state, and the fitted state against the current fittable incumbent (the 0.00-success null
    incumbent before any promotion). Evidence from a screening or probe manifest is refused.
    """
    from lares.eval.promotion import compare_to_incumbent

    eligible, reason = fittable_eligibility(lane, selected_step)
    out = {"candidate_id": candidate_id, "lane": lane, "selected_step": int(selected_step),
           "eligible": eligible, "promoted": False, "reason": reason,
           "fitting_gain": None, "incumbent_comparison": None}
    if not eligible:
        return out
    if step0_result is None or fitted_result is None:
        raise ValueError(f"{candidate_id}: the fittable incumbent needs fitted and step-0 results on the promotion manifest")
    if step0_result.manifest_id != fitted_result.manifest_id:
        raise ValueError(f"{candidate_id}: fitted and step-0 results come from different manifests")
    gain = compare_to_incumbent(fitted_result, step0_result, candidate_id, f"{candidate_id}@step0",
                                excluded_manifest_ids=excluded_manifest_ids)
    versus = compare_to_incumbent(fitted_result, incumbent_result, candidate_id, incumbent_id,
                                  excluded_manifest_ids=excluded_manifest_ids)
    out["fitting_gain"] = gain.to_dict()
    out["incumbent_comparison"] = versus.to_dict()
    if not gain.promoted:
        out["reason"] = f"{FITTABLE_GAIN_REJECTED} ({gain.reason})"
    elif not versus.promoted:
        out["reason"] = f"{FITTABLE_INCUMBENT_NOT_BEATEN} ({versus.reason})"
    else:
        out["promoted"], out["reason"] = True, FITTABLE_PROMOTED
    return out


# ---------------------------------------------------------------------------
#  Resources by lane
# ---------------------------------------------------------------------------


def record_lane(record) -> str:
    return (record.lineage or {}).get("lane") or ""


def lane_resource_totals(records, ledger=None) -> dict:
    """Per-lane totals from saved records alone, with LLM usage on unique call ids.

    Every non-LLM counter and wall-time stage is summed over the lane's records. LLM usage is
    the ledger total over the unique calls the lane's records reference, so a lane's shared
    design call is counted once. A call referenced by records of both lanes would make the
    split meaningless, and is listed rather than divided.
    """
    from lares.search.resources import COUNTER_KEYS, WALL_TIME_STAGES

    llm_keys = ("llm_calls", "llm_prompt_tokens", "llm_completion_tokens", "llm_total_tokens")
    out = {}
    call_lanes: dict = {}
    for lane in LANES:
        mine = [r for r in records if record_lane(r) == lane]
        counters = {k: sum(int(r.resources.get(k, 0) or 0) for r in mine) for k in COUNTER_KEYS if k not in llm_keys}
        wall = {s: sum(float((r.resources.get("wall_time_seconds") or {}).get(s, 0.0)) for r in mine) for s in WALL_TIME_STAGES}
        calls = sorted({c for r in mine for c in (r.llm_call_ids or [])})
        for c in calls:
            call_lanes.setdefault(c, set()).add(lane)
        llm = ledger.totals(calls) if ledger is not None else {}
        out[lane] = {
            "records": len(mine),
            "slots": len({r.candidate_id for r in mine}),
            **counters,
            **{k: int(llm.get(k, 0)) for k in llm_keys},
            "llm_call_ids": calls,
            "simulator_steps": sum(v for k, v in counters.items() if k.endswith("_environment_steps")),
            "wall_time_seconds": wall,
        }
    out["calls_crossing_lanes"] = sorted(c for c, lanes in call_lanes.items() if len(lanes) > 1)
    out["unlaned_records"] = sorted(r.experiment_id for r in records if record_lane(r) not in LANES)
    return out


# ---------------------------------------------------------------------------
#  Expected budget
# ---------------------------------------------------------------------------


def expected_budget(plan: TwoLanePlan, screen_episodes: int, promotion_episodes: int, probe_episodes: int,
                    horizon: int, max_repair_per_candidate: int = 1, expected_correction_rate: float = 0.13,
                    design_retry_rate: float = 0.0) -> dict:
    """What one live run should cost, from the plan and the evaluation settings alone.

    Simulator steps are upper bounds (MetaWorld runs every episode to the horizon). The fittable
    lane screens three checkpoints (zero-shot, halfway, selected), probes each scheduled
    checkpoint and fits for the budget; the exploration lane screens once. Promotion runs the
    performance challenger once and, for the fittable challenger, its fitted and step-0 states;
    the upper bound assumes the two challengers differ.
    """
    nf, ne = len(plan.fittable_slots), len(plan.exploration_slots)
    fittable_screen = nf * 3 * screen_episodes
    exploration_screen = ne * screen_episodes
    probe = nf * len(plan.checkpoint_steps) * probe_episodes
    promotion_max = 3 * promotion_episodes
    design_calls = 2
    implementation_calls = nf + ne
    expected_corrections = expected_correction_rate * (nf + ne)
    return {
        "fittable_lane": {
            "candidates": nf,
            "screen_episodes": fittable_screen,
            "probe_episodes": probe,
            "bc_gradient_steps": nf * plan.fittable_budget,
            "simulator_steps_max": (fittable_screen + probe) * horizon,
        },
        "exploration_lane": {
            "candidates": ne,
            "screen_episodes": exploration_screen,
            "probe_episodes": 0,
            "bc_gradient_steps": 0,
            "simulator_steps_max": exploration_screen * horizon,
        },
        "promotion": {
            "episodes_min": 2 * promotion_episodes,
            "episodes_max": promotion_max,
            "note": "performance challenger fitted state; fittable challenger fitted and step-0 states (shared when "
                    "the two challengers are the same candidate); no fittable challenger if every fittable slot "
                    "rejected fitting",
        },
        "total": {
            "episodes_max": fittable_screen + exploration_screen + probe + promotion_max,
            "simulator_steps_max": (fittable_screen + exploration_screen + probe + promotion_max) * horizon,
            "bc_gradient_steps": nf * plan.fittable_budget,
        },
        "llm": {
            "design_calls": design_calls,
            "implementation_calls": implementation_calls,
            "expected_correction_calls": round(expected_corrections, 2),
            "expected_calls": round(design_calls * (1 + design_retry_rate) + implementation_calls + expected_corrections, 2),
            "max_calls": 2 * design_calls + implementation_calls + int(max_repair_per_candidate) * (nf + ne),
        },
    }


__all__ = [
    "ALL", "DAMPED_PUSH_REFERENCE", "DAMPED_PUSH_SCOPES", "DATA_ROLES", "EXPLORATION_SLOTS", "FITTABLE_CHECKPOINTS",
    "FITTABLE_INCUMBENT", "FITTABLE_SLOTS", "LanePlanError", "MutationScope", "PERFORMANCE_INCUMBENT",
    "ReferenceStructure", "SlotSpec", "TOTAL_CANDIDATES", "TwoLanePlan", "active_parent_ids",
    "check_bounded_mutation", "check_fixed_allocation", "check_mutation_declaration", "control_ids",
    "decide_fittable_incumbent", "expected_budget", "fitting_metrics", "fittable_eligibility",
    "lane_resource_totals", "load_structure_roles", "load_two_lane_plan", "slots_prompt_block",
]
