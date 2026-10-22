"""The two-lane search, mocked end to end (decided 2026-09-14).

Nothing here calls a model or a real simulator. The LLM is scripted, the subprocess harness is
stubbed, and the world is the kinematic push world from ``tests.test_two_generation_mock``. The
orchestrator, generation, in-process validation, the bounded-mutation contract, fitting, checkpoint
probes, screening, both promotion tests, records, the ledger and reconstruction are all real.

The fittable lane's six children are the real seed-bank references with the action-scale parameter
removed and one declared change each. One of them (a weakened reach gain) needs fitting and gets a
fitted checkpoint; the others keep step 0, which the records must call fitting_rejected.

Run from the project root::

    python -m unittest tests.test_two_lane_search
"""

import copy
import json
import os
import re
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))

import yaml  # noqa: E402

from lares.core import policy_generation as pg  # noqa: E402
from lares.core.obs_schema import get_obs_schema  # noqa: E402
from lares.core.training_pipeline import EvolutionOrchestrator  # noqa: E402
from lares.eval import PipelineEnvs, PromotionPolicy  # noqa: E402
from lares.eval.manifest import synthetic_manifest  # noqa: E402
from lares.eval.promotion import NULL_INCUMBENT_ID, PROMOTED_INTERVAL_ABOVE_ZERO  # noqa: E402
from lares.fitting.optimizers import state_parameters_sha256  # noqa: E402
from lares.fitting.structure_bank import STRUCTURES_DIR, instantiate_source, load_bank  # noqa: E402
from lares.search import STATUS_INVALID, LLMCallLedger, SchemaError, load_records  # noqa: E402
from lares.search import lanes  # noqa: E402
from lares.search.generation import (  # noqa: E402
    StructuredOutputError,
    parse_fittable_mutations_payload,
    parse_proposals_payload,
    sha256_file,
)
from lares.search.provenance import (  # noqa: E402
    load_tensor_artifact,
    reconcile_resources,
    reconstruct_record,
    traceability_summary,
)
from lares.search.schemas import (  # noqa: E402
    EXPLORATION_LABEL,
    FITTABLE_LABEL,
    FITTING_REJECTED,
    LANE_EXPLORATION,
    LANE_FITTABLE,
    LINEAGE_COMPLETE,
    MODE_FITTABLE_MUTATION,
)
from test_evaluation_report import make_record, make_result  # noqa: E402
from test_two_generation_mock import PARAMS_C, POLICY_C, PushWorld, expert_buffer  # noqa: E402
from test_two_phase_policy_generation import (  # noqa: E402
    PARAMS_A,
    PARAMS_B,
    POLICY_A,
    POLICY_B,
    ScriptedLLM,
    fence,
    idea_dict,
    proposal_dict,
)

SCHEMA = get_obs_schema("push-v2")
TWO_LANE_CONFIG = os.path.join(_PROJECT_ROOT, "config", "two_lane_search.yaml")
BASELINE_CONFIG = os.path.join(_PROJECT_ROOT, "config", "run_full_evolution.yaml")
ROLES = os.path.join(_PROJECT_ROOT, "config", "fittable_bank", "structure_roles.yaml")
SEED_BANK = os.path.join(_PROJECT_ROOT, "config", "fittable_bank", "seed_bank.json")
GRIP = "self.grip_bias = nn.Parameter(torch.tensor(-0.5))"


def config_block():
    with open(TWO_LANE_CONFIG, encoding="utf-8") as f:
        return copy.deepcopy(yaml.safe_load(f)["two_lane_search"])


def edit(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def strip_scale(source):
    """A reference as the fittable lane must implement it: no log_std, a fixed std."""
    out = re.sub(r"\n\s*self\.log_std = nn\.Parameter\(torch\.tensor\([^)]*\)\)", "", source)
    out = re.sub(r'\n\s*"log_std": \([^)]*\),', "", out)
    out = edit(out, "std = torch.exp(self.log_std) * torch.ones_like(mean)", "std = self.fixed_std(mean)")
    assert "log_std" not in out
    return out


def children(plan):
    """``{slot: (source, declared parameter changes, structural change)}`` for the six fittable slots."""
    src = {s.slot: strip_scale(s.reference.source) for s in plan.fittable_slots}
    geometry = edit(src[5], GRIP, GRIP + "\n        self.approach_height = nn.Parameter(torch.tensor(0.0))")
    geometry = edit(geometry, "to_obj = obj - tcp",
                    "lift = torch.cat([torch.zeros_like(obj[:, 0:2]), self.approach_height * torch.ones_like(obj[:, 2:3])], dim=1)\n"
                    "        to_obj = obj + lift - tcp")
    geometry = edit(geometry, '"grip_bias": (-2.0, 2.0),', '"grip_bias": (-2.0, 2.0),\n            "approach_height": (-0.05, 0.05),')
    return {
        0: (edit(src[0], "self.w_reach = nn.Parameter(torch.tensor(3.0))", "self.w_reach = nn.Parameter(torch.tensor(0.2))"),
            {"changes": ["w_reach"]}, "start the reach gain low so fitting has to set it"),
        1: (edit(src[1], GRIP, GRIP.replace("-0.5", "-0.4")), {"changes": ["grip_bias"]}, "loosen the gripper bias"),
        2: (edit(src[2], GRIP, GRIP.replace("-0.5", "-0.3")), {"changes": ["grip_bias"]}, "loosen the gripper bias further"),
        3: (edit(src[3], GRIP, GRIP.replace("-0.5", "-0.45")), {"changes": ["grip_bias"]}, "nudge the gripper bias"),
        4: (edit(src[4], "self.brake_radius = nn.Parameter(torch.tensor(0.05))", "self.brake_radius = nn.Parameter(torch.tensor(0.04))"),
            {"changes": ["brake_radius"]}, "brake closer to the goal"),
        5: (geometry, {"adds": ["approach_height"]}, "aim the reach at a height offset above the puck"),
    }


def undeclared_slot3(plan):
    """Slot 3's child with an extra, undeclared change to the gate sharpness."""
    source = children(plan)[3][0]
    return edit(source, "self.sharpness = nn.Parameter(torch.tensor(120.0))", "self.sharpness = nn.Parameter(torch.tensor(150.0))")


def idea_for(name, source):
    policy = instantiate_source(source, 39, 4, schema=SCHEMA)
    ranges = policy.get_param_ranges()
    params = [(n, float(p.detach()), float(ranges[n][0]), float(ranges[n][1])) for n, p in policy.named_parameters()]
    return idea_dict(name, params)


def mutation_entry(spec, child, metric="fitting.validation_loss_end", **overrides):
    source, changes, change = child
    entry = {
        "slot": spec.slot,
        "parent_id": spec.reference.structure_id,
        "mutation_scope": spec.scope.scope_id,
        "structural_change": change,
        "parameter_changes": {"adds": [], "removes": [], "changes": [], **changes},
        "proposal": {
            "observation": "fitting improves this reference, but its declared constants sit far from the fitted ones",
            "hypothesis": "one constant limits how quickly the fitted checkpoint appears",
            "suspected_component": "one gain or gate constant",
            "intervention": change,
            "predicted_metric": metric,
            "predicted_direction": "decrease",
            "protected_behaviors": ["reaches the puck", "pushes it toward the goal"],
        },
        "idea": idea_for(f"mutation of slot {spec.slot}", source),
    }
    entry.update(overrides)
    return entry


def llm_script(plan):
    kids = children(plan)
    mutations = [mutation_entry(spec, kids[spec.slot]) for spec in plan.fittable_slots]
    ideas = {"ideas": [idea_dict("A", PARAMS_A), idea_dict("B", PARAMS_B), idea_dict("C", PARAMS_C)]}
    implement = [fence(kids[slot][0]) for slot in range(6)]
    implement[3] = fence(undeclared_slot3(plan))
    implement += [fence(POLICY_A), fence(POLICY_B), fence(POLICY_C)]
    return {
        "policy_fittable_mutations": [json.dumps({"mutations": mutations})],
        "policy_ideas": [json.dumps(ideas)],
        "policy_implement": implement,
        "policy_correction": [fence(kids[3][0])],
    }


def two_lane_args():
    return SimpleNamespace(model="stub-model", reasoning_effort="high", policy_gen_two_phase=True,
                           policy_impl_mode="per_idea")


# ---------------------------------------------------------------------------
#  The plan
# ---------------------------------------------------------------------------


class TestPlan(unittest.TestCase):
    def refused(self, block, text):
        with self.assertRaises(lanes.LanePlanError) as ctx:
            lanes.load_two_lane_plan(block)
        self.assertIn(text, str(ctx.exception))

    def test_the_configured_plan_is_nine_slots_in_the_fixed_4_2_split(self):
        plan = lanes.load_two_lane_plan(config_block())
        self.assertEqual(plan.n, 9)
        self.assertEqual([s.lane for s in plan.slots], [LANE_FITTABLE] * 6 + [LANE_EXPLORATION] * 3)
        references = [s.reference.structure_id for s in plan.fittable_slots]
        reach = [r for r in references if r != "simple_damped_push"]
        self.assertEqual(sorted(reach), ["simple_reach_push", "simple_reach_push_damped",
                                         "simple_reach_push_shared_gain", "simple_reach_push_sharp_gate"])
        self.assertEqual(len(set(reach)), 4)
        damped = [s.scope.scope_id for s in plan.fittable_slots if s.reference.structure_id == "simple_damped_push"]
        self.assertEqual(sorted(damped), ["gate_control", "geometry"])
        self.assertEqual(plan.checkpoint_steps, (0, 500, 1000))
        self.assertEqual(plan.fittable_budget, 1000)
        self.assertTrue(all(s.reference is None and s.scope is None for s in plan.exploration_slots))

    def test_any_other_count_is_refused(self):
        for key, value in (("candidates", 10), ("reference_seeded_bounded_mutations", 5),
                           ("open_explorations", 4), ("generations", 2)):
            block = config_block()
            block[key] = value
            with self.assertRaises(lanes.LanePlanError):
                lanes.load_two_lane_plan(block)
        block = config_block()
        block["fittable_lane"]["slots"].append({"reference": "simple_reach_push", "scope": "single_bounded_change"})
        with self.assertRaises(lanes.LanePlanError):
            lanes.load_two_lane_plan(block)

    def test_the_allocation_never_depends_on_ranking_the_references(self):
        baseline = lanes.load_two_lane_plan(config_block())
        with open(SEED_BANK, encoding="utf-8") as f:
            bank = json.load(f)
        members = dict(reversed(list(bank["members"].items())))
        for i, member in enumerate(members.values()):
            member["mean"] = -1.0 * i
            member["zero_shot_success"] = 1.0 - 0.2 * i
            member["selected_success_by_seed"] = [0.0] * 5
        bank["members"] = members
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "seed_bank.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(bank, f)
            block = config_block()
            block["seed_bank"] = path
            scrambled = lanes.load_two_lane_plan(block)
        self.assertEqual([(s.slot, s.lane, s.reference and s.reference.structure_id, s.scope and s.scope.scope_id)
                          for s in scrambled.slots],
                         [(s.slot, s.lane, s.reference and s.reference.structure_id, s.scope and s.scope.scope_id)
                          for s in baseline.slots])

    def test_an_unbalanced_allocation_is_refused(self):
        block = config_block()
        block["fittable_lane"]["slots"][0] = {"reference": "simple_damped_push", "scope": "single_bounded_change"}
        self.refused(block, "exactly one slot")
        block = config_block()
        block["fittable_lane"]["slots"][5]["scope"] = "gate_control"
        self.refused(block, "gate_control")
        block = config_block()
        block["fittable_lane"]["slots"][1] = {"reference": "simple_reach_push", "scope": "single_bounded_change"}
        self.refused(block, "simple_reach_push_damped must have exactly one slot")

    def test_controls_and_retired_structures_are_never_parents(self):
        for sid, role in (("simple_standoff", "non_fittable_control"),
                          ("simple_damped_push_smooth_gate", "non_fittable_control"),
                          ("simple_standoff_phase_gains", "retired"),
                          ("simple_standoff_moderate_geometry", "retired")):
            block = config_block()
            block["fittable_lane"]["slots"][0] = {"reference": sid, "scope": "single_bounded_change"}
            self.refused(block, f"{sid} is a {role}")
        block = config_block()
        block["fittable_lane"]["slots"][0] = {"reference": "gen_0_cand0", "scope": "single_bounded_change"}
        self.refused(block, "not in the structure roles file")

    def test_checkpoints_budget_and_data_roles_are_fixed(self):
        block = config_block()
        block["fittable_lane"]["checkpoints"] = [0, 1000]
        self.refused(block, "checkpoints")
        block = config_block()
        block["fittable_lane"]["fitting_budget"] = 2000
        self.refused(block, "fitting_budget")
        block = config_block()
        block["checkpoint_selection_data"] = "screen"
        self.refused(block, "checkpoint_selection_data")
        block = config_block()
        block["promotion_data"] = "screen"
        self.refused(block, "promotion_data")
        block = config_block()
        block["open_exploration_fitting"] = "enabled"
        self.refused(block, "open_exploration_fitting")


# ---------------------------------------------------------------------------
#  Structure roles: standoff and smooth-gate out of parent selection, nothing deleted
# ---------------------------------------------------------------------------


class TestStructureRoles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.roles = lanes.load_structure_roles(ROLES)
        with open(SEED_BANK, encoding="utf-8") as f:
            cls.bank = json.load(f)

    def test_the_standoff_family_leaves_parent_selection_with_one_control(self):
        structures = self.roles["structures"]
        standoff = [sid for sid, e in structures.items() if e["family"] == "standoff"]
        self.assertEqual(sorted(standoff), ["simple_standoff", "simple_standoff_moderate_geometry",
                                            "simple_standoff_phase_gains"])
        active = lanes.active_parent_ids(self.roles)
        self.assertFalse(set(standoff) & set(active))
        controls = lanes.control_ids(self.roles)
        self.assertEqual([sid for sid in controls if structures[sid]["family"] == "standoff"], ["simple_standoff"])
        self.assertEqual(sorted(controls), ["simple_damped_push_smooth_gate", "simple_standoff"])
        self.assertNotIn("simple_damped_push_smooth_gate", active)
        self.assertEqual({sid for sid, e in structures.items() if e["role"] == "retired"},
                         {"simple_standoff_moderate_geometry", "simple_standoff_phase_gains"})

    def test_active_parents_are_exactly_the_qualified_seed_bank(self):
        self.assertEqual(set(lanes.active_parent_ids(self.roles)), set(self.bank["members"]))
        plan = lanes.load_two_lane_plan(config_block())
        self.assertEqual(set(plan.reference_ids()), set(self.bank["members"]))
        self.assertEqual(sorted(plan.controls), ["simple_damped_push_smooth_gate", "simple_standoff"])

    def test_no_historical_structure_or_result_was_deleted(self):
        for entry in self.roles["structures"].values():
            self.assertTrue(os.path.isfile(os.path.join(_PROJECT_ROOT, entry["source_path"])))
        self.assertEqual(set(self.bank["excluded"]), {"simple_damped_push_smooth_gate", "simple_standoff",
                                                      "simple_standoff_moderate_geometry", "simple_standoff_phase_gains"})
        self.assertEqual(len(self.bank["members"]), 5)
        self.assertEqual(len(load_bank(STRUCTURES_DIR)), 18)
        with open(os.path.join(_PROJECT_ROOT, "config", "fittable_bank", "qualification.yaml"), encoding="utf-8") as f:
            v1 = yaml.safe_load(f)
        self.assertIn("simple_standoff", v1["structures"]["references"])


# ---------------------------------------------------------------------------
#  Bounded-mutation contract and the fittable-lane parser
# ---------------------------------------------------------------------------


class TestMutationContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = lanes.load_two_lane_plan(config_block())
        cls.kids = children(cls.plan)
        cls.slots = [s.as_parse_slot() for s in cls.plan.fittable_slots]

    def declaration(self, slot, **changes):
        return {"structural_change": "one change", "adds": [], "removes": [], "changes": [], **changes}

    def build(self, source):
        return instantiate_source(source, 39, 4, schema=SCHEMA)

    def test_every_declared_child_passes(self):
        for slot, (source, changes, change) in self.kids.items():
            declaration = self.declaration(slot, structural_change=change, **changes)
            self.assertEqual(self.plan.child_problems(slot, self.build(source), declaration, SCHEMA), [], slot)

    def test_an_undeclared_change_is_refused(self):
        problems = self.plan.child_problems(3, self.build(undeclared_slot3(self.plan)),
                                            self.declaration(3, changes=["grip_bias"]), SCHEMA)
        self.assertTrue(any("sharpness" in p for p in problems), problems)

    def test_a_change_outside_the_slot_scope_is_refused(self):
        problems = self.plan.declaration_problems(self.slots[5], self.declaration(5, changes=["sharpness"]), None, SCHEMA)
        self.assertTrue(any("outside scope 'geometry'" in p for p in problems), problems)
        problems = self.plan.declaration_problems(self.slots[4], self.declaration(4, changes=["grip_bias"]), None, SCHEMA)
        self.assertTrue(any("outside scope 'gate_control'" in p for p in problems), problems)

    def test_a_wholesale_rewrite_is_refused(self):
        declaration = self.declaration(0, changes=["w_reach", "w_push", "sharpness"])
        problems = self.plan.declaration_problems(self.slots[0], declaration, None, SCHEMA)
        self.assertTrue(any("wholesale rewrite" in p for p in problems), problems)

    def test_more_than_the_bounded_number_of_changes_is_refused(self):
        declaration = self.declaration(4, changes=["sharpness", "contact_radius", "brake_radius", "w_reach"])
        problems = self.plan.declaration_problems(self.slots[4], declaration, None, SCHEMA)
        self.assertTrue(any("above the bound of 3" in p for p in problems), problems)

    def test_a_trainable_action_scale_is_refused(self):
        spec = self.plan.slot(3)
        with_scale = edit(spec.reference.source, GRIP, GRIP.replace("-0.5", "-0.45"))
        problems = self.plan.child_problems(3, self.build(with_scale), self.declaration(3, changes=["grip_bias"]), SCHEMA)
        self.assertTrue(any("action-scale" in p and "log_std" in p for p in problems), problems)

    def test_the_idea_must_declare_every_parameter_the_mutation_implies(self):
        source = self.kids[5][0]
        idea = SimpleNamespace(parameters=[SimpleNamespace(name=n) for n, _ in self.build(source).named_parameters()
                                           if n != "approach_height"])
        problems = self.plan.declaration_problems(self.slots[5], self.declaration(5, adds=["approach_height"]), idea, SCHEMA)
        self.assertTrue(any("the idea declares" in p for p in problems), problems)

    def payload(self, entries):
        return json.dumps({"mutations": entries})

    def entries(self):
        return [mutation_entry(spec, self.kids[spec.slot]) for spec in self.plan.fittable_slots]

    def parse(self, entries):
        return parse_fittable_mutations_payload(
            self.payload(entries), self.slots, 0, schema=SCHEMA,
            declaration_check=lambda slot, d, idea: self.plan.declaration_problems(slot, d, idea, SCHEMA),
        )

    def test_the_parser_assigns_ids_mode_and_parent(self):
        parses = self.parse(self.entries())
        for spec, sp in zip(self.plan.fittable_slots, parses):
            self.assertTrue(sp.ok, sp.problems)
            self.assertEqual(sp.proposal.proposal_id, f"gen0_prop{spec.slot}")
            self.assertEqual(sp.proposal.mode, MODE_FITTABLE_MUTATION)
            self.assertEqual(sp.proposal.parent_ids, [spec.reference.structure_id])
            self.assertEqual(sp.idea.idea_id, f"gen0_idea{spec.slot}")

    def test_the_model_cannot_choose_another_parent_or_scope(self):
        entries = self.entries()
        entries[0]["parent_id"] = "simple_damped_push"
        entries[4]["mutation_scope"] = "geometry"
        parses = self.parse(entries)
        self.assertTrue(any("not this slot's reference" in p for p in parses[0].problems))
        self.assertTrue(any("not this slot's scope" in p for p in parses[4].problems))
        self.assertTrue(all(sp.ok for i, sp in enumerate(parses) if i not in (0, 4)))

    def test_the_prediction_must_name_a_fitting_diagnostic(self):
        entries = self.entries()
        entries[2] = mutation_entry(self.plan.slot(2), self.kids[2], metric="signed_goal_progress")
        parses = self.parse(entries)
        self.assertFalse(parses[2].ok)
        self.assertTrue(any("fitting diagnostic" in p for p in parses[2].problems), parses[2].problems)

    def test_a_wrong_number_of_mutations_raises(self):
        with self.assertRaises(StructuredOutputError):
            self.parse(self.entries()[:5])

    def test_ordinary_proposals_cannot_claim_the_fittable_mode(self):
        text = json.dumps({"proposals": [{"proposal": proposal_dict(MODE_FITTABLE_MUTATION, ["gen0_cand0"],
                                                                    predicted_metric="fitting.validation_loss_end"),
                                          "idea": idea_dict("A", PARAMS_A)}]})
        parses = parse_proposals_payload(text, 1, 1, ["gen0_cand0"], schema=SCHEMA)
        self.assertTrue(any("only by the fittable lane" in p for p in parses[0].problems), parses[0].problems)


# ---------------------------------------------------------------------------
#  Incumbent rules
# ---------------------------------------------------------------------------


class TestIncumbentRules(unittest.TestCase):
    EXCLUDED = ("screen", "probe")

    def result(self, successes, manifest="promotion", name="x"):
        result = make_result([make_record(case_id=f"c{i}", success=s) for i, s in enumerate(successes)], actor=name)
        result.manifest_id = manifest
        return result

    def decide(self, lane, step, fitted, step0, incumbent=None, incumbent_id=None):
        return lanes.decide_fittable_incumbent("cand", lane, step, fitted, step0, incumbent, incumbent_id, self.EXCLUDED)

    def test_an_exploration_can_never_be_the_fittable_incumbent(self):
        d = self.decide(LANE_EXPLORATION, 1000, self.result([1.0] * 12), self.result([0.0] * 12))
        self.assertEqual((d["eligible"], d["promoted"], d["reason"]), (False, False, lanes.NOT_FITTABLE_LANE))
        self.assertIsNone(d["fitting_gain"])

    def test_selected_step_zero_is_fitting_rejected_and_ineligible(self):
        d = self.decide(LANE_FITTABLE, 0, self.result([1.0] * 12), self.result([0.0] * 12))
        self.assertEqual((d["eligible"], d["promoted"], d["reason"]), (False, False, FITTING_REJECTED))

    def test_a_real_fitted_gain_over_its_own_step_zero_promotes(self):
        d = self.decide(LANE_FITTABLE, 500, self.result([1.0] * 8 + [0.0] * 4), self.result([0.0] * 12))
        self.assertTrue(d["promoted"], d["reason"])
        self.assertEqual(d["reason"], lanes.FITTABLE_PROMOTED)
        gain = d["fitting_gain"]
        self.assertEqual(gain["reason"], PROMOTED_INTERVAL_ABOVE_ZERO)
        self.assertGreater(gain["paired_success"]["ci_low"], 0.0)
        self.assertEqual(gain["evidence_manifest_id"], "promotion")
        self.assertEqual(d["incumbent_comparison"]["incumbent_id"], NULL_INCUMBENT_ID)

    def test_tied_success_never_promotes(self):
        same = [1.0, 0.0] * 6
        d = self.decide(LANE_FITTABLE, 1000, self.result(same), self.result(same))
        self.assertFalse(d["promoted"])
        self.assertIn("tied_paired_success", d["reason"])

    def test_an_interval_that_includes_zero_does_not_promote(self):
        d = self.decide(LANE_FITTABLE, 1000, self.result([1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0]),
                        self.result([0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]))
        self.assertFalse(d["promoted"])
        self.assertIn("interval_includes_zero", d["reason"])

    def test_the_fitted_state_must_also_beat_the_current_fittable_incumbent(self):
        fitted = self.result([1.0] * 10 + [0.0] * 2)
        d = self.decide(LANE_FITTABLE, 1000, fitted, self.result([0.0] * 12), incumbent=self.result([1.0] * 10 + [0.0] * 2),
                        incumbent_id="previous")
        self.assertTrue(d["fitting_gain"]["promoted"])
        self.assertFalse(d["promoted"])
        self.assertIn(lanes.FITTABLE_INCUMBENT_NOT_BEATEN, d["reason"])

    def test_screening_or_probe_evidence_is_refused(self):
        with self.assertRaises(ValueError):
            self.decide(LANE_FITTABLE, 500, self.result([1.0] * 12, "probe"), self.result([0.0] * 12, "probe"))

    def test_fitted_and_step_zero_must_be_scored_on_the_same_manifest(self):
        with self.assertRaises(ValueError):
            self.decide(LANE_FITTABLE, 500, self.result([1.0] * 12, "promotion"), self.result([0.0] * 12, "other"))


# ---------------------------------------------------------------------------
#  Refusals and configuration
# ---------------------------------------------------------------------------


class TestRefusalsAndConfigs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = lanes.load_two_lane_plan(config_block())
        manifest = synthetic_manifest(20, label="push-v2")
        cls.dev, cls.probe = manifest.head(16), manifest.tail(4)

    def orchestrator(self, **overrides):
        kwargs = dict(env_name="push-v2", num_generations=1, pop_size=9, bc_steps=0, dev_manifest=self.dev,
                      probe_manifest=self.probe, lane_plan=self.plan, log_dir=tempfile.gettempdir(),
                      promotion_policy=PromotionPolicy(screen_episodes=4, expanded_episodes=12))
        kwargs.update(overrides)
        return EvolutionOrchestrator(**kwargs)

    def test_the_orchestrator_refuses_a_lane_plan_it_cannot_honour(self):
        for overrides in ({"num_generations": 2}, {"pop_size": 8}, {"bc_steps": 100}, {"promotion_challengers": 2}):
            with self.assertRaises(ValueError, msg=str(overrides)):
                self.orchestrator(**overrides)

    def test_checkpoint_selection_screening_and_promotion_must_be_separate(self):
        self.orchestrator().check_manifest_isolation()
        with self.assertRaises(ValueError):
            self.orchestrator(probe_manifest=self.dev.head(4)).check_manifest_isolation()
        with self.assertRaises(ValueError):
            self.orchestrator(probe_manifest=None).check_manifest_isolation()

    def test_the_two_lane_generator_refuses_batched_implementation(self):
        with self.assertRaises(ValueError):
            pg.generate_two_lane_candidates(MagicMock(), two_lane_args(), tempfile.gettempdir(), 39, 4, "push-v2", {},
                                            self.plan, "unused.pkl", MagicMock(), impl_mode=pg.POLICY_IMPL_BATCHED)

    def test_the_two_lane_config_loads_through_the_run_script(self):
        import run_full_evolution

        cfg = run_full_evolution.load_config(TWO_LANE_CONFIG)
        self.assertEqual((cfg.num_generations, cfg.pop_size, cfg.bc_steps), (1, 9, 0))
        self.assertEqual((cfg.model, cfg.reasoning_effort), ("gpt-6.1-sol", "high"))
        self.assertEqual(cfg.max_llm_calls_per_generation, 22)
        self.assertEqual(cfg.policy_impl_mode, "per_idea")
        self.assertEqual(cfg.two_lane_search["fittable_lane"]["checkpoints"], [0, 500, 1000])
        self.assertEqual(cfg.probe_episodes + cfg.screen_episodes + cfg.expanded_episodes, 50)

    def test_the_run_script_refuses_an_inconsistent_two_lane_config(self):
        import run_full_evolution

        with open(TWO_LANE_CONFIG, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        for key, value in (("pop_size", 8), ("bc_steps", 500), ("num_generations", 2), ("policy_impl_mode", "batched")):
            broken = dict(raw, **{key: value})
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
                yaml.safe_dump(broken, f)
            try:
                with self.assertRaises(ValueError, msg=key):
                    run_full_evolution.load_config(f.name)
            finally:
                os.remove(f.name)

    def test_the_baseline_is_nine_unseeded_zero_shot_candidates_on_the_new_model(self):
        import run_full_evolution

        cfg = run_full_evolution.load_config(BASELINE_CONFIG)
        self.assertIsNone(cfg.two_lane_search)
        self.assertEqual((cfg.num_generations, cfg.pop_size, cfg.bc_steps, cfg.checkpoint_steps), (1, 9, 0, []))
        self.assertEqual((cfg.model, cfg.reasoning_effort), ("gpt-6.1-sol", "high"))
        self.assertEqual(cfg.log_dir, "./logs/baseline_gpt61_sol_high")
        two = run_full_evolution.load_config(TWO_LANE_CONFIG)
        for key in ("screen_episodes", "expanded_episodes", "probe_episodes", "eval_max_steps", "demo_buffer_path",
                    "promotion_challengers", "max_llm_tokens_per_generation", "max_repair_per_candidate"):
            self.assertEqual(getattr(cfg, key), getattr(two, key), key)
        self.assertNotEqual(cfg.log_dir, two.log_dir)

    def test_the_expected_budget_of_one_live_run(self):
        import run_full_evolution

        cfg = run_full_evolution.load_config(TWO_LANE_CONFIG)
        budget = lanes.expected_budget(self.plan, cfg.screen_episodes, cfg.expanded_episodes, cfg.probe_episodes,
                                       cfg.eval_max_steps, cfg.max_repair_per_candidate)
        self.assertEqual(budget["fittable_lane"]["screen_episodes"], 180)
        self.assertEqual(budget["fittable_lane"]["probe_episodes"], 180)
        self.assertEqual(budget["fittable_lane"]["bc_gradient_steps"], 6000)
        self.assertEqual(budget["exploration_lane"], {"candidates": 3, "screen_episodes": 30, "probe_episodes": 0,
                                                      "bc_gradient_steps": 0, "simulator_steps_max": 4500})
        self.assertEqual(budget["promotion"]["episodes_max"], 90)
        self.assertEqual(budget["total"]["episodes_max"], 480)
        self.assertEqual(budget["total"]["simulator_steps_max"], 72000)
        self.assertEqual(budget["llm"]["max_calls"], cfg.max_llm_calls_per_generation)
        self.assertEqual((budget["llm"]["design_calls"], budget["llm"]["implementation_calls"]), (2, 9))


# ---------------------------------------------------------------------------
#  One mocked two-lane run through the real orchestrator
# ---------------------------------------------------------------------------


class TestTwoLaneRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = cls.tmp.name
        manifest = synthetic_manifest(20, label="push-v2")
        # 4-case screen, 12-case promotion set, 4-case probe: three disjoint case sets.
        cls.dev, cls.probe = manifest.head(16), manifest.tail(4)
        cls.plan = lanes.load_two_lane_plan(config_block())
        cls.orchestrator = EvolutionOrchestrator(
            env_name="push-v2", obs_dim=39, action_dim=4, num_generations=1, pop_size=9, elite_num=2, bc_steps=0,
            log_dir=cls.root, record_demo_gif=False, dev_manifest=cls.dev, probe_manifest=cls.probe,
            eval_max_steps=30, bc_batch_size=32, bc_lr=5e-2, max_llm_calls_per_generation=22,
            promotion_policy=PromotionPolicy(screen_episodes=4, expanded_episodes=12), lane_plan=cls.plan,
        )
        envs = PipelineEnvs(development=PushWorld(), rl=PushWorld(), gif=PushWorld(), probe=PushWorld())
        cls.llm = ScriptedLLM(llm_script(cls.plan))
        train = synthetic_manifest(10, label="push-v2-train").episodes
        with patch.object(pg, "_call_llm", side_effect=cls.llm), patch.object(
            pg, "_run_policy_validation_subprocess", return_value=(True, "Success!", "full")
        ):
            cls.best = cls.orchestrator.run(MagicMock(), envs, expert_buffer(train), two_lane_args())
        cls.records = load_records(os.path.join(cls.root, "experiments"))
        cls.by_id = {r.experiment_id: r for r in cls.records}
        latest = {}
        for r in cls.records:
            if r.candidate_id not in latest or r.lineage["attempt"] > latest[r.candidate_id].lineage["attempt"]:
                latest[r.candidate_id] = r
        cls.final = [latest[k] for k in sorted(latest, key=lambda cid: latest[cid].lineage["slot"])]
        cls.ledger = LLMCallLedger(cls.root)

        def load(name):
            with open(os.path.join(cls.root, name), encoding="utf-8") as f:
                return json.load(f)

        cls.run_manifest = load("run_manifest.json")
        cls.incumbents = load("incumbents.json")
        cls.lane_resources = load("lane_resources.json")
        cls.screening = load(os.path.join("gen_0", "screening.json"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def lane(self, name):
        return [r for r in self.final if r.lineage["lane"] == name]

    # -- requirement 1: exactly nine candidates ---------------------------------

    def test_exactly_nine_candidates_six_fittable_and_three_explorations(self):
        self.assertEqual([r.candidate_id for r in self.final], [f"gen0_cand{i}" for i in range(9)])
        self.assertEqual([r.lineage["lane"] for r in self.final], [LANE_FITTABLE] * 6 + [LANE_EXPLORATION] * 3)
        self.assertTrue(all(r.status != STATUS_INVALID for r in self.final))
        self.assertEqual({r.generation for r in self.records}, {0})
        self.assertEqual(self.run_manifest["two_lane_plan"]["candidates"], 9)

    # -- requirement 2: fixed 4:2 allocation --------------------------------------

    def test_the_fittable_slots_follow_the_fixed_allocation(self):
        declared = [(s["reference"], s["scope"]) for s in config_block()["fittable_lane"]["slots"]]
        recorded = [(r.lineage["lane_metadata"]["reference"]["structure_id"],
                     r.lineage["lane_metadata"]["mutation_scope"]["id"]) for r in self.lane(LANE_FITTABLE)]
        self.assertEqual(recorded, declared)
        self.assertEqual(sorted(scope for ref, scope in recorded if ref == "simple_damped_push"), ["gate_control", "geometry"])

    # -- requirement 3: only the fittable lane fits --------------------------------

    def test_only_the_fittable_lane_fits_with_checkpoints_0_500_1000(self):
        for r in self.lane(LANE_FITTABLE):
            self.assertEqual([h["step"] for h in r.fitting["checkpoint_history"]], [0, 500, 1000])
            self.assertEqual(r.resources["bc_gradient_steps"], 1000)
            self.assertEqual(r.resources["probe_episodes"], 3 * len(self.probe))
            self.assertEqual(r.fitting["probe_manifest_id"], self.probe.manifest_id)
        for r in self.lane(LANE_EXPLORATION):
            for key in ("bc_gradient_steps", "bc_objective_evaluations", "probe_episodes", "probe_environment_steps"):
                self.assertEqual(r.resources[key], 0, key)
            self.assertEqual(r.resources["wall_time_seconds"]["fitting"], 0.0)
            self.assertEqual(r.fitting["checkpoint_history"], [])
            self.assertIsNone(r.fitting["optimizer"])
            self.assertIsNone(r.fitting["probe_manifest_id"])
            self.assertEqual(set(r.evaluation["reports"]) - {"fitted_expanded"}, {"zero_shot"})
            self.assertEqual(r.resources["screen_episodes"], len(self.orchestrator.screen_manifest))

    def test_step_zero_is_selectable_and_recorded_as_fitting_rejected(self):
        steps = {r.candidate_id: r.fitting["selected_step"] for r in self.lane(LANE_FITTABLE)}
        self.assertIn(0, steps.values())
        self.assertTrue(any(s > 0 for s in steps.values()), steps)
        for r in self.lane(LANE_FITTABLE):
            rejected = r.fitting["selected_step"] == 0
            self.assertIs(r.fitting["fitting_rejected"], rejected)
            initial = load_tensor_artifact(self.root, r.fitting["initial_state"])
            selected = load_tensor_artifact(self.root, r.fitting["selected_state"])
            self.assertEqual(state_parameters_sha256(initial) == state_parameters_sha256(selected), rejected)
            if rejected:
                self.assertEqual(r.fitting["selection_outcome"], FITTING_REJECTED)

    # -- requirement 4: separate incumbents ---------------------------------------

    def test_the_performance_challenger_is_ranked_across_both_lanes(self):
        decisions = self.screening["decisions"]
        self.assertEqual({d["candidate_id"] for d in decisions}, {r.candidate_id for r in self.final})
        cleared = [d for d in decisions if d["screen_cleared"]]
        best = max(cleared, key=lambda d: (d["success_rate"], d["progress"], d["mean_return"]))
        tested = [d for d in decisions if d["promotion_tested"]]
        self.assertEqual([d["candidate_id"] for d in tested], [best["candidate_id"]])
        comparison = tested[0]["incumbent_comparison"]
        self.assertEqual(comparison["evidence_manifest_id"], self.orchestrator.promotion_manifest.manifest_id)
        self.assertEqual(set(comparison["excluded_manifest_ids"]),
                         {self.orchestrator.screen_manifest.manifest_id, self.probe.manifest_id})
        if comparison["promoted"]:
            self.assertGreater(comparison["paired_success"]["ci_low"], 0.0)
            self.assertEqual(self.incumbents["performance_incumbent"]["candidate_id"], best["candidate_id"])
            self.assertEqual(self.best["candidate_id"], best["candidate_id"])

    def test_the_fittable_incumbent_needs_a_fitted_gain_on_independent_evidence(self):
        fittable = self.incumbents["fittable_incumbent"]
        self.assertIsNotNone(fittable, self.incumbents["fittable_decisions"])
        record = next(r for r in self.final if r.candidate_id == fittable["candidate_id"])
        self.assertEqual(record.lineage["lane"], LANE_FITTABLE)
        self.assertGreater(record.fitting["selected_step"], 0)
        decision = record.evaluation["fittable_incumbent"]
        self.assertTrue(decision["promoted"])
        promotion = self.orchestrator.promotion_manifest.manifest_id
        for key in ("fitting_gain", "incumbent_comparison"):
            test = decision[key]
            self.assertTrue(test["promoted"], key)
            self.assertGreater(test["paired_success"]["mean_difference"], 0.0)
            self.assertGreater(test["paired_success"]["ci_low"], 0.0)
            self.assertEqual(test["evidence_manifest_id"], promotion)
            self.assertEqual(set(test["excluded_manifest_ids"]),
                             {self.orchestrator.screen_manifest.manifest_id, self.probe.manifest_id})
        self.assertEqual(decision["fitting_gain"]["incumbent_id"], f"{record.candidate_id}@step0")
        self.assertIn("zero_shot_expanded", record.evaluation["reports"])
        self.assertLess(record.evaluation["promotion_step0_success_rate"], record.evaluation["expanded_success_rate"])
        self.assertTrue(record.evaluation["incumbents"]["fittable_incumbent"])
        self.assertEqual(self.orchestrator.fittable_incumbent["candidate_id"], record.candidate_id)

    def test_ineligible_candidates_are_recorded_with_their_reason(self):
        for r in self.final:
            decision = r.evaluation["fittable_incumbent"]
            if r.lineage["lane"] == LANE_EXPLORATION:
                self.assertEqual((decision["eligible"], decision["reason"]), (False, lanes.NOT_FITTABLE_LANE))
            elif r.fitting["selected_step"] == 0:
                self.assertEqual((decision["eligible"], decision["reason"]), (False, FITTING_REJECTED))
            self.assertIn("performance_incumbent", r.evaluation["incumbents"])

    def test_the_incumbents_are_kept_apart(self):
        self.assertEqual(set(self.incumbents) >= {"performance_incumbent", "fittable_incumbent", "rules"}, True)
        performance = self.incumbents["performance_incumbent"]
        if performance is not None:
            self.assertIn(performance["lane"], (LANE_FITTABLE, LANE_EXPLORATION))
        self.assertIn("selected_step > 0", self.incumbents["rules"]["fittable_incumbent"])

    # -- requirement 5: shared manifests, separate data ----------------------------

    def test_both_lanes_share_screen_and_promotion_manifests_and_probe_data_is_separate(self):
        screen = self.orchestrator.screen_manifest.manifest_id
        promotion = self.orchestrator.promotion_manifest.manifest_id
        for r in self.final:
            self.assertEqual(r.evaluation["manifest_ids"][0] if len(r.evaluation["manifest_ids"]) == 1 else screen, screen)
            self.assertTrue(set(r.evaluation["manifest_ids"]) <= {screen, promotion})
            self.assertIn(screen, r.evaluation["manifest_ids"])
        cases = {name: set(m["case_ids"]) for name, m in self.run_manifest["manifests"].items()}
        self.assertFalse(cases["probe"] & cases["screen"])
        self.assertFalse(cases["probe"] & cases["promotion"])
        self.assertFalse(cases["screen"] & cases["promotion"])
        self.assertTrue(self.run_manifest["probe_is_isolated"])
        self.assertEqual(self.run_manifest["two_lane_plan"]["checkpoint_selection_data"], "probe")

    # -- requirement 6: provenance and resources by lane ----------------------------

    def test_fittable_records_carry_parent_proposal_and_scope(self):
        for r in self.lane(LANE_FITTABLE):
            meta = r.lineage["lane_metadata"]
            reference = meta["reference"]["structure_id"]
            self.assertEqual(meta["label"], FITTABLE_LABEL)
            self.assertEqual(r.parent_ids, [reference])
            self.assertEqual(r.lineage["proposal"]["mode"], MODE_FITTABLE_MUTATION)
            self.assertTrue(r.lineage["proposal"]["predicted_metric"].startswith("fitting."))
            self.assertEqual(r.lineage["context_parent_ids"], self.plan.reference_ids())
            self.assertEqual(meta["reference"]["source_sha256"],
                             sha256_file(os.path.join(_PROJECT_ROOT, meta["reference"]["source_path"])))
            self.assertEqual(meta["reference"]["seed_bank_sha256"], sha256_file(SEED_BANK))
            self.assertTrue(meta["mutation"]["structural_change"])
            self.assertEqual(r.lineage["status"], LINEAGE_COMPLETE)
            self.assertEqual(r.lineage["problems"], [])

    def test_explorations_are_labelled_unseeded(self):
        for r in self.lane(LANE_EXPLORATION):
            self.assertEqual(r.lineage["lane_metadata"]["label"], EXPLORATION_LABEL)
            self.assertIs(r.lineage["lane_metadata"]["seeded"], False)
            self.assertEqual((r.parent_ids, r.proposal_id, r.lineage["proposal"]), ([], "", None))
            self.assertEqual(r.lineage["context_parent_ids"], [])
            self.assertEqual(r.lineage["status"], LINEAGE_COMPLETE)

    def test_an_undeclared_change_fails_the_mutation_contract_and_is_corrected(self):
        failed, fixed = self.by_id["gen0_cand3_a0"], self.by_id["gen0_cand3_a1"]
        self.assertEqual(failed.status, STATUS_INVALID)
        self.assertEqual(failed.validation["stage"], pg.STAGE_MUTATION_CONTRACT)
        self.assertTrue(any("sharpness" in e for e in failed.validation["errors"]))
        self.assertEqual(failed.lineage["lane"], LANE_FITTABLE)
        self.assertEqual(fixed.lineage["previous_attempt_id"], "gen0_cand3_a0")
        self.assertNotEqual(fixed.status, STATUS_INVALID)

    def test_resources_are_tracked_separately_by_lane(self):
        totals = lanes.lane_resource_totals(self.records, self.ledger)
        self.assertEqual(json.loads(json.dumps(totals)), self.lane_resources)
        self.assertEqual(totals["calls_crossing_lanes"], [])
        self.assertEqual(totals["unlaned_records"], [])
        fittable, exploration = totals[LANE_FITTABLE], totals[LANE_EXPLORATION]
        self.assertEqual((fittable["slots"], exploration["slots"]), (6, 3))
        self.assertEqual(fittable["bc_gradient_steps"], 6000)
        self.assertEqual(fittable["probe_episodes"], 6 * 3 * len(self.probe))
        for key in ("bc_gradient_steps", "bc_objective_evaluations", "probe_episodes", "probe_environment_steps"):
            self.assertEqual(exploration[key], 0, key)
        self.assertEqual(exploration["wall_time_seconds"]["fitting"], 0.0)
        self.assertEqual(fittable["llm_calls"] + exploration["llm_calls"], len(self.ledger))
        labels = {c.call_id: c.purpose for c in self.ledger.records()}
        self.assertIn("mutations", {labels[c] for c in fittable["llm_call_ids"]})
        self.assertIn("ideas", {labels[c] for c in exploration["llm_call_ids"]})
        self.assertNotIn("mutations", {labels[c] for c in exploration["llm_call_ids"]})

    def test_every_record_validates_reconstructs_and_reconciles(self):
        for r in self.records:
            self.assertTrue(r.validate())
        for r in self.final:
            self.assertTrue(reconstruct_record(r, self.root)["identical"], r.experiment_id)
        summary = traceability_summary(self.records, self.root)
        for name, field in summary.items():
            self.assertTrue(field["complete"], (name, field["gaps"]))
        reconciliation = reconcile_resources(self.records, self.ledger, self.root)
        self.assertTrue(reconciliation["llm"]["reconciles"], reconciliation["llm"])
        self.assertTrue(reconciliation["episodes"]["reconciles"], reconciliation["episodes"])

    def test_missing_lane_provenance_is_refused(self):
        fittable = next(r for r in self.lane(LANE_FITTABLE) if r.fitting["selected_step"] == 0)
        exploration = self.lane(LANE_EXPLORATION)[0]
        mutations = (
            (fittable, lambda r: setattr(r, "parent_ids", [])),
            (fittable, lambda r: r.lineage["lane_metadata"].pop("mutation_scope")),
            (fittable, lambda r: r.lineage["lane_metadata"]["reference"].pop("source_sha256")),
            (fittable, lambda r: r.lineage["proposal"].update(predicted_metric="signed_goal_progress")),
            (fittable, lambda r: r.fitting.update(fitting_rejected=False)),
            (exploration, lambda r: r.lineage["lane_metadata"].update(label="")),
            (exploration, lambda r: setattr(r, "proposal_id", "gen0_prop6")),
            (exploration, lambda r: r.resources.update(probe_episodes=3)),
            (exploration, lambda r: r.resources.update(bc_gradient_steps=10)),
        )
        for original, mutate in mutations:
            record = copy.deepcopy(original)
            mutate(record)
            with self.assertRaises(SchemaError):
                record.validate()

    def test_the_summary_script_reports_resources_by_lane(self):
        import summarise_experiments

        out = os.path.join(self.root, "summary.json")
        with patch.object(sys, "argv", ["summarise_experiments.py", "--log-dir", self.root, "--out", out,
                                        "--no-reconstruct"]):
            self.assertEqual(summarise_experiments.main(), 0)
        with open(out, encoding="utf-8") as f:
            summary = json.load(f)
        self.assertEqual(summary["slots"], 9)
        self.assertEqual((summary["lanes"]["fittable"]["slots"], summary["lanes"]["exploration"]["slots"]), (6, 3))
        self.assertEqual(summary["lanes"]["exploration"]["bc_gradient_steps"], 0)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
#  R-018: the single-phase, rollout-searched exploration lane, through the real orchestrator
# ---------------------------------------------------------------------------


def v2_block():
    block = config_block()
    block["exploration_lane"] = {"generator": "single_phase", "treatment": "rollout_search", "search_candidates": 24}
    return block


class TestSearchedExplorationRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lares.search.schemas import EXPLORATION_SEARCHED_LABEL, IDEA_ORIGIN_CODE, ZERO_SHOT_NOT_ASSERTED

        cls.labels = (EXPLORATION_SEARCHED_LABEL, IDEA_ORIGIN_CODE, ZERO_SHOT_NOT_ASSERTED)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = cls.tmp.name
        manifest = synthetic_manifest(20, label="push-v2")
        cls.dev, cls.probe = manifest.head(16), manifest.tail(4)
        cls.plan = lanes.load_two_lane_plan(v2_block())
        cls.orchestrator = EvolutionOrchestrator(
            env_name="push-v2", obs_dim=39, action_dim=4, num_generations=1, pop_size=9, elite_num=2, bc_steps=0,
            log_dir=cls.root, record_demo_gif=False, dev_manifest=cls.dev, probe_manifest=cls.probe,
            eval_max_steps=30, bc_batch_size=32, bc_lr=5e-2, max_llm_calls_per_generation=22,
            promotion_policy=PromotionPolicy(screen_episodes=4, expanded_episodes=12), lane_plan=cls.plan, fit_seed=3101,
        )
        envs = PipelineEnvs(development=PushWorld(), rl=PushWorld(), gif=PushWorld(), probe=PushWorld())
        script = llm_script(cls.plan)
        script.pop("policy_ideas")
        script["policy_implement"] = script["policy_implement"][:6]
        script["policy_generation"] = [fence(POLICY_A), fence(POLICY_B), fence(POLICY_C)]
        cls.llm = ScriptedLLM(script)
        train = synthetic_manifest(10, label="push-v2-train").episodes
        with patch.object(pg, "_call_llm", side_effect=cls.llm), patch.object(
            pg, "_run_policy_validation_subprocess", return_value=(True, "Success!", "full")
        ):
            cls.best = cls.orchestrator.run(MagicMock(), envs, expert_buffer(train), two_lane_args())
        cls.records = load_records(os.path.join(cls.root, "experiments"))
        cls.explorations = [r for r in cls.records if r.lineage["lane"] == LANE_EXPLORATION]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_lane_makes_no_design_call_and_one_code_call_per_slot(self):
        self.assertNotIn("policy_ideas", self.llm.labels)
        self.assertEqual(self.llm.labels.count("policy_generation"), 3)

    def test_explorations_carry_code_derived_ideas_and_the_searched_label(self):
        label, origin, not_asserted = self.labels
        self.assertEqual(len(self.explorations), 3)
        for r in self.explorations:
            self.assertEqual(r.lineage["lane_metadata"]["label"], label)
            self.assertEqual(r.lineage["idea"]["origin"], origin)
            self.assertEqual(r.lineage["idea"]["zero_shot_behavior"], not_asserted)
            self.assertEqual(r.lineage["status"], "complete")

    def test_explorations_are_searched_on_the_probe_and_never_fitted(self):
        for r in self.explorations:
            search = r.fitting["search"]
            self.assertEqual(len(search["trace"]), 24)
            self.assertEqual(r.resources["probe_episodes"], 24 * len(self.probe))
            self.assertEqual(r.resources["bc_gradient_steps"], 0)
            self.assertEqual(r.resources["wall_time_seconds"]["fitting"], 0.0)
            self.assertEqual(r.fitting["optimizer"], "rollout_search")
            self.assertEqual(r.fitting["checkpoint_history"], [])
            self.assertIn("fitted_screen", r.evaluation["reports"])
            selected = load_tensor_artifact(self.root, r.fitting["selected_state"])
            initial = load_tensor_artifact(self.root, r.fitting["initial_state"])
            self.assertEqual(state_parameters_sha256(selected) == state_parameters_sha256(initial),
                             search["selected_index"] == 0)

    def test_the_fittable_lane_is_unchanged(self):
        for r in (r for r in self.records if r.lineage["lane"] == LANE_FITTABLE and r.status != STATUS_INVALID):
            self.assertEqual([h["step"] for h in r.fitting["checkpoint_history"]], [0, 500, 1000])

    def test_a_searched_record_without_its_trace_is_refused(self):
        record = copy.deepcopy(self.explorations[0])
        record.fitting["search"] = {}
        with self.assertRaises(SchemaError):
            record.validate()

    def test_the_budget_counts_the_search(self):
        budget = lanes.expected_budget(self.plan, 4, 12, 4, 30)
        self.assertEqual(budget["exploration_lane"]["probe_episodes"], 3 * 24 * 4)
        self.assertEqual(budget["llm"]["design_calls"], 1)
