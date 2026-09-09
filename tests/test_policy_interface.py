#!/usr/bin/env python
"""Tests for the trusted policy interface and validator (``spec.md`` FR-2 / AC-1).

AC-1 requires unit tests that intentionally trigger every validator failure class,
so each class here is a policy written to be wrong in exactly one way.

Run from the project root::

    python -m unittest tests.test_policy_interface
"""

import os
import sys
import unittest

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from lares.core.obs_schema import (  # noqa: E402
    ObsField,
    ObsSchema,
    active_schema,
    get_active_schema,
    get_obs_schema,
    set_active_schema,
)
from lares.core.policy_validator import (  # noqa: E402
    BATCH_COUPLING,
    FORBIDDEN_MODULE,
    GATE_TELEMETRY,
    GateAggregator,
    INIT_OUT_OF_RANGE,
    INSENSITIVE_INPUT,
    INVALID_RANGE,
    MISSING_SCHEMA,
    NON_FINITE,
    NON_POSITIVE_STD,
    RANGES_NOT_A_DICT,
    RAW_OBS_INDEXING,
    UNDECLARED_PARAMETER,
    UNKNOWN_RANGE_ENTRY,
    WRONG_SHAPE,
    check_source,
    validate_policy,
)
from lares.core.symbolic_policy import SymbolicPolicy  # noqa: E402

try:
    import metaworld  # noqa: F401

    HAS_METAWORLD = True
except ImportError:
    HAS_METAWORLD = False

OBS_DIM = 39
ACT_DIM = 4
SCHEMA = get_obs_schema("push-v2")


# ---------------------------------------------------------------------------
#  Policies: one correct, then one defect each
# ---------------------------------------------------------------------------


class GoodPolicy(SymbolicPolicy):
    """Two-phase reach-then-push controller. The reference for every other case."""

    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.w_reach = nn.Parameter(torch.tensor(3.0))
        self.w_push = nn.Parameter(torch.tensor(2.0))
        self.sharp = nn.Parameter(torch.tensor(40.0))
        self.thresh = nn.Parameter(torch.tensor(0.06))
        self.grip = nn.Parameter(torch.tensor(-0.5))
        self.log_std = nn.Parameter(torch.tensor(-1.5))

    def _terms(self, obs):
        tcp = self.obs_field(obs, "tcp")
        obj = self.obs_field(obs, "obj")
        goal = self.obs_field(obs, "goal")
        to_obj = obj - tcp
        d_obj = torch.norm(to_obj, dim=-1, keepdim=True) + 1e-8
        to_goal = goal - obj
        d_goal = torch.norm(to_goal, dim=-1, keepdim=True) + 1e-8
        contact = torch.sigmoid(self.sharp * (self.thresh - d_obj))
        return to_obj / d_obj, to_goal / d_goal, contact

    def forward(self, obs):
        dir_obj, dir_goal, contact = self._terms(obs)
        self.record_gate("contact", contact)
        move = (1 - contact) * self.w_reach * dir_obj + contact * self.w_push * dir_goal
        grip = self.grip.unsqueeze(0).expand(obs.shape[0], 1)
        mean = torch.cat([move, grip], dim=1)
        return mean, torch.exp(self.log_std) * torch.ones_like(mean)

    def get_param_ranges(self):
        return {
            "w_reach": (0.1, 10.0),
            "w_push": (0.1, 10.0),
            "sharp": (1.0, 200.0),
            "thresh": (0.01, 0.3),
            "grip": (-2.0, 2.0),
            "log_std": (-5.0, 0.0),
        }


class UndeclaredParamPolicy(GoodPolicy):
    def get_param_ranges(self):
        ranges = super().get_param_ranges()
        del ranges["log_std"]
        return ranges


class UnknownRangePolicy(GoodPolicy):
    def get_param_ranges(self):
        return {**super().get_param_ranges(), "phantom": (0.0, 1.0)}


class InvertedRangePolicy(GoodPolicy):
    def get_param_ranges(self):
        return {**super().get_param_ranges(), "sharp": (200.0, 1.0)}


class InitOutOfRangePolicy(GoodPolicy):
    def get_param_ranges(self):
        return {**super().get_param_ranges(), "thresh": (0.1, 0.3)}


class RangesNotADictPolicy(GoodPolicy):
    def get_param_ranges(self):
        return [("w_reach", (0.1, 10.0))]


class WrongShapePolicy(GoodPolicy):
    def forward(self, obs):
        mean, std = super().forward(obs)
        return mean[:, :3], std[:, :3]


class NonFinitePolicy(GoodPolicy):
    def forward(self, obs):
        tcp = self.obs_field(obs, "tcp")
        # Dividing by a zero norm, the classic missing-epsilon bug.
        zero = tcp - tcp
        move = zero / torch.norm(zero, dim=-1, keepdim=True)
        grip = self.grip.unsqueeze(0).expand(obs.shape[0], 1)
        mean = torch.cat([move, grip], dim=1)
        return mean, torch.exp(self.log_std) * torch.ones_like(mean)


class NonPositiveStdPolicy(GoodPolicy):
    def forward(self, obs):
        mean, std = super().forward(obs)
        return mean, torch.zeros_like(std)


class BatchCoupledPolicy(GoodPolicy):
    def forward(self, obs):
        tcp = self.obs_field(obs, "tcp")
        obj = self.obs_field(obs, "obj")
        goal = self.obs_field(obs, "goal")
        # Batch-wide mean: row i now depends on every other row.
        to_obj = obj - tcp.mean(dim=0, keepdim=True)
        d_obj = torch.norm(to_obj, dim=-1, keepdim=True) + 1e-8
        move = self.w_reach * (to_obj / d_obj) + 0.0 * goal
        grip = self.grip.unsqueeze(0).expand(obs.shape[0], 1)
        mean = torch.cat([move, grip], dim=1)
        return mean, torch.exp(self.log_std) * torch.ones_like(mean)


class IgnoresGoalPolicy(GoodPolicy):
    def forward(self, obs):
        dir_obj, _, contact = self._terms(obs)
        move = self.w_reach * dir_obj + contact * self.w_push * dir_obj
        grip = self.grip.unsqueeze(0).expand(obs.shape[0], 1)
        mean = torch.cat([move, grip], dim=1)
        return mean, torch.exp(self.log_std) * torch.ones_like(mean)


class AttachedGatePolicy(GoodPolicy):
    def forward(self, obs):
        dir_obj, dir_goal, contact = self._terms(obs)
        # Bypasses record_gate, so the tensor is still attached to the graph.
        self.last_gates = {"contact": contact}
        move = (1 - contact) * self.w_reach * dir_obj + contact * self.w_push * dir_goal
        grip = self.grip.unsqueeze(0).expand(obs.shape[0], 1)
        mean = torch.cat([move, grip], dim=1)
        return mean, torch.exp(self.log_std) * torch.ones_like(mean)


class ForbiddenModulePolicy(GoodPolicy):
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.fc = nn.Linear(obs_dim, action_dim)


def build(cls):
    with active_schema(SCHEMA):
        return cls(OBS_DIM, ACT_DIM)


def categories_for(cls, **kwargs):
    policy = build(cls)
    return validate_policy(policy, OBS_DIM, ACT_DIM, schema=SCHEMA, **kwargs).categories


# ---------------------------------------------------------------------------
#  Schema
# ---------------------------------------------------------------------------


class TestObsSchema(unittest.TestCase):
    def test_named_slices_match_the_declared_ranges(self):
        obs = torch.arange(OBS_DIM, dtype=torch.float32).unsqueeze(0)
        self.assertTrue(torch.equal(SCHEMA.slice(obs, "tcp"), obs[:, 0:3]))
        self.assertTrue(torch.equal(SCHEMA.slice(obs, "obj"), obs[:, 4:7]))
        self.assertTrue(torch.equal(SCHEMA.slice(obs, "goal"), obs[:, 36:39]))

    def test_unbatched_observation_is_treated_as_one_row(self):
        obs = torch.arange(OBS_DIM, dtype=torch.float32)
        self.assertEqual(tuple(SCHEMA.slice(obs, "tcp").shape), (1, 3))

    def test_unknown_field_names_what_is_available(self):
        with self.assertRaises(KeyError) as ctx:
            SCHEMA.slice(torch.zeros(1, OBS_DIM), "target")
        self.assertIn("goal", str(ctx.exception))

    def test_wrong_observation_width_is_refused(self):
        with self.assertRaises(ValueError):
            SCHEMA.slice(torch.zeros(1, 12), "tcp")

    def test_unknown_task_refuses_rather_than_returning_a_blank_layout(self):
        from lares.core.obs_schema import get_obs_schema as lookup

        with self.assertRaises(KeyError):
            lookup("sweep-v2")

    def test_overlapping_or_out_of_bounds_fields_are_refused(self):
        with self.assertRaises(ValueError):
            ObsSchema("x", 10, (ObsField("a", 0, 20, "too wide"),))
        with self.assertRaises(ValueError):
            ObsSchema("x", 10, (ObsField("a", 0, 3, "ok"), ObsField("a", 3, 6, "dup")))

    def test_required_fields_are_the_ones_the_task_needs(self):
        self.assertEqual(set(SCHEMA.required_names), {"tcp", "obj", "goal"})

    def test_unverifiable_fields_are_named_not_silently_passed(self):
        self.assertIn("prev_obj", SCHEMA.unverifiable_names())

    def test_active_schema_is_restored_after_the_context(self):
        before = get_active_schema()
        with active_schema(SCHEMA):
            self.assertIs(get_active_schema(), SCHEMA)
        self.assertIs(get_active_schema(), before)


# ---------------------------------------------------------------------------
#  Validator failure classes
# ---------------------------------------------------------------------------


class TestValidatorFailureClasses(unittest.TestCase):
    def test_a_correct_policy_passes_every_check(self):
        policy = build(GoodPolicy)
        report = validate_policy(policy, OBS_DIM, ACT_DIM, schema=SCHEMA)
        self.assertTrue(report.ok, report.summary())
        for name in (
            "source",
            "parameters",
            "outputs",
            "batch_independence",
            "field_sensitivity",
            "gate_telemetry",
            "serialization",
        ):
            self.assertIn(name, report.checked + ["source"])

    def test_raw_observation_indexing_is_rejected(self):
        source = (
            "class GeneratedPolicy(SymbolicPolicy):\n"
            "    def forward(self, obs):\n"
            "        tcp = obs[:, 0:3]\n"
            "        return tcp, tcp\n"
        )
        self.assertIn(RAW_OBS_INDEXING, check_source(source).categories)

    def test_slicing_helpers_count_as_raw_indexing(self):
        source = (
            "class GeneratedPolicy(SymbolicPolicy):\n"
            "    def forward(self, obs):\n"
            "        a = obs.narrow(1, 0, 3)\n"
            "        b = torch.split(obs, 3)\n"
            "        return a, b\n"
        )
        self.assertIn(RAW_OBS_INDEXING, check_source(source).categories)

    def test_batch_size_lookup_is_not_raw_indexing(self):
        source = (
            "class GeneratedPolicy(SymbolicPolicy):\n"
            "    def forward(self, obs):\n"
            "        n = obs.shape[0]\n"
            "        tcp = self.obs_field(obs, 'tcp')\n"
            "        return tcp, tcp\n"
        )
        self.assertTrue(check_source(source).ok)

    def test_unparseable_source_is_reported_not_raised(self):
        self.assertFalse(check_source("class GeneratedPolicy(:\n").ok)

    def test_missing_schema_is_rejected(self):
        set_active_schema(None)
        try:
            policy = GoodPolicy(OBS_DIM, ACT_DIM)
        finally:
            set_active_schema(None)
        report = validate_policy(policy, OBS_DIM, ACT_DIM, require_schema=True)
        self.assertIn(MISSING_SCHEMA, report.categories)

    def test_schema_of_the_wrong_width_is_rejected(self):
        policy = build(GoodPolicy)
        narrow = ObsSchema("x", 12, (ObsField("tcp", 0, 3, "tcp", True),))
        report = validate_policy(policy, OBS_DIM, ACT_DIM, schema=narrow)
        self.assertIn(MISSING_SCHEMA, report.categories)

    def test_forbidden_module_is_rejected(self):
        self.assertIn(FORBIDDEN_MODULE, categories_for(ForbiddenModulePolicy))

    def test_undeclared_parameter_is_rejected(self):
        self.assertIn(UNDECLARED_PARAMETER, categories_for(UndeclaredParamPolicy))

    def test_range_entry_naming_no_parameter_is_rejected(self):
        self.assertIn(UNKNOWN_RANGE_ENTRY, categories_for(UnknownRangePolicy))

    def test_inverted_range_is_rejected(self):
        self.assertIn(INVALID_RANGE, categories_for(InvertedRangePolicy))

    def test_initial_value_outside_its_range_is_rejected(self):
        self.assertIn(INIT_OUT_OF_RANGE, categories_for(InitOutOfRangePolicy))

    def test_non_dict_ranges_is_rejected(self):
        self.assertIn(RANGES_NOT_A_DICT, categories_for(RangesNotADictPolicy))

    def test_wrong_output_shape_is_rejected(self):
        self.assertIn(WRONG_SHAPE, categories_for(WrongShapePolicy))

    def test_non_finite_output_is_rejected(self):
        self.assertIn(NON_FINITE, categories_for(NonFinitePolicy))

    def test_non_positive_std_is_rejected(self):
        self.assertIn(NON_POSITIVE_STD, categories_for(NonPositiveStdPolicy))

    def test_batch_coupling_is_rejected(self):
        self.assertIn(BATCH_COUPLING, categories_for(BatchCoupledPolicy))

    def test_ignoring_a_required_field_is_rejected(self):
        self.assertIn(INSENSITIVE_INPUT, categories_for(IgnoresGoalPolicy))

    def test_attached_gate_telemetry_is_rejected(self):
        self.assertIn(GATE_TELEMETRY, categories_for(AttachedGatePolicy))

    def test_report_collects_every_defect_not_just_the_first(self):
        report = validate_policy(build(UndeclaredParamPolicy), OBS_DIM, ACT_DIM, schema=SCHEMA)
        self.assertGreaterEqual(len(report.errors), 1)
        self.assertFalse(report.ok)
        self.assertIn("undeclared_parameter", report.summary())

    def test_report_serialises_for_the_experiment_record(self):
        d = validate_policy(build(GoodPolicy), OBS_DIM, ACT_DIM, schema=SCHEMA).to_dict()
        self.assertIn("ok", d)
        self.assertIn("checked", d)


# ---------------------------------------------------------------------------
#  Gate telemetry
# ---------------------------------------------------------------------------


class TestGateTelemetry(unittest.TestCase):
    def setUp(self):
        self.policy = build(GoodPolicy)

    def test_recorded_gates_are_detached_and_batch_shaped(self):
        self.policy.reset_diagnostics()
        obs = 0.05 * torch.randn(6, OBS_DIM)
        self.policy(obs)
        gates = self.policy.last_gates
        self.assertIn("contact", gates)
        self.assertFalse(gates["contact"].requires_grad)
        self.assertEqual(gates["contact"].shape[0], 6)

    def test_recording_does_not_change_the_action(self):
        obs = 0.05 * torch.randn(4, OBS_DIM)
        self.policy.reset_diagnostics()
        mean_a, std_a = self.policy(obs)
        self.policy.reset_diagnostics()
        mean_b, std_b = self.policy(obs)
        self.assertTrue(torch.equal(mean_a, mean_b))
        self.assertTrue(torch.equal(std_a, std_b))

    def test_reset_diagnostics_clears_without_touching_parameters(self):
        obs = 0.05 * torch.randn(4, OBS_DIM)
        self.policy(obs)
        before = {k: v.detach().clone() for k, v in self.policy.named_parameters()}
        self.policy.reset_diagnostics()
        self.assertEqual(self.policy.last_gates, {})
        for name, value in self.policy.named_parameters():
            self.assertTrue(torch.equal(before[name], value.detach()))

    def test_gradient_still_flows_after_recording(self):
        obs = 0.05 * torch.randn(4, OBS_DIM)
        self.policy.zero_grad()
        mean, std = self.policy(obs)
        (mean.sum() + std.sum()).backward()
        self.assertTrue(
            any(p.grad is not None and p.grad.abs().sum() > 0 for p in self.policy.parameters())
        )

    def test_aggregator_reports_every_required_statistic(self):
        agg = GateAggregator()
        for value in (0.1, 0.2, 0.8, 0.9, 0.2, 0.7):
            agg.observe({"contact": torch.tensor([[value]])})
        stats = agg.statistics()["contact"]
        self.assertAlmostEqual(stats["min"], 0.1, places=5)
        self.assertAlmostEqual(stats["max"], 0.9, places=5)
        self.assertAlmostEqual(stats["occupancy_above_half"], 3 / 6, places=5)
        self.assertEqual(stats["first_crossing_step"], 2)
        self.assertEqual(stats["transitions"], 3)
        self.assertEqual(stats["num_steps"], 6)

    def test_aggregator_reports_a_gate_that_never_fires(self):
        agg = GateAggregator()
        for _ in range(5):
            agg.observe({"push": torch.tensor([[0.02]])})
        stats = agg.statistics()["push"]
        self.assertIsNone(stats["first_crossing_step"])
        self.assertEqual(stats["occupancy_above_half"], 0.0)
        self.assertEqual(stats["transitions"], 0)

    def test_aggregator_ignores_a_policy_that_exposes_nothing(self):
        agg = GateAggregator()
        agg.observe(None)
        agg.observe({})
        self.assertEqual(agg.statistics(), {})


# ---------------------------------------------------------------------------
#  Schema against the simulator
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAS_METAWORLD, "metaworld is required to verify the layout")
class TestSchemaAgainstSimulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("MUJOCO_GL", "egl")
        from lares.eval import EvaluationManifest, make_manifest_env

        path = os.path.join(_PROJECT_ROOT, "config", "manifests", "push-v2_development.yaml")
        if not os.path.isfile(path):
            raise unittest.SkipTest("development manifest missing; run scripts/lock_baseline.py")
        cls.manifest = EvaluationManifest.load(path)
        cls.env = make_manifest_env(cls.manifest, cls.manifest.resolve_pool(), 200)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_declared_layout_matches_what_the_simulator_reports(self):
        for case in self.manifest.episodes[:3]:
            self.assertEqual(SCHEMA.check_against_env(self.env, case), [])

    def test_a_wrong_slice_is_caught(self):
        wrong = ObsSchema(
            "push-v2",
            OBS_DIM,
            (
                ObsField("tcp", 0, 3, "tcp", True),
                ObsField("obj", 7, 10, "quaternion, not position", True),
                ObsField("goal", 36, 39, "goal", True),
            ),
        )
        problems = wrong.check_against_env(self.env, self.manifest.episodes[0])
        self.assertTrue(any("obj" in p for p in problems), problems)


if __name__ == "__main__":
    unittest.main()
