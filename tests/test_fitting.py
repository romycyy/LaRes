#!/usr/bin/env python
"""Tests for the structure bank, fitting methods and sensitivity (FR-6, AC-3).

Run from the project root::

    python -m unittest tests.test_fitting
"""

import os
import sys
import unittest

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lares.core.obs_schema import get_obs_schema  # noqa: E402
from lares.core.policy_validator import validate_policy  # noqa: E402
from lares.core.training_pipeline import DemoBuffer  # noqa: E402
from lares.fitting import (  # noqa: E402
    ADAM_BASELINE,
    ADAM_SCALED,
    CHECKPOINT_BEST_VALIDATION,
    CHECKPOINT_FINAL,
    DIFFERENTIAL_EVOLUTION,
    METHODS,
    MULTISTART_ADAM,
    action_sensitivity,
    bc_objective,
    check_port_equivalence,
    dead_parameters,
    fit,
    gradient_sensitivity,
    instantiate_source,
    load_bank,
    perturb_within_ranges,
    port_raw_indices,
    prepare_data,
    summarise_sensitivity,
)
from lares.fitting.benchmark import (  # noqa: E402
    LOSS_BASED_CHECKPOINT_RULES,
    SELECTION_RULE,
    BenchmarkConfig,
    aggregate,
    format_table,
    loss_versus_success,
    run_benchmark,
    select_default,
)

OBS_DIM = 39
ACT_DIM = 4
SCHEMA = get_obs_schema("push-v2")

RAW_SOURCE = '''
class GeneratedPolicy(SymbolicPolicy):
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.w = nn.Parameter(torch.tensor(2.0))
        self.wg = nn.Parameter(torch.tensor(1.0))
        self.log_std = nn.Parameter(torch.tensor(-1.0))

    def forward(self, obs):
        tcp = obs[:, 0:3]
        obj = obs[:, 4:7]
        goal = obs[:, 36:39]
        move = self.w * (obj - tcp) + self.wg * (goal - obj)
        grip = torch.zeros(obs.shape[0], 1)
        mean = torch.cat([move, grip], dim=1)
        return mean, torch.exp(self.log_std) * torch.ones_like(mean)

    def get_param_ranges(self):
        return {"w": (0.0, 10.0), "wg": (0.0, 10.0), "log_std": (-5.0, 0.0)}
'''


def make_buffer(episodes=12, steps=25, seed=0):
    rng = np.random.default_rng(seed)
    buf = DemoBuffer()
    for ep in range(episodes):
        for _ in range(steps):
            obs = (0.1 * rng.standard_normal(OBS_DIM)).astype(np.float32)
            action = np.clip(rng.standard_normal(ACT_DIM), -1, 1)
            buf.add(obs, action, 0.0, obs, 0.0, episode_id=f"ep{ep}")
    return buf


# ---------------------------------------------------------------------------
#  Structure bank
# ---------------------------------------------------------------------------


class TestPorting(unittest.TestCase):
    def test_known_slices_become_named_accessors(self):
        ported, unported = port_raw_indices(RAW_SOURCE, SCHEMA)
        self.assertEqual(unported, [])
        self.assertNotIn("obs[:,", ported)
        for name in ("tcp", "obj", "goal"):
            self.assertIn(f'self.obs_field(obs, "{name}")', ported)

    def test_a_slice_matching_no_field_is_reported_not_guessed(self):
        source = RAW_SOURCE.replace("obs[:, 4:7]", "obs[:, 5:9]")
        ported, unported = port_raw_indices(source, SCHEMA)
        self.assertTrue(unported)
        self.assertIn("obs[:, 5:9]", ported)

    def test_the_port_computes_exactly_what_the_original_did(self):
        ported, _ = port_raw_indices(RAW_SOURCE, SCHEMA)
        self.assertTrue(
            check_port_equivalence(RAW_SOURCE, ported, OBS_DIM, ACT_DIM, SCHEMA)
        )

    def test_a_wrong_port_is_caught(self):
        wrong = RAW_SOURCE.replace('obs[:, 4:7]', 'self.obs_field(obs, "goal")')
        self.assertFalse(
            check_port_equivalence(RAW_SOURCE, wrong, OBS_DIM, ACT_DIM, SCHEMA)
        )

    def test_ported_source_passes_the_full_validator(self):
        ported, _ = port_raw_indices(RAW_SOURCE, SCHEMA)
        policy = instantiate_source(ported, OBS_DIM, ACT_DIM, SCHEMA)
        report = validate_policy(policy, OBS_DIM, ACT_DIM, source=ported, schema=SCHEMA)
        self.assertTrue(report.ok, report.summary())


class TestStructureBank(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = load_bank()

    def test_the_bank_is_not_empty_and_every_structure_validates(self):
        self.assertGreaterEqual(len(self.bank), 4)
        for s in self.bank:
            report = validate_policy(
                s.build(SCHEMA), OBS_DIM, ACT_DIM, source=s.source, schema=SCHEMA
            )
            self.assertTrue(report.ok, f"{s.structure_id}: {report.summary()}")

    def test_both_parameter_families_are_represented(self):
        families = {s.family for s in self.bank}
        self.assertEqual(families, {"simple", "complex"})

    def test_family_follows_the_declared_parameter_threshold(self):
        for s in self.bank:
            expected = "simple" if s.num_parameters <= 10 else "complex"
            self.assertEqual(s.family, expected, s.structure_id)

    def test_every_structure_records_its_provenance(self):
        for s in self.bank:
            self.assertTrue(s.provenance, s.structure_id)

    def test_filtering_by_family_returns_only_that_family(self):
        simple = load_bank(family="simple")
        self.assertTrue(simple)
        self.assertTrue(all(s.family == "simple" for s in simple))

    def test_build_returns_a_fresh_policy_each_time(self):
        s = self.bank[0]
        a, b = s.build(SCHEMA), s.build(SCHEMA)
        self.assertIsNot(a, b)
        with torch.no_grad():
            list(a.parameters())[0].add_(1.0)
        self.assertNotEqual(
            float(list(a.parameters())[0].reshape(-1)[0]),
            float(list(b.parameters())[0].reshape(-1)[0]),
        )


# ---------------------------------------------------------------------------
#  Fitting methods
# ---------------------------------------------------------------------------


class TestFittingMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = prepare_data(make_buffer(), val_fraction=0.25, split_seed=0)
        cls.structure = next(s for s in load_bank(family="simple"))

    def test_data_is_split_by_episode(self):
        self.assertEqual(self.data.split_info["mode"], "by_episode")
        self.assertTrue(self.data.has_validation)

    def test_every_method_runs_and_reports_what_it_cost(self):
        for method in METHODS:
            with self.subTest(method=method):
                result = fit(
                    method, self.structure, self.data, budget=60, batch_size=32,
                    seed=0, schema=SCHEMA,
                )
                self.assertEqual(result.method, method)
                self.assertGreater(result.objective_evaluations, 0)
                self.assertGreater(result.transition_evaluations, 0)
                self.assertGreater(result.wall_time_seconds, 0.0)
                self.assertTrue(np.isfinite(result.train_loss_final))
                self.assertTrue(np.isfinite(result.validation_loss_final))

    def test_gradient_methods_receive_exactly_the_declared_budget(self):
        budget = 50
        counts = {
            method: fit(
                method, self.structure, self.data, budget=budget, batch_size=16,
                seed=0, schema=SCHEMA,
            ).objective_evaluations
            for method in (ADAM_BASELINE, ADAM_SCALED, MULTISTART_ADAM)
        }
        self.assertEqual(set(counts.values()), {budget})

    def test_derivative_free_reports_its_actual_count_when_granularity_differs(self):
        result = fit(
            DIFFERENTIAL_EVOLUTION, self.structure, self.data, budget=100,
            batch_size=16, seed=0, schema=SCHEMA,
        )
        self.assertEqual(result.notes["requested_budget"], 100)
        # scipy evaluates whole generations, so the count lands near the budget
        # rather than exactly on it. Reporting both is the point.
        self.assertGreater(result.objective_evaluations, 0)

    def test_baseline_reproduces_the_live_configuration(self):
        result = fit(
            ADAM_BASELINE, self.structure, self.data, budget=30, batch_size=16,
            seed=0, schema=SCHEMA,
        )
        self.assertEqual(result.notes["lr"], 1e-3)
        self.assertEqual(result.notes["scheduler"], "none")
        self.assertEqual(result.restarts, 1)

    def test_multistart_uses_the_declared_number_of_restarts(self):
        result = fit(
            MULTISTART_ADAM, self.structure, self.data, budget=40, batch_size=16,
            seed=0, schema=SCHEMA, restarts=4,
        )
        self.assertEqual(result.restarts, 4)
        self.assertEqual(result.notes["steps_per_restart"], [10, 10, 10, 10])
        self.assertEqual(result.objective_evaluations, 40)

    def test_multistart_spends_an_uneven_budget_exactly(self):
        result = fit(
            MULTISTART_ADAM, self.structure, self.data, budget=50, batch_size=16,
            seed=0, schema=SCHEMA, restarts=4,
        )
        self.assertEqual(sum(result.notes["steps_per_restart"]), 50)
        self.assertEqual(result.objective_evaluations, 50)

    def test_fitting_actually_reduces_the_loss(self):
        policy = self.structure.build(SCHEMA)
        before = float(
            bc_objective(policy, self.data.train_obs, self.data.train_actions)
        )
        result = fit(
            ADAM_BASELINE, self.structure, self.data, budget=200, batch_size=32,
            seed=0, schema=SCHEMA,
        )
        self.assertLess(result.train_loss_final, before)

    def test_checkpoint_rules_select_different_state_when_they_differ(self):
        result = fit(
            ADAM_BASELINE, self.structure, self.data, budget=60, batch_size=16,
            seed=0, schema=SCHEMA, val_every=10,
        )
        self.assertIsNotNone(result.state_for(CHECKPOINT_FINAL))
        self.assertIsNotNone(result.state_for(CHECKPOINT_BEST_VALIDATION))

    def test_the_optimizer_comparison_uses_only_the_loss_based_rules(self):
        # Without a rollout probe during fitting the best-rollout rule falls back
        # to the final parameters, so including it would add a duplicate row.
        self.assertEqual(BenchmarkConfig().checkpoint_rules, LOSS_BASED_CHECKPOINT_RULES)

    def test_the_best_rollout_rule_falls_back_when_nothing_probed_rollouts(self):
        result = fit(
            ADAM_BASELINE, self.structure, self.data, budget=10, batch_size=16,
            seed=0, schema=SCHEMA,
        )
        from lares.fitting.optimizers import CHECKPOINT_BEST_ROLLOUT

        self.assertIs(
            result.state_for(CHECKPOINT_BEST_ROLLOUT), result.state_for(CHECKPOINT_FINAL)
        )

    def test_an_unknown_checkpoint_rule_is_refused(self):
        result = fit(
            ADAM_BASELINE, self.structure, self.data, budget=10, batch_size=16,
            seed=0, schema=SCHEMA,
        )
        with self.assertRaises(ValueError):
            result.state_for("whatever_looks_best")

    def test_an_unknown_method_is_refused(self):
        with self.assertRaises(ValueError):
            fit("simulated_annealing", self.structure, self.data)

    def test_result_serialises_without_the_parameter_tensors(self):
        result = fit(
            ADAM_BASELINE, self.structure, self.data, budget=10, batch_size=16,
            seed=0, schema=SCHEMA,
        )
        d = result.to_dict()
        self.assertNotIn("final_state", d)
        self.assertIn("wall_time_seconds", d)

    def test_bound_activity_is_reported_per_parameter(self):
        result = fit(
            ADAM_BASELINE, self.structure, self.data, budget=20, batch_size=16,
            seed=0, schema=SCHEMA,
        )
        self.assertIsNotNone(result.bound_activity)
        for entry in result.bound_activity.values():
            self.assertIn("at_lower", entry)
            self.assertIn("range", entry)

    def test_perturbation_stays_inside_the_declared_ranges(self):
        policy = self.structure.build(SCHEMA)
        perturb_within_ranges(policy, 0.9, torch.Generator().manual_seed(0))
        ranges = policy.get_param_ranges()
        for name, param in policy.named_parameters():
            lo, hi = ranges[name]
            self.assertGreaterEqual(float(param.detach().min()), lo - 1e-6, name)
            self.assertLessEqual(float(param.detach().max()), hi + 1e-6, name)


# ---------------------------------------------------------------------------
#  Sensitivity
# ---------------------------------------------------------------------------


class TestSensitivity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.structure = next(
            s for s in load_bank(family="simple") if s.structure_id == "simple_reach_push"
        )
        cls.policy = cls.structure.build(SCHEMA)
        cls.obs = 0.1 * torch.randn(64, OBS_DIM, generator=torch.Generator().manual_seed(0))
        cls.actions = torch.zeros(64, ACT_DIM)

    def test_the_scale_parameter_does_not_move_the_executed_action(self):
        # Deterministic execution is tanh(mean); std is ignored. A dead scale
        # parameter is the correct finding, not a bug.
        stats = action_sensitivity(self.policy, self.obs)
        self.assertTrue(stats["log_std"]["dead"])
        self.assertIn("log_std", dead_parameters(stats))

    def test_control_parameters_do_move_the_action(self):
        stats = action_sensitivity(self.policy, self.obs)
        for name in ("w_reach", "w_push", "contact_radius"):
            self.assertFalse(stats[name]["dead"], name)
            self.assertGreater(stats[name]["mean_abs_action_change"], 0.0)

    def test_sensitivity_leaves_the_parameters_where_it_found_them(self):
        before = {n: p.detach().clone() for n, p in self.policy.named_parameters()}
        action_sensitivity(self.policy, self.obs)
        for name, param in self.policy.named_parameters():
            self.assertTrue(torch.allclose(before[name], param.detach()), name)

    def test_relative_position_locates_a_parameter_inside_its_range(self):
        stats = action_sensitivity(self.policy, self.obs)
        position = stats["contact_radius"]["relative_position"]
        self.assertGreaterEqual(position, 0.0)
        self.assertLessEqual(position, 1.0)

    def test_gradient_sensitivity_scales_by_the_declared_range(self):
        stats = gradient_sensitivity(self.policy, self.obs, self.actions)
        for name, entry in stats.items():
            self.assertIn("range_scaled_gradient", entry)
            self.assertIn("no_gradient", entry)

    def test_gradient_sensitivity_clears_the_gradients_it_used(self):
        gradient_sensitivity(self.policy, self.obs, self.actions)
        for _, param in self.policy.named_parameters():
            self.assertTrue(param.grad is None or float(param.grad.abs().sum()) == 0.0)

    def test_summary_orders_by_influence_and_flags_dead_parameters(self):
        stats = action_sensitivity(self.policy, self.obs)
        text = summarise_sensitivity(stats, gradient_sensitivity(self.policy, self.obs, self.actions))
        self.assertIn("no influence", text)
        self.assertIn("w_push", text)


# ---------------------------------------------------------------------------
#  Benchmark aggregation and selection
# ---------------------------------------------------------------------------


def row(method, rule, success, progress=0.0, val=1.0, wall=1.0, family="simple"):
    return {
        "structure_id": "s",
        "family": family,
        "num_parameters": 6,
        "method": method,
        "seed": 0,
        "checkpoint_rule": rule,
        "train_loss": val,
        "validation_loss": val,
        "objective_evaluations": 100,
        "transition_evaluations": 100,
        "wall_time_seconds": wall,
        "converged_at": 10,
        "success_rate": success,
        "mean_return": 1.0,
        "signed_goal_progress": progress,
        "failure_labels": None,
        "bound_activity_fraction": 0.0,
        "dead_parameters": [],
    }


class TestBenchmarkAggregation(unittest.TestCase):
    def test_rows_group_by_method_and_rule(self):
        rows = [
            row(ADAM_BASELINE, CHECKPOINT_FINAL, 0.2),
            row(ADAM_BASELINE, CHECKPOINT_FINAL, 0.4),
            row(ADAM_SCALED, CHECKPOINT_FINAL, 0.6),
        ]
        out = {(e["method"], e["checkpoint_rule"]): e for e in aggregate(rows)}
        self.assertAlmostEqual(out[(ADAM_BASELINE, CHECKPOINT_FINAL)]["success_rate"], 0.3)
        self.assertEqual(out[(ADAM_BASELINE, CHECKPOINT_FINAL)]["n"], 2)

    def test_missing_rollouts_aggregate_to_unavailable_not_zero(self):
        r = row(ADAM_BASELINE, CHECKPOINT_FINAL, None)
        self.assertIsNone(aggregate([r])[0]["success_rate"])

    def test_selection_follows_the_declared_primary_metric(self):
        rows = [
            row(ADAM_BASELINE, CHECKPOINT_FINAL, 0.2, progress=0.9),
            row(ADAM_SCALED, CHECKPOINT_FINAL, 0.5, progress=0.1),
        ]
        chosen = select_default(aggregate(rows))
        self.assertEqual(chosen["selected"]["method"], ADAM_SCALED)
        self.assertEqual(chosen["rule"], SELECTION_RULE)

    def test_progress_breaks_a_success_tie(self):
        rows = [
            row(ADAM_BASELINE, CHECKPOINT_FINAL, 0.5, progress=0.1),
            row(ADAM_SCALED, CHECKPOINT_FINAL, 0.5, progress=0.9),
        ]
        self.assertEqual(select_default(aggregate(rows))["selected"]["method"], ADAM_SCALED)

    def test_wall_time_breaks_a_remaining_tie(self):
        rows = [
            row(ADAM_BASELINE, CHECKPOINT_FINAL, 0.5, progress=0.5, wall=9.0),
            row(ADAM_SCALED, CHECKPOINT_FINAL, 0.5, progress=0.5, wall=1.0),
        ]
        self.assertEqual(select_default(aggregate(rows))["selected"]["method"], ADAM_SCALED)

    def test_selection_with_no_results_says_so(self):
        self.assertIsNone(select_default([])["selected"])

    def test_loss_versus_success_needs_variation(self):
        rows = [row(ADAM_BASELINE, CHECKPOINT_FINAL, 0.5, val=1.0) for _ in range(4)]
        self.assertIn("unavailable", loss_versus_success(rows))

    def test_loss_versus_success_reports_a_correlation(self):
        rows = [
            row(ADAM_BASELINE, CHECKPOINT_FINAL, s, val=v)
            for s, v in ((0.1, 0.9), (0.3, 0.7), (0.6, 0.4), (0.9, 0.1))
        ]
        out = loss_versus_success(rows)
        self.assertIn("pearson_r", out)
        self.assertLess(out["pearson_r"], 0.0)

    def test_table_renders_unavailable_columns(self):
        text = format_table(aggregate([row(ADAM_BASELINE, CHECKPOINT_FINAL, None)]))
        self.assertIn("n/a", text)


class TestBenchmarkRun(unittest.TestCase):
    def test_a_loss_only_benchmark_leaves_the_rollout_columns_unmeasured(self):
        data = prepare_data(make_buffer(), val_fraction=0.25, split_seed=0)
        structures = load_bank(family="simple")[:1]
        config = BenchmarkConfig(
            methods=(ADAM_BASELINE, ADAM_SCALED), budget=20, batch_size=16, seeds=(0,)
        )
        payload = run_benchmark(structures, data, SCHEMA, config)
        self.assertFalse(payload["rollouts_scored"])
        expected = len(config.methods) * len(config.checkpoint_rules) * len(config.seeds)
        self.assertEqual(len(payload["rows"]), expected)
        for r in payload["rows"]:
            self.assertIsNone(r["success_rate"])
        self.assertIn(structures[0].structure_id, payload["sensitivity"])
        self.assertIn("unavailable", payload["loss_versus_success"])


if __name__ == "__main__":
    unittest.main()
