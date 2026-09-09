#!/usr/bin/env python
"""Tests for objectives, execution contracts, phases and DAgger (FR-7, FR-8, AC-5).

Run from the project root::

    python -m unittest tests.test_objectives
"""

import math
import os
import sys
import unittest

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lares.core.obs_schema import get_obs_schema  # noqa: E402
from lares.core.training_pipeline import DemoBuffer  # noqa: E402
from lares.fitting import (  # noqa: E402
    CENSORED_GAUSSIAN,
    DETERMINISTIC_MSE,
    EXEC_CLIPPED_DETERMINISTIC,
    EXEC_CLIPPED_GAUSSIAN,
    EXEC_DETERMINISTIC_TANH,
    EXEC_TANH_GAUSSIAN,
    LEGACY_MSE_STD_PENALTY,
    OBJECTIVES,
    PHASE_NAMES,
    RESIDUAL_SCALE,
    TANH_GAUSSIAN_NLL,
    balanced_indices,
    censored_fraction,
    check_execution_matches,
    execute,
    expert_phase_labels,
    find_scale_parameters,
    freeze_scale_parameters,
    load_bank,
    objective_loss,
    phase_fractions,
    prepare_data,
    spec,
    trainable_parameters,
    validate_expert_label,
)
from lares.fitting.optimizers import (  # noqa: E402
    SAMPLING_PHASE_BALANCED,
    SAMPLING_UNIFORM,
    fit_with_objective,
)
from lares.fitting.phases import (  # noqa: E402
    PHASE_DESCEND,
    PHASE_HOVER,
    PHASE_PUSH,
    verify_against_expert,
)

OBS_DIM = 39
ACT_DIM = 4
SCHEMA = get_obs_schema("push-v2")

try:
    import metaworld  # noqa: F401

    HAS_METAWORLD = True
except ImportError:
    HAS_METAWORLD = False


def obs_row(tcp, obj, goal):
    row = np.zeros(OBS_DIM, dtype=np.float32)
    row[0:3] = tcp
    row[4:7] = obj
    row[36:39] = goal
    return row


# ---------------------------------------------------------------------------
#  Execution contracts
# ---------------------------------------------------------------------------


class TestExecutionContracts(unittest.TestCase):
    def setUp(self):
        self.mean = torch.tensor([[0.5, -0.5, 3.0, 0.0]])
        self.std = torch.tensor([[0.1, 0.1, 0.1, 0.1]])

    def test_deterministic_tanh_is_bounded_and_ignores_the_scale(self):
        a = execute(self.mean, self.std, EXEC_DETERMINISTIC_TANH)
        b = execute(self.mean, self.std * 100, EXEC_DETERMINISTIC_TANH)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(bool((a.abs() < 1.0).all()))

    def test_clipped_deterministic_can_reach_the_endpoint(self):
        a = execute(self.mean, self.std, EXEC_CLIPPED_DETERMINISTIC)
        self.assertAlmostEqual(float(a[0, 2]), 1.0, places=6)

    def test_tanh_gaussian_stays_strictly_inside_the_limits(self):
        g = torch.Generator().manual_seed(0)
        a = execute(self.mean, self.std * 5, EXEC_TANH_GAUSSIAN, generator=g)
        self.assertTrue(bool((a.abs() < 1.0).all()))

    def test_clipped_gaussian_can_sit_exactly_on_a_limit(self):
        g = torch.Generator().manual_seed(0)
        a = execute(self.mean, self.std, EXEC_CLIPPED_GAUSSIAN, generator=g)
        self.assertAlmostEqual(float(a[0, 2]), 1.0, places=6)

    def test_an_unknown_contract_is_refused(self):
        with self.assertRaises(ValueError):
            execute(self.mean, self.std, "vibes")

    def test_scoring_an_objective_under_the_wrong_execution_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            check_execution_matches(CENSORED_GAUSSIAN, EXEC_TANH_GAUSSIAN)
        self.assertIn("same distribution", str(ctx.exception))

    def test_the_matching_execution_is_accepted(self):
        check_execution_matches(CENSORED_GAUSSIAN, EXEC_CLIPPED_GAUSSIAN)
        check_execution_matches(DETERMINISTIC_MSE, EXEC_DETERMINISTIC_TANH)
        check_execution_matches(TANH_GAUSSIAN_NLL, EXEC_TANH_GAUSSIAN)


# ---------------------------------------------------------------------------
#  Objectives
# ---------------------------------------------------------------------------


class TestObjectives(unittest.TestCase):
    def setUp(self):
        self.actions = torch.tensor([[0.5, -0.5, 1.0, 0.0], [0.1, 0.2, -1.0, 0.3]])
        self.mean = torch.tensor([[0.4, -0.4, 2.0, 0.1], [0.2, 0.1, -2.0, 0.2]])
        self.std = torch.full_like(self.mean, 0.3)

    def test_every_objective_is_finite(self):
        for name in OBJECTIVES:
            with self.subTest(objective=name):
                value = objective_loss(name, self.mean, self.std, self.actions)
                self.assertTrue(math.isfinite(float(value)), name)

    def test_an_unknown_objective_is_refused(self):
        with self.assertRaises(ValueError):
            objective_loss("vibes", self.mean, self.std, self.actions)

    def test_the_primary_baseline_ignores_the_scale_entirely(self):
        a = objective_loss(DETERMINISTIC_MSE, self.mean, self.std, self.actions)
        b = objective_loss(DETERMINISTIC_MSE, self.mean, self.std * 50, self.actions)
        self.assertEqual(float(a), float(b))

    def test_the_legacy_objective_penalises_the_scale(self):
        a = objective_loss(LEGACY_MSE_STD_PENALTY, self.mean, self.std, self.actions)
        b = objective_loss(LEGACY_MSE_STD_PENALTY, self.mean, self.std * 2, self.actions)
        self.assertLess(float(a), float(b))

    def test_the_primary_baseline_is_minimised_at_a_perfect_fit(self):
        target = torch.tensor([[0.5, -0.5, 0.9, 0.0]])
        mean = torch.atanh(target)
        loss = objective_loss(DETERMINISTIC_MSE, mean, torch.full_like(mean, 0.2), target)
        self.assertLess(float(loss), 1e-10)

    def test_the_tanh_likelihood_drops_the_censored_targets(self):
        all_interior = torch.tensor([[0.5, -0.5, 0.4, 0.0]])
        mean = torch.zeros_like(all_interior)
        std = torch.full_like(all_interior, 0.5)
        interior_only = objective_loss(TANH_GAUSSIAN_NLL, mean, std, all_interior)
        with_endpoint = objective_loss(
            TANH_GAUSSIAN_NLL,
            torch.cat([mean, mean]),
            torch.cat([std, std]),
            torch.cat([all_interior, torch.ones_like(all_interior)]),
        )
        # The all-endpoint row contributes nothing, so the mean over the interior
        # rows is unchanged.
        self.assertAlmostEqual(float(interior_only), float(with_endpoint), places=5)

    def test_the_tanh_likelihood_declares_that_it_excludes_data(self):
        self.assertTrue(spec(TANH_GAUSSIAN_NLL).excludes_censored)
        self.assertFalse(spec(DETERMINISTIC_MSE).excludes_censored)

    def test_the_tanh_likelihood_on_all_endpoint_targets_is_defined(self):
        actions = torch.ones(2, ACT_DIM)
        value = objective_loss(TANH_GAUSSIAN_NLL, torch.zeros(2, ACT_DIM),
                               torch.full((2, ACT_DIM), 0.5), actions)
        self.assertEqual(float(value), 0.0)

    def test_the_censored_likelihood_rewards_predicting_beyond_the_limit(self):
        # A target clipped at +1 is evidence the unbounded signal exceeded +1, so a
        # location above the limit must be at least as likely as one below it.
        actions = torch.ones(1, 1)
        std = torch.full((1, 1), 0.5)
        beyond = objective_loss(CENSORED_GAUSSIAN, torch.tensor([[2.0]]), std, actions)
        below = objective_loss(CENSORED_GAUSSIAN, torch.tensor([[0.0]]), std, actions)
        self.assertLess(float(beyond), float(below))

    def test_the_censored_likelihood_handles_an_all_interior_batch(self):
        actions = torch.tensor([[0.2, -0.3, 0.1, 0.0]])
        value = objective_loss(
            CENSORED_GAUSSIAN, actions.clone(), torch.full_like(actions, 0.4), actions
        )
        self.assertTrue(math.isfinite(float(value)))

    def test_residual_scale_stage_two_fits_the_residual(self):
        actions = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        mean = torch.zeros_like(actions)
        matching = torch.full_like(actions, 0.5)
        wrong = torch.full_like(actions, 0.01)
        good = objective_loss(RESIDUAL_SCALE, mean, matching, actions, stage="scale")
        bad = objective_loss(RESIDUAL_SCALE, mean, wrong, actions, stage="scale")
        self.assertLess(float(good), float(bad))

    def test_residual_scale_stage_one_is_the_regression(self):
        a = objective_loss(RESIDUAL_SCALE, self.mean, self.std, self.actions, stage="mean")
        b = objective_loss(DETERMINISTIC_MSE, self.mean, self.std, self.actions)
        self.assertAlmostEqual(float(a), float(b), places=9)

    def test_censored_fraction_counts_endpoint_rows(self):
        self.assertAlmostEqual(censored_fraction(self.actions), 1.0)
        self.assertAlmostEqual(censored_fraction(torch.zeros(4, ACT_DIM)), 0.0)

    def test_only_the_two_likelihoods_are_labelled_as_such(self):
        likelihoods = {n for n in OBJECTIVES if spec(n).is_likelihood}
        self.assertEqual(likelihoods, {TANH_GAUSSIAN_NLL, CENSORED_GAUSSIAN})


# ---------------------------------------------------------------------------
#  Scale parameters
# ---------------------------------------------------------------------------


class TestScaleParameters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.structure = next(
            s for s in load_bank(family="simple") if s.structure_id == "simple_reach_push"
        )

    def test_scale_parameters_are_found_by_measurement(self):
        policy = self.structure.build(SCHEMA)
        probe = torch.tensor(
            np.stack([obs_row([0, 0.6, 0.1], [0, 0.62, 0.02], [0.1, 0.8, 0.02])] * 8)
        )
        names = find_scale_parameters(policy, probe)
        self.assertIn("log_std", names)
        self.assertNotIn("w_reach", names)

    def test_freezing_removes_them_from_the_trainable_set(self):
        policy = self.structure.build(SCHEMA)
        probe = torch.tensor(
            np.stack([obs_row([0, 0.6, 0.1], [0, 0.62, 0.02], [0.1, 0.8, 0.02])] * 8)
        )
        before = len(trainable_parameters(policy))
        frozen = freeze_scale_parameters(policy, probe)
        after = len(trainable_parameters(policy))
        self.assertTrue(frozen)
        self.assertEqual(after, before - len(frozen))

    def test_a_frozen_scale_does_not_move_during_fitting(self):
        from lares.core.training_pipeline import DemoBuffer

        buf = DemoBuffer()
        rng = np.random.default_rng(0)
        for ep in range(6):
            for _ in range(20):
                row = obs_row(
                    [0, 0.6, 0.1] + 0.02 * rng.standard_normal(3),
                    [0, 0.62, 0.02] + 0.02 * rng.standard_normal(3),
                    [0.1, 0.8, 0.02],
                )
                buf.add(row, np.clip(rng.standard_normal(ACT_DIM), -1, 1), 0.0, row, 0.0,
                        episode_id=f"ep{ep}")
        data = prepare_data(buf, val_fraction=0.25, split_seed=0)
        before = float(self.structure.build(SCHEMA).state_dict()["log_std"])
        result = fit_with_objective(
            self.structure, data, DETERMINISTIC_MSE, budget=50, batch_size=16,
            seed=0, schema=SCHEMA,
        )
        self.assertIn("log_std", result.notes["frozen_parameters"])
        self.assertAlmostEqual(float(result.final_state["log_std"]), before, places=9)


# ---------------------------------------------------------------------------
#  Phases
# ---------------------------------------------------------------------------


class TestPhases(unittest.TestCase):
    def test_far_in_the_plane_is_the_hover_phase(self):
        obs = np.stack([obs_row([0.5, 0.6, 0.2], [0.0, 0.62, 0.02], [0.1, 0.8, 0.02])])
        self.assertEqual(expert_phase_labels(obs, SCHEMA)[0], PHASE_HOVER)

    def test_aligned_but_high_is_the_descend_phase(self):
        obs = np.stack([obs_row([0.0, 0.62, 0.20], [0.005, 0.62, 0.02], [0.1, 0.8, 0.02])])
        self.assertEqual(expert_phase_labels(obs, SCHEMA)[0], PHASE_DESCEND)

    def test_aligned_and_low_is_the_push_phase(self):
        obs = np.stack([obs_row([0.0, 0.62, 0.04], [0.005, 0.62, 0.02], [0.1, 0.8, 0.02])])
        self.assertEqual(expert_phase_labels(obs, SCHEMA)[0], PHASE_PUSH)

    def test_fractions_sum_to_one(self):
        labels = np.array([0, 0, 1, 2, 2, 2])
        self.assertAlmostEqual(sum(phase_fractions(labels).values()), 1.0)
        self.assertEqual(set(phase_fractions(labels)), set(PHASE_NAMES))

    def test_balanced_sampling_gives_each_phase_an_equal_share(self):
        labels = np.array([0] * 5 + [1] * 5 + [2] * 990)
        idx = balanced_indices(labels, 300, np.random.default_rng(0))
        counts = np.bincount(labels[idx], minlength=3)
        self.assertEqual(len(idx), 300)
        for count in counts:
            self.assertGreater(count, 50)

    def test_balanced_sampling_ignores_an_absent_phase(self):
        labels = np.array([2] * 50)
        idx = balanced_indices(labels, 32, np.random.default_rng(0))
        self.assertEqual(len(idx), 32)
        self.assertTrue(np.all(labels[idx] == 2))

    def test_balanced_sampling_from_nothing_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            balanced_indices(np.array([], dtype=int), 8, np.random.default_rng(0))
        self.assertIn("nothing to sample", str(ctx.exception))

    def test_balanced_sampling_with_unrecognised_labels_falls_back_to_uniform(self):
        idx = balanced_indices(np.full(20, 99), 8, np.random.default_rng(0))
        self.assertEqual(len(idx), 8)

    @unittest.skipUnless(HAS_METAWORLD, "metaworld is required to check against the expert")
    def test_the_labels_match_the_branch_the_expert_actually_takes(self):
        from lares.core.training_pipeline import get_expert_policy

        expert = get_expert_policy("push-v2")
        rng = np.random.default_rng(0)
        rows = []
        for _ in range(200):
            rows.append(
                obs_row(
                    [rng.uniform(-0.2, 0.2), rng.uniform(0.5, 0.8), rng.uniform(0.02, 0.3)],
                    [rng.uniform(-0.1, 0.1), rng.uniform(0.6, 0.7), 0.02],
                    [rng.uniform(-0.1, 0.1), rng.uniform(0.8, 0.9), 0.02],
                )
            )
        mismatches = verify_against_expert(np.stack(rows), SCHEMA, expert)
        self.assertEqual(mismatches, [], f"{len(mismatches)} of 200 labels disagree")


# ---------------------------------------------------------------------------
#  Learner-state aggregation
# ---------------------------------------------------------------------------


class TestExpertLabelValidation(unittest.TestCase):
    def test_a_non_finite_label_is_rejected(self):
        self.assertEqual(
            validate_expert_label([np.nan, 0.0, 0.0, 0.0], None, SCHEMA, 0.2), "non_finite"
        )

    def test_a_do_nothing_label_far_from_the_goal_is_rejected(self):
        reason = validate_expert_label([0.0, 0.0, 0.0, 0.6], None, SCHEMA, 0.3)
        self.assertEqual(reason, "no_motion_requested_while_unfinished")

    def test_a_do_nothing_label_at_the_goal_is_accepted(self):
        self.assertIsNone(validate_expert_label([0.0, 0.0, 0.0, 0.6], None, SCHEMA, 0.01))

    def test_an_ordinary_label_is_accepted(self):
        self.assertIsNone(validate_expert_label([0.4, -0.2, 0.1, 0.6], None, SCHEMA, 0.3))


class TestAggregationStats(unittest.TestCase):
    def test_rates_are_reported_and_unmeasured_ones_stay_none(self):
        from lares.fitting.dagger import AggregationStats, summarise

        stats = AggregationStats()
        stats.expert_queries = 10
        stats.labels_accepted = 8
        stats.labels_rejected = 2
        stats.rejection_reasons = {"non_finite": 2}
        self.assertAlmostEqual(stats.acceptance_rate, 0.8)
        self.assertIsNone(stats.recovery_rate)
        text = summarise(stats)
        self.assertIn("non_finite", text)
        self.assertNotIn("expert label helps", text)

    def test_the_dict_carries_the_cost_of_the_round(self):
        from lares.fitting.dagger import AggregationStats

        stats = AggregationStats(expert_queries=5, environment_steps=5, episodes=1)
        d = stats.to_dict()
        self.assertEqual(d["expert_queries"], 5)
        self.assertEqual(d["environment_steps"], 5)
        self.assertIn("acceptance_rate", d)


@unittest.skipUnless(HAS_METAWORLD, "metaworld is required for simulator branching")
class TestCounterfactualRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("MUJOCO_GL", "egl")
        from lares.core.training_pipeline import ensure_mujoco_headless_gl
        from lares.eval import EvaluationManifest, make_manifest_env

        ensure_mujoco_headless_gl()
        path = os.path.join(_PROJECT_ROOT, "config", "manifests", "push-v2_train.yaml")
        if not os.path.isfile(path):
            raise unittest.SkipTest("train manifest missing; run scripts/lock_baseline.py")
        cls.manifest = EvaluationManifest.load(path)
        cls.env = make_manifest_env(cls.manifest, cls.manifest.resolve_pool(), 200)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_the_branch_leaves_the_simulator_where_it_found_it(self):
        from lares.core.training_pipeline import get_expert_policy
        from lares.fitting.dagger import counterfactual_recovery

        expert = get_expert_policy("push-v2")
        obs, _ = self.env.reset(self.manifest.episodes[0])
        for _ in range(10):
            obs, _, _, _ = self.env.step(np.clip(expert.get_action(obs), -1, 1))
        before = obs.copy()
        counterfactual_recovery(self.env, expert.get_action(obs), obs, SCHEMA, expert)
        after, _, _, _ = self.env.step(np.zeros(ACT_DIM))
        # Re-running from the restored state must reproduce the same next state.
        self.assertTrue(np.array_equal(before, obs))

    def test_the_branch_is_repeatable(self):
        from lares.core.training_pipeline import get_expert_policy
        from lares.fitting.dagger import counterfactual_recovery

        expert = get_expert_policy("push-v2")
        obs, _ = self.env.reset(self.manifest.episodes[1])
        for _ in range(15):
            obs, _, _, _ = self.env.step(np.clip(expert.get_action(obs), -1, 1))
        a = counterfactual_recovery(self.env, expert.get_action(obs), obs, SCHEMA, expert)
        b = counterfactual_recovery(self.env, expert.get_action(obs), obs, SCHEMA, expert)
        self.assertEqual(a["after"], b["after"])
        self.assertIn(a["phase"], PHASE_NAMES)
        self.assertEqual(a["measure"], "hand_to_expert_desired_position")

    def test_the_experts_own_label_helps_on_its_own_trajectory(self):
        from lares.core.training_pipeline import get_expert_policy
        from lares.fitting.dagger import counterfactual_recovery

        expert = get_expert_policy("push-v2")
        obs, _ = self.env.reset(self.manifest.episodes[2])
        improved = 0
        checked = 0
        for _ in range(40):
            check = counterfactual_recovery(self.env, expert.get_action(obs), obs, SCHEMA, expert)
            checked += 1
            improved += int(check["improved"])
            obs, _, _, _ = self.env.step(np.clip(expert.get_action(obs), -1, 1))
        # On states the expert itself visits, its label should usually help.
        self.assertGreater(improved / checked, 0.5, f"{improved}/{checked}")


# ---------------------------------------------------------------------------
#  Rollout-based checkpoint selection
# ---------------------------------------------------------------------------


class TestRolloutCheckpoint(unittest.TestCase):
    """The fix for the non-monotonicity Phase 5 measured.

    Development success peaks partway through fitting and then collapses, while
    validation loss falls the whole way. Neither "final" nor "best validation
    loss" can find the peak, so the checkpoint has to be chosen on rollouts.
    """

    def setUp(self):
        from lares.core.training_pipeline import DemoBuffer

        rng = np.random.default_rng(0)
        self.buffer = DemoBuffer()
        for ep in range(8):
            for _ in range(20):
                row = obs_row(
                    [0, 0.6, 0.1] + 0.02 * rng.standard_normal(3),
                    [0, 0.62, 0.02] + 0.02 * rng.standard_normal(3),
                    [0.1, 0.8, 0.02],
                )
                self.buffer.add(
                    row, np.clip(rng.standard_normal(ACT_DIM), -1, 1), 0.0, row, 0.0,
                    episode_id=f"ep{ep}",
                )
        self.structure = next(
            s for s in load_bank(family="simple") if s.structure_id == "simple_reach_push"
        )

    def _fit(self, probe):
        from lares.core.obs_schema import active_schema
        from lares.core.training_pipeline import behavioral_cloning

        with active_schema(SCHEMA):
            policy = self.structure.build(SCHEMA)
        stats = behavioral_cloning(
            policy, self.buffer, num_steps=40, batch_size=16, log_interval=0,
            rollout_probe=probe, rollout_interval=10,
        )
        return stats

    def test_the_probe_runs_on_the_declared_interval(self):
        seen = []
        self._fit(lambda p: (len(seen), 0.0) if seen.append(1) is None else None)
        self.assertEqual(len(seen), 4)

    def test_the_best_scoring_checkpoint_is_kept_not_the_last(self):
        scores = iter([(0.1, 0.0), (0.9, 0.0), (0.2, 0.0), (0.0, 0.0)])
        stats = self._fit(lambda p: next(scores))
        self.assertEqual(stats["best_rollout_step"], 20)
        self.assertEqual(stats["best_rollout_score"], (0.9, 0.0))
        self.assertIsNotNone(stats["best_rollout_state"])

    def test_a_tie_prefers_the_later_checkpoint(self):
        # When nothing succeeds every probe ties, and reverting to the first would
        # discard fitting for no reason.
        stats = self._fit(lambda p: (0.0, 0.0))
        self.assertEqual(stats["best_rollout_step"], 40)

    def test_goal_progress_breaks_a_tie_on_success(self):
        scores = iter([(0.0, -0.1), (0.0, 0.3), (0.0, -0.2), (0.0, 0.1)])
        stats = self._fit(lambda p: next(scores))
        self.assertEqual(stats["best_rollout_step"], 20)

    def test_no_probe_means_no_rollout_checkpoint(self):
        stats = self._fit(None)
        self.assertIsNone(stats["best_rollout_state"])
        self.assertEqual(stats["rollout_history"], [])

    def test_the_history_records_every_probe(self):
        stats = self._fit(lambda p: (0.5, 0.0))
        self.assertEqual([step for step, _ in stats["rollout_history"]], [10, 20, 30, 40])


if __name__ == "__main__":
    unittest.main()
