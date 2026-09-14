"""Objective-aware fitting of an existing policy and checkpoint consistency.

``debug_spec.md`` Stage B (D3, D4). Run from the project root::

    python -m unittest tests.test_fitting_provenance
"""

import copy
import io
import os
import sys
import unittest

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from lares.core.obs_schema import active_schema, get_obs_schema  # noqa: E402
from lares.core.symbolic_policy import SymbolicPolicy  # noqa: E402
from lares.core.training_pipeline import DemoBuffer  # noqa: E402
from lares.fitting.checkpoints import (  # noqa: E402
    CHECKPOINT_TIE_RULE,
    RolloutScore,
    default_checkpoint_steps,
    is_better_checkpoint,
    rollout_score,
)
from lares.fitting.objectives import DETERMINISTIC_MSE, LEGACY_MSE_STD_PENALTY, objective_loss  # noqa: E402
from lares.fitting.optimizers import (  # noqa: E402
    fit_policy_with_objective,
    fit_with_objective,
    prepare_data,
    state_parameters_sha256,
)
from lares.fitting.structure_bank import load_bank  # noqa: E402

SCHEMA = get_obs_schema("push-v2")
OBS_DIM, ACT_DIM = 39, 4


class LogStdPolicy(SymbolicPolicy):
    """A control law plus a learnable scale the deterministic objective must not touch."""

    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.gain = nn.Parameter(torch.tensor(2.0))
        self.goal_gain = nn.Parameter(torch.tensor(0.5))
        self.grip = nn.Parameter(torch.tensor(0.0))
        self.log_std = nn.Parameter(torch.tensor(-1.0))

    def forward(self, obs):
        tcp = self.obs_field(obs, "tcp")
        obj = self.obs_field(obs, "obj")
        goal = self.obs_field(obs, "goal")
        move = self.gain * (obj - tcp) + self.goal_gain * (goal - obj)
        grip = self.grip.unsqueeze(0).expand(obs.shape[0], 1)
        mean = torch.cat([move, grip], dim=1)
        return mean, torch.exp(self.log_std) * torch.ones_like(mean)

    def get_param_ranges(self):
        return {"gain": (0.1, 20.0), "goal_gain": (0.0, 5.0), "grip": (-1.0, 1.0), "log_std": (-5.0, 0.0)}


def make_buffer(episodes=8, steps=20, seed=0):
    rng = np.random.default_rng(seed)
    buffer = DemoBuffer()
    for ep in range(episodes):
        for _ in range(steps):
            obs = np.zeros(OBS_DIM, dtype=np.float32)
            tcp = np.array([0.0, 0.6, 0.1]) + 0.05 * rng.standard_normal(3)
            obj = np.array([0.0, 0.65, 0.02]) + 0.02 * rng.standard_normal(3)
            goal = np.array([0.1, 0.8, 0.02])
            for name, value in (("tcp", tcp), ("obj", obj), ("goal", goal)):
                f = SCHEMA.field(name)
                obs[f.start : f.stop] = value
            action = np.clip(np.concatenate([8.0 * (obj - tcp), [0.3]]), -1, 1)
            buffer.add(obs, action, 0.0, obs, 0.0, episode_id=f"ep{ep}")
    return buffer


def build(cls=LogStdPolicy):
    with active_schema(SCHEMA):
        return cls(OBS_DIM, ACT_DIM)


class FittingCase(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.data = prepare_data(make_buffer(), val_fraction=0.25, split_seed=0)

    def fit(self, probe=None, steps=(0, 10, 20, 30, 40), budget=40, objective=DETERMINISTIC_MSE, policy=None):
        policy = policy or build()
        result = fit_policy_with_objective(
            policy, self.data, objective, budget=budget, batch_size=16, lr=5e-2, seed=0,
            rollout_probe=probe, checkpoint_steps=steps if probe is not None else None,
        )
        return policy, result


class TestScaleIsNotFitted(FittingCase):
    def test_a_learnable_log_std_is_left_unchanged_under_deterministic_mse(self):
        policy, result = self.fit()
        self.assertIn("log_std", result.frozen_parameters)
        self.assertEqual(float(policy.log_std), -1.0)
        self.assertNotEqual(float(policy.gain), 2.0)
        self.assertEqual(result.objective_config["objective"], DETERMINISTIC_MSE)
        self.assertFalse(result.objective_config["optimises_scale"])

    def test_the_legacy_objective_does_move_it_which_is_why_it_is_not_the_default(self):
        policy, _ = self.fit(objective=LEGACY_MSE_STD_PENALTY)
        self.assertLess(float(policy.log_std), -1.0)

    def test_the_fixed_scale_helper_is_not_a_parameter(self):
        policy = build()
        mean = torch.zeros(3, ACT_DIM)
        std = policy.fixed_std(mean)
        self.assertTrue(torch.all(std == SymbolicPolicy.DEFAULT_ACTION_SCALE))
        self.assertNotIn("fixed_action_scale", dict(policy.named_parameters()))
        self.assertNotIn("fixed_action_scale", policy.state_dict())


class TestReportedLosses(FittingCase):
    def test_final_loss_is_the_named_objective_on_the_final_state(self):
        _, result = self.fit()
        fresh = build()
        fresh.load_state_dict(result.final_state)
        with torch.no_grad():
            mean, std = fresh(self.data.train_obs)
            direct = float(objective_loss(DETERMINISTIC_MSE, mean, std, self.data.train_actions))
            v_mean, v_std = fresh(self.data.val_obs)
            direct_val = float(objective_loss(DETERMINISTIC_MSE, v_mean, v_std, self.data.val_actions))
        self.assertAlmostEqual(result.train_loss_final, direct, places=6)
        self.assertAlmostEqual(result.validation_loss_final, direct_val, places=6)
        # The legacy value would include 0.01 * exp(-1): it must not be what was reported.
        self.assertNotAlmostEqual(result.train_loss_final, direct + 0.01 * float(np.exp(-1.0)), places=4)


class TestCheckpointSelection(FittingCase):
    def test_a_non_final_best_checkpoint_reports_its_own_parameters_and_diagnostics(self):
        scores = iter([(0.0, 0.0), (0.9, 0.0), (0.2, 0.0), (0.1, 0.0), (0.0, 0.0)])
        policy, result = self.fit(probe=lambda p: next(scores))
        self.assertEqual(result.selected_step, 10)
        self.assertEqual(result.checkpoint_history[1]["step"], 10)
        selected_hash = state_parameters_sha256(result.selected_state)
        self.assertEqual(selected_hash, result.checkpoint_history[1]["parameters_sha256"])
        self.assertEqual(result.selected_diagnostics["parameters_sha256"], selected_hash)
        self.assertEqual(state_parameters_sha256(policy.state_dict()), selected_hash)
        self.assertNotEqual(result.final_diagnostics["parameters_sha256"], selected_hash)
        self.assertEqual(result.final_diagnostics["parameters_sha256"], state_parameters_sha256(result.final_state))
        with torch.no_grad():
            mean, std = policy(self.data.train_obs)
            direct = float(objective_loss(DETERMINISTIC_MSE, mean, std, self.data.train_actions))
        self.assertAlmostEqual(result.selected_diagnostics["train_loss"], direct, places=6)
        self.assertEqual(result.selected_diagnostics["parameters"]["gain"], [float(policy.gain)])

    def test_step_zero_is_probed_and_can_be_selected(self):
        scores = iter([(0.5, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)])
        policy, result = self.fit(probe=lambda p: next(scores))
        self.assertEqual([h["step"] for h in result.checkpoint_history], [0, 10, 20, 30, 40])
        self.assertEqual(result.selected_step, 0)
        self.assertEqual(state_parameters_sha256(result.selected_state), state_parameters_sha256(result.initial_state))
        self.assertEqual(float(policy.gain), 2.0)

    def test_an_exact_tie_selects_the_earlier_checkpoint(self):
        _, result = self.fit(probe=lambda p: (0.0, 0.0))
        self.assertEqual(result.selected_step, 0)
        self.assertEqual(result.objective_config["checkpoint_tie_rule"], CHECKPOINT_TIE_RULE)

    def test_progress_breaks_a_success_tie(self):
        scores = iter([(0.0, -0.1), (0.0, 0.2), (0.0, 0.3), (0.0, 0.3), (0.0, 0.1)])
        _, result = self.fit(probe=lambda p: next(scores))
        self.assertEqual(result.selected_step, 20)

    def test_without_a_probe_the_final_state_is_selected(self):
        policy, result = self.fit()
        self.assertEqual(result.selected_step, 40)
        self.assertEqual(result.checkpoint_history, [])
        self.assertEqual(state_parameters_sha256(result.selected_state), state_parameters_sha256(result.final_state))

    def test_the_three_states_are_distinct_objects(self):
        _, result = self.fit(probe=lambda p: (0.0, 0.0))
        self.assertIsNot(result.initial_state, result.final_state)
        self.assertNotEqual(state_parameters_sha256(result.initial_state), state_parameters_sha256(result.final_state))

    def test_the_selected_state_round_trips_through_serialisation(self):
        scores = iter([(0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (0.0, 0.0), (0.0, 0.0)])
        policy, result = self.fit(probe=lambda p: next(scores))
        buffer = io.BytesIO()
        torch.save(result.selected_state, buffer)
        reloaded = build()
        reloaded.load_state_dict(torch.load(io.BytesIO(buffer.getvalue()), weights_only=True))
        with torch.no_grad():
            self.assertTrue(torch.equal(torch.tanh(policy(self.data.train_obs)[0]),
                                        torch.tanh(reloaded(self.data.train_obs)[0])))

    def test_the_serialised_result_carries_no_tensors(self):
        _, result = self.fit(probe=lambda p: (0.0, 0.0))
        d = result.to_dict()
        for key in ("initial_state", "final_state", "selected_state", "checkpoint_states", "best_rollout_state"):
            self.assertNotIn(key, d)
        self.assertEqual(d["selected_step"], 0)
        self.assertIn("selected_diagnostics", d)

    def test_a_zero_step_budget_disables_fitting_and_keeps_the_initial_state(self):
        policy, result = self.fit(budget=0, steps=None)
        self.assertEqual(result.selected_step, 0)
        self.assertEqual(result.bc_gradient_steps, 0)
        self.assertEqual(state_parameters_sha256(result.selected_state), state_parameters_sha256(result.initial_state))
        self.assertEqual(float(policy.gain), 2.0)

    def test_checkpoint_steps_outside_the_budget_are_refused(self):
        with self.assertRaises(ValueError):
            self.fit(probe=lambda p: (0.0, 0.0), steps=(0, 50))


class TestRolloutScore(unittest.TestCase):
    def test_tuple_scores_compare_lexicographically(self):
        self.assertGreater(rollout_score((0.2, -1.0)), rollout_score((0.1, 5.0)))
        self.assertGreater(rollout_score((0.1, 0.3)), rollout_score((0.1, 0.2)))
        self.assertEqual(rollout_score([0.1, 0.2]), RolloutScore(0.1, 0.2))

    def test_a_bare_number_is_success_without_progress(self):
        self.assertEqual(rollout_score(0.5), RolloutScore(0.5, 0.0))

    def test_only_a_strict_improvement_replaces_the_incumbent(self):
        self.assertTrue(is_better_checkpoint((0.1, 0.0), None))
        self.assertFalse(is_better_checkpoint((0.1, 0.0), (0.1, 0.0)))
        self.assertTrue(is_better_checkpoint((0.1, 0.01), (0.1, 0.0)))

    def test_a_malformed_score_is_refused(self):
        with self.assertRaises(ValueError):
            rollout_score((0.1, 0.2, 0.3))

    def test_the_default_schedule_includes_step_zero(self):
        self.assertEqual(default_checkpoint_steps(2000, 4), (0, 500, 1000, 1500, 2000))
        self.assertEqual(default_checkpoint_steps(2000, 0), ())


class TestCompatibilityWrapper(unittest.TestCase):
    def test_fit_with_objective_still_fits_a_bank_structure_with_a_float_probe(self):
        data = prepare_data(make_buffer(), val_fraction=0.25, split_seed=0)
        structure = next(s for s in load_bank(family="simple") if s.structure_id == "simple_standoff")
        scores = iter([0.1, 0.4, 0.4, 0.2])
        result = fit_with_objective(
            structure, data, DETERMINISTIC_MSE, budget=40, batch_size=16, schema=SCHEMA,
            rollout_probe=lambda p: next(scores), rollout_every=10,
        )
        self.assertEqual(result.best_rollout_step, 20)
        self.assertEqual(result.best_rollout_score, 0.4)
        self.assertEqual(result.selected_step, 20)
        self.assertEqual(result.structure_id, "simple_standoff")
        self.assertIn("log_std", result.frozen_parameters)


if __name__ == "__main__":
    unittest.main()
