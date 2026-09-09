#!/usr/bin/env python
"""Tests for manifest replay, RNG isolation and paired comparison (AC-0).

Run from the project root::

    python -m unittest tests.test_manifest_runner
"""

import os
import sys
import unittest

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from lares.core.symbolic_policy import SymbolicPolicy  # noqa: E402
from lares.eval.manifest import (  # noqa: E402
    SPLIT_DEVELOPMENT,
    build_manifest,
    synthetic_manifest,
)
from lares.eval.runner import (  # noqa: E402
    ACTION_MODE_DETERMINISTIC,
    ACTION_MODE_SAMPLED,
    ConstantActor,
    ExpertActor,
    PipelineEnvs,
    PolicyActor,
    evaluate_manifest,
    make_manifest_env,
    paired_difference,
    run_episode,
)

try:
    import metaworld  # noqa: F401

    HAS_METAWORLD = True
except ImportError:
    HAS_METAWORLD = False

ENV_ID = "push-v2"
OBS_DIM = 39
ACT_DIM = 4


# ---------------------------------------------------------------------------
#  Mocks
# ---------------------------------------------------------------------------


class _Space:
    def __init__(self, dim):
        self.shape = (dim,)
        self.high = np.ones(dim, dtype=np.float32)
        self.low = -np.ones(dim, dtype=np.float32)


class SeededMockEnv:
    """Mock whose whole episode is a function of ``case.reset_seed``.

    Makes the isolation property testable without a simulator: if a reset ever
    picked up hidden state, the returns would move.
    """

    def __init__(self, obs_dim=OBS_DIM, action_dim=ACT_DIM, episode_length=5):
        self.observation_space = _Space(obs_dim)
        self.action_space = _Space(action_dim)
        self.obs_dim = obs_dim
        self._episode_length = episode_length
        self._rng = None
        self._step = 0

    def reset(self, case=None):
        if case is None:
            raise ValueError("case required")
        self._rng = np.random.default_rng(int(case.reset_seed))
        self._step = 0
        return self._rng.standard_normal(self.obs_dim).astype(np.float32), {}

    def step(self, action):
        self._step += 1
        obs = self._rng.standard_normal(self.obs_dim).astype(np.float32)
        # Reward depends on the action as well as the seeded stream, so a change
        # of policy shows up in the return while the case still fixes the episode.
        reward = float(self._rng.standard_normal() + np.sum(action))
        done = self._step >= self._episode_length
        info = {"success": 1.0 if done and reward > 0 else 0.0, "obj_to_target": abs(reward)}
        return obs, reward, done, info


class TinyPolicy(SymbolicPolicy):
    def __init__(self, obs_dim, action_dim, bias=0.0):
        super().__init__(obs_dim, action_dim)
        self.w = nn.Parameter(torch.tensor(0.5))
        self.b = nn.Parameter(torch.full((action_dim,), float(bias)))
        self.log_std = nn.Parameter(torch.tensor(-1.0))

    def forward(self, obs):
        move = self.w * obs[:, : self.action_dim] + self.b
        std = torch.exp(self.log_std) * torch.ones_like(move)
        return move, std

    def get_param_ranges(self):
        return {"w": (0.0, 5.0), "b": (-2.0, 2.0), "log_std": (-5.0, 0.0)}


# ---------------------------------------------------------------------------
#  Mock-level tests
# ---------------------------------------------------------------------------


class TestRunnerWithMocks(unittest.TestCase):
    def setUp(self):
        self.manifest = synthetic_manifest(6, label="runner", action_dim=ACT_DIM)
        self.env = SeededMockEnv()

    def test_replay_follows_manifest_order(self):
        result = evaluate_manifest(
            ConstantActor(np.zeros(ACT_DIM)), self.manifest, self.env, max_steps=5
        )
        self.assertEqual(
            [e.case_id for e in result.episodes],
            [c.case_id for c in self.manifest.episodes],
        )

    def test_same_manifest_replays_identically(self):
        actor = ConstantActor(np.zeros(ACT_DIM))
        a = evaluate_manifest(actor, self.manifest, self.env, max_steps=5)
        b = evaluate_manifest(actor, self.manifest, self.env, max_steps=5)
        self.assertEqual(
            [e.episode_return for e in a.episodes], [e.episode_return for e in b.episodes]
        )

    def test_intervening_work_does_not_shift_later_cases(self):
        actor = ConstantActor(np.zeros(ACT_DIM))
        before = evaluate_manifest(actor, self.manifest, self.env, max_steps=5)
        # Simulate dataset collection and a debug rollout between evaluations.
        for case in synthetic_manifest(4, label="noise").episodes:
            run_episode(ConstantActor(np.ones(ACT_DIM)), self.env, case, max_steps=5)
        after = evaluate_manifest(actor, self.manifest, self.env, max_steps=5)
        self.assertEqual(
            [e.episode_return for e in before.episodes],
            [e.episode_return for e in after.episodes],
        )

    def test_two_policies_receive_the_same_cases(self):
        a = evaluate_manifest(
            PolicyActor(TinyPolicy(OBS_DIM, ACT_DIM, 0.0), name="a"),
            self.manifest,
            self.env,
            max_steps=5,
        )
        b = evaluate_manifest(
            PolicyActor(TinyPolicy(OBS_DIM, ACT_DIM, 0.4), name="b"),
            self.manifest,
            self.env,
            max_steps=5,
        )
        self.assertEqual(
            [e.task_id for e in a.episodes], [e.task_id for e in b.episodes]
        )
        diff = paired_difference(a, b, "reward")
        self.assertEqual(diff["n_paired"], len(self.manifest))

    def test_paired_difference_refuses_unpaired_results(self):
        a = evaluate_manifest(
            ConstantActor(np.zeros(ACT_DIM)), self.manifest, self.env, max_steps=5
        )
        other = synthetic_manifest(3, label="other")
        b = evaluate_manifest(ConstantActor(np.zeros(ACT_DIM)), other, self.env, max_steps=5)
        with self.assertRaises(ValueError):
            paired_difference(a, b, "success")

    def test_sampled_actor_is_reproducible_and_differs_from_deterministic(self):
        policy = TinyPolicy(OBS_DIM, ACT_DIM, 0.3)
        sampled = PolicyActor(policy, deterministic=False, name="sampled")
        self.assertEqual(sampled.action_mode, ACTION_MODE_SAMPLED)
        a = evaluate_manifest(sampled, self.manifest, self.env, max_steps=5)
        b = evaluate_manifest(sampled, self.manifest, self.env, max_steps=5)
        self.assertEqual(
            [e.episode_return for e in a.episodes], [e.episode_return for e in b.episodes]
        )
        det = PolicyActor(policy, deterministic=True, name="det")
        self.assertEqual(det.action_mode, ACTION_MODE_DETERMINISTIC)
        d = evaluate_manifest(det, self.manifest, self.env, max_steps=5)
        self.assertNotEqual(
            [e.episode_return for e in a.episodes], [e.episode_return for e in d.episodes]
        )

    def test_episode_record_carries_the_case_identity(self):
        case = self.manifest.episodes[2]
        rec = run_episode(ConstantActor(np.zeros(ACT_DIM)), self.env, case, max_steps=5)
        self.assertEqual(rec.case_id, case.case_id)
        self.assertEqual(rec.task_id, case.task_id)
        self.assertIsNotNone(rec.terminal_obj_to_target)

    def test_pipeline_envs_rejects_a_shared_instance(self):
        env = SeededMockEnv()
        with self.assertRaises(ValueError):
            PipelineEnvs(env, env, env)
        PipelineEnvs(env, env, env, allow_shared=True)  # explicit opt-in for mocks


# ---------------------------------------------------------------------------
#  Simulator-level tests
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAS_METAWORLD, "metaworld is required for simulator replay tests")
class TestRunnerWithMetaWorld(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("MUJOCO_GL", "egl")
        cls.manifest, cls.pool = build_manifest(
            ENV_ID, SPLIT_DEVELOPMENT, [2000], horizon=150, num_episodes=3, tasks_per_seed=3
        )
        cls.env = make_manifest_env(cls.manifest, cls.pool, episode_length=200)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_expert_replay_is_bit_identical(self):
        actor = ExpertActor(ENV_ID)
        a = evaluate_manifest(actor, self.manifest, self.env, max_steps=60)
        b = evaluate_manifest(actor, self.manifest, self.env, max_steps=60)
        for x, y in zip(a.episodes, b.episodes):
            self.assertEqual(x.episode_return, y.episode_return)
            self.assertEqual(x.terminal_obj_to_target, y.terminal_obj_to_target)

    def test_reset_without_a_case_is_refused(self):
        with self.assertRaises(ValueError):
            self.env.reset()

    def test_reset_with_an_unknown_task_is_refused(self):
        from lares.eval.manifest import EpisodeCase

        bogus = EpisodeCase("x-0000", "MT1:push-v3:s9999:000", "h", 1, 2)
        with self.assertRaises(KeyError):
            self.env.reset(bogus)

    def test_case_order_does_not_change_an_episode(self):
        actor = ExpertActor(ENV_ID)
        forward = evaluate_manifest(actor, self.manifest, self.env, max_steps=60)
        reversed_manifest = self.manifest.head(len(self.manifest))
        reversed_manifest.episodes = list(reversed(reversed_manifest.episodes))
        backward = evaluate_manifest(actor, reversed_manifest, self.env, max_steps=60)
        by_case = {e.case_id: e.episode_return for e in backward.episodes}
        for e in forward.episodes:
            self.assertEqual(e.episode_return, by_case[e.case_id])


if __name__ == "__main__":
    unittest.main()
