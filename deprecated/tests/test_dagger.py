#!/usr/bin/env python
"""DEPRECATED: learner-state imitation (``lares.fitting.dagger``) tests, cut from tests/test_objectives.py.

Archived 2026-11-09 with lares/fitting/dagger.py, whose only caller was scripts/compare_objectives.py (Phase 5). Kept for reference, not run: its imports name the module's original location, so restore the
module first (see deprecated/README.md).
"""

import os
import sys
import unittest

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from lares.core.obs_schema import get_obs_schema  # noqa: E402
from lares.fitting.dagger import validate_expert_label  # noqa: E402

SCHEMA = get_obs_schema("push-v2")

try:
    import metaworld  # noqa: F401

    HAS_METAWORLD = True
except ImportError:
    HAS_METAWORLD = False


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


if __name__ == "__main__":
    unittest.main()
