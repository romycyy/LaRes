"""R-106: the exact-success ARS study's placements, slot layout, A/A statistics and frozen files (no Isaac)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lares.envs.isaac_pen_spin import success_ars as sa  # noqa: E402
from lares.envs.isaac_pen_spin import symbolic_reward as sr  # noqa: E402
from lares.utils.cli import load_yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL = load_yaml(os.path.join(ROOT, "config", "shadowhand_spin_stages.yaml"))["research_protocols"]
CFG = ALL[sa.STUDY]


class TestLayout(unittest.TestCase):
    def test_layout_and_budget(self):
        lay = sa.layout(CFG)
        self.assertEqual((lay["slots"], lay["per"], lay["num_envs"], lay["rollouts"], lay["updates"]), (4, 512, 2048, 2, 12))
        est = 2 * lay["updates"] * lay["rollouts"] * lay["num_envs"] + 2 * 2 * 2048 + 2 * 2 * 12 * 128
        self.assertEqual(est, 112640)
        self.assertLessEqual(est + CFG["prior_starts"], CFG["start_ceiling"])
        with self.assertRaises(SystemExit):
            sa.layout({**CFG, "training": {**CFG["training"], "rollouts_per_update": 3}})

    def test_training_placements_are_disjoint_slices(self):
        a, b = sa.training_placements(CFG, 0), sa.training_placements(CFG, 11)
        self.assertEqual(a.shape, (512, 29))
        self.assertFalse(set(sa.placement_hashes(a)) & set(sa.placement_hashes(b)))
        np.testing.assert_array_equal(a, sa.training_placements(CFG, 0))

    def test_slots_share_placements_and_hold_one_candidate_each(self):
        v = sa.tile(np.arange(6.0).reshape(3, 2), 4)
        self.assertEqual(v.shape, (12, 2))
        np.testing.assert_array_equal(v[0:3], v[9:12])
        th = sa.slot_thetas(np.arange(8.0).reshape(4, 2), 3)
        np.testing.assert_array_equal(th[3:6], np.tile([2.0, 3.0], (3, 1)))

    def test_candidates_round_trip_through_the_rollouts(self):
        lay = sa.layout(CFG)
        ars = sr.ARS(CFG["training"]["seed"], CFG["ars"], 0.10)
        cand = ars.ask(0)                                    # +d0..+d3, -d0..-d3
        r0, r1 = sa.rollout_candidates(cand, 0, lay), sa.rollout_candidates(cand, 1, lay)
        np.testing.assert_array_equal(r0, cand[[0, 4, 1, 5]])
        np.testing.assert_array_equal(r1, cand[[2, 6, 3, 7]])
        np.testing.assert_allclose(r0[0] + r0[1], 0.0, atol=1e-12)                    # an antithetic pair about theta = 0
        scores = sa.ars_scores([np.array([0.1, 0.2, 0.3, 0.4]), np.array([0.5, 0.6, 0.7, 0.8])], lay)
        np.testing.assert_allclose(scores, [0.1, 0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8])

    def test_ars_follows_a_success_count(self):
        lay = sa.layout(CFG)
        ars = sr.ARS(1, CFG["ars"], 0.10)
        w = np.zeros(32)
        w[0] = 1.0
        for u in range(12):
            cand = ars.ask(u)
            slot = [np.array([0.4 + 2.0 * c[0] for c in sa.rollout_candidates(cand, r, lay)]) for r in range(2)]
            ars.tell(u, sa.ars_scores(slot, lay))
        self.assertGreater(ars.theta[0], 0.02)               # twelve updates move a coefficient well off zero
        self.assertLessEqual(np.abs(ars.theta).max(), 0.10)

    def test_scorer_equals_the_evaluators_reducer(self):
        from lares.envs.isaac_pen_spin import cases

        T, n = 360, 6
        pred = np.ones((T, n), dtype=bool)
        length = np.array([359, 359, 359, 120, 359, 40])
        completed = np.array([True, True, True, False, False, False])
        pred[358, 1] = False                                  # fails on the last recorded step
        pred[299, 2] = False                                  # the step just before the final window of a 359-step episode
        pred[300:, 3] = False
        for i, L in enumerate(length):
            pred[L:, i] = False                               # nothing is recorded after an episode ends
        got = sa.first_episode_scores(pred, length, completed)
        trace = {"case_ids": list(range(n)), "task_hashes": list(range(n)), "length": length, "completed": completed, "pred": pred,
                 "spin_rate": np.ones((T, n)), "transverse_rate": np.ones((T, n)), "center_dist": np.zeros((T, n)),
                 "reward": np.zeros((T, n))}
        rows = cases.reduce_episodes(trace)
        self.assertEqual(list(got["sustained"]), [r["sustained"] for r in rows])
        np.testing.assert_allclose(got["tail_hit"], [r["tail_hit_fraction"] for r in rows])
        self.assertEqual(list(got["sustained"]), [True, False, False, False, False, False])
        pred[298, 2] = False
        self.assertTrue(sa.first_episode_scores(pred, length, completed)["sustained"][0])
        pred[299, 2], pred[298, 2] = True, False              # outside the final 60 of a 359-step episode
        self.assertTrue(sa.first_episode_scores(pred, length, completed)["sustained"][2])

    def test_aa_statistics(self):
        s = np.zeros((4, 8), dtype=bool)
        s[0, :4] = s[1, :4] = s[2, :4] = True
        s[3, 1:5] = True
        aa = sa.aa_statistics(s)
        self.assertEqual(aa["slot_success"], [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(aa["pairwise_discordance"], [0.0, 0.0, 0.25, 0.0, 0.25, 0.25])
        self.assertEqual(aa["max_abs_slot_difference"], 0.0)


class TestFrozenFiles(unittest.TestCase):
    def test_family_is_r103s(self):
        r103 = ALL[sr.STUDY]
        self.assertEqual(CFG["actor"], r103["actor"])
        self.assertEqual(CFG["reference_residual"], r103["reference_residual"])
        cwd = os.getcwd()
        os.chdir(ROOT)
        try:
            self.assertEqual(len(sa.normalization(CFG)["rms_velocity"]), 20)
            self.assertEqual(sr.check_reference(CFG)["residual_params_sha256"], CFG["reference_residual_sha256"])
            self.assertEqual(sa.build_manifests(CFG), [])
            prov = sa.placement_provenance(CFG)
            self.assertEqual((prov["training_cases"], prov["smoke_cases"]), (6144, 512))
        finally:
            os.chdir(cwd)

    def test_r103_config_hash_is_unchanged_by_the_alias(self):
        self.assertEqual(sr.config_sha256(ALL[sr.STUDY])[:12], "61f9824fc694")

    def test_registry_and_controllers(self):
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import run_shadowhand_spin_stages as rs

        self.assertEqual(rs.PEN_SPIN_STUDIES[sa.STUDY], "success_ars")
        with self.assertRaises(ValueError):
            sa.make_controller("draw3", CFG)


if __name__ == "__main__":
    unittest.main()
