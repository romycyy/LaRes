"""R-110: the slow-mode archive audit's classes, gate, grouping and bootstrap (no Isaac)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lares.envs.isaac_pen_spin import slow_mode_audit as sm  # noqa: E402
from lares.utils.cli import load_yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = load_yaml(os.path.join(ROOT, "config", "shadowhand_spin_stages.yaml"))["research_protocols"][sm.STUDY]


class TestAudit(unittest.TestCase):
    def test_labels(self):
        rows = [{"sustained": True, "completed": True, "final_center_dist": 0.02},
                {"sustained": False, "completed": True, "final_center_dist": 0.03},
                {"sustained": False, "completed": True, "final_center_dist": 0.09},
                {"sustained": False, "completed": False, "final_center_dist": 0.31}]
        self.assertEqual(list(sm.labels(rows, 0.08)), [sm.SUCCESS, sm.SLOW, sm.EXCLUDED, sm.EXCLUDED])

    def test_features(self):
        self.assertEqual(sm.reset_features(np.ones((4, 29)) * 2).shape, (4, 58))
        spin = np.tile(np.arange(60.0)[:, None], (1, 3)) * np.array([1.0, -1.0, 0.0])
        f = sm.prefix_features(spin, np.full((60, 3), 0.02), 30)
        np.testing.assert_allclose(f[:, 0], [14.5, 14.5, 0.0])
        np.testing.assert_allclose(f[:, 1], [14.5, -14.5, 0.0])
        np.testing.assert_allclose(f[:, 2], [29.0, 29.0, 0.0])

    def _data(self, signal, n=1800, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.uniform(-1, 1, (n, 29))
        y = (rng.random(n) < 1 / (1 + np.exp(-signal * (X[:, 3] + X[:, 7] ** 2 - 1 / 3)))).astype(int)
        return sm.reset_features(X), y, np.repeat(np.arange(36), n // 36)

    def test_gate_finds_a_real_dependence_and_not_a_null(self):
        cfg = {**CFG, "resamples": 200}
        strong = sm.gate(*self._data(6.0), cfg)
        self.assertGreater(strong["advantage"], 0.25)
        self.assertEqual(sm.verdict(strong["advantage_ci95"], 0.15), "supported")
        null = sm.gate(*self._data(0.0), cfg)
        self.assertLess(abs(null["advantage"]), 0.06)
        self.assertEqual(sm.verdict(null["advantage_ci95"], 0.15), "refuted")
        self.assertTrue(strong["advantage_ci95"][0] <= strong["advantage"] <= strong["advantage_ci95"][1])

    def test_predictions_are_held_out_by_manifest_group(self):
        X, y, g = self._data(3.0, n=360)
        base = sm.held_out_probabilities(X, y, g, 6, 1.0)
        flipped = y.copy()
        fold0 = g % 6 == 0
        flipped[fold0] = 1 - flipped[fold0]                                       # change only fold 0's own labels
        again = sm.held_out_probabilities(X, flipped, g, 6, 1.0)
        np.testing.assert_array_equal(base[fold0], again[fold0])                  # its predictions never saw them
        self.assertFalse(np.array_equal(base[~fold0], again[~fold0]))

    def test_metrics(self):
        y = np.array([1, 1, 0, 0])
        self.assertEqual(sm.balanced_accuracy(y, np.array([0.9, 0.2, 0.1, 0.6])), 0.5)
        self.assertEqual(sm.auc(y, np.array([0.9, 0.8, 0.1, 0.2])), 1.0)
        self.assertEqual(sm.auc(y, np.array([0.5, 0.5, 0.5, 0.5])), 0.5)
        a, b = (sm.bootstrap_manifests(np.array([1, 0, 1, 0]), np.array([0.9, 0.2, 0.1, 0.6]), np.array([0, 0, 1, 1]), 50, 3) for _ in range(2))
        self.assertEqual(a, b)

    def test_registered_and_no_controller(self):
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import run_shadowhand_spin_stages as rs

        self.assertEqual(rs.PEN_SPIN_STUDIES[sm.STUDY], "slow_mode_audit")
        with self.assertRaises(SystemExit):
            sm.make_controller("x", CFG)
        self.assertEqual((CFG["folds"], CFG["penalty"], CFG["seed"], CFG["threshold"]), (6, 1.0, 4892, 0.15))


if __name__ == "__main__":
    unittest.main()
