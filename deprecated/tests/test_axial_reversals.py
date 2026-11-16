#!/usr/bin/env python
"""DEPRECATED: R-071 axial-reversal tests, cut from tests/test_pen_spin.py.

Archived 2026-11-09 with lares/envs/isaac_pen_spin/axial_reversals.py and scripts/isaac/axial_reversals.py (R-071 decided). Kept for reference, not run: its imports name the module's original location, so restore the
module first (see deprecated/README.md).
"""

import os
import sys
import unittest

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)


class TestAxialReversals(unittest.TestCase):
    """R-071: held directions, reversals and the paired rate."""

    def test_held_direction_and_reversals(self):
        from lares.envs.isaac_pen_spin.axial_reversals import held_directions, reversals

        spin = [0.5, 2, 2, 2, 0.2, -2, -2, -2, 2, -2, 2, 2, 2]
        self.assertEqual(held_directions(spin, 1.0, 3).tolist(), [0, 0, 0, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1])
        self.assertEqual(reversals(spin, 1.0, 3), 2)
        self.assertEqual(reversals([5.0] * 100, 1.0, 3), 0)
        self.assertEqual(reversals([0.0] * 100, 1.0, 3), 0)

    def test_pair_rates_use_the_common_window(self):
        from lares.envs.isaac_pen_spin.axial_reversals import CONTROL_HZ, pair_rates

        T = 120
        a = {"length": np.array([120, 60]), "spin_rate": np.zeros((T, 2))}
        b = {"length": np.array([90, 120]), "spin_rate": np.full((T, 2), 5.0)}
        a["spin_rate"][:, 0] = [5.0 if (t // 10) % 2 == 0 else -5.0 for t in range(T)]   # reversal every 10 steps
        r = pair_rates(a, b, 1.0, 3)
        self.assertEqual(r["window_steps"], [90, 60])
        self.assertAlmostEqual(r["rate"]["a"][0], 8 / (90 / CONTROL_HZ))
        self.assertEqual(r["rate"]["b"], [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
