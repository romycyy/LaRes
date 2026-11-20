"""The settled-window contact gate's paired statistics (pure functions; no simulator)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lares.envs.isaac_pen_spin import contact_control as cc  # noqa: E402
from lares.envs.isaac_pen_spin import contact_settled_gate as sg  # noqa: E402

TH = {"expected_contrasts": 15, "min_eligible_pairs": 20, "paired_sign_agreement_min": 0.8,
      "paired_gain_median_min": 0.3, "paired_gain_median_max": 1.5, "jacobian_rel_error_median_max": 0.15,
      "retention_diff_min": -0.10}


def cell(gain, eligible=None):
    gain = list(gain)
    return {"gain": gain, "eligible_mask": [True] * len(gain) if eligible is None else list(eligible)}


def contrasts(true_gain=0.5, common=5.0, n=40, eligible=None):
    """Every contrast: the request moves the tip by ``true_gain``; a common motion adds ``common`` to both cells."""
    return {sg.contrast_name(k, a): sg.paired_contrast(cell([true_gain + common] * n, eligible),
                                                      cell([true_gain - common] * n, eligible))
            for k, a in sg.CONTRASTS}


class TestPairedContrast(unittest.TestCase):
    def test_common_motion_cancels(self):
        c = sg.paired_contrast(cell([5.5, 6.5]), cell([-4.5, -5.5]))
        self.assertEqual(c["d"], [0.5, 0.5])
        self.assertEqual(c["sign_agreement"], 1.0)
        self.assertAlmostEqual(c["common_mode_median"], 5.5)

    def test_pair_needs_both_cells_eligible(self):
        c = sg.paired_contrast(cell([1.0, 1.0, 1.0], [True, False, True]), cell([1.0, 1.0, 1.0], [True, True, False]))
        self.assertEqual(c["eligible_pairs"], 1)
        self.assertEqual(c["pair_mask"], [True, False, False])

    def test_ineligible_nan_is_ignored_and_eligible_nan_refused(self):
        c = sg.paired_contrast(cell([None, 1.0], [False, True]), cell([1.0, 1.0]))
        self.assertEqual(c["eligible_pairs"], 1)
        with self.assertRaises(ValueError):
            sg.paired_contrast(cell([None, 1.0]), cell([1.0, 1.0]))

    def test_no_pairs(self):
        c = sg.paired_contrast(cell([1.0], [False]), cell([1.0]))
        self.assertIsNone(c["sign_agreement"])
        self.assertIsNone(c["gain_median"])


class TestPairedVerdict(unittest.TestCase):
    def test_pass_despite_large_common_motion(self):
        v = sg.paired_verdict(contrasts(), [0.01, 0.02], 0.8, 0.0, TH)
        self.assertEqual(v["verdict"], cc.PASS)
        self.assertEqual(v["eligible_pairs"], 600)

    def test_wrong_direction_fails(self):
        self.assertEqual(sg.paired_verdict(contrasts(true_gain=-0.5), [0.01], 0.8, 0.0, TH)["verdict"], cc.FAIL)

    def test_low_gain_fails(self):
        self.assertEqual(sg.paired_verdict(contrasts(true_gain=0.1), [0.01], 0.8, 0.0, TH)["verdict"], cc.FAIL)

    def test_coverage_is_inconclusive(self):
        few = [True] * 19 + [False] * 21
        v = sg.paired_verdict(contrasts(eligible=few), [0.01], 0.8, 0.0, TH)
        self.assertEqual(v["verdict"], cc.INCONCLUSIVE)
        self.assertEqual(len(v["checks"]["coverage"]["value"]["under_minimum"]), 15)

    def test_missing_contrast_is_inconclusive(self):
        c = contrasts()
        c.pop("th_axial")
        self.assertEqual(sg.paired_verdict(c, [0.01], 0.8, 0.0, TH)["verdict"], cc.INCONCLUSIVE)

    def test_jacobian_or_retention_failure_fails_whatever_the_coverage(self):
        few = [True] * 5 + [False] * 35
        self.assertEqual(sg.paired_verdict(contrasts(eligible=few), [0.5], 0.8, 0.0, TH)["verdict"], cc.FAIL)
        self.assertEqual(sg.paired_verdict(contrasts(), [0.01], 0.5, 0.9, TH)["verdict"], cc.FAIL)

    def test_missing_measurements_are_inconclusive(self):
        self.assertEqual(sg.paired_verdict(contrasts(), [], 0.8, 0.0, TH)["verdict"], cc.INCONCLUSIVE)
        self.assertEqual(sg.paired_verdict(contrasts(), [0.01], None, 0.0, TH)["verdict"], cc.INCONCLUSIVE)

    def test_thresholds_must_be_complete(self):
        with self.assertRaises(ValueError):
            sg.paired_verdict(contrasts(), [0.01], 0.8, 0.0, {k: v for k, v in TH.items() if k != "min_eligible_pairs"})

    def test_verdict_is_json(self):
        import json

        json.dumps(sg.paired_verdict(contrasts(), np.array([0.01]), 0.8, 0.0, TH))


if __name__ == "__main__":
    unittest.main()
