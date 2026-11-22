"""R-114: the recovery diagnostic's schedule, overrides and statistics (no Isaac)."""

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lares.envs.isaac_pen_spin import recovery_authority as ra  # noqa: E402
from lares.envs.isaac_pen_spin import symbolic_reward as sr  # noqa: E402
from lares.utils.cli import load_yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
CFG = load_yaml(os.path.join(ROOT, "config", "shadowhand_spin_stages.yaml"))["research_protocols"][ra.STUDY]


class TestSchedule(unittest.TestCase):
    def test_slots_rotate_and_every_role_visits_every_slot(self):
        seen = [ra.slot_roles(b) for b in range(4)]
        self.assertEqual(seen[0], list(ra.ROLES))
        for k in range(4):
            self.assertEqual({s[k] for s in seen}, set(ra.ROLES))

    def test_beta_ramps_in_and_out_and_is_zero_outside(self):
        self.assertEqual([ra.beta(t, CFG) for t in (0, 119, 180, 300)], [0.0] * 4)
        self.assertAlmostEqual(ra.beta(120, CFG), 1 / 20)
        self.assertAlmostEqual(ra.beta(139, CFG), 1.0)
        self.assertAlmostEqual(ra.beta(140, CFG), 1.0)
        self.assertAlmostEqual(ra.beta(179, CFG), 1 / 40)
        self.assertTrue(all(0 < ra.beta(t, CFG) <= 1 for t in range(120, 180)))

    def test_eligibility(self):
        e = ra.eligible(torch.tensor([True, True, True, False]), torch.tensor([0.02, 0.09, 0.02, 0.02]),
                        torch.tensor([1.0, 1.0, 5.0, 1.0]), CFG)
        self.assertEqual(e.tolist(), [True, False, False, False])

    def test_overrides_touch_only_eligible_probe_environments_inside_the_window(self):
        n = 8
        role_of = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        elig = torch.tensor([True, True, True, True, False, False, False, False])
        a = torch.full((n, 20), 0.5)
        ref = torch.full((n, 20), 0.45)
        parts = {"s": torch.ones(n, 1), "vhat": torch.zeros(n, 20), "ehat": torch.zeros(n, 20)}
        theta = torch.zeros(32)
        theta[:20] = 0.05                                    # residual 0.05 -> x3 gives 0.15 (cap 0.30)
        neutral = torch.zeros(n, 20)
        for t in (0, 119, 180, 359):                         # outside the window nothing changes: the same tensor
            self.assertIs(ra.sent_action(a, ref, parts, theta, role_of, elig, t, neutral, CFG, 0.10), a)
        out = ra.sent_action(a, ref, parts, theta, role_of, elig, 139, neutral, CFG, 0.10)
        self.assertTrue(torch.equal(out[[0, 2, 4, 5, 6, 7]], a[[0, 2, 4, 5, 6, 7]]))      # control, stroke slot, ineligible
        torch.testing.assert_close(out[1], torch.zeros(20))                               # fully at the reset posture
        torch.testing.assert_close(out[3], torch.full((20,), 0.45 + 0.15))
        half = ra.sent_action(a, ref, parts, theta, role_of, elig, 129, neutral, CFG, 0.10)
        torch.testing.assert_close(half[1], torch.full((20,), 0.25))

    def test_per_env_gain_at_scale_one_is_the_constant(self):
        g = ra.PerEnvGain(1.4081491479357957, 5)
        spin = torch.randn(5, 1)
        self.assertTrue(torch.equal(g(spin), torch.full_like(spin, 1.4081491479357957)))
        g.scale[2] = 1.5
        self.assertAlmostEqual(float(g(spin)[2]), 1.4081491479357957 * 1.5, places=5)


def fake_batches(recapture_rate, per=512, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for b in range(4):
        elig = rng.random(per) < 0.31
        arr = {k: np.zeros(4 * per, dtype=int) for k in ("eligible", "sustained", "completed")}
        arr["tail_abs_spin"] = np.zeros(4 * per)
        for slot, role in enumerate(ra.slot_roles(b)):
            sl = slice(slot * per, (slot + 1) * per)
            e = np.where(rng.random(per) < 0.05, ~elig, elig)                    # slots agree on about 95% of placements
            sus = ~e & (rng.random(per) < 0.75)
            if role == "recapture":
                sus = sus | (e & (rng.random(per) < recapture_rate))
            arr["eligible"][sl], arr["sustained"][sl], arr["completed"][sl] = e, sus, 1
        out.append({k: v.tolist() for k, v in arr.items()})
    return out


class TestStatistics(unittest.TestCase):
    def test_a_working_probe_is_detected_and_a_null_is_refuted(self):
        s = ra.probe_statistics(fake_batches(0.30), 512, CFG)
        self.assertGreater(s["recapture"]["pairs"], 300)
        self.assertGreater(s["recapture"]["ci95"][0], 0.05)
        self.assertTrue(s["recapture"]["verdict"]["useful_scripted_authority"])
        self.assertFalse(s["stroke_boost"]["verdict"]["positive_case_paired"])
        self.assertTrue(s["stroke_boost"]["verdict"]["magnitude_refuted"])
        null = ra.probe_statistics(fake_batches(0.0), 512, CFG)
        self.assertTrue(null["recapture"]["verdict"]["magnitude_refuted"])
        self.assertFalse(null["recapture"]["verdict"]["useful_scripted_authority"])

    def test_control_and_agreement(self):
        b = fake_batches(0.0)
        c = ra.control_summary(b, 512)
        self.assertEqual(c["cases"], 2048)
        self.assertEqual(c["untreated_conversion"], 0.0)
        self.assertGreater(ra.eligibility_agreement(b, 512), 0.85)

    def test_frozen_files(self):
        self.assertEqual(len(ra.finalist_theta(CFG)), 32)
        prov = ra.placement_provenance(CFG)
        self.assertEqual(sorted(prov["batches"]), ["4907", "4923", "4924", "4925"])
        self.assertEqual(sr.check_reference(CFG)["residual_params_sha256"], CFG["reference_residual_sha256"])
        self.assertLessEqual(4 * 2 * 2048 + 4 * 300, CFG["start_ceiling"])


if __name__ == "__main__":
    unittest.main()
