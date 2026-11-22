"""R-101's transfer audit: the progress, direction and authority statistics, the coherent controller, the config."""

import json
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lares.envs.isaac_pen_spin import contact_control as cc  # noqa: E402
from lares.envs.isaac_pen_spin import rolling_decoder_gate as rg  # noqa: E402
from lares.envs.isaac_pen_spin import transfer_audit as ta  # noqa: E402
from lares.utils.cli import load_yaml  # noqa: E402
from tests.test_rolling_decoder_gate import ACTUATED, FROZEN, SIM_JOINTS, Incumbent, fake_ctx, synthetic_obs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TH = {"min_eligible_cases": 20, "positive_fraction_min": 0.8, "median_ratio_min": 0.3}


def protocol_cfg() -> dict:
    return load_yaml(os.path.join(ROOT, "config", "shadowhand_spin_stages.yaml"))["research_protocols"]["transfer_audit"]


class TestStatistics(unittest.TestCase):
    def test_progress_counts_the_original_episode_only(self):
        spin = np.full((20, 3), 2.0)
        spin[12:, 1] = 99.0                                   # case 1 ended at 12: what follows is an auto-reset
        p = ta.roll_progress(spin, [20, 12, 5], 10, 16, dt=0.5)
        np.testing.assert_allclose(p, [6.0, 2.0, 0.0])

    def test_direction_and_eligibility_read_the_window(self):
        spin = np.stack([np.full(30, 1.0), np.full(30, -1.0), np.full(30, 0.1), np.full(30, 1.0)], axis=1)
        d, mean, e = ta.case_direction(spin, [30, 30, 30, 15], (10, 20), 0.25)
        self.assertEqual(d.tolist(), [1.0, -1.0, 1.0, 0.0])
        self.assertEqual(e.tolist(), [True, True, False, False])     # too slow; ended before the window closed

    def test_authority_needs_sign_interval_and_size(self):
        e = np.ones(30, dtype=bool)
        good = ta.authority(np.full(30, 0.01) + np.linspace(-1e-3, 1e-3, 30), e, 0.02, TH)
        self.assertTrue(good["adequate"])
        self.assertAlmostEqual(good["median_ratio"], 0.5, places=2)
        self.assertFalse(ta.authority(np.full(30, 0.002) + np.linspace(-1e-4, 1e-4, 30), e, 0.02, TH)["adequate"])
        mixed = np.where(np.arange(30) % 3 == 0, -0.01, 0.01)
        self.assertFalse(ta.authority(mixed, e, 0.02, TH)["adequate"])
        self.assertFalse(ta.authority(np.full(30, 0.01), np.zeros(30, dtype=bool), 0.02, TH)["adequate"])


class TestController(unittest.TestCase):
    def run_ctrl(self, directions, steps=9, n=3):
        ctrl = ta.CoherentPulseController("x", FROZEN, directions, inner=Incumbent())
        ctx, obs = fake_ctx(n), synthetic_obs(n)
        ctrl.reset(ctx)
        return ctrl, [ctrl(obs, t, ctx) for t in range(steps)]

    def test_zero_request_is_the_incumbent(self):
        ctrl, actions = self.run_ctrl(None)
        for a in actions:
            self.assertTrue(torch.equal(a, torch.full((3, 20), 0.1)))
        self.assertEqual(float(ctrl.extra_trace()["identity_max_abs"]), 0.0)

    def test_both_fingers_move_and_nothing_else(self):
        d = [[(0, ta.CIRC, 1.0), (2, ta.CIRC, 1.0)]] * 3
        ctrl, pos = self.run_ctrl(d)
        _, neg = self.run_ctrl([[(0, ta.CIRC, -1.0), (2, ta.CIRC, -1.0)]] * 3)
        base = torch.full((3, 20), 0.1)
        for t in (0, 1, 2, 6, 7, 8):
            self.assertTrue(torch.equal(pos[t], base), t)
        torch.testing.assert_close(pos[4] - base, -(neg[4] - base), atol=1e-6, rtol=0)
        names = [SIM_JOINTS[j] for j in ACTUATED]
        moved = {names[i][:2] for i in torch.nonzero((pos[4] - base).abs().sum(0) > 0).flatten().tolist()}
        self.assertEqual(moved, {"FF", "RF"})
        extra = ctrl.extra_trace()
        self.assertEqual(extra["audit_feasible_tips"].shape, (3, 3, 5, 3))
        for key in ta.BOUNDS + ("incumbent_saturated",):
            self.assertEqual(extra[f"audit_{key}"].shape[:2], (3, 3))
        self.assertEqual(extra["audit_q"].shape[0], 3)

    def test_single_direction_form_still_works(self):
        ctrl = rg.RollingPulseController("x", FROZEN, [(0, 0, 1.0)] * 2, inner=Incumbent())
        ctx = fake_ctx(2)
        ctrl.reset(ctx)
        self.assertEqual(ctrl.commanded.sum().item(), 2)
        self.assertEqual(ctrl.tip.tolist(), [0, 0])

    def test_chain_fractions_by_type(self):
        ctrl, _ = self.run_ctrl([[(0, ta.CIRC, 1.0), (2, ta.CIRC, 1.0)]] * 3)
        extra = ctrl.extra_trace()
        extra["audit_cap"][:] = False
        ff = [i for i, j in enumerate(ctrl.ctrl.tolist()) if "FF" in SIM_JOINTS[j]]
        extra["audit_cap"][0, :, ff[0]] = True               # the first pulse step, forefinger only
        out = ta.chain_fractions(extra, [0, 2], [300, 300, 300], 3)
        self.assertAlmostEqual(out["ff"]["cap"], 1 / 3)
        self.assertEqual(out["rf"]["cap"], 0.0)


def write_cell(path, obs, spin, length, extra):
    C = obs.shape[0]
    rows = [{"sustained": False, "completed": bool(length[i] >= spin.shape[0]), "length": int(length[i])} for i in range(C)]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"episodes": rows}, f)
    np.savez(path.replace(".json", ".npz"), obs=obs, spin_rate=spin, length=np.asarray(length))
    np.savez(path.replace(".json", ".extra.npz"), **extra)


class TestSummary(unittest.TestCase):
    """The summary on synthetic cells: a pen at the origin along +X, both fingertips above it moving along +/-Y."""

    def cells(self, tmp, transfer=1.0, tip_speed=0.012, n=24, prefix_noise=0.0):
        from lares.envs.isaac_pen_spin import cases

        cfg = protocol_cfg()
        sl, T, p0, p1 = cases.obs_slices(), 215, 150, 180
        paths = {}
        for name, s in (("zero", 0.0), ("coherent_pos", 1.0), ("coherent_neg", -1.0)):
            obs = np.zeros((n, T, 157), dtype=np.float32)
            obs[:, :, sl["obj_rot"]] = [1.0, 0.0, 0.0, 0.0]
            tips = np.zeros((n, T, 5, 3), dtype=np.float32)
            tips[..., 2] = 0.02                                   # above the pen: the normal is +Z
            obs[:, :, sl["fingertip_pos"]] = tips.reshape(n, T, 15)
            quat = np.zeros((n, T, 5, 4), dtype=np.float32)
            quat[..., 0] = 1.0
            obs[:, :, sl["fingertip_rot"]] = quat.reshape(n, T, 20)
            vel = np.zeros((n, T, 5, 6), dtype=np.float32)
            # circumferential = axial x normal = X x Z = -Y
            vel[:, p0 + 1:p1 + 1, 0, 1] = -s * tip_speed
            vel[:, p0 + 1:p1 + 1, 2, 1] = -s * tip_speed
            obs[:, :, sl["fingertip_vel"]] = vel.reshape(n, T, 30)
            if name != "zero":
                obs[:, 100, 0] += prefix_noise
            spin = np.full((T, n), 2.0)
            spin[:, n // 2:] = -2.0
            spin[p0:p1] += s * transfer
            extra = {"identity_max_abs": np.float64(0.0), "chain": np.zeros((5, 18), dtype=bool),
                     **{f"audit_{k}": np.zeros((30, n, 18), dtype=bool) for k in ta.BOUNDS + ("incumbent_saturated",)},
                     "audit_q": np.zeros((30, n, 18), dtype=np.float32)}
            paths[name] = os.path.join(tmp, f"{name}.json")
            write_cell(paths[name], obs, spin, np.full(n, T), extra)
        return paths, cfg

    def summary(self, **kw):
        with tempfile.TemporaryDirectory() as tmp:
            paths, cfg = self.cells(tmp, **kw)
            return ta.audit_summary(paths, cfg)

    def test_supported_when_authority_and_transfer_are_present(self):
        s = self.summary()
        self.assertEqual(s["outcome"], "supported")
        self.assertEqual(s["eligibility"]["eligible"], 24)
        self.assertEqual(s["eligibility"]["positive_direction"], 12)
        self.assertAlmostEqual(s["pro_minus_anti_progress_rad"]["mean"], 2 * 30 / 60, places=5)   # +/-1 rad/s for 0.5 s
        self.assertAlmostEqual(s["authority"]["by_fingertip"]["ff"]["median_ratio"], 0.6, places=3)
        self.assertAlmostEqual(s["secondary"]["pro_minus_zero_rad"]["mean"], 0.5, places=5)
        self.assertAlmostEqual(s["secondary"]["anti_minus_zero_rad"]["mean"], -0.5, places=5)

    def test_refuted_when_authority_is_adequate_but_nothing_transfers(self):
        self.assertEqual(self.summary(transfer=0.0)["outcome"], "refuted")

    def test_inadequate_authority_is_not_a_refutation(self):
        self.assertEqual(self.summary(transfer=0.0, tip_speed=0.002)["outcome"], ta.INADEQUATE)

    def test_coverage_and_void(self):
        self.assertEqual(self.summary(n=10)["outcome"], "inconclusive_coverage")
        self.assertEqual(self.summary(prefix_noise=1e-3)["outcome"], ta.VOID)


class TestConfig(unittest.TestCase):
    def test_block_keeps_the_frozen_hash_and_names_its_own_thresholds(self):
        p = protocol_cfg()
        self.assertEqual(ta.require_frozen(p)["windows"]["pulse_start"], 150)
        self.assertEqual(p["frozen_sha256"][:8], "eb108114")
        self.assertEqual((p["cases"], p["pool_seed"], p["fingertips"]), (40, 4750, ["ff", "rf"]))
        self.assertEqual(set(p["audit_thresholds"]), set(TH))
        self.assertGreater(p["follow_through_stop"], p["windows"]["pulse_stop"])

    def test_registered(self):
        with open(os.path.join(ROOT, "scripts", "run_shadowhand_spin_stages.py"), encoding="utf-8") as f:
            self.assertIn('"transfer_audit": "transfer_audit"', f.read())

    def test_unknown_controller(self):
        with self.assertRaises(ValueError):
            ta.make_controller("nope", protocol_cfg())


if __name__ == "__main__":
    unittest.main()
