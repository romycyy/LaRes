"""R-097's rolling decoder gate: the paired statistics, the verdict, the selection rule and the controller (no Isaac)."""

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lares.envs.isaac_pen_spin import cases  # noqa: E402
from lares.envs.isaac_pen_spin import contact_control as cc  # noqa: E402
from lares.envs.isaac_pen_spin import contact_gate as cg  # noqa: E402
from lares.envs.isaac_pen_spin import rolling_decoder_gate as rg  # noqa: E402
from lares.utils.cli import load_yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SL = cases.obs_slices()
SIM_JOINTS = ["WRJ1", "WRJ0", "FFJ3", "MFJ3", "RFJ3", "LFJ4", "THJ4", "FFJ2", "MFJ2", "RFJ2", "LFJ3", "THJ3", "FFJ1",
              "MFJ1", "RFJ1", "LFJ2", "THJ2", "FFJ0", "MFJ0", "RFJ0", "LFJ1", "THJ1", "LFJ0", "THJ0"]
ACTUATED = list(range(17)) + [20, 21, 23]
TH = {"expected_contrasts": 15, "min_eligible_pairs": 20, "paired_sign_agreement_min": 0.8,
      "contrast_sign_agreement_min": 0.7, "paired_gain_median_min": 0.3, "paired_gain_median_max": 1.5,
      "feasible_ratio_median_min": 0.7, "bound_active_fraction_max": 0.2, "tracking_error_median_max": 0.7,
      "jacobian_rel_error_median_max": 0.15, "retention_diff_min": -0.10}
WIN = {"pulse_start": 3, "pulse_stop": 6, "pulse_speed": 0.02, "record_stop": 8, "jacobian_steps": [4],
       "jacobian_min_displacement": 0.0002}
FROZEN = {"decoder": {"damping": 1e-4, "tip_weights": [1.0] * 5, "max_joint_speed": 6.0, "control_dt": 1 / 60,
                      "lookahead": 0.1, "joint_limit_margin": 0.02, "request_scale": 0.05, "max_request_speed": 0.05},
          "controller": {"cap": 0.3, "control_wrist": False, "coupling": 1.0}, "windows": WIN}


def protocol_cfg() -> dict:
    cfg = load_yaml(os.path.join(ROOT, "config", "shadowhand_spin_stages.yaml"))
    return cfg["research_protocols"]["rolling_decoder_gate"]


def pairs(d=0.6, n=30, ratio=0.9, error=0.3, bound=0.05, contrasts=rg.CONTRASTS):
    return [{"case": i, "contrast": rg.contrast_name(k, a), "d": d, "feasible_ratio": ratio, "tracking_error": error,
             "bound_fraction": bound, "pulse_drops": 0} for k, a in contrasts for i in range(n)]


def verdict(p, jac=(0.01,), zero=0.9, parent=0.9, identity=0.0, prefix=0.0):
    return rg.gate_verdict(p, list(jac), zero, parent, identity, prefix, TH)


class TestVerdict(unittest.TestCase):
    def test_pass(self):
        v = verdict(pairs())
        self.assertEqual(v["verdict"], cc.PASS)
        self.assertEqual(v["pooled"]["eligible_pairs"], 450)
        self.assertEqual(v["prediction"]["outcome"], "supported")

    def test_each_tracking_check_can_fail(self):
        for kw in ({"d": -0.6}, {"d": 0.1}, {"ratio": 0.5}, {"error": 0.9}, {"bound": 0.5}):
            self.assertEqual(verdict(pairs(**kw))["verdict"], cc.FAIL, kw)

    def test_one_weak_contrast_fails_despite_pooled_agreement(self):
        p = pairs(contrasts=rg.CONTRASTS[1:]) + pairs(d=-0.6, contrasts=rg.CONTRASTS[:1])
        v = verdict(p)
        self.assertGreaterEqual(v["checks"]["paired_sign_agreement"]["value"], 0.8)
        self.assertFalse(v["checks"]["contrast_sign_agreement"]["ok"])
        self.assertEqual(v["verdict"], cc.FAIL)

    def test_hard_checks_fail_whatever_the_coverage(self):
        short = pairs(n=5)
        self.assertEqual(verdict(short, jac=(0.5,))["verdict"], cc.FAIL)
        self.assertEqual(verdict(short, zero=0.5)["verdict"], cc.FAIL)
        self.assertEqual(verdict(short, identity=1e-6)["verdict"], cc.FAIL)

    def test_inconclusive_on_coverage_or_a_void_pairing(self):
        self.assertEqual(verdict(pairs(n=19))["verdict"], cc.INCONCLUSIVE)
        self.assertEqual(verdict(pairs(contrasts=rg.CONTRASTS[:14]))["verdict"], cc.INCONCLUSIVE)
        self.assertEqual(verdict(pairs(), prefix=1e-4)["verdict"], cc.INCONCLUSIVE)

    def test_prediction_refuted_when_the_interval_is_below_target(self):
        self.assertEqual(verdict(pairs(d=-0.6))["prediction"]["outcome"], "refuted")

    def test_clustered_interval_resamples_cases(self):
        p = [{"case": c, "contrast": "x", "d": 1.0 if c < 5 else -1.0} for c in range(10) for _ in range(15)]
        low, high = rg.clustered_interval(p)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)
        self.assertGreater(high - low, 0.3)      # 150 pairs but only 10 cases: the interval stays wide


class TestSelection(unittest.TestCase):
    def cell(self, **kw):
        return {"sign_agreement": 0.9, "tracking_error_median": 0.5, "bound_active_fraction": 0.1,
                "feasible_ratio_median": 0.9, **kw}

    def test_qualifying_settings_first_then_agreement_then_error(self):
        cells = {"cal0": self.cell(sign_agreement=1.0, bound_active_fraction=0.5),
                 "cal1": self.cell(sign_agreement=0.90, tracking_error_median=0.6),
                 "cal2": self.cell(sign_agreement=0.89, tracking_error_median=0.4),
                 "cal3": self.cell(sign_agreement=0.80, tracking_error_median=0.1)}
        s = rg.select_setting(cells, TH)
        self.assertEqual(s["met_bound_and_feasible"], ["cal1", "cal2", "cal3"])
        self.assertEqual(s["tied_on_agreement"], ["cal1", "cal2"])
        self.assertEqual(s["chosen"], "cal2")

    def test_falls_back_to_all_settings(self):
        cells = {"cal0": self.cell(bound_active_fraction=0.9, sign_agreement=0.7),
                 "cal1": self.cell(bound_active_fraction=0.9, sign_agreement=0.8)}
        self.assertEqual(rg.select_setting(cells, TH)["chosen"], "cal1")


class TestStatistics(unittest.TestCase):
    def test_wilson(self):
        low, high = rg.wilson(35, 80)
        self.assertAlmostEqual(low, 0.334, places=2)
        self.assertAlmostEqual(high, 0.547, places=2)

    def test_roll_metrics_follow_each_case_direction(self):
        T = 40
        spin = np.stack([np.full(T, 3.0), np.full(T, -3.0), np.full(T, 3.0)], axis=1)
        spin[30:, 2] = -3.0
        m = rg.roll_metrics(spin, [T, T, 10], (10, 20), dt=0.1, smooth=5)
        self.assertEqual([x["direction"] for x in m], [1.0, -1.0, 0.0])      # the third case ended before the window
        self.assertAlmostEqual(m[0]["aligned_roll_after_window"], 6.0)
        self.assertAlmostEqual(m[1]["aligned_roll_after_window"], 6.0)       # unsigned: its own direction
        self.assertEqual(m[0]["reversals"], 0)

    def motion(self, achieved, feasible, length=(300, 300), spin=3.0):
        n = len(achieved)
        return {"achieved_world": np.array(achieved, dtype=float), "feasible_local": np.array(feasible, dtype=float),
                "R_first": np.tile(np.eye(3), (n, 1, 1)), "bound_fraction": np.zeros(n), "length": np.array(length),
                "completed": np.ones(n, dtype=bool), "spin_rate": np.full((200, n), spin),
                "obs_prefix": np.zeros((n, 4, 3))}

    def test_paired_motion_cancels_the_common_stroke(self):
        stroke = np.array([0.2, -0.1, 0.05])
        pos = self.motion([stroke + [0.012, 0, 0], stroke + [0, 0.012, 0]], [[0.018, 0, 0], [0, 0.018, 0]])
        neg = self.motion([stroke - [0.012, 0, 0], stroke - [0, 0.012, 0]], [[-0.018, 0, 0], [0, -0.018, 0]])
        win = {"pulse_start": 150, "pulse_stop": 180, "pulse_speed": 0.02}
        pm = rg.paired_motion(pos, neg, [0, 1], win, {"roll_window": [120, 150], "min_abs_axial_rate": 1.0})
        np.testing.assert_allclose(pm["d"], [0.6, 0.6])
        np.testing.assert_allclose(pm["feasible_ratio"], [0.9, 0.9])
        np.testing.assert_allclose(pm["tracking_error"], [1 / 3, 1 / 3])
        self.assertTrue(pm["eligible"].all())
        self.assertEqual(pm["prefix_max_abs_diff"], 0.0)

    def test_prefix_ignores_steps_after_a_case_ended(self):
        a, b = np.zeros((2, 6, 3)), np.zeros((2, 6, 3))
        a[1, 2:], b[1, 2:] = 7.0, -3.0                     # case 1 ended at step 2: what follows is another episode
        self.assertEqual(np.abs(rg.in_play_prefix(a, [9, 2], 4) - rg.in_play_prefix(b, [9, 2], 4)).max(), 0.0)
        b[0, 3] = 1e-3                                     # an in-play difference is seen
        self.assertGreater(np.abs(rg.in_play_prefix(a, [9, 2], 4) - rg.in_play_prefix(b, [9, 2], 4)).max(), 0.0)
        self.assertGreater(np.abs(rg.in_play_prefix(a, [9, 2], 4) - rg.in_play_prefix(a, [9, 4], 4)).max(), 0.0)

    def test_eligibility_reads_only_the_prefix(self):
        win = {"pulse_start": 150, "pulse_stop": 180, "pulse_speed": 0.02}
        el = {"roll_window": [120, 150], "min_abs_axial_rate": 1.0}
        slow = self.motion([[0, 0, 0]] * 2, [[0, 0, 0]] * 2, spin=0.5)
        self.assertFalse(rg.paired_motion(slow, slow, [0, 0], win, el)["eligible"].any())
        pos = self.motion([[0, 0, 0]] * 2, [[0, 0, 0]] * 2, length=(160, 100))
        pos["completed"] = np.array([False, False])
        pm = rg.paired_motion(pos, pos, [0, 0], win, el)
        self.assertEqual(pm["eligible"].tolist(), [True, False])     # a drop inside the pulse stays; one before it does not
        self.assertEqual(pm["pulse_drops"].tolist(), [2, 0])


class Incumbent:
    params = {"name": "fake"}

    def reset(self, ctx):
        pass

    def __call__(self, obs, t, ctx):
        return torch.full((obs.shape[0], 20), 0.1)


def fake_ctx(n: int):
    J = torch.zeros(n, 5, 3, 24)
    for k, j in enumerate((7, 8, 9, 10, 11)):
        J[:, k, :, j] = 0.05
    return SimpleNamespace(num_envs=n, device="cpu", joint_names=[f"robot0_{x}" for x in SIM_JOINTS],
                           actuated_dof_indices=list(ACTUATED), lower_all=torch.full((n, 24), -1.57),
                           upper_all=torch.full((n, 24), 1.57), joint_pos=lambda: torch.zeros(n, 24),
                           fingertip_jacobians=lambda: J)


def synthetic_obs(n: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    obs = torch.zeros(n, 157)
    obs[:, SL["obj_rot"]] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    obs[:, SL["fingertip_pos"]] = 0.03 * torch.randn(n, 15, generator=g)
    tq = torch.randn(n, 5, 4, generator=g)
    obs[:, SL["fingertip_rot"]] = (tq / tq.norm(dim=-1, keepdim=True)).reshape(n, 20)
    return obs


class TestController(unittest.TestCase):
    def run_ctrl(self, directions, steps=9):
        n = 3
        ctrl = rg.RollingPulseController("x", FROZEN, directions, inner=Incumbent())
        ctx, obs = fake_ctx(n), synthetic_obs(n)
        ctrl.reset(ctx)
        return ctrl, [ctrl(obs, t, ctx) for t in range(steps)]

    def test_zero_request_is_the_incumbent_exactly(self):
        ctrl, actions = self.run_ctrl(None)
        for a in actions:
            self.assertTrue(torch.equal(a, torch.full((3, 20), 0.1)))
        self.assertEqual(float(ctrl.extra_trace()["identity_max_abs"]), 0.0)

    def test_pulse_integrates_inside_the_window_only_and_is_antisymmetric(self):
        d = [(0, 0, 1.0)] * 3
        ctrl, pos = self.run_ctrl(d)
        _, neg = self.run_ctrl([(0, 0, -1.0)] * 3)
        base = torch.full((3, 20), 0.1)
        for t in (0, 1, 2, 6, 7, 8):
            self.assertTrue(torch.equal(pos[t], base), t)
        inc = [(pos[t] - base).abs().max().item() for t in (3, 4, 5)]
        self.assertGreater(inc[0], 0.0)
        self.assertGreater(inc[2], inc[0])                       # the offset accumulates
        torch.testing.assert_close(pos[4] - base, -(neg[4] - base), atol=1e-6, rtol=0)
        moved = (pos[4] - base).abs().sum(0) > 0
        names = [SIM_JOINTS[j] for j in ACTUATED]
        self.assertTrue(all("FF" in names[i] for i in torch.nonzero(moved).flatten().tolist()))   # that finger only
        extra = ctrl.extra_trace()
        self.assertEqual(extra["feasible"].shape, (3, 3, 3))
        self.assertEqual(extra["jacobian"].shape[0], 1)

    def test_cap_bounds_the_offset_and_is_reported(self):
        frozen = {**FROZEN, "controller": {**FROZEN["controller"], "cap": 1e-4}}
        ctrl = rg.RollingPulseController("x", frozen, [(0, 0, 1.0)] * 2, inner=Incumbent())
        ctx, obs = fake_ctx(2), synthetic_obs(2)
        ctrl.reset(ctx)
        for t in range(6):
            ctrl(obs, t, ctx)
        self.assertLessEqual(float(ctrl.delta.abs().max()), 1e-4 + 1e-9)
        self.assertTrue(ctrl.extra_trace()["bound_active"].any())


class TestConfig(unittest.TestCase):
    def test_block_is_complete_and_its_grid_fits_the_budget(self):
        p = protocol_cfg()
        self.assertEqual(set(rg.THRESHOLDS), set(p["thresholds"]))
        self.assertEqual(2 * len(p["calibration"]["settings"]) * int(p["cases"]), 800)
        for i in range(len(p["calibration"]["settings"])):
            c = rg.candidate(p, i)
            cc.DecoderConfig.from_dict(c["decoder"])
            self.assertIn("cap", c["controller"])
        self.assertEqual(len(rg.calibration_contrasts(80)), 80)
        self.assertEqual(len(set(rg.calibration_contrasts(80))), 15)

    def test_frozen_hash_is_enforced(self):
        p = dict(protocol_cfg())
        c = rg.candidate(p, 0)
        block = {"decoder": c["decoder"], "controller": c["controller"], "windows": dict(p["windows"]),
                 "thresholds": dict(p["thresholds"]), "eligibility": dict(p["eligibility"])}
        p.update(frozen={"decoder": c["decoder"], "controller": c["controller"]}, frozen_sha256=cg.frozen_sha256(block))
        self.assertEqual(rg.require_frozen(p)["controller"]["cap"], c["controller"]["cap"])
        p["frozen_sha256"] = "0" * 64
        with self.assertRaises(SystemExit):
            rg.require_frozen(p)


if __name__ == "__main__":
    unittest.main()
