"""R-134 / R-135: the two archive-only follow-up audits (no Isaac)."""

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lares.envs.isaac_pen_spin import reset_geometry_audit as rg  # noqa: E402
from lares.envs.isaac_pen_spin import stage_b_release_audit as sb  # noqa: E402
from lares.utils.cli import load_yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
ALL = load_yaml(os.path.join(ROOT, "config", "shadowhand_spin_stages.yaml"))["research_protocols"]


class TestRelease(unittest.TestCase):
    def test_change_is_measured_around_each_lineages_own_support_end(self):
        cfg = ALL[sb.STUDY]
        tl = sb.timeline({"decision_step": 120.2, "ramp_in": 20.0, "ramp_out": 39.5, "threshold": 3.0}, cfg)
        self.assertEqual((tl["decision_step"], tl["support_steps"], tl["support_end"]), (120, 60, 179))
        n, b = 4, tl["support_end"]
        spin = np.full((360, n), 1.0)
        spin[b + 1:, 0] = 8.0                                  # rolls only after support ends: change +1
        spin[b - 29:, 1] = 8.0                                 # already rolling before: change 0
        spin[100:, 2] = 8.0                                    # never slow at the decision: no trigger
        spin[b + 1:, 3] = 8.0
        length = np.array([359, 359, 359, b + 10])             # the last one drops inside the window: not eligible
        trace = {"spin_rate": spin, "center_dist": np.full((360, n), 0.02), "pred": spin > 4.0, "length": length}
        elig, change = sb.release_change(trace, tl, 30)
        self.assertEqual(elig.tolist(), [True, True, False, False])
        self.assertEqual([float(change[0]), float(change[1])], [1.0, 0.0])
        late = sb.timeline({"decision_step": 180.0, "ramp_in": 40.0, "ramp_out": 80.0, "threshold": 3.0}, cfg)
        self.assertFalse(sb.release_change(trace, late, 60)[0].any())   # the window would run past the episode

    def test_contrast_and_registry(self):
        rng = np.random.default_rng(0)
        per = {k: 0.1 + 0.2 * rng.standard_normal(400) for k in range(5)}
        per[2] = per[2] + 0.3
        c = sb.contrast(per, 2)
        self.assertAlmostEqual(c["estimate"], 0.3, delta=0.05)
        self.assertTrue(c["ci95"][0] > 0.10)
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import run_shadowhand_spin_stages as rs

        self.assertEqual((rs.PEN_SPIN_STUDIES[sb.STUDY], rs.PEN_SPIN_STUDIES[rg.STUDY]), ("stage_b_release_audit", "reset_geometry_audit"))


class TestGeometry(unittest.TestCase):
    def test_decomposition(self):
        v = np.zeros((3, 25))
        v[0, 0:3] = [1.0, 0.0, 0.0]                            # along the untilted pen: purely axial
        v[1, 0:3] = [0.0, 2.0, 0.0]                            # across it: purely transverse
        v[2, 0:3] = [1.0, 0.0, 0.0]
        v[2, 4] = 2.0 / 1.5                                    # scaled to b = pi/2: the axis is -z, so x is transverse
        g = rg.geometry(v, 1.5)
        np.testing.assert_allclose(g["axial"], [1.5, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(g["transverse"], [0.0, 3.0, 1.5], atol=1e-9)
        np.testing.assert_allclose(g["tilt"], [0.0, 0.0, math.pi / 2], atol=1e-9)
        self.assertEqual(rg.rank_bins([5, 1, 3, 2, 4, 0, 7, 6], 4).tolist(), [2, 0, 1, 1, 2, 0, 3, 3])

    def test_standardized_contrast(self):
        cfg = ALL[rg.STUDY]
        rng = np.random.default_rng(1)
        n = 1600
        geo = {"transverse": rng.random(n), "axial": rng.random(n), "tilt": rng.random(n)}
        ok = rng.random(n) < (0.7 - 0.5 * geo["transverse"] - 0.1 * geo["axial"])
        out = rg.audit(ok, geo, cfg)
        self.assertEqual((out["standardized"]["shared_strata"], out["standardized"]["verdict"]), (9, "supported"))
        self.assertAlmostEqual(out["standardized"]["difference"], -0.375, delta=0.08)
        flat = rg.audit(rng.random(n) < 0.5, geo, cfg)
        self.assertEqual(flat["standardized"]["verdict"], "refuted")
        thin = rg.standardized(ok[:40].astype(float), rg.rank_bins(geo["transverse"][:40], 4), np.arange(40), cfg)
        self.assertEqual(thin["verdict"], "inconclusive")


class TestTiltConditions(unittest.TestCase):
    def test_row_groups_and_scalar_equivalence(self):
        from lares.envs.isaac_pen_spin import cases
        from lares.envs.isaac_pen_spin import tilt_reset_robustness as tr

        cfg = ALL[tr.STUDY]
        v = np.random.default_rng(0).uniform(-1, 1, (16, cases.PLACEMENT_DIM))
        np.testing.assert_array_equal(cases.scale_placements(v, {}), v)
        np.testing.assert_array_equal(cases.scale_placements(v, None), v)
        t = cases.scale_placements(v, {"reset_tilt_scale": 1.5})
        np.testing.assert_array_equal(t[:, 3:5], 1.5 * v[:, 3:5])
        np.testing.assert_array_equal(np.delete(t, [3, 4], axis=1), np.delete(v, [3, 4], axis=1))
        o = cases.scale_placements(v, {"reset_other_scale": 1.5})
        np.testing.assert_array_equal(o[:, 3:5], v[:, 3:5])
        np.testing.assert_array_equal(np.delete(o, [3, 4], axis=1), 1.5 * np.delete(v, [3, 4], axis=1))
        np.testing.assert_array_equal(cases.scale_placements(v, {"reset_tilt_scale": 1.5, "reset_other_scale": 1.5}), cases.scale_placements(v, {"reset_scale": 1.5}))
        self.assertEqual(v.max() <= 1.0, True)                                    # the input is untouched
        self.assertEqual(sorted(tr.conditions(cfg)), ["full", "nominal", "position_only", "tilt_only"])
        self.assertEqual((len(tr.plan(cfg)), len(tr.plan(cfg)) * 128), (48, 6144))
        self.assertEqual(tr.build_manifests(cfg), [])
        d = tr.delivery(cfg)
        self.assertAlmostEqual(d["tilt_only"]["mean_tilt_rad"] / d["nominal"]["mean_tilt_rad"], 1.5, delta=0.05)
        self.assertAlmostEqual(d["position_only"]["mean_tilt_rad"], d["nominal"]["mean_tilt_rad"])
        self.assertAlmostEqual(d["position_only"]["mean_abs_joint_draw"] / d["nominal"]["mean_abs_joint_draw"], 1.5)
        self.assertEqual(d["full"]["mean_tilt_rad"], d["tilt_only"]["mean_tilt_rad"])
        for bad in ("joint", "transplant@nominal", "joint@wide"):
            with self.assertRaises(ValueError):
                tr.make_controller(bad, cfg)

    def test_primary_rule(self):
        from lares.envs.isaac_pen_spin import tilt_reset_robustness as tr

        cfg = ALL[tr.STUDY]
        rng = np.random.default_rng(0)

        def rows(prob):
            out = {}
            for s, c, k in tr.plan(cfg):
                u = np.random.default_rng(s).random(128)
                out[(s, c, k)] = [{"task_hash": f"h{s}_{i}", "sustained": bool(u[i] < prob(c, k)), "completed": True, "tail_hit_fraction": 1.0,
                                   "length": 359, "final_center_dist": 0.01} for i in range(128)]
            return out

        a = tr.analyse(rows(lambda c, k: 0.8 - (0.35 if k in ("tilt_only", "full") else 0.03 if k == "position_only" else 0.0)), cfg)
        self.assertEqual((a["primary"]["verdict"], a["primary"]["pairs"]), ("supported", 512))
        b = tr.analyse(rows(lambda c, k: 0.8 - (0.05 if k != "nominal" else 0.0)), cfg)
        self.assertEqual(b["primary"]["verdict"], "refuted")
        del rng


if __name__ == "__main__":
    unittest.main()
