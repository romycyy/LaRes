"""R-121: widened ranges, joint residual coordinates, matched step sizes and the Holm family (no Isaac)."""

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lares.envs.isaac_pen_spin import recovery_authority as ra  # noqa: E402
from lares.envs.isaac_pen_spin import recovery_joint as rj  # noqa: E402
from lares.envs.isaac_pen_spin import recovery_module as rm  # noqa: E402
from lares.utils.cli import load_yaml  # noqa: E402
from tests.test_recovery_module import Incumbent, ctx, obs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
CFG = load_yaml(os.path.join(ROOT, "config", "shadowhand_spin_stages.yaml"))["research_protocols"][rj.STUDY]


def make(arm, x=None, n=4):
    c = rm.build(CFG, arm, rj.initial_x(CFG, arm) if x is None else x, inner=Incumbent(), original_bounds=arm == "module_original")
    c.reset(ctx(n))
    return c


class TestJoint(unittest.TestCase):
    def test_initial_points_are_the_parent_in_each_arms_coordinates(self):
        x0, xw, xj = (rj.initial_x(CFG, a) for a in rj.ARMS)
        self.assertEqual((len(x0), len(xw), len(xj)), (11, 11, 43))
        p0, pw = rm.to_params(x0), rm.to_params(xw, *rm.bounds(CFG))
        np.testing.assert_allclose(pw, p0, atol=1e-12)
        self.assertAlmostEqual(p0[0], 4.0)                                   # the parent's threshold sits on the old bound
        self.assertLess(xw[0], 1.0)                                          # and inside the widened range
        np.testing.assert_array_equal(xj[:11], xw)
        np.testing.assert_allclose(rm.RESIDUAL_SPAN * xj[11:] - 0.10, ra.finalist_theta(CFG), atol=1e-12)

    def test_all_three_initial_policies_act_as_the_parent(self):
        ctrls = {a: make(a) for a in rj.ARMS}
        for t in range(260):
            o = obs(4, [1.0, 6.0, 3.5, 2.0])
            acts = {a: c(o, t, None) for a, c in ctrls.items()}
            self.assertLessEqual(float((acts["module_wide"] - acts["module_original"]).abs().max()), rm.SCRIPT_TOLERANCE)
            self.assertLessEqual(float((acts["joint"] - acts["module_original"]).abs().max()), rm.SCRIPT_TOLERANCE)
        self.assertTrue(torch.equal(ctrls["joint"].root.theta[0], ctrls["module_original"].root.theta))   # the residual exactly
        self.assertEqual(ctrls["joint"].triggered, ctrls["module_original"].triggered)

    def test_widened_threshold_and_residual_coordinates_act(self):
        x = rj.initial_x(CFG, "joint")
        x[0] = 1.0                                                           # threshold 6 rad/s
        c = make("joint", x)
        for t in range(200):
            c(obs(4, [5.0, 5.0, 7.0, 5.0]), t, None)
        self.assertEqual(c.elig.tolist(), [True, True, False, True])         # 5 rad/s now triggers, 7 does not
        xa, xb = rj.initial_x(CFG, "joint"), rj.initial_x(CFG, "joint")
        xb[11] = np.clip(xb[11] + 0.25, 0, 1)
        a, b = make("joint", xa), make("joint", xb)
        o = obs(4, [1.0] * 4)
        self.assertFalse(torch.equal(a(o, 0, None), b(o, 0, None)))
        self.assertAlmostEqual(float(b.root.theta[0, 0] - a.root.theta[0, 0]), 0.05, places=5)
        per_env = np.stack([xa, xb, xa, xb])
        c = make("joint", per_env)
        self.assertTrue(torch.equal(c.root.theta[1], b.root.theta[0]))
        self.assertTrue(torch.equal(c.root.theta[0], a.root.theta[0]))

    def test_step_sizes_are_physically_matched(self):
        s = rj.coord_scale(CFG, "module_wide")
        self.assertAlmostEqual(s[0], 3.0 / 5.0)
        self.assertAlmostEqual(s[3], 70.0 / 77.0)
        self.assertEqual([float(v) for v in np.delete(s, [0, 3])], [1.0] * 9)
        self.assertEqual(len(rj.coord_scale(CFG, "joint")), 43)
        low, high = rm.bounds(CFG)
        d = np.ones((1, 11))
        x0, xw = rj.initial_x(CFG, "module_original") * 0 + 0.5, rj.initial_x(CFG, "module_wide") * 0 + 0.5
        phys_o = rm.to_params(rm.ars_candidates(x0, d, 0.1)[0]) - rm.to_params(x0)
        phys_w = rm.to_params(rm.ars_candidates(xw, d, 0.1, s)[0], low, high) - rm.to_params(xw, low, high)
        np.testing.assert_allclose(phys_w, phys_o, atol=1e-12)               # the same physical perturbation in both arms
        x1, _ = rm.ars_step(xw, d, [1.0, 0.0], 0.05, s)
        x2, _ = rm.ars_step(x0, d, [1.0, 0.0], 0.05)
        np.testing.assert_allclose(rm.to_params(x1, low, high) - rm.to_params(xw, low, high), rm.to_params(x2) - rm.to_params(x0), atol=1e-12)

    def test_records_and_hashes(self):
        for arm in rj.ARMS:
            b = rm.bounds(None if arm == "module_original" else CFG)
            rec = rm.x_record(rj.initial_x(CFG, arm), *b)
            self.assertAlmostEqual(rec["params"]["threshold"], 4.0)
            self.assertAlmostEqual(rec["params"]["ramp_out"], 10.78, places=1)

    def test_frozen_files_registry_and_budget(self):
        self.assertEqual(rj.build_manifests(CFG), [])
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import run_shadowhand_spin_stages as rs

        self.assertEqual(rs.PEN_SPIN_STUDIES[rj.STUDY], "recovery_joint")
        self.assertLessEqual(3 * 24 * 2 * 2 * 2048 + 3 * 2 * 2048 + 2 * 6 * 12 * 128, CFG["start_ceiling"])
        self.assertEqual(len(rj.FAMILY), 8)

    def test_contrasts_and_qualification(self):
        rng = np.random.default_rng(0)
        rows = {}
        for m in range(12):
            base = rng.random(128) < 0.43
            mk = lambda s: [{"task_hash": f"h{m}_{i}", "sustained": bool(v), "completed": True, "tail_hit_fraction": float(v)}   # noqa: E731
                            for i, v in enumerate(s)]
            par = base | (rng.random(128) < 0.5)
            rows[f"s{m}"] = {"reference": mk(base), "r106": mk(base), "continue": mk(par), "module_original": mk(par),
                             "module_wide": mk(par), "joint": mk(par | (rng.random(128) < 0.3))}
        c = rj.contrasts(rows, 0.05, 0.05)
        self.assertEqual(len(c["contrasts"]), 10)
        self.assertTrue(c["qualification"]["joint"]["qualifies"])
        self.assertFalse(c["qualification"]["module_wide"]["qualifies"])       # equal to its parent: no positive lower bound
        self.assertTrue(c["qualification"]["module_wide"]["against_reference"])
        self.assertIn("holm", c["contrasts"]["joint_minus_module_wide"])
        self.assertNotIn("holm", c["contrasts"]["continue_minus_r106"])


if __name__ == "__main__":
    unittest.main()
