"""R-126: block combinations, endpoint reproduction, sizing and contrasts (no Isaac)."""

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lares.envs.isaac_pen_spin import block_transplant as bt  # noqa: E402
from lares.envs.isaac_pen_spin import recovery_module as rm  # noqa: E402
from lares.utils.cli import load_yaml  # noqa: E402
from tests.test_recovery_module import Incumbent, ctx, obs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
CFG = load_yaml(os.path.join(ROOT, "config", "shadowhand_spin_stages.yaml"))["research_protocols"][bt.STUDY]


def make(x, original=False):
    c = rm.build(CFG, "t", x, inner=Incumbent(), original_bounds=original)
    c.reset(ctx(4))
    return c


class TestTransplant(unittest.TestCase):
    def test_endpoints_and_shapes(self):
        e = bt.check_endpoints(CFG)
        self.assertTrue(e["passes"] and e["joint_matches_asset"])
        self.assertEqual([len(bt.compose(CFG, n)) for n in bt.HYBRIDS], [11, 11, 43, 43])
        np.testing.assert_array_equal(bt.compose(CFG, "transplant"), bt.compose(CFG, "joint")[:11])
        np.testing.assert_array_equal(bt.compose(CFG, "reciprocal")[11:], bt.compose(CFG, "joint")[11:])
        np.testing.assert_array_equal(bt.compose(CFG, "reciprocal")[:11], bt.compose(CFG, "r118"))
        with self.assertRaises(SystemExit):
            bt.check_endpoints({**CFG, "assets": {**CFG["assets"], "r121": {**CFG["assets"]["r121"], "sha256": "0" * 64}}})

    def test_composed_r118_acts_as_the_asset_and_blocks_carry_their_residuals(self):
        asset = make(bt._asset(CFG, "r118"), original=True)
        ctrls = {n: make(bt.compose(CFG, n)) for n in bt.HYBRIDS}
        for t in range(230):
            o = obs(4, [1.0, 6.0, 3.0, 2.0])
            a = asset(o, t, None)
            out = {n: c(o, t, None) for n, c in ctrls.items()}
            self.assertLessEqual(float((out["r118"] - a).abs().max()), 1e-5, t)
        th = {n: c.root.theta for n, c in ctrls.items()}
        self.assertTrue(torch.equal(th["r118"], th["transplant"]))
        self.assertTrue(torch.equal(th["joint"], th["reciprocal"]))
        self.assertFalse(torch.equal(th["joint"].expand_as(th["joint"]), th["r118"].expand_as(th["joint"])))
        self.assertTrue(torch.allclose(ctrls["joint"].threshold, ctrls["transplant"].threshold))
        self.assertTrue(torch.allclose(ctrls["reciprocal"].threshold, ctrls["r118"].threshold))

    def test_sizing_rule(self):
        s = bt.confirmation_size(0.197, CFG)
        self.assertEqual((s["pairs_required"], s["manifests"], s["within_cap"]), (3867, 31, True))
        self.assertEqual(bt.confirmation_size(0.0, CFG)["manifests"], 8)
        self.assertFalse(bt.confirmation_size(0.30, CFG)["within_cap"])
        self.assertEqual(bt.confirmation_size(0.30, CFG)["cap"], 40)

    def test_manifests_registry_contrasts(self):
        self.assertEqual(bt.build_manifests(CFG), [])
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import run_shadowhand_spin_stages as rs

        self.assertEqual(rs.PEN_SPIN_STUDIES[bt.STUDY], "block_transplant")
        self.assertLessEqual((2 + 40) * 5 * 128, 26880)
        rng = np.random.default_rng(0)
        rows = {}
        for m in range(8):
            base = rng.random(128) < 0.75
            mk = lambda s: [{"task_hash": f"h{m}_{i}", "sustained": bool(v), "completed": True, "tail_hit_fraction": float(v), "length": 359,
                             "final_center_dist": 0.02} for i, v in enumerate(s)]   # noqa: E731
            rows[f"s{m}"] = {"reference": mk(rng.random(128) < 0.4), "r118": mk(base), "transplant": mk(base), "reciprocal": mk(base),
                             "joint": mk(base | (rng.random(128) < 0.3))}
        c = bt.contrasts(rows, CFG)
        self.assertEqual(set(c), {f"{a}_minus_{b}" for a, b in bt.FAMILY})
        self.assertTrue(c[bt.PRIMARY]["holm"]["rejects"])
        self.assertFalse(c["transplant_minus_r118"]["holm"]["rejects"])
        self.assertGreater(bt.interaction(rows)["low"], 0.0)
        self.assertGreater(bt.discordance(rows, "joint", "transplant"), 0.0)
        self.assertEqual(bt.discordance(rows, "transplant", "r118"), 0.0)


if __name__ == "__main__":
    unittest.main()
