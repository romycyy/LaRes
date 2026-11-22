"""R-122: start-up offsets inside the residual clamp and the two-arm study (no Isaac)."""

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lares.envs.isaac_pen_spin import recovery_early as re_  # noqa: E402
from lares.envs.isaac_pen_spin import recovery_module as rm  # noqa: E402
from lares.utils.cli import load_yaml  # noqa: E402
from tests.test_recovery_module import Incumbent, ctx, obs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
CFG = load_yaml(os.path.join(ROOT, "config", "shadowhand_spin_stages.yaml"))["research_protocols"][re_.STUDY]


def make(x, n=4):
    c = rm.build(CFG, "t", x, inner=Incumbent())
    c.reset(ctx(n))
    return c


class TestEarly(unittest.TestCase):
    def test_initial_points(self):
        a, b = re_.initial_x(CFG, "continue"), re_.initial_x(CFG, "early")
        self.assertEqual((len(a), len(b)), (43, 63))
        np.testing.assert_array_equal(b[:43], a)
        np.testing.assert_array_equal(b[43:], 0.5)
        self.assertEqual(len(re_.coord_scale(CFG, "early")), 63)
        self.assertAlmostEqual(re_.coord_scale(CFG, "continue")[0], 0.6)

    def test_zero_offsets_are_the_parent_exactly(self):
        parent, early = make(re_.initial_x(CFG, "continue")), make(re_.initial_x(CFG, "early"))
        for t in range(230):
            o = obs(4, [1.0, 6.0, 3.0, 2.0])
            self.assertTrue(torch.equal(parent(o, t, None), early(o, t, None)), t)

    def test_offsets_act_in_the_first_sixty_steps_only_and_inside_the_clamp(self):
        x = re_.initial_x(CFG, "early")
        x[43:51] = 1.0                                        # +0.10 on eight joints at step 0
        parent, probe = make(re_.initial_x(CFG, "continue")), make(x)
        diffs = []
        for t in range(120):
            o = obs(4, [6.0] * 4)
            diffs.append(float((probe(o, t, None) - parent(o, t, None)).abs().max()))
        self.assertGreater(diffs[0], 0.0)
        self.assertLessEqual(max(diffs), 0.20 + 1e-6)         # two clamped residuals can differ by at most the clamp's width
        self.assertTrue(all(d == 0.0 for d in diffs[60:]))
        self.assertGreater(diffs[10], diffs[50])              # the gate decays
        gate = probe.root.startup_gate
        self.assertEqual(float(gate.max()), 0.0)

    def test_reset_restarts_the_gate_per_environment(self):
        x = re_.initial_x(CFG, "early")
        x[43:] = 1.0
        c = make(x)
        for t in range(100):
            c(obs(4, [6.0] * 4), t, None)
        c.reset_envs(torch.tensor([True, False, False, False]))
        c(obs(4, [6.0] * 4), 100, None)
        self.assertEqual(c.root.startup_gate[:, 0].tolist(), [1.0, 0.0, 0.0, 0.0])

    def test_frozen_files_registry_budget_and_contrasts(self):
        self.assertEqual(re_.build_manifests(CFG), [])
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import run_shadowhand_spin_stages as rs

        self.assertEqual(rs.PEN_SPIN_STUDIES[re_.STUDY], "recovery_early")
        self.assertLessEqual(2 * 24 * 2 * 2 * 2048 + 3 * 2 * 2048 + 2 * 4 * 12 * 128, CFG["start_ceiling"])
        self.assertEqual(len(re_.FAMILY), 5)
        rng = np.random.default_rng(0)
        rows = {}
        for m in range(12):
            base = rng.random(128) < 0.43
            mk = lambda s: [{"task_hash": f"h{m}_{i}", "sustained": bool(v), "completed": True, "tail_hit_fraction": float(v), "length": 359,   # noqa: E731
                             "final_center_dist": 0.02} for i, v in enumerate(s)]
            par = base | (rng.random(128) < 0.6)
            rows[f"s{m}"] = {"reference": mk(base), "joint": mk(par), "continue": mk(par), "early": mk(par | (rng.random(128) < 0.4))}
        c = re_.contrasts(rows, 0.05, 0.05)
        self.assertTrue(c["qualification"]["early"]["qualifies"])
        self.assertFalse(c["qualification"]["continue"]["qualifies"])
        self.assertEqual(sum(re_.label_table(rows)["early"].values()), 1536)


if __name__ == "__main__":
    unittest.main()
