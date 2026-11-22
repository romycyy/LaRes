"""R-099's anonymous joint-motion residual: the controller, the draws, the ranking and the fresh contrast (no Isaac)."""

import os
import sys
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lares.envs.isaac_pen_spin import cases  # noqa: E402
from lares.envs.isaac_pen_spin import incumbent_residual as ir  # noqa: E402
from lares.utils.cli import load_yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SL = cases.obs_slices()


def protocol_cfg() -> dict:
    return load_yaml(os.path.join(ROOT, "config", "shadowhand_spin_stages.yaml"))["research_protocols"]["incumbent_residual"]


class Incumbent:
    params = {"name": "fake"}

    def __call__(self, obs, t, ctx):
        return torch.full((obs.shape[0], 20), 0.5)


def ctx(n: int):
    return SimpleNamespace(lower=torch.full((n, 20), -1.0), upper=torch.full((n, 20), 1.0))


def obs_with_velocity(n: int, v: float) -> torch.Tensor:
    obs = torch.zeros(n, 157)
    obs[:, SL["hand_dof_vel"]] = v * 0.2          # the observation carries rad/s times VEL_OBS_SCALE
    return obs


class TestController(unittest.TestCase):
    def run_ctrl(self, params, v, steps=3):
        c = ir.JointMotionResidual("x", params, inner=Incumbent())
        c.reset(ctx(2))
        return c, [c(obs_with_velocity(2, v), t, None) for t in range(steps)]

    def test_zero_gain_is_the_incumbent_exactly(self):
        _, actions = self.run_ctrl({"gain": 0.0, "deadband": 0.1, "tau": 0.1, "cap": 0.1}, 2.0)
        for a in actions:
            self.assertTrue(torch.equal(a, torch.full((2, 20), 0.5)))

    def test_reinforces_motion_in_its_own_direction(self):
        p = {"gain": 0.05, "deadband": 0.5, "tau": 0.1, "cap": 1.0}
        _, pos = self.run_ctrl(p, 2.0)
        _, neg = self.run_ctrl(p, -2.0)
        torch.testing.assert_close(pos[0], torch.full((2, 20), 0.5 + 0.05 * 1.5))     # span 2: delta rad = delta action
        torch.testing.assert_close(neg[0], torch.full((2, 20), 0.5 - 0.05 * 1.5))

    def test_deadband_cap_and_action_clamp(self):
        _, quiet = self.run_ctrl({"gain": 0.05, "deadband": 0.5, "tau": 0.1, "cap": 1.0}, 0.3)
        self.assertTrue(torch.equal(quiet[0], torch.full((2, 20), 0.5)))
        _, capped = self.run_ctrl({"gain": 0.1, "deadband": 0.0, "tau": 0.1, "cap": 0.04}, 3.0)
        torch.testing.assert_close(capped[0], torch.full((2, 20), 0.54))
        c, clipped = self.run_ctrl({"gain": 0.5, "deadband": 0.0, "tau": 0.1, "cap": 2.0}, 3.0)
        self.assertTrue(torch.equal(clipped[0], torch.ones(2, 20)))
        self.assertEqual(c.activity()["clipped_by_action_clamp_fraction"], 1.0)

    def test_filter_lags_a_change(self):
        c = ir.JointMotionResidual("x", {"gain": 0.05, "deadband": 0.0, "tau": 0.25, "cap": 1.0}, inner=Incumbent())
        c.reset(ctx(1))
        first = c(obs_with_velocity(1, 2.0), 0, None)
        second = c(obs_with_velocity(1, 0.0), 1, None)
        self.assertGreater(float(second[0, 0]), 0.5)          # still reinforcing the earlier motion
        self.assertLess(float(second[0, 0]), float(first[0, 0]))


class TestSearchAndContrast(unittest.TestCase):
    def test_draws_are_seeded_inside_the_declared_ranges(self):
        p = protocol_cfg()
        a, b = ir.draw_params(p), ir.draw_params(p)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 24)
        self.assertEqual(len({ir.params_sha256(d) for d in a}), 24)
        for d in a:
            for k in ir.PARAMS:
                self.assertTrue(p["ranges"][k][0] <= d[k] <= p["ranges"][k][1])

    def test_ranking_is_tail_hit_then_successes_then_index(self):
        rows = [{"tail_hit_fraction": 0.4, "successes": 30}, {"tail_hit_fraction": 0.5, "successes": 10},
                {"tail_hit_fraction": 0.5, "successes": 12}, {"tail_hit_fraction": 0.5, "successes": 12}]
        self.assertEqual(sorted(range(4), key=lambda i: ir.rank_key(rows[i], i)), [2, 3, 1, 0])

    def rows(self, sustained, completed=None):
        completed = completed or [True] * len(sustained)
        return [{"task_hash": str(i), "sustained": s, "completed": c, "tail_hit_fraction": float(s)}
                for i, (s, c) in enumerate(zip(sustained, completed))]

    def test_strict_rule_and_drop_bound(self):
        base = self.rows([False] * 90)
        better = self.rows([True] * 30 + [False] * 60)
        c = ir.fresh_contrast(better, base, 0.10, 0.05)
        self.assertTrue(c["promotes"] and c["adopts"] and c["prediction"]["supported"])
        dropping = self.rows([True] * 30 + [False] * 60, [True] * 60 + [False] * 30)
        c = ir.fresh_contrast(dropping, base, 0.10, 0.05)
        self.assertTrue(c["promotes"])
        self.assertFalse(c["adopts"])                          # more drops than the margin allows
        tie = ir.fresh_contrast(base, base, 0.10, 0.05)
        self.assertFalse(tie["promotes"])
        self.assertTrue(tie["prediction"]["refuted"])

    def test_unpaired_rows_are_refused(self):
        with self.assertRaises(ValueError):
            ir.fresh_contrast(self.rows([True, False]), list(reversed(self.rows([True, False]))), 0.1, 0.05)

    def test_selected_hash_is_enforced(self):
        p = dict(protocol_cfg())
        with self.assertRaises(SystemExit):
            ir.require_selected({**p, "selected": None, "selected_sha256": None})
        d = ir.draw_params(p)[3]
        self.assertEqual(ir.require_selected({**p, "selected": d, "selected_sha256": ir.params_sha256(d)}), d)
        with self.assertRaises(SystemExit):
            ir.require_selected({**p, "selected": d, "selected_sha256": "0" * 64})


if __name__ == "__main__":
    unittest.main()
