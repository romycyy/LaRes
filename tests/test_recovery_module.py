"""R-116: the learned re-capture module: parameters, schedule, controller state and ARS (no Isaac)."""

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lares.envs.isaac_pen_spin import cases  # noqa: E402
from lares.envs.isaac_pen_spin import recovery_authority as ra  # noqa: E402
from lares.envs.isaac_pen_spin import recovery_module as rm  # noqa: E402
from lares.envs.isaac_pen_spin import symbolic_reward as sr  # noqa: E402
from lares.utils.cli import load_yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
CFG = load_yaml(os.path.join(ROOT, "config", "shadowhand_spin_stages.yaml"))["research_protocols"][rm.STUDY]
SL = cases.obs_slices()


class Policy:
    def __init__(self):
        self._stroke_gain = SimpleNamespace(constant=1.4081491479357957)


class Incumbent:
    """A stand-in incumbent whose action depends on the observation and on its (replaceable) stroke gain."""

    params = {"name": "fake"}

    def __init__(self):
        self.policy = Policy()

    def __call__(self, obs, t, ctx):
        g = self.policy._stroke_gain(obs[:, :1].cpu()) if callable(self.policy._stroke_gain) else 1.4081491479357957
        return (0.3 * torch.tanh(obs[:, :20]) + 0.1 * torch.as_tensor(g, dtype=obs.dtype) - 0.1).clamp(-1.0, 1.0)


def ctx(n):
    return SimpleNamespace(lower=torch.full((n, 20), -1.0), upper=torch.full((n, 20), 1.0), neutral_action=torch.zeros(n, 20))


def obs(n, rate, centre=0.0):
    """Pen axis along world x, spinning at ``rate`` rad/s about it, ``centre`` metres from the in-hand position."""
    o = torch.zeros(n, 157)
    o[:, SL["obj_rot"]] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    o[:, SL["obj_angvel"]] = torch.tensor([1.0, 0.0, 0.0]) * torch.as_tensor(rate, dtype=torch.float32).reshape(-1, 1) * 0.2
    o[:, SL["obj_pos"]] = torch.tensor([1.0, 0.0, 0.0]) * centre
    return o


class TestParameters(unittest.TestCase):
    def test_script_is_the_initial_point(self):
        p = rm.x_record(rm.SCRIPT_X)["params"]
        self.assertEqual((p["threshold"], p["decision_step"], p["ramp_in"], p["ramp_out"], p["stroke_multiplier"]), (3.0, 120.0, 20.0, 40.0, 1.0))
        self.assertEqual([p[f"depth_{g}"] for g in sr.GROUPS], [1.0] * 6)
        self.assertEqual(rm.DIM, 11)
        np.testing.assert_allclose(rm.to_params(np.full(11, 2.0)), rm.HIGH)       # projection

    def test_blend_is_the_d080_schedule_at_the_script(self):
        cfg = {"t0": 120, "window": 60, "ramp_in": 20}
        for t in range(100, 200):
            got = float(rm.blend(torch.tensor([float(t - 120)]), torch.tensor([20.0]), torch.tensor([40.0])))
            self.assertAlmostEqual(got, ra.beta(t, cfg), places=6)

    def test_ars_moves_toward_better_candidates_and_stays_in_the_cube(self):
        x = rm.SCRIPT_X.copy()
        rng = np.random.default_rng(0)
        for _ in range(30):
            d = rng.standard_normal((4, rm.DIM))
            cand = rm.ars_candidates(x, d, 0.1)
            x, _ = rm.ars_step(x, d, [-(c[0] - 0.2) ** 2 for c in cand], 0.05)   # prefers a threshold coordinate of 0.2
        self.assertLess(abs(x[0] - 0.2), abs(rm.SCRIPT_X[0] - 0.2))
        self.assertTrue(((x >= 0) & (x <= 1)).all())
        same, sigma = rm.ars_step(x, d, [0.5] * 8, 0.05)
        np.testing.assert_array_equal(same, x)


class TestController(unittest.TestCase):
    def make(self, x=rm.SCRIPT_X, enabled=True, n=4):
        c = rm.build(CFG, "t", x, enabled=enabled, inner=Incumbent())
        c.reset(ctx(n))
        return c

    def run_episode(self, c, rates, steps=200, centre=0.0):
        sent, root = [], []
        for t in range(steps):
            sent.append(c(obs(len(rates), rates, centre), t, None).clone())
            root.append(c.last_root.clone())
        return sent, root

    def test_only_slow_centred_environments_trigger_and_follow_the_script(self):
        c = self.make()
        sent, root = self.run_episode(c, [1.0, 6.0, 2.9, 3.1])
        self.assertEqual(c.elig.tolist(), [True, False, True, False])
        cfg = {"t0": 120, "window": 60, "ramp_in": 20}
        for t in range(200):
            b = ra.beta(t, cfg)
            for i in range(4):
                want = ((1 - b) * root[t][i]).clamp(-1, 1) if (b > 0 and i in (0, 2)) else root[t][i]   # neutral action is 0
                self.assertLessEqual(float((sent[t][i] - want).abs().max()), rm.SCRIPT_TOLERANCE, (t, i))
                if b == 0 or i in (1, 3):
                    self.assertTrue(torch.equal(sent[t][i], root[t][i]), (t, i))       # untouched actions are the root's exactly
        self.assertEqual(c.triggered, 2)

    def test_off_centre_or_disabled_never_triggers(self):
        c = self.make()
        self.run_episode(c, [1.0, 1.0, 1.0, 1.0], centre=0.2)
        self.assertFalse(bool(c.elig.any()))
        off = self.make(enabled=False)
        sent, root = self.run_episode(off, [1.0, 1.0, 1.0, 1.0])
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(sent, root)))

    def test_disabled_module_is_the_root_controller_bit_for_bit(self):
        off = self.make(enabled=False, n=6)
        plain = sr.build_controller(CFG, "p", ra.finalist_theta(CFG), ra.normalization(CFG), inner=Incumbent())
        plain.reset(ctx(6))
        g = torch.Generator().manual_seed(0)
        for t in range(40):
            o = torch.randn(6, 157, generator=g) * 0.3
            self.assertTrue(torch.equal(off(o, t, None), plain(o, t, None)))

    def test_per_environment_parameters_and_depths(self):
        x = np.tile(rm.SCRIPT_X, (4, 1))
        x[1, 1] = 0.0                                         # environment 1 decides at step 60
        x[2, 4:10] = 0.0                                      # environment 2 blends nowhere
        x[3, 4] = 0.0                                         # environment 3 leaves the wrist alone
        c = self.make(x)
        sent, root = self.run_episode(c, [1.0] * 4, steps=140)
        self.assertEqual(c.T.tolist(), [120, 60, 120, 120])
        self.assertFalse(torch.equal(sent[70][1], root[70][1]))
        self.assertTrue(torch.equal(sent[70][0], root[70][0]))
        self.assertTrue(torch.equal(sent[139][2], root[139][2]))
        wrist = [i for i, j in enumerate(sr.JOINTS) if j.startswith("WR")]
        self.assertTrue(torch.equal(sent[139][3][wrist], root[139][3][wrist]))
        self.assertFalse(torch.equal(sent[139][3], root[139][3]))

    def test_stroke_multiplier_acts_only_after_the_blend(self):
        x = rm.SCRIPT_X.copy()
        x[10] = 1.0                                           # multiplier 2
        c = self.make(x)
        seen = []
        for t in range(250):
            c(obs(4, [1.0, 6.0, 1.0, 6.0]), t, None)
            seen.append(c.gain.scale[:, 0].clone())
        self.assertTrue(all(float(s.max()) == 1.0 for s in seen[:180]))
        self.assertEqual(seen[180].tolist(), [2.0, 1.0, 2.0, 1.0])
        self.assertEqual(seen[239].tolist(), [2.0, 1.0, 2.0, 1.0])
        self.assertEqual(seen[240].tolist(), [1.0] * 4)

    def test_reset_is_per_environment(self):
        c = self.make()
        self.run_episode(c, [1.0] * 4, steps=130)
        self.assertTrue(bool(c.elig.all()))
        c.reset_envs(torch.tensor([True, False, False, True]))
        self.assertEqual(c.step.tolist(), [0, 130, 130, 0])
        self.assertEqual(c.elig.tolist(), [False, True, True, False])
        self.assertEqual(float(c.hist[0].abs().sum()), 0.0)
        for t in range(125):                                  # a reset environment decides again on its own clock
            c(obs(4, [1.0] * 4), t, None)
        self.assertEqual(c.elig.tolist(), [True, True, True, True])
        self.assertEqual(c.triggered, 6)


class TestFrozenFiles(unittest.TestCase):
    def test_manifests_registry_budget(self):
        self.assertEqual(rm.build_manifests(CFG), [])
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import run_shadowhand_spin_stages as rs

        self.assertEqual(rs.PEN_SPIN_STUDIES[rm.STUDY], "recovery_module")
        self.assertLessEqual(12 * 2 * 2 * 2048 + 2 * 2 * 2048 + 2 * 4 * 12 * 128, CFG["start_ceiling"])
        with self.assertRaises(ValueError):
            rm.make_controller("draw3", CFG)

    def test_contrasts(self):
        rng = np.random.default_rng(0)
        rows = {}
        for m in range(12):
            base = rng.random(128) < 0.42
            mk = lambda s: [{"task_hash": f"h{m}_{i}", "sustained": bool(v), "completed": True, "tail_hit_fraction": float(v)}   # noqa: E731
                            for i, v in enumerate(s)]
            root = base | (rng.random(128) < 0.2)
            rows[f"s{m}"] = {"reference": mk(base), "r106": mk(root), "scripted": mk(root), "trained": mk(root | (rng.random(128) < 0.2))}
        c = rm.contrasts(rows, 0.05, 0.05)
        self.assertEqual(len(c), 6)
        self.assertTrue(c["trained_minus_r106"]["pooled"]["adopts"])
        self.assertEqual(c["scripted_minus_r106"]["pooled"]["success"]["mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
