"""R-112 (P3): the memory states, the three matched arms and the contrasts (no Isaac)."""

import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lares.envs.isaac_pen_spin import memory_study as ms  # noqa: E402
from lares.envs.isaac_pen_spin import structure_search as ss  # noqa: E402
from lares.envs.isaac_pen_spin import structure_terms as st  # noqa: E402
from lares.envs.isaac_pen_spin import symbolic_reward as sr  # noqa: E402
from lares.utils.cli import load_yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
CFG = load_yaml(os.path.join(ROOT, "config", "shadowhand_spin_stages.yaml"))["research_protocols"][ms.STUDY]
NORM = ss.normalization(CFG)


class Incumbent:
    params = {"name": "fake"}

    def __call__(self, obs, t, ctx):
        return (0.3 * torch.tanh(obs[:, :20]) + 0.2).clamp(-1.0, 1.0)


def ctx(n):
    return SimpleNamespace(lower=torch.full((n, 20), -1.0), upper=torch.full((n, 20), 1.0))


def obs_batch(n, seed):
    return torch.randn(n, 157, generator=torch.Generator().manual_seed(seed)) * 0.3


class TestMemory(unittest.TestCase):
    def test_arms_differ_only_in_the_gate(self):
        pols = {a: ms.arm_policy(CFG, a) for a in ms.ARMS}
        root = ss.root_policy(CFG)
        for a, p in pols.items():
            self.assertEqual(len(p["theta"]), 58)
            self.assertEqual(p["theta"][:32], root["theta"])
            self.assertEqual(p["theta"][32:], [0.0] * 26)
            self.assertEqual([t["target"] for t in p["terms"][3:]], ["joints", "groups"])
            self.assertEqual(p["terms"][3]["expr"], ms.GATES[a])
        self.assertEqual(len({p["sha256"] for p in pols.values()}), 3)

    def test_zero_new_coefficients_reproduce_the_r106_controller(self):
        root = ss.root_policy(CFG)
        a = sr.build_controller(CFG, "a", root["theta"], NORM, inner=Incumbent())
        b = st.build_controller(CFG, "b", ms.arm_policy(CFG, "two_timescale"), NORM, inner=Incumbent())
        a.reset(ctx(16))
        b.reset(ctx(16))
        for t in range(10):
            o = obs_batch(16, t)
            self.assertTrue(torch.equal(a(o, t, None), b(o, t, None)))

    def test_states_follow_their_definitions(self):
        c = st.build_controller(CFG, "c", ms.arm_policy(CFG, "two_timescale"), NORM, inner=Incumbent())
        c.reset(ctx(4))
        fast = slow = None
        for t in range(6):
            p = c.parts(obs_batch(4, t), t, None)
            s = p["s"]
            fast = s.clone() if fast is None else fast + c.ref.alpha * (s - fast)
            slow = (fast / 60.0) if slow is None else slow + (fast - slow) / 60.0
            self.assertTrue(torch.equal(p["feats"]["deficit_fast"], fast))
            self.assertTrue(torch.equal(p["feats"]["deficit_slow"], slow))
        self.assertLess(float(slow.max()), 6.0 / 60.0 + 1e-6)             # the slow state cannot rise during a start

    def test_reset_is_per_environment(self):
        c = st.build_controller(CFG, "c", ms.arm_policy(CFG, "two_timescale"), NORM, inner=Incumbent())
        c.reset(ctx(4))
        for t in range(8):
            c.parts(obs_batch(4, t), t, None)
        fast0, slow0 = c._fast.clone(), c._slow.clone()
        mask = torch.tensor([True, False, False, True])
        c.reset_envs(mask)
        o = obs_batch(4, 50)
        p = c.parts(o, 8, None)
        fresh = st.build_controller(CFG, "f", ms.arm_policy(CFG, "two_timescale"), NORM, inner=Incumbent())
        fresh.reset(ctx(4))
        fresh.parts(o, 0, None)
        self.assertTrue(torch.equal(c._fast[mask], fresh._fast[mask]))
        self.assertTrue(torch.equal(c._slow[mask], fresh._slow[mask]))
        keep = ~mask
        exp_fast = fast0 + c.ref.alpha * (p["s"] - fast0)
        self.assertTrue(torch.equal(c._fast[keep], exp_fast[keep]))
        self.assertTrue(torch.equal(c._slow[keep], (slow0 + (exp_fast - slow0) / 60.0)[keep]))

    def test_new_coefficients_act_through_their_gate_only(self):
        pol = ms.arm_policy(CFG, "two_timescale")
        c = st.build_controller(CFG, "c", pol, NORM, inner=Incumbent())
        c.reset(ctx(8))
        for t in range(30):
            p = c.parts(obs_batch(8, t), t, None)
        th = c.theta.clone()
        th[32:52] = 0.05
        d = c.mean(p, th) - c.mean(p, c.theta)
        self.assertGreater(float(d.abs().max()), 0.0)
        self.assertLessEqual(float(d.abs().max()), 0.05 * float(p["feats"]["deficit_slow"].max()) + 0.2 + 1e-6)

    def test_state_primitives_are_not_sampled_by_the_grammar(self):
        root = ss.root_policy(CFG)
        for seed in range(200):
            e = st.sample_edit(np.random.default_rng(seed), root)
            self.assertFalse(any(s in str(e["expr"]) for s in st.STATE_PRIMITIVES))

    def test_config_schedule_manifests_and_budget(self):
        self.assertEqual(ss.schedule(CFG)["slices"], 12)
        self.assertEqual(ss.direction_seed(CFG, 0, 0, 3), 4894 + 2000 + 300)
        self.assertEqual(ss.training_placements(CFG, 0, 0, 11).shape, (512, 29))
        self.assertEqual(ms.build_manifests(CFG), [])
        est = 3 * 12 * 8200 + 8200 + 2 * 5 * 12 * 128
        self.assertLessEqual(est, CFG["start_ceiling"])
        self.assertEqual(ss.config_sha256(CFG), ss.config_sha256({**CFG, "config_path": "x", "study": "y"}))
        r109 = load_yaml(os.path.join(ROOT, "config", "pen_spin_plateau_repair.yaml"))[ss.CONFIG_KEY]
        self.assertEqual(ss.config_sha256(r109)[:12], "7716c159132e")              # R-109's amended block is untouched

    def test_contrasts(self):
        rng = np.random.default_rng(0)
        rows = {}
        for m in range(12):
            base = rng.random(128) < 0.42
            mk = lambda s: [{"task_hash": f"h{m}_{i}", "sustained": bool(x), "completed": True, "tail_hit_fraction": float(x)}   # noqa: E731
                            for i, x in enumerate(s)]
            root = base | (rng.random(128) < 0.2)
            rows[f"s{m}"] = {"reference": mk(base), "r106": mk(root), "instant": mk(root), "single_filter": mk(root),
                             "two_timescale": mk(root | (rng.random(128) < 0.15))}
        c = ms.contrasts(rows, 0.05, 0.05)
        self.assertEqual(len(c), 9)
        self.assertTrue(c["two_timescale_minus_r106"]["useful_against_root"])
        self.assertFalse(c["instant_minus_r106"]["useful_against_root"])
        self.assertGreater(c["two_timescale_minus_single_filter"]["pooled"]["success"]["low"], 0.05)


if __name__ == "__main__":
    unittest.main()
