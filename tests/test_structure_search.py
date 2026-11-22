"""R-109 (P2): the term-list actor, the edit language, the selection gate and the schedule (no Isaac, no API key)."""

import json
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lares.envs.isaac_pen_spin import structure_search as ss  # noqa: E402
from lares.envs.isaac_pen_spin import structure_terms as st  # noqa: E402
from lares.envs.isaac_pen_spin import symbolic_reward as sr  # noqa: E402
from lares.utils.cli import load_yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
CFG = load_yaml(os.path.join(ROOT, "config", "pen_spin_plateau_repair.yaml"))[ss.CONFIG_KEY]
NORM = ss.normalization(CFG)


class Incumbent:
    params = {"name": "fake"}

    def __call__(self, obs, t, ctx):
        return (0.3 * torch.tanh(obs[:, :20]) + 0.2).clamp(-1.0, 1.0)


def ctx(n):
    return SimpleNamespace(lower=torch.full((n, 20), -1.0), upper=torch.full((n, 20), 1.0))


def obs_batch(n, seed):
    return torch.randn(n, 157, generator=torch.Generator().manual_seed(seed)) * 0.3


def sample(n=600):
    obs = torch.cat([obs_batch(100, s) for s in range(n // 100)])
    return st.sample_features({"obs": obs.numpy(), "ref": np.zeros((n, 20), np.float32), "v_f": 0.2 * np.ones((n, 20), np.float32)},
                              NORM, CFG["actor"])


class TestActor(unittest.TestCase):
    def setUp(self):
        self.root = ss.root_policy(CFG)

    def test_base_structure_is_the_r106_controller_bit_for_bit(self):
        a = sr.build_controller(CFG, "a", self.root["theta"], NORM, inner=Incumbent())
        b = st.build_controller(CFG, "b", self.root, NORM, inner=Incumbent())
        a.reset(ctx(32))
        b.reset(ctx(32))
        for t in range(8):
            o = obs_batch(32, t)
            self.assertTrue(torch.equal(a(o, t, None), b(o, t, None)))
        self.assertEqual(st.n_coefficients(st.BASE_TERMS), 32)

    def test_child_inherits_exactly_and_new_coefficients_act(self):
        edit = st.validate_edit({"expr": ["mul", "tip_dist", "spin_deficit"], "target": "chain:TH"}, self.root)
        self.assertEqual(edit["expr"], ["mul", "spin_deficit", "tip_dist"])           # canonical argument order
        child = st.extend(self.root, edit)
        self.assertEqual(child["theta"][:32], self.root["theta"])
        self.assertEqual(len(child["theta"]), 37)
        p, c = st.build_controller(CFG, "p", self.root, NORM, inner=Incumbent()), st.build_controller(CFG, "c", child, NORM, inner=Incumbent())
        p.reset(ctx(16))
        c.reset(ctx(16))
        o = obs_batch(16, 3)
        pp, cp = p.parts(o, 0, None), c.parts(o, 0, None)
        self.assertTrue(torch.equal(p.mean(pp, p.theta), c.mean(cp, c.theta)))
        moved = c.theta.clone()
        moved[-5:] = 0.05
        d = (c.mean(cp, moved) - c.mean(cp, c.theta)).abs()
        thumb = [i for i, j in enumerate(sr.JOINTS) if j.startswith("TH")]
        self.assertGreater(float(d[:, thumb].max()), 0.0)
        self.assertEqual(float(d[:, [i for i in range(20) if i not in thumb]].max()), 0.0)

    def test_union_layout_embeds_each_policy_exactly(self):
        e0 = st.validate_edit({"expr": "tilt_up", "target": "wrist"}, self.root)
        e1 = st.validate_edit({"expr": ["abs", "centre_y"], "target": "groups"}, self.root)
        c0, c1 = st.extend(self.root, e0), st.extend(self.root, e1)
        c1 = st.policy_record(c1["terms"], c1["theta"][:32] + [0.03] * 6)
        terms, thetas = ss.union_layout([self.root, c0, c1])
        self.assertEqual(len(terms), 5)
        self.assertEqual([len(t) for t in thetas], [40, 40, 40])
        np.testing.assert_array_equal(thetas[0][32:], 0.0)
        np.testing.assert_allclose(thetas[2][34:], 0.03, rtol=1e-6)
        union = st.build_controller(CFG, "u", st.policy_record(terms, [0.0] * 40), NORM, inner=Incumbent())
        own = st.build_controller(CFG, "o", c1, NORM, inner=Incumbent())
        union.reset(ctx(8))
        own.reset(ctx(8))
        o = obs_batch(8, 5)
        th = torch.tensor(np.stack([thetas[2]] * 8), dtype=torch.float32)
        self.assertTrue(torch.equal(union.mean(union.parts(o, 0, None), th), own.mean(own.parts(o, 0, None), own.theta)))

    def test_stored_policy_must_match_its_hash(self):
        bad = dict(self.root, theta=[0.0] * 32)
        with self.assertRaises(SystemExit):
            st.check_policy(bad)


class TestLanguage(unittest.TestCase):
    def setUp(self):
        self.root = ss.root_policy(CFG)

    def test_types_depth_and_limits(self):
        self.assertEqual(st.expr_type(["mul", "spin_deficit", "tip_dist"]), "unit")
        self.assertEqual(st.expr_type(["mul", "spin_deficit", "vel"]), "signed")
        self.assertEqual(st.depth(["gate", ["comp", "tip_dist"], 0.5, "above"]), 3)
        for bad in (["gate", "vel", 0.5, "above"], ["gate", "tip_dist", 0.4, "above"], ["comp", "vel"], ["abs", "tip_dist"],
                    ["mul", "vel"], "torque", ["exp", "vel"], ["gate", ["gate", ["comp", "tip_dist"], 0.5, "above"], 0.5, "below"]):
            with self.assertRaises(st.EditError):
                st.validate_edit({"expr": bad, "target": "groups"}, self.root)
        with self.assertRaises(st.EditError):
            st.validate_edit({"expr": "tilt_up", "target": "joints"}, self.root)           # 20 new coefficients
        with self.assertRaises(st.EditError):
            st.validate_edit({"expr": "vel", "target": "groups"}, self.root)               # already in the structure
        with self.assertRaises(st.EditError):
            st.validate_edit({"expr": "tilt_up", "target": "wrist"}, self.root, tried=[{"expr": "tilt_up", "target": "wrist"}])
        full = st.policy_record(list(self.root["terms"]) + [{"expr": "tilt_up", "target": "groups"}, {"expr": "pos", "target": "groups"}],
                                self.root["theta"] + [0.0] * 12)
        with self.assertRaises(st.EditError):
            st.validate_edit({"expr": "ref", "target": "groups"}, full)                    # 44 + 6 > 48
        st.validate_edit({"expr": "ref", "target": "chain:LF"}, full)

    def test_activation_check(self):
        feats = sample()
        with self.assertRaises(st.EditError):                                              # vel is 0.2 / (3 rms): never above 0.75
            st.validate_edit({"expr": ["gate", ["abs", "vel"], 0.75, "above"], "target": "wrist"},
                             self.root, feats=st.sample_features({"obs": np.zeros((50, 157), np.float32), "ref": np.zeros((50, 20), np.float32),
                                                                  "v_f": np.zeros((50, 20), np.float32)}, NORM, CFG["actor"]))
        self.assertGreater(st.activation_fraction({"expr": "spin_deficit", "target": "groups"}, feats), 0.9)

    def test_every_primitive_evaluates_in_its_range(self):
        feats = sample()
        for name, (kind, _) in st.PRIMITIVES.items():
            v = feats[name]
            self.assertIn(v.shape[1], (1, 20), name)
            self.assertTrue(bool(torch.isfinite(v).all()), name)
            self.assertLessEqual(float(v.max()), 1.0 + 1e-6, name)
            self.assertGreaterEqual(float(v.min()), (0.0 if kind == "unit" else -1.0) - 1e-6, name)
        self.assertEqual(float(feats["tip_dist"][:, :2].abs().max()), 0.0)                 # the wrist has no fingertip

    def test_grammar_is_seeded_valid_and_varied(self):
        feats = sample()
        a = [st.sample_edit(np.random.default_rng(s), self.root, (), feats) for s in range(40)]
        b = [st.sample_edit(np.random.default_rng(s), self.root, (), feats) for s in range(40)]
        self.assertEqual(a, b)
        for e in a:
            self.assertEqual(st.validate_edit(e, self.root, (), feats), e)
            self.assertLessEqual(st.n_slots(e["target"]), 8)
        self.assertGreater(len({json.dumps(e) for e in a}), 25)
        self.assertEqual({st.depth(e["expr"]) for e in a}, {1, 2, 3})

    def test_ars_at_any_dimension(self):
        theta = np.zeros(37)
        target = np.full(37, 0.05)
        for u in range(60):
            d = st.ars_directions(u, 4, 37)
            cand = st.ars_candidates(theta, d, 0.02, 0.10)
            theta, _ = st.ars_step(theta, d, [-np.sum((c - target) ** 2) for c in cand], 0.01, 0.10, 4)
        self.assertLess(np.linalg.norm(theta - target), 0.8 * np.linalg.norm(target))
        same, info = st.ars_step(theta, d, [1.0] * 8, 0.01, 0.10)
        np.testing.assert_array_equal(same, theta)


def rows(parent, c0, c1, dup=None, drops=None):
    mk = lambda s, d=None: {"sustained": list(map(int, s)), "completed": [1] * len(s) if d is None else list(map(int, d))}   # noqa: E731
    return {"parent": mk(parent), "c0": mk(c0, drops), "c1": mk(c1), "parent_dup": mk(parent if dup is None else dup)}


class TestSelectionAndSchedule(unittest.TestCase):
    def test_gate(self):
        rng = np.random.default_rng(0)
        par = rng.random(1536) < 0.5
        better = par | (rng.random(1536) < 0.12)
        noise = np.where(rng.random(1536) < 0.1, ~par, par)
        d = ss.selection_decision(rows(par, noise, better), 0.05)
        self.assertEqual(d["selected"], "c1")
        self.assertFalse(d["candidates"]["c0"]["passes"])
        self.assertEqual(ss.selection_decision(rows(par, noise, par), 0.05)["selected"], "parent")       # a tie never replaces
        both = ss.selection_decision(rows(par, better, par | (rng.random(1536) < 0.3)), 0.05)
        self.assertEqual(both["selected"], "c1")                                                           # the larger gain
        dropped = np.ones(1536, dtype=int)
        dropped[:200] = 0
        self.assertEqual(ss.selection_decision(rows(par, better, par, drops=dropped), 0.05)["selected"], "parent")
        self.assertEqual(d["duplicate_parent"]["discordance"], 0.0)

    def test_amended_rule_keeps_the_best_within_the_drop_bound(self):
        rng = np.random.default_rng(1)
        par = rng.random(1536) < 0.5
        def shifted(gain, loss):                                                            # net +gain-loss, many discordant cases
            out = par.copy()
            out[np.flatnonzero(~par)[:gain]] = True
            out[np.flatnonzero(par)[:loss]] = False
            return out

        small, smaller = shifted(150, 120), shifted(130, 120)                               # +30 and +10: not Holm-significant
        r = rows(par, smaller, small)
        self.assertEqual(ss.selection_decision(r, 0.05)["selected"], "parent")
        d = ss.selection_decision(r, 0.05, rule=ss.RULE_BEST)
        self.assertEqual((d["selected"], d["selected_under_registered_strict_rule"]), ("c1", "parent"))
        self.assertEqual(ss.selection_decision(rows(par, par, par), 0.05, rule=ss.RULE_BEST)["selected"], "parent")   # ties: parent
        self.assertEqual(ss.selection_decision(rows(par, small, small), 0.05, rule=ss.RULE_BEST)["selected"], "c0")   # then the earlier
        dropped = np.ones(1536, dtype=int)
        dropped[:200] = 0
        self.assertEqual(ss.selection_decision(rows(par, small, smaller, drops=dropped), 0.05, rule=ss.RULE_BEST)["selected"], "c1")
        self.assertEqual(CFG["selection_rule"], ss.RULE_BEST)

    def test_update_mapping_gives_no_candidate_more_than_six_updates(self):
        jobs = ss._jobs(CFG, "grammar", ss.root_policy(CFG), [{"expr": "tilt_up", "target": "wrist"}, None])
        self.assertEqual([j["updates"] for j in jobs], [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]])
        self.assertEqual(len(jobs[0]["policy"]["theta"]), 34)
        self.assertEqual(jobs[1]["policy"]["sha256"], ss.root_policy(CFG)["sha256"])                      # an invalid edit: inherited no-op
        fixed = ss._jobs(CFG, "fixed", ss.root_policy(CFG), [None, None])
        self.assertEqual((fixed[0]["updates"], fixed[0]["snapshots"]), (list(range(12)), [6, 12]))
        a, b = ss.training_placements(CFG, 0, 0, 11), ss.training_placements(CFG, 0, 1, 0)
        self.assertFalse(np.array_equal(a, b))
        self.assertEqual(len({ss.direction_seed(CFG, b, r, u) for b in range(3) for r in range(2) for u in range(12)}), 72)

    def test_frozen_files_and_budget(self):
        self.assertEqual(ss.freeze_manifests(CFG)["written"], [])
        self.assertEqual(ss.placement_provenance(CFG)["placements"], 3 * 12288 + 18 * 512 + 512)
        self.assertEqual(sorted(s for b in CFG["selection_seeds"] for r in b for s in r), list(range(4838, 4856)))
        self.assertEqual(sorted(s for b in CFG["confirmation_seeds"] for s in b), list(range(4856, 4892)))
        est = 3 * 3 * 24 * 8200 + 3 * 3 * 2 * 3 * 4100 + 2 * 5 * 36 * 128 + 8200
        self.assertLessEqual(est, CFG["start_ceiling"])
        self.assertEqual(ss.config_sha256(CFG), ss.config_sha256({**CFG, "model": "x", "isaac_env": "y"}))

    def test_prompt_and_parsing(self):
        root = ss.root_policy(CFG)
        text = ss.edit_prompt(root, {"success": 0.52, "drops": 20}, [{"edit": {"expr": "tilt_up", "target": "wrist"}, "outcome": "not selected"}])
        for needle in ("EDIT LANGUAGE", "tip_dist", "term 2", "not selected", '"success": 0.52'):
            self.assertIn(needle, text)
        self.assertNotIn("confirm", text.lower())
        got = ss.parse_edit('Here: {"expr": ["mul", "spin_deficit", "tip_dist"], "target": "chain:TH", "rationale": "x"}')
        self.assertEqual(got["target"], "chain:TH")
        with self.assertRaises(st.EditError):
            ss.parse_edit("no json here")

    def test_generator_uses_one_correction_then_gives_a_no_op(self):
        import tempfile

        replies = iter(['{"expr": "torque", "target": "groups"}', '{"expr": "tilt_up", "target": "wrist", "rationale": "r"}',
                        "nonsense", "still nonsense"])

        class Client:
            class chat:
                class completions:
                    @staticmethod
                    def create(**kw):
                        msg = SimpleNamespace(message=SimpleNamespace(content=next(replies)))
                        return SimpleNamespace(choices=[msg], usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2))

        with tempfile.TemporaryDirectory() as d:
            gen = ss.EditGenerator(Client(), {**CFG, "model": "m", "reasoning_effort": "high"}, d)
            root = ss.root_policy(CFG)
            ok = gen.propose(root, {}, [], None, "t")
            self.assertEqual((ok["edit"], len(ok["calls"]), len(ok["errors"])), ({"expr": "tilt_up", "target": "wrist"}, 2, 1))
            bad = gen.propose(root, {}, [], None, "t")
            self.assertIsNone(bad["edit"])
            self.assertEqual(len(bad["calls"]), 2)
            self.assertEqual(gen.ledger.totals()["llm_calls"], 4)


if __name__ == "__main__":
    unittest.main()
