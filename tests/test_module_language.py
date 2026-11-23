"""R-131: the recovery-module language: schema, bounds, identity inheritance, memory reset, the sampler, and the
interpreter against the existing hand-written module (no Isaac)."""

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lares.envs.isaac_pen_spin import module_language as ml  # noqa: E402
from lares.envs.isaac_pen_spin import module_search as ms  # noqa: E402
from lares.envs.isaac_pen_spin import recovery_module as rm  # noqa: E402
from lares.utils.cli import load_yaml  # noqa: E402
from tests.test_recovery_module import Incumbent, ctx, obs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
CFG = load_yaml(os.path.join(ROOT, "config", "pen_spin_plateau_repair.yaml"))[ms.CONFIG_KEY]


def signals(n=64, steps=360, seed=0):
    """Rolling, slow and mixed cases, a few episode ends."""
    rng = np.random.default_rng(seed)
    base = rng.choice([1.0, 2.5, 3.5, 8.0], size=n)
    rate = np.abs(base[None] + 0.8 * rng.standard_normal((steps, n)))
    rate[200:] += rng.choice([0.0, 5.0], size=n)[None]
    done = np.zeros((steps, n), dtype=bool)
    done[rng.integers(50, 300, 6), rng.integers(0, n, 6)] = True
    return {"rate": rate.astype(np.float32), "centre": np.abs(0.03 + 0.03 * rng.standard_normal((steps, n))).astype(np.float32),
            "transverse": np.abs(1.5 + rng.standard_normal((steps, n))).astype(np.float32), "done": done}


def with_values(spec, **values):
    """A record of ``spec`` at its identity point with some slots set to physical values."""
    idx = ml.active(spec)
    x = ml.initial_x(spec)
    for name, v in values.items():
        k = ml.I[name]
        x[idx.index(k)] = (v - ml.LOW[k]) / (ml.HIGH[k] - ml.LOW[k])
    return ml.record(spec, x)


def edited(*edits):
    spec = ml.TEMPLATE
    for e in edits:
        spec = ml.apply_edit(spec, e)
    return spec


class TestLanguage(unittest.TestCase):
    def test_template_is_the_scripted_module(self):
        t = ml.template_record()
        np.testing.assert_allclose(t["x"], rm.SCRIPT_X)
        self.assertEqual((t["parameters"], t["edits"], ml.N_SLOTS), (11, 0, 27))
        self.assertEqual([s[:3] for s in ml.SLOTS[:11]], [p[:3] for p in rm.PARAMS])
        self.assertEqual(ml.check_record(t)["sha256"], t["sha256"])
        with self.assertRaises(SystemExit):
            ml.check_record({**t, "x": [0.1] * 11})

    def test_edits_schema_and_caps(self):
        edits = ml.valid_edits(ml.TEMPLATE)
        self.assertEqual(len(edits), 13)
        for e in edits:
            child = ml.apply_edit(ml.TEMPLATE, e)
            added = len(ml.active(child)) - 11
            self.assertTrue(1 <= added <= 4, e)
            self.assertEqual(ml.n_edits(child), 1)
            with self.assertRaises(ml.EditError):                 # the same component cannot be changed twice
                ml.apply_edit(child, e)
        for bad in ({"component": "schedule", "value": "single"}, {"component": "nope", "value": 1}, {"component": "target"},
                    {"component": "conditions", "add": {"primitive": "rate", "op": "lt"}}, {"component": "attempts", "value": 3},
                    {"component": "schedule", "value": "periodic", "period": 3}, "x"):
            with self.assertRaises(ml.EditError):
                ml.apply_edit(ml.TEMPLATE, bad)
        two = edited({"component": "conditions", "add": {"primitive": "trend", "op": "gt"}}, {"component": "conditions", "add": {"primitive": "deficit", "op": "lt"}})
        with self.assertRaises(ml.EditError):                     # at most two extra conditions
            ml.apply_edit(two, {"component": "conditions", "add": {"primitive": "transverse", "op": "lt"}})
        three = edited({"component": "schedule", "value": "periodic"}, {"component": "boost_end", "value": "event"}, {"component": "attempts", "value": 2})
        self.assertEqual(len(ml.active(three)), 17)
        with self.assertRaises(ml.EditError):                     # three edits deep at most
            ml.apply_edit(three, {"component": "target", "value": "hold"})
        self.assertLessEqual(max(len(ml.active(ml.apply_edit(ml.apply_edit(ml.TEMPLATE, a), b))) for a in edits for b in edits
                                 if ml.edit_key(b) in {ml.edit_key(e) for e in ml.valid_edits(ml.apply_edit(ml.TEMPLATE, a))}), ml.LIMITS["parameters"])

    def test_parameters_stay_in_their_bounds(self):
        spec = edited({"component": "boost_end", "value": "event"})
        p = ml.full_params(spec, np.array([[-3.0] * 13, [7.0] * 13]))
        idx = ml.active(spec)
        np.testing.assert_array_equal(p[0, idx], ml.LOW[idx])
        np.testing.assert_array_equal(p[1, idx], ml.HIGH[idx])
        self.assertEqual(p[0, ml.I["max_hold"]], 0.0)             # an inactive slot stays at its identity value
        with self.assertRaises(ml.EditError):
            ml.full_params(spec, [0.5] * 11)

    def test_inheritance_keeps_the_parents_parameters(self):
        parent = ml.record(ml.TEMPLATE, np.linspace(0.1, 0.9, 11))
        child = ml.apply_edit(ml.TEMPLATE, {"component": "schedule", "value": "two_step"})
        x = ml.inherit(parent, child)
        self.assertEqual(len(x), 13)
        np.testing.assert_array_equal(x[:11], parent["x"])
        self.assertAlmostEqual(ml.full_params(child, x)[0, ml.I["second_scale"]], 0.0)
        grand = ml.apply_edit(child, {"component": "conditions", "add": {"primitive": "transverse", "op": "lt"}})
        gx = ml.inherit(ml.record(child, np.r_[parent["x"], 0.3, 0.6]), grand)
        full = ml.full_params(grand, gx)[0]
        self.assertAlmostEqual(full[ml.I["second_scale"]], 0.6)
        self.assertAlmostEqual(full[ml.I["level_transverse"]], 10.0)   # "lt" starts at the top: always true

    def test_every_edit_is_an_identity_until_trained(self):
        sig = signals()
        parent = ml.record(ml.TEMPLATE, np.r_[0.9, 0.5, 0.4, 0.2, np.full(6, 0.7), 0.5])
        self.assertGreater(ml.replay(parent["spec"], parent["x"], sig)["fired"].sum(), 5)
        for e in ml.valid_edits(ml.TEMPLATE):
            child = ml.apply_edit(ml.TEMPLATE, e)
            r = ml.replay_identity(parent, child, sig)
            self.assertTrue(r["passes"], (e, r))
            for e2 in ml.valid_edits(child)[:4]:                    # and a second edit on a trained first edit
                p2 = ml.record(child, np.clip(ml.inherit(parent, child) + 0.2, 0, 1))
                self.assertTrue(ml.replay_identity(p2, ml.apply_edit(child, e2), sig)["passes"], (e, e2))

    def test_each_component_acts_when_trained(self):
        sig = signals()
        base = ml.record(ml.TEMPLATE, np.r_[0.9, 0.5, 0.4, 0.2, np.full(6, 0.7), 0.5])
        ref = ml.replay(base["spec"], base["x"], sig)

        def out(rec):
            return ml.replay(rec["spec"], rec["x"], sig)

        def like(spec, **v):
            x = ml.inherit(base, spec)
            idx = ml.active(spec)
            for name, val in v.items():
                k = ml.I[name]
                x[idx.index(k)] = (val - ml.LOW[k]) / (ml.HIGH[k] - ml.LOW[k])
            return ml.record(spec, x)

        self.assertGreater(out(like(edited({"component": "schedule", "value": "periodic"}), extra_window=120.0))["fired"].sum(), ref["fired"].sum())
        self.assertGreater(out(like(edited({"component": "schedule", "value": "two_step"}), second_scale=1.0))["fired"].sum(), ref["fired"].sum())
        self.assertGreater(out(like(edited({"component": "boost_end", "value": "event"}), max_extra=100.0, release_level=10.0))["boost"].sum(), ref["boost"].sum())
        self.assertGreater(out(like(edited({"component": "support_end", "value": "event"}), max_hold=40.0, settle_level=0.0))["beta"].sum(), ref["beta"].sum())
        self.assertGreater(out(like(edited({"component": "attempts", "value": 2}), retry_depth=1.0))["beta"].sum(), ref["beta"].sum())
        for prim, op, lvl in (("transverse", "lt", 0.0), ("trend", "gt", 4.0), ("deficit", "gt", 1.0)):
            blocked = out(like(edited({"component": "conditions", "add": {"primitive": prim, "op": op}}), **{f"level_{prim}": lvl}))
            self.assertLess(blocked["fired"].sum(), ref["fired"].sum(), prim)

    def test_memory_resets_per_environment(self):
        st = ml.ModuleState(4, "cpu")
        for _ in range(140):
            st(torch.full((4,), 1.0), torch.full((4,), 0.01), torch.full((4,), 2.0))
        self.assertTrue(st.fired.all() and (st.deficit > 0).all())
        st.reset(torch.tensor([True, False, True, False]))
        self.assertEqual(st.fired.tolist(), [False, True, False, True])
        self.assertEqual(st.step.tolist(), [0, 140, 0, 140])
        for name in ("hist", "thist", "deficit", "hold_used", "extra_used", "boost_steps"):
            self.assertEqual(float(getattr(st, name)[0].abs().sum()), 0.0, name)
            self.assertGreater(float(getattr(st, "hist")[1].abs().sum()), 0.0)

    def test_sampler_is_uniform_without_replacement(self):
        rng = np.random.default_rng(3)
        draws = [ml.edit_key(ml.sample_edits(rng, ml.TEMPLATE, 1)[0]) for _ in range(2600)]
        counts = np.array([draws.count(ml.edit_key(e)) for e in ml.valid_edits(ml.TEMPLATE)])
        self.assertTrue((np.abs(counts - 200) < 60).all(), counts)
        two = ml.sample_edits(np.random.default_rng(1), ml.TEMPLATE, 2)
        self.assertNotEqual(ml.edit_key(two[0]), ml.edit_key(two[1]))
        left = ml.sample_edits(np.random.default_rng(1), ml.TEMPLATE, 13, tried=two)
        self.assertEqual(len(left), 11)
        self.assertEqual(ml.sample_edits(np.random.default_rng(1), ml.TEMPLATE, 2), two)


class TestController(unittest.TestCase):
    def _pair(self, x, retry=False):
        spec = edited({"component": "attempts", "value": 2}) if retry else ml.TEMPLATE
        mine = ms.build_controller(CFG, "mine", ml.record(spec, x), inner=Incumbent())
        theirs = rm.build(CFG, "theirs", x, inner=Incumbent(), original_bounds=True)
        mine.reset(ctx(4))
        theirs.reset(ctx(4))
        return mine, theirs

    def _same(self, x, retry=False):
        mine, theirs = self._pair(x, retry)
        worst, fired = 0.0, 0
        for t in range(340):
            o = obs(4, [1.0 if t < 250 else 9.0, 6.0, 3.0 if t < 200 else 1.0, 2.0])
            a, b = mine(o, t, None), theirs(o, t, None)
            worst = max(worst, float((a - b).abs().max()))
        return worst, mine.totals["fired"], int(theirs.triggered), int(theirs.triggered2)

    def test_template_is_the_hand_written_module(self):
        worst, mine, theirs, _ = self._same(rm.SCRIPT_X)
        self.assertLessEqual(worst, 1e-5)
        self.assertEqual((mine, theirs), (2, 2))
        worst, mine, theirs, _ = self._same(np.r_[0.7, 0.3, 0.37, 0.13, 0.4, 0.9, 0.6, 0.7, 0.0, 0.3, 0.45])
        self.assertLessEqual(worst, 1e-5)
        self.assertEqual(mine, theirs)

    def test_second_attempt_is_r118s(self):
        x = np.r_[rm.SCRIPT_X, 0.25, 0.8]
        worst, _, _, second = self._same(x, retry=True)
        self.assertLessEqual(worst, 1e-5)
        self.assertGreater(second, 0)

    def test_hold_target_and_counters(self):
        spec = edited({"component": "target", "value": "hold"})
        c = ms.build_controller(CFG, "c", with_values(spec, mix_hold=1.0), inner=Incumbent())
        plain = ms.build_controller(CFG, "p", ml.template_record(), inner=Incumbent())
        c.reset(ctx(2))
        plain.reset(ctx(2))
        diff = 0.0
        for t in range(200):
            o = obs(2, [1.0, 9.0])
            o[:, :20] = 0.01 * t                                   # the root action drifts, so a held action differs from the reset posture
            diff = max(diff, float((c(o, t, None) - plain(o, t, None)).abs().max()))
        self.assertGreater(diff, 1e-3)
        self.assertEqual((c.totals["fired"], c.totals["steps"]), (1, 400))
        c.extra_trace()
        self.assertEqual(c.params["recovery"]["triggered_environment_episodes"], 1)
        self.assertEqual(c.params["module"]["spec"]["target"], "hold")


if __name__ == "__main__":
    unittest.main()
