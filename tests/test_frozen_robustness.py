"""R-129: the panel, the frozen assets, the condition hook, the six-second rule inside longer runs and the inference
(no Isaac)."""

import json
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lares.envs.isaac_pen_spin import cases  # noqa: E402
from lares.envs.isaac_pen_spin import frozen_robustness as fr  # noqa: E402
from lares.envs.isaac_pen_spin import recovery_module as rm  # noqa: E402
from lares.utils.cli import load_yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
CFG = load_yaml(os.path.join(ROOT, "config", "shadowhand_spin_stages.yaml"))["research_protocols"][fr.STUDY]


def trace(n=4, steps=359, hit=None, length=None):
    T = steps + 1
    pred = np.ones((T, n), dtype=bool) if hit is None else hit
    L = np.full(n, steps) if length is None else np.asarray(length)
    rows = [{"task_hash": f"h{i}", "sustained": cases.sustained(pred[:L[i], i], int(L[i]), bool(L[i] == steps)),
             "tail_hit_fraction": float(pred[max(0, L[i] - 60):L[i], i].mean()), "final_center_dist": 0.01} for i in range(n)]
    return {"pred": pred, "length": L, "completed": L == steps, "spin_rate": np.full((T, n), 6.0)}, rows


class TestPlan(unittest.TestCase):
    def test_cells_and_cost_are_the_registered_ones(self):
        cells = fr.plan(CFG)
        self.assertEqual(len(cells), 204)
        self.assertEqual(len(set(cells)), 204)
        steps = [fr.horizon_steps(CFG, k) + 1 for _, _, k in cells]
        self.assertEqual((steps.count(360), steps.count(720), steps.count(1800)), (172, 16, 16))
        self.assertEqual(128 * sum(steps), 13_086_720)
        self.assertEqual(sum(1 for c in cells if c[1:] == fr.PRIMARY), 10)
        self.assertEqual(len(fr.conditions(CFG)), 12)
        self.assertEqual(fr.build_manifests(CFG), [])
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import run_shadowhand_spin_stages as rs

        self.assertEqual(rs.PEN_SPIN_STUDIES[fr.STUDY], "frozen_robustness")

    def test_assets_are_the_frozen_ones(self):
        self.assertEqual((len(fr.asset_x(CFG, "joint")), len(fr.asset_x(CFG, "transplant"))), (43, 11))
        np.testing.assert_array_equal(fr.asset_x(CFG, "transplant"), fr.asset_x(CFG, "joint")[:11])
        with self.assertRaises(SystemExit):
            fr.asset_x({**CFG, "assets": {**CFG["assets"], "joint": {**CFG["assets"]["joint"], "sha256": "0" * 64}}}, "joint")
        for bad in ("joint", "joint@nope", "x@nominal"):
            with self.assertRaises(ValueError):
                fr.make_controller(bad, CFG)

    def test_a_cell_carries_its_condition_and_leaves_the_controller_alone(self):
        inner = SimpleNamespace(params={"a": 1}, reset=mock.Mock(), __call__=None)
        with mock.patch.object(fr, "base_controller", return_value=inner):
            c = fr.make_controller("joint@friction_0.8", CFG)
        self.assertEqual((c.name, c.env_condition, c.params), ("joint@friction_0.8", {"friction": 0.8}, {"a": 1, "condition": {"friction": 0.8}}))
        c.reset("ctx")
        inner.reset.assert_called_once_with("ctx")
        with mock.patch.object(fr, "base_controller", return_value=inner):
            self.assertEqual(fr.make_controller("ppo@nominal", CFG).env_condition, {})


class TestCondition(unittest.TestCase):
    def _cfg(self):
        mat = SimpleNamespace(static_friction=1.2, dynamic_friction=1.0)
        mat.replace = lambda **kw: SimpleNamespace(**{**vars(mat), **kw})
        spawn = SimpleNamespace(mass_props=SimpleNamespace(mass=0.035), physics_material=mat, radius=0.012, height=0.14)
        return SimpleNamespace(object_cfg=SimpleNamespace(spawn=spawn), episode_length_s=6.0,
                               goal_object_cfg=SimpleNamespace(markers={"goal": SimpleNamespace(radius=0.012, height=0.14)}))

    def test_each_factor_changes_only_itself(self):
        from lares.envs.isaac_pen_spin.cases import apply_condition

        for name, cond in fr.conditions(CFG).items():
            cfg = self._cfg()
            apply_condition(cfg, cond)
            sp, want = cfg.object_cfg.spawn, fr.expected_physical(CFG, name)
            got = {"mass": sp.mass_props.mass, "static_friction": sp.physics_material.static_friction,
                   "dynamic_friction": sp.physics_material.dynamic_friction, "radius": sp.radius, "length": sp.height,
                   "max_episode_length": round(cfg.episode_length_s * 60)}
            for k in want:
                self.assertAlmostEqual(got[k], want[k], places=9, msg=(name, k))
            self.assertTrue(fr.preflight_check(CFG, name, {**got, "control_dt": CFG["control_dt"], "errors": []})["passes"], name)
        with self.assertRaises(ValueError):
            apply_condition(self._cfg(), {"gravity": 2.0})
        bad = fr.preflight_check(CFG, "mass_1.2", {**fr.expected_physical(CFG, "nominal"), "control_dt": CFG["control_dt"], "errors": []})
        self.assertFalse(bad["passes"])
        self.assertIn("mass", bad["problems"][0])


class TestMetrics(unittest.TestCase):
    def test_six_second_rule_is_the_evaluators_on_a_nominal_trace(self):
        rng = np.random.default_rng(0)
        hit = rng.random((360, 64)) < 0.995
        length = np.where(rng.random(64) < 0.2, rng.integers(1, 359, 64), 359)
        tr, rows = trace(64, 359, hit, length)
        six = fr.six_second_success(tr["pred"], tr["length"])
        self.assertEqual([bool(v) for v in six], [r["sustained"] for r in rows])
        self.assertTrue(0 < six.sum() < 64)

    def test_a_long_run_scores_its_first_six_seconds_on_the_same_steps(self):
        hit = np.ones((720, 4), dtype=bool)
        hit[298, 0] = False                                   # just outside the window: still a six-second success
        hit[299, 1] = False                                   # inside it
        hit[700, 2] = False                                   # fails only the 12 s endpoint
        tr, rows = trace(4, 719, hit, [719, 719, 719, 400])
        m = fr.case_metrics(tr, rows, 719)
        self.assertEqual([r["six_second"] for r in m], [True, False, True, True])
        self.assertEqual([r["terminal"] for r in m], [True, True, False, False])
        self.assertEqual([r["drop_step"] for r in m], [None, None, None, 400])
        self.assertEqual(m[0]["longest_rolling_steps"], 420)
        self.assertEqual(len(m[0]["windows"]), 12)
        self.assertEqual((m[1]["windows"][4], m[1]["windows"][5], m[0]["windows"][4], m[0]["windows"][5]), (True, False, False, True))
        s = fr.cell_summary(m, 719)
        self.assertEqual((s["six_second_success"], s["terminal_success"], s["drops"], s["six_second_labels"]), (0.75, 0.5, 1, None))
        self.assertEqual(s["drops_by_second"][6], 1)
        self.assertEqual(fr.cell_summary(fr.case_metrics(*trace(), 359), 359)["six_second_labels"]["success"], 4)

    def test_window_alignment(self):
        w = fr.window_success(np.ones((360, 1), dtype=bool), 359)
        self.assertEqual(w.shape, (6, 1))
        p = np.ones((360, 1), dtype=bool)
        p[58] = False
        self.assertEqual([bool(v) for v in fr.window_success(p, 359)[:, 0]], [False, True, True, True, True, True])


class TestInference(unittest.TestCase):
    def _cells(self, rate, seed=0):
        rng = np.random.default_rng(seed)
        cells = {}
        for s, c, k in fr.plan(CFG):
            u = np.random.default_rng(s).random(128)          # shared case difficulty across controllers and conditions
            steps = fr.horizon_steps(CFG, k)
            q = rate(c, k)
            six = u < q
            term = six & (rng.random(128) < (0.9 if steps > 359 else 1.0))
            cells[(s, c, k)] = [{"task_hash": f"h{s}_{i}", "sustained": bool(six[i]), "six_second": bool(six[i]), "terminal": bool(term[i]),
                                 "completed": True, "length": steps, "survived_six_seconds": True, "drop_step": None,
                                 "rolling_fraction": 0.8, "longest_rolling_steps": 300, "final_second_abs_spin": 7.0,
                                 "tail_hit_fraction": float(six[i]), "final_center_dist": 0.01, "windows": [bool(six[i])] * ((steps + 1) // 60)}
                                for i in range(128)]
        return cells

    def test_verdict_rule(self):
        self.assertEqual(fr.primary_verdict(-0.03, 0.01, -0.05), "supported")
        self.assertEqual(fr.primary_verdict(-0.12, -0.07, -0.05), "refuted")
        self.assertEqual(fr.primary_verdict(-0.07, -0.02, -0.05), "inconclusive")

    def test_analysis_family_and_primary(self):
        base = {"joint": 0.79, "transplant": 0.78, "d072": 0.43, "ppo": 0.97}
        out = fr.analyse(self._cells(lambda c, k: base[c] - (0.2 if k == "mass_1.2" else 0.0)), CFG)
        self.assertEqual(out["primary"]["pairs"], 1280)
        self.assertEqual(out["primary"]["verdict"], "supported")
        self.assertEqual(out["secondary_tests"], 4 * 9 - 1 + 4 * 2 + 3 * 12)
        self.assertIn("joint@mass_1.2_minus_joint@nominal", out["secondary_rejections"])
        self.assertIn("joint@horizon_30s_terminal_minus_six_second", out["secondary_rejections"])
        self.assertNotIn("joint@radius_0.9_minus_joint@nominal", out["secondary_rejections"])
        self.assertEqual(out["nominal_prefix_consistency"]["joint@horizon_12s"]["agreement"], 1.0)
        self.assertEqual(out["table"]["joint"]["nominal"]["cases"], 1280)
        self.assertEqual(out["table"]["ppo"]["nominal"]["cases"], 512)
        json.dumps(out)
        bad = fr.analyse(self._cells(lambda c, k: base[c] - (0.15 if k == "friction_0.8" else 0.0)), CFG)
        self.assertEqual(bad["primary"]["verdict"], "refuted")
        self.assertAlmostEqual(bad["paired_success_difference"], -0.15, delta=0.03)


if __name__ == "__main__":
    unittest.main()
