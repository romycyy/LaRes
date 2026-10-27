"""Pen-spin evaluation contract (R-038): manifests, reducers, rotation helpers. No Isaac Lab needed."""

import math
import os
import re
import sys
import tempfile
import unittest

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from lares.envs.isaac_pen_spin import cases  # noqa: E402
from lares.eval.manifest import EvaluationManifest  # noqa: E402

_CFG_SRC = os.path.join(_PROJECT_ROOT, "lares", "envs", "isaac_pen_spin", "shadow_hand_spin_env_cfg.py")


def _axis_angle_quat(axis, angle):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    return np.concatenate([[math.cos(angle / 2)], math.sin(angle / 2) * axis])


class TestManifests(unittest.TestCase):
    def test_pure_function_of_seed(self):
        a = cases.build_manifest(1, 4000, 10)
        b = cases.build_manifest(1, 4000, 10)
        self.assertEqual(a.to_dict(), b.to_dict())
        c = cases.build_manifest(1, 4001, 10)
        self.assertFalse(a.task_hashes() & c.task_hashes())

    def test_roundtrip_and_resolve(self):
        m = cases.build_manifest(2, 4001, 5)
        with tempfile.TemporaryDirectory() as d:
            path = m.save(os.path.join(d, "m.yaml"))
            loaded = EvaluationManifest.load(path)
        vecs = cases.resolve_placements(loaded)
        self.assertEqual(vecs.shape, (5, cases.PLACEMENT_DIM))
        self.assertTrue(np.all(np.abs(vecs) <= 1.0))
        self.assertEqual(cases.stage_of(loaded), 2)

    def test_drift_refused(self):
        m = cases.build_manifest(1, 4000, 3)
        d = m.to_dict()
        d["episodes"][1]["task_hash"] = "0" * 16
        with self.assertRaises(ValueError):
            cases.resolve_placements(EvaluationManifest.from_dict(d))

    def test_head_keeps_pool_identity(self):
        m = cases.build_manifest(3, 4002, 10)
        self.assertEqual(len(cases.resolve_placements(m.head(4))), 4)

    def test_metaworld_resolve_pool_refused(self):
        with self.assertRaises(Exception):
            cases.build_manifest(1, 4000, 2).resolve_pool()


class TestStageThresholds(unittest.TestCase):
    def test_thresholds_match_vendored_cfg(self):
        with open(_CFG_SRC, encoding="utf-8") as f:
            src = f.read()
        blocks = re.split(r"\nclass ", src)
        values = {}
        for block in blocks[1:]:
            name = block.split("(")[0]
            values[name] = {k: float(v) for k, v in re.findall(r"^\s+(success_spin_rate|success_center_tolerance|fall_dist) = ([0-9.]+)", block, re.M)}
        cur = {}
        for k in (1, 2, 3):
            cur.update(values[cases.STAGES[k].cfg_class])  # stages inherit from the previous one
            s = cases.STAGES[k]
            self.assertEqual(s.success_spin_rate, cur["success_spin_rate"])
            self.assertEqual(s.success_center_tolerance, cur["success_center_tolerance"])
            self.assertEqual(s.fall_dist, cur["fall_dist"])


class TestObsLayout(unittest.TestCase):
    def test_157_contiguous(self):
        sl = cases.obs_slices()
        self.assertEqual(sl["hand_dof_pos"], slice(0, 24))
        self.assertEqual(sl["obj_angvel"], slice(58, 61))
        self.assertEqual(sl["last_action"], slice(137, 157))


class TestRotation(unittest.TestCase):
    def test_roll_versus_end_over_end(self):
        # Pen tilted 30 degrees about z; its long axis is the rotated x.
        q = _axis_angle_quat([0, 0, 1], math.radians(30))
        axis = cases.pen_axis(q)
        np.testing.assert_allclose(axis, [math.cos(math.radians(30)), math.sin(math.radians(30)), 0], atol=1e-12)
        roll = 5.0 * axis
        axial, transverse = cases.spin_components(q, roll)
        self.assertAlmostEqual(float(axial), 5.0, places=9)
        self.assertAlmostEqual(float(transverse), 0.0, places=9)
        twirl = np.array([0.0, 0.0, 5.0])  # end over end in the horizontal plane
        axial, transverse = cases.spin_components(q, twirl)
        self.assertAlmostEqual(float(axial), 0.0, places=9)
        self.assertAlmostEqual(float(transverse), 5.0, places=9)

    def test_batched(self):
        q = np.stack([_axis_angle_quat([0, 1, 0], a) for a in (0.0, 0.5, 1.0)])
        w = np.tile([1.0, 2.0, 3.0], (3, 1))
        axial, transverse = cases.spin_components(q, w)
        np.testing.assert_allclose(axial ** 2 + transverse ** 2, np.full(3, 14.0))


class TestReducers(unittest.TestCase):
    def test_transient_hit_counts_only_for_any_hit(self):
        pred = np.zeros(359, dtype=bool)
        pred[3:8] = True
        self.assertTrue(cases.any_hit(pred, 359))
        self.assertFalse(cases.sustained(pred, 359, completed=True))

    def test_sustained_tail(self):
        pred = np.zeros(359, dtype=bool)
        pred[-cases.SUSTAIN_WINDOW:] = True
        self.assertTrue(cases.sustained(pred, 359, completed=True))
        pred[-1] = False
        self.assertFalse(cases.sustained(pred, 359, completed=True))

    def test_fall_is_failure(self):
        pred = np.ones(200, dtype=bool)
        self.assertTrue(cases.any_hit(pred, 200))
        self.assertFalse(cases.sustained(pred, 200, completed=False))

    def test_any_hit_ignores_steps_after_length(self):
        pred = np.zeros(359, dtype=bool)
        pred[300] = True
        self.assertFalse(cases.any_hit(pred, 100))


class TestAggregate(unittest.TestCase):
    def test_pooled_difference(self):
        import json

        from lares.envs.isaac_pen_spin.protocols import aggregate_audit

        def ep(a, s):
            return {"any_hit": a, "sustained": s, "completed": s, "length": 359, "hit_steps": 3, "tail_hit_fraction": 0.5,
                    "mean_abs_spin": 1.0, "max_abs_spin": 2.0, "mean_transverse": 0.1, "first_hit_step": 1}

        with tempfile.TemporaryDirectory() as d:
            for st in (1, 2):
                for ctrl, eps in (("neutral_pose", [ep(True, False)]), ("open_loop_digit_oscillation", [ep(True, True)])):
                    with open(os.path.join(d, f"stage{st}_{ctrl}.json"), "w") as f:
                        json.dump({"episodes": eps}, f)
            out = aggregate_audit(d, stages=(1, 2))
        self.assertEqual(out["episodes"], 4)
        self.assertAlmostEqual(out["success_rate_any_hit"], 1.0)
        self.assertAlmostEqual(out["success_rate_sustained"], 0.5)
        self.assertAlmostEqual(out["paired_difference_sustained_minus_any_hit"]["mean"], -0.5)


if __name__ == "__main__":
    unittest.main()
