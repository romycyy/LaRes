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


class TestQualification(unittest.TestCase):
    def _cfg(self, d):
        return {"log_dir": d, "cases_per_stage": 40, "pool_seeds": {1: 4010, 2: 4011}, "control_seed": 4098,
                "manifests": {1: os.path.join(d, "m1.yaml"), 2: os.path.join(d, "m2.yaml")},
                "baseline": "osc", "candidate": "gait", "controllers": ["osc", "gait"]}

    def test_manifests_built_once_and_refuse_drift(self):
        from lares.envs.isaac_pen_spin.protocols import build_qualification_manifests

        with tempfile.TemporaryDirectory() as d:
            cfg = self._cfg(d)
            self.assertEqual(len(build_qualification_manifests(cfg)), 4)  # 2 development + 2 control
            self.assertEqual(build_qualification_manifests(cfg), [])
            m = EvaluationManifest.load(cfg["manifests"][1])
            self.assertEqual(len(m), 40)
            cfg["pool_seeds"][1] = 4013
            with self.assertRaises(ValueError):
                build_qualification_manifests(cfg)

    def test_aggregate(self):
        import json

        from lares.envs.isaac_pen_spin.protocols import aggregate_qualification

        def eps(successes):
            return [{"sustained": bool(x), "tail_hit_fraction": 0.5, "completed": True, "mean_abs_spin": 2.0,
                     "mean_transverse": 0.1} for x in successes]

        with tempfile.TemporaryDirectory() as d:
            cfg = self._cfg(d)
            for st in (1, 2):
                gait = [1] * 40 if st == 1 else [1] * 20 + [0] * 20
                for ctrl, succ in (("osc", [0] * 40), ("gait", gait)):
                    with open(os.path.join(d, f"stage{st}_{ctrl}.json"), "w") as f:
                        json.dump({"episodes": eps(succ)}, f)
            out = aggregate_qualification(cfg, d)
        s1, s2 = out["per_stage"]["stage1"], out["per_stage"]["stage2"]
        self.assertTrue(s1["qualifies"]["gait"])
        self.assertFalse(s2["qualifies"]["gait"])  # promotion cases 10-39 hold 10 successes
        self.assertEqual(s2["promotion"]["gait_successes"], 10)
        self.assertEqual(out["pooled_promotion_pairs"], 60)
        self.assertAlmostEqual(out["paired_success_difference"], 40 / 60)
        self.assertTrue(s1["promotes"])


class TestFeedbackAblation(unittest.TestCase):
    def _row(self, h, ok, length=359):
        return {"task_hash": h, "case_id": h, "sustained": ok, "completed": True, "tail_hit_fraction": 1.0 if ok else 0.5,
                "hit_steps": length, "length": length, "mean_abs_spin": 10.0, "mean_transverse": 1.0}

    def test_select_first_sustained_in_manifest_order(self):
        from lares.envs.isaac_pen_spin.feedback_ablation import select_demonstration

        rows = [self._row("a", False), self._row("b", True, 300), self._row("c", True)]
        acts = np.arange(3 * 359 * 20, dtype=np.float32).reshape(3, 359, 20)  # (cases, T, 20), as the runner stores it
        idx, tape = select_demonstration(rows, acts)
        self.assertEqual(idx, 1)
        self.assertEqual(tape.shape, (300, 20))
        np.testing.assert_array_equal(tape, acts[1, :300])
        with self.assertRaises(ValueError):
            select_demonstration([self._row("a", False)], acts[:1])
        with self.assertRaises(ValueError):  # the (T, cases, 20) layout is refused, not silently mis-sliced
            select_demonstration(rows, acts.transpose(1, 0, 2))

    def test_tape_shape_checked(self):
        from lares.envs.isaac_pen_spin.feedback_ablation import ActionTape

        with self.assertRaises(ValueError):
            ActionTape(np.zeros((10, 19)), {})
        self.assertEqual(ActionTape(np.zeros((10, 20)), {}).params["steps"], 10)

    def test_aggregate_paired(self):
        from lares.envs.isaac_pen_spin.feedback_ablation import aggregate

        closed = [self._row(h, True) for h in "abcd"]
        tape = [self._row("a", True)] + [self._row(h, False) for h in "bcd"]
        s = aggregate(closed, tape)
        self.assertAlmostEqual(s["paired_success_difference"], 0.75)
        self.assertEqual(s["closed_loop"]["successes"], 4)
        self.assertEqual(s["action_tape"]["successes"], 1)
        with self.assertRaises(ValueError):
            aggregate(closed, list(reversed(tape)))

    def test_manifests_match_config(self):
        import yaml

        from lares.envs.isaac_pen_spin import cases
        from lares.envs.isaac_pen_spin.feedback_ablation import manifest_paths
        from lares.eval.manifest import EvaluationManifest

        cfg = yaml.safe_load(open("config/shadowhand_spin_stages.yaml"))["research_protocols"]["feedback_ablation"]
        for label, path in manifest_paths(cfg).items():
            if os.path.exists(path):
                m = cases.build_manifest(3, int(cfg["pool_seeds"][label]), int(cfg["cases"]), label=label)
                self.assertEqual(EvaluationManifest.load(path).to_dict(), m.to_dict())


class TestFeedbackLesions(unittest.TestCase):
    def _row(self, h, ok, completed=True, length=359):
        return {"task_hash": h, "case_id": h, "sustained": ok, "completed": completed,
                "tail_hit_fraction": 1.0 if ok else 0.5, "hit_steps": length, "length": length,
                "mean_abs_spin": 10.0, "mean_transverse": 1.0}

    def test_lesion_columns_cover_exactly_the_named_slices(self):
        from lares.envs.isaac_pen_spin.feedback_lesions import LESIONS, lesion_index

        sl = cases.obs_slices()
        pen, hand = lesion_index(LESIONS["pen_state_clamped"]), lesion_index(LESIONS["hand_state_clamped"])
        self.assertEqual(len(pen), 3 + 4 + 3 + 3 + 4)
        self.assertEqual(len(hand), 24 + 24 + 15 + 20 + 30)
        self.assertFalse(set(pen) & set(hand))
        live = set(range(sl["last_action"].start, sl["last_action"].stop))
        self.assertFalse(live & (set(pen) | set(hand)))  # action history stays live in both lesions

    def test_wrapper_holds_first_step_values_and_passes_the_rest(self):
        import torch

        from lares.envs.isaac_pen_spin.feedback_lesions import LesionedController, lesion_index, LESIONS

        seen = []

        class Inner:
            params = {"checkpoint": "x"}

            def reset(self, ctx):
                pass

            def __call__(self, obs, t, ctx):
                seen.append(obs.clone())
                return torch.zeros(obs.shape[0], 20)

        ctrl = LesionedController(Inner(), "pen_state_clamped")
        ctrl.reset(None)
        o0, o1 = torch.rand(3, 157), torch.rand(3, 157)
        ctrl(o0, 0, None)
        ctrl(o1, 1, None)
        cols = lesion_index(LESIONS["pen_state_clamped"])
        other = np.setdiff1d(np.arange(157), cols)
        np.testing.assert_array_equal(seen[1][:, cols].numpy(), o0[:, cols].numpy())
        np.testing.assert_array_equal(seen[1][:, other].numpy(), o1[:, other].numpy())
        ctrl.reset(None)  # a new episode holds its own first observation
        ctrl(o1, 0, None)
        np.testing.assert_array_equal(seen[2].numpy(), o1.numpy())
        with self.assertRaises(ValueError):
            LesionedController(Inner(), "goal_clamped")

    def test_aggregate_pairs_and_splits_failures(self):
        from lares.envs.isaac_pen_spin.feedback_lesions import aggregate

        expert = [self._row(h, True) for h in "abcd"]
        pen = [self._row("a", True), self._row("b", False, completed=False, length=40),
               self._row("c", False, completed=False, length=60), self._row("d", False)]
        s = aggregate({"expert": expert, "pen_state_clamped": pen, "hand_state_clamped": expert})
        self.assertAlmostEqual(s["paired_success_difference"], -0.75)
        self.assertEqual(s["arms"]["pen_state_clamped"]["drops"], 2)
        self.assertEqual(s["arms"]["pen_state_clamped"]["tail_spin_failures"], 1)
        self.assertEqual(s["arms"]["pen_state_clamped"]["median_fall_step"], 50.0)
        self.assertAlmostEqual(s["hand_state_clamped_minus_expert"]["mean"], 0.0)
        with self.assertRaises(ValueError):
            aggregate({"expert": expert, "pen_state_clamped": list(reversed(pen))})

    def test_manifest_matches_config(self):
        import yaml

        cfg = yaml.safe_load(open("config/shadowhand_spin_stages.yaml"))["research_protocols"]["feedback_lesions"]
        m = cases.build_manifest(3, int(cfg["pool_seed"]), int(cfg["cases"]), label="lesion")
        self.assertEqual(EvaluationManifest.load(cfg["manifest"]).to_dict(), m.to_dict())
        self.assertEqual(cfg["pool_seed"], 4140)


class TestFeedbackDelay(unittest.TestCase):
    class Inner:
        params = {"checkpoint": "x"}

        def __init__(self):
            self.seen = []

        def reset(self, ctx):
            pass

        def __call__(self, obs, t, ctx):
            self.seen.append(obs.clone())
            return obs[:, :20] * 0

    def test_delay_feeds_the_group_from_k_steps_earlier_and_passes_the_rest(self):
        import torch

        from lares.envs.isaac_pen_spin.feedback_delay import DELAYS, DelayedController
        from lares.envs.isaac_pen_spin.feedback_lesions import lesion_index

        inner = self.Inner()
        ctrl = DelayedController(inner, "pen_state_delayed4", delay=4)
        cols = lesion_index(DELAYS["pen_state_delayed4"])
        other = np.setdiff1d(np.arange(157), cols)
        obs = [torch.rand(3, 157) for _ in range(7)]
        ctrl.reset(None)
        for t, o in enumerate(obs):
            ctrl(o, t, None)
        for t in range(7):
            src = obs[max(0, t - 4)]  # the buffer starts filled with the reset observation
            np.testing.assert_array_equal(inner.seen[t][:, cols].numpy(), src[:, cols].numpy())
            np.testing.assert_array_equal(inner.seen[t][:, other].numpy(), obs[t][:, other].numpy())
        ctrl.reset(None)  # a new episode starts its own history
        ctrl(obs[6], 0, None)
        np.testing.assert_array_equal(inner.seen[-1].numpy(), obs[6].numpy())
        with self.assertRaises(ValueError):
            DelayedController(inner, "pen_state_clamped")

    def test_groups_match_the_r044_lesions_and_leave_action_history_live(self):
        from lares.envs.isaac_pen_spin.feedback_delay import DELAYS
        from lares.envs.isaac_pen_spin.feedback_lesions import LESIONS, lesion_index

        self.assertEqual(DELAYS["pen_state_delayed4"], LESIONS["pen_state_clamped"])
        self.assertEqual(DELAYS["hand_state_delayed4"], LESIONS["hand_state_clamped"])
        sl = cases.obs_slices()
        live = set(range(sl["last_action"].start, sl["last_action"].stop))
        self.assertFalse(live & set(lesion_index(DELAYS["hand_state_delayed4"])))

    def test_aggregate_reports_each_delay_against_the_expert(self):
        from lares.envs.isaac_pen_spin.feedback_delay import aggregate

        row = TestFeedbackLesions._row
        expert = [row(None, h, True) for h in "abcd"]
        pen = [row(None, "a", True), row(None, "b", False, completed=False, length=80)] + expert[2:]
        s = aggregate({"expert": expert, "pen_state_delayed4": pen, "hand_state_delayed4": expert})
        self.assertAlmostEqual(s["paired_success_difference"], -0.25)
        self.assertEqual(s["arms"]["pen_state_delayed4"]["drops"], 1)
        self.assertAlmostEqual(s["hand_state_delayed4_minus_expert"]["mean"], 0.0)

    def test_manifest_matches_config(self):
        import yaml

        cfg = yaml.safe_load(open("config/shadowhand_spin_stages.yaml"))["research_protocols"]["feedback_delay"]
        m = cases.build_manifest(3, int(cfg["pool_seed"]), int(cfg["cases"]), label="delay")
        self.assertEqual(EvaluationManifest.load(cfg["manifest"]).to_dict(), m.to_dict())
        self.assertEqual((cfg["pool_seed"], cfg["delay_steps"]), (4200, 4))


class TestLearnerRecovery(unittest.TestCase):
    class Const:
        def __init__(self, value):
            self.value = value
            self.params = {"value": value}

        def reset(self, ctx):
            pass

        def __call__(self, obs, t, ctx):
            return obs[:, :20] * 0 + self.value

    class Ctx:
        device = "cpu"

    def test_branch_rule_first_retained_failure_in_window_else_120_else_never(self):
        from lares.envs.isaac_pen_spin.learner_recovery import NEVER, branch_steps

        T, n = 360, 4
        pred = np.ones((T, n), dtype=bool)
        spin = np.full((T, n), 10.0)
        centre = np.full((T, n), 0.01)
        length = np.array([360, 360, 50, 100])
        pred[70, 0], spin[70, 0] = False, 2.0       # case 0: slow spin at 70 -> expert from 71
        pred[59, 1], centre[59, 1] = False, 0.2     # case 1: off-centre before the window only -> 120
        spin[:, 2] = 0.0                            # case 2: dropped before the window -> never
        pred[80:, 3], spin[80:, 3] = False, 0.0     # case 3: slow at 80 but still in play -> 81
        steps = branch_steps(length, pred, centre, spin)
        self.assertEqual(steps.tolist(), [71, 120, NEVER, 81])
        length[3] = 81                              # case 3 ends at step 80: no switch
        self.assertEqual(int(branch_steps(length, pred, centre, spin)[3]), NEVER)

    def test_handover_switches_each_row_at_its_own_step(self):
        import torch

        from lares.envs.isaac_pen_spin.learner_recovery import NEVER, HandoverController

        ctrl = HandoverController(self.Const(-0.5), self.Const(0.5), [2, NEVER, 0], "takeover")
        ctrl.reset(self.Ctx())
        obs = torch.zeros(3, 157)
        rows = [ctrl(obs, t, None)[:, 0].tolist() for t in range(4)]
        self.assertEqual(rows, [[-0.5, -0.5, 0.5], [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5], [0.5, -0.5, 0.5]])
        self.assertEqual(ctrl.params["switched_cases"], 2)

    def test_gates_and_transitions(self):
        from lares.envs.isaac_pen_spin.learner_recovery import NEVER, identity_gate, prefix_gate, transitions

        rng = np.random.default_rng(0)
        obs = rng.normal(size=(2, 10, 157))
        a = {"obs": obs, "length": np.array([10, 4]), "actions": np.zeros((2, 10, 20))}
        rows = [{"sustained": True, "completed": True, "length": 10}, {"sustained": False, "completed": False, "length": 4}]
        b = {**a, "obs": obs.copy()}
        b["obs"][1, 6] += 1.0                       # after case 1 ended: ignored
        self.assertTrue(identity_gate(a, rows, b, [dict(r) for r in rows])["passes"])
        b["obs"][0, 3] += 1e-7
        g = identity_gate(a, rows, b, rows)
        self.assertFalse(g["passes"])
        self.assertEqual(g["observation_mismatch_cases"], [0])

        take = {"obs": obs.copy(), "length": np.array([10, 4]), "actions": np.ones((2, 10, 20))}
        take["obs"][0, 6:] += 1.0                   # diverges after its first expert action at step 5
        self.assertTrue(prefix_gate(a, take, [5, NEVER])["passes"])
        bad = prefix_gate(a, take, [6, NEVER])
        self.assertFalse(bad["passes"])
        self.assertEqual(bad["mismatches"][0]["first_mismatch_step"], 6)
        tr = transitions(a, take, [5, NEVER], k=3)
        self.assertEqual(tr["step"].tolist(), [5, 6, 7])
        np.testing.assert_array_equal(tr["expert_next_obs"][0], take["obs"][0, 6])
        np.testing.assert_array_equal(tr["learner_next_obs"][0], obs[0, 6])

    def test_frozen_learner_hashes_and_manifest(self):
        import yaml

        from lares.envs.isaac_pen_spin import learner_recovery

        with open("config/shadowhand_spin_stages.yaml") as f:
            pc = yaml.safe_load(f)["research_protocols"]["learner_recovery"]
        learner_recovery.check_learner(pc)
        m = EvaluationManifest.load(pc["manifest"])
        self.assertEqual((len(m), cases.stage_of(m)), (32, 3))
        self.assertEqual(learner_recovery.build_manifests(pc), [])


class TestLearnerAxialSign(unittest.TestCase):
    def test_reflection_negates_axial_keeps_transverse_and_other_columns(self):
        import torch

        from lares.envs.isaac_pen_spin.axial_sign import reflect_axial

        sl = cases.obs_slices()
        obs = torch.rand(5, 157, dtype=torch.float64)
        q = torch.randn(5, 4, dtype=torch.float64)
        obs[:, sl["obj_rot"]] = q / q.norm(dim=1, keepdim=True)
        out = reflect_axial(obs)
        ax0, tr0 = cases.spin_components(obs[:, sl["obj_rot"]].numpy(), obs[:, sl["obj_angvel"]].numpy())
        ax1, tr1 = cases.spin_components(out[:, sl["obj_rot"]].numpy(), out[:, sl["obj_angvel"]].numpy())
        np.testing.assert_allclose(ax1, -ax0, atol=1e-12)
        np.testing.assert_allclose(tr1, tr0, atol=1e-12)
        keep = np.setdiff1d(np.arange(157), np.arange(sl["obj_angvel"].start, sl["obj_angvel"].stop))
        np.testing.assert_array_equal(out[:, keep].numpy(), obs[:, keep].numpy())
        self.assertTrue(torch.equal(reflect_axial(obs, "identity"), obs))

    def test_source_audit_and_manifest(self):
        import yaml

        from lares.envs.isaac_pen_spin import axial_sign

        with open("config/shadowhand_spin_stages.yaml") as f:
            pc = yaml.safe_load(f)["research_protocols"]["learner_axial_sign"]
        audit = axial_sign.source_audit(pc)
        self.assertTrue(audit["reads_obj_angvel"] and audit["axial_projection"] and audit["axial_drive"])
        m = EvaluationManifest.load(pc["manifest"])
        self.assertEqual((len(m), cases.stage_of(m)), (32, 3))
        self.assertEqual(axial_sign.build_manifests(pc), [])


if __name__ == "__main__":
    unittest.main()


class TestScalarTuning(unittest.TestCase):
    """R-058: parameter space, samplers, ranking, tiling, saturation and aggregation, no Isaac."""

    @classmethod
    def setUpClass(cls):
        import yaml

        with open("config/shadowhand_spin_stages.yaml") as f:
            cls.pc = yaml.safe_load(f)["research_protocols"]["r058_scalar_cma_vs_random"]

    def test_space_reads_bounds_and_values_from_the_frozen_parent(self):
        from lares.envs.isaac_pen_spin import scalar_tuning as st

        space = st.parameter_space(self.pc)
        self.assertEqual(space.names, st.whitelist_names(self.pc))
        self.assertEqual(space.dim, 12)
        i = space.names.index("spin_target")
        self.assertAlmostEqual(space.parent[i], 12.156, places=2)       # the frozen operating point (D-046)
        self.assertEqual((space.lo[i], space.hi[i]), (5.5, 14.0))
        z = space.normalise(space.parent)
        self.assertTrue(np.all((z >= 0) & (z <= 1)))
        np.testing.assert_allclose(space.denormalise(z), space.parent)
        with self.assertRaises(ValueError):
            st.ParameterSpace(["a"], [0.0], [1.0], [2.0])

    def test_samplers_are_deterministic_bounded_and_the_random_arm_never_adapts(self):
        from lares.envs.isaac_pen_spin import scalar_tuning as st

        space = st.ParameterSpace([f"p{i}" for i in range(12)], [0.0] * 12, [1.0] * 12, [0.5] * 12)
        opts = {"sigma0": 0.2, "popsize": 8, "generations": 3, "bounds": [0.0, 1.0], "random_seed_base": 100}
        a, b = st.make_sampler("cma", space, opts, 4510), st.make_sampler("cma", space, opts, 4510)
        xa, xb = a.ask(), b.ask()
        self.assertEqual(xa.shape, (8, 12))
        np.testing.assert_allclose(xa, xb)
        self.assertTrue(np.all((xa >= 0) & (xa <= 1)))
        a.tell(list(range(8)))
        b.tell(list(reversed(range(8))))
        self.assertFalse(np.allclose(a.ask(), b.ask()))              # the adaptive arm depends on the ranks
        r1, r2 = st.make_sampler("random", space, opts, 4510), st.make_sampler("random", space, opts, 4510)
        g0 = r1.ask()
        r1.tell(list(range(8)))
        g1 = r1.ask()
        r2.ask()
        r2.tell(list(reversed(range(8))))
        np.testing.assert_allclose(g1, r2.ask())                      # ranks never reach the random arm
        self.assertFalse(np.allclose(g0, g1))
        fresh = st._new_es(space, opts, 100 + 4510 * 100 + 1).ask()  # generation 0 = one fresh instance, asked once
        np.testing.assert_allclose(g0, np.stack(fresh))

    def test_ranking_is_lexicographic_with_index_tie_break(self):
        from lares.envs.isaac_pen_spin import scalar_tuning as st

        m = [{"successes": 1, "drops": 2, "tail_hit_fraction": 0.5},
             {"successes": 2, "drops": 3, "tail_hit_fraction": 0.1},
             {"successes": 1, "drops": 1, "tail_hit_fraction": 0.2},
             {"successes": 1, "drops": 1, "tail_hit_fraction": 0.9},
             {"successes": 1, "drops": 1, "tail_hit_fraction": 0.9}]
        self.assertEqual(st.rank_candidates(m), [4, 0, 3, 1, 2])
        rows = [{"sustained": True, "completed": True, "tail_hit_fraction": 1.0, "mean_abs_spin": 5.0},
                {"sustained": False, "completed": False, "tail_hit_fraction": 0.0, "mean_abs_spin": None}]
        self.assertEqual(st.screen_metrics(rows)["successes"], 1)
        self.assertEqual(st.screen_metrics(rows)["drops"], 1)
        self.assertEqual(st.split_rows(list(range(6)), 3), [[0, 1], [2, 3], [4, 5]])

    def test_tiled_manifest_and_tiled_controller_route_rows_by_candidate(self):
        import torch

        from lares.envs.isaac_pen_spin import scalar_tuning as st
        from lares.envs.isaac_pen_spin.learner_recovery import make_learner

        m = EvaluationManifest.load(st.manifest_path(self.pc, 4510))
        role = cases.role_manifests(m, st.roles(self.pc), expected_cases=80, stage=3)
        self.assertEqual([len(role[r]) for r in ("screen", "promotion", "fresh")], [10, 30, 40])
        tiled = st.tiled_manifest(role["screen"], 8)
        self.assertEqual(len(tiled), 80)
        self.assertEqual(len({c.case_id for c in tiled.episodes}), 80)
        self.assertEqual(cases.resolve_placements(tiled).shape, (80, cases.PLACEMENT_DIM))
        np.testing.assert_array_equal(cases.resolve_placements(tiled)[10:20], cases.resolve_placements(role["screen"]))

        cands = [{"center_gain": 0.05}, {"center_gain": 0.5}]   # centering acts on every observation
        ctrl = st.TiledCandidates([make_learner(self.pc), make_learner(self.pc)], cands, "t")
        self.assertAlmostEqual(float(ctrl.learners[1].policy.center_gain), 0.5)
        obs = torch.randn(6, cases.OBS_DIM)
        obs[:, cases.obs_slices()["obj_rot"]] = torch.tensor([1.0, 0, 0, 0])
        out = ctrl(obs, 0, None)
        single = st.TiledCandidates([make_learner(self.pc)], [cands[1]], "s")(obs[3:], 0, None)
        torch.testing.assert_close(out[3:], single)
        self.assertFalse(torch.allclose(out[:3], st.TiledCandidates([make_learner(self.pc)], [cands[1]], "s")(obs[:3], 0, None)))

    def test_candidates_file_freezes_and_controller_rebuilds_from_it(self):
        from lares.envs.isaac_pen_spin import scalar_tuning as st

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "c.json")
            st.write_candidates(p, "x", [{"spin_target": 8.0}], {"normalised": [[0.3]]})
            st.write_candidates(p, "x", [{"spin_target": 8.0}])          # identical: accepted
            with self.assertRaises(SystemExit):
                st.write_candidates(p, "x", [{"spin_target": 8.5}])
            ctrl = st.make_controller("candidates:c.json", self.pc, d)
            self.assertEqual(ctrl.tile, 1)
            self.assertAlmostEqual(float(ctrl.learners[0].policy.spin_target), 8.0)
            with self.assertRaises(ValueError):
                st.make_controller("parent", self.pc, d)

    def test_saturation_from_recorded_traces(self):
        from lares.envs.isaac_pen_spin import scalar_tuning as st

        sl = cases.obs_slices()
        obs = np.zeros((2, 4, cases.OBS_DIM))
        obs[:, :, sl["obj_rot"]] = [1.0, 0.0, 0.0, 0.0]               # long axis = world x
        obs[0, :, sl["obj_angvel"]] = [2.0, 0.0, 0.0]                  # spin = 5 * 2 = 10 rad/s
        obs[1, :, sl["obj_angvel"]] = [0.0, 0.0, 0.0]
        d = st.drive_from_obs(obs[0], spin_target=12.0, speed_gain=0.8)
        np.testing.assert_allclose(d, 0.25 + 0.75 * np.tanh(0.8 * 2.0))
        actions = np.zeros((2, 4, 20))
        actions[0, :, 0] = 1.0
        actions[1, :2, :] = -1.0
        trace = {"obs": obs, "actions": actions, "length": np.array([4, 2])}
        s = st.saturation(trace, {"spin_target": 12.0, "speed_gain": 0.8})
        self.assertEqual(s["in_play_steps"], 6)
        self.assertAlmostEqual(s["drive_saturated_fraction"], 2 / 6)   # case 1 (spin 0) saturates, case 0 does not
        self.assertEqual(s["drive_unsaturated_fraction_per_case"], [1.0, 0.0])
        self.assertAlmostEqual(s["action_saturated_fraction"], (4 + 40) / 120)
        self.assertAlmostEqual(s["action_saturated_by_axis"][0], 1.0)

    def test_block_summary_and_decision(self):
        from lares.envs.isaac_pen_spin import scalar_tuning as st

        def rows(succ, drops, n):
            out = []
            for i in range(n):
                out.append({"task_hash": f"h{i}", "sustained": i < succ, "completed": i < n - drops,
                            "tail_hit_fraction": 1.0 if i < succ else 0.0, "hit_steps": 60, "length": 360,
                            "mean_abs_spin": 5.0, "mean_transverse": 1.0})
            return out

        def sat(unsat):
            return {"in_play_steps": 1, "drive_saturated_fraction": 1 - unsat, "action_saturated_fraction": 0.5,
                    "action_saturated_by_axis": [0.5] * 20, "drive_unsaturated_fraction_per_case": [unsat] * 40,
                    "note": ""}

        r = {"cma_winner": {"promotion": rows(20, 0, 30), "fresh": rows(30, 0, 40)},
             "random_winner": {"promotion": rows(6, 2, 30), "fresh": rows(8, 3, 40)},
             "parent": {"promotion": rows(6, 2, 30), "fresh": rows(8, 2, 40)}}
        s = {"cma_winner": {"promotion": sat(0.5), "fresh": sat(0.5)},
             "random_winner": {"promotion": sat(0.1), "fresh": sat(0.1)},
             "parent": {"promotion": sat(0.1), "fresh": sat(0.1)}}
        winners = {"cma": {"params": {"spin_target": 9.0}}, "random": {"params": {"spin_target": 11.0}}}
        b = st.block_summary(r, s, winners, {"successes": 2}, {})
        self.assertTrue(b["promotion"]["cma_winner_minus_parent"]["promotes"])
        self.assertFalse(b["promotion"]["random_winner_minus_parent"]["promotes"])   # tie never promotes
        self.assertTrue(b["fresh"]["cma_winner_qualifies_repair_parent"])
        self.assertFalse(b["fresh"]["random_winner_qualifies_repair_parent"])        # one more drop than the parent
        self.assertEqual(b["fresh"]["cma_minus_random"]["drop_counts"], [0, 3])
        self.assertTrue(b["fresh"]["cma_winner_feedback"]["activated"])    # +0.4 on every case: interval above zero
        self.assertFalse(b["fresh"]["random_winner_feedback"]["activated"])
        import json

        b["seed"] = 4510
        b2 = json.loads(json.dumps(b))
        b2["seed"] = 4511
        summary = st.summarise([b, b2])
        self.assertAlmostEqual(summary["paired_success_difference"], 30 / 40 - 8 / 40)
        self.assertEqual(summary["promotions"]["cma_winner"], [4510, 4511])
        self.assertEqual(summary["qualifying_repair_parents"]["random_winner"], [])
        dec = st.decision_artifact(summary, [b, b2])
        self.assertEqual(dec["repair_parent"]["arm"], "cma_winner")
        self.assertTrue(dec["route"].startswith("R-056"))

    def test_manifests_match_config_and_are_disjoint(self):
        from lares.envs.isaac_pen_spin import scalar_tuning as st
        from lares.eval.manifest import assert_disjoint

        self.assertEqual(st.build_manifests(self.pc), [])
        ms = [EvaluationManifest.load(st.manifest_path(self.pc, s)) for s in self.pc["seeds"]]
        self.assertEqual([len(m) for m in ms], [80] * 8)
        assert_disjoint(ms)
        self.assertEqual(self.pc["optimizer"]["popsize"] * self.pc["optimizer"]["generations"], 64)


class TestArchiveRescreen(unittest.TestCase):
    """R-060: archive shortlist, re-ranking, manifests, aggregation and the offline clipping diagnostic."""

    @classmethod
    def setUpClass(cls):
        import yaml

        with open("config/shadowhand_spin_stages.yaml") as f:
            cls.pc = yaml.safe_load(f)["research_protocols"]["r060_archive_rescreen"]

    def _fake_archive(self, d, block_seed=4510, generations=2, popsize=4):
        import json

        adir = os.path.join(d, f"block_s{block_seed}", "cma")
        os.makedirs(adir)
        idx = 0
        best = None
        for g in range(generations):
            cands = [{"spin_target": 6.0 + idx + k} for k in range(popsize)]
            metrics = [{"successes": (idx + k) % 3, "drops": 0, "tail_hit_fraction": 0.1 * k} for k in range(popsize)]
            json.dump({"label": "x", "tile": popsize, "candidates": cands}, open(os.path.join(adir, f"gen{g}_candidates.json"), "w"))
            json.dump({"generation": g, "metrics": metrics, "candidate_indices": list(range(idx, idx + popsize))},
                      open(os.path.join(adir, f"gen{g}_screen.json"), "w"))
            idx += popsize
        # the frozen ranking: successes 2 first; among those higher tail; index 2 (succ 2, tail .2) vs 5 (succ 2, tail .1)
        json.dump({"index": 2}, open(os.path.join(adir, "winner.json"), "w"))
        return adir

    def test_shortlist_reproduces_the_winner_and_orders_by_the_frozen_ranking(self):
        import json

        from lares.envs.isaac_pen_spin import archive_rescreen as ar

        with tempfile.TemporaryDirectory() as d:
            adir = self._fake_archive(d)
            arch = ar.load_archive_arm(d, 4510, "cma", 2, 4)
            self.assertEqual(len(arch["params"]), 8)
            sl = ar.shortlist(arch, 3)
            self.assertEqual(sl["indices"][0], 2)
            self.assertEqual(sl["indices"], [2, 5, 7])   # succ 2: idx 2 (tail .2), 5 (.1); then succ 1 with the highest tail: 7
            self.assertEqual(sl["params"][0], {"spin_target": 8.0})
            json.dump({"index": 5}, open(os.path.join(adir, "winner.json"), "w"))
            with self.assertRaises(SystemExit):
                ar.shortlist(ar.load_archive_arm(d, 4510, "cma", 2, 4), 3)
            with self.assertRaises(SystemExit):
                ar.load_archive_arm(d, 4511, "cma", 2, 4)

    def test_block_seed_pairing_and_manifests(self):
        from lares.envs.isaac_pen_spin import archive_rescreen as ar
        from lares.eval.manifest import assert_disjoint

        self.assertEqual(ar.archive_block_seed(self.pc, 4550), 4510)
        self.assertEqual(ar.archive_block_seed(self.pc, 4557), 4517)
        self.assertEqual(ar.build_manifests(self.pc), [])
        ms = [EvaluationManifest.load(ar.manifest_path(self.pc, s)) for s in self.pc["seeds"]]
        self.assertEqual([len(m) for m in ms], [110] * 8)
        assert_disjoint(ms + [EvaluationManifest.load(p) for p in self.pc["disjoint_from"]])
        role = cases.role_manifests(ms[0], ar.roles(self.pc), expected_cases=110, stage=3)
        self.assertEqual([len(role[r]) for r in ("rescreen", "promotion", "fresh")], [40, 30, 40])

    def test_block_summary_summary_and_decision(self):
        import json

        from lares.envs.isaac_pen_spin import archive_rescreen as ar

        def rows(succ, drops, n):
            return [{"task_hash": f"h{i}", "sustained": i < succ, "completed": i < n - drops,
                     "tail_hit_fraction": 1.0 if i < succ else 0.0, "hit_steps": 60, "length": 360,
                     "mean_abs_spin": 5.0, "mean_transverse": 1.0} for i in range(n)]

        def sat(u):
            return {"in_play_steps": 1, "drive_saturated_fraction": 1 - u, "action_saturated_fraction": 0.5,
                    "action_saturated_by_axis": [0.5] * 20, "drive_unsaturated_fraction_per_case": [u] * 40, "note": ""}

        r = {"cma_reranked": {"promotion": rows(20, 0, 30), "fresh": rows(30, 0, 40)},
             "cma_original": {"promotion": rows(6, 2, 30), "fresh": rows(8, 2, 40)},
             "random_reranked": {"promotion": rows(6, 2, 30), "fresh": rows(8, 2, 40)},
             "random_original": {"promotion": rows(6, 2, 30), "fresh": rows(8, 2, 40)},
             "parent": {"promotion": rows(6, 2, 30), "fresh": rows(8, 2, 40)}}
        s = {c: {"promotion": sat(0.1), "fresh": sat(0.1)} for c in r}
        rescreen = {"cases": 40, "archive_block_seed": 4510, "parent": {"successes": 8},
                    "cma": {"metrics": [{"successes": 12}, {"successes": 9}]}, "random": {"metrics": [{"successes": 7}]}}
        winners = {"cma": {"reranked_index": 44, "original_index": 10}, "random": {"reranked_index": 3, "original_index": 3}}
        b = ar.block_summary(r, s, rescreen, winners)
        self.assertTrue(b["fresh"]["cma_reranked_minus_cma_original"]["success"]["mean"] > 0.5)
        self.assertTrue(b["fresh"]["random_reranked_minus_random_original"]["same_candidate"])
        self.assertAlmostEqual(b["fresh"]["reranked_minus_original_mean_over_arms"], (22 / 40 + 0) / 2)
        self.assertTrue(b["fresh"]["cma_reranked_qualifies_repair_parent"])
        self.assertFalse(b["fresh"]["cma_original_qualifies_repair_parent"])
        b["seed"] = 4550
        b2 = json.loads(json.dumps(b))
        b2["seed"] = 4551
        summ = ar.summarise([b, b2])
        self.assertAlmostEqual(summ["paired_success_difference"], 11 / 40)
        self.assertEqual(summ["same_candidate_blocks"]["random"], [4550, 4551])
        self.assertEqual(summ["rescreen"]["shortlist_best_success_per_block"]["cma"], [0.3, 0.3])
        dec = ar.decision_artifact(summ, [b, b2])
        self.assertEqual(dec["repair_parent"]["controller"], "cma_reranked")

    def test_clipping_diagnostic_measures_pre_minus_post_clip_variation(self):
        import torch

        from lares.envs.isaac_pen_spin import archive_rescreen as ar

        pol = ar.instrumented_policy(self.pc, {})
        sl = cases.obs_slices()
        obs = torch.zeros(3, cases.OBS_DIM)
        obs[:, sl["obj_rot"]] = torch.tensor([1.0, 0, 0, 0])
        obs[:, sl["fingertip_rot"]] = torch.tensor([1.0, 0, 0, 0]).repeat(5)
        obs[:, sl["in_hand_pos"]] = torch.tensor([0.0, 0.0, 0.1])
        obs[:, sl["obj_pos"]] = torch.tensor([0.0, 0.0, 0.1])  # at the centre: retained gate open, pre-clip mean unclamped
        pol._drive_delta = 0.0
        m0, _ = pol(obs)
        pol._drive_delta = 0.3
        m1, _ = pol(obs)
        self.assertEqual(m0.shape, (3, 20))
        self.assertFalse(torch.allclose(m0, m1))                     # the override reaches the output
        trace = {"obs": obs.unsqueeze(1).repeat(1, 4, 1).numpy(), "length": np.array([4, 2, 0])}
        d = ar.clipping_diagnostic(self.pc, [trace], [{}], delta=0.5)
        self.assertEqual(d["episodes"], 2)
        self.assertEqual(len(d["reduction_by_axis"]), 20)
        self.assertTrue(all(x >= y - 1e-6 for x, y in zip(d["action_variation_by_axis_pre_clip"], d["action_variation_by_axis_post_clip"])))


class TestStrokeLesion(unittest.TestCase):
    """R-062: source audit, substitution, identity, aggregation and the frozen manifest."""

    @classmethod
    def setUpClass(cls):
        import yaml

        with open("config/shadowhand_spin_stages.yaml") as f:
            cls.pc = yaml.safe_load(f)["research_protocols"]["r062_stroke_lesion"]

    def _contact_obs(self, n=3):
        import torch

        sl = cases.obs_slices()
        obs = torch.zeros(n, cases.OBS_DIM)
        obs[:, sl["obj_rot"]] = torch.tensor([1.0, 0, 0, 0])
        obs[:, sl["in_hand_pos"]] = torch.tensor([0.0, 0.0, 0.1])
        obs[:, sl["obj_pos"]] = torch.tensor([0.0, 0.0, 0.1])          # retained
        tips = torch.tensor([[0.02, 0.0, 0.1], [0.0, 0.02, 0.1], [-0.02, 0.0, 0.1], [0.0, -0.02, 0.1], [0.0, 0.0, 0.12]])
        obs[:, sl["fingertip_pos"]] = tips.flatten()                     # in contact: mean tip distance 0.02
        obs[:, sl["fingertip_rot"]] = torch.tensor([1.0, 0, 0, 0]).repeat(5)
        obs[:, sl["obj_angvel"]] = torch.tensor([0.4, 0.0, 0.0])
        return obs

    def test_audit_substitution_identity_and_lesion(self):
        import torch

        from lares.envs.isaac_pen_spin import stroke_lesion as sl_
        from lares.envs.isaac_pen_spin.learner_recovery import make_learner

        with open(self.pc["learner"]["source"]) as f:
            src = f.read()
        audit = sl_.source_audit(src)
        self.assertTrue(audit["passes"], audit)
        self.assertIn("+ 0.0 * rolling * gait", sl_.substituted_source(src, 0.0))
        with self.assertRaises(SystemExit):
            sl_.substituted_source(src.replace("+ rolling * gait\n", "+ gait\n"), 0.0)
        obs = self._contact_obs()
        parent, ident, lesion = (make_learner(self.pc), sl_.make_controller("parent_identity", self.pc),
                                 sl_.make_controller("stroke_lesion", self.pc))
        a, b, c = parent(obs, 0, None), ident(obs, 0, None), lesion(obs, 0, None)
        self.assertTrue(torch.equal(a, b))                               # identity factor is bit-identical
        self.assertFalse(torch.allclose(a, c))                           # the stroke contributes in contact
        self.assertEqual(lesion.params["gait_factor"], 0.0)
        far = obs.clone()
        far[:, cases.obs_slices()["obj_pos"]] = torch.tensor([0.3, 0.3, 0.3])  # out of the safe region: gate closed
        self.assertTrue(torch.allclose(parent(far, 0, None), lesion(far, 0, None)))

    def test_aggregate_interpretation_rule(self):
        from lares.envs.isaac_pen_spin import stroke_lesion as sl_

        def rows(succ, drops, n, spin):
            return [{"task_hash": f"h{i}", "sustained": i < succ, "completed": i < n - drops, "tail_hit_fraction": 0.0,
                     "hit_steps": 0, "length": 360, "mean_abs_spin": spin, "mean_transverse": 1.0} for i in range(n)]

        sat = {"in_play_steps": 1, "drive_saturated_fraction": 0.9, "action_saturated_fraction": 0.1,
               "action_saturated_by_axis": [0.1] * 20, "drive_unsaturated_fraction_per_case": [0.1] * 80, "note": ""}
        parent = rows(24, 10, 80, 6.0)
        a = sl_.aggregate(parent, rows(0, 10, 80, 1.0), sat, sat)
        self.assertTrue(a["interpretation"]["stroke_is_spin_source"])
        self.assertAlmostEqual(a["paired_success_difference"], -0.3)
        self.assertEqual(a["parent_sustained_cases"], 24)
        b = sl_.aggregate(parent, rows(0, 60, 80, 1.0), sat, sat)       # loss carried by drops
        self.assertTrue(b["interpretation"]["drop_driven_inconclusive"])
        self.assertFalse(b["interpretation"]["stroke_is_spin_source"])
        c = sl_.aggregate(parent, rows(22, 10, 80, 6.0), sat, sat)      # no loss: refuted
        self.assertTrue(c["interpretation"]["prediction_refuted"])

    def test_manifest_matches_config(self):
        from lares.envs.isaac_pen_spin import stroke_lesion as sl_

        self.assertEqual(sl_.build_manifests(self.pc), [])
        m = EvaluationManifest.load(self.pc["manifest"])
        self.assertEqual((len(m), cases.stage_of(m)), (80, 3))


class TestChannelLesions(unittest.TestCase):
    """R-065: audit, three substitutions, offline identity gate, tail split and interpretation."""

    @classmethod
    def setUpClass(cls):
        import yaml

        with open("config/shadowhand_spin_stages.yaml") as f:
            cls.pc = yaml.safe_load(f)["research_protocols"]["r065_channel_lesions"]

    def test_audit_substitutions_and_offline_gate(self):
        import torch

        from lares.envs.isaac_pen_spin import channel_lesions as cl
        from lares.envs.isaac_pen_spin.learner_recovery import make_learner

        with open(self.pc["learner"]["source"]) as f:
            src = f.read()
        audit = cl.source_audit(src)
        self.assertTrue(audit["passes"], audit)
        self.assertIn("rotation_correction = 0.0 * (", cl.substituted_source(src, "rotation_correction_lesion", 0.0))
        self.assertIn("error = 0.0 * (home - obj)", cl.substituted_source(src, "centering_clamp", 0.0))
        self.assertIn("extra_curl = 0.0 * self.capture_curl * (", cl.substituted_source(src, "curl_clamp", 0.0))
        obs = TestStrokeLesion._contact_obs(self, 4)
        obs[:, cases.obs_slices()["obj_pos"]] = torch.tensor([0.01, 0.0, 0.1])   # off centre: centering active
        obs[:, cases.obs_slices()["obj_rot"]] = torch.tensor([0.96, 0.0, 0.28, 0.0])  # tilted: rotation correction active
        parent = make_learner(self.pc)
        a = parent(obs, 0, None)
        for lesion in cl.LESIONS:
            ident = cl.lesioned_learner(self.pc, lesion, 1.0, "i")
            self.assertTrue(torch.equal(a, ident(obs, 0, None)), lesion)
            self.assertFalse(torch.allclose(a, cl.make_controller(lesion, self.pc)(obs, 0, None)), lesion)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.npz")
            np.savez(p, obs=obs.unsqueeze(1).repeat(1, 3, 1).numpy(), length=np.array([3, 2, 0, 3]))
            gate = cl.offline_identity_gate(self.pc, p)
            self.assertTrue(gate["passes"])
            self.assertEqual(gate["lesions"]["centering_clamp"]["rows"], 8)

    def test_tail_split_and_interpretation(self):
        from lares.envs.isaac_pen_spin import channel_lesions as cl

        def rows(succ, drops, n, spin, off=0):
            out = []
            for i in range(n):
                tail = succ <= i < n - drops
                out.append({"task_hash": f"h{i}", "sustained": i < succ, "completed": i < n - drops, "tail_hit_fraction": 0.0,
                            "hit_steps": 0, "length": 360, "mean_abs_spin": spin, "mean_transverse": 1.0,
                            "final_center_dist": 0.2 if (tail and i < succ + off) else 0.01})
            return out

        sat = {"in_play_steps": 1, "drive_saturated_fraction": 0.9, "action_saturated_fraction": 0.1,
               "action_saturated_by_axis": [0.1] * 20, "drive_unsaturated_fraction_per_case": [0.1] * 80, "note": ""}
        parent = rows(24, 10, 80, 6.0)
        self.assertEqual(cl.tail_failure_split(rows(24, 10, 80, 6.0, off=5)), {"tail_failures": 46, "off_centre": 5, "slow_spin": 41})
        spin = cl.arm_contrast(parent, rows(4, 10, 80, 1.0), sat, sat)
        self.assertTrue(spin["interpretation"]["spin_channel"])
        ret = cl.arm_contrast(parent, rows(4, 60, 80, 6.0), sat, sat)
        self.assertTrue(ret["interpretation"]["retention_channel"])
        self.assertFalse(ret["interpretation"]["spin_channel"])
        none = cl.arm_contrast(parent, rows(23, 10, 80, 6.0), sat, sat)
        self.assertTrue(none["interpretation"]["prediction_refuted"])
        agg = cl.aggregate({"parent": parent, **{k: rows(4, 10, 80, 1.0) for k in cl.LESIONS}}, {k: sat for k in cl.ARMS})
        self.assertAlmostEqual(agg["paired_success_difference"], -0.25)
        self.assertEqual(agg["primary"], "rotation_correction_lesion")

    def test_manifest_matches_config(self):
        from lares.envs.isaac_pen_spin import channel_lesions as cl

        self.assertEqual(cl.build_manifests(self.pc), [])
        m = EvaluationManifest.load(self.pc["manifest"])
        self.assertEqual((len(m), cases.stage_of(m)), (80, 3))


class TestCurlConfirmation(unittest.TestCase):
    """R-068: the zero-curl controller, the manifest and the qualification rule."""

    @classmethod
    def setUpClass(cls):
        import yaml

        with open("config/shadowhand_spin_stages.yaml") as f:
            cls.pc = yaml.safe_load(f)["research_protocols"]["r068_capture_curl_confirmation"]

    def test_controller_is_the_r065_curl_clamp_and_manifest_is_disjoint(self):
        import torch

        from lares.envs.isaac_pen_spin import channel_lesions as cl
        from lares.envs.isaac_pen_spin import curl_confirmation as cc

        obs = TestStrokeLesion._contact_obs(self, 2)
        obs[:, cases.obs_slices()["fingertip_pos"]] += 0.03       # partial contact: the capture curl is active
        a = cc.make_controller("zero_curl", self.pc)(obs, 0, None)
        b = cl.make_controller("curl_clamp", self.pc)(obs, 0, None)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.allclose(a, cc.make_controller("parent", self.pc)(obs, 0, None)))
        self.assertEqual(cc.build_manifests(self.pc), [])
        m = EvaluationManifest.load(self.pc["manifest"])
        self.assertEqual((len(m), cases.stage_of(m)), (200, 3))
        role = cases.role_manifests(m, cc.roles(self.pc), expected_cases=200, stage=3)
        self.assertEqual(cc.roles(self.pc), {"promotion": (10, 40), "fresh": (40, 200)})
        self.assertEqual([len(role[p]) for p in cc.PANELS], [30, 160])
        self.assertGreaterEqual(len(self.pc["disjoint_from"]), 20)

    def test_aggregate_rule(self):
        from lares.envs.isaac_pen_spin import curl_confirmation as cc

        def rows(succ, drops, n):
            return [{"task_hash": f"h{i}", "sustained": i < succ, "completed": i < n - drops, "tail_hit_fraction": 0.0,
                     "hit_steps": 0, "length": 360, "mean_abs_spin": 5.0, "mean_transverse": 1.0, "final_center_dist": 0.01}
                    for i in range(n)]

        sat = {"in_play_steps": 1, "drive_saturated_fraction": 0.9, "action_saturated_fraction": 0.1,
               "action_saturated_by_axis": [0.1] * 20, "drive_unsaturated_fraction_per_case": [0.1], "note": ""}
        sats = {a: {p: sat for p in cc.PANELS} for a in cc.ARMS}
        good = cc.aggregate({"parent": {"promotion": rows(6, 4, 30), "fresh": rows(30, 20, 160)},
                             "zero_curl": {"promotion": rows(14, 2, 30), "fresh": rows(60, 10, 160)}}, sats)
        self.assertTrue(good["prediction"]["supported"])
        self.assertTrue(good["qualifies_repair_parent"])
        self.assertAlmostEqual(good["paired_success_difference"], 30 / 160)
        flat = cc.aggregate({"parent": {"promotion": rows(6, 4, 30), "fresh": rows(30, 20, 160)},
                             "zero_curl": {"promotion": rows(7, 4, 30), "fresh": rows(32, 20, 160)}}, sats)
        self.assertTrue(flat["prediction"]["refuted"])
        self.assertFalse(flat["qualifies_repair_parent"])
        droppy = cc.aggregate({"parent": {"promotion": rows(6, 4, 30), "fresh": rows(30, 10, 160)},
                               "zero_curl": {"promotion": rows(14, 2, 30), "fresh": rows(60, 40, 160)}}, sats)
        self.assertTrue(droppy["prediction"]["retention_refuted"])
        self.assertFalse(droppy["qualifies_repair_parent"])
