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


if __name__ == "__main__":
    unittest.main()
