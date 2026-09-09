#!/usr/bin/env python
"""Tests for structured evaluation, diagnostics and promotion (FR-3/FR-4, AC-2).

Run from the project root::

    python -m unittest tests.test_evaluation_report
"""

import json
import os
import sys
import tempfile
import unittest

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from lares.core.obs_schema import ObsField, ObsSchema, active_schema, get_obs_schema  # noqa: E402
from lares.core.policy_validator import validate_policy  # noqa: E402
from lares.core.symbolic_policy import SymbolicPolicy  # noqa: E402
from lares.core.training_pipeline import (  # noqa: E402
    DemoBuffer,
    behavioral_cloning,
    split_buffer_by_episode,
)
from lares.eval.diagnostics import (  # noqa: E402
    MOVED_THRESHOLD,
    REACH_RADIUS,
    RolloutGeometry,
    aggregate_by_axis,
    aggregate_gate_statistics,
)
from lares.eval.promotion import (  # noqa: E402
    PROMOTED_ON_PROGRESS,
    PROMOTED_ON_SUCCESS,
    REJECTED_BELOW_RULE,
    REJECTED_INVALID,
    PromotionPolicy,
    compare_to_incumbent,
)
from lares.eval.report import (  # noqa: E402
    CHECKPOINT_FITTED,
    CHECKPOINT_ZERO_SHOT,
    EvaluationReport,
    build_report,
    episode_table,
)
from lares.eval.runner import EpisodeRecord, ManifestResult  # noqa: E402

OBS_DIM = 39
ACT_DIM = 4
SCHEMA = get_obs_schema("push-v2")


def make_record(**overrides):
    base = dict(
        case_id="development-0000",
        task_id="MT1:push-v3:s2000:000",
        success=0.0,
        episode_return=100.0,
        length=150,
        terminal_obj_to_target=0.2,
        initial_obj_to_target=0.3,
        near_object_rate=0.1,
        first_near_object_step=5,
        grasp_success_rate=0.0,
        action_saturation_rate=0.2,
        timed_out=True,
    )
    base.update(overrides)
    return EpisodeRecord(**base)


def make_result(records, actor="cand", manifest="m", schema_id="push-v2"):
    return ManifestResult(
        actor_name=actor,
        manifest_id=manifest,
        action_mode="deterministic",
        episodes=list(records),
        schema_id=schema_id,
    )


# ---------------------------------------------------------------------------
#  Geometry
# ---------------------------------------------------------------------------


class TestRolloutGeometry(unittest.TestCase):
    def _obs(self, tcp, obj, goal):
        row = np.zeros(OBS_DIM, dtype=np.float32)
        row[0:3] = tcp
        row[4:7] = obj
        row[36:39] = goal
        return torch.tensor(row).unsqueeze(0)

    def test_reports_unavailable_without_a_schema(self):
        geo = RolloutGeometry(None, ACT_DIM)
        self.assertFalse(geo.available)
        self.assertIsNone(geo.geometry_statistics()["final_goal_distance"])
        self.assertIsNone(geo.failure_label(0.0))

    def test_progress_and_drift_decompose_the_object_motion(self):
        geo = RolloutGeometry(SCHEMA, ACT_DIM)
        # Object starts 1.0 from the goal along x, ends 0.4 along x and 0.3 off axis.
        geo.observe(self._obs([0, 0, 0], [0, 0, 0], [1.0, 0, 0]), np.zeros(ACT_DIM))
        geo.finish(self._obs([0, 0, 0], [0.6, 0.3, 0], [1.0, 0, 0]))
        stats = geo.geometry_statistics()
        self.assertAlmostEqual(stats["initial_goal_distance"], 1.0, places=5)
        self.assertAlmostEqual(stats["final_goal_distance"], 0.5, places=5)
        self.assertAlmostEqual(stats["signed_goal_progress"], 0.5, places=5)
        self.assertAlmostEqual(stats["lateral_drift"], 0.3, places=5)
        self.assertAlmostEqual(stats["object_displacement"], 0.67082, places=4)

    def test_min_separation_is_the_closest_approach_not_the_last(self):
        geo = RolloutGeometry(SCHEMA, ACT_DIM)
        geo.observe(self._obs([1, 0, 0], [0, 0, 0], [1, 0, 0]), np.zeros(ACT_DIM))
        geo.observe(self._obs([0.01, 0, 0], [0, 0, 0], [1, 0, 0]), np.zeros(ACT_DIM))
        geo.finish(self._obs([2, 0, 0], [0, 0, 0], [1, 0, 0]))
        stats = geo.geometry_statistics()
        self.assertAlmostEqual(stats["min_tcp_object_distance"], 0.01, places=5)
        self.assertAlmostEqual(stats["final_tcp_object_distance"], 2.0, places=5)

    def test_per_axis_action_statistics(self):
        geo = RolloutGeometry(SCHEMA, ACT_DIM)
        for a in ([1.0, 0.0, 0.0, 0.0], [1.0, 0.5, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]):
            geo.observe(self._obs([0, 0, 0], [0, 0, 0], [1, 0, 0]), np.array(a))
        stats = geo.action_statistics()
        self.assertEqual(stats["action_saturation_by_axis"][0], 1.0)
        self.assertEqual(stats["action_saturation_by_axis"][1], 0.0)
        self.assertAlmostEqual(stats["action_variation_by_axis"][1], 0.5, places=5)
        self.assertAlmostEqual(stats["action_variation_by_axis"][0], 0.0, places=5)

    def _labelled(self, obj_start, obj_end, tcp_min, goal=(1.0, 0.0, 0.0)):
        geo = RolloutGeometry(SCHEMA, ACT_DIM)
        geo.observe(self._obs([5, 0, 0], obj_start, goal), np.zeros(ACT_DIM))
        geo.observe(self._obs(np.asarray(obj_start) + [tcp_min, 0, 0], obj_start, goal),
                    np.zeros(ACT_DIM))
        geo.finish(self._obs([5, 0, 0], obj_end, goal))
        return geo

    def test_failure_label_success(self):
        geo = self._labelled([0, 0, 0], [1, 0, 0], 0.01)
        self.assertEqual(geo.failure_label(1.0), "success")

    def test_failure_label_never_reached_object(self):
        geo = self._labelled([0, 0, 0], [0, 0, 0], REACH_RADIUS + 0.1)
        self.assertEqual(geo.failure_label(0.0), "never_reached_object")

    def test_failure_label_object_not_moved(self):
        geo = self._labelled([0, 0, 0], [MOVED_THRESHOLD / 2, 0, 0], 0.01)
        self.assertEqual(geo.failure_label(0.0), "object_not_moved")

    def test_failure_label_pushed_away(self):
        geo = self._labelled([0, 0, 0], [-0.2, 0, 0], 0.01)
        self.assertEqual(geo.failure_label(0.0), "pushed_away_from_goal")

    def test_failure_label_stopped_short(self):
        geo = self._labelled([0, 0, 0], [0.3, 0, 0], 0.01)
        self.assertEqual(geo.failure_label(0.0), "stopped_short_of_goal")

    def test_thresholds_are_reported_with_the_label(self):
        geo = RolloutGeometry(SCHEMA, ACT_DIM)
        self.assertIn("reach_radius_m", geo.thresholds())


class TestAggregation(unittest.TestCase):
    def test_by_axis_ignores_episodes_that_measured_nothing(self):
        self.assertEqual(aggregate_by_axis([[1.0, 3.0], None, [3.0, 1.0]], 2), [2.0, 2.0])
        self.assertIsNone(aggregate_by_axis([None, None], 2))

    def test_gate_aggregation_separates_never_crossed_from_crossed_late(self):
        crossed = {
            "contact": {
                "mean": 0.4, "min": 0.0, "max": 1.0, "occupancy_above_half": 0.5,
                "first_crossing_step": 10, "transitions": 2, "num_steps": 20,
            }
        }
        never = {
            "contact": {
                "mean": 0.01, "min": 0.0, "max": 0.02, "occupancy_above_half": 0.0,
                "first_crossing_step": None, "transitions": 0, "num_steps": 20,
            }
        }
        agg = aggregate_gate_statistics([crossed, never])["contact"]
        self.assertEqual(agg["episodes_measured"], 2)
        self.assertEqual(agg["episodes_with_crossing"], 1)
        self.assertEqual(agg["mean_first_crossing_step"], 10.0)

    def test_gate_aggregation_is_none_when_no_policy_exposed_gates(self):
        self.assertIsNone(aggregate_gate_statistics([None, None]))


# ---------------------------------------------------------------------------
#  Report
# ---------------------------------------------------------------------------


class TestEvaluationReport(unittest.TestCase):
    def test_unknown_checkpoint_is_refused(self):
        with self.assertRaises(ValueError):
            EvaluationReport("c", "m", "after_lunch", "deterministic")

    def test_report_carries_a_row_per_episode(self):
        result = make_result([make_record(case_id=f"c{i}") for i in range(4)])
        report = build_report(result, "cand", CHECKPOINT_FITTED)
        self.assertEqual(len(report.episodes), 4)
        self.assertEqual(report.rollout.num_episodes, 4)

    def test_success_is_primary_and_distances_are_secondary(self):
        result = make_result(
            [
                make_record(success=1.0, final_goal_distance=0.01, signed_goal_progress=0.2),
                make_record(case_id="c1", success=0.0, final_goal_distance=0.2,
                            signed_goal_progress=0.0),
            ]
        )
        report = build_report(result, "cand", CHECKPOINT_FITTED)
        self.assertAlmostEqual(report.rollout.success_rate, 0.5)
        self.assertAlmostEqual(report.rollout.final_goal_distance["mean"], 0.105, places=5)
        self.assertAlmostEqual(report.rollout.signed_goal_progress["mean"], 0.1, places=5)

    def test_unmeasured_geometry_is_unavailable_not_zero(self):
        report = build_report(make_result([make_record()]), "cand", CHECKPOINT_FITTED)
        self.assertIsNone(report.rollout.final_goal_distance)
        self.assertIn("rollout.final_goal_distance", report.unavailable_fields)
        self.assertIn("rollout.gate_statistics", report.unavailable_fields)
        self.assertIn("fitting", report.unavailable_fields)
        self.assertIn("validity", report.unavailable_fields)

    def test_gate_statistics_become_phase_occupancy_and_transitions(self):
        gates = {
            "contact": {
                "mean": 0.6, "min": 0.0, "max": 1.0, "occupancy_above_half": 0.75,
                "first_crossing_step": 3, "transitions": 2, "num_steps": 12,
            }
        }
        result = make_result([make_record(gate_statistics=gates)])
        report = build_report(result, "cand", CHECKPOINT_FITTED)
        self.assertAlmostEqual(report.rollout.phase_occupancy["contact"], 0.75)
        self.assertAlmostEqual(report.rollout.phase_transitions["contact"], 2.0)
        self.assertNotIn("rollout.gate_statistics", report.unavailable_fields)

    def test_failure_labels_are_counted(self):
        result = make_result(
            [
                make_record(failure_label="never_reached_object"),
                make_record(case_id="c1", failure_label="never_reached_object"),
                make_record(case_id="c2", failure_label="pushed_away_from_goal"),
            ]
        )
        report = build_report(result, "cand", CHECKPOINT_FITTED)
        self.assertEqual(report.rollout.failure_labels["never_reached_object"], 2)
        self.assertEqual(report.rollout.failure_labels["pushed_away_from_goal"], 1)

    def test_validity_section_mirrors_the_validator_categories(self):
        class Bad(SymbolicPolicy):
            def __init__(self, o, a):
                super().__init__(o, a)
                self.w = nn.Parameter(torch.tensor(1.0))
                self.hidden = nn.Parameter(torch.tensor(1.0))

            def forward(self, obs):
                tcp = self.obs_field(obs, "tcp")
                mean = self.w * tcp
                return mean, torch.ones_like(mean)

            def get_param_ranges(self):
                return {"w": (0.0, 5.0)}

        with active_schema(SCHEMA):
            policy = Bad(OBS_DIM, ACT_DIM)
        vr = validate_policy(policy, OBS_DIM, ACT_DIM, schema=SCHEMA)
        report = build_report(
            make_result([make_record()]), "cand", CHECKPOINT_ZERO_SHOT, validation_report=vr
        )
        self.assertFalse(report.validity.range_coverage_ok)
        self.assertFalse(report.validity.shape_ok)
        self.assertTrue(report.validity.errors)
        self.assertNotIn("validity", report.unavailable_fields)

    def test_measurement_notes_flag_proximity_and_the_trivial_timeout_rate(self):
        report = build_report(make_result([make_record()]), "cand", CHECKPOINT_FITTED)
        self.assertIn("proximity", report.measurement_notes["near_object"])
        self.assertIn("does not terminate early", report.measurement_notes["timeout_rate"])

    def test_report_round_trips_through_json(self):
        result = make_result([make_record(final_goal_distance=0.2)])
        report = build_report(result, "cand", CHECKPOINT_FITTED)
        with tempfile.TemporaryDirectory() as tmp:
            path = report.save(os.path.join(tmp, "r.json"))
            loaded = EvaluationReport.load(path)
        self.assertEqual(loaded["candidate_id"], "cand")
        self.assertEqual(len(loaded["episodes"]), 1)
        self.assertEqual(loaded["checkpoint"], CHECKPOINT_FITTED)

    def test_episode_table_renders_missing_values_as_not_available(self):
        report = build_report(make_result([make_record()]), "cand", CHECKPOINT_FITTED)
        table = episode_table(report)
        self.assertIn("n/a", table)
        self.assertIn("development-0000", table)

    def test_headline_names_the_checkpoint_and_action_mode(self):
        report = build_report(make_result([make_record()]), "cand", CHECKPOINT_ZERO_SHOT)
        self.assertIn(CHECKPOINT_ZERO_SHOT, report.headline())
        self.assertIn("deterministic", report.headline())


# ---------------------------------------------------------------------------
#  Fitting diagnostics
# ---------------------------------------------------------------------------


class SmallPolicy(SymbolicPolicy):
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.w = nn.Parameter(torch.tensor(1.0))
        self.wg = nn.Parameter(torch.tensor(0.5))
        self.sharp = nn.Parameter(torch.tensor(40.0))
        self.th = nn.Parameter(torch.tensor(0.06))
        self.grip = nn.Parameter(torch.tensor(0.0))
        self.log_std = nn.Parameter(torch.tensor(-1.0))

    def forward(self, obs):
        tcp = self.obs_field(obs, "tcp")
        obj = self.obs_field(obs, "obj")
        goal = self.obs_field(obs, "goal")
        d = torch.norm(obj - tcp, dim=-1, keepdim=True) + 1e-8
        gate = torch.sigmoid(self.sharp * (self.th - d))
        self.record_gate("contact", gate)
        move = (1 - gate) * self.w * ((obj - tcp) / d) + gate * self.wg * (goal - obj)
        grip = self.grip.unsqueeze(0).expand(obs.shape[0], 1)
        mean = torch.cat([move, grip], dim=1)
        return mean, torch.exp(self.log_std) * torch.ones_like(mean)

    def get_param_ranges(self):
        return {
            "w": (0.1, 10.0), "wg": (0.0, 5.0), "sharp": (1.0, 200.0),
            "th": (0.01, 0.3), "grip": (-2.0, 2.0), "log_std": (-5.0, 0.0),
        }


def make_buffer(episodes=10, steps=20, seed=0, labelled=True):
    rng = np.random.default_rng(seed)
    buf = DemoBuffer()
    for ep in range(episodes):
        for _ in range(steps):
            obs = (0.1 * rng.standard_normal(OBS_DIM)).astype(np.float32)
            action = np.clip(rng.standard_normal(ACT_DIM), -1, 1)
            buf.add(obs, action, 0.0, obs, 0.0, episode_id=f"ep{ep}" if labelled else "")
    return buf


class TestFittingDiagnostics(unittest.TestCase):
    def test_split_is_by_episode_and_never_overlaps(self):
        buf = make_buffer()
        train, val, info = split_buffer_by_episode(buf, 0.2, seed=0)
        self.assertEqual(info["mode"], "by_episode")
        self.assertEqual(set(train) & set(val), set())
        self.assertEqual(len(train) + len(val), len(buf))
        # No episode may straddle the split.
        groups = buf.episode_index()
        train_set, val_set = set(train), set(val)
        for idx in groups.values():
            self.assertTrue(set(idx) <= train_set or set(idx) <= val_set)

    def test_split_is_reproducible(self):
        buf = make_buffer()
        a = split_buffer_by_episode(buf, 0.2, seed=3)[1]
        b = split_buffer_by_episode(buf, 0.2, seed=3)[1]
        self.assertTrue(np.array_equal(a, b))

    def test_unlabelled_buffer_refuses_a_trajectory_split_and_says_why(self):
        train, val, info = split_buffer_by_episode(make_buffer(labelled=False))
        self.assertIsNone(val)
        self.assertEqual(info["mode"], "none")
        self.assertIn("episode ids", info["reason"])

    def test_bc_reports_the_fitting_diagnostics_the_report_needs(self):
        with active_schema(SCHEMA):
            policy = SmallPolicy(OBS_DIM, ACT_DIM)
        stats = behavioral_cloning(
            policy, make_buffer(), num_steps=40, batch_size=16, log_interval=0
        )
        self.assertEqual(len(stats["train_loss_by_axis"]), ACT_DIM)
        self.assertEqual(len(stats["validation_loss_by_axis"]), ACT_DIM)
        self.assertIn("mean", stats["gradient_norms"])
        self.assertIn("fraction_clipped", stats["gradient_norms"])
        self.assertIn("w", stats["bound_activity"])
        self.assertIn("contact", stats["loss_by_phase"])
        self.assertIsNotNone(stats["train_loss_start"])

    def test_bc_captures_the_intermediate_checkpoint(self):
        with active_schema(SCHEMA):
            policy = SmallPolicy(OBS_DIM, ACT_DIM)
        seen = {}
        behavioral_cloning(
            policy, make_buffer(), num_steps=40, batch_size=16, log_interval=0,
            checkpoint_callback=lambda step, pol: seen.update(step=step),
        )
        self.assertEqual(seen["step"], 20)

    def test_fitting_section_is_populated_from_bc_stats(self):
        with active_schema(SCHEMA):
            policy = SmallPolicy(OBS_DIM, ACT_DIM)
        stats = behavioral_cloning(
            policy, make_buffer(), num_steps=40, batch_size=16, log_interval=0
        )
        report = build_report(
            make_result([make_record()]), "cand", CHECKPOINT_FITTED, bc_stats=stats
        )
        self.assertIsNotNone(report.fitting.train_loss_by_axis)
        self.assertIsNotNone(report.fitting.validation_loss_by_axis)
        self.assertIsNotNone(report.fitting.bound_activity)
        self.assertNotIn("fitting", report.unavailable_fields)


# ---------------------------------------------------------------------------
#  Promotion
# ---------------------------------------------------------------------------


def screening_entry(cid, success, progress, ret, valid=True):
    record = make_record(success=success, signed_goal_progress=progress, episode_return=ret)
    report = build_report(make_result([record], actor=cid), cid, CHECKPOINT_FITTED)
    return {"candidate_id": cid, "report": report, "valid": valid}


class TestPromotionPolicy(unittest.TestCase):
    def setUp(self):
        self.policy = PromotionPolicy(screen_episodes=10, expanded_episodes=30)

    def test_budgets_default_to_the_documented_ten_and_thirty(self):
        self.assertEqual(self.policy.screen_episodes, 10)
        self.assertEqual(self.policy.expanded_episodes, 30)

    def test_any_success_clears_the_screen(self):
        decisions = self.policy.screen(
            [screening_entry("a", 0.2, -1.0, 1.0), screening_entry("b", 0.0, 5.0, 900.0)]
        )
        by_id = {d["candidate_id"]: d for d in decisions}
        self.assertTrue(by_id["a"]["promoted"])
        self.assertEqual(by_id["a"]["reason"], PROMOTED_ON_SUCCESS)
        self.assertFalse(by_id["b"]["promoted"])
        self.assertEqual(by_id["b"]["reason"], REJECTED_BELOW_RULE)

    def test_when_nothing_succeeds_the_best_progress_still_advances(self):
        decisions = self.policy.screen(
            [
                screening_entry("a", 0.0, 0.01, 10.0),
                screening_entry("b", 0.0, 0.30, 20.0),
                screening_entry("c", 0.0, 0.20, 30.0),
            ]
        )
        promoted = {d["candidate_id"] for d in decisions if d["promoted"]}
        self.assertEqual(promoted, {"b", "c"})
        for d in decisions:
            if d["promoted"]:
                self.assertEqual(d["reason"], PROMOTED_ON_PROGRESS)

    def test_invalid_candidates_never_advance(self):
        decisions = self.policy.screen([screening_entry("a", 0.9, 1.0, 1.0, valid=False)])
        self.assertFalse(decisions[0]["promoted"])
        self.assertEqual(decisions[0]["reason"], REJECTED_INVALID)

    def test_policy_serialises_so_the_record_shows_the_rule(self):
        d = self.policy.to_dict()
        self.assertIn("screen_min_success", d)
        self.assertIn("fallback_top_k", d)


class TestIncumbentComparison(unittest.TestCase):
    def _result(self, successes, returns, name):
        return make_result(
            [
                make_record(case_id=f"c{i}", success=s, episode_return=r)
                for i, (s, r) in enumerate(zip(successes, returns))
            ],
            actor=name,
        )

    def test_first_contender_is_promoted_with_no_incumbent(self):
        d = compare_to_incumbent(self._result([0, 0], [1, 1], "a"), None, "a", None)
        self.assertTrue(d.promoted)
        self.assertEqual(d.reason, "no_incumbent")
        self.assertIsNone(d.paired_success)

    def test_higher_paired_success_promotes(self):
        d = compare_to_incumbent(
            self._result([1, 1], [10, 10], "a"),
            self._result([0, 0], [50, 50], "b"),
            "a",
            "b",
        )
        self.assertTrue(d.promoted)
        self.assertEqual(d.reason, "higher_paired_success")
        self.assertAlmostEqual(d.paired_success["mean_difference"], 1.0)

    def test_lower_paired_success_does_not_promote_even_with_better_return(self):
        d = compare_to_incumbent(
            self._result([0, 0], [900, 900], "a"),
            self._result([1, 1], [10, 10], "b"),
            "a",
            "b",
        )
        self.assertFalse(d.promoted)
        self.assertEqual(d.reason, "lower_paired_success")

    def test_return_breaks_a_tie_only_when_success_is_equal(self):
        better = compare_to_incumbent(
            self._result([0, 0], [20, 20], "a"),
            self._result([0, 0], [10, 10], "b"),
            "a",
            "b",
        )
        self.assertTrue(better.promoted)
        self.assertEqual(better.reason, "tied_success_higher_paired_return")
        worse = compare_to_incumbent(
            self._result([0, 0], [5, 5], "a"),
            self._result([0, 0], [10, 10], "b"),
            "a",
            "b",
        )
        self.assertFalse(worse.promoted)

    def test_comparison_reports_an_interval_for_the_difference(self):
        d = compare_to_incumbent(
            self._result([1, 0, 1, 0], [1, 2, 3, 4], "a"),
            self._result([0, 0, 0, 0], [1, 1, 1, 1], "b"),
            "a",
            "b",
        )
        self.assertEqual(d.paired_success["n_paired"], 4)
        self.assertLess(d.paired_success["ci95_low"], d.paired_success["mean_difference"])
        self.assertGreater(d.paired_success["ci95_high"], d.paired_success["mean_difference"])

    def test_comparison_serialises_for_the_experiment_record(self):
        d = compare_to_incumbent(
            self._result([1], [1], "a"), self._result([0], [1], "b"), "a", "b"
        )
        self.assertIn("paired_success", json.dumps(d.to_dict(), default=str))


if __name__ == "__main__":
    unittest.main()
