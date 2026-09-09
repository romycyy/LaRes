#!/usr/bin/env python
"""Tests for intervention-guided repair (``spec.md`` FR-9, AC-6).

Run from the project root::

    python -m unittest tests.test_repair
"""

import os
import sys
import unittest

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lares.core.obs_schema import get_obs_schema  # noqa: E402
from lares.eval.report import CHECKPOINT_FITTED, build_report  # noqa: E402
from lares.eval.runner import EpisodeRecord, ManifestResult  # noqa: E402
from lares.fitting import load_bank  # noqa: E402
from lares.repair import (  # noqa: E402
    MIN_CLUSTER_SIZE,
    PROTECTED_REGRESSION_TOLERANCE,
    SUBSTITUTION_EXPERT_AXIS,
    SUBSTITUTION_EXPERT_PHASE,
    SUBSTITUTION_EXPERT_PREFIX,
    SUBSTITUTION_FORCE_GATE,
    SUBSTITUTION_NONE,
    Explanation,
    InterventionActor,
    Substitution,
    available_gates,
    competing_explanations,
    episode_metric_mean,
    evaluate_repair,
    find_clusters,
    gate_override_is_effective,
    rank_explanations,
)
from lares.repair.hypotheses import InterventionOutcome  # noqa: E402
from lares.repair.repair import (  # noqa: E402
    COMPONENT_SCOPES,
    PROTECTED_PROGRESS_TOLERANCE,
    REPAIR_SEARCH,
    BoundedRepair,
    accept_repair,
    apply_repair,
    bounded_change,
    gate_parameters,
    phase_subset,
    progress_regression,
    propose_repair,
    protected_cases,
    repair_stayed_in_scope,
    scope_for,
    scoped_sensitivity,
    scalar_parameters,
    search_repair,
    select_parameters,
)

OBS_DIM = 39
ACT_DIM = 4
SCHEMA = get_obs_schema("push-v2")

try:
    import metaworld  # noqa: F401

    HAS_METAWORLD = True
except ImportError:
    HAS_METAWORLD = False


def episode(case_id, label, success=0.0, **overrides):
    base = dict(
        case_id=case_id,
        task_id="t",
        success=success,
        episode_return=100.0,
        length=150,
        terminal_obj_to_target=0.2,
        initial_obj_to_target=0.3,
        near_object_rate=0.1,
        first_near_object_step=None,
        grasp_success_rate=0.0,
        action_saturation_rate=0.0,
        timed_out=True,
        failure_label=label,
        min_tcp_object_distance=0.08,
        object_displacement=0.03,
        signed_goal_progress=0.01,
        final_goal_distance=0.2,
        lateral_drift=0.02,
    )
    base.update(overrides)
    return EpisodeRecord(**base)


def result_of(episodes, name="r"):
    return ManifestResult(
        actor_name=name, manifest_id="m", action_mode="deterministic",
        episodes=list(episodes), schema_id="push-v2",
    )


def report_of(episodes):
    return build_report(result_of(episodes), "c", CHECKPOINT_FITTED)


# ---------------------------------------------------------------------------
#  Substitutions
# ---------------------------------------------------------------------------


class TestSubstitution(unittest.TestCase):
    def test_an_unknown_kind_is_refused(self):
        with self.assertRaises(ValueError):
            Substitution(kind="hope")

    def test_each_kind_requires_the_field_it_needs(self):
        with self.assertRaises(ValueError):
            Substitution(kind=SUBSTITUTION_EXPERT_AXIS)
        with self.assertRaises(ValueError):
            Substitution(kind=SUBSTITUTION_EXPERT_PHASE)
        with self.assertRaises(ValueError):
            Substitution(kind=SUBSTITUTION_EXPERT_PREFIX, prefix_steps=0)
        with self.assertRaises(ValueError):
            Substitution(kind=SUBSTITUTION_FORCE_GATE)

    def test_only_the_control_is_not_an_intervention(self):
        self.assertFalse(Substitution(kind=SUBSTITUTION_NONE).is_intervention)
        self.assertTrue(Substitution(kind=SUBSTITUTION_EXPERT_AXIS, axes=(2,)).is_intervention)
        self.assertTrue(
            Substitution(kind=SUBSTITUTION_FORCE_GATE, gate="contact").is_intervention
        )

    def test_the_label_names_the_axis_rather_than_its_index(self):
        self.assertIn("dz", Substitution(kind=SUBSTITUTION_EXPERT_AXIS, axes=(2,)).label())
        self.assertIn(
            "gripper", Substitution(kind=SUBSTITUTION_EXPERT_AXIS, axes=(3,)).label()
        )

    def test_it_serialises_with_the_intervention_flag(self):
        d = Substitution(kind=SUBSTITUTION_EXPERT_PREFIX, prefix_steps=40).to_dict()
        self.assertTrue(d["is_intervention"])
        self.assertIn("label", d)


# ---------------------------------------------------------------------------
#  Gate overrides
# ---------------------------------------------------------------------------


class TestGateOverride(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.structure = next(
            s for s in load_bank(family="simple") if s.structure_id == "simple_reach_push"
        )
        cls.obs = 0.05 * torch.randn(8, OBS_DIM, generator=torch.Generator().manual_seed(0))

    def test_forcing_a_gate_changes_the_action(self):
        policy = self.structure.build(SCHEMA)
        with torch.no_grad():
            before = torch.tanh(policy(self.obs)[0]).clone()
        policy.set_gate_override("contact", 1.0)
        with torch.no_grad():
            after = torch.tanh(policy(self.obs)[0])
        self.assertFalse(torch.allclose(before, after))

    def test_clearing_the_override_restores_the_action(self):
        policy = self.structure.build(SCHEMA)
        with torch.no_grad():
            before = torch.tanh(policy(self.obs)[0]).clone()
        policy.set_gate_override("contact", 0.0)
        policy(self.obs)
        policy.clear_gate_overrides()
        with torch.no_grad():
            after = torch.tanh(policy(self.obs)[0])
        self.assertTrue(torch.equal(before, after))

    def test_resetting_diagnostics_does_not_clear_an_override(self):
        # An intervention is set deliberately and must survive an episode boundary.
        policy = self.structure.build(SCHEMA)
        policy.set_gate_override("contact", 1.0)
        policy.reset_diagnostics()
        self.assertEqual(policy.gate_overrides, {"contact": 1.0})

    def test_telemetry_reports_the_forced_value(self):
        policy = self.structure.build(SCHEMA)
        policy.set_gate_override("contact", 1.0)
        policy(self.obs)
        self.assertAlmostEqual(float(policy.last_gates["contact"].mean()), 1.0, places=6)

    def test_effectiveness_is_checked_rather_than_assumed(self):
        policy = self.structure.build(SCHEMA)
        self.assertTrue(gate_override_is_effective(policy, "contact", self.obs))
        # A gate the policy never records is not a lever.
        self.assertFalse(gate_override_is_effective(policy, "imaginary", self.obs))

    def test_the_effectiveness_check_leaves_no_override_behind(self):
        policy = self.structure.build(SCHEMA)
        gate_override_is_effective(policy, "contact", self.obs)
        self.assertEqual(policy.gate_overrides, {})

    def test_available_gates_marks_which_are_forceable(self):
        policy = self.structure.build(SCHEMA)
        gates = available_gates(policy, self.obs)
        self.assertTrue(gates)
        self.assertEqual(gates[0]["gate"], "contact")
        self.assertTrue(gates[0]["forceable"])


# ---------------------------------------------------------------------------
#  Clusters
# ---------------------------------------------------------------------------


class TestClusters(unittest.TestCase):
    def test_a_recurrent_failure_becomes_a_cluster(self):
        report = report_of(
            [episode(f"c{i}", "never_reached_object") for i in range(5)]
        )
        clusters = find_clusters(report)
        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0].label, "never_reached_object")
        self.assertEqual(clusters[0].size, 5)
        self.assertTrue(clusters[0].is_recurrent)

    def test_a_one_off_failure_is_not_worth_a_diagnostic_budget(self):
        report = report_of(
            [episode("c0", "pushed_away_from_goal")]
            + [episode(f"c{i}", "never_reached_object") for i in range(1, 5)]
        )
        labels = {c.label for c in find_clusters(report)}
        self.assertEqual(labels, {"never_reached_object"})

    def test_successes_never_form_a_cluster(self):
        report = report_of([episode(f"c{i}", "success", success=1.0) for i in range(6)])
        self.assertEqual(find_clusters(report), [])

    def test_clusters_are_ordered_largest_first(self):
        report = report_of(
            [episode(f"a{i}", "never_reached_object") for i in range(3)]
            + [episode(f"b{i}", "pushed_away_from_goal") for i in range(6)]
        )
        self.assertEqual(find_clusters(report)[0].label, "pushed_away_from_goal")

    def test_a_cluster_carries_the_geometry_the_explanation_reasons_about(self):
        report = report_of(
            [episode(f"c{i}", "never_reached_object", min_tcp_object_distance=0.11)
             for i in range(4)]
        )
        summary = find_clusters(report)[0].summary
        self.assertAlmostEqual(summary["min_tcp_object_distance"], 0.11, places=6)
        self.assertIn("lateral_drift", summary)

    def test_the_minimum_size_is_declared_not_inline(self):
        self.assertGreaterEqual(MIN_CLUSTER_SIZE, 2)


# ---------------------------------------------------------------------------
#  Competing explanations
# ---------------------------------------------------------------------------


class TestExplanations(unittest.TestCase):
    def _cluster(self, label, size=5):
        return find_clusters(report_of([episode(f"c{i}", label) for i in range(size)]))[0]

    def test_every_failure_label_gets_at_least_two_accounts(self):
        # Derived from the diagnostics module rather than listed here, so a new
        # failure label cannot quietly produce a cluster nothing can explain.
        from lares.eval.diagnostics import FAILURE_LABELS

        for label in FAILURE_LABELS:
            if label == "success":
                continue
            with self.subTest(label=label):
                self.assertGreaterEqual(len(competing_explanations(self._cluster(label))), 2)

    def test_the_two_accounts_blame_different_components(self):
        explanations = competing_explanations(self._cluster("never_reached_object"))
        suspects = {e.suspected_component for e in explanations}
        self.assertEqual(len(suspects), len(explanations))

    def test_each_account_carries_a_substitution_that_tests_it(self):
        for e in competing_explanations(self._cluster("pushed_away_from_goal")):
            self.assertTrue(e.substitution.is_intervention)
            self.assertIn(e.predicted_direction, ("increase", "decrease"))

    def test_a_forceable_gate_adds_an_account(self):
        cluster = self._cluster("never_reached_object")
        without = competing_explanations(cluster)
        with_gate = competing_explanations(
            cluster, gates=[{"gate": "contact", "forceable": True}]
        )
        self.assertEqual(len(with_gate), len(without) + 1)
        self.assertTrue(any("contact" in e.name for e in with_gate))

    def test_a_telemetry_only_gate_adds_nothing(self):
        cluster = self._cluster("never_reached_object")
        self.assertEqual(
            len(competing_explanations(cluster, gates=[{"gate": "g", "forceable": False}])),
            len(competing_explanations(cluster)),
        )

    def test_an_account_is_supported_only_when_the_metric_moves_as_predicted(self):
        e = competing_explanations(self._cluster("never_reached_object"))[0]
        self.assertTrue(e.check(0.09, 0.03)["supported"])
        self.assertFalse(e.check(0.09, 0.11)["supported"])

    def test_an_unmeasurable_outcome_is_not_support(self):
        e = competing_explanations(self._cluster("never_reached_object"))[0]
        self.assertFalse(e.check(None, 0.03)["measurable"])


class TestRanking(unittest.TestCase):
    def make_outcome(self, name, supported, delta):
        return InterventionOutcome(
            explanation=name,
            substitution={},
            focused_metric="min_tcp_object_distance",
            control_value=0.09,
            intervened_value=0.09 + delta,
            check={
                "measurable": True, "supported": supported, "delta": delta,
                "metric": "min_tcp_object_distance", "before": 0.09,
                "after": 0.09 + delta, "predicted_direction": "decrease",
            },
            control_success=0.0,
            intervened_success=0.0,
        )

    def test_supported_accounts_come_first(self):
        ranked = rank_explanations(
            [self.make_outcome("a", False, 0.01), self.make_outcome("b", True, -0.03)]
        )
        self.assertEqual(ranked[0].explanation, "b")

    def test_among_supported_the_larger_movement_wins(self):
        ranked = rank_explanations(
            [self.make_outcome("small", True, -0.01), self.make_outcome("big", True, -0.05)]
        )
        self.assertEqual(ranked[0].explanation, "big")

    def test_an_unmeasurable_outcome_ranks_last(self):
        unmeasurable = self.make_outcome("u", False, 0.0)
        unmeasurable.check = {"measurable": False}
        ranked = rank_explanations([unmeasurable, self.make_outcome("ok", True, -0.02)])
        self.assertEqual(ranked[0].explanation, "ok")


# ---------------------------------------------------------------------------
#  Repair acceptance
# ---------------------------------------------------------------------------


class TestRepairAcceptance(unittest.TestCase):
    def setUp(self):
        self.focused = ["f0", "f1", "f2"]
        self.protected = ["p0", "p1", "p2", "p3"]

    def _pair(self, focused_after, protected_after_success):
        before = result_of(
            [episode(c, "never_reached_object", min_tcp_object_distance=0.09)
             for c in self.focused]
            + [episode(c, "success", success=1.0) for c in self.protected]
        )
        after = result_of(
            [episode(c, "never_reached_object", min_tcp_object_distance=focused_after)
             for c in self.focused]
            + [episode(c, "success", success=s)
               for c, s in zip(self.protected, protected_after_success)]
        )
        return before, after

    def test_a_repair_that_helps_and_holds_is_accepted(self):
        before, after = self._pair(0.03, [1.0, 1.0, 1.0, 1.0])
        decision = evaluate_repair(
            before, after, "min_tcp_object_distance", "decrease",
            self.focused, self.protected,
        )
        self.assertTrue(decision.accepted)
        self.assertTrue(decision.focused_check["as_predicted"])
        self.assertTrue(decision.protected_check["passed"])

    def test_a_repair_that_does_not_move_the_metric_is_rejected(self):
        before, after = self._pair(0.095, [1.0, 1.0, 1.0, 1.0])
        decision = evaluate_repair(
            before, after, "min_tcp_object_distance", "decrease",
            self.focused, self.protected,
        )
        self.assertFalse(decision.accepted)
        self.assertIn("did not move", decision.reason)

    def test_a_repair_that_breaks_protected_cases_is_rejected(self):
        # Fixes the cluster, destroys everything that already worked.
        before, after = self._pair(0.01, [0.0, 0.0, 0.0, 0.0])
        decision = evaluate_repair(
            before, after, "min_tcp_object_distance", "decrease",
            self.focused, self.protected,
        )
        self.assertFalse(decision.accepted)
        self.assertIn("regressed", decision.reason)
        self.assertFalse(decision.protected_check["passed"])

    def test_a_small_regression_inside_the_tolerance_is_allowed(self):
        before, after = self._pair(0.01, [1.0, 1.0, 1.0, 1.0])
        decision = evaluate_repair(
            before, after, "min_tcp_object_distance", "decrease",
            self.focused, self.protected, tolerance=0.25,
        )
        self.assertTrue(decision.accepted)

    def test_an_unmeasurable_focused_metric_blocks_acceptance(self):
        before = result_of([episode(c, "never_reached_object", min_tcp_object_distance=None)
                            for c in self.focused])
        decision = evaluate_repair(
            before, before, "min_tcp_object_distance", "decrease", self.focused, []
        )
        self.assertFalse(decision.accepted)
        self.assertIn("unavailable", decision.reason)

    def test_the_tolerance_is_declared_in_the_module(self):
        self.assertGreater(PROTECTED_REGRESSION_TOLERANCE, 0.0)
        self.assertLess(PROTECTED_REGRESSION_TOLERANCE, 0.5)

    def test_the_decision_serialises_with_both_checks(self):
        before, after = self._pair(0.03, [1.0, 1.0, 1.0, 1.0])
        d = evaluate_repair(
            before, after, "min_tcp_object_distance", "decrease",
            self.focused, self.protected,
        ).to_dict()
        self.assertIn("focused_check", d)
        self.assertIn("protected_check", d)


class TestMetricRestriction(unittest.TestCase):
    def test_the_mean_is_restricted_to_the_named_cases(self):
        result = result_of(
            [episode("a", "never_reached_object", min_tcp_object_distance=0.10),
             episode("b", "never_reached_object", min_tcp_object_distance=0.20)]
        )
        self.assertAlmostEqual(
            episode_metric_mean(result, "min_tcp_object_distance", ["a"]), 0.10
        )
        self.assertAlmostEqual(
            episode_metric_mean(result, "min_tcp_object_distance"), 0.15
        )

    def test_an_unmeasured_field_returns_nothing(self):
        result = result_of([episode("a", "x", min_tcp_object_distance=None)])
        self.assertIsNone(episode_metric_mean(result, "min_tcp_object_distance"))


class TestExperimentRecords(unittest.TestCase):
    """FR-10: an intervention must be traceable and must never be rankable."""

    @classmethod
    def setUpClass(cls):
        cls.structure = next(
            s for s in load_bank(family="simple") if s.structure_id == "simple_standoff"
        )

    def _payload(self, with_repair=True, accepted=True):
        payload = {
            "structure_id": "simple_standoff",
            "manifest_id": "dev#head10",
            "studies": [{
                "cluster": {"label": "never_reached_object", "case_ids": ["a", "b", "c"]},
                "supported": "approach_control",
                "outcomes": [
                    {"explanation": "approach_control",
                     "substitution": {"label": "expert drives the first 40 steps"},
                     "focused_metric": "min_tcp_object_distance",
                     "intervened_success": 0.0, "check": {"supported": True}},
                    {"explanation": "vertical_axis",
                     "substitution": {"label": "expert drives dz"},
                     "focused_metric": "min_tcp_object_distance",
                     "intervened_success": 0.0, "check": {"supported": False}},
                ],
            }],
        }
        if with_repair:
            payload["repair"] = {
                "attempted": True, "success_after": 0.1,
                "repair": {"parameters": ["gain"]},
                "decision": {"accepted": accepted, "reason": ""},
            }
        return payload

    def test_one_record_is_written_per_substitution(self):
        from lares.repair import records_from_study

        records = records_from_study(self._payload(with_repair=False), self.structure)
        self.assertEqual(len(records), 2)
        self.assertTrue(all(r.intervention for r in records))

    def test_no_intervention_record_can_be_ranked(self):
        from lares.repair import records_from_study

        records = records_from_study(self._payload(), self.structure)
        for record in records:
            if record.intervention:
                self.assertFalse(record.eligible_for_ranking)

    def test_every_intervention_record_says_what_was_substituted(self):
        from lares.repair import records_from_study

        for record in records_from_study(self._payload(), self.structure):
            if record.intervention:
                self.assertTrue(record.intervention_detail)

    def test_the_repair_itself_is_rankable_because_no_expert_helped_it(self):
        from lares.repair import records_from_study

        repair = [r for r in records_from_study(self._payload(), self.structure)
                  if r.candidate_id.endswith(":repaired")]
        self.assertEqual(len(repair), 1)
        self.assertTrue(repair[0].eligible_for_ranking)
        self.assertEqual(repair[0].final_disposition, "accepted")

    def test_a_rejected_repair_is_recorded_as_rejected(self):
        from lares.repair import records_from_study

        repair = [r for r in records_from_study(self._payload(accepted=False), self.structure)
                  if r.candidate_id.endswith(":repaired")][0]
        self.assertEqual(repair.final_disposition, "rejected")

    def test_rejected_explanations_are_recorded_alongside_the_supported_one(self):
        from lares.repair import records_from_study

        dispositions = {r.experiment_id.split(":")[-1]: r.final_disposition
                        for r in records_from_study(self._payload(), self.structure)
                        if r.intervention}
        self.assertEqual(dispositions["approach_control"], "supported")
        self.assertEqual(dispositions["vertical_axis"], "rejected")

    def test_the_repair_record_carries_what_the_aggregate_report_needs(self):
        from lares.repair import records_from_study

        payload = self._payload()
        payload["repair"]["return_after"] = 120.0
        repair = [r for r in records_from_study(payload, self.structure)
                  if r.candidate_id.endswith(":repaired")][0]
        self.assertEqual(repair.evaluation["mean_return"], 120.0)
        self.assertEqual(repair.code["num_parameters"], self.structure.num_parameters)

    def test_every_record_ties_back_to_exact_code(self):
        from lares.repair import records_from_study

        for record in records_from_study(self._payload(), self.structure):
            self.assertTrue(record.code.get("source_hash"))
            self.assertTrue(record.evaluation.get("manifest_ids"))
            self.assertTrue(record.evaluation.get("action_mode"))


class TestRunRepair(unittest.TestCase):
    """The whole step 4-5 cycle: propose, apply, validate."""

    @classmethod
    def setUpClass(cls):
        from lares.eval.manifest import synthetic_manifest

        cls.structure = next(
            s for s in load_bank(family="simple") if s.structure_id == "simple_standoff"
        )
        cls.manifest = synthetic_manifest(4, label="push-v2")
        cls.case_ids = [c.case_id for c in cls.manifest.episodes]
        cls.data = demo_data(128)

    def _payload(self, supported="approach_control"):
        cluster = find_clusters(report_of(
            [episode(c, "never_reached_object") for c in self.case_ids]))[0]
        explanations = competing_explanations(cluster)
        return {
            "studies": [{
                "supported": supported,
                "cluster": cluster.to_dict(),
                "explanations": [e.to_dict() for e in explanations],
            }]
        }

    def _run(self, payload, **kwargs):
        from lares.repair.study import run_repair

        policy = self.structure.build(SCHEMA)
        return run_repair(
            policy, self.structure, payload, self.data, SCHEMA, self.manifest,
            SeededMockEnv(), max_steps=5, search_episodes=16, population=2, **kwargs
        )

    def test_a_study_with_no_supported_explanation_produces_no_repair(self):
        record, repaired = self._run(self._payload(supported=None))
        self.assertFalse(record["attempted"])
        self.assertIsNone(repaired)

    def test_the_search_operator_produces_a_validated_repair(self):
        record, repaired = self._run(self._payload())
        self.assertTrue(record["attempted"])
        self.assertEqual(record["operator"], "search")
        self.assertIn("accepted", record["decision"])
        self.assertIsNotNone(repaired)

    def test_the_refit_operator_produces_a_validated_repair(self):
        record, _ = self._run(self._payload(), operator="refit", budget=20)
        self.assertEqual(record["operator"], "refit")
        self.assertIn("focused_check", record["decision"])

    def test_an_unknown_operator_is_refused(self):
        with self.assertRaises(ValueError):
            self._run(self._payload(), operator="wishful")

    def test_the_repair_search_is_not_counted_as_an_intervention(self):
        # Only expert-assisted or gate-forced rollouts are interventions. A
        # repair search runs the policy unaided and must not be excluded from
        # ranking along with them.
        from lares.repair.study import run_study

        policy = self.structure.build(SCHEMA)
        payload = run_study(
            policy, self.structure, self.manifest, SeededMockEnv(), "push-v2", SCHEMA,
            max_steps=5, max_clusters=1,
        )
        self.assertEqual(
            payload["intervention_episodes"],
            payload["rollout_episodes"] - len(self.manifest),
        )

    def test_the_repair_reports_every_episode_it_spent(self):
        record, _ = self._run(self._payload())
        # Search evaluations, the control run and the after run all cost cases.
        self.assertGreaterEqual(record["rollout_episodes"], 2 * len(self.manifest))

    def test_held_out_cases_are_scored_but_take_no_part_in_the_decision(self):
        record, _ = self._run(self._payload(), held_out=self.manifest)
        self.assertIn("held_out", record)
        self.assertIn("took no part", record["held_out"]["note"])

    def test_the_edit_stays_inside_the_component_it_named(self):
        record, _ = self._run(self._payload())
        repair = record["repair"]
        self.assertTrue(set(repair["changed"]) <= set(repair["parameters"]))


# ---------------------------------------------------------------------------
#  Intervention actors against the simulator
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAS_METAWORLD, "metaworld is required for intervention rollouts")
class TestInterventionActor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("MUJOCO_GL", "egl")
        from lares.core.training_pipeline import ensure_mujoco_headless_gl
        from lares.eval import EvaluationManifest, make_manifest_env

        ensure_mujoco_headless_gl()
        path = os.path.join(_PROJECT_ROOT, "config", "manifests", "push-v2_development.yaml")
        if not os.path.isfile(path):
            raise unittest.SkipTest("development manifest missing")
        cls.manifest = EvaluationManifest.load(path).head(2)
        cls.env = make_manifest_env(
            EvaluationManifest.load(path), EvaluationManifest.load(path).resolve_pool(), 200
        )
        cls.structure = next(
            s for s in load_bank(family="simple") if s.structure_id == "simple_standoff"
        )
        cls.policy = cls.structure.build(SCHEMA)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def _run(self, substitution, steps=30):
        from lares.eval.runner import evaluate_manifest

        actor = InterventionActor(self.policy, substitution, "push-v2", SCHEMA)
        return evaluate_manifest(actor, self.manifest, self.env, max_steps=steps)

    def test_the_unmodified_control_matches_the_plain_policy(self):
        from lares.eval.runner import PolicyActor, evaluate_manifest

        control = self._run(Substitution(kind=SUBSTITUTION_NONE))
        plain = evaluate_manifest(
            PolicyActor(self.policy, True, name="plain"), self.manifest, self.env,
            max_steps=30,
        )
        self.assertEqual(
            [e.episode_return for e in control.episodes],
            [e.episode_return for e in plain.episodes],
        )

    def test_an_expert_prefix_changes_the_trajectory(self):
        control = self._run(Substitution(kind=SUBSTITUTION_NONE))
        prefixed = self._run(
            Substitution(kind=SUBSTITUTION_EXPERT_PREFIX, prefix_steps=20)
        )
        self.assertNotEqual(
            [e.episode_return for e in control.episodes],
            [e.episode_return for e in prefixed.episodes],
        )

    def test_replacing_one_axis_changes_less_than_replacing_all(self):
        control = self._run(Substitution(kind=SUBSTITUTION_NONE))
        one_axis = self._run(Substitution(kind=SUBSTITUTION_EXPERT_AXIS, axes=(2,)))
        every_axis = self._run(
            Substitution(kind=SUBSTITUTION_EXPERT_AXIS, axes=(0, 1, 2, 3))
        )
        base = np.array([e.episode_return for e in control.episodes])
        self.assertLessEqual(
            np.abs(np.array([e.episode_return for e in one_axis.episodes]) - base).sum(),
            np.abs(np.array([e.episode_return for e in every_axis.episodes]) - base).sum(),
        )

    def test_the_actor_clears_its_override_between_episodes(self):
        self._run(Substitution(kind=SUBSTITUTION_NONE))
        self.assertEqual(self.policy.gate_overrides, {})

    def test_an_intervention_run_is_paired_with_the_control_on_the_same_cases(self):
        control = self._run(Substitution(kind=SUBSTITUTION_NONE))
        intervened = self._run(
            Substitution(kind=SUBSTITUTION_EXPERT_PREFIX, prefix_steps=20)
        )
        self.assertEqual(
            [e.case_id for e in control.episodes],
            [e.case_id for e in intervened.episodes],
        )


# ---------------------------------------------------------------------------
#  Scopes and measured parameter selection
# ---------------------------------------------------------------------------


class _Space:
    def __init__(self, dim):
        self.shape = (dim,)
        self.high = np.ones(dim, dtype=np.float32)
        self.low = -np.ones(dim, dtype=np.float32)


class SeededMockEnv:
    """Episode is a function of the case seed and of the actions taken."""

    def __init__(self, obs_dim=OBS_DIM, action_dim=ACT_DIM, episode_length=5):
        self.observation_space = _Space(obs_dim)
        self.action_space = _Space(action_dim)
        self.obs_dim = obs_dim
        self._episode_length = episode_length
        self._rng = None
        self._step = 0

    def reset(self, case=None):
        if case is None:
            raise ValueError("case required")
        self._rng = np.random.default_rng(int(case.reset_seed))
        self._step = 0
        return self._rng.standard_normal(self.obs_dim).astype(np.float32), {}

    def step(self, action):
        self._step += 1
        obs = self._rng.standard_normal(self.obs_dim).astype(np.float32)
        reward = float(self._rng.standard_normal() + np.sum(action))
        done = self._step >= self._episode_length
        return obs, reward, done, {"success": 1.0 if done and reward > 0 else 0.0,
                                   "obj_to_target": abs(reward)}


class _Data:
    def __init__(self, obs, actions):
        self.train_obs = obs
        self.train_actions = actions


def demo_data(n=256, seed=0):
    generator = torch.Generator().manual_seed(seed)
    obs = 0.1 * torch.randn(n, OBS_DIM, generator=generator)
    actions = torch.tanh(torch.randn(n, ACT_DIM, generator=generator))
    return _Data(obs, actions)


class TestScopes(unittest.TestCase):
    def test_every_declared_component_has_axes_and_phases(self):
        for component, scope in COMPONENT_SCOPES.items():
            with self.subTest(component=component):
                self.assertIn("axes", scope)
                self.assertIn("phases", scope)

    def test_every_explanation_names_a_component_a_repair_can_act_on(self):
        # An explanation whose component has no scope can be supported and then
        # produce no repair, which would make the diagnosis unusable.
        report = report_of([episode(f"c{i}", "never_reached_object") for i in range(4)]
                           + [episode(f"d{i}", "pushed_away_from_goal") for i in range(4)]
                           + [episode(f"e{i}", "object_not_moved") for i in range(4)])
        for cluster in find_clusters(report):
            for explanation in competing_explanations(cluster):
                with self.subTest(explanation=explanation.name):
                    self.assertIsNotNone(scope_for(explanation.suspected_component))

    def test_a_gate_component_resolves_to_the_gate_name(self):
        scope = scope_for("the contact gate threshold")
        self.assertEqual(scope["gate"], "contact")

    def test_an_unknown_component_has_no_scope(self):
        self.assertIsNone(scope_for("vibes"))


class TestParameterSelection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = {s.structure_id: s for s in load_bank(family="simple")}
        cls.obs = 0.1 * torch.randn(64, OBS_DIM, generator=torch.Generator().manual_seed(1))

    def test_sensitivity_is_restricted_to_the_scoped_axes(self):
        policy = self.bank["simple_standoff"].build(SCHEMA)
        vertical = scoped_sensitivity(policy, self.obs, axes=(2,))
        gripper = scoped_sensitivity(policy, self.obs, axes=(3,))
        # grip_bias drives only the gripper; height_offset only the vertical axis.
        self.assertGreater(gripper["grip_bias"], vertical["grip_bias"])
        self.assertGreater(vertical["height_offset"], gripper["height_offset"])

    def test_selection_keeps_the_parameters_that_move_the_scoped_axes(self):
        policy = self.bank["simple_standoff"].build(SCHEMA)
        chosen = select_parameters(policy, self.obs, axes=(0, 1, 2))
        self.assertIn("gain", chosen)
        self.assertNotIn("grip_bias", chosen)

    def test_selection_ignores_parameters_with_no_declared_range(self):
        policy = self.bank["simple_standoff"].build(SCHEMA)
        ranges = set(policy.get_param_ranges())
        self.assertTrue(set(select_parameters(policy, self.obs)) <= ranges)

    def test_gate_parameters_are_found_by_moving_the_gate(self):
        policy = self.bank["simple_reach_push"].build(SCHEMA)
        found = gate_parameters(policy, "contact", self.obs)
        self.assertTrue(found)
        self.assertTrue(set(found) <= set(policy.get_param_ranges()))

    def test_an_unrecorded_gate_yields_no_parameters(self):
        policy = self.bank["simple_reach_push"].build(SCHEMA)
        self.assertEqual(gate_parameters(policy, "imaginary", self.obs), [])


class TestPhaseSubset(unittest.TestCase):
    def test_no_phase_restriction_keeps_every_row(self):
        data = demo_data(32)
        rows, idx = phase_subset(data.train_obs, (), SCHEMA)
        self.assertEqual(rows.shape[0], 32)
        self.assertEqual(len(idx), 32)

    def test_a_restriction_keeps_a_subset_in_order(self):
        data = demo_data(64)
        rows, idx = phase_subset(data.train_obs, ("push",), SCHEMA)
        self.assertLessEqual(rows.shape[0], 64)
        self.assertEqual(list(idx), sorted(idx))

    def test_an_empty_restriction_falls_back_rather_than_fitting_on_nothing(self):
        data = demo_data(8)
        rows, _ = phase_subset(data.train_obs, ("hover", "descend", "push"), SCHEMA)
        self.assertEqual(rows.shape[0], 8)


# ---------------------------------------------------------------------------
#  Bounded edits
# ---------------------------------------------------------------------------


class TestBoundedRefit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.structure = next(
            s for s in load_bank(family="simple") if s.structure_id == "simple_standoff"
        )
        cls.data = demo_data(256)

    def _repair(self, parameters=("gain",), axes=(0, 1, 2)):
        return BoundedRepair(
            explanation="approach_control", suspected_component="the approach term",
            parameters=list(parameters), axes=tuple(axes), phases=(),
            focused_metric="min_tcp_object_distance", predicted_direction="decrease",
        )

    def test_only_the_declared_parameters_move(self):
        policy = self.structure.build(SCHEMA)
        repair = self._repair()
        apply_repair(policy, repair, self.data, SCHEMA, budget=30)
        self.assertTrue(repair_stayed_in_scope(repair))
        self.assertEqual(set(repair.changed), {"gain"})

    def test_the_original_policy_is_left_alone(self):
        policy = self.structure.build(SCHEMA)
        before = {n: p.detach().clone() for n, p in policy.named_parameters()}
        apply_repair(policy, self._repair(), self.data, SCHEMA, budget=30)
        self.assertEqual(bounded_change(before, policy), {})

    def test_an_empty_repair_costs_nothing_and_changes_nothing(self):
        policy = self.structure.build(SCHEMA)
        repair = self._repair(parameters=())
        repaired = apply_repair(policy, repair, self.data, SCHEMA, budget=30)
        self.assertEqual(repair.cost["objective_evaluations"], 0)
        self.assertEqual(bounded_change(
            {n: p.detach().clone() for n, p in policy.named_parameters()}, repaired), {})

    def test_the_refit_reports_what_it_cost(self):
        policy = self.structure.build(SCHEMA)
        repair = self._repair()
        apply_repair(policy, repair, self.data, SCHEMA, budget=25, batch_size=32)
        self.assertEqual(repair.cost["objective_evaluations"], 25)
        self.assertEqual(repair.cost["transition_evaluations"], 25 * 32)

    def test_a_repaired_parameter_stays_inside_its_declared_range(self):
        policy = self.structure.build(SCHEMA)
        repaired = apply_repair(policy, self._repair(), self.data, SCHEMA, budget=50)
        ranges = repaired.get_param_ranges()
        for name, param in repaired.named_parameters():
            lo, hi = ranges[name]
            self.assertGreaterEqual(float(param.detach().min()), lo - 1e-6)
            self.assertLessEqual(float(param.detach().max()), hi + 1e-6)

    def test_propose_selects_the_scope_of_the_supported_explanation(self):
        policy = self.structure.build(SCHEMA)
        cluster = find_clusters(report_of(
            [episode(f"c{i}", "never_reached_object") for i in range(4)]))[0]
        explanation = competing_explanations(cluster)[0]
        repair = propose_repair(policy, explanation, self.data, SCHEMA)
        self.assertEqual(repair.axes, (0, 1, 2))
        self.assertEqual(repair.phases, ("hover", "descend"))
        self.assertFalse(repair.is_empty)

    def test_propose_refuses_a_component_it_cannot_localise(self):
        policy = self.structure.build(SCHEMA)
        explanation = Explanation(
            name="x", statement="", substitution=Substitution(kind=SUBSTITUTION_NONE),
            focused_metric="signed_goal_progress", predicted_direction="increase",
            suspected_component="something unnamed",
        )
        repair = propose_repair(policy, explanation, self.data, SCHEMA)
        self.assertTrue(repair.is_empty)
        self.assertIn("no scope", repair.note)


class TestScopedSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lares.eval.manifest import synthetic_manifest

        cls.structure = next(
            s for s in load_bank(family="simple") if s.structure_id == "simple_standoff"
        )
        cls.manifest = synthetic_manifest(4, label="push-v2")

    def _repair(self, parameters=("gain", "standoff")):
        return BoundedRepair(
            explanation="approach_control", suspected_component="the approach term",
            parameters=list(parameters), axes=(0, 1, 2), phases=(),
            focused_metric="signed_goal_progress", predicted_direction="increase",
        )

    def test_the_search_spends_its_whole_episode_budget(self):
        policy = self.structure.build(SCHEMA)
        repair = self._repair()
        search_repair(policy, repair, self.manifest, SeededMockEnv(),
                      budget_episodes=40, population=3, max_steps=5)
        self.assertEqual(repair.cost["rollout_episodes"], 40)
        self.assertEqual(repair.cost["evaluations"], 10)

    def test_a_budget_that_does_not_divide_the_population_is_still_spent(self):
        policy = self.structure.build(SCHEMA)
        repair = self._repair()
        search_repair(policy, repair, self.manifest, SeededMockEnv(),
                      budget_episodes=36, population=4, max_steps=5)
        self.assertEqual(repair.cost["evaluations"], 9)

    def test_the_search_moves_only_the_parameters_in_scope(self):
        policy = self.structure.build(SCHEMA)
        repair = self._repair()
        search_repair(policy, repair, self.manifest, SeededMockEnv(),
                      budget_episodes=40, population=3, max_steps=5)
        self.assertTrue(repair_stayed_in_scope(repair))

    def test_the_search_leaves_the_original_policy_alone(self):
        policy = self.structure.build(SCHEMA)
        before = {n: p.detach().clone() for n, p in policy.named_parameters()}
        search_repair(policy, self._repair(), self.manifest, SeededMockEnv(),
                      budget_episodes=40, population=3, max_steps=5)
        self.assertEqual(bounded_change(before, policy), {})

    def test_the_search_records_its_kind(self):
        policy = self.structure.build(SCHEMA)
        repair = self._repair()
        search_repair(policy, repair, self.manifest, SeededMockEnv(),
                      budget_episodes=20, population=2, max_steps=5)
        self.assertEqual(repair.kind, REPAIR_SEARCH)

    def test_an_empty_scope_spends_nothing(self):
        policy = self.structure.build(SCHEMA)
        repair = self._repair(parameters=())
        search_repair(policy, repair, self.manifest, SeededMockEnv(),
                      budget_episodes=40, population=3, max_steps=5)
        self.assertEqual(repair.cost["rollout_episodes"], 0)

    def test_a_vector_parameter_is_dropped_rather_than_flattened(self):
        # Several bank structures declare a four-element log_std. Moving it
        # along one coordinate would collapse it to a single value.
        vector_structure = next(
            s for s in load_bank(env_id="push-v2")
            if any(p.numel() > 1 for _, p in s.build(SCHEMA).named_parameters())
        )
        policy = vector_structure.build(SCHEMA)
        vector = [n for n, p in policy.named_parameters() if p.numel() > 1]
        repair = BoundedRepair(
            explanation="global", suspected_component="every parameter",
            parameters=sorted(policy.get_param_ranges()), axes=(0, 1, 2, 3), phases=(),
            focused_metric="signed_goal_progress", predicted_direction="increase",
        )
        before = {n: p.detach().clone() for n, p in policy.named_parameters()}
        repaired, _ = search_repair(policy, repair, self.manifest, SeededMockEnv(),
                                    budget_episodes=20, population=2, max_steps=5)
        self.assertFalse(set(vector) & set(repair.parameters))
        self.assertIn("vector parameter", repair.note)
        for name in vector:
            self.assertTrue(torch.equal(
                before[name], dict(repaired.named_parameters())[name].detach()))

    def test_scalar_parameters_keeps_only_the_single_valued_ones(self):
        policy = self.structure.build(SCHEMA)
        names = sorted(policy.get_param_ranges())
        self.assertEqual(scalar_parameters(policy, names), names)

    def test_the_search_never_leaves_a_declared_range(self):
        policy = self.structure.build(SCHEMA)
        repaired, _ = search_repair(
            policy, self._repair(), self.manifest, SeededMockEnv(),
            budget_episodes=40, population=3, sigma=2.0, max_steps=5,
        )
        ranges = repaired.get_param_ranges()
        for name, param in repaired.named_parameters():
            lo, hi = ranges[name]
            self.assertGreaterEqual(float(param.detach().min()), lo - 1e-6)
            self.assertLessEqual(float(param.detach().max()), hi + 1e-6)


# ---------------------------------------------------------------------------
#  Acceptance with the vacuity guard
# ---------------------------------------------------------------------------


class TestAcceptRepair(unittest.TestCase):
    def _repair(self):
        return BoundedRepair(
            explanation="approach_control", suspected_component="the approach term",
            parameters=["gain"], axes=(0, 1, 2), phases=(),
            focused_metric="min_tcp_object_distance", predicted_direction="decrease",
        )

    def _results(self, focused_after, other_progress_after, other_success=(1.0, 1.0)):
        before = result_of(
            [episode(f"f{i}", "never_reached_object", min_tcp_object_distance=0.09,
                     signed_goal_progress=0.02) for i in range(3)]
            + [episode(f"o{i}", "success", success=s, signed_goal_progress=0.05)
               for i, s in enumerate(other_success)]
        )
        after = result_of(
            [episode(f"f{i}", "never_reached_object", min_tcp_object_distance=focused_after,
                     signed_goal_progress=0.02) for i in range(3)]
            + [episode(f"o{i}", "success", success=s,
                       signed_goal_progress=other_progress_after)
               for i, s in enumerate(other_success)]
        )
        return before, after

    def test_the_protected_set_is_the_cases_that_already_succeeded(self):
        before, _ = self._results(0.03, 0.05)
        ids, basis = protected_cases(before, ["f0", "f1", "f2"])
        self.assertEqual(basis, "success")
        self.assertEqual(sorted(ids), ["o0", "o1"])

    def test_with_no_successes_the_protected_set_is_everything_outside_the_cluster(self):
        before, _ = self._results(0.03, 0.05, other_success=(0.0, 0.0))
        ids, basis = protected_cases(before, ["f0", "f1", "f2"])
        self.assertEqual(sorted(ids), ["o0", "o1"])
        self.assertIn("cannot fail", basis)

    def test_a_vacuous_success_check_falls_back_to_a_continuous_one(self):
        # No case succeeded, so no repair can fail the success threshold. The
        # progress check is what stops a repair from wrecking the other cases.
        before, after = self._results(0.01, -0.20, other_success=(0.0, 0.0))
        decision = accept_repair(before, after, self._repair(), ["f0", "f1", "f2"])
        self.assertFalse(decision.accepted)
        self.assertIn("signed_goal_progress", decision.reason)
        self.assertFalse(decision.protected_check["secondary"]["passed"])

    def test_a_repair_that_holds_progress_is_accepted(self):
        before, after = self._results(0.01, 0.05, other_success=(0.0, 0.0))
        decision = accept_repair(before, after, self._repair(), ["f0", "f1", "f2"])
        self.assertTrue(decision.accepted)

    def test_an_edit_that_escaped_its_scope_is_refused(self):
        before, after = self._results(0.01, 0.05)
        repair = self._repair()
        repair.changed = {"gain": 0.2, "grip_bias": 0.1}
        decision = accept_repair(before, after, repair, ["f0", "f1", "f2"])
        self.assertFalse(decision.accepted)
        self.assertIn("outside its scope", decision.reason)

    def test_the_basis_is_recorded_on_the_decision(self):
        before, after = self._results(0.01, 0.05)
        decision = accept_repair(before, after, self._repair(), ["f0", "f1", "f2"])
        self.assertEqual(decision.protected_check["basis"], "success")

    def test_a_cluster_covering_every_case_leaves_nothing_to_protect(self):
        # The protected set must not silently become the cluster itself.
        before = result_of([episode(f"f{i}", "never_reached_object",
                                    min_tcp_object_distance=0.09) for i in range(3)])
        after = result_of([episode(f"f{i}", "never_reached_object",
                                   min_tcp_object_distance=0.02) for i in range(3)])
        decision = accept_repair(before, after, self._repair(), ["f0", "f1", "f2"])
        self.assertEqual(decision.protected_check["n_cases"], 0)
        self.assertIn("no protected set", decision.protected_check["basis"])
        self.assertTrue(decision.accepted)

    def test_the_progress_tolerance_is_declared_in_the_module(self):
        self.assertGreater(PROTECTED_PROGRESS_TOLERANCE, 0.0)

    def test_progress_regression_reports_the_drop(self):
        before, after = self._results(0.01, 0.01, other_success=(0.0, 0.0))
        check = progress_regression(before, after, ["o0", "o1"])
        self.assertAlmostEqual(check["drop"], 0.04, places=6)
        self.assertFalse(check["passed"])


if __name__ == "__main__":
    unittest.main()
