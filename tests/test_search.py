#!/usr/bin/env python
"""Tests for the search schemas, archive and feedback (FR-5, FR-10, AC-4).

Run from the project root::

    python -m unittest tests.test_search
"""

import os
import sys
import tempfile
import unittest

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from lares.core.obs_schema import get_obs_schema  # noqa: E402
from lares.eval.report import CHECKPOINT_FITTED, build_report  # noqa: E402
from lares.eval.runner import EpisodeRecord, ManifestResult  # noqa: E402
from lares.search import (  # noqa: E402
    MODE_CROSSOVER,
    MODE_EXPLORATION,
    MODE_PARAMETER_ONLY,
    MODE_REPAIR,
    SLOT_DISTINCT,
    SLOT_ROBUST,
    SLOT_SIMPLEST,
    STATUS_EVALUATED,
    STATUS_GENERATED,
    STATUS_INVALID,
    Archive,
    CandidateEvidence,
    ExperimentRecord,
    IdeaParameter,
    IdeaPhase,
    MutationProposal,
    PolicyIdea,
    SchemaError,
    behavior_descriptor,
    behavioral_distance,
    build_feedback,
    choose_informative_parents,
    load_records,
    measurable_metrics,
    source_hash,
)

SCHEMA = get_obs_schema("push-v2")


def make_report(candidate="c", success=0.0, labels=None, progress=0.0, params=6):
    records = []
    labels = labels or {"never_reached_object": 5}
    i = 0
    for name, count in labels.items():
        for _ in range(count):
            records.append(
                EpisodeRecord(
                    case_id=f"case-{i}",
                    task_id="t",
                    success=1.0 if name == "success" else 0.0,
                    episode_return=10.0,
                    length=150,
                    terminal_obj_to_target=0.2,
                    initial_obj_to_target=0.3,
                    near_object_rate=0.1,
                    first_near_object_step=None,
                    grasp_success_rate=0.0,
                    action_saturation_rate=0.0,
                    timed_out=True,
                    failure_label=name,
                    signed_goal_progress=progress,
                    final_goal_distance=0.2,
                    lateral_drift=0.01,
                    min_tcp_object_distance=0.08,
                    object_displacement=0.05,
                    action_saturation_by_axis=[0.0, 0.0, 0.0, 0.0],
                    action_variation_by_axis=[0.01, 0.01, 0.01, 0.0],
                )
            )
            i += 1
    if success > 0:
        for r in records[: max(1, int(round(success * len(records))))]:
            r.success = 1.0
    result = ManifestResult(
        actor_name=candidate,
        manifest_id="m",
        action_mode="deterministic",
        episodes=records,
        schema_id="push-v2",
    )
    return build_report(result, candidate, CHECKPOINT_FITTED)


def good_idea(**overrides):
    base = dict(
        idea_id="idea-0",
        name="staged push with standoff",
        summary="Approach a standoff point behind the puck, then drive it at the goal.",
        named_observations=["tcp", "obj", "goal"],
        phases=[
            IdeaPhase("approach", "distance to object above the contact radius", "move toward the standoff point"),
            IdeaPhase("push", "distance to object below the contact radius", "drive along the object-to-goal direction"),
        ],
        parameters=[
            IdeaParameter("w_reach", "approach speed", "gain", "m/step", 3.0, (0.1, 10.0)),
            IdeaParameter("contact_radius", "contact threshold", "structural", "m", 0.06, (0.01, 0.2)),
        ],
        zero_shot_behavior="the arm should already move to the puck and push it roughly at the goal",
        expected_failure="may overshoot; the brake is untuned",
        protected_behaviors=["reaches the object within 40 steps"],
    )
    base.update(overrides)
    return PolicyIdea(**base)


# ---------------------------------------------------------------------------
#  PolicyIdea
# ---------------------------------------------------------------------------


class TestPolicyIdea(unittest.TestCase):
    def test_a_complete_idea_validates(self):
        self.assertTrue(good_idea().validate(SCHEMA))

    def test_an_idea_must_say_what_the_untrained_policy_does(self):
        with self.assertRaises(SchemaError) as ctx:
            good_idea(zero_shot_behavior="").validate()
        self.assertIn("untrained", str(ctx.exception))

    def test_an_idea_with_no_phases_is_refused(self):
        with self.assertRaises(SchemaError):
            good_idea(phases=[]).validate()

    def test_an_idea_with_no_parameters_is_refused(self):
        with self.assertRaises(SchemaError):
            good_idea(parameters=[]).validate()

    def test_a_parameter_initialising_outside_its_range_is_refused(self):
        bad = IdeaParameter("g", "gain", "gain", "m", 99.0, (0.1, 10.0))
        with self.assertRaises(SchemaError) as ctx:
            good_idea(parameters=[bad]).validate()
        self.assertIn("outside its own range", str(ctx.exception))

    def test_an_unknown_parameter_type_is_refused(self):
        bad = IdeaParameter("g", "gain", "vibe", "m", 1.0, (0.1, 10.0))
        with self.assertRaises(SchemaError):
            good_idea(parameters=[bad]).validate()

    def test_duplicate_parameter_names_are_refused(self):
        p = IdeaParameter("g", "gain", "gain", "m", 1.0, (0.1, 10.0))
        with self.assertRaises(SchemaError):
            good_idea(parameters=[p, p]).validate()

    def test_an_observation_field_that_does_not_exist_is_refused(self):
        with self.assertRaises(SchemaError) as ctx:
            good_idea(named_observations=["tcp", "puck_velocity"]).validate(SCHEMA)
        self.assertIn("puck_velocity", str(ctx.exception))

    def test_a_phase_missing_its_activation_is_refused(self):
        with self.assertRaises(SchemaError):
            good_idea(phases=[IdeaPhase("push", "", "drive at the goal")]).validate()

    def test_complexity_is_measured_not_assumed(self):
        c = good_idea().complexity
        self.assertEqual(c["num_parameters"], 2)
        self.assertEqual(c["num_phases"], 2)
        self.assertEqual(c["num_structural"], 1)
        self.assertEqual(c["num_gains"], 1)

    def test_declared_parameters_are_checked_against_the_generated_code(self):
        import torch
        import torch.nn as nn

        from lares.core.obs_schema import active_schema
        from lares.core.symbolic_policy import SymbolicPolicy

        class P(SymbolicPolicy):
            def __init__(self, o, a):
                super().__init__(o, a)
                self.w_reach = nn.Parameter(torch.tensor(3.0))
                self.undeclared = nn.Parameter(torch.tensor(1.0))

            def forward(self, obs):
                m = self.w_reach * self.obs_field(obs, "tcp")
                return m, torch.ones_like(m)

            def get_param_ranges(self):
                return {}

        with active_schema(SCHEMA):
            policy = P(39, 4)
        problems = good_idea().check_against_policy(policy)
        self.assertTrue(any("undeclared" in p for p in problems))
        self.assertTrue(any("contact_radius" in p for p in problems))

    def test_round_trips_through_a_dict(self):
        idea = good_idea()
        self.assertEqual(PolicyIdea.from_dict(idea.to_dict()).to_dict(), idea.to_dict())

    def test_implementation_brief_names_every_parameter(self):
        brief = good_idea().implementation_brief()
        self.assertIn("w_reach", brief)
        self.assertIn("contact_radius", brief)
        self.assertIn("Untrained behaviour", brief)


# ---------------------------------------------------------------------------
#  MutationProposal
# ---------------------------------------------------------------------------


def good_proposal(**overrides):
    base = dict(
        proposal_id="p-0",
        parent_ids=["gen0_cand0"],
        mode=MODE_REPAIR,
        observation="17 of 30 episodes labelled never_reached_object",
        hypothesis="the approach gain is too small to close the initial distance",
        suspected_component="the approach term",
        intervention="raise the approach gain range and its initial value",
        predicted_metric="min_tcp_object_distance",
        predicted_direction="decrease",
        protected_behaviors=["does not start pushing before contact"],
        focused_cases=["development-0000"],
    )
    base.update(overrides)
    return MutationProposal(**base)


class TestMutationProposal(unittest.TestCase):
    def test_a_complete_repair_proposal_validates(self):
        self.assertTrue(good_proposal().validate())

    def test_a_prediction_naming_an_unmeasured_metric_is_refused(self):
        with self.assertRaises(SchemaError) as ctx:
            good_proposal(predicted_metric="elegance").validate()
        self.assertIn("could never be wrong", str(ctx.exception))

    def test_a_proposal_without_a_predicted_direction_is_refused(self):
        with self.assertRaises(SchemaError):
            good_proposal(predicted_direction="").validate()

    def test_a_proposal_without_protected_behaviours_is_refused(self):
        with self.assertRaises(SchemaError) as ctx:
            good_proposal(protected_behaviors=[]).validate()
        self.assertIn("regression is detectable", str(ctx.exception))

    def test_a_repair_edits_exactly_one_parent(self):
        with self.assertRaises(SchemaError):
            good_proposal(parent_ids=["a", "b"]).validate()

    def test_a_crossover_needs_two_parents(self):
        with self.assertRaises(SchemaError):
            good_proposal(mode=MODE_CROSSOVER, parent_ids=["a"]).validate()

    def test_exploration_is_allowed_to_be_broad(self):
        proposal = MutationProposal(
            proposal_id="p-1",
            parent_ids=[],
            mode=MODE_EXPLORATION,
            observation="the whole population stalls before contact",
            hypothesis="",
            suspected_component="",
            intervention="",
        )
        self.assertTrue(proposal.validate())

    def test_an_unknown_mode_is_refused(self):
        with self.assertRaises(SchemaError):
            good_proposal(mode="vibes").validate()

    def test_parameter_only_search_bypasses_the_generator(self):
        proposal = good_proposal(mode=MODE_PARAMETER_ONLY)
        self.assertTrue(proposal.validate())
        self.assertTrue(proposal.bypasses_llm)
        self.assertFalse(good_proposal().bypasses_llm)

    def test_the_outcome_check_can_report_the_prediction_was_wrong(self):
        proposal = good_proposal()
        wrong = proposal.check_outcome(before=0.05, after=0.09)
        self.assertTrue(wrong["measurable"])
        self.assertFalse(wrong["as_predicted"])
        right = proposal.check_outcome(before=0.09, after=0.05)
        self.assertTrue(right["as_predicted"])

    def test_an_unmeasurable_outcome_says_so(self):
        self.assertFalse(good_proposal().check_outcome(None, 0.1)["measurable"])

    def test_measurable_metrics_come_from_the_report_itself(self):
        metrics = measurable_metrics()
        self.assertIn("success_rate", metrics)
        self.assertIn("signed_goal_progress", metrics)
        self.assertIn("fitting.bound_activity", metrics)


# ---------------------------------------------------------------------------
#  ExperimentRecord
# ---------------------------------------------------------------------------


class TestExperimentRecord(unittest.TestCase):
    def _record(self, **overrides):
        base = dict(
            experiment_id="gen0_cand0",
            candidate_id="gen0_cand0",
            generation=0,
            status=STATUS_EVALUATED,
            code={"source_hash": source_hash("class GeneratedPolicy: pass")},
            evaluation={"manifest_ids": ["m"], "action_mode": "deterministic"},
        )
        base.update(overrides)
        return ExperimentRecord(**base)

    def test_a_complete_record_validates(self):
        self.assertTrue(self._record().validate())

    def test_an_evaluated_record_must_name_its_manifests(self):
        with self.assertRaises(SchemaError) as ctx:
            self._record(evaluation={"action_mode": "deterministic"}).validate()
        self.assertIn("manifests", str(ctx.exception))

    def test_an_evaluated_record_must_name_its_action_mode(self):
        with self.assertRaises(SchemaError):
            self._record(evaluation={"manifest_ids": ["m"]}).validate()

    def test_a_record_past_generation_must_carry_a_code_hash(self):
        with self.assertRaises(SchemaError) as ctx:
            self._record(code={}).validate()
        self.assertIn("source hash", str(ctx.exception))

    def test_a_freshly_generated_record_needs_no_hash_yet(self):
        self.assertTrue(
            self._record(status=STATUS_GENERATED, code={}, evaluation={}).validate()
        )

    def test_an_invalid_candidate_still_gets_a_record(self):
        record = self._record(
            status=STATUS_INVALID,
            evaluation={},
            validation={"passed": False, "errors": ["shape mismatch"]},
        )
        self.assertTrue(record.validate())

    def test_an_intervention_must_say_what_was_substituted(self):
        with self.assertRaises(SchemaError):
            self._record(intervention=True).validate()

    def test_intervention_results_are_never_eligible_for_ranking(self):
        record = self._record(intervention=True, intervention_detail="expert drove the z axis")
        record.validate()
        self.assertFalse(record.eligible_for_ranking)
        self.assertTrue(self._record().eligible_for_ranking)

    def test_an_unknown_status_is_refused(self):
        with self.assertRaises(SchemaError):
            self._record(status="looked_promising").validate()

    def test_records_round_trip_through_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._record().save(tmp)
            self._record(experiment_id="gen0_cand1", candidate_id="gen0_cand1").save(tmp)
            loaded = load_records(tmp)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].candidate_id, "gen0_cand0")
        self.assertEqual(loaded[0].evaluation["action_mode"], "deterministic")

    def test_loading_a_missing_directory_returns_nothing(self):
        self.assertEqual(load_records("/nonexistent/path/for/records"), [])


# ---------------------------------------------------------------------------
#  Archive
# ---------------------------------------------------------------------------


class TestArchive(unittest.TestCase):
    def test_behaviour_distance_separates_different_failure_profiles(self):
        never = behavior_descriptor(make_report(labels={"never_reached_object": 10}))
        away = behavior_descriptor(make_report(labels={"pushed_away_from_goal": 10}))
        same = behavior_descriptor(make_report(labels={"never_reached_object": 10}))
        self.assertGreater(behavioral_distance(never, away), 0.5)
        self.assertAlmostEqual(behavioral_distance(never, same), 0.0, places=9)

    def test_the_best_candidate_takes_the_robust_slot(self):
        archive = Archive()
        archive.add("a", 0, make_report("a", labels={"never_reached_object": 10}), 8, "h1")
        archive.add("b", 0, make_report("b", labels={"success": 5, "never_reached_object": 5}), 20, "h2")
        self.assertEqual(archive.slot(SLOT_ROBUST).candidate_id, "b")

    def test_the_simplest_competitive_candidate_takes_the_simple_slot(self):
        archive = Archive(competitive_margin=0.5)
        archive.add("big", 0, make_report("big", labels={"success": 6, "never_reached_object": 4}), 30, "h1")
        archive.add("small", 0, make_report("small", labels={"success": 5, "never_reached_object": 5}), 6, "h2")
        self.assertEqual(archive.slot(SLOT_ROBUST).candidate_id, "big")
        self.assertEqual(archive.slot(SLOT_SIMPLEST).candidate_id, "small")

    def test_a_simpler_but_uncompetitive_candidate_does_not_take_the_slot(self):
        archive = Archive(competitive_margin=0.01)
        archive.add("big", 0, make_report("big", labels={"success": 10}), 30, "h1")
        archive.add("small", 0, make_report("small", labels={"never_reached_object": 10}), 4, "h2")
        self.assertEqual(archive.slot(SLOT_SIMPLEST).candidate_id, "big")

    def test_the_distinct_slot_goes_to_a_different_failure_profile(self):
        archive = Archive()
        archive.add("a", 0, make_report("a", labels={"success": 5, "never_reached_object": 5}), 8, "h1")
        archive.add("b", 0, make_report("b", labels={"never_reached_object": 10}), 8, "h2")
        archive.add("c", 0, make_report("c", labels={"pushed_away_from_goal": 10}), 8, "h3")
        self.assertEqual(archive.slot(SLOT_ROBUST).candidate_id, "a")
        self.assertEqual(archive.slot(SLOT_DISTINCT).candidate_id, "c")

    def test_diversity_is_behavioural_not_textual(self):
        # Two candidates with identical behaviour cannot both be diverse, whatever
        # their source looks like.
        archive = Archive()
        archive.add("a", 0, make_report("a", labels={"never_reached_object": 10}), 8, "hash-one")
        archive.add("b", 0, make_report("b", labels={"never_reached_object": 10}), 8, "hash-two")
        self.assertIsNone(archive.slot(SLOT_DISTINCT))

    def test_intervention_results_are_recorded_but_never_occupy_a_slot(self):
        archive = Archive()
        archive.add("normal", 0, make_report("normal", labels={"never_reached_object": 10}), 8, "h1")
        archive.add(
            "assisted", 0, make_report("assisted", labels={"success": 10}), 8, "h2",
            intervention=True,
        )
        self.assertEqual(len(archive), 2)
        self.assertEqual(archive.slot(SLOT_ROBUST).candidate_id, "normal")
        self.assertIn("intervention result", archive.summary())

    def test_an_empty_archive_reports_unoccupied_slots(self):
        archive = Archive()
        self.assertIsNone(archive.slot(SLOT_ROBUST))
        self.assertIn("unoccupied", archive.summary())

    def test_archive_serialises_every_entry_not_just_the_slots(self):
        archive = Archive()
        for name in ("a", "b", "c"):
            archive.add(name, 0, make_report(name, labels={"never_reached_object": 10}), 8, name)
        d = archive.to_dict()
        self.assertEqual(d["num_entries"], 3)
        self.assertEqual(len(d["entries"]), 3)


# ---------------------------------------------------------------------------
#  Feedback
# ---------------------------------------------------------------------------


def evidence(candidate, labels, params=6, dead=None, code="class GeneratedPolicy: pass"):
    return CandidateEvidence(
        candidate_id=candidate,
        code=code,
        report=make_report(candidate, labels=labels),
        num_parameters=params,
        dead_parameters=dead or [],
    )


class TestFeedback(unittest.TestCase):
    def test_parents_are_chosen_to_differ_not_merely_to_score_well(self):
        leader = evidence("leader", {"success": 6, "never_reached_object": 4})
        twin = evidence("twin", {"success": 5, "never_reached_object": 5})
        other = evidence("other", {"pushed_away_from_goal": 10})
        parents = choose_informative_parents([leader, twin, other], limit=2)
        self.assertEqual(parents[0].candidate_id, "leader")
        self.assertEqual(parents[1].candidate_id, "other")

    def test_choosing_parents_from_nothing_returns_nothing(self):
        self.assertEqual(choose_informative_parents([]), [])

    def test_the_block_names_the_manifest_so_pairing_is_visible(self):
        block = build_feedback(
            [evidence("a", {"never_reached_object": 10})], 0, "push-v2:development#head10"
        )
        self.assertIn("push-v2:development#head10", block)
        self.assertIn("identical ordered case list", block)

    def test_the_block_carries_full_code_for_the_parents(self):
        block = build_feedback(
            [evidence("a", {"never_reached_object": 10}, code="class GeneratedPolicy:\n    marker = 1")],
            0,
            "m",
        )
        self.assertIn("marker = 1", block)

    def test_every_candidate_appears_in_the_population_table(self):
        block = build_feedback(
            [
                evidence("a", {"never_reached_object": 10}),
                evidence("b", {"pushed_away_from_goal": 10}),
                evidence("c", {"stopped_short_of_goal": 10}),
            ],
            0,
            "m",
        )
        for name in ("a", "b", "c"):
            self.assertIn(name, block)

    def test_dead_parameters_are_reported_as_dead_weight(self):
        block = build_feedback(
            [evidence("a", {"never_reached_object": 10}, dead=["log_std", "k_slip"])], 0, "m"
        )
        self.assertIn("k_slip", block)
        self.assertIn("dead weight", block)

    def test_candidates_that_never_compiled_are_reported_with_their_error(self):
        broken = CandidateEvidence(
            candidate_id="broken", code="", report=None, error="shape mismatch on axis 1"
        )
        block = build_feedback([evidence("a", {"never_reached_object": 10}), broken], 0, "m")
        self.assertIn("never reached evaluation", block)
        self.assertIn("shape mismatch", block)

    def test_the_loss_caveat_is_always_present(self):
        block = build_feedback([evidence("a", {"never_reached_object": 10})], 0, "m")
        self.assertIn("wrong sign", block)
        self.assertIn("not as evidence of quality", block)

    def test_fitting_making_a_candidate_worse_is_called_out(self):
        e = evidence("a", {"never_reached_object": 10})
        e.zero_shot_report = make_report("a", labels={"success": 10})
        block = build_feedback([e], 0, "m")
        self.assertIn("Fitting made it worse", block)

    def test_the_archive_appears_when_one_is_supplied(self):
        archive = Archive()
        archive.add("a", 0, make_report("a", labels={"never_reached_object": 10}), 8, "h")
        block = build_feedback(
            [evidence("a", {"never_reached_object": 10})], 0, "m", archive=archive
        )
        self.assertIn("### Archive", block)

    def test_history_lines_are_included_when_supplied(self):
        block = build_feedback(
            [evidence("a", {"never_reached_object": 10})], 1, "m",
            history=["generation 0: 5 valid, 0 invalid, best success 0.100"],
        )
        self.assertIn("generation 0:", block)


if __name__ == "__main__":
    unittest.main()
