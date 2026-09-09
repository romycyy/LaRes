#!/usr/bin/env python
"""Tests for regenerating the aggregate report from saved artifacts (AC-7).

The property under test is that the report says only what the files support. A
report that quietly fills a gap with zero is the failure this suite exists to
prevent.

Run from the project root::

    python -m unittest tests.test_final_report
"""

import json
import os
import sys
import tempfile
import unittest

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from lares.eval.final_report import (  # noqa: E402
    REQUIRED_QUANTITIES,
    ablation_section,
    build_final_report,
    final_test_section,
    load_records,
    render,
    repair_section,
    search_section,
    validity_section,
)
from lares.search.schemas import (  # noqa: E402
    STATUS_EVALUATED,
    STATUS_INVALID,
    ExperimentRecord,
)


def a_record(candidate="c0", status=STATUS_EVALUATED, success=0.3, intervention=False, **kw):
    record = ExperimentRecord(
        experiment_id=f"e_{candidate}",
        candidate_id=candidate,
        generation=0,
        status=status,
        code={"source_hash": "h", "num_parameters": 6},
        evaluation={
            "manifest_ids": ["dev"], "action_mode": "deterministic",
            "success_rate": success, "mean_return": 100.0,
        } if status == STATUS_EVALUATED else {},
        resources={"environment_steps": 1500},
        fitting={"steps": 2000},
        intervention=intervention,
        intervention_detail="expert drives dz" if intervention else "",
        **kw,
    )
    return record


def a_study(accepted=True, attempted=True, supported=1, rejected=1):
    outcomes = (
        [{"explanation": f"s{i}", "check": {"supported": True}} for i in range(supported)]
        + [{"explanation": f"r{i}", "check": {"supported": False}} for i in range(rejected)]
    )
    study = {
        "structure_id": "simple_standoff",
        "intervention_episodes": 20,
        "studies": [{"cluster": {"label": "never_reached_object"}, "outcomes": outcomes}],
    }
    if attempted:
        study["repair"] = {"attempted": True, "decision": {"accepted": accepted}}
    return study


def an_ablation(budget=200, arms=("full", "optimizer_only"), success=0.2):
    return {
        "budget_screen_episodes": budget,
        "rows": [
            {
                "arm": arm, "structure_id": "s", "seed": 0,
                "held_out_success": success, "baseline_success": 0.0,
                "held_out_progress": 0.01,
                "cost": {"screen_episodes": budget, "adam_steps": 0,
                         "expert_assisted_episodes": 20 if arm == "full" else 0},
            }
            for arm in arms
        ],
    }


class TestSearchSection(unittest.TestCase):
    def test_intervention_records_are_counted_but_never_scored(self):
        records = [a_record("c0", success=0.4), a_record("c1", success=1.0, intervention=True)]
        section = search_section(records)
        self.assertEqual(section["interventions"], 1)
        self.assertEqual(section["scored"], 1)
        self.assertAlmostEqual(section["mean_success"], 0.4)

    def test_an_empty_directory_reports_nothing_rather_than_zero(self):
        section = search_section([])
        self.assertIsNone(section["mean_success"])
        self.assertIsNone(section["best_success"])

    def test_failed_candidates_stay_in_the_denominator_of_the_attempt_count(self):
        records = [a_record("c0"), a_record("c1", status=STATUS_INVALID)]
        self.assertEqual(search_section(records)["records"], 2)


class TestValiditySection(unittest.TestCase):
    def test_the_invalid_rate_is_over_attempted_candidates(self):
        records = [a_record("c0"), a_record("c1", status=STATUS_INVALID),
                   a_record("c2", status=STATUS_INVALID)]
        self.assertAlmostEqual(validity_section(records)["invalid_program_rate"], 2 / 3)

    def test_intervention_records_are_not_generated_candidates(self):
        records = [a_record("c0"), a_record("c1", intervention=True)]
        self.assertEqual(validity_section(records)["attempted"], 1)

    def test_no_candidates_means_no_rate(self):
        self.assertIsNone(validity_section([])["invalid_program_rate"])


class TestRepairSection(unittest.TestCase):
    def test_acceptance_is_over_attempted_repairs(self):
        section = repair_section([a_study(accepted=True), a_study(accepted=False)])
        self.assertAlmostEqual(section["repair_acceptance_rate"], 0.5)

    def test_rejected_explanations_are_reported_not_dropped(self):
        section = repair_section([a_study(supported=1, rejected=3)])
        self.assertEqual(section["explanations_rejected"], 3)

    def test_a_study_with_no_repair_is_not_an_attempt(self):
        section = repair_section([a_study(attempted=False)])
        self.assertEqual(section["repairs_attempted"], 0)
        self.assertIsNone(section["repair_acceptance_rate"])

    def test_intervention_episodes_are_summed_across_studies(self):
        self.assertEqual(repair_section([a_study(), a_study()])["intervention_episodes"], 40)


class TestAblationSection(unittest.TestCase):
    def test_runs_at_different_budgets_are_never_pooled(self):
        section = ablation_section([an_ablation(budget=60, success=0.9),
                                    an_ablation(budget=200, success=0.1)])
        self.assertEqual(section["budget_screen_episodes"], 200)
        self.assertEqual(section["budgets_excluded"], [60])
        self.assertAlmostEqual(section["arms"]["full"]["held_out_success"], 0.1)

    def test_a_budget_can_be_named_explicitly(self):
        section = ablation_section([an_ablation(budget=60, success=0.9),
                                    an_ablation(budget=200, success=0.1)], budget=60)
        self.assertAlmostEqual(section["arms"]["full"]["held_out_success"], 0.9)

    def test_replicates_at_one_budget_are_pooled(self):
        section = ablation_section([an_ablation(), an_ablation()])
        self.assertEqual(section["arms"]["full"]["replicates"], 2)

    def test_no_runs_reports_none(self):
        self.assertEqual(ablation_section([])["runs"], 0)

    def test_the_cost_of_each_arm_is_carried_alongside_its_score(self):
        arms = ablation_section([an_ablation()])["arms"]
        self.assertEqual(arms["full"]["expert_assisted_episodes"], 20)
        self.assertEqual(arms["optimizer_only"]["expert_assisted_episodes"], 0)


class TestFinalTestSection(unittest.TestCase):
    def test_an_untouched_final_split_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            section = final_test_section(tmp)
        self.assertEqual(section["looks"], 0)
        self.assertFalse(section["frozen"])
        self.assertIn("never been evaluated", section["note"])

    def test_every_look_is_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "final_test_ledger.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"key": "a", "rescore": False}) + "\n")
                f.write(json.dumps({"key": "a", "rescore": True}) + "\n")
            section = final_test_section(tmp)
        self.assertEqual(section["looks"], 2)
        self.assertEqual(section["distinct_candidates"], 1)
        self.assertEqual(section["rescores"], 1)


class TestAssembly(unittest.TestCase):
    def _tree(self, tmp):
        experiments = os.path.join(tmp, "experiments")
        studies = os.path.join(tmp, "studies")
        ablations = os.path.join(tmp, "ablations")
        final = os.path.join(tmp, "final")
        for d in (experiments, studies, ablations, final):
            os.makedirs(d)
        a_record("c0").save(experiments)
        with open(os.path.join(studies, "s.json"), "w", encoding="utf-8") as f:
            json.dump(a_study(), f)
        with open(os.path.join(ablations, "a.json"), "w", encoding="utf-8") as f:
            json.dump(an_ablation(), f)
        return experiments, studies, ablations, final

    def test_it_assembles_from_files_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = build_final_report(*self._tree(tmp))
        self.assertEqual(report.sources["experiment_records"], 1)
        self.assertEqual(report.repair["repairs_accepted"], 1)
        self.assertEqual(report.ablation["replicates"], 2)

    def test_the_same_artifacts_give_the_same_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = self._tree(tmp)
            first = build_final_report(*tree).to_dict()
            second = build_final_report(*tree).to_dict()
        first.pop("generated_at")
        second.pop("generated_at")
        self.assertEqual(first, second)

    def test_missing_directories_are_survivable(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = build_final_report(
                os.path.join(tmp, "nope"), os.path.join(tmp, "nope2"),
                os.path.join(tmp, "nope3"), os.path.join(tmp, "nope4"),
            )
        self.assertEqual(report.sources["experiment_records"], 0)
        self.assertIsNone(report.search["mean_success"])

    def test_unmeasured_quantities_are_named_with_where_they_belong(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = build_final_report(*self._tree(tmp))
        self.assertIn("llm_tokens", report.unavailable)
        self.assertIn("inference_latency", report.unavailable)
        for quantity, source in report.unavailable.items():
            self.assertEqual(source, REQUIRED_QUANTITIES[quantity])

    def test_a_measured_quantity_is_not_listed_as_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = build_final_report(*self._tree(tmp))
        self.assertNotIn("success", report.unavailable)
        self.assertNotIn("complexity", report.unavailable)

    def test_the_rendering_states_the_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = render(build_final_report(*self._tree(tmp)))
        self.assertIn("Not measured", text)
        self.assertIn("absent, not zero", text)

    def test_records_load_from_several_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = os.path.join(tmp, "a"), os.path.join(tmp, "b")
            os.makedirs(first)
            os.makedirs(second)
            a_record("c0").save(first)
            a_record("c1").save(second)
            report = build_final_report([first, second], tmp, tmp, tmp)
        self.assertEqual(report.sources["experiment_records"], 2)

    def test_every_ac7_quantity_is_either_reported_or_named_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = build_final_report(*self._tree(tmp))
        text = render(report)
        for quantity in REQUIRED_QUANTITIES:
            with self.subTest(quantity=quantity):
                self.assertTrue(
                    quantity in report.unavailable or quantity.split("_")[0] in text.lower()
                )


class TestLoadRecords(unittest.TestCase):
    def test_records_round_trip_through_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            a_record("c0").save(tmp)
            loaded = load_records(tmp)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].candidate_id, "c0")

    def test_an_empty_directory_loads_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_records(tmp), [])


if __name__ == "__main__":
    unittest.main()
