"""R-005's comparison script: pair discovery, the placement-level statistic, seed reservation.

No rollouts and no MetaWorld. Run from the project root::

    python -m unittest tests.test_child_vs_parent
"""

import json
import os
import sys
import tempfile
import unittest

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))

import compare_child_vs_parent as cvp  # noqa: E402
from lares.utils.hashing import sha256_file, sha256_text  # noqa: E402

PARENTS = {
    "simple_reach_push": "lares/fitting/structures/simple_reach_push.py",
    "simple_damped_push": "lares/fitting/structures/simple_damped_push.py",
}
FIT = {"objective": "deterministic_mse", "sampling": "uniform", "budget_gradient_steps": 1000, "batch_size": 256,
       "lr": 0.001, "clip_grad_norm": 1.0, "checkpoint_steps": [0, 500, 1000], "fixed_action_scale": 0.1,
       "val_fraction": 0.2, "split_seed": 0, "seed": 0}


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(payload, str):
            f.write(payload)
        else:
            json.dump(payload, f)


def make_run(root, children, lane=cvp.LANE_FITTABLE, fit=None):
    """A minimal two-lane run: ``children`` is ``[(attempt_id, parent_id)]``."""
    for attempt_id, parent_id in children:
        source = f"# child {attempt_id}\n"
        source_rel = f"gen_0/candidates/{attempt_id}/policy_source.py"
        _write(os.path.join(root, source_rel), source)
        _write(os.path.join(root, "experiments", f"{attempt_id}.json"), {
            "experiment_id": attempt_id, "candidate_id": attempt_id.rsplit("_", 1)[0],
            "code": {"source_path": source_rel, "source_sha256": sha256_text(source)},
            "lineage": {"lane": lane, "lane_metadata": {
                "reference": {"structure_id": parent_id, "source_path": PARENTS[parent_id],
                              "source_sha256": sha256_file(os.path.join(_PROJECT_ROOT, PARENTS[parent_id]))},
                "mutation_scope": {"id": "single_bounded_change"}}},
            "fitting": {"objective_config": dict(fit or FIT), "probe_manifest_id": "dev#tail10", "selected_step": 1000,
                        "initial_state": {"path": f"gen_0/candidates/{attempt_id}/initial_state.pt"}},
        })
    _write(os.path.join(root, "incumbents.json"), {"performance_incumbent": {"attempt_id": children[0][0]}})


class TestDiscovery(unittest.TestCase):
    def test_pairs_name_each_child_and_its_seed_bank_parent(self):
        with tempfile.TemporaryDirectory() as d:
            make_run(d, [("gen0_cand0_a0", "simple_reach_push"), ("gen0_cand4_a0", "simple_damped_push"),
                         ("gen0_cand5_a0", "simple_damped_push")])
            pairs = cvp.discover_pairs(d)
            self.assertEqual([(p["child_id"], p["parent_id"]) for p in pairs],
                             [("gen0_cand0_a0", "simple_reach_push"), ("gen0_cand4_a0", "simple_damped_push"),
                              ("gen0_cand5_a0", "simple_damped_push")])
            self.assertEqual(cvp.common_fitting(pairs)["budget_gradient_steps"], 1000)
            self.assertEqual(cvp.performance_incumbent(d), "gen0_cand0_a0")

    def test_exploration_candidates_are_not_pairs(self):
        with tempfile.TemporaryDirectory() as d:
            make_run(d, [("gen0_cand6_a0", "simple_reach_push")], lane="exploration")
            with self.assertRaises(SystemExit):
                cvp.discover_pairs(d)

    def test_a_changed_child_source_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            make_run(d, [("gen0_cand0_a0", "simple_reach_push")])
            _write(os.path.join(d, "gen_0/candidates/gen0_cand0_a0/policy_source.py"), "# edited\n")
            with self.assertRaises(SystemExit):
                cvp.discover_pairs(d)

    def test_children_fitted_differently_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            make_run(d, [("gen0_cand0_a0", "simple_reach_push")])
            make_run(d, [("gen0_cand1_a0", "simple_reach_push")], fit={**FIT, "lr": 0.01})
            with self.assertRaises(SystemExit):
                cvp.common_fitting(cvp.discover_pairs(d))

    def test_baseline_best_is_its_promotion_tested_candidate(self):
        with tempfile.TemporaryDirectory() as d:
            for eid, tested in (("gen0_cand0_a0", False), ("gen0_cand2_a1", True)):
                _write(os.path.join(d, "experiments", f"{eid}.json"), {
                    "experiment_id": eid, "code": {"source_path": f"gen_0/candidates/{eid}/policy_source.py"},
                    "fitting": {"selected_state": {"path": f"gen_0/candidates/{eid}/selected_state.pt"}, "selected_step": 0},
                    "evaluation": {"screening": {"promotion_tested": tested}}})
            self.assertEqual(cvp.baseline_best(d)["id"], "gen0_cand2_a1")


class TestStatistic(unittest.TestCase):
    PAIRS = [{"child_id": "c0", "parent_id": "p0", "scope": "s"}, {"child_id": "c1", "parent_id": "p1", "scope": "s"},
             {"child_id": "c2", "parent_id": "p1", "scope": "s"}]

    def per_case(self):
        return {
            "c0": {"a": True, "b": True, "c": False},
            "c1": {"a": True, "b": False, "c": False},
            "c2": {"a": False, "b": True, "c": True},
            "p0": {"a": False, "b": True, "c": False},
            "p1": {"a": True, "b": True, "c": False},
            "base": {"a": False, "b": False, "c": False},
        }

    def test_differences_average_over_pairs_within_each_placement(self):
        by_case = cvp.placement_differences(self.per_case(), self.PAIRS)
        # a: (1-0) + (1-1) + (0-1) = 0; b: 0 + (0-1) + 0 = -1; c: 0 + 0 + 1 = 1, each over 3 pairs
        self.assertAlmostEqual(by_case["a"], 0.0)
        self.assertAlmostEqual(by_case["b"], -1 / 3)
        self.assertAlmostEqual(by_case["c"], 1 / 3)

    def test_interval_is_over_placements_not_episodes(self):
        s = cvp.summarise(self.per_case(), self.PAIRS, "c0", "base")
        self.assertEqual(s["primary"]["n"], 3)
        self.assertEqual(s["primary"]["n_pairs"], 3)
        self.assertAlmostEqual(s["paired_success_difference"], 0.0)
        self.assertEqual(len(s["per_pair"]), 3)
        self.assertAlmostEqual(s["secondary_r001"]["mean"], 2 / 3)

    def test_policies_on_different_cases_are_refused(self):
        per_case = self.per_case()
        del per_case["p1"]["c"]
        with self.assertRaises(ValueError):
            cvp.placement_differences(per_case, self.PAIRS)


class TestNestedScreen(unittest.TestCase):
    """R-007: Stage F's acceptance rule replayed on subsets of a larger screen."""

    def test_the_rule_needs_strictly_better_success_or_progress(self):
        child = {"a": (1.0, 0.2), "b": (0.0, 0.1)}
        self.assertEqual(cvp.screen_decision(child, {"a": (0.0, 0.2), "b": (0.0, 0.1)}, ["a", "b"]), cvp.BRANCH_ACCEPTED)
        self.assertEqual(cvp.screen_decision(child, {"a": (1.0, 0.3), "b": (1.0, 0.1)}, ["a", "b"]),
                         cvp.BRANCH_LOWER_SUCCESS)
        self.assertEqual(cvp.screen_decision(child, dict(child), ["a", "b"]), cvp.BRANCH_TIED_SUCCESS_LOWER_PROGRESS)

    def test_a_child_better_only_on_part_of_the_screen_is_rejected_by_some_subsets(self):
        cases = [f"s{i}" for i in range(30)]
        # The child wins 6 of 30 screen cases and ties the rest at slightly lower progress.
        child = {c: ((1.0 if i < 6 else 0.0), 0.10) for i, c in enumerate(cases)}
        parent = {c: (0.0, 0.11) for c in cases}
        conf = [f"c{i}" for i in range(20)]
        child_conf = {c: (1.0 if i < 8 else 0.0) for i, c in enumerate(conf)}
        parent_conf = {c: (1.0 if i < 4 else 0.0) for i, c in enumerate(conf)}
        out = cvp.nested_screen_analysis(child, parent, child_conf, parent_conf, replays=500, seed=1)
        self.assertEqual(out["full_screen_decision"], cvp.BRANCH_ACCEPTED)
        self.assertTrue(0.0 < out["ten_case_acceptance_rate"] < 1.0)
        self.assertEqual(sum(out["ten_case_branches"].values()), 500)
        expected = (1 - out["ten_case_acceptance_rate"]) * 0.2      # 4 of 20 cases differ by 1
        self.assertAlmostEqual(out["paired_success_difference"], expected)
        self.assertAlmostEqual(out["refitted_child_minus_parent_confirmation"]["mean"], 0.2)

    def test_unpaired_cases_are_refused(self):
        with self.assertRaises(ValueError):
            cvp.nested_screen_analysis({"a": (1, 0)}, {"b": (1, 0)}, {"c": 1}, {"c": 1}, replays=10, seed=0, subset=1)


class TestSeedRobustness(unittest.TestCase):
    RULE = {"improves_in_at_least": 3, "worst_seed_difference_at_least": -0.05, "mean_difference_greater_than": 0.0}

    def test_reports_placement_and_seed_level_intervals_and_the_childs_qualification(self):
        cases = [f"c{i}" for i in range(4)]
        child = {s: {c: float(i < 3) for i, c in enumerate(cases)} for s in (1, 2, 3)}
        parent = {1: {c: float(i < 2) for i, c in enumerate(cases)}, 2: {c: float(i < 3) for i, c in enumerate(cases)},
                  3: {c: float(i < 1) for i, c in enumerate(cases)}}
        step0 = {c: float(i < 1) for i, c in enumerate(cases)}
        out = cvp.seed_robustness_analysis(child, parent, step0, step0, self.RULE)
        self.assertEqual(out["seed_level"]["by_seed"], {1: 0.25, 2: 0.0, 3: 0.5})
        self.assertAlmostEqual(out["paired_success_difference"], 0.25)
        self.assertEqual(out["placement_level"]["n"], 4)
        self.assertEqual(out["child_qualification_report_only"]["fitted_minus_step0_by_seed"], [0.5, 0.5, 0.5])
        self.assertTrue(out["child_qualification_report_only"]["qualifies"])

    def test_mismatched_seeds_are_refused(self):
        with self.assertRaises(ValueError):
            cvp.seed_robustness_analysis({1: {"a": 1}}, {2: {"a": 1}}, {"a": 0}, {"a": 0}, self.RULE)


class TestSeeds(unittest.TestCase):
    def test_reserves_once_then_recognises_its_own_reservation(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = os.path.join(d, "ledger.jsonl")
            open(ledger, "w").close()
            out = os.path.join(_PROJECT_ROOT, "logs", "child_vs_parent_test")
            self.assertEqual(cvp.ensure_reserved(99001, 99002, out, ledger_path=ledger), "reserved")
            self.assertEqual(cvp.ensure_reserved(99001, 99002, out, ledger_path=ledger),
                             "already reserved for this out-dir")
            with self.assertRaises(SystemExit):
                cvp.ensure_reserved(99001, 99003, out + "_other", ledger_path=ledger)

    def test_a_spent_seed_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            ledger = os.path.join(d, "ledger.jsonl")
            _write(ledger, json.dumps({"seed": 99001, "namespace": "fitting", "use": "x", "artifact": "y"}) + "\n")
            with self.assertRaises(SystemExit):
                cvp.ensure_reserved(99001, 99002, os.path.join(d, "out"), ledger_path=ledger)


if __name__ == "__main__":
    unittest.main()


class TestAddedParameterAblation(unittest.TestCase):
    def test_child_only_parameters_and_clamping(self):
        from lares.core.obs_schema import get_obs_schema
        from lares.fitting.structure_bank import instantiate_source

        schema = get_obs_schema("push-v2")
        child = os.path.join(_PROJECT_ROOT, "logs", "two_lane_search", "gen_0", "candidates", "gen0_cand5_a0",
                             "policy_source.py")
        if not os.path.exists(child):
            self.skipTest("the two-lane run is not on this machine")
        with open(child) as f:
            c = instantiate_source(f.read(), 39, 4, schema)
        with open(os.path.join(_PROJECT_ROOT, PARENTS["simple_damped_push"])) as f:
            p = instantiate_source(f.read(), 39, 4, schema)
        self.assertEqual(cvp.child_only_parameters(c, p), ["approach_standoff"])
        cvp.with_ranges(c, {"approach_standoff": (0.02, 0.02)})
        import torch

        with torch.no_grad():
            c.approach_standoff.add_(0.01)
        c.clip_params()
        self.assertAlmostEqual(float(c.approach_standoff), 0.02, places=6)

    def test_ablation_analysis_pairs_by_seed_and_placement(self):
        trainable = {1: {"a": 1.0, "b": 1.0}, 2: {"a": 1.0, "b": 0.0}}
        clamped = {1: {"a": 0.0, "b": 1.0}, 2: {"a": 1.0, "b": 0.0}}
        out = cvp.ablation_analysis(trainable, clamped)
        self.assertAlmostEqual(out["paired_success_difference"], 0.25)
        self.assertEqual(out["seed_level"]["by_seed"], {1: 0.5, 2: 0.0})
