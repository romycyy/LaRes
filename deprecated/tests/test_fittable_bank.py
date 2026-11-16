"""The fittable-bank variants and qualification rule, checked before any rollout.

Run from the project root::

    python -m unittest tests.test_fittable_bank
"""

import os
import sys
import unittest

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))

import qualify_fittable_bank as qfb  # noqa: E402
from lares.core.obs_schema import get_obs_schema  # noqa: E402
from lares.core.policy_validator import validate_policy  # noqa: E402
from lares.fitting.structure_bank import SIMPLE_FAMILY_MAX_PARAMS, STRUCTURES_DIR, load_bank  # noqa: E402

SCHEMA = get_obs_schema("push-v2")


class TestVariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = qfb.load_config()
        cls.references = qfb.reference_structures(cls.cfg)
        cls.variants = qfb.variant_structures(cls.cfg)

    def test_there_are_six_variants_over_the_three_references(self):
        self.assertEqual(len(self.variants), 6)
        self.assertEqual(set(self.references), {"simple_reach_push", "simple_damped_push", "simple_standoff"})
        self.assertEqual({v["parent"] for v in self.cfg["structures"]["variants"]}, set(self.references))

    def test_every_variant_passes_the_full_validator_with_levers_on(self):
        for sid, structure in self.variants.items():
            report = validate_policy(structure.build(SCHEMA), 39, 4, source=structure.source, schema=SCHEMA)
            self.assertTrue(report.ok, f"{sid}: {report.summary()}")
            self.assertIn("gate_levers", report.checked)

    def test_each_variant_differs_from_its_parent_only_as_declared(self):
        for declaration in self.cfg["structures"]["variants"]:
            parent = self.references[declaration["parent"]].build(SCHEMA)
            variant = self.variants[declaration["id"]].build(SCHEMA)
            self.assertEqual(qfb.declared_difference_problems(parent, variant, declaration), [], declaration["id"])

    def test_an_undeclared_difference_is_caught(self):
        declaration = dict(self.cfg["structures"]["variants"][0], changes=[])
        parent = self.references[declaration["parent"]].build(SCHEMA)
        variant = self.variants[declaration["id"]].build(SCHEMA)
        self.assertTrue(qfb.declared_difference_problems(parent, variant, declaration))

    def test_variants_stay_in_the_simple_family_and_name_their_provenance(self):
        for sid, structure in self.variants.items():
            self.assertLessEqual(structure.num_parameters, SIMPLE_FAMILY_MAX_PARAMS, sid)
            self.assertIn("variant of", structure.provenance, sid)

    def test_the_phase_three_bank_is_unchanged(self):
        ids = {s.structure_id for s in load_bank(STRUCTURES_DIR)}
        self.assertEqual(len(ids), 18)
        self.assertFalse(ids & set(self.variants))

    def test_selection_and_evaluation_use_different_manifests(self):
        self.assertNotEqual(self.cfg["checkpoint_selection_manifest"]["path"], self.cfg["evaluation_manifest"]["path"])
        self.assertEqual(self.cfg["fitting"]["checkpoint_steps"], [0, 500, 1000])


class TestQualificationRule(unittest.TestCase):
    RULE = {"improves_in_at_least": 3, "worst_seed_difference_at_least": -0.05, "mean_difference_greater_than": 0.0}

    def test_three_improvements_no_large_regression_and_a_positive_mean_qualify(self):
        self.assertTrue(qfb.qualify([0.2, 0.1, 0.04, 0.0, -0.04], self.RULE)["qualifies"])

    def test_fewer_than_three_improvements_do_not(self):
        result = qfb.qualify([0.5, 0.5, 0.0, 0.0, 0.0], self.RULE)
        self.assertFalse(result["qualifies"])
        self.assertFalse(result["checks"]["improves_in_enough_seeds"])

    def test_one_regression_beyond_five_points_disqualifies(self):
        result = qfb.qualify([0.3, 0.3, 0.3, 0.3, -0.06], self.RULE)
        self.assertFalse(result["qualifies"])
        self.assertFalse(result["checks"]["no_seed_regresses_beyond_tolerance"])

    def test_a_regression_of_exactly_five_points_is_tolerated(self):
        self.assertTrue(qfb.qualify([0.1, 0.1, 0.1, 0.0, -0.05], self.RULE)["qualifies"])

    def test_rejected_fitting_is_never_an_improvement(self):
        result = qfb.qualify([0.0, 0.0, 0.0, 0.0, 0.0], self.RULE)
        self.assertEqual(result["improvements"], 0)
        self.assertFalse(result["qualifies"])

    def test_the_mean_must_be_positive(self):
        result = qfb.qualify([0.02, 0.02, 0.02, -0.05, -0.05], self.RULE)
        self.assertFalse(result["checks"]["mean_difference_positive"])


class TestReplicationComparison(unittest.TestCase):
    """R-004: success_se, its paired bootstrap, and the seed reservation. No rollouts."""

    def rows(self):
        out = []
        for rep, (a, b) in enumerate([(0.9, 0.80), (0.7, 0.82), (0.95, 0.81), (0.6, 0.79), (0.85, 0.80)]):
            for sid, offset in (("s1", 0.05), ("s2", -0.05)):
                out.append({"arm": "data_varying", "replicate": rep, "structure_id": sid, "selected_success": a + offset})
                out.append({"arm": "minibatch_only", "replicate": rep, "structure_id": sid, "selected_success": b + offset})
        return out

    def test_bank_mean_weights_structures_equally_per_replicate(self):
        self.assertEqual([round(x, 6) for x in qfb.bank_means(self.rows(), "data_varying")], [0.9, 0.7, 0.95, 0.6, 0.85])

    def test_success_se_is_sd_over_replicates_divided_by_root_n(self):
        import numpy as np

        values = [0.9, 0.7, 0.95, 0.6, 0.85]
        self.assertAlmostEqual(qfb.success_se(values), float(np.std(values, ddof=1) / np.sqrt(5)))

    def test_bootstrap_difference_is_positive_when_data_varies_more(self):
        rows = self.rows()
        result = qfb.paired_se_bootstrap(qfb.bank_means(rows, "data_varying"), qfb.bank_means(rows, "minibatch_only"),
                                         resamples=2000)
        self.assertGreater(result["difference"], 0.05)
        self.assertLessEqual(result["low"], result["difference"])
        self.assertGreaterEqual(result["high"], result["low"])

    def test_bootstrap_refuses_unpaired_arms(self):
        with self.assertRaises(ValueError):
            qfb.paired_se_bootstrap([0.1, 0.2, 0.3], [0.1, 0.2])

    def test_reservation_is_idempotent_for_its_out_dir_and_refuses_reuse(self):
        import tempfile

        cfg = qfb.load_config(os.path.join(_PROJECT_ROOT, "config", "fittable_bank", "qualification_v2.yaml"))
        cfg["seeds"] = {"data_seeds": [99100, 99101], "probe_pool_seeds": [99110, 99111],
                        "qualification_pool_seeds": [99120, 99121]}
        cfg["minibatch_control"] = {"data_seed": 99130, "probe_pool_seed": 99131, "fitting_seeds": [99132, 99133]}
        with tempfile.TemporaryDirectory() as d:
            ledger = os.path.join(d, "ledger.jsonl")
            open(ledger, "w").close()
            out = os.path.join(_PROJECT_ROOT, "logs", "r004_test")
            self.assertEqual(qfb.ensure_replication_reserved(cfg, out, ledger_path=ledger), "reserved")
            self.assertEqual(qfb.ensure_replication_reserved(cfg, out, ledger_path=ledger),
                             "already reserved for this out-dir")
            with self.assertRaises(SystemExit):
                qfb.ensure_replication_reserved(cfg, out + "_other", ledger_path=ledger)


if __name__ == "__main__":
    unittest.main()
