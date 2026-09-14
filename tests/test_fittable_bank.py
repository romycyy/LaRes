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


if __name__ == "__main__":
    unittest.main()
