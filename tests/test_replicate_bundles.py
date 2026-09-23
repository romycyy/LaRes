"""Replicate bundles vary the data, and every structure in a replicate shares one (decided 2026-09-14).

No simulator and no model: demonstrations come from the kinematic push world's scripted expert and
manifests from a synthetic factory that gives each pool seed its own placements.

Run from the project root::

    python -m unittest tests.test_replicate_bundles
"""

import os
import sys
import unittest

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from lares.eval.manifest import EvaluationManifest, synthetic_cases, synthetic_manifest  # noqa: E402
from lares.fitting.replicates import (  # noqa: E402
    ReplicateBundleError,
    build_replicate_bundles,
    check_shared_bundles,
    qualification_checks,
    run_replicate_qualification,
)
from test_two_generation_mock import expert_buffer  # noqa: E402

RULE = {"improves_in_at_least": 3, "worst_seed_difference_at_least": -0.05, "mean_difference_greater_than": 0.0}
DATA_SEEDS = [2400, 2401, 2402, 2403, 2404]
PROBE_SEEDS = [2410, 2411, 2412, 2413, 2414]
QUALIFICATION_SEEDS = [2420, 2421, 2422, 2423, 2424]


def factory(role, pool_seed, episodes, replicate):
    base = synthetic_manifest(episodes, label=f"push-v2-{role}-s{pool_seed}")
    cases = synthetic_cases(episodes, prefix=f"{role}-s{pool_seed}", task_id=f"{role}:{pool_seed}", base_seed=pool_seed * 100)
    return EvaluationManifest(manifest_id=base.manifest_id, environment=base.environment, pool=base.pool,
                              split=base.split, episodes=cases)


def bundles_for(buffer, **overrides):
    kwargs = dict(data_seeds=DATA_SEEDS, demos_per_replicate=10, val_fraction=0.2, manifest_factory=factory,
                  probe_pool_seeds=PROBE_SEEDS, qualification_pool_seeds=QUALIFICATION_SEEDS, probe_episodes=3,
                  qualification_episodes=5)
    kwargs.update(overrides)
    return build_replicate_bundles(buffer, **kwargs)


class TestReplicateBundles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.buffer = expert_buffer(synthetic_manifest(14, label="push-v2-train").episodes, horizon=10)
        cls.bundles = bundles_for(cls.buffer)

    def test_each_replicate_samples_its_own_demonstrations(self):
        pool = set(self.buffer.episode_index())
        samples = [b.demo_episode_ids for b in self.bundles]
        self.assertEqual(len(set(samples)), 5)
        for sample in samples:
            self.assertEqual(len(sample), 10)
            self.assertEqual(len(set(sample)), 10)
            self.assertTrue(set(sample) <= pool)

    def test_bundles_are_a_function_of_their_seeds(self):
        again = bundles_for(self.buffer)
        self.assertEqual([b.bundle_id for b in again], [b.bundle_id for b in self.bundles])
        self.assertEqual(len({b.bundle_id for b in self.bundles}), 5)

    def test_each_replicate_has_its_own_trajectory_level_split(self):
        splits = set()
        for b in self.bundles:
            self.assertFalse(set(b.train_episode_ids) & set(b.val_episode_ids))
            self.assertEqual(set(b.train_episode_ids) | set(b.val_episode_ids), set(b.demo_episode_ids))
            self.assertEqual(len(b.val_episode_ids), 2)
            splits.add((b.train_episode_ids, b.val_episode_ids))
        self.assertEqual(len(splits), 5)

    def test_fitting_data_contains_only_the_bundle_episodes_split_by_trajectory(self):
        groups = self.buffer.episode_index()
        tensors = []
        for b in self.bundles:
            data = b.fitting_data(self.buffer)
            train = sum(len(groups[e]) for e in b.train_episode_ids)
            val = sum(len(groups[e]) for e in b.val_episode_ids)
            self.assertEqual(int(data.train_obs.shape[0]), train)
            self.assertEqual(int(data.val_obs.shape[0]), val)
            self.assertEqual(data.split_info["mode"], "by_episode")
            self.assertEqual(data.split_info["bundle_id"], b.bundle_id)
            tensors.append(data.train_obs)
        self.assertFalse(any(torch.equal(tensors[0], t) for t in tensors[1:] if t.shape == tensors[0].shape))

    def test_each_replicate_has_its_own_disjoint_probe_and_qualification_manifests(self):
        ids = []
        for b in self.bundles:
            probe = {(c.task_id, c.reset_seed) for c in b.probe_manifest.episodes}
            qualification = {(c.task_id, c.reset_seed) for c in b.qualification_manifest.episodes}
            self.assertFalse(probe & qualification)
            self.assertEqual((len(b.probe_manifest), len(b.qualification_manifest)), (3, 5))
            ids += [b.probe_manifest.manifest_id, b.qualification_manifest.manifest_id]
        self.assertEqual(len(set(ids)), 10)

    def test_a_probe_that_overlaps_the_qualification_manifest_is_refused(self):
        def overlapping(role, pool_seed, episodes, replicate):
            return factory("same", 1, episodes, replicate)

        with self.assertRaises(ReplicateBundleError):
            bundles_for(self.buffer, manifest_factory=overlapping)

    def test_repeated_seeds_are_refused(self):
        with self.assertRaises(ReplicateBundleError):
            bundles_for(self.buffer, data_seeds=[2400, 2400, 2402, 2403, 2404])
        with self.assertRaises(ReplicateBundleError):
            bundles_for(self.buffer, qualification_pool_seeds=[2410, 2421, 2422, 2423, 2424])

    def test_every_structure_in_a_replicate_shares_its_bundle(self):
        seen = {}

        def fit_and_score(sid, bundle, data):
            seen[(sid, bundle.replicate)] = (id(bundle), id(data))
            return {"zero_shot_success": 0.2, "selected_success": 0.6, "selected_step": 500}

        structures = ["simple_reach_push", "simple_damped_push"]
        result = run_replicate_qualification(structures, self.bundles, self.buffer, fit_and_score, RULE,
                                             controls=["simple_standoff"])
        for b in self.bundles:
            shared = {seen[(sid, b.replicate)] for sid in structures + ["simple_standoff"]}
            self.assertEqual(len(shared), 1, b.replicate)
        self.assertEqual(len({seen[("simple_reach_push", b.replicate)] for b in self.bundles}), 5)
        rows = [row for s in result["structures"].values() for row in s["rows"]]
        self.assertEqual(check_shared_bundles(rows, self.bundles), [])
        rows[0] = dict(rows[0], bundle_id=self.bundles[1].bundle_id)
        self.assertTrue(check_shared_bundles(rows, self.bundles))

    def test_fitting_rejected_contributes_nothing_and_controls_never_qualify(self):
        def fit_and_score(sid, bundle, data):
            if sid == "rejects":
                return {"zero_shot_success": 0.0, "selected_success": 0.9, "selected_step": 0}
            return {"zero_shot_success": 0.1, "selected_success": 0.5, "selected_step": 1000}

        result = run_replicate_qualification(["helps", "rejects"], self.bundles, self.buffer, fit_and_score, RULE,
                                             controls=["simple_standoff"])
        self.assertTrue(result["structures"]["helps"]["qualifies"])
        self.assertEqual([r["difference"] for r in result["structures"]["rejects"]["rows"]], [0.0] * 5)
        self.assertFalse(result["structures"]["rejects"]["qualifies"])
        self.assertTrue(result["structures"]["simple_standoff"]["control"])
        self.assertFalse(result["structures"]["simple_standoff"]["qualifies"])
        self.assertEqual(result["seed_bank"], ["helps"])

    def test_the_script_rule_is_the_shared_rule(self):
        import qualify_fittable_bank

        for differences in ([0.2, 0.1, 0.04, 0.0, -0.04], [0.3, 0.3, 0.3, 0.3, -0.06], [0.0] * 5):
            shared = qualification_checks(differences, RULE)
            script = qualify_fittable_bank.qualify(differences, RULE)
            self.assertEqual({k: script[k] for k in shared}, shared)
            self.assertIn("interval", script)


class TestQualificationV2Config(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(_PROJECT_ROOT, "config", "fittable_bank", "qualification_v2.yaml"), encoding="utf-8") as f:
            cls.cfg = yaml.safe_load(f)
        with open(os.path.join(_PROJECT_ROOT, "config", "fittable_bank", "qualification.yaml"), encoding="utf-8") as f:
            cls.v1 = yaml.safe_load(f)

    def test_replicates_vary_data_and_share_bundles(self):
        bundle = self.cfg["replicate_bundle"]
        self.assertTrue(bundle["shared_by_all_structures_in_a_replicate"])
        self.assertEqual(bundle["split"]["level"], "trajectory")
        seeds = self.cfg["seeds"]
        for key in ("data_seeds", "probe_pool_seeds", "qualification_pool_seeds"):
            self.assertEqual(len(seeds[key]), self.cfg["replicates"])
        every = seeds["data_seeds"] + seeds["probe_pool_seeds"] + seeds["qualification_pool_seeds"]
        self.assertEqual(len(set(every)), len(every))
        spent = set(self.v1["seeds"]["fitting_seeds"]) | {self.v1["seeds"]["evaluation_pool_seed"]}
        self.assertFalse(set(every) & spent)
        self.assertEqual(self.cfg["status"], "predeclared_not_run")

    def test_controls_are_scored_but_standoff_variants_are_retired(self):
        structures = self.cfg["structures"]
        self.assertEqual(sorted(structures["non_fittable_controls"]), ["simple_damped_push_smooth_gate", "simple_standoff"])
        self.assertEqual(sorted(structures["retired"]), ["simple_standoff_moderate_geometry", "simple_standoff_phase_gains"])
        self.assertFalse(any("standoff" in sid for sid in structures["fittable_seeds"]))
        self.assertEqual(self.cfg["fitting"]["checkpoint_steps"], [0, 500, 1000])


if __name__ == "__main__":
    unittest.main()
