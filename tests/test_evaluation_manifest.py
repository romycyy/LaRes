#!/usr/bin/env python
"""Tests for the evaluation manifest contract (``spec.md`` FR-1 / AC-0).

Run from the project root::

    python -m unittest tests.test_evaluation_manifest
"""

import os
import sys
import tempfile
import unittest

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from lares.eval.manifest import (  # noqa: E402
    SPLIT_DEVELOPMENT,
    SPLIT_FINAL_TEST,
    SPLIT_TRAIN,
    EpisodeCase,
    EvaluationManifest,
    assert_disjoint,
    build_manifest,
    build_task_pool,
    synthetic_cases,
    synthetic_manifest,
    to_v3,
)

try:
    import metaworld  # noqa: F401

    HAS_METAWORLD = True
except ImportError:
    HAS_METAWORLD = False

MANIFEST_DIR = os.path.join(_PROJECT_ROOT, "config", "manifests")
ENV_ID = "push-v2"


class TestManifestSchema(unittest.TestCase):
    """Schema behaviour that needs no simulator."""

    def _manifest(self, n=4, split=SPLIT_DEVELOPMENT):
        return synthetic_manifest(n, label="unit", split=split)

    def test_name_mapping_keeps_v2_spelling_in_configs(self):
        self.assertEqual(to_v3("push-v2"), "push-v3")
        self.assertEqual(to_v3("push-v3"), "push-v3")

    def test_yaml_roundtrip_is_lossless(self):
        manifest = self._manifest()
        with tempfile.TemporaryDirectory() as tmp:
            path = manifest.save(os.path.join(tmp, "m.yaml"))
            loaded = EvaluationManifest.load(path)
        self.assertEqual(loaded.to_dict(), manifest.to_dict())

    def test_head_is_a_nested_prefix_not_a_resample(self):
        manifest = self._manifest(n=10)
        head = manifest.head(3)
        self.assertEqual(len(head), 3)
        self.assertEqual(
            [c.case_id for c in head.episodes],
            [c.case_id for c in manifest.episodes[:3]],
        )
        # Screening must stay inside the expanded set, so a candidate promoted on
        # 10 cases is re-scored on those same 10 plus more.
        self.assertTrue(set(head.task_hashes()) <= set(manifest.task_hashes()))

    def test_head_beyond_length_is_refused(self):
        with self.assertRaises(ValueError):
            self._manifest(n=2).head(5)

    def test_tail_is_the_disjoint_complement_of_a_head(self):
        manifest = self._manifest(n=10)
        head, tail = manifest.head(4), manifest.tail(6)
        self.assertEqual(len(head) + len(tail), len(manifest))
        self.assertEqual(
            set(c.case_id for c in head.episodes) & set(c.case_id for c in tail.episodes),
            set(),
        )

    def test_tail_keeps_the_last_cases_in_order(self):
        manifest = self._manifest(n=10)
        self.assertEqual(
            [c.case_id for c in manifest.tail(3).episodes],
            [c.case_id for c in manifest.episodes[-3:]],
        )

    def test_tail_beyond_length_is_refused(self):
        with self.assertRaises(ValueError):
            self._manifest(n=2).tail(5)

    def test_narrowing_twice_replaces_the_suffix_rather_than_stacking(self):
        manifest = self._manifest(n=10)
        self.assertEqual(manifest.head(8).tail(3).manifest_id.count("#"), 1)

    def test_duplicate_case_ids_are_refused(self):
        case = EpisodeCase("dup", "t", "h", 1, 2)
        manifest = self._manifest()
        with self.assertRaises(ValueError):
            EvaluationManifest(
                manifest_id="bad",
                environment=manifest.environment,
                pool=manifest.pool,
                split=SPLIT_DEVELOPMENT,
                episodes=[case, case],
            )

    def test_unknown_split_is_refused(self):
        manifest = self._manifest()
        with self.assertRaises(ValueError):
            EvaluationManifest(
                manifest_id="bad",
                environment=manifest.environment,
                pool=manifest.pool,
                split="holdout",
                episodes=[],
            )

    def test_synthetic_manifest_cannot_be_resolved_to_a_pool(self):
        # A synthetic manifest carries no version lock, so it must never stand in
        # for a real one in a reported result.
        with self.assertRaises(ValueError):
            self._manifest().resolve_pool()

    def test_synthetic_cases_are_deterministic(self):
        a = synthetic_cases(5, prefix="x", base_seed=7)
        b = synthetic_cases(5, prefix="x", base_seed=7)
        self.assertEqual([c.to_dict() for c in a], [c.to_dict() for c in b])

    def test_assert_disjoint_detects_a_shared_placement(self):
        env = self._manifest().environment
        pool = self._manifest().pool
        shared = EpisodeCase("a-0000", "t0", "HASH", 1, 2)
        left = EvaluationManifest("left", env, pool, SPLIT_TRAIN, [shared])
        right = EvaluationManifest(
            "right", env, pool, SPLIT_FINAL_TEST, [EpisodeCase("b-0000", "t0", "HASH", 3, 4)]
        )
        with self.assertRaises(ValueError):
            assert_disjoint([left, right])


@unittest.skipUnless(HAS_METAWORLD, "metaworld is required to build task pools")
class TestTaskPool(unittest.TestCase):
    """Placement generation, the part that has to be pinned."""

    def test_same_seed_reproduces_the_same_placements(self):
        a = build_task_pool(ENV_ID, [2000], tasks_per_seed=5)
        b = build_task_pool(ENV_ID, [2000], tasks_per_seed=5)
        self.assertEqual(
            [e.task_hash for e in a.entries], [e.task_hash for e in b.entries]
        )
        self.assertEqual([e.rand_vec for e in a.entries], [e.rand_vec for e in b.entries])

    def test_different_seeds_give_different_placements(self):
        a = build_task_pool(ENV_ID, [2000], tasks_per_seed=5)
        b = build_task_pool(ENV_ID, [3000], tasks_per_seed=5)
        self.assertFalse(
            set(e.task_hash for e in a.entries) & set(e.task_hash for e in b.entries)
        )

    def test_no_pool_seeds_is_refused(self):
        with self.assertRaises(ValueError):
            build_task_pool(ENV_ID, [])

    def test_unknown_task_id_names_the_pool_it_is_missing_from(self):
        pool = build_task_pool(ENV_ID, [2000], tasks_per_seed=3)
        with self.assertRaises(KeyError):
            pool.entry("MT1:push-v3:s2000:999")

    def test_manifest_binds_to_the_pool_that_built_it(self):
        manifest, pool = build_manifest(
            ENV_ID, SPLIT_DEVELOPMENT, [2000], horizon=150, num_episodes=5, tasks_per_seed=5
        )
        for case in manifest.episodes:
            self.assertEqual(pool.entry(case.task_id).task_hash, case.task_hash)

    def test_resolve_pool_rejects_drifted_placements(self):
        manifest, _ = build_manifest(
            ENV_ID, SPLIT_DEVELOPMENT, [2000], horizon=150, num_episodes=3, tasks_per_seed=3
        )
        drifted = manifest.episodes[0]
        manifest.episodes[0] = EpisodeCase(
            drifted.case_id, drifted.task_id, "0" * 16, drifted.reset_seed, drifted.policy_seed
        )
        with self.assertRaises(ValueError) as ctx:
            manifest.resolve_pool()
        self.assertIn("drift", str(ctx.exception))

    def test_environment_spec_records_the_installed_success_rule(self):
        manifest, _ = build_manifest(
            ENV_ID, SPLIT_DEVELOPMENT, [2000], horizon=150, num_episodes=1, tasks_per_seed=1
        )
        env = manifest.environment
        self.assertEqual(env.env_id, "push-v2")
        self.assertEqual(env.env_id_v3, "push-v3")
        self.assertIn("TARGET_RADIUS", env.success_rule)
        self.assertNotEqual(env.version_or_commit, "unknown")


@unittest.skipUnless(HAS_METAWORLD, "metaworld is required to resolve committed manifests")
class TestCommittedManifests(unittest.TestCase):
    """The three splits checked into ``config/manifests`` are the contract."""

    @classmethod
    def setUpClass(cls):
        cls.manifests = {}
        for split in (SPLIT_TRAIN, SPLIT_DEVELOPMENT, SPLIT_FINAL_TEST):
            path = os.path.join(MANIFEST_DIR, f"{ENV_ID}_{split}.yaml")
            if not os.path.isfile(path):
                raise unittest.SkipTest(
                    f"missing {path}; run scripts/lock_baseline.py --build-manifests"
                )
            cls.manifests[split] = EvaluationManifest.load(path)

    def test_all_three_splits_are_present_and_non_empty(self):
        for split, manifest in self.manifests.items():
            self.assertEqual(manifest.split, split)
            self.assertGreater(len(manifest), 0)

    def test_splits_share_no_placement(self):
        assert_disjoint(list(self.manifests.values()))

    def test_final_test_has_enough_episodes_for_the_ac7_target(self):
        # AC-7 asks for at least 100 isolated final-test episodes.
        self.assertGreaterEqual(len(self.manifests[SPLIT_FINAL_TEST]), 100)

    def test_development_covers_the_screening_and_expanded_budgets(self):
        # spec.md section 12: a 10-case screen, then a 30-case expansion.
        self.assertGreaterEqual(len(self.manifests[SPLIT_DEVELOPMENT]), 30)

    def test_reset_and_policy_seed_streams_never_collide(self):
        resets, policies = set(), set()
        for manifest in self.manifests.values():
            for case in manifest.episodes:
                resets.add(case.reset_seed)
                policies.add(case.policy_seed)
        self.assertFalse(resets & policies)

    def test_every_placement_still_regenerates(self):
        # Verifies the recorded hashes against the installed metaworld build.
        self.manifests[SPLIT_DEVELOPMENT].resolve_pool()


if __name__ == "__main__":
    unittest.main()
