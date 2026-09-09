#!/usr/bin/env python
"""Tests for the method freeze and the final-test lock (FR-4, AC-7).

FR-4's last requirement is that winner selection and final estimation use
different data. These tests are what makes that a property of the code rather
than a promise in a document.

Run from the project root::

    python -m unittest tests.test_freeze
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

from lares.eval.freeze import (  # noqa: E402
    FinalTestLocked,
    FreezeRecord,
    final_test_session,
    hash_text,
    ledger_entries,
    load_freeze,
    record_final_test,
    session_is_open,
)
from lares.eval.manifest import (  # noqa: E402
    SPLIT_DEVELOPMENT,
    SPLIT_FINAL_TEST,
    EvaluationManifest,
    synthetic_manifest,
)
from lares.eval.runner import (  # noqa: E402
    ConstantActor,
    ManifestResult,
    evaluate_manifest,
)

OBS_DIM = 39
ACT_DIM = 4
MANIFEST_DIR = os.path.join(_PROJECT_ROOT, "config", "manifests")


class _Space:
    def __init__(self, dim):
        self.shape = (dim,)
        self.high = np.ones(dim, dtype=np.float32)
        self.low = -np.ones(dim, dtype=np.float32)


class MockEnv:
    def __init__(self, episode_length=3):
        self.observation_space = _Space(OBS_DIM)
        self.action_space = _Space(ACT_DIM)
        self._episode_length = episode_length
        self._step = 0

    def reset(self, case=None):
        if case is None:
            raise ValueError("case required")
        self._step = 0
        return np.zeros(OBS_DIM, dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        done = self._step >= self._episode_length
        return (np.zeros(OBS_DIM, dtype=np.float32), 1.0, done,
                {"success": 0.0, "obj_to_target": 0.2})


def a_record(**overrides):
    base = dict(
        freeze_id="t1",
        created_at="2026-01-01T00:00:00",
        models={"generation_model": "claude-opus-5"},
        prompts={"initial_system.txt": hash_text("a")},
        code={"lares/core/symbolic_policy.py": hash_text("b")},
        manifests={"push-v2_final_test.yaml": hash_text("c")},
        optimizer={"objective": "deterministic_mse"},
        selection={"screen_episodes": 10},
        budgets={"final_test_episodes": 100},
    )
    base.update(overrides)
    return FreezeRecord(**base)


def final_manifest(n=3):
    manifest = synthetic_manifest(n, label="push-v2")
    return EvaluationManifest(
        manifest_id="synthetic:final", environment=manifest.environment,
        pool=manifest.pool, split=SPLIT_FINAL_TEST, episodes=manifest.episodes,
    )


class TestFreezeRecord(unittest.TestCase):
    def test_a_record_missing_a_section_is_refused(self):
        for section in ("models", "prompts", "code", "manifests", "optimizer",
                        "selection", "budgets"):
            with self.subTest(section=section):
                with self.assertRaises(ValueError):
                    a_record(**{section: {}}).validate()

    def test_a_record_must_name_the_final_test_manifest_it_locks(self):
        with self.assertRaises(ValueError):
            a_record(manifests={"push-v2_development.yaml": "x"}).validate()

    def test_a_complete_record_validates(self):
        self.assertTrue(a_record().validate())

    def test_it_round_trips_through_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            a_record().save(tmp)
            self.assertEqual(load_freeze(tmp).to_dict(), a_record().to_dict())

    def test_a_freeze_is_written_once(self):
        # Rewriting it after a final-test number has been seen would erase what
        # the run was actually frozen against.
        with tempfile.TemporaryDirectory() as tmp:
            a_record().save(tmp)
            with self.assertRaises(FileExistsError):
                a_record().save(tmp)

    def test_a_missing_freeze_says_how_to_write_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as caught:
                load_freeze(tmp)
        self.assertIn("freeze_method.py", str(caught.exception))


class TestDriftDetection(unittest.TestCase):
    def test_an_unchanged_tree_matches(self):
        record = a_record()
        report = record.verify(prompts=dict(record.prompts), code=dict(record.code))
        self.assertTrue(report["matches"])

    def test_a_changed_file_is_named_not_merely_counted(self):
        record = a_record()
        report = record.verify(prompts={"initial_system.txt": hash_text("different")})
        self.assertFalse(report["matches"])
        self.assertEqual(report["drift"]["prompts"]["changed"], ["initial_system.txt"])

    def test_a_deleted_file_is_reported_as_missing(self):
        report = a_record().verify(prompts={})
        self.assertEqual(report["drift"]["prompts"]["missing"], ["initial_system.txt"])

    def test_a_new_file_is_reported_as_added(self):
        record = a_record()
        report = record.verify(prompts={**record.prompts, "new.txt": "z"})
        self.assertEqual(report["drift"]["prompts"]["added"], ["new.txt"])

    def test_a_section_that_is_not_supplied_is_not_judged(self):
        self.assertTrue(a_record().verify()["matches"])


class TestFinalTestLock(unittest.TestCase):
    """The property FR-4 asks for: held-out data cannot be looked at casually."""

    def test_the_committed_final_test_manifest_is_refused_by_default(self):
        path = os.path.join(MANIFEST_DIR, "push-v2_final_test.yaml")
        if not os.path.isfile(path):
            self.skipTest("committed manifests missing")
        manifest = EvaluationManifest.load(path)
        with self.assertRaises(FinalTestLocked):
            evaluate_manifest(ConstantActor(np.zeros(ACT_DIM)), manifest, MockEnv())

    def test_development_is_never_locked(self):
        manifest = synthetic_manifest(2, label="push-v2", split=SPLIT_DEVELOPMENT)
        result = evaluate_manifest(
            ConstantActor(np.zeros(ACT_DIM)), manifest, MockEnv(), max_steps=3
        )
        self.assertEqual(len(result.episodes), 2)

    def test_an_open_session_permits_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            with final_test_session(a_record(), tmp):
                result = evaluate_manifest(
                    ConstantActor(np.zeros(ACT_DIM)), final_manifest(), MockEnv(),
                    max_steps=3,
                )
        self.assertEqual(len(result.episodes), 3)

    def test_the_lock_closes_again_after_the_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            with final_test_session(a_record(), tmp):
                pass
        self.assertFalse(session_is_open())
        with self.assertRaises(FinalTestLocked):
            evaluate_manifest(ConstantActor(np.zeros(ACT_DIM)), final_manifest(), MockEnv())

    def test_the_lock_closes_even_when_the_block_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                with final_test_session(a_record(), tmp):
                    raise RuntimeError("boom")
        self.assertFalse(session_is_open())

    def test_two_overlapping_sessions_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with final_test_session(a_record(), tmp):
                with self.assertRaises(FinalTestLocked):
                    with final_test_session(a_record(), tmp):
                        pass

    def test_a_session_needs_a_complete_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                with final_test_session(a_record(models={}), tmp):
                    pass


class TestLedger(unittest.TestCase):
    """How often the held-out data was looked at is the number nobody records."""

    def _result(self, name="cand", successes=(1.0, 0.0)):
        from lares.eval.runner import EpisodeRecord

        episodes = [
            EpisodeRecord(
                case_id=f"c{i}", task_id="t", success=s, episode_return=1.0, length=3,
                terminal_obj_to_target=0.2, initial_obj_to_target=0.3,
                near_object_rate=0.0, first_near_object_step=None,
                grasp_success_rate=0.0, action_saturation_rate=0.0, timed_out=True,
            )
            for i, s in enumerate(successes)
        ]
        return ManifestResult(
            actor_name=name, manifest_id="synthetic:final",
            action_mode="deterministic", episodes=episodes,
        )

    def test_every_look_is_appended(self):
        with tempfile.TemporaryDirectory() as tmp:
            with final_test_session(a_record(), tmp):
                record_final_test(self._result("a"), code_hash="hash_a")
                record_final_test(self._result("b"), code_hash="hash_b")
            entries = ledger_entries(os.path.join(tmp, "final_test_ledger.jsonl"))
        self.assertEqual([e["key"] for e in entries], ["hash_a", "hash_b"])

    def test_the_entry_carries_what_a_reader_needs(self):
        with tempfile.TemporaryDirectory() as tmp:
            with final_test_session(a_record(), tmp):
                entry = record_final_test(self._result(), code_hash="h")
        for key in ("freeze_id", "manifest_id", "action_mode", "episodes",
                    "success_rate", "at"):
            self.assertIn(key, entry)
        self.assertEqual(entry["episodes"], 2)
        self.assertAlmostEqual(entry["success_rate"], 0.5)

    def test_scoring_the_same_code_twice_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with final_test_session(a_record(), tmp):
                record_final_test(self._result(), code_hash="h")
                with self.assertRaises(FinalTestLocked) as caught:
                    record_final_test(self._result(), code_hash="h")
        self.assertIn("already been scored", str(caught.exception))

    def test_a_deliberate_rescore_is_allowed_and_written_down(self):
        with tempfile.TemporaryDirectory() as tmp:
            with final_test_session(a_record(), tmp):
                record_final_test(self._result(), code_hash="h")
                entry = record_final_test(self._result(), code_hash="h", allow_rescore=True)
            entries = ledger_entries(os.path.join(tmp, "final_test_ledger.jsonl"))
        self.assertTrue(entry["rescore"])
        self.assertEqual(entry["previous_looks"], 1)
        self.assertEqual(len(entries), 2)

    def test_the_ledger_survives_the_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            with final_test_session(a_record(), tmp):
                record_final_test(self._result(), code_hash="h")
            with final_test_session(a_record(), tmp):
                with self.assertRaises(FinalTestLocked):
                    record_final_test(self._result(), code_hash="h")

    def test_recording_outside_a_session_is_refused(self):
        with self.assertRaises(FinalTestLocked):
            record_final_test(self._result(), code_hash="h")

    def test_an_empty_ledger_reads_as_no_looks(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(ledger_entries(os.path.join(tmp, "missing.jsonl")), [])


if __name__ == "__main__":
    unittest.main()
