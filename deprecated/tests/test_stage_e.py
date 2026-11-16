"""Stage E protocol pieces that must hold before any Stage E case exists.

Run from the project root::

    python -m unittest tests.test_stage_e
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))

import stage_e  # noqa: E402
from lares.eval.manifest import TaskPoolSpec, synthetic_manifest  # noqa: E402


class TestDeduplication(unittest.TestCase):
    def test_identical_source_is_dropped_in_favour_of_the_higher_rank(self):
        candidates = [{"id": "a", "source_sha256": "x"}, {"id": "b", "source_sha256": "x"}, {"id": "c", "source_sha256": "y"}]
        kept, dropped = stage_e.deduplicate(candidates, lambda a, b: False, lambda a, b: False)
        self.assertEqual([k["id"] for k in kept], ["a", "c"])
        self.assertEqual(dropped[0]["duplicate_of"], "a")
        self.assertEqual(dropped[0]["reason"], "identical source")

    def test_identical_behaviour_needs_both_actions_and_outcomes(self):
        candidates = [{"id": "a", "source_sha256": "x"}, {"id": "b", "source_sha256": "y"}]
        kept, _ = stage_e.deduplicate(candidates, lambda a, b: True, lambda a, b: False)
        self.assertEqual(len(kept), 2)
        kept, dropped = stage_e.deduplicate(candidates, lambda a, b: True, lambda a, b: True)
        self.assertEqual([k["id"] for k in kept], ["a"])
        self.assertIn("identical actions", dropped[0]["reason"])

    def test_development_behaviour_compares_success_and_return_per_case(self):
        a = {"fitted_screen": {"c0": (1.0, 100.0), "c1": (0.0, 20.0)}}
        self.assertTrue(stage_e.same_development_behaviour(a, {"fitted_screen": {"c0": (1.0, 100.0), "c1": (0.0, 20.0)}}))
        self.assertFalse(stage_e.same_development_behaviour(a, {"fitted_screen": {"c0": (1.0, 100.0), "c1": (0.0, 20.5)}}))
        self.assertFalse(stage_e.same_development_behaviour(a, {"other": {"c0": (1.0, 100.0)}}))


class TestSeedLedger(unittest.TestCase):
    def test_a_seed_in_the_ledger_or_a_saved_manifest_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = os.path.join(tmp, "ledger.jsonl")
            stage_e.reserve_seeds([{"seed": 2203, "namespace": "fitting", "use": "earlier study"}], ledger)
            manifest = synthetic_manifest(3, label="push-v2")
            manifest.pool = TaskPoolSpec("MT1", "push-v2", (2200,), 50)
            manifest.save(os.path.join(tmp, "m.yaml"))
            conflicts = stage_e.seed_conflicts([2200], [2201, 2203], ledger, [os.path.join(tmp, "*.yaml")])
            self.assertEqual(len(conflicts), 2)
            self.assertTrue(any("2203" in c for c in conflicts))
            self.assertTrue(any("pool seed" in c for c in conflicts))
            self.assertEqual(stage_e.seed_conflicts([2300], [2301], ledger, [os.path.join(tmp, "*.yaml")]), [])

    def test_saved_copies_of_one_manifest_count_once_but_a_conflicting_id_is_refused(self):
        a = synthetic_manifest(4, label="push-v2")
        copy_of_a = synthetic_manifest(4, label="push-v2")
        other = synthetic_manifest(5, label="push-v2")
        unique = stage_e.unique_manifests([a, copy_of_a, other])
        self.assertEqual([m.manifest_id for m in unique], [a.manifest_id, other.manifest_id])
        clash = synthetic_manifest(4, label="push-v2")
        from dataclasses import replace

        clash.episodes = [replace(clash.episodes[0], task_hash="different")] + clash.episodes[1:]
        with self.assertRaises(SystemExit):
            stage_e.unique_manifests([a, clash])

    def test_reservations_are_appended_not_rewritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = os.path.join(tmp, "ledger.jsonl")
            stage_e.reserve_seeds([{"seed": 1, "namespace": "fitting", "use": "a"}], ledger)
            stage_e.reserve_seeds([{"seed": 2, "namespace": "fitting", "use": "b"}], ledger)
            self.assertEqual([e["seed"] for e in stage_e.read_ledger(ledger)], [1, 2])

    def test_the_committed_ledger_does_not_already_hold_the_stage_e_seeds(self):
        entries = [e for e in stage_e.read_ledger() if not e.get("backfilled", True)]
        used = {int(e["seed"]) for e in stage_e.read_ledger() if e.get("backfilled")}
        self.assertFalse(used & {2200, 2201, 2202, 2203, 2204})
        self.assertIsInstance(entries, list)


class TestCleanWorktree(unittest.TestCase):
    def test_uncommitted_work_is_refused_and_a_clean_tree_returns_its_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = lambda *a: subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", *a], cwd=tmp,  # noqa: E731
                                            check=True, capture_output=True)
            run("init", "-q")
            with open(os.path.join(tmp, "a.txt"), "w") as f:
                f.write("x")
            run("add", "a.txt")
            run("commit", "-qm", "init")
            sha = stage_e.require_clean_worktree(tmp)
            self.assertEqual(len(sha), 40)
            with open(os.path.join(tmp, "b.txt"), "w") as f:
                f.write("y")
            with self.assertRaises(SystemExit):
                stage_e.require_clean_worktree(tmp)


def structure_result(sid, family, zero, fitted, seeds=(1, 2, 3)):
    """A minimal structure result: zero-shot success and per-condition success per seed."""
    outcomes = {"z": {"success_rate": zero}}
    seed_entries = []
    for i, seed in enumerate(seeds):
        conditions = {stage_e.ZERO_SHOT: {"parameters_sha256": "z", "selected_step": 0}}
        for c in stage_e.FITTED_CONDITIONS:
            key = f"{c}-{seed}"
            outcomes[key] = {"success_rate": fitted[c][i]}
            conditions[c] = {"parameters_sha256": key, "selected_step": 2000}
        seed_entries.append({"seed": seed, "conditions": conditions})
    return {"structure_id": sid, "family": family, "outcomes_by_parameters": outcomes, "seeds": seed_entries,
            "git_sha": "0" * 40, "resources": {}}


class TestDecisionRule(unittest.TestCase):
    def _results(self, lift):
        out = []
        for k in range(6):
            zero = 0.2 + 0.05 * k
            fitted = {
                stage_e.MSE_FIXED: [zero - 0.2] * 3,
                stage_e.MSE_ROLLOUT: [zero + lift + 0.01 * k, zero + lift, zero + lift - 0.01 * k],
                stage_e.PHASE_ROLLOUT: [zero] * 3,
            }
            out.append(structure_result(f"s{k}", stage_e.FAMILY_BANK if k < 3 else stage_e.FAMILY_EVOLVED, zero, fitted))
        return out

    def test_fitting_is_kept_only_when_an_interval_clears_zero(self):
        analysis = stage_e.analyse(self._results(lift=0.1))
        self.assertTrue(analysis["decision"]["keep_fitting"])
        self.assertEqual(analysis["decision"]["fitted_conditions_beating_zero_shot"], [stage_e.MSE_ROLLOUT])
        self.assertGreater(analysis["conditions"][stage_e.MSE_ROLLOUT]["structure_level"]["low"], 0.0)

    def test_otherwise_fitting_is_disabled_by_default(self):
        analysis = stage_e.analyse(self._results(lift=0.0))
        self.assertFalse(analysis["decision"]["keep_fitting"])
        self.assertIn("disabled", analysis["decision"]["default_after_stage_e"])

    def test_structure_is_the_unit_and_families_are_reported(self):
        analysis = stage_e.analyse(self._results(lift=0.1))
        level = analysis["conditions"][stage_e.MSE_FIXED]["structure_level"]
        self.assertEqual(level["n"], 6)
        self.assertAlmostEqual(level["mean"], -0.2)
        self.assertEqual(set(analysis["families"]), {stage_e.FAMILY_BANK, stage_e.FAMILY_EVOLVED})
        self.assertIn("holm", analysis)
        self.assertEqual(sorted(h["rank"] for h in analysis["holm"].values()), [1, 2, 3])


class TestFrozenConfig(unittest.TestCase):
    def test_the_frozen_protocol_names_every_condition_and_the_decision_rule(self):
        if not os.path.isfile(stage_e.CONFIG_PATH):
            self.skipTest("config/stage_e/stage_e.yaml not written yet")
        cfg = stage_e.load_config()
        self.assertEqual([c["name"] for c in cfg["conditions"]], list(stage_e.CONDITIONS))
        self.assertEqual(cfg["seeds"]["confirmation_pool_seed"], 2200)
        self.assertEqual(cfg["seeds"]["fitting_seeds"], [2200, 2201, 2202, 2203, 2204])
        self.assertEqual(cfg["fitting"]["checkpoint_steps"], [0, 500, 1000, 1500, 2000])
        self.assertNotEqual(cfg["checkpoint_selection_manifest"]["path"], cfg["confirmation_manifest"]["path"])
        self.assertEqual(cfg["structures"]["bank"], ["simple_reach_push", "simple_damped_push", "simple_standoff"])


if __name__ == "__main__":
    unittest.main()
