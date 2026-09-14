"""Mocked two-generation run through the real orchestrator (``debug_spec.md`` Stage D, mocked).

Everything except the LLM, the subprocess harness and the simulator is real: the
orchestrator, generation, validation, objective-aware fitting, checkpoint
probes, screening, expanded evaluation, archive, records, ledger, feedback
bundles, reconstruction and the summary checks. The simulator is a small
kinematic push world laid out in the push-v2 observation schema, so geometry
diagnostics, failure labels and success are all computed.

The test then walks the Stage D acceptance list against the saved files alone.
Run from the project root::

    python -m unittest tests.test_two_generation_mock
"""

import copy
import io
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from lares.core import policy_generation as pg  # noqa: E402
from lares.core.obs_schema import get_obs_schema  # noqa: E402
from lares.core.training_pipeline import DemoBuffer, EvolutionOrchestrator  # noqa: E402
from lares.eval import PipelineEnvs, PromotionPolicy  # noqa: E402
from lares.eval.manifest import SPLIT_FINAL_TEST, synthetic_manifest  # noqa: E402
from lares.fitting.objectives import DETERMINISTIC_MSE  # noqa: E402
from lares.fitting.optimizers import state_parameters_sha256  # noqa: E402
from lares.search import (  # noqa: E402
    LINEAGE_COMPLETE,
    LINEAGE_FAILED_VALIDATION,
    STATUS_EVALUATED,
    STATUS_INVALID,
    STATUS_PROMOTED,
    STATUS_REJECTED,
    LLMCallLedger,
    SchemaError,
    load_records,
)
from lares.search.provenance import (  # noqa: E402
    lineage_status,
    load_tensor_artifact,
    reconcile_resources,
    reconstruct_record,
    traceability_gaps,
    traceability_summary,
)
from test_two_phase_policy_generation import (  # noqa: E402
    PARAMS_A,
    PARAMS_B,
    POLICY_A,
    POLICY_B,
    POLICY_RAW,
    ScriptedLLM,
    fence,
    idea_dict,
    proposal_dict,
)

SCHEMA = get_obs_schema("push-v2")
CHECKPOINTS = (0, 10, 20, 30, 40)

POLICY_C = POLICY_A.replace("w_move", "c_move").replace("w_goal", "c_goal").replace("grip_bias", "c_grip")
PARAMS_C = [("c_move", 3.0, 0.1, 10.0), ("c_goal", 1.0, 0.0, 5.0), ("c_grip", 0.0, -1.0, 1.0)]


class PushWorld:
    """Kinematic push in the push-v2 observation layout.

    The end effector moves 2 cm per unit action; the puck follows it in the plane
    while they are within 3 cm; success is the puck within 5 cm of the goal.
    """

    def __init__(self, horizon=30):
        self.horizon = horizon
        self.action_space = SimpleNamespace(shape=(4,), high=np.ones(4, dtype=np.float32), low=-np.ones(4, dtype=np.float32))
        self.observation_space = SimpleNamespace(shape=(39,))

    def reset(self, case=None):
        rng = np.random.default_rng(0 if case is None else case.reset_seed)
        self.tcp = np.array([0.0, 0.55, 0.05]) + np.r_[rng.uniform(-0.03, 0.03, 2), 0.0]
        self.obj = np.array([0.0, 0.62, 0.02]) + np.r_[rng.uniform(-0.02, 0.02, 2), 0.0]
        self.goal = np.array([0.03, 0.70, 0.02]) + np.r_[rng.uniform(-0.02, 0.02, 2), 0.0]
        self.t = 0
        return self._obs(), {}

    def _obs(self):
        obs = np.zeros(39, dtype=np.float32)
        for name, value in (("tcp", self.tcp), ("obj", self.obj), ("goal", self.goal)):
            f = SCHEMA.field(name)
            obs[f.start : f.stop] = value
        return obs

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        near = np.linalg.norm(self.tcp - self.obj) < 0.03
        self.tcp = self.tcp + 0.02 * a[:3]
        if near:
            self.obj[:2] = self.obj[:2] + 0.02 * a[:2]
        self.t += 1
        d = float(np.linalg.norm(self.obj - self.goal))
        info = {"success": float(d < 0.05), "obj_to_target": d, "near_object": float(near), "grasp_success": 0.0}
        return self._obs(), -d, self.t >= self.horizon, info

    def close(self):
        pass


def expert_buffer(cases, horizon=30):
    env = PushWorld(horizon)
    buffer = DemoBuffer()
    for case in cases:
        obs, _ = env.reset(case)
        for _ in range(horizon):
            target = env.obj - 0.02 * (env.goal - env.obj) / (np.linalg.norm(env.goal - env.obj) + 1e-8)
            action = np.clip(np.r_[10.0 * (target - env.tcp), 0.0], -1, 1)
            if np.linalg.norm(env.tcp - env.obj) < 0.025:
                action = np.clip(np.r_[10.0 * (env.goal - env.obj), 0.0], -1, 1)
            next_obs, reward, done, _ = env.step(action)
            buffer.add(obs, action, reward, next_obs, float(done), episode_id=case.case_id)
            obs = next_obs
    return buffer


def llm_script():
    ideas = json.dumps({"ideas": [idea_dict("A", PARAMS_A), idea_dict("B", PARAMS_B), idea_dict("C", PARAMS_C)]})
    proposals = json.dumps({"proposals": [
        {"proposal": proposal_dict("repair", ["gen0_cand0"]), "idea": idea_dict("A-repaired", PARAMS_A)},
        {"proposal": proposal_dict("crossover", ["gen0_cand0", "gen0_cand1"]), "idea": idea_dict("AB-cross", PARAMS_B)},
        {"proposal": proposal_dict("exploration", [], no_parent_rationale="no parent moves the puck toward the goal"),
         "idea": idea_dict("C-explore", PARAMS_C)},
    ]})
    return {
        "policy_ideas": [ideas],
        "policy_proposals": [proposals],
        # gen 0: slot 0 is corrected once; gen 1: slot 2 fails its correction too.
        "policy_implement": [fence(POLICY_RAW), fence(POLICY_B), fence(POLICY_C),
                             fence(POLICY_A), fence(POLICY_B), fence(POLICY_RAW)],
        "policy_correction": [fence(POLICY_A), fence(POLICY_RAW)],
    }


class TestTwoGenerationRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = cls.tmp.name
        # 4-case screen, 6-case promotion set, 3-case probe: three disjoint streams.
        manifest = synthetic_manifest(13, label="push-v2")
        cls.dev = manifest.head(10)
        cls.probe = manifest.tail(3)
        train = synthetic_manifest(10, label="push-v2-train").episodes
        cls.orchestrator = EvolutionOrchestrator(
            env_name="push-v2", obs_dim=39, action_dim=4, num_generations=2, pop_size=3, elite_num=2,
            bc_steps=40, log_dir=cls.root, record_demo_gif=False, dev_manifest=cls.dev,
            probe_manifest=cls.probe, eval_max_steps=30, checkpoint_steps=CHECKPOINTS, bc_batch_size=32,
            bc_lr=5e-2, promotion_policy=PromotionPolicy(screen_episodes=4, expanded_episodes=6),
        )
        envs = PipelineEnvs(development=PushWorld(), rl=PushWorld(), gif=PushWorld(), probe=PushWorld())
        args = SimpleNamespace(model="stub-model", policy_gen_two_phase=True, policy_impl_mode="per_idea")
        cls.llm = ScriptedLLM(llm_script())
        with patch.object(pg, "_call_llm", side_effect=cls.llm), patch.object(
            pg, "_run_policy_validation_subprocess", return_value=(True, "Success!", "full")
        ):
            cls.best = cls.orchestrator.run(MagicMock(), envs, expert_buffer(train), args)
        cls.records = load_records(os.path.join(cls.root, "experiments"))
        cls.by_id = {r.experiment_id: r for r in cls.records}
        cls.ledger = LLMCallLedger(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def evaluated(self, generation=None):
        return [r for r in self.records if r.status != STATUS_INVALID
                and (generation is None or r.generation == generation)]

    # -- identity and lineage -------------------------------------------------

    def test_every_attempt_has_a_unique_record(self):
        expected = {
            "gen0_cand0_a0", "gen0_cand0_a1", "gen0_cand1_a0", "gen0_cand2_a0",
            "gen1_cand0_a0", "gen1_cand1_a0", "gen1_cand2_a0", "gen1_cand2_a1",
        }
        self.assertEqual(set(self.by_id), expected)
        self.assertEqual(len(self.records), len(expected))
        for record in self.records:
            self.assertTrue(record.validate())

    def test_the_corrected_attempt_links_to_the_failed_one(self):
        failed, fixed = self.by_id["gen0_cand0_a0"], self.by_id["gen0_cand0_a1"]
        self.assertEqual(failed.status, STATUS_INVALID)
        self.assertEqual(fixed.lineage["previous_attempt_id"], "gen0_cand0_a0")
        correction = [c for c in fixed.llm["calls"] if c["label"] == "correction"]
        self.assertEqual(len(correction), 1)
        self.assertIn("superseded", failed.final_disposition)
        still_failed = self.by_id["gen1_cand2_a1"]
        self.assertEqual(still_failed.status, STATUS_INVALID)
        self.assertEqual(still_failed.lineage["status"], LINEAGE_FAILED_VALIDATION)

    def test_generation_zero_candidates_have_a_valid_idea_and_no_parent(self):
        for record in self.evaluated(0):
            self.assertTrue(record.idea_id.startswith("gen0_idea"))
            self.assertEqual(record.parent_ids, [])
            self.assertEqual(record.proposal_id, "")
            self.assertEqual(lineage_status(record), LINEAGE_COMPLETE)

    def test_generation_one_candidates_have_proposal_feedback_and_valid_parents(self):
        bundle = self.orchestrator.feedback_bundles[0]
        self.assertEqual(bundle.context_parent_ids, ["gen0_cand0", "gen0_cand1", "gen0_cand2"])
        modes = {}
        for record in self.evaluated(1):
            self.assertTrue(record.proposal_id.startswith("gen1_prop"))
            self.assertEqual(record.lineage["feedback_id"], bundle.feedback_id)
            self.assertEqual(record.lineage["context_parent_ids"], bundle.context_parent_ids)
            self.assertTrue(set(record.parent_ids) <= set(bundle.context_parent_ids))
            self.assertEqual(record.lineage["problems"], [])
            modes[record.lineage["mode"]] = record.parent_ids
        self.assertEqual(modes, {"repair": ["gen0_cand0"], "crossover": ["gen0_cand0", "gen0_cand1"]})

    # -- LLM provenance ---------------------------------------------------------

    def test_every_prompt_and_response_is_stored_and_hashed(self):
        self.assertEqual(self.ledger.verify(), {})
        self.assertEqual(len(self.ledger), len(self.llm.labels))
        for record in self.records:
            for call in record.llm["calls"]:
                self.assertEqual(call["prompt_sha256"], self.ledger.get(call["call_id"]).prompt_sha256)

    def test_token_totals_reconcile_on_unique_call_ids(self):
        llm = reconcile_resources(self.records, self.ledger, self.root)["llm"]
        self.assertTrue(llm["reconciles"], llm)
        self.assertEqual(llm["unreferenced_call_ids"], [])
        ideation = [c for c in self.ledger.records() if c.label == "ideation"]
        self.assertEqual(len(ideation), 2)
        self.assertEqual(llm["ledger_total_tokens"], 150 + 180 + 15 * (len(self.ledger) - 2))

    # -- fitting and checkpoints ------------------------------------------------

    def test_fitting_uses_the_named_objective_and_a_fixed_scale(self):
        for record in self.evaluated():
            self.assertEqual(record.fitting["objective"], DETERMINISTIC_MSE)
            self.assertEqual(record.fitting["action_mode"], "deterministic_tanh")
            self.assertEqual(record.fitting["fixed_action_scale"], 0.1)
            self.assertEqual(record.fitting["frozen_parameters"], [])
            self.assertNotIn("log_std", record.fitting["selected_diagnostics"]["parameters"])

    def test_step_zero_and_every_scheduled_checkpoint_is_probed(self):
        for record in self.evaluated():
            self.assertEqual([h["step"] for h in record.fitting["checkpoint_history"]], list(CHECKPOINTS))
            self.assertIn(record.fitting["selected_step"], CHECKPOINTS)
            self.assertEqual(record.resources["probe_episodes"], len(CHECKPOINTS) * len(self.probe))

    def test_selected_parameters_diagnostics_and_report_agree(self):
        for record in self.evaluated():
            state = load_tensor_artifact(self.root, record.fitting["selected_state"])
            digest = state_parameters_sha256(state)
            self.assertEqual(record.fitting["selected_diagnostics"]["parameters_sha256"], digest)
            step = record.fitting["selected_step"]
            probed = next(h for h in record.fitting["checkpoint_history"] if h["step"] == step)
            self.assertEqual(probed["parameters_sha256"], digest)
            with open(os.path.join(self.root, record.evaluation["reports"]["fitted_screen"]["path"])) as f:
                fitting = json.load(f)["fitting"]
            self.assertEqual(fitting["parameters_sha256"], digest)
            self.assertEqual(fitting["checkpoint_step"], step)
            self.assertAlmostEqual(fitting["train_loss_end"], record.fitting["selected_diagnostics"]["train_loss"])

    def test_probe_screen_and_expanded_cases_are_isolated(self):
        with open(os.path.join(self.root, "run_manifest.json")) as f:
            run = json.load(f)
        self.assertTrue(run["probe_is_isolated"])
        self.assertEqual(run["probe_stream"], "probe")
        probe_cases = set(run["manifests"]["probe"]["case_ids"])
        promotion_cases = set(run["manifests"]["promotion"]["case_ids"])
        screen_cases = set(run["manifests"]["screen"]["case_ids"])
        self.assertFalse(probe_cases & promotion_cases)
        self.assertFalse(screen_cases & promotion_cases)
        self.assertEqual(len(promotion_cases), 6)
        self.assertEqual(run["environment"]["resolved_registry_env_id"], "push-v3")
        for m in run["manifests"].values():
            self.assertNotEqual(m["split"], SPLIT_FINAL_TEST)

    # -- resources, archive, reconstruction ------------------------------------

    def test_simulator_and_gradient_work_are_separate(self):
        for record in self.evaluated():
            self.assertEqual(record.resources["bc_gradient_steps"], 40)
            self.assertNotIn("environment_steps", record.resources)
            self.assertEqual(record.resources["screen_episodes"], 3 * 4)
            self.assertEqual(record.resources["screen_environment_steps"], 3 * 4 * 30)
        episodes = reconcile_resources(self.records, self.ledger, self.root)["episodes"]
        self.assertTrue(episodes["reconciles"], episodes)

    def test_expanded_results_update_the_archive_and_the_records(self):
        expanded = [r for r in self.evaluated() if r.evaluation.get("expanded_success_rate") is not None]
        self.assertTrue(expanded)
        for record in expanded:
            self.assertIn(record.status, (STATUS_PROMOTED, STATUS_REJECTED))
            self.assertEqual(record.evaluation["archive"]["metrics_source"], "confirmed")
            entry = self.orchestrator.archive.get(record.candidate_id)
            self.assertAlmostEqual(entry.success_rate, record.evaluation["expanded_success_rate"])
            self.assertEqual(record.resources["expanded_episodes"], 6)
            self.assertTrue(os.path.isfile(os.path.join(self.root, "experiments", "history", f"{record.experiment_id}.v1.json")))

    def test_only_the_best_screened_challenger_is_tested_and_only_real_evidence_promotes(self):
        from lares.eval.promotion import PROMOTED_INTERVAL_ABOVE_ZERO

        promotion_id = self.orchestrator.promotion_manifest.manifest_id
        excluded = {self.orchestrator.screen_manifest.manifest_id, self.probe.manifest_id}
        for generation in (0, 1):
            tested = [r for r in self.evaluated(generation) if r.evaluation.get("incumbent_comparison")]
            self.assertEqual(len(tested), 1)
            comparison = tested[0].evaluation["incumbent_comparison"]
            self.assertEqual(comparison["evidence_manifest_id"], promotion_id)
            self.assertNotIn(comparison["evidence_manifest_id"], excluded)
            ps = comparison["paired_success"]
            if comparison["promoted"]:
                self.assertEqual(comparison["reason"], PROMOTED_INTERVAL_ABOVE_ZERO)
                self.assertGreater(ps["mean_difference"], 0.0)
                self.assertGreater(ps["ci_low"], 0.0)
                self.assertEqual(tested[0].status, STATUS_PROMOTED)
            else:
                self.assertEqual(tested[0].status, STATUS_REJECTED)
            for record in self.evaluated(generation):
                if record is not tested[0]:
                    self.assertIsNone(record.evaluation.get("expanded_success_rate"))

    def test_overlapping_promotion_evidence_is_refused(self):
        overlapping = EvolutionOrchestrator(
            env_name="push-v2", dev_manifest=self.dev, log_dir=self.root, probe_manifest=self.dev.slice(4, 7),
            promotion_policy=PromotionPolicy(screen_episodes=4, expanded_episodes=6),
        )
        with self.assertRaises(ValueError):
            overlapping.check_manifest_isolation()

    def test_candidates_from_both_generations_reconstruct_exactly(self):
        for generation in (0, 1):
            records = self.evaluated(generation)
            self.assertTrue(records)
            for record in records:
                outcome = reconstruct_record(record, self.root)
                self.assertTrue(outcome["identical"], (record.experiment_id, outcome))
                self.assertEqual(record.reconstruction["status"], "verified")

    # -- summaries ----------------------------------------------------------------

    def test_the_summary_reports_complete_traceability_only_when_it_is(self):
        summary = traceability_summary(self.records, self.root)
        for name, field in summary.items():
            self.assertTrue(field["complete"], (name, field["gaps"]))

    def test_the_summary_script_regenerates_from_saved_files(self):
        import summarise_experiments  # noqa: F401  (scripts/ is on sys.path below)

        out = os.path.join(self.root, "summary.json")
        with patch.object(sys, "argv", ["summarise_experiments.py", "--log-dir", self.root, "--out", out]):
            self.assertEqual(summarise_experiments.main(), 0)
        with open(out) as f:
            summary = json.load(f)
        self.assertEqual(summary["attempts"], 8)
        self.assertEqual(summary["slots"], 6)
        self.assertEqual(summary["validated_attempts"], 5)
        self.assertEqual(summary["corrections"]["attempted"], 2)
        self.assertEqual(summary["corrections"]["succeeded"], 1)
        self.assertTrue(all(v["complete"] for v in summary["traceability"].values()))
        self.assertEqual(summary["lineage_status"]["complete"], 5)

    def test_missing_evidence_breaks_traceability_and_validation(self):
        original = self.evaluated(1)[0]
        for mutate, field in (
            (lambda r: r.fitting.pop("selected_state"), "state_artifacts"),
            (lambda r: r.llm.update(calls=[]), "prompt_hashes"),
            (lambda r: r.resources.pop("probe_environment_steps"), "resources"),
        ):
            record = copy.deepcopy(original)
            mutate(record)
            with self.assertRaises(SchemaError):
                record.validate()
            self.assertTrue(traceability_gaps(record, self.root, self.ledger, verify_reconstruction=False)[field])

    def test_a_deleted_state_file_is_not_reported_complete(self):
        record = self.evaluated(0)[0]
        path = os.path.join(self.root, record.fitting["final_state"]["path"])
        saved = open(path, "rb").read()
        os.remove(path)
        try:
            gaps = traceability_gaps(record, self.root, self.ledger, verify_reconstruction=False)
            self.assertTrue(gaps["state_artifacts"])
        finally:
            with open(path, "wb") as f:
                f.write(saved)

    def test_the_orchestrator_refuses_what_it_cannot_trace(self):
        with self.assertRaises(ValueError):
            EvolutionOrchestrator(env_name="push-v2", rl_iterations=3)
        orchestrator = EvolutionOrchestrator(env_name="push-v2", dev_manifest=self.dev, log_dir=self.root)
        envs = PipelineEnvs(development=PushWorld(), rl=PushWorld(), gif=PushWorld(), probe=PushWorld())
        with self.assertRaises(ValueError):
            orchestrator.run(MagicMock(), envs, expert_buffer(self.dev.episodes[:2]),
                             SimpleNamespace(model="m", policy_gen_two_phase=False))
        no_probe = PipelineEnvs(development=PushWorld(), rl=PushWorld(), gif=PushWorld())
        with self.assertRaises(ValueError):
            orchestrator.run(MagicMock(), no_probe, expert_buffer(self.dev.episodes[:2]),
                             SimpleNamespace(model="m", policy_gen_two_phase=True))


sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))

if __name__ == "__main__":
    unittest.main()
