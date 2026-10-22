"""Correction-LLM repair and the clean ablation, against mocks (``debug_spec.md`` Stage F, D10).

The LLM and the subprocess harness are scripted; everything else is real:
generation with lineage, the in-process validator, objective-aware fitting with
rollout-selected checkpoints, the substitution study (MetaWorld's push expert
on a kinematic push world), evaluation, the ledgers and the chain files.

    python -m unittest tests.test_stage_f
"""

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

from lares.core import policy_generation as pg  # noqa: E402
from lares.core.obs_schema import get_obs_schema  # noqa: E402
from lares.core.training_pipeline import load_policy_prompt_assets  # noqa: E402
from lares.eval.manifest import synthetic_manifest  # noqa: E402
from lares.fitting.optimizers import prepare_data  # noqa: E402
from lares.fitting.structure_bank import instantiate_source  # noqa: E402
from lares.repair import correction as sf  # noqa: E402
from test_two_generation_mock import PushWorld, expert_buffer  # noqa: E402
from test_two_phase_policy_generation import (  # noqa: E402
    PARAMS_B,
    POLICY_B,
    ScriptedLLM,
    fence,
    idea_dict,
    proposal_dict,
)

SCHEMA = get_obs_schema("push-v2")
PARENT = "simple_parent"
#: A fault fitting cannot remove (the push gain is bounded positive): the push drives the puck away.
PARENT_SOURCE = POLICY_B.replace("contact * self.k_push * to_goal / dg", "contact * self.k_push * (-to_goal) / dg")
#: The repair: the push term restored, every declared parameter kept.
CHILD = POLICY_B
CHILD_RAW = CHILD.replace('tcp = self.obs_field(obs, "tcp")', "tcp = obs[:, 0:3]")

try:  # the substitution study drives MetaWorld's scripted push expert
    from lares.core.training_pipeline import get_expert_policy

    get_expert_policy("push-v2")
    HAVE_EXPERT = True
except Exception:  # noqa: BLE001
    HAVE_EXPERT = False


def proposals(mode="repair"):
    return json.dumps({"proposals": [
        {"proposal": proposal_dict(mode, [PARENT]), "idea": idea_dict("sharper-contact", PARAMS_B)}
    ]})


def stub_study(policy, structure, manifest, env, env_name, schema, max_steps=150, max_clusters=1,
               control_result=None, max_substitutions=None):
    """Two substitutions' worth of accounting with no expert, for machines without MetaWorld."""
    n = len(manifest)
    return {"structure_id": structure.structure_id, "manifest_id": manifest.manifest_id,
            "control": {"success_rate": control_result.success_rate, "mean_return": 0.0, "failure_labels": {}},
            "gates": [], "clusters_found": [], "studies": [], "rollout_episodes": 2 * n,
            "intervention_episodes": 2 * n, "intervention_environment_steps": 2 * n * 30, "control_reused": True}


class StageFCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        manifest = synthetic_manifest(20, label="push-v2")
        self.screen, self.probe, self.held_out = manifest.slice(0, 6), manifest.slice(6, 10), manifest.slice(10, 20)
        train = synthetic_manifest(8, label="push-v2-train").episodes
        self.data = prepare_data(expert_buffer(train), split_seed=0)
        self.budget = sf.StageFBudget(screen_episodes=6, probe_episodes=4, fit_budget=40,
                                      checkpoint_steps=(0, 20, 40), max_substitutions=2)
        self.env = PushWorld(horizon=30)

    def context(self, seed=0, study_fn=None):
        return sf.ArmContext(
            env=self.env, env_name="push-v2", schema=SCHEMA, data=self.data, screen=self.screen,
            probe=self.probe, budget=self.budget, seed=seed, max_steps=30, client=MagicMock(),
            llm_args=SimpleNamespace(model="stub-model", reasoning_effort="high"),
            prompts=load_policy_prompt_assets("push-v2"), param_budget=(2, 10),
            study_fn=study_fn if study_fn is not None else (None if HAVE_EXPERT else stub_study),
        )

    def build(self):
        return instantiate_source(PARENT_SOURCE, 39, 4, schema=SCHEMA)

    def run_replicate(self, replies, arms=sf.ARMS, seed=0, study_fn=None):
        llm = ScriptedLLM(replies)
        with patch.object(pg, "_call_llm", side_effect=llm), patch.object(
            pg, "_run_policy_validation_subprocess", return_value=(True, "Success!", "full")
        ):
            out = os.path.join(self.root, f"rep_{seed}")
            rep = sf.run_replicate(PARENT, PARENT_SOURCE, self.build, self.context(seed, study_fn), self.held_out,
                                   out, arms=arms)
        return rep, out, llm


class TestBudget(unittest.TestCase):
    def test_the_cap_is_the_most_an_llm_arm_can_spend(self):
        b = sf.StageFBudget(screen_episodes=10, probe_episodes=10, fit_budget=1000,
                            checkpoint_steps=(0, 500, 1000), max_substitutions=2)
        self.assertEqual(b.repair_episode_cap, 2 * 10 + 3 * 10 + 10)
        self.assertEqual(b.search_candidates, 6)

    def test_without_fitting_no_probe_is_charged(self):
        b = sf.StageFBudget(screen_episodes=10, probe_episodes=10, fit_budget=0, checkpoint_steps=())
        self.assertEqual(b.refit_probe_episodes, 0)
        self.assertEqual(b.repair_episode_cap, 30)

    def test_overlapping_streams_are_refused(self):
        m = synthetic_manifest(10, label="push-v2")
        with self.assertRaises(ValueError):
            sf.check_isolation(m.slice(0, 5), m.slice(4, 7), m.slice(7, 10))
        sf.check_isolation(m.slice(0, 4), m.slice(4, 7), m.slice(7, 10))

    def test_evidence_levels_nest(self):
        text = sf.build_evidence_text(sf.EVIDENCE_NONE, PARENT, POLICY_B)
        self.assertIn("No evaluation evidence", text)
        self.assertIn("k_reach", text)
        with self.assertRaises(ValueError):
            sf.build_evidence_text(sf.EVIDENCE_NUMERIC, PARENT, POLICY_B)

    def test_replicate_interval_is_a_t_interval_over_replicates(self):
        out = sf.replicate_interval([0.1, 0.2, 0.0, 0.1])
        self.assertAlmostEqual(out["mean_difference"], 0.1)
        self.assertLess(out["ci95_low"], 0.1)
        self.assertGreater(out["ci95_high"], 0.1)
        self.assertEqual((out["wins"], out["losses"], out["ties"]), (3, 0, 1))


class TestFullReplicate(StageFCase):
    """Six arms from one preprocessed parent: a corrected repair, a non-repair, a clean repair."""

    def setUp(self):
        super().setUp()
        replies = {
            "policy_proposals": [proposals(), proposals("simplification"), proposals()],
            # evidence-free: raw indexing, corrected; numeric: valid but not a repair; intervention: valid.
            "policy_implement": [fence(CHILD_RAW), fence(CHILD), fence(CHILD)],
            "policy_correction": [fence(CHILD)],
        }
        self.rep, self.out, self.llm = self.run_replicate(replies)
        self.rows = {r["arm"]: r for r in self.rep["rows"]}

    def test_every_arm_ran_from_the_same_preprocessed_parent(self):
        self.assertEqual(set(self.rows), set(sf.ARMS))
        prep = self.rep["common_preprocessing"]
        self.assertIn(prep["selected_step"], (0, 20, 40))
        hashes = {r["parent"]["preprocessing"]["parameters_sha256"] for r in self.rows.values()}
        self.assertEqual(hashes, {prep["parameters_sha256"]})
        self.assertEqual(self.rows[sf.ARM_NO_REPAIR]["final_parameters_sha256"], prep["parameters_sha256"])

    def test_no_arm_exceeds_the_matched_cap(self):
        cap = self.budget.repair_episode_cap
        for arm, row in self.rows.items():
            self.assertLessEqual(row["repair_episodes_spent"], cap, arm)
            self.assertEqual(row["repair_episode_cap"], cap)
        for arm in sf.SEARCH_ARMS:
            self.assertEqual(self.rows[arm]["repair_episodes_spent"], cap, arm)
            self.assertEqual(len(self.rows[arm]["candidates"]), self.budget.search_candidates)

    def test_only_llm_arms_spend_llm_calls_and_only_llm_children_are_refitted(self):
        for arm, row in self.rows.items():
            calls = row["resources"]["llm_calls"]
            if arm in sf.LLM_ARM_EVIDENCE:
                self.assertGreater(calls, 0, arm)
            else:
                self.assertEqual(calls, 0, arm)
                self.assertEqual(row["resources"]["bc_gradient_steps"], 0, arm)
        for arm in (sf.ARM_LLM_EVIDENCE_FREE, sf.ARM_LLM_INTERVENTION):
            row = self.rows[arm]
            self.assertEqual(row["resources"]["bc_gradient_steps"], self.budget.fit_budget)
            self.assertEqual(row["resources"]["probe_episodes"], self.budget.refit_probe_episodes)
            self.assertIn(row["child"]["preprocessing"]["selected_step"], (0, 20, 40))

    def test_each_llm_arm_is_shown_only_its_evidence_level(self):
        texts = {}
        for arm in (a for a in sf.LLM_ARM_EVIDENCE if a in self.rows):
            packet = self.rows[arm]["evidence_packet"]
            with open(os.path.join(self.out, arm, packet["text_path"]), encoding="utf-8") as f:
                texts[arm] = f.read()
        self.assertIn("No evaluation evidence", texts[sf.ARM_LLM_EVIDENCE_FREE])
        self.assertNotIn("Numeric evidence", texts[sf.ARM_LLM_EVIDENCE_FREE])
        self.assertIn("Numeric evidence", texts[sf.ARM_LLM_NUMERIC])
        self.assertNotIn("Controlled substitutions", texts[sf.ARM_LLM_NUMERIC])
        self.assertIn("Controlled substitutions", texts[sf.ARM_LLM_INTERVENTION])

    def test_only_the_intervention_arm_is_expert_assisted(self):
        for arm, row in self.rows.items():
            assisted = row["resources"]["expert_assisted_episodes"]
            if arm == sf.ARM_LLM_INTERVENTION:
                self.assertEqual(assisted, row["study"]["intervention_episodes"])
                if HAVE_EXPERT:  # the faulty parent fails in a cluster, so substitutions really run
                    self.assertGreater(assisted, 0)
                self.assertLessEqual(assisted, self.budget.max_substitutions * self.budget.screen_episodes)
            else:
                self.assertEqual(assisted, 0, arm)

    def test_the_corrected_attempt_links_to_the_failed_one(self):
        attempts = self.rows[sf.ARM_LLM_EVIDENCE_FREE]["attempts"]
        self.assertEqual(len(attempts), 2)
        self.assertFalse(attempts[0]["validation"]["passed"])
        self.assertTrue(attempts[1]["validation"]["passed"])
        self.assertEqual(attempts[1]["previous_attempt_id"], attempts[0]["attempt_id"])
        self.assertEqual(self.rows[sf.ARM_LLM_EVIDENCE_FREE]["resources"]["llm_calls"], 3)

    def test_a_proposal_that_is_not_a_repair_keeps_the_parent(self):
        row = self.rows[sf.ARM_LLM_NUMERIC]
        self.assertEqual(row["outcome"], sf.OUTCOME_NOT_A_REPAIR)
        self.assertEqual(row["final_parameters_sha256"], row["parent"]["preprocessing"]["parameters_sha256"])
        self.assertEqual(row["resources"]["bc_gradient_steps"], 0)

    def test_repair_chains_trace_to_parent_packet_and_ledger(self):
        for arm in (a for a in sf.LLM_ARM_EVIDENCE if a in self.rows):
            row = self.rows[arm]
            self.assertEqual(sf.verify_chain(row, os.path.join(self.out, arm)), [], arm)
            for attempt in row["attempts"]:
                lineage = attempt["lineage"]
                self.assertEqual(lineage["feedback_id"], row["evidence_packet"]["feedback_id"])
                self.assertEqual(lineage["parent_ids"], [PARENT])
        repaired = self.rows[sf.ARM_LLM_INTERVENTION]
        self.assertEqual(repaired["proposal"]["mode"], "repair")
        self.assertIn("as_predicted", repaired["prediction"])

    def test_a_tampered_packet_breaks_the_chain(self):
        row = self.rows[sf.ARM_LLM_NUMERIC]
        path = os.path.join(self.out, sf.ARM_LLM_NUMERIC, row["evidence_packet"]["text_path"])
        with open(path, "a", encoding="utf-8") as f:
            f.write("edited afterwards\n")
        self.assertTrue(sf.verify_chain(row, os.path.join(self.out, sf.ARM_LLM_NUMERIC)))

    def test_the_repair_fixes_the_fault_and_is_kept(self):
        parent = self.rows[sf.ARM_NO_REPAIR]["held_out"]["success"]
        repaired = self.rows[sf.ARM_LLM_INTERVENTION]
        self.assertEqual(repaired["outcome"], sf.OUTCOME_REPLACED)
        self.assertGreater(repaired["held_out"]["success"], parent)

    def test_replacement_follows_the_declared_rule(self):
        for arm, row in self.rows.items():
            parent = row["parent"]["screen"]
            if row["outcome"] != sf.OUTCOME_REPLACED:
                self.assertEqual(row["final_parameters_sha256"], row["parent"]["preprocessing"]["parameters_sha256"],
                                 arm)
                continue
            best = row["child"]["screen"] if "child" in row else row["best_screen"]
            self.assertGreater((best["success"], best["progress"]), (parent["success"], parent["progress"]), arm)

    def test_chain_files_and_replicate_summary_are_written(self):
        for arm in sf.ARMS:
            self.assertTrue(os.path.isfile(os.path.join(self.out, arm, "chain.json")), arm)
        with open(os.path.join(self.out, "replicate.json"), encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(saved["manifests"]["held_out"], self.held_out.manifest_id)
        summary = sf.summarise_replicates([saved])
        self.assertEqual(set(summary["paired"]), set(sf.ARMS) - {sf.ARM_NO_REPAIR})
        for arm, row in self.rows.items():
            self.assertEqual(len(row["held_out"]["per_case"]), len(self.held_out))


def stub_vision(policy, parent_id, control_result, ctx, run_dir):
    """Two media reruns and one fixed analysis block, with no renderer and no model."""
    return {"block": "Visual analysis: the gripper stops short of the puck in both episodes.",
            "run": {"model_id": "scripted", "episodes_analysed": 2, "episodes_rerun": 2, "frames_shown": 10,
                    "analyses_returned": 2, "analyses_rejected": 0, "conflicts": 0},
            "episodes_rerun": 2, "merged": []}


class TestVisionArm(StageFCase):
    """R-006: the vision arm is the numeric arm plus one section, charged for its media reruns."""

    ARMS = (sf.ARM_NO_REPAIR, sf.ARM_LLM_NUMERIC, sf.ARM_LLM_VISION)

    def context(self, seed=0, study_fn=None):
        ctx = super().context(seed, study_fn)
        ctx.vision_fn = stub_vision
        return ctx

    def setUp(self):
        super().setUp()
        replies = {"policy_proposals": [proposals(), proposals()],
                   "policy_implement": [fence(CHILD), fence(CHILD)], "policy_correction": []}
        self.rep, self.out, self.llm = self.run_replicate(replies, arms=self.ARMS)
        self.rows = {r["arm"]: r for r in self.rep["rows"]}

    def text(self, arm):
        packet = self.rows[arm]["evidence_packet"]
        with open(os.path.join(self.out, arm, packet["text_path"]), encoding="utf-8") as f:
            return f.read()

    def test_the_vision_packet_is_the_numeric_packet_plus_one_section(self):
        numeric, vision = self.text(sf.ARM_LLM_NUMERIC), self.text(sf.ARM_LLM_VISION)
        self.assertNotIn("Visual failure analysis", numeric)
        self.assertIn("Visual failure analysis", vision)
        self.assertIn("stops short of the puck", vision)
        self.assertTrue(vision.startswith(numeric.rstrip("\n")))
        self.assertEqual(self.rows[sf.ARM_LLM_VISION]["evidence_level"], sf.EVIDENCE_VISION)

    def test_media_reruns_are_charged_to_the_vision_arm_and_kept_under_the_cap(self):
        vision, numeric = self.rows[sf.ARM_LLM_VISION], self.rows[sf.ARM_LLM_NUMERIC]
        self.assertEqual(vision["resources"]["repair_episodes"] - numeric["resources"]["repair_episodes"], 2)
        self.assertLessEqual(vision["repair_episodes_spent"], self.budget.repair_episode_cap)
        self.assertEqual(vision["vision"]["analyses_returned"], 2)
        self.assertTrue(os.path.isfile(os.path.join(self.out, sf.ARM_LLM_VISION, "vision.json")))

    def test_the_vision_chain_verifies(self):
        self.assertEqual(sf.verify_chain(self.rows[sf.ARM_LLM_VISION], os.path.join(self.out, sf.ARM_LLM_VISION)), [])

    def test_the_vision_arm_without_a_vision_fn_is_refused(self):
        with self.assertRaises(ValueError):
            sf.build_evidence_text(sf.EVIDENCE_VISION, PARENT, PARENT_SOURCE, MagicMock(), MagicMock())


class TestGenerationFailure(StageFCase):
    def test_a_repair_that_never_validates_keeps_the_parent(self):
        replies = {"policy_proposals": [proposals()], "policy_implement": [fence(CHILD_RAW)],
                   "policy_correction": [fence(CHILD_RAW)]}
        rep, out, _ = self.run_replicate(replies, arms=(sf.ARM_NO_REPAIR, sf.ARM_LLM_EVIDENCE_FREE))
        row = {r["arm"]: r for r in rep["rows"]}[sf.ARM_LLM_EVIDENCE_FREE]
        self.assertEqual(row["outcome"], sf.OUTCOME_GENERATION_FAILED)
        self.assertEqual(row["resources"]["bc_gradient_steps"], 0)
        self.assertEqual(row["final_parameters_sha256"], rep["common_preprocessing"]["parameters_sha256"])
        self.assertEqual(sf.verify_chain(row, os.path.join(out, sf.ARM_LLM_EVIDENCE_FREE)), [])

    def test_an_unknown_arm_is_refused(self):
        with self.assertRaises(ValueError):
            self.run_replicate({}, arms=("optimizer_only",))


if __name__ == "__main__":
    unittest.main()


class TestVisionSummary(unittest.TestCase):
    """R-006's report: vision minus numeric paired per cell, case resampling, packet audits."""

    def test_pairs_within_cells_and_counts_packets(self):
        sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))
        import stage_f

        def per_case(rate):
            return {f"c{j}": float(j < round(rate * 10)) for j in range(10)}

        reps = []
        for seed, (v, n) in enumerate([(0.6, 0.5), (0.4, 0.4), (0.7, 0.6)]):
            reps.append({"structure_id": "s", "seed": seed, "rows": [
                {"arm": sf.ARM_LLM_VISION, "held_out": {"success": v, "per_case": per_case(v)},
                 "vision": {"analyses_returned": seed, "analyses_rejected": 1, "conflicts": 0, "episodes_rerun": 3}},
                {"arm": sf.ARM_LLM_NUMERIC, "held_out": {"success": n, "per_case": per_case(n)}},
                {"arm": sf.ARM_NO_REPAIR, "held_out": {"success": 0.3, "per_case": per_case(0.3)}}]})
        summary = stage_f.vision_summary(reps, resamples=200)
        self.assertAlmostEqual(summary["paired_vs_llm_numeric"]["mean_difference"], 0.2 / 3)
        self.assertEqual(summary["paired_vs_llm_numeric"]["n_replicates"], 3)
        self.assertEqual(summary["packets"]["with_valid_analysis"], 2)
        self.assertEqual(summary["packets"]["episodes_rerun"], 9)
        self.assertLessEqual(summary["held_out_case_bootstrap"]["low"], summary["held_out_case_bootstrap"]["high"])
