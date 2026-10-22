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


class TestSearchVsFit(unittest.TestCase):
    """R-008: the fixed search order and the placement-averaged statistic."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))
        import stage_f

        cls.stage_f = stage_f

    def test_candidates_follow_the_declared_order_and_stay_in_range(self):
        import numpy as np
        import torch

        original = instantiate_source(PARENT_SOURCE, 39, 4, schema=SCHEMA)
        cands = self.stage_f.search_candidates(original, np.random.default_rng(2812))
        self.assertEqual([k for k, _ in cands], list(self.stage_f.SEARCH_ORDER))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(cands[0][1].parameters(), original.parameters())))
        ranges = original.get_param_ranges()
        for _, cand in cands[1:]:
            for name, p in cand.named_parameters():
                if name in ranges:
                    self.assertTrue(float(p.min()) >= ranges[name][0] - 1e-6 and float(p.max()) <= ranges[name][1] + 1e-6)
        again = self.stage_f.search_candidates(original, np.random.default_rng(2812))
        self.assertTrue(all(torch.equal(a, b) for (_, x), (_, y) in zip(cands, again)
                            for a, b in zip(x.parameters(), y.parameters())))

    def test_the_statistic_averages_structures_within_each_placement(self):
        def row(sid, search, fitting):
            return {"structure_id": sid, "episodes": {"search_selection": 80, "fitting_selection": 80, "confirmation": 100},
                    "search": {"success_rate": sum(search.values()) / len(search), "per_case": search,
                               "selected_kind": "global"},
                    "fitting": {"success_rate": sum(fitting.values()) / len(fitting), "per_case": fitting,
                                "selected_step": 500}}

        results = [row("a", {"c0": 1.0, "c1": 0.0}, {"c0": 0.0, "c1": 0.0}),
                   row("b", {"c0": 1.0, "c1": 1.0}, {"c0": 1.0, "c1": 0.0})]
        summary = self.stage_f.search_vs_fit_summary(results)
        self.assertAlmostEqual(summary["paired_success_difference"], 0.5)   # c0: (1+0)/2, c1: (0+1)/2
        self.assertEqual(summary["primary"]["n_placements"], 2)
        self.assertEqual(summary["episodes"]["confirmation"], 200)


class TestPooledOptimizer(unittest.TestCase):
    """R-011: nested candidate sets, one tie rule, the paired summary and the frozen subject rule."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))
        import stage_f

        cls.stage_f = stage_f

    def test_orders_cover_fifteen_candidates_and_pooled_is_a_subset_of_both_controls(self):
        sf_ = self.stage_f
        for order in (sf_.POOLED_ORDER, sf_.SEARCH_CONTROL_ORDER, sf_.IMITATION_CONTROL_ORDER):
            self.assertEqual(len(order), 15)
            self.assertEqual(len(set(order)), 15)
            self.assertEqual(order[0], "original")
        self.assertTrue(set(sf_.POOLED_ORDER) <= set(sf_.SEARCH_CONTROL_ORDER) | set(sf_.IMITATION_CONTROL_ORDER))
        self.assertEqual(len(set(sf_.SEARCH_CONTROL_ORDER) | set(sf_.IMITATION_CONTROL_ORDER)), 29)

    def test_search_draws_are_reproducible_per_stream(self):
        import torch

        original = instantiate_source(PARENT_SOURCE, 39, 4, schema=SCHEMA)
        a = self.stage_f.pooled_search_draws(original, 2910)
        b = self.stage_f.pooled_search_draws(original, 2910)
        self.assertEqual(sorted(a), sorted([f"L{i}" for i in range(1, 7)] + [f"G{i}" for i in range(1, 9)]))
        for label in a:
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(a[label].parameters(), b[label].parameters())))

    def test_ties_go_to_the_earlier_label_in_each_selectors_order(self):
        from lares.fitting.checkpoints import RolloutScore

        scores = {"original": RolloutScore(0.5, 0.1), "L1": RolloutScore(0.5, 0.1), "fit125": RolloutScore(0.4, 0.9)}
        self.assertEqual(self.stage_f.select_by_order(scores, ("original", "L1", "fit125")), "original")
        self.assertEqual(self.stage_f.select_by_order(scores, ("fit125", "L1")), "L1")

    def test_summary_pairs_within_placements_and_takes_the_weaker_comparison(self):
        def row(sid, group, pooled, search, imitation):
            confirmed = {"P": {"per_case": pooled, "success_rate": 0}, "S": {"per_case": search, "success_rate": 0},
                         "I": {"per_case": imitation, "success_rate": 0}}
            return {"structure_id": sid, "group": group, "confirmed": confirmed,
                    "selected": {"pooled": "P", "search_only": "S", "imitation_only": "I"},
                    "selector_success": {"pooled": 1.0, "search_only": 0.5, "imitation_only": 0.5},
                    "episodes": {"probe": 290, "confirmation": 150}}

        one = {"a": 1.0, "b": 1.0}
        zero = {"a": 0.0, "b": 0.0}
        results = [row("x", "anchor", one, zero, one), row("y", "additional", one, one, zero)]
        out = self.stage_f.pooled_summary(results)
        self.assertAlmostEqual(out["comparisons"]["search_only"]["mean"], 0.5)
        self.assertAlmostEqual(out["comparisons"]["imitation_only"]["mean"], 0.5)
        self.assertAlmostEqual(out["paired_success_difference"], 0.5)
        self.assertEqual(out["episodes"], {"probe": 580, "confirmation": 300})
        self.assertIn("holm_p_value", out["comparisons"]["search_only"])

    def test_additional_subjects_follow_the_frozen_rule_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "experiments"))
            recs = [("gen0_cand0_a0", 0.0, "evaluated"), ("gen0_cand1_a0", 0.0, "evaluated"),
                    ("gen0_cand2_a0", 0.6, "evaluated"), ("gen0_cand3_a0", 0.2, "invalid"), ("gen0_cand4_a0", 0.3, "evaluated")]
            for eid, success, status in recs:
                with open(os.path.join(d, "experiments", f"{eid}.json"), "w") as f:
                    json.dump({"experiment_id": eid, "status": status, "code": {"source_path": f"{eid}.py"},
                               "evaluation": {"success_rate": success}, "reconstruction": {"status": "verified"}}, f)
            picked = self.stage_f.additional_subjects(d, {"gen0_cand1_a0"})
            self.assertEqual([r["experiment_id"] for r in picked], ["gen0_cand0_a0", "gen0_cand4_a0"])
            with self.assertRaises(SystemExit):
                self.stage_f.additional_subjects(d, {"gen0_cand0_a0", "gen0_cand1_a0"})


class TestMutationScopeTransfer(unittest.TestCase):
    """R-012: a fittable-only, code-only plan; generation with no exploration lane; the ITT summary."""

    PARENTS = ["simple_reach_push_sharp_gate", "simple_reach_push_shared_gain"]

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))
        import stage_f

        cls.stage_f = stage_f
        cls.plan = stage_f.transfer_plan(cls.PARENTS, 1)

    def test_the_plan_is_fittable_only_code_only_and_ordered(self):
        slots = self.plan.slots
        self.assertEqual(len(slots), 4)
        self.assertEqual([(s.reference.structure_id, s.scope.scope_id) for s in slots],
                         [(p, sc) for p in self.PARENTS for sc in (self.stage_f.SCOPE_GEOMETRY, self.stage_f.SCOPE_CONTROL)])
        self.assertEqual(self.plan.exploration_slots, [])
        for s in slots:
            self.assertEqual((s.scope.may_change, s.scope.may_remove, s.scope.allow_adds), ((), (), False))

    def test_generation_needs_no_exploration_call_and_enforces_code_only_children(self):
        import re

        from test_two_lane_search import idea_for, mutation_entry  # noqa: E402

        def strip(source):
            out = re.sub(r"\n\s*self\.log_std = nn\.Parameter\(torch\.tensor\([^)]*\)\)", "", source)
            out = re.sub(r'\n\s*"log_std": \([^)]*\),', "", out)
            return out.replace("std = torch.exp(self.log_std) * torch.ones_like(mean)", "std = self.fixed_std(mean)")

        geometry = ("to_obj = obj - tcp", "to_obj = obj - 0.02 * (goal - obj) / (torch.norm(goal - obj, dim=-1, "
                    "keepdim=True) + 1e-8) - tcp")
        control = ("(to_goal / d_goal)", "(to_goal / d_goal) * torch.clamp(d_goal / 0.05, max=1.0)")
        kids = {}
        for s in self.plan.slots:
            old, new = geometry if s.scope.scope_id == self.stage_f.SCOPE_GEOMETRY else control
            kids[s.slot] = (strip(s.reference.source).replace(old, new, 1), {}, f"{s.scope.scope_id} code-only edit")
        # slot 3 smuggles in a changed initial value, which the code-only scope refuses
        bad = kids[3][0].replace("torch.tensor(-0.5)", "torch.tensor(-0.4)")
        mutations = [mutation_entry(s, kids[s.slot]) for s in self.plan.slots]
        script = {"policy_fittable_mutations": [json.dumps({"mutations": mutations})],
                  "policy_implement": [fence(kids[0][0]), fence(kids[1][0]), fence(kids[2][0]), fence(bad)]}
        llm = ScriptedLLM(script)
        with tempfile.TemporaryDirectory() as d, patch.object(pg, "_call_llm", side_effect=llm), patch.object(
                pg, "_run_policy_validation_subprocess", return_value=(True, "Success!", "full")):
            import pickle

            import numpy as np

            from lares.search.generation import LLMCallLedger

            pkl = os.path.join(d, "data.pkl")
            with open(pkl, "wb") as f:
                pickle.dump([{"obs": np.zeros(39)}], f)
            cands = pg.generate_two_lane_candidates(
                MagicMock(), SimpleNamespace(model="stub", reasoning_effort="high"), d, 39, 4, "push-v2",
                load_policy_prompt_assets("push-v2"), self.plan, pkl, LLMCallLedger(d),
                max_total_calls=self.plan.n + 1, max_repair_per_candidate=0)
        self.assertEqual([c.valid for c in cands], [True, True, True, False])
        self.assertIn("changes", " ".join(cands[3].validation["errors"]))
        self.assertNotIn("policy_ideas", [c.get("label") for c in getattr(llm, "calls", [])])

    def test_invalid_children_score_as_their_parent_and_contrasts_are_matched(self):
        def res(rate_a, rate_b, step=1000):
            return {"confirmation": {"per_case": {"a": [rate_a, 0], "b": [rate_b, 0]}}, "selected_step": step}

        children = [
            {"attempt_id": "g0", "parent": "P", "scope": self.stage_f.SCOPE_GEOMETRY, "draw": 0, "valid": True},
            {"attempt_id": "c0", "parent": "P", "scope": self.stage_f.SCOPE_CONTROL, "draw": 0, "valid": False},
        ]
        results = {("P", 1): res(0.0, 1.0), ("g0", 1): res(1.0, 1.0)}
        out = self.stage_f.transfer_summary(children, results, [1])
        self.assertAlmostEqual(out["geometry_minus_control"]["mean"], 0.5)   # control falls back to the parent
        self.assertAlmostEqual(out["control_minus_parent"]["mean"], 0.0)
        self.assertEqual(out["validity"], {"valid": 1, "total": 2, "invalid": [
            {"attempt_id": "c0", "scope": self.stage_f.SCOPE_CONTROL, "stage": None, "errors": None}]})


class TestBoundsOnly(unittest.TestCase):
    """R-009: narrowed ranges only, validated strictly; random controls matched in width; clipping honours them."""

    RANGES = {"a": (0.0, 10.0), "b": (-1.0, 1.0), "c": (0.0, 1.0)}
    INITIAL = {"a": 3.0, "b": 0.0, "c": 0.5}

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))
        import stage_f

        cls.stage_f = stage_f

    def test_valid_narrowing_is_accepted(self):
        edits, problems = self.stage_f.validate_bound_edits(
            {"edits": [{"parameter": "a", "range": [2.0, 5.0]}, {"parameter": "b", "range": [-0.5, 0.5]}]},
            self.INITIAL, self.RANGES)
        self.assertEqual((edits, problems), ({"a": (2.0, 5.0), "b": (-0.5, 0.5)}, []))

    def test_any_bad_edit_makes_the_reply_a_no_op(self):
        for payload in ({"edits": [{"parameter": "a", "range": [4.0, 5.0]}]},          # excludes the initial value
                        {"edits": [{"parameter": "a", "range": [-1.0, 5.0]}]},         # leaves the original range
                        {"edits": [{"parameter": "a", "range": [0.0, 10.0]}]},         # does not narrow
                        {"edits": [{"parameter": "z", "range": [0.0, 1.0]}]},          # unknown parameter
                        {"edits": [{"parameter": n, "range": [self.INITIAL[n] - 0.1, self.INITIAL[n] + 0.1]}
                                   for n in ("a", "b", "c")]},                          # more than two
                        None):
            edits, problems = self.stage_f.validate_bound_edits(payload, self.INITIAL, self.RANGES)
            self.assertEqual(edits, {})
            self.assertTrue(problems)

    def test_random_bounds_match_width_and_contain_the_initial_value(self):
        import numpy as np

        edits = {"a": (2.0, 5.0)}
        for seed in range(20):
            out = self.stage_f.random_matched_bounds(edits, self.INITIAL, self.RANGES, np.random.default_rng(seed))
            lo, hi = out["a"]
            self.assertAlmostEqual(hi - lo, 3.0)
            self.assertTrue(0.0 <= lo <= 3.0 <= hi <= 10.0)

    def test_overridden_ranges_are_what_clipping_enforces(self):
        import torch

        policy = instantiate_source(PARENT_SOURCE, 39, 4, schema=SCHEMA)
        name = next(n for n in policy.get_param_ranges() if n != "log_std")
        value = float(dict(policy.named_parameters())[name].detach().reshape(-1)[0])
        self.stage_f.with_ranges(policy, {name: (value - 1e-3, value + 1e-3)})
        with torch.no_grad():
            dict(policy.named_parameters())[name].add_(5.0)
        policy.clip_params()
        self.assertAlmostEqual(float(dict(policy.named_parameters())[name].reshape(-1)[0]), value + 1e-3, places=5)

    def test_summary_contrasts_average_cells_within_placements(self):
        cells = [{"parent": "P", "fit_seed": s, "edits": {}, "random_bounds": {}, "edit_problems": [],
                  "selected_steps": {}, "confirmation": {"unedited": {"x": 0.0, "y": 0.0}, "llm": {"x": 1.0, "y": 0.0},
                                                         "random": {"x": 0.0, "y": 0.0}}} for s in (1, 2)]
        out = self.stage_f.bounds_summary(cells)
        self.assertAlmostEqual(out["llm_minus_random"]["mean"], 0.5)
        self.assertAlmostEqual(out["random_minus_unedited"]["mean"], 0.0)


class TestAdaptiveSearch(unittest.TestCase):
    """R-016: the adaptive proposal stays in range, concentrates near elites, keeps a global share."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))
        import stage_f

        cls.stage_f = stage_f

    def test_vector_round_trip_and_batch_shape(self):
        import numpy as np

        from lares.fitting.checkpoints import RolloutScore

        sf_ = self.stage_f
        original = instantiate_source(PARENT_SOURCE, 39, 4, schema=SCHEMA)
        names, x, lo, hi = sf_.parameter_vector(original)
        copy_ = sf_.set_parameter_vector(instantiate_source(PARENT_SOURCE, 39, 4, schema=SCHEMA), names, x)
        self.assertTrue(np.allclose(sf_.parameter_vector(copy_)[1], x))
        evaluated = [(sf_.set_parameter_vector(instantiate_source(PARENT_SOURCE, 39, 4, schema=SCHEMA), names,
                                               lo + f * (hi - lo)), RolloutScore(s, 0.0))
                     for f, s in ((0.9, 1.0), (0.85, 0.8), (0.8, 0.6), (0.1, 0.0))]
        batch = sf_.adaptive_batch(evaluated, original, np.random.default_rng(0))
        self.assertEqual(len(batch), sf_.ADAPTIVE_BATCH_SIZE)
        zs = [(sf_.parameter_vector(b)[1] - lo) / np.where(hi > lo, hi - lo, 1.0) for b in batch]
        for z in zs:
            self.assertTrue(np.all(z >= -1e-6) and np.all(z <= 1 + 1e-6))
        gaussian = np.array(zs[int(round(sf_.ADAPTIVE_GLOBAL_FRACTION * sf_.ADAPTIVE_BATCH_SIZE)):])
        self.assertGreater(float(np.median(gaussian)), 0.6)   # pulled toward the elites near 0.85

    def test_summary_pairs_cells_within_placements(self):
        cells = [{"structure_id": "a", "search_seed": 1, "finalist_index": {}, "episodes": {"probe": 160, "confirmation": 100},
                  "confirmation": {"adaptive": {"x": 1.0, "y": 0.0}, "independent": {"x": 0.0, "y": 0.0}},
                  "success": {"adaptive": 0.5, "independent": 0.0}},
                 {"structure_id": "b", "search_seed": 1, "finalist_index": {}, "episodes": {"probe": 160, "confirmation": 50},
                  "confirmation": {"adaptive": {"x": 0.0, "y": 0.0}, "independent": {"x": 0.0, "y": 0.0}},
                  "success": {"adaptive": 0.0, "independent": 0.0}}]
        out = self.stage_f.adaptive_summary(cells)
        self.assertAlmostEqual(out["paired_success_difference"], 0.25)
        self.assertEqual(out["episodes"], {"probe": 320, "confirmation": 150})


class TestSeedlessSearchability(unittest.TestCase):
    """R-015: the contract survives str.format; invalid slots score zero; blocks pair the two prompts."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))
        import compare_generation_modes as cgm

        cls.cgm = cgm

    def test_the_contract_is_safe_inside_a_format_template(self):
        self.assertNotIn("{", self.cgm.SEARCHABILITY_CONTRACT)
        self.assertNotIn("}", self.cgm.SEARCHABILITY_CONTRACT)
        prompts = load_policy_prompt_assets("push-v2")
        (prompts["initial_user"] + self.cgm.SEARCHABILITY_CONTRACT).format(
            task="t", obs_dim=39, action_dim=4, obs_description="o", input_dict_string="i")

    def test_block_pairing_and_invalid_slots(self):
        slots = [{"slot_id": "b1_baseline", "block": 1, "arm": "baseline", "valid": True},
                 {"slot_id": "b1_contract", "block": 1, "arm": "contract", "valid": True},
                 {"slot_id": "b2_baseline", "block": 2, "arm": "baseline", "valid": True},
                 {"slot_id": "b2_contract", "block": 2, "arm": "contract", "valid": False}]
        one, zero = {"x": 1.0, "y": 1.0}, {"x": 0.0, "y": 0.0}
        results = {("b1_baseline", 7): {"searched": zero, "original": zero}, ("b1_contract", 7): {"searched": one, "original": zero},
                   ("b2_baseline", 7): {"searched": one, "original": one}}
        out = self.cgm.searchability_summary(slots, results, [7])
        self.assertEqual(out["block_level"]["by_block"], {1: 1.0, 2: -1.0})
        self.assertAlmostEqual(out["paired_success_difference"], 0.0)
        self.assertEqual(out["step0_block_level"]["by_block"], {1: 0.0, 2: -1.0})
