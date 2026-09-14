"""Structured two-phase generation with lineage (``debug_spec.md`` Stage A).

Run from the project root::

    python -m unittest tests.test_two_phase_policy_generation

The LLM is replaced by a scripted responder keyed on the call's transcript label,
and the subprocess harness by a stub, so these tests exercise the parsing,
mapping, lineage, correction and ledger logic exactly and run in seconds. The
in-process validator is real.
"""

import json
import os
import pickle
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from lares.core import policy_generation as pg  # noqa: E402
from lares.core.obs_schema import get_obs_schema  # noqa: E402
from lares.core.training_pipeline import load_policy_prompt_assets  # noqa: E402
from lares.search.generation import (  # noqa: E402
    LABEL_CORRECTION,
    LABEL_GENERATION,
    LABEL_IDEATION,
    LINEAGE_COMPLETE,
    LINEAGE_FAILED_VALIDATION,
    LLMCallLedger,
    StructuredOutputError,
    check_semantic_parents,
    parse_ideas_payload,
    parse_proposals_payload,
    write_feedback_bundle,
)
from lares.search.schemas import MutationProposal  # noqa: E402

SCHEMA = get_obs_schema("push-v2")

# Two valid policies with *different* parameter names, so a mapping error between
# an idea and the code implementing it is caught by the idea contract.
POLICY_A = """
class GeneratedPolicy(SymbolicPolicy):
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.w_move = nn.Parameter(torch.tensor(3.0))
        self.w_goal = nn.Parameter(torch.tensor(1.0))
        self.grip_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, obs):
        tcp = self.obs_field(obs, "tcp")
        obj = self.obs_field(obs, "obj")
        goal = self.obs_field(obs, "goal")
        diff = obj - tcp
        dist = torch.norm(diff, dim=-1, keepdim=True) + 1e-8
        move = self.w_move * diff / dist + self.w_goal * (goal - obj)
        grip = self.grip_bias.unsqueeze(0).expand(obs.shape[0], 1)
        mean = torch.cat([move, grip], dim=1)
        return mean, self.fixed_std(mean)

    def get_param_ranges(self):
        return {"w_move": (0.1, 10.0), "w_goal": (0.0, 5.0), "grip_bias": (-1.0, 1.0)}
""".strip()

POLICY_B = """
class GeneratedPolicy(SymbolicPolicy):
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.k_reach = nn.Parameter(torch.tensor(4.0))
        self.k_push = nn.Parameter(torch.tensor(2.0))
        self.radius = nn.Parameter(torch.tensor(0.05))
        self.grip = nn.Parameter(torch.tensor(-0.5))

    def forward(self, obs):
        tcp = self.obs_field(obs, "tcp")
        obj = self.obs_field(obs, "obj")
        goal = self.obs_field(obs, "goal")
        to_obj = obj - tcp
        d = torch.norm(to_obj, dim=-1, keepdim=True) + 1e-8
        to_goal = goal - obj
        dg = torch.norm(to_goal, dim=-1, keepdim=True) + 1e-8
        contact = torch.sigmoid(40.0 * (self.radius - d))
        contact = self.record_gate("contact", contact)
        move = (1 - contact) * self.k_reach * to_obj / d + contact * self.k_push * to_goal / dg
        grip = self.grip.unsqueeze(0).expand(obs.shape[0], 1)
        mean = torch.cat([move, grip], dim=1)
        return mean, self.fixed_std(mean)

    def get_param_ranges(self):
        return {"k_reach": (0.1, 10.0), "k_push": (0.1, 10.0), "radius": (0.01, 0.2), "grip": (-1.0, 1.0)}
""".strip()

# Reads the observation by raw index, which the validator rejects.
POLICY_RAW = POLICY_A.replace('tcp = self.obs_field(obs, "tcp")', "tcp = obs[:, 0:3]")

PARAMS_A = [("w_move", 3.0, 0.1, 10.0), ("w_goal", 1.0, 0.0, 5.0), ("grip_bias", 0.0, -1.0, 1.0)]
PARAMS_B = [("k_reach", 4.0, 0.1, 10.0), ("k_push", 2.0, 0.1, 10.0), ("radius", 0.05, 0.01, 0.2), ("grip", -0.5, -1.0, 1.0)]


def idea_dict(name, params, **overrides):
    d = {
        "idea_id": "ignored-by-the-parser",
        "name": name,
        "summary": f"{name}: drive the end effector to the puck, then push the puck toward the goal.",
        "named_observations": ["tcp", "obj", "goal"],
        "phases": [{"name": "approach", "activation": "far from the puck", "action_intent": "move to the puck"}],
        "parameters": [
            {"name": n, "role": "gain", "type": "gain", "units": "1", "initial_value": i, "range": [lo, hi]}
            for n, i, lo, hi in params
        ],
        "zero_shot_behavior": "moves toward the puck and pushes it roughly at the goal",
        "expected_failure": "overshoot",
        "protected_behaviors": ["reaches the puck"],
        "complexity_justification": "",
    }
    d.update(overrides)
    return d


def proposal_dict(mode, parents, **overrides):
    d = {
        "parent_ids": parents,
        "mode": mode,
        "observation": "the arm reaches the puck but pushes it sideways",
        "hypothesis": "the push direction ignores the goal offset",
        "suspected_component": "push term",
        "intervention": "aim the push along the object-to-goal direction",
        "predicted_metric": "signed_goal_progress",
        "predicted_direction": "increase",
        "protected_behaviors": ["reaches the puck"],
        "no_parent_rationale": "",
    }
    d.update(overrides)
    return d


def fence(code, label=None):
    head = f"# idea: {label}\n" if label else ""
    return f"```python\n{head}{code}\n```"


class ScriptedLLM:
    """Replaces ``_call_llm``: returns queued replies for each transcript label."""

    TOKENS = {"policy_ideas": (100, 50, 150), "policy_proposals": (120, 60, 180)}

    def __init__(self, replies):
        self.replies = {k: list(v) for k, v in replies.items()}
        self.labels = []

    def __call__(self, client, sample_size, model, messages, temperature, **kwargs):
        label = kwargs.get("transcript_label", "")
        self.labels.append(label)
        key = label if label in self.replies else label.rsplit("_", 1)[0]
        queue = self.replies.get(key)
        if not queue:
            raise AssertionError(f"unexpected LLM call {label!r}")
        text = queue.pop(0)
        choice = MagicMock()
        choice.message = MagicMock(content=text)
        pt, ct, tt = self.TOKENS.get(label.replace("_retry", ""), (10, 5, 15))
        return [choice] * sample_size, pt, ct, tt


class GenerationCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.gen_dir = os.path.join(self.root, "gen")
        os.makedirs(self.gen_dir)
        self.data_pkl = os.path.join(self.root, "data.pkl")
        with open(self.data_pkl, "wb") as f:
            pickle.dump([{"obs": np.zeros(39)}], f)
        self.ledger = LLMCallLedger(self.root)
        self.prompts = load_policy_prompt_assets("push-v2")

    def generate(self, replies, n=2, impl_mode=pg.POLICY_IMPL_PER_IDEA, generation=0, bundle=None,
                 text=None, parent_sources=None, max_repair=1, max_calls=50, max_tokens=None):
        llm = ScriptedLLM(replies)
        with patch.object(pg, "_call_llm", side_effect=llm), patch.object(
            pg, "_run_policy_validation_subprocess", return_value=(True, "Success!", "full")
        ):
            candidates = pg.generate_candidates(
                MagicMock(), SimpleNamespace(model="stub-model"), self.gen_dir, generation, 39, 4,
                "push-v2", self.prompts, n, self.data_pkl, self.ledger, feedback_bundle=bundle,
                feedback_text=text, parent_sources=parent_sources, impl_mode=impl_mode,
                max_total_calls=max_calls, max_repair_per_candidate=max_repair,
                max_total_tokens=max_tokens,
            )
        return candidates, llm


# ---------------------------------------------------------------------------
#  Strict parsing
# ---------------------------------------------------------------------------


class TestParseIdeas(unittest.TestCase):
    def payload(self, ideas):
        return json.dumps({"ideas": ideas})

    def test_exact_output_gives_that_many_distinct_ideas_with_code_assigned_ids(self):
        parsed = parse_ideas_payload(
            self.payload([idea_dict("A", PARAMS_A), idea_dict("B", PARAMS_B), idea_dict("C", PARAMS_A)]),
            3, generation=0, schema=SCHEMA,
        )
        self.assertEqual(len(parsed), 3)
        self.assertTrue(all(sp.ok for sp in parsed))
        self.assertEqual([sp.idea.idea_id for sp in parsed], ["gen0_idea0", "gen0_idea1", "gen0_idea2"])
        self.assertEqual(len({sp.idea.name for sp in parsed}), 3)

    def test_a_missing_idea_fails_rather_than_being_padded(self):
        with self.assertRaises(StructuredOutputError) as ctx:
            parse_ideas_payload(self.payload([idea_dict("A", PARAMS_A)]), 2, generation=0)
        self.assertIn("exactly 2", str(ctx.exception))

    def test_an_extra_idea_fails_too(self):
        with self.assertRaises(StructuredOutputError):
            parse_ideas_payload(self.payload([idea_dict("A", PARAMS_A)] * 3), 2, generation=0)

    def test_the_old_free_text_shape_is_refused(self):
        with self.assertRaises(StructuredOutputError):
            parse_ideas_payload('{"ideas": ["move to the puck", "push it"]}'[:-1], 2, generation=0)
        parsed = parse_ideas_payload('{"ideas": ["move to the puck", "push it"]}', 2, generation=0)
        self.assertFalse(any(sp.ok for sp in parsed))

    def test_non_json_is_refused(self):
        with self.assertRaises(StructuredOutputError):
            parse_ideas_payload("1. approach\n2. push", 2, generation=0)

    def test_a_malformed_idea_fails_only_its_own_slot(self):
        bad = idea_dict("B", PARAMS_B)
        del bad["zero_shot_behavior"]
        parsed = parse_ideas_payload(self.payload([idea_dict("A", PARAMS_A), bad]), 2, generation=0)
        self.assertTrue(parsed[0].ok)
        self.assertFalse(parsed[1].ok)
        self.assertIn("zero_shot_behavior", " ".join(parsed[1].problems))

    def test_a_duplicate_idea_is_refused(self):
        parsed = parse_ideas_payload(self.payload([idea_dict("A", PARAMS_A), idea_dict("A", PARAMS_B)]), 2, generation=0)
        self.assertTrue(parsed[0].ok)
        self.assertFalse(parsed[1].ok)
        self.assertIn("duplicates", parsed[1].problems[0])

    def test_an_unknown_observation_field_is_refused(self):
        bad = idea_dict("A", PARAMS_A, named_observations=["tcp", "puck_velocity"])
        parsed = parse_ideas_payload(self.payload([bad]), 1, generation=0, schema=SCHEMA)
        self.assertFalse(parsed[0].ok)

    def test_a_policy_above_the_parameter_budget_needs_a_justification(self):
        big = [(f"p{i}", 1.0, 0.0, 2.0) for i in range(12)]
        refused = parse_ideas_payload(self.payload([idea_dict("A", big)]), 1, generation=0)
        self.assertFalse(refused[0].ok)
        self.assertIn("complexity_justification", refused[0].problems[0])
        allowed = parse_ideas_payload(
            self.payload([idea_dict("A", big, complexity_justification="needs per-axis gains")]), 1, generation=0
        )
        self.assertTrue(allowed[0].ok)

    def test_a_fenced_payload_is_unwrapped(self):
        parsed = parse_ideas_payload("```json\n" + self.payload([idea_dict("A", PARAMS_A)]) + "\n```", 1, generation=0)
        self.assertTrue(parsed[0].ok)


class TestProposalParentRules(unittest.TestCase):
    CONTEXT = ["gen0_cand0", "gen0_cand1"]

    def problems(self, mode, parents, **kw):
        proposal = MutationProposal.from_dict({**proposal_dict(mode, parents, **kw), "proposal_id": "p"})
        return check_semantic_parents(proposal, self.CONTEXT)

    def test_a_repair_has_exactly_one_parent(self):
        self.assertEqual(self.problems("repair", ["gen0_cand0"]), [])
        self.assertTrue(self.problems("repair", self.CONTEXT))
        self.assertTrue(self.problems("repair", []))

    def test_a_simplification_has_exactly_one_parent(self):
        self.assertEqual(self.problems("simplification", ["gen0_cand1"]), [])
        self.assertTrue(self.problems("simplification", self.CONTEXT))

    def test_a_crossover_has_at_least_two_parents(self):
        self.assertEqual(self.problems("crossover", self.CONTEXT), [])
        self.assertTrue(self.problems("crossover", ["gen0_cand0"]))

    def test_a_parentless_exploration_must_say_why(self):
        self.assertTrue(self.problems("exploration", []))
        self.assertEqual(self.problems("exploration", [], no_parent_rationale="every parent never reaches the puck"), [])

    def test_a_parent_outside_the_feedback_context_is_refused(self):
        found = self.problems("repair", ["gen0_cand7"])
        self.assertIn("feedback context", found[0])

    def test_parameter_only_edits_are_not_accepted_from_the_generator(self):
        self.assertTrue(self.problems("parameter_only", ["gen0_cand0"]))

    def test_the_payload_parser_applies_the_rules_per_slot(self):
        payload = json.dumps({"proposals": [
            {"proposal": proposal_dict("repair", ["gen0_cand0"]), "idea": idea_dict("A", PARAMS_A)},
            {"proposal": proposal_dict("crossover", ["gen0_cand0"]), "idea": idea_dict("B", PARAMS_B)},
        ]})
        parsed = parse_proposals_payload(payload, 2, generation=1, context_parent_ids=self.CONTEXT)
        self.assertTrue(parsed[0].ok)
        self.assertEqual(parsed[0].proposal.proposal_id, "gen1_prop0")
        self.assertFalse(parsed[1].ok)

    def test_an_unmeasurable_prediction_is_refused(self):
        payload = json.dumps({"proposals": [
            {"proposal": proposal_dict("repair", ["gen0_cand0"], predicted_metric="vibes"), "idea": idea_dict("A", PARAMS_A)},
        ]})
        parsed = parse_proposals_payload(payload, 1, generation=1, context_parent_ids=self.CONTEXT)
        self.assertFalse(parsed[0].ok)


class TestExtractNPolicyCodes(unittest.TestCase):
    def test_two_fences(self):
        text = f"Intro\n{fence(POLICY_A)}\n\n{fence(POLICY_A)}\n"
        codes = pg._extract_n_policy_codes(text, 2)
        self.assertEqual(len(codes), 2)
        self.assertIn("class GeneratedPolicy", codes[0])

    def test_labelled_fences_map_by_label_not_by_position(self):
        text = f"{fence(POLICY_B, 'gen0_idea1')}\n{fence(POLICY_A, 'gen0_idea0')}"
        codes, mode, found = pg._extract_labelled_policy_codes(text, ["gen0_idea0", "gen0_idea1"])
        self.assertEqual(mode, "labelled")
        self.assertIn("w_move", codes["gen0_idea0"])
        self.assertIn("k_reach", codes["gen0_idea1"])


# ---------------------------------------------------------------------------
#  Generation end to end with a scripted LLM
# ---------------------------------------------------------------------------


class TestGenerationZero(GenerationCase):
    def ideas(self):
        return json.dumps({"ideas": [idea_dict("A", PARAMS_A), idea_dict("B", PARAMS_B)]})

    def test_per_idea_attaches_each_implementation_to_its_own_idea(self):
        candidates, llm = self.generate({
            "policy_ideas": [self.ideas()],
            "policy_implement": [fence(POLICY_A), fence(POLICY_B)],
        })
        self.assertEqual([c.candidate_id for c in candidates], ["gen0_cand0", "gen0_cand1"])
        self.assertTrue(all(c.valid for c in candidates))
        for c in candidates:
            self.assertEqual(c.idea.check_against_policy(c.policy), [])
            self.assertEqual(c.mapping["idea_id"], c.idea.idea_id)
            self.assertEqual(c.lineage_status(), LINEAGE_COMPLETE)
            self.assertIsNone(c.proposal)
        self.assertEqual(len(self.ledger), 3)

    def test_batched_mode_maps_fences_by_label(self):
        batched = f"{fence(POLICY_B, 'gen0_idea1')}\n\n{fence(POLICY_A, 'gen0_idea0')}"
        candidates, _ = self.generate(
            {"policy_ideas": [self.ideas()], "policy_implement_batched": [batched]},
            impl_mode=pg.POLICY_IMPL_BATCHED,
        )
        self.assertTrue(all(c.valid for c in candidates))
        self.assertEqual(candidates[0].mapping["mode"], "batched")
        self.assertIn("w_move", candidates[0].code)
        self.assertIn("k_reach", candidates[1].code)

    def test_a_missing_batched_fence_is_filled_for_that_idea(self):
        batched = fence(POLICY_A, "gen0_idea0")
        candidates, llm = self.generate(
            {"policy_ideas": [self.ideas()], "policy_implement_batched": [batched], "policy_implement": [fence(POLICY_B)]},
            impl_mode=pg.POLICY_IMPL_BATCHED,
        )
        self.assertTrue(all(c.valid for c in candidates))
        self.assertEqual(candidates[1].mapping["mode"], "fill")
        self.assertIn("policy_implement_1", llm.labels)

    def test_a_correction_is_one_linked_attempt(self):
        candidates, llm = self.generate({
            "policy_ideas": [self.ideas()],
            "policy_implement": [fence(POLICY_RAW), fence(POLICY_B)],
            "policy_correction": [fence(POLICY_A)],
        })
        slot0 = [c for c in candidates if c.slot == 0]
        self.assertEqual([c.attempt for c in slot0], [0, 1])
        failed, fixed = slot0
        self.assertFalse(failed.valid)
        self.assertEqual(failed.validation["stage"], pg.STAGE_INSTANTIATION)
        self.assertEqual(failed.lineage_status(), LINEAGE_FAILED_VALIDATION)
        self.assertTrue(fixed.valid)
        self.assertEqual(fixed.previous_attempt_id, "gen0_cand0_a0")
        correction = self.ledger.get(fixed.mapping["call_id"])
        self.assertEqual(correction.label, LABEL_CORRECTION)
        self.assertIn(correction.call_id, fixed.llm_call_ids)
        self.assertEqual(correction.attempt_ids, ["gen0_cand0_a1"])
        self.assertEqual(len({c.attempt_id for c in candidates}), len(candidates))

    def test_a_failed_correction_is_not_retried_again(self):
        candidates, llm = self.generate({
            "policy_ideas": [self.ideas()],
            "policy_implement": [fence(POLICY_RAW), fence(POLICY_B)],
            "policy_correction": [fence(POLICY_RAW), fence(POLICY_A)],
        })
        slot0 = [c for c in candidates if c.slot == 0]
        self.assertEqual(len(slot0), 2)
        self.assertFalse(any(c.valid for c in slot0))
        self.assertEqual(llm.labels.count("policy_correction_0"), 1)

    def test_code_implementing_the_wrong_idea_is_rejected(self):
        candidates, _ = self.generate(
            {"policy_ideas": [self.ideas()], "policy_implement": [fence(POLICY_B), fence(POLICY_B)],
             "policy_correction": [fence(POLICY_B)]},
        )
        slot0 = [c for c in candidates if c.slot == 0]
        self.assertEqual(slot0[0].validation["stage"], pg.STAGE_IDEA_MISMATCH)

    def test_malformed_ideation_fails_every_slot_with_ids_intact(self):
        candidates, llm = self.generate({"policy_ideas": ["not json", '{"ideas": []}']})
        self.assertEqual(llm.labels, ["policy_ideas", "policy_ideas_retry"])
        self.assertEqual([c.attempt_id for c in candidates], ["gen0_cand0_a0", "gen0_cand1_a0"])
        for c in candidates:
            self.assertFalse(c.valid)
            self.assertEqual(c.validation["stage"], pg.STAGE_IDEATION)
            self.assertEqual(len(c.llm_call_ids), 2)
            self.assertTrue(c.validation["errors"])
        with open(os.path.join(self.gen_dir, "Iter_0_designs.json")) as f:
            self.assertEqual(json.load(f)["raw_responses"][0], "not json")

    def test_a_retry_that_fixes_the_payload_is_used(self):
        candidates, _ = self.generate({
            "policy_ideas": ['{"ideas": []}', self.ideas()],
            "policy_implement": [fence(POLICY_A), fence(POLICY_B)],
        })
        self.assertTrue(all(c.valid for c in candidates))
        self.assertEqual(len(candidates[0].llm_call_ids), 3)

    def test_a_shared_ideation_call_is_counted_once(self):
        candidates, _ = self.generate({
            "policy_ideas": [self.ideas()],
            "policy_implement": [fence(POLICY_A), fence(POLICY_B)],
        })
        ideation = [r for r in self.ledger.records() if r.label == LABEL_IDEATION]
        self.assertEqual(len(ideation), 1)
        self.assertEqual(ideation[0].attempt_ids, ["gen0_cand0_a0", "gen0_cand1_a0"])
        self.assertTrue(ideation[0].shared)
        referenced = [cid for c in candidates for cid in c.llm_call_ids]
        self.assertGreater(len(referenced), len(set(referenced)))
        totals = self.ledger.totals(referenced)
        self.assertEqual(totals["llm_calls"], 3)
        self.assertEqual(totals["llm_total_tokens"], 150 + 15 + 15)
        self.assertEqual(totals, {**self.ledger.totals(), "unknown_call_ids": []})

    def test_prompts_and_responses_are_stored_and_hashed(self):
        self.generate({"policy_ideas": [self.ideas()], "policy_implement": [fence(POLICY_A), fence(POLICY_B)]})
        self.assertEqual(self.ledger.verify(), {})
        for record in self.ledger.records():
            self.assertIn(record.label, (LABEL_IDEATION, LABEL_GENERATION))
            self.assertTrue(os.path.isfile(os.path.join(self.root, record.prompt_path)))

    def test_the_token_budget_bounds_the_generation(self):
        # Ideation costs 150 tokens and each implementation 15: after the first slot the
        # 160-token budget is spent, so the second slot is recorded as unreachable.
        candidates, llm = self.generate(
            {"policy_ideas": [self.ideas()], "policy_implement": [fence(POLICY_A)]}, max_tokens=160
        )
        self.assertTrue(candidates[0].valid)
        self.assertEqual(candidates[1].validation["stage"], pg.STAGE_BUDGET)
        self.assertIn("165/160 tokens", candidates[1].validation["errors"][0])

    def test_the_call_budget_bounds_the_generation(self):
        candidates, llm = self.generate(
            {"policy_ideas": [self.ideas()], "policy_implement": [fence(POLICY_A)]}, max_calls=2
        )
        self.assertTrue(candidates[0].valid)
        self.assertEqual(candidates[1].validation["stage"], pg.STAGE_BUDGET)


class TestLaterGenerations(GenerationCase):
    def bundle(self, context=("gen0_cand0", "gen0_cand1")):
        text = "## Evidence from generation 0\n(population table)"
        return write_feedback_bundle(self.root, 0, "dev#head10", list(context), text), text

    def test_generation_one_candidates_carry_proposals_and_valid_parents(self):
        bundle, text = self.bundle()
        proposals = json.dumps({"proposals": [
            {"proposal": proposal_dict("repair", ["gen0_cand0"]), "idea": idea_dict("A2", PARAMS_A)},
            {"proposal": proposal_dict("crossover", ["gen0_cand0", "gen0_cand1"]), "idea": idea_dict("B2", PARAMS_B)},
            {"proposal": proposal_dict("exploration", [], no_parent_rationale="both parents stall"), "idea": idea_dict("C2", PARAMS_A)},
        ]})
        candidates, llm = self.generate(
            {"policy_proposals": [proposals], "policy_implement": [fence(POLICY_A), fence(POLICY_B), fence(POLICY_A)]},
            n=3, generation=1, bundle=bundle, text=text,
            parent_sources={"gen0_cand0": POLICY_A, "gen0_cand1": POLICY_B},
        )
        self.assertEqual([c.mode for c in candidates], ["repair", "crossover", "exploration"])
        for c in candidates:
            self.assertTrue(c.valid, c.validation)
            self.assertEqual(c.feedback_id, bundle.feedback_id)
            self.assertEqual(c.context_parent_ids, ["gen0_cand0", "gen0_cand1"])
            self.assertEqual(c.lineage_problems(bundle), [])
            self.assertEqual(c.lineage_status(bundle), LINEAGE_COMPLETE)
        self.assertEqual(candidates[0].parent_ids, ["gen0_cand0"])
        self.assertEqual(candidates[2].parent_ids, [])
        implement = self.ledger.get(candidates[0].mapping["call_id"])
        with open(os.path.join(self.root, implement.prompt_path)) as f:
            prompt = f.read()
        self.assertIn("### Parent gen0_cand0 source", prompt)
        self.assertIn("self.w_move", prompt)
        design = self.ledger.get(candidates[0].llm_call_ids[0])
        with open(os.path.join(self.root, design.prompt_path)) as f:
            self.assertIn('[\\"gen0_cand0\\", \\"gen0_cand1\\"]', f.read())

    def test_a_proposal_breaking_its_parent_rule_fails_its_slot_only(self):
        bundle, text = self.bundle()
        proposals = json.dumps({"proposals": [
            {"proposal": proposal_dict("repair", ["gen0_cand0", "gen0_cand1"]), "idea": idea_dict("A2", PARAMS_A)},
            {"proposal": proposal_dict("repair", ["gen0_cand1"]), "idea": idea_dict("B2", PARAMS_B)},
        ]})
        bad_again = proposals
        candidates, _ = self.generate(
            {"policy_proposals": [proposals, bad_again], "policy_implement": [fence(POLICY_B)]},
            generation=1, bundle=bundle, text=text,
        )
        self.assertFalse(candidates[0].valid)
        self.assertEqual(candidates[0].validation["stage"], pg.STAGE_PROPOSAL)
        self.assertTrue(candidates[1].valid)

    def test_generation_one_without_a_feedback_bundle_is_refused(self):
        with self.assertRaises(ValueError):
            self.generate({}, generation=1)

    def test_feedback_text_that_is_not_the_hashed_text_is_refused(self):
        bundle, _ = self.bundle()
        with self.assertRaises(ValueError):
            self.generate({}, generation=1, bundle=bundle, text="something else")


class TestLegacyEntryPoint(GenerationCase):
    def test_get_symbolic_policies_two_phase_returns_candidates(self):
        args = SimpleNamespace(model="stub-model", policy_gen_two_phase=True, policy_impl_mode=pg.POLICY_IMPL_PER_IDEA)
        llm = ScriptedLLM({
            "policy_ideas": [json.dumps({"ideas": [idea_dict("A", PARAMS_A), idea_dict("B", PARAMS_B)]})],
            "policy_implement": [fence(POLICY_A), fence(POLICY_B)],
        })
        p = self.prompts
        with patch.object(pg, "_call_llm", side_effect=llm), patch.object(
            pg, "_run_policy_validation_subprocess", return_value=(True, "Success!", "full")
        ):
            candidates = pg.get_symbolic_policies(
                client=MagicMock(), dir_path=self.gen_dir, llm_iter=0, args=args, obs_dim=39, action_dim=4,
                env_name="push-v2", initial_system=p["initial_system"], initial_user=p["initial_user"],
                task_description=p["task_description"], obs_description=p["obs_description"],
                input_dict_string=p["input_dict_string"], code_output_tip=p["code_output_tip"],
                data_pkl_path=self.data_pkl, real_num=2, ideas_system=p["ideas_system"], ideas_user=p["ideas_user"],
            )
        policies, codes, responses = pg.valid_policies(candidates)
        self.assertEqual(len(policies), 2)
        self.assertEqual(len(codes), 2)
        self.assertTrue(os.path.isfile(os.path.join(self.gen_dir, "llm_calls.jsonl")))


if __name__ == "__main__":
    unittest.main()
