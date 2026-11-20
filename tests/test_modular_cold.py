"""R-092's mocked gate: the modular-versus-monolithic cold study (``lares/envs/isaac_pen_spin/modular_cold.py``).

Real: manifest freezing and role slices, the slot generator with the module contract as its extra source check, the
linked correction, ``llm_pool`` (saved generations), the stream's request/response protocol and the child's policy
rebuild, the 24-candidate rollout search (one stream call per candidate), strict promotion against the scored null,
fresh evaluation and the block-paired summary. Mocked: Isaac (the toy pen world of ``test_pen_spin_pipeline``), the
LLM and the policy-validation subprocess.

Scripted replies: the monolithic arm always gets a controller that cannot spin the toy pen. The modular arm gets a
contract-compliant controller that does not spin at its initial values but does after a parameter draw; its first slot
of each block first violates the contract (``forward`` reads the observation), and the correction repairs it in block 0
and repeats the violation in block 1 (that slot stays invalid and runs no episode).

    python -m unittest tests.test_modular_cold
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))

import torch  # noqa: E402
import yaml  # noqa: E402

from lares.core import policy_generation as pg  # noqa: E402
from lares.core.module_contract import MODULAR_REQUIREMENT, check_module_contract  # noqa: E402
from lares.envs.isaac_pen_spin import cases, modular_cold  # noqa: E402
from lares.envs.isaac_pen_spin import pipeline_backend as pb  # noqa: E402
from lares.eval.manifest import EvaluationManifest  # noqa: E402
from test_pen_spin_pipeline import CONFIG, NEUTRAL, POLICY_FLAT, FakeChild  # noqa: E402

#: Contract-compliant; the knuckles move apart by ``k_flex`` (0 at first, so only a searched draw spins the toy pen).
POLICY_MODULAR = f"""
class GeneratedPolicy(SymbolicPolicy):
    MODULE_SPEC = {{
        "sense": {{"inputs": ["obj_pos", "obj_rot", "obj_angvel", "hand_dof_pos"], "outputs": ["trim"],
                  "units": {{"trim": "normalised joint offset"}}, "dependencies": []}},
        "drive": {{"inputs": [], "outputs": ["base"], "units": "normalised joint targets", "dependencies": ["sense"]}},
    }}

    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.k_flex = nn.Parameter(torch.tensor(0.0))
        self.k_wrist = nn.Parameter(torch.tensor(0.2))

    def sense(self, obs):
        pos = self.obs_field(obs, "obj_pos")
        rot = self.obs_field(obs, "obj_rot")
        w = self.obs_field(obs, "obj_angvel")
        q = self.obs_field(obs, "hand_dof_pos")
        return 0.01 * torch.tanh(pos.sum(-1) + rot.sum(-1) + w.sum(-1) + q.sum(-1))

    def drive(self, trim):
        base = torch.tensor({NEUTRAL}).expand(trim.shape[0], -1).clone()
        base[:, 1] = base[:, 1] + trim
        base[:, 7] = self.k_flex
        base[:, 8] = -self.k_flex
        base[:, 0] = base[:, 0] * 0 + self.k_wrist
        return base

    def forward(self, obs):
        base = self.drive(self.sense(obs))
        return base, self.fixed_std(base)

    def get_param_ranges(self):
        return {{"k_flex": (-0.5, 0.5), "k_wrist": (-0.5, 0.5)}}
""".strip()

#: Declares the modules but lets ``forward`` read the observation as well: a bypass.
POLICY_BYPASS = POLICY_MODULAR.replace(
    "        base = self.drive(self.sense(obs))",
    '        base = self.drive(self.sense(obs)) + 0.0 * self.obs_field(obs, "hand_dof_vel").sum(-1, keepdim=True)')

BLOCKS = 2


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class ArmLLM:
    """Replaces ``_call_llm``: the reply depends on the arm (whether the prompt carries the requirement) and label."""

    def __init__(self):
        self.modular = {"policy_generation": [POLICY_BYPASS, POLICY_MODULAR, POLICY_MODULAR, POLICY_MODULAR] * BLOCKS,
                        "policy_repair": [POLICY_MODULAR, POLICY_BYPASS]}
        self.calls = []

    def __call__(self, client, sample_size, model, messages, temperature, **kwargs):
        label = kwargs.get("transcript_label", "")
        modular = MODULAR_REQUIREMENT in messages[1]["content"]
        self.calls.append({"label": label, "modular": modular, "messages": messages, "model": model,
                           "reasoning_effort": kwargs.get("reasoning_effort")})
        if modular:
            key = next(k for k in self.modular if label.startswith(k))
            text = self.modular[key].pop(0)
        else:
            if not label.startswith("policy_generation"):
                raise AssertionError(f"the monolithic arm needs no correction, got {label!r}")
            text = POLICY_FLAT
        choice = MagicMock()
        choice.message = MagicMock(content=f"```python\n{text}\n```")
        return [choice] * sample_size, 10, 5, 15


class ModularRun:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        c = dict(yaml.safe_load(read(CONFIG))["modular_cold"])
        self.committed = dict(c)
        c["study_dir"] = os.path.join(root, "study")
        c["block_seeds"] = c["block_seeds"][:BLOCKS]
        c["llm_workers"] = 1  # serial, so the scripted reply order is the slot order
        c.update(model="gpt-6.1-sol", reasoning_effort="high")
        self.ccfg, self.study_dir, self.mdir = c, c["study_dir"], os.path.join(root, "manifests")
        self.child, self.llm = FakeChild(), ArmLLM()
        self.summary = self.run()

    def run(self):
        child = self.child
        factory = lambda stage, name, work: pb.IsaacPenSpinStream(stage, work, name, runner=child)  # noqa: E731
        with patch.object(pg, "_call_llm", side_effect=self.llm), \
                patch.object(pg, "_run_policy_validation_subprocess", return_value=(True, "Success!", "full")), \
                patch.object(modular_cold, "manifest_file",
                             lambda seed: cases.manifest_path(3, modular_cold.LABEL, seed, self.mdir)):
            return modular_cold.run_study(self.ccfg, factory, MagicMock())

    def arm(self, b, arm):
        return json.loads(read(os.path.join(self.study_dir, f"block{b}", arm, "arm.json")))

    def role_requests(self, role):
        return [r for r in self.child.requests
                if EvaluationManifest.load(r["manifest_path"]).manifest_id.endswith(f":{role}")]

    def close(self):
        self.tmp.cleanup()


class TestModularColdStudy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = ModularRun()

    @classmethod
    def tearDownClass(cls):
        cls.r.close()

    def test_fixtures_are_what_they_claim(self):
        self.assertEqual(check_module_contract(POLICY_MODULAR), [])
        self.assertIn("`forward` calls `self.obs_field`", "\n".join(check_module_contract(POLICY_BYPASS)))
        self.assertTrue(check_module_contract(POLICY_FLAT))

    def test_the_committed_block_is_the_protocol(self):
        c = self.r.committed
        self.assertEqual(c["block_seeds"], list(range(4710, 4718)))
        self.assertEqual(c["study_dir"], "logs/pen_spin/r092_modular_cold")
        self.assertEqual((c["max_calls_per_slot"], c["llm_workers"]), (2, 16))
        self.assertEqual(modular_cold.manifest_file(4710), "config/manifests/pen_spin_stage3_modular_cold_s4710.yaml")
        self.assertEqual(modular_cold.ROLES, {"screen": (0, 10), "promotion": (10, 40), "probe": (40, 50),
                                              "fresh": (50, 80)})
        self.assertEqual((modular_cold.ARMS, modular_cold.PRIMARY, modular_cold.SLOTS),
                         (("monolithic", "modular"), ("modular", "monolithic"), 4))
        self.assertEqual(8 * (80 + 2 * modular_cold.MAX_ARM_EPISODES), 6080)
        self.assertEqual(8 * 2 * modular_cold.SLOTS * c["max_calls_per_slot"], 128)

    def test_manifests_are_frozen_stage3_blocks_of_eighty(self):
        for seed in self.r.ccfg["block_seeds"]:
            m = EvaluationManifest.load(cases.manifest_path(3, "modular_cold", seed, self.r.mdir))
            self.assertEqual(len(m), 80)
            self.assertEqual(cases.stage_of(m), 3)

    def test_prompts_differ_only_in_the_requirement(self):
        for b in range(BLOCKS):
            d = os.path.join(self.r.study_dir, f"block{b}")
            base = read(os.path.join(d, "monolithic", "user_prompt.txt"))
            treat = read(os.path.join(d, "modular", "user_prompt.txt"))
            self.assertNotEqual(base, treat)
            self.assertEqual(treat.replace("\n" + MODULAR_REQUIREMENT, ""), base)
        first = {m: next(c for c in self.r.llm.calls if c["modular"] is m and c["label"].startswith("policy_generation"))
                 for m in (True, False)}
        self.assertEqual(first[True]["messages"][0], first[False]["messages"][0])  # the same system prompt
        self.assertEqual(first[True]["messages"][1]["content"].replace("\n" + MODULAR_REQUIREMENT, ""),
                         first[False]["messages"][1]["content"])
        self.assertEqual({(c["model"], c["reasoning_effort"]) for c in self.r.llm.calls}, {("gpt-6.1-sol", "high")})

    def test_a_contract_violation_gets_exactly_one_correction(self):
        repairs = [c for c in self.r.llm.calls if c["label"].startswith("policy_repair")]
        self.assertEqual(len(repairs), BLOCKS)
        for c in repairs:
            self.assertTrue(c["modular"])
            self.assertIn("`forward` calls `self.obs_field`", c["messages"][-1]["content"])
        ok, bad = self.r.arm(0, "modular"), self.r.arm(1, "modular")
        self.assertEqual([a["valid"] for a in ok["slots"][0]["attempts"]], [False, True])
        self.assertEqual(ok["slots"][0]["attempts"][0]["validation"]["stage"], pg.STAGE_MODULE_CONTRACT)
        self.assertEqual(ok["valid_candidates"], 4)
        self.assertEqual([a["valid"] for a in bad["slots"][0]["attempts"]], [False, False])
        self.assertEqual(bad["valid_candidates"], 3)
        self.assertNotIn("0", bad["screens"])  # the slot still invalid after its correction runs no episode
        self.assertEqual(sorted(bad["screens"]), ["1", "2", "3"])

    def test_the_monolithic_arm_never_sees_the_contract(self):
        for b in range(BLOCKS):
            a = self.r.arm(b, "monolithic")
            self.assertEqual(a["valid_candidates"], 4)
            for s in a["slots"]:
                self.assertEqual(len(s["attempts"]), 1)
                self.assertNotIn("source_check", s["attempts"][0]["validation"])
                self.assertNotIn("MODULE_SPEC", s["code"])

    def test_the_search_scores_24_candidates_in_24_processes(self):
        from lares.search import rollout_search

        for b, seed in enumerate(self.r.ccfg["block_seeds"]):
            for arm in modular_cold.ARMS:
                a = self.r.arm(b, arm)
                s = a["search"]
                self.assertEqual(len(s["trace"]), 24)
                self.assertEqual([t["kind"] for t in s["trace"]],
                                 list(rollout_search.FIRST_ORDER + rollout_search.REST_ORDER))
                self.assertEqual((s["seed"], s["candidates"], s["slot"]), (seed, 24, a["selected_slot"]))
                self.assertEqual(a["isaac_processes"], a["valid_candidates"] + 24 + 2)
                for t in s["trace"]:
                    self.assertEqual(len(t["per_case"]), 10)
                    self.assertEqual(set(t) >= {"kind", "success", "tail_hit_fraction", "state_sha256"}, True)
                self.assertEqual(len({t["state_sha256"] for t in s["trace"]}), 24)
        # per block: the null once, then 24 candidates per arm, each its own request on the probe role
        self.assertEqual(len(self.r.role_requests("probe")), BLOCKS * (1 + 2 * 24))

    def test_the_winner_follows_the_tie_rule_and_its_state_is_frozen(self):
        for b in range(BLOCKS):
            for arm in modular_cold.ARMS:
                a = self.r.arm(b, arm)
                s, adir = a["search"], os.path.join(self.r.study_dir, f"block{b}", arm)
                keys = [(t["success"], t["tail_hit_fraction"], -t["index"]) for t in s["trace"]]
                self.assertEqual(s["selected_index"], keys.index(max(keys)))
                self.assertEqual(s["selected_state"]["sha256"], s["trace"][s["selected_index"]]["state_sha256"])
                self.assertEqual(json.loads(read(os.path.join(adir, "search.json"))), s)
                state = torch.load(os.path.join(adir, "selected_state.pt"), weights_only=True)
                self.assertEqual({k: v.tolist() for k, v in state.items()}, s["selected_state"]["parameters"])
        mono, mod = self.r.arm(0, "monolithic")["search"], self.r.arm(0, "modular")["search"]
        self.assertEqual(mono["selected_index"], 0)  # nothing spins: the unchanged candidate keeps every tie
        self.assertEqual(mono["candidates_with_probe_success"], 0)
        self.assertGreater(mod["selected_index"], 0)  # the original does not spin; a draw does
        self.assertGreater(mod["candidates_with_probe_success"], 0)

    def test_promotion_is_against_the_scored_null_and_fresh_follows_it(self):
        for blk in self.r.summary["blocks"]:
            for arm in modular_cold.ARMS:
                a = blk["arms"][arm]
                d = a["promotion_decision"]
                self.assertEqual(d["incumbent_id"], "neutral_pose")
                self.assertEqual(len(a["promotion"]["per_case"]), 30)
                self.assertEqual(len(a["fresh_selected"]["per_case"]), 30)
                want = a["fresh_selected"]["success"] if a["promoted"] else blk["null"]["fresh"]["success"]
                self.assertEqual(a["fresh_incumbent"]["success"], want)
                self.assertEqual(a["promoted"], arm == "modular")
            self.assertEqual(blk["arms"]["monolithic"]["fresh_incumbent"]["candidate"], "null")
            self.assertEqual(blk["null"]["fresh"]["success"], 0.0)

    def test_the_promoted_actor_is_the_searched_state(self):
        a = self.r.arm(0, "modular")
        fresh = [r for r in self.r.role_requests("fresh") if r["actor"].get("kind") == "symbolic"
                 and "MODULE_SPEC" in r["actor"]["source"]][0]
        self.assertEqual(fresh["actor"]["source"], a["selected_code"])
        self.assertNotEqual(a["search"]["selected_state"]["sha256"], a["search"]["trace"][0]["state_sha256"])

    def test_episode_and_call_accounting(self):
        eps = sum(len(EvaluationManifest.load(r["manifest_path"])) for r in self.r.child.episodes())
        want = BLOCKS * 80 + 2 * 340 + 340 + 330  # block 1's modular arm screens three slots
        self.assertEqual(eps, want)
        self.assertEqual(self.r.summary["isaac_episodes"], want)
        self.assertEqual(self.r.summary["isaac_processes"], len(self.r.child.requests))
        self.assertEqual([self.r.arm(b, arm)["isaac_episodes"] for b in range(BLOCKS) for arm in modular_cold.ARMS],
                         [340, 340, 340, 330])
        self.assertEqual(len(self.r.llm.calls), BLOCKS * (4 + 5))
        self.assertEqual(self.r.summary["llm_calls"], len(self.r.llm.calls))
        self.assertEqual(self.r.summary["arms"]["modular"]["llm_calls"], 10)
        self.assertEqual(self.r.summary["llm_tokens"], 15 * len(self.r.llm.calls))

    def test_generation_is_saved_before_any_episode(self):
        for b in range(BLOCKS):
            for arm in modular_cold.ARMS:
                adir = os.path.join(self.r.study_dir, f"block{b}", arm)
                g = json.loads(read(os.path.join(adir, "generated.json")))
                self.assertEqual(len(g["slots"]), 4)
                self.assertLess(os.path.getmtime(os.path.join(adir, "generated.json")),
                                os.path.getmtime(os.path.join(adir, "arm.json")))

    def test_summary_fields(self):
        s = self.r.summary
        self.assertEqual(json.loads(read(os.path.join(self.r.study_dir, "summary.json")))["paired_success_difference"],
                         s["paired_success_difference"])
        self.assertTrue(os.path.exists(os.path.join(self.r.study_dir, "summary.partial.json")))
        primary = s["modular_minus_monolithic_fresh_incumbent"]
        self.assertEqual(len(primary["per_block"]), BLOCKS)
        self.assertEqual(s["paired_success_difference"], primary["mean"])
        self.assertGreater(s["paired_success_difference"], 0.0)
        for key in ("fresh_selected", "fresh_selected_tail_hit_fraction", "fresh_selected_drops",
                    "candidates_with_screen_success", "valid_slots", "search_candidates_with_probe_success"):
            self.assertEqual(len(s[f"modular_minus_monolithic_{key}"]["per_block"]), BLOCKS, key)
        self.assertEqual(s["modular_minus_monolithic_valid_slots"]["per_block"], [0.0, -1.0])
        mono, mod = s["arms"]["monolithic"], s["arms"]["modular"]
        self.assertEqual((mono["slots"], mono["valid_slots"], mono["slots_valid_within_one_correction"]), (8, 8, 1.0))
        self.assertEqual((mod["slots"], mod["valid_slots"], mod["slots_valid_within_one_correction"]), (8, 7, 7 / 8))
        self.assertEqual(s["slots_valid_within_one_correction"], {"monolithic": 1.0, "modular": 7 / 8})
        self.assertEqual((mod["first_attempt_valid_slots"], mod["module_contract_defect_slots"]), (6, 2))
        self.assertEqual((mono["promoted_blocks"], mono["plateau_blocks"]), (0, BLOCKS))
        self.assertEqual((mod["promoted_blocks"], mod["plateau_blocks"]), (BLOCKS, 0))
        self.assertEqual(mono["fresh_selected"], 0.0)
        self.assertEqual(sum(mono["fresh_selected_failure_labels"].values()), 30 * BLOCKS)
        self.assertEqual(mono["fresh_selected_drops"], mono["fresh_selected_failure_labels"].get("dropped", 0))
        for arm in (mono, mod):
            for key in ("fresh_incumbent", "fresh_selected_tail_hit_fraction", "candidates_with_screen_success",
                        "search_candidates_with_probe_success", "llm_tokens", "isaac_processes", "isaac_episodes"):
                self.assertIn(key, arm)

    def test_zz_a_rerun_resumes_without_a_new_episode_or_call(self):
        before, calls = len(self.r.child.requests), len(self.r.llm.calls)
        again = self.r.run()
        self.assertEqual(len(self.r.child.requests), before)
        self.assertEqual(len(self.r.llm.calls), calls)
        self.assertEqual(again["paired_success_difference"], self.r.summary["paired_success_difference"])

    def test_zz_an_interrupted_arm_is_not_rerun(self):
        adir = os.path.join(self.r.study_dir, "block0", "modular")
        os.rename(os.path.join(adir, "arm.json"), os.path.join(adir, "arm.json.bak"))
        try:
            with self.assertRaises(SystemExit):
                self.r.run()
        finally:
            os.rename(os.path.join(adir, "arm.json.bak"), os.path.join(adir, "arm.json"))


class TestNoValidCandidate(unittest.TestCase):
    def test_an_arm_with_no_valid_slot_keeps_the_null_and_runs_no_episode(self):
        with tempfile.TemporaryDirectory() as d:
            null = {"fresh": {"success": 0.1, "per_case": [0.0] * 30}}
            g = {"llm_calls": 8, "llm_tokens": 120, "prompt_sha256": "x",
                 "slots": [{"slot": k, "valid": False, "policy": None, "code": None, "attempts": []} for k in range(4)]}

            def no_stream(*a):
                raise AssertionError("no episode may run")

            a = modular_cold.run_arm("modular", {}, 4710, {}, null, d, no_stream, None, generated=g)
            self.assertEqual((a["valid_candidates"], a["isaac_episodes"], a["promoted"]), (0, 0, False))
            self.assertEqual(a["fresh_incumbent"]["success"], 0.1)
            row = {"null": {"fresh": null["fresh"]}, "arms": {"modular": a, "monolithic": a}}
            s = modular_cold.summarise([row, row])
            self.assertEqual(s["paired_success_difference"], 0.0)
            self.assertEqual(s["modular_minus_monolithic_fresh_selected"]["per_block"], [])
            self.assertEqual(s["arms"]["modular"]["slots_valid_within_one_correction"], 0.0)


if __name__ == "__main__":
    unittest.main()
