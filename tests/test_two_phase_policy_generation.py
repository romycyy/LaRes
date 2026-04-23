"""Unit tests for two-phase symbolic policy LLM generation (run on server with pytest)."""

import os
import pickle
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

import sys
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from lares.core import policy_generation as pg
from lares.core.training_pipeline import load_policy_prompt_assets


# Minimal valid GeneratedPolicy (matches new_code_output_tip reach-style; obs_dim>=7, action_dim=4)
_VALID_POLICY = """
class GeneratedPolicy(SymbolicPolicy):
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.w_move = nn.Parameter(torch.tensor(3.0))
        self.grip_bias = nn.Parameter(torch.tensor(0.0))
        self.log_std = nn.Parameter(torch.tensor(-1.0))

    def forward(self, obs):
        tcp = obs[:, 0:3]
        obj = obs[:, 4:7]
        diff = obj - tcp
        dist = torch.norm(diff, dim=-1, keepdim=True) + 1e-8
        direction = diff / dist
        move = self.w_move * direction
        grip = self.grip_bias.unsqueeze(0).expand(obs.shape[0], 1)
        mean = torch.cat([move, grip], dim=1)
        std = torch.exp(self.log_std) * torch.ones_like(mean)
        return (mean, std)

    def get_param_ranges(self):
        return {"w_move": (0.1, 10.0), "grip_bias": (-1.0, 1.0), "log_std": (-5.0, 0.0)}
"""


def _make_llm_return(content):
    ch = MagicMock()
    ch.message = MagicMock(content=content)
    usage = MagicMock()
    usage.prompt_tokens = 1
    usage.completion_tokens = 1
    usage.total_tokens = 2
    resp = MagicMock()
    resp.choices = [ch]
    resp.usage = usage
    return ([ch], 1, 1, 2)


class TestParseIdeas(unittest.TestCase):
    def test_json_array(self):
        raw = '{"ideas": ["first idea here", "second idea here", "third"]}'
        out = pg._parse_ideas_response(raw, 3)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0], "first idea here")

    def test_json_in_fence(self):
        raw = '```json\n{"ideas": ["a", "b"]}\n```'
        out = pg._parse_ideas_response(raw, 2)
        self.assertEqual(out, ["a", "b"])

    def test_pad_short(self):
        out = pg._parse_ideas_response("{}", 2)
        self.assertEqual(len(out), 2)


class TestExtractNPolicyCodes(unittest.TestCase):
    def test_two_fences(self):
        body = _VALID_POLICY.strip()
        text = f"Intro\n```python\n{body}\n```\n\n```python\n{body}\n```\n"
        codes = pg._extract_n_policy_codes(text, 2)
        self.assertEqual(len(codes), 2)
        self.assertIn("class GeneratedPolicy", codes[0])


class TestTwoPhaseIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir_path = self.tmp.name
        self.data_pkl = os.path.join(self.dir_path, "data.pkl")
        with open(self.data_pkl, "wb") as f:
            pickle.dump([{"obs": np.zeros(39)}], f)

    def _prompts(self):
        return load_policy_prompt_assets("reach-v2")

    def test_batched_two_phase(self):
        prompts = self._prompts()
        ideas_json = (
            '{"ideas": ["Hypothesis A: reach-style proportional move with smooth grip.", '
            '"Hypothesis B: two-phase gating between approach and fine positioning."]}'
        )
        body = _VALID_POLICY.strip()
        batched = f"Here are two policies.\n```python\n{body}\n```\n\n```python\n{body}\n```\n"
        # ideas call + batched impl (up to 3 retry rounds if parsing fails)
        queue = [ideas_json, batched, batched, batched]

        def fake_call_llm(client, sample_size, model, messages, temperature, **kwargs):
            self.assertTrue(queue, "unexpected extra _call_llm (exhausted canned responses)")
            return _make_llm_return(queue.pop(0))

        args = SimpleNamespace(
            model="stub-model",
            policy_gen_two_phase=True,
            policy_impl_mode=pg.POLICY_IMPL_BATCHED,
        )

        with patch.object(pg, "_call_llm", side_effect=fake_call_llm):
            with patch.object(
                pg,
                "_run_policy_validation_subprocess",
                return_value=(True, "Success!", "full"),
            ):
                pop, codes, resps = pg.get_symbolic_policies(
                    client=MagicMock(),
                    dir_path=self.dir_path,
                    llm_iter=0,
                    args=args,
                    obs_dim=39,
                    action_dim=4,
                    initial_system=prompts["initial_system"],
                    initial_user=prompts["initial_user"],
                    task_description=prompts["task_description"],
                    obs_description=prompts["obs_description"],
                    input_dict_string=prompts["input_dict_string"],
                    code_output_tip=prompts["code_output_tip"],
                    data_pkl_path=self.data_pkl,
                    real_num=2,
                    max_total_attempts=20,
                    ideas_system=prompts["ideas_system"],
                    ideas_user=prompts["ideas_user"],
                )
        self.assertEqual(len(pop), 2)
        self.assertEqual(len(codes), 2)
        self.assertEqual(len(resps), 2)
        ideas_path = os.path.join(self.dir_path, "Iter_0_ideas.json")
        self.assertTrue(os.path.isfile(ideas_path))

    def test_per_idea_two_phase(self):
        prompts = self._prompts()
        ideas_json = (
            '{"ideas": ["First distinct control approach for the arm.", '
            '"Second distinct control approach with different gating."]}'
        )
        body = _VALID_POLICY.strip()
        one = f"```python\n{body}\n```"
        queue = [ideas_json, one, one]

        def fake_call_llm(client, sample_size, model, messages, temperature, **kwargs):
            return _make_llm_return(queue.pop(0))

        args = SimpleNamespace(
            model="stub-model",
            policy_gen_two_phase=True,
            policy_impl_mode=pg.POLICY_IMPL_PER_IDEA,
        )

        with patch.object(pg, "_call_llm", side_effect=fake_call_llm):
            with patch.object(
                pg,
                "_run_policy_validation_subprocess",
                return_value=(True, "Success!", "full"),
            ):
                pop, codes, resps = pg.get_symbolic_policies(
                    client=MagicMock(),
                    dir_path=self.dir_path,
                    llm_iter=1,
                    args=args,
                    obs_dim=39,
                    action_dim=4,
                    initial_system=prompts["initial_system"],
                    initial_user=prompts["initial_user"],
                    task_description=prompts["task_description"],
                    obs_description=prompts["obs_description"],
                    input_dict_string=prompts["input_dict_string"],
                    code_output_tip=prompts["code_output_tip"],
                    data_pkl_path=self.data_pkl,
                    real_num=2,
                    max_total_attempts=30,
                    ideas_system=prompts["ideas_system"],
                    ideas_user=prompts["ideas_user"],
                )
        self.assertEqual(len(pop), 2)


if __name__ == "__main__":
    unittest.main()
