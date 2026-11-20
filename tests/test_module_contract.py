"""R-092: the module contract of the modular arm (``lares/core/module_contract.py``).

A compliant modular controller passes; each rule broken on its own is caught with a message naming the defect; a
monolithic controller fails; and the compliant controller also passes the real generation validator on the 157-d
pen-spin schema, so the requirement text is satisfiable. The opt-in hook in ``policy_generation`` gives a contract
failure the slot's one linked correction and leaves every other path unchanged.

    python -m unittest tests.test_module_contract
"""

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
from lares.core.module_contract import MODULAR_REQUIREMENT, check_module_contract  # noqa: E402
from lares.envs.isaac_pen_spin import cases  # noqa: E402

ENV_ID = cases.STAGES[3].task_id

#: A compliant modular controller on the 157-d pen-spin schema (20 actions): two modules, one feeding the other.
MODULAR = '''
class GeneratedPolicy(SymbolicPolicy):
    MODULE_SPEC = {
        "pen_terms": {
            "inputs": ["obj_pos", "in_hand_pos", "obj_rot", "obj_angvel"],
            "outputs": ["offset", "rate"],
            "units": {"offset": "m", "rate": "scaled rad/s"},
            "dependencies": [],
        },
        "joint_targets": {
            "inputs": ["hand_dof_pos"],
            "outputs": ["targets"],
            "units": "normalised joint targets",
            "dependencies": ["pen_terms"],
        },
    }

    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.k_pos = nn.Parameter(torch.tensor(0.5))
        self.k_rate = nn.Parameter(torch.tensor(0.2))
        self.k_joint = nn.Parameter(torch.tensor(0.1))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def pen_terms(self, obs):
        rel = self.obs_field(obs, "obj_pos") - self.obs_field(obs, "in_hand_pos")
        rot = self.obs_field(obs, "obj_rot")
        w = self.obs_field(obs, "obj_angvel")
        offset = self.k_pos * rel.sum(dim=-1, keepdim=True) + 0.1 * rot.sum(dim=-1, keepdim=True)
        rate = self.k_rate * w.sum(dim=-1, keepdim=True)
        return offset, rate

    def joint_targets(self, obs, offset, rate):
        q = self.obs_field(obs, "hand_dof_pos")
        return self.k_joint * q[:, :20] + torch.tanh(offset - rate)

    def forward(self, obs):
        offset, rate = self.pen_terms(obs)
        targets = self.joint_targets(obs, offset, rate)
        mean = targets + self.bias * torch.ones(obs.shape[0], 1, device=obs.device, dtype=obs.dtype)
        return mean, self.fixed_std(mean)

    def get_param_ranges(self):
        return {"k_pos": (0.0, 2.0), "k_rate": (0.0, 1.0), "k_joint": (0.0, 1.0), "bias": (-0.5, 0.5)}
'''.strip()

MONOLITHIC = '''
class GeneratedPolicy(SymbolicPolicy):
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.k = nn.Parameter(torch.tensor(0.1))

    def forward(self, obs):
        q = self.obs_field(obs, "hand_dof_pos")
        pen = torch.cat([self.obs_field(obs, "obj_pos"), self.obs_field(obs, "obj_rot"),
                         self.obs_field(obs, "obj_angvel")], dim=1)
        mean = self.k * q[:, :20] + torch.tanh(pen.sum(dim=-1, keepdim=True))
        return mean, self.fixed_std(mean)

    def get_param_ranges(self):
        return {"k": (0.0, 1.0)}
'''.strip()


def edit(old, new, source=MODULAR):
    assert source.count(old) == 1, old
    return source.replace(old, new)


class TestContract(unittest.TestCase):
    def defects(self, source, *needles):
        problems = check_module_contract(source)
        self.assertTrue(problems, "the contract accepted a violating source")
        text = "\n".join(problems)
        for needle in needles:
            self.assertIn(needle, text)
        for p in problems:
            self.assertNotIn("\n", p)  # one line each: the correction prompt keeps the first 20 lines
        return problems

    def test_a_compliant_modular_class_passes(self):
        self.assertEqual(check_module_contract(MODULAR), [])

    def test_a_monolithic_class_fails(self):
        self.defects(MONOLITHIC, "no class-level `MODULE_SPEC`")

    def test_source_that_does_not_parse_or_has_no_class(self):
        self.defects("class GeneratedPolicy(:", "does not parse")
        self.defects("x = 1", "no class GeneratedPolicy")

    def test_the_spec_must_be_a_pure_literal(self):
        self.defects(edit('"inputs": ["hand_dof_pos"]', '"inputs": list(FIELDS)'), "not a pure literal")

    def test_at_least_two_modules(self):
        one = MODULAR.split('        "joint_targets": {')[0] + "    }\n" + MODULAR.split("        },\n    }\n")[1]
        self.defects(one, "at least 2 are required")

    def test_every_spec_key_is_required(self):
        self.defects(edit('            "units": "normalised joint targets",\n', ""), "is missing ['units']")
        self.defects(edit('"outputs": ["targets"]', '"outputs": []'), "is empty")
        self.defects(edit('"inputs": ["hand_dof_pos"]', '"inputs": "hand_dof_pos"'), "must be a list of strings")

    def test_a_declared_module_must_be_a_method(self):
        self.defects(edit('"joint_targets": {', '"joint_goals": {'), "has no method of that name")

    def test_reserved_methods_cannot_be_modules(self):
        self.defects(edit('"joint_targets": {', '"forward": {'), "`forward` cannot be a module")

    def test_reads_must_equal_the_declared_inputs(self):
        p = self.defects(edit('"inputs": ["hand_dof_pos"]', '"inputs": ["hand_dof_pos", "hand_dof_vel"]'),
                         "module `joint_targets` reads the observation fields ['hand_dof_pos'] but declares the inputs "
                         "['hand_dof_pos', 'hand_dof_vel']")
        self.assertEqual(len(p), 1)
        self.defects(edit('"inputs": ["obj_pos", "in_hand_pos", "obj_rot", "obj_angvel"]',
                          '"inputs": ["obj_pos", "obj_rot", "obj_angvel"]'), "module `pen_terms` reads")

    def test_a_field_name_must_be_a_literal(self):
        self.defects(edit('self.obs_field(obs, "obj_rot")', 'self.obs_field(obs, "obj_" + "rot")'),
                     "not a string literal")

    def test_forward_may_not_read_the_observation(self):
        src = edit("        mean = targets +", '        v = self.obs_field(obs, "hand_dof_vel")\n        mean = v[:, :20] * 0 + targets +')
        self.defects(src, "`forward` calls `self.obs_field`")

    def test_any_other_use_of_obs_in_forward_is_a_bypass(self):
        for bad in ("mean = targets + obs.sum(dim=-1, keepdim=True) * 0 + self.bias",
                    "mean = targets + torch.tanh(obs).mean(dim=-1, keepdim=True) + self.bias",
                    "x = obs\n        mean = targets + self.bias",
                    "mean = targets + self.helper(obs) + self.bias",
                    "mean = targets + (obs * 0.0).sum(dim=-1, keepdim=True) + self.bias"):
            with self.subTest(bad=bad):
                src = edit("mean = targets + self.bias * torch.ones(obs.shape[0], 1, device=obs.device, dtype=obs.dtype)",
                           bad)
                self.defects(src, "undeclared observation-to-action bypass", "`forward`")

    def test_forward_may_not_reassign_obs(self):
        self.defects(edit("        offset, rate = self.pen_terms(obs)", "        obs = self.pen_terms(obs)[0]\n"
                          "        offset, rate = self.pen_terms(obs)"), "reassigns the observation")

    def test_the_observation_argument_may_have_any_name(self):
        src = MODULAR.replace("def forward(self, obs):", "def forward(self, x):").replace(
            "self.pen_terms(obs)", "self.pen_terms(x)").replace("self.joint_targets(obs, offset", "self.joint_targets(x, offset") \
            .replace("torch.ones(obs.shape[0], 1, device=obs.device, dtype=obs.dtype)", "torch.ones(x.shape[0], 1)")
        self.assertEqual(check_module_contract(src), [])
        self.defects(src.replace("torch.ones(x.shape[0], 1)", "x.mean(dim=-1, keepdim=True)"), "bypass")

    def test_keyword_arguments_are_followed(self):
        src = MODULAR.replace("self.pen_terms(obs)", "self.pen_terms(obs=obs)").replace(
            'self.obs_field(obs, "hand_dof_pos")', 'self.obs_field(obs=obs, name="hand_dof_pos")').replace(
            "self.joint_targets(obs, offset, rate)", "self.joint_targets(obs, rate=rate, offset=offset)")
        self.assertEqual(check_module_contract(src), [])
        self.defects(src.replace("self.joint_targets(obs, rate=rate, offset=offset)",
                                 "self.joint_targets(obs, rate=obs, offset=offset)"), "module `joint_targets`", "bypass")

    def test_a_module_may_use_obs_only_through_obs_field(self):
        self.defects(edit("return self.k_joint * q[:, :20] + torch.tanh(offset - rate)",
                          "return self.k_joint * q[:, :20] + torch.tanh(offset - rate) + obs.mean(dim=-1, keepdim=True)"),
                     "undeclared observation-to-action bypass", "module `joint_targets`")

    def test_a_helper_that_reads_the_observation_is_a_defect(self):
        src = edit("    def forward(self, obs):", "    def helper(self, obs):\n        return self.obs_field(obs, "
                   '"obj_linvel")\n\n    def forward(self, obs):')
        self.defects(src, "`helper` reads the observation", "not a declared module")

    def test_obs_field_may_not_be_aliased(self):
        self.defects(edit('q = self.obs_field(obs, "hand_dof_pos")', 'f = self.obs_field\n        q = f(obs, "hand_dof_pos")'),
                     "without calling it directly")

    def test_every_module_is_called_from_forward(self):
        src = edit("        targets = self.joint_targets(obs, offset, rate)", "        targets = offset - rate")
        self.defects(src, "module `joint_targets` is declared but never called from `forward`")

    def test_dependencies_must_equal_what_the_module_receives(self):
        self.defects(edit('"dependencies": ["pen_terms"]', '"dependencies": []'),
                     "module `joint_targets` declares the dependencies [] but receives the outputs of ['pen_terms']")
        self.defects(edit('            "dependencies": [],', '            "dependencies": ["joint_targets"],'),
                     "module `pen_terms` declares the dependencies ['joint_targets'] but receives the outputs of []")
        self.defects(edit('"dependencies": ["pen_terms"]', '"dependencies": ["pen_terms", "other"]'),
                     "`other`, which is not a declared module")

    def test_dependencies_follow_values_through_forward_arithmetic(self):
        src = edit("        targets = self.joint_targets(obs, offset, rate)",
                   "        mixed = 2.0 * (offset + rate)\n        targets = self.joint_targets(obs, mixed, mixed)")
        self.assertEqual(check_module_contract(src), [])
        self.defects(src.replace('"dependencies": ["pen_terms"]', '"dependencies": []'), "receives the outputs of")

    def test_a_direct_call_inside_a_module_is_a_dependency(self):
        src = edit("    def joint_targets(self, obs, offset, rate):\n", "    def joint_targets(self, obs):\n"
                   "        offset, rate = self.pen_terms(obs)\n")
        src = src.replace("self.joint_targets(obs, offset, rate)", "self.joint_targets(obs) + 0 * (offset + rate)")
        self.assertEqual(check_module_contract(src), [])
        self.defects(src.replace('"dependencies": ["pen_terms"]', '"dependencies": []'), "modules it calls itself")

    def test_dependencies_are_direct_only(self):
        three = MODULAR.replace('''        "joint_targets": {''', '''        "blend": {
            "inputs": [], "outputs": ["mean"], "units": "normalised joint targets", "dependencies": ["joint_targets"],
        },
        "joint_targets": {''').replace("    def forward(self, obs):", "    def blend(self, targets):\n"
                                         "        return torch.tanh(targets)\n\n    def forward(self, obs):") \
            .replace("mean = targets +", "mean = self.blend(targets) +")
        self.assertEqual(check_module_contract(three), [])


class TestRequirementText(unittest.TestCase):
    def test_no_role_is_hinted(self):
        low = MODULAR_REQUIREMENT.lower()
        for word in ("retention", "retain", "spin", "damp", "support", "grasp", "grip", "finger", "pen", "contact",
                     "stabil", "phase", "gait", "stroke"):
            self.assertNotRegex(low, rf"\b{word}", word)

    def test_it_states_the_schema_and_the_rules(self):
        for needle in ("MODULE_SPEC", '"inputs"', '"outputs"', '"units"', '"dependencies"', "at least two modules",
                       "obs.shape", "obs.device", "obs.dtype", "self.obs_field", "discarded", "fixed_std",
                       "get_param_ranges"):
            self.assertIn(needle, MODULAR_REQUIREMENT)

    def test_it_survives_the_prompt_template(self):
        from lares.core.training_pipeline import load_policy_prompt_assets

        prompts = load_policy_prompt_assets(ENV_ID)
        user = prompts["initial_user"].format(task=prompts["task_description"] + "\n" + MODULAR_REQUIREMENT, obs_dim=157,
                                              action_dim=20, obs_description=prompts["obs_description"],
                                              input_dict_string=prompts["input_dict_string"])
        self.assertIn(MODULAR_REQUIREMENT, user)


class TestSatisfiable(unittest.TestCase):
    """The requirement can be met: the compliant controller passes the contract and the real validator."""

    def test_the_compliant_policy_passes_the_generation_validator(self):
        policy, report = pg.instantiate_generated_policy(MODULAR, cases.OBS_DIM, cases.NUM_ACTIONS, env_name=ENV_ID)
        self.assertTrue(report.ok, [str(e) for e in report.errors])
        self.assertEqual(check_module_contract(MODULAR), [])
        self.assertIsInstance(policy.MODULE_SPEC, dict)

    def test_it_passes_the_subprocess_harness_too(self):
        with tempfile.TemporaryDirectory() as d:
            validate = pg._SlotValidator(d, None, cases.OBS_DIM, cases.NUM_ACTIONS, ENV_ID, forbid_scale_parameters=False,
                                         source_check=check_module_contract)
            policy, validation = validate(MODULAR, None, "gen0_cand0_a0")
        self.assertTrue(validation["passed"], validation["errors"])
        self.assertEqual(validation["source_check"], [])
        self.assertIsNotNone(policy)


class Scripted:
    """Replaces ``_call_llm``: queued replies, every call's messages kept."""

    def __init__(self, replies):
        self.replies, self.messages = list(replies), []

    def __call__(self, client, sample_size, model, messages, temperature, **kwargs):
        self.messages.append(messages)
        choice = MagicMock()
        choice.message = MagicMock(content=f"```python\n{self.replies.pop(0)}\n```")
        return [choice] * sample_size, 10, 5, 15


class TestGenerationHook(unittest.TestCase):
    def generate(self, replies, check):
        from lares.envs.isaac_pen_spin.block_study import make_slot_generator

        llm = Scripted(replies)
        with tempfile.TemporaryDirectory() as d, patch.object(pg, "_call_llm", side_effect=llm), \
                patch.object(pg, "_run_policy_validation_subprocess", return_value=(True, "Success!", "full")):
            gen, ledger = make_slot_generator(d, ENV_ID, MagicMock(), "stub-model", "high", 1, 2,
                                              extra_source_check=check)
            attempts = gen.exploration(0)
            return attempts, llm, len(ledger)

    def test_a_contract_failure_gets_exactly_one_correction_showing_the_defects(self):
        attempts, llm, calls = self.generate([MONOLITHIC, MODULAR], check_module_contract)
        self.assertEqual([a.valid for a in attempts], [False, True])
        self.assertEqual(attempts[0].validation["stage"], pg.STAGE_MODULE_CONTRACT)
        self.assertIsNone(attempts[0].policy)
        self.assertEqual(calls, 2)
        self.assertIn("no class-level `MODULE_SPEC`", llm.messages[1][-1]["content"])

    def test_a_slot_still_violating_after_its_correction_is_invalid(self):
        attempts, llm, calls = self.generate([MONOLITHIC, MONOLITHIC, MODULAR], check_module_contract)
        self.assertEqual([a.valid for a in attempts], [False, False])
        self.assertEqual(calls, 2)
        self.assertEqual(len(llm.replies), 1)

    def test_other_defects_and_contract_defects_reach_the_same_correction(self):
        broken = MONOLITHIC.replace('return {"k": (0.0, 1.0)}', "return {}")
        attempts, llm, _ = self.generate([broken, MODULAR], check_module_contract)
        self.assertEqual(attempts[0].validation["stage"], pg.STAGE_INSTANTIATION)
        self.assertIn("MODULE_SPEC", llm.messages[1][-1]["content"])
        self.assertGreater(len(attempts[0].validation["errors"]), len(attempts[0].validation["source_check"]))

    def test_without_the_check_nothing_changes(self):
        attempts, llm, calls = self.generate([MONOLITHIC], None)
        self.assertEqual([a.valid for a in attempts], [True])
        self.assertEqual(calls, 1)
        self.assertNotIn("source_check", attempts[0].validation)
        self.assertEqual(set(attempts[0].validation), {"passed", "stage", "errors", "validator"})


if __name__ == "__main__":
    unittest.main()
