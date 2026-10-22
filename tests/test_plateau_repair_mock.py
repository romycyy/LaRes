"""R-037's mocked end-to-end gate: plateau-triggered trace repair through ``scripts/run_full_evolution.py``.

Everything except the LLM, the subprocess harness, MetaWorld and the scripted expert is real: the entry point, its
config loading and overrides, task-manifest construction and freezing, the orchestrator, single-phase generation, the
24-candidate rollout search, screening, strict paired promotion, the plateau trigger, demonstration collection and
qualification, the trace-repair slot with its linked correction, records, lineage, the ledger and the accounting.

The simulator is a kinematic lift world in the MetaWorld observation layout: the object moves only while the gripper
is closed near it, and the goal is 10 cm above the table, so a controller that never closes its gripper can never
succeed whatever its parameters (a plateau the search cannot leave), and one that grasps does.

Run from the project root::

    python -m unittest tests.test_plateau_repair_mock
"""

import copy
import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from lares.core import policy_generation as pg  # noqa: E402
from lares.core.obs_schema import get_obs_schema  # noqa: E402
from lares.eval.manifest import EvaluationManifest, synthetic_manifest  # noqa: E402
from lares.repair import traces as trace_source  # noqa: E402
from lares.search import STATUS_INVALID, LLMCallLedger, SchemaError, load_records  # noqa: E402
from lares.search import lanes  # noqa: E402
from lares.search.schemas import IDEA_ORIGIN_CODE, LANE_EXPLORATION, LANE_TRACE_REPAIR, TRACE_REPAIR_LABEL  # noqa: E402
from test_two_phase_policy_generation import POLICY_RAW, ScriptedLLM, fence  # noqa: E402

SCHEMA = get_obs_schema("assembly-v2")
HORIZON = 40
CONFIG = os.path.join(_PROJECT_ROOT, "config", "assembly_two_lane_repair.yaml")

#: Never closes its gripper: no parameter value can lift the object, so every search candidate fails.
POLICY_PUSHER = """
class GeneratedPolicy(SymbolicPolicy):
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.k_reach = nn.Parameter(torch.tensor(4.0))
        self.k_goal = nn.Parameter(torch.tensor(2.0))

    def forward(self, obs):
        tcp = self.obs_field(obs, "tcp")
        obj = self.obs_field(obs, "obj")
        goal = self.obs_field(obs, "goal")
        move = self.k_reach * (obj - tcp) + self.k_goal * (goal - obj)
        grip = -torch.ones(obs.shape[0], 1)
        mean = torch.cat([move, grip], dim=1)
        return mean, self.fixed_std(mean)

    def get_param_ranges(self):
        return {"k_reach": (0.5, 10.0), "k_goal": (0.0, 5.0)}
""".strip()

#: Reaches, closes its gripper near the object and carries it to the goal.
POLICY_LIFTER = """
class GeneratedPolicy(SymbolicPolicy):
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.k_reach = nn.Parameter(torch.tensor(20.0))
        self.k_carry = nn.Parameter(torch.tensor(20.0))
        self.radius = nn.Parameter(torch.tensor(0.02))

    def forward(self, obs):
        tcp = self.obs_field(obs, "tcp")
        obj = self.obs_field(obs, "obj")
        goal = self.obs_field(obs, "goal")
        dist = torch.norm(obj - tcp, dim=-1, keepdim=True)
        near = torch.sigmoid((self.radius - dist) * 400.0)
        move = (1 - near) * self.k_reach * (obj - tcp) + near * self.k_carry * (goal - obj)
        grip = 2 * near - 1
        mean = torch.cat([move, grip], dim=1)
        return mean, self.fixed_std(mean)

    def get_param_ranges(self):
        return {"k_reach": (10.0, 40.0), "k_carry": (10.0, 40.0), "radius": (0.015, 0.03)}
""".strip()


class LiftWorld:
    """Kinematic pick-and-lift in the MetaWorld layout; success is the object within 3 cm of a raised goal."""

    def __init__(self):
        self.action_space = SimpleNamespace(shape=(4,), high=np.ones(4, dtype=np.float32), low=-np.ones(4, dtype=np.float32))
        self.observation_space = SimpleNamespace(shape=(39,))

    def reset(self, case=None):
        rng = np.random.default_rng(0 if case is None else case.reset_seed)
        self.tcp = np.array([0.0, 0.55, 0.10]) + np.r_[rng.uniform(-0.03, 0.03, 2), 0.0]
        self.obj = np.array([0.0, 0.62, 0.02]) + np.r_[rng.uniform(-0.02, 0.02, 2), 0.0]
        self.goal = self.obj + np.array([0.0, 0.0, 0.10])
        self.t = 0
        return self._obs(), {}

    def _obs(self):
        obs = np.zeros(39, dtype=np.float32)
        for name, value in (("tcp", self.tcp), ("obj", self.obj), ("goal", self.goal)):
            f = SCHEMA.field(name)
            obs[f.start:f.stop] = value
        return obs

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        held = a[3] > 0 and np.linalg.norm(self.tcp - self.obj) < 0.03
        delta = 0.01 * a[:3]
        self.tcp = self.tcp + delta
        if held:
            self.obj = self.obj + delta
        self.t += 1
        d = float(np.linalg.norm(self.obj - self.goal))
        info = {"success": float(d < 0.03), "obj_to_target": d, "near_object": float(held), "grasp_success": float(held)}
        return self._obs(), -d, self.t >= HORIZON, info

    def close(self):
        pass


class LiftExpert:
    """The mock's scripted expert: reach, grasp, carry. ``grasp=False`` never closes (fails qualification)."""

    action_mode = "deterministic"

    def __init__(self, grasp=True):
        self.grasp = grasp
        self.name = "expert:mock"

    def reset(self, case):
        pass

    def act(self, obs):
        o = np.asarray(obs, dtype=np.float64)
        tcp, obj, goal = (o[SCHEMA.field(n).start:SCHEMA.field(n).stop] for n in ("tcp", "obj", "goal"))
        if np.linalg.norm(obj - tcp) > 0.02:
            return np.clip(np.r_[20 * (obj - tcp), -1.0], -1, 1)
        return np.clip(np.r_[20 * (goal - obj), 1.0 if self.grasp else -1.0], -1, 1)


def placed_manifest(manifest_id, n, label):
    """A synthetic manifest whose cases carry distinct placement hashes, as a built one would."""
    m = synthetic_manifest(n, label=label)
    return replace(m, manifest_id=manifest_id,
                   episodes=[replace(c, task_hash=f"{label}-{i:04d}") for i, c in enumerate(m.episodes)])


def fake_build_manifest(env_name, split, seeds, horizon=150, num_episodes=0, manifest_id=None, **_):
    return placed_manifest(manifest_id, num_episodes, f"{env_name}-s{seeds[0]}"), None


class GateRun:
    """One full ``run_full_evolution.main()`` in a temporary directory, with the mocks R-037 names."""

    def __init__(self, replies, env_name="assembly-v2", repair="enabled", expert=None, seed=3900, task_manifests=None):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        with open(CONFIG, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg.update(env_name=env_name, log_dir=os.path.join(root, "run"))
        cfg["task_manifests"] = dict(cfg["task_manifests"], manifest_dir=os.path.join(root, "manifests"),
                                     **(task_manifests or {}))
        self.config_path = os.path.join(root, "config.yaml")
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)
        self.log_dir = cfg["log_dir"]
        self.llm = ScriptedLLM(replies)
        self.expert = expert or LiftExpert()
        self.loaded_paths, self.demo_calls = [], []
        real_load, real_collect = EvaluationManifest.load, trace_source.collect_expert_traces

        def load(path, *a, **k):
            self.loaded_paths.append(str(path))
            return real_load(path, *a, **k)

        def collect(*a, **k):
            self.demo_calls.append(a)
            return real_collect(*a, **k)

        import run_full_evolution as rfe

        argv = ["run_full_evolution.py", "--config", self.config_path, "--seed", str(seed), "--repair", repair]
        with patch.object(sys, "argv", argv), patch.dict(os.environ, {"OPENAI_API_KEY": "sk-mock-not-a-key"}), \
                patch.object(pg, "_call_llm", side_effect=self.llm), \
                patch.object(pg, "_run_policy_validation_subprocess", return_value=(True, "Success!", "full")), \
                patch("lares.eval.manifest.build_manifest", side_effect=fake_build_manifest), \
                patch.object(EvaluationManifest, "resolve_pool", lambda self: None), \
                patch.object(EvaluationManifest, "load", staticmethod(load)), \
                patch.object(rfe, "make_env", lambda cfg, pool: LiftWorld()), \
                patch("lares.eval.ExpertActor", lambda task: self.expert), \
                patch.object(trace_source, "collect_expert_traces", side_effect=collect):
            rfe.main()
        self.records = load_records(os.path.join(self.log_dir, "experiments"))
        self.ledger = LLMCallLedger(self.log_dir)
        with open(os.path.join(self.log_dir, "gen_0", "plateau_repair.json"), encoding="utf-8") as f:
            self.plateau = json.load(f)
        with open(os.path.join(self.log_dir, "run_manifest.json"), encoding="utf-8") as f:
            self.run_manifest = json.load(f)

    def by_lane(self, lane, evaluated=True):
        return [r for r in self.records if r.lineage["lane"] == lane and (r.status != STATUS_INVALID) == evaluated]

    def close(self):
        self.tmp.cleanup()


def repair_prompt(gate, record) -> str:
    """The user message of the record's first (repair) call, read back from the ledger's stored prompt."""
    call = gate.ledger.get(record.llm_call_ids[0])
    with open(os.path.join(gate.log_dir, call.prompt_path), encoding="utf-8") as f:
        messages = json.load(f)
    messages = messages.get("messages", messages) if isinstance(messages, dict) else messages
    return [m for m in messages if m["role"] == "user"][0]["content"]


def plateau_replies(repair_code=POLICY_LIFTER, corrected=False, continuation=None):
    replies = {"policy_generation": [fence(POLICY_PUSHER)] + ([fence(continuation)] if continuation else [])}
    if repair_code is not None:
        replies["trace_repair"] = [fence(POLICY_RAW if corrected else repair_code)]
        if corrected:
            replies["trace_repair_correction"] = [fence(repair_code)]
    return replies


class TestPlateauWithRepair(unittest.TestCase):
    """The plateau path, repair enabled; the repair's first draft fails validation and its one correction passes."""

    @classmethod
    def setUpClass(cls):
        cls.gate = GateRun(plateau_replies(corrected=True))

    @classmethod
    def tearDownClass(cls):
        cls.gate.close()

    def test_the_trigger_fires_on_a_searched_plateau(self):
        trigger = self.gate.plateau["trigger"]
        self.assertTrue(trigger["fired"])
        self.assertEqual((trigger["probe_successes"], trigger["candidates"], trigger["prefix_promoted"]), (0, 24, False))
        self.assertEqual(self.gate.plateau["continuation"], lanes.CONTINUATION_TRACE_REPAIR)

    def test_the_repair_edits_the_prefix_and_carries_its_lineage(self):
        prefix = self.gate.by_lane(LANE_EXPLORATION)[0]
        (repair,) = self.gate.by_lane(LANE_TRACE_REPAIR)
        meta = repair.lineage["lane_metadata"]
        self.assertEqual(meta["label"], TRACE_REPAIR_LABEL)
        self.assertEqual(repair.parent_ids, [prefix.candidate_id])
        self.assertEqual(meta["edited_candidate"]["candidate_id"], prefix.candidate_id)
        self.assertEqual(meta["edited_candidate"]["source_hash"], prefix.code["source_hash"])
        self.assertTrue(meta["trigger"]["fired"])
        self.assertTrue(meta["demonstrations"]["qualified"])
        self.assertEqual(len(meta["demonstrations"]["traces_case_ids"]), 2)
        self.assertEqual(repair.lineage["status"], "complete")
        prompt = repair_prompt(self.gate, self.gate.by_lane(LANE_TRACE_REPAIR, evaluated=False)[0])
        self.assertIn("It does not reliably seat the nut on the peg.", prompt)
        self.assertIn("Successful expert episodes after the expert lifted the nut (every 4th step", prompt)
        self.assertIn("Episode 2:", prompt)
        self.assertIn("self.k_goal * (goal - obj)", prompt)     # the edited prefix's own code
        self.assertEqual(repair.lineage["idea"]["origin"], IDEA_ORIGIN_CODE)

    def test_the_failed_draft_is_recorded_and_the_correction_links_to_it(self):
        (failed,) = self.gate.by_lane(LANE_TRACE_REPAIR, evaluated=False)
        (repair,) = self.gate.by_lane(LANE_TRACE_REPAIR)
        self.assertEqual(failed.lineage["attempt"], 0)
        self.assertEqual(repair.lineage["previous_attempt_id"], failed.experiment_id)
        self.assertEqual(failed.parent_ids, repair.parent_ids)
        self.assertEqual(self.gate.llm.labels, ["policy_generation", "trace_repair", "trace_repair_correction"])

    def test_the_repair_is_searched_screened_and_promoted_by_the_strict_rule(self):
        (repair,) = self.gate.by_lane(LANE_TRACE_REPAIR)
        self.assertEqual(len(repair.fitting["search"]["trace"]), 24)
        self.assertEqual(repair.resources["bc_gradient_steps"], 0)
        comparison = repair.evaluation["incumbent_comparison"]
        self.assertTrue(comparison["promoted"])
        self.assertGreater(comparison["paired_success"]["ci_low"], 0)
        self.assertEqual(self.gate.plateau["final_incumbent"], repair.candidate_id)

    def test_episode_and_call_accounting(self):
        acc = self.gate.plateau["accounting"]
        budget = lanes.expected_plateau_budget(lanes.PlateauRepairPlan(True), 10, 30, 5, 150)
        # Two slots of (zero-shot screen + selected screen + 24 x 5 probe + 30 promotion), plus 10 demonstrations.
        self.assertEqual(acc["learner_episodes"], {"screen": 40, "probe": 240, "expanded": 60})
        self.assertEqual(acc["demonstration_episodes"], 10)
        self.assertEqual(acc["episodes_total"], 350)
        self.assertLessEqual(acc["episodes_total"], budget["episodes_max"])
        self.assertEqual(acc["llm_calls"], 3)
        self.assertLessEqual(acc["llm_calls"], budget["llm"]["max_calls"])

    def test_no_final_test_manifest_is_touched(self):
        self.assertFalse([p for p in self.gate.loaded_paths if "final_test" in p])
        self.assertTrue(all(m["split"] == "development" for m in self.gate.run_manifest["manifests"].values()))
        for r in self.gate.records:
            for mid in (r.evaluation or {}).get("manifest_ids", []):
                self.assertNotIn("final_test", mid)

    def test_the_run_never_loads_or_builds_the_fresh_evaluation_manifest(self):
        self.assertFalse([p for p in self.gate.loaded_paths if "confirmation" in p])
        self.assertFalse(os.path.exists(os.path.join(self.gate.tmp.name, "manifests", "assembly-v2_confirmation_s3904.yaml")))

    def test_task_manifests_are_built_frozen_and_disjoint(self):
        ms = self.gate.run_manifest["manifests"]
        self.assertEqual(ms["screen"]["case_ids"][0], "assembly-v2-s3902-0000")
        self.assertEqual(len(ms["screen"]["case_ids"]), 10)
        self.assertEqual(ms["promotion"]["case_ids"][0], "assembly-v2-s3902-0010")
        self.assertEqual(len(ms["promotion"]["case_ids"]), 30)
        self.assertEqual(ms["probe"]["case_ids"], [f"assembly-v2-s3902-{i:04d}" for i in range(40, 45)])
        demo = self.gate.run_manifest["demonstration_manifest"]
        self.assertEqual(demo["manifest_id"], "assembly-v2:development_demonstration:s3903:n10")
        self.assertEqual(len(self.gate.demo_calls), 1)

    def test_every_record_is_fully_traceable_from_saved_files(self):
        from lares.search.provenance import traceability_summary

        for name, field in traceability_summary(self.gate.records, self.gate.log_dir).items():
            self.assertTrue(field["complete"], (name, field["gaps"]))

    def test_a_repair_record_without_its_trigger_is_refused(self):
        (repair,) = self.gate.by_lane(LANE_TRACE_REPAIR)
        record = copy.deepcopy(repair)
        record.lineage["lane_metadata"]["trigger"]["fired"] = False
        with self.assertRaises(SchemaError):
            record.validate()
        record = copy.deepcopy(repair)
        record.lineage["parent_ids"] = []
        record.parent_ids = []
        with self.assertRaises(SchemaError):
            record.validate()


class TestPlateauWithRepairDisabled(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gate = GateRun(plateau_replies(repair_code=None, continuation=POLICY_PUSHER), repair="disabled", seed=3901)

    @classmethod
    def tearDownClass(cls):
        cls.gate.close()

    def test_the_trigger_fires_but_the_continuation_is_a_fresh_exploration(self):
        self.assertTrue(self.gate.plateau["trigger"]["fired"])
        self.assertEqual(self.gate.plateau["continuation"], lanes.CONTINUATION_EXPLORATION)
        self.assertEqual(len(self.gate.by_lane(LANE_EXPLORATION)), 2)
        self.assertFalse(self.gate.by_lane(LANE_TRACE_REPAIR))

    def test_no_demonstration_is_ever_requested(self):
        self.assertEqual(self.gate.demo_calls, [])
        self.assertIsNone(self.gate.plateau["demonstrations"])
        self.assertEqual(self.gate.plateau["accounting"]["demonstration_episodes"], 0)
        self.assertIsNone(self.gate.run_manifest["demonstration_manifest"])

    def test_nothing_promotes_and_the_incumbent_stays_null(self):
        self.assertIsNone(self.gate.plateau["final_incumbent"])
        self.assertEqual(self.gate.plateau["accounting"]["llm_calls"], 2)


class TestNoPlateau(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        replies = {"policy_generation": [fence(POLICY_LIFTER), fence(POLICY_PUSHER)]}
        cls.gate = GateRun(replies)

    @classmethod
    def tearDownClass(cls):
        cls.gate.close()

    def test_a_searchable_prefix_does_not_trigger_and_no_demonstration_is_collected(self):
        trigger = self.gate.plateau["trigger"]
        self.assertFalse(trigger["fired"])
        self.assertTrue(trigger["probe_successes"] > 0 or trigger["prefix_promoted"])
        self.assertEqual(self.gate.plateau["continuation"], lanes.CONTINUATION_EXPLORATION)
        self.assertEqual(self.gate.demo_calls, [])
        self.assertEqual(self.gate.plateau["final_incumbent"], "gen0_cand0")


class TestFailedExpertQualification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gate = GateRun({"policy_generation": [fence(POLICY_PUSHER)]}, expert=LiftExpert(grasp=False))

    @classmethod
    def tearDownClass(cls):
        cls.gate.close()

    def test_the_run_stops_inconclusive_without_spending_the_continuation(self):
        demos = self.gate.plateau["demonstrations"]
        self.assertFalse(demos["qualified"])
        self.assertEqual(demos["successes"], 0)
        self.assertEqual(self.gate.plateau["continuation"], lanes.CONTINUATION_STOPPED)
        self.assertTrue(self.gate.plateau["outcome"].startswith("inconclusive"))
        self.assertEqual(self.gate.llm.labels, ["policy_generation"])
        self.assertEqual(len(self.gate.records), 1)


class TestSecondMappedTask(unittest.TestCase):
    """push-v2 has a scripted expert and the same schema fields, so its traces are whole-episode rows."""

    @classmethod
    def setUpClass(cls):
        cls.gate = GateRun(plateau_replies(), env_name="push-v2", task_manifests={"development_pool_seed": 3990,
                                                                                 "demonstration_pool_seed": 3991})

    @classmethod
    def tearDownClass(cls):
        cls.gate.close()

    def test_the_task_gets_its_own_manifests_and_whole_episode_traces(self):
        ms = self.gate.run_manifest["manifests"]
        self.assertTrue(ms["screen"]["case_ids"][0].startswith("push-v2-s3990"))
        (repair,) = self.gate.by_lane(LANE_TRACE_REPAIR)
        rule = repair.lineage["lane_metadata"]["trace_rule"]
        self.assertEqual(rule["extraction"], trace_source.EXTRACTION_WHOLE_EPISODE)
        prompt = repair_prompt(self.gate, repair)
        self.assertIn("It does not reliably complete the task.", prompt)
        self.assertIn("Successful expert episodes (every 4th step", prompt)
        self.assertNotIn("nut", prompt)


class TestFreezeManifests(unittest.TestCase):
    """``--freeze-manifests`` builds every manifest of a block, refuses a mismatching frozen file and runs nothing."""

    def test_freeze_builds_all_three_then_verifies(self):
        import run_full_evolution as rfe

        with tempfile.TemporaryDirectory() as root:
            with open(CONFIG, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            cfg["task_manifests"]["manifest_dir"] = os.path.join(root, "m")
            path = os.path.join(root, "c.yaml")
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f)
            argv = ["run_full_evolution.py", "--config", path, "--freeze-manifests", "--dev-pool-seed", "3912",
                    "--demo-pool-seed", "3913", "--fresh-pool-seed", "3914"]
            with patch.object(sys, "argv", argv), patch("lares.eval.manifest.build_manifest", side_effect=fake_build_manifest), \
                    patch.object(pg, "_call_llm", side_effect=AssertionError("no LLM call while freezing")):
                rfe.main()
                self.assertEqual(sorted(os.listdir(os.path.join(root, "m"))), [
                    "assembly-v2_confirmation_s3914.yaml", "assembly-v2_demonstration_s3913_n10.yaml",
                    "assembly-v2_development_s3912_n45.yaml"])
                rfe.main()                                       # a second freeze loads and checks, never rebuilds
            frozen = os.path.join(root, "m", "assembly-v2_development_s3912_n45.yaml")
            m = EvaluationManifest.load(frozen)
            replace(m, manifest_id="assembly-v2:development:s9999:n45").save(frozen)
            with patch.object(sys, "argv", argv), self.assertRaises(ValueError):
                rfe.main()


class TestTraceSource(unittest.TestCase):
    def test_an_unmapped_task_fails_explicitly(self):
        with self.assertRaises(trace_source.UnsupportedTraceTask):
            trace_source.trace_rule("not-a-task-v2")

    def test_the_assembly_request_is_byte_identical_to_r036(self):
        import compare_generation_modes as cgm

        rows = [{"t": t, "tcp": [0.1, 0.2, 0.3], "gripper": 0.5, "obj": [0.0, 0.6, 0.05], "goal": [0.0, 0.8, 0.1],
                 "action": [0.1, -0.2, 0.3, 1.0]} for t in range(12)]
        episodes = [{"case": f"c{i}", "success": 1.0, "rows": rows} for i in range(3)]
        selected = trace_source.select_traces({"episodes": episodes})
        ours = trace_source.repair_request("assembly-v2", "class GeneratedPolicy: pass\n", 10, 10, selected)
        block = cgm.trace_block({"episodes": episodes}).replace(
            "Successful expert continuations after your lift", "Successful expert episodes after the expert lifted the nut")
        theirs = cgm.WHOLE_CONTROLLER_REQUEST.replace("{successes}", "10").replace("{episodes}", "10").replace(
            "{parent}", "class GeneratedPolicy: pass") + "\n" + block
        self.assertEqual(ours, theirs)

    def test_the_plan_refuses_other_shapes(self):
        with open(CONFIG, encoding="utf-8") as f:
            block = yaml.safe_load(f)["plateau_repair"]
        lanes.load_plateau_repair_plan(block)
        for bad in ({"structure_slots": 3}, {"fittable_lane": "enabled"}, {"max_triggers": 2},
                    {"exploration_lane": {"generator": "two_phase", "treatment": "zero_shot"}}):
            with self.assertRaises(lanes.LanePlanError):
                lanes.load_plateau_repair_plan({**block, **bad})

    def test_the_verified_budget_fits_the_approved_cap(self):
        plan = lanes.PlateauRepairPlan(True)
        on = lanes.expected_plateau_budget(plan, 10, 30, 5, 150)
        off = lanes.expected_plateau_budget(lanes.PlateauRepairPlan(False), 10, 30, 5, 150)
        fresh = 2 * 2 * 30            # two incumbents per block on 30 fresh cases, two blocks
        self.assertLessEqual(2 * (on["episodes_max"] + off["episodes_max"]) + fresh, 1660)
        self.assertLessEqual(2 * (on["llm"]["max_calls"] + off["llm"]["max_calls"]), 40)


if __name__ == "__main__":
    unittest.main()
