"""R-042's mocked gate: the pen-spin keyframe-repair study through ``scripts/run_full_evolution.py --study``.

Real: the entry point and config, manifest freezing, the stream's request/response protocol, the child's policy
rebuild from source and its ``clamp(mean, -1, 1)`` execution, the D-034 sustained reducer, the orchestrator with its
24-candidate rollout search, screening, strict promotion, the plateau trigger and the R-042 fork (prefix state, copy,
continuation arms), the demonstration gate, both keyframe rules and the trace packet, records and lineage, fresh
evaluation and the block-paired summary. Mocked: Isaac (a toy pen world in the real 157-d layout behind the same
child protocol), the LLM, and the policy-validation subprocess.

Toy world: the pen's roll rate follows ``w <- 0.9 w + 10 (a[FFJ2] - a[MFJ2])``, so a controller that moves both
knuckles together never spins it whatever its parameters (a plateau); a wrist command beyond 0.9 drops the pen.

    python -m unittest tests.test_pen_spin_pipeline
"""

import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from lares.core import policy_generation as pg  # noqa: E402
from lares.core.obs_schema import get_obs_schema  # noqa: E402
from lares.envs.isaac_pen_spin import cases  # noqa: E402
from lares.envs.isaac_pen_spin import pipeline_backend as pb  # noqa: E402
from lares.envs.isaac_pen_spin import study  # noqa: E402
from lares.envs.isaac_pen_spin.pipeline_child import SymbolicController, build_controller  # noqa: E402
from lares.eval.manifest import EvaluationManifest  # noqa: E402
from lares.repair import pen_spin_traces as pst  # noqa: E402
from lares.search import load_records  # noqa: E402
from test_two_phase_policy_generation import ScriptedLLM, fence  # noqa: E402

ENV_ID = cases.STAGES[3].task_id
CONFIG = os.path.join(_PROJECT_ROOT, "config", "pen_spin_plateau_repair.yaml")
SL = cases.obs_slices()
FF2, MF2, WR1 = 7, 8, 0
NEUTRAL = [0.555, 0.176, 0, 0, 0, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, -1, 0, 1]

_FIELDS = f"""
        pos = self.obs_field(obs, "obj_pos")
        rot = self.obs_field(obs, "obj_rot")
        w = self.obs_field(obs, "obj_angvel")
        q = self.obs_field(obs, "hand_dof_pos")
        sense = 0.01 * torch.tanh(pos.sum(-1) + rot.sum(-1) + w.sum(-1) + q.sum(-1))
        base = torch.tensor({NEUTRAL}).expand(obs.shape[0], -1).clone()
        base[:, 1] = base[:, 1] + sense
"""

#: Moves both knuckles together, so no parameter value can roll the pen.
POLICY_FLAT = f"""
class GeneratedPolicy(SymbolicPolicy):
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.k_flex = nn.Parameter(torch.tensor(0.0))
        self.k_wrist = nn.Parameter(torch.tensor(0.2))

    def forward(self, obs):
{_FIELDS}
        base[:, 7] = self.k_flex
        base[:, 8] = self.k_flex
        base[:, 0] = base[:, 0] * 0 + self.k_wrist
        return base, self.fixed_std(base)

    def get_param_ranges(self):
        return {{"k_flex": (-0.5, 0.5), "k_wrist": (-0.5, 0.5)}}
""".strip()

#: Moves the two knuckles apart: the pen rolls at about 100 * 2 * k_flex rad/s.
POLICY_SPINNER = f"""
class GeneratedPolicy(SymbolicPolicy):
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.k_flex = nn.Parameter(torch.tensor(0.3))
        self.k_wrist = nn.Parameter(torch.tensor(0.2))

    def forward(self, obs):
{_FIELDS}
        base[:, 7] = self.k_flex
        base[:, 8] = -self.k_flex
        base[:, 0] = base[:, 0] * 0 + self.k_wrist
        return base, self.fixed_std(base)

    def get_param_ranges(self):
        return {{"k_flex": (0.2, 0.6), "k_wrist": (-0.5, 0.5)}}
""".strip()


# ---------------------------------------------------------------------------------------------------------------------
#  The fake Isaac child
# ---------------------------------------------------------------------------------------------------------------------


class ToyExpert:
    """Stands in for the PPO expert: knuckles apart, with roll-rate feedback."""

    def reset(self, ctx):
        pass

    def __call__(self, obs, t, ctx):
        a = torch.tensor(NEUTRAL, dtype=torch.float32).expand(obs.shape[0], -1).clone()
        w = obs[:, SL["obj_angvel"]][:, 0] / cases.VEL_OBS_SCALE
        a[:, FF2] = 0.3 + 0.01 * (20.0 - w)
        a[:, MF2] = -0.3
        return a


def toy_obs(w, offset, act):
    n = len(w)
    obs = np.zeros((n, 157), dtype=np.float32)
    in_hand = np.array([0.0, -0.39, 0.6])
    obs[:, SL["in_hand_pos"]] = in_hand
    obs[:, SL["obj_pos"]] = in_hand + offset
    obs[:, SL["obj_rot"]] = [1.0, 0.0, 0.0, 0.0]
    obs[:, SL["obj_angvel"].start] = 0.2 * w
    obs[:, SL["hand_dof_pos"].start:SL["hand_dof_pos"].start + 20] = act
    obs[:, SL["last_action"]] = act
    return obs


def toy_rollout(manifest, controller, device="cpu"):
    vecs = cases.resolve_placements(manifest)
    n, T = len(vecs), cases.HORIZON
    w, offset = 2.0 * vecs[:, 0], 0.01 * vecs[:, 1:4]
    act = np.zeros((n, 20), dtype=np.float32)
    ctx = SimpleNamespace(device=device, num_envs=n, control_dt=1 / 60,
                          neutral_action=torch.tensor(NEUTRAL, dtype=torch.float32).expand(n, -1).clone())
    controller.reset(ctx)
    alive, completed = np.ones(n, bool), np.zeros(n, bool)
    length = np.zeros(n, int)
    rec = {k: np.full((T, n), np.nan) for k in ("center_dist", "spin_rate", "transverse_rate", "reward")}
    pred = np.zeros((T, n), bool)
    obs_tr, act_tr = [], []
    for t in range(T - 1):
        obs = toy_obs(w, offset, act)
        a = controller(torch.as_tensor(obs), t, ctx).clamp(-1, 1).detach().numpy()
        obs_tr.append(obs)
        act_tr.append(a)
        w = 0.9 * w + 10.0 * (a[:, FF2] - a[:, MF2])
        act = a
        fall = np.abs(a[:, WR1]) > 0.9
        cd = np.linalg.norm(offset, axis=-1)
        for k, v in (("center_dist", cd), ("spin_rate", w), ("transverse_rate", np.zeros(n)), ("reward", np.abs(w))):
            rec[k][t] = np.where(alive, v, rec[k][t])
        pred[t] = alive & (cd < 0.08) & (np.abs(w) > 4.0)
        length = np.where(alive, length + 1, length)
        done = fall | (t == T - 2)
        completed = np.where(alive & done, ~fall, completed)
        alive = alive & ~done
    return {"stage": 3, "manifest_id": manifest.manifest_id, "case_ids": [c.case_id for c in manifest.episodes],
            "task_hashes": [c.task_hash for c in manifest.episodes], "max_episode_length": T, "length": length,
            "completed": completed, "pred": pred, **rec, "obs": np.stack(obs_tr, axis=1),
            "actions": np.stack(act_tr, axis=1)}


class FakeChild:
    """The request/response protocol of ``pipeline_child`` over the toy world; counts processes and episodes."""

    def __init__(self):
        self.requests = []

    def __call__(self, request_path):
        req = json.load(open(request_path))
        self.requests.append(req)
        m = EvaluationManifest.load(req["manifest_path"])
        if req["mode"] == "reset_obs":
            vecs = cases.resolve_placements(m)
            obs = toy_obs(2.0 * vecs[:, 0], 0.01 * vecs[:, 1:4], np.zeros((len(m), 20)))
            json.dump({"observations": obs.tolist(), "episodes": []}, open(req["response_path"], "w"))
            return
        spec = req["actor"]
        if spec["kind"] == "ppo":
            ctrl = ToyExpert()
            if spec.get("shadow_gait"):
                from lares.envs.isaac_pen_spin.gait_scaffold import ShadowGait

                ctrl = ShadowGait(ctrl)
        else:
            ctrl = build_controller(spec, ENV_ID)
        trace = toy_rollout(m, ctrl)
        resp = {"request_path": request_path, "episodes": pb.rows_from_trace(trace)}
        if req["record"]:
            path = req["response_path"].replace(".response.json", ".trace.npz")
            extra = {"gait_actions": ctrl.gait_actions()} if hasattr(ctrl, "gait_actions") else {}
            np.savez_compressed(path, **{k: trace[k] for k in ("obs", "actions", "length", "completed", "pred",
                                                              "spin_rate", "center_dist")}, **extra)
            resp["trace_path"] = path
        json.dump(resp, open(req["response_path"], "w"), default=float)

    def episodes(self, kind=None):
        return [r for r in self.requests if r["mode"] != "reset_obs" and (kind is None or r["actor"]["kind"] == kind)]


# ---------------------------------------------------------------------------------------------------------------------
#  Unit checks
# ---------------------------------------------------------------------------------------------------------------------


def make_policy(source):
    policy, report = pg.instantiate_generated_policy(source, 157, 20, env_name=ENV_ID)
    assert report.ok, report.errors
    return policy


class TestSchemaAndBackend(unittest.TestCase):
    def test_schema_matches_the_observation_layout(self):
        schema = get_obs_schema(ENV_ID)
        self.assertEqual(schema.obs_dim, 157)
        for name, s in SL.items():
            f = schema.field(name)
            self.assertEqual((f.start, f.stop), (s.start, s.stop))
        self.assertEqual(set(schema.required_names), {"obj_pos", "obj_rot", "obj_angvel", "hand_dof_pos"})
        from lares.envs.isaac_pen_spin.layout import STAGE_TASK_IDS

        self.assertEqual(STAGE_TASK_IDS, {k: s.task_id for k, s in cases.STAGES.items()})

    def test_cases_imports_first_without_a_cycle(self):
        import subprocess

        code = "import sys; sys.path.insert(0, '.'); import lares.envs.isaac_pen_spin.cases"
        self.assertEqual(subprocess.run([sys.executable, "-c", code], cwd=_PROJECT_ROOT).returncode, 0)

    def test_generated_policies_carry_their_source(self):
        self.assertEqual(make_policy(POLICY_SPINNER)._generated_source, POLICY_SPINNER)

    def test_actor_spec_refuses_what_cannot_be_rebuilt(self):
        from lares.eval.runner import PolicyActor

        policy = make_policy(POLICY_SPINNER)
        with self.assertRaises(ValueError):
            pb.actor_spec(PolicyActor(policy, deterministic=False))
        del policy._generated_source
        with self.assertRaises(ValueError):
            pb.actor_spec(PolicyActor(policy))

    def test_the_child_executes_the_clamped_mean(self):
        from lares.eval.runner import PolicyActor

        policy = make_policy(POLICY_SPINNER)
        with torch.no_grad():
            policy.k_wrist.fill_(3.0)  # outside its range on purpose: the executed action must still be clamped
        spec, state = pb.actor_spec(PolicyActor(policy))
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(state)
        spec["state_path"] = f.name
        ctrl = SymbolicController(spec, ENV_ID)
        ctrl.reset(SimpleNamespace(device="cpu"))
        obs = torch.as_tensor(toy_obs(np.zeros(2), np.zeros((2, 3)), np.zeros((2, 20))))
        a = ctrl(obs, 0, None)
        mean, _ = policy(obs)
        self.assertTrue(torch.allclose(a, mean.clamp(-1, 1)))
        self.assertEqual(float(a[0, 0]), 1.0)
        os.unlink(f.name)

    def test_evaluate_manifest_hands_a_whole_manifest_to_the_stream(self):
        from lares.eval.runner import PolicyActor, evaluate_manifest

        child = FakeChild()
        with tempfile.TemporaryDirectory() as tmp:
            stream = pb.IsaacPenSpinStream(3, tmp, "dev", runner=child)
            m = cases.build_manifest(3, 77, 5, label="unit")
            res = evaluate_manifest(PolicyActor(make_policy(POLICY_SPINNER)), m, stream)
            self.assertEqual(len(child.requests), 1)
            self.assertEqual([e.case_id for e in res.episodes], [c.case_id for c in m.episodes])
            self.assertEqual(res.success_rate, 1.0)
            self.assertTrue(all(e.signed_goal_progress == 1.0 and e.timed_out for e in res.episodes))
            flat = evaluate_manifest(PolicyActor(make_policy(POLICY_FLAT)), m, stream)
            self.assertEqual(flat.success_rate, 0.0)
            self.assertEqual({e.failure_label for e in flat.episodes}, {"slow_spin"})
            obs = stream.reset_observations(m)
            self.assertEqual(obs.shape, (5, 157))

    def test_failure_labels(self):
        row = {"sustained": False, "completed": False, "tail_hit_fraction": 0.0, "final_center_dist": 0.2}
        self.assertEqual(pb.failure_label(row), "dropped")
        self.assertEqual(pb.failure_label(dict(row, completed=True)), "off_centre")
        self.assertEqual(pb.failure_label(dict(row, completed=True, final_center_dist=0.01)), "slow_spin")
        self.assertIsNone(pb.failure_label(dict(row, sustained=True)))


class TestKeyframes(unittest.TestCase):
    def rows(self, L=359, spikes=(50, 52, 200)):
        rng = np.random.default_rng(0)
        out = []
        for t in range(L):
            a = np.full(20, 0.1) + (1.0 if t in spikes else 0.0) + 0.001 * rng.standard_normal(20)
            out.append({"t": t, "action": a.tolist(), "pen_angvel": [0.0, 0.0, 1.0], "pen_offset": [0, 0, 0],
                        "pen_axis": [1, 0, 0], "roll": 1.0, "hand": [0.0] * 24})
        return out

    def test_uniform(self):
        k = pst.uniform_keyframes(359)
        self.assertEqual(len(k), 16)
        self.assertEqual((k[0], k[-1]), (0, 358))
        self.assertEqual(k, sorted(set(k)))

    def test_event_keeps_endpoints_spacing_and_the_spikes(self):
        k = pst.event_keyframes(self.rows())
        self.assertEqual(len(k), 16)
        self.assertEqual((k[0], k[-1]), (0, 358))
        self.assertTrue(all(b - a >= pst.MIN_SPACING for a, b in zip(k, k[1:])))
        # The spikes change the action at 50-53 and 200-201: one step of each cluster is kept (spacing drops the rest).
        self.assertEqual(len([t for t in k if 50 <= t <= 53]), 1)
        self.assertEqual(len([t for t in k if 200 <= t <= 201]), 1)

    def test_ties_go_to_the_earlier_step(self):
        rows = [{"t": t, "action": [float(t % 2)] * 20, "pen_angvel": [0, 0, 0]} for t in range(100)]
        # Every interior score ties, so steps are taken earliest first, six apart from step 0.
        self.assertEqual(pst.event_keyframes(rows), [0] + list(range(6, 85, 6)) + [99])

    def test_both_packets_render_the_same_fields_and_count(self):
        sel = [{"case": "c0", "rows": self.rows()}, {"case": "c1", "rows": self.rows(spikes=(10,))}]
        u, mu = pst.repair_request("class X: pass", 10, 10, sel, pst.UNIFORM, parameters={"k": 0.5, "v": [1.0, 2.0]})
        e, me = pst.repair_request("class X: pass", 10, 10, sel, pst.EVENT, parameters={"k": 0.5, "v": [1.0, 2.0]})
        self.assertEqual(u.count("ACTION="), 32)
        self.assertEqual(e.count("ACTION="), 32)
        self.assertNotEqual(mu["kept_steps"], me["kept_steps"])
        self.assertIn("k=0.5", u)
        self.assertIn("v=[1, 2]", e)
        self.assertEqual(u.split("Episode 1")[0].split("```")[0], e.split("Episode 1")[0].split("```")[0])


# ---------------------------------------------------------------------------------------------------------------------
#  The whole study, mocked
# ---------------------------------------------------------------------------------------------------------------------


class StudyRun:
    def __init__(self, blocks=3):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        cfg = yaml.safe_load(open(CONFIG))
        cfg["pen_spin"]["study_dir"] = os.path.join(root, "study")
        cfg["pen_spin"]["blocks"] = cfg["pen_spin"]["blocks"][:blocks]
        cfg["log_dir"] = cfg["pen_spin"]["study_dir"]
        self.config_path = os.path.join(root, "config.yaml")
        yaml.safe_dump(cfg, open(self.config_path, "w"))
        self.study_dir = cfg["pen_spin"]["study_dir"]
        self.child = FakeChild()
        # Each block: prefix exploration, no_trace exploration, uniform repair, event repair.
        self.llm = ScriptedLLM({"policy_generation": [fence(POLICY_FLAT)] * (2 * blocks),
                                "trace_repair": [fence(POLICY_SPINNER)] * (2 * blocks)})
        import run_full_evolution as rfe

        self.rfe = rfe
        mdir = os.path.join(root, "manifests")

        def stream(cfg_, name, work):
            return pb.IsaacPenSpinStream(3, work, name, runner=self.child)

        def run_cell_reference(ps, manifest_path, out, config_path):
            m = EvaluationManifest.load(manifest_path)
            tr = toy_rollout(m, ToyExpert())
            return {"rows": pb.rows_from_trace(tr), "spin_rate": tr["spin_rate"], "center_dist": tr["center_dist"]}

        def run(cmd, check=True, **kw):
            args = cmd[cmd.index("--block"):]
            block = int(args[1])
            phase = args[3] if "--phase" in args else None
            arm = args[3] if "--arm" in args else None
            cfg_ = rfe.load_config(self.config_path)
            rfe.run_pen_spin_block(cfg_, self.config_path, block, phase, arm)
            return SimpleNamespace(returncode=0)

        argv = ["run_full_evolution.py", "--config", self.config_path, "--study", "keyframe_repair"]
        with patch.object(sys, "argv", argv), patch.dict(os.environ, {"OPENAI_API_KEY": "sk-mock-not-a-key"}), \
                patch.object(pg, "_call_llm", side_effect=self.llm), \
                patch.object(pg, "_run_policy_validation_subprocess", return_value=(True, "Success!", "full")), \
                patch.object(study, "manifest_file", lambda label, seed: cases.manifest_path(3, label, seed, mdir)), \
                patch.object(study, "run_cell_reference", run_cell_reference), \
                patch.object(rfe, "pen_spin_stream", stream), \
                patch("subprocess.run", side_effect=run):
            rfe.main()
        self.summary = json.load(open(os.path.join(self.study_dir, "summary.json")))

    def close(self):
        self.tmp.cleanup()


class TestStudy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sr = StudyRun(blocks=3)

    @classmethod
    def tearDownClass(cls):
        cls.sr.close()

    def test_parity_gate_ran_at_three_batch_sizes_and_passed(self):
        p = self.sr.summary["parity"]
        self.assertTrue(p["passes"])
        self.assertEqual(sorted(k for k in p if k.startswith("batch")), ["batch10", "batch30", "batch4"])

    def test_every_block_triggered_and_forked_into_three_arms(self):
        self.assertEqual(self.sr.summary["eligible_blocks"], 3)
        for b in range(3):
            bdir = os.path.join(self.sr.study_dir, f"block{b}")
            state = json.load(open(os.path.join(bdir, "prefix", "gen_0", "prefix_state.json")))
            self.assertTrue(state["trigger"]["fired"])
            self.assertIsNone(state["incumbent_id"])
            for arm in study.ARMS:
                self.assertTrue(os.path.isfile(os.path.join(bdir, arm, "final_incumbent.json")))

    def test_trace_arms_promote_and_the_no_trace_arm_stays_null(self):
        for blk in self.sr.summary["blocks"]:
            self.assertEqual(blk["fresh"]["no_trace"]["candidate_id"], None)
            self.assertEqual(blk["fresh"]["no_trace"]["episodes"], 0)
            for arm in ("uniform", "event"):
                self.assertEqual(blk["fresh"][arm]["success"], 1.0)
                self.assertEqual(blk["fresh"][arm]["episodes"], 30)
        self.assertEqual(self.sr.summary["uniform_minus_no_trace"]["per_block"], [1.0, 1.0, 1.0])
        self.assertEqual(self.sr.summary["paired_success_difference"], 0.0)
        self.assertTrue(self.sr.summary["conclusive"])

    def test_demonstrations_were_collected_once_and_qualified(self):
        demos = json.load(open(os.path.join(self.sr.study_dir, "demonstrations.json")))
        self.assertTrue(demos["qualified"])
        self.assertEqual(len(demos["traces_case_ids"]), 2)
        demo_requests = [r for r in self.sr.child.episodes("ppo") if ":demo:" in r["manifest_id"]]
        self.assertEqual(len(demo_requests), 1)

    def test_repair_requests_differ_only_in_kept_steps(self):
        bdir = os.path.join(self.sr.study_dir, "block0")
        u = open(os.path.join(bdir, "uniform", "gen_0", "repair_request_uniform.txt")).read()
        e = open(os.path.join(bdir, "event", "gen_0", "repair_request_event.txt")).read()
        self.assertEqual(u.count("ACTION="), 32)
        self.assertEqual(e.count("ACTION="), 32)
        self.assertEqual(u.split("Episode 1")[0], e.split("Episode 1")[0])
        self.assertNotEqual(u, e)

    def test_arm_records_keep_the_prefix_and_link_the_repair_to_it(self):
        bdir = os.path.join(self.sr.study_dir, "block0")
        prefix = json.load(open(os.path.join(bdir, "prefix", "gen_0", "prefix_state.json")))
        recs = load_records(os.path.join(bdir, "event", "experiments"))
        repairs = [r for r in recs if r.lineage["lane"] == "trace_repair"]
        self.assertTrue(repairs)
        edited = prefix["edited_candidate"]["candidate_id"]
        self.assertEqual(repairs[0].parent_ids, [edited])
        meta = repairs[0].lineage["lane_metadata"]
        self.assertEqual(meta["edited_candidate"]["candidate_id"], edited)
        self.assertTrue(meta["trigger"]["fired"])
        self.assertEqual(meta["demonstrations"]["traces_case_ids"], ["demo-0000", "demo-0001"])
        self.assertTrue(any(r.candidate_id == edited for r in recs))

    def test_every_record_in_every_arm_is_fully_traceable(self):
        from lares.search.provenance import traceability_summary

        for arm in ("prefix",) + study.ARMS:
            run_dir = os.path.join(self.sr.study_dir, "block1", arm)
            recs = load_records(os.path.join(run_dir, "experiments"))
            for name, field in traceability_summary(recs, run_dir).items():
                self.assertTrue(field["complete"], (arm, name, field["gaps"]))

    def test_every_evaluation_was_its_own_process_and_rows_match_the_manifest(self):
        for r in self.sr.child.requests:
            m = EvaluationManifest.load(r["manifest_path"])
            self.assertEqual(cases.stage_of(m), 3)
        # One request per manifest scored: 24 search settings x 4 slots per block at least.
        probe = [r for r in self.sr.child.requests if ":probe:" in r["manifest_id"]]
        self.assertGreaterEqual(len(probe), 3 * 4 * 24)
        self.assertTrue(all(len(EvaluationManifest.load(r["manifest_path"])) == 4 for r in probe))

    def test_no_llm_call_beyond_the_budget(self):
        self.assertEqual(len(self.sr.llm.labels), 4 * 3)

    def test_an_interrupted_arm_is_not_rerun(self):
        bdir = os.path.join(self.sr.study_dir, "block0", "event")
        os.rename(os.path.join(bdir, "final_incumbent.json"), os.path.join(bdir, "final_incumbent.moved"))
        try:
            with patch.object(sys, "argv", ["x", "--config", self.sr.config_path, "--study", "keyframe_repair"]), \
                    patch.object(study, "manifest_file",
                                 lambda label, seed: cases.manifest_path(3, label, seed, os.path.join(
                                     os.path.dirname(self.sr.config_path), "manifests"))), \
                    patch.object(self.sr.rfe, "pen_spin_stream",
                                 lambda c, n, w: pb.IsaacPenSpinStream(3, w, n, runner=self.sr.child)):
                with self.assertRaises(SystemExit):
                    self.sr.rfe.main()
        finally:
            os.rename(os.path.join(bdir, "final_incumbent.moved"), os.path.join(bdir, "final_incumbent.json"))


class TestFreeze(unittest.TestCase):
    def test_the_committed_config_names_disjoint_fresh_manifests(self):
        ps = yaml.safe_load(open(CONFIG))["pen_spin"]
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(study, "manifest_file", lambda label, seed: cases.manifest_path(3, label, seed, tmp)):
            built = study.freeze_manifests(ps)
            self.assertEqual(len(built), 2 + 3 * len(ps["blocks"]))
            again = study.freeze_manifests(ps)
            self.assertEqual(built, again)
        seeds = [s for _, s, _, _ in study.manifest_specs(ps)]
        self.assertEqual(sorted(seeds), list(range(4124, 4138)))


# ---------------------------------------------------------------------------------------------------------------------
#  R-045: dense expert-label fitting vs rollout search
# ---------------------------------------------------------------------------------------------------------------------

#: Two independent knuckle commands, both at 0: no roll until they move apart. The toy expert holds them near +0.3 / -0.3,
#: so fitting to its labels can reach a spinning setting.
POLICY_SPLIT = f"""
class GeneratedPolicy(SymbolicPolicy):
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.k_ff = nn.Parameter(torch.tensor(0.0))
        self.k_mf = nn.Parameter(torch.tensor(0.0))

    def forward(self, obs):
{_FIELDS}
        base[:, 7] = base[:, 7] * 0 + self.k_ff
        base[:, 8] = base[:, 8] * 0 + self.k_mf
        return base, self.fixed_std(base)

    def get_param_ranges(self):
        return {{"k_ff": (-0.6, 0.6), "k_mf": (-0.6, 0.6)}}
""".strip()


class DenseRun:
    def __init__(self, blocks=2):
        from lares.envs.isaac_pen_spin import dense_fit
        from lares.utils.hashing import sha256_file

        self.dense_fit = dense_fit
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        cfg = yaml.safe_load(open(CONFIG))
        d = cfg["dense_feedback_fit"]
        d["study_dir"] = os.path.join(root, "study")
        d["block_seeds"] = d["block_seeds"][:blocks]
        students = []
        for b in range(blocks):
            src, st = os.path.join(root, f"student{b}.py"), os.path.join(root, f"student{b}.pt")
            open(src, "w").write(POLICY_SPLIT)
            torch.save(make_policy(POLICY_SPLIT).state_dict(), st)
            students.append({"source": src, "source_sha256": sha256_file(src), "state": st,
                             "state_sha256": sha256_file(st)})
        d["students"] = students
        self.cfg = d
        self.config_path = os.path.join(root, "config.yaml")
        yaml.safe_dump(cfg, open(self.config_path, "w"))
        self.study_dir = d["study_dir"]
        self.child = FakeChild()
        self.mdir = os.path.join(root, "manifests")
        self.run()
        self.summary = json.load(open(os.path.join(self.study_dir, "summary.json")))

    def run(self):
        import run_full_evolution as rfe

        def stream(cfg_, name, work):
            return pb.IsaacPenSpinStream(3, work, name, runner=self.child)

        argv = ["run_full_evolution.py", "--config", self.config_path, "--study", "dense_feedback_fit"]
        with patch.object(sys, "argv", argv), \
                patch.object(pg, "_call_llm", side_effect=AssertionError("R-045 makes no LLM call")), \
                patch.object(self.dense_fit, "manifest_file",
                             lambda label, seed: cases.manifest_path(3, label, seed, self.mdir)), \
                patch.object(rfe, "pen_spin_stream", stream):
            rfe.main()

    def close(self):
        self.tmp.cleanup()


class TestDenseFit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dr = DenseRun(blocks=2)

    @classmethod
    def tearDownClass(cls):
        cls.dr.close()

    def test_parity_ran_twice_on_forty_cases_and_passed(self):
        p = self.dr.summary["parity"]
        self.assertTrue(p["passes"])
        self.assertEqual(p["episodes"], 80)

    def test_both_arms_probe_four_settings_on_the_same_cases(self):
        for blk in self.dr.summary["blocks"]:
            probes = {a: blk["arms"][a]["probe"] for a in self.dr.dense_fit.ARMS}
            self.assertEqual([p["label"] for p in probes["dense_fit"]],
                             ["fit_step0", "fit_step250", "fit_step500", "fit_step1000"])
            self.assertEqual([p["label"] for p in probes["rollout_search"]],
                             ["search0_original", "search1_local", "search2_global", "search3_local"])
            ids = {p["manifest_id"] for a in probes for p in probes[a]}
            self.assertEqual(len(ids), 1)
            # Step 0 and the original are the same student, so they score the same.
            self.assertEqual(probes["dense_fit"][0]["per_case"], probes["rollout_search"][0]["per_case"])
            self.assertEqual(probes["dense_fit"][0]["parameters"], probes["rollout_search"][0]["parameters"])

    def test_fitting_moves_toward_the_expert_and_wins_by_rollouts(self):
        for blk in self.dr.summary["blocks"]:
            fit = blk["arms"]["dense_fit"]
            loss = [r["mse"] for r in fit["fit"]["loss"]]
            self.assertLess(loss[-1], loss[0])
            self.assertNotEqual(fit["selected_label"], "fit_step0")
            self.assertTrue(fit["promoted"])
            self.assertGreater(blk["dense_fit"]["fresh_incumbent"], 0.5)
        self.assertGreater(self.dr.summary["paired_success_difference"], -1.0)

    def test_slices_are_disjoint_and_unpromoted_arms_score_the_null_incumbent(self):
        for blk in self.dr.summary["blocks"]:
            for a in self.dr.dense_fit.ARMS:
                arm = blk["arms"][a]
                ids = [arm["screen"]["manifest_id"], arm["promotion"]["manifest_id"], arm["fresh_selected"]["manifest_id"],
                       arm["probe"][0]["manifest_id"]]
                self.assertEqual(len(set(ids)), 4)
                if not arm["promoted"]:
                    self.assertEqual(arm["fresh_incumbent"]["success"], 0.0)

    def test_episode_budget_and_one_process_per_manifest(self):
        n = sum(len(EvaluationManifest.load(r["manifest_path"])) for r in self.dr.child.episodes())
        blocks = len(self.dr.summary["blocks"])
        self.assertEqual(n, 80 + blocks * (12 + 2 * (40 + 10 + 30 + 30)))
        self.assertEqual(len(self.dr.child.episodes()), 2 + blocks * (1 + 2 * (4 + 3)))

    def test_a_rerun_resumes_without_a_new_episode(self):
        before = len(self.dr.child.requests)
        self.dr.run()
        self.assertEqual(len(self.dr.child.requests), before)

    def test_a_changed_student_is_refused(self):
        s = dict(self.dr.cfg["students"][0], source_sha256="0" * 64)
        with self.assertRaises(ValueError):
            self.dr.dense_fit.check_student(s)

    def test_the_committed_config_pins_four_students_and_unused_seeds(self):
        d = yaml.safe_load(open(CONFIG))["dense_feedback_fit"]
        self.assertEqual(d["block_seeds"], [4170, 4171, 4172, 4173])
        self.assertEqual(d["parity_seed"], 4174)
        self.assertEqual(len(d["students"]), 4)
        for s in d["students"]:
            self.assertEqual(len(s["source_sha256"]), 64)


# ---------------------------------------------------------------------------------------------------------------------
#  R-046: gait-scaffold continuation
# ---------------------------------------------------------------------------------------------------------------------

#: A correction that reads gait_action and splits the two knuckles apart (rolls the toy pen).
POLICY_CORRECTION = f"""
class GeneratedPolicy(SymbolicPolicy):
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.k_split = nn.Parameter(torch.tensor(0.3))
        self.k_follow = nn.Parameter(torch.tensor(0.0))

    def forward(self, obs):
{_FIELDS}
        g = self.obs_field(obs, "gait_action")
        delta = torch.zeros_like(g)
        delta[:, 1] = sense
        delta[:, 7] = self.k_split + self.k_follow * g[:, 7]
        delta[:, 8] = -self.k_split
        return delta, self.fixed_std(delta)

    def get_param_ranges(self):
        return {{"k_split": (0.0, 0.6), "k_follow": (-0.5, 0.5)}}
""".strip()


def toy_ctx(n):
    return SimpleNamespace(device="cpu", num_envs=n, control_dt=1 / 60,
                           neutral_action=torch.tensor(NEUTRAL, dtype=torch.float32).expand(n, -1).clone())


class TestGaitScaffold(unittest.TestCase):
    def setUp(self):
        from lares.envs.isaac_pen_spin import gait_scaffold as gs

        self.gs = gs
        rng = np.random.default_rng(0)
        self.obs = [torch.as_tensor(toy_obs(rng.normal(size=3), 0.01 * rng.normal(size=(3, 3)),
                                            np.zeros((3, 20))), dtype=torch.float32) for _ in range(5)]

    def run_ctrl(self, ctrl):
        ctx = toy_ctx(3)
        ctrl.reset(ctx)
        return [ctrl(o, t, ctx) for t, o in enumerate(self.obs)]

    def test_zero_correction_reproduces_the_unchanged_gait(self):
        gait = self.run_ctrl(self.gs.ScaffoldController("unchanged"))
        for mode in ("residual", "whole"):
            out = self.run_ctrl(self.gs.ScaffoldController(mode, lambda x: torch.zeros(x.shape[0], 20)))
            for a, b in zip(out, gait):
                torch.testing.assert_close(a, b)

    def test_residual_correction_is_clipped_to_its_bound(self):
        gait = self.run_ctrl(self.gs.ScaffoldController("unchanged"))
        out = self.run_ctrl(self.gs.ScaffoldController("residual", lambda x: torch.full((x.shape[0], 20), 5.0)))
        for a, g in zip(out, gait):
            torch.testing.assert_close(a, (g + 0.15).clamp(-1, 1))

    def test_whole_edit_can_cancel_the_gait_and_reach_any_action(self):
        target = torch.linspace(-0.9, 0.9, 20).expand(3, -1)

        def cancel(x):  # delta = target - gait_action, read from the augmented observation
            return target - x[:, 157:177]

        for a in self.run_ctrl(self.gs.ScaffoldController("whole", cancel)):
            torch.testing.assert_close(a, target)
        out = self.run_ctrl(self.gs.ScaffoldController("residual", cancel))
        self.assertFalse(all(torch.allclose(a, target) for a in out))  # the residual arm cannot

    def test_the_policy_sees_the_gait_action_it_is_corrected_from(self):
        seen = []
        gait = self.run_ctrl(self.gs.ScaffoldController("unchanged"))
        self.run_ctrl(self.gs.ScaffoldController("whole", lambda x: seen.append(x) or torch.zeros(x.shape[0], 20)))
        for x, g, o in zip(seen, gait, self.obs):
            self.assertEqual(x.shape[1], 177)
            torch.testing.assert_close(x[:, 157:], g)
            torch.testing.assert_close(x[:, :157], o)

    def test_gait_state_resets_with_every_run(self):
        ctrl = self.gs.ScaffoldController("unchanged")
        torch.testing.assert_close(torch.stack(self.run_ctrl(ctrl)), torch.stack(self.run_ctrl(ctrl)))

    def test_the_shadow_gait_logs_the_gait_without_executing_it(self):
        inner = ToyExpert()
        shadow = self.gs.ShadowGait(inner)
        out = self.run_ctrl(shadow)
        ctx = toy_ctx(3)
        for a, o, t in zip(out, self.obs, range(5)):
            torch.testing.assert_close(a, inner(o, t, ctx))
        gait = torch.stack(self.run_ctrl(self.gs.ScaffoldController("unchanged")), dim=1).numpy()
        np.testing.assert_allclose(shadow.gait_actions(), gait, rtol=0, atol=1e-6)

    def test_the_gait_is_pinned(self):
        cfg = yaml.safe_load(open(CONFIG))["gait_preserving_repair"]
        self.gs.check_gait(cfg["gait_sha256"])
        with self.assertRaises(ValueError):
            self.gs.check_gait("0" * 64)
        self.assertEqual(cfg["block_seeds"], [4190, 4191, 4192, 4193])
        self.assertEqual(cfg["parity_gate_version"], 2)

    def test_the_scaffold_schema_extends_the_stage3_layout(self):
        from lares.envs.isaac_pen_spin.layout import GAIT_SCAFFOLD_TASK_ID

        base, aug = get_obs_schema(ENV_ID), get_obs_schema(GAIT_SCAFFOLD_TASK_ID)
        self.assertEqual(aug.obs_dim, 177)
        self.assertEqual([f.name for f in aug.fields][:-1], [f.name for f in base.fields])
        self.assertFalse(aug.fields[-1].required)


class GaitRun:
    def __init__(self, blocks=2):
        from lares.envs.isaac_pen_spin import gait_repair

        self.gr = gait_repair
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        cfg = yaml.safe_load(open(CONFIG))
        g = cfg["gait_preserving_repair"]
        g["study_dir"] = os.path.join(root, "study")
        g["block_seeds"] = g["block_seeds"][:blocks]
        self.config_path = os.path.join(root, "config.yaml")
        yaml.safe_dump(cfg, open(self.config_path, "w"))
        self.study_dir = g["study_dir"]
        self.child = FakeChild()
        self.mdir = os.path.join(root, "manifests")
        self.llm = ScriptedLLM({"policy_generation": [fence(POLICY_FLAT)] * blocks,
                                "trace_repair": [fence(POLICY_CORRECTION)] * (2 * blocks)})
        self.run()
        self.summary = json.load(open(os.path.join(self.study_dir, "summary.json")))

    def run(self):
        import run_full_evolution as rfe

        from lares.envs.isaac_pen_spin.scripted_expert import frozen_gait

        def stream(cfg_, name, work):
            return pb.IsaacPenSpinStream(3, work, name, runner=self.child)

        def run_cell_gait(gcfg, manifest_path, out, config_path):
            m = EvaluationManifest.load(manifest_path)
            tr = toy_rollout(m, frozen_gait())
            rows = pb.rows_from_trace(tr)
            json.dump({"episodes": rows}, open(out, "w"), default=float)
            np.savez(out.replace(".json", ".npz"), spin_rate=tr["spin_rate"], center_dist=tr["center_dist"])
            return {"rows": rows, "spin_rate": tr["spin_rate"], "center_dist": tr["center_dist"]}

        def run(cmd, check=True, **kw):
            block = int(cmd[cmd.index("--gait-block") + 1])
            rfe.run_gait_prefix(rfe.load_config(self.config_path), block)
            return SimpleNamespace(returncode=0)

        argv = ["run_full_evolution.py", "--config", self.config_path, "--study", "gait_preserving_repair"]
        with patch.object(sys, "argv", argv), patch.dict(os.environ, {"OPENAI_API_KEY": "sk-mock-not-a-key"}), \
                patch.object(pg, "_call_llm", side_effect=self.llm), \
                patch.object(pg, "_run_policy_validation_subprocess", return_value=(True, "Success!", "full")), \
                patch.object(self.gr, "manifest_file", lambda label, seed: cases.manifest_path(3, label, seed, self.mdir)), \
                patch.object(self.gr, "run_cell_gait", run_cell_gait), \
                patch.object(rfe, "pen_spin_stream", stream), \
                patch("subprocess.run", side_effect=run):
            rfe.main()

    def close(self):
        self.tmp.cleanup()


class TestGaitStudy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gr = GaitRun(blocks=2)

    @classmethod
    def tearDownClass(cls):
        cls.gr.close()

    def test_parity_compares_the_gait_with_its_wrapped_implementation(self):
        p = self.gr.summary["parity"]
        self.assertTrue(p["passes"])
        self.assertEqual(p["version"], 2)
        self.assertEqual(p["v1"]["episodes"], 80)
        self.assertEqual(p["diverged_envs"], 0)
        gait_reqs = [r for r in self.gr.child.requests if r["actor"]["kind"] == "gait"]
        self.assertTrue(gait_reqs)

    def test_every_eligible_block_runs_three_arms_against_a_null_prefix_incumbent(self):
        for blk in self.gr.summary["blocks"]:
            self.assertTrue(blk["eligible"])
            self.assertIsNone(blk["prefix_incumbent"])
            self.assertEqual(set(blk["arms"]), set(self.gr.gr.ARMS))
            for arm in self.gr.gr.LLM_ARMS:
                a = blk["arms"][arm]
                self.assertTrue(a["valid"])
                self.assertEqual([p["label"] for p in a["probe"]][0], "search0_original")
                self.assertEqual(len(a["probe"]), 6)
            self.assertEqual(blk["arms"]["unchanged_gait"]["probe"], [])

    def test_the_two_requests_differ_only_in_the_execution_rule(self):
        for b in range(2):
            texts = {arm: open(os.path.join(self.gr.study_dir, f"block{b}", arm, "request.txt")).read()
                     for arm in self.gr.gr.LLM_ARMS}
            r, w = texts["bounded_residual"], texts["whole_edit"]
            rule_r = self.gr.gr.EXECUTION_RULE["residual"].format(bound=0.15)
            rule_w = self.gr.gr.EXECUTION_RULE["whole"].format(bound=0.15)
            self.assertEqual(r.replace(rule_r, "RULE"), w.replace(rule_w, "RULE"))
            self.assertIn("GAIT=", r)
            self.assertIn("class FeedbackGait", r)

    def test_expert_traces_carry_the_shadow_gait_on_the_expert_trajectory(self):
        ex = json.load(open(os.path.join(self.gr.study_dir, "block0", "expert.json")))
        self.assertTrue(ex["qualified"])
        self.assertEqual(len(ex["traced"]), 2)
        self.assertEqual(ex["traced_case_ids"][0], "gait-0062")  # expert slice case 10 (index 52 + 10)
        row = ex["traced"][0]["rows"][5]
        self.assertEqual(len(row["gait_action"]), 20)
        self.assertNotEqual(row["gait_action"], row["action"])

    def test_scaffold_actors_reach_the_child_with_mode_and_schema(self):
        modes = {r["actor"]["scaffold"]["mode"] for r in self.gr.child.requests if r["actor"].get("scaffold")}
        self.assertEqual(modes, {"residual", "whole"})
        for r in self.gr.child.requests:
            if r["actor"].get("scaffold"):
                self.assertEqual(r["actor"]["schema_env_id"], "Isaac-ShadowHandSpin-Stage3-GaitScaffold-v0")
                self.assertEqual(r["actor"]["obs_dim"], 177)

    def test_episode_budget(self):
        arm_eps, other = 0, 0
        for r in self.gr.child.episodes():
            n = len(EvaluationManifest.load(r["manifest_path"]))
            arm_eps += n if any(f"/{a}/" in r["manifest_path"] for a in self.gr.gr.ARMS) else 0
        blocks = len(self.gr.summary["blocks"])
        self.assertEqual(arm_eps, blocks * (70 + 2 * (6 * 8 + 70)))
        expert = [r for r in self.gr.child.requests if r["actor"].get("shadow_gait")]
        self.assertEqual(len(expert), blocks)

    def test_role_slices_are_disjoint_and_promotion_never_sees_screen_or_probe_evidence(self):
        for blk in self.gr.summary["blocks"]:
            for arm in self.gr.gr.ARMS:
                a = blk["arms"][arm]
                ids = {a["screen"]["manifest_id"], a["promotion"]["manifest_id"], a["fresh_selected"]["manifest_id"]}
                self.assertEqual(len(ids), 3)
                excluded = a["promotion_decision"]["excluded_manifest_ids"]
                self.assertIn(a["screen"]["manifest_id"], excluded)
                self.assertNotIn(a["promotion"]["manifest_id"], excluded)

    def test_a_rerun_resumes_without_a_new_episode_or_call(self):
        before, calls = len(self.gr.child.requests), len(self.gr.llm.labels)
        self.gr.run()
        self.assertEqual(len(self.gr.child.requests), before)
        self.assertEqual(len(self.gr.llm.labels), calls)


class TestWrapperGateV2(unittest.TestCase):
    def write(self, d, spin_a, spin_b, sus_a, sus_b):
        from lares.envs.isaac_pen_spin import gait_repair  # noqa: F401

        T, n = spin_a.shape
        rows = lambda sus: [{"sustained": s, "length": T} for s in sus]  # noqa: E731
        np.savez(os.path.join(d, "run_cell_gait.npz"), spin_rate=spin_b, center_dist=np.zeros((T, n)))
        json.dump({"episodes": rows(sus_b)}, open(os.path.join(d, "run_cell_gait.json"), "w"))
        np.savez(os.path.join(d, "parity_0001.trace.npz"), spin_rate=spin_a, center_dist=np.zeros((T, n)))
        json.dump({"episodes": rows(sus_a), "trace_path": os.path.join(d, "parity_0001.trace.npz")},
                  open(os.path.join(d, "parity_0001.response.json"), "w"))

    def test_late_divergence_passes_and_early_divergence_fails(self):
        from lares.envs.isaac_pen_spin.gait_repair import wrapper_gate_v2

        with tempfile.TemporaryDirectory() as d:
            a = np.ones((359, 5))
            b = a.copy()
            b[260:, 2] += 1.0
            self.write(d, a, b, [True] * 5, [True, True, False, True, True])
            g = wrapper_gate_v2(d, 150, 4)
            self.assertTrue(g["passes"])
            self.assertEqual(g["first_divergence_step"], {"2": 260})
            self.assertEqual(g["sustained_outcome_flips"], 1)
            b[100:, 3] += 1.0
            self.write(d, a, b, [True] * 5, [True] * 5)
            self.assertFalse(wrapper_gate_v2(d, 150, 4)["passes"])


if __name__ == "__main__":
    unittest.main()
