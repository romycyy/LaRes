"""R-094's mocked gate: contact-frame velocity rules versus direct joint targets
(``lares/envs/isaac_pen_spin/contact_rules.py``).

Real: the 182-d schema and ``contact_control.contact_features``, the validation harness (subprocess included, in
``TestValidationHarness``), ``actor_spec`` and the child's controller construction for both actor kinds,
``contact_gate.ContactFrameController`` and the decoder, the C1 prerequisite, manifest freezing and role slices, the
slot generator at each arm's output width, the linked correction, ``llm_pool``, the 24-candidate search, strict
promotion against the scored null, the hand-authored diagnostic and the block-paired summary. Mocked: Isaac (the toy
pen world of ``test_pen_spin_pipeline`` plus a fake simulator context whose fingertip Jacobian maps each fingertip's
world-x velocity onto one joint), the LLM, and in the study run the validation subprocess.

In the toy world the pen rolls when action 7 (FFJ2) and action 8 (MFJ2) differ. The scripted direct-joint controller
moves them together (no draw can spin). The scripted contact rule requests opposite axial velocities for ff and mf
scaled by ``k_roll`` (0 at first, so only a searched draw spins): the decoder turns them into opposite FFJ2 / MFJ2
targets. The contact arm's first slot of each block first answers with the 20-output controller (the wrong width for
its contract); the correction repairs it in block 0 and repeats it in block 1.

    python -m unittest tests.test_contact_rules
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
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from lares.core import policy_generation as pg  # noqa: E402
from lares.core.obs_schema import get_obs_schema  # noqa: E402
from lares.envs.isaac_pen_spin import cases, contact_gate, contact_rules  # noqa: E402
from lares.envs.isaac_pen_spin import contact_control as cc  # noqa: E402
from lares.envs.isaac_pen_spin import pipeline_backend as pb  # noqa: E402
from lares.envs.isaac_pen_spin.layout import CONTACT_FIELDS, CONTACT_FRAME_TASK_ID  # noqa: E402
from lares.envs.isaac_pen_spin.pipeline_child import build_controller, write_queried_actions  # noqa: E402
from lares.eval.manifest import EvaluationManifest  # noqa: E402
from lares.eval.runner import PolicyActor  # noqa: E402
from test_pen_spin_pipeline import CONFIG, NEUTRAL, POLICY_FLAT, FakeChild, toy_rollout  # noqa: E402

ENV_ID = CONTACT_FRAME_TASK_ID
SL = cases.obs_slices()
SIM_JOINTS = ["WRJ1", "WRJ0", "FFJ3", "MFJ3", "RFJ3", "LFJ4", "THJ4", "FFJ2", "MFJ2", "RFJ2", "LFJ3", "THJ3", "FFJ1",
              "MFJ1", "RFJ1", "LFJ2", "THJ2", "FFJ0", "MFJ0", "RFJ0", "LFJ1", "THJ1", "LFJ0", "THJ0"]
ACTUATED = list(range(17)) + [20, 21, 23]
#: The fake Jacobian: fingertip k's world-x velocity is 0.05 m/rad of one joint (FFJ2, MFJ2, RFJ2, LFJ3, THJ3).
TIP_JOINT = (7, 8, 9, 10, 11)
BLOCKS = 2

_SENSE = """
        pos = self.obs_field(obs, "obj_pos")
        rot = self.obs_field(obs, "obj_rot")
        w = self.obs_field(obs, "obj_angvel")
        q = self.obs_field(obs, "hand_dof_pos")
        d = self.obs_field(obs, "contact_distance")
        sense = 0.01 * torch.tanh(pos.sum(-1) + rot.sum(-1) + w.sum(-1) + q.sum(-1) + d.sum(-1))
"""

#: 20 joint targets; both knuckles move together, so no parameter value rolls the toy pen.
POLICY_DIRECT = f"""
class GeneratedPolicy(SymbolicPolicy):
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.k_flex = nn.Parameter(torch.tensor(0.0))
        self.k_wrist = nn.Parameter(torch.tensor(0.2))

    def forward(self, obs):
{_SENSE}
        base = torch.tensor({NEUTRAL}).expand(obs.shape[0], -1).clone()
        base[:, 1] = base[:, 1] + sense
        base[:, 7] = self.k_flex
        base[:, 8] = self.k_flex
        base[:, 0] = base[:, 0] * 0 + self.k_wrist
        return base, self.fixed_std(base)

    def get_param_ranges(self):
        return {{"k_flex": (-0.5, 0.5), "k_wrist": (-0.5, 0.5)}}
""".strip()

#: 15 requests; ff and mf ask for opposite axial velocities scaled by ``k_roll`` (0 at first).
POLICY_RULES = f"""
class GeneratedPolicy(SymbolicPolicy):
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.k_press = nn.Parameter(torch.tensor(20.0))
        self.k_roll = nn.Parameter(torch.tensor(0.0))

    def forward(self, obs):
{_SENSE}
        normal = -torch.clamp(self.k_press * d, -1.0, 1.0) + sense.unsqueeze(-1)
        circumferential = 0.1 * torch.tanh(self.obs_field(obs, "contact_rel_vel").reshape(obs.shape[0], 5, 3)[:, :, 1])
        sign = torch.tensor([1.0, -1.0, 0.0, 0.0, 0.0])
        axial = self.k_roll * sign.expand(obs.shape[0], -1)
        mean = torch.stack([normal, circumferential, axial], dim=-1).reshape(obs.shape[0], -1)
        return mean, self.fixed_std(mean)

    def get_param_ranges(self):
        return {{"k_press": (0.0, 100.0), "k_roll": (-1.0, 1.0)}}
""".strip()

RULES_MARK = "Your controller does not output joint targets."


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def frozen_block() -> dict:
    """R-093's frozen constants with its windows: a block of the shape a passing gate would freeze."""
    p = yaml.safe_load(read(os.path.join(_PROJECT_ROOT, "config", "shadowhand_spin_stages.yaml")))
    g = p["research_protocols"]["contact_decoder_gate"]
    return json.loads(json.dumps({**g["frozen"], "windows": g["windows"], "thresholds": g["thresholds"],
                                  "eligibility": g["eligibility"]}))


def decoder_block(root: str, verdict: str = "pass", **summary_overrides) -> dict:
    """A config ``decoder`` block with a gate summary written under ``root``."""
    frozen = frozen_block()
    sha = contact_gate.frozen_sha256(frozen)
    summary = {"study": "contact_decoder_gate", "verdict": verdict, "frozen_sha256": sha,
               "decoder_sha256": cc.DecoderConfig.from_dict(frozen["decoder"]).sha256(), **summary_overrides}
    path = os.path.join(root, f"gate_summary_{verdict}_{len(summary_overrides)}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f)
    return {"frozen": frozen, "frozen_sha256": sha, "gate_summary": path}


def make_policy(source: str, action_dim: int):
    policy, report = pg.instantiate_generated_policy(source, contact_rules.OBS_DIM, action_dim, env_name=ENV_ID,
                                                     forbid_scale_parameters=False)
    assert report.ok, report.errors
    return policy


def fake_sim(ctx):
    """Give a toy context what a trusted controller reads from the simulator (``runner.ControllerContext``)."""
    n = ctx.num_envs
    J = torch.zeros(n, 5, 3, 24)
    for k, j in enumerate(TIP_JOINT):
        J[:, k, 0, j] = 0.05
    ctx.joint_names = [f"robot0_{name}" for name in SIM_JOINTS]
    ctx.actuated_dof_indices = list(ACTUATED)
    ctx.lower_all = torch.full((n, 24), -1.57)
    ctx.upper_all = torch.full((n, 24), 1.57)
    ctx.joint_pos = lambda: torch.zeros(n, 24)
    ctx.fingertip_jacobians = lambda: J
    return ctx


class WithFakeSim:
    """A controller whose context is completed by :func:`fake_sim` at reset."""

    def __init__(self, ctrl):
        self.ctrl = ctrl

    def reset(self, ctx):
        self.ctrl.reset(fake_sim(ctx))

    def __call__(self, obs, t, ctx):
        return self.ctrl(obs, t, ctx)


class ContactChild(FakeChild):
    """``FakeChild`` plus the R-094 actor kinds: the real ``build_controller`` over the toy world and the fake sim."""

    def __call__(self, request_path):
        req = json.load(open(request_path))
        if req["mode"] in ("reset_obs", "query") or not req["actor"].get("contact_actor"):
            return super().__call__(request_path)
        self.requests.append(req)
        m = EvaluationManifest.load(req["manifest_path"])
        ctrl = build_controller(req["actor"], cases.STAGES[cases.stage_of(m)].task_id)
        trace = toy_rollout(m, WithFakeSim(ctrl))
        resp = {"request_path": request_path, "episodes": pb.rows_from_trace(trace),
                "controller_diagnostics": ctrl.pipeline_diagnostics(trace)}
        json.dump(resp, open(req["response_path"], "w"), default=float)


def synthetic_obs(n: int = 4, seed: int = 0) -> torch.Tensor:
    """Plausible 157-d observations: a pen near the origin, five fingertips a few centimetres around it."""
    g = torch.Generator().manual_seed(seed)
    obs = torch.zeros(n, 157)
    obs[:, SL["obj_pos"]] = 0.01 * torch.randn(n, 3, generator=g)
    q = torch.randn(n, 4, generator=g)
    obs[:, SL["obj_rot"]] = q / q.norm(dim=-1, keepdim=True)
    obs[:, SL["obj_linvel"]] = 0.05 * torch.randn(n, 3, generator=g)
    obs[:, SL["obj_angvel"]] = 0.2 * torch.randn(n, 3, generator=g)
    obs[:, SL["fingertip_pos"]] = 0.04 * torch.randn(n, 15, generator=g)
    tq = torch.randn(n, 5, 4, generator=g)
    obs[:, SL["fingertip_rot"]] = (tq / tq.norm(dim=-1, keepdim=True)).reshape(n, 20)
    obs[:, SL["fingertip_vel"]] = 0.1 * torch.randn(n, 30, generator=g)
    obs[:, SL["hand_dof_pos"]] = 0.3 * torch.randn(n, 24, generator=g)
    return obs


# ---------------------------------------------------------------------------------------------------------------------
#  Schema and features
# ---------------------------------------------------------------------------------------------------------------------


class TestSchemaAndFeatures(unittest.TestCase):
    def test_the_schema_is_stage3_plus_the_contact_fields(self):
        schema, base = get_obs_schema(ENV_ID), get_obs_schema(cases.STAGES[3].task_id)
        self.assertEqual((schema.obs_dim, contact_rules.OBS_DIM, contact_rules.REQUEST_DIM), (182, 182, 15))
        self.assertEqual(schema.fields[:len(base.fields)], base.fields)
        self.assertEqual(set(schema.required_names), set(base.required_names))
        start = 157
        for name, size, description in CONTACT_FIELDS:
            f = schema.field(name)
            self.assertEqual((f.start, f.stop, f.required, f.description), (start, start + size, False, description))
            start += size
        self.assertEqual([(n, s) for n, s, _ in CONTACT_FIELDS],
                         [("contact_distance", 5), ("contact_axial_coord", 5), ("contact_rel_vel", 15)])
        text = schema.describe()
        for phrase in ("proximity only", "counter-clockwise seen from the pen's +x end", "circumferential = axial x normal",
                       "normal points from the pen surface outward toward the fingertip"):
            self.assertIn(phrase, text)

    def test_other_schemas_are_untouched(self):
        self.assertEqual(get_obs_schema(cases.STAGES[3].task_id).obs_dim, 157)
        self.assertEqual(get_obs_schema("Isaac-ShadowHandSpin-Stage3-ParentScaffold-v0").obs_dim, 177)
        self.assertEqual(get_obs_schema("Isaac-ShadowHandSpin-Stage3-GaitScaffold-v0").obs_dim, 177)

    def test_features_are_the_capsule_frame_quantities(self):
        obs = synthetic_obs(6)
        feats = cc.contact_features(obs)
        self.assertEqual(tuple(feats.shape), (6, 25))
        s = cc.state_from_obs(obs)
        fr = cc.capsule_frames(s["pen_pos"], s["pen_quat"], s["tip_pos"])
        rel = cc.relative_local_velocity(cc.tip_origin_linvel(s), fr, s["pen_pos"], s["pen_linvel"], s["pen_angvel"])
        torch.testing.assert_close(feats[:, 0:5], fr["distance"])
        torch.testing.assert_close(feats[:, 5:10], fr["axial_coord"])
        torch.testing.assert_close(feats[:, 10:25].reshape(6, 5, 3), rel)
        # a wider observation (the 182-d one) gives the same features: only the first 157 columns are read
        torch.testing.assert_close(cc.contact_features(torch.cat([obs, torch.randn(6, 25)], dim=-1)), feats)
        aug = contact_rules.augmented(obs)
        self.assertEqual(tuple(aug.shape), (6, 182))
        torch.testing.assert_close(aug[:, :157], obs)
        torch.testing.assert_close(aug[:, 157:], feats)
        schema = get_obs_schema(ENV_ID)
        torch.testing.assert_close(schema.slice(aug, "contact_rel_vel"), rel.reshape(6, 15))

    def test_the_sign_convention_of_circumferential_motion(self):
        # Pen at the origin, identity orientation, rolling at +w about its +x; ff still at 8 mm above the surface on +y.
        w = 5.0
        obs = torch.zeros(1, 157)
        obs[0, SL["obj_rot"].start] = 1.0
        obs[0, SL["obj_angvel"].start] = cases.VEL_OBS_SCALE * w
        obs[0, SL["fingertip_pos"].start + 1] = 0.02
        f = cc.contact_features(obs)[0]
        self.assertAlmostEqual(float(f[0]), 0.008, places=6)                    # distance
        self.assertAlmostEqual(float(f[5]), 0.0, places=6)                      # axial coordinate
        normal, circ, axial = (float(x) for x in f[10:13])
        self.assertAlmostEqual(circ, -w * cc.PEN_RADIUS, places=6)
        self.assertAlmostEqual(normal, 0.0, places=6)
        self.assertAlmostEqual(axial, 0.0, places=6)

    def test_non_finite_values_are_zero_filled(self):
        obs = synthetic_obs(3)
        obs[0, SL["obj_pos"].start] = float("nan")
        obs[1, SL["fingertip_vel"].start + 2] = float("inf")
        obs[2, SL["obj_rot"]] = 0.0
        feats = cc.contact_features(obs)
        self.assertTrue(bool(torch.isfinite(feats).all()))
        self.assertTrue(bool(torch.isfinite(contact_rules.augmented(obs)[:, 157:]).all()))


# ---------------------------------------------------------------------------------------------------------------------
#  The real validation harness on the 182-d schema
# ---------------------------------------------------------------------------------------------------------------------


class TestValidationHarness(unittest.TestCase):
    def validate(self, source, action_dim, d):
        v = pg._SlotValidator(d, None, contact_rules.OBS_DIM, action_dim, ENV_ID, forbid_scale_parameters=False)
        return v(source, None, f"a{action_dim}")

    def test_a_15_output_rule_and_a_20_output_controller_both_pass(self):
        with tempfile.TemporaryDirectory() as d:
            for source, dim in ((POLICY_RULES, 15), (POLICY_DIRECT, 20), (contact_rules.HAND_AUTHORED_SOURCE, 15)):
                policy, validation = self.validate(source, dim, os.path.join(d, str(dim)))
                if source is contact_rules.HAND_AUTHORED_SOURCE:
                    # trusted, not a candidate: it ignores the required fields, and that is all the validator objects to
                    self.assertFalse(validation["passed"])
                    self.assertTrue(all("required" in e or "sensitiv" in e.lower() or "respond" in e.lower()
                                        for e in validation["errors"]), validation["errors"])
                    continue
                self.assertTrue(validation["passed"], validation)
                mean, std = policy(torch.zeros(3, contact_rules.OBS_DIM))
                self.assertEqual(tuple(mean.shape), (3, dim))
                self.assertEqual((policy.obs_dim, policy.action_dim), (182, dim))

    def test_the_wrong_width_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            policy, validation = self.validate(POLICY_DIRECT, 15, d)
            self.assertIsNone(policy)
            self.assertFalse(validation["passed"])


# ---------------------------------------------------------------------------------------------------------------------
#  Actor specs, the query refusal and the child's controllers
# ---------------------------------------------------------------------------------------------------------------------


class TestActorsAndControllers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.decoder = contact_rules.require_gate_pass({"decoder": decoder_block(cls.tmp.name)})

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def spec(self, source, arm, name="state.pt"):
        policy = contact_rules.contact_actor(make_policy(source, contact_rules.ACTION_DIM[arm]), arm, self.decoder)
        spec, state = pb.actor_spec(PolicyActor(policy, deterministic=True, name=arm))
        path = os.path.join(self.tmp.name, f"{arm}_{name}")
        with open(path, "wb") as f:
            f.write(state)
        spec["state_path"] = path
        return json.loads(json.dumps(spec)), policy

    def test_actor_spec_carries_kind_dims_and_decoder_hash(self):
        direct, _ = self.spec(POLICY_DIRECT, "direct_joint")
        rules, _ = self.spec(POLICY_RULES, "contact_rules")
        for spec, kind, dim in ((direct, "contact_features", 20), (rules, "contact_frame", 15)):
            self.assertEqual((spec["kind"], spec["obs_dim"], spec["action_dim"], spec["schema_env_id"]),
                             ("symbolic", 182, dim, ENV_ID))
            self.assertEqual((spec["contact_actor"]["kind"], spec["contact_actor"]["policy_output_dim"]), (kind, dim))
            self.assertEqual(spec["execution"], spec["contact_actor"]["execution"])
        self.assertNotIn("frozen", direct["contact_actor"])
        self.assertEqual(rules["contact_actor"]["frozen"], frozen_block())
        self.assertEqual(rules["contact_actor"]["frozen_sha256"], contact_gate.frozen_sha256(frozen_block()))
        self.assertIn(rules["contact_actor"]["frozen_sha256"][:16], rules["execution"])

    def test_an_unmarked_policy_has_the_old_spec(self):
        policy, report = pg.instantiate_generated_policy(POLICY_FLAT, 157, 20, env_name=cases.STAGES[3].task_id)
        self.assertTrue(report.ok)
        spec, _ = pb.actor_spec(PolicyActor(policy, deterministic=True, name="flat"))
        self.assertEqual(set(spec), {"kind", "source", "obs_dim", "action_dim", "fixed_action_scale", "execution"})
        self.assertEqual(spec["execution"], pb.EXECUTION)
        self.assertFalse(pb.is_contact_frame(PolicyActor(policy, deterministic=True, name="flat")))

    def test_a_policy_of_the_wrong_shape_cannot_be_marked(self):
        with self.assertRaises(ValueError):
            contact_rules.contact_actor(make_policy(POLICY_DIRECT, 20), "contact_rules", self.decoder)

    def test_query_actions_refuses_contact_frame_and_spawns_nothing(self):
        child = FakeChild()
        stream = pb.IsaacPenSpinStream(3, os.path.join(self.tmp.name, "q"), "q", runner=child)
        _, policy = self.spec(POLICY_RULES, "contact_rules")
        manifest = cases.build_manifest(3, 4999, 2, label="query")
        with self.assertRaisesRegex(ValueError, "cannot be queried offline"):
            stream.query_actions(PolicyActor(policy, deterministic=True, name="r"), np.zeros((2, 157), np.float32), manifest)
        self.assertEqual((child.requests, stream.calls), ([], 0))
        rules, _ = self.spec(POLICY_RULES, "contact_rules")
        with self.assertRaisesRegex(SystemExit, "cannot be queried offline"):
            write_queried_actions({"actor": rules}, 3)

    def test_query_actions_still_works_for_the_expert(self):
        child = FakeChild()
        stream = pb.IsaacPenSpinStream(3, os.path.join(self.tmp.name, "qe"), "qe", runner=child)
        manifest = cases.build_manifest(3, 4999, 2, label="query")
        acts = stream.query_actions(pb.IsaacExpertActor("x.pt", "0" * 64, 1), np.zeros((3, 157), np.float32), manifest)
        self.assertEqual(acts.shape, (3, 20))
        self.assertIsNone(stream.last_diagnostics)

    def ctx(self, n):
        return fake_sim(SimpleNamespace(device="cpu", num_envs=n, control_dt=1 / 60,
                                        neutral_action=torch.tensor(NEUTRAL, dtype=torch.float32).expand(n, -1).clone()))

    def test_the_direct_controller_is_the_clamped_policy_on_augmented_observations(self):
        spec, policy = self.spec(POLICY_DIRECT, "direct_joint")
        ctrl = build_controller(spec, cases.STAGES[3].task_id)
        self.assertIsInstance(ctrl, contact_rules.ContactFeatureController)
        obs = synthetic_obs(5)
        ctrl.reset(self.ctx(5))
        action = ctrl(obs, 0, None)
        with torch.no_grad():
            want = policy(contact_rules.augmented(obs))[0].clamp(-1.0, 1.0)
        torch.testing.assert_close(action, want)
        d = ctrl.pipeline_diagnostics({"length": np.array([1] * 5)})
        self.assertEqual((d["kind"], d["steps"]), ("contact_features", 5))
        self.assertEqual(d["output_saturation_fraction"], float((want.abs() >= 1.0).float().mean()))

    def test_the_contact_frame_controller_gives_finite_bounded_joint_targets(self):
        spec, policy = self.spec(POLICY_RULES, "contact_rules")
        with torch.no_grad():
            policy.k_roll.fill_(0.8)
        spec, _ = pb.actor_spec(PolicyActor(policy, deterministic=True, name="r"))
        path = os.path.join(self.tmp.name, "rolled.pt")
        torch.save(policy.state_dict(), path)
        spec = json.loads(json.dumps({**spec, "state_path": path}))
        ctrl = build_controller(spec, cases.STAGES[3].task_id)
        self.assertIsInstance(ctrl, contact_rules.ContactRuleController)
        self.assertEqual(ctrl.params["frozen_sha256"], self.decoder["frozen_sha256"])
        n, obs = 6, synthetic_obs(6, seed=3)
        ctx = self.ctx(n)
        ctrl.reset(ctx)
        actions = [ctrl(obs, t, ctx) for t in range(40)]
        for a in actions:
            self.assertEqual(tuple(a.shape), (n, 20))
            self.assertTrue(bool(torch.isfinite(a).all()))
            self.assertTrue(bool((a.abs() <= 1.0).all()))
        # the wrist is held at neutral; the requests reach the joints the fake Jacobian maps them to
        torch.testing.assert_close(actions[-1][:, :2], ctx.neutral_action[:, :2])
        self.assertFalse(torch.allclose(actions[-1][:, 7], actions[0][:, 7]))
        d = ctrl.pipeline_diagnostics({"length": np.array([40, 40, 40, 10, 40, 40])})
        self.assertEqual((d["kind"], d["steps"], d["frozen_sha256"]), ("contact_frame", 5 * 40 + 10,
                                                                      self.decoder["frozen_sha256"]))
        for key in ("residual_mean_mps", "residual_p95_mps", "bound_active_fraction", "any_bound_step_fraction",
                    "output_saturation_fraction", "distance_mean_m"):
            self.assertTrue(np.isfinite(d[key]), key)
        self.assertGreater(d["residual_mean_mps"], 0.0)   # the fake Jacobian cannot produce normal or circumferential motion
        self.assertEqual(ctrl.extra_trace()["residual"].shape, (40, n, 5))

    def test_non_finite_observations_still_give_bounded_actions(self):
        spec, _ = self.spec(POLICY_RULES, "contact_rules")
        ctrl = build_controller(spec, cases.STAGES[3].task_id)
        obs = synthetic_obs(3)
        obs[0, SL["obj_pos"]] = float("nan")
        obs[1, SL["fingertip_pos"]] = float("inf")
        ctx = self.ctx(3)
        ctrl.reset(ctx)
        a = ctrl(obs, 0, ctx)
        self.assertTrue(bool(torch.isfinite(a).all()) and bool((a.abs() <= 1.0).all()))

    def test_the_child_reverifies_the_decoder_hash(self):
        spec, _ = self.spec(POLICY_RULES, "contact_rules")
        spec["contact_actor"]["frozen"]["decoder"]["damping"] = 0.5
        with self.assertRaisesRegex(SystemExit, "frozen decoder block hash"):
            build_controller(spec, cases.STAGES[3].task_id)
        spec, _ = self.spec(POLICY_RULES, "contact_rules")
        spec["contact_actor"]["policy_output_dim"] = 20
        with self.assertRaises(SystemExit):
            build_controller(spec, cases.STAGES[3].task_id)

    def test_the_hand_authored_controller_is_a_frozen_contact_frame_policy(self):
        p = contact_rules.hand_authored_policy(self.decoder)
        self.assertEqual((p.obs_dim, p.action_dim, p._contact_actor["kind"]), (182, 15, "contact_frame"))
        self.assertEqual(p._generated_source, contact_rules.HAND_AUTHORED_SOURCE)
        obs = torch.zeros(2, 182)
        obs[:, 157:162] = torch.tensor([0.05, 0.01, 0.0, -0.01, 0.02])
        with torch.no_grad():
            mean = p(obs)[0].reshape(2, 5, 3)
        torch.testing.assert_close(mean[0, :, 0], torch.tensor([-1.0, -0.5, 0.0, 0.5, -1.0]))   # toward the surface
        torch.testing.assert_close(mean[0, :, 1], torch.full((5,), 0.5))                         # one roll direction
        torch.testing.assert_close(mean[0, :, 2], torch.zeros(5))


# ---------------------------------------------------------------------------------------------------------------------
#  The C1 prerequisite
# ---------------------------------------------------------------------------------------------------------------------


class TestGatePrerequisite(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_the_committed_block_is_the_protocol_with_the_qualified_decoder(self):
        c = yaml.safe_load(read(CONFIG))["contact_rules"]
        self.assertEqual(c["block_seeds"], list(range(4730, 4738)))   # eight blocks since the re-triage after D-067
        self.assertEqual(c["study_dir"], "logs/pen_spin/r094_contact_rules")
        self.assertEqual((c["max_calls_per_slot"], c["llm_workers"]), (2, 16))
        # R-096's gate qualified this block (D-067); the study verifies the hash and that gate's recorded pass.
        self.assertEqual(c["decoder"]["frozen_sha256"], "7219ec6d5075b4457a6642cb0199de17fdd9bea423a95b0eb17d20d885ede250")
        self.assertEqual(c["decoder"]["gate_summary"], "logs/pen_spin/r096_contact_settled_gate_80/summary.json")
        self.assertEqual(contact_gate.frozen_sha256(c["decoder"]["frozen"]), c["decoder"]["frozen_sha256"])
        self.assertEqual(contact_rules.manifest_file(4730), "config/manifests/pen_spin_stage3_contact_rules_s4730.yaml")
        self.assertEqual(contact_rules.ROLES, {"screen": (0, 10), "promotion": (10, 40), "probe": (40, 50),
                                               "fresh": (50, 80)})
        self.assertEqual((contact_rules.ARMS, contact_rules.PRIMARY, contact_rules.SLOTS,
                          contact_rules.SEARCH_CANDIDATES), (("direct_joint", "contact_rules"),
                                                             ("contact_rules", "direct_joint"), 4, 24))
        self.assertEqual(len(c["block_seeds"]) * (80 + 80 + 2 * contact_rules.MAX_ARM_EPISODES), 6720)
        self.assertEqual(len(c["block_seeds"]) * 2 * contact_rules.SLOTS * c["max_calls_per_slot"], 128)
        with self.assertRaisesRegex(SystemExit, "conditional on a recorded C1 pass"):
            contact_rules.require_gate_pass({**c, "decoder": {"frozen": None, "frozen_sha256": None, "gate_summary": None}})

    def test_a_placeholder_block_runs_nothing(self):
        c = dict(yaml.safe_load(read(CONFIG))["contact_rules"], study_dir=os.path.join(self.tmp.name, "study"),
                 model="gpt-6.1-sol", reasoning_effort="high",
                 decoder={"frozen": None, "frozen_sha256": None, "gate_summary": None})
        mdir = os.path.join(self.tmp.name, "manifests")

        def nothing(*a, **k):
            raise AssertionError("nothing may run before the gate passes")

        with patch.object(pg, "_call_llm", side_effect=nothing), \
                patch.object(contact_rules, "manifest_file", lambda seed: cases.manifest_path(3, "contact_rules", seed, mdir)):
            with self.assertRaises(SystemExit):
                contact_rules.run_study(c, nothing, MagicMock())
            skipped = contact_rules.freeze_manifests(c)
        self.assertEqual((skipped["frozen"], skipped["study"]), (False, "contact_rules"))
        self.assertIn("C1 pass", skipped["skipped"])
        self.assertFalse(os.path.exists(mdir))
        self.assertFalse(os.path.exists(c["study_dir"]))

    def test_each_way_the_prerequisite_can_fail(self):
        root = self.tmp.name
        good = decoder_block(root)
        self.assertEqual(contact_rules.require_gate_pass({"decoder": good})["gate_verdict"], "pass")
        cases_ = {
            "not 'pass'": decoder_block(root, verdict="fail"),
            "not 'pass' ": decoder_block(root, verdict="inconclusive"),
            "is the gate of frozen block": decoder_block(root, frozen_sha256="0" * 64),
            "records decoder": decoder_block(root, decoder_sha256="0" * 64, study="other"),
            "frozen_sha256": {**good, "frozen_sha256": "0" * 64},
            "no gate summary": {**good, "gate_summary": os.path.join(root, "missing.json")},
            "lacks": {**good, "frozen": {"decoder": good["frozen"]["decoder"]}},
            "conditional on": {**good, "gate_summary": None},
        }
        for message, block in cases_.items():
            with self.assertRaisesRegex(SystemExit, message.strip(), msg=message):
                contact_rules.require_gate_pass({"decoder": block})
        changed = json.loads(json.dumps(good))
        changed["frozen"]["decoder"]["lookahead"] = 0.2          # an edited constant no longer matches the hash
        with self.assertRaisesRegex(SystemExit, "frozen decoder block hash"):
            contact_rules.require_gate_pass({"decoder": changed})

    def test_r093s_recorded_gate_does_not_qualify(self):
        path = os.path.join(_PROJECT_ROOT, "logs", "pen_spin", "r093_contact_decoder_gate", "summary.json")
        if not os.path.exists(path):
            self.skipTest("the R-093 gate summary is not on this machine")
        g = yaml.safe_load(read(os.path.join(_PROJECT_ROOT, "config", "shadowhand_spin_stages.yaml")))
        block = {"frozen": frozen_block(), "frozen_sha256": g["research_protocols"]["contact_decoder_gate"]["frozen_sha256"],
                 "gate_summary": path}
        with self.assertRaisesRegex(SystemExit, "not 'pass'"):
            contact_rules.require_gate_pass({"decoder": block})


# ---------------------------------------------------------------------------------------------------------------------
#  Prompts
# ---------------------------------------------------------------------------------------------------------------------


class TestPrompts(unittest.TestCase):
    def test_the_arms_differ_only_in_the_output_contract(self):
        frozen = frozen_block()
        direct = contact_rules.task_notes("direct_joint", frozen)
        rules = contact_rules.task_notes("contact_rules", frozen)
        d_par, r_par = contact_rules.output_contract("direct_joint"), contact_rules.output_contract("contact_rules", frozen)
        self.assertEqual(direct, pg.TASK_NOTES[cases.STAGES[3].task_id] + "\n" + contact_rules.CONTACT_FEATURES_NOTE)
        self.assertEqual((direct.count(d_par), rules.count(r_par)), (1, 1))
        self.assertEqual(rules.replace(r_par, d_par), direct)
        self.assertTrue(d_par.startswith("Actions are 20-dimensional") and d_par.endswith("\n"))
        self.assertTrue(r_par.startswith(RULES_MARK) and r_par.endswith("\n"))

    def test_the_request_contract_states_the_frozen_decoder_as_facts(self):
        frozen = frozen_block()
        text = contact_rules.output_contract("contact_rules", frozen)
        for phrase in ("15 normalised requests", "ff, mf, rf, lf, th", "[normal, circumferential, axial]",
                       "multiplied by 0.05 m/s", "capped at 0.05 m/s", "limits them to 3 rad/s",
                       "should move with the pen rather than be pinned in the world",
                       "a request of zero holds the fingertip still in the world",        # follow_pen is false in R-093's block
                       "The two wrist joints are held", "No fingertip has a predefined role"):
            self.assertIn(phrase, " ".join(text.split()))
        following = json.loads(json.dumps(frozen))
        following["controller"].update(follow_pen=True, control_wrist=True)
        other = " ".join(contact_rules.output_contract("contact_rules", following).split())
        self.assertIn("a request of zero makes that fingertip move with the pen's surface", other)
        self.assertIn("relative to the pen-surface point nearest to it", other)
        self.assertIn("The two wrist joints are driven by the decoder", other)
        self.assertIn("should move with the pen rather than be pinned in the world", other)
        with self.assertRaises(ValueError):
            contact_rules.output_contract("modular", frozen)


# ---------------------------------------------------------------------------------------------------------------------
#  The mocked study
# ---------------------------------------------------------------------------------------------------------------------


class ArmLLM:
    """Replaces ``_call_llm``: the reply depends on the arm (whether the prompt carries the request contract)."""

    def __init__(self):
        self.rules = {"policy_generation": [POLICY_DIRECT, POLICY_RULES, POLICY_RULES, POLICY_RULES] * BLOCKS,
                      "policy_repair": [POLICY_RULES, POLICY_DIRECT]}
        self.calls = []

    def __call__(self, client, sample_size, model, messages, temperature, **kwargs):
        label = kwargs.get("transcript_label", "")
        rules = RULES_MARK in messages[1]["content"]
        self.calls.append({"label": label, "rules": rules, "messages": messages, "model": model,
                           "reasoning_effort": kwargs.get("reasoning_effort")})
        if rules:
            text = self.rules[next(k for k in self.rules if label.startswith(k))].pop(0)
        else:
            if not label.startswith("policy_generation"):
                raise AssertionError(f"the direct arm needs no correction, got {label!r}")
            text = POLICY_DIRECT
        choice = MagicMock()
        choice.message = MagicMock(content=f"```python\n{text}\n```")
        return [choice] * sample_size, 10, 5, 15


class ContactRun:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.tmp.name
        c = dict(yaml.safe_load(read(CONFIG))["contact_rules"])
        c["study_dir"] = os.path.join(root, "study")
        c["block_seeds"] = c["block_seeds"][:BLOCKS]
        c["llm_workers"] = 1  # serial, so the scripted reply order is the slot order
        c["decoder"] = decoder_block(root)
        c.update(model="gpt-6.1-sol", reasoning_effort="high")
        self.ccfg, self.study_dir, self.mdir = c, c["study_dir"], os.path.join(root, "manifests")
        self.child, self.llm = ContactChild(), ArmLLM()
        self.summary = self.run()

    def run(self):
        child = self.child
        factory = lambda stage, name, work: pb.IsaacPenSpinStream(stage, work, name, runner=child)  # noqa: E731
        with patch.object(pg, "_call_llm", side_effect=self.llm), \
                patch.object(pg, "_run_policy_validation_subprocess", return_value=(True, "Success!", "full")), \
                patch.object(contact_rules, "manifest_file",
                             lambda seed: cases.manifest_path(3, contact_rules.LABEL, seed, self.mdir)):
            return contact_rules.run_study(self.ccfg, factory, MagicMock())

    def arm(self, b, arm):
        return json.loads(read(os.path.join(self.study_dir, f"block{b}", arm, "arm.json")))

    def requests(self, kind):
        return [r for r in self.child.requests if (r["actor"].get("contact_actor") or {}).get("kind") == kind]

    def close(self):
        self.tmp.cleanup()


class TestContactRulesStudy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = ContactRun()

    @classmethod
    def tearDownClass(cls):
        cls.r.close()

    def test_manifests_are_frozen_stage3_blocks_of_eighty(self):
        for seed in self.r.ccfg["block_seeds"]:
            m = EvaluationManifest.load(cases.manifest_path(3, "contact_rules", seed, self.r.mdir))
            self.assertEqual((len(m), cases.stage_of(m)), (80, 3))

    def test_the_decoder_record_is_frozen_before_anything_else(self):
        d = json.loads(read(os.path.join(self.r.study_dir, "decoder.json")))
        self.assertEqual(d["frozen_sha256"], self.r.ccfg["decoder"]["frozen_sha256"])
        self.assertEqual((d["gate_verdict"], d["frozen"]), ("pass", frozen_block()))
        self.assertEqual(self.r.summary["decoder"]["frozen_sha256"], d["frozen_sha256"])
        first = min(os.path.getmtime(os.path.join(self.r.study_dir, f"block{b}", arm, "generated.json"))
                    for b in range(BLOCKS) for arm in contact_rules.ARMS)
        self.assertLessEqual(os.path.getmtime(os.path.join(self.r.study_dir, "decoder.json")), first)

    def test_user_prompts_differ_only_in_the_contract_and_the_output_width(self):
        frozen = frozen_block()
        d_par, r_par = contact_rules.output_contract("direct_joint"), contact_rules.output_contract("contact_rules", frozen)
        for b in range(BLOCKS):
            d = os.path.join(self.r.study_dir, f"block{b}")
            base = read(os.path.join(d, "direct_joint", "user_prompt.txt"))
            treat = read(os.path.join(d, "contact_rules", "user_prompt.txt"))
            self.assertEqual((base.count(d_par), treat.count(r_par)), (1, 1))
            for text in (base, treat):
                self.assertEqual(text.count(contact_rules.CONTACT_FEATURES_NOTE), 1)
                self.assertIn(get_obs_schema(ENV_ID).describe(), text)
                self.assertIn("Observation space: 182 dimensions", text)
            pairs = [(x, y) for x, y in zip(treat.replace(r_par, d_par).splitlines(), base.splitlines()) if x != y]
            self.assertEqual(len(treat.replace(r_par, d_par).splitlines()), len(base.splitlines()))
            # what is left is the output width the shared template prints: "Action space: N" and "(batch_size, N)"
            self.assertEqual(len(pairs), 2)
            for x, y in pairs:
                self.assertEqual(x.replace("15", "20"), y)
                self.assertTrue("Action space:" in x or "(batch_size, 15)" in x, x)
        first = {m: next(c for c in self.r.llm.calls if c["rules"] is m and c["label"].startswith("policy_generation"))
                 for m in (True, False)}
        self.assertEqual(first[True]["messages"][0], first[False]["messages"][0])  # the same system prompt
        self.assertEqual({(c["model"], c["reasoning_effort"]) for c in self.r.llm.calls}, {("gpt-6.1-sol", "high")})
        self.assertNotEqual(self.r.arm(0, "direct_joint")["output_contract_sha256"],
                            self.r.arm(0, "contact_rules")["output_contract_sha256"])

    def test_a_wrong_width_reply_gets_exactly_one_correction(self):
        repairs = [c for c in self.r.llm.calls if c["label"].startswith("policy_repair")]
        self.assertEqual(len(repairs), BLOCKS)
        self.assertTrue(all(c["rules"] for c in repairs))
        ok, bad = self.r.arm(0, "contact_rules"), self.r.arm(1, "contact_rules")
        self.assertEqual([a["valid"] for a in ok["slots"][0]["attempts"]], [False, True])
        self.assertEqual(ok["valid_candidates"], 4)
        self.assertEqual([a["valid"] for a in bad["slots"][0]["attempts"]], [False, False])
        self.assertEqual(bad["valid_candidates"], 3)
        self.assertEqual(sorted(bad["screens"]), ["1", "2", "3"])
        for b in range(BLOCKS):
            a = self.r.arm(b, "direct_joint")
            self.assertEqual(a["valid_candidates"], 4)
            self.assertTrue(all(len(s["attempts"]) == 1 for s in a["slots"]))

    def test_each_arm_runs_through_its_own_actor_kind(self):
        direct, rules = self.r.requests("contact_features"), self.r.requests("contact_frame")
        # per block: direct 4 screens + 24 + 2; rules 4 (3 in block 1) + 24 + 2, plus the hand-authored's 4 roles
        self.assertEqual(len(direct), BLOCKS * 30)
        self.assertEqual(len(rules), 30 + 29 + BLOCKS * 4)
        for r in direct:
            self.assertEqual((r["actor"]["obs_dim"], r["actor"]["action_dim"], r["actor"]["schema_env_id"]),
                             (182, 20, ENV_ID))
            self.assertNotIn("frozen", r["actor"]["contact_actor"])
        sha = self.r.ccfg["decoder"]["frozen_sha256"]
        for r in rules:
            self.assertEqual((r["actor"]["obs_dim"], r["actor"]["action_dim"], r["actor"]["schema_env_id"]),
                             (182, 15, ENV_ID))
            self.assertEqual(r["actor"]["contact_actor"]["frozen_sha256"], sha)
        for b in range(BLOCKS):
            self.assertEqual((self.r.arm(b, "direct_joint")["actor_kind"], self.r.arm(b, "contact_rules")["actor_kind"]),
                             ("contact_features", "contact_frame"))
            self.assertEqual((self.r.arm(b, "direct_joint")["policy_output_dim"],
                              self.r.arm(b, "contact_rules")["policy_output_dim"]), (20, 15))

    def test_zero_shot_is_the_screen_and_the_search_is_separate(self):
        for b, seed in enumerate(self.r.ccfg["block_seeds"]):
            for arm in contact_rules.ARMS:
                a = self.r.arm(b, arm)
                z, s = a["zero_shot"], a["search"]
                self.assertEqual(z["slot"], a["selected_slot"])
                self.assertEqual(z["screen"], a["screens"][str(a["selected_slot"])])
                self.assertEqual(len(z["screen"]["per_case"]), 10)
                self.assertEqual(z["probe_original"]["per_case"], s["trace"][0]["per_case"])
                self.assertEqual(z["screen"]["success"], 0.0)      # nothing spins at its declared initial values
                self.assertEqual((len(s["trace"]), s["seed"], s["trace"][0]["kind"]), (24, seed, "original"))
                self.assertEqual(len({t["state_sha256"] for t in s["trace"]}), 24)
                self.assertEqual(a["isaac_processes"], a["valid_candidates"] + 24 + 2)
                adir = os.path.join(self.r.study_dir, f"block{b}", arm)
                self.assertEqual(json.loads(read(os.path.join(adir, "search.json"))), s)
                state = torch.load(os.path.join(adir, "selected_state.pt"), weights_only=True)
                self.assertEqual({k: v.tolist() for k, v in state.items()}, s["selected_state"]["parameters"])
        direct, rules = self.r.arm(0, "direct_joint")["search"], self.r.arm(0, "contact_rules")["search"]
        self.assertEqual((direct["selected_index"], direct["candidates_with_probe_success"]), (0, 0))
        self.assertGreater(rules["selected_index"], 0)             # a draw of k_roll spins through the decoder
        self.assertGreater(rules["candidates_with_probe_success"], 0)

    def test_promotion_is_against_the_scored_null_and_fresh_follows_it(self):
        for blk in self.r.summary["blocks"]:
            for arm in contact_rules.ARMS:
                a = blk["arms"][arm]
                self.assertEqual(a["promotion_decision"]["incumbent_id"], "neutral_pose")
                self.assertEqual((len(a["promotion"]["per_case"]), len(a["fresh_selected"]["per_case"])), (30, 30))
                want = a["fresh_selected"]["success"] if a["promoted"] else blk["null"]["fresh"]["success"]
                self.assertEqual(a["fresh_incumbent"]["success"], want)
                self.assertEqual(a["promoted"], arm == "contact_rules")
            self.assertEqual(blk["arms"]["direct_joint"]["fresh_incumbent"]["candidate"], "null")
            self.assertEqual(blk["null"]["fresh"]["success"], 0.0)

    def test_the_hand_authored_controller_is_scored_once_per_block_and_never_ranked(self):
        for b, blk in enumerate(self.r.summary["blocks"]):
            h = blk["hand_authored"]
            self.assertEqual((h["cases"], h["isaac_episodes"], h["isaac_processes"]), (80, 80, 4))
            self.assertEqual(h, json.loads(read(os.path.join(self.r.study_dir, f"block{b}", "hand_authored.json"))))
            for role, (lo, hi) in contact_rules.ROLES.items():
                self.assertEqual(len(h[role]["per_case"]), hi - lo)
                self.assertEqual(h[role]["controller_diagnostics"]["kind"], "contact_frame")
            for arm in contact_rules.ARMS:
                a = blk["arms"][arm]
                self.assertNotIn("hand_authored", json.dumps({k: a[k] for k in ("fresh_incumbent", "promotion_decision",
                                                                                 "screens", "zero_shot")}))
                self.assertNotEqual(a["selected_code"], contact_rules.HAND_AUTHORED_SOURCE)
        hand = [r for r in self.r.requests("contact_frame") if r["actor"]["source"] == contact_rules.HAND_AUTHORED_SOURCE]
        self.assertEqual(len(hand), BLOCKS * 4)
        s = self.r.summary["hand_authored"]
        self.assertEqual(len(s["success_per_block"]), BLOCKS)
        self.assertIn("never ranked", s["role"])
        self.assertIsNotNone(s["diagnostics"]["residual_mean_mps"])

    def test_decoder_diagnostics_reach_the_records(self):
        for b in range(BLOCKS):
            rules, direct = self.r.arm(b, "contact_rules"), self.r.arm(b, "direct_joint")
            for rec in list(rules["screens"].values()) + [rules["promotion"], rules["fresh_selected"]]:
                d = rec["controller_diagnostics"]
                self.assertEqual((d["kind"], d["frozen_sha256"]), ("contact_frame", self.r.ccfg["decoder"]["frozen_sha256"]))
                for key in ("residual_mean_mps", "bound_active_fraction", "output_saturation_fraction"):
                    self.assertTrue(0.0 <= d[key] < 10.0, key)
                self.assertGreater(d["steps"], 0)
            for rec in list(direct["screens"].values()) + [direct["promotion"], direct["fresh_selected"]]:
                self.assertEqual(set(rec["controller_diagnostics"]), {"kind", "steps", "output_saturation_fraction"})
        s = self.r.summary
        self.assertIsNotNone(s["arms"]["contact_rules"]["fresh_selected_diagnostics"]["residual_mean_mps"])
        self.assertIsNone(s["arms"]["direct_joint"]["fresh_selected_diagnostics"]["residual_mean_mps"])
        self.assertIsNotNone(s["arms"]["direct_joint"]["fresh_selected_diagnostics"]["output_saturation_fraction"])
        self.assertEqual(s["decoder_diagnostics"]["contact_rules_fresh_selected"],
                         s["arms"]["contact_rules"]["fresh_selected_diagnostics"])

    def test_episode_and_call_accounting(self):
        eps = sum(len(EvaluationManifest.load(r["manifest_path"])) for r in self.r.child.episodes())
        want = BLOCKS * (80 + 80) + 3 * 340 + 330   # block 1's contact arm screens three slots
        self.assertEqual(eps, want)
        self.assertEqual(self.r.summary["isaac_episodes"], want)
        self.assertEqual(self.r.summary["isaac_processes"], len(self.r.child.requests))
        self.assertEqual([self.r.arm(b, arm)["isaac_episodes"] for b in range(BLOCKS) for arm in contact_rules.ARMS],
                         [340, 340, 340, 330])
        self.assertEqual(len(self.r.llm.calls), BLOCKS * (4 + 5))
        self.assertEqual(self.r.summary["llm_calls"], len(self.r.llm.calls))
        self.assertEqual(self.r.summary["llm_tokens"], 15 * len(self.r.llm.calls))

    def test_summary_fields(self):
        s = self.r.summary
        self.assertEqual(json.loads(read(os.path.join(self.r.study_dir, "summary.json")))["paired_success_difference"],
                         s["paired_success_difference"])
        self.assertTrue(os.path.exists(os.path.join(self.r.study_dir, "summary.partial.json")))
        self.assertEqual(s["primary"], "contact_rules_minus_direct_joint_fresh_incumbent")
        primary = s[s["primary"]]
        self.assertEqual(len(primary["per_block"]), BLOCKS)
        self.assertEqual(s["paired_success_difference"], primary["mean"])
        self.assertGreater(s["paired_success_difference"], 0.0)
        for key in s["zero_shot_contrasts"] + s["searched_contrasts"]:
            self.assertEqual(len(s[key]["per_block"]), BLOCKS, key)
        self.assertEqual(s["zero_shot_contrasts"], [
            "contact_rules_minus_direct_joint_zero_shot_screen_success",
            "contact_rules_minus_direct_joint_zero_shot_screen_tail_hit_fraction",
            "contact_rules_minus_direct_joint_zero_shot_probe_success"])
        self.assertEqual(s["contact_rules_minus_direct_joint_zero_shot_screen_success"]["per_block"], [0.0] * BLOCKS)
        self.assertEqual(s["contact_rules_minus_direct_joint_valid_slots"]["per_block"], [0.0, -1.0])
        direct, rules = s["arms"]["direct_joint"], s["arms"]["contact_rules"]
        self.assertEqual((direct["slots"], direct["valid_slots"]), (8, 8))
        self.assertEqual((rules["slots"], rules["valid_slots"], rules["first_attempt_valid_slots"]), (8, 7, 6))
        self.assertEqual(s["slots_valid_within_one_correction"], {"direct_joint": 1.0, "contact_rules": 7 / 8})
        self.assertEqual((direct["promoted_blocks"], direct["plateau_blocks"]), (0, BLOCKS))
        self.assertEqual((rules["promoted_blocks"], rules["plateau_blocks"]), (BLOCKS, 0))
        self.assertEqual((direct["zero_shot_screen_success"], rules["zero_shot_screen_success"]), (0.0, 0.0))
        self.assertGreater(rules["fresh_selected"], 0.0)
        self.assertEqual(direct["fresh_selected"], 0.0)

    def test_generation_is_saved_before_any_episode(self):
        for b in range(BLOCKS):
            for arm in contact_rules.ARMS:
                adir = os.path.join(self.r.study_dir, f"block{b}", arm)
                g = json.loads(read(os.path.join(adir, "generated.json")))
                self.assertEqual(len(g["slots"]), 4)
                self.assertLess(os.path.getmtime(os.path.join(adir, "generated.json")),
                                os.path.getmtime(os.path.join(adir, "arm.json")))

    def test_zz_a_rerun_resumes_without_a_new_episode_or_call(self):
        before, calls = len(self.r.child.requests), len(self.r.llm.calls)
        again = self.r.run()
        self.assertEqual((len(self.r.child.requests), len(self.r.llm.calls)), (before, calls))
        self.assertEqual(again["paired_success_difference"], self.r.summary["paired_success_difference"])

    def test_zz_saved_generations_rebuild_at_each_arms_width_without_a_call(self):
        from lares.envs.isaac_pen_spin import llm_pool

        def no_call(*a, **k):
            raise AssertionError("a saved generation is never regenerated")

        for arm, dim in contact_rules.ACTION_DIM.items():
            with patch.object(pg, "_call_llm", side_effect=no_call):
                g = contact_rules.generate_arm(arm, self.r.ccfg, os.path.join(self.r.study_dir, "block0"), MagicMock(),
                                               contact_rules.require_gate_pass(self.r.ccfg))
            self.assertEqual([s["valid"] for s in g["slots"]], [True] * 4)
            for s in g["slots"]:
                self.assertEqual((s["policy"].obs_dim, s["policy"].action_dim), (182, dim))
            self.assertIsNotNone(llm_pool.load_generation(os.path.join(self.r.study_dir, "block0", arm), ENV_ID,
                                                          obs_dim=182, action_dim=dim))

    def test_zz_an_interrupted_arm_is_not_rerun(self):
        adir = os.path.join(self.r.study_dir, "block0", "contact_rules")
        os.rename(os.path.join(adir, "arm.json"), os.path.join(adir, "arm.json.bak"))
        try:
            with self.assertRaises(SystemExit):
                self.r.run()
        finally:
            os.rename(os.path.join(adir, "arm.json.bak"), os.path.join(adir, "arm.json"))

    def test_zz_a_changed_decoder_refuses_the_existing_study(self):
        other = json.loads(json.dumps(self.r.ccfg["decoder"]))
        other["frozen"]["decoder"]["lookahead"] = 0.2
        other["frozen_sha256"] = contact_gate.frozen_sha256(other["frozen"])
        path = os.path.join(self.r.tmp.name, "other_gate.json")
        with open(path, "w") as f:
            json.dump({"verdict": "pass", "frozen_sha256": other["frozen_sha256"],
                       "decoder_sha256": cc.DecoderConfig.from_dict(other["frozen"]["decoder"]).sha256()}, f)
        other["gate_summary"] = path
        saved, self.r.ccfg["decoder"] = self.r.ccfg["decoder"], other
        try:
            with self.assertRaisesRegex(SystemExit, "frozen decoder record"):
                self.r.run()
        finally:
            self.r.ccfg["decoder"] = saved


class TestNoValidCandidate(unittest.TestCase):
    def test_an_arm_with_no_valid_slot_keeps_the_null_and_runs_no_episode(self):
        with tempfile.TemporaryDirectory() as d:
            null = {"fresh": {"success": 0.1, "per_case": [0.0] * 30}}
            g = {"llm_calls": 8, "llm_tokens": 120, "prompt_sha256": "x",
                 "slots": [{"slot": k, "valid": False, "policy": None, "code": None, "attempts": []} for k in range(4)]}

            def no_stream(*a):
                raise AssertionError("no episode may run")

            a = contact_rules.run_arm("contact_rules", {}, 4730, {}, null, d, no_stream, None, {}, generated=g)
            self.assertEqual((a["valid_candidates"], a["isaac_episodes"], a["promoted"], a["zero_shot"]),
                             (0, 0, False, None))
            self.assertEqual(a["fresh_incumbent"]["success"], 0.1)
            row = {"null": {"fresh": null["fresh"]}, "arms": {"contact_rules": a, "direct_joint": a}}
            s = contact_rules.summarise([row, row])
            self.assertEqual(s["paired_success_difference"], 0.0)
            self.assertEqual(s["contact_rules_minus_direct_joint_fresh_selected"]["per_block"], [])
            self.assertEqual(s["contact_rules_minus_direct_joint_zero_shot_screen_success"]["per_block"], [])
            self.assertEqual(s["hand_authored"]["success"], None)


if __name__ == "__main__":
    unittest.main()
