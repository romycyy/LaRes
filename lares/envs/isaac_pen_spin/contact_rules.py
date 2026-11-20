"""R-094 (C2): do cold-generated contact-frame velocity rules, executed through the trusted C1 decoder, beat cold-generated
direct joint targets after a matched rollout search?

Driven by ``scripts/run_full_evolution.py --config config/pen_spin_plateau_repair.yaml --study contact_rules``.
Conditional on R-093 (C1): :func:`require_gate_pass` refuses the study (and :func:`freeze_manifests` writes nothing)
until the config's ``decoder`` block names a frozen decoder block whose hash matches ``frozen_sha256`` and a
``gate_summary`` file whose verdict is ``pass`` for that same hash. One block per pool seed in ``block_seeds``, each one
frozen 80-case Stage-3 manifest split by index into R-092's roles (``ROLES``).

Both arms read the same 182-d observation (schema ``layout.CONTACT_FRAME_TASK_ID``): the 157-d layout plus 25 trusted
per-fingertip contact-frame features (``contact_control.contact_features``), appended in the Isaac child before the
generated policy is called. Both are monolithic, cold and single-phase. They differ only in the output contract, which
is one paragraph of the prompt (:func:`output_contract`) and the actor kind the child builds:

* ``direct_joint`` (actor kind ``contact_features``): the policy emits 20 normalised joint-position targets, executed
  as ``clamp(mean, -1, 1)`` like every other symbolic controller.
* ``contact_rules`` (actor kind ``contact_frame``): the policy emits 15 normalised requests, 5 fingertips x [normal,
  circumferential, axial]; the child executes ``contact_gate.ContactFrameController`` with the frozen decoder block
  (shipped in the request with its hash, which the child re-verifies) on ``scale_request(clamp(mean, -1, 1))``.
  Generated code never touches the simulator. An offline observation-only query of this actor is refused
  (``IsaacPenSpinStream.query_actions``): its action depends on simulator Jacobians and joint state.

Per block (each step skipped if its output exists; an interrupted step that spent episodes is never rerun):

1. ``null``: the neutral pose on all four roles, 80 episodes: the scored null incumbent (``modular_cold.run_null``).
2. ``hand_authored``: one frozen hand-written contact controller (:data:`HAND_AUTHORED_SOURCE`) through the same
   ``contact_frame`` path on all four roles, 80 episodes. A decoder diagnostic only: never ranked, never selected,
   never an incumbent.
3. ``direct_joint`` and ``contact_rules``: four cold slots each (one call plus at most one linked correction).
   * screen: every valid slot on the 10 screen cases. The best slot's screen result *is* the arm's zero-shot result
     (declared initial parameters); it is recorded as ``zero_shot`` together with the unchanged candidate's probe
     outcome (search candidate 0).
   * search: ``modular_cold.probe_search`` of that slot (the unchanged candidate plus 23 draws, seeded by the block
     seed, one fresh process each on the 10 probe cases); the winner's state is written before anything below runs.
   * promotion (30) against the null incumbent under the strict paired rule, then fresh (30). Nothing is selected on
     fresh outcomes.
   At most 340 episodes per arm, so 80 + 80 + 680 = 840 per block.
4. ``summary.json``: primary ``paired_success_difference`` = contact_rules minus direct_joint on fresh final incumbents
   (native sustained success), paired over blocks. Zero-shot is compared with zero-shot (screen best, and the
   unchanged candidate on the probe cases) and searched with searched (the searched winner's fresh success). The
   other secondaries are R-092's, plus the controllers' diagnostics (decoder residual, bound activity, output
   saturation) for the contact arm and the hand-authored controller.

Every (block, arm) generation is queued at the start and runs concurrently (``llm_pool``); its slots are saved to
``generated.json`` before any of its episodes run. Isaac steps run one process at a time.
"""

from __future__ import annotations

import json
import os

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin import contact_control as cc
from lares.envs.isaac_pen_spin import modular_cold
from lares.envs.isaac_pen_spin.block_study import (arm_needs_generation, attempts_record, make_slot_generator,
                                                   parent_like, started)
from lares.envs.isaac_pen_spin.gait_repair import public, score
from lares.envs.isaac_pen_spin.layout import CONTACT_FEATURE_DIM, CONTACT_FRAME_TASK_ID, STAGE_TASK_IDS

LABEL = "contact_rules"
STAGE = 3
ENV_ID = CONTACT_FRAME_TASK_ID
OBS_DIM = cases.OBS_DIM + CONTACT_FEATURE_DIM                    # 182
REQUEST_DIM = 3 * len(cc.FINGERTIPS)                             # 15
#: R-092's roles, search allocation and selection rules, unchanged.
ROLES = dict(modular_cold.ROLES)
BLOCK_CASES = modular_cold.BLOCK_CASES
SLOTS = modular_cold.SLOTS
SEARCH_CANDIDATES = modular_cold.SEARCH_CANDIDATES
MAX_ARM_EPISODES = modular_cold.MAX_ARM_EPISODES
ARMS = ("direct_joint", "contact_rules")
PRIMARY = ("contact_rules", "direct_joint")
#: What each arm's policy emits, and the actor kind the child builds for it.
ACTION_DIM = {"direct_joint": cases.NUM_ACTIONS, "contact_rules": REQUEST_DIM}
ACTOR_KIND = {"direct_joint": "contact_features", "contact_rules": "contact_frame"}
#: The keys a frozen decoder block must carry (``ContactFrameController`` reads these three; the hash covers all).
FROZEN_KEYS = ("decoder", "controller", "windows")
#: Actuated-joint (action) order, as ``TASK_NOTES`` states it.
ACTION_ORDER = ("WRJ1, WRJ0, FFJ3, MFJ3, RFJ3, LFJ4, THJ4, FFJ2, MFJ2, RFJ2, LFJ3, THJ3, FFJ1, MFJ1, RFJ1, LFJ2, THJ2, "
                "LFJ1, THJ1, THJ0")


# ---------------------------------------------------------------------------------------------------------------------
#  The C1 prerequisite
# ---------------------------------------------------------------------------------------------------------------------


def require_gate_pass(ccfg: dict) -> dict:
    """The frozen decoder this study may use, or ``SystemExit``.

    ``ccfg["decoder"]`` must give ``frozen`` (the block the gate froze: at least ``decoder``, ``controller`` and
    ``windows``), ``frozen_sha256`` (``contact_gate.frozen_sha256`` of that block) and ``gate_summary`` (the path of
    the gate's ``summary.json``). The summary must say ``"verdict": "pass"`` and record the same ``frozen_sha256``
    and the same ``decoder_sha256``, so the decoder that runs here is the one that passed.
    """
    from lares.envs.isaac_pen_spin.contact_gate import frozen_sha256
    from lares.utils.hashing import sha256_file

    d = ccfg.get("decoder") or {}
    frozen, sha, path = d.get("frozen"), d.get("frozen_sha256"), d.get("gate_summary")
    if not frozen or not sha or not path:
        raise SystemExit("contact_rules (R-094) is conditional on a recorded C1 pass: the config's decoder block needs "
                         "frozen, frozen_sha256 and gate_summary (they are placeholders until a decoder gate passes)")
    missing = [k for k in FROZEN_KEYS if k not in frozen]
    if missing:
        raise SystemExit(f"the frozen decoder block lacks {missing}")
    got = frozen_sha256(frozen)
    if got != sha:
        raise SystemExit(f"frozen decoder block hash {got} != the config's frozen_sha256 {sha}")
    if not os.path.isfile(path):
        raise SystemExit(f"no gate summary at {path}: contact_rules needs a recorded C1 pass")
    with open(path, encoding="utf-8") as f:
        summary = json.load(f)
    if summary.get("verdict") != cc.PASS:
        raise SystemExit(f"{path}: the decoder gate's verdict is {summary.get('verdict')!r}, not 'pass'; "
                         "contact_rules does not run on an unqualified decoder")
    if summary.get("frozen_sha256") != sha:
        raise SystemExit(f"{path} is the gate of frozen block {summary.get('frozen_sha256')}, not of {sha}")
    decoder_sha = cc.DecoderConfig.from_dict(frozen["decoder"]).sha256()
    if summary.get("decoder_sha256") != decoder_sha:
        raise SystemExit(f"{path} records decoder {summary.get('decoder_sha256')}, the config's block is {decoder_sha}")
    return {"frozen": json.loads(json.dumps(frozen)), "frozen_sha256": sha, "decoder_sha256": decoder_sha,
            "gate_summary": path, "gate_summary_sha256": sha256_file(path), "gate_study": summary.get("study"),
            "gate_verdict": summary["verdict"]}


# ---------------------------------------------------------------------------------------------------------------------
#  Actors: the mark the backend ships, and the controllers the child builds
# ---------------------------------------------------------------------------------------------------------------------


def contact_actor(policy, arm: str, decoder: dict):
    """Mark a policy on the 182-d schema for the pipeline backend (``actor_spec`` ships the mark as ``contact_actor``)."""
    kind = ACTOR_KIND[arm]
    if int(policy.action_dim) != ACTION_DIM[arm] or int(policy.obs_dim) != OBS_DIM:
        raise ValueError(f"{arm}: a policy of shape ({policy.obs_dim} -> {policy.action_dim}), the arm's is "
                         f"({OBS_DIM} -> {ACTION_DIM[arm]})")
    mark = {"kind": kind, "schema_env_id": ENV_ID, "policy_output_dim": ACTION_DIM[arm]}
    if kind == "contact_frame":
        mark.update(frozen=decoder["frozen"], frozen_sha256=decoder["frozen_sha256"],
                    execution=f"contact-frame decoder {decoder['frozen_sha256'][:16]}: "
                              "scale_request(clamp(mean, -1, 1)) decoded to 20 joint targets")
    else:
        mark["execution"] = "clamp(mean, -1, 1) on the observation plus contact features"
    policy._contact_actor = mark
    return policy


def augmented(obs):
    """``[obs157, contact_features(obs157)]`` as float32, ``(B, 182)``: what both arms' policies read."""
    import torch

    base = obs.detach().float()[:, :cases.OBS_DIM]
    return torch.cat([base, cc.contact_features(base).to(base.device)], dim=-1)


def _in_play(values, length) -> np.ndarray:
    """Entries of a ``(T, B, ...)`` per-step record at steps each case was still running (``t < length[b]``)."""
    values = np.asarray(values)
    mask = np.arange(values.shape[0])[:, None] < np.asarray(length)[None, :values.shape[1]]
    return values[mask]


def _mean(values) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()) if values.size else None


class ContactFeatureController:
    """``direct_joint``: a generated policy on ``[obs, contact features]``, executed as ``clamp(mean, -1, 1)``."""

    name = "contact_features"

    def __init__(self, policy_controller):
        self.inner = policy_controller                       # pipeline_child.SymbolicController on the 182-d schema
        self.params = {"execution": "clamp(mean, -1, 1) on [obs, contact_features(obs)]", "schema_env_id": ENV_ID}
        self._saturated = []

    def reset(self, ctx):
        self.inner.reset(ctx)
        self._saturated = []

    def __call__(self, obs, t, ctx):
        mean = self.inner.mean(augmented(obs))
        self._saturated.append((mean.abs() >= 1.0).float().mean(-1).numpy())
        return mean.clamp(-1.0, 1.0).to(obs.device)

    def pipeline_diagnostics(self, trace: dict) -> dict:
        sat = _in_play(np.stack(self._saturated), trace["length"]) if self._saturated else np.zeros(0)
        return {"kind": "contact_features", "steps": int(sat.size), "output_saturation_fraction": _mean(sat)}


class ContactRuleController:
    """``contact_rules``: 15 normalised requests from a generated policy, decoded by the frozen C1 controller.

    The policy sees only ``[obs, contact features]``; everything that reads the simulator (Jacobians, joint state,
    limits) is ``contact_gate.ContactFrameController``, unchanged. Its pulse windows do not apply (the request comes
    from the policy at every step); only its recording window is widened to the whole episode, with no Jacobian
    dumps, so :meth:`pipeline_diagnostics` covers every in-play step.
    """

    name = "contact_rules"

    def __init__(self, policy_controller, frozen: dict, frozen_sha256: str):
        from lares.envs.isaac_pen_spin import contact_gate

        got = contact_gate.frozen_sha256(frozen)
        if got != frozen_sha256:
            raise SystemExit(f"frozen decoder block hash {got} != the request's {frozen_sha256}")
        self.policy = policy_controller
        self.decoder_cfg = cc.DecoderConfig.from_dict(frozen["decoder"])
        self.inner = contact_gate.ContactFrameController(self.name, frozen, request_fn=self._request)
        self.inner.win = {**self.inner.win, "record_stop": cases.HORIZON + 1, "jacobian_steps": ()}
        self.params = {**self.inner.params, "schema_env_id": ENV_ID,
                       "execution": "decode(scale_request(clamp(mean, -1, 1))) -> 20 joint position targets"}
        self._saturated = []

    def _request(self, obs, t, frames):
        mean = self.policy.mean(augmented(obs))
        if mean.shape[-1] != REQUEST_DIM:
            raise ValueError(f"a contact_frame policy emits {REQUEST_DIM} requests, got {tuple(mean.shape)}")
        self._saturated.append((mean.abs() >= 1.0).float().mean(-1).numpy())
        return cc.scale_request(mean, self.decoder_cfg).to(obs.device)

    def reset(self, ctx):
        self.policy.reset(ctx)
        self.inner.reset(ctx)
        self._saturated = []

    def __call__(self, obs, t, ctx):
        return self.inner(obs, t, ctx)

    def extra_trace(self) -> dict:
        return self.inner.extra_trace()

    def pipeline_diagnostics(self, trace: dict) -> dict:
        """In-play means over the run: decoder residual (m/s), bound activity, request saturation, tip distance."""
        if not self._saturated:
            return {"kind": "contact_frame", "steps": 0}
        length, extra = trace["length"], self.inner.extra_trace()
        residual = _in_play(extra["residual"], length)                       # (steps, 5)
        bound = _in_play(extra["bound_active"], length)                      # (steps, controlled joints)
        return {"kind": "contact_frame", "frozen_sha256": self.inner.params["frozen_sha256"],
                "steps": int(residual.shape[0]),
                "residual_mean_mps": _mean(residual),
                "residual_p95_mps": float(np.percentile(residual, 95)) if residual.size else None,
                "bound_active_fraction": _mean(bound),
                "any_bound_step_fraction": _mean(bound.any(-1)) if bound.size else None,
                "output_saturation_fraction": _mean(_in_play(np.stack(self._saturated), length)),
                "distance_mean_m": _mean(_in_play(extra["distance"], length))}


def build_contact_controller(spec: dict, policy_controller):
    """The child's controller for a spec carrying ``contact_actor`` (``pipeline_child.build_controller``)."""
    ca = spec["contact_actor"]
    if int(spec["obs_dim"]) != OBS_DIM or int(spec["action_dim"]) != int(ca["policy_output_dim"]):
        raise SystemExit(f"contact actor {ca['kind']!r}: policy shape ({spec['obs_dim']} -> {spec['action_dim']}) does "
                         f"not match ({OBS_DIM} -> {ca['policy_output_dim']})")
    if ca["kind"] == "contact_features" and int(spec["action_dim"]) == cases.NUM_ACTIONS:
        return ContactFeatureController(policy_controller)
    if ca["kind"] == "contact_frame" and int(spec["action_dim"]) == REQUEST_DIM:
        return ContactRuleController(policy_controller, ca["frozen"], ca["frozen_sha256"])
    raise ValueError(f"unknown contact actor {ca['kind']!r} with {spec['action_dim']} outputs")


# ---------------------------------------------------------------------------------------------------------------------
#  The hand-authored contact controller (decoder diagnostic only)
# ---------------------------------------------------------------------------------------------------------------------

#: Frozen 2026-11-20, before any run. Every fingertip closes its distance to the pen surface proportionally (a
#: normalised normal request of ``-approach_gain * distance``, saturating at 2 cm) and all five request the same
#: circumferential velocity. By the frame convention a fingertip moving along +circumferential rolls a pen with a
#: fixed centre positively about its +x wherever it touches, so opposing contacts need the *same* circumferential
#: sign, not opposite ones. No axial request. It is trusted code in the generated-policy shape so that it runs
#: through the identical ``contact_frame`` path; it is not validated, searched, ranked or selected.
HAND_AUTHORED_SOURCE = '''
class GeneratedPolicy(SymbolicPolicy):
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.approach_gain = nn.Parameter(torch.tensor(50.0))
        self.roll_request = nn.Parameter(torch.tensor(0.5))

    def forward(self, obs):
        distance = self.obs_field(obs, "contact_distance")
        normal = -torch.clamp(self.approach_gain * distance, -1.0, 1.0)
        circumferential = self.roll_request * torch.ones_like(distance)
        axial = torch.zeros_like(distance)
        mean = torch.stack([normal, circumferential, axial], dim=-1).reshape(obs.shape[0], -1)
        return mean, self.fixed_std(mean)

    def get_param_ranges(self):
        return {"approach_gain": (0.0, 200.0), "roll_request": (-1.0, 1.0)}
'''.strip()


def hand_authored_policy(decoder: dict):
    """The frozen hand-written controller as a marked ``contact_frame`` policy."""
    from lares.core.obs_schema import active_schema, get_obs_schema
    from lares.core.policy_generation import GENERATED_IMPORTS
    from lares.envs.isaac_pen_spin.dense_fit import build_policy

    ns: dict = {}
    exec(GENERATED_IMPORTS + HAND_AUTHORED_SOURCE, ns)
    with active_schema(get_obs_schema(ENV_ID)):
        state = ns["GeneratedPolicy"](OBS_DIM, REQUEST_DIM).state_dict()
    policy = build_policy(HAND_AUTHORED_SOURCE, state, ENV_ID, 0.1, obs_dim=OBS_DIM, action_dim=REQUEST_DIM)
    return contact_actor(policy, "contact_rules", decoder)


# ---------------------------------------------------------------------------------------------------------------------
#  Prompts: identical except the output-contract paragraph (and the output width the templates print)
# ---------------------------------------------------------------------------------------------------------------------

#: Appended to both arms' task notes, word for word.
CONTACT_FEATURES_NOTE = (
    "contact_distance, contact_axial_coord and contact_rel_vel are computed from this same observation by fixed,\n"
    "trusted geometry: the pen is treated as an ideal capsule (radius 0.012 m, a straight section 0.14 m long with a\n"
    "hemispherical cap at each end) and each fingertip as one point, the origin of its distal link, which is not a\n"
    "point on the finger pad. Fingertip order is ff, mf, rf, lf, th (first, middle, ring, little finger, thumb).\n"
    "contact_rel_vel is in m/s and is not multiplied by 0.2. On an end cap the normal is the cap's own surface normal\n"
    "and the third axis is the pen's axis projected on the tangent plane. Proximity is not contact: a small\n"
    "contact_distance does not mean the finger touches, supports or grips the pen."
)


def _joint_contract(notes: str) -> tuple[int, int]:
    """Where the Stage-3 task notes state the joint-target output contract (one paragraph, newline-terminated)."""
    return notes.index("Actions are 20-dimensional"), notes.index("Control runs at 60 Hz")


def output_contract(arm: str, frozen: dict | None = None) -> str:
    """The one paragraph in which the arms' prompts differ.

    ``direct_joint``: the Stage-3 notes' own joint-target paragraph, verbatim. ``contact_rules``: the request
    contract, with the frozen decoder's scales and its two behavioural switches (``follow_pen``, ``control_wrist``)
    stated as facts. No controller, strategy or finger role is given.
    """
    from lares.core.policy_generation import TASK_NOTES

    notes = TASK_NOTES[STAGE_TASK_IDS[STAGE]]
    start, end = _joint_contract(notes)
    if arm == "direct_joint":
        return notes[start:end]
    if arm != "contact_rules":
        raise ValueError(f"unknown arm {arm!r}; one of {ARMS}")
    dec = cc.DecoderConfig.from_dict(frozen["decoder"])
    ctl = frozen["controller"]
    if ctl.get("follow_pen"):
        reference = "relative to the pen-surface point nearest to it"
        follow = ("The decoder adds the velocity of the pen-surface point to every request, so a request of zero makes\n"
                  "that fingertip move with the pen's surface there (it is not pinned in the world) and a non-zero\n"
                  "request is motion relative to that surface. A fingertip that supports the pen while the pen moves\n"
                  "should move with the pen rather than be pinned in the world; here that is the zero request.\n")
    else:
        reference = "in the world"
        follow = ("The decoder does not add the pen's own velocity: a request is that fingertip's velocity in the world,\n"
                  "expressed along its local frame axes, so a request of zero holds the fingertip still in the world,\n"
                  "while contact_rel_vel is measured relative to the moving pen surface. A fingertip that supports the\n"
                  "pen while the pen moves should move with the pen rather than be pinned in the world; here that\n"
                  "motion has to be requested.\n")
    wrist = ("The two wrist joints are driven by the decoder as well." if ctl.get("control_wrist")
             else "The two wrist joints are held at their default pose; the decoder moves the 18 finger and thumb joints.")
    return (
        "Your controller does not output joint targets. Its output (the policy mean) is 15 normalised requests, three\n"
        "per fingertip in the order ff, mf, rf, lf, th and, within a fingertip, [normal, circumferential, axial]: the\n"
        f"velocity you ask that fingertip point to have {reference}, expressed in the local frame of\n"
        "contact_rel_vel (normal points from the pen surface outward toward the fingertip, axial is the pen's local +x,\n"
        "circumferential = axial x normal: positive is counter-clockwise around the pen seen from its +x end).\n"
        f"Each request is clamped to [-1, 1] (no tanh) and multiplied by {dec.request_scale:g} m/s; a fingertip's\n"
        f"request is then capped at {dec.max_request_speed:g} m/s in norm.\n"
        "A fixed, trusted decoder, the same for every candidate, turns the 15 requests into the 20 joint-position\n"
        "targets at every step: with the hand's fingertip Jacobians it solves a damped least-squares problem for the\n"
        f"joint speeds that best give the requested fingertip velocities, limits them to {dec.max_joint_speed:g} rad/s\n"
        "and to the joint limits, and integrates them into joint targets. It is a kinematic approximation: a request\n"
        "may be tracked only in part, and it controls neither force nor contact.\n"
        f"{wrist}\n"
        f"{follow}"
        "You cannot read or change the decoder, the Jacobians or the simulator; you write only the rule from the\n"
        "observation to the 15 requests. No fingertip has a predefined role: which ones press, hold or drive, and when,\n"
        "is yours to decide.\n"
        "last_action is the 20 joint targets the decoder executed on the previous step, normalised to [-1, 1], in\n"
        f"actuated-joint order:\n  {ACTION_ORDER}\n"
        "  (FF/MF/RF/LF = first/middle/ring/little finger, TH = thumb, WR = wrist; J3 = knuckle side-to-side for\n"
        "  FF/MF/RF, J2 = knuckle flexion, J1 = middle-joint flexion; LFJ4 = little-finger palm cup; the distal J0\n"
        "  joints of the four fingers are coupled to J1 and not actuated).\n"
    )


def task_notes(arm: str, frozen: dict | None = None) -> str:
    """The Stage-3 task notes with the arm's output contract in place, then the shared contact-feature note."""
    from lares.core.policy_generation import TASK_NOTES

    notes = TASK_NOTES[STAGE_TASK_IDS[STAGE]]
    start, end = _joint_contract(notes)
    return notes[:start] + output_contract(arm, frozen) + notes[end:] + "\n" + CONTACT_FEATURES_NOTE


def arm_prompts(arm: str, decoder: dict) -> dict:
    """The arm's prompt assets: the Stage-3 task and templates, the 182-d layout, and :func:`task_notes`."""
    from lares.core.obs_schema import get_obs_schema
    from lares.core.training_pipeline import load_policy_prompt_assets

    prompts = dict(load_policy_prompt_assets(STAGE_TASK_IDS[STAGE]))
    schema = get_obs_schema(ENV_ID)
    prompts.update(obs_description=f"{schema.describe()}\n\n{task_notes(arm, decoder['frozen'])}", obs_schema=schema)
    return prompts


# ---------------------------------------------------------------------------------------------------------------------
#  Manifests
# ---------------------------------------------------------------------------------------------------------------------


def manifest_file(seed: int) -> str:
    return cases.manifest_path(STAGE, LABEL, seed)


def _freeze_manifests(ccfg: dict) -> dict:
    return cases.freeze_manifest_set(
        (manifest_file(int(seed)), cases.build_manifest(STAGE, int(seed), BLOCK_CASES, label=LABEL), f"block{b}")
        for b, seed in enumerate(ccfg["block_seeds"]))


def freeze_manifests(ccfg: dict) -> dict:
    """Freeze the block manifests, but only once the C1 prerequisite holds; before that nothing is written.

    ``--freeze-manifests`` walks every study block a config declares, so an unqualified decoder must not make this
    one reserve its seeds (or stop the others): it reports why it was skipped instead.
    """
    try:
        require_gate_pass(ccfg)
    except SystemExit as exc:
        return {"study": LABEL, "frozen": False, "skipped": str(exc)}
    return _freeze_manifests(ccfg)


def block_roles(seed: int) -> dict:
    from lares.eval.manifest import EvaluationManifest

    return cases.role_manifests(EvaluationManifest.load(manifest_file(seed)), ROLES, expected_cases=BLOCK_CASES,
                                stage=STAGE)


# ---------------------------------------------------------------------------------------------------------------------
#  Steps
# ---------------------------------------------------------------------------------------------------------------------


def scored(actor, manifest, stream) -> dict:
    """``gait_repair.score`` plus the child's ``controller_diagnostics`` for that process, when it returned any."""
    s = score(actor, manifest, stream)
    diag = getattr(stream, "last_diagnostics", None)
    if diag is not None:
        s["controller_diagnostics"] = diag
    return s


def run_hand_authored(roles: dict, bdir: str, stream_factory, decoder: dict) -> dict:
    """The frozen hand-written contact controller on all four roles (80 episodes). Diagnostic only."""
    from lares.eval.runner import PolicyActor

    path = os.path.join(bdir, "hand_authored.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    started(bdir, "hand_authored")
    stream = stream_factory(STAGE, "hand_authored", os.path.join(bdir, "hand_authored"))
    actor = PolicyActor(hand_authored_policy(decoder), deterministic=True, name="hand_authored_contact")
    out = {role: public(scored(actor, roles[role], stream)) for role in ROLES}
    per_case = [x for role in ROLES for x in out[role]["per_case"]]
    labels = [x for role in ROLES for x in out[role]["failure_labels"]]
    out.update(success=float(np.mean(per_case)), drops=labels.count("dropped"), cases=len(per_case),
               role="decoder diagnostic: never ranked, never selected, never an incumbent",
               frozen_sha256=decoder["frozen_sha256"], isaac_processes=stream.calls, isaac_episodes=stream.episodes)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    return out


def generate_slots(arm: str, adir: str, client, model: str, reasoning_effort: str, decoder: dict,
                   max_calls_per_slot: int = 2) -> dict:
    """Four single-phase cold slots from the arm's prompt, validated at the arm's output width on the 182-d schema."""
    from lares.utils.hashing import sha256_text

    gen, ledger = make_slot_generator(
        adir, ENV_ID, client, model, reasoning_effort, SLOTS, int(max_calls_per_slot), obs_dim=OBS_DIM,
        prompts=arm_prompts(arm, decoder), slot_metadata=lambda k: {"study": LABEL, "arm": arm, "slot": k},
        action_dim=ACTION_DIM[arm])
    with open(os.path.join(adir, "user_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(gen.user_content)
    slots = [{"slot": k, **attempts_record(gen.exploration(k))} for k in range(SLOTS)]
    return {"slots": slots, "llm_calls": len(ledger), "llm_tokens": ledger.totals()["llm_total_tokens"],
            "prompt_sha256": sha256_text(gen.user_content),
            "output_contract_sha256": sha256_text(output_contract(arm, decoder["frozen"]))}


def generate_arm(arm: str, ccfg: dict, bdir: str, client, decoder: dict) -> dict:
    """The arm's four slots; no episode. Safe on a worker thread (``llm_pool``)."""
    from lares.envs.isaac_pen_spin import llm_pool

    adir = os.path.join(bdir, arm)
    os.makedirs(adir, exist_ok=True)
    return llm_pool.generate_once(
        adir, ENV_ID, lambda: generate_slots(arm, adir, client, ccfg["model"], ccfg["reasoning_effort"], decoder,
                                             int(ccfg.get("max_calls_per_slot", 2))),
        obs_dim=OBS_DIM, action_dim=ACTION_DIM[arm])


def run_arm(arm: str, ccfg: dict, seed: int, roles: dict, null: dict, bdir: str, stream_factory, client,
            decoder: dict, generated: dict | None = None) -> dict:
    import torch

    from lares.eval.promotion import compare_to_incumbent
    from lares.eval.runner import PolicyActor
    from lares.fitting.checkpoints import RolloutScore, is_better_checkpoint

    adir = os.path.join(bdir, arm)
    path = os.path.join(adir, "arm.json")
    if not arm_needs_generation(bdir, arm):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    g = generated or generate_arm(arm, ccfg, bdir, client, decoder)
    os.makedirs(adir, exist_ok=True)
    out = {"arm": arm, "actor_kind": ACTOR_KIND[arm], "policy_output_dim": ACTION_DIM[arm], "stage": STAGE,
           "seed": int(seed), "llm_calls": g["llm_calls"], "llm_tokens": g["llm_tokens"],
           "prompt_sha256": g["prompt_sha256"], "output_contract_sha256": g.get("output_contract_sha256"),
           "slots": [{k: v for k, v in s.items() if k != "policy"} for s in g["slots"]]}
    valid = [(s["slot"], contact_actor(s["policy"], arm, decoder)) for s in g["slots"] if s["valid"]]
    null_fresh = null["fresh"]
    out["valid_candidates"] = len(valid)
    if not valid:
        out.update(promoted=False, selected_slot=None, zero_shot=None, isaac_processes=0, isaac_episodes=0, screens={},
                   fresh_incumbent={"success": null_fresh["success"], "per_case": null_fresh["per_case"],
                                    "candidate": "null", "rule": "no valid candidate: the null incumbent"})
        with open(path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        return out
    stream = stream_factory(STAGE, arm, os.path.join(adir, "isaac"))
    screens, best, best_k = {}, None, None
    for k, policy in valid:
        s = scored(PolicyActor(policy, deterministic=True, name=f"{arm}:slot{k}"), roles["screen"], stream)
        screens[str(k)] = public(s)
        r = RolloutScore(s["success"], s["tail_hit_fraction"])
        if is_better_checkpoint(r, best):
            best, best_k = r, k
    # The marked policy is deep-copied per search candidate, so every candidate carries the arm's actor kind.
    found = modular_cold.probe_search(dict(valid)[best_k], seed, f"{arm}:slot{best_k}", roles["probe"], stream)
    selected = found.pop("policy")
    original = found["trace"][0]
    zero_shot = {"slot": best_k, "candidate": f"{arm}:slot{best_k}", "screen": screens[str(best_k)],
                 "probe_original": {k: original[k] for k in ("success", "tail_hit_fraction", "per_case",
                                                             "failure_labels")},
                 "rule": "the best screened slot at its declared initial parameters: its screen result (the slot was "
                         "chosen on these cases) and the unchanged candidate's probe result (search candidate 0)"}
    # The selected state is frozen on disk before any promotion or fresh episode runs.
    state = modular_cold.state_record(selected)
    torch.save(selected.state_dict(), os.path.join(adir, "selected_state.pt"))
    search = {**found, "slot": best_k, "selected_state": {"file": "selected_state.pt", **state}}
    with open(os.path.join(adir, "search.json"), "w") as f:
        json.dump(search, f, indent=2, default=str)
    name = f"{arm}:slot{best_k}:search{found['selected_index']}"
    actor = PolicyActor(selected, deterministic=True, name=name)
    promo = scored(actor, roles["promotion"], stream)
    incumbent = parent_like(null["promotion"], roles["promotion"], null["promotion_returns"], "neutral_pose")
    decision = compare_to_incumbent(promo["_result"], incumbent, name, "neutral_pose",
                                    excluded_manifest_ids=[roles["screen"].manifest_id, roles["probe"].manifest_id])
    fresh = scored(actor, roles["fresh"], stream)
    promoted = bool(decision.promoted)
    want = 10 * len(valid) + 10 * SEARCH_CANDIDATES + 60
    if stream.episodes != want or stream.episodes > MAX_ARM_EPISODES:
        raise RuntimeError(f"{arm}: {stream.episodes} episodes ran, the protocol's count is {want} "
                           f"(at most {MAX_ARM_EPISODES})")
    out.update(screens=screens, selected_slot=best_k, zero_shot=zero_shot, search=search, promotion=public(promo),
               promotion_decision=decision.to_dict(), promoted=promoted, fresh_selected=public(fresh),
               fresh_incumbent={"success": fresh["success"] if promoted else null_fresh["success"],
                                "per_case": fresh["per_case"] if promoted else null_fresh["per_case"],
                                "candidate": name if promoted else "null",
                                "rule": "the arm's searched winner if it promoted against the null incumbent, else "
                                        "that incumbent (the neutral pose)"},
               isaac_processes=stream.calls, isaac_episodes=stream.episodes,
               selected_code=selected._generated_source)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    return out


# ---------------------------------------------------------------------------------------------------------------------
#  Study
# ---------------------------------------------------------------------------------------------------------------------

#: The controller diagnostics carried into the summary (``None`` where a controller does not report one).
DIAGNOSTIC_KEYS = ("residual_mean_mps", "residual_p95_mps", "bound_active_fraction", "any_bound_step_fraction",
                   "output_saturation_fraction", "distance_mean_m")


def _diagnostics(record: dict | None) -> dict:
    d = (record or {}).get("controller_diagnostics") or {}
    return {k: d.get(k) for k in DIAGNOSTIC_KEYS}


def _arm_stats(a: dict) -> dict:
    """One arm of one block reduced to the pre-declared secondaries (``None`` where the arm ran no candidate)."""
    screens, fresh, search, zero = a.get("screens", {}), a.get("fresh_selected"), a.get("search"), a.get("zero_shot")
    labels = fresh["failure_labels"] if fresh else []
    attempts = [s["attempts"] for s in a["slots"]]
    return {"slots": len(a["slots"]), "valid_slots": a.get("valid_candidates", 0),
            "first_attempt_valid_slots": sum(1 for at in attempts if at and at[0]["valid"]),
            "candidates_with_screen_success": sum(1 for s in screens.values() if s["success"] > 0),
            "zero_shot_screen_success": zero["screen"]["success"] if zero else None,
            "zero_shot_screen_tail_hit_fraction": zero["screen"]["tail_hit_fraction"] if zero else None,
            "zero_shot_probe_success": zero["probe_original"]["success"] if zero else None,
            "search_candidates_with_probe_success": search["candidates_with_probe_success"] if search else 0,
            "search_selected_index": search["selected_index"] if search else None,
            "plateau": bool(not a.get("promoted") and all(s["success"] == 0 for s in screens.values())
                            and not (search and search["candidates_with_probe_success"])),
            "promoted": bool(a.get("promoted")),
            "fresh_selected_success": fresh["success"] if fresh else None,
            "fresh_selected_tail_hit_fraction": fresh["tail_hit_fraction"] if fresh else None,
            "fresh_selected_drops": labels.count("dropped") if fresh else None,
            "fresh_selected_failure_labels": {k: labels.count(k) for k in sorted({x for x in labels if x})},
            "fresh_selected_diagnostics": _diagnostics(fresh),
            "fresh_incumbent_success": a["fresh_incumbent"]["success"],
            "llm_calls": a["llm_calls"], "llm_tokens": a["llm_tokens"],
            "isaac_processes": a.get("isaac_processes", 0), "isaac_episodes": a.get("isaac_episodes", 0)}


#: Per-block quantities contrasted contact_rules minus direct_joint (a block is skipped where either arm has no value).
ZERO_SHOT_CONTRASTS = ("zero_shot_screen_success", "zero_shot_screen_tail_hit_fraction", "zero_shot_probe_success")
SEARCHED_CONTRASTS = ("fresh_incumbent_success", "fresh_selected_success", "fresh_selected_tail_hit_fraction",
                      "fresh_selected_drops")
OTHER_CONTRASTS = ("candidates_with_screen_success", "valid_slots", "search_candidates_with_probe_success")
CONTRASTS = ZERO_SHOT_CONTRASTS + SEARCHED_CONTRASTS + OTHER_CONTRASTS


def _mean_of(rows: list[dict]) -> dict:
    """Per-key mean over the rows that report the key (``None`` where none does)."""
    out = {}
    for key in DIAGNOSTIC_KEYS:
        vals = [r[key] for r in rows if r.get(key) is not None]
        out[key] = float(np.mean(vals)) if vals else None
    return out


def summarise(blocks: list[dict], decoder: dict | None = None) -> dict:
    from lares.eval.stats import block_contrast

    stats = [{arm: _arm_stats(blk["arms"][arm]) for arm in ARMS} for blk in blocks]
    a, b = PRIMARY
    out = {"study": LABEL, "blocks": blocks, "n_blocks": len(blocks), "primary": f"{a}_minus_{b}_fresh_incumbent",
           "success_rule": "native sustained success; tail-hit fraction is a secondary and never replaces it",
           "comparison_rule": "zero-shot with zero-shot (the best screened slot at its declared initial parameters), "
                              "searched with searched (the 24-candidate winner on fresh cases)",
           "zero_shot_contrasts": [f"{a}_minus_{b}_{k}" for k in ZERO_SHOT_CONTRASTS],
           "searched_contrasts": [f"{a}_minus_{b}_fresh_incumbent", f"{a}_minus_{b}_fresh_selected",
                                  f"{a}_minus_{b}_fresh_selected_tail_hit_fraction",
                                  f"{a}_minus_{b}_fresh_selected_drops"]}
    if decoder is not None:
        out["decoder"] = {k: decoder[k] for k in ("frozen_sha256", "decoder_sha256", "gate_summary",
                                                  "gate_summary_sha256", "gate_verdict")}
    for key in CONTRASTS:
        out[f"{a}_minus_{b}_{key}"] = block_contrast(
            s[a][key] - s[b][key] for s in stats if s[a][key] is not None and s[b][key] is not None)
    out[f"{a}_minus_{b}_fresh_incumbent"] = out.pop(f"{a}_minus_{b}_fresh_incumbent_success")
    out[f"{a}_minus_{b}_fresh_selected"] = out.pop(f"{a}_minus_{b}_fresh_selected_success")
    out["paired_success_difference"] = out[f"{a}_minus_{b}_fresh_incumbent"].get("mean")
    out["null_fresh"] = float(np.mean([blk["null"]["fresh"]["success"] for blk in blocks])) if blocks else None
    arms = {}
    for arm in ARMS:
        st = [s[arm] for s in stats]

        def mean(key):
            vals = [s[key] for s in st if s[key] is not None]
            return float(np.mean(vals)) if vals else None

        total = lambda key: int(sum(s[key] or 0 for s in st))  # noqa: E731
        labels: dict = {}
        for s in st:
            for k, n in s["fresh_selected_failure_labels"].items():
                labels[k] = labels.get(k, 0) + n
        slots = total("slots")
        arms[arm] = {
            "fresh_incumbent": mean("fresh_incumbent_success"),
            "fresh_selected": mean("fresh_selected_success"),
            "fresh_selected_tail_hit_fraction": mean("fresh_selected_tail_hit_fraction"),
            "zero_shot_screen_success": mean("zero_shot_screen_success"),
            "zero_shot_screen_tail_hit_fraction": mean("zero_shot_screen_tail_hit_fraction"),
            "zero_shot_probe_success": mean("zero_shot_probe_success"),
            "blocks_with_a_searched_winner": sum(1 for s in st if s["fresh_selected_success"] is not None),
            "candidates_with_screen_success": total("candidates_with_screen_success"),
            "slots": slots, "valid_slots": total("valid_slots"),
            "slots_valid_within_one_correction": total("valid_slots") / slots if slots else None,
            "first_attempt_valid_slots": total("first_attempt_valid_slots"),
            "search_candidates_with_probe_success": total("search_candidates_with_probe_success"),
            "fresh_selected_drops": total("fresh_selected_drops"),
            "fresh_selected_failure_labels": labels,
            "fresh_selected_diagnostics": _mean_of([s["fresh_selected_diagnostics"] for s in st]),
            "promoted_blocks": sum(s["promoted"] for s in st), "plateau_blocks": sum(s["plateau"] for s in st),
            "llm_calls": total("llm_calls"), "llm_tokens": total("llm_tokens"),
            "isaac_processes": total("isaac_processes"), "isaac_episodes": total("isaac_episodes")}
    out["arms"] = arms
    out["slots_valid_within_one_correction"] = {arm: arms[arm]["slots_valid_within_one_correction"] for arm in ARMS}
    hand = [blk["hand_authored"] for blk in blocks if blk.get("hand_authored")]
    out["hand_authored"] = {
        "role": "decoder diagnostic: never ranked, never selected, never an incumbent",
        "success_per_block": [h["success"] for h in hand], "drops_per_block": [h["drops"] for h in hand],
        "success": float(np.mean([h["success"] for h in hand])) if hand else None,
        "diagnostics": _mean_of([_diagnostics(h[role]) for h in hand for role in ROLES])}
    out["decoder_diagnostics"] = {"contact_rules_fresh_selected": arms["contact_rules"]["fresh_selected_diagnostics"],
                                  "hand_authored": out["hand_authored"]["diagnostics"]}
    extra = ("null", "hand_authored")
    out["isaac_episodes"] = (sum((blk.get(k) or {}).get("isaac_episodes", 0) for blk in blocks for k in extra)
                             + sum(arms[arm]["isaac_episodes"] for arm in ARMS))
    out["isaac_processes"] = (sum((blk.get(k) or {}).get("isaac_processes", 0) for blk in blocks for k in extra)
                              + sum(arms[arm]["isaac_processes"] for arm in ARMS))
    out["llm_calls"] = sum(arms[arm]["llm_calls"] for arm in ARMS)
    out["llm_tokens"] = sum(arms[arm]["llm_tokens"] for arm in ARMS)
    return out


def run_study(ccfg: dict, stream_factory, client) -> dict:
    from lares.envs.isaac_pen_spin import llm_pool
    from lares.envs.isaac_pen_spin.protocols import freeze_json

    decoder = require_gate_pass(ccfg)        # before any manifest, call or episode
    sdir = ccfg["study_dir"]
    os.makedirs(sdir, exist_ok=True)
    freeze_json(os.path.join(sdir, "decoder.json"), decoder, "frozen decoder record")
    _freeze_manifests(ccfg)
    seeds = [int(s) for s in ccfg["block_seeds"]]
    blocks = []
    with llm_pool.pool_for(ccfg) as pool:
        # Cold prompts depend on no episode, so every (block, arm) generation is queued before any Isaac work.
        for b in range(len(seeds)):
            bdir = os.path.join(sdir, f"block{b}")
            for arm in ARMS:
                if arm_needs_generation(bdir, arm):
                    pool.submit((b, arm), generate_arm, arm, ccfg, bdir, client, decoder)
        for b, seed in enumerate(seeds):
            roles = block_roles(seed)
            bdir = os.path.join(sdir, f"block{b}")
            null = modular_cold.run_null(roles, bdir, stream_factory)
            hand = run_hand_authored(roles, bdir, stream_factory, decoder)
            row = {"block": b, "stage": STAGE, "seed": seed,
                   "null": {**{k: null[k] for k in ROLES}, "isaac_processes": null["isaac_processes"],
                            "isaac_episodes": null["isaac_episodes"]},
                   "hand_authored": hand}
            row["arms"] = {arm: run_arm(arm, ccfg, seed, roles, null, bdir, stream_factory, client, decoder,
                                        generated=pool.result((b, arm)) if (b, arm) in pool.pending() else None)
                           for arm in ARMS}
            blocks.append(row)
            with open(os.path.join(sdir, "summary.partial.json"), "w") as f:
                json.dump(summarise(blocks, decoder), f, indent=2, default=str)
    summary = summarise(blocks, decoder)
    with open(os.path.join(sdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


__all__ = ["ACTION_DIM", "ACTOR_KIND", "ARMS", "CONTACT_FEATURES_NOTE", "ContactFeatureController",
           "ContactRuleController", "ENV_ID", "HAND_AUTHORED_SOURCE", "LABEL", "MAX_ARM_EPISODES", "OBS_DIM", "PRIMARY",
           "REQUEST_DIM", "ROLES", "SEARCH_CANDIDATES", "SLOTS", "STAGE", "arm_prompts", "augmented", "block_roles",
           "build_contact_controller", "contact_actor", "freeze_manifests", "generate_arm", "generate_slots",
           "hand_authored_policy", "output_contract", "require_gate_pass", "run_arm", "run_hand_authored", "run_study",
           "scored", "summarise", "task_notes"]
