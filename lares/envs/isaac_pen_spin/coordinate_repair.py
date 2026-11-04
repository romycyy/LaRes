"""R-084 (the R-043 branch): anatomical synergy coordinates versus joint coordinates for LLM residual repair.

The parent is the D-058 incumbent (``config/pen_spin/d058_fixed_stroke_incumbent/``: the qualified zero-curl clamp of
the frozen R-046 learner with its stroke scaled by a fixed constant). An LLM writes a residual policy on the 177-d
parent-scaffold schema (the observation plus ``parent_action``); its mean ``c`` is a 20-d vector in decoder
coordinates and the executed action is ``clamp(parent + clamp(D c, -bound, bound), -1, 1)``. The two arms differ only
in ``D``: the identity (``joint``) or a frozen orthonormal synergy basis (``anatomical``); both see anonymous labels
``c0``-``c19`` and the same documentation, the same bound and clipping in joint space, the same request.

Per arm: 8 slots (one call plus at most one linked correction each), each candidate screened on the frozen 20-case
manifest in its own process, the finalist chosen by sustained successes, then fewer drops, then slot order, frozen
before any confirmation episode; the two finalists and the parent on the confirmation manifest's promotion (10-39) and
fresh (40-119) cases, one process each, traces recorded.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin.layout import PARENT_SCAFFOLD_TASK_ID
from lares.envs.isaac_pen_spin.protocols import arm_summary, failure_split

LABEL = "r084_coordinate_repair"
STAGE = 3
ARMS = ("joint", "anatomical")
PRIMARY = ("anatomical", "joint")
SLOTS = 8
RESIDUAL_BOUND = 0.15
ROLES = {"promotion": (10, 40), "fresh": (40, 120)}
CONFIRMATION_CASES, SCREEN_CASES = 120, 20
GAIN_THRESHOLD = 0.05
#: Verified action order (Isaac Lab sorts actuated_dof_indices; see controllers.py).
ACTION_ORDER = ("WRJ1", "WRJ0", "FFJ3", "MFJ3", "RFJ3", "LFJ4", "THJ4", "FFJ2", "MFJ2", "RFJ2", "LFJ3", "THJ3",
                "FFJ1", "MFJ1", "RFJ1", "LFJ2", "THJ2", "LFJ1", "THJ1", "THJ0")
#: The synergies, in this order, Gram-Schmidt-orthonormalised then completed by the remaining joint directions. A
#: synergy already in the span of the earlier ones (lf_curl = all_finger_curl minus the other three curls) is skipped
#: and recorded; the basis is still full rank and orthonormal.
SYNERGIES = (
    ("all_finger_curl", ("FFJ2", "FFJ1", "MFJ2", "MFJ1", "RFJ2", "RFJ1", "LFJ3", "LFJ2", "LFJ1")),
    ("ff_curl", ("FFJ2", "FFJ1")), ("mf_curl", ("MFJ2", "MFJ1")), ("rf_curl", ("RFJ2", "RFJ1")),
    ("lf_curl", ("LFJ3", "LFJ2", "LFJ1")),
    ("ff_abduction", ("FFJ3",)), ("mf_abduction", ("MFJ3",)), ("rf_abduction", ("RFJ3",)),
    ("lf_palm", ("LFJ4",)),
    ("thumb_opposition", ("THJ4", "THJ3")), ("thumb_flexion", ("THJ2", "THJ1", "THJ0")),
    ("wrist", ("WRJ1", "WRJ0")),
)


# ---------------------------------------------------------------------------------------------------------------------
#  Decoder
# ---------------------------------------------------------------------------------------------------------------------


def synergy_decoder() -> tuple[np.ndarray, list[str]]:
    """A full-rank orthonormal 20 x 20 matrix whose columns are the synergies (Gram-Schmidt) completed by joints.

    ``D @ c`` maps decoder coordinates to joint-target space; ``D`` is orthogonal, so the residual's norm is kept.
    """
    n = len(ACTION_ORDER)
    idx = {name: i for i, name in enumerate(ACTION_ORDER)}
    cols, names = [], []
    for name, joints in SYNERGIES:
        v = np.zeros(n)
        for j in joints:
            v[idx[j]] = 1.0
        for u in cols:
            v = v - np.dot(u, v) * u
        if np.linalg.norm(v) < 1e-9:
            continue  # spanned by the earlier synergies (recorded by its absence from ``names``)
        cols.append(v / np.linalg.norm(v))
        names.append(name)
    for j in ACTION_ORDER:
        v = np.zeros(n)
        v[idx[j]] = 1.0
        for u in cols:
            v = v - np.dot(u, v) * u
        if np.linalg.norm(v) > 1e-9:
            cols.append(v / np.linalg.norm(v))
            names.append(f"residual_{j}")
        if len(cols) == n:
            break
    D = np.stack(cols, axis=1)
    if D.shape != (n, n) or not np.allclose(D.T @ D, np.eye(n), atol=1e-10):
        raise ValueError("the synergy decoder is not orthonormal")
    return D, names


def decoder_for(arm: str) -> tuple[np.ndarray, list[str]]:
    if arm == "joint":
        return np.eye(len(ACTION_ORDER)), [f"joint_{j}" for j in ACTION_ORDER]
    if arm == "anatomical":
        return synergy_decoder()
    raise ValueError(f"unknown arm {arm!r}; one of {ARMS}")


def freeze_decoders(sdir: str) -> dict:
    """Write (or verify) ``decoders.json`` with both matrices and their hashes, before any call."""
    from lares.utils.hashing import sha256_text

    path = os.path.join(sdir, "decoders.json")
    payload = {}
    for arm in ARMS:
        D, names = decoder_for(arm)
        text = json.dumps(D.tolist())
        payload[arm] = {"matrix": D.tolist(), "names": names, "sha256": sha256_text(text), "bound": RESIDUAL_BOUND,
                        "action_order": list(ACTION_ORDER)}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
        if {a: old[a]["sha256"] for a in ARMS} != {a: payload[a]["sha256"] for a in ARMS}:
            raise SystemExit(f"{path} exists with different decoders")
        return old
    os.makedirs(sdir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    return payload


# ---------------------------------------------------------------------------------------------------------------------
#  The incumbent (D-058) as a controller and as a pipeline actor
# ---------------------------------------------------------------------------------------------------------------------


def check_asset(asset_dir: str) -> dict:
    """The incumbent's files against ``incumbent.json``; the evaluated source must equal the rebuilt substitution."""
    from lares.envs.isaac_pen_spin import channel_lesions as cl
    from lares.envs.isaac_pen_spin import stroke_feedback as sf
    from lares.utils.hashing import sha256_file, sha256_text

    with open(os.path.join(asset_dir, "incumbent.json"), encoding="utf-8") as f:
        inc = json.load(f)
    for name, expected in inc["files"].items():
        got = sha256_file(os.path.join(asset_dir, name))
        if got != expected:
            raise SystemExit(f"incumbent asset {name} hash {got} != frozen {expected}")
    with open(os.path.join(asset_dir, "parent_source.py"), encoding="utf-8") as f:
        parent_src = f.read()
    rebuilt = sf.substituted_source(cl.substituted_source(parent_src, "curl_clamp", 0.0))
    with open(os.path.join(asset_dir, "evaluated_source.py"), encoding="utf-8") as f:
        evaluated = f.read()
    if sha256_text(rebuilt) != sha256_text(evaluated):
        raise SystemExit("the incumbent's evaluated source is not the rebuilt substitution of its parent source")
    with open(os.path.join(asset_dir, "constant_gain.json"), encoding="utf-8") as f:
        if json.load(f)["constant"] != inc["constant"]:
            raise SystemExit("incumbent.json and constant_gain.json disagree on the constant")
    return inc


def incumbent_spec(asset_dir: str) -> dict:
    """The child-process actor spec of the incumbent: evaluated source, state file, fixed gain constant."""
    inc = check_asset(asset_dir)
    with open(os.path.join(asset_dir, "evaluated_source.py"), encoding="utf-8") as f:
        source = f.read()
    return {"kind": "symbolic", "source": source, "state_path": os.path.abspath(os.path.join(asset_dir, "state.pt")),
            "obs_dim": cases.OBS_DIM, "action_dim": cases.NUM_ACTIONS, "fixed_action_scale": 0.1,
            "execution": "clamp(mean, -1, 1)", "stroke_gain_constant": float(inc["constant"]),
            "name": inc["name"]}


def incumbent_controller(asset_dir: str):
    """The incumbent behind the runner's controller interface (used by the child and by the offline gate)."""
    from lares.envs.isaac_pen_spin.pipeline_child import SymbolicController
    from lares.envs.isaac_pen_spin.stroke_feedback import StrokeGain

    spec = incumbent_spec(asset_dir)
    ctrl = SymbolicController(spec, cases.STAGES[STAGE].task_id)
    ctrl.policy._stroke_gain = StrokeGain("constant", spec["stroke_gain_constant"])
    ctrl.name = spec["name"]
    ctrl.params = {**ctrl.params, "stroke_gain_constant": spec["stroke_gain_constant"]}
    return ctrl


class IncumbentActor:
    """The incumbent as a pipeline actor: the stream ships its spec; nothing is stepped in this process."""

    def __init__(self, asset_dir: str):
        self._spec = incumbent_spec(asset_dir)
        self.name = self._spec["name"]
        self.deterministic = True

    def spec(self) -> dict:
        return dict(self._spec)

    def reset(self, case=None):  # pragma: no cover - the child drives it
        return None

    def act(self, obs):  # pragma: no cover - never stepped in this process
        raise RuntimeError("the incumbent runs in the Isaac child only")


class ParentResidualController:
    """``clamp(parent + clamp(D c, -bound, bound), -1, 1)`` with ``c`` the residual policy's mean on ``[obs, parent]``."""

    name = "parent_residual"

    def __init__(self, parent, policy_mean, decoder, bound: float = RESIDUAL_BOUND, arm: str | None = None):
        import torch

        self.parent, self.policy_mean, self.bound = parent, policy_mean, float(bound)
        self.decoder = torch.as_tensor(np.asarray(decoder, dtype=np.float32))
        self.arm = arm
        self.residual_sum, self.residual_max, self.count = 0.0, 0.0, 0
        self.params = {"arm": arm, "bound": self.bound, "parent": getattr(parent, "params", None),
                       "execution": "clamp(parent + clamp(D c, -bound, bound), -1, 1)"}

    def reset(self, ctx):
        if hasattr(self.parent, "reset"):
            self.parent.reset(ctx)

    def __call__(self, obs, t, ctx):
        import torch

        g = self.parent(obs, t, ctx).to(obs.device)
        aug = torch.cat([obs.detach().float(), g.detach().float()], dim=-1)
        c = self.policy_mean(aug).to(obs.device)
        delta = (c @ self.decoder.to(obs.device).T).clamp(-self.bound, self.bound)
        with torch.no_grad():
            norms = delta.norm(dim=-1)
            self.residual_sum += float(norms.sum())
            self.residual_max = max(self.residual_max, float(norms.max()) if norms.numel() else 0.0)
            self.count += int(norms.numel())
        return (g + delta).clamp(-1.0, 1.0)

    def residual_stats(self) -> dict:
        return {"mean_norm": self.residual_sum / max(1, self.count), "max_norm": self.residual_max, "rows": self.count}


def residual_policy(policy, arm: str, decoders: dict, asset_dir: str, bound: float = RESIDUAL_BOUND):
    """Mark a generated residual policy for the pipeline backend (the child rebuilds the whole controller)."""
    policy._parent_residual = {"asset": os.path.abspath(asset_dir), "decoder": decoders[arm]["matrix"],
                               "decoder_name": arm, "decoder_sha256": decoders[arm]["sha256"], "bound": float(bound),
                               "arm": arm, "schema_env_id": PARENT_SCAFFOLD_TASK_ID}
    return policy


def offline_identity_gate(asset_dir: str, archive_npz: str, max_steps: int | None = None) -> dict:
    """The zero residual (both decoders) reproduces the incumbent's actions bit for bit on archived observations."""
    import torch

    data = np.load(archive_npz)
    obs, length = data["obs"], data["length"]
    parent = incumbent_controller(asset_dir)
    out = {"archive": archive_npz, "arms": {}}

    def zero_mean(aug):
        return torch.zeros(aug.shape[0], cases.NUM_ACTIONS)

    for arm in ARMS:
        D, _ = decoder_for(arm)
        ctrl = ParentResidualController(incumbent_controller(asset_dir), zero_mean, D, RESIDUAL_BOUND, arm)
        mism, rows = [], 0
        for i in range(obs.shape[0]):
            L = int(min(length[i], obs.shape[1]))
            if max_steps:
                L = min(L, int(max_steps))
            if L <= 0:
                continue
            o = torch.as_tensor(obs[i, :L], dtype=torch.float32)
            rows += L
            if not torch.equal(parent(o, 0, None), ctrl(o, 0, None)):
                mism.append(i)
        out["arms"][arm] = {"rows": rows, "mismatch_cases": mism, "passes": not mism}
    out["passes"] = all(v["passes"] for v in out["arms"].values())
    return out


# ---------------------------------------------------------------------------------------------------------------------
#  Manifests
# ---------------------------------------------------------------------------------------------------------------------


def manifest_files(rcfg: dict) -> dict:
    return {"screen": cases.manifest_path(STAGE, "r084_screen", int(rcfg["screen_seed"])),
            "confirmation": cases.manifest_path(STAGE, "r084_confirmation", int(rcfg["confirmation_seed"]))}


def freeze_manifests(rcfg: dict) -> dict:
    from lares.eval.manifest import EvaluationManifest, assert_disjoint

    files = manifest_files(rcfg)
    specs = [(files["screen"], cases.build_manifest(STAGE, int(rcfg["screen_seed"]), SCREEN_CASES, label="r084_screen"), "screen"),
             (files["confirmation"], cases.build_manifest(STAGE, int(rcfg["confirmation_seed"]), CONFIRMATION_CASES,
                                                          label="r084_confirmation"), "confirmation")]
    out = cases.freeze_manifest_set(specs)
    assert_disjoint([m for _, m, _ in specs] + [EvaluationManifest.load(p) for p in rcfg.get("disjoint_from", [])])
    return out


def load_manifests(rcfg: dict) -> dict:
    from lares.eval.manifest import EvaluationManifest

    files = manifest_files(rcfg)
    screen = EvaluationManifest.load(files["screen"])
    conf = EvaluationManifest.load(files["confirmation"])
    roles = cases.role_manifests(conf, ROLES, expected_cases=CONFIRMATION_CASES, stage=STAGE)
    return {"screen": screen, **roles}


# ---------------------------------------------------------------------------------------------------------------------
#  Request and generation
# ---------------------------------------------------------------------------------------------------------------------

COORDINATE_DOC = """Your policy's mean is a 20-d vector c = (c0, ..., c19). It is NOT a joint command. It is expressed in
20 fixed, mutually orthogonal unit directions of the 20-d joint-target space (the same space as parent_action and the
executed action). Their identities are not disclosed; they are the same for every candidate in this run. The executed
action is clamp(parent_action + clamp(D c, -{bound}, {bound}), -1, 1), where D is the fixed orthonormal 20 x 20 matrix
of those directions, so |c_i| = 1 moves the joint targets by at most {bound} along direction i, and the total
correction never exceeds {bound} per joint. c = 0 reproduces the parent exactly."""

REQUEST_TEMPLATE = """Below is the current controller (the parent). It keeps the pen in hand well but spins it too
slowly: on 80 fresh evaluation cases it succeeded on {successes} of 80; its failures were {failures}. Its executed
action is available to you as the observation field parent_action (20 values, action order, in [-1, 1]).

Write a RESIDUAL correction to it, not a replacement. {coordinate_doc}

Your controller is evaluated exactly at the initial parameter values its code declares, with no fitting or search.
Keep the class name GeneratedPolicy, read observations only through self.obs_field (parent_action included), declare
every parameter with get_param_ranges, and return a mean of shape (batch, 20) with std = self.fixed_std(mean). Make the
correction depend on the pen's state (its position relative to in_hand_pos, its orientation and angular velocity) and
on the hand's joint positions, so that it acts when and where the parent is losing spin.

```python
{parent}
```"""


def failure_evidence(rcfg: dict) -> dict:
    """The parent's archived fresh outcome (R-081's constant-gain rows), the failure evidence both arms see."""
    from lares.envs.isaac_pen_spin import channel_lesions as cl

    with open(rcfg["evidence_rows"], encoding="utf-8") as f:
        rows = json.load(f)["episodes"]
    split = {**failure_split(rows), **cl.tail_failure_split(rows)}
    return {"successes": int(sum(r["sustained"] for r in rows)), "cases": len(rows), "drops": split["drops"],
            "off_centre": split["off_centre"], "slow_spin": split["slow_spin"], "source": rcfg["evidence_rows"]}


def repair_request(parent_source: str, evidence: dict) -> tuple[str, dict]:
    from lares.utils.hashing import sha256_text

    failures = (f"slow spin x{evidence['slow_spin']}, off-centre x{evidence['off_centre']}, "
                f"dropped the pen x{evidence['drops']}")
    text = REQUEST_TEMPLATE.format(successes=evidence["successes"], failures=failures,
                                   coordinate_doc=COORDINATE_DOC.format(bound=RESIDUAL_BOUND), parent=parent_source.strip())
    return text, {"request_sha256": sha256_text(text), "request_chars": len(text), "evidence": evidence}


def generate_slot(arm: str, slot: int, request: str, packet: dict, adir: str, client, model: str,
                  reasoning_effort: str, max_calls: int = 2) -> dict:
    """One slot: one call plus at most one linked correction, its own generator, ledger and directory."""
    import pickle

    from lares.core.policy_generation import PlateauSlotGenerator
    from lares.core.training_pipeline import load_policy_prompt_assets
    from lares.search.generation import LLMCallLedger

    sdir = os.path.join(adir, f"slot{slot}")
    os.makedirs(sdir, exist_ok=True)
    data_pkl = os.path.join(sdir, "data.pkl")
    with open(data_pkl, "wb") as f:
        pickle.dump([{"obs": np.zeros(cases.OBS_DIM + cases.NUM_ACTIONS)}], f)
    ledger = LLMCallLedger(sdir)
    gen = PlateauSlotGenerator(client, SimpleNamespace(model=model, reasoning_effort=reasoning_effort),
                               os.path.join(sdir, "gen_0"), cases.OBS_DIM + cases.NUM_ACTIONS, cases.NUM_ACTIONS,
                               PARENT_SCAFFOLD_TASK_ID, load_policy_prompt_assets(PARENT_SCAFFOLD_TASK_ID),
                               SimpleNamespace(n=1), data_pkl, ledger,
                               llm_transcript_path=os.path.join(sdir, "llm_transcript.log"),
                               max_total_calls=int(max_calls), max_repair_per_candidate=1)
    attempts = gen.trace_repair(0, {"candidate_id": "d058_fixed_stroke_incumbent"}, request,
                                {"lane": LABEL, "arm": arm, "slot": slot, "parent": "d058_fixed_stroke_incumbent", **packet})
    final = attempts[-1]
    return {"slot": slot, "attempts": [{"attempt_id": c.attempt_id, "valid": c.valid, "validation": c.validation,
                                       "llm_call_ids": c.llm_call_ids} for c in attempts],
            "valid": bool(final.valid), "policy": final.policy if final.valid else None,
            "code": final.code if final.valid else None, "llm_calls": len(ledger),
            "llm_tokens": ledger.totals()["llm_total_tokens"]}


def slot_dir(sdir: str, arm: str, slot: int) -> str:
    return os.path.join(sdir, arm, f"slot{slot}")


def build_residual_policy(source: str, state: dict):
    """A generated residual policy on the 177-d parent-scaffold schema, rebuilt from its source and state."""
    from lares.core.obs_schema import active_schema, get_obs_schema
    from lares.core.policy_generation import GENERATED_IMPORTS

    ns: dict = {}
    exec(GENERATED_IMPORTS + source, ns)
    with active_schema(get_obs_schema(PARENT_SCAFFOLD_TASK_ID)):
        policy = ns["GeneratedPolicy"](cases.OBS_DIM + cases.NUM_ACTIONS, cases.NUM_ACTIONS)
    policy.fixed_action_scale = 0.1
    policy.load_state_dict(state)
    policy._generated_source = source
    policy.eval()
    return policy


def generated_slot(sdir: str, arm: str, slot: int, request: str, packet: dict, rcfg: dict, client) -> dict:
    """The slot's generation, saved before any episode; rebuilt from disk on a rerun (never regenerated)."""
    import torch

    d = slot_dir(sdir, arm, slot)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "generated.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            g = json.load(f)
        g["policy"] = (build_residual_policy(g["code"], torch.load(os.path.join(d, "generated_state.pt"), weights_only=True))
                       if g["valid"] else None)
        return g
    g = generate_slot(arm, slot, request, packet, os.path.join(sdir, arm), client, rcfg["model"], rcfg["reasoning_effort"],
                      int(rcfg.get("max_calls_per_slot", 2)))
    if g["valid"]:
        torch.save(g["policy"].state_dict(), os.path.join(d, "generated_state.pt"))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in g.items() if k != "policy"}, f, indent=1, default=str)
    os.replace(tmp, path)
    return g


# ---------------------------------------------------------------------------------------------------------------------
#  Scoring
# ---------------------------------------------------------------------------------------------------------------------


def score_rows(actor, manifest, stream, record: bool = False) -> dict:
    """Rows (and traces when recorded) of one actor on one manifest in one fresh process."""
    if record:
        got = stream.collect_traces(actor, manifest)
        rows = got["episodes"]
        return {"rows": rows, "trace_path": got["trace_path"], "obs": got["obs"], "actions": got["actions"]}
    response = stream._call(actor, manifest, record=False)
    return {"rows": response["episodes"]}


def screen_metrics(rows: list[dict]) -> dict:
    return {"successes": int(sum(r["sustained"] for r in rows)), "drops": int(sum(not r["completed"] for r in rows)),
            "tail_hit_fraction": float(np.mean([r["tail_hit_fraction"] for r in rows])) if rows else 0.0,
            "cases": len(rows)}


def finalist_key(m: dict, slot: int) -> tuple:
    return (-int(m["successes"]), int(m["drops"]), int(slot))


def _contrast(rows_a, rows_b) -> dict:
    from lares.envs.isaac_pen_spin import scalar_tuning as st

    c = st.contrast(rows_a, rows_b)
    return {**c, "drops_not_increased": bool(c["drop_counts"][0] <= c["drop_counts"][1]),
            "drop_increase_established": bool(c["drops"]["low"] is not None and c["drops"]["low"] > 0.0)}


def residual_norms(asset_dir: str, trace: dict) -> dict:
    """Executed minus parent action norms on recorded in-play steps, the parent recomputed offline."""
    import torch

    parent = incumbent_controller(asset_dir)
    obs, actions, length = np.asarray(trace["obs"]), np.asarray(trace["actions"]), np.asarray(trace["length"])
    norms = []
    for i in range(obs.shape[0]):
        L = int(min(length[i], obs.shape[1]))
        if L <= 0:
            continue
        with torch.no_grad():
            g = parent(torch.as_tensor(obs[i, :L], dtype=torch.float32), 0, None).numpy()
        norms.append(np.linalg.norm(actions[i, :L] - g, axis=1))
    n = np.concatenate(norms) if norms else np.zeros(0)
    return {"steps": int(n.size), "mean_norm": float(n.mean()) if n.size else None,
            "p95_norm": float(np.percentile(n, 95)) if n.size else None, "max_norm": float(n.max()) if n.size else None}


def aggregate(rows: dict, sats: dict, screens: dict, finalists: dict, norms: dict) -> dict:
    """``rows[controller][panel]`` for controllers ``parent``, ``joint``, ``anatomical``."""
    from lares.envs.isaac_pen_spin import channel_lesions as cl

    ctrls = ("parent",) + ARMS
    out = {"controllers": {c: {p: {**arm_summary(rows[c][p]), **failure_split(rows[c][p]), **cl.tail_failure_split(rows[c][p]),
                                  "saturation": {k: v for k, v in sats[c][p].items() if k != "drive_unsaturated_fraction_per_case"},
                                  "residual_norms": norms.get(c, {}).get(p)}
                              for p in ROLES} for c in ctrls},
           "screens": screens, "finalists": finalists, "contrasts": {}}
    for p in ROLES:
        out["contrasts"][p] = {"anatomical_minus_joint": _contrast(rows["anatomical"][p], rows["joint"][p]),
                               "anatomical_minus_parent": _contrast(rows["anatomical"][p], rows["parent"][p]),
                               "joint_minus_parent": _contrast(rows["joint"][p], rows["parent"][p])}
    t = out["contrasts"]["fresh"]["anatomical_minus_joint"]["success"]
    out["paired_success_difference"] = t["mean"]
    out["prediction"] = {"threshold": GAIN_THRESHOLD,
                         "supported": bool(t["mean"] is not None and t["mean"] >= GAIN_THRESHOLD and t["low"] is not None and t["low"] > 0.0),
                         "refuted": bool(t["high"] is not None and t["high"] < GAIN_THRESHOLD)}
    out["adoption"] = {}
    for arm in ARMS:
        pro, fr = out["contrasts"]["promotion"][f"{arm}_minus_parent"], out["contrasts"]["fresh"][f"{arm}_minus_parent"]
        out["adoption"][arm] = bool(pro["promotes"] and fr["success"]["low"] is not None and fr["success"]["low"] > 0.0
                                    and pro["drops_not_increased"] and fr["drops_not_increased"])
    return out


# ---------------------------------------------------------------------------------------------------------------------
#  Study
# ---------------------------------------------------------------------------------------------------------------------


def _started(path_dir: str, name: str) -> None:
    started = os.path.join(path_dir, f"{name}.started")
    if os.path.exists(started):
        raise SystemExit(f"{started}: this evaluation was interrupted earlier; its cases are never rerun")
    os.makedirs(path_dir, exist_ok=True)
    open(started, "w").close()


def run_study(rcfg: dict, stream_factory, client) -> dict:
    from lares.envs.isaac_pen_spin import llm_pool
    from lares.envs.isaac_pen_spin import scalar_tuning as st
    from lares.eval.runner import PolicyActor
    from lares.utils.hashing import sha256_file

    sdir = rcfg["study_dir"]
    os.makedirs(sdir, exist_ok=True)
    asset = rcfg["incumbent_asset"]
    inc = check_asset(asset)
    freeze_manifests(rcfg)
    decoders = freeze_decoders(sdir)
    gate_path = os.path.join(sdir, "identity_gate.json")
    if os.path.exists(gate_path):
        with open(gate_path, encoding="utf-8") as f:
            gate = json.load(f)
    else:
        gate = offline_identity_gate(asset, rcfg["identity_archive"])
        with open(gate_path, "w", encoding="utf-8") as f:
            json.dump(gate, f, indent=2)
    if not gate["passes"]:
        raise SystemExit(f"offline identity gate failed, no call and no episode: {gate}")
    manifests = load_manifests(rcfg)
    with open(os.path.join(asset, "evaluated_source.py"), encoding="utf-8") as f:
        parent_source = f.read()
    evidence = failure_evidence(rcfg)
    request, packet = repair_request(parent_source, evidence)
    with open(os.path.join(sdir, "request.txt"), "w", encoding="utf-8") as f:
        f.write(request)
    with open(os.path.join(sdir, "request.json"), "w", encoding="utf-8") as f:
        json.dump(packet, f, indent=2)

    # Generation: 16 independent slots on the pool (no episode until every slot of an arm is saved).
    slots = {arm: {} for arm in ARMS}
    with llm_pool.pool_for(rcfg) as pool:
        order = [(arm, k) for k in range(SLOTS) for arm in ARMS]
        rng = np.random.default_rng(int(rcfg["generation_seed"]))
        order = [order[i] for i in rng.permutation(len(order))]
        for arm, k in order:
            pool.submit((arm, k), generated_slot, sdir, arm, k, request, packet, rcfg, client)
        for arm, k in order:
            slots[arm][k] = pool.result((arm, k))

    # Screening: every valid candidate in its own process on the 20 screen cases; invalid ones score 0 (charged).
    screens, finalists = {}, {}
    fpath = os.path.join(sdir, "finalists.json")
    for arm in ARMS:
        adir = os.path.join(sdir, arm)
        spath = os.path.join(adir, "screens.json")
        if os.path.exists(spath):
            with open(spath, encoding="utf-8") as f:
                screens[arm] = json.load(f)
        else:
            _started(adir, "screen")
            stream = stream_factory(f"{arm}_screen", os.path.join(adir, "isaac_screen"))
            screens[arm] = {}
            for k in range(SLOTS):
                s = slots[arm][k]
                if not s["valid"]:
                    screens[arm][str(k)] = {"valid": False, "successes": 0, "drops": SCREEN_CASES, "tail_hit_fraction": 0.0,
                                            "cases": SCREEN_CASES, "charged": True}
                    continue
                actor = PolicyActor(residual_policy(s["policy"], arm, decoders, asset), deterministic=True, name=f"{arm}:slot{k}")
                got = score_rows(actor, manifests["screen"], stream)
                screens[arm][str(k)] = {"valid": True, **screen_metrics(got["rows"]),
                                        "failure_labels": [r.get("failure_label") for r in got["rows"]]}
            screens[arm]["isaac_processes"] = stream.calls
            with open(spath, "w", encoding="utf-8") as f:
                json.dump(screens[arm], f, indent=1)
        valid = [k for k in range(SLOTS) if screens[arm][str(k)].get("valid")]
        if not valid:
            finalists[arm] = None
        else:
            best = min(valid, key=lambda k: finalist_key(screens[arm][str(k)], k))
            finalists[arm] = {"slot": best, "screen": {k: v for k, v in screens[arm][str(best)].items() if k != "failure_labels"},
                              "rule": "sustained successes, then fewer drops, then slot order"}
    if os.path.exists(fpath):
        with open(fpath, encoding="utf-8") as f:
            if json.load(f) != json.loads(json.dumps(finalists)):
                raise SystemExit("finalists.json exists and differs from the selection this run makes")
    else:
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(finalists, f, indent=1)  # frozen before any confirmation episode
        import torch

        for arm in ARMS:
            if finalists[arm] is not None:
                k = finalists[arm]["slot"]
                torch.save(slots[arm][k]["policy"].state_dict(), os.path.join(sdir, arm, "finalist_state.pt"))
                with open(os.path.join(sdir, arm, "finalist_source.py"), "w", encoding="utf-8") as f:
                    f.write(slots[arm][k]["code"])

    # Confirmation: the two finalists and the parent on promotion and fresh, one process each, traces recorded.
    cdir = os.path.join(sdir, "confirmation")
    cpath = os.path.join(cdir, "confirmation.json")
    if os.path.exists(cpath):
        with open(cpath, encoding="utf-8") as f:
            conf = json.load(f)
    else:
        _started(cdir, "confirmation")
        stream = stream_factory("confirmation", os.path.join(cdir, "isaac"))
        conf = {"rows": {}, "traces": {}}
        for c in ("parent",) + ARMS:
            if c == "parent":
                actor = IncumbentActor(asset)
            elif finalists[c] is None:
                conf["rows"][c] = None
                continue
            else:
                actor = PolicyActor(residual_policy(slots[c][finalists[c]["slot"]]["policy"], c, decoders, asset),
                                    deterministic=True, name=f"{c}:slot{finalists[c]['slot']}")
            conf["rows"][c], conf["traces"][c] = {}, {}
            for p in ROLES:
                got = score_rows(actor, manifests[p], stream, record=True)
                conf["rows"][c][p] = got["rows"]
                conf["traces"][c][p] = got["trace_path"]
        conf["isaac_processes"] = stream.calls
        with open(cpath, "w", encoding="utf-8") as f:
            json.dump(conf, f, indent=1)
    for arm in ARMS:
        if conf["rows"].get(arm) is None:  # an arm with no valid candidate keeps the parent (scored once)
            conf["rows"][arm] = conf["rows"]["parent"]
            conf["traces"][arm] = conf["traces"]["parent"]
    import torch

    sd = torch.load(os.path.join(asset, "state.pt"), weights_only=True)
    params = {"spin_target": float(sd["spin_target"]), "speed_gain": float(sd["speed_gain"])}
    sats, norms = {}, {}
    for c in ("parent",) + ARMS:
        sats[c], norms[c] = {}, {}
        for p in ROLES:
            trace = np.load(conf["traces"][c][p])
            sats[c][p] = st.saturation(trace, params)
            norms[c][p] = residual_norms(asset, trace) if c != "parent" else None
    summary = aggregate(conf["rows"], sats, screens, finalists, norms)
    summary["generation"] = {arm: {"valid_candidates": int(sum(1 for k in range(SLOTS) if slots[arm][k]["valid"])),
                                   "llm_calls": int(sum(slots[arm][k].get("llm_calls", 0) for k in range(SLOTS))),
                                   "llm_tokens": int(sum(slots[arm][k].get("llm_tokens", 0) for k in range(SLOTS)))}
                             for arm in ARMS}
    summary["request"] = packet
    summary["decoders"] = {arm: {"sha256": decoders[arm]["sha256"], "names": decoders[arm]["names"]} for arm in ARMS}
    summary["identity_gate"] = gate
    summary["incumbent"] = {"asset": asset, "constant": inc["constant"], "files": inc["files"]}
    summary["manifests"] = {k: {"path": v, "sha256": sha256_file(v)} for k, v in manifest_files(rcfg).items()}
    screened = sum(1 for arm in ARMS for k in range(SLOTS) if screens[arm][str(k)].get("valid"))
    confirmed = 1 + sum(1 for arm in ARMS if finalists[arm] is not None)
    summary["episodes"] = int(screened * SCREEN_CASES + confirmed * sum(len(manifests[p]) for p in ROLES))
    summary["decision"] = {"adopt": [a for a in ARMS if summary["adoption"][a]],
                           "route": ("adopt " + ", ".join(a for a in ARMS if summary["adoption"][a]) if any(summary["adoption"].values())
                                     else "keep the D-058 incumbent"),
                           "rule": "strict promotion against the parent on cases 10-39, fresh lower bound above zero, no more observed drops"}
    summary["limitation"] = ("one generated cohort of 8 per arm with anonymous coordinates; tests this frozen repair "
                             "procedure, not anatomical coordinates in general; each candidate screened in its own "
                             "20-environment process")
    with open(os.path.join(sdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


__all__ = ["ACTION_ORDER", "ARMS", "GAIN_THRESHOLD", "LABEL", "PRIMARY", "RESIDUAL_BOUND", "ROLES", "SLOTS", "SYNERGIES",
           "IncumbentActor", "ParentResidualController", "aggregate", "check_asset", "decoder_for", "failure_evidence",
           "finalist_key", "freeze_decoders", "freeze_manifests", "incumbent_controller", "incumbent_spec",
           "load_manifests", "offline_identity_gate", "repair_request", "residual_norms", "residual_policy", "run_study",
           "screen_metrics", "synergy_decoder"]
