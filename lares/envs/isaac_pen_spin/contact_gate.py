"""R-093 (C1): the contact-frame decoder gate. No LLM; engineering checks, not learned-policy results.

Two registry names serve this module (``protocol_cfg["study"]`` says which):

* ``contact_decoder_calibration``: the declared calibration on the seed-4721 manifest. Each cell is one whole
  manifest in a fresh process under one candidate setting of the controller constants; case ``i`` pulses the
  direction ``i mod 30`` (five fingertips x three local axes x two signs), so one cell exercises every direction.
  At most 20 cells (800 episodes). Its outcomes choose the constants and are never gate evidence.
* ``contact_decoder_gate``: 32 cells on the seed-4720 manifest (1,280 episodes, no retries): the neutral-pose hold,
  the zero-request hold, and one cell per fingertip-axis-sign pulsing that one direction on all 40 cases. It refuses
  to run unless the config's ``frozen`` block (decoder, controller, windows, thresholds) matches
  ``frozen_sha256``; the block is written to ``frozen.json`` before the first cell.

Controller (trusted code; it reads the simulator through ``ControllerContext``, generated code never does). Per step:
capsule frames at the five fingertip body origins from the observation; fingertip Jacobians w.r.t. the 24 joints,
folded onto the controlled joints (the actuated finger joints; each passive distal joint ``J0`` moves with its
``J1``, coefficient 1, the tendon coupling measured at 0.86-0.98 on archived traces; the two wrist joints are held
unless ``control_wrist``); damped least squares for the joint speeds that give the requested fingertip velocity
relative to the pen surface point (``contact_control.decode``); a joint *reference* integrated from those speeds
(``q_ref += dt * qdot``, kept within ``max_reference_error`` of the measured joints and inside the limits); the action
is the position target ``q_ref + lookahead * qdot``. The reference starts at the default joint positions moved
``reference_margin`` inside the joint limits. Fingertip points are the distal link *origins*: the Jacobian
(``ControllerContext.fingertip_jacobians``) and the observed velocity (``contact_control.tip_origin_linvel``) are
both corrected from PhysX's centre-of-mass quantities. The env applies targets unsmoothed (``act_moving_average`` 1.0).

Windows. The request is nonzero for control steps ``pulse_start <= t < pulse_stop`` (10 to 40). Observed relative
velocity is read from the observations at steps ``pulse_start + 1 .. pulse_stop`` (each is the state after a
commanded step) and projected on the commanded fingertip's frame at ``pulse_start``.

Eligibility never reads observed motion: the pen has not fallen by ``pulse_stop`` and, on at least
``min_inactive_fraction`` of the pulse steps, no decoder bound (joint speed, joint limit, reference clamp) was active
on the commanded fingertip's chain.

Jacobian audit. At ``jacobian_steps`` the controller records the simulator's Jacobians; offline, the fingertip
displacement over the next control step is compared with ``J dq`` over all 24 measured joints, for samples that
moved at least ``jacobian_min_displacement``: ``|dp - J dq| / |dp|``.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin import contact_control as cc
from lares.envs.isaac_pen_spin.protocols import load_cell, study_cell
from lares.utils.hashing import sha256_text

STAGE = 3
CALIBRATION, GATE = "contact_decoder_calibration", "contact_decoder_gate"
LABELS = {CALIBRATION: "contact_decoder_calibration", GATE: "contact_decoder_gate"}
SIGNS = (("pos", 1.0), ("neg", -1.0))
#: The 30 directions in a fixed order: (fingertip index, local axis index, sign).
DIRECTIONS = tuple((k, a, s) for k in range(len(cc.FINGERTIPS)) for a in range(3) for _, s in SIGNS)
FINGER_TAGS = ("FF", "MF", "RF", "LF", "TH")  # joint-name tags of the chains, fingertip order
MAX_CALIBRATION_CELLS = 20


def direction_name(k: int, a: int, s: float) -> str:
    return f"{cc.FINGERTIPS[k]}_{cc.LOCAL_AXES[a]}_{'pos' if s > 0 else 'neg'}"


DIRECTION_NAMES = tuple(direction_name(*d) for d in DIRECTIONS)
GATE_CELLS = ("neutral", "hold") + DIRECTION_NAMES


def frozen_sha256(frozen: dict) -> str:
    return sha256_text(json.dumps(frozen, sort_keys=True, separators=(",", ":")))


def settings_of(frozen: dict) -> tuple:
    """``(DecoderConfig, controller dict, windows dict)`` of a frozen / candidate block."""
    return cc.DecoderConfig.from_dict(frozen["decoder"]), dict(frozen["controller"]), dict(frozen["windows"])


# ---------------------------------------------------------------------------------------------------------------------
#  Manifests
# ---------------------------------------------------------------------------------------------------------------------


def build_manifests(protocol_cfg: dict) -> list[str]:
    """Both manifests (calibration 4721, gate 4720), disjoint from each other and from every committed Stage-3 one."""
    from lares.eval.manifest import EvaluationManifest, assert_disjoint

    n = int(protocol_cfg["cases"])
    mine = {protocol_cfg["manifest"]: cases.build_manifest(STAGE, int(protocol_cfg["pool_seed"]), n, label=LABELS[GATE]),
            protocol_cfg["calibration_manifest"]: cases.build_manifest(
                STAGE, int(protocol_cfg["calibration_pool_seed"]), n, label=LABELS[CALIBRATION])}
    others = [EvaluationManifest.load(p) for p in sorted(glob.glob("config/manifests/pen_spin_stage3_*.yaml"))
              if p not in mine]
    assert_disjoint(list(mine.values()) + others)
    return [p for p, m in mine.items() if cases.freeze_manifest(p, m)]


# ---------------------------------------------------------------------------------------------------------------------
#  Controller
# ---------------------------------------------------------------------------------------------------------------------


class ContactFrameController:
    """Decoded contact-frame velocity requests behind the runner's controller interface (module docstring).

    ``directions``: per case ``(fingertip, axis, sign)`` or ``None`` (no pulse). ``request(t)`` may be replaced by a
    subclass or a callable ``request_fn(obs, t, frames) -> (B, 5, 3)`` m/s (R-094's policy path).
    """

    def __init__(self, name: str, frozen: dict, directions=None, request_fn=None):
        self.name = name
        self.decoder, self.ctl, self.win = settings_of(frozen)
        self.directions = directions
        self.request_fn = request_fn
        self.params = {"frozen_sha256": frozen_sha256(frozen), **frozen}
        self._rec = None

    # -- set-up ------------------------------------------------------------------------------------------------------

    def reset(self, ctx):
        import torch

        names = ctx.joint_names
        act = list(ctx.actuated_dof_indices)
        wrist = [i for i in act if "WRJ" in names[i]]
        ctrl = [i for i in act if self.ctl.get("control_wrist") or i not in wrist]
        n_all, dev = len(names), ctx.device
        C = torch.zeros(n_all, len(ctrl), device=dev)
        for col, j in enumerate(ctrl):
            C[j, col] = 1.0
        coupled = {}
        for j, nm in enumerate(names):                      # a passive distal joint moves with its finger's J1
            if j not in act and nm.endswith("J0"):
                partner = names.index(nm[:-1] + "1")
                if partner in ctrl:
                    C[j, ctrl.index(partner)] = float(self.ctl["coupling"])
                    coupled[nm] = names[partner]
        self.C, self.ctrl = C, torch.tensor(ctrl, device=dev)
        self.ctrl_in_action = torch.tensor([act.index(j) for j in ctrl], device=dev)
        self.lo, self.hi = ctx.lower_all[:, self.ctrl], ctx.upper_all[:, self.ctrl]
        # The default joint positions are zero, which is the lower limit of every flexion joint: from there half of
        # all requests are infeasible. The reference starts ``reference_margin`` inside the limits instead (capped
        # at mid-range), so the hand closes slightly from the first step.
        m, mid = float(self.ctl.get("reference_margin", 0.0)), 0.5 * (self.lo + self.hi)
        self.q_ref = torch.zeros_like(self.lo).clamp(torch.minimum(self.lo + m, mid), torch.maximum(self.hi - m, mid))
        self.neutral = ctx.neutral_action.clone()
        self.chain = torch.stack([torch.tensor([tag in names[j] or "WRJ" in names[j] for j in ctrl], device=dev)
                                  for tag in FINGER_TAGS])               # (5, n_ctrl)
        self.frames = None
        b = ctx.num_envs
        self.cmd = torch.zeros(b, len(cc.FINGERTIPS), 3, device=dev)
        self.commanded = torch.zeros(b, len(cc.FINGERTIPS), dtype=torch.bool, device=dev)
        if self.directions is not None:
            for i, d in enumerate(self.directions):
                if d is not None:
                    k, a, s = d
                    self.cmd[i, k, a] = float(s) * float(self.win["pulse_speed"])
                    self.commanded[i, k] = True
        self._rec = {"bound_active": [], "residual": [], "jacobian": [], "jacobian_step": [], "distance": []}
        self._meta = {"controlled_joints": [names[j] for j in ctrl], "coupled": coupled,
                      "lower_all": ctx.lower_all[0].cpu().numpy(), "upper_all": ctx.upper_all[0].cpu().numpy()}

    # -- one step ----------------------------------------------------------------------------------------------------

    def __call__(self, obs, t, ctx):
        import torch

        s = cc.state_from_obs(obs)
        self.frames = frames = cc.capsule_frames(s["pen_pos"], s["pen_quat"], s["tip_pos"], prev=self.frames)
        pulse = int(self.win["pulse_start"]) <= t < int(self.win["pulse_stop"])
        if self.request_fn is not None:
            u = self.request_fn(obs, t, frames)
            commanded = torch.ones_like(self.commanded)
        else:
            u = self.cmd if pulse else torch.zeros_like(self.cmd)
            commanded = self.commanded & pulse
        near = frames["distance"] < float(self.ctl["active_distance"])
        active = commanded | (near if self.ctl["follow_pen"] else torch.zeros_like(near))
        zero = torch.zeros_like(s["pen_linvel"])
        follow = bool(self.ctl["follow_pen"])
        J_full = ctx.fingertip_jacobians()
        J = cc.effective_jacobian(J_full, self.C)
        qdot, diag = cc.decode(J, frames, u, s["pen_pos"], s["pen_linvel"] if follow else zero,
                               s["pen_angvel"] if follow else zero, self.q_ref, self.lo, self.hi, self.decoder,
                               active=active)
        dt = self.decoder.control_dt
        q = ctx.joint_pos()[:, self.ctrl]
        err = float(self.ctl["max_reference_error"])
        want = self.q_ref + dt * qdot
        self.q_ref = torch.maximum(torch.minimum(want, q + err), q - err).clamp(self.lo, self.hi)
        ref_clamped = (self.q_ref - want).abs() > 1e-7
        action = self.neutral.clone()
        action[:, self.ctrl_in_action] = cc.position_targets(self.q_ref, qdot, self.decoder, self.lo, self.hi)

        if t < int(self.win["record_stop"]):
            bound = diag["vel_bound_active"] | diag["limit_active"] | ref_clamped
            self._rec["bound_active"].append(bound.cpu().numpy())
            self._rec["residual"].append(diag["residual"].cpu().numpy().astype(np.float32))
            self._rec["distance"].append(frames["distance"].cpu().numpy().astype(np.float32))
            if t in self.win["jacobian_steps"]:
                self._rec["jacobian"].append(J_full.cpu().numpy().astype(np.float32))
                self._rec["jacobian_step"].append(t)
        return action

    def extra_trace(self) -> dict:
        r = self._rec
        out = {"bound_active": np.stack(r["bound_active"]), "residual": np.stack(r["residual"]),
               "distance": np.stack(r["distance"]), "chain": self.chain.cpu().numpy(),
               "lower_all": self._meta["lower_all"], "upper_all": self._meta["upper_all"],
               "jacobian_step": np.array(r["jacobian_step"], dtype=np.int64)}
        if r["jacobian"]:
            out["jacobian"] = np.stack(r["jacobian"])
        return out


def calibration_directions(n: int) -> list:
    return [DIRECTIONS[i % len(DIRECTIONS)] for i in range(n)]


def candidate(protocol_cfg: dict, index: int) -> dict:
    """Calibration setting ``index``: the config's ``calibration.base`` with that setting's overrides merged in."""
    cal = protocol_cfg["calibration"]
    over = cal["settings"][index]
    out = {key: {**cal["base"][key], **over.get(key, {})} for key in ("decoder", "controller")}
    out["windows"] = dict(protocol_cfg["windows"])
    return out


def require_frozen(protocol_cfg: dict) -> dict:
    frozen = protocol_cfg.get("frozen")
    if not frozen or not protocol_cfg.get("frozen_sha256"):
        raise SystemExit("the gate has no frozen decoder block: run the calibration and freeze its constants first")
    frozen = {**frozen, "windows": dict(protocol_cfg["windows"]), "thresholds": dict(protocol_cfg["thresholds"]),
              "eligibility": dict(protocol_cfg["eligibility"])}
    got = frozen_sha256(frozen)
    if got != protocol_cfg["frozen_sha256"]:
        raise SystemExit(f"frozen block hash {got} != the config's frozen_sha256 {protocol_cfg['frozen_sha256']}")
    return frozen


def make_controller(name: str, protocol_cfg: dict, out_dir: str | None = None):
    n = int(protocol_cfg["cases"])
    if name == "neutral":
        from lares.envs.isaac_pen_spin.controllers import NeutralPose

        return NeutralPose()
    if name.startswith("cal"):                                # cal<idx> (pulses) or calhold<idx> (zero request)
        hold = name.startswith("calhold")
        frozen = candidate(protocol_cfg, int(name[7 if hold else 3:]))
        return ContactFrameController(name, frozen, None if hold else calibration_directions(n))
    frozen = require_frozen(protocol_cfg)
    if name == "hold":
        return ContactFrameController(name, frozen)
    if name in DIRECTION_NAMES:
        return ContactFrameController(name, frozen, [DIRECTIONS[DIRECTION_NAMES.index(name)]] * n)
    raise ValueError(f"unknown controller {name!r}")


# ---------------------------------------------------------------------------------------------------------------------
#  Offline analysis of one cell
# ---------------------------------------------------------------------------------------------------------------------


def timeout_rate(rows: list[dict]) -> float:
    return float(np.mean([bool(r["completed"]) for r in rows]))


def analyse_pulses(cell: str, directions: list, windows: dict, eligibility: dict) -> dict:
    """Per direction present in the cell: :func:`contact_control.cell_tracking`; plus the Jacobian audit samples."""
    import torch

    trace, rows = load_cell(cell)
    extra = dict(np.load(cell.replace(".json", ".extra.npz")))
    obs = trace["obs"]                                            # (C, T, 157)
    C, T = obs.shape[:2]
    p0, p1 = int(windows["pulse_start"]), int(windows["pulse_stop"])
    length, completed = trace["length"], trace["completed"]
    fall_step = np.where(completed, -1, length).astype(np.int64)
    if T <= p1:                                                   # every case ended before the window closed
        fall_step = np.where(fall_step < 0, T, fall_step)
    tip = np.array([d[0] for d in directions])
    chain = extra["chain"][tip]                                   # (C, n_ctrl)
    bound = extra["bound_active"][p0:p1]                          # (steps, C, n_ctrl)
    inactive = 1.0 - (bound & chain[None]).any(-1).mean(0) if len(bound) else np.zeros(C)
    eligible, labels = cc.pair_eligibility(fall_step, inactive, pulse_stop=p1,
                                           min_inactive_fraction=float(eligibility["min_inactive_fraction"]))
    stop = min(p1, T - 1)
    flat = torch.as_tensor(obs[:, p0:stop + 1].reshape(-1, obs.shape[-1]), dtype=torch.float64)
    s = cc.state_from_obs(flat)
    fr = cc.capsule_frames(s["pen_pos"], s["pen_quat"], s["tip_pos"])
    rel = cc.relative_world_velocity(cc.tip_origin_linvel(s), fr["point"], s["pen_pos"], s["pen_linvel"],
                                     s["pen_angvel"])
    steps = stop + 1 - p0
    rel = rel.reshape(C, steps, len(cc.FINGERTIPS), 3).numpy()
    R = fr["R"].reshape(C, steps, len(cc.FINGERTIPS), 3, 3).numpy()
    idx = np.arange(C)
    rel_cmd, R_first = np.nan_to_num(rel[idx, 1:, tip]), R[idx, 0, tip]   # obs p0+1..p1, frame at p0
    if rel_cmd.shape[1] == 0:
        rel_cmd, eligible = np.zeros((C, 1, 3)), np.zeros(C, dtype=bool)
    cells = {}
    for d in sorted(set(directions)):
        m = np.array([x == d for x in directions])
        cells[direction_name(*d)] = cc.cell_tracking(rel_cmd[m], R_first[m], d[1], d[2], eligible[m],
                                                     [l for l, keep in zip(labels, m) if keep],
                                                     requested_speed=float(windows["pulse_speed"]))
    return {"cells": cells, "jacobian_errors": jacobian_errors(obs, extra, length, windows),
            "timeout_rate": timeout_rate(rows), "sustained": int(sum(r["sustained"] for r in rows)),
            "bound_active_fraction": float(bound.any(-1).mean()) if len(bound) else None,
            "residual_mean": float(np.nanmean(extra["residual"][p0:p1])) if len(bound) else None}


def jacobian_errors(obs, extra: dict, length, windows: dict) -> list:
    """``|dp - J dq| / |dp|`` for every (step, case, fingertip) sample that moved enough over one control step."""
    if "jacobian" not in extra:
        return []
    sl = cases.obs_slices()
    lo, hi = extra["lower_all"], extra["upper_all"]
    out = []
    for J, t in zip(extra["jacobian"], extra["jacobian_step"]):   # J (C, 5, 3, 24)
        t = int(t)
        if t + 1 >= obs.shape[1]:
            continue
        q = lo + 0.5 * (obs[:, t:t + 2, sl["hand_dof_pos"]].astype(np.float64) + 1.0) * (hi - lo)
        p = obs[:, t:t + 2, sl["fingertip_pos"]].astype(np.float64).reshape(obs.shape[0], 2, 5, 3)
        dp, dq = p[:, 1] - p[:, 0], q[:, 1] - q[:, 0]
        pred = np.einsum("ckin,cn->cki", J.astype(np.float64), dq)
        moved = np.linalg.norm(dp, axis=-1)
        ok = (moved >= float(windows["jacobian_min_displacement"])) & (np.asarray(length) > t + 1)[:, None]
        out.extend((np.linalg.norm(dp - pred, axis=-1)[ok] / moved[ok]).tolist())
    return out


def pooled(cells: dict) -> dict:
    g = np.array([x for c in cells.values() for x, e in zip(c["gain"], c["eligible_mask"]) if e], dtype=np.float64)
    return {"eligible_pairs": int(len(g)), "pairs": int(sum(c["cases"] for c in cells.values())),
            "sign_agreement": float((g > 0).mean()) if len(g) else None,
            "gain_median": float(np.median(g)) if len(g) else None,
            "min_eligible_fraction": min((c["eligible"] / c["cases"] for c in cells.values()), default=None)}


# ---------------------------------------------------------------------------------------------------------------------
#  Drivers
# ---------------------------------------------------------------------------------------------------------------------


def run_calibration(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    cal = protocol_cfg["calibration"]
    plan = [(f"calhold{i}" if s.get("hold") else f"cal{i}") for i, s in enumerate(cal["settings"])]
    plan = ["neutral"] + plan
    if len(plan) > MAX_CALIBRATION_CELLS:
        raise SystemExit(f"{len(plan)} calibration cells exceed the declared {MAX_CALIBRATION_CELLS}")
    mpath = cases.require_frozen(protocol_cfg["calibration_manifest"])
    n = int(protocol_cfg["cases"])
    out = {"manifest": cases.manifest_record(mpath), "cells": {}, "episodes": 0,
           "note": "calibration only (seed 4721): chooses decoder constants, never gate evidence"}
    for name in plan:
        cell = os.path.join(out_dir, f"{name}.json")
        study_cell(protocol_cfg, config_path, mpath, name, cell)
        out["episodes"] += n
        if name == "neutral" or name.startswith("calhold"):
            _, rows = load_cell(cell)
            out["cells"][name] = {"timeout_rate": timeout_rate(rows), "sustained": int(sum(r["sustained"] for r in rows))}
            if name != "neutral":
                out["cells"][name]["setting"] = candidate(protocol_cfg, int(name[7:]))
            continue
        a = analyse_pulses(cell, calibration_directions(n), protocol_cfg["windows"], protocol_cfg["eligibility"])
        errs = np.array(a["jacobian_errors"])
        out["cells"][name] = {"setting": candidate(protocol_cfg, int(name[3:])), **pooled(a["cells"]),
                              "timeout_rate": a["timeout_rate"], "bound_active_fraction": a["bound_active_fraction"],
                              "residual_mean": a["residual_mean"],
                              "jacobian_rel_error_median": float(np.median(errs)) if len(errs) else None,
                              "jacobian_samples": int(len(errs)),
                              "per_direction": {k: {"gain": [g for g, e in zip(c["gain"], c["eligible_mask"]) if e],
                                                    "excluded": c.get("excluded")} for k, c in a["cells"].items()}}
        with open(os.path.join(out_dir, "summary.partial.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def run_gate(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    from lares.envs.isaac_pen_spin.protocols import freeze_json

    os.makedirs(out_dir, exist_ok=True)
    frozen = require_frozen(protocol_cfg)
    freeze_json(os.path.join(out_dir, "frozen.json"), {"frozen": frozen, "sha256": protocol_cfg["frozen_sha256"]},
                "frozen decoder block")
    mpath = cases.require_frozen(protocol_cfg["manifest"])
    n = int(protocol_cfg["cases"])
    cells, errors, per_cell, rates = {}, [], {}, {}
    for name in GATE_CELLS:
        cell = os.path.join(out_dir, f"{name}.json")
        study_cell(protocol_cfg, config_path, mpath, name, cell)
        if name in ("neutral", "hold"):
            _, rows = load_cell(cell)
            rates[name] = timeout_rate(rows)
            per_cell[name] = {"timeout_rate": rates[name], "sustained": int(sum(r["sustained"] for r in rows))}
            continue
        d = DIRECTIONS[DIRECTION_NAMES.index(name)]
        a = analyse_pulses(cell, [d] * n, frozen["windows"], frozen["eligibility"])
        cells[name] = a["cells"][name]
        errors.extend(a["jacobian_errors"])
        per_cell[name] = {k: a[k] for k in ("timeout_rate", "sustained", "bound_active_fraction", "residual_mean")}
    verdict = cc.gate_verdict(cells, errors, rates["hold"], rates["neutral"], frozen["thresholds"])
    summary = {"study": GATE, "verdict": verdict["verdict"], "gate": verdict, "cells": per_cell,
               "timeout_rate": rates["hold"], "neutral_timeout_rate": rates["neutral"],
               "timeout_rate_difference": rates["hold"] - rates["neutral"],
               "jacobian_samples": len(errors), "frozen_sha256": protocol_cfg["frozen_sha256"],
               "decoder_sha256": cc.DecoderConfig.from_dict(frozen["decoder"]).sha256(),
               "manifest": cases.manifest_record(mpath), "episodes": n * len(GATE_CELLS),
               "limitation": "kinematic tracking of fingertip body origins near an ideal capsule; proximity is not "
                             "contact, and a pass says nothing about task success"}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def run(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    return (run_calibration if protocol_cfg["study"] == CALIBRATION else run_gate)(protocol_cfg, config_path, out_dir)


__all__ = ["CALIBRATION", "ContactFrameController", "DIRECTIONS", "DIRECTION_NAMES", "GATE", "GATE_CELLS",
           "analyse_pulses", "build_manifests", "calibration_directions", "candidate", "frozen_sha256",
           "jacobian_errors", "make_controller", "require_frozen", "run", "run_calibration", "run_gate"]
