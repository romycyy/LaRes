"""R-101: does coherent forefinger + ring-finger circumferential motion transfer to pen roll during rolling?

D-069's all-finger rolling gate failed, but the forefinger and ring finger tracked (0.994 and 1.000 paired sign
agreement). This audit asks the next question for those two alone, with nothing tuned: R-097's frozen block
(``rolling_decoder_gate.frozen``, hash verified) pulses both fingertips together along ``+`` or ``-`` circumferential
for steps 150-180 on top of the unchanged D-058 incumbent.

Three cells on one fresh manifest, each its own Isaac process: ``zero`` (the zero-request residual, which must equal
the incumbent's action exactly), ``coherent_pos`` and ``coherent_neg``.

* **Eligibility** reads only the zero cell's prefix: in play through the roll window and ``|mean axial rate|`` over it
  at least ``min_abs_mean_rate``. No case is dropped for what a pulse does to it.
* **Roll progress** of a cell is the pen's axial rate integrated over ``[pulse_start, follow_through_stop)`` of the
  original episode; a step at or after the case's end contributes zero (auto-reset motion is never read).
* **Pro / anti.** By ``contact_control``'s convention a fingertip moving along ``+circumferential`` rolls the pen
  positively about its +X, so with ``d`` the zero cell's roll sign the pro cell is ``coherent_pos`` where ``d > 0``
  and ``coherent_neg`` otherwise. The primary statistic, direction-signed progress pro minus anti, is therefore
  ``progress_pos - progress_neg`` for every case; ``d`` enters only through eligibility. The quantities that do
  depend on ``d``, pro minus zero and anti minus zero, are reported as secondary.
* **Authority** (per fingertip, before transfer is read): the achieved world displacement of the fingertip link
  origin along the circumferential direction at ``pulse_start``, ``+`` cell minus ``-`` cell, over the pulse, as a
  fraction of ``2 * pulse_speed * pulse_duration``.

The env has no contact sensor: contact impulses are unavailable and are not estimated.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin import contact_control as cc
from lares.envs.isaac_pen_spin import contact_gate as cg
from lares.envs.isaac_pen_spin import rolling_decoder_gate as rg
from lares.envs.isaac_pen_spin.protocols import freeze_json, load_cell, study_cell

STAGE = 3
STUDY = "transfer_audit"
CELLS = ("zero", "coherent_pos", "coherent_neg")
CIRC = cc.LOCAL_AXES.index("circumferential")
BOUNDS = ("vel_bound", "limit", "cap", "clamp")
VOID, INADEQUATE = "void", "inadequate_authority"


# ---------------------------------------------------------------------------------------------------------------------
#  Controller
# ---------------------------------------------------------------------------------------------------------------------


class CoherentPulseController(rg.RollingPulseController):
    """R-097's pulse controller commanding several fingertips together, with per-step telemetry of every bound by
    type, the incumbent's own saturation and the controlled joint positions over the pulse window."""

    def reset(self, ctx):
        super().reset(ctx)
        self._audit = {key: [] for key in BOUNDS + ("incumbent_saturated", "q", "feasible_tips")}

    def _observe(self, t: int, step: dict) -> None:
        import torch

        a = self._audit
        for key in BOUNDS:
            a[key].append(step[key].cpu().numpy())
        a["incumbent_saturated"].append((step["incumbent"][:, self.ctrl_in_action].abs() >= 1.0).cpu().numpy())
        a["q"].append(step["q"].cpu().numpy().astype(np.float32))
        feas = cc.world_to_local(torch.einsum("bkin,bn->bki", step["J"], step["qdot_eff"].to(step["J"].dtype)),
                                 step["frames"])
        a["feasible_tips"].append(feas.cpu().numpy().astype(np.float32))

    def extra_trace(self) -> dict:
        out = super().extra_trace()
        for key, values in self._audit.items():
            if values:
                out[f"audit_{key}"] = np.stack(values)
        return out


def tip_indices(protocol_cfg: dict) -> list:
    return [cc.FINGERTIPS.index(t) for t in protocol_cfg["fingertips"]]


def require_frozen(protocol_cfg: dict) -> dict:
    """R-097's frozen block, verified against its recorded hash. The audit declares nothing of its own in it."""
    return rg.require_frozen(protocol_cfg)


def make_controller(name: str, protocol_cfg: dict, out_dir: str | None = None):
    asset, n = protocol_cfg["incumbent_asset"], int(protocol_cfg["cases"])
    frozen = require_frozen(protocol_cfg)
    if name == "zero":
        return CoherentPulseController(name, frozen, None, asset)
    if name in ("coherent_pos", "coherent_neg"):
        s = 1.0 if name.endswith("pos") else -1.0
        return CoherentPulseController(name, frozen, [[(k, CIRC, s) for k in tip_indices(protocol_cfg)]] * n, asset)
    raise ValueError(f"unknown controller {name!r}")


def build_manifests(protocol_cfg: dict) -> list[str]:
    from lares.eval.manifest import EvaluationManifest, assert_disjoint

    path = protocol_cfg["manifest"]
    mine = cases.build_manifest(STAGE, int(protocol_cfg["pool_seed"]), int(protocol_cfg["cases"]), label=STUDY)
    others = [EvaluationManifest.load(p) for p in sorted(glob.glob("config/manifests/pen_spin_stage3_*.yaml"))
              if p != path]
    assert_disjoint([mine] + others)
    return [path] if cases.freeze_manifest(path, mine) else []


# ---------------------------------------------------------------------------------------------------------------------
#  Offline statistics (numpy; no Isaac)
# ---------------------------------------------------------------------------------------------------------------------


def roll_progress(spin, length, start: int, stop: int, dt: float) -> np.ndarray:
    """Per case, the axial rate (``spin`` is ``(T, C)``) integrated over ``[start, stop)`` of the original episode:
    steps at or after ``length`` contribute zero."""
    spin = np.nan_to_num(np.asarray(spin, dtype=np.float64))
    t = np.arange(spin.shape[0])[:, None]
    keep = (t >= start) & (t < stop) & (t < np.asarray(length)[None, :])
    return (spin * keep).sum(0) * dt


def case_direction(spin, length, window, min_rate: float) -> tuple:
    """``(direction, mean rate, eligible)`` from the zero cell's prefix alone."""
    w0, w1 = int(window[0]), int(window[1])
    spin = np.nan_to_num(np.asarray(spin, dtype=np.float64))
    mean = spin[w0:w1].mean(0) if spin.shape[0] >= w1 else np.zeros(spin.shape[1])
    in_play = np.asarray(length) >= w1
    return np.where(in_play, np.sign(mean), 0.0), mean, in_play & (np.abs(mean) >= min_rate)


def tip_motion(obs, length, tips, p0: int, p1: int, dt: float) -> dict:
    """Per case and fingertip, over the in-play pulse steps: world displacement of the link origin, of the pen's
    surface point under it, and their difference, each along the circumferential direction of the frame at ``p0``
    (m). ``obs`` is ``(C, T, d)``; observation ``t`` (``p0 < t <= p1``) is the state after commanded step ``t - 1``."""
    import torch

    C, T = obs.shape[:2]
    stop = min(p1, T - 1)
    n = stop + 1 - p0
    flat = torch.as_tensor(np.nan_to_num(obs[:, p0:stop + 1]).reshape(-1, obs.shape[-1]), dtype=torch.float64)
    s = cc.state_from_obs(flat)
    fr = cc.capsule_frames(s["pen_pos"], s["pen_quat"], s["tip_pos"])
    tip_v = cc.tip_origin_linvel(s)
    surf_v = cc.surface_point_velocity(s["pen_pos"], s["pen_linvel"], s["pen_angvel"], fr["point"])
    K = tip_v.shape[1]
    tip_v, surf_v = tip_v.reshape(C, n, K, 3).numpy(), surf_v.reshape(C, n, K, 3).numpy()
    circ0 = fr["R"].reshape(C, n, K, 3, 3).numpy()[:, 0, :, :, CIRC]               # (C, K, 3)
    out = {key: np.zeros((C, len(tips))) for key in ("tip", "surface", "relative")}
    for i in range(C):
        last = int(min(stop, length[i] - 1))
        m = max(0, last - p0)
        for j, k in enumerate(tips):
            if m:
                vt = (tip_v[i, 1:m + 1, k] @ circ0[i, k]).sum() * dt
                vs = (surf_v[i, 1:m + 1, k] @ circ0[i, k]).sum() * dt
                out["tip"][i, j], out["surface"][i, j], out["relative"][i, j] = vt, vs, vt - vs
    return out


def _t(values) -> dict:
    from lares.eval.stats import t_interval

    return t_interval([float(v) for v in values])


def authority(diff, eligible, full: float, thresholds: dict) -> dict:
    """One fingertip's adequacy from its ``+`` minus ``-`` circumferential displacement (m) over eligible cases."""
    d = np.asarray(diff, dtype=np.float64)[np.asarray(eligible, dtype=bool)]
    if d.size < 2:
        return {"n": int(d.size), "adequate": False}
    t = _t(d)
    frac, ratio = float((d > 0).mean()), float(np.median(d) / full)
    ok = bool(frac >= float(thresholds["positive_fraction_min"]) and t["low"] is not None and t["low"] > 0
              and ratio >= float(thresholds["median_ratio_min"]))
    return {"n": int(d.size), "positive_fraction": frac, "displacement_m": t, "median_ratio": ratio, "adequate": ok}


def chain_fractions(extra: dict, tips, length, p0: int) -> dict:
    """Per commanded fingertip: the fraction of in-play pulse steps on which each bound type (and the incumbent's
    own saturation) was active on any joint of that finger's chain, and the mean absolute joint displacement."""
    chain = np.asarray(extra["chain"], dtype=bool)                                   # (5, n_ctrl), wrist included
    out = {}
    for k in tips:
        name = cc.FINGERTIPS[k]
        row = {}
        for key in BOUNDS + ("incumbent_saturated",):
            arr = extra.get(f"audit_{key}")
            if arr is None:
                continue
            steps, C = arr.shape[:2]
            live = (np.arange(steps)[:, None] + p0) < np.asarray(length)[None, :]
            hit = (arr & chain[k][None, None, :]).any(-1)
            row[key] = float(hit[live].mean()) if live.any() else None
        q = extra.get("audit_q")
        if q is not None and q.shape[0] > 1:
            row["mean_abs_joint_travel_rad"] = float(np.abs(q[-1] - q[0])[:, chain[k]].mean())
        out[name] = row
    return out


def audit_summary(cells: dict, protocol_cfg: dict) -> dict:
    """The audit's statistics from the three saved cells (``{name: path}``)."""
    frozen = require_frozen(protocol_cfg)
    win, dt = frozen["windows"], float(protocol_cfg["control_dt"])
    p0, p1, stop = int(win["pulse_start"]), int(win["pulse_stop"]), int(protocol_cfg["follow_through_stop"])
    tips, th = tip_indices(protocol_cfg), protocol_cfg["audit_thresholds"]
    data = {name: load_cell(path) for name, path in cells.items()}
    extras = {name: dict(np.load(path.replace(".json", ".extra.npz"))) for name, path in cells.items()}
    ztrace = data["zero"][0]
    direction, mean_rate, eligible = case_direction(ztrace["spin_rate"], ztrace["length"],
                                                    protocol_cfg["roll_window"], float(protocol_cfg["min_abs_mean_rate"]))
    prefix = {name: rg.in_play_prefix(tr["obs"], tr["length"], p0) for name, (tr, _) in data.items()}
    prefix_diff = max(float(np.abs(prefix[name] - prefix["zero"]).max()) for name in CELLS[1:])
    identity = float(extras["zero"]["identity_max_abs"])

    prog = {name: roll_progress(tr["spin_rate"], tr["length"], p0, stop, dt) for name, (tr, _) in data.items()}
    pulse_only = {name: roll_progress(tr["spin_rate"], tr["length"], p0, p1, dt) for name, (tr, _) in data.items()}
    e = eligible
    pos_neg = prog["coherent_pos"] - prog["coherent_neg"]
    pro = np.where(direction > 0, prog["coherent_pos"], prog["coherent_neg"])
    anti = np.where(direction > 0, prog["coherent_neg"], prog["coherent_pos"])
    primary = _t(pos_neg[e]) if e.sum() >= 2 else {"n": int(e.sum())}

    motion = {name: tip_motion(data[name][0]["obs"], data[name][0]["length"], tips, p0, p1, dt)
              for name in CELLS[1:]}
    full = 2.0 * float(win["pulse_speed"]) * (p1 - p0) * dt
    auth, tip_report = {}, {}
    for j, k in enumerate(tips):
        name = cc.FINGERTIPS[k]
        auth[name] = authority(motion["coherent_pos"]["tip"][:, j] - motion["coherent_neg"]["tip"][:, j], e, full, th)
        tip_report[name] = {key: _t((motion["coherent_pos"][key][:, j] - motion["coherent_neg"][key][:, j])[e])
                            for key in ("surface", "relative")} if e.sum() >= 2 else {}
    adequate = all(a["adequate"] for a in auth.values())

    target = float(protocol_cfg["threshold"])
    if prefix_diff > 0.0 or identity > 0.0:
        outcome = VOID
    elif int(e.sum()) < int(th["min_eligible_cases"]):
        outcome = "inconclusive_coverage"
    elif not adequate:
        outcome = INADEQUATE
    elif primary["mean"] >= target and primary["low"] > 0:
        outcome = "supported"
    elif primary["high"] < target:
        outcome = "refuted"
    else:
        outcome = "inconclusive"

    def cell_row(name):
        rows = data[name][1]
        return {"sustained": int(sum(r["sustained"] for r in rows)), "drops": int(sum(not r["completed"] for r in rows)),
                "drops_by_follow_through": int(sum((not r["completed"]) and r["length"] <= stop for r in rows)),
                "timeout_rate": cg.timeout_rate(rows)}

    return {
        "study": STUDY, "outcome": outcome, "prediction_threshold_rad": target,
        "pro_minus_anti_progress_rad": primary,
        "pro_minus_anti_note": "equals progress(+ cell) - progress(- cell) for every case; the case direction enters "
                               "only through eligibility",
        "secondary": {} if e.sum() < 2 else {
            "pro_minus_zero_rad": _t((direction * (pro - prog["zero"]))[e]),
            "anti_minus_zero_rad": _t((direction * (anti - prog["zero"]))[e]),
            "pulse_window_only_pos_minus_neg_rad": _t((pulse_only["coherent_pos"] - pulse_only["coherent_neg"])[e]),
            "zero_cell_aligned_progress_rad": _t((direction * prog["zero"])[e]),
            "positive_direction_cases_pos_minus_neg_rad": _t(pos_neg[e & (direction > 0)]),
            "negative_direction_cases_pos_minus_neg_rad": _t(pos_neg[e & (direction < 0)])},
        "authority": {"adequate": adequate, "full_displacement_m": full, "by_fingertip": auth},
        "pen_surface_and_relative_displacement_m": tip_report,
        "bounds_by_chain": {name: chain_fractions(extras[name], tips, data[name][0]["length"], p0) for name in CELLS[1:]},
        "eligibility": {"eligible": int(e.sum()), "cases": int(len(e)), "positive_direction": int((e & (direction > 0)).sum()),
                        "negative_direction": int((e & (direction < 0)).sum()),
                        "median_abs_mean_rate": float(np.median(np.abs(mean_rate[e]))) if e.any() else None},
        "checks": {"prefix_max_abs_diff": prefix_diff, "zero_identity_max_abs": identity},
        "cells": {name: cell_row(name) for name in CELLS},
        "unavailable": ["contact impulses (the env has no contact sensor)"],
        "per_case": [{"case": i, "eligible": bool(e[i]), "direction": float(direction[i]),
                      "mean_rate": float(mean_rate[i]), **{f"progress_{n}": float(prog[n][i]) for n in CELLS}}
                     for i in range(len(e))],
        "frozen_sha256": protocol_cfg["frozen_sha256"], "episodes": len(e) * len(CELLS),
        "limitation": "one 0.5 s pulse of 0.02 m/s on two fingertips through one frozen decoder on one incumbent; "
                      "fingertip link origins near an ideal capsule; says nothing about task success",
    }


# ---------------------------------------------------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------------------------------------------------


def run(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    frozen = require_frozen(protocol_cfg)
    freeze_json(os.path.join(out_dir, "frozen.json"), {"frozen": frozen, "sha256": protocol_cfg["frozen_sha256"],
                                                       "fingertips": list(protocol_cfg["fingertips"])}, "frozen block")
    mpath = cases.require_frozen(protocol_cfg["manifest"])
    cells = {}
    for name in CELLS:
        cells[name] = os.path.join(out_dir, f"{name}.json")
        study_cell(protocol_cfg, config_path, mpath, name, cells[name])
    summary = audit_summary(cells, protocol_cfg)
    summary["manifest"] = cases.manifest_record(mpath)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


__all__ = ["CELLS", "CoherentPulseController", "STUDY", "audit_summary", "authority", "build_manifests",
           "case_direction", "chain_fractions", "make_controller", "require_frozen", "roll_progress", "run",
           "tip_motion"]
