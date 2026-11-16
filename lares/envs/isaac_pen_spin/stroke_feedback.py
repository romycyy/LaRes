"""R-081: give spin feedback direct authority over the periodic stroke of the qualified zero-curl learner.

D-049 measured the learner's spin-drive channel as saturated and nearly powerless; D-050 showed the periodic stroke is a
real lever; D-057 qualified the zero-curl clamp as the repair parent, whose failures are now slow spin. This study
multiplies the stroke by a gain that depends on the pen's own axial rate: ``gain = clip(4 / max(|spin|, 1), 0.5, 2)``
with ``spin`` the policy's axial rate (rad/s) from the current observation. The control is the same gain frozen to its
mean over the clamp's archived in-play steps (stronger open-loop stroke, no feedback). The identity form returns
exactly 1.0 and must reproduce the clamp bit for bit offline. Compression stays zero; nothing else changes.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin import channel_lesions as cl
from lares.envs.isaac_pen_spin import scalar_tuning as st
from lares.envs.isaac_pen_spin.learner_recovery import check_learner, learner_source, make_learner
from lares.envs.isaac_pen_spin.protocols import (actions_match, arm_summary, build_single_manifest, cached_json,
                                                 failure_split, in_play, learner_drive_params, load_cell, manifest_path,
                                                 roles, study_cell)

STUDY = "r081_stroke_feedback"
STAGE = 3
ARMS = ("zero_curl", "stroke_feedback", "constant_gain")
PANELS = ("promotion", "fresh")
GAIN_THRESHOLD = 0.05
GAIN_TARGET, GAIN_FLOOR, GAIN_LO, GAIN_HI = 4.0, 1.0, 0.5, 2.0
ACTIVE_BAND = (0.9, 1.1)
_GAIT = re.compile(r"\+ rolling \* gait\n")


def gain_law(spin):
    """``clip(4 / max(|spin|, 1), 0.5, 2)`` elementwise (torch or numpy)."""
    try:
        import torch

        if torch.is_tensor(spin):
            return torch.clamp(GAIN_TARGET / torch.clamp(spin.abs(), min=GAIN_FLOOR), GAIN_LO, GAIN_HI)
    except ImportError:  # pragma: no cover
        pass
    return np.clip(GAIN_TARGET / np.maximum(np.abs(spin), GAIN_FLOOR), GAIN_LO, GAIN_HI)


def substituted_source(clamp_source: str) -> str:
    """``+ rolling * gait`` becomes ``+ self._stroke_gain(spin) * rolling * gait`` in the clamp's source."""
    if len(_GAIT.findall(clamp_source)) != 1:
        raise SystemExit("the clamp source does not contain exactly one '+ rolling * gait' term")
    return _GAIT.sub("+ self._stroke_gain(spin) * rolling * gait\n", clamp_source)


class StrokeGain:
    """Per-step gain with activity accounting. ``mode``: ``feedback``, ``constant`` (fixed value) or ``identity`` (1.0)."""

    def __init__(self, mode: str, constant: float | None = None):
        self.mode = mode
        self.constant = float(constant) if constant is not None else None
        self.sum, self.active, self.count = 0.0, 0, 0

    def __call__(self, spin):
        import torch

        if self.mode == "identity":
            return 1.0
        if self.mode == "constant":
            g = torch.full_like(spin, self.constant)
        elif self.mode == "feedback":
            g = gain_law(spin)
        else:
            raise ValueError(self.mode)
        with torch.no_grad():
            self.sum += float(g.sum())
            self.active += int(((g < ACTIVE_BAND[0]) | (g > ACTIVE_BAND[1])).sum())
            self.count += int(g.numel())
        return g

    def activity(self) -> dict:
        n = max(1, self.count)
        return {"mean_gain": self.sum / n if self.count else None, "active_fraction": self.active / n if self.count else None,
                "values": self.count, "band": list(ACTIVE_BAND)}


def _clamp(source: str) -> str:
    return cl.substituted_source(source, "curl_clamp", 0.0)


def clamp_source(protocol_cfg: dict) -> str:
    return _clamp(learner_source(protocol_cfg))


class StrokeFeedbackController:
    name = "stroke_feedback"

    def __init__(self, protocol_cfg: dict, mode: str, name: str, constant: float | None = None):
        self.inner = make_learner(protocol_cfg, lambda source: substituted_source(_clamp(source)))
        self.gain = StrokeGain(mode, constant)
        self.inner.policy._stroke_gain = self.gain
        self.name = name
        self._params = {**self.inner.params, "arm": name, "mode": mode, "constant": constant,
                        "gain_law": "clip(4 / max(|spin|, 1), 0.5, 2) on the policy's axial rate (rad/s)", "compression": 0.0}

    @property
    def params(self) -> dict:
        return {**self._params, "gain_activity": self.gain.activity()}

    def reset(self, ctx):
        self.inner.reset(ctx)

    def __call__(self, obs, t, ctx):
        return self.inner(obs, t, ctx)


def archived_gain_mean(protocol_cfg: dict, archives: list[str]) -> dict:
    """The feedback gain's arithmetic mean over the clamp's archived in-play steps (its own axial rate from the observations)."""
    vals = []
    sl = cases.obs_slices()
    for path in archives:
        data = np.load(path)
        obs, length = data["obs"], data["length"]
        for i, L in in_play(obs, length):
            o = obs[i, :L].astype(np.float64)
            omega = 5.0 * o[:, sl["obj_angvel"]]
            q = o[:, sl["obj_rot"]]
            q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            axis = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], axis=1)
            axis = axis / (np.linalg.norm(axis, axis=1, keepdims=True) + 1e-8)
            spin = np.sum(omega * axis, axis=1)
            vals.append(gain_law(spin))
    g = np.concatenate(vals) if vals else np.zeros(0)
    if g.size == 0:
        raise SystemExit("no archived in-play step; the constant control cannot be defined")
    return {"constant": float(g.mean()), "steps": int(g.size), "quartiles": np.percentile(g, [25, 50, 75]).tolist(),
            "active_fraction": float(np.mean((g < ACTIVE_BAND[0]) | (g > ACTIVE_BAND[1]))),
            "rule": "arithmetic mean of clip(4 / max(|spin|, 1), 0.5, 2) over the clamp's archived in-play steps"}


def freeze_constant(protocol_cfg: dict, out_dir: str) -> dict:
    path = os.path.join(out_dir, "constant_gain.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    from lares.utils.hashing import sha256_file

    archives = list(protocol_cfg["gain_archives"])
    for a in archives:
        if not os.path.exists(a):
            raise SystemExit(f"archived clamp run {a} missing; the constant control cannot be defined")
    out = archived_gain_mean(protocol_cfg, archives)
    out["archives"] = {a: sha256_file(a) for a in archives}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def preflight(protocol_cfg: dict) -> dict:
    path = protocol_cfg["qualification"]
    if not os.path.exists(path):
        raise SystemExit(f"qualification artifact {path} missing; the study runs nothing")
    with open(path, encoding="utf-8") as f:
        q = json.load(f)
    if not (q.get("qualifies") and q.get("repair_parent_eligible")):
        raise SystemExit(f"the zero-curl clamp is not a qualified repair parent ({path}); the study runs nothing")
    return q


def build_manifests(protocol_cfg: dict) -> list[str]:
    return build_single_manifest(protocol_cfg, "r081_block", STAGE)


def make_controller(arm: str, protocol_cfg: dict, out_dir: str):
    if arm == "zero_curl":
        return cl.lesioned_learner(protocol_cfg, "curl_clamp", 0.0, arm)
    if arm == "stroke_feedback":
        return StrokeFeedbackController(protocol_cfg, "feedback", arm)
    if arm == "constant_gain":
        with open(os.path.join(out_dir, "constant_gain.json"), encoding="utf-8") as f:
            c = json.load(f)["constant"]
        return StrokeFeedbackController(protocol_cfg, "constant", arm, c)
    raise ValueError(f"unknown arm {arm!r}; one of {ARMS}")


def offline_identity_gate(protocol_cfg: dict, archive_npz: str) -> dict:
    data = np.load(archive_npz)
    clamp = cl.lesioned_learner(protocol_cfg, "curl_clamp", 0.0, "clamp")
    ident = StrokeFeedbackController(protocol_cfg, "identity", "identity")
    return {"archive": archive_npz, **actions_match(clamp, ident, data["obs"], data["length"]),
            "reference": "the qualified zero-curl clamp"}


def _panel(rows_a, rows_b) -> dict:
    c = st.contrast(rows_a, rows_b)
    return {**c, "drops_not_increased": bool(c["drop_counts"][0] <= c["drop_counts"][1]),
            "drop_increase_established": bool(c["drops"]["low"] is not None and c["drops"]["low"] > 0.0)}


def aggregate(rows: dict, sats: dict, activity: dict | None = None) -> dict:
    out = {"arms": {a: {p: {**arm_summary(rows[a][p]), **failure_split(rows[a][p]), **cl.tail_failure_split(rows[a][p]),
                            "saturation": {k: v for k, v in sats[a][p].items() if k != "drive_unsaturated_fraction_per_case"}}
                        for p in PANELS} for a in ARMS},
           "contrasts": {p: {"stroke_feedback_minus_constant_gain": _panel(rows["stroke_feedback"][p], rows["constant_gain"][p]),
                             "stroke_feedback_minus_zero_curl": _panel(rows["stroke_feedback"][p], rows["zero_curl"][p]),
                             "constant_gain_minus_zero_curl": _panel(rows["constant_gain"][p], rows["zero_curl"][p])} for p in PANELS},
           "gain_activity": activity or {}}
    t = out["contrasts"]["fresh"]["stroke_feedback_minus_constant_gain"]["success"]
    out["paired_success_difference"] = t["mean"]
    out["prediction"] = {"threshold": GAIN_THRESHOLD,
                         "supported": bool(t["mean"] is not None and t["mean"] >= GAIN_THRESHOLD and t["low"] is not None and t["low"] > 0.0),
                         "refuted": bool(t["high"] is not None and t["high"] < GAIN_THRESHOLD)}
    pro = out["contrasts"]["promotion"]["stroke_feedback_minus_zero_curl"]
    fr = out["contrasts"]["fresh"]["stroke_feedback_minus_zero_curl"]
    out["adopts_stroke_feedback"] = bool(pro["promotes"] and fr["success"]["low"] is not None and fr["success"]["low"] > 0.0
                                         and pro["drops_not_increased"] and fr["drops_not_increased"])
    fb = (activity or {}).get("stroke_feedback", {}).get("fresh") or {}
    out["feedback_activated"] = bool(fb.get("active_fraction") is not None and fb["active_fraction"] >= 0.25)
    out["feedback_activated_without_gain"] = bool(out["feedback_activated"] and not out["prediction"]["supported"])
    return out


def run_stroke_feedback(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    q = preflight(protocol_cfg)
    check_learner(protocol_cfg)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "stroke_feedback_source.py"), "w", encoding="utf-8") as f:
        f.write(substituted_source(clamp_source(protocol_cfg)))
    constant = freeze_constant(protocol_cfg, out_dir)
    gate = cached_json(os.path.join(out_dir, "identity_gate.json"),
                       lambda: offline_identity_gate(protocol_cfg, protocol_cfg["identity_archive"]))
    if not gate["passes"]:
        raise SystemExit(f"offline identity gate failed, no episode run: {gate}")
    mpath = manifest_path(protocol_cfg)
    m = cases.load_frozen(mpath)
    role = cases.role_manifests(m, roles(protocol_cfg), expected_cases=int(protocol_cfg["cases"]), stage=STAGE)
    paths = cases.freeze_roles(role, out_dir)
    params = learner_drive_params(protocol_cfg)
    rows, sats, activity = {a: {} for a in ARMS}, {a: {} for a in ARMS}, {a: {} for a in ARMS}
    for p in PANELS:
        for a in ARMS:
            cell = os.path.join(out_dir, f"{a}_{p}.json")
            study_cell(protocol_cfg, config_path, paths[p], a, cell)
            trace, rows[a][p] = load_cell(cell)
            sats[a][p] = st.saturation(trace, params)
            with open(cell, encoding="utf-8") as f:
                activity[a][p] = (json.load(f).get("controller_params") or {}).get("gain_activity")
    summary = aggregate(rows, sats, activity)
    summary["constant_gain"] = constant
    summary["qualification"] = q
    summary["identity_gate"] = gate
    summary["manifest"] = cases.manifest_record(mpath, m)
    summary["episodes"] = len(ARMS) * sum(len(role[p]) for p in PANELS)
    summary["decision"] = {"adopt": summary["adopts_stroke_feedback"],
                           "route": ("adopt stroke feedback as the new incumbent" if summary["adopts_stroke_feedback"] else
                                     "R-043 next: feedback activates without a success gain" if summary["feedback_activated_without_gain"] else
                                     "keep the zero-curl parent"),
                           "rule": "strict promotion against the qualified parent on cases 10-39, fresh lower bound above zero, no more observed drops"}
    summary["limitation"] = "one block, one learner, one gain law; the constant control fixes the gain at the archived mean only"
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


run = run_stroke_feedback

__all__ = ["ARMS", "GAIN_THRESHOLD", "PANELS", "STUDY", "StrokeFeedbackController", "StrokeGain", "aggregate",
           "archived_gain_mean", "build_manifests", "clamp_source", "freeze_constant", "gain_law", "make_controller",
           "offline_identity_gate", "preflight", "run", "run_stroke_feedback", "substituted_source"]
