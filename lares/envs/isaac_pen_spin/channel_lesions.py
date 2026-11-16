"""R-065: which reactive channel carries the spin that survives the stroke lesion (D-050)?

D-050 removed the frozen R-046 learner's periodic stroke and the pen still spun: the lesion kept 6/80 successes and the
same overall in-play spin rate. Three reactive channels remain, each one audited line of the parent source, each
lesioned by a frozen substitution at factor 0.0 (R-062's machinery) and gated offline at factor 1.0:

* ``rotation_correction_lesion``: ``rotation_correction`` (tilt_gain and swing_damping terms) zeroed.
* ``centering_clamp``: the centering error ``home - obj`` clamped to its zero-error reference, so ``center = 0``; every
  other branch still reads the true observation (``retained`` keeps its own distance).
* ``curl_clamp``: ``extra_curl`` (the capture curl driven by the contact and retained gates) clamped to 0.

Identity gate (exact, no episode): each substitution at factor 1.0 must reproduce the parent's actions bit for bit on
the archived parent observations of D-050. Arms then run one fresh process each on a frozen 80-case Stage-3 manifest.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin import scalar_tuning as st
from lares.envs.isaac_pen_spin.learner_recovery import learner_source, make_learner
from lares.envs.isaac_pen_spin.protocols import (actions_match, arm_summary, build_single_manifest, cached_json,
                                                 failure_split, learner_drive_params, load_cell, manifest_path,
                                                 study_cell)

STUDY = "r065_channel_lesions"
STAGE = 3
LESIONS = {
    "rotation_correction_lesion": (re.compile(r"rotation_correction = \(\n"), "rotation_correction = {f} * (\n"),
    "centering_clamp": (re.compile(r"error = home - obj\n"), "error = {f} * (home - obj)\n"),
    "curl_clamp": (re.compile(r"extra_curl = self\.capture_curl \* \(\n"), "extra_curl = {f} * self.capture_curl * (\n"),
}
ARMS = ("parent",) + tuple(LESIONS)
PRIMARY = "rotation_correction_lesion"
LOSS_THRESHOLD = 0.10
_AUDIT = {
    "rotation_correction_terms": r"rotation_correction = \(\s*self\.tilt_gain \* tilt_correction\s*- self\.swing_damping\s*\* torch\.tanh\(transverse_speed / 4\.0\)\s*\* transverse_direction\s*\)",
    "centering_from_error": r"error = home - obj\n\s*distance = torch\.norm\(error, dim=1, keepdim=True\)",
    "center_term": r"center = \(\s*self\.center_gain\s*\* torch\.tanh\(distance / self\.center_length\)\s*\* center_direction\s*\)",
    "capture_curl": r"extra_curl = self\.capture_curl \* \(\s*\(1\.0 - contact\) \+ 0\.75 \* \(1\.0 - retained\)\s*\)",
    "target_composition": r"self\.grasp_pose\.unsqueeze\(0\)\s*\+ \(extra_curl - 0\.03 \* passive_curl\) \* self\.curl_mask\s*\+ stabilization\s*\+ rolling \* gait",
}


def source_audit(source: str) -> dict:
    out = {k: len(re.findall(p, source)) for k, p in _AUDIT.items()}
    out.update({f"substitution:{name}": len(rx.findall(source)) for name, (rx, _) in LESIONS.items()})
    out["passes"] = all(v == 1 for k, v in out.items() if k != "passes")
    return out


def substituted_source(source: str, lesion: str, factor: float) -> str:
    rx, repl = LESIONS[lesion]
    if len(rx.findall(source)) != 1:
        raise SystemExit(f"the parent source does not contain exactly one {lesion} site")
    return rx.sub(repl.format(f=f"{float(factor):.1f}"), source)


def lesioned_learner(protocol_cfg: dict, lesion: str, factor: float, name: str):
    ctrl = make_learner(protocol_cfg, lambda source: substituted_source(source, lesion, factor))
    ctrl.name = name
    ctrl.params = {**ctrl.params, "arm": name, "lesion": lesion, "factor": float(factor),
                   "substitution": LESIONS[lesion][1]}
    return ctrl


def build_manifests(protocol_cfg: dict) -> list[str]:
    return build_single_manifest(protocol_cfg, "r065_diagnostic", STAGE)


def make_controller(arm: str, protocol_cfg: dict, out_dir: str | None = None):
    if arm == "parent":
        return make_learner(protocol_cfg)
    if arm in LESIONS:
        return lesioned_learner(protocol_cfg, arm, 0.0, arm)
    raise ValueError(f"unknown arm {arm!r}; one of {ARMS}")


def offline_identity_gate(protocol_cfg: dict, archive_npz: str) -> dict:
    """Each lesion at factor 1.0 must reproduce the parent's actions bit for bit on archived observations."""
    data = np.load(archive_npz)
    obs, length = data["obs"], data["length"]
    parent = make_learner(protocol_cfg)
    out = {"archive": archive_npz, "cases": int(obs.shape[0]), "lesions": {}}
    for lesion in LESIONS:
        ctrl = lesioned_learner(protocol_cfg, lesion, 1.0, f"{lesion}:identity")
        out["lesions"][lesion] = actions_match(parent, ctrl, obs, length)
    out["passes"] = all(v["passes"] for v in out["lesions"].values())
    return out


def tail_failure_split(rows: list[dict], stage: int = STAGE) -> dict:
    """Completed-but-not-sustained episodes split by the final centre distance (off-centre) or else slow spin."""
    tol = cases.STAGES[stage].success_center_tolerance
    tail = [r for r in rows if r["completed"] and not r["sustained"]]
    off = [r for r in tail if r["final_center_dist"] is not None and r["final_center_dist"] >= tol]
    return {"tail_failures": len(tail), "off_centre": len(off), "slow_spin": len(tail) - len(off)}


def arm_contrast(parent_rows, rows, parent_sat, sat, threshold: float = LOSS_THRESHOLD) -> dict:
    from lares.eval.stats import t_summary, win_loss_tie

    if [r["task_hash"] for r in parent_rows] != [r["task_hash"] for r in rows]:
        raise ValueError("rows are not paired case for case with the parent")
    ds = [float(l["sustained"]) - float(p["sustained"]) for l, p in zip(rows, parent_rows)]
    dd = [float(not l["completed"]) - float(not p["completed"]) for l, p in zip(rows, parent_rows)]
    ts, td = t_summary(ds), t_summary(dd)
    kept = [i for i, p in enumerate(parent_rows) if p["sustained"]]
    spin_kept = (float(np.mean([rows[i]["mean_abs_spin"] for i in kept if rows[i]["mean_abs_spin"] is not None]))
                 if kept else None)
    spin_thr = cases.STAGES[STAGE].success_spin_rate
    loss = bool(ts["high"] is not None and ts["high"] < -threshold)
    drops_ok = bool(td["low"] is not None and td["low"] <= 0.0)
    slow = bool(spin_kept is not None and spin_kept < spin_thr)
    return {"minus_parent": ts, "win_loss_tie": win_loss_tie(ds), "drops_minus_parent": td,
            "summary": {**arm_summary(rows), **failure_split(rows), **tail_failure_split(rows),
                        "saturation": {k: v for k, v in sat.items() if k != "drive_unsaturated_fraction_per_case"}},
            "mean_abs_spin_on_parent_sustained_cases": spin_kept,
            "successes_on_parent_sustained_cases": int(sum(rows[i]["sustained"] for i in kept)),
            "interpretation": {"threshold": threshold, "loss_at_least_threshold": loss,
                               "loss_established": bool(ts["high"] is not None and ts["high"] < 0.0),
                               "loss_not_carried_by_drops": drops_ok, "spin_below_stage_threshold": slow,
                               "spin_channel": bool(loss and drops_ok and slow),
                               "retention_channel": bool(ts["high"] is not None and ts["high"] < 0.0 and not drops_ok),
                               "prediction_refuted": bool(ts["low"] is not None and ts["low"] > -threshold)}}


def aggregate(rows: dict, sats: dict) -> dict:
    parent_rows = rows["parent"]
    out = {"arms": {arm: arm_contrast(parent_rows, rows[arm], sats["parent"], sats[arm]) for arm in LESIONS},
           "parent": {**arm_summary(parent_rows), **failure_split(parent_rows), **tail_failure_split(parent_rows),
                      "saturation": {k: v for k, v in sats["parent"].items() if k != "drive_unsaturated_fraction_per_case"},
                      "mean_abs_spin_on_own_sustained_cases": float(np.mean(
                          [r["mean_abs_spin"] for r in parent_rows if r["sustained"]])) if any(r["sustained"] for r in parent_rows) else None},
           "primary": PRIMARY}
    out["paired_success_difference"] = out["arms"][PRIMARY]["minus_parent"]["mean"]
    return out


def run_channel_lesions(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    """Driver: audit, offline identity gates, four arms, aggregate. Nothing is rerun if its output exists."""
    source = learner_source(protocol_cfg)
    audit = source_audit(source)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "source_audit.json"), "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    if not audit["passes"]:
        raise SystemExit(f"source audit failed, no episode run: {audit}")
    for lesion in LESIONS:
        with open(os.path.join(out_dir, f"{lesion}_source.py"), "w", encoding="utf-8") as f:
            f.write(substituted_source(source, lesion, 0.0))
    gate = cached_json(os.path.join(out_dir, "identity_gate.json"),
                       lambda: offline_identity_gate(protocol_cfg, protocol_cfg["identity_archive"]))
    if not gate["passes"]:
        raise SystemExit(f"offline identity gate failed, no episode run: {gate}")
    m = cases.require_frozen(manifest_path(protocol_cfg))
    out = {arm: os.path.join(out_dir, f"{arm}.json") for arm in ARMS}
    rows, sats = {}, {}
    params = learner_drive_params(protocol_cfg)
    for arm in ARMS:
        study_cell(protocol_cfg, config_path, m, arm, out[arm])
        trace, rows[arm] = load_cell(out[arm])
        sats[arm] = st.saturation(trace, params)
    summary = aggregate(rows, sats)
    summary["identity_gate"] = gate
    summary["source_audit"] = audit
    summary["manifest"] = cases.manifest_record(m)
    summary["episodes"] = len(ARMS) * int(protocol_cfg["cases"])
    summary["limitation"] = ("single-channel lesions of one learner; redundant or interacting channels cannot be excluded, "
                             "and a loss carried by drops is retention dependence, not spin generation")
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


run = run_channel_lesions

__all__ = ["ARMS", "LESIONS", "LOSS_THRESHOLD", "PRIMARY", "STUDY", "aggregate", "arm_contrast", "build_manifests",
           "lesioned_learner", "make_controller", "offline_identity_gate", "run", "run_channel_lesions", "source_audit",
           "substituted_source", "tail_failure_split"]
