"""R-062: does the frozen R-046 learner generate its axial spin through its periodic stroke?

D-048 and D-049 closed scalar tuning of the learner and measured its spin-feedback channel as saturated and nearly
powerless. This study asks where the spin comes from at all. The parent's joint target is
``grasp_pose + curl terms + stabilization + rolling * gait``, where ``gait`` is built only from the periodic stroke
(``sin_phase`` through ``drive`` and the fingertip tangents) and the periodic load ``2 * stance - 1``; both have zero
cycle mean, so replacing the periodic contribution by its cycle mean is exactly the removal of the ``rolling * gait``
term. The lesion is a frozen source substitution (``+ rolling * gait`` becomes ``+ 0.0 * rolling * gait``; the gates
are still recorded; parameters, grasp, centering, tilt correction and execution unchanged). The same machinery with the
factor ``1.0`` is the identity gate: it must be bit-identical to the parent on every case (D-045's gate) before the
lesion runs. One 80-case Stage-3 diagnostic manifest (seed 4560), one fresh process per arm (D-034).
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin import scalar_tuning as st
from lares.envs.isaac_pen_spin.learner_recovery import _load, check_learner, identity_gate, make_learner
from lares.envs.isaac_pen_spin.protocols import arm_summary, build_single_manifest, child_cell, failure_split

STUDY = "r062_stroke_lesion"
STAGE = 3
ARMS = ("parent", "parent_identity", "stroke_lesion")
FACTORS = {"parent_identity": 1.0, "stroke_lesion": 0.0}
LOSS_THRESHOLD = 0.15

_GAIT_TERM = re.compile(r"\+ rolling \* gait\n")
_AUDIT = {
    "gait_term_in_target": r"\+ rolling \* gait\n\s*\)",
    "rolling_is_retained_times_contact": r"rolling = retained \* contact",
    "stroke_is_periodic": r"stroke = \(\s*self\.stroke_amp\s*\* drive\.unsqueeze\(1\)\s*\* sin_phase\.unsqueeze\(2\)\s*\* tangent",
    "load_is_periodic": r"load = self\.lift_amp \* \(2\.0 \* stance - 1\.0\)",
    "stance_from_cos_phase": r"stance = torch\.sigmoid\(self\.stance_sharpness \* cos_phase\)",
    "target_has_grasp_curl_stabilization": r"self\.grasp_pose\.unsqueeze\(0\)\s*\+ \(extra_curl - 0\.03 \* passive_curl\) \* self\.curl_mask\s*\+ stabilization",
}


def source_audit(source: str) -> dict:
    """Static check that the parent's target decomposes as the lesion assumes; every pattern must match once."""
    out = {k: len(re.findall(p, source)) for k, p in _AUDIT.items()}
    out["passes"] = all(v == 1 for k, v in out.items() if k != "passes")
    out["cycle_mean_argument"] = ("sin_phase averages to zero over a roll cycle and stance = sigmoid(k cos_phase) averages "
                                  "to 0.5, so both periodic terms of gait have zero cycle mean; the cycle-mean replacement "
                                  "is the removal of rolling * gait")
    return out


def substituted_source(source: str, factor: float) -> str:
    if len(_GAIT_TERM.findall(source)) != 1:
        raise SystemExit("the parent source does not contain exactly one '+ rolling * gait' term")
    return _GAIT_TERM.sub(f"+ {float(factor):.1f} * rolling * gait\n", source)


def lesioned_learner(protocol_cfg: dict, factor: float, name: str):
    """The parent with the gait term scaled by ``factor`` (1.0 is the identity), same state, same execution."""
    from lares.envs.isaac_pen_spin.layout import OBS_LAYOUT
    from lares.envs.isaac_pen_spin.pipeline_child import SymbolicController

    lc = check_learner(protocol_cfg)
    with open(lc["source"], encoding="utf-8") as f:
        source = f.read()
    ctrl = SymbolicController({"source": substituted_source(source, factor), "state_path": lc["state"],
                               "obs_dim": sum(size for _, size, _ in OBS_LAYOUT), "action_dim": cases.NUM_ACTIONS,
                               "fixed_action_scale": float(lc["fixed_action_scale"]),
                               "execution": "clamp(mean, -1, 1)"}, cases.STAGES[STAGE].task_id)
    ctrl.name = name
    ctrl.params = {**ctrl.params, "arm": name, "gait_factor": float(factor),
                   "substitution": "+ rolling * gait -> + <factor> * rolling * gait"}
    return ctrl


def manifest_path(protocol_cfg: dict) -> str:
    return protocol_cfg["manifest"]


def build_manifests(protocol_cfg: dict) -> list[str]:
    return build_single_manifest(protocol_cfg, "r062_diagnostic", STAGE)


def make_controller(arm: str, protocol_cfg: dict, out_dir: str | None = None):
    if arm == "parent":
        return make_learner(protocol_cfg)
    if arm in FACTORS:
        return lesioned_learner(protocol_cfg, FACTORS[arm], arm)
    raise ValueError(f"unknown arm {arm!r}; one of {ARMS}")


def _child(config_path: str, manifest: str, arm: str, out_path: str) -> None:
    child_cell(config_path, manifest, arm, out_path, study=STUDY, record_obs=True, skip_existing=True)


def aggregate(parent_rows, lesion_rows, parent_sat: dict, lesion_sat: dict) -> dict:
    from lares.eval.stats import t_summary, win_loss_tie

    if [r["task_hash"] for r in parent_rows] != [r["task_hash"] for r in lesion_rows]:
        raise ValueError("lesion rows are not paired case for case with the parent")
    ds = [float(l["sustained"]) - float(p["sustained"]) for l, p in zip(lesion_rows, parent_rows)]
    dd = [float(not l["completed"]) - float(not p["completed"]) for l, p in zip(lesion_rows, parent_rows)]
    ts, td = t_summary(ds), t_summary(dd)
    kept = [i for i, p in enumerate(parent_rows) if p["sustained"]]
    spin_on_kept = {arm: (float(np.mean([rows[i]["mean_abs_spin"] for i in kept if rows[i]["mean_abs_spin"] is not None]))
                          if kept else None)
                    for arm, rows in (("parent", parent_rows), ("stroke_lesion", lesion_rows))}
    threshold = cases.STAGES[STAGE].success_spin_rate
    loss = bool(ts["high"] is not None and ts["high"] < -LOSS_THRESHOLD)
    drops_ok = bool(td["low"] is not None and td["low"] <= 0.0)
    slow = bool(spin_on_kept["stroke_lesion"] is not None and spin_on_kept["stroke_lesion"] < threshold)
    return {"paired_success_difference": ts["mean"], "lesion_minus_parent": ts, "win_loss_tie": win_loss_tie(ds),
            "drops_lesion_minus_parent": td,
            "arms": {"parent": {**arm_summary(parent_rows), **failure_split(parent_rows),
                                "saturation": {k: v for k, v in parent_sat.items() if k != "drive_unsaturated_fraction_per_case"}},
                     "stroke_lesion": {**arm_summary(lesion_rows), **failure_split(lesion_rows),
                                       "saturation": {k: v for k, v in lesion_sat.items() if k != "drive_unsaturated_fraction_per_case"}}},
            "parent_sustained_cases": len(kept), "mean_abs_spin_on_parent_sustained_cases": spin_on_kept,
            "lesion_successes_on_parent_sustained_cases": int(sum(lesion_rows[i]["sustained"] for i in kept)),
            "interpretation": {"loss_at_least_threshold": loss, "loss_not_carried_by_drops": drops_ok,
                               "lesion_spin_below_stage_threshold": slow, "stage_spin_threshold": threshold,
                               "stroke_is_spin_source": bool(loss and drops_ok and slow),
                               "drop_driven_inconclusive": bool(loss and not drops_ok),
                               "prediction_refuted": bool(ts["low"] is not None and ts["low"] > -LOSS_THRESHOLD)}}


def run_stroke_lesion(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    """Driver: audit, parent, identity (gate), lesion, aggregate. Nothing is rerun if its output exists."""
    from lares.utils.hashing import sha256_file

    lc = check_learner(protocol_cfg)
    with open(lc["source"], encoding="utf-8") as f:
        source = f.read()
    audit = source_audit(source)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "source_audit.json"), "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    if not audit["passes"]:
        raise SystemExit(f"source audit failed, no episode run: {audit}")
    with open(os.path.join(out_dir, "stroke_lesion_source.py"), "w", encoding="utf-8") as f:
        f.write(substituted_source(source, 0.0))
    m = manifest_path(protocol_cfg)
    if not os.path.exists(m):
        raise SystemExit(f"frozen manifest {m} missing; build and commit it before the run")
    out = {arm: os.path.join(out_dir, f"{arm}.json") for arm in ARMS}
    for arm in ("parent", "parent_identity"):
        _child(config_path, m, arm, out[arm])
    base, base_rows = _load(out["parent"])
    ident, ident_rows = _load(out["parent_identity"])
    gate = identity_gate(base, base_rows, ident, ident_rows)
    if not gate["passes"]:
        with open(os.path.join(out_dir, "gate_failed.json"), "w") as f:
            json.dump(gate, f, indent=2)
        raise SystemExit(f"identity gate failed, lesion not run: {gate}")
    _child(config_path, m, "stroke_lesion", out["stroke_lesion"])
    les, les_rows = _load(out["stroke_lesion"])
    learner = make_learner(protocol_cfg)
    sd = learner.policy.state_dict()
    params = {"spin_target": float(sd["spin_target"]), "speed_gain": float(sd["speed_gain"])}
    summary = aggregate(base_rows, les_rows, st.saturation(base, params), st.saturation(les, params))
    summary["identity_gate"] = gate
    summary["source_audit"] = audit
    summary["manifest"] = {"path": m, "sha256": sha256_file(m)}
    summary["episodes"] = len(ARMS) * int(protocol_cfg["cases"])
    summary["limitation"] = ("a lesion measures dependence of this learner's success on its periodic stroke; it does not "
                             "say how the stroke should be changed, nor anything about other controllers")
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


run = run_stroke_lesion

__all__ = ["ARMS", "FACTORS", "LOSS_THRESHOLD", "STUDY", "aggregate", "build_manifests", "lesioned_learner",
           "make_controller", "run", "run_stroke_lesion", "source_audit", "substituted_source"]
