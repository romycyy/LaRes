"""R-068: confirm D-051's capture-curl clamp against the frozen parent on fresh paired cases, under the strict rule.

D-051 selected the clamp on a diagnostic manifest (+0.125 [-0.014, +0.264], drops 13 to 5). Confirmation is the exact
same one-line substitution (R-065's ``curl_clamp``: ``extra_curl = 0.0 * capture_curl * (...)``), audited and
identity-gated offline, against the parent on one frozen 200-case manifest: cases 0-9 unused, a 30-case promotion
panel (cases 10-39, the strict paired rule) and a 160-case fresh panel (cases 40-199, the prediction). Each controller runs each panel in its own fresh process.
"""

from __future__ import annotations

import json
import os

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin import channel_lesions as cl
from lares.envs.isaac_pen_spin import scalar_tuning as st
from lares.envs.isaac_pen_spin.learner_recovery import _load, check_learner, make_learner
from lares.envs.isaac_pen_spin.protocols import arm_summary, child_cell, failure_split
from lares.eval.manifest import EvaluationManifest

STUDY = "r068_capture_curl_confirmation"
STAGE = 3
LESION = "curl_clamp"
ARMS = ("parent", "zero_curl")
PANELS = ("promotion", "fresh")
GAIN_THRESHOLD = 0.10


def manifest_path(protocol_cfg: dict) -> str:
    return protocol_cfg["manifest"]


def roles(protocol_cfg: dict) -> dict:
    return {k: (int(v[0]), int(v[1])) for k, v in protocol_cfg["roles"].items()}


def build_manifests(protocol_cfg: dict) -> list[str]:
    from lares.eval.manifest import assert_disjoint

    path = manifest_path(protocol_cfg)
    m = cases.build_manifest(STAGE, int(protocol_cfg["pool_seed"]), int(protocol_cfg["cases"]), label="r068_block")
    written = cases.freeze_manifest(path, m)
    others = [EvaluationManifest.load(p) for p in protocol_cfg.get("disjoint_from", [])]
    assert_disjoint([m] + others)
    return [path] if written else []


def make_controller(arm: str, protocol_cfg: dict, out_dir: str | None = None):
    if arm == "parent":
        return make_learner(protocol_cfg)
    if arm == "zero_curl":
        return cl.lesioned_learner(protocol_cfg, LESION, 0.0, arm)
    raise ValueError(f"unknown arm {arm!r}; one of {ARMS}")


def panel_contrast(parent_rows, rows, parent_sat, sat) -> dict:
    c = st.contrast(rows, parent_rows)
    return {**c,
            "arms": {"parent": {**arm_summary(parent_rows), **failure_split(parent_rows), **cl.tail_failure_split(parent_rows),
                                "saturation": {k: v for k, v in parent_sat.items() if k != "drive_unsaturated_fraction_per_case"}},
                     "zero_curl": {**arm_summary(rows), **failure_split(rows), **cl.tail_failure_split(rows),
                                   "saturation": {k: v for k, v in sat.items() if k != "drive_unsaturated_fraction_per_case"}}},
            "drops_not_increased": bool(c["drop_counts"][0] <= c["drop_counts"][1]),
            "drop_increase_established": bool(c["drops"]["low"] is not None and c["drops"]["low"] > 0.0)}


def aggregate(rows: dict, sats: dict) -> dict:
    out = {"panels": {p: panel_contrast(rows["parent"][p], rows["zero_curl"][p], sats["parent"][p], sats["zero_curl"][p])
                      for p in PANELS}}
    fresh, promo = out["panels"]["fresh"], out["panels"]["promotion"]
    t = fresh["success"]
    out["paired_success_difference"] = t["mean"]
    out["prediction"] = {"threshold": GAIN_THRESHOLD,
                         "supported": bool(t["mean"] is not None and t["mean"] >= GAIN_THRESHOLD and t["low"] is not None and t["low"] > 0.0),
                         "refuted": bool(t["high"] is not None and t["high"] < GAIN_THRESHOLD),
                         "retention_refuted": bool(fresh["drop_increase_established"] or promo["drop_increase_established"])}
    out["qualifies_repair_parent"] = bool(promo["promotes"] and t["low"] is not None and t["low"] > 0.0
                                          and fresh["drops_not_increased"] and promo["drops_not_increased"])
    return out


def _child(config_path: str, manifest: str, arm: str, out_path: str) -> None:
    child_cell(config_path, manifest, arm, out_path, study=STUDY, record_obs=True, skip_existing=True)


def run_confirmation(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    from lares.utils.hashing import sha256_file

    lc = check_learner(protocol_cfg)
    with open(lc["source"], encoding="utf-8") as f:
        source = f.read()
    audit = cl.source_audit(source)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "source_audit.json"), "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    if not audit["passes"]:
        raise SystemExit(f"source audit failed, no episode run: {audit}")
    with open(os.path.join(out_dir, "zero_curl_source.py"), "w", encoding="utf-8") as f:
        f.write(cl.substituted_source(source, LESION, 0.0))
    gate_path = os.path.join(out_dir, "identity_gate.json")
    if os.path.exists(gate_path):
        with open(gate_path, encoding="utf-8") as f:
            gate = json.load(f)
    else:
        gate = cl.offline_identity_gate(protocol_cfg, protocol_cfg["identity_archive"])
        with open(gate_path, "w", encoding="utf-8") as f:
            json.dump(gate, f, indent=2)
    if not gate["lesions"][LESION]["passes"]:
        raise SystemExit(f"offline identity gate failed for {LESION}, no episode run: {gate}")
    mpath = manifest_path(protocol_cfg)
    if not os.path.exists(mpath):
        raise SystemExit(f"frozen manifest {mpath} missing; build and commit it before the run")
    m = EvaluationManifest.load(mpath)
    role = cases.role_manifests(m, roles(protocol_cfg), expected_cases=int(protocol_cfg["cases"]), stage=STAGE)
    paths = {}
    for p, rm in role.items():
        paths[p] = os.path.join(out_dir, f"{p}.yaml")
        cases.freeze_manifest(paths[p], rm)
    learner = make_learner(protocol_cfg)
    sd = learner.policy.state_dict()
    params = {"spin_target": float(sd["spin_target"]), "speed_gain": float(sd["speed_gain"])}
    rows, sats = {a: {} for a in ARMS}, {a: {} for a in ARMS}
    for p in PANELS:
        for a in ARMS:
            cell = os.path.join(out_dir, f"{a}_{p}.json")
            _child(config_path, paths[p], a, cell)
            trace, rows[a][p] = _load(cell)
            sats[a][p] = st.saturation(trace, params)
    summary = aggregate(rows, sats)
    summary["identity_gate"] = gate["lesions"][LESION]
    summary["source_audit"] = audit
    summary["manifest"] = {"path": mpath, "sha256": sha256_file(mpath), "id": m.manifest_id}
    summary["episodes"] = len(ARMS) * sum(len(role[p]) for p in PANELS)
    summary["decision"] = {"repair_parent": ({"controller": "zero_curl", "source": os.path.join(out_dir, "zero_curl_source.py"),
                                              "fresh_success": summary["panels"]["fresh"]["arms"]["zero_curl"]["success"]}
                                             if summary["qualifies_repair_parent"] else None),
                           "rule": "strict promotion on the 30 promotion cases, fresh lower bound above zero, no observed "
                                   "drop increase on either panel"}
    summary["limitation"] = "one block, one learner; the clamp is confirmed or not for this parent on these panels only"
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


run = run_confirmation

__all__ = ["ARMS", "GAIN_THRESHOLD", "LESION", "PANELS", "STUDY", "aggregate", "build_manifests", "make_controller",
           "panel_contrast", "run", "run_confirmation"]
