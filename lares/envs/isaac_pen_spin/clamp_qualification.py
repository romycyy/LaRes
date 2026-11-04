"""R-077: qualify the exact D-052 zero-curl clamp on a fresh frozen 30-case promotion panel under the standing rule.

Three fresh panels (D-052, D-054, D-055) all favoured the clamp over the parent, but each 30-case promotion panel missed
the strict rule (positive paired difference with the t-interval lower bound above zero) by 0.006 to 0.02. Pooling does
not replace the rule, so this is the rule's own measurement: the parent and the exact clamp, each in its own process, on
cases 10-39 of a frozen 40-case manifest (cases 0-9 unused). No selection, no extension, no retry.
"""

from __future__ import annotations

import json
import os

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin import channel_lesions as cl
from lares.envs.isaac_pen_spin import curl_confirmation as cc
from lares.envs.isaac_pen_spin import scalar_tuning as st
from lares.envs.isaac_pen_spin.learner_recovery import _load, check_learner, make_learner
from lares.envs.isaac_pen_spin.protocols import child_cell
from lares.eval.manifest import EvaluationManifest

STUDY = "r077_clamp_qualification"
STAGE = 3
ARMS = cc.ARMS  # ("parent", "zero_curl")
PANEL = "promotion"
GAIN_THRESHOLD = 0.05


def manifest_path(protocol_cfg: dict) -> str:
    return protocol_cfg["manifest"]


def roles(protocol_cfg: dict) -> dict:
    return {k: (int(v[0]), int(v[1])) for k, v in protocol_cfg["roles"].items()}


def build_manifests(protocol_cfg: dict) -> list[str]:
    from lares.eval.manifest import assert_disjoint

    path = manifest_path(protocol_cfg)
    m = cases.build_manifest(STAGE, int(protocol_cfg["pool_seed"]), int(protocol_cfg["cases"]), label="r077_block")
    written = cases.freeze_manifest(path, m)
    assert_disjoint([m] + [EvaluationManifest.load(p) for p in protocol_cfg.get("disjoint_from", [])])
    return [path] if written else []


make_controller = cc.make_controller


def aggregate(parent_rows, clamp_rows, parent_sat, clamp_sat) -> dict:
    panel = cc.panel_contrast(parent_rows, clamp_rows, parent_sat, clamp_sat)
    t = panel["success"]
    return {"panel": panel, "paired_success_difference": t["mean"],
            "prediction": {"threshold": GAIN_THRESHOLD,
                           "supported": bool(t["mean"] is not None and t["mean"] >= GAIN_THRESHOLD and t["low"] is not None and t["low"] > 0.0),
                           "refuted": bool(t["high"] is not None and t["high"] < GAIN_THRESHOLD)},
            "qualifies": bool(panel["promotes"]),
            "repair_parent_eligible": bool(panel["promotes"] and panel["drops_not_increased"])}


def _child(config_path: str, manifest: str, arm: str, out_path: str) -> None:
    child_cell(config_path, manifest, arm, out_path, study=STUDY, record_obs=True, skip_existing=True)


def run_qualification(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    from lares.utils.hashing import sha256_file

    lc = check_learner(protocol_cfg)
    os.makedirs(out_dir, exist_ok=True)
    with open(lc["source"], encoding="utf-8") as f:
        source = f.read()
    audit = cl.source_audit(source)
    if not audit["passes"]:
        raise SystemExit(f"source audit failed, no episode run: {audit}")
    with open(os.path.join(out_dir, "zero_curl_source.py"), "w", encoding="utf-8") as f:
        f.write(cl.substituted_source(source, cc.LESION, 0.0))
    gate_path = os.path.join(out_dir, "identity_gate.json")
    if os.path.exists(gate_path):
        with open(gate_path, encoding="utf-8") as f:
            gate = json.load(f)
    else:
        gate = cl.offline_identity_gate(protocol_cfg, protocol_cfg["identity_archive"])
        with open(gate_path, "w", encoding="utf-8") as f:
            json.dump(gate, f, indent=2)
    if not gate["lesions"][cc.LESION]["passes"]:
        raise SystemExit(f"offline identity gate failed for {cc.LESION}, no episode run: {gate}")
    mpath = manifest_path(protocol_cfg)
    if not os.path.exists(mpath):
        raise SystemExit(f"frozen manifest {mpath} missing; build and commit it before the run")
    m = EvaluationManifest.load(mpath)
    role = cases.role_manifests(m, roles(protocol_cfg), expected_cases=int(protocol_cfg["cases"]), stage=STAGE)
    ppath = os.path.join(out_dir, f"{PANEL}.yaml")
    cases.freeze_manifest(ppath, role[PANEL])
    sd = make_learner(protocol_cfg).policy.state_dict()
    params = {"spin_target": float(sd["spin_target"]), "speed_gain": float(sd["speed_gain"])}
    rows, sats = {}, {}
    for a in ARMS:
        cell = os.path.join(out_dir, f"{a}_{PANEL}.json")
        _child(config_path, ppath, a, cell)
        trace, rows[a] = _load(cell)
        sats[a] = st.saturation(trace, params)
    summary = aggregate(rows["parent"], rows["zero_curl"], sats["parent"], sats["zero_curl"])
    summary["identity_gate"] = gate["lesions"][cc.LESION]
    summary["source_audit"] = audit
    summary["manifest"] = {"path": mpath, "sha256": sha256_file(mpath), "id": m.manifest_id, "panel": list(roles(protocol_cfg)[PANEL])}
    summary["episodes"] = len(ARMS) * len(role[PANEL])
    summary["decision"] = {"repair_parent": ({"controller": "zero_curl", "source": os.path.join(out_dir, "zero_curl_source.py")}
                                             if summary["repair_parent_eligible"] else None),
                           "rule": "strict promotion on cases 10-39 (positive difference, lower bound above zero) and no observed drop increase"}
    summary["limitation"] = "30 pairs: the paired standard error at this base rate is about 0.10; no retry, no extension"
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


run = run_qualification

__all__ = ["ARMS", "GAIN_THRESHOLD", "PANEL", "STUDY", "aggregate", "build_manifests", "make_controller", "run", "run_qualification"]
