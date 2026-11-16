"""R-073: a frozen constant grasp preload in place of the reactive capture curl.

D-052 confirmed that zeroing the capture curl (``extra_curl = capture_curl * ((1 - contact) + 0.75 (1 - retained))``)
doubles fresh success and cuts drops, leaving slow spin as the failure mode. This study asks whether the *static* part
of that curl (a constant preload) keeps the traction without the reactive tightening that loses the pen. The constant
is the median of the parent's own ``extra_curl`` over the recorded in-play steps of its archived R-068 runs, frozen to
disk with its provenance before any episode; no search. Three arms (parent, exact zero-curl, constant preload) on one
frozen 200-case manifest (promotion 10-39, fresh 40-199), one process per arm per panel.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin import channel_lesions as cl
from lares.envs.isaac_pen_spin import scalar_tuning as st
from lares.envs.isaac_pen_spin.learner_recovery import learner_source, make_learner
from lares.envs.isaac_pen_spin.protocols import (actions_match, arm_summary, build_single_manifest, cached_json,
                                                 failure_split, in_play, learner_drive_params, load_cell, manifest_path,
                                                 roles, study_cell)

STUDY = "r073_constant_preload"
STAGE = 3
ARMS = ("parent", "zero_curl", "constant_preload")
PANELS = ("promotion", "fresh")
GAIN_THRESHOLD = 0.10
POSITIVE = 1e-6
_CURL = re.compile(r"extra_curl = self\.capture_curl \* \(\n")


def substituted_source(source: str, constant: float, factor: float) -> str:
    """``extra_curl = C + f * capture_curl * (...)``; ``C = 0, f = 1`` is the identity, ``C = 0, f = 0`` the zero clamp."""
    if len(_CURL.findall(source)) != 1:
        raise SystemExit("the parent source does not contain exactly one capture-curl site")
    return _CURL.sub(f"extra_curl = {float(constant)!r} + {float(factor):.1f} * self.capture_curl * (\n", source)


def preload_learner(protocol_cfg: dict, constant: float, factor: float, name: str):
    ctrl = make_learner(protocol_cfg, lambda source: substituted_source(source, constant, factor))
    ctrl.name = name
    ctrl.params = {**ctrl.params, "arm": name, "preload": float(constant), "curl_factor": float(factor)}
    return ctrl


def parent_extra_curl(protocol_cfg: dict, archives: list[str]) -> np.ndarray:
    """The parent's extra_curl on every recorded in-play step of the archived runs, from its own gates."""
    import torch

    learner = make_learner(protocol_cfg)
    pol = learner.policy
    cc = float(pol.capture_curl.item())
    values = []
    for path in archives:
        data = np.load(path)
        obs, length = data["obs"], data["length"]
        for i, L in in_play(obs, length):
            with torch.no_grad():
                pol(torch.as_tensor(obs[i, :L], dtype=torch.float32))
            contact = pol.last_gates["contact"].reshape(-1).numpy()
            retained = pol.last_gates["retained"].reshape(-1).numpy()
            values.append(cc * ((1.0 - contact) + 0.75 * (1.0 - retained)))
    return np.concatenate(values) if values else np.zeros(0)


def freeze_preload(protocol_cfg: dict, out_dir: str) -> dict:
    """The constant and its provenance, written before any episode; an existing file must agree."""
    path = os.path.join(out_dir, "preload.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    from lares.utils.hashing import sha256_file

    archives = list(protocol_cfg["preload_archives"])
    for a in archives:
        if not os.path.exists(a):
            raise SystemExit(f"archived parent run {a} missing; the preload cannot be defined")
    ec = parent_extra_curl(protocol_cfg, archives)
    positive = ec[ec > POSITIVE]
    if positive.size == 0:
        raise SystemExit("no in-play step has a positive parent extra_curl; the study stops before any episode")
    out = {"constant": float(np.median(positive)), "rule": f"median of parent extra_curl over archived in-play steps with extra_curl > {POSITIVE}",
           "steps": int(ec.size), "positive_steps": int(positive.size), "mean": float(ec.mean()),
           "quartiles": np.percentile(positive, [25, 50, 75]).tolist(), "max": float(ec.max()),
           "capture_curl": float(make_learner(protocol_cfg).policy.capture_curl.item()),
           "archives": {a: sha256_file(a) for a in archives}}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def build_manifests(protocol_cfg: dict) -> list[str]:
    return build_single_manifest(protocol_cfg, "r073_block", STAGE)


def make_controller(arm: str, protocol_cfg: dict, out_dir: str):
    if arm == "parent":
        return make_learner(protocol_cfg)
    if arm == "zero_curl":
        return cl.lesioned_learner(protocol_cfg, "curl_clamp", 0.0, arm)
    if arm == "constant_preload":
        with open(os.path.join(out_dir, "preload.json"), encoding="utf-8") as f:
            c = json.load(f)["constant"]
        return preload_learner(protocol_cfg, c, 0.0, arm)
    raise ValueError(f"unknown arm {arm!r}; one of {ARMS}")


def offline_identity_gate(protocol_cfg: dict, archive_npz: str) -> dict:
    data = np.load(archive_npz)
    parent = make_learner(protocol_cfg)
    ident = preload_learner(protocol_cfg, 0.0, 1.0, "identity")
    return {"archive": archive_npz, **actions_match(parent, ident, data["obs"], data["length"])}


def _panel(rows_a, rows_b, sat_a, sat_b) -> dict:
    c = st.contrast(rows_a, rows_b)
    return {**c, "drops_not_increased": bool(c["drop_counts"][0] <= c["drop_counts"][1]),
            "drop_increase_established": bool(c["drops"]["low"] is not None and c["drops"]["low"] > 0.0)}


def aggregate(rows: dict, sats: dict) -> dict:
    out = {"arms": {a: {p: {**arm_summary(rows[a][p]), **failure_split(rows[a][p]), **cl.tail_failure_split(rows[a][p]),
                            "saturation": {k: v for k, v in sats[a][p].items() if k != "drive_unsaturated_fraction_per_case"}}
                        for p in PANELS} for a in ARMS},
           "contrasts": {p: {"constant_preload_minus_zero_curl": _panel(rows["constant_preload"][p], rows["zero_curl"][p], sats["constant_preload"][p], sats["zero_curl"][p]),
                             "constant_preload_minus_parent": _panel(rows["constant_preload"][p], rows["parent"][p], sats["constant_preload"][p], sats["parent"][p]),
                             "zero_curl_minus_parent": _panel(rows["zero_curl"][p], rows["parent"][p], sats["zero_curl"][p], sats["parent"][p])}
                         for p in PANELS}}
    t = out["contrasts"]["fresh"]["constant_preload_minus_zero_curl"]["success"]
    out["paired_success_difference"] = t["mean"]
    out["prediction"] = {"threshold": GAIN_THRESHOLD,
                         "supported": bool(t["mean"] is not None and t["mean"] >= GAIN_THRESHOLD and t["low"] is not None and t["low"] > 0.0),
                         "refuted": bool(t["high"] is not None and t["high"] < GAIN_THRESHOLD)}
    q = True
    for ctl in ("zero_curl", "parent"):
        pro = out["contrasts"]["promotion"][f"constant_preload_minus_{ctl}"]
        fr = out["contrasts"]["fresh"][f"constant_preload_minus_{ctl}"]
        q = q and pro["promotes"] and fr["success"]["low"] is not None and fr["success"]["low"] > 0.0 \
            and pro["drops_not_increased"] and fr["drops_not_increased"]
    out["qualifies_repair_parent"] = bool(q)
    return out


def run_constant_preload(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    source = learner_source(protocol_cfg)
    os.makedirs(out_dir, exist_ok=True)
    audit = cl.source_audit(source)
    if not audit["passes"]:
        raise SystemExit(f"source audit failed, no episode run: {audit}")
    preload = freeze_preload(protocol_cfg, out_dir)
    with open(os.path.join(out_dir, "constant_preload_source.py"), "w", encoding="utf-8") as f:
        f.write(substituted_source(source, preload["constant"], 0.0))
    gate = cached_json(os.path.join(out_dir, "identity_gate.json"),
                       lambda: offline_identity_gate(protocol_cfg, protocol_cfg["identity_archive"]))
    if not gate["passes"]:
        raise SystemExit(f"offline identity gate failed, no episode run: {gate}")
    mpath = manifest_path(protocol_cfg)
    m = cases.load_frozen(mpath)
    role = cases.role_manifests(m, roles(protocol_cfg), expected_cases=int(protocol_cfg["cases"]), stage=STAGE)
    paths = cases.freeze_roles(role, out_dir)
    params = learner_drive_params(protocol_cfg)
    rows, sats = {a: {} for a in ARMS}, {a: {} for a in ARMS}
    for p in PANELS:
        for a in ARMS:
            cell = os.path.join(out_dir, f"{a}_{p}.json")
            study_cell(protocol_cfg, config_path, paths[p], a, cell)
            trace, rows[a][p] = load_cell(cell)
            sats[a][p] = st.saturation(trace, params)
    summary = aggregate(rows, sats)
    summary["preload"] = preload
    summary["identity_gate"] = gate
    summary["source_audit"] = audit
    summary["manifest"] = cases.manifest_record(mpath, m)
    summary["episodes"] = len(ARMS) * sum(len(role[p]) for p in PANELS)
    summary["decision"] = {"repair_parent": ({"controller": "constant_preload", "constant": preload["constant"],
                                              "source": os.path.join(out_dir, "constant_preload_source.py")}
                                             if summary["qualifies_repair_parent"] else None),
                           "rule": "strict promotion against both controls, fresh lower bound above zero against both, "
                                   "no observed drop increase against either on either panel"}
    summary["limitation"] = "one block, one learner, one frozen constant; no preload level other than the median was tried"
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


run = run_constant_preload

__all__ = ["ARMS", "GAIN_THRESHOLD", "PANELS", "STUDY", "aggregate", "build_manifests", "freeze_preload",
           "make_controller", "offline_identity_gate", "parent_extra_curl", "preload_learner", "run",
           "run_constant_preload", "substituted_source"]
