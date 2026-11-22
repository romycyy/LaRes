"""R-124: one consolidated confirmation of the frozen construction ladder. No training, no LLM.

Six frozen, hashed controllers on the same 24 fresh manifests, each once per manifest in its own fresh process:

    d058   the saved default incumbent (fixed-stroke controller)
    d072   d058 + D-070's joint-motion residual (the qualified reference of this direction)
    r106   d072 + the 32-coefficient residual trained by ARS on shared placements
    r116   r106 + the learned single-decision re-capture module
    r118   r116's module after 24 more updates (CONTINUE)
    r121   module and residual co-adapted (JOINT)

Every earlier step was confirmed only against its own parent on its own manifests; this measures them together.
"""

from __future__ import annotations

import glob
import os

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin import recovery_authority as ra
from lares.envs.isaac_pen_spin import recovery_module as rm
from lares.envs.isaac_pen_spin import recovery_tail_audit as ta
from lares.envs.isaac_pen_spin import symbolic_reward as sr
from lares.envs.isaac_pen_spin.protocols import freeze_json, load_cell, study_cell
from lares.utils.hashing import sha256_file

STUDY = "stage3_consolidated_ladder"
POLICIES = ("d058", "d072", "r106", "r116", "r118", "r121")
#: one Holm family: the primary, the five adjacent steps, and the top against the saved default
FAMILY = (("r121", "d072"), ("d072", "d058"), ("r106", "d072"), ("r116", "r106"), ("r118", "r116"), ("r121", "r118"), ("r121", "d058"))
PRIMARY = "r121_minus_d072"
LABEL_CFG = {"early_drop_step": 60, "slow_hit_max": 0.2, "near_hit_min": 0.8}


def module_x(cfg: dict, name: str) -> np.ndarray:
    """A frozen module asset's coordinates, verified against the hash the config names."""
    entry = rm._load(cfg["assets"][name]["path"])
    if entry["sha256"] != cfg["assets"][name]["sha256"]:
        raise SystemExit(f"{name}: the asset is not the frozen one")
    return np.asarray(entry["x"], dtype=float)


def asset_record(cfg: dict) -> dict:
    """File hashes of every asset the ladder evaluates (the incumbent directory through its own check)."""
    from lares.envs.isaac_pen_spin.coordinate_repair import check_asset

    out = {"d058": {"asset": cfg["incumbent_asset"], "files": dict(check_asset(cfg["incumbent_asset"])["files"])},
           "d072": {"descriptor": cfg["reference_descriptor"], "sha256": sha256_file(cfg["reference_descriptor"])},
           "r106": {"path": cfg["root_finalist"], "sha256": sha256_file(cfg["root_finalist"]), "theta_sha256": cfg["root_theta_sha256"]}}
    for name in ("r116", "r118", "r121"):
        module_x(cfg, name)
        out[name] = {"path": cfg["assets"][name]["path"], "file_sha256": sha256_file(cfg["assets"][name]["path"]),
                     "module_sha256": cfg["assets"][name]["sha256"]}
    return out


def manifest_for(cfg: dict, seed) -> str:
    return str(cfg["confirmation_manifest_template"]).format(seed=int(seed))


def build_manifests(protocol_cfg: dict) -> list[str]:
    from lares.eval.manifest import EvaluationManifest, assert_disjoint

    mine = {manifest_for(protocol_cfg, s): cases.build_manifest(3, int(s), int(protocol_cfg["confirmation_cases"]), label="ladder_confirm")
            for s in protocol_cfg["confirmation_seeds"]}
    others = [EvaluationManifest.load(p) for p in sorted(glob.glob("config/manifests/pen_spin_stage3_*.yaml")) if p not in mine]
    assert_disjoint(list(mine.values()) + others)
    return [p for p, m in mine.items() if cases.freeze_manifest(p, m)]


def make_controller(name: str, protocol_cfg: dict, out_dir: str | None = None):
    if name == "d058":
        from lares.envs.isaac_pen_spin.coordinate_repair import incumbent_controller

        return incumbent_controller(protocol_cfg["incumbent_asset"])
    if name == "d072":
        return sr.build_reference(protocol_cfg, "d072", resettable=False)
    if name == "r106":
        return sr.build_controller(protocol_cfg, "r106", ra.finalist_theta(protocol_cfg), ra.normalization(protocol_cfg))
    if name in ("r116", "r118"):
        return rm.build(protocol_cfg, name, module_x(protocol_cfg, name), original_bounds=True)
    if name == "r121":
        return rm.build(protocol_cfg, name, module_x(protocol_cfg, name))
    raise ValueError(f"unknown controller {name!r}")


def contrasts(rows: dict, cfg: dict) -> dict:
    """The seven-contrast Holm family: pooled case-paired and manifest-clustered summaries, drop bounds."""
    from lares.envs.isaac_pen_spin.incumbent_residual import confirmation_contrast

    out = {}
    for a, b in FAMILY:
        thr = float(cfg["threshold"]) if f"{a}_minus_{b}" == PRIMARY else float(cfg["step_threshold"])
        out[f"{a}_minus_{b}"] = confirmation_contrast({m: (r[a], r[b]) for m, r in rows.items()}, thr, float(cfg["drop_margin"]))
    h = sr.holm({k: v["pooled"]["success"]["p_value"] for k, v in out.items()})
    for k, v in out.items():
        v["holm"] = h[k]
        v["qualifies"] = bool(h[k]["rejects"] and (v["pooled"]["success"]["low"] or 0) > 0 and v["pooled"]["drops"]["non_inferior"])
    return out


def label_table(rows: dict) -> dict:
    return {p: {name: int(sum(ta.label(r, LABEL_CFG) == name for m in rows.values() for r in m[p])) for name in ta.LABELS} for p in POLICIES}


def run(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    from lares.eval.stats import t_summary

    os.makedirs(out_dir, exist_ok=True)
    descriptor = sr.check_reference(protocol_cfg)
    ra.finalist_theta(protocol_cfg)
    ra.normalization(protocol_cfg)
    cfg_sha = sr.config_sha256(protocol_cfg)
    freeze_json(os.path.join(out_dir, "frozen_config.json"),
                {"config_sha256": cfg_sha, "config": sr.frozen_config(protocol_cfg), "reference": descriptor,
                 "assets": asset_record(protocol_cfg)}, "frozen configuration")
    seeds = [int(s) for s in protocol_cfg["confirmation_seeds"]]
    rows = {}
    for s in seeds:
        mpath = cases.require_frozen(manifest_for(protocol_cfg, s))
        sub = os.path.join(out_dir, "confirmation", f"s{s}")
        os.makedirs(sub, exist_ok=True)
        rows[f"s{s}"] = {}
        for name in POLICIES:
            cell = os.path.join(sub, f"{name}_cell.json")
            study_cell(protocol_cfg, config_path, mpath, name, cell, record_obs=False)
            rows[f"s{s}"][name] = load_cell(cell)[1]
    c = contrasts(rows, protocol_cfg)
    head = c[PRIMARY]["pooled"]["success"]
    n = sum(len(m["d058"]) for m in rows.values())
    rate = {p: {"successes": int(sum(r["sustained"] for m in rows.values() for r in m[p])),
                "drops": int(sum(not r["completed"] for m in rows.values() for r in m[p]))} for p in POLICIES}
    for p in POLICIES:
        per = [float(np.mean([r["sustained"] for r in m[p]])) for m in rows.values()]
        ts = t_summary(per)
        rate[p].update({"success": rate[p]["successes"] / n, "manifest_ci95": [ts["low"], ts["high"]],
                        "manifest_min_max": [float(min(per)), float(max(per))]})
    summary = {"study": STUDY, "config_sha256": cfg_sha, "cases_per_policy": n, "manifests": len(seeds),
               "paired_success_difference": head["mean"], "paired_success_difference_ci95": [head["low"], head["high"]],
               "policies": rate, "contrasts": c, "labels": label_table(rows),
               "accounting": {"cells": len(POLICIES) * len(seeds), "episodes": len(POLICIES) * n, "llm_calls": 0, "training": "none"},
               "limitation": "one evaluation of six frozen controllers; no training and no selection; one fixed set of 24 manifests"}
    rm._dump(os.path.join(out_dir, "summary.json"), summary)
    return summary


__all__ = ["FAMILY", "POLICIES", "PRIMARY", "STUDY", "asset_record", "build_manifests", "contrasts", "label_table", "make_controller",
           "module_x", "run"]
