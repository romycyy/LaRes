"""R-137 (prospective follow-up to D-094): is the frozen JOINT controller's wide-reset weakness an initial-tilt effect?

Frozen JOINT, the PPO expert and D-072 on four fresh manifests under four predeclared reset conditions that scale the
placement vector's rotation draws and its position and joint draws separately: nominal (1, 1), tilt-only (1.5, 1),
position-and-joints-only (1, 1.5) and full (1.5, 1.5). No fitting. The one confirmatory contrast is JOINT under
tilt-only minus JOINT under position-and-joints-only scaling on identical cases (512 pairs); everything else is
descriptive. Cells are ``<controller>@<condition>`` through ``frozen_robustness``.

    python scripts/run_shadowhand_spin_stages.py --study tilt_reset_robustness
"""

from __future__ import annotations

import glob
import os

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin import frozen_robustness as fr
from lares.envs.isaac_pen_spin import recovery_module as rm
from lares.envs.isaac_pen_spin import recovery_tail_audit as ta
from lares.envs.isaac_pen_spin import reset_geometry_audit as rg
from lares.envs.isaac_pen_spin import symbolic_reward as sr
from lares.envs.isaac_pen_spin.protocols import freeze_json, load_cell, study_cell
from lares.eval.manifest import EvaluationManifest
from lares.utils.hashing import sha256_file

STUDY = "tilt_reset_robustness"
CONTROLLERS = ("joint", "ppo", "d072")
PRIMARY = (("joint", "tilt_only"), ("joint", "position_only"))


def conditions(cfg: dict) -> dict:
    return {str(k): dict(v or {}) for k, v in cfg["conditions"].items()}


def plan(cfg: dict) -> list[tuple]:
    return [(int(s), c, k) for s in cfg["panel_seeds"] for k in conditions(cfg) for c in CONTROLLERS]


def manifest_for(cfg: dict, seed) -> str:
    return str(cfg["manifest_template"]).format(seed=int(seed))


def build_manifests(protocol_cfg: dict) -> list[str]:
    from lares.eval.manifest import assert_disjoint

    mine = {manifest_for(protocol_cfg, s): cases.build_manifest(3, int(s), int(protocol_cfg["cases"]), label="tilt")
            for s in protocol_cfg["panel_seeds"]}
    paths = sorted(glob.glob("config/manifests/pen_spin_stage3_*.yaml") + glob.glob("config/manifests/shadowhand_spin_*.yaml"))
    assert_disjoint(list(mine.values()) + [EvaluationManifest.load(p) for p in paths if p not in mine])
    return [p for p, m in mine.items() if cases.freeze_manifest(p, m)]


def make_controller(name: str, protocol_cfg: dict, out_dir: str | None = None):
    controller, _, condition = name.partition("@")
    conds = conditions(protocol_cfg)
    if controller not in CONTROLLERS or condition not in conds:
        raise ValueError(f"unknown cell {name!r}")
    return fr.Conditioned(fr.base_controller(controller, protocol_cfg), name, conds[condition])


def delivery(cfg: dict) -> dict:
    """What each condition does to the frozen cases' initial geometry (reconstructed from the scaled placements)."""
    vecs = np.concatenate([cases.resolve_placements(EvaluationManifest.load(manifest_for(cfg, s))) for s in cfg["panel_seeds"]])
    out = {}
    for name, cond in conditions(cfg).items():
        g = rg.geometry(cases.scale_placements(vecs, cond), 1.0)
        out[name] = {"condition": cond, "mean_tilt_rad": float(g["tilt"].mean()), "mean_transverse": float(g["transverse"].mean()),
                     "mean_axial": float(g["axial"].mean()), "mean_abs_joint_draw": float(np.abs(cases.scale_placements(vecs, cond)[:, 5:]).mean())}
    return out


def analyse(rows: dict, cfg: dict) -> dict:
    """``rows[(seed, controller, condition)]`` -> the one confirmatory contrast and the descriptive table."""
    seeds, thr = [int(s) for s in cfg["panel_seeds"]], -float(cfg["threshold"])

    def contrast(a, b):
        return fr.paired({f"s{s}": (rows[(s, *a)], rows[(s, *b)]) for s in seeds}, 0.0, float(cfg["drop_margin"]))

    head = contrast(*PRIMARY)
    lo, hi = head["ci95"]
    head["verdict"] = "supported" if hi <= thr else ("refuted" if lo > thr else "inconclusive")
    head["threshold"], head["includes_zero"] = thr, bool(lo <= 0 <= hi)
    table = {c: {k: {"success": float(np.mean([r["sustained"] for s in seeds for r in rows[(s, c, k)]])),
                     "drops": int(sum(not r["completed"] for s in seeds for r in rows[(s, c, k)])),
                     "cases": int(sum(len(rows[(s, c, k)]) for s in seeds)),
                     "labels": {lab: int(sum(ta.label(r, fr.LABEL_CFG) == lab for s in seeds for r in rows[(s, c, k)])) for lab in ta.LABELS}}
                 for k in conditions(cfg)} for c in CONTROLLERS}
    descriptive = {f"{c}@{k}_minus_{c}@nominal": contrast((c, k), (c, "nominal")) for c in CONTROLLERS for k in conditions(cfg) if k != "nominal"}
    descriptive.update({f"{c}@tilt_only_minus_{c}@position_only": contrast((c, "tilt_only"), (c, "position_only")) for c in CONTROLLERS if c != "joint"})
    return {"paired_success_difference": head["difference"], "paired_success_difference_ci95": head["ci95"], "primary": head,
            "table": table, "descriptive": descriptive}


def run(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    cfg = protocol_cfg
    os.makedirs(out_dir, exist_ok=True)
    descriptor = sr.check_reference(cfg)
    cfg_sha = sr.config_sha256(cfg)
    mpaths = {int(s): cases.require_frozen(manifest_for(cfg, s)) for s in cfg["panel_seeds"]}
    freeze_json(os.path.join(out_dir, "frozen_config.json"),
                {"config_sha256": cfg_sha, "config": sr.frozen_config(cfg), "reference": descriptor, "assets": fr.asset_record(cfg),
                 "manifests": {str(s): {"path": p, "sha256": sha256_file(p)} for s, p in mpaths.items()}, "delivery": delivery(cfg),
                 "primary": {"a": "joint@tilt_only", "b": "joint@position_only", "threshold": -float(cfg["threshold"]),
                             "rule": "supported: the paired 95% interval wholly at or below the threshold; refuted: wholly above it"}},
                "frozen configuration")
    rows = {}
    for seed, controller, condition in plan(cfg):
        sub = os.path.join(out_dir, "cells", f"s{seed}")
        os.makedirs(sub, exist_ok=True)
        cell = os.path.join(sub, f"{controller}@{condition}_cell.json")
        study_cell(cfg, config_path, mpaths[seed], f"{controller}@{condition}", cell, record_obs=False)
        if rm._load(cell)["condition"] != conditions(cfg)[condition]:
            raise SystemExit(f"{cell}: the cell did not run under its condition")
        rows[(seed, controller, condition)] = load_cell(cell)[1]
    summary = {"study": STUDY, "config_sha256": cfg_sha, "delivery": delivery(cfg), **analyse(rows, cfg),
               "accounting": {"cells": len(rows), "episodes": int(sum(len(v) for v in rows.values())), "llm_calls": 0, "training": "none"},
               "limitation": "one frozen controller lineage, one scale (1.5), 512 pairs; position and joint draws are scaled together"}
    rm._dump(os.path.join(out_dir, "summary.json"), summary)
    return summary


__all__ = ["CONTROLLERS", "PRIMARY", "STUDY", "analyse", "build_manifests", "conditions", "delivery", "make_controller", "plan", "run"]
