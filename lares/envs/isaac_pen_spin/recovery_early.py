"""R-122: learned start-up offsets for the capture transient versus matched continuation of the JOINT finalist.

Both arms start from the frozen R-121 JOINT finalist (43 coordinates) and get 24 ARS updates on the same placements,
sharing direction draws on the common coordinates:

* ``continue`` -- the 43 coordinates;
* ``early``    -- the same plus 20 start-up coordinates: a per-joint offset, inside the residual clamp, multiplied by
  ``max(0, 1 - t/60)``; zero initially, so the initial policy is the parent exactly.

The first 60 steps are where early drops happen and where the slow mode is decided (D-078), and nothing so far has
acted there specifically. The controller and the Isaac child are ``recovery_module``'s.
"""

from __future__ import annotations

import os

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin import recovery_authority as ra
from lares.envs.isaac_pen_spin import recovery_module as rm
from lares.envs.isaac_pen_spin import recovery_tail_audit as ta
from lares.envs.isaac_pen_spin import symbolic_reward as sr
from lares.envs.isaac_pen_spin.protocols import freeze_json, load_cell, study_cell

STUDY = "joint_early_phase"
ARMS = ("continue", "early")
POLICIES = ("reference", "joint", *ARMS)
FAMILY = (("early", "continue"), *((a, "joint") for a in ARMS), *((a, "reference") for a in ARMS))
EXTRA = (("joint", "reference"),)
LABEL_CFG = {"early_drop_step": 60, "slow_hit_max": 0.2, "near_hit_min": 0.8}


def parent_x(cfg: dict) -> np.ndarray:
    entry = rm._load(cfg["module_finalist"])
    if entry["sha256"] != cfg["module_sha256"] or rm.x_record(entry["x"], *rm.bounds(cfg))["sha256"] != entry["sha256"]:
        raise SystemExit("the parent is not the frozen R-121 JOINT finalist")
    return np.asarray(entry["x"], dtype=float)


def initial_x(cfg: dict, arm: str) -> np.ndarray:
    x = parent_x(cfg)
    return x if arm == "continue" else np.concatenate([x, np.full(sr.N_JOINTS, 0.5)])


def coord_scale(cfg: dict, arm: str) -> np.ndarray:
    low, high = rm.bounds(cfg)
    s = np.concatenate([((rm.HIGH - rm.LOW) / (high - low))[:rm.DIM], np.ones(rm.N_RESIDUAL)])
    return s if arm == "continue" else np.concatenate([s, np.ones(sr.N_JOINTS)])


def confirmation_manifest(cfg: dict, seed) -> str:
    return str(cfg["confirmation_manifest_template"]).format(seed=int(seed))


def build_manifests(protocol_cfg: dict) -> list[str]:
    import glob

    from lares.eval.manifest import EvaluationManifest, assert_disjoint

    mine = {confirmation_manifest(protocol_cfg, s): cases.build_manifest(3, int(s), int(protocol_cfg["confirmation_cases"]), label="early_confirm")
            for s in protocol_cfg["confirmation_seeds"]}
    others = [EvaluationManifest.load(p) for p in sorted(glob.glob("config/manifests/pen_spin_stage3_*.yaml")) if p not in mine]
    assert_disjoint(list(mine.values()) + others)
    return [p for p, m in mine.items() if cases.freeze_manifest(p, m)]


def make_controller(name: str, protocol_cfg: dict, out_dir: str | None = None):
    if name == "reference":
        return sr.build_reference(protocol_cfg, "reference", resettable=False)
    if name == "joint":
        return rm.build(protocol_cfg, "joint", parent_x(protocol_cfg))
    if name in ARMS:
        entry = rm._load(os.path.join(out_dir or str(protocol_cfg["log_dir"]), "finalists.json"))["finalists"][name]
        if rm.x_record(entry["x"], *rm.bounds(protocol_cfg))["sha256"] != entry["sha256"]:
            raise SystemExit("a stored finalist does not match its hash")
        return rm.build(protocol_cfg, name, entry["x"])
    raise ValueError(f"unknown controller {name!r}")


def contrasts(rows: dict, threshold: float, drop_margin: float) -> dict:
    """The five-contrast Holm family and each finalist's qualification against both JOINT and D-072."""
    from lares.envs.isaac_pen_spin.incumbent_residual import confirmation_contrast

    out = {f"{a}_minus_{b}": confirmation_contrast({m: (r[a], r[b]) for m, r in rows.items()}, threshold, drop_margin)
           for a, b in (*FAMILY, *EXTRA)}
    names = [f"{a}_minus_{b}" for a, b in FAMILY]
    h = sr.holm({k: out[k]["pooled"]["success"]["p_value"] for k in names})
    for k in names:
        out[k]["holm"] = h[k]
    qual = {}
    for arm in ARMS:
        ok = [bool(out[f"{arm}_minus_{base}"]["holm"]["rejects"] and (out[f"{arm}_minus_{base}"]["pooled"]["success"]["low"] or 0) > 0
                   and out[f"{arm}_minus_{base}"]["pooled"]["drops"]["non_inferior"]) for base in ("joint", "reference")]
        qual[arm] = {"against_parent": ok[0], "against_reference": ok[1], "qualifies": all(ok)}
    return {"contrasts": out, "qualification": qual}


def label_table(rows: dict) -> dict:
    return {p: {name: int(sum(ta.label(r, LABEL_CFG) == name for m in rows.values() for r in m[p])) for name in ta.LABELS} for p in POLICIES}


def run(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    if os.path.normpath(out_dir) != os.path.normpath(str(protocol_cfg["log_dir"])):
        raise SystemExit("evaluation cells read the finalists from the protocol's own log_dir; do not override it")
    os.makedirs(out_dir, exist_ok=True)
    descriptor = sr.check_reference(protocol_cfg)
    ra.normalization(protocol_cfg)
    cfg_sha = sr.config_sha256(protocol_cfg)
    b = rm.bounds(protocol_cfg)
    init = {arm: rm.x_record(initial_x(protocol_cfg, arm), *b) for arm in ARMS}
    scale = {arm: [float(v) for v in coord_scale(protocol_cfg, arm)] for arm in ARMS}
    freeze_json(os.path.join(out_dir, "frozen_config.json"),
                {"config_sha256": cfg_sha, "config": sr.frozen_config(protocol_cfg), "reference": descriptor, "initial": init,
                 "coord_scale": scale}, "frozen configuration")
    seeds = [int(s) for s in protocol_cfg["confirmation_seeds"]]
    mpaths = {s: cases.require_frozen(confirmation_manifest(protocol_cfg, s)) for s in seeds}
    ceiling = int(protocol_cfg["start_ceiling"])
    reserve = 2 * len(POLICIES) * len(seeds) * int(protocol_cfg["confirmation_cases"])
    base = {"config_path": os.path.abspath(config_path), "study": STUDY, "config_sha256": cfg_sha}

    smoke = rm.run_child({**base, "mode": "smoke_early", "response_path": os.path.join(out_dir, "smoke.json")}, os.path.join(out_dir, "smoke"))
    summary = {"study": STUDY, "config_sha256": cfg_sha, "smoke": smoke, "smoke_gate": "pass" if smoke["passes"] else "fail"}
    if not smoke["passes"]:
        summary["note"] = "the smoke gate failed: no training or confirmation ran"
        rm._dump(os.path.join(out_dir, "summary.json"), summary)
        return summary
    started, training, finalists = int(smoke["starts"]), {}, {}
    for arm in ARMS:
        tr = rm.run_child({**base, "mode": "train", "init_x": init[arm]["x"], "direction_dim": rm.DIM_EARLY, "coord_scale": scale[arm],
                           "start_cap": ceiling - reserve - started, "response_path": os.path.join(out_dir, f"train_{arm}.json")},
                          os.path.join(out_dir, f"train_{arm}"))
        started += int(tr["starts"])
        training[arm] = tr
        if not tr["complete"]:
            summary["incomplete"] = f"the start ceiling was reached in the {arm} arm: no confirmation"
            rm._dump(os.path.join(out_dir, "summary.json"), summary)
            return summary
        finalists[arm] = rm.x_record(tr["final_x"], *b)
    freeze_json(os.path.join(out_dir, "finalists.json"), {"rule": "each arm's final ARS mean; no selection", "finalists": finalists}, "finalists")

    rows = {}
    for s in seeds:
        sub = os.path.join(out_dir, "confirmation", f"s{s}")
        os.makedirs(sub, exist_ok=True)
        rows[f"s{s}"] = {}
        for name in POLICIES:
            cell = os.path.join(sub, f"{name}_cell.json")
            study_cell(protocol_cfg, config_path, mpaths[s], name, cell, record_obs=False)
            if name in ARMS and rm._load(cell)["controller_params"]["module"]["sha256"] != finalists[name]["sha256"]:
                raise SystemExit(f"{cell}: the evaluated policy is not the frozen finalist")
            rows[f"s{s}"][name] = load_cell(cell)[1]
    c = contrasts(rows, float(protocol_cfg["threshold"]), float(protocol_cfg["drop_margin"]))
    head = c["contrasts"]["early_minus_continue"]["pooled"]["success"]
    offsets = rm.RESIDUAL_SPAN * (np.asarray(finalists["early"]["x"])[rm.DIM_JOINT:] - 0.5)
    summary.update({
        "paired_success_difference": head["mean"], "paired_success_difference_ci95": [head["low"], head["high"]],
        **c, "labels": label_table(rows),
        "successes": {n: int(sum(r["sustained"] for m in rows.values() for r in m[n])) for n in POLICIES},
        "drops": {n: int(sum(not r["completed"] for m in rows.values() for r in m[n])) for n in POLICIES},
        "finalists": finalists, "initial": init,
        "startup_offsets": dict(zip(sr.JOINTS, [float(v) for v in offsets])),
        "training": training,
        "accounting": {"smoke_starts": smoke["starts"], "training_starts": {a: t["starts"] for a, t in training.items()},
                       "confirmation_charged_starts": reserve, "charged_total": started + reserve, "ceiling": ceiling, "llm_calls": 0},
        "limitation": "one matched pair of ARS trajectories from one parent; a single confirmation",
    })
    rm._dump(os.path.join(out_dir, "summary.json"), summary)
    return summary


__all__ = ["ARMS", "FAMILY", "POLICIES", "STUDY", "build_manifests", "contrasts", "coord_scale", "initial_x", "label_table",
           "make_controller", "parent_x", "run"]
