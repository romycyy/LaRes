"""R-121: does co-adapting the residual with the re-capture module add value beyond continued module fitting?

Three arms from the frozen CONTINUE finalist of R-118 (the R-106 residual plus the single-decision re-capture
module), 24 ARS updates each on the same placements, sharing direction draws on their common coordinates:

* ``module_original`` -- the 11 module parameters in their R-116 ranges (plain continuation);
* ``module_wide``     -- the same with the two saturated ranges widened (threshold up to 6 rad/s, ramp-out down to 3 steps);
* ``joint``           -- ``module_wide`` plus the residual's 32 coefficients (43 coordinates).

A widened coordinate's perturbation and update are scaled back to their original physical size, and residual
coordinates use R-106's step sizes, so the arms differ in what they may change, not in how hard they push.
"""

from __future__ import annotations

import os

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin import recovery_authority as ra
from lares.envs.isaac_pen_spin import recovery_module as rm
from lares.envs.isaac_pen_spin import symbolic_reward as sr
from lares.envs.isaac_pen_spin.protocols import freeze_json, load_cell, study_cell

STUDY = "recovery_joint_matched"
ARMS = ("module_original", "module_wide", "joint")
POLICIES = ("reference", "r106", "continue", *ARMS)
#: one Holm family: the primary, the boundary contrast, each finalist against the parent and against D-072
FAMILY = (("joint", "module_wide"), ("module_wide", "module_original"), *((a, "continue") for a in ARMS), *((a, "reference") for a in ARMS))
EXTRA = (("continue", "r106"), ("continue", "reference"))


def parent_x(cfg: dict) -> np.ndarray:
    entry = rm._load(cfg["module_finalist"])
    if entry["sha256"] != cfg["module_sha256"] or rm.x_record(entry["x"])["sha256"] != entry["sha256"]:
        raise SystemExit("the parent is not the frozen CONTINUE finalist")
    return np.asarray(entry["x"], dtype=float)


def initial_x(cfg: dict, arm: str) -> np.ndarray:
    """The parent in the arm's coordinates: unchanged, re-expressed in the widened ranges, or that plus the residual."""
    x0 = parent_x(cfg)
    if arm == "module_original":
        return x0
    wide = rm.to_unit(rm.to_params(x0), *rm.bounds(cfg))
    return wide if arm == "module_wide" else np.concatenate([wide, rm.residual_origin(ra.finalist_theta(cfg))])


def coord_scale(cfg: dict, arm: str) -> np.ndarray:
    """Original range over the arm's range per module coordinate (1 where unchanged), 1 for residual coordinates."""
    if arm == "module_original":
        return np.ones(rm.DIM)
    low, high = rm.bounds(cfg)
    s = ((rm.HIGH - rm.LOW) / (high - low))[:rm.DIM]
    return s if arm == "module_wide" else np.concatenate([s, np.ones(rm.N_RESIDUAL)])


def confirmation_manifest(cfg: dict, seed) -> str:
    return str(cfg["confirmation_manifest_template"]).format(seed=int(seed))


def build_manifests(protocol_cfg: dict) -> list[str]:
    import glob

    from lares.eval.manifest import EvaluationManifest, assert_disjoint

    mine = {confirmation_manifest(protocol_cfg, s): cases.build_manifest(3, int(s), int(protocol_cfg["confirmation_cases"]), label="joint_confirm")
            for s in protocol_cfg["confirmation_seeds"]}
    others = [EvaluationManifest.load(p) for p in sorted(glob.glob("config/manifests/pen_spin_stage3_*.yaml")) if p not in mine]
    assert_disjoint(list(mine.values()) + others)
    return [p for p, m in mine.items() if cases.freeze_manifest(p, m)]


def make_controller(name: str, protocol_cfg: dict, out_dir: str | None = None):
    if name == "reference":
        return sr.build_reference(protocol_cfg, "reference", resettable=False)
    if name == "r106":
        return sr.build_controller(protocol_cfg, "r106", ra.finalist_theta(protocol_cfg), ra.normalization(protocol_cfg))
    if name == "continue":
        return rm.build(protocol_cfg, "continue", parent_x(protocol_cfg), original_bounds=True)
    if name in ARMS:
        entry = rm._load(os.path.join(out_dir or str(protocol_cfg["log_dir"]), "finalists.json"))["finalists"][name]
        orig = name == "module_original"
        if rm.x_record(entry["x"], *rm.bounds(None if orig else protocol_cfg))["sha256"] != entry["sha256"]:
            raise SystemExit("a stored finalist does not match its hash")
        return rm.build(protocol_cfg, name, entry["x"], original_bounds=orig)
    raise ValueError(f"unknown controller {name!r}")


def contrasts(rows: dict, threshold: float, drop_margin: float) -> dict:
    """The eight-contrast Holm family and each finalist's qualification: Holm p < 0.05 with a strictly positive paired
    lower bound against BOTH the parent and D-072, and drop bounds within the margin against both."""
    from lares.envs.isaac_pen_spin.incumbent_residual import confirmation_contrast

    out = {f"{a}_minus_{b}": confirmation_contrast({m: (r[a], r[b]) for m, r in rows.items()}, threshold, drop_margin)
           for a, b in (*FAMILY, *EXTRA)}
    names = [f"{a}_minus_{b}" for a, b in FAMILY]
    h = sr.holm({k: out[k]["pooled"]["success"]["p_value"] for k in names})
    for k in names:
        out[k]["holm"] = h[k]
    qual = {}
    for arm in ARMS:
        ok = []
        for base in ("continue", "reference"):
            c = out[f"{arm}_minus_{base}"]
            ok.append(bool(c["holm"]["rejects"] and c["pooled"]["success"]["low"] is not None and c["pooled"]["success"]["low"] > 0
                           and c["pooled"]["drops"]["non_inferior"]))
        qual[arm] = {"against_parent": ok[0], "against_reference": ok[1], "qualifies": all(ok)}
    return {"contrasts": out, "qualification": qual}


def run(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    if os.path.normpath(out_dir) != os.path.normpath(str(protocol_cfg["log_dir"])):
        raise SystemExit("evaluation cells read the finalists from the protocol's own log_dir; do not override it")
    os.makedirs(out_dir, exist_ok=True)
    descriptor = sr.check_reference(protocol_cfg)
    ra.normalization(protocol_cfg)
    cfg_sha = sr.config_sha256(protocol_cfg)
    wide_bounds = rm.bounds(protocol_cfg)
    init = {arm: rm.x_record(initial_x(protocol_cfg, arm), *rm.bounds(None if arm == "module_original" else protocol_cfg)) for arm in ARMS}
    scale = {arm: [float(v) for v in coord_scale(protocol_cfg, arm)] for arm in ARMS}
    freeze_json(os.path.join(out_dir, "frozen_config.json"),
                {"config_sha256": cfg_sha, "config": sr.frozen_config(protocol_cfg), "reference": descriptor, "initial": init,
                 "coord_scale": scale, "wide_bounds": [[float(v) for v in b[:rm.DIM]] for b in wide_bounds]}, "frozen configuration")
    seeds = [int(s) for s in protocol_cfg["confirmation_seeds"]]
    mpaths = {s: cases.require_frozen(confirmation_manifest(protocol_cfg, s)) for s in seeds}
    ceiling = int(protocol_cfg["start_ceiling"])
    reserve = 2 * len(POLICIES) * len(seeds) * int(protocol_cfg["confirmation_cases"])
    base = {"config_path": os.path.abspath(config_path), "study": STUDY, "config_sha256": cfg_sha}

    smoke = rm.run_child({**base, "mode": "smoke_joint", "init_wide": init["module_wide"]["x"], "init_joint": init["joint"]["x"],
                          "response_path": os.path.join(out_dir, "smoke.json")}, os.path.join(out_dir, "smoke"))
    summary = {"study": STUDY, "config_sha256": cfg_sha, "smoke": smoke, "smoke_gate": "pass" if smoke["passes"] else "fail"}
    if not smoke["passes"]:
        summary["note"] = "the smoke gate failed: no training or confirmation ran"
        rm._dump(os.path.join(out_dir, "summary.json"), summary)
        return summary
    started, training, finalists = int(smoke["starts"]), {}, {}
    for arm in ARMS:
        orig = arm == "module_original"
        tr = rm.run_child({**base, "mode": "train", "init_x": init[arm]["x"], "direction_dim": rm.DIM_JOINT, "coord_scale": scale[arm],
                           "original_bounds": orig, "start_cap": ceiling - reserve - started,
                           "response_path": os.path.join(out_dir, f"train_{arm}.json")}, os.path.join(out_dir, f"train_{arm}"))
        started += int(tr["starts"])
        training[arm] = tr
        if not tr["complete"]:
            summary["incomplete"] = f"the start ceiling was reached in the {arm} arm: no confirmation"
            rm._dump(os.path.join(out_dir, "summary.json"), summary)
            return summary
        finalists[arm] = rm.x_record(tr["final_x"], *rm.bounds(None if orig else protocol_cfg))
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
    head = c["contrasts"]["joint_minus_module_wide"]["pooled"]["success"]
    theta0 = np.asarray(ra.finalist_theta(protocol_cfg))
    joint_x = np.asarray(finalists["joint"]["x"])
    theta_j = np.clip(theta0 + rm.RESIDUAL_SPAN * (joint_x[rm.DIM:] - rm.residual_origin(theta0)), -0.10, 0.10)
    summary.update({
        "paired_success_difference": head["mean"], "paired_success_difference_ci95": [head["low"], head["high"]],
        **c,
        "successes": {n: int(sum(r["sustained"] for m in rows.values() for r in m[n])) for n in POLICIES},
        "drops": {n: int(sum(not r["completed"] for m in rows.values() for r in m[n])) for n in POLICIES},
        "finalists": finalists, "initial": init,
        "joint_residual": {"l2_change": float(np.linalg.norm(theta_j - theta0)), "max_abs_change": float(np.abs(theta_j - theta0).max()),
                           "theta": dict(zip(sr.COEF_NAMES, [float(v) for v in theta_j]))},
        "training": training,
        "accounting": {"smoke_starts": smoke["starts"], "training_starts": {a: t["starts"] for a, t in training.items()},
                       "confirmation_charged_starts": reserve, "charged_total": started + reserve, "ceiling": ceiling, "llm_calls": 0},
        "limitation": "one matched set of ARS trajectories from one parent; a single confirmation",
    })
    rm._dump(os.path.join(out_dir, "summary.json"), summary)
    return summary


__all__ = ["ARMS", "EXTRA", "FAMILY", "POLICIES", "STUDY", "build_manifests", "contrasts", "coord_scale", "initial_x", "make_controller",
           "parent_x", "run"]
