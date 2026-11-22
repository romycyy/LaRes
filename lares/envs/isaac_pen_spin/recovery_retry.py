"""R-118: a second re-capture attempt versus budget-matched continued training of the single-attempt module.

Both arms start from the frozen R-116 module (D-081) and get 24 ARS updates on the same placements and the same
direction draws on their shared coordinates:

* ``continue`` -- the unchanged 11 parameters, one decision per episode;
* ``retry``    -- the same plus a gap and a second-attempt depth multiplier ``kappa`` (initially 0: the R-116 module
  bit for bit). A second decision is made ``gap`` steps after the first attempt ends, only where a first attempt
  actually fired, the pen is centred and still slow, and the attempt can finish by step 300.

The controller, the learner and the Isaac child are ``recovery_module``'s; this module is the driver.
"""

from __future__ import annotations

import json
import os

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin import recovery_authority as ra
from lares.envs.isaac_pen_spin import recovery_module as rm
from lares.envs.isaac_pen_spin import symbolic_reward as sr
from lares.envs.isaac_pen_spin.protocols import freeze_json, load_cell, study_cell

STUDY = "recovery_retry_vs_continue"
ARMS = ("continue", "retry")
POLICIES = ("reference", "r106", "r116", *ARMS)
CONTRASTS = (("retry", "continue"), ("retry", "r116"), ("continue", "r116"), ("retry", "r106"), ("continue", "r106"),
             ("retry", "reference"), ("continue", "reference"), ("r116", "r106"), ("r116", "reference"))
HOLM_FAMILY = ("retry_minus_r116", "continue_minus_r116")


def parent_x(cfg: dict) -> np.ndarray:
    """The frozen R-116 module's normalised parameters, verified against the hash the config names."""
    entry = rm._load(cfg["module_finalist"])
    if entry["sha256"] != cfg["module_sha256"] or rm.x_record(entry["x"])["sha256"] != entry["sha256"]:
        raise SystemExit("the parent is not the frozen R-116 module")
    return np.asarray(entry["x"], dtype=float)


def initial_x(cfg: dict, arm: str) -> np.ndarray:
    x = parent_x(cfg)
    return x if arm == "continue" else np.concatenate([x, rm.RETRY_INIT_TAIL])


def confirmation_manifest(cfg: dict, seed) -> str:
    return str(cfg["confirmation_manifest_template"]).format(seed=int(seed))


def build_manifests(protocol_cfg: dict) -> list[str]:
    import glob

    from lares.eval.manifest import EvaluationManifest, assert_disjoint

    mine = {confirmation_manifest(protocol_cfg, s): cases.build_manifest(3, int(s), int(protocol_cfg["confirmation_cases"]), label="retry_confirm")
            for s in protocol_cfg["confirmation_seeds"]}
    others = [EvaluationManifest.load(p) for p in sorted(glob.glob("config/manifests/pen_spin_stage3_*.yaml")) if p not in mine]
    assert_disjoint(list(mine.values()) + others)
    return [p for p, m in mine.items() if cases.freeze_manifest(p, m)]


def make_controller(name: str, protocol_cfg: dict, out_dir: str | None = None):
    if name == "reference":
        return sr.build_reference(protocol_cfg, "reference", resettable=False)
    if name == "r106":
        return sr.build_controller(protocol_cfg, "r106", ra.finalist_theta(protocol_cfg), ra.normalization(protocol_cfg))
    if name == "r116":
        return rm.build(protocol_cfg, "r116", parent_x(protocol_cfg))
    if name in ARMS:
        entry = rm._load(os.path.join(out_dir or str(protocol_cfg["log_dir"]), "finalists.json"))["finalists"][name]
        if rm.x_record(entry["x"])["sha256"] != entry["sha256"]:
            raise SystemExit("a stored finalist does not match its hash")
        return rm.build(protocol_cfg, name, entry["x"])
    raise ValueError(f"unknown controller {name!r}")


def contrasts(rows: dict, threshold: float, drop_margin: float) -> dict:
    from lares.envs.isaac_pen_spin.incumbent_residual import confirmation_contrast

    out = {f"{a}_minus_{b}": confirmation_contrast({m: (r[a], r[b]) for m, r in rows.items()}, threshold, drop_margin)
           for a, b in CONTRASTS}
    h = sr.holm({k: out[k]["pooled"]["success"]["p_value"] for k in HOLM_FAMILY})
    for k in HOLM_FAMILY:
        arm = k.split("_minus_")[0]
        out[k]["holm"] = h[k]
        out[k]["useful"] = bool(h[k]["rejects"] and out[k]["pooled"]["adopts"]
                                and out[f"{arm}_minus_reference"]["pooled"]["drops"]["non_inferior"])
    return out


def run(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    if os.path.normpath(out_dir) != os.path.normpath(str(protocol_cfg["log_dir"])):
        raise SystemExit("evaluation cells read the finalists from the protocol's own log_dir; do not override it")
    os.makedirs(out_dir, exist_ok=True)
    descriptor = sr.check_reference(protocol_cfg)
    ra.finalist_theta(protocol_cfg)
    ra.normalization(protocol_cfg)
    cfg_sha = sr.config_sha256(protocol_cfg)
    init = {arm: rm.x_record(initial_x(protocol_cfg, arm)) for arm in ARMS}
    freeze_json(os.path.join(out_dir, "frozen_config.json"),
                {"config_sha256": cfg_sha, "config": sr.frozen_config(protocol_cfg), "reference": descriptor,
                 "parameters": [list(p) for p in rm.ALL_PARAMS], "initial": init}, "frozen configuration")
    seeds = [int(s) for s in protocol_cfg["confirmation_seeds"]]
    mpaths = {s: cases.require_frozen(confirmation_manifest(protocol_cfg, s)) for s in seeds}
    ceiling = int(protocol_cfg["start_ceiling"])
    reserve = 2 * len(POLICIES) * len(seeds) * int(protocol_cfg["confirmation_cases"])
    base = {"config_path": os.path.abspath(config_path), "study": STUDY, "config_sha256": cfg_sha}

    smoke = rm.run_child({**base, "mode": "smoke_retry", "response_path": os.path.join(out_dir, "smoke.json")}, os.path.join(out_dir, "smoke"))
    summary = {"study": STUDY, "config_sha256": cfg_sha, "smoke": smoke, "smoke_gate": "pass" if smoke["passes"] else "fail"}
    if not smoke["passes"]:
        summary["note"] = "the smoke gate failed: no training or confirmation ran"
        rm._dump(os.path.join(out_dir, "summary.json"), summary)
        return summary
    started, training, finalists = int(smoke["starts"]), {}, {}
    for arm in ARMS:
        tr = rm.run_child({**base, "mode": "train", "init_x": init[arm]["x"], "direction_dim": rm.DIM_RETRY,
                           "start_cap": ceiling - reserve - started, "response_path": os.path.join(out_dir, f"train_{arm}.json")},
                          os.path.join(out_dir, f"train_{arm}"))
        started += int(tr["starts"])
        training[arm] = tr
        if not tr["complete"]:
            summary["incomplete"] = f"the start ceiling was reached in the {arm} arm: no confirmation"
            rm._dump(os.path.join(out_dir, "summary.json"), summary)
            return summary
        finalists[arm] = rm.x_record(tr["final_x"])
    freeze_json(os.path.join(out_dir, "finalists.json"), {"rule": "each arm's final ARS mean; no selection", "finalists": finalists}, "finalists")

    rows, attempts = {}, {p: [0, 0] for p in ("r116", *ARMS)}
    for s in seeds:
        sub = os.path.join(out_dir, "confirmation", f"s{s}")
        os.makedirs(sub, exist_ok=True)
        rows[f"s{s}"] = {}
        for name in POLICIES:
            cell = os.path.join(sub, f"{name}_cell.json")
            study_cell(protocol_cfg, config_path, mpaths[s], name, cell, record_obs=False)
            params = rm._load(cell)["controller_params"]
            if name in ARMS and params["module"]["sha256"] != finalists[name]["sha256"]:
                raise SystemExit(f"{cell}: the evaluated module is not the frozen finalist")
            if name in attempts:
                attempts[name][0] += int(params["recovery"]["triggered_environment_episodes"])
                attempts[name][1] += int(params["recovery"]["second_attempts"])
            rows[f"s{s}"][name] = load_cell(cell)[1]
    c = contrasts(rows, float(protocol_cfg["threshold"]), float(protocol_cfg["drop_margin"]))
    head = c["retry_minus_continue"]["pooled"]["success"]
    summary.update({
        "paired_success_difference": head["mean"], "paired_success_difference_ci95": [head["low"], head["high"]],
        "contrasts": c,
        "successes": {n: int(sum(r["sustained"] for m in rows.values() for r in m[n])) for n in POLICIES},
        "drops": {n: int(sum(not r["completed"] for m in rows.values() for r in m[n])) for n in POLICIES},
        "first_and_second_attempts": attempts, "finalists": finalists, "initial": init,
        "training": {a: {k: v for k, v in t.items()} for a, t in training.items()},
        "accounting": {"smoke_starts": smoke["starts"], "training_starts": {a: t["starts"] for a, t in training.items()},
                       "confirmation_charged_starts": reserve, "charged_total": started + reserve, "ceiling": ceiling, "llm_calls": 0},
        "limitation": "one matched pair of ARS trajectories from one parent; a single confirmation",
    })
    rm._dump(os.path.join(out_dir, "summary.json"), summary)
    return summary


__all__ = ["ARMS", "CONTRASTS", "POLICIES", "STUDY", "build_manifests", "contrasts", "initial_x", "make_controller", "parent_x", "run"]
