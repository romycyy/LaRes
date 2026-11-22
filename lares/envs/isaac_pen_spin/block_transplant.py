"""R-126: frozen block transplants between the R-118 and R-121 JOINT controllers. No training, no LLM.

The JOINT controller differs from R-118 in two blocks: the 32 residual coefficients and the 11 re-capture module
parameters. The four combinations, plus the D-072 reference, are evaluated on the same fresh manifests:

    r118        R-118 residual (= the R-106 coefficients) + R-118 module     (the R-118 asset)
    transplant  R-118 residual + JOINT module                                 (module-only transplant)
    reciprocal  JOINT residual + R-118 module
    joint       JOINT residual + JOINT module                                 (the R-121 asset)

The primary contrast is ``joint - transplant``: what JOINT's realised residual changes add once its module is in place.
A two-manifest pilot is read for one number only (the discordance of that pair), which sizes the confirmation by a
frozen formula.
"""

from __future__ import annotations

import glob
import math
import os

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin import recovery_authority as ra
from lares.envs.isaac_pen_spin import recovery_module as rm
from lares.envs.isaac_pen_spin import symbolic_reward as sr
from lares.envs.isaac_pen_spin.protocols import freeze_json, load_cell, study_cell

STUDY = "parameter_block_transplant"
HYBRIDS = ("r118", "transplant", "reciprocal", "joint")
POLICIES = ("reference", *HYBRIDS)
FAMILY = (("joint", "transplant"), ("joint", "reciprocal"), ("transplant", "r118"), ("reciprocal", "r118"), ("joint", "r118"))
PRIMARY = "joint_minus_transplant"
Z = 1.960 + 0.842                                           # two-sided 5% test, 80% power


def _asset(cfg: dict, name: str) -> np.ndarray:
    entry = rm._load(cfg["assets"][name]["path"])
    if entry["sha256"] != cfg["assets"][name]["sha256"]:
        raise SystemExit(f"{name}: the asset is not the frozen one")
    return np.asarray(entry["x"], dtype=float)


def compose(cfg: dict, name: str) -> np.ndarray:
    """A combination's coordinates in the JOINT asset's ranges (the R-118 module is re-expressed in them)."""
    joint = _asset(cfg, "r121")
    module118 = rm.to_unit(rm.to_params(_asset(cfg, "r118")), *rm.bounds(cfg))
    if name == "joint":
        return np.concatenate([joint[:rm.DIM], joint[rm.DIM:]])
    if name == "transplant":
        return joint[:rm.DIM].copy()
    if name == "reciprocal":
        return np.concatenate([module118, joint[rm.DIM:]])
    if name == "r118":
        return module118
    raise ValueError(f"unknown combination {name!r}")


def check_endpoints(cfg: dict) -> dict:
    """The two unmodified endpoints are reproduced exactly by composition (no rollout)."""
    b = rm.bounds(cfg)
    joint_sha = rm.x_record(compose(cfg, "joint"), *b)["sha256"]
    p118 = rm.to_params(_asset(cfg, "r118"))
    err = float(np.abs(rm.to_params(compose(cfg, "r118"), *b) - p118).max())
    err_rec = float(np.abs(rm.to_params(compose(cfg, "reciprocal")[:rm.DIM], *b) - p118).max())
    ok = joint_sha == cfg["assets"]["r121"]["sha256"] and max(err, err_rec) <= 1e-9 and len(compose(cfg, "r118")) == rm.DIM
    out = {"joint_sha256": joint_sha, "joint_matches_asset": joint_sha == cfg["assets"]["r121"]["sha256"],
           "r118_module_max_abs_error": err, "reciprocal_module_max_abs_error": err_rec,
           "r118_residual": "the frozen R-106 coefficients (no residual coordinates)", "passes": bool(ok)}
    if not ok:
        raise SystemExit(f"endpoint reproduction failed: {out}")
    return out


def confirmation_size(discordance: float, cfg: dict) -> dict:
    """Manifests the frozen rule asks for: 80% power for the registered effect against zero."""
    d = max(float(discordance), float(cfg["discordance_floor"]))
    pairs = math.ceil(Z * Z * d / float(cfg["threshold"]) ** 2)
    manifests = max(int(cfg["min_confirmation_manifests"]), math.ceil(pairs / int(cfg["confirmation_cases"])))
    cap = len(cfg["confirmation_seeds"])
    return {"discordance": float(discordance), "discordance_used": d, "pairs_required": pairs, "manifests": manifests, "cap": cap,
            "within_cap": manifests <= cap}


def manifest_for(cfg: dict, role: str, seed) -> str:
    return str(cfg[f"{role}_manifest_template"]).format(seed=int(seed))


def build_manifests(protocol_cfg: dict) -> list[str]:
    from lares.eval.manifest import EvaluationManifest, assert_disjoint

    mine = {manifest_for(protocol_cfg, role, s): cases.build_manifest(3, int(s), int(protocol_cfg["confirmation_cases"]), label=f"blockswap_{role}")
            for role in ("pilot", "confirmation") for s in protocol_cfg[f"{role}_seeds"]}
    others = [EvaluationManifest.load(p) for p in sorted(glob.glob("config/manifests/pen_spin_stage3_*.yaml")) if p not in mine]
    assert_disjoint(list(mine.values()) + others)
    return [p for p, m in mine.items() if cases.freeze_manifest(p, m)]


def make_controller(name: str, protocol_cfg: dict, out_dir: str | None = None):
    if name == "reference":
        return sr.build_reference(protocol_cfg, "reference", resettable=False)
    if name in HYBRIDS:
        return rm.build(protocol_cfg, name, compose(protocol_cfg, name))
    raise ValueError(f"unknown controller {name!r}")


def _cells(protocol_cfg: dict, config_path: str, out_dir: str, role: str, seeds) -> dict:
    rows = {}
    for s in seeds:
        mpath = cases.require_frozen(manifest_for(protocol_cfg, role, s))
        sub = os.path.join(out_dir, role, f"s{s}")
        os.makedirs(sub, exist_ok=True)
        rows[f"s{s}"] = {}
        for name in POLICIES:
            cell = os.path.join(sub, f"{name}_cell.json")
            study_cell(protocol_cfg, config_path, mpath, name, cell, record_obs=False)
            rows[f"s{s}"][name] = load_cell(cell)[1]
    return rows


def discordance(rows: dict, a: str, b: str) -> float:
    pairs = [(x["sustained"], y["sustained"]) for m in rows.values() for x, y in zip(m[a], m[b])]
    return float(np.mean([bool(x) != bool(y) for x, y in pairs]))


def interaction(rows: dict) -> dict:
    """Per-case ``joint - transplant - reciprocal + r118`` (diagnostic)."""
    from lares.eval.stats import t_summary

    v = [float(j["sustained"]) - float(t["sustained"]) - float(r["sustained"]) + float(k["sustained"])
         for m in rows.values() for j, t, r, k in zip(m["joint"], m["transplant"], m["reciprocal"], m["r118"])]
    ts = t_summary(v)
    return {"mean": ts["mean"], "low": ts["low"], "high": ts["high"], "n": len(v)}


def contrasts(rows: dict, cfg: dict) -> dict:
    from lares.envs.isaac_pen_spin.incumbent_residual import confirmation_contrast

    out = {f"{a}_minus_{b}": confirmation_contrast({m: (r[a], r[b]) for m, r in rows.items()}, float(cfg["threshold"]), float(cfg["drop_margin"]))
           for a, b in FAMILY}
    h = sr.holm({k: v["pooled"]["success"]["p_value"] for k, v in out.items()})
    for k, v in out.items():
        v["holm"] = h[k]
    return out


def _rates(rows: dict) -> dict:
    return {p: {"successes": int(sum(r["sustained"] for m in rows.values() for r in m[p])),
                "drops": int(sum(not r["completed"] for m in rows.values() for r in m[p])),
                "cases": int(sum(len(m[p]) for m in rows.values()))} for p in POLICIES}


def run(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    descriptor = sr.check_reference(protocol_cfg)
    ra.finalist_theta(protocol_cfg)
    ra.normalization(protocol_cfg)
    endpoints = check_endpoints(protocol_cfg)
    cfg_sha = sr.config_sha256(protocol_cfg)
    b = rm.bounds(protocol_cfg)
    freeze_json(os.path.join(out_dir, "frozen_config.json"),
                {"config_sha256": cfg_sha, "config": sr.frozen_config(protocol_cfg), "reference": descriptor, "endpoints": endpoints,
                 "policies": {n: rm.x_record(compose(protocol_cfg, n), *b) for n in HYBRIDS}}, "frozen configuration")
    pilot = _cells(protocol_cfg, config_path, out_dir, "pilot", [int(s) for s in protocol_cfg["pilot_seeds"]])
    size = confirmation_size(discordance(pilot, "joint", "transplant"), protocol_cfg)
    freeze_json(os.path.join(out_dir, "sizing.json"), size, "confirmation size")
    summary = {"study": STUDY, "config_sha256": cfg_sha, "endpoints": endpoints, "sizing": size,
               "pilot": {"policies": _rates(pilot), "note": "read for sizing only; excluded from confirmation"}}
    if not size["within_cap"]:
        summary["incomplete"] = "the frozen sizing rule asks for more manifests than the cap: no confirmation ran"
        rm._dump(os.path.join(out_dir, "summary.json"), summary)
        return summary
    seeds = [int(s) for s in protocol_cfg["confirmation_seeds"]][:size["manifests"]]
    rows = _cells(protocol_cfg, config_path, out_dir, "confirmation", seeds)
    c = contrasts(rows, protocol_cfg)
    head = c[PRIMARY]["pooled"]["success"]
    summary.update({
        "confirmation_seeds": seeds, "paired_success_difference": head["mean"], "paired_success_difference_ci95": [head["low"], head["high"]],
        "policies": _rates(rows), "contrasts": c, "interaction": interaction(rows),
        "accounting": {"pilot_episodes": len(POLICIES) * len(pilot) * int(protocol_cfg["confirmation_cases"]),
                       "confirmation_episodes": len(POLICIES) * len(seeds) * int(protocol_cfg["confirmation_cases"]), "llm_calls": 0, "training": "none"},
        "limitation": "frozen counterfactual combinations of two trained endpoints; one confirmation; power is nominal (pilot-estimated "
                      "discordance, Holm adjustment)",
    })
    rm._dump(os.path.join(out_dir, "summary.json"), summary)
    return summary


__all__ = ["FAMILY", "HYBRIDS", "POLICIES", "PRIMARY", "STUDY", "build_manifests", "check_endpoints", "compose", "confirmation_size", "contrasts",
           "discordance", "interaction", "make_controller", "run"]
