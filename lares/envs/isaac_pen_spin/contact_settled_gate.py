"""The settled-window contact decoder gate: R-093's controller, pulsed after the pen has settled, scored by a paired contrast.

R-093 (D-064) failed its sign-agreement threshold because a request-independent relative motion (the pen settling and
the fingers closing at steps 10-40, about 0.17 m/s along the normal) was five times the 0.03 m/s pulse. This gate
changes two things, both declared before any run:

* the pulse window is late (``windows.pulse_start`` / ``pulse_stop`` in the config), after the pen has come to rest in
  the closed reference posture, and the decoder follows the pen;
* tracking is the **paired contrast**: for a fingertip and local axis, the same 40 cases are pulsed ``+`` in one cell
  and ``-`` in another, and per case ``d = (gain_pos + gain_neg) / 2`` (each gain is already signed by its request, so
  ``d`` is half the difference of the observed velocities along the axis, over the requested speed). A motion common
  to both cells cancels. A pair is eligible only if the case is eligible in both cells (R-093's motion-blind rule).

Everything else (controller, frames, Jacobian audit, eligibility, one whole manifest per fresh process) is R-093's,
imported from ``contact_gate``. Two registry names serve this module: ``contact_settled_calibration`` (candidate
settings on the calibration manifest; case ``i`` pulses direction ``i mod 30``, so a cell has no pairs and reports
the unpaired statistics plus how still the pen is) and ``contact_settled_gate`` (neutral hold, zero-request hold and
the 30 directional cells; refuses to run unless the ``frozen`` block matches ``frozen_sha256``).
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin import contact_control as cc
from lares.envs.isaac_pen_spin import contact_gate as cg
from lares.envs.isaac_pen_spin.protocols import freeze_json, load_cell, study_cell

STAGE = 3
CALIBRATION, GATE = "contact_settled_calibration", "contact_settled_gate"
LABELS = {CALIBRATION: "contact_settled_calibration", GATE: "contact_settled_gate"}
CONTRASTS = tuple((k, a) for k in range(len(cc.FINGERTIPS)) for a in range(3))
PAIRED_THRESHOLDS = ("expected_contrasts", "min_eligible_pairs", "paired_sign_agreement_min", "paired_gain_median_min",
                     "paired_gain_median_max", "jacobian_rel_error_median_max", "retention_diff_min")


def contrast_name(k: int, a: int) -> str:
    return f"{cc.FINGERTIPS[k]}_{cc.LOCAL_AXES[a]}"


# ---------------------------------------------------------------------------------------------------------------------
#  Paired statistics (pure functions of two cells' ``cell_tracking`` results)
# ---------------------------------------------------------------------------------------------------------------------


def paired_contrast(pos: dict, neg: dict) -> dict:
    """Per-case ``d = (gain_pos + gain_neg) / 2`` over the cases eligible in both cells of one fingertip-axis."""
    gp = np.array([np.nan if g is None else g for g in pos["gain"]], dtype=np.float64)
    gn = np.array([np.nan if g is None else g for g in neg["gain"]], dtype=np.float64)
    if gp.shape != gn.shape:
        raise ValueError("the two cells score different numbers of cases")
    both = np.array(pos["eligible_mask"], dtype=bool) & np.array(neg["eligible_mask"], dtype=bool)
    d = 0.5 * (gp + gn)
    chosen = d[both]
    if not np.isfinite(chosen).all():
        raise ValueError("an eligible pair has a non-finite gain")
    common = 0.5 * (gp - gn)[both]
    return {"cases": int(len(d)), "eligible_pairs": int(both.sum()), "pair_mask": [bool(b) for b in both],
            "d": [float(x) if np.isfinite(x) else None for x in d],
            "sign_agreement": float((chosen > 0).mean()) if len(chosen) else None,
            "gain_median": float(np.median(chosen)) if len(chosen) else None,
            "common_mode_median": float(np.median(common)) if len(common) else None}


def paired_verdict(contrasts: dict, jacobian_rel_errors, hold_timeout_rate, neutral_timeout_rate,
                   thresholds: dict) -> dict:
    """``pass`` / ``fail`` / ``inconclusive``, with R-093's precedence: a failed Jacobian audit or retention check
    fails whatever the coverage; otherwise missing contrasts or one under the pair minimum is inconclusive; otherwise
    the pooled paired sign agreement and median paired gain decide."""
    missing = set(PAIRED_THRESHOLDS) - set(thresholds)
    if missing:
        raise ValueError(f"thresholds lack {sorted(missing)}")
    th, checks = thresholds, {}
    errors = np.asarray(jacobian_rel_errors, dtype=np.float64).ravel()
    jac = float(np.median(errors)) if len(errors) and np.isfinite(errors).all() else None
    checks["jacobian_rel_error_median"] = {"value": jac, "ok": None if jac is None else jac < th["jacobian_rel_error_median_max"],
                                           "rule": f"< {th['jacobian_rel_error_median_max']}"}
    rates_ok = all(r is not None and np.isfinite(r) for r in (hold_timeout_rate, neutral_timeout_rate))
    diff = float(hold_timeout_rate) - float(neutral_timeout_rate) if rates_ok else None
    checks["retention_diff"] = {"value": diff, "ok": None if diff is None else diff >= th["retention_diff_min"] - 1e-9,
                                "rule": f">= {th['retention_diff_min']} (zero-request hold minus neutral hold timeout rate)"}
    short = sorted(n for n, c in contrasts.items() if c["eligible_pairs"] < th["min_eligible_pairs"])
    covered = len(contrasts) == th["expected_contrasts"] and not short
    checks["coverage"] = {"value": {"contrasts": len(contrasts), "under_minimum": short,
                                    "min_pairs": min((c["eligible_pairs"] for c in contrasts.values()), default=0)},
                          "ok": covered,
                          "rule": f"{th['expected_contrasts']} contrasts, each with >= {th['min_eligible_pairs']} eligible pairs"}
    pooled = np.array([x for c in contrasts.values() for x, m in zip(c["d"], c["pair_mask"]) if m], dtype=np.float64)
    agreement = float((pooled > 0).mean()) if len(pooled) else None
    median = float(np.median(pooled)) if len(pooled) else None
    checks["paired_sign_agreement"] = {"value": agreement,
                                       "ok": None if agreement is None else agreement >= th["paired_sign_agreement_min"],
                                       "rule": f">= {th['paired_sign_agreement_min']} over eligible pairs"}
    checks["paired_gain_median"] = {
        "value": median,
        "ok": None if median is None else th["paired_gain_median_min"] <= median <= th["paired_gain_median_max"],
        "rule": f"in [{th['paired_gain_median_min']}, {th['paired_gain_median_max']}] over eligible pairs"}
    independent = (checks["jacobian_rel_error_median"]["ok"], checks["retention_diff"]["ok"])
    tracking = (checks["paired_sign_agreement"]["ok"], checks["paired_gain_median"]["ok"])
    if False in independent:
        verdict = cc.FAIL
    elif None in independent or not covered or None in tracking:
        verdict = cc.INCONCLUSIVE
    else:
        verdict = cc.PASS if all(tracking) else cc.FAIL
    return {"verdict": verdict, "checks": checks, "eligible_pairs": int(len(pooled)),
            "per_contrast": {n: {k: c[k] for k in ("eligible_pairs", "sign_agreement", "gain_median", "common_mode_median")}
                             for n, c in contrasts.items()},
            "thresholds": dict(th)}


# ---------------------------------------------------------------------------------------------------------------------
#  Config, manifests, controllers
# ---------------------------------------------------------------------------------------------------------------------


def candidate(protocol_cfg: dict, index: int) -> dict:
    """Calibration setting ``index``: ``calibration.base`` with that setting's decoder / controller / windows overrides."""
    cal = protocol_cfg["calibration"]
    over = cal["settings"][index]
    out = {key: {**cal["base"][key], **over.get(key, {})} for key in ("decoder", "controller")}
    out["windows"] = {**protocol_cfg["windows"], **over.get("windows", {})}
    return out


def require_frozen(protocol_cfg: dict) -> dict:
    frozen = protocol_cfg.get("frozen")
    if not frozen or not protocol_cfg.get("frozen_sha256"):
        raise SystemExit("the gate has no frozen block: run the calibration and freeze its constants first")
    frozen = {"decoder": dict(frozen["decoder"]), "controller": dict(frozen["controller"]),
              "windows": dict(frozen["windows"]), "thresholds": dict(protocol_cfg["thresholds"]),
              "eligibility": dict(protocol_cfg["eligibility"])}
    got = cg.frozen_sha256(frozen)
    if got != protocol_cfg["frozen_sha256"]:
        raise SystemExit(f"frozen block hash {got} != the config's frozen_sha256 {protocol_cfg['frozen_sha256']}")
    return frozen


def build_manifests(protocol_cfg: dict) -> list[str]:
    from lares.eval.manifest import EvaluationManifest, assert_disjoint

    n = int(protocol_cfg["cases"])
    mine = {protocol_cfg["manifest"]: cases.build_manifest(STAGE, int(protocol_cfg["pool_seed"]), n, label=LABELS[GATE]),
            protocol_cfg["calibration_manifest"]: cases.build_manifest(
                STAGE, int(protocol_cfg["calibration_pool_seed"]), n, label=LABELS[CALIBRATION])}
    others = [EvaluationManifest.load(p) for p in sorted(glob.glob("config/manifests/pen_spin_stage3_*.yaml"))
              if p not in mine]
    assert_disjoint(list(mine.values()) + others)
    return [p for p, m in mine.items() if cases.freeze_manifest(p, m)]


def make_controller(name: str, protocol_cfg: dict, out_dir: str | None = None):
    n = int(protocol_cfg["cases"])
    if name == "neutral":
        from lares.envs.isaac_pen_spin.controllers import NeutralPose

        return NeutralPose()
    if name.startswith("cal"):
        hold = name.startswith("calhold")
        frozen = candidate(protocol_cfg, int(name[7 if hold else 3:]))
        return cg.ContactFrameController(name, frozen, None if hold else cg.calibration_directions(n))
    frozen = require_frozen(protocol_cfg)
    if name == "hold":
        return cg.ContactFrameController(name, frozen)
    if name in cg.DIRECTION_NAMES:
        return cg.ContactFrameController(name, frozen, [cg.DIRECTIONS[cg.DIRECTION_NAMES.index(name)]] * n)
    raise ValueError(f"unknown controller {name!r}")


def pen_stillness(cell: str, windows: dict) -> dict:
    """Median pen linear and angular speed over the pulse window, over the cases still in play: how settled it is."""
    trace, _ = load_cell(cell)
    sl = cases.obs_slices()
    p0, p1 = int(windows["pulse_start"]), int(windows["pulse_stop"])
    alive = trace["length"] > p1
    obs = trace["obs"][alive, p0:p1 + 1]
    if obs.size == 0:
        return {"alive": int(alive.sum()), "pen_speed_median": None, "pen_angular_speed_median": None}
    return {"alive": int(alive.sum()),
            "pen_speed_median": float(np.median(np.linalg.norm(obs[..., sl["obj_linvel"]], axis=-1))),
            "pen_angular_speed_median": float(np.median(np.linalg.norm(obs[..., sl["obj_angvel"]], axis=-1))
                                              / cc.VEL_OBS_SCALE)}


# ---------------------------------------------------------------------------------------------------------------------
#  Drivers
# ---------------------------------------------------------------------------------------------------------------------


def run_calibration(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    cal = protocol_cfg["calibration"]
    plan = ["neutral"] + [(f"calhold{i}" if s.get("hold") else f"cal{i}") for i, s in enumerate(cal["settings"])]
    if len(plan) > int(cal["max_cells"]):
        raise SystemExit(f"{len(plan)} calibration cells exceed the declared {cal['max_cells']}")
    mpath = cases.require_frozen(protocol_cfg["calibration_manifest"])
    n = int(protocol_cfg["cases"])
    out = {"manifest": cases.manifest_record(mpath), "cells": {}, "episodes": 0,
           "note": "calibration only: chooses constants and the pulse window, never gate evidence"}
    for name in plan:
        cell = os.path.join(out_dir, f"{name}.json")
        study_cell(protocol_cfg, config_path, mpath, name, cell)
        out["episodes"] += n
        _, rows = load_cell(cell)
        row = {"timeout_rate": cg.timeout_rate(rows), "sustained": int(sum(r["sustained"] for r in rows))}
        if name != "neutral":
            setting = candidate(protocol_cfg, int(name[7 if name.startswith("calhold") else 3:]))
            row.update(setting=setting, **pen_stillness(cell, setting["windows"]))
            if not name.startswith("calhold"):
                a = cg.analyse_pulses(cell, cg.calibration_directions(n), setting["windows"], protocol_cfg["eligibility"])
                errs = np.array(a["jacobian_errors"])
                row.update(cg.pooled(a["cells"]), bound_active_fraction=a["bound_active_fraction"],
                           residual_mean=a["residual_mean"],
                           jacobian_rel_error_median=float(np.median(errs)) if len(errs) else None,
                           per_direction={k: {"gain": [g for g, e in zip(c["gain"], c["eligible_mask"]) if e],
                                              "excluded": c.get("excluded")} for k, c in a["cells"].items()})
        out["cells"][name] = row
        with open(os.path.join(out_dir, "summary.partial.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def run_gate(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    frozen = require_frozen(protocol_cfg)
    freeze_json(os.path.join(out_dir, "frozen.json"), {"frozen": frozen, "sha256": protocol_cfg["frozen_sha256"]},
                "frozen block")
    mpath = cases.require_frozen(protocol_cfg["manifest"])
    n = int(protocol_cfg["cases"])
    cells, errors, per_cell, rates = {}, [], {}, {}
    for name in cg.GATE_CELLS:
        cell = os.path.join(out_dir, f"{name}.json")
        study_cell(protocol_cfg, config_path, mpath, name, cell)
        if name in ("neutral", "hold"):
            _, rows = load_cell(cell)
            rates[name] = cg.timeout_rate(rows)
            per_cell[name] = {"timeout_rate": rates[name], "sustained": int(sum(r["sustained"] for r in rows))}
            if name == "hold":
                per_cell[name].update(pen_stillness(cell, frozen["windows"]))
            continue
        d = cg.DIRECTIONS[cg.DIRECTION_NAMES.index(name)]
        a = cg.analyse_pulses(cell, [d] * n, frozen["windows"], frozen["eligibility"])
        cells[name] = a["cells"][name]
        errors.extend(a["jacobian_errors"])
        per_cell[name] = {**{k: a[k] for k in ("timeout_rate", "sustained", "bound_active_fraction", "residual_mean")},
                          **{k: cells[name].get(k) for k in ("eligible", "sign_agreement", "gain_median", "excluded")}}
    contrasts = {contrast_name(k, a): paired_contrast(cells[cg.direction_name(k, a, 1.0)],
                                                      cells[cg.direction_name(k, a, -1.0)]) for k, a in CONTRASTS}
    verdict = paired_verdict(contrasts, errors, rates["hold"], rates["neutral"], frozen["thresholds"])
    by_tip = {}
    for k, tip in enumerate(cc.FINGERTIPS):
        d = np.array([x for a in range(3) for x, m in zip(contrasts[contrast_name(k, a)]["d"],
                                                           contrasts[contrast_name(k, a)]["pair_mask"]) if m])
        by_tip[tip] = {"pairs": int(len(d)), "sign_agreement": float((d > 0).mean()) if len(d) else None,
                       "gain_median": float(np.median(d)) if len(d) else None}
    summary = {"study": GATE, "verdict": verdict["verdict"], "gate": verdict, "by_fingertip": by_tip,
               "unpaired": cg.pooled(cells), "cells": per_cell,
               "timeout_rate": rates["hold"], "neutral_timeout_rate": rates["neutral"],
               "timeout_rate_difference": rates["hold"] - rates["neutral"], "jacobian_samples": len(errors),
               "frozen_sha256": protocol_cfg["frozen_sha256"],
               "decoder_sha256": cc.DecoderConfig.from_dict(frozen["decoder"]).sha256(),
               "manifest": cases.manifest_record(mpath), "episodes": n * len(cg.GATE_CELLS),
               "limitation": "kinematic tracking of fingertip link origins near an ideal capsule, as a paired contrast "
                             "that removes motion common to both signs; proximity is not contact and a pass says "
                             "nothing about task success"}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def run(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    return (run_calibration if protocol_cfg["study"] == CALIBRATION else run_gate)(protocol_cfg, config_path, out_dir)


__all__ = ["CALIBRATION", "CONTRASTS", "GATE", "PAIRED_THRESHOLDS", "build_manifests", "candidate", "contrast_name",
           "make_controller", "paired_contrast", "paired_verdict", "pen_stillness", "require_frozen", "run",
           "run_calibration", "run_gate"]
