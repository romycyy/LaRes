"""R-130 (Q3 of ``docs/NEXT_RESEARCH_DIRECTION_V2.md``, first spend): an archive-only screen of the recovery hand-back.

No Isaac, no episode, no model call. The archived JOINT and transplant cells (R-124, R-126) hold each case's per-step
axial rate, centre distance and success predicate, but no per-case trigger flag, so the frozen module's single
decision is *reconstructed* from the trace, and every count that depends on it is reported with its sensitivity to
the one-step indexing choice.

Timing, read off ``recovery_module.RecoveryController``: the module's counter equals the control step; at step
``T = round(decision_step)`` it triggers when the centre distance is inside the tolerance and the mean absolute axial
rate of its last 30 observations is under the threshold. The observation at step ``t`` is the simulator state the
archive stores at index ``t - 1``. Support (the blend toward the reset posture) covers actions ``T .. T + S - 1`` with
``S = ceil(ramp_in + ramp_out)``; the stroke boost covers the next 60 actions. The two hand-back boundaries are the
archive indices of the last supported and the last boosted action.

    python scripts/run_shadowhand_spin_stages.py --study recovery_handback_archive
"""

from __future__ import annotations

import glob
import json
import math
import os

import numpy as np

from lares.envs.isaac_pen_spin import recovery_module as rm
from lares.envs.isaac_pen_spin import recovery_tail_audit as ta
from lares.envs.isaac_pen_spin.protocols import load_cell
from lares.utils.hashing import sha256_file

STUDY = "recovery_handback_archive"
CATEGORIES = ("success", "untriggered_failure", "reachable", "dropped_during_blend", "never_recaptured", "pre_boundary_loss",
              "later_loss")


def timeline(cfg: dict) -> dict:
    """The frozen module's decision rule and the two hand-back boundaries, as archive indices."""
    entry = rm._load(cfg["module_asset"])
    if entry["sha256"] != cfg["module_sha256"]:
        raise SystemExit("the module asset is not the frozen JOINT finalist")
    p = dict(zip(rm.NAMES, rm.to_params(np.asarray(entry["x"], dtype=float), *rm.bounds(cfg))))
    T = int(round(p["decision_step"]))
    support = int(math.ceil(p["ramp_in"] + p["ramp_out"]))
    return {"decision_step": T, "threshold": float(p["threshold"]), "centre_tolerance": float(cfg["centre_tolerance"]),
            "rate_window": rm.WINDOW, "support_steps": support, "boost_steps": rm.BOOST_STEPS,
            "support_end": T + support - 1, "boost_end": T + support + rm.BOOST_STEPS - 1,
            "stroke_multiplier": float(p["stroke_multiplier"])}


def reconstruct_trigger(spin, centre, length, tl: dict, shift: int = 0) -> np.ndarray:
    """Per case: alive at the decision, centred and slow over the module's 30-observation window. ``shift = 0`` uses
    the archive indices the observations correspond to (``T - 30 .. T - 1``); ``shift = 1`` is the one-step-later
    reading reported as sensitivity."""
    T, w = tl["decision_step"] + shift, tl["rate_window"]
    rate = np.abs(np.nan_to_num(spin[T - w:T])).mean(axis=0)
    return (np.asarray(length) > tl["decision_step"]) & (np.nan_to_num(centre[T - 1], nan=1.0) < tl["centre_tolerance"]) & (rate < tl["threshold"])


def healthy_at(pred, centre, boundary: int, tl: dict, look: int) -> np.ndarray:
    """The predicate true and the pen centred on each of the ``look`` steps ending at the boundary."""
    sl = slice(boundary - look + 1, boundary + 1)
    return pred[sl].all(axis=0) & (np.nan_to_num(centre[sl], nan=1.0) < tl["centre_tolerance"]).all(axis=0)


def first_loss_after(pred, length, boundary: int) -> np.ndarray:
    """Steps from the boundary to the first predicate loss or the drop (a large number when neither happens)."""
    n = pred.shape[1]
    out = np.full(n, 10 ** 6)
    for i in range(n):
        end = int(length[i])
        tail = pred[boundary + 1:end, i]
        miss = np.flatnonzero(~tail)
        if miss.size:
            out[i] = int(miss[0]) + 1
        elif end < pred.shape[0] - 1:                        # dropped before any predicate loss
            out[i] = max(1, end - boundary)
    return out


def classify(trace: dict, rows: list, tl: dict, cfg: dict, window: int | None = None, shift: int = 0) -> list[dict]:
    """One record per case. A failure is *reachable* when the reconstructed trigger fired, the hand-back was healthy
    at either boundary, and the first loss came within ``window`` steps of that boundary."""
    window = int(cfg["loss_window"] if window is None else window)
    look = int(cfg["healthy_steps"])
    pred = np.asarray(trace["pred"], dtype=bool)
    centre, length = np.asarray(trace["center_dist"]), np.asarray(trace["length"])
    trig = reconstruct_trigger(np.asarray(trace["spin_rate"]), centre, length, tl, shift)
    b1, b2 = tl["support_end"], tl["boost_end"]
    h1, h2 = healthy_at(pred, centre, b1, tl, look), healthy_at(pred, centre, b2, tl, look)
    l1, l2 = first_loss_after(pred, length, b1), first_loss_after(pred, length, b2)
    out = []
    for i, r in enumerate(rows):
        L, ok = int(length[i]), bool(r["sustained"])
        near1, near2 = bool(h1[i] and l1[i] <= window), bool(h2[i] and l2[i] <= window)
        if ok:
            cat = "success"
        elif not trig[i]:
            cat = "untriggered_failure"
        elif near1 or near2:
            cat = "reachable"
        elif not r["completed"] and L <= b1 + 1:
            cat = "dropped_during_blend"
        elif not pred[b1 + 1:L, i].any():
            cat = "never_recaptured"
        elif h1[i] or h2[i]:
            cat = "later_loss"
        else:
            cat = "pre_boundary_loss"
        after = pred[(b2 if near2 else b1) + 1:L, i]
        run = best = 0
        for v in after:
            run = run + 1 if v else 0
            best = max(best, run)
        out.append({"category": cat, "triggered": bool(trig[i]), "label": ta.label(r, cfg), "boundary": "boost_end" if near2 else ("support_end" if near1 else None),
                    "loss_delay": int(l2[i] if near2 else l1[i]) if (near1 or near2) else None,
                    "rolling_recovers": bool(best >= look) if cat == "reachable" else None})
    return out


def fraction(flags, manifests) -> dict:
    """An all-case fraction with its case-level and manifest-clustered 95% intervals."""
    from lares.eval.stats import t_summary

    d, m = np.asarray(flags, dtype=float), np.asarray(manifests)
    ts, cl = t_summary(d), t_summary([float(d[m == k].mean()) for k in sorted(set(manifests))])
    return {"count": int(d.sum()), "cases": int(d.size), "fraction": ts["mean"], "ci95": [ts["low"], ts["high"]],
            "manifest_clustered_ci95": [cl["low"], cl["high"]]}


def load(source: dict) -> list:
    """``[(manifest, trace, rows, recorded trigger count, file hash)]`` for one archive."""
    dirs = sorted(glob.glob(os.path.join(source["log_dir"], "confirmation", "s*")))
    if len(dirs) != int(source["manifests"]):
        raise FileNotFoundError(f"{source['log_dir']}: expected {source['manifests']} archived manifests, found {len(dirs)}")
    out = []
    for d in dirs:
        cell = os.path.join(d, f"{source['cell']}_cell.json")
        trace, rows = load_cell(cell)
        rec = (rm._load(cell)["controller_params"].get("recovery") or {}).get("triggered_environment_episodes")
        out.append((os.path.basename(d), trace, rows, rec, sha256_file(cell)))
    return out


def audit(cells: list, tl: dict, cfg: dict) -> dict:
    recs, man, per_cell, alt = [], [], [], []
    for name, trace, rows, recorded, sha in cells:
        c = classify(trace, rows, tl, cfg)
        recs += c
        alt += classify(trace, rows, tl, cfg, shift=1)
        man += [name] * len(rows)
        n_trig = int(sum(r["triggered"] for r in c))
        per_cell.append({"manifest": name, "reconstructed_first_episode_triggers": n_trig, "recorded_all_episode_triggers": recorded,
                         "recorded_minus_reconstructed": None if recorded is None else int(recorded) - n_trig,
                         "exceeds_recorded_upper_bound": None if recorded is None else bool(n_trig > int(recorded)), "sha256": sha})
    cat = np.array([r["category"] for r in recs])
    reach = cat == "reachable"
    windows = {}
    for w in cfg["sensitivity_windows"]:
        flags = [r["category"] == "reachable" for _, trace, rows, _, _ in cells for r in classify(trace, rows, tl, cfg, window=int(w))]
        windows[str(w)] = fraction(flags, man)
    lab = np.array([r["label"] for r in recs])
    delays = [r["loss_delay"] for r in recs if r["category"] == "reachable"]
    trig = np.array([r["triggered"] for r in recs])
    trig_alt = np.array([r["triggered"] for r in alt])
    return {"cases": len(recs), "manifests": len(cells), "reachable": fraction(reach, man),
            "categories": {c: fraction(cat == c, man) for c in CATEGORIES},
            "triggered": {"reconstructed": int(trig.sum()), "successes": int((trig & (cat == "success")).sum()),
                          "failures": int((trig & (cat != "success")).sum()),
                          "recorded_all_episodes": int(sum(c["recorded_all_episode_triggers"] or 0 for c in per_cell)),
                          "cells_exceeding_recorded_upper_bound": int(sum(bool(c["exceeds_recorded_upper_bound"]) for c in per_cell))},
            "indexing_sensitivity": {"trigger_flags_changed": int((trig != trig_alt).sum()),
                                     "reachable_one_step_later": fraction([r["category"] == "reachable" for r in alt], man)},
            "window_sensitivity": windows,
            "reachable_by_boundary": {b: int(sum(r["boundary"] == b for r in recs if r["category"] == "reachable")) for b in ("support_end", "boost_end")},
            "reachable_by_final_label": {name: int((reach & (lab == name)).sum()) for name in ta.LABELS[1:]},
            "failure_labels_by_category": {c: {name: int(((cat == c) & (lab == name)).sum()) for name in ta.LABELS[1:]} for c in CATEGORIES[1:]},
            "reachable_loss_delay_quartiles": [float(v) for v in np.percentile(delays, [25, 50, 75])] if delays else None,
            "reachable_rolling_recovers": int(sum(bool(r["rolling_recovers"]) for r in recs if r["category"] == "reachable")),
            "per_cell": per_cell}


def gate(results: dict, cfg: dict) -> dict:
    """The registered screen: both JOINT archives' reachable fraction against the threshold."""
    thr = float(cfg["threshold"])
    rows = {k: results[k]["reachable"] for k in cfg["gate_archives"]}
    lows = [v["ci95"][0] for v in rows.values()]
    highs = [v["ci95"][1] for v in rows.values()]
    verdict = "supported" if min(lows) >= thr else ("refuted" if min(highs) < thr else "inconclusive")
    return {"threshold": thr, "fractions": {k: v["fraction"] for k, v in rows.items()}, "ci95": {k: v["ci95"] for k, v in rows.items()},
            "verdict": verdict, "simulation_eligible": bool(all(v["fraction"] >= thr for v in rows.values())),
            "rule": "supported: every gate archive's lower 95% bound at or above the threshold; refuted: any gate archive's upper bound "
                    "below it; simulation is eligible for a separate agreement only if every gate archive's fraction reaches the threshold"}


def build_manifests(protocol_cfg: dict) -> list[str]:
    return []


def make_controller(name: str, protocol_cfg: dict, out_dir: str | None = None):
    raise SystemExit("the hand-back audit reads archives only; it has no controller and runs no simulation")


def run(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    cfg = protocol_cfg
    os.makedirs(out_dir, exist_ok=True)
    tl = timeline(cfg)
    summary = {"study": STUDY, "timeline": tl, "simulation": "none",
               "definitions": {"healthy_steps": int(cfg["healthy_steps"]), "loss_window": int(cfg["loss_window"]),
                               "reachable": "reconstructed trigger, eventual failure, healthy hand-back at either boundary (the predicate true "
                                            "and the pen centred on each of the preceding steps), first predicate loss or drop within the window",
                               "caveat": "a temporal-association screen: not evidence that the loss caused the failure or that a hand-back "
                                         "rule could recover it; trigger flags are reconstructed, not recorded"}}
    try:
        results = {name: audit(load(src), tl, cfg) for name, src in cfg["archives"].items()}
    except FileNotFoundError as e:
        summary.update(verdict="inconclusive", missing=str(e))
    else:
        g = gate(results, cfg)
        summary.update({"failure_labels": min(g["fractions"].values()), "gate_statistics": g, "verdict": g["verdict"], "archives": results})
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return summary


__all__ = ["CATEGORIES", "STUDY", "audit", "build_manifests", "classify", "first_loss_after", "fraction", "gate", "healthy_at", "load",
           "make_controller", "reconstruct_trigger", "run", "timeline"]
