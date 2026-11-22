"""R-120: which failures did continued re-capture training remove? Archive only, no Isaac.

Labels every archived confirmation episode of R-118 (five policies on the same 1,536 cases) and compares the R-116
module with the CONTINUE finalist case by case. The registered statistic is the paired difference (R-116 minus
CONTINUE) of the ``persistent_slow`` indicator.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin.protocols import load_cell

STUDY = "recovery_tail_archive_audit"
LABELS = ("success", "drop_early", "drop_late", "off_centre", "persistent_slow", "intermittent", "near_miss")


def label(row: dict, cfg: dict) -> str:
    if row["sustained"]:
        return "success"
    if not row["completed"]:
        return "drop_early" if int(row["length"]) < int(cfg["early_drop_step"]) else "drop_late"
    if row["final_center_dist"] is None or row["final_center_dist"] >= cases.STAGES[3].success_center_tolerance:
        return "off_centre"
    h = float(row["tail_hit_fraction"])
    return "persistent_slow" if h <= float(cfg["slow_hit_max"]) else ("near_miss" if h >= float(cfg["near_hit_min"]) else "intermittent")


def load(cfg: dict) -> dict:
    """``{policy: [(manifest, rows, final-second mean |axial rate| per case)]}`` over the archived manifests."""
    out = {p: [] for p in cfg["policies"]}
    dirs = sorted(glob.glob(os.path.join(cfg["source_log_dir"], "confirmation", "s*")))
    if len(dirs) != int(cfg["manifests"]):
        raise KeyError(f"expected {cfg['manifests']} archived manifests, found {len(dirs)}")
    for d in dirs:
        hashes = None
        for p in cfg["policies"]:
            trace, rows = load_cell(os.path.join(d, f"{p}_cell.json"))
            if hashes is not None and [r["task_hash"] for r in rows] != hashes:
                raise KeyError(f"{d}: policies are not paired case for case")
            hashes = [r["task_hash"] for r in rows]
            spin = np.abs(np.nan_to_num(trace["spin_rate"]))
            tail = np.array([spin[max(0, int(r["length"]) - 60):int(r["length"]), i].mean() if r["length"] else 0.0 for i, r in enumerate(rows)])
            out[p].append((os.path.basename(d), rows, tail))
    return out


def paired(labels_a: list, labels_b: list, manifests: list, name: str) -> dict:
    """``a`` minus ``b`` for the indicator of ``name``: paired t interval over cases and over manifest means."""
    from lares.eval.stats import t_summary

    d = np.array([float(x == name) - float(y == name) for x, y in zip(labels_a, labels_b)])
    m = np.asarray(manifests)
    ts, cl = t_summary(d), t_summary([float(d[m == k].mean()) for k in sorted(set(manifests))])
    return {"difference": ts["mean"], "ci95": [ts["low"], ts["high"]], "manifest_clustered_ci95": [cl["low"], cl["high"]],
            "a_count": int(sum(x == name for x in labels_a)), "b_count": int(sum(y == name for y in labels_b))}


def build_manifests(protocol_cfg: dict) -> list[str]:
    return []


def make_controller(name: str, protocol_cfg: dict, out_dir: str | None = None):
    raise SystemExit("the tail audit reads archives only; it has no controller and runs no simulation")


def run(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    try:
        data = load(protocol_cfg)
    except (KeyError, FileNotFoundError) as e:
        summary = {"study": STUDY, "verdict": "inconclusive", "missing": str(e), "simulation": "none launched"}
    else:
        lab, man, tail = {}, [], {}
        for p, cells in data.items():
            lab[p] = [label(r, protocol_cfg) for _, rows, _ in cells for r in rows]
            tail[p] = np.concatenate([t for _, _, t in cells])
        man = [m for m, rows, _ in data[protocol_cfg["policies"][0]] for _ in rows]
        a, b = protocol_cfg["parent"], protocol_cfg["candidate"]
        head = paired(lab[a], lab[b], man, "persistent_slow")
        thr = float(protocol_cfg["threshold"])
        verdict = "supported" if head["ci95"][0] >= thr else ("refuted" if head["ci95"][1] < thr else "inconclusive")
        la, lb = np.array(lab[a]), np.array(lab[b])
        summary = {"study": STUDY, "failure_labels": head["difference"], "failure_labels_ci95": head["ci95"], "threshold": thr,
                   "verdict": verdict, "persistent_slow": head, "cases": len(man),
                   "label_table": {p: {name: int(sum(x == name for x in ls)) for name in LABELS} for p, ls in lab.items()},
                   "paired_differences_parent_minus_candidate": {name: paired(lab[a], lab[b], man, name) for name in LABELS},
                   "transitions_parent_to_candidate": {f"{x}->{y}": int(((la == x) & (lb == y)).sum()) for x in LABELS for y in LABELS
                                                       if ((la == x) & (lb == y)).any()},
                   "candidate_failure_tail_rate": {name: float(tail[b][lb == name].mean()) for name in LABELS[3:] if (lb == name).any()},
                   "note": "labels are descriptive; the registered statistic is the paired persistent-slow difference", "simulation": "none"}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return summary


__all__ = ["LABELS", "STUDY", "build_manifests", "label", "load", "make_controller", "paired", "run"]
