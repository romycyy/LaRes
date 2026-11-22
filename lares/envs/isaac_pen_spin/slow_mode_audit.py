"""R-110: does the reset geometry predict the R-106 finalist's slow-rolling mode? Archive only, no Isaac.

About 41% of the finalist's episodes settle into a slow roll (about 2 rad/s) instead of the rolling mode (about
8 rad/s). This audit asks, on the 4,608 archived confirmation episodes of R-109, whether a compact gate on the 29
reset placement draws (and their squares) separates slow-spin failures from sustained successes when every
prediction is held out by manifest group. The registered statistic is that gate's held-out balanced accuracy minus
0.50. Prefix gates, divergence timing, roll direction and reference-to-finalist flips are diagnostics.
"""

from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin.protocols import load_cell

STUDY = "slow_mode_archive_audit"
SLOW, SUCCESS, EXCLUDED = 1, 0, -1


def labels(rows: list, tolerance: float) -> np.ndarray:
    """Slow-spin failure (completed, not sustained, centred), sustained success, or excluded (drop / off-centre)."""
    out = np.full(len(rows), EXCLUDED)
    for i, r in enumerate(rows):
        if r["sustained"]:
            out[i] = SUCCESS
        elif r["completed"] and r["final_center_dist"] is not None and r["final_center_dist"] < tolerance:
            out[i] = SLOW
    return out


def reset_features(vecs) -> np.ndarray:
    v = np.asarray(vecs, dtype=float)
    return np.concatenate([v, v * v], axis=1)


def prefix_features(spin, centre, window: int) -> np.ndarray:
    """Five summaries of steps ``[0, window)`` per case from ``(T, cases)`` traces."""
    s, c = np.nan_to_num(np.asarray(spin, float)[:window]), np.nan_to_num(np.asarray(centre, float)[:window])
    return np.stack([np.abs(s).mean(0), s.mean(0), np.abs(s[-1]), c.mean(0), c[-1]], axis=1)


def fit_logistic(X, y, penalty: float) -> tuple:
    """Class-balanced L2 logistic regression: mean weighted log-loss + penalty / (2 n) ||w||^2, intercept free."""
    from scipy.optimize import minimize

    X, y = np.asarray(X, float), np.asarray(y, float)
    n, d = X.shape
    wts = np.where(y == 1, 0.5 / max(y.sum(), 1), 0.5 / max((1 - y).sum(), 1))

    def loss(p):
        z = X @ p[:d] + p[d]
        ll = np.logaddexp(0.0, z) - y * z
        g = wts * (1.0 / (1.0 + np.exp(-z)) - y)
        return float((wts * ll).sum() + penalty / (2.0 * n) * (p[:d] ** 2).sum()), np.concatenate([X.T @ g + penalty / n * p[:d], [g.sum()]])

    res = minimize(loss, np.zeros(d + 1), jac=True, method="L-BFGS-B")
    return res.x[:d], float(res.x[d])


def held_out_probabilities(X, y, groups, folds: int, penalty: float) -> np.ndarray:
    """P(slow) for every case, each predicted by a gate fitted without its manifest group (``group mod folds``)."""
    X, y, fold = np.asarray(X, float), np.asarray(y), np.asarray(groups) % folds
    out = np.zeros(len(y))
    for f in range(folds):
        tr, te = fold != f, fold == f
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        w, b = fit_logistic((X[tr] - mu) / sd, y[tr], penalty)
        out[te] = 1.0 / (1.0 + np.exp(-(((X[te] - mu) / sd) @ w + b)))
    return out


def balanced_accuracy(y, prob, threshold: float = 0.5) -> float:
    y, pred = np.asarray(y), np.asarray(prob) >= threshold
    return float(0.5 * (pred[y == 1].mean() + (~pred[y == 0]).mean()))


def auc(y, score) -> float:
    """P(score of a slow case > score of a success), ties counted half."""
    from scipy.stats import rankdata

    y = np.asarray(y)
    r = rankdata(score)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def bootstrap_manifests(y, prob, groups, resamples: int, seed: int, fn=balanced_accuracy) -> list:
    rng = np.random.default_rng(int(seed))
    y, prob, groups = np.asarray(y), np.asarray(prob), np.asarray(groups)
    ids = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in ids}
    vals = []
    for _ in range(int(resamples)):
        take = np.concatenate([index[g] for g in rng.choice(ids, size=len(ids))])
        vals.append(fn(y[take], prob[take]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def gate(X, y, groups, cfg: dict) -> dict:
    prob = held_out_probabilities(X, y, groups, int(cfg["folds"]), float(cfg["penalty"]))
    ba = balanced_accuracy(y, prob)
    ci = bootstrap_manifests(y, prob, groups, int(cfg["resamples"]), int(cfg["seed"]))
    return {"balanced_accuracy": ba, "advantage": ba - 0.5, "advantage_ci95": [ci[0] - 0.5, ci[1] - 0.5], "auc": auc(y, prob),
            "features": int(np.asarray(X).shape[1])}


def load_archive(cfg: dict) -> dict:
    """Per-case arrays for the root policy and the reference over every confirmation manifest; placements are
    reconstructed from the manifest seed and verified against the archived task hashes."""
    from lares.eval.manifest import _hash_rand_vec

    tol = cases.STAGES[3].success_center_tolerance
    out = {k: [] for k in ("placement", "label", "ref_label", "group", "spin", "centre")}
    dirs = sorted(glob.glob(os.path.join(cfg["source_log_dir"], "confirmation", "b*", "*")),
                  key=lambda p: int(re.search(r"_s(\d+)$", p).group(1)))
    for g, d in enumerate(dirs):
        seed = int(re.search(r"_s(\d+)$", d).group(1))
        trace, rows = load_cell(os.path.join(d, f"{cfg['policy']}_cell.json"))
        _, ref_rows = load_cell(os.path.join(d, "reference_cell.json"))
        vecs = cases.placements(seed, len(rows))
        if [_hash_rand_vec(v) for v in vecs] != [r["task_hash"] for r in rows] or [r["task_hash"] for r in ref_rows] != [r["task_hash"] for r in rows]:
            raise KeyError(f"placements of {d} cannot be reconstructed from seed {seed}")
        T = int(cfg["horizon"])
        pad = lambda a: np.concatenate([a, np.full((max(0, T - a.shape[0]), a.shape[1]), np.nan)])[:T]   # noqa: E731
        out["placement"].append(vecs)
        out["label"].append(labels(rows, tol))
        out["ref_label"].append(labels(ref_rows, tol))
        out["group"].append(np.full(len(rows), g))
        out["spin"].append(pad(trace["spin_rate"]).T)
        out["centre"].append(pad(trace["center_dist"]).T)
    if len(dirs) != int(cfg["manifests"]):
        raise KeyError(f"expected {cfg['manifests']} archived manifests, found {len(dirs)}")
    return {k: np.concatenate(v) for k, v in out.items()}


def diagnostics(a: dict, keep, cfg: dict) -> dict:
    y, g = a["label"][keep], a["group"][keep]
    spin, centre = a["spin"][keep].T, a["centre"][keep].T                 # (T, cases)
    out = {"prefix_gates": {}}
    reset = reset_features(a["placement"][keep])
    for w in cfg["prefix_windows"]:
        pf = prefix_features(spin, centre, int(w))
        out["prefix_gates"][f"first_{w}_steps"] = gate(pf, y, g, cfg)
        out["prefix_gates"][f"first_{w}_steps_plus_reset"] = gate(np.concatenate([pf, reset], axis=1), y, g, cfg)
    trail = int(cfg["trailing_mean"])
    absspin = np.abs(np.nan_to_num(spin))
    timing = {}
    for t in range(trail, absspin.shape[0] + 1, 5):
        timing[t] = auc(y, -absspin[t - trail:t].mean(0))                 # slow cases have the lower rate
    out["divergence"] = {"auc_by_step": {str(t): round(v, 4) for t, v in timing.items()},
                         **{f"first_step_auc_{int(100 * th)}": next((t for t, v in timing.items() if v >= th), None) for th in (0.80, 0.90)}}
    lo, hi = cfg["direction_window"]
    sign = np.sign(np.nan_to_num(spin[int(lo):int(hi)]).mean(0))
    out["roll_direction"] = {s: {"cases": int((sign == v).sum()), "success_rate": float((y[sign == v] == SUCCESS).mean()) if (sign == v).any() else None}
                             for s, v in (("positive", 1.0), ("negative", -1.0))}
    ref, fin = a["ref_label"], a["label"]
    names = {SUCCESS: "success", SLOW: "slow", EXCLUDED: "drop_or_off_centre"}
    out["reference_to_finalist"] = {f"{names[r]}->{names[f]}": int(((ref == r) & (fin == f)).sum()) for r in names for f in names}
    both = (ref != EXCLUDED) & (fin != EXCLUDED)
    out["reference_outcome_as_predictor"] = {"balanced_accuracy": balanced_accuracy(fin[both], (ref[both] == SLOW).astype(float)),
                                             "cases": int(both.sum())}
    X = reset_features(a["placement"][keep])
    w, _ = fit_logistic((X - X.mean(0)) / np.where(X.std(0) > 1e-12, X.std(0), 1.0), y, float(cfg["penalty"]))
    labels_ = [f"p{i}" for i in range(X.shape[1] // 2)] + [f"p{i}^2" for i in range(X.shape[1] // 2)]
    top = np.argsort(-np.abs(w))[:8]
    out["largest_reset_coefficients"] = {labels_[i]: float(w[i]) for i in top}
    out["placement_layout"] = "p0-p2 pen position, p3-p4 pen rotation, p5-p28 hand joint offsets (24 joints, simulator order)"
    return out


def verdict(interval: list, threshold: float) -> str:
    if interval[0] >= threshold:
        return "supported"
    return "refuted" if interval[1] < threshold else "inconclusive"


def build_manifests(protocol_cfg: dict) -> list[str]:
    return []


def make_controller(name: str, protocol_cfg: dict, out_dir: str | None = None):
    raise SystemExit("the slow-mode audit reads archives only; it has no controller and runs no simulation")


def run(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    try:
        a = load_archive(protocol_cfg)
    except (KeyError, FileNotFoundError) as e:
        summary = {"study": STUDY, "verdict": "inconclusive", "missing": str(e), "simulation": "none launched"}
    else:
        keep = a["label"] != EXCLUDED
        y, g = a["label"][keep], a["group"][keep]
        head = gate(reset_features(a["placement"][keep]), y, g, protocol_cfg)
        summary = {"study": STUDY, "gate_statistics": head["advantage"], "gate_statistics_ci95": head["advantage_ci95"],
                   "threshold": float(protocol_cfg["threshold"]),
                   "verdict": verdict(head["advantage_ci95"], float(protocol_cfg["threshold"])), "reset_gate": head,
                   "classes": {"slow": int((y == SLOW).sum()), "success": int((y == SUCCESS).sum()),
                               "excluded_drop_or_off_centre": int((~keep).sum()), "episodes": int(len(keep)),
                               "manifests": int(len(np.unique(a["group"])))},
                   "diagnostics": diagnostics(a, keep, protocol_cfg),
                   "note": "an association on archived episodes of one policy; it cannot establish capture causality", "simulation": "none"}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return summary


__all__ = ["STUDY", "auc", "balanced_accuracy", "bootstrap_manifests", "build_manifests", "diagnostics", "fit_logistic", "gate",
           "held_out_probabilities", "labels", "load_archive", "make_controller", "prefix_features", "reset_features", "run", "verdict"]
