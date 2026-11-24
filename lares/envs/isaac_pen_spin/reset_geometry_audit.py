"""R-135 (archive only): where in the expanded reset distribution does the frozen JOINT controller fail?

Initial states are not recorded in cells but are exactly reconstructable: a case's reset is its manifest placement
vector times the reset scale, written by ``runner._write_placements`` (position offset = rows 0-2 times the env's
position noise; rotation = Rx(0.25 pi r3) Ry(0.25 pi r4) of a pen lying along x). Offsets are in units of the position
noise, which only fixes a scale. Per case: axial displacement (along the tilted pen axis), transverse displacement
(perpendicular to it) and tilt (the rotation angle). Registered: sustained success in the highest transverse quartile
minus the lowest, standardized over the axial x tilt tertile strata the two quartiles share.

    python scripts/run_shadowhand_spin_stages.py --study reset_geometry_audit
"""

from __future__ import annotations

import glob
import json
import math
import os

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.eval.manifest import EvaluationManifest

STUDY = "reset_geometry_audit"


def geometry(vecs, scale: float) -> dict:
    """Axial and transverse displacement (position-noise units) and tilt (rad) of scaled placement vectors."""
    v = np.asarray(vecs, dtype=float) * float(scale)
    a, b = 0.25 * math.pi * v[:, 3], 0.25 * math.pi * v[:, 4]
    axis = np.stack([np.cos(b), np.sin(a) * np.sin(b), -np.cos(a) * np.sin(b)], axis=1)      # Rx(a) Ry(b) x
    off = v[:, 0:3]
    axial = (off * axis).sum(axis=1)
    return {"axial": np.abs(axial), "transverse": np.linalg.norm(off - axial[:, None] * axis, axis=1),
            "tilt": 2.0 * np.arccos(np.clip(np.abs(np.cos(a / 2.0) * np.cos(b / 2.0)), 0.0, 1.0))}


def rank_bins(x, k: int) -> np.ndarray:
    """``k`` equal-count bins by rank (ties by case order)."""
    order = np.argsort(np.asarray(x), kind="stable")
    out = np.empty(len(order), dtype=int)
    out[order] = (np.arange(len(order)) * k) // len(order)
    return out


def standardized(success, quart, strata, cfg: dict) -> dict:
    """Highest minus lowest quartile success over the strata both quartiles occupy, weighted by their pooled count."""
    from scipy import stats

    hi, lo = quart == quart.max(), quart == quart.min()
    rows, used_hi, used_lo = [], 0, 0
    for s in np.unique(strata):
        h, l = success[hi & (strata == s)], success[lo & (strata == s)]
        if len(h) and len(l):
            rows.append((len(h) + len(l), float(h.mean() - l.mean()), float(h.mean() * (1 - h.mean()) / len(h) + l.mean() * (1 - l.mean()) / len(l))))
            used_hi, used_lo = used_hi + len(h), used_lo + len(l)
    n_strata = len(np.unique(strata))
    coverage = min(used_hi / max(1, hi.sum()), used_lo / max(1, lo.sum()))
    out = {"shared_strata": len(rows), "strata": int(n_strata), "coverage": float(coverage), "high_cases": int(hi.sum()), "low_cases": int(lo.sum())}
    if len(rows) < int(cfg["min_shared_strata"]) or coverage < float(cfg["min_coverage"]):
        return {**out, "verdict": "inconclusive", "note": "inadequate common support"}
    w = np.array([r[0] for r in rows], dtype=float)
    w /= w.sum()
    est = float((w * np.array([r[1] for r in rows])).sum())
    se = math.sqrt(float((w ** 2 * np.array([r[2] for r in rows])).sum()))
    z = float(stats.norm.ppf(0.975))
    thr = -float(cfg["threshold"])
    lo_b, hi_b = est - z * se, est + z * se
    return {**out, "difference": est, "ci95": [lo_b, hi_b], "verdict": "supported" if hi_b <= thr else ("refuted" if lo_b > thr else "inconclusive")}


def audit(success, geo: dict, cfg: dict) -> dict:
    success = np.asarray(success, dtype=float)
    q = rank_bins(geo["transverse"], int(cfg["transverse_quantiles"]))
    strata = rank_bins(geo["axial"], int(cfg["axial_strata"])) * 10 + rank_bins(geo["tilt"], int(cfg["tilt_strata"]))
    by = {name: [float(success[rank_bins(geo[name], 4) == j].mean()) for j in range(4)] for name in ("transverse", "axial", "tilt")}
    return {"cases": int(len(success)), "success": float(success.mean()), "standardized": standardized(success, q, strata, cfg),
            "unstandardized_high_minus_low_transverse": by["transverse"][-1] - by["transverse"][0], "success_by_quartile": by,
            "quartile_edges": {name: [float(v) for v in np.quantile(geo[name], [0.25, 0.5, 0.75])] for name in geo}}


def load(source: dict) -> tuple:
    """``(success per case, placement vectors)`` of one archive's frozen-JOINT cells under the expanded scale."""
    cells = sorted(glob.glob(source["cells"]))
    if len(cells) != int(source["manifests"]):
        raise FileNotFoundError(f"{source['cells']}: expected {source['manifests']} cells, found {len(cells)}")
    ok, vecs = [], []
    for c in cells:
        with open(c, encoding="utf-8") as f:
            saved = json.load(f)
        if float((saved.get("condition") or {}).get("reset_scale", 1.0)) != float(source["reset_scale"]):
            raise KeyError(f"{c}: the cell did not run under reset scale {source['reset_scale']}")
        v = cases.resolve_placements(EvaluationManifest.load(saved["manifest_path"]))
        if len(v) != len(saved["episodes"]):
            raise KeyError(f"{c}: cases and placements do not match")
        ok += [bool(r["sustained"]) for r in saved["episodes"]]
        vecs.append(v)
    return np.array(ok), np.concatenate(vecs)


def build_manifests(protocol_cfg: dict) -> list[str]:
    return []


def make_controller(name: str, protocol_cfg: dict, out_dir: str | None = None):
    raise SystemExit("the reset-geometry audit reads archives only; it has no controller and runs no simulation")


def run(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    cfg = protocol_cfg
    os.makedirs(out_dir, exist_ok=True)
    summary = {"study": STUDY, "simulation": "none", "units": "displacements in units of the env's reset position noise; tilt in radians",
               "reconstruction": "placement vector x reset scale, as runner._write_placements writes it; not a recorded state"}
    try:
        res = {}
        for name, src in cfg["archives"].items():
            ok, vecs = load(src)
            res[name] = audit(ok, geometry(vecs, float(src["reset_scale"])), cfg)
        head = res[cfg["primary_archive"]]["standardized"]
        summary.update({"success_rate": head.get("difference"), "success_rate_ci95": head.get("ci95"), "verdict": head["verdict"],
                        "threshold": -float(cfg["threshold"]), "archives": res})
    except (FileNotFoundError, KeyError) as e:
        summary.update(verdict="inconclusive", missing=str(e))
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return summary


__all__ = ["STUDY", "audit", "build_manifests", "geometry", "load", "make_controller", "rank_bins", "run", "standardized"]
