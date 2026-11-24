"""R-134 (archive only): does rolling improve around the end of recovery support more in R-128's weak lineage?

For each lineage's stage-B endpoint the frozen module's own parameters give its decision step and the end of its
support; the per-case trigger is reconstructed from the archived trace (``recovery_handback_audit``). Among triggered
cases still in play 30 steps after support ends, the statistic is the rolling-step fraction in the 30 steps after the
end of support minus the 30 before. Registered: the weak lineage's mean change minus the equal-weight mean of the
other lineages'. One lineage is one lineage: this describes it and says nothing about the recipe.

    python scripts/run_shadowhand_spin_stages.py --study stage_b_release_audit
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from lares.envs.isaac_pen_spin import recovery_handback_audit as ha
from lares.envs.isaac_pen_spin import recovery_module as rm
from lares.envs.isaac_pen_spin.protocols import load_cell

STUDY = "stage_b_release_audit"


def timeline(params: dict, cfg: dict) -> dict:
    T = int(round(params["decision_step"]))
    support = int(math.ceil(params["ramp_in"] + params["ramp_out"]))
    return {"decision_step": T, "threshold": float(params["threshold"]), "centre_tolerance": float(cfg["centre_tolerance"]),
            "rate_window": rm.WINDOW, "support_steps": support, "support_end": T + support - 1}


def release_change(trace: dict, tl: dict, window: int) -> tuple:
    """``(eligible mask, post-minus-pre rolling fraction per case)``; nothing depends on post-release rolling."""
    pred, length = np.asarray(trace["pred"], dtype=bool), np.asarray(trace["length"])
    b = tl["support_end"]
    trig = ha.reconstruct_trigger(np.asarray(trace["spin_rate"]), np.asarray(trace["center_dist"]), length, tl)
    if b + window >= pred.shape[0] - 1 or b - window + 1 < 0:
        return np.zeros_like(trig), np.zeros(len(trig))
    eligible = trig & (length > b + window)
    change = pred[b + 1:b + 1 + window].mean(axis=0) - pred[b - window + 1:b + 1].mean(axis=0)
    return eligible, change


def contrast(per_lineage: dict, weak: int) -> dict:
    """The weak lineage's mean change minus the equal-weight mean of the others', cases treated as independent."""
    from scipy import stats

    means = {k: float(np.mean(v)) for k, v in per_lineage.items()}
    var = {k: float(np.var(v, ddof=1) / len(v)) for k, v in per_lineage.items()}
    others = [k for k in per_lineage if k != weak]
    est = means[weak] - float(np.mean([means[k] for k in others]))
    se = math.sqrt(var[weak] + sum(var[k] for k in others) / len(others) ** 2)
    z = float(stats.norm.ppf(0.975))
    return {"estimate": est, "ci95": [est - z * se, est + z * se], "se": se, "lineage_mean_change": means}


def build_manifests(protocol_cfg: dict) -> list[str]:
    return []


def make_controller(name: str, protocol_cfg: dict, out_dir: str | None = None):
    raise SystemExit("the release audit reads archives only; it has no controller and runs no simulation")


def run(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    from lares.eval.stats import t_summary

    cfg = protocol_cfg
    os.makedirs(out_dir, exist_ok=True)
    src, weak, window = cfg["source_log_dir"], int(cfg["weak_lineage"]), int(cfg["window_steps"])
    summary = {"study": STUDY, "simulation": "none", "window_steps": window, "weak_lineage": weak}
    try:
        seeds = sorted(d for d in os.listdir(os.path.join(src, "confirmation")) if d.startswith("s"))
        if len(seeds) != int(cfg["manifests"]):
            raise FileNotFoundError(f"expected {cfg['manifests']} archived manifests, found {len(seeds)}")
        per, by_manifest, lines = {}, {s: {} for s in seeds}, {}
        for k in range(int(cfg["lineages"])):
            params = rm._load(os.path.join(src, f"lineage_{k}", "endpoints.json"))["endpoints"]["B"]["params"]
            tl = timeline(params, cfg)
            vals, n_cases, n_trig = [], 0, 0
            for s in seeds:
                trace, rows = load_cell(os.path.join(src, "confirmation", s, f"L{k}_B_cell.json"))
                elig, change = release_change(trace, tl, window)
                vals.append(change[elig])
                by_manifest[s][k] = float(change[elig].mean()) if elig.any() else None
                n_cases += len(rows)
                n_trig += int(ha.reconstruct_trigger(np.asarray(trace["spin_rate"]), np.asarray(trace["center_dist"]), np.asarray(trace["length"]), tl).sum())
            per[k] = np.concatenate(vals)
            lines[f"L{k}"] = {"timeline": tl, "stage_b_params": params, "cases": n_cases, "reconstructed_triggers": n_trig, "eligible": int(len(per[k]))}
        if min(len(v) for v in per.values()) < 2:
            raise FileNotFoundError("a lineage has fewer than two eligible cases (its support ends too late for the window)")
        c = contrast(per, weak)
        others = [k for k in per if k != weak]
        man = [v[weak] - float(np.mean([v[k] for k in others])) for v in by_manifest.values() if all(v[k] is not None for k in per)]
        cl = t_summary(man)
        thr = float(cfg["threshold"])
        verdict = "supported" if c["ci95"][0] >= thr else ("refuted" if c["ci95"][1] < thr else "inconclusive")
        summary.update({"gate_statistics": c["estimate"], "gate_statistics_ci95": c["ci95"], "threshold": thr, "verdict": verdict,
                        "spans_zero": bool(c["ci95"][0] <= 0 <= c["ci95"][1]), "lineage_mean_change": {f"L{k}": v for k, v in c["lineage_mean_change"].items()},
                        "manifest_clustered": {"mean": cl["mean"], "ci95": [cl["low"], cl["high"]], "manifests": cl["n"]}, "lineages": lines,
                        "caveat": "triggers are reconstructed from traces; one weak lineage; an association around the end of support, not a cause"})
    except (FileNotFoundError, KeyError) as e:
        summary.update(verdict="inconclusive", missing=str(e))
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return summary


__all__ = ["STUDY", "build_manifests", "contrast", "make_controller", "release_change", "run", "timeline"]
