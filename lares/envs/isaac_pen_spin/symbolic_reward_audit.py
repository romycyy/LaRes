"""R-104: does a late-tail score rank sustained success better than accumulated dense rolling? Archive only, no Isaac.

Reads R-103's fresh evaluation cells (per-step centre distance and axial rate, length, completion, sustained success)
and compares two reconstructible scores of an episode, both **without** the residual penalty (the cells stored no
observations or actions, so neither is R-103's training return)::

    g_t   = exp(-(d_t / d0)^2) * min(|w_t| / w0, 1)
    dense = sum of g_t over in-play steps                                    - P [the episode ended in a fall]
    tail  = H * mean of the k smallest g_t over the last ``window`` scheduled steps (unreached = 0)  - P [fall]

Concordance of a score: over every unordered policy pair of a block and every case on which exactly one of the two
policies sustains, 1 if the score is strictly higher for the sustaining policy, 0.5 on a tie, 0 otherwise; pooled
within block, block values averaged equally. The registered statistic is ``concordance(tail) - concordance(dense)``
with a case-clustered bootstrap (a resampled case brings all its policy comparisons).
"""

from __future__ import annotations

import itertools
import json
import os

import numpy as np

from lares.envs.isaac_pen_spin.protocols import load_cell

STUDY = "symbolic_reward_alignment_audit"
REQUIRED = ("center_dist", "spin_rate", "length")
SCORES = ("dense", "tail")


def smooth_term(center_dist, spin_rate, length, cfg: dict) -> np.ndarray:
    """``g`` as ``(T, cases)``, zero on every step a case did not reach."""
    d, w = np.asarray(center_dist, dtype=float), np.asarray(spin_rate, dtype=float)
    g = np.exp(-(d / float(cfg["center_scale"])) ** 2) * np.minimum(np.abs(w) / float(cfg["spin_cap"]), 1.0)
    reached = np.arange(g.shape[0])[:, None] < np.asarray(length)[None, :]
    return np.where(reached, np.nan_to_num(g), 0.0)


def episode_scores(trace: dict, rows: list[dict], cfg: dict) -> dict:
    """``{"dense", "tail", "sustained", "length"}`` per case of one cell."""
    missing = [k for k in REQUIRED if k not in trace]
    if missing or any("completed" not in r or "sustained" not in r for r in rows):
        raise KeyError(f"archived cell lacks {missing or ['completed/sustained rows']}")
    H, win, k = int(cfg["horizon"]), int(cfg["tail_window"]), int(cfg["tail_smallest"])
    length = np.asarray(trace["length"], dtype=int)
    g = smooth_term(trace["center_dist"], trace["spin_rate"], length, cfg)
    if g.shape[0] < H:
        g = np.concatenate([g, np.zeros((H - g.shape[0], g.shape[1]))])
    fell = np.array([not r["completed"] for r in rows])
    pen = float(cfg["fall_penalty"]) * fell
    tail = H * np.sort(g[H - win:H], axis=0)[:k].mean(axis=0)
    return {"dense": g.sum(axis=0) - pen, "tail": tail - pen, "length": length,
            "sustained": np.array([bool(r["sustained"]) for r in rows]),
            "task_hash": [r["task_hash"] for r in rows]}


def case_tables(policies: dict, penalty_per_step: float) -> dict:
    """Per-case sums over the block's policy pairs: discordant comparisons, concordant ones per score, and the
    comparisons whose score margin is within the two episodes' omitted-penalty bounds."""
    names = sorted(policies)
    n = len(policies[names[0]]["sustained"])
    for a in names:
        if policies[a]["task_hash"] != policies[names[0]]["task_hash"]:
            raise ValueError("policies of a block are not paired case for case")
    out = {"den": np.zeros(n), **{s: np.zeros(n) for s in SCORES}, **{f"{s}_vulnerable": np.zeros(n) for s in SCORES}}
    for a, b in itertools.combinations(names, 2):
        sa, sb = policies[a]["sustained"], policies[b]["sustained"]
        disc = sa != sb
        sign = np.where(sa, 1.0, -1.0)                      # +1 when a is the sustaining policy
        bound = penalty_per_step * (policies[a]["length"] + policies[b]["length"])
        out["den"] += disc
        for s in SCORES:
            m = (policies[a][s] - policies[b][s]) * sign
            out[s] += disc * np.where(m > 0, 1.0, np.where(m == 0, 0.5, 0.0))
            out[f"{s}_vulnerable"] += disc * (np.abs(m) <= bound)
    return out


def concordance(tables: dict, index: dict | None = None) -> dict:
    """Block-averaged concordance per score and their difference; ``index[block]`` resamples cases."""
    per = {s: [] for s in SCORES}
    for b, t in sorted(tables.items()):
        i = slice(None) if index is None else index[b]
        den = t["den"][i].sum()
        for s in SCORES:
            per[s].append(t[s][i].sum() / den if den > 0 else np.nan)
    out = {s: float(np.mean(per[s])) for s in SCORES}
    out["difference"] = out["tail"] - out["dense"]
    out["per_block"] = {s: [float(x) for x in per[s]] for s in SCORES}
    return out


def bootstrap(tables: dict, resamples: int, seed: int) -> dict:
    rng = np.random.default_rng(int(seed))
    draws = {k: [] for k in (*SCORES, "difference")}
    sizes = {b: len(t["den"]) for b, t in tables.items()}
    for _ in range(int(resamples)):
        c = concordance(tables, {b: rng.integers(0, n, n) for b, n in sorted(sizes.items())})
        for k in draws:
            draws[k].append(c[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in draws.items()}


def sizing(policies_by_block: dict, effect: float, resamples: int, seed: int) -> dict:
    """Planning estimates per policy pair: discordance, paired cases for 80% power at ``effect`` (two-sided 0.05),
    and the sampling SD of a paired success difference at 64, 128 and 256 cases."""
    z = 1.959963984540054 + 0.8416212335729143
    rng = np.random.default_rng(int(seed) + 1)
    out = {}
    for b, policies in sorted(policies_by_block.items()):
        for a, c in itertools.combinations(sorted(policies), 2):
            d = policies[a]["sustained"].astype(float) - policies[c]["sustained"].astype(float)
            n = len(d)
            need = lambda x: float(z ** 2 * max(np.mean(np.abs(x)) - effect ** 2, 0.0) / effect ** 2)   # noqa: E731
            boots = [need(d[rng.integers(0, n, n)]) for _ in range(int(resamples))]
            out[f"b{b}:{a}_vs_{c}"] = {
                "discordance": float(np.mean(np.abs(d))), "difference": float(d.mean()),
                "pairs_for_80pct_power": need(d),
                "pairs_for_80pct_power_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                "sd_of_paired_difference_at": {str(m): float(d.std(ddof=1) / np.sqrt(m)) for m in (64, 128, 256)}}
    return out


def load_policies(protocol_cfg: dict) -> dict:
    """``{block: {policy: scores}}`` for the reference and every distinct finalist that has its own fresh cell."""
    src = str(protocol_cfg["source_log_dir"])
    out = {}
    for b in protocol_cfg["blocks"]:
        out[int(b)] = {}
        for name in ("reference", "draws", "ars", "ppo"):
            cell = os.path.join(src, "fresh", f"b{int(b)}", f"{name}_cell.json")
            if os.path.exists(cell):                         # a finalist identical to the reference reused its cell
                trace, rows = load_cell(cell)
                out[int(b)][name] = episode_scores(trace, rows, protocol_cfg)
    return out


def build_manifests(protocol_cfg: dict) -> list[str]:
    return []


def make_controller(name: str, protocol_cfg: dict, out_dir: str | None = None):
    raise SystemExit("the alignment audit reads archives only; it has no controller and runs no simulation")


def verdict(interval: list, threshold: float) -> str:
    if interval[0] >= threshold:
        return "supported"
    return "refuted" if interval[1] < threshold else "inconclusive"


def run(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    try:
        policies = load_policies(protocol_cfg)
    except KeyError as e:
        summary = {"study": STUDY, "verdict": "inconclusive", "missing": str(e), "simulation": "none launched"}
    else:
        pen = float(protocol_cfg["omitted_penalty_per_step"])
        tables = {b: case_tables(p, pen) for b, p in policies.items()}
        point = concordance(tables)
        ci = bootstrap(tables, int(protocol_cfg["resamples"]), int(protocol_cfg["seed"]))
        den = sum(float(t["den"].sum()) for t in tables.values())
        summary = {
            "study": STUDY, "gate_statistics": point["difference"], "gate_statistics_ci95": ci["difference"],
            "threshold": float(protocol_cfg["threshold"]),
            "verdict": verdict(ci["difference"], float(protocol_cfg["threshold"])),
            "concordance": {s: {"value": point[s], "ci95": ci[s], "per_block": point["per_block"][s]} for s in SCORES},
            "discordant_comparisons": {f"b{b}": int(t["den"].sum()) for b, t in sorted(tables.items())},
            "policies": {f"b{b}": sorted(p) for b, p in sorted(policies.items())},
            "omitted_penalty": {"bound_per_episode": pen * int(protocol_cfg["horizon"]),
                                "vulnerable_margin_fraction": {s: sum(float(t[f"{s}_vulnerable"].sum()) for t in tables.values()) / den
                                                               for s in SCORES}},
            "between_policy": {f"b{b}:{name}": {"success": float(p["sustained"].mean()), "mean_dense": float(p["dense"].mean()),
                                                "mean_tail": float(p["tail"].mean())}
                               for b, ps in sorted(policies.items()) for name, p in sorted(ps.items())},
            "sizing": sizing(policies, float(protocol_cfg["effect"]), int(protocol_cfg["resamples"]), int(protocol_cfg["seed"])),
            "note": "both scores omit the residual penalty and neither is R-103's training return; the sizing numbers are "
                    "planning estimates, not promotion budgets; seven selected policies, retrospective only",
            "simulation": "none",
        }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return summary


__all__ = ["STUDY", "bootstrap", "build_manifests", "case_tables", "concordance", "episode_scores", "load_policies",
           "make_controller", "run", "sizing", "smooth_term", "verdict"]
