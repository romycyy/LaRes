"""R-071: does removing the capture curl replace sustained roll with direction reversals? Offline, from archived traces.

Frozen definitions. The signed axial rate is the env's ``spin_rate`` (rad/s). A direction is *held* when ``|spin|``
exceeds the hysteresis band with the same sign on ``persistence`` consecutive steps; a *reversal* is a change of held
direction. The rate is reversals per in-play second (60 Hz) over each pair's common pre-drop window (the shorter of the
two recorded lengths). Nothing is simulated: a missing record ends the analysis.
"""

from __future__ import annotations

import json
import os

import numpy as np

CONTROL_HZ = 60.0
PANELS = ("fresh", "promotion")
ARMS = ("parent", "zero_curl")


def held_directions(spin, hysteresis: float, persistence: int) -> np.ndarray:
    """Per step, the held direction (+1, -1) or 0 before any direction is held; a direction is taken after
    ``persistence`` consecutive steps beyond the band with one sign, and kept until the opposite one is taken."""
    spin = np.asarray(spin, dtype=np.float64)
    held = np.zeros(len(spin), dtype=np.int64)
    current, run_sign, run_len = 0, 0, 0
    for t, s in enumerate(spin):
        sign = 1 if s > hysteresis else (-1 if s < -hysteresis else 0)
        if sign != 0 and sign == run_sign:
            run_len += 1
        else:
            run_sign, run_len = sign, (1 if sign != 0 else 0)
        if sign != 0 and run_len >= persistence and sign != current:
            current = sign
        held[t] = current
    return held


def reversals(spin, hysteresis: float, persistence: int) -> int:
    h = held_directions(spin, hysteresis, persistence)
    nonzero = h[h != 0]
    return int(np.sum(nonzero[1:] != nonzero[:-1])) if nonzero.size > 1 else 0


def pair_rates(a: dict, b: dict, hysteresis: float, persistence: int) -> dict:
    """Reversal rates of two arms' traces on each pair's common pre-drop window."""
    n = len(a["length"])
    out = {"window_steps": [], "rate": {"a": [], "b": []}, "held_fraction": {"a": [], "b": []},
           "mean_abs_spin": {"a": [], "b": []}}
    for i in range(n):
        L = int(min(a["length"][i], b["length"][i]))
        out["window_steps"].append(L)
        for key, tr in (("a", a), ("b", b)):
            s = np.nan_to_num(np.asarray(tr["spin_rate"][:L, i], dtype=np.float64))
            secs = L / CONTROL_HZ
            out["rate"][key].append(reversals(s, hysteresis, persistence) / secs if secs > 0 else 0.0)
            h = held_directions(s, hysteresis, persistence)
            out["held_fraction"][key].append(float(np.mean(h != 0)) if L else 0.0)
            out["mean_abs_spin"][key].append(float(np.mean(np.abs(s))) if L else 0.0)
    return out


def analyse(run_dir: str, hysteresis: float = 1.0, persistence: int = 3) -> dict:
    from lares.eval.stats import t_summary, win_loss_tie

    out = {"definitions": {"hysteresis_rad_s": float(hysteresis), "persistence_steps": int(persistence),
                           "control_hz": CONTROL_HZ, "window": "common pre-drop window: min of the two recorded lengths",
                           "reversal": "change of held direction; a direction is held after persistence consecutive "
                                       "steps beyond the band with one sign"},
           "panels": {}, "source": run_dir}
    for panel in PANELS:
        traces, rows = {}, {}
        for arm in ARMS:
            path = os.path.join(run_dir, f"{arm}_{panel}.json")
            npz = path.replace(".json", ".npz")
            if not (os.path.exists(path) and os.path.exists(npz)):
                raise SystemExit(f"required record {path} / {npz} missing; the analysis collects nothing")
            traces[arm] = np.load(npz)
            with open(path, encoding="utf-8") as f:
                rows[arm] = json.load(f)["episodes"]
        if [r["task_hash"] for r in rows["parent"]] != [r["task_hash"] for r in rows["zero_curl"]]:
            raise SystemExit(f"{panel}: rows are not paired case for case")
        pr = pair_rates(traces["zero_curl"], traces["parent"], hysteresis, persistence)
        d = [x - y for x, y in zip(pr["rate"]["a"], pr["rate"]["b"])]
        out["panels"][panel] = {
            "pairs": len(d), "phase_transitions": t_summary(d), "win_loss_tie": win_loss_tie(d),
            "arms": {arm: {"reversals_per_second": float(np.mean(pr["rate"][k])),
                           "held_direction_fraction": float(np.mean(pr["held_fraction"][k])),
                           "mean_abs_spin_on_window": float(np.mean(pr["mean_abs_spin"][k])),
                           "sustained_successes": int(sum(r["sustained"] for r in rows[arm])),
                           "drops": int(sum(not r["completed"] for r in rows[arm]))}
                     for arm, k in (("zero_curl", "a"), ("parent", "b"))},
            "mean_window_steps": float(np.mean(pr["window_steps"]))}
    out["paired_success_difference"] = None
    out["phase_transitions"] = out["panels"]["fresh"]["phase_transitions"]["mean"]
    return out


__all__ = ["ARMS", "CONTROL_HZ", "PANELS", "analyse", "held_directions", "pair_rates", "reversals"]
