"""Small-sample paired statistics, declared once.

Every stage report, promotion test and replicate summary uses the same
Student-t interval and two-sided p-value over paired differences. A standard
error of exactly zero yields a degenerate interval at the mean.
"""

from __future__ import annotations

import math

import numpy as np


def t_summary(values, level: float = 0.95) -> dict:
    """``{n, mean, se, low, high, p_value}`` for a sample of paired differences.

    ``se``, ``low``, ``high`` and ``p_value`` are ``None`` below two samples.
    """
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    out = {"n": n, "mean": float(arr.mean()) if n else None, "se": None, "low": None, "high": None,
           "p_value": None}
    if n < 2:
        return out
    from scipy import stats

    se = float(arr.std(ddof=1) / math.sqrt(n))
    out["se"] = se
    if se == 0.0:
        out.update(low=out["mean"], high=out["mean"], p_value=0.0 if out["mean"] != 0.0 else 1.0)
        return out
    half = float(stats.t.ppf(0.5 + level / 2.0, n - 1) * se)
    out.update(low=out["mean"] - half, high=out["mean"] + half,
               p_value=float(2.0 * stats.t.sf(abs(out["mean"] / se), n - 1)))
    return out


def block_contrast(values) -> dict:
    """``{"per_block": [...], **t_summary}`` for a per-block paired contrast.

    Below two blocks only ``n`` is reported beside the values: a one-block study has no interval, and the
    recorded ``summary.json`` files of completed studies carry exactly this shape.
    """
    d = [float(v) for v in values]
    return {"per_block": d, **(t_summary(d) if len(d) >= 2 else {"n": len(d)})}


def t_interval(values, level: float = 0.95) -> dict:
    """Mean, median, t interval and p-value: the stage-report shape."""
    arr = np.asarray(values, dtype=float)
    out = t_summary(arr, level)
    out["median"] = float(np.median(arr)) if arr.size else None
    out["level"] = level
    return out


def t_bounds(values, level: float = 0.95):
    """``[low, high]`` of the t interval, or ``None`` below two samples."""
    s = t_summary(values, level)
    return None if s["low"] is None else [s["low"], s["high"]]


def win_loss_tie(values) -> dict:
    arr = np.asarray(values, dtype=float)
    return {"wins": int((arr > 0).sum()), "losses": int((arr < 0).sum()), "ties": int((arr == 0).sum())}
