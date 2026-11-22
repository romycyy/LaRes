"""R-097: reproduce the fixed-stroke incumbent, then gate contact-frame decoder control *during rolling*.

R-096 qualified the decoder for small pulses on a pen at rest; in R-094's rollouts it was far outside that regime
(D-068). This gate asks the question in the regime a residual repair would use: the frozen D-058 incumbent runs
unchanged, and during a mid-episode window a decoded joint increment is **added** to its action.

Three registry names serve this module (``protocol_cfg["study"]`` says which), run in this order:

* ``rolling_baseline`` (S1): the incumbent alone on the gate manifest, one cell. Reports its sustained success with
  a Wilson interval (``reproduces`` = the interval overlaps the recorded band), the failure split, and roll measured
  along each case's own direction. A stop point: nothing else runs unless it reproduces.
* ``rolling_decoder_calibration``: the declared settings on the calibration manifest, two cells each (``+`` and
  ``-``); case ``i`` pulses contrast ``i mod 15``. Chooses the frozen constants by :func:`select_setting`, which never
  reads sustained success. Never gate evidence.
* ``rolling_decoder_gate`` (S2): the zero-request residual and the 30 directional cells on the gate manifest (the
  incumbent-alone cell is S1's, read from ``baseline_log_dir``, not rerun). Refuses to run unless S1 reproduced and
  the ``frozen`` block matches ``frozen_sha256``.

Residual (trusted code). For control steps ``pulse_start <= t < pulse_stop`` the commanded fingertip's request
``u`` (local ``[normal, circumferential, axial]``, m/s) is decoded by ``contact_control.decode`` into joint speeds of
that finger's chain *as an increment*: the decoder's target is ``R u`` alone (no pen following; the incumbent's own
motion is the baseline the increment rides on). The incumbent's action is ``q + 0.8 (target - q)``, so a per-step
offset would give a position step, not a velocity; the offset therefore integrates, ``delta += dt * qdot`` with each
joint's ``|delta| <= cap``, and the action is ``clamp(incumbent + norm(delta + lookahead * qdot), -1, 1)``. Outside
the window ``delta`` is zero and the action is the incumbent's, bit for bit.

Three motions, each as the paired contrast ``(x_pos - x_neg) / 2`` of the same case in the ``+`` and ``-`` cells
(the cells share the pre-pulse prefix, so the incumbent's own stroke cancels): **requested** ``u``; **feasible**, the
decoded joint speed after the speed, limit and cap bounds mapped back through the Jacobian; **achieved**, the
observed link-origin fingertip velocity minus the pen's rigid velocity at the nearest surface point, averaged over
the window and projected on the frame at ``pulse_start``.

Eligibility reads only the common pre-pulse prefix: the pen has not fallen by ``pulse_start`` and the median
``|axial rate|`` over ``roll_window`` is at least ``min_abs_axial_rate``. An eligible pair stays in its contrast
whatever the pulse does afterwards (a pulse-induced drop is a measured outcome, reported, never an exclusion).
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin import contact_control as cc
from lares.envs.isaac_pen_spin import contact_gate as cg
from lares.envs.isaac_pen_spin.protocols import arm_summary, failure_split, freeze_json, load_cell, study_cell

STAGE = 3
BASELINE, CALIBRATION, GATE = "rolling_baseline", "rolling_decoder_calibration", "rolling_decoder_gate"
LABELS = {CALIBRATION: "rolling_decoder_calibration", GATE: "rolling_decoder_gate"}
CONTRASTS = tuple((k, a) for k in range(len(cc.FINGERTIPS)) for a in range(3))
THRESHOLDS = ("expected_contrasts", "min_eligible_pairs", "paired_sign_agreement_min",
              "contrast_sign_agreement_min", "paired_gain_median_min", "paired_gain_median_max",
              "feasible_ratio_median_min", "bound_active_fraction_max", "tracking_error_median_max",
              "jacobian_rel_error_median_max", "retention_diff_min")
BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED = 2000, 0


def contrast_name(k: int, a: int) -> str:
    return f"{cc.FINGERTIPS[k]}_{cc.LOCAL_AXES[a]}"


def calibration_contrasts(n: int) -> list:
    return [CONTRASTS[i % len(CONTRASTS)] for i in range(n)]


# ---------------------------------------------------------------------------------------------------------------------
#  Controller
# ---------------------------------------------------------------------------------------------------------------------


class RollingPulseController:
    """The incumbent plus a decoded, integrating joint increment inside the pulse window (module docstring).

    ``directions``: per case ``(fingertip, axis, sign)`` or ``None`` for the zero-request residual, which runs the
    same decoder with nothing commanded and must return the incumbent's action exactly. ``inner`` is the incumbent
    (injected by tests; built from ``asset_dir`` otherwise).
    """

    def __init__(self, name: str, frozen: dict, directions=None, asset_dir: str | None = None, inner=None):
        if inner is None:
            from lares.envs.isaac_pen_spin.coordinate_repair import incumbent_controller

            inner = incumbent_controller(asset_dir)
        self.inner, self.name = inner, name
        self.decoder, self.ctl, self.win = cg.settings_of(frozen)
        self.directions = directions
        self.params = {"frozen_sha256": cg.frozen_sha256(frozen), **frozen, "parent": getattr(inner, "params", None),
                       "execution": "clamp(incumbent + norm(delta + lookahead * qdot), -1, 1) inside the pulse window"}
        self._rec = None

    def reset(self, ctx):
        import torch

        if hasattr(self.inner, "reset"):
            self.inner.reset(ctx)
        names, act = ctx.joint_names, list(ctx.actuated_dof_indices)
        ctrl = [i for i in act if self.ctl.get("control_wrist") or "WRJ" not in names[i]]
        dev = ctx.device
        C = torch.zeros(len(names), len(ctrl), device=dev)
        for col, j in enumerate(ctrl):
            C[j, col] = 1.0
        for j, nm in enumerate(names):                       # a passive distal joint moves with its finger's J1
            if j not in act and nm.endswith("J0"):
                partner = names.index(nm[:-1] + "1")
                if partner in ctrl:
                    C[j, ctrl.index(partner)] = float(self.ctl["coupling"])
        self.C, self.ctrl = C, torch.tensor(ctrl, device=dev)
        self.ctrl_in_action = torch.tensor([act.index(j) for j in ctrl], device=dev)
        self.lo, self.hi = ctx.lower_all[:, self.ctrl], ctx.upper_all[:, self.ctrl]
        self.chain = torch.stack([torch.tensor([tag in names[j] or "WRJ" in names[j] for j in ctrl], device=dev)
                                  for tag in cg.FINGER_TAGS])
        b, k = ctx.num_envs, len(cc.FINGERTIPS)
        self.cmd = torch.zeros(b, k, 3, device=dev)
        self.commanded = torch.zeros(b, k, dtype=torch.bool, device=dev)
        self.tip = torch.zeros(b, dtype=torch.long, device=dev)
        for i, d in enumerate(self.directions or []):
            if d is not None:
                kk, a, s = d
                self.cmd[i, kk, a] = float(s) * float(self.win["pulse_speed"])
                self.commanded[i, kk] = True
                self.tip[i] = kk
        self.delta = torch.zeros(b, len(ctrl), device=dev)
        self.frames = None
        self._identity = 0.0
        self._rec = {"bound_active": [], "feasible": [], "jacobian": [], "jacobian_step": [], "distance": []}
        self._meta = {"lower_all": ctx.lower_all[0].cpu().numpy(), "upper_all": ctx.upper_all[0].cpu().numpy()}

    def __call__(self, obs, t, ctx):
        import torch

        g = self.inner(obs, t, ctx).to(ctx.device)
        p0, p1 = int(self.win["pulse_start"]), int(self.win["pulse_stop"])
        if not p0 <= t < p1:
            self.delta.zero_()
            self.frames = None
            return g
        s = cc.state_from_obs(obs)
        self.frames = frames = cc.capsule_frames(s["pen_pos"], s["pen_quat"], s["tip_pos"], prev=self.frames)
        J_full = ctx.fingertip_jacobians()
        J = cc.effective_jacobian(J_full, self.C)
        zero = torch.zeros_like(s["pen_linvel"])
        q = ctx.joint_pos()[:, self.ctrl]
        qdot, diag = cc.decode(J, frames, self.cmd, s["pen_pos"], zero, zero, q, self.lo, self.hi, self.decoder,
                               active=self.commanded)
        dt, cap = self.decoder.control_dt, float(self.ctl["cap"])
        want = self.delta + dt * qdot
        new = want.clamp(-cap, cap)
        cap_active = (new - want).abs() > 1e-9
        qdot_eff = (new - self.delta) / dt
        self.delta = new
        incr = 2.0 * (self.delta + self.decoder.lookahead * qdot_eff) / (self.hi - self.lo)
        raw = g.clone()
        raw[:, self.ctrl_in_action] = raw[:, self.ctrl_in_action] + incr.to(raw.dtype)
        action = raw.clamp(-1.0, 1.0)
        clamp_active = (action != raw)[:, self.ctrl_in_action]
        self._identity = max(self._identity, float((action - g).abs().max()) if self.directions is None else 0.0)

        if t < int(self.win["record_stop"]):
            idx = torch.arange(obs.shape[0], device=J.device)
            feasible = cc.world_to_local(torch.einsum("bkin,bn->bki", J, qdot_eff.to(J.dtype)), frames)[idx, self.tip]
            bound = diag["vel_bound_active"] | diag["limit_active"] | cap_active | clamp_active
            self._rec["bound_active"].append(bound.cpu().numpy())
            self._rec["feasible"].append(feasible.cpu().numpy().astype(np.float32))
            self._rec["distance"].append(frames["distance"].cpu().numpy().astype(np.float32))
            if t in self.win["jacobian_steps"]:
                self._rec["jacobian"].append(J_full.cpu().numpy().astype(np.float32))
                self._rec["jacobian_step"].append(t)
        return action

    def extra_trace(self) -> dict:
        r = self._rec
        out = {"chain": self.chain.cpu().numpy(), "lower_all": self._meta["lower_all"],
               "upper_all": self._meta["upper_all"], "identity_max_abs": np.float64(self._identity),
               "jacobian_step": np.array(r["jacobian_step"], dtype=np.int64)}
        for key in ("bound_active", "feasible", "distance"):
            if r[key]:
                out[key] = np.stack(r[key])
        if r["jacobian"]:
            out["jacobian"] = np.stack(r["jacobian"])
        return out


# ---------------------------------------------------------------------------------------------------------------------
#  Offline statistics
# ---------------------------------------------------------------------------------------------------------------------


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (float(centre - half), float(centre + half))


def roll_direction(spin, length, window) -> np.ndarray:
    """Per case ``+1`` / ``-1`` / ``0``: the sign of the median axial rate over ``window`` (``spin`` is ``(T, C)``),
    ``0`` when the case ended before the window closed. Causal: nothing after the window is read."""
    w0, w1 = int(window[0]), int(window[1])
    med = np.nanmedian(np.where(np.isfinite(spin[w0:w1]), spin[w0:w1], np.nan), axis=0) if spin.shape[0] >= w1 else \
        np.zeros(spin.shape[1])
    med = np.nan_to_num(med)
    return np.where(np.asarray(length) >= w1, np.sign(med), 0.0), np.abs(med)


def roll_metrics(spin, length, window, dt: float, smooth: int = 15) -> list:
    """Per case: roll direction and pre-window ``|rate|``, direction-aligned integrated roll after the window (rad),
    the integral of ``|axial rate|`` over the episode (rad) and the sign changes of the smoothed rate."""
    direction, rate = roll_direction(spin, length, window)
    out = []
    for i in range(spin.shape[1]):
        L = int(min(length[i], spin.shape[0]))
        s = np.nan_to_num(spin[:L, i].astype(np.float64))
        sm = np.convolve(s, np.ones(smooth) / smooth, mode="valid") if L >= smooth else s
        sign = np.sign(sm[np.abs(sm) > 1e-6])
        out.append({"direction": float(direction[i]), "pre_window_abs_rate": float(rate[i]),
                    "aligned_roll_after_window": float(direction[i] * s[int(window[1]):].sum() * dt),
                    "abs_roll_integral": float(np.abs(s).sum() * dt),
                    "reversals": int((sign[1:] != sign[:-1]).sum()) if len(sign) > 1 else 0})
    return out


def baseline_summary(cell: str, protocol_cfg: dict) -> dict:
    """S1: the incumbent's success with a Wilson interval, its failure split and its roll, by outcome."""
    from lares.envs.isaac_pen_spin.channel_lesions import tail_failure_split

    trace, rows = load_cell(cell)
    n, k = len(rows), int(sum(r["sustained"] for r in rows))
    low, high = wilson(k, n)
    band = [float(x) for x in protocol_cfg["reproduction_band"]]
    tol = cases.STAGES[STAGE].success_center_tolerance
    roll = roll_metrics(trace["spin_rate"], trace["length"], protocol_cfg["eligibility"]["roll_window"],
                        float(protocol_cfg["control_dt"]))

    def outcome(r):
        if r["sustained"]:
            return "success"
        if not r["completed"]:
            return "drop"
        return "off_centre" if r["final_center_dist"] is not None and r["final_center_dist"] >= tol else "slow_spin"

    groups = {}
    for r, m in zip(rows, roll):
        groups.setdefault(outcome(r), []).append({**m, "mean_abs_spin": r["mean_abs_spin"],
                                                 "tail_hit_fraction": r["tail_hit_fraction"]})
    by_outcome = {o: {"n": len(g), **{key: float(np.median([x[key] for x in g if x[key] is not None]))
                                      for key in ("pre_window_abs_rate", "aligned_roll_after_window",
                                                  "abs_roll_integral", "reversals", "mean_abs_spin",
                                                  "tail_hit_fraction")}} for o, g in sorted(groups.items())}
    eligible = sum(1 for m in roll if m["direction"] != 0.0
                   and m["pre_window_abs_rate"] >= float(protocol_cfg["eligibility"]["min_abs_axial_rate"]))
    return {"study": BASELINE, "n": n, "successes": k, "success": k / n, "success_wilson95": [low, high],
            "reproduction_band": band, "reproduces": bool(low <= band[1] and high >= band[0]),
            "rule": "the Wilson 95% interval of the sustained-success estimate overlaps the recorded band",
            **arm_summary(rows), **failure_split(rows), **tail_failure_split(rows),
            "roll_by_outcome": by_outcome, "roll_directions": {"positive": sum(m["direction"] > 0 for m in roll),
                                                               "negative": sum(m["direction"] < 0 for m in roll),
                                                               "ended_before_window": sum(m["direction"] == 0 for m in roll)},
            "rolling_eligible_cases": int(eligible), "manifest": cases.manifest_record(cell_manifest(cell)),
            "episodes": n}


def cell_manifest(cell: str) -> str:
    with open(cell, encoding="utf-8") as f:
        return json.load(f)["manifest_path"]


def in_play_prefix(obs, length, pulse_start: int) -> np.ndarray:
    """Observations ``0 .. pulse_start`` with every step at or after a case's end zeroed: what a case records after
    it ended is the next episode's auto-reset, which differs between processes and is not part of the pairing."""
    out = np.nan_to_num(np.asarray(obs[:, :pulse_start + 1], dtype=np.float64))
    ended = np.arange(out.shape[1])[None, :] >= np.asarray(length)[:, None]
    out[ended] = 0.0
    return out


def cell_motion(cell: str, tips, windows: dict) -> dict:
    """What one pulse cell measured, per case: the mean achieved relative fingertip velocity (world) and the mean
    feasible velocity (local) over the in-play pulse steps, the frame at ``pulse_start``, bound activity on the
    commanded chain, and the pre-pulse quantities eligibility reads."""
    import torch

    trace, rows = load_cell(cell)
    extra = dict(np.load(cell.replace(".json", ".extra.npz")))
    obs, length = trace["obs"], np.asarray(trace["length"])
    C, T = obs.shape[:2]
    p0, p1 = int(windows["pulse_start"]), int(windows["pulse_stop"])
    tips = np.asarray(tips)
    idx = np.arange(C)
    achieved, feasible, bound_frac = np.zeros((C, 3)), np.zeros((C, 3)), np.full(C, np.nan)
    R_first = np.tile(np.eye(3), (C, 1, 1))
    steps = np.zeros(C, dtype=np.int64)
    stop = min(p1, T - 1)
    if stop > p0:
        flat = torch.as_tensor(np.nan_to_num(obs[:, p0:stop + 1]).reshape(-1, obs.shape[-1]), dtype=torch.float64)
        s = cc.state_from_obs(flat)
        fr = cc.capsule_frames(s["pen_pos"], s["pen_quat"], s["tip_pos"])
        rel = cc.relative_world_velocity(cc.tip_origin_linvel(s), fr["point"], s["pen_pos"], s["pen_linvel"],
                                         s["pen_angvel"]).reshape(C, stop + 1 - p0, len(cc.FINGERTIPS), 3).numpy()
        R = fr["R"].reshape(C, stop + 1 - p0, len(cc.FINGERTIPS), 3, 3).numpy()
        R_first = R[idx, 0, tips]
        feas = extra["feasible"][:p1 - p0] if "feasible" in extra else np.zeros((0, C, 3))   # (steps, C, 3)
        bound = extra["bound_active"][:p1 - p0] if "bound_active" in extra else np.zeros((0, C, 1), dtype=bool)
        chain = extra["chain"][tips]
        for i in range(C):
            # observation t (p0 < t <= p1) is the state after commanded step t - 1; it exists while t < length
            last = int(min(stop, length[i] - 1))
            n_i = max(0, last - p0)
            steps[i] = n_i
            if n_i:
                achieved[i] = np.nan_to_num(rel[i, 1:n_i + 1, tips[i]]).mean(0)
                feasible[i] = np.nan_to_num(feas[:n_i, i]).mean(0)
                bound_frac[i] = float((bound[:n_i, i] & chain[i][None]).any(-1).mean())
    return {"achieved_world": achieved, "feasible_local": feasible, "R_first": R_first, "bound_fraction": bound_frac,
            "pulse_steps": steps, "length": length, "completed": np.asarray(trace["completed"], dtype=bool),
            "spin_rate": trace["spin_rate"], "obs_prefix": in_play_prefix(obs, length, p0), "rows": rows,
            "jacobian_errors": cg.jacobian_errors(obs, extra, length, windows) if "jacobian" in extra else []}


def paired_motion(pos: dict, neg: dict, axes, windows: dict, eligibility: dict) -> dict:
    """The paired contrast of one ``+`` and one ``-`` cell, per case (module docstring). ``axes``: per case."""
    p0, speed = int(windows["pulse_start"]), float(windows["pulse_speed"])
    axes = np.asarray(axes)
    idx = np.arange(len(axes))
    _, rate = roll_direction(pos["spin_rate"], pos["length"], eligibility["roll_window"])
    eligible = (pos["length"] > p0) & (neg["length"] > p0) & (rate >= float(eligibility["min_abs_axial_rate"]))
    achieved = np.einsum("cji,cj->ci", pos["R_first"], 0.5 * (pos["achieved_world"] - neg["achieved_world"]))
    feasible = 0.5 * (pos["feasible_local"] - neg["feasible_local"])
    d = achieved[idx, axes] / speed
    ratio = feasible[idx, axes] / speed
    fn = np.linalg.norm(feasible, axis=1)
    error = np.where(fn > 1e-9, np.linalg.norm(achieved - feasible, axis=1) / np.maximum(fn, 1e-9), np.nan)
    with np.errstate(invalid="ignore"):
        bound = np.nanmean(np.stack([pos["bound_fraction"], neg["bound_fraction"]]), axis=0)
    prefix = float(np.abs(pos["obs_prefix"] - neg["obs_prefix"]).max()) \
        if pos["obs_prefix"].shape == neg["obs_prefix"].shape else float("inf")
    return {"eligible": eligible, "d": d, "feasible_ratio": ratio, "tracking_error": error, "bound_fraction": bound,
            "achieved_local": achieved, "feasible_local": feasible, "prefix_max_abs_diff": prefix,
            "pulse_drops": (~pos["completed"] & (pos["length"] <= int(windows["pulse_stop"])) & eligible).astype(int)
            + (~neg["completed"] & (neg["length"] <= int(windows["pulse_stop"])) & eligible).astype(int)}


def _median(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else None


def pooled(pairs: list) -> dict:
    """Pooled statistics over ``(case index, paired_motion row)`` records of eligible pairs."""
    d = np.array([p["d"] for p in pairs], dtype=np.float64)
    return {"eligible_pairs": int(len(d)),
            "sign_agreement": float((d > 0).mean()) if len(d) else None,
            "gain_median": _median(d),
            "feasible_ratio_median": _median([p["feasible_ratio"] for p in pairs]),
            "tracking_error_median": _median([p["tracking_error"] for p in pairs]),
            "bound_active_fraction": (float(np.nanmean([p["bound_fraction"] for p in pairs]))
                                      if len(d) and np.isfinite([p["bound_fraction"] for p in pairs]).any() else None)}


def clustered_interval(pairs: list, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> list:
    """95% percentile interval of the pooled sign agreement, resampling *cases* (the contrasts share cases)."""
    by_case = {}
    for p in pairs:
        by_case.setdefault(p["case"], []).append(p["d"] > 0)
    keys = sorted(by_case)
    if not keys:
        return [None, None]
    hits = np.array([sum(by_case[k]) for k in keys], dtype=np.float64)
    counts = np.array([len(by_case[k]) for k in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(keys), size=(resamples, len(keys)))
    stat = hits[draw].sum(1) / counts[draw].sum(1)
    return [float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))]


def pair_records(pm: dict, tips, axes) -> list:
    return [{"case": int(i), "contrast": contrast_name(int(tips[i]), int(axes[i])), "d": float(pm["d"][i]),
             "feasible_ratio": float(pm["feasible_ratio"][i]), "tracking_error": float(pm["tracking_error"][i]),
             "bound_fraction": float(pm["bound_fraction"][i]), "pulse_drops": int(pm["pulse_drops"][i])}
            for i in np.flatnonzero(pm["eligible"])]


def _check(value, ok, rule: str) -> dict:
    return {"value": value, "ok": None if value is None else bool(ok), "rule": rule}


def gate_verdict(pairs: list, jacobian_errors, zero_timeout_rate, parent_timeout_rate, identity_max_abs,
                 prefix_max_abs_diff, thresholds: dict) -> dict:
    """``pass`` / ``fail`` / ``inconclusive``. A failed Jacobian audit, retention check or zero-request identity
    fails whatever else; otherwise a prefix that differs between cells (the pairing is void) or a contrast under the
    pair minimum is inconclusive; otherwise every tracking check must hold."""
    missing = set(THRESHOLDS) - set(thresholds)
    if missing:
        raise ValueError(f"thresholds lack {sorted(missing)}")
    th, checks = thresholds, {}
    errs = np.asarray(jacobian_errors, dtype=np.float64).ravel()
    jac = float(np.median(errs)) if len(errs) and np.isfinite(errs).all() else None
    checks["jacobian_rel_error_median"] = _check(jac, jac is not None and jac < th["jacobian_rel_error_median_max"],
                                                 f"< {th['jacobian_rel_error_median_max']}")
    diff = (float(zero_timeout_rate) - float(parent_timeout_rate)
            if zero_timeout_rate is not None and parent_timeout_rate is not None else None)
    checks["retention_diff"] = _check(diff, diff is not None and diff >= th["retention_diff_min"] - 1e-9,
                                      f">= {th['retention_diff_min']} (zero-request residual minus incumbent timeout rate)")
    checks["zero_request_identity"] = _check(identity_max_abs, identity_max_abs == 0.0,
                                             "the zero-request residual's action equals the incumbent's exactly")
    checks["prefix_identity"] = _check(prefix_max_abs_diff, prefix_max_abs_diff == 0.0,
                                       "every pulse cell's observations equal its partner's up to pulse_start")
    per = {}
    for p in pairs:
        per.setdefault(p["contrast"], []).append(p)
    per_contrast = {name: pooled(ps) for name, ps in sorted(per.items())}
    short = sorted(n for n in (contrast_name(k, a) for k, a in CONTRASTS)
                   if per_contrast.get(n, {"eligible_pairs": 0})["eligible_pairs"] < th["min_eligible_pairs"])
    covered = len(per_contrast) == th["expected_contrasts"] and not short
    checks["coverage"] = {"value": {"contrasts": len(per_contrast), "under_minimum": short,
                                    "min_pairs": min((c["eligible_pairs"] for c in per_contrast.values()), default=0)},
                          "ok": covered,
                          "rule": f"{th['expected_contrasts']} contrasts, each with >= {th['min_eligible_pairs']} eligible pairs"}
    pool = pooled(pairs)
    interval = clustered_interval(pairs)
    checks["paired_sign_agreement"] = {**_check(pool["sign_agreement"], pool["sign_agreement"] is not None
                                                and pool["sign_agreement"] >= th["paired_sign_agreement_min"],
                                                f">= {th['paired_sign_agreement_min']} over eligible pairs"),
                                       "case_clustered_interval95": interval}
    weakest = min((c["sign_agreement"] for c in per_contrast.values() if c["sign_agreement"] is not None), default=None)
    checks["contrast_sign_agreement"] = _check(weakest, weakest is not None and weakest >= th["contrast_sign_agreement_min"],
                                               f"every contrast >= {th['contrast_sign_agreement_min']}")
    g = pool["gain_median"]
    checks["paired_gain_median"] = _check(g, g is not None and th["paired_gain_median_min"] <= g <= th["paired_gain_median_max"],
                                          f"in [{th['paired_gain_median_min']}, {th['paired_gain_median_max']}]")
    r = pool["feasible_ratio_median"]
    checks["feasible_ratio_median"] = _check(r, r is not None and r >= th["feasible_ratio_median_min"],
                                             f">= {th['feasible_ratio_median_min']} (feasible / requested)")
    b = pool["bound_active_fraction"]
    checks["bound_active_fraction"] = _check(b, b is not None and b <= th["bound_active_fraction_max"],
                                             f"<= {th['bound_active_fraction_max']} of pulse steps")
    e = pool["tracking_error_median"]
    checks["tracking_error_median"] = _check(e, e is not None and e <= th["tracking_error_median_max"],
                                             f"<= {th['tracking_error_median_max']} (|achieved - feasible| / |feasible|)")
    hard = [checks[k]["ok"] for k in ("jacobian_rel_error_median", "retention_diff", "zero_request_identity")]
    tracking = [checks[k]["ok"] for k in ("paired_sign_agreement", "contrast_sign_agreement", "paired_gain_median",
                                          "feasible_ratio_median", "bound_active_fraction", "tracking_error_median")]
    if False in hard:
        verdict = cc.FAIL
    elif None in hard or not covered or checks["prefix_identity"]["ok"] is not True or None in tracking:
        verdict = cc.INCONCLUSIVE
    else:
        verdict = cc.PASS if all(tracking) else cc.FAIL
    low, high = interval
    target = th["paired_sign_agreement_min"]
    prediction = ("inconclusive" if low is None else "refuted" if high < target else
                  "supported" if low >= target else "inconclusive")
    return {"verdict": verdict, "checks": checks, "pooled": pool, "per_contrast": per_contrast,
            "prediction": {"outcome": prediction, "rule": f"case-clustered interval of the pooled sign agreement "
                                                          f"against {target}: upper below refutes, lower at or above supports"},
            "pulse_drops": int(sum(p["pulse_drops"] for p in pairs)), "thresholds": dict(th)}


def select_setting(cells: dict, thresholds: dict) -> dict:
    """The calibration's declared rule. Among settings meeting the bound-activity and feasible-ratio thresholds (all
    settings if none does): highest pooled sign agreement, values within 0.02 counting as tied; then lowest median
    tracking error; then lowest bound activity; then lowest index. Sustained success is never read."""
    def val(x, default):
        return default if x is None else x

    names = sorted(cells, key=lambda n: int(n[3:]))
    ok = [n for n in names if val(cells[n]["bound_active_fraction"], 1.0) <= thresholds["bound_active_fraction_max"]
          and val(cells[n]["feasible_ratio_median"], 0.0) >= thresholds["feasible_ratio_median_min"]]
    pool = ok or names
    best = max(val(cells[n]["sign_agreement"], 0.0) for n in pool)
    tied = [n for n in pool if val(cells[n]["sign_agreement"], 0.0) >= best - 0.02]
    chosen = min(tied, key=lambda n: (val(cells[n]["tracking_error_median"], float("inf")),
                                      val(cells[n]["bound_active_fraction"], 1.0), int(n[3:])))
    return {"chosen": chosen, "index": int(chosen[3:]), "met_bound_and_feasible": ok, "tied_on_agreement": tied}


# ---------------------------------------------------------------------------------------------------------------------
#  Config, manifests, controllers
# ---------------------------------------------------------------------------------------------------------------------


def candidate(protocol_cfg: dict, index: int) -> dict:
    cal = protocol_cfg["calibration"]
    over = cal["settings"][index]
    out = {key: {**cal["base"][key], **over.get(key, {})} for key in ("decoder", "controller")}
    out["windows"] = dict(protocol_cfg["windows"])
    return out


def require_frozen(protocol_cfg: dict) -> dict:
    frozen = protocol_cfg.get("frozen")
    if not frozen or not protocol_cfg.get("frozen_sha256"):
        raise SystemExit("the gate has no frozen block: run the calibration and freeze its constants first")
    frozen = {"decoder": dict(frozen["decoder"]), "controller": dict(frozen["controller"]),
              "windows": dict(protocol_cfg["windows"]), "thresholds": dict(protocol_cfg["thresholds"]),
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
    asset, n = protocol_cfg["incumbent_asset"], int(protocol_cfg["cases"])
    if name == "parent":
        from lares.envs.isaac_pen_spin.coordinate_repair import incumbent_controller

        return incumbent_controller(asset)
    if name.startswith("cal"):
        index, sign = name[3:].split("_")
        s = 1.0 if sign == "pos" else -1.0
        return RollingPulseController(name, candidate(protocol_cfg, int(index)),
                                      [(k, a, s) for k, a in calibration_contrasts(n)], asset)
    frozen = require_frozen(protocol_cfg)
    if name == "zero":
        return RollingPulseController(name, frozen, None, asset)
    if name in cg.DIRECTION_NAMES:
        return RollingPulseController(name, frozen, [cg.DIRECTIONS[cg.DIRECTION_NAMES.index(name)]] * n, asset)
    raise ValueError(f"unknown controller {name!r}")


# ---------------------------------------------------------------------------------------------------------------------
#  Drivers
# ---------------------------------------------------------------------------------------------------------------------


def _baseline_cell(protocol_cfg: dict) -> str:
    return os.path.join(str(protocol_cfg["baseline_log_dir"]), "parent.json")


def _dump(out_dir: str, name: str, payload: dict) -> None:
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_baseline(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    cell = os.path.join(out_dir, "parent.json")
    if os.path.abspath(cell) != os.path.abspath(_baseline_cell(protocol_cfg)):
        raise SystemExit("the baseline must be written to the config's baseline_log_dir")
    study_cell(protocol_cfg, config_path, cases.require_frozen(protocol_cfg["manifest"]), "parent", cell)
    summary = baseline_summary(cell, protocol_cfg)
    _dump(out_dir, "summary.json", summary)
    return summary


def run_calibration(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    cal = protocol_cfg["calibration"]
    if 2 * len(cal["settings"]) > int(cal["max_cells"]):
        raise SystemExit(f"{2 * len(cal['settings'])} calibration cells exceed the declared {cal['max_cells']}")
    freeze_json(os.path.join(out_dir, "grid.json"), {"base": cal["base"], "settings": cal["settings"],
                                                     "windows": protocol_cfg["windows"]}, "calibration grid")
    mpath = cases.require_frozen(protocol_cfg["calibration_manifest"])
    n = int(protocol_cfg["cases"])
    tips, axes = zip(*calibration_contrasts(n))
    out = {"study": CALIBRATION, "manifest": cases.manifest_record(mpath), "cells": {}, "episodes": 0,
           "note": "calibration only: chooses decoder constants, never gate evidence; sustained success is not read"}
    for i in range(len(cal["settings"])):
        motion = {}
        for sign in ("pos", "neg"):
            cell = os.path.join(out_dir, f"cal{i}_{sign}.json")
            study_cell(protocol_cfg, config_path, mpath, f"cal{i}_{sign}", cell)
            out["episodes"] += n
            motion[sign] = cell_motion(cell, tips, protocol_cfg["windows"])
        pm = paired_motion(motion["pos"], motion["neg"], axes, protocol_cfg["windows"], protocol_cfg["eligibility"])
        pairs = pair_records(pm, tips, axes)
        errs = motion["pos"]["jacobian_errors"] + motion["neg"]["jacobian_errors"]
        per = {}
        for p in pairs:
            per.setdefault(p["contrast"], []).append(p["d"])
        out["cells"][f"cal{i}"] = {"setting": candidate(protocol_cfg, i), **pooled(pairs),
                                   "case_clustered_interval95": clustered_interval(pairs),
                                   "prefix_max_abs_diff": pm["prefix_max_abs_diff"],
                                   "pulse_drops": int(sum(p["pulse_drops"] for p in pairs)),
                                   "jacobian_rel_error_median": float(np.median(errs)) if errs else None,
                                   "per_contrast_d": per}
        _dump(out_dir, "summary.partial.json", out)
    out["selection"] = select_setting(out["cells"], protocol_cfg["thresholds"])
    chosen = candidate(protocol_cfg, out["selection"]["index"])
    block = {"decoder": chosen["decoder"], "controller": chosen["controller"], "windows": chosen["windows"],
             "thresholds": dict(protocol_cfg["thresholds"]), "eligibility": dict(protocol_cfg["eligibility"])}
    out["frozen"] = {"decoder": chosen["decoder"], "controller": chosen["controller"],
                     "frozen_sha256": cg.frozen_sha256(block)}
    _dump(out_dir, "summary.json", out)
    return out


def run_gate(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    base_cell = _baseline_cell(protocol_cfg)
    base_summary = os.path.join(os.path.dirname(base_cell), "summary.json")
    if not (os.path.exists(base_cell) and os.path.exists(base_summary)):
        raise SystemExit("S1 has not run: `--study rolling_baseline` first")
    with open(base_summary, encoding="utf-8") as f:
        baseline = json.load(f)
    if not baseline["reproduces"]:
        raise SystemExit("S1 did not reproduce the incumbent; the gate runs nothing")
    frozen = require_frozen(protocol_cfg)
    freeze_json(os.path.join(out_dir, "frozen.json"), {"frozen": frozen, "sha256": protocol_cfg["frozen_sha256"]},
                "frozen block")
    mpath = cases.require_frozen(protocol_cfg["manifest"])
    if os.path.abspath(cell_manifest(base_cell)) != os.path.abspath(mpath):
        raise SystemExit("S1's cell was scored on another manifest")
    n = int(protocol_cfg["cases"])
    parent_trace, parent_rows = load_cell(base_cell)
    zero = os.path.join(out_dir, "zero.json")
    study_cell(protocol_cfg, config_path, mpath, "zero", zero)
    zero_trace, zero_rows = load_cell(zero)
    identity = float(np.load(zero.replace(".json", ".extra.npz"))["identity_max_abs"])
    per_cell = {"parent": {"timeout_rate": cg.timeout_rate(parent_rows), "sustained": baseline["successes"]},
                "zero": {"timeout_rate": cg.timeout_rate(zero_rows),
                         "sustained": int(sum(r["sustained"] for r in zero_rows)), "identity_max_abs": identity,
                         "reproduces_parent_rows": [r["sustained"] for r in zero_rows] == [r["sustained"] for r in parent_rows]}}
    pairs, errors, prefix = [], [], 0.0
    p0 = int(frozen["windows"]["pulse_start"])
    parent_prefix = in_play_prefix(parent_trace["obs"], parent_trace["length"], p0)
    for k, a in CONTRASTS:
        motion = {}
        for tag, s in cg.SIGNS:
            name = cg.direction_name(k, a, s)
            cell = os.path.join(out_dir, f"{name}.json")
            study_cell(protocol_cfg, config_path, mpath, name, cell)
            motion[tag] = m = cell_motion(cell, [k] * n, frozen["windows"])
            errors.extend(m["jacobian_errors"])
            prefix = max(prefix, float(np.abs(m["obs_prefix"] - parent_prefix).max()))
            per_cell[name] = {"timeout_rate": cg.timeout_rate(m["rows"]),
                              "sustained": int(sum(r["sustained"] for r in m["rows"])),
                              "drops": int(sum(not r["completed"] for r in m["rows"]))}
        pm = paired_motion(motion["pos"], motion["neg"], [a] * n, frozen["windows"], frozen["eligibility"])
        prefix = max(prefix, pm["prefix_max_abs_diff"])
        pairs.extend(pair_records(pm, [k] * n, [a] * n))
        _dump(out_dir, "summary.partial.json", {"cells": per_cell, "pairs": len(pairs)})
    verdict = gate_verdict(pairs, errors, per_cell["zero"]["timeout_rate"], per_cell["parent"]["timeout_rate"],
                           identity, prefix, frozen["thresholds"])
    by_tip = {}
    for k, tip in enumerate(cc.FINGERTIPS):
        by_tip[tip] = pooled([p for p in pairs if p["contrast"].startswith(tip + "_")])
    pulse_rates = [per_cell[nm]["timeout_rate"] for nm in cg.DIRECTION_NAMES]
    summary = {"study": GATE, "verdict": verdict["verdict"], "gate": verdict, "by_fingertip": by_tip,
               "baseline": {k: baseline[k] for k in ("successes", "success", "success_wilson95", "reproduces")},
               "cells": per_cell, "pulse_cell_timeout_rate": {"mean": float(np.mean(pulse_rates)),
                                                              "min": float(np.min(pulse_rates))},
               "jacobian_samples": len(errors), "frozen_sha256": protocol_cfg["frozen_sha256"],
               "decoder_sha256": cc.DecoderConfig.from_dict(frozen["decoder"]).sha256(),
               "manifest": cases.manifest_record(mpath), "episodes": n * (1 + len(cg.DIRECTION_NAMES)),
               "limitation": "directional tracking of fingertip link origins near an ideal capsule during the "
                             "incumbent's stroke, as a paired contrast; retention under a nonzero pulse is reported "
                             "per cell, not established; a pass says nothing about task success"}
    _dump(out_dir, "summary.json", summary)
    return summary


def run(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    return {BASELINE: run_baseline, CALIBRATION: run_calibration, GATE: run_gate}[protocol_cfg["study"]](
        protocol_cfg, config_path, out_dir)


__all__ = ["BASELINE", "CALIBRATION", "CONTRASTS", "GATE", "RollingPulseController", "THRESHOLDS",
           "baseline_summary", "build_manifests", "calibration_contrasts", "candidate", "cell_motion",
           "clustered_interval", "contrast_name", "gate_verdict", "in_play_prefix", "make_controller", "paired_motion", "pair_records",
           "pooled", "require_frozen", "roll_direction", "roll_metrics", "run", "run_baseline", "run_calibration",
           "run_gate", "select_setting", "wilson"]
