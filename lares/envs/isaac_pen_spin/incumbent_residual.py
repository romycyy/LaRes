"""R-099 (anonymous-only fallback): a bounded joint-motion residual on the fixed-stroke incumbent. No LLM.

R-097 (D-069) reproduced the incumbent and failed the rolling decoder gate, so the contact arm of this idea is
untested and only the predeclared anonymous arm runs. The residual reads joint velocities and anonymous joint
indices only (no fingertip, no pen frame, no roll sign) and reinforces the motion each actuated joint already has::

    v_f   <- v_f + alpha (v - v_f),  alpha = dt / (tau + dt)            (rad/s, per actuated joint)
    delta  = clip(g * sign(v_f) * max(|v_f| - v0, 0), -b, b)            (rad)
    action = clamp(incumbent + 2 delta / (hi - lo), -1, 1)

Four free scalars ``(g, v0, tau, b)``. With ``g = 0`` the action is the incumbent's, bit for bit.

Two registry names serve this module (``protocol_cfg["study"]`` says which), separately gated:

* ``incumbent_residual`` (S3): the incumbent and ``draws`` parameter draws (seeded uniform in the declared ranges),
  each on the whole probe manifest in its own process. Ranks the draws by mean final-window hit fraction, then
  sustained successes, then index; writes the winner and its hash and **stops**. No fresh case is touched.
* ``incumbent_residual_fresh`` (S4): refuses to run unless the config's ``selected`` block matches
  ``selected_sha256`` and S3's own record; then the incumbent and the frozen winner on the fresh manifest. Adoption
  needs the strict paired sustained-success rule and a one-sided 95% upper bound on the paired drop-rate difference
  no greater than ``drop_margin``.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin.layout import VEL_OBS_SCALE
from lares.envs.isaac_pen_spin.protocols import arm_summary, failure_split, freeze_json, load_cell, study_cell
from lares.utils.hashing import sha256_text

STAGE = 3
SEARCH, FRESH = "incumbent_residual", "incumbent_residual_fresh"
CONFIRMATION = "incumbent_residual_confirmation"
PARAMS = ("gain", "deadband", "tau", "cap")
ACTUATED = tuple(range(17)) + (20, 21, 23)     # the 20 actuated joints among the 24 observed (R-093)


def params_sha256(params: dict) -> str:
    return sha256_text(json.dumps({k: float(params[k]) for k in PARAMS}, sort_keys=True, separators=(",", ":")))


def draw_params(protocol_cfg: dict) -> list[dict]:
    """``draws`` parameter sets: uniform in ``ranges``, from ``numpy.random.default_rng(draw_seed)``."""
    rng = np.random.default_rng(int(protocol_cfg["draw_seed"]))
    u = rng.random((int(protocol_cfg["draws"]), len(PARAMS)))
    r = protocol_cfg["ranges"]
    return [{k: float(r[k][0] + u[i, j] * (r[k][1] - r[k][0])) for j, k in enumerate(PARAMS)} for i in range(len(u))]


class JointMotionResidual:
    """The incumbent plus the anonymous joint-motion residual (module docstring). ``inner`` is injected by tests."""

    def __init__(self, name: str, params: dict, asset_dir: str | None = None, inner=None, control_dt: float = 1.0 / 60.0):
        if inner is None:
            from lares.envs.isaac_pen_spin.coordinate_repair import incumbent_controller

            inner = incumbent_controller(asset_dir)
        self.inner, self.name = inner, name
        self.p = {k: float(params[k]) for k in PARAMS}
        self.alpha = control_dt / (self.p["tau"] + control_dt)
        self.params = {**self.p, "params_sha256": params_sha256(self.p), "parent": getattr(inner, "params", None),
                       "execution": "clamp(incumbent + 2 clip(g sign(v_f) max(|v_f| - v0, 0), -b, b) / (hi - lo), -1, 1)"}
        self._stats = {"steps": 0, "abs_delta": 0.0, "at_cap": 0, "clipped": 0, "values": 0}

    def reset(self, ctx):
        if hasattr(self.inner, "reset"):
            self.inner.reset(ctx)
        self.v_f = None
        self.span = (ctx.upper - ctx.lower)
        self.sl = cases.obs_slices()["hand_dof_vel"]

    def __call__(self, obs, t, ctx):
        import torch

        g = self.inner(obs, t, ctx).to(self.span.device)
        v = (obs[:, self.sl][:, list(ACTUATED)] / VEL_OBS_SCALE).to(self.span.device, self.span.dtype)
        v = torch.nan_to_num(v)
        self.v_f = v.clone() if self.v_f is None else self.v_f + self.alpha * (v - self.v_f)
        p = self.p
        delta = (p["gain"] * torch.sign(self.v_f) * (self.v_f.abs() - p["deadband"]).clamp(min=0.0)).clamp(-p["cap"], p["cap"])
        raw = g + (2.0 * delta / self.span).to(g.dtype)
        action = raw.clamp(-1.0, 1.0)
        s = self._stats
        s["steps"] += 1
        s["values"] += int(delta.numel())
        s["abs_delta"] += float(delta.abs().sum())
        s["at_cap"] += int((delta.abs() >= p["cap"] - 1e-9).sum()) if p["cap"] > 0 else 0
        s["clipped"] += int(((action != raw) & (delta != 0)).sum())
        self.params["residual_activity"] = self.activity()
        return action

    def activity(self) -> dict:
        s = self._stats
        n = max(1, s["values"])
        return {"mean_abs_delta_rad": s["abs_delta"] / n, "at_cap_fraction": s["at_cap"] / n,
                "clipped_by_action_clamp_fraction": s["clipped"] / n, "values": s["values"]}


# ---------------------------------------------------------------------------------------------------------------------
#  Manifests and controllers
# ---------------------------------------------------------------------------------------------------------------------


def build_manifests(protocol_cfg: dict) -> list[str]:
    from lares.eval.manifest import EvaluationManifest, assert_disjoint

    if protocol_cfg.get("study") == CONFIRMATION:
        mine = {confirmation_manifest(protocol_cfg, seed): cases.build_manifest(
            STAGE, int(seed), int(protocol_cfg["confirmation_cases"]), label="residual_confirmation")
            for seed in protocol_cfg["confirmation_seeds"]}
        others = [EvaluationManifest.load(p) for p in sorted(glob.glob("config/manifests/pen_spin_stage3_*.yaml"))
                  if p not in mine]
        assert_disjoint(list(mine.values()) + others)
        return [p for p, m in mine.items() if cases.freeze_manifest(p, m)]
    mine = {protocol_cfg["probe_manifest"]: cases.build_manifest(STAGE, int(protocol_cfg["probe_seed"]),
                                                                 int(protocol_cfg["probe_cases"]), label="residual_probe"),
            protocol_cfg["fresh_manifest"]: cases.build_manifest(STAGE, int(protocol_cfg["fresh_seed"]),
                                                                 int(protocol_cfg["fresh_cases"]), label="residual_fresh")}
    others = [EvaluationManifest.load(p) for p in sorted(glob.glob("config/manifests/pen_spin_stage3_*.yaml"))
              if p not in mine]
    assert_disjoint(list(mine.values()) + others)
    return [p for p, m in mine.items() if cases.freeze_manifest(p, m)]


def require_selected(protocol_cfg: dict) -> dict:
    sel = protocol_cfg.get("selected")
    if not sel or not protocol_cfg.get("selected_sha256"):
        raise SystemExit("no selected block: run the search (S3) and freeze its winner first")
    got = params_sha256(sel)
    if got != protocol_cfg["selected_sha256"]:
        raise SystemExit(f"selected block hash {got} != the config's selected_sha256 {protocol_cfg['selected_sha256']}")
    return {k: float(sel[k]) for k in PARAMS}


def make_controller(name: str, protocol_cfg: dict, out_dir: str | None = None):
    asset = protocol_cfg["incumbent_asset"]
    if name == "parent":
        from lares.envs.isaac_pen_spin.coordinate_repair import incumbent_controller

        return incumbent_controller(asset)
    if name.startswith("draw"):
        return JointMotionResidual(name, draw_params(protocol_cfg)[int(name[4:])], asset)
    if name == "selected":
        return JointMotionResidual(name, require_selected(protocol_cfg), asset)
    raise ValueError(f"unknown controller {name!r}")


# ---------------------------------------------------------------------------------------------------------------------
#  Statistics
# ---------------------------------------------------------------------------------------------------------------------


def rank_key(row: dict, index: int) -> tuple:
    """Higher mean final-window hit fraction, then more sustained successes, then the lower index."""
    return (-float(row["tail_hit_fraction"]), -int(row["successes"]), int(index))


def unsigned_roll(trace: dict, dt: float) -> float:
    """Mean over cases of the integral of ``|axial rate|`` over the episode (rad): roll whatever its direction."""
    spin, length = trace["spin_rate"], trace["length"]
    return float(np.mean([np.abs(np.nan_to_num(spin[:int(min(L, spin.shape[0])), i])).sum() * dt
                          for i, L in enumerate(length)]))


def cell_row(cell: str, dt: float) -> dict:
    from lares.envs.isaac_pen_spin.channel_lesions import tail_failure_split

    trace, rows = load_cell(cell)
    with open(cell, encoding="utf-8") as f:
        activity = (json.load(f).get("controller_params") or {}).get("residual_activity")
    return {**arm_summary(rows), **failure_split(rows), **tail_failure_split(rows),
            "unsigned_roll_rad": unsigned_roll(trace, dt), "residual_activity": activity}


def fresh_contrast(a_rows: list, b_rows: list, threshold: float, drop_margin: float) -> dict:
    """``a`` minus ``b`` paired by case: the strict success rule, the prediction, and the drop non-inferiority bound
    (the one-sided 95% upper bound of the paired drop-rate difference, i.e. the upper end of a two-sided 90%
    interval)."""
    from lares.eval.stats import t_summary, win_loss_tie

    if [r["task_hash"] for r in a_rows] != [r["task_hash"] for r in b_rows]:
        raise ValueError("rows are not paired case for case")
    ds = [float(a["sustained"]) - float(b["sustained"]) for a, b in zip(a_rows, b_rows)]
    dd = [float(not a["completed"]) - float(not b["completed"]) for a, b in zip(a_rows, b_rows)]
    dh = [float(a["tail_hit_fraction"]) - float(b["tail_hit_fraction"]) for a, b in zip(a_rows, b_rows)]
    ts, td = t_summary(ds), t_summary(dd, level=0.90)
    promotes = bool(ts["mean"] > 0 and ts["low"] is not None and ts["low"] > 0)
    drops_ok = bool(td["high"] is not None and td["high"] <= drop_margin + 1e-12)
    return {"success": ts, "success_win_loss_tie": win_loss_tie(ds), "tail_hit_fraction": t_summary(dh),
            "drops": {**td, "one_sided_95_upper": td["high"], "margin": drop_margin, "non_inferior": drops_ok,
                      "counts": [int(sum(not r["completed"] for r in a_rows)), int(sum(not r["completed"] for r in b_rows))]},
            "promotes": promotes, "adopts": bool(promotes and drops_ok),
            "prediction": {"threshold": threshold,
                           "supported": bool(ts["mean"] >= threshold and ts["low"] is not None and ts["low"] > 0),
                           "refuted": bool(ts["high"] is not None and ts["high"] < threshold)}}


# ---------------------------------------------------------------------------------------------------------------------
#  Drivers
# ---------------------------------------------------------------------------------------------------------------------


def _dump(out_dir: str, name: str, payload: dict) -> None:
    with open(os.path.join(out_dir, name), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run_search(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    draws = draw_params(protocol_cfg)
    freeze_json(os.path.join(out_dir, "draws.json"), {"draw_seed": int(protocol_cfg["draw_seed"]),
                                                      "ranges": protocol_cfg["ranges"], "draws": draws}, "parameter draws")
    mpath = cases.require_frozen(protocol_cfg["probe_manifest"])
    n, dt = int(protocol_cfg["probe_cases"]), float(protocol_cfg["control_dt"])
    out = {"study": SEARCH, "manifest": cases.manifest_record(mpath), "cells": {}, "episodes": 0}
    for name in ["parent"] + [f"draw{i}" for i in range(len(draws))]:
        cell = os.path.join(out_dir, f"{name}.json")
        study_cell(protocol_cfg, config_path, mpath, name, cell)
        out["episodes"] += n
        out["cells"][name] = cell_row(cell, dt)
        if name != "parent":
            out["cells"][name]["params"] = draws[int(name[4:])]
        _dump(out_dir, "summary.partial.json", out)
    order = sorted(range(len(draws)), key=lambda i: rank_key(out["cells"][f"draw{i}"], i))
    best = order[0]
    out["ranking"] = [f"draw{i}" for i in order]
    out["selected"] = {"draw": f"draw{best}", "params": draws[best], "selected_sha256": params_sha256(draws[best]),
                       "rule": "mean final-window hit fraction, then sustained successes, then the lower index"}
    out["note"] = ("S3 stops here: the winner is chosen on the probe manifest only and is not evidence of "
                   "usefulness; no fresh case has been run")
    freeze_json(os.path.join(out_dir, "selected.json"), out["selected"], "selected residual")
    _dump(out_dir, "summary.json", out)
    return out


def run_fresh(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    selected = require_selected(protocol_cfg)
    record = os.path.join(str(protocol_cfg["search_log_dir"]), "selected.json")
    if not os.path.exists(record):
        raise SystemExit("S3 has not run: `--study incumbent_residual` first")
    with open(record, encoding="utf-8") as f:
        if json.load(f)["selected_sha256"] != protocol_cfg["selected_sha256"]:
            raise SystemExit("the config's selected block is not S3's recorded winner")
    # not "selected.json": that is the ``selected`` controller's cell, which a file of that name would skip
    freeze_json(os.path.join(out_dir, "frozen_selected.json"), {"params": selected,
                                                        "selected_sha256": protocol_cfg["selected_sha256"]}, "selected residual")
    mpath = cases.require_frozen(protocol_cfg["fresh_manifest"])
    n, dt = int(protocol_cfg["fresh_cases"]), float(protocol_cfg["control_dt"])
    rows, cells = {}, {}
    for name in ("parent", "selected"):
        cell = os.path.join(out_dir, f"{name}.json")
        study_cell(protocol_cfg, config_path, mpath, name, cell)
        _, rows[name] = load_cell(cell)
        cells[name] = cell_row(cell, dt)
    c = fresh_contrast(rows["selected"], rows["parent"], float(protocol_cfg["threshold"]), float(protocol_cfg["drop_margin"]))
    summary = {"study": FRESH, "cells": cells, "selected_minus_parent": c,
               "paired_success_difference": c["success"]["mean"],
               "paired_success_difference_ci95": [c["success"]["low"], c["success"]["high"]],
               "adopts": c["adopts"], "selected": selected, "selected_sha256": protocol_cfg["selected_sha256"],
               "manifest": cases.manifest_record(mpath), "episodes": 2 * n,
               "limitation": "one residual family, one incumbent, one 24-draw random search; the contact arm of this "
                             "idea was not run (R-097 failed its gate)"}
    _dump(out_dir, "summary.json", summary)
    return summary


def confirmation_manifest(protocol_cfg: dict, seed) -> str:
    return str(protocol_cfg["confirmation_manifest_template"]).format(seed=int(seed))


def confirmation_contrast(by_manifest: dict, threshold: float, drop_margin: float) -> dict:
    """R-102: ``{manifest: (selected rows, parent rows)}`` -> the pooled paired contrast over every case (the primary:
    strict rule and drop bound, :func:`fresh_contrast`) and, beside it, the manifest-clustered check (a t interval
    over the per-manifest mean differences), because the cases of one batch are not independent."""
    from lares.eval.stats import t_summary

    sel = [r for a, _ in by_manifest.values() for r in a]
    par = [r for _, b in by_manifest.values() for r in b]
    pooled = fresh_contrast(sel, par, threshold, drop_margin)
    per = {m: {"n": len(a), "selected": int(sum(r["sustained"] for r in a)), "parent": int(sum(r["sustained"] for r in b)),
               "difference": float(np.mean([float(x["sustained"]) - float(y["sustained"]) for x, y in zip(a, b)])),
               "drops": [int(sum(not r["completed"] for r in a)), int(sum(not r["completed"] for r in b))]}
           for m, (a, b) in by_manifest.items()}
    clustered = t_summary([v["difference"] for v in per.values()])
    return {"pooled": pooled, "per_manifest": per, "manifest_clustered": clustered,
            "intervals_agree_on_sign": bool((pooled["success"]["low"] > 0) == (clustered["low"] is not None and clustered["low"] > 0)),
            "pairs": len(sel), "successes": [int(sum(r["sustained"] for r in sel)), int(sum(r["sustained"] for r in par))]}


def run_confirmation(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    """R-102: the frozen D-070 residual, unchanged, against the incumbent on every confirmation manifest; each
    manifest once per arm, each in a fresh process. The last replication whatever it shows."""
    os.makedirs(out_dir, exist_ok=True)
    selected = require_selected(protocol_cfg)
    record = os.path.join(str(protocol_cfg["search_log_dir"]), "selected.json")
    if not os.path.exists(record):
        raise SystemExit("S3's record is missing: the confirmation only runs the residual that search froze")
    with open(record, encoding="utf-8") as f:
        if json.load(f)["selected_sha256"] != protocol_cfg["selected_sha256"]:
            raise SystemExit("the config's selected block is not S3's recorded winner")
    seeds = [int(s) for s in protocol_cfg["confirmation_seeds"]]
    freeze_json(os.path.join(out_dir, "frozen_selected.json"),
                {"params": selected, "selected_sha256": protocol_cfg["selected_sha256"], "seeds": seeds,
                 "cases_per_manifest": int(protocol_cfg["confirmation_cases"])}, "confirmation plan")
    dt = float(protocol_cfg["control_dt"])
    by_manifest, cells, manifests = {}, {}, {}
    for seed in seeds:
        mpath = cases.require_frozen(confirmation_manifest(protocol_cfg, seed))
        sub = os.path.join(out_dir, f"s{seed}")
        os.makedirs(sub, exist_ok=True)
        rows = {}
        for name in ("parent", "selected"):
            cell = os.path.join(sub, f"{name}.json")
            study_cell(protocol_cfg, config_path, mpath, name, cell)
            _, rows[name] = load_cell(cell)
            cells[f"s{seed}/{name}"] = cell_row(cell, dt)
        by_manifest[f"s{seed}"] = (rows["selected"], rows["parent"])
        manifests[f"s{seed}"] = cases.manifest_record(mpath)
    c = confirmation_contrast(by_manifest, float(protocol_cfg["threshold"]), float(protocol_cfg["drop_margin"]))
    p = c["pooled"]
    summary = {"study": CONFIRMATION, "selected_minus_parent": c, "cells": cells,
               "paired_success_difference": p["success"]["mean"],
               "paired_success_difference_ci95": [p["success"]["low"], p["success"]["high"]],
               "adopts": p["adopts"], "selected": selected, "selected_sha256": protocol_cfg["selected_sha256"],
               "manifests": manifests, "episodes": 2 * c["pairs"],
               "limitation": "one frozen residual on one incumbent; the final replication, with no retuning; D-070's "
                             "90 pairs are not pooled in"}
    _dump(out_dir, "summary.json", summary)
    return summary


def run(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    return {SEARCH: run_search, FRESH: run_fresh, CONFIRMATION: run_confirmation}[protocol_cfg["study"]](
        protocol_cfg, config_path, out_dir)


__all__ = ["ACTUATED", "CONFIRMATION", "FRESH", "JointMotionResidual", "PARAMS", "SEARCH", "build_manifests", "cell_row",
           "confirmation_contrast", "confirmation_manifest", "draw_params", "fresh_contrast", "make_controller", "params_sha256", "rank_key", "require_selected", "run",
           "run_confirmation", "run_fresh", "run_search", "unsigned_roll"]
