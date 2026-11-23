"""R-129 (Q2 of ``docs/NEXT_RESEARCH_DIRECTION_V2.md``): frozen controllers beyond the nominal operating point.

Four frozen, hashed controllers (the R-121 JOINT asset, the R-126 transplant, the D-072 reference and the PPO expert
as a competence control) on a predeclared one-factor panel: nominal; 12 s and 30 s horizons; pen mass, friction,
radius and length each scaled down and up; and reset vectors scaled by 1.5. Nothing is fitted or selected. A cell
is one controller under one condition on one whole manifest in a fresh process; a cell's controller name is
``<controller>@<condition>``, and the condition reaches the simulator through ``runner.apply_condition``.

The six-second endpoint is the same in every cell: no termination before step 359 and the per-step predicate on
recorded steps 299..358 (what ``cases.sustained`` computes on a nominal 359-step trace). Longer-horizon endpoints are
reported separately and never replace it.

    python scripts/run_shadowhand_spin_stages.py --study q2_frozen_robustness

The module is also its own Isaac child for the no-rollout preflight of each condition.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import time
import traceback

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lares.envs.isaac_pen_spin import cases  # noqa: E402
from lares.envs.isaac_pen_spin import recovery_authority as ra  # noqa: E402
from lares.envs.isaac_pen_spin import recovery_module as rm  # noqa: E402
from lares.envs.isaac_pen_spin import recovery_tail_audit as ta  # noqa: E402
from lares.envs.isaac_pen_spin import symbolic_reward as sr  # noqa: E402
from lares.envs.isaac_pen_spin.protocols import freeze_json, load_cell, study_cell  # noqa: E402
from lares.utils.hashing import sha256_file  # noqa: E402

STUDY = "q2_frozen_robustness"
CONTROLLERS = ("joint", "transplant", "d072", "ppo")
NOMINAL = "nominal"
PRIMARY = ("joint", "friction_0.8")
SIX_S_STEPS, WINDOW = 359, 60                                # a six-second time-out trace has 359 recorded steps
LABEL_CFG = {"early_drop_step": 60, "slow_hit_max": 0.2, "near_hit_min": 0.8}
MAX_RERUNS = 2
PREFLIGHT_TOL = 1e-4


# ---------------------------------------------------------------------------------------------------------------------
#  Plan
# ---------------------------------------------------------------------------------------------------------------------


def conditions(cfg: dict) -> dict:
    out = {str(k): dict(v or {}) for k, v in cfg["conditions"].items()}
    if NOMINAL not in out or out[NOMINAL] or PRIMARY[1] not in out:
        raise SystemExit("the panel needs an empty nominal condition and the primary condition")
    return out


def horizon_steps(cfg: dict, name: str) -> int:
    """Recorded steps of a time-out episode under a condition (the env times out one step early)."""
    return int(round(float(conditions(cfg)[name].get("horizon_s", 6.0)) / float(cfg["control_dt"]))) - 1


def is_six_second(cfg: dict, name: str) -> bool:
    return horizon_steps(cfg, name) == SIX_S_STEPS


def cell_name(controller: str, condition: str) -> str:
    return f"{controller}@{condition}"


def plan(cfg: dict) -> list[tuple]:
    """Every ``(seed, controller, condition)`` cell: the full panel on the panel seeds, then the primary pair on
    the extra seeds."""
    cells = [(int(s), c, k) for s in cfg["panel_seeds"] for k in conditions(cfg) for c in CONTROLLERS]
    cells += [(int(s), PRIMARY[0], k) for s in cfg["primary_extra_seeds"] for k in (NOMINAL, PRIMARY[1])]
    return cells


def manifest_for(cfg: dict, seed) -> str:
    return str(cfg["manifest_template"]).format(seed=int(seed))


def all_seeds(cfg: dict) -> list[int]:
    return [int(s) for s in (*cfg["panel_seeds"], *cfg["primary_extra_seeds"])]


def build_manifests(protocol_cfg: dict) -> list[str]:
    from lares.eval.manifest import EvaluationManifest, assert_disjoint

    mine = {manifest_for(protocol_cfg, s): cases.build_manifest(3, int(s), int(protocol_cfg["cases"]), label="robust")
            for s in all_seeds(protocol_cfg)}
    paths = sorted(glob.glob("config/manifests/pen_spin_stage3_*.yaml") + glob.glob("config/manifests/shadowhand_spin_*.yaml"))
    assert_disjoint(list(mine.values()) + [EvaluationManifest.load(p) for p in paths if p not in mine])
    return [p for p, m in mine.items() if cases.freeze_manifest(p, m)]


# ---------------------------------------------------------------------------------------------------------------------
#  Controllers
# ---------------------------------------------------------------------------------------------------------------------


def asset_x(cfg: dict, name: str) -> np.ndarray:
    """A frozen module asset's coordinates (both are stored in the JOINT ranges), verified by hash."""
    entry = rm._load(cfg["assets"][name]["path"])
    if entry["sha256"] != cfg["assets"][name]["sha256"] or rm.x_record(entry["x"], *rm.bounds(cfg))["sha256"] != entry["sha256"]:
        raise SystemExit(f"{name}: the asset is not the frozen one")
    return np.asarray(entry["x"], dtype=float)


def asset_record(cfg: dict) -> dict:
    from lares.envs.isaac_pen_spin.coordinate_repair import check_asset
    from lares.envs.isaac_pen_spin.feedback_ablation import check_checkpoint

    out = {"d072": {"descriptor": cfg["reference_descriptor"], "sha256": sha256_file(cfg["reference_descriptor"]),
                    "incumbent_files": dict(check_asset(cfg["incumbent_asset"])["files"])},
           "root_residual": {"path": cfg["root_finalist"], "sha256": sha256_file(cfg["root_finalist"]), "theta_sha256": cfg["root_theta_sha256"]},
           "ppo": {"checkpoint": check_checkpoint(cfg), "sha256": cfg["checkpoint_sha256"], "seed": int(cfg["expert_seed"])},
           "normalization": {"path": cfg["normalization_path"], "sha256": cfg["normalization_sha256"]}}
    ra.finalist_theta(cfg)
    ra.normalization(cfg)
    for name in ("joint", "transplant"):
        x = asset_x(cfg, name)
        out[name] = {"path": cfg["assets"][name]["path"], "file_sha256": sha256_file(cfg["assets"][name]["path"]),
                     "module_sha256": cfg["assets"][name]["sha256"], "coordinates": len(x),
                     "decision_step": float(rm.to_params(x, *rm.bounds(cfg))[1])}
    here = os.path.dirname(os.path.abspath(__file__))
    out["code"] = {f: sha256_file(os.path.join(here, f)) for f in ("runner.py", "protocols.py", "cases.py", "recovery_module.py",
                                                                   "symbolic_reward.py", "ppo_expert.py", "shadow_hand_spin_env.py",
                                                                   "shadow_hand_spin_env_cfg.py", "frozen_robustness.py")}
    return out


def base_controller(name: str, cfg: dict):
    if name == "d072":
        return sr.build_reference(cfg, "d072", resettable=False)
    if name in ("joint", "transplant"):
        return rm.build(cfg, name, asset_x(cfg, name))
    if name == "ppo":
        from lares.envs.isaac_pen_spin.feedback_ablation import check_checkpoint
        from lares.envs.isaac_pen_spin.ppo_expert import PPOController

        return PPOController(check_checkpoint(cfg), int(cfg["expert_seed"]))
    raise ValueError(f"unknown controller {name!r}")


class Conditioned:
    """A frozen controller plus the condition its cell's simulator is built with (``protocols.run_cell`` reads
    ``env_condition``). The controller itself is untouched: no time constant is rescaled."""

    def __init__(self, inner, name: str, condition: dict):
        self.inner, self.name, self.env_condition = inner, name, dict(condition)

    @property
    def params(self):
        return {**(getattr(self.inner, "params", None) or {}), "condition": self.env_condition}

    def attach(self, env):
        if hasattr(self.inner, "attach"):
            self.inner.attach(env)

    def reset(self, ctx):
        if hasattr(self.inner, "reset"):
            self.inner.reset(ctx)

    def __call__(self, obs, t, ctx):
        return self.inner(obs, t, ctx)

    def extra_trace(self) -> dict:
        """The inner controller's diagnostics (a recovery module adds its trigger count to ``params`` here; R-129's
        own cells predate this and have no count)."""
        return self.inner.extra_trace() if callable(getattr(self.inner, "extra_trace", None)) else {}


def make_controller(name: str, protocol_cfg: dict, out_dir: str | None = None):
    controller, _, condition = name.partition("@")
    conds = conditions(protocol_cfg)
    if controller not in CONTROLLERS or condition not in conds:
        raise ValueError(f"unknown cell {name!r}")
    return Conditioned(base_controller(controller, protocol_cfg), name, conds[condition])


# ---------------------------------------------------------------------------------------------------------------------
#  Per-case metrics from a saved cell
# ---------------------------------------------------------------------------------------------------------------------


def six_second_success(pred, length) -> np.ndarray:
    """No termination before step 359 and the predicate on recorded steps 299..358, per case."""
    pred, length = np.asarray(pred, dtype=bool), np.asarray(length)
    return (length >= SIX_S_STEPS) & pred[SIX_S_STEPS - WINDOW:SIX_S_STEPS].all(axis=0)


def longest_run(col) -> int:
    best = run = 0
    for v in col:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def window_success(pred, steps: int) -> np.ndarray:
    """``(windows, cases)``: the predicate on every step of each second, aligned so the last window of a six-second
    trace is steps 299..358 (window ``j`` covers ``[60 j - 1, 60 j + 59)``, the first ``[0, 59)``)."""
    pred = np.asarray(pred, dtype=bool)
    return np.stack([pred[max(0, WINDOW * j - 1):WINDOW * j + WINDOW - 1].all(axis=0) for j in range((steps + 1) // WINDOW)])


def case_metrics(trace: dict, rows: list, steps: int) -> list[dict]:
    """One row per case: the six-second endpoint, the horizon's own endpoint and the retention measures."""
    pred, length = np.asarray(trace["pred"], dtype=bool), np.asarray(trace["length"])
    spin, completed = np.abs(np.asarray(trace["spin_rate"])), np.asarray(trace["completed"], dtype=bool)
    six, win = six_second_success(pred, length), window_success(pred, steps)
    out = []
    for i, r in enumerate(rows):
        L = int(length[i])
        tail = spin[max(0, L - WINDOW):L, i]
        out.append({"task_hash": r["task_hash"], "sustained": bool(six[i]), "six_second": bool(six[i]),
                    "terminal": bool(r["sustained"]), "completed": bool(completed[i]), "length": L,
                    "survived_six_seconds": bool(L >= SIX_S_STEPS), "drop_step": None if completed[i] else L,
                    "rolling_fraction": float(pred[:steps, i].sum() / steps),
                    "longest_rolling_steps": longest_run(pred[:L, i]),
                    "final_second_abs_spin": float(tail.mean()) if L else None,
                    "tail_hit_fraction": float(r["tail_hit_fraction"]), "final_center_dist": r["final_center_dist"],
                    "windows": [bool(v) for v in win[:, i]]})
    return out


def cell_summary(rows: list, steps: int) -> dict:
    n = len(rows)
    drops = [r["drop_step"] for r in rows if r["drop_step"] is not None]
    spin = np.array([r["final_second_abs_spin"] for r in rows if r["completed"] and r["final_second_abs_spin"] is not None])
    win = np.array([r["windows"] for r in rows], dtype=float)
    labels = {name: 0 for name in ta.LABELS}
    for r in rows:                                           # the archived six-second failure labels, where they apply
        if steps == SIX_S_STEPS:
            labels[ta.label({**r, "sustained": r["six_second"]}, LABEL_CFG)] += 1
    return {"cases": n, "six_second_success": float(np.mean([r["six_second"] for r in rows])),
            "terminal_success": float(np.mean([r["terminal"] for r in rows])),
            "survival": float(np.mean([r["completed"] for r in rows])), "drops": len(drops),
            "survival_six_seconds": float(np.mean([r["survived_six_seconds"] for r in rows])),
            "rolling_fraction": float(np.mean([r["rolling_fraction"] for r in rows])),
            "longest_rolling_steps": {"mean": float(np.mean([r["longest_rolling_steps"] for r in rows])),
                                      "median": float(np.median([r["longest_rolling_steps"] for r in rows]))},
            "drop_step_quartiles": [float(v) for v in np.percentile(drops, [25, 50, 75])] if drops else None,
            "drops_by_second": [int(sum(WINDOW * j <= d < WINDOW * (j + 1) for d in drops)) for j in range((steps + 1) // WINDOW)],
            "final_second_abs_spin_quartiles": [float(v) for v in np.percentile(spin, [10, 25, 50, 75, 90])] if spin.size else None,
            "window_success_by_second": [float(v) for v in win.mean(axis=0)],
            "six_second_labels": labels if steps == SIX_S_STEPS else None}


# ---------------------------------------------------------------------------------------------------------------------
#  Inference
# ---------------------------------------------------------------------------------------------------------------------


def paired(by_manifest: dict, margin: float, drop_margin: float) -> dict:
    """``{manifest: (rows_a, rows_b)}`` -> the pooled paired contrast, the manifest-clustered check and the drop bound."""
    from lares.envs.isaac_pen_spin.incumbent_residual import confirmation_contrast

    c = confirmation_contrast(by_manifest, margin, drop_margin)
    p = c["pooled"]
    return {"difference": p["success"]["mean"], "ci95": [p["success"]["low"], p["success"]["high"]], "p_value": p["success"]["p_value"],
            "manifest_clustered_ci95": [c["manifest_clustered"]["low"], c["manifest_clustered"]["high"]],
            "pairs": c["pairs"], "successes": c["successes"], "win_loss_tie": p["success_win_loss_tie"],
            "drop_difference": p["drops"]["mean"], "drop_upper_95": p["drops"]["one_sided_95_upper"]}


def primary_verdict(low: float, high: float, margin: float) -> str:
    """Non-inferiority at ``margin`` (negative): supported with the lower bound above it, refuted with the upper
    bound below it."""
    return "supported" if low > margin else ("refuted" if high < margin else "inconclusive")


def analyse(cells: dict, cfg: dict) -> dict:
    """``cells[(seed, controller, condition)]`` = per-case metric rows -> every reported quantity."""
    from lares.eval.stats import t_summary

    conds, margin, dm = conditions(cfg), float(cfg["margin"]), float(cfg["drop_margin"])
    panel = [int(s) for s in cfg["panel_seeds"]]

    def rows(seeds, controller, condition):
        return {f"s{s}": cells[(s, controller, condition)] for s in seeds}

    def contrast(seeds, a, b):
        ra_, rb_ = rows(seeds, *a), rows(seeds, *b)
        return paired({m: (ra_[m], rb_[m]) for m in ra_}, margin, dm)

    head = contrast(all_seeds(cfg), PRIMARY, (PRIMARY[0], NOMINAL))
    head["verdict"] = primary_verdict(head["ci95"][0], head["ci95"][1], margin)
    head["margin"] = margin
    table = {c: {k: cell_summary([r for s in (all_seeds(cfg) if (c, k) in (PRIMARY, (PRIMARY[0], NOMINAL)) else panel)
                                  for r in cells[(s, c, k)]], horizon_steps(cfg, k)) for k in conds} for c in CONTROLLERS}
    secondary = {}
    for c in CONTROLLERS:
        for k in conds:
            if k == NOMINAL:
                continue
            if is_six_second(cfg, k):
                if (c, k) != PRIMARY:                        # a shifted condition against the same controller at nominal
                    secondary[f"{c}@{k}_minus_{c}@{NOMINAL}"] = contrast(panel, (c, k), (c, NOMINAL))
            else:                                            # retention: the horizon's endpoint against the same cell's six-second one
                d = [float(r["terminal"]) - float(r["six_second"]) for s in panel for r in cells[(s, c, k)]]
                ts = t_summary(d)
                per = t_summary([float(np.mean([float(r["terminal"]) - float(r["six_second"]) for r in cells[(s, c, k)]])) for s in panel])
                secondary[f"{c}@{k}_terminal_minus_six_second"] = {"difference": ts["mean"], "ci95": [ts["low"], ts["high"]], "p_value": ts["p_value"],
                                                                   "manifest_clustered_ci95": [per["low"], per["high"]], "pairs": ts["n"]}
    for k in conds:                                          # the other controllers against JOINT, six-second endpoint
        for c in CONTROLLERS:
            if c != "joint":
                secondary[f"{c}@{k}_minus_joint@{k}"] = contrast(panel, (c, k), ("joint", k))
    h = sr.holm({k: (1.0 if v["p_value"] is None else v["p_value"]) for k, v in secondary.items()})
    for k, v in secondary.items():
        v["holm"] = h[k]
    prefix = {}
    for c in CONTROLLERS:                                    # a longer run's first six seconds against the nominal cell
        for k in conds:
            if not is_six_second(cfg, k) and set(conds[k]) == {"horizon_s"}:
                same = [a["six_second"] == b["six_second"] for s in panel for a, b in zip(cells[(s, c, k)], cells[(s, c, NOMINAL)])]
                prefix[f"{c}@{k}"] = {"cases": len(same), "agree": int(sum(same)), "agreement": float(np.mean(same))}
    return {"paired_success_difference": head["difference"], "paired_success_difference_ci95": head["ci95"], "primary": head,
            "table": table, "secondary": secondary, "secondary_tests": len(secondary),
            "secondary_rejections": sorted(k for k, v in secondary.items() if v["holm"]["rejects"]),
            "nominal_prefix_consistency": prefix}


# ---------------------------------------------------------------------------------------------------------------------
#  Preflight (no rollout) and driver
# ---------------------------------------------------------------------------------------------------------------------


def expected_physical(cfg: dict, name: str) -> dict:
    nom, c = cfg["nominal_pen"], conditions(cfg)[name]
    return {"mass": float(nom["mass"]) * float(c.get("mass", 1.0)),
            "static_friction": float(nom["static_friction"]) * float(c.get("friction", 1.0)),
            "dynamic_friction": float(nom["dynamic_friction"]) * float(c.get("friction", 1.0)),
            "radius": float(nom["radius"]) * float(c.get("radius", 1.0)),
            "length": float(nom["length"]) * float(c.get("length", 1.0)),
            "max_episode_length": horizon_steps(cfg, name) + 1}


def preflight_check(cfg: dict, name: str, record: dict) -> dict:
    """The simulator holds what the condition says, read back with no rollout."""
    want = expected_physical(cfg, name)
    bad = [f"{k}: simulator {record.get(k)!r}, expected {v!r}" for k, v in want.items()
           if record.get(k) is None or abs(float(record[k]) - float(v)) > PREFLIGHT_TOL * max(1.0, abs(float(v)))]
    bad += list(record.get("errors") or [])
    if abs(float(record.get("control_dt", 0.0)) - float(cfg["control_dt"])) > 1e-9:
        bad.append("the control rate changed")
    return {"condition": name, "expected": want, "simulator": record, "problems": bad, "passes": not bad}


def _spawn(module_args: list, log: str) -> int:
    with open(log, "w", encoding="utf-8") as f:
        return subprocess.run([sys.executable, "-u", *module_args], stdout=f, stderr=subprocess.STDOUT, timeout=1800).returncode


def run_preflight(cfg: dict, config_path: str, out_dir: str) -> dict:
    sub = os.path.join(out_dir, "preflight")
    os.makedirs(sub, exist_ok=True)
    out = {}
    for name in conditions(cfg):
        path = os.path.join(sub, f"{name}.json")
        if not os.path.exists(path):
            req = os.path.join(sub, f"{name}.request.json")
            rm._dump(req, {"config_path": os.path.abspath(config_path), "study": STUDY, "condition": name, "response_path": path})
            code = _spawn(["-m", "lares.envs.isaac_pen_spin.frozen_robustness", req], os.path.join(sub, f"{name}.log"))
            if code != 0 or not os.path.exists(path):
                raise RuntimeError(f"preflight child for {name} failed (exit {code}); see {sub}/{name}.log")
        out[name] = preflight_check(cfg, name, rm._load(path))
    return out


def run(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    cfg = protocol_cfg
    os.makedirs(out_dir, exist_ok=True)
    descriptor = sr.check_reference(cfg)
    cfg_sha = sr.config_sha256(cfg)
    cells_plan = plan(cfg)
    mpaths = {s: cases.require_frozen(manifest_for(cfg, s)) for s in all_seeds(cfg)}
    scheduled = sum(int(cfg["cases"]) * (horizon_steps(cfg, k) + 1) for _, _, k in cells_plan)
    freeze_json(os.path.join(out_dir, "frozen_config.json"),
                {"config_sha256": cfg_sha, "config": sr.frozen_config(cfg), "reference": descriptor, "assets": asset_record(cfg),
                 "manifests": {str(s): {"path": p, "sha256": sha256_file(p)} for s, p in mpaths.items()},
                 "cells": len(cells_plan), "scheduled_case_transitions": scheduled,
                 "primary": {"a": cell_name(*PRIMARY), "b": cell_name(PRIMARY[0], NOMINAL), "margin": float(cfg["margin"]),
                             "rule": "supported: lower bound of the paired 95% interval above the margin; refuted: upper bound below it"},
                 "six_second_rule": "no termination before step 359 and the per-step predicate on recorded steps 299..358",
                 "friction_note": "the pen material's coefficients are scaled; contact friction is PhysX's combination with the hand's material"},
                "frozen configuration")
    summary = {"study": STUDY, "config_sha256": cfg_sha, "cells_planned": len(cells_plan), "scheduled_case_transitions": scheduled}
    pre = run_preflight(cfg, config_path, out_dir)
    summary["preflight"] = pre
    if not all(v["passes"] for v in pre.values()):
        summary["note"] = "the preflight failed: no outcome was collected"
        rm._dump(os.path.join(out_dir, "summary.json"), summary)
        return summary

    clock_path = os.path.join(out_dir, "accounting.json")
    clock = rm._load(clock_path) if os.path.exists(clock_path) else {"wall_s": 0.0, "cells_run": 0, "reruns": []}
    cells, missing, scored = {}, [], 0
    for seed, controller, condition in cells_plan:
        name = cell_name(controller, condition)
        sub = os.path.join(out_dir, "cells", f"s{seed}")
        os.makedirs(sub, exist_ok=True)
        cell = os.path.join(sub, f"{name}_cell.json")
        for attempt in range(1 + MAX_RERUNS):
            if os.path.exists(cell) or clock["wall_s"] >= 60.0 * float(cfg["max_minutes"]):
                break
            t0 = time.time()
            try:
                study_cell(cfg, config_path, mpaths[seed], name, cell, record_obs=False)
            except subprocess.CalledProcessError:
                clock["reruns"].append({"cell": f"s{seed}/{name}", "attempt": attempt + 1})
            clock["wall_s"] += time.time() - t0
            clock["cells_run"] += int(os.path.exists(cell))
            rm._dump(clock_path, clock)
        if not os.path.exists(cell):
            missing.append(f"s{seed}/{name}")
            continue
        saved = rm._load(cell)
        check = preflight_check(cfg, condition, saved["physical"])
        if not check["passes"] or saved["condition"] != conditions(cfg)[condition]:
            raise SystemExit(f"{cell}: the cell did not run under its condition: {check['problems']}")
        trace, rows = load_cell(cell)
        cells[(seed, controller, condition)] = case_metrics(trace, rows, horizon_steps(cfg, condition))
        scored += int(np.asarray(trace["length"]).sum())
    summary["accounting"] = {"cells": len(cells), "missing": missing, "first_episodes": int(sum(len(v) for v in cells.values())),
                             "scored_transitions": scored, "wall_minutes": clock["wall_s"] / 60.0, "max_minutes": float(cfg["max_minutes"]),
                             "reruns": clock["reruns"], "llm_calls": 0, "training": "none"}
    if missing:
        summary["incomplete"] = f"{len(missing)} cells have no result; no inference"
        rm._dump(os.path.join(out_dir, "summary.json"), summary)
        return summary
    summary.update(analyse(cells, cfg))
    summary["triggered"] = {c: {k: sum((rm._load(os.path.join(out_dir, "cells", f"s{s}", f"{cell_name(c, k)}_cell.json"))["controller_params"]
                                         .get("recovery") or {}).get("triggered_environment_episodes", float("nan")) for s in cfg["panel_seeds"])
                                for k in conditions(cfg)} for c in ("joint", "transplant")}
    summary["limitation"] = ("frozen controllers from one training lineage on a one-factor panel; no combined shifts; secondary cells have "
                             "512 cases; robustness here says nothing about which component provides it")
    rm._dump(os.path.join(out_dir, "summary.json"), summary)
    return summary


def child_main(request_path: str) -> None:
    """Build one condition's simulator, reset it, read the pen's physical record back. No step is taken."""
    from lares.envs.isaac_lab_adapter import _ensure_isaaclab_on_path, close_isaac_app, launch_isaac_app
    from lares.utils.cli import load_yaml

    req = rm._load(request_path)
    full = load_yaml(req["config_path"])
    cfg = full["research_protocols"][req["study"]]
    _ensure_isaaclab_on_path(str(full["isaaclab_root"]))
    launch_isaac_app(headless=True, device=str(full["device"]))
    code = 0
    try:
        from lares.envs.isaac_pen_spin.runner import make_env, physical_record

        env = make_env(3, int(cfg["cases"]), str(full["device"]), condition=conditions(cfg)[req["condition"]])
        try:
            env.reset()
            rm._dump(req["response_path"], physical_record(env))
        finally:
            env.close()
        print("wrote", req["response_path"], flush=True)
    except BaseException:  # Isaac's close swallows tracebacks: print before it runs
        traceback.print_exc()
        sys.stdout.flush()
        code = 1
    finally:
        if code:
            os._exit(code)
        close_isaac_app()


__all__ = ["CONTROLLERS", "NOMINAL", "PRIMARY", "STUDY", "Conditioned", "analyse", "asset_record", "asset_x", "build_manifests",
           "case_metrics", "cell_name", "cell_summary", "conditions", "expected_physical", "horizon_steps", "make_controller",
           "plan", "preflight_check", "primary_verdict", "run", "six_second_success", "window_success"]

if __name__ == "__main__":
    child_main(sys.argv[1])
