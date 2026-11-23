"""R-128 (Q1 of ``docs/NEXT_RESEARCH_DIRECTION_V2.md``): the successful construction recipe, replicated across
independent training lineages.

Each lineage repeats the four archived stages from the frozen D-072 reference with its own seeds and nothing else
changed: A the 32-coefficient residual (R-106), B the 11-parameter re-capture module from the scripted point (R-116),
C its continuation (R-118 CONTINUE), D joint adaptation of the module and the residual in widened module ranges
(R-121 JOINT). The training children are the archived ones (``success_ars_child``, ``recovery_module``); a lineage's
seeds, schedule and root reach them through ``cfg_overrides`` in the request. Every terminal mean is frozen with no
selection, and every endpoint, a D-072 cell and the frozen historical JOINT asset are evaluated on the same fresh
manifests, one manifest per fresh process.

    python scripts/run_shadowhand_spin_stages.py --study q1_training_replication

The module is also its own Isaac child for the four-rollout smoke gate (``python -m ... <request.json>``).
"""

from __future__ import annotations

import glob
import json
import math
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

STUDY = "q1_training_replication"
STAGES = ("A", "B", "C", "D")
REFERENCE, HISTORICAL = "d072", "r121"
SLOTS, ROLLOUTS, HORIZON = 4, 2, 360
MAX_RERUNS = 2                                               # infrastructure deaths only, per stage or cell
LABEL_CFG = {"early_drop_step": 60, "slow_hit_max": 0.2, "near_hit_min": 0.8}
CHILDREN = {"A": "lares.envs.isaac_pen_spin.success_ars_child", "B": "lares.envs.isaac_pen_spin.recovery_module",
            "C": "lares.envs.isaac_pen_spin.recovery_module", "D": "lares.envs.isaac_pen_spin.recovery_module",
            "smoke": "lares.envs.isaac_pen_spin.training_replication"}


# ---------------------------------------------------------------------------------------------------------------------
#  Lineages: seeds, per-stage configuration, inheritance
# ---------------------------------------------------------------------------------------------------------------------


def lineages(cfg: dict) -> list[dict]:
    """``[{"A": seed, "B": seed, "C": seed, "D": seed}, ...]``, one per lineage, all seeds distinct."""
    out = [dict(zip(STAGES, (int(s) for s in row))) for row in cfg["lineages"]]
    flat = [s for lin in out for s in lin.values()] + [int(cfg["smoke_seed"])]
    if any(len(row) != len(STAGES) for row in cfg["lineages"]) or len(set(flat)) != len(flat):
        raise SystemExit("every lineage needs one distinct seed per stage, disjoint from the smoke seed")
    return out


def endpoint_name(k: int, stage: str) -> str:
    return f"L{k}_{stage}"


def policies(cfg: dict) -> list[str]:
    return [REFERENCE, HISTORICAL] + [endpoint_name(k, s) for k in range(len(cfg["lineages"])) for s in STAGES]


def updates(cfg: dict, stage: str) -> int:
    return int(cfg["stages"][stage]["updates"])


def scored_per_update(cfg: dict) -> int:
    return SLOTS * int(cfg["cases_per_slot"]) * ROLLOUTS


def stage_overrides(cfg: dict, stage: str, seed: int, root_path: str | None = None, root_sha: str | None = None) -> dict:
    """What a training child's config block becomes for one lineage stage: the archived schedule, this lineage's seed
    and (stages B to D) this lineage's own stage-A residual as the root."""
    st = cfg["stages"][stage]
    if stage == "A":
        return {"training": {"seed": int(seed), "slots": SLOTS, "cases_per_slot": int(cfg["cases_per_slot"]),
                             "rollouts_per_update": ROLLOUTS, "horizon": HORIZON},
                "ars": {"iterations": int(st["updates"]), "directions": 4, "top": int(st["top"]), "nu": float(st["nu"]),
                        "alpha": float(st["alpha"]), "rng_offset": int(st["rng_offset"])}}
    return {"training_seed": int(seed), "root_finalist": root_path, "root_theta_sha256": root_sha,
            "ars": {"iterations": int(st["updates"]), "directions": 4, "nu": float(st["nu"]), "alpha": float(st["alpha"])}}


def lineage_cfg(cfg: dict, root_path: str, root_sha: str) -> dict:
    """The study block with one lineage's stage-A residual as the recovery controller's root."""
    return {**cfg, "root_finalist": root_path, "root_theta_sha256": root_sha}


def joint_initial(cfg: dict, x_parent, theta) -> np.ndarray:
    """R-121's inheritance: the parent module re-expressed in the widened ranges, then the residual's own coordinates."""
    wide = rm.to_unit(rm.to_params(np.asarray(x_parent, dtype=float)), *rm.bounds(cfg))
    return np.concatenate([wide, rm.residual_origin(theta)])


def joint_scale(cfg: dict) -> np.ndarray:
    """Original range over widened range per module coordinate, 1 for the residual coordinates (R-121)."""
    low, high = rm.bounds(cfg)
    return np.concatenate([((rm.HIGH - rm.LOW) / (high - low))[:rm.DIM], np.ones(rm.N_RESIDUAL)])


def stage_request(cfg: dict, stage: str, seed: int, parent: dict | None, root: dict | None) -> dict:
    """The stage-specific part of a training request. ``parent`` is the previous stage's endpoint record, ``root``
    ``{"path", "theta", "theta_sha256"}`` of the lineage's stage-A residual."""
    if stage == "A":
        return {"cfg_overrides": stage_overrides(cfg, "A", seed)}
    req = {"cfg_overrides": stage_overrides(cfg, stage, seed, root["path"], root["theta_sha256"])}
    if stage == "B":                                         # from the scripted point, original ranges, 11 draws
        return {**req, "original_bounds": True}
    if stage == "C":                                         # R-118 CONTINUE shared its draws with a 13-coordinate arm
        return {**req, "init_x": list(parent["x"]), "direction_dim": rm.DIM_RETRY, "original_bounds": True}
    return {**req, "init_x": [float(v) for v in joint_initial(cfg, parent["x"], root["theta"])], "direction_dim": rm.DIM_JOINT,
            "coord_scale": [float(v) for v in joint_scale(cfg)], "original_bounds": False}


def endpoint_record(cfg: dict, stage: str, train: dict) -> dict:
    """A terminal mean as frozen: the residual's coefficients (A) or the module coordinates in the stage's ranges."""
    if stage == "A":
        return {"theta": train["final_theta"], "theta_sha256": train["final_theta_sha256"]}
    return rm.x_record(train["final_x"], *rm.bounds(cfg if stage == "D" else None))


def divergent(stage: str, train: dict, bound: float) -> str | None:
    """Why a terminal state is unusable, or ``None``. Never a reason to rerun."""
    v = np.asarray(train.get("final_theta" if stage == "A" else "final_x") or [np.nan], dtype=float)
    if not np.isfinite(v).all():
        return "non-finite terminal mean"
    lo, hi = (-bound, bound) if stage == "A" else (0.0, 1.0)
    if v.min() < lo - 1e-9 or v.max() > hi + 1e-9:
        return "terminal mean outside its range"
    return None


def placement_provenance(cfg: dict) -> dict:
    """Every training and smoke placement of every lineage, hashed, disjoint from each other and from every committed
    pen-spin manifest (this study's own included)."""
    from lares.envs.isaac_pen_spin.success_ars import placement_hashes
    from lares.eval.manifest import EvaluationManifest
    from lares.utils.hashing import sha256_text

    per, out, seen = int(cfg["cases_per_slot"]), {}, []
    for k, lin in enumerate(lineages(cfg)):
        for s in STAGES:
            h = placement_hashes(cases.placements(lin[s], per * updates(cfg, s)))
            out[endpoint_name(k, s)] = {"seed": lin[s], "direction_rng_seed": lin[s] + 2000, "cases": len(h), "sha256": sha256_text("".join(h))}
            seen += h
    h = placement_hashes(cases.placements(int(cfg["smoke_seed"]), per))
    out["smoke"] = {"seed": int(cfg["smoke_seed"]), "cases": len(h), "sha256": sha256_text("".join(h))}
    seen += h
    committed = set()
    for p in sorted(glob.glob("config/manifests/pen_spin_*.yaml") + glob.glob("config/manifests/shadowhand_spin_*.yaml")):
        committed |= {c.task_hash for c in EvaluationManifest.load(p).episodes}
    if len(set(seen)) != len(seen) or committed & set(seen):
        raise SystemExit("training or smoke placements collide with each other or with a committed manifest")
    rng = [v["direction_rng_seed"] for v in out.values() if "direction_rng_seed" in v]
    if len(set(rng)) != len(rng) or set(rng) & {v["seed"] for v in out.values()}:
        raise SystemExit("an optimizer direction stream shares its seed with another stream")
    return {"streams": out, "training_cases": len(seen) - per, "disjoint_from_committed_pen_spin_cases": len(committed)}


# ---------------------------------------------------------------------------------------------------------------------
#  Manifests and controllers
# ---------------------------------------------------------------------------------------------------------------------


def manifest_for(cfg: dict, seed) -> str:
    return str(cfg["confirmation_manifest_template"]).format(seed=int(seed))


def build_manifests(protocol_cfg: dict) -> list[str]:
    from lares.eval.manifest import EvaluationManifest, assert_disjoint

    mine = {manifest_for(protocol_cfg, s): cases.build_manifest(3, int(s), int(protocol_cfg["confirmation_cases"]), label="q1_replication")
            for s in protocol_cfg["confirmation_seeds"]}
    paths = sorted(glob.glob("config/manifests/pen_spin_stage3_*.yaml") + glob.glob("config/manifests/shadowhand_spin_*.yaml"))
    assert_disjoint(list(mine.values()) + [EvaluationManifest.load(p) for p in paths if p not in mine])
    return [p for p, m in mine.items() if cases.freeze_manifest(p, m)]


def lineage_dir(out_dir: str, k: int) -> str:
    return os.path.join(out_dir, f"lineage_{k}")


def root_path(out_dir: str, k: int) -> str:
    return os.path.join(lineage_dir(out_dir, k), "stage_A_finalist.json")


def load_endpoints(out_dir: str, k: int) -> dict:
    path = os.path.join(lineage_dir(out_dir, k), "endpoints.json")
    return rm._load(path)["endpoints"] if os.path.exists(path) else {}


def historical_x(cfg: dict) -> np.ndarray:
    entry = rm._load(cfg["assets"][HISTORICAL]["path"])
    if entry["sha256"] != cfg["assets"][HISTORICAL]["sha256"] or rm.x_record(entry["x"], *rm.bounds(cfg))["sha256"] != entry["sha256"]:
        raise SystemExit("the historical JOINT asset is not the frozen one")
    return np.asarray(entry["x"], dtype=float)


def make_controller(name: str, protocol_cfg: dict, out_dir: str | None = None):
    cfg = protocol_cfg
    if name == REFERENCE:
        return sr.build_reference(cfg, REFERENCE, resettable=False)
    if name == HISTORICAL:
        return rm.build(cfg, HISTORICAL, historical_x(cfg))
    try:
        k, stage = int(name[1:].split("_")[0]), name.split("_")[1]
        if name != endpoint_name(k, stage) or stage not in STAGES:
            raise ValueError
    except (ValueError, IndexError):
        raise ValueError(f"unknown controller {name!r}") from None
    out_dir = out_dir or str(cfg["log_dir"])
    ends = load_endpoints(out_dir, k)
    if stage not in ends:
        raise SystemExit(f"{name}: this lineage has no frozen stage-{stage} endpoint")
    a = ends["A"]
    if sr.theta_sha256(a["theta"]) != a["theta_sha256"]:
        raise SystemExit(f"{name}: the stored residual does not match its hash")
    if stage == "A":
        return sr.build_controller(cfg, name, a["theta"], ra.normalization(cfg))
    e, orig = ends[stage], stage != "D"
    if rm.x_record(e["x"], *rm.bounds(None if orig else cfg))["sha256"] != e["sha256"]:
        raise SystemExit(f"{name}: the stored module does not match its hash")
    return rm.build(lineage_cfg(cfg, root_path(out_dir, k), a["theta_sha256"]), name, e["x"], original_bounds=orig)


# ---------------------------------------------------------------------------------------------------------------------
#  Inference
# ---------------------------------------------------------------------------------------------------------------------


def two_level(matrix, threshold: float, level: float = 0.95) -> dict:
    """Lineages (rows) by manifests (columns) of manifest-mean paired differences: random lineage and manifest
    intercepts plus a residual, by the balanced two-way ANOVA moment estimators truncated at zero (REML for this
    layout when non-negative). The grand mean's variance is ``s_L/L + s_M/M + s_R/(L M)``; t on ``L - 1`` df."""
    from scipy import stats

    y = np.asarray(matrix, dtype=float)
    L, M = y.shape
    if L < 2 or M < 2:
        return {"mean": float(y.mean()) if y.size else None, "low": None, "high": None, "lineages": L, "manifests": M,
                "threshold": float(threshold), "verdict": "inconclusive", "note": "fewer than two lineages or manifests"}
    grand, rows, cols = float(y.mean()), y.mean(axis=1), y.mean(axis=0)
    ms_l = M * float(((rows - grand) ** 2).sum()) / (L - 1)
    ms_m = L * float(((cols - grand) ** 2).sum()) / (M - 1)
    ms_e = float(((y - rows[:, None] - cols[None, :] + grand) ** 2).sum()) / ((L - 1) * (M - 1))
    s_r, s_l, s_m = ms_e, max(0.0, (ms_l - ms_e) / M), max(0.0, (ms_m - ms_e) / L)
    se = math.sqrt(s_l / L + s_m / M + s_r / (L * M))
    half = float(stats.t.ppf(0.5 + level / 2.0, L - 1)) * se
    low, high = grand - half, grand + half
    plain = float(stats.t.ppf(0.5 + level / 2.0, L - 1)) * float(rows.std(ddof=1)) / math.sqrt(L)
    return {"mean": grand, "se": se, "low": low, "high": high, "df": L - 1, "lineages": L, "manifests": M,
            "variance_components": {"lineage": s_l, "manifest": s_m, "residual": s_r},
            "lineage_sd": math.sqrt(s_l), "lineage_means": [float(v) for v in rows],
            "lineage_t_interval": [grand - plain, grand + plain],
            "threshold": float(threshold),
            "verdict": "supported" if low >= threshold else ("refuted" if high < threshold else "inconclusive")}


def spread(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)),
            "sd": float(v.std(ddof=1)) if v.size > 1 else None, "min": float(v.min()), "max": float(v.max())}


def difference_matrix(rows: dict, names: list, base: str = REFERENCE) -> np.ndarray:
    """``(len(names), manifests)`` of manifest-mean paired success differences, ``name`` minus ``base``."""
    return np.array([[float(np.mean([float(a["sustained"]) - float(b["sustained"]) for a, b in zip(m[n], m[base])]))
                      for m in rows.values()] for n in names])


def primary(rows: dict, cfg: dict, complete: list, missing: list) -> dict:
    """The registered contrast: each lineage's stage-D endpoint minus D-072, equal weight per lineage. A lineage
    with no stage-D endpoint enters as bounds (every case a failure, then every case a success), never as data."""
    thr = float(cfg["threshold"])
    names = [endpoint_name(k, "D") for k in complete]
    obs = difference_matrix(rows, names) if names else np.zeros((0, len(rows)))
    if not missing:
        return {**two_level(obs, thr), "missing_lineages": []}
    ref = np.array([float(np.mean([float(r["sustained"]) for r in m[REFERENCE]])) for m in rows.values()])
    lo = two_level(np.vstack([obs] + [0.0 - ref] * len(missing)), thr)
    hi = two_level(np.vstack([obs] + [1.0 - ref] * len(missing)), thr)
    if lo.get("low") is None:
        return {**lo, "missing_lineages": list(missing)}
    verdict = "supported" if lo["low"] >= thr else ("refuted" if hi["high"] < thr else "inconclusive")
    return {"mean": lo["mean"], "low": lo["low"], "high": hi["high"], "verdict": verdict, "threshold": thr,
            "missing_lineages": list(missing), "missing_as_all_failures": lo, "missing_as_all_successes": hi,
            "observed_only": two_level(obs, thr) if len(names) > 1 else None,
            "note": "bounds, not observations: a missing endpoint is scored first as all failures, then as all successes"}


def parent_of(k: int, stage: str) -> str:
    return REFERENCE if stage == "A" else endpoint_name(k, STAGES[STAGES.index(stage) - 1])


def qualification(rows: dict, cfg: dict, ends: dict) -> dict:
    """One Holm family: every available milestone against its construction parent, and every stage-D endpoint against
    D-072. A contrast qualifies with a Holm rejection, a strictly positive paired interval and the drop bound."""
    from lares.envs.isaac_pen_spin.incumbent_residual import confirmation_contrast

    pairs = [(endpoint_name(k, s), parent_of(k, s)) for k, have in ends.items() for s in STAGES if s in have]
    pairs += [(endpoint_name(k, "D"), REFERENCE) for k, have in ends.items() if "D" in have]
    out = {f"{a}_minus_{b}": confirmation_contrast({m: (r[a], r[b]) for m, r in rows.items()}, float(cfg["step_threshold"]),
                                                   float(cfg["drop_margin"])) for a, b in pairs}
    h = sr.holm({k: v["pooled"]["success"]["p_value"] for k, v in out.items()})
    for k, v in out.items():
        v["holm"] = h[k]
        v["qualifies"] = bool(h[k]["rejects"] and (v["pooled"]["success"]["low"] or 0) > 0 and v["pooled"]["drops"]["non_inferior"])
    return out


def historical_comparison(rows: dict, cfg: dict, complete: list) -> dict:
    """Descriptive only: each replicated stage-D endpoint against the frozen historical JOINT asset."""
    from lares.envs.isaac_pen_spin.incumbent_residual import confirmation_contrast

    out = {}
    for k in complete:
        c = confirmation_contrast({m: (r[endpoint_name(k, "D")], r[HISTORICAL]) for m, r in rows.items()}, 0.0, float(cfg["drop_margin"]))
        out[endpoint_name(k, "D")] = {"difference": c["pooled"]["success"]["mean"], "ci95": [c["pooled"]["success"]["low"], c["pooled"]["success"]["high"]],
                                      "manifest_clustered": [c["manifest_clustered"]["low"], c["manifest_clustered"]["high"]],
                                      "win_loss_tie": c["pooled"]["success_win_loss_tie"]}
    return out


def policy_table(rows: dict, names: list) -> dict:
    n = sum(len(m[REFERENCE]) for m in rows.values())
    out = {}
    for p in names:
        per = [float(np.mean([r["sustained"] for r in m[p]])) for m in rows.values()]
        out[p] = {"successes": int(sum(r["sustained"] for m in rows.values() for r in m[p])),
                  "success": float(np.mean([r["sustained"] for m in rows.values() for r in m[p]])),
                  "drops": int(sum(not r["completed"] for m in rows.values() for r in m[p])), "cases": n,
                  "manifest_min_max": [float(min(per)), float(max(per))],
                  "labels": {name: int(sum(ta.label(r, LABEL_CFG) == name for m in rows.values() for r in m[p])) for name in ta.LABELS}}
    return out


def analyse(rows: dict, cfg: dict, ends: dict) -> dict:
    """``rows[manifest][policy]`` and ``ends[k]`` (the stages lineage ``k`` froze) -> every reported quantity."""
    n_lin = len(cfg["lineages"])
    complete = [k for k in range(n_lin) if "D" in ends.get(k, {})]
    missing = [k for k in range(n_lin) if k not in complete]
    names = [REFERENCE, HISTORICAL] + [endpoint_name(k, s) for k in range(n_lin) for s in STAGES if s in ends.get(k, {})]
    table = policy_table(rows, names)
    head = primary(rows, cfg, complete, missing)
    milestones = {}
    for s in STAGES:
        ks = [k for k in range(n_lin) if s in ends.get(k, {})]
        if ks:
            d = difference_matrix(rows, [endpoint_name(k, s) for k in ks]).mean(axis=1)
            milestones[s] = {"lineages": ks, "success": spread([table[endpoint_name(k, s)]["success"] for k in ks]),
                             "gain_over_d072": spread(d), "drops": [table[endpoint_name(k, s)]["drops"] for k in ks],
                             "per_lineage_success": [table[endpoint_name(k, s)]["success"] for k in ks]}
    qual = qualification(rows, cfg, {k: v for k, v in ends.items() if v})
    return {"paired_success_difference": head["mean"], "paired_success_difference_ci95": [head["low"], head["high"]],
            "primary": head, "milestones": milestones, "policies": table, "qualification": qual,
            "qualified": {k: v["qualifies"] for k, v in qual.items()},
            "all_available_contrasts_qualify": bool(qual) and all(v["qualifies"] for v in qual.values()),
            "replicated_joint_minus_historical_joint": historical_comparison(rows, cfg, complete),
            "complete_lineages": complete, "missing_lineages": missing}


# ---------------------------------------------------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------------------------------------------------


class Ledger:
    """Charged episode starts, wall time and every child attempt, persisted so a resumed driver keeps the totals."""

    def __init__(self, path: str, ceiling: int, max_minutes: float):
        self.path, self.ceiling, self.max_s = path, int(ceiling), 60.0 * float(max_minutes)
        self.data = rm._load(path) if os.path.exists(path) else {"starts": {}, "wall_s": 0.0, "attempts": []}

    def save(self) -> None:
        rm._dump(self.path, self.data)

    @property
    def charged(self) -> int:
        return int(sum(self.data["starts"].values()))

    def charge(self, key: str, starts: int, wall_s: float = 0.0) -> None:
        self.data["starts"][key] = int(self.data["starts"].get(key, 0)) + int(starts)
        self.data["wall_s"] = float(self.data["wall_s"]) + float(wall_s)
        self.save()

    def attempt(self, record: dict) -> None:
        self.data["attempts"].append(record)
        self.save()

    def room(self, need: int) -> bool:
        return self.charged + int(need) <= self.ceiling and float(self.data["wall_s"]) < self.max_s


def _log_starts(log: str) -> int:
    """Starts a dead training child had made, from its last ``update`` line (0 when it never finished one)."""
    last = 0
    if os.path.exists(log):
        with open(log, encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("update "):
                    try:
                        last = int(json.loads(line[7:])["starts_so_far"])
                    except (ValueError, KeyError):
                        pass
    return last


def run_child(kind: str, request: dict, stem: str, ledger: Ledger, key: str) -> dict | None:
    """One Isaac child. A child that dies without a response (infrastructure) is rerun with the identical request, at
    most ``MAX_RERUNS`` times, each attempt logged and its starts charged. ``None`` when every attempt died."""
    if os.path.exists(request["response_path"]):
        print("exists", request["response_path"], flush=True)
        return rm._load(request["response_path"])
    rm._dump(stem + ".request.json", request)
    for attempt in range(1 + MAX_RERUNS):
        log = f"{stem}.log" if attempt == 0 else f"{stem}.attempt{attempt + 1}.log"
        cmd = [sys.executable, "-u", "-m", CHILDREN[kind], stem + ".request.json"]
        print("running", " ".join(cmd), f"(attempt {attempt + 1})", flush=True)
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as f:
            try:
                code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=4 * 3600).returncode
            except subprocess.TimeoutExpired:
                code = "timeout"
        wall = time.time() - t0
        ok = code == 0 and os.path.exists(request["response_path"])
        ledger.attempt({"key": key, "attempt": attempt + 1, "exit": code, "ok": ok, "wall_s": wall, "log": log,
                        "at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        if ok:
            out = rm._load(request["response_path"])
            ledger.charge(key, int(out["starts"]), wall)
            return out
        ledger.charge(key + ":died", _log_starts(log), wall)
        if os.path.exists(request["response_path"]):
            os.remove(request["response_path"])
    return None


def run_lineage(cfg: dict, base: dict, out_dir: str, k: int, seeds: dict, ledger: Ledger, reserve: int) -> dict:
    """The four stages of one lineage in order; a stage that diverges, dies for good or meets a ceiling ends it."""
    sub = lineage_dir(out_dir, k)
    os.makedirs(sub, exist_ok=True)
    path = os.path.join(sub, "endpoints.json")
    state = rm._load(path) if os.path.exists(path) else {"lineage": k, "seeds": seeds, "endpoints": {}, "training": {}, "ended": None,
                                                        "rule": "each stage's final ARS mean; no selection, no rerun of a seed"}
    root, parent = None, None
    for stage in STAGES:
        if state["ended"]:
            break
        if stage in state["endpoints"]:
            parent = state["endpoints"][stage]
        else:
            need = 2 * scored_per_update(cfg) * updates(cfg, stage)
            if not ledger.room(need + reserve):
                state["ended"] = {"stage": stage, "reason": "the start or wall-time ceiling leaves no room for this stage"}
                rm._dump(path, state)
                break
            key = endpoint_name(k, stage)
            stem = os.path.join(sub, f"train_{stage}")
            req = {**base, "mode": "train", "out_dir": sub, "start_cap": ledger.ceiling - ledger.charged - reserve,
                   "response_path": stem + ".json", **stage_request(cfg, stage, seeds[stage], parent, root)}
            tr = run_child(stage, req, stem, ledger, key)
            why = "the child died on every attempt (infrastructure)" if tr is None else (
                (tr.get("note") or "incomplete") if not tr["complete"] else divergent(stage, tr, float(cfg["actor"]["bound"])))
            if tr is not None and why is None and len(tr["updates"]) * scored_per_update(cfg) != updates(cfg, stage) * scored_per_update(cfg):
                why = "the stage did not score the archived number of episodes"
            if why:
                state["ended"] = {"stage": stage, "reason": why}
                rm._dump(path, state)
                break
            parent = endpoint_record(cfg, stage, tr)
            state["endpoints"][stage] = parent
            state["training"][stage] = {"seed": seeds[stage], "starts": tr["starts"], "transitions": tr["transitions"],
                                        "scored_episodes": len(tr["updates"]) * scored_per_update(cfg), "wall_time_s": tr["wall_time_s"],
                                        "mean_candidate_success": [u["mean_candidate_success"] for u in tr["updates"]]}
            rm._dump(path, state)
        if stage == "A":
            freeze_json(root_path(out_dir, k), {**parent, "rule": "the lineage's stage-A terminal mean; the root of its stages B to D"},
                        "stage-A finalist")
            root = {"path": root_path(out_dir, k), **parent}
    return state


def evaluate(cfg: dict, config_path: str, out_dir: str, ends: dict, ledger: Ledger) -> tuple:
    """Every available endpoint, D-072 and the historical JOINT asset on every manifest, one fresh process each."""
    names = [REFERENCE, HISTORICAL] + [endpoint_name(k, s) for k in sorted(ends) for s in STAGES if s in ends[k]]
    rows, failed = {}, []
    for s in (int(v) for v in cfg["confirmation_seeds"]):
        mpath = cases.require_frozen(manifest_for(cfg, s))
        sub = os.path.join(out_dir, "confirmation", f"s{s}")
        os.makedirs(sub, exist_ok=True)
        rows[f"s{s}"] = {}
        for name in names:
            cell = os.path.join(sub, f"{name}_cell.json")
            for attempt in range(1 + MAX_RERUNS):
                if os.path.exists(cell) or float(ledger.data["wall_s"]) >= ledger.max_s:
                    break
                t0 = time.time()
                try:
                    study_cell(cfg, config_path, mpath, name, cell, record_obs=False)
                    ok = os.path.exists(cell)
                except subprocess.CalledProcessError:
                    ok = False
                ledger.attempt({"key": f"s{s}/{name}", "attempt": attempt + 1, "ok": ok, "wall_s": time.time() - t0})
                ledger.charge("confirmation" if ok else "confirmation:died", 2 * int(cfg["confirmation_cases"]), time.time() - t0)
            if not os.path.exists(cell):
                failed.append(f"s{s}/{name}")
                continue
            params = rm._load(cell)["controller_params"]
            if name[0] == "L":
                k, stage = int(name[1:].split("_")[0]), name.split("_")[1]
                want = ends[k]["A"]["theta_sha256"] if stage == "A" else ends[k][stage]["sha256"]
                got = params["theta_sha256"] if stage == "A" else params["module"]["sha256"]
                if got != want or (stage != "A" and params["root_theta_sha256"] != ends[k]["A"]["theta_sha256"]):
                    raise SystemExit(f"{cell}: the evaluated policy is not the frozen endpoint")
            rows[f"s{s}"][name] = load_cell(cell)[1]
    return rows, failed


def run(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    cfg = protocol_cfg
    if os.path.normpath(out_dir) != os.path.normpath(str(cfg["log_dir"])):
        raise SystemExit("evaluation cells read the endpoints from the protocol's own log_dir; do not override it")
    os.makedirs(out_dir, exist_ok=True)
    descriptor = sr.check_reference(cfg)
    ra.normalization(cfg)
    historical_x(cfg)
    lins = lineages(cfg)
    cfg_sha = sr.config_sha256(cfg)
    per_update = {"signed_candidates": 2 * 4, "placements_per_candidate": int(cfg["cases_per_slot"]), "rollouts": ROLLOUTS,
                  "environments_per_rollout": SLOTS * int(cfg["cases_per_slot"]), "scored_episodes": scored_per_update(cfg),
                  "archived": cfg["archived_accounting"]}
    freeze_json(os.path.join(out_dir, "frozen_config.json"),
                {"config_sha256": cfg_sha, "config": sr.frozen_config(cfg), "reference": descriptor, "lineages": lins,
                 "per_update": per_update, "placements": placement_provenance(cfg),
                 "joint_coord_scale": [float(v) for v in joint_scale(cfg)], "script": rm.x_record(rm.SCRIPT_X),
                 "policies": policies(cfg), "manifests": [cases.require_frozen(manifest_for(cfg, s)) for s in cfg["confirmation_seeds"]]},
                "frozen configuration")
    ledger = Ledger(os.path.join(out_dir, "accounting.json"), int(cfg["start_ceiling"]), float(cfg["max_minutes"]))
    base = {"config_path": os.path.abspath(config_path), "study": STUDY, "config_sha256": cfg_sha}
    summary = {"study": STUDY, "config_sha256": cfg_sha, "lineages": lins}

    smoke = run_child("smoke", {**base, "mode": "smoke", "out_dir": out_dir, "response_path": os.path.join(out_dir, "smoke.json")},
                      os.path.join(out_dir, "smoke"), ledger, "smoke")
    summary["smoke"] = smoke
    train_scored = scored_per_update(cfg) * sum(updates(cfg, s) for s in STAGES) * len(lins)
    cells = len(policies(cfg)) * len(cfg["confirmation_seeds"])
    reserve = 2 * cells * int(cfg["confirmation_cases"])
    if smoke is not None:
        projected = (train_scored * HORIZON / smoke["throughput"]["transitions_per_s"] + 4 * len(lins) * 60.0 + cells * 35.0) / 60.0
        summary["projected_minutes"] = projected
    ok = smoke is not None and bool(smoke["passes"]) and projected <= float(cfg["max_minutes"])
    summary["smoke_gate"] = "pass" if ok else "fail"
    if not ok:
        summary["note"] = "the smoke gate failed: no training or confirmation ran"
        rm._dump(os.path.join(out_dir, "summary.json"), summary)
        return summary

    states = [run_lineage(cfg, base, out_dir, k, seeds, ledger, reserve) for k, seeds in enumerate(lins)]
    ends = {k: st["endpoints"] for k, st in enumerate(states)}
    rows, failed = evaluate(cfg, config_path, out_dir, ends, ledger)
    summary["training"] = {f"lineage_{k}": {"seeds": st["seeds"], "ended": st["ended"], "stages": st["training"],
                                            "endpoints": st["endpoints"]} for k, st in enumerate(states)}
    summary["accounting"] = {"starts": ledger.data["starts"], "charged_total": ledger.charged, "ceiling": ledger.ceiling,
                             "wall_minutes": ledger.data["wall_s"] / 60.0, "max_minutes": float(cfg["max_minutes"]),
                             "training_scored_episodes": int(sum(v["scored_episodes"] for st in states for v in st["training"].values())),
                             "training_transitions": int(sum(v["transitions"] for st in states for v in st["training"].values())),
                             "confirmation_cells": cells - len(failed), "reruns": [a for a in ledger.data["attempts"] if a["attempt"] > 1 or not a["ok"]],
                             "llm_calls": 0}
    if failed:
        summary["incomplete"] = f"cells with no result after {1 + MAX_RERUNS} attempts: {failed}; no inference"
        rm._dump(os.path.join(out_dir, "summary.json"), summary)
        return summary
    summary.update(analyse(rows, cfg, ends))
    summary["limitation"] = ("five training lineages on one task, simulator and reference: algorithm-level precision is that of "
                             "five runs (4 df), whatever the number of evaluation cases")
    rm._dump(os.path.join(out_dir, "summary.json"), summary)
    return summary


# ---------------------------------------------------------------------------------------------------------------------
#  Isaac child: the four-rollout smoke gate
# ---------------------------------------------------------------------------------------------------------------------


def _smoke(req: dict, cfg: dict, device: str) -> dict:
    """One rollout per stage on the smoke placements, through the same constructions training and evaluation use:
    A the zero residual is the D-072 reference bit for bit; B a disabled module on a *derived* root (a probe residual
    saved as a lineage's stage-A finalist) is that residual's controller bit for bit; C/D the joint policy's initial
    point is its scripted parent on that root; D per-slot candidates act, and equal slots agree."""
    import torch

    from lares.envs.isaac_pen_spin import success_ars as sa
    from lares.envs.isaac_pen_spin.runner import ControllerContext
    from lares.envs.isaac_pen_spin.success_ars_child import rollout as residual_rollout
    from lares.envs.isaac_pen_spin.symbolic_reward_child import make_training_env

    per, seed = int(cfg["cases_per_slot"]), int(cfg["smoke_seed"])
    n = SLOTS * per
    lay = {"slots": SLOTS, "per": per}
    env = make_training_env(n, device, seed)
    out = {"mode": "smoke", "seed": seed, "gates": {}}
    try:
        ctx = ControllerContext(env)
        out["gates"]["joint_order"] = {"passes": sr.executed_joint_order(ctx) == list(sr.JOINTS)}
        vecs = sa.tile(cases.placements(seed, per), SLOTS)
        norm = ra.normalization(cfg)
        t0 = time.time()
        # A: the residual family at zero is the reference
        ctrl = sr.build_controller(cfg, "smoke_A", None, norm)
        a = residual_rollout(env, ctx, ctrl, vecs, torch.zeros((n, sr.N_COEF), device=device),
                             plain=sr.build_reference(cfg, "smoke_plain", resettable=False))
        sus = a["sustained"].reshape(SLOTS, per)
        out["gates"]["zero_theta_identity"] = {**a["identity"], "passes": a["identity"]["mismatches"] == 0 and a["identity"]["compared"] > 0}
        out["gates"]["scorer_live"] = {"reference_successes": int(sus.sum()), "passes": bool(sus.any())}
        out["aa"] = sa.aa_statistics(sus)
        # B: a derived root, written and read as a lineage's stage-A finalist is
        probe = sr.theta_record(0.05 * np.random.default_rng(seed).choice([-1.0, 1.0], size=sr.N_COEF))
        sub = os.path.join(req["out_dir"], "smoke_lineage")
        os.makedirs(sub, exist_ok=True)
        rpath = os.path.join(sub, "stage_A_finalist.json")
        rm._dump(rpath, {"theta": probe, "theta_sha256": sr.theta_sha256(probe)})
        lcfg = lineage_cfg(cfg, rpath, sr.theta_sha256(probe))
        off = rm.build(lcfg, "smoke_off", rm.SCRIPT_X, enabled=False, original_bounds=True)
        b = rm._rollout(env, ctx, off, vecs, rm.SCRIPT_X, plain=sr.build_controller(cfg, "smoke_root", probe, norm))
        out["gates"]["derived_root_identity"] = {**b["identity"], "triggered": b["triggered"],
                                                 "root_theta_sha256": off.params["root_theta_sha256"],
                                                 "passes": b["identity"]["mismatches"] == 0 and b["identity"]["compared"] > 0
                                                 and b["triggered"] == 0 and off.params["root_theta_sha256"] == sr.theta_sha256(probe)}
        # C/D: the joint policy's initial point is its parent module on the same root
        joint = joint_initial(cfg, rm.SCRIPT_X, probe)
        c = rm._rollout(env, ctx, rm.build(lcfg, "smoke_joint", joint), vecs, joint,
                        plain=rm.build(lcfg, "smoke_parent", rm.SCRIPT_X, original_bounds=True), plain_tol=rm.SCRIPT_TOLERANCE)
        out["gates"]["joint_inheritance"] = {**c["identity"], "triggered": c["triggered"],
                                             "passes": c["identity"]["mismatches"] == 0 and c["identity"]["compared"] > 0 and c["triggered"] > 0}
        # D: per-slot candidates through the training path
        moved = joint.copy()
        moved[0] = 1.0                                       # threshold at its widened upper bound
        moved[rm.DIM:rm.DIM + 8] = np.clip(moved[rm.DIM:rm.DIM + 8] + 0.25, 0.0, 1.0)
        cand = np.stack([joint, moved, joint, moved])
        d = rm._rollout(env, ctx, rm.build(lcfg, "smoke_slots", joint), vecs, sa.slot_thetas(cand, per))
        ss = d["sustained"].reshape(SLOTS, per).mean(axis=1)
        out["gates"]["per_slot_candidates"] = {"slot_success": [float(v) for v in ss], "triggered": d["triggered"],
                                               "equal_slots_max_difference": float(max(abs(ss[0] - ss[2]), abs(ss[1] - ss[3]))),
                                               "passes": bool(max(abs(ss[0] - ss[2]), abs(ss[1] - ss[3])) <= float(cfg["smoke_aa_tolerance"])
                                                              and d["triggered"] > c["triggered"])}
        out["scripted_on_probe_root"] = {"success": float(c["sustained"].mean()), "drops": int((~c["completed"]).sum())}
        out["starts"] = a["starts"] + b["starts"] + c["starts"] + d["starts"]
        steps = a["env_steps"] + b["env_steps"] + c["env_steps"] + d["env_steps"]
        out["throughput"] = {"transitions_per_s": steps / (time.time() - t0), "wall_time_s": time.time() - t0}
        out["passes"] = all(g["passes"] for g in out["gates"].values())
        return out
    finally:
        env.close()


def child_main(request_path: str) -> None:
    from lares.envs.isaac_lab_adapter import _ensure_isaaclab_on_path, close_isaac_app, launch_isaac_app
    from lares.utils.cli import load_yaml

    req = rm._load(request_path)
    full = load_yaml(req["config_path"])
    cfg = full["research_protocols"][req["study"]]
    device = str(full["device"])
    _ensure_isaaclab_on_path(str(full["isaaclab_root"]))
    launch_isaac_app(headless=True, device=device)
    code = 0
    try:
        if sr.config_sha256(cfg) != req["config_sha256"]:
            raise SystemExit("the config block changed since the driver froze it")
        rm._dump(req["response_path"], _smoke(req, cfg, device))
        print("wrote", req["response_path"], flush=True)
    except BaseException:  # Isaac's close swallows tracebacks: print before it runs
        traceback.print_exc()
        sys.stdout.flush()
        code = 1
    finally:
        if code:
            os._exit(code)
        close_isaac_app()


__all__ = ["HISTORICAL", "REFERENCE", "STAGES", "STUDY", "Ledger", "analyse", "build_manifests", "difference_matrix", "divergent",
           "endpoint_name", "endpoint_record", "joint_initial", "joint_scale", "lineage_cfg", "lineages", "make_controller",
           "placement_provenance", "policies", "primary", "qualification", "run", "stage_overrides", "stage_request", "two_level"]

if __name__ == "__main__":
    child_main(sys.argv[1])
