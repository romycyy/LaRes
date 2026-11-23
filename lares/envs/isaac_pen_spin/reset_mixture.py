"""R-133 (follow-up to D-089): does training on a mixture of nominal and expanded resets buy capture from wider resets?

Five paired blocks from the frozen R-121 JOINT controller, 36 ARS updates of its 43 coordinates each. The two arms of
a block share their optimizer draws and their latent placements and differ in one thing: ``nominal`` writes all 512
placements of an update as drawn; ``mixed`` writes the last 256 rows multiplied by 1.5 (the evaluator's reset-scale
semantics, ``recovery_module.mixed_rows``). Terminal means are frozen without selection. The ten endpoints and the
unchanged JOINT are evaluated on twelve fresh nominal manifests and twelve other fresh manifests under reset scale
1.5, one manifest per fresh process (cells are ``<policy>@<condition>``, as in ``frozen_robustness``).

    python scripts/run_shadowhand_spin_stages.py --study recovery_reset_mixture

The module is also its own Isaac child for the two-rollout smoke.
"""

from __future__ import annotations

import glob
import os
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
from lares.envs.isaac_pen_spin import training_replication as tr  # noqa: E402
from lares.envs.isaac_pen_spin.frozen_robustness import Conditioned  # noqa: E402
from lares.envs.isaac_pen_spin.protocols import freeze_json, load_cell, study_cell  # noqa: E402

STUDY = "recovery_reset_mixture"
ARMS = ("nominal", "mixed")
JOINT = "joint"
CONDITIONS = ("nominal", "expanded")
LABEL_CFG = {"early_drop_step": 60, "slow_hit_max": 0.2, "near_hit_min": 0.8}


def joint_x(cfg: dict) -> np.ndarray:
    entry = rm._load(cfg["joint_asset"])
    if entry["sha256"] != cfg["joint_sha256"] or rm.x_record(entry["x"], *rm.bounds(cfg))["sha256"] != entry["sha256"]:
        raise SystemExit("the parent is not the frozen R-121 JOINT finalist")
    return np.asarray(entry["x"], dtype=float)


def blocks(cfg: dict) -> list[dict]:
    opt, pl = [int(s) for s in cfg["optimizer_seeds"]], [int(s) for s in cfg["placement_seeds"]]
    if len(opt) != len(pl) or len(set(opt + pl)) != len(opt) + len(pl):
        raise SystemExit("every block needs its own optimizer seed and its own placement seed")
    return [{"optimizer_seed": o, "placement_seed": p} for o, p in zip(opt, pl)]


def reset_mix(cfg: dict, arm: str) -> dict | None:
    return None if arm == "nominal" else {"rows": int(cfg["mixed_rows"]), "scale": float(cfg["expanded_scale"])}


def train_request(cfg: dict, block: dict, arm: str) -> dict:
    """One arm's training request: the JOINT parent, the block's shared streams, the arm's reset allocation."""
    return {"mode": "train", "init_x": [float(v) for v in joint_x(cfg)], "direction_dim": rm.DIM_JOINT,
            "coord_scale": [float(v) for v in tr.joint_scale(cfg)], "original_bounds": False,
            "placement_seed": block["placement_seed"], "reset_mix": reset_mix(cfg, arm),
            "cfg_overrides": {"training_seed": block["optimizer_seed"],
                              "ars": {"iterations": int(cfg["updates"]), "directions": 4, "nu": float(cfg["ars"]["nu"]), "alpha": float(cfg["ars"]["alpha"])}}}


def endpoint(k: int, arm: str) -> str:
    return f"b{k}_{arm}"


def policies(cfg: dict) -> list[str]:
    return [JOINT] + [endpoint(k, a) for k in range(len(cfg["optimizer_seeds"])) for a in ARMS]


def condition(cfg: dict, name: str) -> dict:
    return {} if name == "nominal" else {"reset_scale": float(cfg["expanded_scale"])}


def seeds_for(cfg: dict, cond: str) -> list[int]:
    return [int(s) for s in cfg[f"{cond}_confirmation_seeds"]]


def manifest_for(cfg: dict, seed) -> str:
    return str(cfg["confirmation_manifest_template"]).format(seed=int(seed))


def build_manifests(protocol_cfg: dict) -> list[str]:
    from lares.eval.manifest import EvaluationManifest, assert_disjoint

    mine = {manifest_for(protocol_cfg, s): cases.build_manifest(3, int(s), int(protocol_cfg["confirmation_cases"]), label="resetmix")
            for c in CONDITIONS for s in seeds_for(protocol_cfg, c)}
    paths = sorted(glob.glob("config/manifests/pen_spin_stage3_*.yaml") + glob.glob("config/manifests/shadowhand_spin_*.yaml"))
    assert_disjoint(list(mine.values()) + [EvaluationManifest.load(p) for p in paths if p not in mine])
    return [p for p, m in mine.items() if cases.freeze_manifest(p, m)]


def placement_provenance(cfg: dict) -> dict:
    from lares.envs.isaac_pen_spin.success_ars import placement_hashes
    from lares.eval.manifest import EvaluationManifest
    from lares.utils.hashing import sha256_text

    per = int(cfg["cases_per_slot"])
    hashes = [h for b in blocks(cfg) for h in placement_hashes(cases.placements(b["placement_seed"], per * int(cfg["updates"])))]
    hashes += placement_hashes(cases.placements(int(cfg["smoke_seed"]), per))
    committed = set()
    for p in sorted(glob.glob("config/manifests/pen_spin_*.yaml") + glob.glob("config/manifests/shadowhand_spin_*.yaml")):
        committed |= {c.task_hash for c in EvaluationManifest.load(p).episodes}
    if len(set(hashes)) != len(hashes) or committed & set(hashes):
        raise SystemExit("training or smoke placements collide with each other or with a committed manifest")
    return {"latent_placements": len(hashes), "sha256": sha256_text("".join(hashes)), "disjoint_from_committed_pen_spin_cases": len(committed),
            "shared": "both arms of a block draw the same latent placements and the same optimizer directions"}


def finalists(out_dir: str) -> dict:
    path = os.path.join(out_dir, "finalists.json")
    return rm._load(path)["finalists"] if os.path.exists(path) else {}


def make_controller(name: str, protocol_cfg: dict, out_dir: str | None = None):
    policy, _, cond = name.partition("@")
    if cond not in CONDITIONS or policy not in policies(protocol_cfg):
        raise ValueError(f"unknown cell {name!r}")
    if policy == JOINT:
        x = joint_x(protocol_cfg)
    else:
        entry = finalists(out_dir or str(protocol_cfg["log_dir"]))[policy]
        if rm.x_record(entry["x"], *rm.bounds(protocol_cfg))["sha256"] != entry["sha256"]:
            raise SystemExit("a stored endpoint does not match its hash")
        x = entry["x"]
    return Conditioned(rm.build(protocol_cfg, name, x), name, condition(protocol_cfg, cond))


# ---------------------------------------------------------------------------------------------------------------------
#  Inference
# ---------------------------------------------------------------------------------------------------------------------


def diff_matrix(rows: dict, a: list, b: list) -> np.ndarray:
    """``(len(a), manifests)`` of manifest-mean paired success differences, ``a[k]`` minus ``b[k]``."""
    return np.array([[float(np.mean([float(x["sustained"]) - float(y["sustained"]) for x, y in zip(m[p], m[q])])) for m in rows.values()]
                     for p, q in zip(a, b)])


def analyse(rows: dict, cfg: dict) -> dict:
    """``rows[condition][manifest][policy]`` -> the registered contrast, the usefulness rules and the tables."""
    from lares.envs.isaac_pen_spin.incumbent_residual import confirmation_contrast

    ks = range(len(cfg["optimizer_seeds"]))
    mixed, nominal = [endpoint(k, "mixed") for k in ks], [endpoint(k, "nominal") for k in ks]
    thr = float(cfg["threshold"])
    head = tr.two_level(diff_matrix(rows["expanded"], mixed, nominal), thr)
    head["contains_zero"] = bool(head["low"] <= 0.0 <= head["high"])
    on_nominal = tr.two_level(diff_matrix(rows["nominal"], mixed, nominal), 0.0)
    family, use = {}, {}
    for name in (*mixed, *nominal):
        c = {cond: confirmation_contrast({m: (r[name], r[JOINT]) for m, r in rows[cond].items()}, 0.0, float(cfg["drop_margin"])) for cond in CONDITIONS}
        e, n = c["expanded"]["pooled"], c["nominal"]["pooled"]
        family[f"{name}_minus_joint@expanded"] = {"mean": e["success"]["mean"], "ci95": [e["success"]["low"], e["success"]["high"]], "p_value": e["success"]["p_value"]}
        use[name] = {"expanded_gain_over_joint": [e["success"]["mean"], e["success"]["low"], e["success"]["high"]],
                     "nominal_difference_from_joint": [n["success"]["mean"], n["success"]["low"], n["success"]["high"]],
                     "nominal_non_inferior": bool(n["success"]["low"] > -float(cfg["nominal_margin"])),
                     "drops_non_inferior": bool(e["drops"]["non_inferior"] and n["drops"]["non_inferior"]),
                     "drop_upper_95": {"expanded": e["drops"]["one_sided_95_upper"], "nominal": n["drops"]["one_sided_95_upper"]}}
    h = sr.holm({k: (1.0 if v["p_value"] is None else v["p_value"]) for k, v in family.items()})
    for name in use:
        hk = h[f"{name}_minus_joint@expanded"]
        use[name].update(holm_p=hk["p_holm"], useful=bool(hk["rejects"] and use[name]["expanded_gain_over_joint"][1] > 0
                                                         and use[name]["nominal_non_inferior"] and use[name]["drops_non_inferior"]))
    table = {cond: {p: {"success": float(np.mean([r["sustained"] for m in rows[cond].values() for r in m[p]])),
                        "drops": int(sum(not r["completed"] for m in rows[cond].values() for r in m[p])),
                        "cases": int(sum(len(m[p]) for m in rows[cond].values())),
                        "labels": {lab: int(sum(ta.label(r, LABEL_CFG) == lab for m in rows[cond].values() for r in m[p])) for lab in ta.LABELS}}
                    for p in policies(cfg)} for cond in CONDITIONS}
    arms = {cond: {a: tr.spread([table[cond][endpoint(k, a)]["success"] for k in ks]) for a in ARMS} for cond in CONDITIONS}
    return {"paired_success_difference": head["mean"], "paired_success_difference_ci95": [head["low"], head["high"]], "primary": head,
            "mixed_minus_nominal_on_nominal_resets": on_nominal, "usefulness": use, "useful_mixed_endpoints": int(sum(use[p]["useful"] for p in mixed)),
            "policies": table, "arms": arms, "joint": {cond: table[cond][JOINT]["success"] for cond in CONDITIONS},
            "precision_note": "five paired training blocks: block-level intervals have 4 degrees of freedom"}


# ---------------------------------------------------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------------------------------------------------


def run(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    cfg = protocol_cfg
    if os.path.normpath(out_dir) != os.path.normpath(str(cfg["log_dir"])):
        raise SystemExit("evaluation cells read the endpoints from the protocol's own log_dir; do not override it")
    os.makedirs(out_dir, exist_ok=True)
    descriptor = sr.check_reference(cfg)
    ra.finalist_theta(cfg)
    ra.normalization(cfg)
    x0 = joint_x(cfg)
    cfg_sha = sr.config_sha256(cfg)
    bl = blocks(cfg)
    mpaths = {c: {s: cases.require_frozen(manifest_for(cfg, s)) for s in seeds_for(cfg, c)} for c in CONDITIONS}
    freeze_json(os.path.join(out_dir, "frozen_config.json"),
                {"config_sha256": cfg_sha, "config": sr.frozen_config(cfg), "reference": descriptor, "parent": rm.x_record(x0, *rm.bounds(cfg)),
                 "blocks": bl, "placements": placement_provenance(cfg), "policies": policies(cfg),
                 "allocation": {a: reset_mix(cfg, a) for a in ARMS}}, "frozen configuration")
    base = {"config_path": os.path.abspath(config_path), "study": STUDY, "config_sha256": cfg_sha}
    per_update = 2 * 4 * int(cfg["cases_per_slot"])
    cells = len(policies(cfg)) * sum(len(v) for v in mpaths.values())
    reserve = 2 * cells * int(cfg["confirmation_cases"])
    ceiling = int(cfg["start_ceiling"])
    summary = {"study": STUDY, "config_sha256": cfg_sha}
    t_start = time.time()

    smoke = rm.run_child({**base, "mode": "smoke", "out_dir": out_dir, "response_path": os.path.join(out_dir, "smoke.json")},
                         os.path.join(out_dir, "smoke"), module="lares.envs.isaac_pen_spin.reset_mixture")
    summary.update(smoke=smoke, smoke_gate="pass" if smoke["passes"] else "fail")
    if not smoke["passes"]:
        summary["note"] = "the smoke gate failed: no training or confirmation ran"
        rm._dump(os.path.join(out_dir, "summary.json"), summary)
        return summary
    started, training, ends = int(smoke["starts"]), {}, {}
    for k, block in enumerate(bl):
        for arm in ARMS:
            name = endpoint(k, arm)
            t = rm.run_child({**base, **train_request(cfg, block, arm), "start_cap": ceiling - reserve - started,
                              "response_path": os.path.join(out_dir, f"train_{name}.json")}, os.path.join(out_dir, f"train_{name}"))
            started += int(t["starts"])
            training[name] = {"starts": t["starts"], "transitions": t["transitions"], "wall_time_s": t["wall_time_s"], "complete": t["complete"],
                              "mean_candidate_success": [u["mean_candidate_success"] for u in t["updates"]]}
            if not t["complete"] or len(t["updates"]) != int(cfg["updates"]):
                summary.update(training=training, incomplete=f"{name}: {t.get('note')}; no confirmation")
                rm._dump(os.path.join(out_dir, "summary.json"), summary)
                return summary
            ends[name] = rm.x_record(t["final_x"], *rm.bounds(cfg))
    freeze_json(os.path.join(out_dir, "finalists.json"), {"rule": "each run's final ARS mean; no selection", "finalists": ends}, "finalists")

    rows = {c: {} for c in CONDITIONS}
    for cond in CONDITIONS:
        for s, mpath in mpaths[cond].items():
            sub = os.path.join(out_dir, "confirmation", cond, f"s{s}")
            os.makedirs(sub, exist_ok=True)
            rows[cond][f"s{s}"] = {}
            for p in policies(cfg):
                cell = os.path.join(sub, f"{p}_cell.json")
                study_cell(cfg, config_path, mpath, f"{p}@{cond}", cell, record_obs=False)
                saved = rm._load(cell)
                want = rm.x_record(x0, *rm.bounds(cfg))["sha256"] if p == JOINT else ends[p]["sha256"]
                if saved["controller_params"]["module"]["sha256"] != want or saved["condition"] != condition(cfg, cond):
                    raise SystemExit(f"{cell}: not the frozen policy under its condition")
                rows[cond][f"s{s}"][p] = load_cell(cell)[1]
    summary.update(analyse(rows, cfg))
    summary.update(training=training, finalists=ends,
                   accounting={"smoke_starts": smoke["starts"], "training_starts": int(sum(t["starts"] for t in training.values())),
                               "training_scored_episodes": len(training) * int(cfg["updates"]) * per_update,
                               "training_transitions": int(sum(t["transitions"] for t in training.values())),
                               "confirmation_cells": cells, "confirmation_episodes": cells * int(cfg["confirmation_cases"]),
                               "charged_total": started + reserve, "ceiling": ceiling,
                               "driver_wall_minutes_this_launch": (time.time() - t_start) / 60.0, "max_minutes": float(cfg["max_minutes"]), "llm_calls": 0},
                   limitation="five paired blocks from one frozen parent; one expanded scale (1.5); one mixture ratio")
    rm._dump(os.path.join(out_dir, "summary.json"), summary)
    return summary


# ---------------------------------------------------------------------------------------------------------------------
#  Isaac child: the two-rollout smoke
# ---------------------------------------------------------------------------------------------------------------------


def _smoke(req: dict, cfg: dict, device: str) -> dict:
    """Rollout 1: the JOINT parent on four identical nominal slots (it is the frozen asset; the scorer is live; the
    slots agree to within their noise). Rollout 2: the mixed allocation; the simulator's written pen offsets are read
    back before the rollout and the scaled rows start 1.5 times further from the nominal pose."""
    import torch

    from lares.envs.isaac_pen_spin import success_ars as sa
    from lares.envs.isaac_pen_spin.runner import ControllerContext, _write_placements
    from lares.envs.isaac_pen_spin.symbolic_reward_child import make_training_env

    per, seed = int(cfg["cases_per_slot"]), int(cfg["smoke_seed"])
    env = make_training_env(4 * per, device, seed)
    out = {"mode": "smoke", "gates": {}}
    try:
        ctx = ControllerContext(env)
        out["gates"]["joint_order"] = {"passes": sr.executed_joint_order(ctx) == list(sr.JOINTS)}
        x = joint_x(cfg)
        base = cases.placements(seed, per)
        ctrl = rm.build(cfg, "smoke", x)
        out["gates"]["parent_is_the_asset"] = {"sha256": ctrl.params["module"]["sha256"], "passes": ctrl.params["module"]["sha256"] == cfg["joint_sha256"]}
        t0 = time.time()
        a = rm._rollout(env, ctx, ctrl, sa.tile(base, 4), x)
        sus = a["sustained"].reshape(4, per)
        aa = sa.aa_statistics(sus)
        bound = float(cfg["smoke_aa_sd_multiplier"]) * float(np.sqrt(aa["mean_discordance"] / aa["cases"]))
        out["aa"] = aa
        out["gates"]["scorer_live"] = {"successes": int(sus.sum()), "passes": bool(sus.mean() > 0.5)}
        out["gates"]["equal_slots_agree"] = {"max_abs_slot_difference": aa["max_abs_slot_difference"], "noise_scaled_bound": bound,
                                             "passes": bool(aa["max_abs_slot_difference"] <= bound)}
        mix = reset_mix(cfg, "mixed")
        vecs = sa.tile(rm.mixed_rows(base, mix), 4)
        env.reset()
        _write_placements(env, vecs)
        off = (env.object.data.root_pos_w - env.scene.env_origins - env.object.data.default_root_state[:, :3]).cpu().numpy()
        want = float(env.cfg.reset_position_noise) * vecs[:, 0:3]
        rows = np.arange(4 * per) % per >= per - int(mix["rows"])
        ratio = float(np.abs(off[rows]).mean() / np.abs(off[~rows]).mean())
        out["gates"]["mixed_rows_written"] = {"max_abs_position_error": float(np.abs(off - want).max()), "scaled_rows": int(rows.sum()),
                                              "mean_offset_ratio_scaled_over_nominal": ratio,
                                              "passes": bool(np.abs(off - want).max() < 1e-5 and rows.sum() == 4 * int(mix["rows"])
                                                             and abs(ratio - float(mix["scale"])) < 0.15)}
        b = rm._rollout(env, ctx, ctrl, vecs, x)
        s_nom, s_exp = float(b["sustained"][~rows].mean()), float(b["sustained"][rows].mean())
        out["gates"]["expanded_rows_are_harder"] = {"nominal_rows_success": s_nom, "expanded_rows_success": s_exp, "passes": bool(s_exp < s_nom)}
        out["starts"] = a["starts"] + b["starts"] + 4 * per
        out["throughput"] = {"transitions_per_s": (a["env_steps"] + b["env_steps"]) / (time.time() - t0)}
        out["passes"] = all(g["passes"] for g in out["gates"].values())
        del torch
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


__all__ = ["ARMS", "CONDITIONS", "JOINT", "STUDY", "analyse", "blocks", "build_manifests", "condition", "diff_matrix", "endpoint", "joint_x",
           "make_controller", "placement_provenance", "policies", "reset_mix", "run", "train_request"]

if __name__ == "__main__":
    child_main(sys.argv[1])
