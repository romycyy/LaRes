"""Research protocols on the pen-spin task, driven by ``scripts/run_shadowhand_spin_stages.py``.

``episode_rule_audit`` (R-038): ten frozen probe cases per stage, two diagnostic controllers, and both
episode reducers scored offline from the same trajectories. A second rollout in one process is not
reproducible (see ``runner``), so the driver runs each (stage, controller) in its own child process
(``--stage --controller``) and aggregates their JSON afterwards.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

from lares.envs.isaac_pen_spin import cases

EPISODE_RULE_AUDIT = "episode_rule_audit"


def audit_manifest_paths(protocol_cfg: dict) -> dict:
    return {int(k): v for k, v in protocol_cfg["manifests"].items()}


def build_audit_manifests(protocol_cfg: dict) -> list[str]:
    """Write the probe manifests the protocol names; refuses to overwrite a different one."""
    from lares.eval.manifest import EvaluationManifest

    written = []
    n = int(protocol_cfg["cases_per_stage"])
    for stage, path in audit_manifest_paths(protocol_cfg).items():
        seed = int(protocol_cfg["pool_seeds"][stage])
        m = cases.build_manifest(stage, seed, n, label="probe")
        if os.path.exists(path):
            old = EvaluationManifest.load(path)
            if old.to_dict() != m.to_dict():
                raise ValueError(f"{path} exists and differs from the manifest this protocol builds")
            continue
        m.save(path)
        written.append(path)
    return written


def run_audit_cell(stage: int, controller: str, protocol_cfg: dict, out_dir: str, device: str) -> str:
    """Child process body: one controller on one stage's probe manifest (one run per process)."""
    from lares.envs.isaac_pen_spin.controllers import AUDIT_CONTROLLERS
    from lares.envs.isaac_pen_spin.runner import PenSpinBatchRunner, reduce_episodes
    from lares.eval.manifest import EvaluationManifest

    ctrl = {c.name: c for c in AUDIT_CONTROLLERS}[controller]()
    manifest = EvaluationManifest.load(audit_manifest_paths(protocol_cfg)[stage])
    runner = PenSpinBatchRunner(stage, len(manifest), device=device)
    try:
        trace = runner.run(manifest, ctrl)
    finally:
        runner.close()
    np.savez_compressed(
        os.path.join(out_dir, f"trace_stage{stage}_{controller}.npz"),
        **{k: trace[k] for k in ("length", "completed", "pred", "center_dist", "spin_rate",
                                 "transverse_rate", "reward")},
    )
    result = {"stage": stage, "controller": controller, "manifest_id": manifest.manifest_id,
              "max_episode_length": int(trace["max_episode_length"]), "episodes": reduce_episodes(trace)}
    path = _cell_path(out_dir, stage, controller)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return path


def _cell_path(out_dir: str, stage: int, controller: str) -> str:
    return os.path.join(out_dir, f"stage{stage}_{controller}.json")


def aggregate_audit(out_dir: str, stages=(1, 2, 3)) -> dict:
    """Both reducers per stage and controller, and the pooled paired difference sustained - any_hit."""
    from lares.eval.stats import t_summary

    from lares.envs.isaac_pen_spin.controllers import AUDIT_CONTROLLERS

    per_cell, diffs, any_all, sus_all = {}, [], [], []
    for stage in stages:
        for ctrl in (c.name for c in AUDIT_CONTROLLERS):
            with open(_cell_path(out_dir, stage, ctrl), encoding="utf-8") as f:
                eps = json.load(f)["episodes"]
            a = [float(e["any_hit"]) for e in eps]
            s = [float(e["sustained"]) for e in eps]
            per_cell[f"stage{stage}/{ctrl}"] = {
                "n": len(eps),
                "any_hit_rate": float(np.mean(a)),
                "sustained_rate": float(np.mean(s)),
                "completed_rate": float(np.mean([e["completed"] for e in eps])),
                "mean_length": float(np.mean([e["length"] for e in eps])),
                "mean_hit_steps": float(np.mean([e["hit_steps"] for e in eps])),
                "mean_tail_hit_fraction": float(np.mean([e["tail_hit_fraction"] for e in eps])),
                "mean_abs_spin": float(np.mean([e["mean_abs_spin"] for e in eps])),
                "max_abs_spin": float(np.max([e["max_abs_spin"] for e in eps])),
                "mean_transverse": float(np.mean([e["mean_transverse"] for e in eps])),
                "first_hit_steps": [e["first_hit_step"] for e in eps],
            }
            diffs += [si - ai for si, ai in zip(s, a)]
            any_all += a
            sus_all += s
    return {
        "per_stage_controller": per_cell,
        "episodes": len(diffs),
        "success_rate_any_hit": float(np.mean(any_all)),
        "success_rate_sustained": float(np.mean(sus_all)),
        "paired_difference_sustained_minus_any_hit": t_summary(diffs),
    }


def run_episode_rule_audit(protocol_cfg: dict, config_path: str, out_dir: str, device: str) -> dict:
    from lares.envs.isaac_pen_spin.controllers import AUDIT_CONTROLLERS

    os.makedirs(out_dir, exist_ok=True)
    stages = tuple(sorted(audit_manifest_paths(protocol_cfg)))
    for stage in stages:
        for ctrl in (c.name for c in AUDIT_CONTROLLERS):
            if os.path.exists(_cell_path(out_dir, stage, ctrl)):
                continue
            cmd = [sys.executable, "-u", sys.argv[0], "--config", config_path, "--research-protocol",
                   EPISODE_RULE_AUDIT, "--stage", str(stage), "--controller", ctrl, "--log-dir", out_dir]
            print("running", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
    summary = aggregate_audit(out_dir, stages)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------------------
#  R-039: scripted expert qualification
# ---------------------------------------------------------------------------

SCRIPTED_EXPERT_QUALIFICATION = "scripted_expert_qualification"
QUALIFY_MIN = 27  # of 30 promotion cases per stage
SCREEN = (0, 10)
PROMOTION = (10, 40)


def controller_factory(name: str):
    from lares.envs.isaac_pen_spin.controllers import AUDIT_CONTROLLERS
    from lares.envs.isaac_pen_spin.scripted_expert import FeedbackGait, frozen_gait

    table = {c.name: c for c in AUDIT_CONTROLLERS}
    table[FeedbackGait.name] = frozen_gait
    return table[name]()


def build_qualification_manifests(protocol_cfg: dict) -> list[str]:
    from lares.eval.manifest import EvaluationManifest

    written = []
    specs = [(int(k), v, int(protocol_cfg["pool_seeds"][k]), "development") for k, v in protocol_cfg["manifests"].items()]
    specs.append((None, None, int(protocol_cfg["control_seed"]), "debug"))
    for stage, path, seed, label in specs:
        stages = [stage] if stage is not None else [int(k) for k in protocol_cfg["manifests"]]
        for st in stages:
            p = path or os.path.join(protocol_cfg["log_dir"], "control_manifests", f"pen_spin_stage{st}_debug_s{seed}.yaml")
            m = cases.build_manifest(st, seed, int(protocol_cfg["cases_per_stage"]), label=label)
            if os.path.exists(p):
                if EvaluationManifest.load(p).to_dict() != m.to_dict():
                    raise ValueError(f"{p} exists and differs from the manifest this protocol builds")
                continue
            m.save(p)
            written.append(p)
    return written


def run_cell(manifest_path: str, controller: str, out_path: str, device: str, record_obs: bool = False) -> str:
    """Child process body: one controller on one whole manifest, saved as JSON plus an .npz trace."""
    from lares.envs.isaac_pen_spin.runner import PenSpinBatchRunner, reduce_episodes
    from lares.eval.manifest import EvaluationManifest
    from lares.utils.hashing import sha256_file

    manifest = EvaluationManifest.load(manifest_path)
    ctrl = controller_factory(controller)
    runner = PenSpinBatchRunner(cases.stage_of(manifest), len(manifest), device=device)
    try:
        trace = runner.run(manifest, ctrl, record_obs=record_obs)
    finally:
        runner.close()
    keys = ["length", "completed", "pred", "center_dist", "spin_rate", "transverse_rate", "reward"]
    if record_obs:
        keys += ["obs", "actions"]
    np.savez_compressed(out_path.replace(".json", ".npz"), **{k: trace[k] for k in keys})
    result = {
        "stage": cases.stage_of(manifest), "controller": controller, "manifest_id": manifest.manifest_id,
        "manifest_path": manifest_path, "manifest_sha256": sha256_file(manifest_path),
        "controller_params": getattr(ctrl, "params", None),
        "reducer": "sustained: time-out without a fall and the per-step predicate on all final "
                   f"{cases.SUSTAIN_WINDOW} steps (D-034)",
        "max_episode_length": int(trace["max_episode_length"]), "episodes": reduce_episodes(trace),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return out_path


def _child(config_path: str, protocol: str, manifest_path: str, controller: str, out_path: str, record_obs: bool):
    cmd = [sys.executable, "-u", sys.argv[0], "--config", config_path, "--research-protocol", protocol,
           "--cell-manifest", manifest_path, "--controller", controller, "--cell-out", out_path]
    if record_obs:
        cmd.append("--record-obs")
    print("running", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def reproducibility_control(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    """Each (stage, controller) on the debug manifest in two fresh processes; spin and centre must be bit-identical."""
    ctrl_dir = os.path.join(out_dir, "control")
    os.makedirs(ctrl_dir, exist_ok=True)
    report = {}
    for stage in sorted(int(k) for k in protocol_cfg["manifests"]):
        mpath = os.path.join(protocol_cfg["log_dir"], "control_manifests",
                             f"pen_spin_stage{stage}_debug_s{int(protocol_cfg['control_seed'])}.yaml")
        for ctrl in protocol_cfg["controllers"]:
            outs = []
            for rep in ("a", "b"):
                out = os.path.join(ctrl_dir, f"stage{stage}_{ctrl}_{rep}.json")
                if not os.path.exists(out):
                    _child(config_path, SCRIPTED_EXPERT_QUALIFICATION, mpath, ctrl, out, False)
                outs.append(np.load(out.replace(".json", ".npz")))
            same = all(np.array_equal(outs[0][k], outs[1][k], equal_nan=True)
                       for k in ("spin_rate", "center_dist", "length", "completed"))
            report[f"stage{stage}/{ctrl}"] = bool(same)
    with open(os.path.join(ctrl_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def aggregate_qualification(protocol_cfg: dict, out_dir: str) -> dict:
    from lares.eval.stats import t_summary

    base, cand = protocol_cfg["baseline"], protocol_cfg["candidate"]
    per_stage, pooled = {}, []
    for stage in sorted(int(k) for k in protocol_cfg["manifests"]):
        rows = {}
        for ctrl in (base, cand):
            with open(os.path.join(out_dir, f"stage{stage}_{ctrl}.json"), encoding="utf-8") as f:
                rows[ctrl] = json.load(f)["episodes"]
        block = {}
        for split, (lo, hi) in (("screen", SCREEN), ("promotion", PROMOTION)):
            b = [float(e["sustained"]) for e in rows[base][lo:hi]]
            c = [float(e["sustained"]) for e in rows[cand][lo:hi]]
            d = [ci - bi for ci, bi in zip(c, b)]
            block[split] = {
                "n": len(d),
                f"{base}_success": float(np.mean(b)), f"{cand}_success": float(np.mean(c)),
                f"{cand}_successes": int(sum(c)),
                f"{base}_tail_hit_fraction": float(np.mean([e["tail_hit_fraction"] for e in rows[base][lo:hi]])),
                f"{cand}_tail_hit_fraction": float(np.mean([e["tail_hit_fraction"] for e in rows[cand][lo:hi]])),
                f"{cand}_completed": float(np.mean([e["completed"] for e in rows[cand][lo:hi]])),
                f"{cand}_mean_abs_spin": float(np.mean([e["mean_abs_spin"] for e in rows[cand][lo:hi]])),
                f"{cand}_mean_transverse": float(np.mean([e["mean_transverse"] for e in rows[cand][lo:hi]])),
                "paired_difference": t_summary(d),
            }
            if split == "promotion":
                pooled += d
        prom = block["promotion"]
        block["qualifies"] = {c: int(sum(float(e["sustained"]) for e in rows[c][PROMOTION[0]:PROMOTION[1]])) >= QUALIFY_MIN
                              for c in (base, cand)}
        lo = prom["paired_difference"]["low"]
        block["promotes"] = bool(prom["paired_difference"]["mean"] > 0 and lo is not None and lo > 0)
        per_stage[f"stage{stage}"] = block
    return {"per_stage": per_stage, "pooled_promotion_pairs": len(pooled),
            "paired_success_difference": float(np.mean(pooled)) if pooled else None,
            "paired_success_difference_t": t_summary(pooled)}


def run_scripted_expert_qualification(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    control = reproducibility_control(protocol_cfg, config_path, out_dir)
    if not all(control.values()):
        raise SystemExit(f"reproducibility control failed, evaluation not run: {control}")
    for stage in sorted(int(k) for k in protocol_cfg["manifests"]):
        mpath = protocol_cfg["manifests"][stage]
        for ctrl in protocol_cfg["controllers"]:
            out = os.path.join(out_dir, f"stage{stage}_{ctrl}.json")
            if not os.path.exists(out):
                _child(config_path, SCRIPTED_EXPERT_QUALIFICATION, mpath, ctrl, out, ctrl == protocol_cfg["candidate"])
    summary = {"reproducibility_control": control, **aggregate_qualification(protocol_cfg, out_dir)}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
