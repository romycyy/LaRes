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
