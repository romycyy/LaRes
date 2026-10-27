"""R-044: which state feedback does the qualified PPO expert rely on? Observation-clamping lesions on Stage 3.

D-037 removed all feedback at once (an action replay scored 0/40). This protocol keeps the policy closed loop and
holds one group of observation slices at the case's reset values (the first observation the controller sees) for the
whole episode, before inference; every other slice, including ``last_action``, stays live, and the simulator is not
touched. Held values come from a real observation, so the quaternions stay valid.

* ``pen_state_clamped``: the pen's position, orientation, linear and angular velocity and its goal-relative rotation.
* ``hand_state_clamped``: hand joint positions and velocities and the fingertip poses and velocities that re-encode them.

One frozen 40-case manifest (seed 4140). Arms in order: ``expert``, ``expert_repeat`` (the reproducibility gate: zero
sustained flips, at most 4/40 diverged environments; the lesions are not run otherwise), then the two lesions. Each
arm is its own Isaac process (D-034). Failures are split into drops (the episode ended by a fall) and tail-spin
failures (time-out reached, final-60-step predicate failed). Conclusions are limited to this checkpoint's dependence
under clamping, a distribution-shift intervention: not information-theoretic necessity, nor sufficiency.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin.feedback_ablation import _arm_summary, check_checkpoint, reproducibility_gate

FEEDBACK_LESIONS = "feedback_lesions"
STAGE = 3
LESIONS = {
    "pen_state_clamped": ("obj_pos", "obj_rot", "obj_linvel", "obj_angvel", "rel_rot"),
    "hand_state_clamped": ("hand_dof_pos", "hand_dof_vel", "fingertip_pos", "fingertip_rot", "fingertip_vel"),
}
ARMS = ("expert", "expert_repeat") + tuple(LESIONS)


def lesion_index(fields) -> np.ndarray:
    """Observation columns of the named slices."""
    sl = cases.obs_slices()
    return np.concatenate([np.arange(sl[f].start, sl[f].stop) for f in fields])


class LesionedController:
    """Wraps a controller; holds the named observation columns at their first-step values."""

    def __init__(self, inner, lesion: str):
        if lesion not in LESIONS:
            raise ValueError(f"unknown lesion {lesion!r}; one of {tuple(LESIONS)}")
        self.inner = inner
        self.lesion = lesion
        self.name = lesion
        self.columns = lesion_index(LESIONS[lesion])
        self.params = {"lesion": lesion, "fields": list(LESIONS[lesion]), "columns": int(len(self.columns)),
                       "hold": "the first observation of the episode", **getattr(inner, "params", {})}

    def attach(self, env):
        if hasattr(self.inner, "attach"):
            self.inner.attach(env)

    def reset(self, ctx):
        self._held = None
        self.inner.reset(ctx)

    def __call__(self, obs, t, ctx):
        import torch

        if self._held is None:
            self._held = obs.clone()
            self._cols = torch.as_tensor(self.columns, device=obs.device, dtype=torch.long)
        lesioned = obs.clone()
        lesioned[:, self._cols] = self._held[:, self._cols]
        return self.inner(lesioned, t, ctx)


def manifest_path(protocol_cfg: dict) -> str:
    return protocol_cfg["manifest"]


def build_manifests(protocol_cfg: dict) -> list[str]:
    from lares.eval.manifest import EvaluationManifest

    path = manifest_path(protocol_cfg)
    m = cases.build_manifest(STAGE, int(protocol_cfg["pool_seed"]), int(protocol_cfg["cases"]), label="lesion")
    if os.path.exists(path):
        if EvaluationManifest.load(path).to_dict() != m.to_dict():
            raise ValueError(f"{path} exists and differs from the manifest this protocol builds")
        return []
    m.save(path)
    return [path]


def make_controller(arm: str, protocol_cfg: dict):
    from lares.envs.isaac_pen_spin.ppo_expert import PPOController

    expert = PPOController(check_checkpoint(protocol_cfg), int(protocol_cfg["training_seed"]))
    if arm in ("expert", "expert_repeat"):
        return expert
    return LesionedController(expert, arm)


def _child(config_path: str, manifest: str, arm: str, out_path: str) -> None:
    if os.path.exists(out_path):
        print("exists", out_path, flush=True)
        return
    cmd = [sys.executable, "-u", sys.argv[0], "--config", config_path, "--study", FEEDBACK_LESIONS,
           "--cell-manifest", manifest, "--controller", arm, "--cell-out", out_path]
    print("running", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def failure_split(rows: list[dict]) -> dict:
    drops = [r for r in rows if not r["completed"]]
    tail = [r for r in rows if r["completed"] and not r["sustained"]]
    return {"drops": len(drops), "tail_spin_failures": len(tail),
            "median_fall_step": float(np.median([r["length"] for r in drops])) if drops else None}


def aggregate(rows: dict) -> dict:
    """Per-arm summaries and each lesion minus the expert, paired by case."""
    from lares.eval.stats import t_summary

    base = rows["expert"]
    out = {"arms": {}}
    for arm, rs in rows.items():
        if [r["task_hash"] for r in rs] != [r["task_hash"] for r in base]:
            raise ValueError(f"{arm} rows are not paired case for case with the expert")
        out["arms"][arm] = {**_arm_summary(rs), **failure_split(rs)}
    for lesion in LESIONS:
        if lesion in rows:
            d = [float(x["sustained"]) - float(e["sustained"]) for x, e in zip(rows[lesion], base)]
            out[f"{lesion}_minus_expert"] = t_summary(d)
    pen = out.get("pen_state_clamped_minus_expert")
    out["paired_success_difference"] = pen["mean"] if pen else None
    return out


def run_feedback_lesions(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    """Driver: expert twice, the gate, the two lesions, aggregate. Nothing is rerun if its output exists."""
    from lares.utils.hashing import sha256_file

    check_checkpoint(protocol_cfg)
    os.makedirs(out_dir, exist_ok=True)
    m = manifest_path(protocol_cfg)
    out = {arm: os.path.join(out_dir, f"{arm}.json") for arm in ARMS}
    for arm in ("expert", "expert_repeat"):
        _child(config_path, m, arm, out[arm])
    gate = reproducibility_gate(out["expert"], out["expert_repeat"])
    if not gate["passes"]:
        with open(os.path.join(out_dir, "gate_failed.json"), "w") as f:
            json.dump(gate, f, indent=2)
        raise SystemExit(f"reproducibility gate failed, lesions not run: {gate}")
    for arm in LESIONS:
        _child(config_path, m, arm, out[arm])
    rows = {arm: json.load(open(out[arm]))["episodes"] for arm in ARMS}
    summary = aggregate({a: rows[a] for a in ("expert",) + tuple(LESIONS)})
    summary["arms"]["expert_repeat"] = {**_arm_summary(rows["expert_repeat"]), **failure_split(rows["expert_repeat"])}
    summary["reproducibility_control"] = gate
    summary["lesions"] = {k: list(v) for k, v in LESIONS.items()}
    summary["manifest"] = {"path": m, "sha256": sha256_file(m)}
    summary["episodes"] = sum(len(r) for r in rows.values())
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


__all__ = ["ARMS", "FEEDBACK_LESIONS", "LESIONS", "LesionedController", "aggregate", "build_manifests",
           "failure_split", "lesion_index", "run_feedback_lesions"]
