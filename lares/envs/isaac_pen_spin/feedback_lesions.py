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

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin.feedback_ablation import check_checkpoint, reproducibility_gate
from lares.envs.isaac_pen_spin.protocols import (aggregate_vs_expert, arm_summary, build_single_manifest,
                                                 failure_split, manifest_path, study_cell)

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


def build_manifests(protocol_cfg: dict) -> list[str]:
    return build_single_manifest(protocol_cfg, "lesion", STAGE)


def make_controller(arm: str, protocol_cfg: dict, out_dir: str | None = None):
    from lares.envs.isaac_pen_spin.ppo_expert import PPOController

    expert = PPOController(check_checkpoint(protocol_cfg), int(protocol_cfg["training_seed"]))
    if arm in ("expert", "expert_repeat"):
        return expert
    return LesionedController(expert, arm)


def aggregate(rows: dict) -> dict:
    """Per-arm summaries and each lesion minus the expert, paired by case."""
    return aggregate_vs_expert(rows, LESIONS, "pen_state_clamped")


def run_feedback_lesions(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    """Driver: expert twice, the gate, the two lesions, aggregate. Nothing is rerun if its output exists."""
    check_checkpoint(protocol_cfg)
    os.makedirs(out_dir, exist_ok=True)
    m = manifest_path(protocol_cfg)
    out = {arm: os.path.join(out_dir, f"{arm}.json") for arm in ARMS}
    for arm in ("expert", "expert_repeat"):
        study_cell(protocol_cfg, config_path, m, arm, out[arm], record_obs=False)
    gate = reproducibility_gate(out["expert"], out["expert_repeat"])
    if not gate["passes"]:
        with open(os.path.join(out_dir, "gate_failed.json"), "w") as f:
            json.dump(gate, f, indent=2)
        raise SystemExit(f"reproducibility gate failed, lesions not run: {gate}")
    for arm in LESIONS:
        study_cell(protocol_cfg, config_path, m, arm, out[arm], record_obs=False)
    rows = {arm: json.load(open(out[arm]))["episodes"] for arm in ARMS}
    summary = aggregate({a: rows[a] for a in ("expert",) + tuple(LESIONS)})
    summary["arms"]["expert_repeat"] = {**arm_summary(rows["expert_repeat"]), **failure_split(rows["expert_repeat"])}
    summary["reproducibility_control"] = gate
    summary["lesions"] = {k: list(v) for k, v in LESIONS.items()}
    summary["manifest"] = cases.manifest_record(m)
    summary["episodes"] = sum(len(r) for r in rows.values())
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


run = run_feedback_lesions

__all__ = ["ARMS", "LESIONS", "LesionedController", "aggregate", "build_manifests", "lesion_index", "run",
           "run_feedback_lesions"]
