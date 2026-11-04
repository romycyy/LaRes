"""R-047: how late can the qualified PPO expert's state feedback arrive? Observation-delay lesions on Stage 3.

R-044 (D-040) held a group of observation slices at their reset values. This protocol instead feeds the group's real
values from ``DELAY_STEPS`` control steps earlier, so the inputs stay on the episode's own trajectory and only their
timing changes. Each environment's history buffer starts filled with its own reset observation, so the first
``DELAY_STEPS`` steps see the reset values. Every other slice, including ``last_action``, stays live, and the wrapper
checks that it changed nothing outside the delayed group. The simulator is not touched.

* ``pen_state_delayed4``: the pen's position, orientation, linear and angular velocity and its goal-relative rotation.
* ``hand_state_delayed4``: hand joint positions and velocities and the fingertip poses and velocities.

One frozen 40-case manifest (pool seed 4200). Arms in order: ``expert``, ``expert_repeat`` (the reproducibility gate
from R-044; the delays are not run otherwise), then the two delays, each in its own Isaac process (D-034). Failures
are split into drops and tail-spin failures as in R-044. A delayed group paired with live other groups is a temporal
joint-distribution shift the expert never trained on: the result characterizes this checkpoint, not a universal
feedback-bandwidth requirement.
"""

from __future__ import annotations

import json
import os

from lares.envs.isaac_pen_spin.feedback_ablation import _arm_summary, check_checkpoint, reproducibility_gate
from lares.envs.isaac_pen_spin.feedback_lesions import LESIONS, _child, failure_split, lesion_index
from lares.envs.isaac_pen_spin.protocols import aggregate_vs_expert, build_single_manifest

FEEDBACK_DELAY = "feedback_delay"
STAGE = 3
DELAY_STEPS = 4
DELAYS = {
    "pen_state_delayed4": LESIONS["pen_state_clamped"],
    "hand_state_delayed4": LESIONS["hand_state_clamped"],
}
ARMS = ("expert", "expert_repeat") + tuple(DELAYS)


class DelayedController:
    """Wraps a controller; feeds the named observation columns from ``delay`` steps earlier."""

    def __init__(self, inner, arm: str, delay: int = DELAY_STEPS):
        if arm not in DELAYS:
            raise ValueError(f"unknown delay arm {arm!r}; one of {tuple(DELAYS)}")
        if delay < 1:
            raise ValueError("delay must be at least one control step")
        self.inner = inner
        self.name = arm
        self.delay = int(delay)
        self.columns = lesion_index(DELAYS[arm])
        self.params = {"delay_arm": arm, "fields": list(DELAYS[arm]), "columns": int(len(self.columns)),
                       "delay_steps": self.delay, "history_init": "the episode's reset observation",
                       **getattr(inner, "params", {})}

    def attach(self, env):
        if hasattr(self.inner, "attach"):
            self.inner.attach(env)

    def reset(self, ctx):
        self._history = None
        self.inner.reset(ctx)

    def __call__(self, obs, t, ctx):
        import torch

        if self._history is None:
            self._cols = torch.as_tensor(self.columns, device=obs.device, dtype=torch.long)
            self._history = [obs[:, self._cols].clone() for _ in range(self.delay)]
        delayed = self._history.pop(0)
        self._history.append(obs[:, self._cols].clone())
        out = obs.clone()
        out[:, self._cols] = delayed
        keep = torch.ones(obs.shape[1], dtype=torch.bool, device=obs.device)
        keep[self._cols] = False
        if not torch.equal(out[:, keep], obs[:, keep]):
            raise AssertionError("the delay changed an observation column outside its group")
        return self.inner(out, t, ctx)


def manifest_path(protocol_cfg: dict) -> str:
    return protocol_cfg["manifest"]


def build_manifests(protocol_cfg: dict) -> list[str]:
    return build_single_manifest(protocol_cfg, "delay", STAGE)


def make_controller(arm: str, protocol_cfg: dict, out_dir: str | None = None):
    from lares.envs.isaac_pen_spin.ppo_expert import PPOController

    expert = PPOController(check_checkpoint(protocol_cfg), int(protocol_cfg["training_seed"]))
    if arm in ("expert", "expert_repeat"):
        return expert
    return DelayedController(expert, arm, int(protocol_cfg.get("delay_steps", DELAY_STEPS)))


def aggregate(rows: dict) -> dict:
    """Per-arm summaries and each delay minus the expert, paired by case."""
    return aggregate_vs_expert(rows, DELAYS, "pen_state_delayed4")


def run_feedback_delay(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    """Driver: expert twice, the gate, the two delays, aggregate. Nothing is rerun if its output exists."""
    from lares.utils.hashing import sha256_file

    check_checkpoint(protocol_cfg)
    os.makedirs(out_dir, exist_ok=True)
    m = manifest_path(protocol_cfg)
    if not os.path.exists(m):
        raise SystemExit(f"frozen manifest {m} missing; build and commit it before the run")
    out = {arm: os.path.join(out_dir, f"{arm}.json") for arm in ARMS}
    for arm in ("expert", "expert_repeat"):
        _child(config_path, m, arm, out[arm], study=FEEDBACK_DELAY)
    gate = reproducibility_gate(out["expert"], out["expert_repeat"])
    if not gate["passes"]:
        with open(os.path.join(out_dir, "gate_failed.json"), "w") as f:
            json.dump(gate, f, indent=2)
        raise SystemExit(f"reproducibility gate failed, delays not run: {gate}")
    for arm in DELAYS:
        _child(config_path, m, arm, out[arm], study=FEEDBACK_DELAY)
    rows = {arm: json.load(open(out[arm]))["episodes"] for arm in ARMS}
    summary = aggregate({a: rows[a] for a in ("expert",) + tuple(DELAYS)})
    summary["arms"]["expert_repeat"] = {**_arm_summary(rows["expert_repeat"]), **failure_split(rows["expert_repeat"])}
    summary["reproducibility_control"] = gate
    summary["delays"] = {k: list(v) for k, v in DELAYS.items()}
    summary["delay_steps"] = int(protocol_cfg.get("delay_steps", DELAY_STEPS))
    summary["manifest"] = {"path": m, "sha256": sha256_file(m)}
    summary["episodes"] = sum(len(r) for r in rows.values())
    summary["limitation"] = ("a delayed group beside live groups is a temporal joint-distribution shift; this measures "
                             "the checkpoint's sensitivity, not a feedback-bandwidth requirement for any controller")
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


run = run_feedback_delay

__all__ = ["ARMS", "DELAYS", "DELAY_STEPS", "DelayedController", "FEEDBACK_DELAY", "aggregate", "build_manifests",
           "run", "run_feedback_delay"]
