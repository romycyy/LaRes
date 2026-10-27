"""R-039 scripted feedback finger gait for rolling the pen about its long axis.

Reads the observation only through the named layout (``cases.obs_slices``). Action indices follow the
order Isaac Lab actually uses: ``InHandManipulationEnv`` *sorts* ``actuated_dof_indices``, so action i
drives the i-th actuated joint in ``hand.joint_names`` order (verified on the box 2026-10-27 with
``scripts/isaac/tune_pen_spin_gait.py --describe``), not ``cfg.actuated_joint_names`` order.

The gait: every long finger's MCP and PIP flexion follow a sinusoid around a flexion bias, PIP lagging
MCP so the fingertip traces an ellipse (a stroke in contact, a return out of contact), with a phase
wave across the four fingers so strokes overlap. Two feedback terms: the measured roll rate about the
pen axis scales the stroke amplitude up when the pen spins below the target rate, and the pen centre's
offset along the finger direction (y) shifts the flexion bias to push it back toward ``in_hand_pos``.
The thumb runs the same oscillator with its own amplitude and phase. A third feedback term regulates the
signed roll rate about world x (the pen lies across the palm, its axis along x) by tilting the palm with
WRJ1, PI on the error to ``roll_dir * target_rate``: a ball-on-beam rate loop that rolls the pen one way
instead of rocking it. Every parameter can be given per environment, so one batched rollout can score
a sweep.
"""

from __future__ import annotations

import math

import torch

from lares.envs.isaac_pen_spin import cases

# Action indices (sorted actuated order): see module docstring.
A = {name: i for i, name in enumerate([
    "WRJ1", "WRJ0", "FFJ3", "MFJ3", "RFJ3", "LFJ4", "THJ4", "FFJ2", "MFJ2", "RFJ2",
    "LFJ3", "THJ3", "FFJ1", "MFJ1", "RFJ1", "LFJ2", "THJ2", "LFJ1", "THJ1", "THJ0",
])}
#: (MCP flexion, PIP flexion) per long finger, in wave order.
FINGER_FLEX = (("FFJ2", "FFJ1"), ("MFJ2", "MFJ1"), ("RFJ2", "RFJ1"), ("LFJ2", "LFJ1"))
THUMB_FLEX = ("THJ3", "THJ1", "THJ0")

DEFAULTS = {
    "flex_bias": -0.6,        # normalised flexion (-1 = straight, +1 = fully flexed)
    "amp": 0.3,               # stroke amplitude, normalised units
    "freq_hz": 2.0,
    "wave_lag": 0.5 * math.pi,  # phase lag between consecutive fingers
    "pip_lag": 0.5 * math.pi,   # PIP lags MCP: elliptical fingertip path
    "thumb_bias": 0.0,
    "thumb_amp": 0.0,
    "thumb_phase": math.pi,
    "wrist_bias": 0.0,        # added to the neutral WRJ1 action
    "wrist0_bias": 0.0,       # added to the neutral WRJ0 action
    "target_rate": 2.0,       # rad/s the amplitude feedback aims for
    "k_rate": 0.0,            # amplitude gain per unit relative rate shortfall
    "k_center": 0.0,          # flexion bias shift per metre of +y pen offset
    "amp_max": 0.8,
    # Roll regulation by palm tilt (ball-on-beam): WRJ1 tilts the palm about world x, the pen's axis.
    "roll_dir": 1.0,          # +1 / -1: desired sign of the roll rate about world x
    "k_tilt_p": 0.0,          # WRJ1 action per rad/s of roll-rate error
    "k_tilt_i": 0.0,          # WRJ1 action per rad of integrated roll-rate error
    "tilt_max": 0.5,          # bound on the tilt command, normalised units
    # Roll regulation by finger slope: the same rate error shifts the long fingers' flexion bias
    # (straight fingers slope down toward the fingertips; flexing them raises a barrier).
    "k_flex_p": 0.0,
    "k_flex_i": 0.0,
}


def quat_rotate_x(q: torch.Tensor) -> torch.Tensor:
    """World direction of the local x axis for quaternions (N, 4), w first."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)], dim=-1)


class FeedbackGait:
    name = "feedback_gait"

    def __init__(self, **params):
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown gait parameters {sorted(unknown)}")
        self.params = {**DEFAULTS, **params}
        self._t = None

    @classmethod
    def batched(cls, settings: list[dict]) -> "FeedbackGait":
        """One controller whose parameters differ per environment (``settings[i]`` for env i)."""
        g = cls()
        g.params = {k: [float({**DEFAULTS, **s}[k]) for s in settings] for k in DEFAULTS}
        return g

    def reset(self, ctx):
        dev = ctx.device
        self._p = {k: torch.as_tensor(v, dtype=torch.float32, device=dev).reshape(-1)
                   for k, v in self.params.items()}
        self._sl = cases.obs_slices()
        self._integral = torch.zeros(ctx.num_envs, device=dev)
        self._dt = ctx.control_dt

    def roll_rate(self, obs: torch.Tensor) -> torch.Tensor:
        axis = quat_rotate_x(obs[:, self._sl["obj_rot"]])
        axis = axis / axis.norm(dim=-1, keepdim=True)
        angvel = obs[:, self._sl["obj_angvel"]] / cases.VEL_OBS_SCALE
        return (angvel * axis).sum(-1)

    def __call__(self, obs, t, ctx):
        p = self._p
        a = ctx.neutral_action.clone()
        rate = self.roll_rate(obs).abs()
        shortfall = ((p["target_rate"] - rate) / p["target_rate"]).clamp(0.0, 1.0)
        amp = (p["amp"] * (1.0 + p["k_rate"] * shortfall)).clamp(max=p["amp_max"])
        dy = obs[:, self._sl["obj_pos"]][:, 1] - obs[:, self._sl["in_hand_pos"]][:, 1]
        roll_x = obs[:, self._sl["obj_angvel"]][:, 0] / cases.VEL_OBS_SCALE
        err = p["roll_dir"] * p["target_rate"] - roll_x
        self._integral = (self._integral + err * self._dt).clamp(-5.0, 5.0)
        bias = (p["flex_bias"] + p["k_center"] * dy
                + p["k_flex_p"] * err + p["k_flex_i"] * self._integral).clamp(-1.0, 1.0)
        phase0 = 2.0 * math.pi * p["freq_hz"] * (t * ctx.control_dt)
        for k, (mcp, pip) in enumerate(FINGER_FLEX):
            ph = phase0 - k * p["wave_lag"]
            a[:, A[mcp]] = bias + amp * torch.sin(ph)
            a[:, A[pip]] = bias + amp * torch.sin(ph - p["pip_lag"])
        th = phase0 + p["thumb_phase"]
        for j in THUMB_FLEX:
            a[:, A[j]] = a[:, A[j]] + p["thumb_bias"] + p["thumb_amp"] * torch.sin(th)
        tilt = (p["k_tilt_p"] * err + p["k_tilt_i"] * self._integral).clamp(-p["tilt_max"], p["tilt_max"])
        a[:, A["WRJ1"]] = a[:, A["WRJ1"]] + p["wrist_bias"] + tilt
        a[:, A["WRJ0"]] = a[:, A["WRJ0"]] + p["wrist0_bias"]
        return a.clamp(-1.0, 1.0)

    def describe(self) -> dict:
        return {"name": self.name, "params": self.params}


#: R-039 frozen setting, chosen on debug seed 4098 only (1720 tuning episodes; sweeps under
#: logs/pen_spin/r039_scripted_expert/sweeps). Committed before any development case was run.
#: Debug estimate on 40 cases: sustained 0.65 / 0.57 / 0.47 (stage 1/2/3). Debugging also showed that a
#: case's outcome depends on batch composition (same setting and cases: 0.75 vs 0.60 in two sweeps).
FROZEN_R039 = {
    "flex_bias": -0.82, "target_rate": 2.5, "roll_dir": -1.0, "k_flex_p": 0.08, "k_flex_i": -1.8,
    "k_center": 3.0, "amp": 0.3, "freq_hz": 3.0,
}


def frozen_gait() -> FeedbackGait:
    return FeedbackGait(**FROZEN_R039)
