"""Diagnostic controllers for the pen-spin episode-rule audit (R-038).

Neither is meant to spin the pen. They fix two reference behaviours whose any-hit and sustained
scores show how much transient post-reset motion the two reducers admit. Constants are frozen here
before any probe case is run.
"""

from __future__ import annotations

import math

# Index blocks of the action vector. CAUTION (found 2026-10-27, after R-038 ran): Isaac Lab sorts
# actuated_dof_indices, so action i drives the i-th actuated joint in hand.joint_names order (WRJ1, WRJ0,
# FFJ3, MFJ3, RFJ3, LFJ4, THJ4, FFJ2, ...), not ShadowHandEnvCfg.actuated_joint_names order. These blocks
# therefore do not group joints by finger: "ff" (2-4) is the FF/MF/RF abduction joints, "mf" (5-7) is
# LFJ4, THJ4, FFJ2, and so on. The oscillator is kept exactly as it ran in R-038 (it is R-039's frozen
# baseline); only this description changed. lares/envs/isaac_pen_spin/scripted_expert.A has the true map.
WRIST = (0, 1)
FINGERS = {
    "ff": (2, 3, 4),
    "mf": (5, 6, 7),
    "rf": (8, 9, 10),
    "lf": (11, 12, 13, 14),
    "th": (15, 16, 17, 18, 19),
}
OSC_AMPLITUDE = 0.35  # in normalised action units
OSC_FREQUENCY_HZ = 1.5
OSC_PHASES = {"ff": 0.0, "mf": 0.5 * math.pi, "rf": math.pi, "lf": 1.5 * math.pi, "th": 0.25 * math.pi}


class NeutralPose:
    """Hold every joint at its default (zero) position."""

    name = "neutral_pose"

    def reset(self, ctx):
        pass

    def __call__(self, obs, t, ctx):
        return ctx.neutral_action.clone()


class OpenLoopDigitOscillation:
    """Neutral pose plus a sinusoid on action indices 2-19 in five phase-staggered blocks; the wrist is held.

    The blocks are index ranges, not fingers (see the caution above).
    """

    name = "open_loop_digit_oscillation"

    def __init__(self, amplitude: float = OSC_AMPLITUDE, frequency_hz: float = OSC_FREQUENCY_HZ):
        self.amplitude = float(amplitude)
        self.frequency_hz = float(frequency_hz)

    def __call__(self, obs, t, ctx):
        a = ctx.neutral_action.clone()
        time_s = t * ctx.control_dt
        for finger, joints in FINGERS.items():
            delta = self.amplitude * math.sin(2.0 * math.pi * self.frequency_hz * time_s + OSC_PHASES[finger])
            a[:, list(joints)] += delta
        return a.clamp(-1.0, 1.0)


AUDIT_CONTROLLERS = (NeutralPose, OpenLoopDigitOscillation)


def describe() -> dict:
    return {
        "neutral_pose": "joint targets at default position for all 20 actuated joints",
        "open_loop_digit_oscillation": {
            "amplitude": OSC_AMPLITUDE, "frequency_hz": OSC_FREQUENCY_HZ,
            "phases": OSC_PHASES, "wrist": "held at neutral",
        },
    }
