"""Diagnostic controllers for the pen-spin episode-rule audit (R-038).

Neither is meant to spin the pen. They fix two reference behaviours whose any-hit and sustained
scores show how much transient post-reset motion the two reducers admit. Constants are frozen here
before any probe case is run.
"""

from __future__ import annotations

import math

# Actuated joint index groups, in ShadowHandEnvCfg.actuated_joint_names order.
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

    def __call__(self, obs, t, ctx):
        return ctx.neutral_action.clone()


class OpenLoopDigitOscillation:
    """Neutral pose plus a phase-staggered sinusoid on every finger joint; the wrist is held."""

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
