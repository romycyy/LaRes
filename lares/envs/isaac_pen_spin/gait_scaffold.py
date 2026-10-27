"""R-046: the frozen R-039 gait as a scaffold that generated corrections are added to.

Every arm runs the frozen gait (``scripted_expert.FROZEN_R039``, hash-checked) with its own clock and integral, reset
when the batch starts (one process runs one manifest, so every case starts its episode there) and updated once per
control step on that arm's own trajectory. A generated policy reads the 157-d observation plus ``gait_action`` (the
gait's action this step, 177-d in all) and its mean is a correction ``delta``:

* ``unchanged``: execute ``g`` (no policy).
* ``residual``: execute ``clamp(g + clamp(delta, -bound, bound), -1, 1)``.
* ``whole``: execute ``clamp(g + delta, -1, 1)``; ``delta = 0`` keeps the gait and ``delta = target - g`` replaces it.

:class:`ShadowGait` runs the gait beside another controller without executing it, so an expert's demonstration carries
the gait's action at every step, with the integral history it would have had along that trajectory.
"""

from __future__ import annotations

import json

from lares.utils.hashing import sha256_text

MODES = ("unchanged", "residual", "whole")
RESIDUAL_BOUND = 0.15


def gait_params_sha256() -> str:
    from lares.envs.isaac_pen_spin.scripted_expert import FROZEN_R039

    return sha256_text(json.dumps(FROZEN_R039, sort_keys=True))


def check_gait(expected_sha256: str) -> None:
    got = gait_params_sha256()
    if got != expected_sha256:
        raise ValueError(f"FROZEN_R039 hash {got} != pinned {expected_sha256}; the scaffold must not be retuned")


def combine(g, delta, mode: str, bound: float = RESIDUAL_BOUND):
    """The executed action from the gait's action and the policy's correction."""
    if mode == "unchanged":
        return g.clamp(-1.0, 1.0)
    if mode == "residual":
        return (g + delta.clamp(-bound, bound)).clamp(-1.0, 1.0)
    if mode == "whole":
        return (g + delta).clamp(-1.0, 1.0)
    raise ValueError(f"unknown scaffold mode {mode!r}; one of {MODES}")


class ScaffoldController:
    """The frozen gait plus an optional generated correction, behind the runner's controller interface."""

    name = "gait_scaffold"

    def __init__(self, mode: str, policy_mean=None, bound: float = RESIDUAL_BOUND, gait_sha256: str | None = None):
        from lares.envs.isaac_pen_spin.scripted_expert import frozen_gait

        if mode not in MODES:
            raise ValueError(f"unknown scaffold mode {mode!r}; one of {MODES}")
        if (policy_mean is None) != (mode == "unchanged"):
            raise ValueError("only the unchanged scaffold runs without a policy")
        if gait_sha256 is not None:
            check_gait(gait_sha256)
        self.mode, self.bound, self.policy_mean = mode, float(bound), policy_mean
        self.gait = frozen_gait()
        self.params = {"mode": mode, "bound": self.bound if mode == "residual" else None,
                       "gait_sha256": gait_params_sha256()}

    def reset(self, ctx):
        self.gait.reset(ctx)

    def __call__(self, obs, t, ctx):
        import torch

        g = self.gait(obs, t, ctx)
        if self.policy_mean is None:
            return combine(g, None, self.mode)
        aug = torch.cat([obs.detach().float(), g.detach().float()], dim=-1)
        delta = self.policy_mean(aug).to(obs.device)
        return combine(g, delta, self.mode, self.bound)


class ShadowGait:
    """Runs another controller and, beside it, the frozen gait on the same observations; logs the gait's actions."""

    def __init__(self, inner):
        from lares.envs.isaac_pen_spin.scripted_expert import frozen_gait

        self.inner = inner
        self.name = getattr(inner, "name", "controller")
        self.params = dict(getattr(inner, "params", {}) or {}, shadow_gait_sha256=gait_params_sha256())
        self.gait = frozen_gait()
        self.log = []

    def attach(self, env):
        if hasattr(self.inner, "attach"):
            self.inner.attach(env)

    def reset(self, ctx):
        self.inner.reset(ctx)
        self.gait.reset(ctx)
        self.log = []

    def __call__(self, obs, t, ctx):
        self.log.append(self.gait(obs, t, ctx).detach().cpu().numpy())
        return self.inner(obs, t, ctx)

    def gait_actions(self):
        """``(cases, T, 20)``, the layout the runner uses for actions."""
        import numpy as np

        return np.stack(self.log, axis=1)


__all__ = ["MODES", "RESIDUAL_BOUND", "ScaffoldController", "ShadowGait", "check_gait", "combine",
           "gait_params_sha256"]
