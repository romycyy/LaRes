"""Controlled substitutions for diagnosing a failure (``spec.md`` FR-9, AC-6).

A failure label says *what* went wrong. It does not say which part of the
controller is responsible. A substitution answers that by replacing exactly one
part with a known-good source and re-running the same cases: if the failure
disappears, the replaced part was the fault; if it survives, it was not.

Four substitutions, each isolating a different suspect:

``expert_axis``   one or more action dimensions come from the scripted expert.
``expert_phase``  the whole action comes from the expert while a named phase is active.
``expert_prefix`` the expert drives the first N steps, then the policy takes over.
``force_gate``    a named phase gate is clamped open or shut.

Every result these produce describes an intervention, not a policy. They are
labelled so, and :class:`~lares.search.schemas.ExperimentRecord` and
:class:`~lares.search.archive.Archive` refuse to let them enter fitness ranking.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from lares.eval.runner import ACTION_MODE_DETERMINISTIC, Actor

SUBSTITUTION_NONE = "none"
SUBSTITUTION_EXPERT_AXIS = "expert_axis"
SUBSTITUTION_EXPERT_PHASE = "expert_phase"
SUBSTITUTION_EXPERT_PREFIX = "expert_prefix"
SUBSTITUTION_FORCE_GATE = "force_gate"
SUBSTITUTIONS = (
    SUBSTITUTION_NONE,
    SUBSTITUTION_EXPERT_AXIS,
    SUBSTITUTION_EXPERT_PHASE,
    SUBSTITUTION_EXPERT_PREFIX,
    SUBSTITUTION_FORCE_GATE,
)

AXIS_NAMES = ("dx", "dy", "dz", "gripper")


@dataclass
class Substitution:
    """One controlled substitution, described precisely enough to reproduce."""

    kind: str
    #: Action dimensions the expert drives, for ``expert_axis``.
    axes: tuple = ()
    #: Phase name the expert drives, for ``expert_phase``.
    phase: str = ""
    #: Steps the expert drives from the start, for ``expert_prefix``.
    prefix_steps: int = 0
    #: Gate name and forced value, for ``force_gate``.
    gate: str = ""
    gate_value: float = 1.0

    def __post_init__(self):
        if self.kind not in SUBSTITUTIONS:
            raise ValueError(f"kind must be one of {SUBSTITUTIONS}, got {self.kind!r}")
        if self.kind == SUBSTITUTION_EXPERT_AXIS and not self.axes:
            raise ValueError("expert_axis needs at least one axis")
        if self.kind == SUBSTITUTION_EXPERT_PHASE and not self.phase:
            raise ValueError("expert_phase needs a phase name")
        if self.kind == SUBSTITUTION_EXPERT_PREFIX and self.prefix_steps <= 0:
            raise ValueError("expert_prefix needs a positive step count")
        if self.kind == SUBSTITUTION_FORCE_GATE and not self.gate:
            raise ValueError("force_gate needs a gate name")

    @property
    def is_intervention(self) -> bool:
        """Anything but the unmodified control is expert-assisted or gate-forced."""
        return self.kind != SUBSTITUTION_NONE

    def label(self) -> str:
        if self.kind == SUBSTITUTION_NONE:
            return "unmodified"
        if self.kind == SUBSTITUTION_EXPERT_AXIS:
            names = ",".join(AXIS_NAMES[a] if a < len(AXIS_NAMES) else str(a) for a in self.axes)
            return f"expert drives {names}"
        if self.kind == SUBSTITUTION_EXPERT_PHASE:
            return f"expert drives the {self.phase} phase"
        if self.kind == SUBSTITUTION_EXPERT_PREFIX:
            return f"expert drives the first {self.prefix_steps} steps"
        return f"gate {self.gate} forced to {self.gate_value}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["axes"] = list(self.axes)
        d["label"] = self.label()
        d["is_intervention"] = self.is_intervention
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Substitution":
        """Rebuild from :meth:`to_dict`, dropping the derived fields."""
        return cls(
            kind=d["kind"],
            axes=tuple(d.get("axes", ()) or ()),
            phase=d.get("phase", ""),
            prefix_steps=int(d.get("prefix_steps", 0) or 0),
            gate=d.get("gate", ""),
            gate_value=float(d.get("gate_value", 1.0)),
        )


class InterventionActor(Actor):
    """Runs a policy with one part replaced by a known-good source.

    Shares the manifest runner with every other actor, so an intervention result
    is directly paired with the unmodified control on the same ordered cases.
    """

    action_mode = ACTION_MODE_DETERMINISTIC

    def __init__(self, policy, substitution: Substitution, env_name: str, schema, name=None):
        from lares.core.training_pipeline import get_expert_policy

        self.policy = policy
        self.substitution = substitution
        self.schema = schema
        self.expert = get_expert_policy(env_name)
        self.name = name or f"intervention:{substitution.label()}"
        self._step = 0
        self.policy.eval()

    def reset(self, case) -> None:
        self._step = 0
        reset_diagnostics = getattr(self.policy, "reset_diagnostics", None)
        if callable(reset_diagnostics):
            reset_diagnostics()
        clear = getattr(self.policy, "clear_gate_overrides", None)
        if callable(clear):
            clear()
        if self.substitution.kind == SUBSTITUTION_FORCE_GATE:
            self.policy.set_gate_override(
                self.substitution.gate, self.substitution.gate_value
            )

    def _policy_action(self, obs):
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean, _ = self.policy(obs_t)
        return torch.tanh(mean).squeeze(0).numpy().astype(np.float64)

    def _expert_action(self, obs):
        return np.clip(
            self.expert.get_action(np.asarray(obs, dtype=np.float64)), -1.0, 1.0
        )

    def act(self, obs) -> np.ndarray:
        from lares.fitting.phases import PHASE_NAMES, expert_phase_labels

        sub = self.substitution
        step = self._step
        self._step += 1

        if sub.kind in (SUBSTITUTION_NONE, SUBSTITUTION_FORCE_GATE):
            return self._policy_action(obs)

        if sub.kind == SUBSTITUTION_EXPERT_PREFIX:
            if step < sub.prefix_steps:
                return self._expert_action(obs)
            return self._policy_action(obs)

        if sub.kind == SUBSTITUTION_EXPERT_PHASE:
            phase = PHASE_NAMES[int(expert_phase_labels(np.asarray(obs)[None, :], self.schema)[0])]
            if phase == sub.phase:
                return self._expert_action(obs)
            return self._policy_action(obs)

        action = self._policy_action(obs)
        expert = self._expert_action(obs)
        for axis in sub.axes:
            action[axis] = expert[axis]
        return action

    def gates(self):
        return getattr(self.policy, "last_gates", None) or None


def gate_override_is_effective(policy, gate: str, obs, value: float = 1.0) -> bool:
    """Does forcing this gate actually change the action?

    A policy that calls ``record_gate`` without using the returned value has
    telemetry but no lever. Forcing such a gate changes nothing, and a null
    intervention that produced no change would otherwise read as evidence that
    the gate is not the fault.
    """
    with torch.no_grad():
        before = torch.tanh(policy(obs)[0]).clone()
    previous = dict(getattr(policy, "gate_overrides", {}))
    policy.set_gate_override(gate, value)
    try:
        with torch.no_grad():
            after = torch.tanh(policy(obs)[0])
        changed = not torch.allclose(before, after, atol=1e-9)
    finally:
        policy.gate_overrides = previous
    return changed


def available_gates(policy, obs) -> list:
    """Gate names this policy exposes, with a note on whether each is a lever."""
    reset = getattr(policy, "reset_diagnostics", None)
    if callable(reset):
        reset()
    with torch.no_grad():
        policy(obs)
    out = []
    for name in sorted(getattr(policy, "last_gates", {}) or {}):
        out.append(
            {"gate": name, "forceable": gate_override_is_effective(policy, name, obs)}
        )
    return out
