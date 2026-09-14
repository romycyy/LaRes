"""The one checkpoint score and tie rule every fitter uses (``debug_spec.md`` D4).

A rollout probe used to return whatever the caller liked: a float in one path, a
``(success, progress)`` tuple in another, forced through ``float()`` in a third,
and the tie rule preferred the later checkpoint in one place and the earlier in
another. The rule is now declared here once, before any run.

Tie rule: higher success, then higher mean signed goal progress, then the
*earlier* checkpoint. An exact tie keeps the checkpoint reached with less fitting.
"""

from __future__ import annotations

from dataclasses import dataclass

CHECKPOINT_TIE_RULE = "higher success, then higher progress, then earlier checkpoint"


@dataclass(frozen=True, order=True)
class RolloutScore:
    """``(success, progress)``, compared lexicographically."""

    success: float
    progress: float

    def to_dict(self) -> dict:
        return {"success": float(self.success), "progress": float(self.progress)}


def rollout_score(value) -> RolloutScore:
    """Coerce a probe's return value to a :class:`RolloutScore`.

    A bare number is read as success with no progress information, which keeps
    older probes that return a single float ordered the same way as before.
    """
    if isinstance(value, RolloutScore):
        return value
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"a rollout score is (success, progress), got {value!r}")
        return RolloutScore(float(value[0]), float(value[1]))
    return RolloutScore(float(value), 0.0)


def is_better_checkpoint(candidate, incumbent) -> bool:
    """True when ``candidate`` strictly beats ``incumbent``.

    Probes run in increasing step order, so "strictly better" is what makes an
    exact tie keep the earlier checkpoint.
    """
    if incumbent is None:
        return True
    return rollout_score(candidate) > rollout_score(incumbent)


def default_checkpoint_steps(budget: int, probes: int) -> tuple:
    """Step 0 plus ``probes`` evenly spaced steps ending at ``budget``.

    ``default_checkpoint_steps(2000, 4) == (0, 500, 1000, 1500, 2000)``.
    """
    budget, probes = int(budget), int(probes)
    if probes <= 0 or budget <= 0:
        return ()
    steps = {0} | {int(round(budget * k / probes)) for k in range(1, probes + 1)}
    return tuple(sorted(s for s in steps if 0 <= s <= budget))
