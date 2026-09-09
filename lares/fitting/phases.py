"""Ground-truth expert phases (``spec.md`` FR-8, AC-5).

MetaWorld's scripted expert for ``push`` is not a black box: its desired position
has three explicit branches, and which branch fires is a function of the
observation alone. That makes the phase label ground truth rather than something
inferred from a learned gate, so phase-balanced sampling can be built on it
without circularity.

From ``metaworld.policies.sawyer_push_v3_policy.SawyerPushV3Policy._desired_pos``:

1. ``hover``   -- planar error to the puck above 0.02 m: go 0.2 m above the puck.
2. ``descend`` -- planar error settled, vertical error above 0.04 m: drop to 0.03 m above it.
3. ``push``    -- both settled: drive the hand at the goal.

The thresholds here are copied from that policy. If it changes, this drifts, so
:func:`verify_against_expert` compares the labels against the expert's own
branch decision and is exercised in the tests.
"""

from __future__ import annotations

import numpy as np

PHASE_HOVER = 0
PHASE_DESCEND = 1
PHASE_PUSH = 2
PHASE_NAMES = ("hover", "descend", "push")

#: Copied from the expert. A puck offset of -0.005 m in x is applied there too.
PLANAR_TOLERANCE = 0.02
VERTICAL_TOLERANCE = 0.04
PUCK_X_OFFSET = -0.005


def expert_phase_labels(obs, schema) -> np.ndarray:
    """Phase index per row, following the expert's own branch conditions."""
    tcp = np.asarray(schema.slice(obs, "tcp"), dtype=np.float64)
    obj = np.asarray(schema.slice(obs, "obj"), dtype=np.float64)
    puck = obj.copy()
    puck[:, 0] += PUCK_X_OFFSET

    planar = np.linalg.norm(tcp[:, :2] - puck[:, :2], axis=1)
    vertical = np.abs(tcp[:, 2] - puck[:, 2])

    labels = np.full(tcp.shape[0], PHASE_PUSH, dtype=np.int64)
    labels[vertical > VERTICAL_TOLERANCE] = PHASE_DESCEND
    labels[planar > PLANAR_TOLERANCE] = PHASE_HOVER
    return labels


def phase_counts(labels) -> dict:
    arr = np.asarray(labels)
    return {name: int((arr == i).sum()) for i, name in enumerate(PHASE_NAMES)}


def phase_fractions(labels) -> dict:
    counts = phase_counts(labels)
    total = sum(counts.values()) or 1
    return {name: count / total for name, count in counts.items()}


def verify_against_expert(obs_rows, schema, expert) -> list:
    """Compare the labels against the branch the expert actually took.

    The expert exposes no branch id, so the check reconstructs it from the
    desired position it returns: hovering targets 0.2 m above the puck,
    descending targets 0.03 m above it, pushing targets the goal.
    Returns the indices where the label disagrees.
    """
    obs = np.asarray(obs_rows, dtype=np.float64)
    labels = expert_phase_labels(obs, schema)
    goal = np.asarray(schema.slice(obs, "goal"), dtype=np.float64)
    obj = np.asarray(schema.slice(obs, "obj"), dtype=np.float64)

    mismatches = []
    for i, row in enumerate(obs):
        parsed = expert._parse_obs(row)
        desired = expert._desired_pos(parsed)
        puck = obj[i] + np.array([PUCK_X_OFFSET, 0.0, 0.0])
        if np.allclose(desired, puck + np.array([0.0, 0.0, 0.2]), atol=1e-9):
            actual = PHASE_HOVER
        elif np.allclose(desired, puck + np.array([0.0, 0.0, 0.03]), atol=1e-9):
            actual = PHASE_DESCEND
        elif np.allclose(desired, goal[i], atol=1e-9):
            actual = PHASE_PUSH
        else:
            mismatches.append(i)
            continue
        if actual != labels[i]:
            mismatches.append(i)
    return mismatches


def balanced_indices(labels, batch_size: int, generator) -> np.ndarray:
    """Sample a minibatch with equal mass per occupied phase.

    Uniform sampling sees the phases in whatever proportion the expert spends
    time in them, so a short phase contributes few gradients however important
    it is. This gives each occupied phase an equal share.
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        raise ValueError(
            "cannot draw a minibatch from an empty label array; there is nothing to "
            "sample, and returning indices into an empty dataset would fail later "
            "somewhere less obvious"
        )
    occupied = [i for i in range(len(PHASE_NAMES)) if np.any(labels == i)]
    if not occupied:
        return generator.integers(0, len(labels), size=batch_size)

    per_phase = max(1, batch_size // len(occupied))
    picked = []
    for phase in occupied:
        pool = np.flatnonzero(labels == phase)
        picked.append(pool[generator.integers(0, len(pool), size=per_phase)])
    out = np.concatenate(picked)
    if len(out) < batch_size:
        extra = generator.integers(0, len(labels), size=batch_size - len(out))
        out = np.concatenate([out, extra])
    return out[:batch_size]
