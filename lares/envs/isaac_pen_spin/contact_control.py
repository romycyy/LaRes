"""R-093 (C1): capsule contact frames, a damped least-squares velocity decoder and the gate's statistics. No Isaac.

Everything here is batched torch (frames, decoder) or numpy over recorded arrays (eligibility, gate statistics) and
imports on a machine with no simulator. The Isaac-attached controller that supplies Jacobians, joint state and
limits is written on top of this module.

**What is trusted geometry and what the gate measures.** The frames are exact geometry of an ideal capsule and a
fingertip *point*: nothing here knows whether the finger touches the pen. A small ``distance`` is proximity, not
contact, support or friction. The decoder is a kinematic approximation (one linear solve, then bounds); whether a
decoded request produces the requested local motion in the simulator is what the gate measures, with the pure
functions at the bottom of this file, which are written and tested before any run.

Conventions
-----------
* Quaternions are scalar-first ``(w, x, y, z)``, as Isaac Lab; positions are in the env frame (origin subtracted),
  as the observation. All functions take physical units (m, m/s, rad, rad/s).
* The pen is a capsule whose long axis is its local +X (``CapsuleCfg(axis="X")``), ``radius`` 0.012 m. Isaac Lab's
  ``CapsuleCfg(height=0.14)`` is written to a ``UsdGeom.Capsule`` prim, whose ``height`` attribute is "the length of
  the capsule's spine along the specified axis excluding the size of the two half spheres" (OpenUSD, UsdGeomCapsule).
  So the axis segment has half-length 0.07 m and the pen is 0.164 m end to end. Source: the USD schema text and
  Isaac Lab's ``spawn_capsule``, from memory of both; the prim's extent is to be read on the box.
* Fingertips are ordered (ff, mf, rf, lf, th), ``cfg.fingertip_body_names`` of the inherited env. A fingertip
  "point" is the distal link's body origin, which is not a point on the finger pad.
* A frame ``R`` has columns ``[normal, circumferential, axial]`` in the env frame, right-handed
  (``normal x circumferential = axial``, det +1):

  - ``normal`` points from the pen surface outward toward the fingertip;
  - ``axial`` is the pen's local +X on the cylinder section;
  - ``circumferential = axial x normal``. A fingertip with positive circumferential relative velocity moves around
    the pen counter-clockwise seen from the +X end, the right-hand sense about +X. Equivalently, with the fingertip
    still, a pen rolling at ``+w`` about its +X gives a circumferential relative velocity of ``-w * radius``; a
    non-slipping fingertip moving along +circumferential rolls a pen with a fixed centre positively about +X.

* **End caps** (``on_cap``: the tip's axial coordinate is beyond the segment). The normal is the true surface
  normal, radial from the cap centre, so it gains an axial component. The frame stays orthonormal:
  ``circumferential = normalize(axis x normal)`` (unchanged in direction) and the third column is
  ``normal x circumferential``, the pen axis projected on the tangent plane. This is continuous across the
  cylinder/cap boundary.
* **Degenerate** inputs never give NaN. (a) Tip on the axis segment: the normal is the previous normal projected
  perpendicular to the axis when ``prev`` is given, else the pen's local +Z. (b) Tip on the axis line beyond an end
  (cap pole, normal = +/-axis): the circumferential is the previous one projected on the tangent plane when ``prev``
  is given, else the pen's local +Y. (c) Non-finite input or a zero quaternion: inputs are zero-filled / the
  identity rotation is used. All three set ``degenerate``; no frame is continuous through (a) or (b).

To verify on the box (flagged, not checked here): the capsule extent; that ``fingertip_vel`` is five rows of
(linear 3, angular 3) from ``body_vel_w`` and is *not* multiplied by ``vel_obs_scale`` (Isaac Lab's
``compute_full_observations`` scales only the joint velocities and the object's angular velocity, as ``layout.py``
also says; ``state_from_obs`` takes the scale as an argument); and ``act_moving_average`` (see ``position_targets``).
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import torch

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin.layout import VEL_OBS_SCALE
from lares.utils.hashing import sha256_text

PEN_RADIUS = 0.012
PEN_HALF_LENGTH = 0.07
FINGERTIPS = ("ff", "mf", "rf", "lf", "th")
LOCAL_AXES = ("normal", "circumferential", "axial")

#: Centre of mass of each distal link in its own frame (m), read on the box 2026-11-20 (``body_com_pos_b``). The
#: observation's fingertip *position* is the link origin, but its linear *velocity* (``body_vel_w``) and the PhysX
#: Jacobian are for the centre of mass: on the R-093 calibration traces the finite-difference origin velocity
#: differs from the observed velocity by a median 12-21% and from ``v - w x r`` by 4-6%.
TIP_COM_OFFSET_B = ((0.0, 0.0, 0.015), (0.0, 0.0, 0.015), (0.0, 0.0, 0.015), (0.0, 0.0, 0.015), (0.0, 0.0, 0.016))

_EPS_LENGTH = 1e-6   # m: below this the tip is on the axis segment
_EPS_POLE = 1e-4     # sine of the angle between the normal and the axis: below this the tip is at a cap pole


# ---------------------------------------------------------------------------
#  Observation -> physical state
# ---------------------------------------------------------------------------


def state_from_obs(obs, fingertip_vel_scale: float = 1.0) -> dict:
    """Physical-unit pen and fingertip state from a ``(B, 157+)`` observation batch, by the named layout.

    ``obj_angvel`` and ``hand_dof_vel`` are divided by ``VEL_OBS_SCALE``; ``obj_linvel`` is unscaled in the layout.
    ``fingertip_vel`` is read as ``(B, 5, 6)`` = linear then angular per fingertip and divided by
    ``fingertip_vel_scale`` (1.0: believed unscaled; to verify on the box). ``hand_dof_pos`` stays normalised to
    [-1, 1] over all 24 joints, as observed; the caller owns the joint limits.
    """
    obs = torch.as_tensor(obs)
    s = cases.obs_slices()
    b = obs.shape[0]
    k = len(FINGERTIPS)
    tip_vel = obs[:, s["fingertip_vel"]].reshape(b, k, 6) / fingertip_vel_scale
    return {
        "pen_pos": obs[:, s["obj_pos"]],
        "pen_quat": obs[:, s["obj_rot"]],
        "pen_linvel": obs[:, s["obj_linvel"]],
        "pen_angvel": obs[:, s["obj_angvel"]] / VEL_OBS_SCALE,
        "tip_pos": obs[:, s["fingertip_pos"]].reshape(b, k, 3),
        "tip_quat": obs[:, s["fingertip_rot"]].reshape(b, k, 4),
        "tip_linvel": tip_vel[..., :3],
        "tip_angvel": tip_vel[..., 3:],
        "hand_dof_pos_normalised": obs[:, s["hand_dof_pos"]],
        "hand_dof_vel": obs[:, s["hand_dof_vel"]] / VEL_OBS_SCALE,
    }


def tip_origin_linvel(state: dict, com_offset_b=TIP_COM_OFFSET_B) -> torch.Tensor:
    """Linear velocity of the fingertip link *origins* (B, 5, 3): ``v_com - w x (R r_b)`` from a ``state_from_obs`` dict."""
    r_b = torch.as_tensor(com_offset_b, dtype=state["tip_linvel"].dtype, device=state["tip_linvel"].device)
    r_w = torch.einsum("bkij,kj->bki", quat_to_matrix(state["tip_quat"]).to(r_b.dtype), r_b)
    return state["tip_linvel"] - torch.linalg.cross(state["tip_angvel"], r_w)


# ---------------------------------------------------------------------------
#  Capsule frames
# ---------------------------------------------------------------------------


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """Rotation matrices ``(..., 3, 3)`` of scalar-first quaternions; a zero or non-finite one gives the identity."""
    q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    n = q.norm(dim=-1, keepdim=True)
    ident = torch.zeros_like(q)
    ident[..., 0] = 1.0
    q = torch.where(n > 1e-8, q / n.clamp_min(1e-8), ident)
    w, x, y, z = q.unbind(-1)
    rows = (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y))
    return torch.stack(rows, dim=-1).reshape(q.shape[:-1] + (3, 3))


def _unit(v: torch.Tensor, fallback: torch.Tensor, eps: float):
    """``v`` normalised, or ``fallback`` where ``|v| <= eps``; also the mask of rows that were normalised."""
    n = v.norm(dim=-1, keepdim=True)
    ok = n > eps
    return torch.where(ok, v / n.clamp_min(eps), fallback.expand_as(v)), ok.squeeze(-1)


def _reject(v: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return v - (v * n).sum(-1, keepdim=True) * n


def capsule_frames(pen_pos, pen_quat, tip_pos, radius: float = PEN_RADIUS, half_length: float = PEN_HALF_LENGTH,
                   prev: dict | None = None) -> dict:
    """Nearest-surface frames of a capsule at ``K`` fingertip points.

    ``pen_pos`` (B, 3), ``pen_quat`` (B, 4), ``tip_pos`` (B, K, 3). Returns a dict of ``point`` (B, K, 3) nearest
    point on the surface, ``distance`` (B, K) from the tip to the surface (negative inside), ``R`` (B, K, 3, 3)
    with columns [normal, circumferential, axial] (module docstring), ``axis`` (B, 3) the pen's +X, ``axial_coord``
    (B, K) the tip's coordinate along the axis from the centre, and the masks ``on_cap`` and ``degenerate`` (B, K).
    ``prev`` is an earlier result for the same envs and fingertips; it is used only where the frame is degenerate.
    """
    finite = (torch.isfinite(tip_pos).all(-1) & torch.isfinite(pen_pos).all(-1, keepdim=True)
              & torch.isfinite(pen_quat).all(-1, keepdim=True) & (pen_quat.norm(dim=-1, keepdim=True) > 1e-8))
    pen_pos = torch.nan_to_num(pen_pos, nan=0.0, posinf=0.0, neginf=0.0)
    tip_pos = torch.nan_to_num(tip_pos, nan=0.0, posinf=0.0, neginf=0.0)
    rot = quat_to_matrix(pen_quat).to(tip_pos.dtype)
    a, y, z = (rot[:, None, :, i] for i in range(3))            # (B, 1, 3) each, orthonormal

    d = tip_pos - pen_pos[:, None, :]
    t = (d * a).sum(-1)
    foot = pen_pos[:, None, :] + t.clamp(-half_length, half_length)[..., None] * a
    radial = tip_pos - foot

    normal_fallback = z
    if prev is not None:
        normal_fallback, _ = _unit(_reject(prev["R"][..., 0], a), z, _EPS_LENGTH)
    normal, normal_ok = _unit(radial, normal_fallback, _EPS_LENGTH)

    circ_fallback, _ = _unit(_reject(y.expand_as(normal), normal), z, 0.5)
    if prev is not None:
        circ_fallback, _ = _unit(_reject(prev["R"][..., 1], normal), circ_fallback, 0.5)
    circ, circ_ok = _unit(torch.linalg.cross(a.expand_as(normal), normal), circ_fallback, _EPS_POLE)
    third = torch.linalg.cross(normal, circ)

    return {
        "point": foot + radius * normal,
        "distance": radial.norm(dim=-1) - radius,
        "R": torch.stack((normal, circ, third), dim=-1),
        "axis": a[:, 0, :],
        "axial_coord": t,
        "on_cap": t.abs() > half_length,
        "degenerate": ~(normal_ok & circ_ok & finite),
    }


# ---------------------------------------------------------------------------
#  Relative velocity
# ---------------------------------------------------------------------------


def surface_point_velocity(pen_pos, pen_linvel, pen_angvel, point) -> torch.Tensor:
    """``v + w x (p - c)``: velocity of the pen's material point at ``point`` (B, K, 3); pen state is (B, 3)."""
    return pen_linvel[:, None, :] + torch.linalg.cross(pen_angvel[:, None, :].expand_as(point),
                                                       point - pen_pos[:, None, :])


def relative_world_velocity(tip_linvel, point, pen_pos, pen_linvel, pen_angvel) -> torch.Tensor:
    """Fingertip velocity minus the velocity of the pen material point at ``point``, in the env frame (B, K, 3)."""
    return tip_linvel - surface_point_velocity(pen_pos, pen_linvel, pen_angvel, point)


def relative_local_velocity(tip_linvel, frames: dict, pen_pos, pen_linvel, pen_angvel) -> torch.Tensor:
    """The gate's observed quantity: relative velocity in the local [normal, circumferential, axial] coordinates.

    The nearest surface point changes identity as the tip moves; this is the velocity relative to the material
    point currently under the tip, not the rate of change of any surface coordinate.
    """
    rel = relative_world_velocity(tip_linvel, frames["point"], pen_pos, pen_linvel, pen_angvel)
    return world_to_local(rel, frames)


def world_to_local(v_world, frames: dict) -> torch.Tensor:
    return torch.einsum("bkji,bkj->bki", frames["R"], v_world)


def local_to_world(u_local, frames: dict) -> torch.Tensor:
    """Inverse of :func:`world_to_local`: a local request (B, K, 3) as a world relative velocity."""
    return torch.einsum("bkij,bkj->bki", frames["R"], u_local)


# ---------------------------------------------------------------------------
#  Decoder
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DecoderConfig:
    """Decoder constants. The defaults are placeholders, not calibrated values; R-093 freezes its own in the config.

    ``damping`` is the ``lambda`` of ``lambda * |qdot|^2`` (not its square root). ``lookahead`` is in seconds:
    ``q_target = q + lookahead * qdot``; it is also the horizon of the joint-limit bound. ``request_scale`` is m/s
    per unit of normalised request, and ``max_request_speed`` caps the norm of each fingertip's request (m/s).
    """

    damping: float = 1e-4
    tip_weights: tuple = (1.0, 1.0, 1.0, 1.0, 1.0)
    max_joint_speed: float = 2.0          # rad/s
    control_dt: float = 1.0 / 60.0        # s
    lookahead: float = 1.0 / 60.0         # s
    joint_limit_margin: float = 0.02      # rad
    request_scale: float = 0.05           # m/s per unit request
    max_request_speed: float = 0.05       # m/s

    def __post_init__(self):
        object.__setattr__(self, "tip_weights", tuple(float(w) for w in self.tip_weights))
        for name in ("damping", "max_joint_speed", "control_dt", "lookahead", "request_scale", "max_request_speed"):
            value = getattr(self, name)
            if not (np.isfinite(value) and value > 0):
                raise ValueError(f"DecoderConfig.{name} must be finite and positive, got {value!r}")
        if not (np.isfinite(self.joint_limit_margin) and self.joint_limit_margin >= 0):
            raise ValueError("DecoderConfig.joint_limit_margin must be finite and non-negative")
        if not self.tip_weights or any(not np.isfinite(w) or w < 0 for w in self.tip_weights):
            raise ValueError("DecoderConfig.tip_weights must be finite and non-negative")

    def to_dict(self) -> dict:
        out = {f.name: float(getattr(self, f.name)) for f in dataclasses.fields(self) if f.name != "tip_weights"}
        out["tip_weights"] = list(self.tip_weights)
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "DecoderConfig":
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - names
        if unknown:
            raise ValueError(f"unknown DecoderConfig keys: {sorted(unknown)}")
        return cls(**data)

    def sha256(self) -> str:
        return sha256_text(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")))


def scale_request(request, cfg: DecoderConfig) -> torch.Tensor:
    """Normalised requests (B, K, 3) or (B, 3K) to m/s: ``clamp(request, -1, 1) * request_scale``."""
    request = torch.nan_to_num(request, nan=0.0, posinf=1.0, neginf=-1.0)
    return request.reshape(request.shape[0], -1, 3).clamp(-1.0, 1.0) * cfg.request_scale


def effective_jacobian(J_full, coupling) -> torch.Tensor:
    """Fold coupled joints in: ``J_full`` (B, K, 3, N_full) times ``coupling`` (N_full, N), ``qdot_full = C qdot``."""
    return J_full @ coupling.to(J_full.dtype)


def _limit_rates(q, q_lo, q_hi, cfg: DecoderConfig):
    """Joint-speed interval that keeps ``q + lookahead * qdot`` inside the margined limits.

    A joint already outside the margin may not move further out and is not forced back. A range narrower than two
    margins collapses to its midpoint.
    """
    mid = 0.5 * (q_lo + q_hi)
    lo = torch.minimum(q_lo + cfg.joint_limit_margin, mid)
    hi = torch.maximum(q_hi - cfg.joint_limit_margin, mid)
    return ((lo - q) / cfg.lookahead).clamp(max=0.0), ((hi - q) / cfg.lookahead).clamp(min=0.0)


def decode(J, frames: dict, u_local, pen_pos, pen_linvel, pen_angvel, q, q_lo, q_hi, cfg: DecoderConfig,
           active=None):
    """Joint speeds that move each fingertip at ``surface-point velocity + R u`` in the damped least-squares sense.

    ``J`` (B, K, 3, N): linear fingertip Jacobians w.r.t. the N controlled joint coordinates (the caller folds
    coupled joints in). ``u_local`` (B, K, 3): requested relative velocities, m/s, local coordinates; each
    fingertip's request is capped at ``max_request_speed`` in norm. ``q``, ``q_lo``, ``q_hi`` (B, N) or (N,).
    ``active`` (B, K) zeroes a fingertip's weight.

    Solves ``(sum_i w_i J_i^T J_i + lambda I) qdot = sum_i w_i J_i^T target_i``, then clips to
    ``+/- max_joint_speed`` and to the joint-limit rates. The bounds are applied after the solve and the problem is
    not re-solved, so a bounded joint's share is lost rather than redistributed; ``residual`` shows it. Non-finite
    inputs are zero-filled and reported in ``nonfinite_input``; the output is always finite and bounded.

    Returns ``(qdot (B, N), diag)``; ``diag`` has ``target`` (B, K, 3), ``qdot_unbounded``, ``residual_unbounded``
    and ``residual`` (B, K) = ``|J_i qdot - target_i|`` in m/s, ``vel_bound_active`` and ``limit_active`` (B, N),
    ``request_clipped`` (B, K) and ``nonfinite_input`` (B,).
    """
    b, k, _, n = J.shape
    if len(cfg.tip_weights) != k:
        raise ValueError(f"cfg.tip_weights has {len(cfg.tip_weights)} entries for {k} fingertips")
    dtype, device = J.dtype, J.device
    q, q_lo, q_hi = (torch.as_tensor(x, dtype=dtype, device=device).expand(b, n) for x in (q, q_lo, q_hi))

    surface = surface_point_velocity(pen_pos, pen_linvel, pen_angvel, frames["point"]).to(dtype)
    raw = (J, u_local, surface, q)
    nonfinite = torch.zeros(b, dtype=torch.bool, device=device)
    for x in raw:
        nonfinite |= ~torch.isfinite(x).reshape(b, -1).all(-1)
    J, u_local, surface, q = (torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) for x in raw)

    speed = u_local.norm(dim=-1, keepdim=True)
    clipped = speed > cfg.max_request_speed
    u = torch.where(clipped, u_local * (cfg.max_request_speed / speed.clamp_min(1e-12)), u_local).to(dtype)
    target = surface + local_to_world(u, frames).to(dtype)

    w = torch.tensor(cfg.tip_weights, dtype=dtype, device=device).expand(b, k)
    if active is not None:
        w = w * active.to(dtype)
    Jw = J * w[:, :, None, None]
    A = torch.einsum("bkin,bkim->bnm", Jw, J) + cfg.damping * torch.eye(n, dtype=dtype, device=device)
    rhs = torch.einsum("bkin,bki->bn", Jw, target)
    solved = torch.linalg.solve(A, rhs)
    qdot_raw = torch.nan_to_num(solved, nan=0.0, posinf=0.0, neginf=0.0)
    nonfinite |= ~torch.isfinite(solved).all(-1)

    vel_active = qdot_raw.abs() > cfg.max_joint_speed
    qdot = qdot_raw.clamp(-cfg.max_joint_speed, cfg.max_joint_speed)
    rate_lo, rate_hi = _limit_rates(q, q_lo, q_hi, cfg)
    limited = torch.maximum(torch.minimum(qdot, rate_hi), rate_lo)
    limit_active = limited != qdot

    def residual(rates):
        return (torch.einsum("bkin,bn->bki", J, rates) - target).norm(dim=-1)

    diag = {
        "target": target,
        "qdot_unbounded": qdot_raw,
        "residual_unbounded": residual(qdot_raw),
        "residual": residual(limited),
        "vel_bound_active": vel_active,
        "limit_active": limit_active,
        "request_clipped": clipped.squeeze(-1),
        "nonfinite_input": nonfinite,
    }
    return limited, diag


def position_targets(q, qdot, cfg: DecoderConfig, q_lo, q_hi) -> torch.Tensor:
    """Normalised position-target action in [-1, 1]: ``2 (q + lookahead qdot - lo) / (hi - lo) - 1``, clamped.

    This inverts Isaac Lab's ``scale(action, lower, upper)`` over the *actuated* joints, so ``q`` and the limits
    must be in action order. The inherited env may also smooth targets
    (``act_moving_average * new + (1 - act_moving_average) * previous``) and clamps them to the limits; whether the
    average is 1.0 for this task is to be verified on the box before any tracking claim.
    """
    target = q + cfg.lookahead * qdot
    return (2.0 * (target - q_lo) / (q_hi - q_lo).clamp_min(1e-9) - 1.0).clamp(-1.0, 1.0)


def targets_from_action(action, q_lo, q_hi) -> torch.Tensor:
    """Inverse of :func:`position_targets` up to its clamp: the joint target an action names."""
    return q_lo + 0.5 * (action + 1.0) * (q_hi - q_lo)


# ---------------------------------------------------------------------------
#  Jacobian audit
# ---------------------------------------------------------------------------


def numerical_jacobian(fk, q, eps: float = 1e-4) -> torch.Tensor:
    """Central-difference Jacobian (B, K, 3, N) of ``fk``: (B, N) joint coordinates -> (B, K, 3) points."""
    cols = []
    for j in range(q.shape[-1]):
        step = torch.zeros_like(q)
        step[:, j] = eps
        cols.append((fk(q + step) - fk(q - step)) / (2.0 * eps))
    return torch.stack(cols, dim=-1)


def jacobian_relative_error(J, J_reference) -> torch.Tensor:
    """``|J - J_ref|_F / |J_ref|_F`` per env and fingertip (B, K); the reference is the finite-difference one."""
    num = (J - J_reference).flatten(-2).norm(dim=-1)
    return num / J_reference.flatten(-2).norm(dim=-1).clamp_min(1e-12)


# ---------------------------------------------------------------------------
#  Eligibility (never reads observed motion)
# ---------------------------------------------------------------------------

ELIGIBLE, FELL, BOUNDS_ACTIVE = "eligible", "fell", "bounds_active"


def bound_inactive_fraction(diag_sequence, chain_joint_mask) -> torch.Tensor:
    """Fraction (B,) of the given steps on which no joint of the chain had a velocity or joint-limit bound active.

    ``diag_sequence`` is the decoder's ``diag`` for each pulse step, in order; ``chain_joint_mask`` (N,) or (B, N)
    selects the commanded fingertip's joints among the controlled coordinates.
    """
    diag_sequence = list(diag_sequence)
    if not diag_sequence:
        raise ValueError("no pulse steps recorded")
    mask = torch.as_tensor(chain_joint_mask, dtype=torch.bool)
    if not mask.any():
        raise ValueError("chain_joint_mask selects no joint")
    active = torch.stack([torch.as_tensor(d["vel_bound_active"]).cpu() | torch.as_tensor(d["limit_active"]).cpu()
                          for d in diag_sequence])                               # (T, B, N)
    return 1.0 - (active & mask).any(-1).to(torch.float64).mean(0)


def pair_eligibility(fall_step, inactive_fraction, *, pulse_stop: int, min_inactive_fraction: float):
    """Per case: ``(eligible (C,) bool, labels)`` with labels ``eligible`` / ``fell`` / ``bounds_active``.

    ``fall_step`` (C,): the step at which the pen fell, negative for never. A pair is excluded as ``fell`` if
    ``0 <= fall_step <= pulse_stop``, else as ``bounds_active`` if ``inactive_fraction < min_inactive_fraction``.
    """
    fall_step = np.asarray(fall_step, dtype=np.int64)
    fraction = np.asarray(inactive_fraction, dtype=np.float64)
    fell = (fall_step >= 0) & (fall_step <= int(pulse_stop))
    bounded = ~(fraction >= float(min_inactive_fraction))        # a NaN fraction is not eligible
    labels = [FELL if f else BOUNDS_ACTIVE if b else ELIGIBLE for f, b in zip(fell, bounded)]
    return ~(fell | bounded), labels


# ---------------------------------------------------------------------------
#  Gate statistics (pure functions of recorded arrays)
# ---------------------------------------------------------------------------

#: R-093's thresholds as approved. The run reads its own copy from the hashed config; this is the reference.
GATE_THRESHOLDS = {
    "expected_cells": 30,
    "min_eligible_per_cell": 20,
    "sign_agreement_min": 0.80,
    "gain_median_min": 0.3,
    "gain_median_max": 1.5,
    "jacobian_rel_error_median_max": 0.15,
    "retention_diff_min": -0.10,
}
PASS, FAIL, INCONCLUSIVE = "pass", "fail", "inconclusive"
_RATE_TOLERANCE = 1e-9   # 0.3 - 0.4 is -0.10000000000000003 in floats; rates are k/40, so this changes no outcome


def cell_tracking(rel_world, R_first, axis: int, sign: float, eligible, labels=None,
                  requested_speed: float = 0.03) -> dict:
    """One fingertip-axis-sign cell: per-case gain, and sign agreement and median gain over its eligible cases.

    ``rel_world`` (C, T, 3): observed relative velocity (:func:`relative_world_velocity`) on the pulse steps only.
    ``R_first`` (C, 3, 3): the commanded fingertip's frame at the pulse's first step; every step is projected on
    it. ``gain`` = ``sign * mean_t(rel . R_first[:, :, axis]) / requested_speed``, so 1.0 is exact tracking and a
    positive value agrees in sign. Statistics are ``None`` when no case is eligible.
    """
    rel_world = np.asarray(rel_world, dtype=np.float64)
    R_first = np.asarray(R_first, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    if rel_world.ndim != 3 or rel_world.shape[1] == 0 or R_first.shape != (rel_world.shape[0], 3, 3):
        raise ValueError(f"bad shapes: rel_world {rel_world.shape}, R_first {R_first.shape}")
    along = np.einsum("cti,ci->ct", rel_world, R_first[:, :, int(axis)])
    gain = float(np.sign(sign)) * along.mean(axis=1) / float(requested_speed)
    chosen = gain[eligible]
    if not np.isfinite(chosen).all():
        raise ValueError("an eligible case has a non-finite observed velocity")
    out = {
        "cases": int(len(gain)),
        "eligible": int(eligible.sum()),
        "gain": [float(g) if np.isfinite(g) else None for g in gain],
        "eligible_mask": [bool(e) for e in eligible],
        "sign_agreement": float((chosen > 0).mean()) if len(chosen) else None,
        "gain_median": float(np.median(chosen)) if len(chosen) else None,
    }
    if labels is not None:
        out["excluded"] = {reason: int(sum(1 for x in labels if x == reason)) for reason in (FELL, BOUNDS_ACTIVE)}
    return out


def _check(value, ok, rule: str) -> dict:
    return {"value": value, "ok": ok, "rule": rule}


def gate_verdict(cells: dict, jacobian_rel_errors, hold_timeout_rate, neutral_timeout_rate, thresholds: dict) -> dict:
    """``pass`` / ``fail`` / ``inconclusive`` for the C1 gate, with every threshold's value and outcome.

    ``cells``: name -> :func:`cell_tracking` result. Sign agreement and the median gain are pooled over all
    eligible (cell, case) pairs; per-cell values are reported beside them. Precedence, as the idea states it:

    1. a measured Jacobian error at or above its bound, or a hold-minus-neutral timeout rate below its bound,
       fails the gate whatever the coverage;
    2. otherwise a missing cell, a cell under the eligible minimum, no eligible pair, or a missing Jacobian or
       retention measurement makes it inconclusive (never a pass);
    3. otherwise the pooled sign agreement and median gain decide.
    """
    missing = set(GATE_THRESHOLDS) - set(thresholds)
    if missing:
        raise ValueError(f"thresholds lack {sorted(missing)}")
    th = thresholds
    checks = {}

    errors = np.asarray(jacobian_rel_errors, dtype=np.float64).ravel()
    errors_ok = len(errors) > 0 and bool(np.isfinite(errors).all())
    jac = float(np.median(errors)) if errors_ok else None
    checks["jacobian_rel_error_median"] = _check(
        jac, None if jac is None else jac < th["jacobian_rel_error_median_max"],
        f"< {th['jacobian_rel_error_median_max']}")

    rates = [hold_timeout_rate, neutral_timeout_rate]
    rates_ok = all(r is not None and np.isfinite(r) for r in rates)
    diff = float(hold_timeout_rate) - float(neutral_timeout_rate) if rates_ok else None
    checks["retention_diff"] = _check(
        diff, None if diff is None else diff >= th["retention_diff_min"] - _RATE_TOLERANCE,
        f">= {th['retention_diff_min']} (zero-request hold minus neutral hold timeout rate)")

    short = sorted(name for name, c in cells.items() if c["eligible"] < th["min_eligible_per_cell"])
    covered = len(cells) == th["expected_cells"] and not short
    checks["coverage"] = _check(
        {"cells": len(cells), "under_minimum": short,
         "min_eligible": min((c["eligible"] for c in cells.values()), default=0)},
        covered, f"{th['expected_cells']} cells, each with >= {th['min_eligible_per_cell']} eligible cases")

    pooled = np.array([g for c in cells.values() for g, e in zip(c["gain"], c["eligible_mask"]) if e],
                      dtype=np.float64)
    agreement = float((pooled > 0).mean()) if len(pooled) else None
    median = float(np.median(pooled)) if len(pooled) else None
    checks["sign_agreement"] = _check(
        agreement, None if agreement is None else agreement >= th["sign_agreement_min"],
        f">= {th['sign_agreement_min']} over eligible pairs")
    checks["gain_median"] = _check(
        median, None if median is None else th["gain_median_min"] <= median <= th["gain_median_max"],
        f"in [{th['gain_median_min']}, {th['gain_median_max']}] over eligible pairs")

    independent = (checks["jacobian_rel_error_median"]["ok"], checks["retention_diff"]["ok"])
    tracking = (checks["sign_agreement"]["ok"], checks["gain_median"]["ok"])
    if False in independent:
        verdict = FAIL
    elif None in independent or not covered or None in tracking:
        verdict = INCONCLUSIVE
    else:
        verdict = PASS if all(tracking) else FAIL
    return {
        "verdict": verdict,
        "checks": checks,
        "eligible_pairs": int(len(pooled)),
        "per_cell": {name: {key: c.get(key) for key in ("cases", "eligible", "sign_agreement", "gain_median",
                                                         "excluded")} for name, c in cells.items()},
        "thresholds": dict(th),
    }
