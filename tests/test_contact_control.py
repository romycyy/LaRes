"""R-093 contact-frame geometry, decoder and gate statistics. No Isaac Lab; the toy hand below exists only here."""

import math
import os
import sys
import unittest

import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from lares.envs.isaac_pen_spin import cases  # noqa: E402
from lares.envs.isaac_pen_spin import contact_control as cc  # noqa: E402

D = torch.float64
RADIUS, HALF = cc.PEN_RADIUS, cc.PEN_HALF_LENGTH


def _rand_quat(gen, b):
    q = torch.randn(b, 4, generator=gen, dtype=D)
    return q / q.norm(dim=-1, keepdim=True)


def _axis_quat(axis, angle):
    axis = torch.tensor(axis, dtype=D)
    axis = axis / axis.norm()
    return torch.cat([torch.tensor([math.cos(angle / 2)], dtype=D), math.sin(angle / 2) * axis])[None]


def _assert_frames_valid(test, frames, atol=1e-9):
    R = frames["R"]
    test.assertTrue(torch.isfinite(R).all())
    test.assertTrue(torch.isfinite(frames["point"]).all())
    test.assertTrue(torch.isfinite(frames["distance"]).all())
    eye = torch.eye(3, dtype=R.dtype).expand_as(R)
    test.assertLess((R.transpose(-1, -2) @ R - eye).abs().max().item(), atol)
    test.assertLess((torch.linalg.det(R) - 1.0).abs().max().item(), atol)
    test.assertLess((torch.linalg.cross(R[..., 0], R[..., 1]) - R[..., 2]).abs().max().item(), atol)


def _adversarial(dtype=D):
    """Identity pen at the origin; tips on the axis, at the ends, at a cap centre, at a pole, inside, far away."""
    tips = torch.tensor([[
        [0.0, 0.0, 0.0],                   # 0 centre of the pen (on the segment)
        [0.03, 0.0, 0.0],                  # 1 on the segment
        [HALF, 0.0, 0.0],                  # 2 a cap centre (segment endpoint)
        [HALF + RADIUS, 0.0, 0.0],         # 3 the cap pole, on the surface
        [HALF + 0.5, 0.0, 0.0],            # 4 on the axis line, beyond the end
        [-HALF - 0.01, 0.0, 0.0],          # 5 same, other end, inside the cap
        [0.01, 0.004, 0.0],                # 6 inside the cylinder
        [50.0, -80.0, 30.0],               # 7 far away
        [0.0, RADIUS, 0.0],                # 8 on the surface
        [HALF, 0.0, 0.02],                 # 9 exactly on the cylinder/cap boundary plane
    ]], dtype=dtype)
    pos = torch.zeros(1, 3, dtype=dtype)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype)
    return pos, quat, tips


class TestCapsuleFrames(unittest.TestCase):
    def test_random_frames_are_orthonormal_right_handed(self):
        gen = torch.Generator().manual_seed(0)
        b, k = 64, 5
        pos = torch.randn(b, 3, generator=gen, dtype=D) * 0.2
        tips = pos[:, None] + torch.randn(b, k, 3, generator=gen, dtype=D) * 0.06
        frames = cc.capsule_frames(pos, _rand_quat(gen, b), tips)
        _assert_frames_valid(self, frames)
        self.assertFalse(frames["degenerate"].any())
        self.assertTrue(frames["on_cap"].any() and (~frames["on_cap"]).any())

    def test_adversarial_inputs_are_finite_and_flagged(self):
        for dtype, atol in ((D, 1e-9), (torch.float32, 1e-5)):
            pos, quat, tips = _adversarial(dtype)
            frames = cc.capsule_frames(pos, quat, tips)
            _assert_frames_valid(self, frames, atol)
            self.assertEqual(frames["degenerate"][0].tolist(),
                             [True, True, True, True, True, True, False, False, False, False])
            self.assertEqual(frames["on_cap"][0].tolist(),
                             [False, False, False, True, True, True, False, True, False, False])
        frames = cc.capsule_frames(*_adversarial())
        # on the segment: the declared fallback normal is the pen's local +Z, and the distance is -radius
        self.assertTrue(torch.allclose(frames["R"][0, 0, :, 0], torch.tensor([0.0, 0.0, 1.0], dtype=D)))
        self.assertAlmostEqual(frames["distance"][0, 0].item(), -RADIUS)
        # at the pole: normal = +axis, circumferential = local +Y, third = normal x circumferential = +Z
        self.assertTrue(torch.allclose(frames["R"][0, 3], torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=D)))
        self.assertAlmostEqual(frames["distance"][0, 3].item(), 0.0)
        self.assertAlmostEqual(frames["distance"][0, 5].item(), 0.01 - RADIUS)
        self.assertLess(frames["distance"][0, 6].item(), 0.0)

    def test_nonfinite_and_zero_quaternion_inputs(self):
        pos, quat, tips = _adversarial()
        tips = tips.clone()
        tips[0, 7] = float("nan")
        frames = cc.capsule_frames(pos, torch.zeros(1, 4, dtype=D), tips)
        _assert_frames_valid(self, frames)
        self.assertTrue(frames["degenerate"].all())
        bad = cc.capsule_frames(torch.full((1, 3), float("inf"), dtype=D), quat, tips)
        _assert_frames_valid(self, bad)

    def test_cylinder_frame_axes_and_rotation_sense(self):
        pos, quat, _ = _adversarial()
        tips = torch.tensor([[[0.0, 0.03, 0.0], [0.02, 0.0, -0.05]]], dtype=D)
        R = cc.capsule_frames(pos, quat, tips)["R"]
        self.assertTrue(torch.allclose(R[0, 0], torch.tensor(      # columns n=+Y, c = X x Y = +Z, a=+X
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=D)))
        self.assertTrue(torch.allclose(R[0, 1, :, 0], torch.tensor([0.0, 0.0, -1.0], dtype=D)))
        self.assertTrue(torch.allclose(R[0, 1, :, 2], torch.tensor([1.0, 0.0, 0.0], dtype=D)))

    def test_nearest_point_matches_brute_force(self):
        gen = torch.Generator().manual_seed(1)
        # a dense sample of the capsule surface in the pen frame
        phi = torch.linspace(0, 2 * math.pi, 361, dtype=D)[:-1]
        x = torch.linspace(-HALF, HALF, 281, dtype=D)
        cyl = torch.stack([x[:, None].expand(-1, len(phi)), RADIUS * torch.cos(phi).expand(len(x), -1),
                           RADIUS * torch.sin(phi).expand(len(x), -1)], dim=-1).reshape(-1, 3)
        th = torch.linspace(0, math.pi / 2, 61, dtype=D)
        capx = HALF + RADIUS * torch.cos(th)[:, None].expand(-1, len(phi))
        rho = RADIUS * torch.sin(th)[:, None]
        cap = torch.stack([capx, rho * torch.cos(phi), rho * torch.sin(phi)], dim=-1).reshape(-1, 3)
        local = torch.cat([cyl, cap, cap * torch.tensor([-1.0, 1.0, 1.0], dtype=D)])

        b, k = 6, 5
        pos = torch.randn(b, 3, generator=gen, dtype=D) * 0.1
        quat = _rand_quat(gen, b)
        tips = pos[:, None] + torch.randn(b, k, 3, generator=gen, dtype=D) * 0.07
        frames = cc.capsule_frames(pos, quat, tips)
        surface = torch.einsum("bij,sj->bsi", cc.quat_to_matrix(quat), local) + pos[:, None]
        dist = (tips[:, :, None, :] - surface[:, None, :, :]).norm(dim=-1)         # (B, K, S)
        brute, arg = dist.min(dim=-1)
        self.assertLess((brute - frames["distance"].abs()).abs().max().item(), 2e-5)
        nearest = torch.gather(surface, 1, arg[..., None].expand(-1, -1, 3))
        self.assertLess((nearest - frames["point"]).norm(dim=-1).max().item(), 5e-4)
        self.assertTrue(frames["on_cap"].any() and (~frames["on_cap"]).any())
        # the point is on the surface and the normal joins it to the tip
        rebuilt = frames["point"] + frames["distance"][..., None] * frames["R"][..., 0]
        self.assertLess((rebuilt - tips).abs().max().item(), 1e-12)

    def test_continuous_across_cap_boundary(self):
        pos, quat, _ = _adversarial()
        for offset in (0.004, RADIUS, 0.03):                 # inside, on, outside the surface
            for eps in (1e-6, 1e-9):
                tips = torch.tensor([[[HALF - eps, offset, 0.0], [HALF + eps, offset, 0.0]]], dtype=D)
                f = cc.capsule_frames(pos, quat, tips)
                self.assertEqual(f["on_cap"][0].tolist(), [False, True])
                self.assertLess((f["R"][0, 0] - f["R"][0, 1]).abs().max().item(), 10 * eps / offset)
                self.assertLess((f["point"][0, 0] - f["point"][0, 1]).abs().max().item(), 10 * eps)
                self.assertLess((f["distance"][0, 0] - f["distance"][0, 1]).abs().item(), 10 * eps)

    def test_cap_frame_tilts_but_keeps_the_circumferential(self):
        pos, quat, _ = _adversarial()
        tips = torch.tensor([[[HALF + 0.02, 0.02, 0.0]]], dtype=D)
        f = cc.capsule_frames(pos, quat, tips)
        s = math.sqrt(0.5)
        self.assertTrue(torch.allclose(f["R"][0, 0, :, 0], torch.tensor([s, s, 0.0], dtype=D)))
        self.assertTrue(torch.allclose(f["R"][0, 0, :, 1], torch.tensor([0.0, 0.0, 1.0], dtype=D)))
        self.assertTrue(torch.allclose(f["R"][0, 0, :, 2], torch.tensor([s, -s, 0.0], dtype=D)))
        self.assertGreater(f["R"][0, 0, 0, 2].item(), 0.0)   # the third column keeps a positive axial component

    def test_prev_gives_continuity_where_degenerate(self):
        pos, quat, _ = _adversarial()
        before = cc.capsule_frames(pos, quat, torch.tensor([[[0.01, 1e-3, -2e-3], [HALF + 0.02, 1e-3, 2e-3]]], dtype=D))
        on_axis = torch.tensor([[[0.01, 0.0, 0.0], [HALF + 0.02, 0.0, 0.0]]], dtype=D)
        cold = cc.capsule_frames(pos, quat, on_axis)
        warm = cc.capsule_frames(pos, quat, on_axis, prev=before)
        _assert_frames_valid(self, warm)
        self.assertTrue(warm["degenerate"].all())
        # segment: the previous normal is kept; cold start falls back to +Z
        self.assertLess((warm["R"][0, 0, :, 0] - before["R"][0, 0, :, 0]).abs().max().item(), 1e-12)
        self.assertGreater((cold["R"][0, 0, :, 0] - before["R"][0, 0, :, 0]).abs().max().item(), 0.5)
        # pole: the previous circumferential is kept (up to the small tilt of the normal)
        self.assertLess((warm["R"][0, 1, :, 1] - before["R"][0, 1, :, 1]).abs().max().item(), 1e-9)
        # prev is ignored where the frame is well defined
        tips = torch.tensor([[[0.0, 0.03, 0.0], [0.02, 0.0, -0.05]]], dtype=D)
        self.assertTrue(torch.equal(cc.capsule_frames(pos, quat, tips)["R"],
                                    cc.capsule_frames(pos, quat, tips, prev=before)["R"]))


class TestRelativeVelocity(unittest.TestCase):
    def test_rigidly_attached_fingertip_has_zero_relative_velocity(self):
        gen = torch.Generator().manual_seed(2)
        b, k = 16, 5
        pos = torch.randn(b, 3, generator=gen, dtype=D) * 0.2
        quat = _rand_quat(gen, b)
        v = torch.randn(b, 3, generator=gen, dtype=D)
        w = torch.randn(b, 3, generator=gen, dtype=D) * 5
        tips = pos[:, None] + torch.randn(b, k, 3, generator=gen, dtype=D) * 0.05
        frames = cc.capsule_frames(pos, quat, tips)
        # the tip rides with the pen, but the observed quantity is relative to the surface point, not to the tip:
        # a rigidly attached surface-contact tip is at the surface point
        on_surface = frames["point"]
        frames = cc.capsule_frames(pos, quat, on_surface + 1e-9 * frames["R"][..., 0])
        tip_v = v[:, None] + torch.linalg.cross(w[:, None].expand(b, k, 3), on_surface - pos[:, None])
        rel = cc.relative_local_velocity(tip_v, frames, pos, v, w)
        self.assertLess(rel.abs().max().item(), 1e-6)

    def test_attached_tip_off_the_surface_differs_only_by_the_lever(self):
        """A tip rigidly attached at distance d from the surface: relative velocity = w x (d * normal)."""
        gen = torch.Generator().manual_seed(3)
        b, k = 8, 5
        pos = torch.randn(b, 3, generator=gen, dtype=D) * 0.2
        quat = _rand_quat(gen, b)
        v = torch.randn(b, 3, generator=gen, dtype=D)
        w = torch.randn(b, 3, generator=gen, dtype=D) * 5
        tips = pos[:, None] + torch.randn(b, k, 3, generator=gen, dtype=D) * 0.05
        frames = cc.capsule_frames(pos, quat, tips)
        tip_v = v[:, None] + torch.linalg.cross(w[:, None].expand(b, k, 3), tips - pos[:, None])
        rel = cc.relative_world_velocity(tip_v, frames["point"], pos, v, w)
        lever = torch.linalg.cross(w[:, None].expand(b, k, 3), frames["distance"][..., None] * frames["R"][..., 0])
        self.assertLess((rel - lever).abs().max().item(), 1e-12)
        self.assertLess(cc.world_to_local(rel, frames)[..., 0].abs().max().item(), 1e-12)   # never along the normal

    def test_pure_roll_is_pure_negative_circumferential(self):
        gen = torch.Generator().manual_seed(4)
        b, k, rate = 8, 5, 6.0
        pos = torch.randn(b, 3, generator=gen, dtype=D) * 0.2
        quat = _rand_quat(gen, b)
        axis = cc.quat_to_matrix(quat)[:, :, 0]
        # stationary tips beside the cylinder section, at various distances
        t = (torch.rand(b, k, 1, generator=gen, dtype=D) - 0.5) * 2 * HALF * 0.9
        radial = torch.linalg.cross(axis[:, None].expand(b, k, 3), torch.randn(b, k, 3, generator=gen, dtype=D))
        radial = radial / radial.norm(dim=-1, keepdim=True) * (RADIUS + torch.rand(b, k, 1, generator=gen, dtype=D) * 0.02)
        tips = pos[:, None] + t * axis[:, None] + radial
        frames = cc.capsule_frames(pos, quat, tips)
        self.assertFalse(frames["on_cap"].any())
        rel = cc.relative_local_velocity(torch.zeros(b, k, 3, dtype=D), frames, pos,
                                         torch.zeros(b, 3, dtype=D), rate * axis)
        expected = torch.zeros(b, k, 3, dtype=D)
        expected[..., 1] = -rate * RADIUS          # pen rolling +w about +X, tip still: -w * radius
        self.assertLess((rel - expected).abs().max().item(), 1e-12)

    def test_local_world_round_trip(self):
        gen = torch.Generator().manual_seed(5)
        pos = torch.zeros(4, 3, dtype=D)
        frames = cc.capsule_frames(pos, _rand_quat(gen, 4), torch.randn(4, 5, 3, generator=gen, dtype=D) * 0.05)
        u = torch.randn(4, 5, 3, generator=gen, dtype=D)
        self.assertLess((cc.world_to_local(cc.local_to_world(u, frames), frames) - u).abs().max().item(), 1e-12)
        self.assertLess((cc.local_to_world(u, frames) - (frames["R"] @ u[..., None])[..., 0]).abs().max().item(), 1e-12)

    def test_surface_point_velocity(self):
        v = cc.surface_point_velocity(torch.tensor([[1.0, 0.0, 0.0]], dtype=D), torch.tensor([[0.0, 0.0, 2.0]], dtype=D),
                                      torch.tensor([[0.0, 0.0, 3.0]], dtype=D), torch.tensor([[[2.0, 0.0, 0.0]]], dtype=D))
        self.assertTrue(torch.allclose(v, torch.tensor([[[0.0, 3.0, 2.0]]], dtype=D)))


class TestObservation(unittest.TestCase):
    def test_state_from_obs_uses_the_named_layout(self):
        s = cases.obs_slices()
        obs = torch.arange(2 * cases.OBS_DIM, dtype=D).reshape(2, cases.OBS_DIM)
        state = cc.state_from_obs(obs)
        self.assertTrue(torch.equal(state["pen_pos"], obs[:, s["obj_pos"]]))
        self.assertTrue(torch.equal(state["pen_quat"], obs[:, s["obj_rot"]]))
        self.assertTrue(torch.equal(state["pen_linvel"], obs[:, s["obj_linvel"]]))
        self.assertTrue(torch.allclose(state["pen_angvel"], obs[:, s["obj_angvel"]] / 0.2))
        self.assertTrue(torch.allclose(state["hand_dof_vel"], obs[:, s["hand_dof_vel"]] / 0.2))
        self.assertEqual(tuple(state["tip_pos"].shape), (2, 5, 3))
        start = s["fingertip_vel"].start
        self.assertTrue(torch.equal(state["tip_pos"][:, 4], obs[:, s["fingertip_pos"]][:, 12:15]))       # thumb last
        self.assertTrue(torch.equal(state["tip_linvel"][:, 1], obs[:, start + 6:start + 9]))
        self.assertTrue(torch.equal(state["tip_angvel"][:, 1], obs[:, start + 9:start + 12]))
        scaled = cc.state_from_obs(obs, fingertip_vel_scale=0.2)
        self.assertTrue(torch.allclose(scaled["tip_linvel"], state["tip_linvel"] / 0.2))
        self.assertEqual(tuple(cc.state_from_obs(obs.numpy())["tip_linvel"].shape), (2, 5, 3))


# ---------------------------------------------------------------------------
#  A toy hand: K fingers, each z-yaw + three y-pitch joints; the last pitch joint is coupled (q_passive = q_previous).
# ---------------------------------------------------------------------------

K_TOY, FULL_PER, CTRL_PER = 3, 4, 3
LINKS = (0.045, 0.03, 0.025)
BASES = torch.tensor([[0.0, -0.03, 0.0], [0.0, 0.0, 0.0], [0.0, 0.03, 0.0]], dtype=D)


def toy_coupling(dtype=D):
    """(N_full, N_ctrl): full joint velocities from controlled ones; each finger's 4th joint copies its 3rd."""
    c = torch.zeros(K_TOY * FULL_PER, K_TOY * CTRL_PER, dtype=dtype)
    for f in range(K_TOY):
        for j in range(CTRL_PER):
            c[f * FULL_PER + j, f * CTRL_PER + j] = 1.0
        c[f * FULL_PER + 3, f * CTRL_PER + 2] = 1.0
    return c


def toy_fk_full(q_full):
    """(B, 12) -> (B, 3, 3) fingertip positions."""
    q = q_full.reshape(q_full.shape[0], K_TOY, FULL_PER)
    yaw, pitch = q[..., 0], torch.cumsum(q[..., 1:], dim=-1)
    reach = sum(l * torch.cos(pitch[..., i]) for i, l in enumerate(LINKS))
    height = sum(l * torch.sin(pitch[..., i]) for i, l in enumerate(LINKS))
    tip = torch.stack([reach * torch.cos(yaw), reach * torch.sin(yaw), height], dim=-1)
    return tip + BASES.to(q_full.dtype)


def toy_fk(q):
    return toy_fk_full(q @ toy_coupling(q.dtype).T)


def toy_jacobian(q):
    """Effective (B, K, 3, N_ctrl) Jacobian: autograd on the full chain, coupling folded in by the module."""
    q_full = q @ toy_coupling(q.dtype).T
    per_env = [torch.autograd.functional.jacobian(lambda x: toy_fk_full(x[None])[0], row) for row in q_full]
    return cc.effective_jacobian(torch.stack(per_env), toy_coupling(q.dtype))


def toy_q(b, gen):
    base = torch.tensor([0.1, 0.5, 0.4] * K_TOY, dtype=D)
    return base + 0.15 * torch.randn(b, K_TOY * CTRL_PER, generator=gen, dtype=D)


TOY_CFG = cc.DecoderConfig(damping=1e-8, tip_weights=(1.0,) * K_TOY, max_joint_speed=20.0, control_dt=1e-3,
                           lookahead=1e-3, joint_limit_margin=0.0, request_scale=0.05, max_request_speed=0.05)
WIDE = (torch.full((K_TOY * CTRL_PER,), -10.0, dtype=D), torch.full((K_TOY * CTRL_PER,), 10.0, dtype=D))


def toy_pen(b):
    pos = torch.tensor([[0.07, 0.0, 0.08]], dtype=D).expand(b, 3).clone()
    quat = _axis_quat([0.0, 0.0, 1.0], 1.3).expand(b, 4).clone()          # axis roughly along world Y
    return pos, quat


class TestJacobian(unittest.TestCase):
    def test_effective_jacobian_matches_finite_differences(self):
        q = toy_q(5, torch.Generator().manual_seed(6))
        J = toy_jacobian(q)
        J_fd = cc.numerical_jacobian(toy_fk, q, eps=1e-5)
        self.assertEqual(tuple(J.shape), (5, K_TOY, 3, K_TOY * CTRL_PER))
        self.assertLess(cc.jacobian_relative_error(J, J_fd).max().item(), 1e-8)
        # leaving the coupled joint out is a visible error, which is what the audit must be able to see
        full = torch.stack([torch.autograd.functional.jacobian(lambda x: toy_fk_full(x[None])[0], row)
                            for row in q @ toy_coupling().T])
        dropped = full[..., [i for i in range(K_TOY * FULL_PER) if i % FULL_PER != 3]]
        self.assertGreater(cc.jacobian_relative_error(dropped, J_fd).min().item(), 0.05)

    def test_relative_error_is_scale_relative(self):
        J = torch.randn(2, 3, 3, 4, dtype=D)
        self.assertTrue(torch.allclose(cc.jacobian_relative_error(1.1 * J, J), torch.full((2, 3), 0.1, dtype=D)))


class TestDecoder(unittest.TestCase):
    def _rollout(self, u_local, pen_linvel, pen_angvel, steps=40, b=4):
        gen = torch.Generator().manual_seed(7)
        q = toy_q(b, gen)
        pos, quat = toy_pen(b)
        dt = TOY_CFG.control_dt
        frames, worst_residual, observed = None, 0.0, []
        for _ in range(steps):
            tips = toy_fk(q)
            frames = cc.capsule_frames(pos, quat, tips, prev=frames)
            qdot, diag = cc.decode(toy_jacobian(q), frames, u_local, pos, pen_linvel, pen_angvel, q, *WIDE, TOY_CFG)
            self.assertFalse(diag["vel_bound_active"].any() or diag["limit_active"].any())
            worst_residual = max(worst_residual, diag["residual"].max().item())
            q_next = q + dt * qdot
            tip_v = (toy_fk(q_next) - tips) / dt                          # what the kinematic toy actually did
            observed.append(cc.relative_local_velocity(tip_v, frames, pos, pen_linvel, pen_angvel))
            q = q_next
            # the pen is advanced rigidly (first order; the rotation is left fixed over this short horizon)
            pos = pos + dt * pen_linvel
        return torch.stack(observed), worst_residual

    def test_tracks_a_local_request_on_the_toy_chain(self):
        b = 4
        u = torch.zeros(b, K_TOY, 3, dtype=D)
        u[:, 0, 1] = 0.03          # finger 0: +circumferential
        u[:, 1, 2] = -0.03         # finger 1: -axial
        u[:, 2, 0] = 0.02          # finger 2: +normal
        zero = torch.zeros(b, 3, dtype=D)
        observed, residual = self._rollout(u, zero, zero)
        self.assertLess(residual, 1e-4)
        self.assertLess((observed - u).abs().max().item(), 1.5e-3)        # 5% of the 0.03 m/s request

    def test_zero_request_follows_a_moving_pen(self):
        b = 4
        u = torch.zeros(b, K_TOY, 3, dtype=D)
        pen_v = torch.tensor([[0.02, -0.01, 0.015]], dtype=D).expand(b, 3)
        observed, residual = self._rollout(u, pen_v, torch.zeros(b, 3, dtype=D))
        self.assertLess(residual, 1e-4)
        self.assertLess(observed.abs().max().item(), 1.5e-3)

    def test_target_is_surface_velocity_plus_rotated_request(self):
        gen = torch.Generator().manual_seed(8)
        b = 3
        q = toy_q(b, gen)
        pos, quat = toy_pen(b)
        v, w = torch.randn(b, 3, generator=gen, dtype=D) * 0.05, torch.randn(b, 3, generator=gen, dtype=D)
        frames = cc.capsule_frames(pos, quat, toy_fk(q))
        u = torch.randn(b, K_TOY, 3, generator=gen, dtype=D) * 0.01
        _, diag = cc.decode(toy_jacobian(q), frames, u, pos, v, w, q, *WIDE, TOY_CFG)
        expected = cc.surface_point_velocity(pos, v, w, frames["point"]) + cc.local_to_world(u, frames)
        self.assertLess((diag["target"] - expected).abs().max().item(), 1e-12)
        self.assertFalse(diag["request_clipped"].any())

    def test_solution_is_the_damped_least_squares_minimiser(self):
        gen = torch.Generator().manual_seed(9)
        b, k, n = 2, 3, 9
        J = torch.randn(b, k, 3, n, generator=gen, dtype=D) * 0.05
        cfg = cc.DecoderConfig(damping=1e-3, tip_weights=(1.0, 0.5, 2.0), max_joint_speed=1e6, lookahead=1e-3,
                               joint_limit_margin=0.0, max_request_speed=1.0)
        pos, quat = toy_pen(b)
        frames = cc.capsule_frames(pos, quat, pos[:, None] + torch.randn(b, k, 3, generator=gen, dtype=D) * 0.04)
        u = torch.randn(b, k, 3, generator=gen, dtype=D) * 0.02
        zero = torch.zeros(b, 3, dtype=D)
        q = torch.zeros(b, n, dtype=D)
        big = (torch.full((n,), -1e6, dtype=D), torch.full((n,), 1e6, dtype=D))
        qdot, diag = cc.decode(J, frames, u, pos, zero, zero, q, *big, cfg)
        w = torch.tensor(cfg.tip_weights, dtype=D)

        def cost(x):
            err = torch.einsum("bkin,bn->bki", J, x) - diag["target"]
            return (w * err.pow(2).sum(-1)).sum(-1) + cfg.damping * x.pow(2).sum(-1)

        base = cost(qdot)
        for _ in range(20):
            self.assertTrue((cost(qdot + 1e-3 * torch.randn(b, n, generator=gen, dtype=D)) >= base).all())
        # an inactive fingertip does not shape the solution
        active = torch.tensor([[True, False, True]] * b)
        masked, _ = cc.decode(J, frames, u, pos, zero, zero, q, *big, cfg, active=active)
        u2 = u.clone()
        u2[:, 1] = 0.9
        masked2, _ = cc.decode(J, frames, u2, pos, zero, zero, q, *big, cfg, active=active)
        self.assertLess((masked - masked2).abs().max().item(), 1e-10)
        self.assertGreater((masked - qdot).abs().max().item(), 1e-6)

    def test_infeasible_and_huge_requests_stay_bounded_and_finite(self):
        gen = torch.Generator().manual_seed(10)
        cfg = cc.DecoderConfig(damping=1e-6, tip_weights=(1.0,) * K_TOY, max_joint_speed=1.5, lookahead=0.05,
                               joint_limit_margin=0.02, max_request_speed=0.05)
        for dtype in (D, torch.float32):
            b = 6
            q = toy_q(b, gen).to(dtype)
            J = toy_jacobian(q.to(D)).to(dtype)
            J[1] = 0.0                                             # no authority at all
            J[2, :, :, 1:] = J[2, :, :, :1]                        # rank one
            pos, quat = (x.to(dtype) for x in toy_pen(b))
            frames = cc.capsule_frames(pos, quat, toy_fk(q.to(D)).to(dtype))
            u = torch.full((b, K_TOY, 3), 1e8, dtype=dtype)
            u[3] = float("nan")
            u[4] = float("inf")
            pen_v = torch.zeros(b, 3, dtype=dtype)
            pen_v[5] = 1e6                                         # the pen "moves" at an absurd speed
            lo, hi = q - 0.3, q + 0.3
            qdot, diag = cc.decode(J, frames, u, pos, pen_v, torch.zeros(b, 3, dtype=dtype), q, lo, hi, cfg)
            self.assertEqual(qdot.dtype, dtype)
            self.assertTrue(torch.isfinite(qdot).all())
            self.assertLessEqual(qdot.abs().max().item(), cfg.max_joint_speed + 1e-6)
            for key in ("target", "residual", "residual_unbounded", "qdot_unbounded"):
                self.assertTrue(torch.isfinite(diag[key]).all(), key)
            self.assertTrue(diag["request_clipped"][0].all())
            self.assertEqual(diag["nonfinite_input"].tolist(), [False, False, False, True, True, False])
            self.assertEqual(qdot[1].abs().max().item(), 0.0)
            self.assertTrue(diag["vel_bound_active"][5].any())
            self.assertGreater(diag["residual"][5].min().item(), 1e3)      # the miss is reported, not hidden
            action = cc.position_targets(q, qdot, cfg, lo, hi)
            self.assertTrue(torch.isfinite(action).all() and (action.abs() <= 1).all())

    def test_joint_limits_bound_the_predicted_position(self):
        gen = torch.Generator().manual_seed(11)
        b = 4
        cfg = cc.DecoderConfig(damping=1e-8, tip_weights=(1.0,) * K_TOY, max_joint_speed=50.0, lookahead=0.1,
                               joint_limit_margin=0.03, max_request_speed=0.05)
        q = toy_q(b, gen)
        pos, quat = toy_pen(b)
        frames = cc.capsule_frames(pos, quat, toy_fk(q))
        u = torch.zeros(b, K_TOY, 3, dtype=D)
        u[:, :, 2] = 0.05
        zero = torch.zeros(b, 3, dtype=D)
        free, free_diag = cc.decode(toy_jacobian(q), frames, u, pos, zero, zero, q, *WIDE, cfg)
        self.assertFalse(free_diag["limit_active"].any())
        lo, hi = q - 0.04, q + 0.04                         # 0.01 rad of travel left inside the margin
        qdot, diag = cc.decode(toy_jacobian(q), frames, u, pos, zero, zero, q, lo, hi, cfg)
        predicted = q + cfg.lookahead * qdot
        self.assertTrue((predicted >= lo + cfg.joint_limit_margin - 1e-12).all())
        self.assertTrue((predicted <= hi - cfg.joint_limit_margin + 1e-12).all())
        self.assertTrue(torch.equal(diag["limit_active"], free.abs() > 0.1 + 1e-12))
        self.assertTrue(diag["limit_active"].any())
        self.assertFalse(diag["vel_bound_active"].any())
        self.assertTrue((diag["residual"] >= diag["residual_unbounded"] - 1e-12).all())
        # a joint already beyond the margin may come back but not go further out, and is not forced back
        outside_hi = q + 0.01                                # q is 0.02 past (hi - margin)
        qdot_out, diag_out = cc.decode(toy_jacobian(q), frames, u, pos, zero, zero, q, q - 5.0, outside_hi, cfg)
        self.assertTrue((qdot_out <= 1e-12).all())
        self.assertTrue(torch.equal(qdot_out, free.clamp(max=0.0)))
        self.assertTrue(torch.equal(diag_out["limit_active"], free > 0))

    def test_velocity_bound_is_recorded(self):
        gen = torch.Generator().manual_seed(12)
        cfg = cc.DecoderConfig(damping=1e-8, tip_weights=(1.0,) * K_TOY, max_joint_speed=0.2, lookahead=1e-3,
                               joint_limit_margin=0.0, max_request_speed=0.05)
        q = toy_q(2, gen)
        pos, quat = toy_pen(2)
        frames = cc.capsule_frames(pos, quat, toy_fk(q))
        u = torch.zeros(2, K_TOY, 3, dtype=D)
        u[:, 0, 1] = 0.05
        zero = torch.zeros(2, 3, dtype=D)
        qdot, diag = cc.decode(toy_jacobian(q), frames, u, pos, zero, zero, q, *WIDE, cfg)
        self.assertTrue(torch.equal(diag["vel_bound_active"], diag["qdot_unbounded"].abs() > 0.2))
        self.assertTrue(diag["vel_bound_active"][:, :CTRL_PER].any())
        self.assertFalse(diag["vel_bound_active"][:, CTRL_PER:].any())     # only finger 0 was asked to move
        self.assertLessEqual(qdot.abs().max().item(), 0.2)

    def test_weight_count_must_match(self):
        with self.assertRaises(ValueError):
            cc.decode(torch.zeros(1, 2, 3, 4), {"point": torch.zeros(1, 2, 3), "R": torch.eye(3).expand(1, 2, 3, 3)},
                      torch.zeros(1, 2, 3), torch.zeros(1, 3), torch.zeros(1, 3), torch.zeros(1, 3),
                      torch.zeros(4), -torch.ones(4), torch.ones(4), cc.DecoderConfig())

    def test_scale_request(self):
        cfg = cc.DecoderConfig(request_scale=0.04)
        out = cc.scale_request(torch.tensor([[0.5, -2.0, float("nan"), 1.0, 0.0, float("inf")]]), cfg)
        self.assertEqual(tuple(out.shape), (1, 2, 3))
        self.assertTrue(torch.allclose(out, torch.tensor([[[0.02, -0.04, 0.0], [0.04, 0.0, 0.04]]])))


class TestPositionTargets(unittest.TestCase):
    def test_round_trip_and_clamp(self):
        cfg = cc.DecoderConfig(lookahead=0.1)
        lo, hi = torch.tensor([-1.0, 0.0, -0.5], dtype=D), torch.tensor([1.0, 1.6, 0.5], dtype=D)
        q = torch.tensor([[0.2, 0.8, 0.0]], dtype=D)
        qdot = torch.tensor([[1.0, -2.0, 30.0]], dtype=D)
        action = cc.position_targets(q, qdot, cfg, lo, hi)
        self.assertTrue(torch.allclose(action, torch.tensor([[0.3, -0.25, 1.0]], dtype=D)))
        target = cc.targets_from_action(action, lo, hi)
        self.assertTrue(torch.allclose(target, torch.tensor([[0.3, 0.6, 0.5]], dtype=D)))   # the third is clamped
        self.assertTrue(torch.allclose(cc.position_targets(target, torch.zeros_like(q), cfg, lo, hi), action))


class TestConfig(unittest.TestCase):
    def test_hash_is_stable_and_sensitive(self):
        a = cc.DecoderConfig()
        b = cc.DecoderConfig.from_dict(a.to_dict())
        self.assertEqual(a, b)
        self.assertEqual(a.sha256(), b.sha256())
        self.assertEqual(a.sha256(), cc.DecoderConfig(tip_weights=[1, 1, 1, 1, 1], max_joint_speed=2).sha256())
        self.assertEqual(len(a.sha256()), 64)
        self.assertNotEqual(a.sha256(), cc.DecoderConfig(damping=2e-4).sha256())
        self.assertNotEqual(a.sha256(), cc.DecoderConfig(tip_weights=(1, 1, 1, 1, 0)).sha256())
        import json
        self.assertEqual(cc.DecoderConfig.from_dict(json.loads(json.dumps(a.to_dict()))), a)
        # a literal digest, so an accidental change of the serialisation is caught
        self.assertEqual(cc.DecoderConfig(damping=0.5, tip_weights=(1.0,), max_joint_speed=1.0, control_dt=0.5,
                                          lookahead=0.5, joint_limit_margin=0.0, request_scale=1.0,
                                          max_request_speed=1.0).sha256(), _LITERAL_DIGEST)

    def test_refuses_bad_values(self):
        for bad in ({"damping": 0.0}, {"lookahead": 0.0}, {"max_joint_speed": -1.0}, {"joint_limit_margin": -0.1},
                    {"tip_weights": ()}, {"tip_weights": (1.0, -1.0)}, {"damping": float("nan")}):
            with self.assertRaises(ValueError):
                cc.DecoderConfig(**bad)
        with self.assertRaises(ValueError):
            cc.DecoderConfig.from_dict({"lambda": 1.0})
        with self.assertRaises(Exception):
            cc.DecoderConfig().damping = 1.0


def _literal_digest():
    import hashlib
    text = ('{"control_dt":0.5,"damping":0.5,"joint_limit_margin":0.0,"lookahead":0.5,"max_joint_speed":1.0,'
            '"max_request_speed":1.0,"request_scale":1.0,"tip_weights":[1.0]}')
    return hashlib.sha256(text.encode()).hexdigest()


_LITERAL_DIGEST = _literal_digest()


class TestEligibility(unittest.TestCase):
    def _diags(self):
        n, steps, b = 6, 10, 3
        chain = torch.tensor([True, True, False, False, False, False])
        out = []
        for t in range(steps):
            vel = torch.zeros(b, n, dtype=torch.bool)
            lim = torch.zeros(b, n, dtype=torch.bool)
            vel[0, 3] = True                 # env 0: always bounded, but off the chain
            vel[1, 0] = t < 2                # env 1: velocity bound on the chain in 2 of 10 steps
            lim[2, 1] = t < 2                # env 2: limit on the chain in 2 steps, velocity in another
            vel[2, 0] = t == 5
            out.append({"vel_bound_active": vel, "limit_active": lim})
        return out, chain

    def test_bound_inactive_fraction(self):
        diags, chain = self._diags()
        frac = cc.bound_inactive_fraction(diags, chain)
        self.assertTrue(torch.allclose(frac, torch.tensor([1.0, 0.8, 0.7], dtype=torch.float64)))
        per_env = chain.expand(3, 6).clone()
        per_env[2] = False
        per_env[2, 5] = True
        self.assertEqual(cc.bound_inactive_fraction(diags, per_env).tolist(), [1.0, 0.8, 1.0])
        with self.assertRaises(ValueError):
            cc.bound_inactive_fraction([], chain)
        with self.assertRaises(ValueError):
            cc.bound_inactive_fraction(diags, torch.zeros(6, dtype=torch.bool))

    def test_decoder_diags_feed_the_fraction(self):
        gen = torch.Generator().manual_seed(13)
        cfg = cc.DecoderConfig(damping=1e-8, tip_weights=(1.0,) * K_TOY, max_joint_speed=0.2, lookahead=1e-3,
                               joint_limit_margin=0.0, max_request_speed=0.05)
        q = toy_q(2, gen)
        pos, quat = toy_pen(2)
        frames = cc.capsule_frames(pos, quat, toy_fk(q))
        u = torch.zeros(2, K_TOY, 3, dtype=D)
        u[0, 0, 1] = 0.05                                   # env 0 saturates finger 0; env 1 asks for nothing
        zero = torch.zeros(2, 3, dtype=D)
        _, diag = cc.decode(toy_jacobian(q), frames, u, pos, zero, zero, q, *WIDE, cfg)
        finger0 = torch.arange(K_TOY * CTRL_PER) < CTRL_PER
        self.assertEqual(cc.bound_inactive_fraction([diag] * 4, finger0).tolist(), [0.0, 1.0])
        self.assertEqual(cc.bound_inactive_fraction([diag] * 4, ~finger0).tolist(), [1.0, 1.0])

    def test_pair_eligibility_labels(self):
        eligible, labels = cc.pair_eligibility([-1, 12, 40, 41, -1, 5, -1], [1.0, 1.0, 1.0, 0.8, 0.79, 0.1, float("nan")],
                                               pulse_stop=40, min_inactive_fraction=0.8)
        self.assertEqual(labels, ["eligible", "fell", "fell", "eligible", "bounds_active", "fell", "bounds_active"])
        self.assertEqual(eligible.tolist(), [True, False, False, True, False, False, False])


def _cell(gains, eligible=None, speed=0.03, axis=1, sign=1.0, steps=30, labels=None):
    """A synthetic cell whose per-case gain along the requested axis is ``gains``; the frame is a fixed rotation."""
    gains = np.asarray(gains, dtype=float)
    c = len(gains)
    R = cc.quat_to_matrix(_axis_quat([1.0, 2.0, -0.5], 0.9)).numpy()[0]
    rng = np.random.default_rng(0)
    rel = np.zeros((c, steps, 3))
    wobble = rng.normal(size=(c, steps)) * 0.01
    wobble -= wobble.mean(axis=1, keepdims=True)             # zero-mean in time: does not move the mean
    rel += (sign * gains[:, None] * speed + wobble)[..., None] * R[:, axis]
    rel += rng.normal(size=(c, steps, 1)) * 0.02 * R[:, (axis + 1) % 3]     # motion on another axis is ignored
    eligible = np.ones(c, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    return cc.cell_tracking(rel, np.broadcast_to(R, (c, 3, 3)), axis, sign, eligible, labels=labels,
                            requested_speed=speed)


def _cells(gains, n=30, **kw):
    return {f"cell{i:02d}": _cell(gains, **kw) for i in range(n)}


class TestGateStatistics(unittest.TestCase):
    TH = dict(cc.GATE_THRESHOLDS)

    def test_cell_tracking(self):
        gains = np.linspace(0.5, 1.2, 40)
        for axis in range(3):
            for sign in (1.0, -1.0):
                cell = _cell(gains, axis=axis, sign=sign)
                self.assertTrue(np.allclose(cell["gain"], gains))
                self.assertEqual(cell["sign_agreement"], 1.0)
                self.assertAlmostEqual(cell["gain_median"], float(np.median(gains)))
        mixed = np.array([1.0] * 30 + [-0.5] * 10)
        eligible = np.array([True] * 25 + [False] * 5 + [True] * 10)
        cell = _cell(mixed, eligible, labels=["eligible"] * 25 + ["fell"] * 2 + ["bounds_active"] * 3 + ["eligible"] * 10)
        self.assertEqual((cell["cases"], cell["eligible"]), (40, 35))
        self.assertAlmostEqual(cell["sign_agreement"], 25 / 35)
        self.assertAlmostEqual(cell["gain_median"], 1.0)
        self.assertEqual(cell["excluded"], {"fell": 2, "bounds_active": 3})
        empty = _cell(mixed, np.zeros(40, dtype=bool))
        self.assertIsNone(empty["sign_agreement"])
        self.assertIsNone(empty["gain_median"])

    def test_projection_uses_the_first_step_frame(self):
        """The tip moves along the first-step circumferential; a later, rotated frame is never consulted."""
        pos = torch.zeros(1, 3, dtype=D)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=D)
        first = cc.capsule_frames(pos, quat, torch.tensor([[[0.0, 0.02, 0.0]]], dtype=D))
        rel = np.tile(0.03 * first["R"][0, 0, :, 1].numpy(), (1, 30, 1))
        cell = cc.cell_tracking(rel, first["R"][:, 0].numpy(), 1, 1.0, [True])
        self.assertAlmostEqual(cell["gain"][0], 1.0)
        self.assertAlmostEqual(cc.cell_tracking(rel, first["R"][:, 0].numpy(), 1, -1.0, [True])["gain"][0], -1.0)

    def test_cell_tracking_refuses_bad_input(self):
        with self.assertRaises(ValueError):
            cc.cell_tracking(np.zeros((2, 0, 3)), np.zeros((2, 3, 3)), 0, 1.0, [True, True])
        rel = np.zeros((2, 5, 3))
        rel[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            cc.cell_tracking(rel, np.tile(np.eye(3), (2, 1, 1)), 0, 1.0, [True, True])
        fine = cc.cell_tracking(rel, np.tile(np.eye(3), (2, 1, 1)), 0, 1.0, [False, True])   # NaN only where excluded
        self.assertIsNone(fine["gain"][0])

    def _verdict(self, cells, jac=(0.05, 0.08, 0.2), hold=0.9, neutral=0.95, th=None):
        return cc.gate_verdict(cells, jac, hold, neutral, th or self.TH)

    def test_pass(self):
        out = self._verdict(_cells(np.linspace(0.4, 1.3, 40)))
        self.assertEqual(out["verdict"], "pass")
        self.assertTrue(all(c["ok"] for c in out["checks"].values()))
        self.assertEqual(out["eligible_pairs"], 1200)
        self.assertEqual(set(out["checks"]), {"jacobian_rel_error_median", "retention_diff", "coverage",
                                              "sign_agreement", "gain_median"})
        import json
        json.dumps(out)                                           # the verdict is a plain record

    def test_tracking_failures(self):
        low_agreement = self._verdict(_cells(np.array([1.0] * 31 + [-1.0] * 9)))          # 0.775
        self.assertEqual(low_agreement["verdict"], "fail")
        self.assertFalse(low_agreement["checks"]["sign_agreement"]["ok"])
        self.assertTrue(low_agreement["checks"]["gain_median"]["ok"])
        at_bound = self._verdict(_cells(np.array([1.0] * 32 + [-1.0] * 8)))               # exactly 0.80
        self.assertEqual(at_bound["verdict"], "pass")
        for gains in (np.full(40, 0.2), np.full(40, 1.8)):
            out = self._verdict(_cells(gains))
            self.assertEqual(out["verdict"], "fail")
            self.assertFalse(out["checks"]["gain_median"]["ok"])
            self.assertTrue(out["checks"]["sign_agreement"]["ok"])

    def test_jacobian_and_retention_fail_independently(self):
        good = _cells(np.linspace(0.4, 1.3, 40))
        self.assertEqual(self._verdict(good, jac=(0.15, 0.15, 0.01))["verdict"], "fail")  # median 0.15 is not < 0.15
        self.assertEqual(self._verdict(good, jac=(0.149, 0.149, 0.9))["verdict"], "pass")
        self.assertEqual(self._verdict(good, hold=0.8, neutral=0.925)["verdict"], "fail")
        # exactly -0.10 passes, including the pairs whose float difference lands just below it
        for k in range(4, 41):
            out = self._verdict(good, hold=(k - 4) / 40, neutral=k / 40)
            self.assertEqual(out["verdict"], "pass", k)
            if k >= 5:
                self.assertEqual(self._verdict(good, hold=(k - 5) / 40, neutral=k / 40)["verdict"], "fail", k)
        # they fail the gate even when coverage alone would have made it inconclusive
        thin = _cells(np.linspace(0.4, 1.3, 40), eligible=np.arange(40) < 10)
        self.assertEqual(self._verdict(thin)["verdict"], "inconclusive")
        self.assertEqual(self._verdict(thin, jac=(0.5,))["verdict"], "fail")
        self.assertEqual(self._verdict(thin, hold=0.2, neutral=0.9)["verdict"], "fail")

    def test_inconclusive_paths(self):
        gains = np.linspace(0.4, 1.3, 40)
        cells = _cells(gains)
        cells["cell07"] = _cell(gains, eligible=np.arange(40) < 19)                       # one cell at 19 of 40
        out = self._verdict(cells)
        self.assertEqual(out["verdict"], "inconclusive")
        self.assertEqual(out["checks"]["coverage"]["value"]["under_minimum"], ["cell07"])
        self.assertTrue(out["checks"]["sign_agreement"]["ok"])                            # reported, not decisive
        cells["cell07"] = _cell(gains, eligible=np.arange(40) < 20)                       # exactly 20 is enough
        self.assertEqual(self._verdict(cells)["verdict"], "pass")
        self.assertEqual(self._verdict(_cells(gains, n=29))["verdict"], "inconclusive")   # a missing cell
        none = self._verdict(_cells(gains, eligible=np.zeros(40, dtype=bool)))            # no eligible pair at all
        self.assertEqual(none["verdict"], "inconclusive")
        self.assertIsNone(none["checks"]["sign_agreement"]["value"])
        self.assertEqual(none["eligible_pairs"], 0)
        # even terrible tracking on too few pairs is inconclusive, never a pass and not a tracking failure
        self.assertEqual(self._verdict(_cells(-gains, eligible=np.arange(40) < 5))["verdict"], "inconclusive")
        self.assertEqual(self._verdict(_cells(gains), jac=())["verdict"], "inconclusive")
        self.assertEqual(self._verdict(_cells(gains), jac=(0.1, float("nan")))["verdict"], "inconclusive")
        self.assertEqual(self._verdict(_cells(gains), hold=None)["verdict"], "inconclusive")
        self.assertEqual(self._verdict({})["verdict"], "inconclusive")

    def test_thresholds_come_from_the_argument(self):
        gains = np.full(40, 0.2)
        loose = dict(self.TH, gain_median_min=0.1)
        self.assertEqual(self._verdict(_cells(gains), th=loose)["verdict"], "pass")
        self.assertEqual(self._verdict(_cells(gains), th=loose)["thresholds"]["gain_median_min"], 0.1)
        with self.assertRaises(ValueError):
            cc.gate_verdict(_cells(gains), (0.1,), 0.9, 0.9, {"sign_agreement_min": 0.8})
        self.assertEqual(cc.GATE_THRESHOLDS, {
            "expected_cells": 30, "min_eligible_per_cell": 20, "sign_agreement_min": 0.80, "gain_median_min": 0.3,
            "gain_median_max": 1.5, "jacobian_rel_error_median_max": 0.15, "retention_diff_min": -0.10})


if __name__ == "__main__":
    unittest.main()
