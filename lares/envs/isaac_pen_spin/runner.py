"""Batched, manifest-driven rollouts of the pen-spin task (R-038). Needs a launched Isaac app.

One Isaac env holds one environment per case. After ``env.reset()`` each environment's state is
overwritten with its case's placement using the same formulas as the two
``_reset_idx`` methods, so an episode is a function of its case rather than of Isaac's global RNG.
Environments that end are auto-reset by ``DirectRLEnv``; the runner stops recording them at their
first done.

Per step it records the env's own ``center_dist`` and ``spin_rate`` (computed in ``_get_dones``
before any auto-reset), the transverse (end-over-end) angular speed, the per-step success
predicate, and done flags.

**One run per process.** Measured 2026-10-27 (``scripts/isaac/debug_pen_spin_runner.py``): the first
``run`` in a fresh process is bit-identical across processes, but a second ``run`` in the same process is
not reproducible even though its observation at t=0 is identical. State the solver keeps from the
earlier episode (and its RNG-drawn auto-resets) changes the pen's landing contact around step 8, and the
trajectories diverge from there. PhysX ``enable_enhanced_determinism`` does not remove it. So a runner
refuses a second ``run`` unless ``allow_reuse`` is set (debugging only), and protocols start one child
process per (stage, controller).
"""

from __future__ import annotations

import numpy as np
import torch

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin.shadow_hand_spin_env import ShadowHandSpinEnv


class MeasuredShadowHandSpinEnv(ShadowHandSpinEnv):
    """The vendored env plus a stored transverse angular speed; reward and dones are unchanged."""

    def _compute_spin_terms(self):
        super()._compute_spin_terms()
        local_axis = self.x_unit_tensor
        from isaaclab.utils.math import quat_apply

        axis = torch.nn.functional.normalize(quat_apply(self.object_rot, local_axis), dim=-1)
        perp = self.object_angvel - self.spin_rate.unsqueeze(-1) * axis
        self.transverse_rate = torch.norm(perp, dim=-1)


apply_condition = cases.apply_condition   # numpy-only, so tests can exercise it without Isaac


def physical_record(env) -> dict:
    """What the simulator actually holds for environment 0's pen: mass, inertia, material, capsule size, horizon."""
    out = {"max_episode_length": int(env.max_episode_length), "episode_length_s": float(env.cfg.episode_length_s),
           "control_dt": float(env.cfg.sim.dt * env.cfg.decimation), "errors": []}
    view = env.object.root_physx_view
    try:
        out["mass"] = float(view.get_masses().reshape(env.num_envs, -1)[0, 0])
        out["inertia"] = [float(x) for x in view.get_inertias().reshape(env.num_envs, -1)[0]]
        mat = view.get_material_properties().reshape(env.num_envs, -1, 3)[0]
        out["static_friction"], out["dynamic_friction"] = float(mat[0, 0]), float(mat[0, 1])
        out["collision_shapes"] = int(mat.shape[0])
    except Exception as exc:  # noqa: BLE001 - recorded, the preflight gate fails on it
        out["errors"].append(f"physx view: {exc!r}")
    try:
        import omni.usd
        from pxr import Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        caps = [p for p in Usd.PrimRange(stage.GetPrimAtPath("/World/envs/env_0/object")) if p.IsA(UsdGeom.Capsule)]
        out["capsules"] = len(caps)
        out["radius"], out["length"] = float(caps[0].GetAttribute("radius").Get()), float(caps[0].GetAttribute("height").Get())
        out["capsule_axis"] = str(caps[0].GetAttribute("axis").Get())
    except Exception as exc:  # noqa: BLE001
        out["errors"].append(f"usd capsule: {exc!r}")
    return out


def make_env(stage: int, num_envs: int, device: str = "cuda:0", enhanced_determinism: bool = False,
             condition: dict | None = None):
    import importlib

    cfg_mod = importlib.import_module("lares.envs.isaac_pen_spin.shadow_hand_spin_env_cfg")
    cfg = getattr(cfg_mod, cases.STAGES[stage].cfg_class)()
    if condition:
        apply_condition(cfg, condition)
    cfg.scene.num_envs = int(num_envs)
    cfg.sim.device = device
    cfg.sim.physx.enable_enhanced_determinism = bool(enhanced_determinism)
    return MeasuredShadowHandSpinEnv(cfg=cfg)


def _write_placements(env, vecs: np.ndarray) -> None:
    """Overwrite every environment's reset state with its case's placement draws."""
    from isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env import randomize_rotation

    dev = env.device
    ids = torch.arange(env.num_envs, device=dev)
    v = torch.as_tensor(vecs, dtype=torch.float32, device=dev)
    pos_n, rot_n, dof_n = v[:, 0:3], v[:, 3:5], v[:, 5:5 + cases.NUM_HAND_DOFS]

    state = env.object.data.default_root_state.clone()[ids]
    state[:, 0:3] = state[:, 0:3] + env.scene.env_origins[ids] + env.cfg.reset_position_noise * pos_n
    state[:, 3:7] = randomize_rotation(0.25 * rot_n[:, 0], 0.25 * rot_n[:, 1], env.x_unit_tensor[ids], env.y_unit_tensor[ids])
    state[:, 7:] = 0.0
    env.object.write_root_pose_to_sim(state[:, :7], ids)
    env.object.write_root_velocity_to_sim(state[:, 7:], ids)

    default = env.hand.data.default_joint_pos[ids]
    delta_max = env.hand_dof_upper_limits[ids] - default
    delta_min = env.hand_dof_lower_limits[ids] - default
    dof_pos = default + env.cfg.reset_dof_pos_noise * (delta_min + (delta_max - delta_min) * 0.5 * dof_n)
    dof_vel = env.hand.data.default_joint_vel[ids].clone()
    env.prev_targets[ids] = dof_pos
    env.cur_targets[ids] = dof_pos
    env.hand_dof_targets[ids] = dof_pos
    env.hand.set_joint_position_target(dof_pos, env_ids=ids)
    env.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=ids)
    env.successes[ids] = 0
    env.episode_length_buf[ids] = 0
    # The previous run's last action is part of the observation (``last_action``); a fresh episode has none.
    env.actions = torch.zeros((env.num_envs, cases.NUM_ACTIONS), device=dev)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(dt=0.0)
    env._compute_intermediate_values()


class ControllerContext:
    """What a controller may know besides the observation: limits, neutral action and timing."""

    def __init__(self, env):
        idx = env.actuated_dof_indices
        lo = env.hand_dof_lower_limits[:, idx]
        hi = env.hand_dof_upper_limits[:, idx]
        default = env.hand.data.default_joint_pos[:, idx]
        self.num_envs = env.num_envs
        self.device = env.device
        self.control_dt = float(env.cfg.sim.dt * env.cfg.decimation)
        self.actuated_joint_names = list(env.cfg.actuated_joint_names)
        self.lower = lo
        self.upper = hi
        #: the action whose joint target is the default (zero) joint position
        self.neutral_action = ((2.0 * default - hi - lo) / (hi - lo)).clamp(-1.0, 1.0)
        # R-093: what a *trusted* controller (never generated code) may read from the simulator besides limits.
        self._env = env
        self.joint_names = list(env.hand.joint_names)                  # all 24, simulator order
        self.actuated_dof_indices = [int(i) for i in idx]              # sorted: action i drives joint_names[idx[i]]
        self.lower_all = env.hand_dof_lower_limits                     # (num_envs, 24)
        self.upper_all = env.hand_dof_upper_limits

    def joint_pos(self):
        """Measured positions of all 24 hand joints, radians, simulator order: ``(num_envs, 24)``."""
        return self._env.hand.data.joint_pos

    def fingertip_jacobians(self):
        """Linear Jacobians of the five fingertip link *origins* (ff, mf, rf, lf, th) w.r.t. all 24 hand joints, world frame:
        ``(num_envs, 5, 3, 24)``. The hand is fixed-base, so PhysX omits the root link and body ``b`` is row ``b - 1``
        (checked on the box 2026-11-20: 26 bodies, Jacobian shape ``(N, 25, 6, 24)``)."""
        hand = self._env.hand
        if not hand.is_fixed_base:
            raise RuntimeError("fingertip_jacobians assumes the fixed-base hand")
        rows = [int(b) - 1 for b in self._env.finger_bodies]
        J = hand.root_physx_view.get_jacobians()[:, rows]
        # PhysX differentiates each link's centre of mass; the observed fingertip position is the link origin
        # (found in the R-093 calibration). v_origin = v_com - w x r, r = com - origin in the world frame.
        bodies = list(self._env.finger_bodies)
        r = (hand.data.body_com_pos_w[:, bodies] - hand.data.body_link_pos_w[:, bodies]).unsqueeze(-1)
        return J[:, :, :3, :] - torch.linalg.cross(J[:, :, 3:, :], r.expand_as(J[:, :, 3:, :]), dim=2)


class PenSpinBatchRunner:
    """Run a controller on every case of a pen-spin manifest in one batched Isaac env."""

    def __init__(self, stage: int, num_envs: int, device: str = "cuda:0", enhanced_determinism: bool = False,
                 condition: dict | None = None):
        self.stage = int(stage)
        self.enhanced_determinism = bool(enhanced_determinism)
        self.condition = dict(condition or {})
        self.env = make_env(self.stage, num_envs, device, self.enhanced_determinism, self.condition)
        self.num_envs = int(num_envs)
        self.runs = 0

    def close(self):
        self.env.close()

    def run(self, manifest, controller, *, record_obs: bool = False, allow_reuse: bool = False) -> dict:
        if self.runs and not allow_reuse:
            raise RuntimeError("a second run in one process is not reproducible; start a fresh process per run")
        self.runs += 1
        if cases.stage_of(manifest) != self.stage:
            raise ValueError(f"{manifest.manifest_id} is not a stage-{self.stage} manifest")
        vecs = cases.resolve_placements(manifest) * float(self.condition.get("reset_scale", 1.0))
        n = len(vecs)
        if n != self.num_envs:
            raise ValueError(f"manifest has {n} cases, runner holds {self.num_envs} envs")
        env = self.env
        spec = cases.STAGES[self.stage]

        env.reset()
        _write_placements(env, vecs)
        obs = env._get_observations()["policy"]
        ctx = ControllerContext(env)
        if hasattr(controller, "reset"):
            controller.reset(ctx)

        T = int(env.max_episode_length)
        alive = torch.ones(n, dtype=torch.bool, device=env.device)
        length = torch.zeros(n, dtype=torch.long, device=env.device)
        completed = torch.zeros(n, dtype=torch.bool, device=env.device)
        rec = {k: torch.full((T, n), float("nan"), device=env.device)
               for k in ("center_dist", "spin_rate", "transverse_rate", "reward")}
        pred = torch.zeros((T, n), dtype=torch.bool, device=env.device)
        obs_trace = [] if record_obs else None
        act_trace = [] if record_obs else None

        for t in range(T):
            action = controller(obs, t, ctx).clamp(-1.0, 1.0)
            if record_obs:
                obs_trace.append(obs.detach().cpu().numpy())
                act_trace.append(action.detach().cpu().numpy())
            obs_d, reward, terminated, truncated, _ = env.step(action)
            obs = obs_d["policy"]
            live = alive.clone()
            rec["center_dist"][t] = torch.where(live, env.center_dist, rec["center_dist"][t])
            rec["spin_rate"][t] = torch.where(live, env.spin_rate, rec["spin_rate"][t])
            rec["transverse_rate"][t] = torch.where(live, env.transverse_rate, rec["transverse_rate"][t])
            rec["reward"][t] = torch.where(live, reward, rec["reward"][t])
            step_ok = (env.center_dist < spec.success_center_tolerance) & (env.spin_rate.abs() > spec.success_spin_rate)
            pred[t] = live & step_ok
            length = torch.where(live, length + 1, length)
            done = terminated | truncated
            completed = torch.where(live & done, truncated & ~terminated, completed)
            alive = live & ~done
            if not bool(alive.any()):
                break

        out = {
            "stage": self.stage,
            "manifest_id": manifest.manifest_id,
            "case_ids": [c.case_id for c in manifest.episodes],
            "task_hashes": [c.task_hash for c in manifest.episodes],
            "max_episode_length": T,
            "length": length.cpu().numpy(),
            "completed": completed.cpu().numpy(),
            "pred": pred.cpu().numpy(),
            **{k: v.cpu().numpy() for k, v in rec.items()},
        }
        if record_obs:
            # (cases, T, dim): case-major, unlike the (T, cases) per-step arrays above.
            out["obs"] = np.stack(obs_trace, axis=1)
            out["actions"] = np.stack(act_trace, axis=1)
        return out


# Moved to ``cases`` (numpy only) so the pipeline backend can reduce rows without Isaac; re-exported here.
reduce_episodes = cases.reduce_episodes
