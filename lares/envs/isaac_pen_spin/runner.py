"""Batched, manifest-driven rollouts of the pen-spin task (R-038). Needs a launched Isaac app.

One Isaac env holds one environment per case. After ``env.reset()`` each environment's state is
overwritten with its case's placement (``cases.split_placement``) using the same formulas as the two
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


def make_env(stage: int, num_envs: int, device: str = "cuda:0", enhanced_determinism: bool = False):
    import importlib

    cfg_mod = importlib.import_module("lares.envs.isaac_pen_spin.shadow_hand_spin_env_cfg")
    cfg = getattr(cfg_mod, cases.STAGES[stage].cfg_class)()
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


class PenSpinBatchRunner:
    """Run a controller on every case of a pen-spin manifest in one batched Isaac env."""

    def __init__(self, stage: int, num_envs: int, device: str = "cuda:0", enhanced_determinism: bool = False):
        self.stage = int(stage)
        self.enhanced_determinism = bool(enhanced_determinism)
        self.env = make_env(self.stage, num_envs, device, self.enhanced_determinism)
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
        vecs = cases.resolve_placements(manifest)
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


def reduce_episodes(trace: dict) -> list[dict]:
    """Per-episode rows: both reducers plus the geometry a reader needs to check them."""
    rows = []
    for i, cid in enumerate(trace["case_ids"]):
        L = int(trace["length"][i])
        p = trace["pred"][:L, i]
        spin = np.abs(trace["spin_rate"][:L, i])
        rows.append({
            "case_id": cid,
            "task_hash": trace["task_hashes"][i],
            "length": L,
            "completed": bool(trace["completed"][i]),
            "any_hit": cases.any_hit(p, L),
            "sustained": cases.sustained(p, L, bool(trace["completed"][i])),
            "hit_steps": int(p.sum()),
            "first_hit_step": int(np.argmax(p)) if p.any() else None,
            "tail_hit_fraction": float(p[-cases.SUSTAIN_WINDOW:].mean()) if L else 0.0,
            "mean_abs_spin": float(spin.mean()) if L else None,
            "max_abs_spin": float(spin.max()) if L else None,
            "mean_transverse": float(np.mean(trace["transverse_rate"][:L, i])) if L else None,
            "final_center_dist": float(trace["center_dist"][L - 1, i]) if L else None,
            "return": float(np.nansum(trace["reward"][:L, i])),
        })
    return rows
