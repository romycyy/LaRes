# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Shadow Hand pen-spinning task.

This environment reuses the standard in-hand manipulation scene and action /
observation pipeline, but changes the objective from target-pose reorientation
to sustained spinning of a capsule-shaped pen while keeping it in hand.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils.math import quat_apply
from isaaclab_tasks.direct.inhand_manipulation.inhand_manipulation_env import (
    InHandManipulationEnv,
    randomize_rotation,
)


class ShadowHandSpinEnv(InHandManipulationEnv):
    """Direct RL task for spinning a pen-like object in the Shadow Hand."""

    def _reset_target_pose(self, env_ids: Sequence[int]):
        # The spin task does not sample arbitrary orientation goals. Keep a
        # stable marker at the nominal in-hand location so inherited rendering
        # and reset paths remain valid.
        self.goal_rot[env_ids] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=self.device)
        goal_pos = self.goal_pos + self.scene.env_origins
        self.goal_markers.visualize(goal_pos, self.goal_rot)
        self.reset_goal_buf[env_ids] = 0

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES

        # Start pens close to horizontal. This mirrors the IsaacGymEnvs pen
        # reset idea while keeping the implementation in Isaac Lab primitives.
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        object_state = self.object.data.default_root_state.clone()[env_ids_t]
        object_state[:, 0:3] = object_state[:, 0:3] + self.scene.env_origins[env_ids_t]
        object_state[:, 0:3] += self.cfg.reset_position_noise * torch.empty(
            (len(env_ids_t), 3), device=self.device
        ).uniform_(-1.0, 1.0)
        rand = torch.empty((len(env_ids_t), 2), device=self.device).uniform_(-1.0, 1.0)
        object_state[:, 3:7] = randomize_rotation(
            0.25 * rand[:, 0],
            0.25 * rand[:, 1],
            self.x_unit_tensor[env_ids_t],
            self.y_unit_tensor[env_ids_t],
        )
        object_state[:, 7:] = 0.0
        self.object.write_root_pose_to_sim(object_state[:, :7], env_ids_t)
        self.object.write_root_velocity_to_sim(object_state[:, 7:], env_ids_t)
        self._compute_intermediate_values()

    def _get_rewards(self) -> torch.Tensor:
        self._compute_spin_terms()

        spin_abs = torch.abs(self.spin_rate)
        spin_reward = torch.clamp(spin_abs / max(self.cfg.target_spin_rate, 1e-6), 0.0, 2.0)
        center_reward = torch.exp(-self.cfg.center_reward_sharpness * self.center_dist)
        axis_reward = torch.exp(-self.cfg.axis_reward_sharpness * self.axis_error)
        action_penalty = torch.sum(self.actions**2, dim=-1)

        reward = (
            self.cfg.spin_reward_scale * spin_reward
            + self.cfg.center_reward_scale * center_reward
            + self.cfg.axis_reward_scale * axis_reward
            + self.cfg.action_penalty_scale * action_penalty
        )
        reward = torch.where(self.center_dist >= self.cfg.fall_dist, reward + self.cfg.fall_penalty, reward)

        success = (self.center_dist < self.cfg.success_center_tolerance) & (
            spin_abs > self.cfg.success_spin_rate
        )
        self.successes = self.successes + success.float()
        self.reset_goal_buf[:] = success

        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["spin_rate"] = spin_abs.mean()
        self.extras["log"]["center_dist"] = self.center_dist.mean()
        self.extras["log"]["axis_error"] = self.axis_error.mean()
        self.extras["log"]["success_rate"] = success.float().mean()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        self._compute_spin_terms()
        out_of_reach = self.center_dist >= self.cfg.fall_dist
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return out_of_reach, time_out

    def _compute_spin_terms(self):
        # Capsule local X is the pen long axis because the object config uses
        # axis="X". Quaternions are scalar-first in Isaac Lab.
        local_axis = self.x_unit_tensor
        pen_axis_w = quat_apply(self.object_rot, local_axis)
        pen_axis = torch.nn.functional.normalize(pen_axis_w, dim=-1)

        self.spin_rate = torch.sum(self.object_angvel * pen_axis, dim=-1)
        self.center_dist = torch.norm(self.object_pos - self.in_hand_pos, p=2, dim=-1)

        target_axis = self.cfg.target_axis
        target_axis_t = torch.tensor(target_axis, dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        target_axis_t = torch.nn.functional.normalize(target_axis_t, dim=-1)
        # Sign-invariant axis error; a pen pointing left/right is equivalent.
        alignment = torch.abs(torch.sum(pen_axis * target_axis_t, dim=-1)).clamp(0.0, 1.0)
        self.axis_error = 1.0 - alignment
