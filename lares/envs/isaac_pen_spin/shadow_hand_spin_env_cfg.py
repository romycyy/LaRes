# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ShadowHandSpin staged pen task."""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg import ShadowHandEnvCfg


_PEN_MATERIAL = RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0, restitution=0.0)


@configclass
class ShadowHandSpinStage1EnvCfg(ShadowHandEnvCfg):
    """Stage 1: keep the pen in hand with a gentle spin target."""

    episode_length_s = 6.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=0.75, replicate_physics=True)

    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.CapsuleCfg(
            radius=0.012,
            height=0.14,
            axis="X",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=1,
                sleep_threshold=0.001,
                stabilization_threshold=0.001,
                max_depenetration_velocity=1000.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.035),
            physics_material=_PEN_MATERIAL,
            semantic_tags=[("class", "pen")],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.39, 0.6), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/pen_goal_marker",
        markers={
            "goal": sim_utils.CapsuleCfg(
                radius=0.012,
                height=0.14,
                axis="X",
                physics_material=RigidBodyMaterialCfg(static_friction=0.0, dynamic_friction=0.0),
            )
        },
    )

    # spin-specific reward parameters
    target_axis = (1.0, 0.0, 0.0)
    target_spin_rate = 2.0
    success_spin_rate = 1.0
    success_center_tolerance = 0.10
    spin_reward_scale = 0.15
    center_reward_scale = 3.0
    center_reward_sharpness = 18.0
    axis_reward_scale = 0.75
    axis_reward_sharpness = 4.0
    action_penalty_scale = -0.0002
    fall_penalty = -5.0
    fall_dist = 0.30


@configclass
class ShadowHandSpinStage2EnvCfg(ShadowHandSpinStage1EnvCfg):
    """Stage 2: reward stronger sustained spin."""

    target_spin_rate = 5.0
    success_spin_rate = 3.0
    spin_reward_scale = 0.5
    center_reward_scale = 2.5


@configclass
class ShadowHandSpinStage3EnvCfg(ShadowHandSpinStage2EnvCfg):
    """Stage 3: spin while preserving in-hand stability and axis alignment."""

    target_spin_rate = 7.0
    success_spin_rate = 4.0
    spin_reward_scale = 0.75
    center_reward_scale = 3.0
    axis_reward_scale = 1.25
    success_center_tolerance = 0.08
