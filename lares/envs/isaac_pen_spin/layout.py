"""The pen-spin observation layout and stage ids, importable with nothing else (``obs_schema`` and ``cases`` share it)."""

#: (name, size, description). Velocities in the observation are multiplied by vel_obs_scale = 0.2.
OBS_LAYOUT = (
    ("hand_dof_pos", 24, "hand joint positions, unscaled to [-1, 1] by joint limits"),
    ("hand_dof_vel", 24, "hand joint velocities x 0.2"),
    ("obj_pos", 3, "pen centre position in the env frame (m)"),
    ("obj_rot", 4, "pen orientation quaternion (w, x, y, z); the long axis is local x"),
    ("obj_linvel", 3, "pen linear velocity (m/s)"),
    ("obj_angvel", 3, "pen angular velocity x 0.2 (world frame)"),
    ("in_hand_pos", 3, "nominal in-hand pen position (success centre)"),
    ("goal_rot", 4, "goal quaternion, fixed to identity in the spin task"),
    ("rel_rot", 4, "obj_rot * conj(goal_rot)"),
    ("fingertip_pos", 15, "five fingertip positions (ff, mf, rf, lf, th)"),
    ("fingertip_rot", 20, "five fingertip quaternions"),
    ("fingertip_vel", 30, "five fingertip linear+angular velocities"),
    ("last_action", 20, "previous action"),
)
VEL_OBS_SCALE = 0.2

#: Gym ids of the three stages, the env ids the observation schema is registered under.
STAGE_TASK_IDS = {
    1: "Isaac-ShadowHandSpin-Stage1-Direct-v0",
    2: "Isaac-ShadowHandSpin-Stage2-Direct-v0",
    3: "Isaac-ShadowHandSpin-Stage3-Direct-v0",
}

#: R-046: the gait-scaffold continuation's schema id. Its observation is the 157-d layout plus the frozen R-039 gait's
#: action for this step (20, normalised [-1, 1], action order), appended by the child before the policy is called.
GAIT_SCAFFOLD_TASK_ID = "Isaac-ShadowHandSpin-Stage3-GaitScaffold-v0"
GAIT_FIELD = ("gait_action", 20, "the frozen R-039 gait's action this step (normalised [-1, 1], action order)")

#: R-084: the parent-scaffold schema id. The observation is the 157-d layout plus the frozen D-058 incumbent's action
#: for this step (20, normalised [-1, 1], action order), appended by the child before the residual policy is called.
PARENT_SCAFFOLD_TASK_ID = "Isaac-ShadowHandSpin-Stage3-ParentScaffold-v0"
PARENT_FIELD = ("parent_action", 20, "the frozen parent controller's action this step (normalised [-1, 1], action order)")
