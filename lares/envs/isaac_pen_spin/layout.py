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

#: R-094: the contact-frame schema id. The observation is the 157-d layout plus trusted per-fingertip contact-frame
#: features (25), computed from that same observation by ``contact_control.contact_features`` and appended by the child
#: before the generated policy is called. Both arms of the study read the same 182-d observation.
CONTACT_FRAME_TASK_ID = "Isaac-ShadowHandSpin-Stage3-ContactFrame-v0"
CONTACT_FIELDS = (
    ("contact_distance", 5,
     "distance (m) from each fingertip point (ff, mf, rf, lf, th) to the nearest point of the pen's surface, negative "
     "inside the pen; a small distance is proximity only, it does not establish contact, support or friction"),
    ("contact_axial_coord", 5,
     "each fingertip point's coordinate (m) along the pen's long axis (its local +x) measured from the pen centre; "
     "the pen's straight section spans -0.07 to +0.07, beyond it the nearest surface is an end cap"),
    ("contact_rel_vel", 15,
     "per fingertip (ff, mf, rf, lf, th) the three components [normal, circumferential, axial] (m/s, unscaled) of the "
     "fingertip point's velocity relative to the pen-surface point nearest to it, in that fingertip's local frame: "
     "normal points from the pen surface outward toward the fingertip, axial is the pen's local +x, and "
     "circumferential = axial x normal, so a positive circumferential value is the fingertip moving around the pen "
     "counter-clockwise seen from the pen's +x end; with the fingertip still, a pen rolling at +w rad/s about its "
     "+x axis gives a circumferential value of -w * 0.012"),
)
CONTACT_FEATURE_DIM = sum(size for _, size, _ in CONTACT_FIELDS)
