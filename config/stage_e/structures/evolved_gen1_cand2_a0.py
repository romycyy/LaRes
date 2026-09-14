class GeneratedPolicy(SymbolicPolicy):
    """
    Standoff Vector Field Push with Slip-Aware Lateral Correction (gen1_idea2)
    - Same four phases as parent
    - Lateral correction in push phase is attenuated when object doesn't move forward (slip)
    """
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        # Gains
        self.k_approach = nn.Parameter(torch.tensor(4.0))     # (0.5, 8.0)
        self.k_push = nn.Parameter(torch.tensor(3.5))         # (0.5, 8.0)
        self.k_lateral = nn.Parameter(torch.tensor(2.5))      # (0.1, 8.0)
        self.k_finish = nn.Parameter(torch.tensor(1.2))       # (0.1, 5.0)
        # Geometry / thresholds
        self.standoff_dist = nn.Parameter(torch.tensor(0.06))     # (0.02, 0.12)
        self.contact_radius = nn.Parameter(torch.tensor(0.03))    # (0.01, 0.06)
        self.approach_tol = nn.Parameter(torch.tensor(0.035))     # (0.01, 0.08)
        self.goal_tol = nn.Parameter(torch.tensor(0.02))          # (0.005, 0.06)
        self.z_offset = nn.Parameter(torch.tensor(0.005))         # (0.0, 0.03)
        # Gripper command (kept closing for pushing)
        self.grip_cmd = nn.Parameter(torch.tensor(-0.8))          # (-1.0, 0.0)

    def forward(self, obs):
        B = obs.shape[0]
        eps = 1e-8

        # Read required observation fields by NAME
        tcp = self.obs_field(obs, "tcp")           # (B,3)
        obj = self.obs_field(obs, "obj")           # (B,3)
        goal = self.obs_field(obs, "goal")         # (B,3)
        prev_tcp = self.obs_field(obs, "prev_tcp") # (B,3)
        prev_obj = self.obs_field(obs, "prev_obj") # (B,3)

        # 2D planar components for pushing
        tcp_xy = tcp[:, 0:2]     # (B,2)
        obj_xy = obj[:, 0:2]     # (B,2)
        goal_xy = goal[:, 0:2]   # (B,2)

        # Goal direction from object to target (2D, normalized)
        v_goal = goal_xy - obj_xy
        v_goal_norm = torch.norm(v_goal, dim=-1, keepdim=True) + eps
        dir_goal = v_goal / v_goal_norm  # (B,2)

        # Standoff point behind the object along push direction (2D)
        standoff_xy = obj_xy - self.standoff_dist * dir_goal  # (B,2)

        # Distances for gating (2D planar distances)
        diff_standoff = standoff_xy - tcp_xy
        dist_standoff = torch.norm(diff_standoff, dim=-1, keepdim=True)  # (B,1)

        diff_tcp_obj = obj_xy - tcp_xy
        dist_tcp_obj = torch.norm(diff_tcp_obj, dim=-1, keepdim=True)    # (B,1)

        dist_obj_goal = torch.norm(v_goal, dim=-1, keepdim=True)         # (B,1)

        # Smooth phase gates (sigmoid). Higher beta -> sharper transitions; fixed constants (not learnable).
        beta = 80.0
        beta_goal = 80.0
        beta_contact = 80.0

        # Finish when object is within goal_tol of the goal
        g_finish = torch.sigmoid(beta_goal * (self.goal_tol - dist_obj_goal))  # near-goal -> ~1
        # Contact when tcp is within contact_radius of the object
        g_contact = torch.sigmoid(beta_contact * (self.contact_radius - dist_tcp_obj))  # in-contact -> ~1
        # Far from standoff when tcp is beyond approach_tol to standoff point
        g_far = torch.sigmoid(beta * (dist_standoff - self.approach_tol))  # far -> ~1

        # Phase weights (ensure they sum softly; all in [0,1])
        w_approach = (1.0 - g_finish) * (1.0 - g_contact) * g_far
        w_acquire = (1.0 - g_finish) * (1.0 - g_contact) * (1.0 - g_far)
        w_push = (1.0 - g_finish) * g_contact
        w_finish = g_finish

        # Record and use gates (enables external diagnostics/forcing)
        w_approach = self.record_gate("approach_standoff", w_approach)
        w_acquire = self.record_gate("contact_acquire", w_acquire)
        w_push = self.record_gate("push", w_push)
        w_finish = self.record_gate("finish", w_finish)

        # Approach: move toward standoff point (normalized direction)
        dir_to_standoff = diff_standoff / (dist_standoff + eps)
        move_approach_xy = self.k_approach * dir_to_standoff

        # Contact acquire: move gently toward pre-contact point along push line
        pre_contact_xy = obj_xy - 0.5 * self.contact_radius * dir_goal
        diff_acquire = pre_contact_xy - tcp_xy
        dist_acquire = torch.norm(diff_acquire, dim=-1, keepdim=True)
        dir_acquire = diff_acquire / (dist_acquire + eps)
        move_acquire_xy = self.k_approach * dir_acquire

        # Push: forward along goal direction + slip-aware lateral error correction
        forward_xy = self.k_push * dir_goal

        # Lateral error: perpendicular distance from tcp to the push line through obj along dir_goal
        delta_tcp_from_line = tcp_xy - obj_xy
        proj_along = torch.sum(delta_tcp_from_line * dir_goal, dim=-1, keepdim=True)  # scalar along dir_goal
        parallel_component = proj_along * dir_goal
        perp = delta_tcp_from_line - parallel_component
        perp_norm = torch.norm(perp, dim=-1, keepdim=True)
        perp_dir = perp / (perp_norm + eps)
        # Magnitude scaled and saturated by tanh to avoid instability; uses contact_radius for scale
        side_mag = torch.tanh(perp_norm / (self.contact_radius + eps))

        # Slip-aware attenuation: compare object's forward motion vs tcp forward motion
        prev_tcp_xy = prev_tcp[:, 0:2]
        prev_obj_xy = prev_obj[:, 0:2]
        v_tcp_xy = tcp_xy - prev_tcp_xy
        v_obj_xy = obj_xy - prev_obj_xy
        v_tcp_along = torch.sum(v_tcp_xy * dir_goal, dim=-1, keepdim=True)
        v_obj_along = torch.sum(v_obj_xy * dir_goal, dim=-1, keepdim=True)
        vel_eps = 1e-6
        eff_raw = torch.clamp_min(v_obj_along, 0.0) / (torch.abs(v_tcp_along) + vel_eps)
        eff = torch.clamp(eff_raw, 0.0, 1.0)  # 0=no effectiveness (attenuate), 1=good transfer

        side_correction_xy = - self.k_lateral * eff * side_mag * perp_dir
        move_push_xy = forward_xy + side_correction_xy

        # Finish: gentle nudge along goal direction
        move_finish_xy = self.k_finish * dir_goal

        # Blend phases smoothly
        move_xy = (
            w_approach * move_approach_xy
            + w_acquire * move_acquire_xy
            + w_push * move_push_xy
            + w_finish * move_finish_xy
        )

        # Z regulation: maintain end-effector at obj.z + z_offset (planar push posture)
        target_z = obj[:, 2:3] + self.z_offset
        tcp_z = tcp[:, 2:3]
        k_z = 6.0  # fixed vertical gain (not learnable)
        move_z = k_z * (target_z - tcp_z)  # (B,1)

        # Gripper: keep closing (constant parameter, expanded over batch)
        grip = self.grip_cmd.unsqueeze(0).expand(B, 1)  # (B,1)

        # Assemble final action mean: (dx, dy, dz, gripper)
        move_3d = torch.cat([move_xy, move_z], dim=1)  # (B,3)
        mean = torch.cat([move_3d, grip], dim=1)       # (B,4)

        # Fixed, non-trainable action scale from the framework
        std = self.fixed_std(mean)
        return (mean, std)

    def get_param_ranges(self):
        return {
            "k_approach": (0.5, 8.0),
            "k_push": (0.5, 8.0),
            "k_lateral": (0.1, 8.0),
            "k_finish": (0.1, 5.0),
            "standoff_dist": (0.02, 0.12),
            "contact_radius": (0.01, 0.06),
            "approach_tol": (0.01, 0.08),
            "goal_tol": (0.005, 0.06),
            "z_offset": (0.0, 0.03),
            "grip_cmd": (-1.0, 0.0),
        }