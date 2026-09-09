# Frozen structure for the Phase 3 fitting comparison. Do not edit by hand:
# regenerate with scripts/build_structure_bank.py.
# provenance: gen_2 candidate 4 of the 2026-08-28 search; reported score 82.56
# ported: raw observation slices rewritten to named accessors; outputs verified
# identical to the original on random observations at two scales.

class GeneratedPolicy(SymbolicPolicy):
    """
    Hypothesis 4:
    Hierarchical four-stage gating with history smoothing and anticipatory braking.
    """
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)

        # Geometry
        self.r_back = nn.Parameter(torch.tensor(0.07))
        self.z_clear = nn.Parameter(torch.tensor(0.10))
        self.h_push = nn.Parameter(torch.tensor(0.02))

        # Stage gains
        self.kp_app = nn.Parameter(torch.tensor(6.0))
        self.kd_app = nn.Parameter(torch.tensor(0.5))
        self.kp_align = nn.Parameter(torch.tensor(2.0))
        self.kd_align = nn.Parameter(torch.tensor(0.5))
        self.v_push = nn.Parameter(torch.tensor(2.0))
        self.k_lat = nn.Parameter(torch.tensor(3.0))
        self.kd_push = nn.Parameter(torch.tensor(0.6))
        self.v_brake = nn.Parameter(torch.tensor(0.8))
        self.k_hold = nn.Parameter(torch.tensor(3.0))
        self.v_side = nn.Parameter(torch.tensor(0.5))  # disengage sideways speed

        # Anticipatory braking
        self.a_dec = nn.Parameter(torch.tensor(2.0))   # deceleration estimate
        self.alpha_stop = nn.Parameter(torch.tensor(40.0))

        # Gates thresholds and sharpness
        self.r_app = nn.Parameter(torch.tensor(0.12))
        self.r_align = nn.Parameter(torch.tensor(0.06))
        self.r_contact = nn.Parameter(torch.tensor(0.06))
        self.r_goal = nn.Parameter(torch.tensor(0.05))
        self.alpha_app = nn.Parameter(torch.tensor(60.0))
        self.alpha_align = nn.Parameter(torch.tensor(80.0))
        self.alpha_contact = nn.Parameter(torch.tensor(80.0))
        self.alpha_goal = nn.Parameter(torch.tensor(80.0))

        # Smoothing coefficient for stage weights
        self.beta_smooth = nn.Parameter(torch.tensor(0.3))  # 0=no smoothing, 1=all prev

        # Vertical gains
        self.kp_z = nn.Parameter(torch.tensor(12.0))
        self.kd_z = nn.Parameter(torch.tensor(0.7))
        self.z_lift = nn.Parameter(torch.tensor(0.02))

        # Speed cap
        self.v_cap = nn.Parameter(torch.tensor(2.0))

        # Gripper and noise
        self.grip_bias = nn.Parameter(torch.tensor(-0.6))
        self.log_std_vec = nn.Parameter(torch.tensor([-1.8, -1.8, -2.2, -2.2]))
        self.std_push_brake = nn.Parameter(torch.tensor(0.6))  # shrink during push & brake

    def _compute_stage_weights(self, tcp, obj, goal):
        eps = 1e-8
        g_xy = (goal - obj)[:, 0:2]
        d_goal = torch.norm(g_xy, dim=-1, keepdim=True) + eps
        dir_xy = g_xy / d_goal

        anchor_xy = obj[:, 0:2] - self.r_back * dir_xy
        p_above = torch.cat([anchor_xy, obj[:, 2:3] + self.z_clear], dim=1)

        d_to_above = torch.norm(tcp - p_above, dim=-1, keepdim=True) + eps
        d_xy_anchor = torch.norm(tcp[:, 0:2] - anchor_xy, dim=-1, keepdim=True) + eps
        d_contact = torch.norm(tcp[:, 0:2] - obj[:, 0:2], dim=-1, keepdim=True) + eps

        g_app = 1.0 - torch.sigmoid(self.alpha_app * (self.r_app - d_to_above))   # high when far
        g_align = torch.sigmoid(self.alpha_align * (self.r_align - d_xy_anchor))  # high when xy aligned
        g_contact = torch.sigmoid(self.alpha_contact * (self.r_contact - d_contact))  # near object
        g_goal = torch.sigmoid(self.alpha_goal * (self.r_goal - d_goal))          # near goal

        # Anticipatory stop based on current object velocity (if available via prev)
        return g_app, g_align, g_contact, g_goal, dir_xy, anchor_xy, d_goal

    def forward(self, obs):
        B = obs.shape[0]
        eps = 1e-8

        # Current and previous states
        tcp = self.obs_field(obs, "tcp")
        obj = self.obs_field(obs, "obj")
        goal = self.obs_field(obs, "goal")
        prev_tcp = self.obs_field(obs, "prev_tcp")
        prev_obj = self.obs_field(obs, "prev_obj")

        v_tcp = tcp - prev_tcp
        v_tcp_xy = v_tcp[:, 0:2]
        v_tcp_z = v_tcp[:, 2:3]
        v_obj_xy = (obj - prev_obj)[:, 0:2]

        # Current stage features
        g_app, g_align, g_contact, g_goal, dir_xy, anchor_xy, d_goal = self._compute_stage_weights(tcp, obj, goal)

        # Previous-step stage features for smoothing
        g_app_p, g_align_p, g_contact_p, g_goal_p, dir_xy_p, anchor_xy_p, d_goal_p = self._compute_stage_weights(prev_tcp, prev_obj, goal)

        # Anticipatory stop: compare stop distance vs remaining
        v_obj_speed = torch.norm(v_obj_xy, dim=-1, keepdim=True)
        d_stop = (v_obj_speed * v_obj_speed) / (2.0 * (self.a_dec + 1e-6))
        g_stop = torch.sigmoid(self.alpha_stop * (d_stop - d_goal))

        # Raw stage activations
        w_app_raw = g_app * (1.0 - g_align) * (1.0 - g_goal)     # far from above or misaligned
        w_desc_raw = g_align * (1.0 - g_contact) * (1.0 - g_goal)
        w_push_raw = g_contact * (1.0 - g_goal) * (1.0 - g_stop)
        w_brake_raw = torch.clamp(g_goal + g_stop, 0.0, 1.0)

        # Prev raw activations
        w_app_prev = g_app_p * (1.0 - g_align_p) * (1.0 - g_goal_p)
        w_desc_prev = g_align_p * (1.0 - g_contact_p) * (1.0 - g_goal_p)
        w_push_prev = g_contact_p * (1.0 - g_goal_p)
        w_brake_prev = g_goal_p

        # Smooth with previous via convex combination
        b = torch.clamp(self.beta_smooth, 0.0, 0.95)
        w_app_s = (1.0 - b) * w_app_raw + b * w_app_prev
        w_desc_s = (1.0 - b) * w_desc_raw + b * w_desc_prev
        w_push_s = (1.0 - b) * w_push_raw + b * w_push_prev
        w_brake_s = (1.0 - b) * w_brake_raw + b * w_brake_prev

        # Normalize
        w_sum = w_app_s + w_desc_s + w_push_s + w_brake_s + eps
        w_app = w_app_s / w_sum
        w_desc = w_desc_s / w_sum
        w_push = w_push_s / w_sum
        w_brake = w_brake_s / w_sum

        # Stage commands
        # 1) Approach-Above: go to p_above
        p_above = torch.cat([anchor_xy, obj[:, 2:3] + self.z_clear], dim=1)
        u_app = -self.kp_app * (tcp - p_above) - self.kd_app * v_tcp
        u_app_xy = u_app[:, 0:2]
        u_app_z = u_app[:, 2:3]

        # 2) Descend-Align: align XY to anchor, descend to push height
        u_desc_xy = -self.kp_align * (tcp[:, 0:2] - anchor_xy) - self.kd_align * v_tcp_xy
        z_push = obj[:, 2:3] + self.h_push
        u_desc_z = -self.kp_z * (tcp[:, 2:3] - z_push) - self.kd_z * v_tcp_z

        # 3) Push: forward along dir, lateral funnel, damping
        err_xy = tcp[:, 0:2] - anchor_xy
        along = torch.sum(err_xy * dir_xy, dim=-1, keepdim=True) * dir_xy
        side = err_xy - along
        u_push_xy = self.v_push * dir_xy - self.k_lat * side - self.kd_push * v_tcp_xy
        u_push_z = -self.kp_z * (tcp[:, 2:3] - z_push) - self.kd_z * v_tcp_z

        # 4) Brake-Disengage: reduce forward, pull to goal, move sideways and lift
        n_xy = torch.stack([-dir_xy[:, 1], dir_xy[:, 0]], dim=1)
        u_brake_xy = -self.v_brake * dir_xy - self.k_hold * (obj[:, 0:2] - goal[:, 0:2]) + self.v_side * n_xy - self.kd_push * v_tcp_xy
        u_brake_z = -self.kp_z * (tcp[:, 2:3] - (z_push + self.z_lift)) - self.kd_z * v_tcp_z

        # Blend
        move_xy = w_app * u_app_xy + w_desc * u_desc_xy + w_push * u_push_xy + w_brake * u_brake_xy
        dz = w_app * u_app_z + w_desc * u_desc_z + w_push * u_push_z + w_brake * u_brake_z

        # Soft cap XY
        mag = torch.norm(move_xy, dim=-1, keepdim=True) + eps
        cap = self.v_cap
        scaled = cap * torch.tanh(mag / (cap + eps))
        move_xy = move_xy * (scaled / mag)

        move3 = torch.cat([move_xy, dz], dim=1)

        # Gripper
        grip = self.grip_bias.unsqueeze(0).expand(B, 1)

        mean = torch.cat([move3, grip], dim=1)

        # Std: shrink during push and brake
        base_std = torch.exp(self.log_std_vec.unsqueeze(0).expand(B, 4))
        shrink = 1.0 - self.std_push_brake * torch.clamp(w_push + w_brake, 0.0, 1.0)
        shrink = torch.clamp(shrink, 0.1, 1.0)
        std = base_std * shrink

        return (mean, std)

    def get_param_ranges(self):
        return {
            "r_back": (0.02, 0.20),
            "z_clear": (0.03, 0.25),
            "h_push": (0.005, 0.06),
            "kp_app": (0.1, 20.0),
            "kd_app": (0.0, 3.0),
            "kp_align": (0.0, 10.0),
            "kd_align": (0.0, 3.0),
            "v_push": (0.0, 8.0),
            "k_lat": (0.0, 20.0),
            "kd_push": (0.0, 3.0),
            "v_brake": (0.0, 5.0),
            "k_hold": (0.0, 15.0),
            "v_side": (0.0, 3.0),
            "a_dec": (0.5, 10.0),
            "alpha_stop": (5.0, 200.0),
            "r_app": (0.02, 0.40),
            "r_align": (0.01, 0.20),
            "r_contact": (0.01, 0.20),
            "r_goal": (0.01, 0.20),
            "alpha_app": (5.0, 200.0),
            "alpha_align": (5.0, 200.0),
            "alpha_contact": (5.0, 200.0),
            "alpha_goal": (5.0, 200.0),
            "beta_smooth": (0.0, 0.95),
            "kp_z": (0.5, 30.0),
            "kd_z": (0.0, 3.0),
            "z_lift": (0.0, 0.06),
            "v_cap": (0.2, 5.0),
            "grip_bias": (-2.0, 2.0),
            "log_std_vec": (-5.0, 0.0),
            "std_push_brake": (0.0, 0.95),
        }
