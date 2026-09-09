# Frozen structure for the Phase 3 fitting comparison. Do not edit by hand:
# regenerate with scripts/build_structure_bank.py.
# provenance: gen_1 candidate 2 of the 2026-08-28 search; reported score 80.84
# ported: raw observation slices rewritten to named accessors; outputs verified
# identical to the original on random observations at two scales.

class GeneratedPolicy(SymbolicPolicy):
    """
    Hypothesis 1:
    Four-phase, behind-contact push with soft blending:
      - Approach-above -> Descend -> Push -> Brake/Settle
    """
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        # Geometric anchors
        self.d_back = nn.Parameter(torch.tensor(0.07))       # behind offset [m]
        self.h_above = nn.Parameter(torch.tensor(0.12))      # approach height [m]
        self.z_contact = nn.Parameter(torch.tensor(0.02))    # contact height [m]

        # Approach-above (XY) gains
        self.kp_app = nn.Parameter(torch.tensor(5.0))
        self.kd_app = nn.Parameter(torch.tensor(0.6))

        # Descend phase gains
        self.kp_align = nn.Parameter(torch.tensor(2.0))      # small XY alignment to A_con
        self.kd_align = nn.Parameter(torch.tensor(0.4))
        self.kpz_desc = nn.Parameter(torch.tensor(10.0))
        self.kdz_desc = nn.Parameter(torch.tensor(0.6))

        # Push phase gains
        self.v_push = nn.Parameter(torch.tensor(2.0))        # feedforward push speed
        self.k_lat = nn.Parameter(torch.tensor(3.0))         # lateral centering stiffness
        self.kd_push = nn.Parameter(torch.tensor(0.6))       # damping

        # Brake/settle near goal
        self.v_brake = nn.Parameter(torch.tensor(0.8))
        self.k_hold = nn.Parameter(torch.tensor(3.0))

        # Z control at approach/push/brake
        self.kpz_push = nn.Parameter(torch.tensor(12.0))
        self.kdz_push = nn.Parameter(torch.tensor(0.7))

        # Gating thresholds and sharpness
        self.r_pre = nn.Parameter(torch.tensor(0.10))        # distance to A_pre to switch to descend
        self.r_con = nn.Parameter(torch.tensor(0.06))        # tcp-object distance for contact
        self.r_goal = nn.Parameter(torch.tensor(0.04))       # obj-goal distance for braking
        self.alpha_pre = nn.Parameter(torch.tensor(60.0))
        self.alpha_con = nn.Parameter(torch.tensor(80.0))
        self.alpha_goal = nn.Parameter(torch.tensor(80.0))

        # Speed cap for XY via tanh
        self.v_cap = nn.Parameter(torch.tensor(2.0))

        # Gripper control (closed-ish)
        self.grip_bias = nn.Parameter(torch.tensor(-0.5))

        # Per-action log std (dx, dy, dz, grip)
        self.log_std_vec = nn.Parameter(torch.tensor([-1.6, -1.6, -2.0, -2.0]))

    def forward(self, obs):
        B = obs.shape[0]
        eps = 1e-8

        # Observations and finite-difference velocities
        tcp = self.obs_field(obs, "tcp")
        obj = self.obs_field(obs, "obj")
        goal = self.obs_field(obs, "goal")
        prev_tcp = self.obs_field(obs, "prev_tcp")
        prev_obj = self.obs_field(obs, "prev_obj")
        v_tcp = tcp - prev_tcp
        v_obj = obj - prev_obj

        # Geometry
        g_vec = goal - obj
        d_goal = torch.norm(g_vec, dim=-1, keepdim=True) + eps
        u = g_vec / d_goal  # (B,3)
        u_xy = u[:, 0:2]
        u_xy = u_xy / (torch.norm(u_xy, dim=-1, keepdim=True) + eps)

        # Anchors
        A_pre = obj - self.d_back * u
        A_pre = torch.cat([A_pre[:, 0:2], self.h_above.unsqueeze(0).expand(B, 1)], dim=1)
        A_con = obj - self.d_back * u
        A_con = torch.cat([A_con[:, 0:2], self.z_contact.unsqueeze(0).expand(B, 1)], dim=1)

        # Distances for gating
        d_pre = torch.norm(tcp - A_pre, dim=-1, keepdim=True) + eps
        d_con = torch.norm(tcp - obj, dim=-1, keepdim=True) + eps

        g_preclose = torch.sigmoid(self.alpha_pre * (self.r_pre - d_pre))
        g_contact = torch.sigmoid(self.alpha_con * (self.r_con - d_con))
        g_goal = torch.sigmoid(self.alpha_goal * (self.r_goal - d_goal))

        # Phase weights
        w_brake = g_goal
        w_push = (1.0 - g_goal) * g_contact
        w_desc = (1.0 - g_goal) * (1.0 - g_contact) * g_preclose
        w_pre = 1.0 - w_push - w_desc - w_brake
        # Normalize to sum to 1
        w_sum = w_pre + w_desc + w_push + w_brake + eps
        w_pre = w_pre / w_sum
        w_desc = w_desc / w_sum
        w_push = w_push / w_sum
        w_brake = w_brake / w_sum

        # Controllers
        tcp_xy = tcp[:, 0:2]
        v_tcp_xy = v_tcp[:, 0:2]
        A_pre_xy = A_pre[:, 0:2]
        A_con_xy = A_con[:, 0:2]
        tcp_z = tcp[:, 2:3]
        v_tcp_z = v_tcp[:, 2:3]

        # 1) Approach-above (XY to A_pre_xy), keep Z at h_above
        move_pre_xy = -self.kp_app * (tcp_xy - A_pre_xy) - self.kd_app * v_tcp_xy
        dz_pre = 0.0 * tcp_z + 0.0  # implicit: held by A_pre_z in blending through descend controller

        # 2) Descend: align XY to A_con_xy (small), Z -> z_contact
        move_desc_xy = -self.kp_align * (tcp_xy - A_con_xy) - self.kd_align * v_tcp_xy
        dz_desc = -self.kpz_desc * (tcp_z - self.z_contact.unsqueeze(0).expand(B, 1)) - self.kdz_desc * v_tcp_z

        # 3) Push: feedforward along u + lateral centering to A_con, Z hold at contact
        # Lateral error: component orthogonal to u in XY of (tcp - A_con)
        err_xy = tcp_xy - A_con_xy
        along = torch.sum(err_xy * u_xy, dim=-1, keepdim=True) * u_xy
        side = err_xy - along
        move_push_xy = self.v_push * u_xy - self.k_lat * side - self.kd_push * v_tcp_xy
        dz_push = -self.kpz_push * (tcp_z - self.z_contact.unsqueeze(0).expand(B, 1)) - self.kdz_push * v_tcp_z

        # 4) Brake/Settle near goal: pull back slightly and pin object to goal
        obj_goal_xy = (obj[:, 0:2] - goal[:, 0:2])
        move_brake_xy = -self.v_brake * u_xy - self.k_hold * obj_goal_xy - self.kd_push * v_tcp_xy
        dz_brake = dz_push  # keep contact height while braking

        # Blend
        move_xy = w_pre * move_pre_xy + w_desc * move_desc_xy + w_push * move_push_xy + w_brake * move_brake_xy
        dz = w_pre * dz_pre + w_desc * dz_desc + w_push * dz_push + w_brake * dz_brake

        # Soft speed cap on XY
        mag = torch.norm(move_xy, dim=-1, keepdim=True) + eps
        cap = self.v_cap
        scaled_mag = cap * torch.tanh(mag / (cap + eps))
        move_xy = move_xy * (scaled_mag / mag)

        move3 = torch.cat([move_xy, dz], dim=1)

        # Gripper mean (closed bias)
        grip = self.grip_bias.unsqueeze(0).expand(B, 1)

        mean = torch.cat([move3, grip], dim=1)

        # Std
        log_std = self.log_std_vec.unsqueeze(0).expand(B, 4)
        std = torch.exp(log_std)

        return (mean, std)

    def get_param_ranges(self):
        return {
            "d_back": (0.02, 0.20),
            "h_above": (0.05, 0.30),
            "z_contact": (0.0, 0.08),
            "kp_app": (0.1, 15.0),
            "kd_app": (0.0, 3.0),
            "kp_align": (0.0, 8.0),
            "kd_align": (0.0, 3.0),
            "kpz_desc": (0.5, 30.0),
            "kdz_desc": (0.0, 3.0),
            "v_push": (0.0, 8.0),
            "k_lat": (0.0, 15.0),
            "kd_push": (0.0, 3.0),
            "v_brake": (0.0, 5.0),
            "k_hold": (0.0, 15.0),
            "kpz_push": (0.5, 30.0),
            "kdz_push": (0.0, 3.0),
            "r_pre": (0.02, 0.30),
            "r_con": (0.01, 0.15),
            "r_goal": (0.01, 0.20),
            "alpha_pre": (5.0, 200.0),
            "alpha_con": (5.0, 200.0),
            "alpha_goal": (5.0, 200.0),
            "v_cap": (0.2, 5.0),
            "grip_bias": (-2.0, 2.0),
            "log_std_vec": (-5.0, 0.0),
        }
