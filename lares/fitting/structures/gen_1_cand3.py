# Frozen structure for the Phase 3 fitting comparison. Do not edit by hand:
# regenerate with scripts/build_structure_bank.py.
# provenance: gen_1 candidate 3 of the 2026-08-28 search; reported score 76.23
# ported: raw observation slices rewritten to named accessors; outputs verified
# identical to the original on random observations at two scales.

class GeneratedPolicy(SymbolicPolicy):
    """
    Hypothesis 2:
    Moving-anchor compliance field with contact gating and feedforward.
    """
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)

        # Geometry
        self.d_back = nn.Parameter(torch.tensor(0.07))     # behind offset
        self.h_above = nn.Parameter(torch.tensor(0.12))    # high (no contact)
        self.z_contact = nn.Parameter(torch.tensor(0.02))  # contact height

        # Compliance gains
        self.kp_xy = nn.Parameter(torch.tensor(6.0))
        self.kd_xy = nn.Parameter(torch.tensor(0.6))
        self.gamma_obj = nn.Parameter(torch.tensor(0.3))   # couple to object velocity

        # Lateral error rejection to keep centered behind puck
        self.k_lat = nn.Parameter(torch.tensor(3.0))

        # Feedforward push
        self.v_ff = nn.Parameter(torch.tensor(2.5))
        self.alpha_goal = nn.Parameter(torch.tensor(80.0))
        self.r_goal = nn.Parameter(torch.tensor(0.05))
        self.k_goal = nn.Parameter(torch.tensor(4.0))      # hold at goal

        # Vertical impedance and saturation
        self.kp_z = nn.Parameter(torch.tensor(12.0))
        self.kd_z = nn.Parameter(torch.tensor(0.7))
        self.z_cap = nn.Parameter(torch.tensor(2.0))       # cap for dz via tanh

        # Contact gating
        self.r_c = nn.Parameter(torch.tensor(0.06))
        self.alpha_c = nn.Parameter(torch.tensor(80.0))

        # Speed cap for XY
        self.v_cap = nn.Parameter(torch.tensor(2.2))

        # Gripper
        self.grip_bias = nn.Parameter(torch.tensor(-0.6))

        # Std
        self.log_std_vec = nn.Parameter(torch.tensor([-1.6, -1.6, -2.1, -2.0]))

    def forward(self, obs):
        B = obs.shape[0]
        eps = 1e-8

        # Observations
        tcp = self.obs_field(obs, "tcp")
        obj = self.obs_field(obs, "obj")
        goal = self.obs_field(obs, "goal")
        prev_tcp = self.obs_field(obs, "prev_tcp")
        prev_obj = self.obs_field(obs, "prev_obj")

        v_tcp = tcp - prev_tcp
        v_obj = obj - prev_obj

        # Direction to goal from object
        g_vec = goal - obj
        d_goal = torch.norm(g_vec, dim=-1, keepdim=True) + eps
        u = g_vec / d_goal
        u_xy = u[:, 0:2]
        u_xy = u_xy / (torch.norm(u_xy, dim=-1, keepdim=True) + eps)

        # Contact confidence based on tcp-object distance
        d_to_obj = torch.norm(tcp - obj, dim=-1, keepdim=True) + eps
        c = torch.sigmoid(self.alpha_c * (self.r_c - d_to_obj))

        # Moving anchor A(c) = obj - d_back*u at height z(c)
        A = obj - self.d_back * u
        z_des = (1.0 - c) * self.h_above + c * self.z_contact
        A = torch.cat([A[:, 0:2], z_des], dim=1)

        # Planar compliance about A with object-velocity coupling
        err = tcp - A
        err_xy = err[:, 0:2]
        v_eff_xy = v_tcp[:, 0:2] - self.gamma_obj * v_obj[:, 0:2]
        delta_xy = -self.kp_xy * err_xy - self.kd_xy * v_eff_xy

        # Lateral error rejection relative to line (obj - d_back*u)
        base_pt = obj - self.d_back * u
        base_xy = base_pt[:, 0:2]
        diff_xy = tcp[:, 0:2] - base_xy
        along = torch.sum(diff_xy * u_xy, dim=-1, keepdim=True) * u_xy
        side = diff_xy - along
        delta_xy = delta_xy - self.k_lat * side

        # Feedforward push scaled by contact and reduced near goal
        s_goal = 1.0 - torch.sigmoid(self.alpha_goal * (self.r_goal - d_goal))
        delta_xy = delta_xy + (self.v_ff * c * s_goal) * u_xy

        # Near goal holding
        delta_xy = delta_xy - self.k_goal * torch.sigmoid(self.alpha_goal * (self.r_goal - d_goal)) * (obj[:, 0:2] - goal[:, 0:2])

        # Soft cap XY
        mag = torch.norm(delta_xy, dim=-1, keepdim=True) + eps
        vcap = self.v_cap
        scaled = vcap * torch.tanh(mag / (vcap + eps))
        move_xy = delta_xy * (scaled / mag)

        # Vertical impedance with tanh saturation
        tcp_z = tcp[:, 2:3]
        v_tcp_z = v_tcp[:, 2:3]
        dz_raw = -self.kp_z * (tcp_z - z_des) - self.kd_z * v_tcp_z
        zcap = self.z_cap
        dz = zcap * torch.tanh(dz_raw / (zcap + eps))

        move3 = torch.cat([move_xy, dz], dim=1)

        # Gripper
        grip = self.grip_bias.unsqueeze(0).expand(B, 1)

        mean = torch.cat([move3, grip], dim=1)
        std = torch.exp(self.log_std_vec.unsqueeze(0).expand(B, 4))
        return (mean, std)

    def get_param_ranges(self):
        return {
            "d_back": (0.02, 0.20),
            "h_above": (0.05, 0.30),
            "z_contact": (0.0, 0.08),
            "kp_xy": (0.1, 20.0),
            "kd_xy": (0.0, 3.0),
            "gamma_obj": (0.0, 1.0),
            "k_lat": (0.0, 15.0),
            "v_ff": (0.0, 8.0),
            "alpha_goal": (5.0, 200.0),
            "r_goal": (0.01, 0.20),
            "k_goal": (0.0, 15.0),
            "kp_z": (0.5, 30.0),
            "kd_z": (0.0, 3.0),
            "z_cap": (0.2, 5.0),
            "r_c": (0.01, 0.15),
            "alpha_c": (5.0, 200.0),
            "v_cap": (0.2, 5.0),
            "grip_bias": (-2.0, 2.0),
            "log_std_vec": (-5.0, 0.0),
        }
