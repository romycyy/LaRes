# Frozen structure for the Phase 3 fitting comparison. Do not edit by hand:
# regenerate with scripts/build_structure_bank.py.
# provenance: gen_1 candidate 1 of the 2026-08-28 search; reported score 614.85
# ported: raw observation slices rewritten to named accessors; outputs verified
# identical to the original on random observations at two scales.

class GeneratedPolicy(SymbolicPolicy):
    """
    Hypothesis 3:
    Slip-aware push with lateral alignment, compression control, and smooth braking.
    """
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)

        # Geometry and heights
        self.d_back = nn.Parameter(torch.tensor(0.07))
        self.d_comp = nn.Parameter(torch.tensor(0.01))    # compression target along u at contact
        self.h_above = nn.Parameter(torch.tensor(0.12))
        self.z_contact = nn.Parameter(torch.tensor(0.02))

        # Lateral centering and compression
        self.k_lat = nn.Parameter(torch.tensor(3.5))
        self.k_comp = nn.Parameter(torch.tensor(3.0))

        # Push drive and slip feedback
        self.v_push = nn.Parameter(torch.tensor(2.5))
        self.k_slip = nn.Parameter(torch.tensor(1.0))
        self.s_thr = nn.Parameter(torch.tensor(0.02))     # slip threshold for extra drive

        # Damping
        self.kd_xy = nn.Parameter(torch.tensor(0.6))

        # Brake and hold near goal
        self.v_brake = nn.Parameter(torch.tensor(0.8))
        self.k_hold = nn.Parameter(torch.tensor(4.0))
        self.r_goal = nn.Parameter(torch.tensor(0.05))
        self.alpha_b = nn.Parameter(torch.tensor(80.0))

        # Contact gate
        self.r_c = nn.Parameter(torch.tensor(0.06))
        self.alpha_c = nn.Parameter(torch.tensor(80.0))

        # Vertical impedance
        self.kp_z = nn.Parameter(torch.tensor(12.0))
        self.kd_z = nn.Parameter(torch.tensor(0.7))

        # Speed cap
        self.v_cap = nn.Parameter(torch.tensor(2.2))

        # Gripper
        self.grip_bias = nn.Parameter(torch.tensor(-0.6))

        # Std
        self.log_std_vec = nn.Parameter(torch.tensor([-1.6, -1.6, -2.1, -2.1]))

    def forward(self, obs):
        B = obs.shape[0]
        eps = 1e-8

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
        u = g_vec / d_goal
        u_xy = u[:, 0:2]
        u_xy = u_xy / (torch.norm(u_xy, dim=-1, keepdim=True) + eps)

        # Contact gate
        d_to_obj = torch.norm(tcp - obj, dim=-1, keepdim=True) + eps
        c = torch.sigmoid(self.alpha_c * (self.r_c - d_to_obj))

        # Base point behind puck
        A = obj - self.d_back * u
        A_xy = A[:, 0:2]

        # Lateral centering relative to A (orthogonal to u)
        diff_xy = tcp[:, 0:2] - A_xy
        along = torch.sum(diff_xy * u_xy, dim=-1, keepdim=True) * u_xy
        side = diff_xy - along
        center_xy = -self.k_lat * side

        # Compression along u: target at obj - d_comp*u
        comp_err = torch.sum((tcp - (obj - self.d_comp * u)) * u, dim=-1, keepdim=True)
        comp_term_xy = -self.k_comp * c * comp_err * u_xy

        # Slip-aware push drive
        rel_along = torch.sum((v_tcp - v_obj) * u, dim=-1, keepdim=True)
        extra = self.k_slip * torch.relu(rel_along - self.s_thr)
        push_xy = c * (self.v_push + extra) * u_xy

        # Damping
        damp_xy = -self.kd_xy * v_tcp[:, 0:2]

        # Combine planar
        move_xy = center_xy + comp_term_xy + push_xy + damp_xy

        # Braking near goal
        b = torch.sigmoid(self.alpha_b * (self.r_goal - d_goal))
        move_xy = (1.0 - b) * move_xy + b * (-self.v_brake * u_xy - self.k_hold * (obj[:, 0:2] - goal[:, 0:2]) + damp_xy)

        # Speed cap
        mag = torch.norm(move_xy, dim=-1, keepdim=True) + eps
        cap = self.v_cap
        scaled = cap * torch.tanh(mag / (cap + eps))
        move_xy = move_xy * (scaled / mag)

        # Vertical control: blend above height and contact height by c
        z_des = (1.0 - c) * self.h_above + c * self.z_contact
        tcp_z = tcp[:, 2:3]
        v_tcp_z = v_tcp[:, 2:3]
        dz = -self.kp_z * (tcp_z - z_des) - self.kd_z * v_tcp_z

        move3 = torch.cat([move_xy, dz], dim=1)

        # Gripper
        grip = self.grip_bias.unsqueeze(0).expand(B, 1)
        mean = torch.cat([move3, grip], dim=1)

        std = torch.exp(self.log_std_vec.unsqueeze(0).expand(B, 4))
        return (mean, std)

    def get_param_ranges(self):
        return {
            "d_back": (0.02, 0.20),
            "d_comp": (0.0, 0.05),
            "h_above": (0.05, 0.30),
            "z_contact": (0.0, 0.08),
            "k_lat": (0.0, 15.0),
            "k_comp": (0.0, 15.0),
            "v_push": (0.0, 8.0),
            "k_slip": (0.0, 5.0),
            "s_thr": (0.0, 0.10),
            "kd_xy": (0.0, 3.0),
            "v_brake": (0.0, 5.0),
            "k_hold": (0.0, 15.0),
            "r_goal": (0.01, 0.20),
            "alpha_b": (5.0, 200.0),
            "r_c": (0.01, 0.15),
            "alpha_c": (5.0, 200.0),
            "kp_z": (0.5, 30.0),
            "kd_z": (0.0, 3.0),
            "v_cap": (0.2, 5.0),
            "grip_bias": (-2.0, 2.0),
            "log_std_vec": (-5.0, 0.0),
        }
