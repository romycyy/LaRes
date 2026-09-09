# Frozen structure for the Phase 3 fitting comparison. Do not edit by hand:
# regenerate with scripts/build_structure_bank.py.
# provenance: gen_1 candidate 4 of the 2026-08-28 search; reported score 51.71
# ported: raw observation slices rewritten to named accessors; outputs verified
# identical to the original on random observations at two scales.

class GeneratedPolicy(SymbolicPolicy):
    """
    Hypothesis 4:
    Progress-based waypoint tracking with planned overshoot and retract to pin.
    """
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)

        # Waypoints and overshoot
        self.d_back = nn.Parameter(torch.tensor(0.07))
        self.h_above = nn.Parameter(torch.tensor(0.12))
        self.z_contact = nn.Parameter(torch.tensor(0.02))
        self.o_dist = nn.Parameter(torch.tensor(0.04))     # overshoot distance beyond goal

        # Progress shaping p = 1 - d/(d + den)
        self.p_den = nn.Parameter(torch.tensor(0.20))
        self.p_sw = nn.Parameter(torch.tensor(0.5))        # switch to push-through
        self.alpha2 = nn.Parameter(torch.tensor(20.0))

        # Approach gate to W1
        self.r_app = nn.Parameter(torch.tensor(0.10))
        self.alpha1 = nn.Parameter(torch.tensor(60.0))

        # Controller gains
        self.kp_xy = nn.Parameter(torch.tensor(6.0))
        self.kd_xy = nn.Parameter(torch.tensor(0.6))
        self.v_ff = nn.Parameter(torch.tensor(2.0))        # along u during push-through

        # Brake/settle near goal
        self.r_goal = nn.Parameter(torch.tensor(0.05))
        self.alpha_b = nn.Parameter(torch.tensor(80.0))
        self.k_goal = nn.Parameter(torch.tensor(4.0))
        self.v_brake = nn.Parameter(torch.tensor(0.8))

        # Z tracking and saturation
        self.kp_z = nn.Parameter(torch.tensor(12.0))
        self.kd_z = nn.Parameter(torch.tensor(0.7))
        self.z_cap = nn.Parameter(torch.tensor(2.0))

        # Speed cap
        self.v_cap = nn.Parameter(torch.tensor(2.2))

        # Gripper
        self.grip_bias = nn.Parameter(torch.tensor(-0.6))

        # Std
        self.log_std_vec = nn.Parameter(torch.tensor([-1.6, -1.6, -2.1, -2.0]))

    def forward(self, obs):
        B = obs.shape[0]
        eps = 1e-8

        tcp = self.obs_field(obs, "tcp")
        obj = self.obs_field(obs, "obj")
        goal = self.obs_field(obs, "goal")
        prev_tcp = self.obs_field(obs, "prev_tcp")

        v_tcp = tcp - prev_tcp

        # Geometry and direction
        g_vec = goal - obj
        d_goal = torch.norm(g_vec, dim=-1, keepdim=True) + eps
        u = g_vec / d_goal
        u_xy = u[:, 0:2]
        u_xy = u_xy / (torch.norm(u_xy, dim=-1, keepdim=True) + eps)

        # Progress proxy p in [0,1)
        den = torch.clamp(self.p_den, min=0.05, max=1.0)
        p = 1.0 - d_goal / (d_goal + den)

        # Overshoot target
        G_ov = goal + self.o_dist * u

        # Waypoints
        W1 = obj - self.d_back * u
        W1 = torch.cat([W1[:, 0:2], self.h_above.unsqueeze(0).expand(B, 1)], dim=1)
        W2 = G_ov - self.d_back * u
        W2 = torch.cat([W2[:, 0:2], self.z_contact.unsqueeze(0).expand(B, 1)], dim=1)

        # Blend W = (1-g2)*W1 + g2*W2
        g2 = torch.sigmoid(self.alpha2 * (p - self.p_sw))
        W = (1.0 - g2) * W1 + g2 * W2

        # Approach gate to reduce v_ff until near W1
        d_app = torch.norm(tcp - W1, dim=-1, keepdim=True) + eps
        g_app = 1.0 - torch.sigmoid(self.alpha1 * (d_app - self.r_app))  # high when close

        # Planar control to W plus feedforward along u during push-through
        err_xy = tcp[:, 0:2] - W[:, 0:2]
        move_xy = -self.kp_xy * err_xy - self.kd_xy * v_tcp[:, 0:2] + (self.v_ff * g2 * g_app) * u_xy

        # Brake near goal
        b = torch.sigmoid(self.alpha_b * (self.r_goal - d_goal))
        move_xy = (1.0 - b) * move_xy + b * (-self.k_goal * (obj[:, 0:2] - goal[:, 0:2]) - self.v_brake * u_xy - self.kd_xy * v_tcp[:, 0:2])

        # Speed cap
        mag = torch.norm(move_xy, dim=-1, keepdim=True) + eps
        cap = self.v_cap
        scaled = cap * torch.tanh(mag / (cap + eps))
        move_xy = move_xy * (scaled / mag)

        # Vertical tracking to W_z with saturation
        tcp_z = tcp[:, 2:3]
        v_tcp_z = v_tcp[:, 2:3]
        dz_raw = -self.kp_z * (tcp_z - W[:, 2:3]) - self.kd_z * v_tcp_z
        dz = self.z_cap * torch.tanh(dz_raw / (self.z_cap + eps))

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
            "o_dist": (0.0, 0.20),
            "p_den": (0.05, 1.0),
            "p_sw": (0.1, 0.9),
            "alpha2": (1.0, 100.0),
            "r_app": (0.02, 0.30),
            "alpha1": (5.0, 200.0),
            "kp_xy": (0.1, 20.0),
            "kd_xy": (0.0, 3.0),
            "v_ff": (0.0, 8.0),
            "r_goal": (0.01, 0.20),
            "alpha_b": (5.0, 200.0),
            "k_goal": (0.0, 15.0),
            "v_brake": (0.0, 5.0),
            "kp_z": (0.5, 30.0),
            "kd_z": (0.0, 3.0),
            "z_cap": (0.2, 5.0),
            "v_cap": (0.2, 5.0),
            "grip_bias": (-2.0, 2.0),
            "log_std_vec": (-5.0, 0.0),
        }
