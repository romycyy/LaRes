# Frozen structure for the Phase 3 fitting comparison. Do not edit by hand:
# regenerate with scripts/build_structure_bank.py.
# provenance: gen_1 candidate 0 of the 2026-08-28 search; reported score 638.77
# ported: raw observation slices rewritten to named accessors; outputs verified
# identical to the original on random observations at two scales.

class GeneratedPolicy(SymbolicPolicy):
    """
    Hypothesis 5:
    Frenet-frame path following with lookahead and object-velocity damping.
    """
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)

        # Geometry and heights
        self.d_back = nn.Parameter(torch.tensor(0.07))
        self.z_contact = nn.Parameter(torch.tensor(0.02))
        self.h_above = nn.Parameter(torch.tensor(0.12))

        # Frenet-frame gains
        self.k_t = nn.Parameter(torch.tensor(4.0))   # tangential (along u)
        self.k_n = nn.Parameter(torch.tensor(5.0))   # normal (lateral)
        self.kd_xy = nn.Parameter(torch.tensor(0.6))
        self.beta_obj = nn.Parameter(torch.tensor(0.3))  # couple to object vel

        # Feedforward lookahead
        self.v0 = nn.Parameter(torch.tensor(1.0))
        self.v1 = nn.Parameter(torch.tensor(3.0))
        self.alpha_d = nn.Parameter(torch.tensor(40.0))
        self.r_d = nn.Parameter(torch.tensor(0.10))

        # Approach gate
        self.r_app = nn.Parameter(torch.tensor(0.10))
        self.alpha_a = nn.Parameter(torch.tensor(60.0))

        # Stop gate near goal
        self.r_goal = nn.Parameter(torch.tensor(0.05))
        self.alpha_s = nn.Parameter(torch.tensor(80.0))
        self.k_goal = nn.Parameter(torch.tensor(4.0))
        self.v_brake = nn.Parameter(torch.tensor(0.8))

        # Vertical control
        self.kp_z = nn.Parameter(torch.tensor(12.0))
        self.kd_z = nn.Parameter(torch.tensor(0.7))

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
        prev_obj = self.obs_field(obs, "prev_obj")

        v_tcp = tcp - prev_tcp
        v_obj = obj - prev_obj

        # Geometry and Frenet frame
        g_vec = goal - obj
        d_goal = torch.norm(g_vec, dim=-1, keepdim=True) + eps
        u = g_vec / d_goal
        u_xy = u[:, 0:2]
        u_xy = u_xy / (torch.norm(u_xy, dim=-1, keepdim=True) + eps)

        # Desired contact anchor A = obj - d_back * u at z_contact
        A = obj - self.d_back * u
        A_xy = A[:, 0:2]

        # Frenet errors
        e_vec_xy = tcp[:, 0:2] - A_xy
        e_t = torch.sum(e_vec_xy * u_xy, dim=-1, keepdim=True)  # along u
        e_n_xy = e_vec_xy - e_t * u_xy                          # lateral

        # Feedforward lookahead velocity along u
        d = d_goal
        look = 1.0 - torch.sigmoid(self.alpha_d * (d - self.r_d))  # increases as d decreases
        v_ff = self.v0 + (self.v1 - self.v0) * look

        # Approach gate: hold off strong feedforward until near and aligned
        d_app = torch.norm(tcp - (obj + torch.tensor([0.0, 0.0, 0.0], device=obs.device)), dim=-1, keepdim=True) + eps
        g_app = 1.0 - torch.sigmoid(self.alpha_a * (d_app - self.r_app))

        # Stop gate near goal
        g_stop = torch.sigmoid(self.alpha_s * (self.r_goal - d_goal))

        # Planar command
        move_xy = (-self.k_t * e_t) * u_xy - self.k_n * e_n_xy - self.kd_xy * (v_tcp[:, 0:2] - self.beta_obj * v_obj[:, 0:2])
        move_xy = move_xy + (v_ff * g_app * (1.0 - g_stop)) * u_xy
        move_xy = move_xy + g_stop * (-self.k_goal * (obj[:, 0:2] - goal[:, 0:2]) - self.v_brake * u_xy)

        # Speed cap
        mag = torch.norm(move_xy, dim=-1, keepdim=True) + eps
        cap = self.v_cap
        scaled = cap * torch.tanh(mag / (cap + eps))
        move_xy = move_xy * (scaled / mag)

        # Z control: blend by approach gate
        z_des = (1.0 - g_app) * self.z_contact + g_app * self.h_above
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
            "z_contact": (0.0, 0.08),
            "h_above": (0.05, 0.30),
            "k_t": (0.1, 12.0),
            "k_n": (0.1, 20.0),
            "kd_xy": (0.0, 3.0),
            "beta_obj": (0.0, 1.0),
            "v0": (0.0, 5.0),
            "v1": (0.0, 8.0),
            "alpha_d": (5.0, 200.0),
            "r_d": (0.02, 0.30),
            "r_app": (0.02, 0.30),
            "alpha_a": (5.0, 200.0),
            "r_goal": (0.01, 0.20),
            "alpha_s": (5.0, 200.0),
            "k_goal": (0.0, 15.0),
            "v_brake": (0.0, 5.0),
            "kp_z": (0.5, 30.0),
            "kd_z": (0.0, 3.0),
            "v_cap": (0.2, 5.0),
            "grip_bias": (-2.0, 2.0),
            "log_std_vec": (-5.0, 0.0),
        }
