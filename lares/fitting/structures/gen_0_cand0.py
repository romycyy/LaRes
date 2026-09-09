# Frozen structure for the Phase 3 fitting comparison. Do not edit by hand:
# regenerate with scripts/build_structure_bank.py.
# provenance: gen_0 candidate 0 of the 2026-08-28 search; reported score 102.60
# ported: raw observation slices rewritten to named accessors; outputs verified
# identical to the original on random observations at two scales.

class GeneratedPolicy(SymbolicPolicy):
    """
    Hypothesis 1:
    Phase-gated approach–contact–push controller with smooth blending using object/goal geometry and finite-difference velocities.
    """
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)

        # Geometric/height parameters
        self.m_back = nn.Parameter(torch.tensor(0.06))        # standoff behind puck [m]
        self.m_pen = nn.Parameter(torch.tensor(0.01))         # small penetration during contact [m]
        self.z_approach = nn.Parameter(torch.tensor(0.12))    # safe approach height [m]
        self.z_contact = nn.Parameter(torch.tensor(0.02))     # contact height [m]

        # Gains (phase-specific)
        self.k_lat_app = nn.Parameter(torch.tensor(4.0))      # approach lateral P
        self.b_lat_app = nn.Parameter(torch.tensor(0.5))      # approach lateral D (tcp vel damping)
        self.kz_app = nn.Parameter(torch.tensor(6.0))         # approach Z P
        self.bz_app = nn.Parameter(torch.tensor(0.5))         # approach Z D

        self.k_lat_con = nn.Parameter(torch.tensor(5.0))      # contact lateral P
        self.b_lat_con = nn.Parameter(torch.tensor(0.6))      # contact lateral D
        self.kz_con = nn.Parameter(torch.tensor(8.0))         # contact Z P
        self.bz_con = nn.Parameter(torch.tensor(0.6))         # contact Z D

        self.k_push = nn.Parameter(torch.tensor(3.0))         # push along-track gain
        self.k_side_push = nn.Parameter(torch.tensor(2.0))    # lateral alignment during push
        self.k_slip = nn.Parameter(torch.tensor(1.0))         # slip damping along d_hat
        self.b_lat_push = nn.Parameter(torch.tensor(0.5))     # push lateral damping
        self.kz_push = nn.Parameter(torch.tensor(10.0))       # push Z P
        self.bz_push = nn.Parameter(torch.tensor(0.7))        # push Z D

        # Saturation for XY speed (soft via tanh)
        self.v_cap = nn.Parameter(torch.tensor(2.0))

        # Phase gating thresholds and sharpness
        self.d_contact = nn.Parameter(torch.tensor(0.06))     # distance threshold for contact
        self.v_contact = nn.Parameter(torch.tensor(0.01))     # relative normal vel threshold
        self.gate_sharp = nn.Parameter(torch.tensor(80.0))    # sharpness for distance gate
        self.align_sharp = nn.Parameter(torch.tensor(40.0))   # sharpness for alignment/behindness

        # Gripper control (closed bias) and its std
        self.grip_bias = nn.Parameter(torch.tensor(-0.5))

        # Per-phase log-std vectors (4D: dx, dy, dz, gripper)
        self.log_std_app = nn.Parameter(torch.tensor([-0.7, -0.7, -0.7, -0.5]))
        self.log_std_con = nn.Parameter(torch.tensor([-1.2, -1.2, -1.0, -1.2]))
        self.log_std_push = nn.Parameter(torch.tensor([-1.5, -1.5, -1.2, -1.2]))

    def forward(self, obs):
        B = obs.shape[0]
        eps = 1e-8

        # Parse observations
        tcp = self.obs_field(obs, "tcp")
        obj = self.obs_field(obs, "obj")
        goal = self.obs_field(obs, "goal")

        prev_tcp = self.obs_field(obs, "prev_tcp")
        prev_obj = self.obs_field(obs, "prev_obj")

        v_tcp = tcp - prev_tcp
        v_obj = obj - prev_obj

        # Geometry
        to_obj = obj - tcp
        dist_to_obj = torch.norm(to_obj, dim=-1, keepdim=True) + eps
        u_hat = to_obj / dist_to_obj

        to_goal_from_obj = goal - obj
        dist_obj_goal = torch.norm(to_goal_from_obj, dim=-1, keepdim=True) + eps
        d_hat = to_goal_from_obj / dist_obj_goal

        # Pre-push waypoint behind the puck
        W = obj - self.m_back * d_hat

        # Contact confidence: close distance AND low relative normal velocity
        rel_vel_along_d = torch.sum((v_tcp - v_obj) * d_hat, dim=-1, keepdim=True)
        g_close = torch.sigmoid(self.gate_sharp * (self.d_contact - dist_to_obj))
        g_low_vel = torch.sigmoid(self.gate_sharp * (self.v_contact - torch.abs(rel_vel_along_d)))
        g_contact = g_close * g_low_vel

        # Behindness (positive if tcp is behind puck along d_hat)
        behindness = torch.sum((obj - tcp) * d_hat, dim=-1, keepdim=True)
        g_push = g_contact * torch.sigmoid(self.align_sharp * (behindness - 0.5 * self.m_back))

        # Phase weights (normalized)
        w_app = 1.0 - g_contact
        w_con = g_contact * (1.0 - g_push)
        w_push = g_push
        w_sum = w_app + w_con + w_push + eps
        w_app = w_app / w_sum
        w_con = w_con / w_sum
        w_push = w_push / w_sum

        # XY controls per phase
        # Approach: drive to waypoint W laterally
        err_app = (W - tcp)
        err_app_xy = err_app[:, 0:2]
        v_tcp_xy = v_tcp[:, 0:2]
        move_app_xy = self.k_lat_app * err_app_xy - self.b_lat_app * v_tcp_xy

        # Contact: align to object center with slight penetration along -d_hat
        target_con = obj - self.m_pen * d_hat
        err_con = (target_con - tcp)
        err_con_xy = err_con[:, 0:2]
        move_con_xy = self.k_lat_con * err_con_xy - self.b_lat_con * v_tcp_xy

        # Push: along d_hat with side alignment and slip damping
        # Side error: component of (obj - tcp) orthogonal to d_hat
        obj_tcp = obj - tcp
        along_comp = torch.sum(obj_tcp * d_hat, dim=-1, keepdim=True) * d_hat
        side_vec = obj_tcp - along_comp
        side_xy = side_vec[:, 0:2]

        push_mag_ff = self.k_push * torch.tanh(dist_obj_goal)  # bounded by tanh
        slip = rel_vel_along_d
        push_mag = push_mag_ff - self.k_slip * slip
        push_dir_xy = d_hat[:, 0:2] / (torch.norm(d_hat[:, 0:2], dim=-1, keepdim=True) + eps)
        move_push_xy = push_mag * push_dir_xy + self.k_side_push * side_xy - self.b_lat_push * v_tcp_xy

        # Soft speed cap via tanh for each phase XY
        def cap_xy(v_xy):
            mag = torch.norm(v_xy, dim=-1, keepdim=True) + eps
            cap = self.v_cap
            scaled_mag = cap * torch.tanh(mag / (cap + eps))
            return v_xy * (scaled_mag / mag)

        move_app_xy = cap_xy(move_app_xy)
        move_con_xy = cap_xy(move_con_xy)
        move_push_xy = cap_xy(move_push_xy)

        # Z controls per phase
        tcp_z = tcp[:, 2:3]
        v_tcp_z = v_tcp[:, 2:3]
        dz_app = self.kz_app * (self.z_approach - tcp_z) - self.bz_app * v_tcp_z
        dz_con = self.kz_con * (self.z_contact - tcp_z) - self.bz_con * v_tcp_z
        dz_push = self.kz_push * (self.z_contact - tcp_z) - self.bz_push * v_tcp_z

        # Blend phases
        move_xy = w_app * move_app_xy + w_con * move_con_xy + w_push * move_push_xy
        dz = w_app * dz_app + w_con * dz_con + w_push * dz_push

        # Assemble 3D movement
        move3 = torch.cat([move_xy, dz], dim=1)

        # Gripper mean
        grip = self.grip_bias.unsqueeze(0).expand(B, 1)

        mean = torch.cat([move3, grip], dim=1)

        # Std blending (log-stds mixed by phase weights)
        log_std_app = self.log_std_app.unsqueeze(0).expand(B, 4)
        log_std_con = self.log_std_con.unsqueeze(0).expand(B, 4)
        log_std_push = self.log_std_push.unsqueeze(0).expand(B, 4)
        log_std = w_app * log_std_app + w_con * log_std_con + w_push * log_std_push
        std = torch.exp(log_std)

        return (mean, std)

    def get_param_ranges(self):
        return {
            "m_back": (0.01, 0.15),
            "m_pen": (0.0, 0.05),
            "z_approach": (0.05, 0.30),
            "z_contact": (0.0, 0.08),
            "k_lat_app": (0.1, 12.0),
            "b_lat_app": (0.0, 3.0),
            "kz_app": (0.5, 20.0),
            "bz_app": (0.0, 3.0),
            "k_lat_con": (0.1, 15.0),
            "b_lat_con": (0.0, 3.0),
            "kz_con": (0.5, 25.0),
            "bz_con": (0.0, 3.0),
            "k_push": (0.1, 10.0),
            "k_side_push": (0.0, 10.0),
            "k_slip": (0.0, 5.0),
            "b_lat_push": (0.0, 3.0),
            "kz_push": (0.5, 30.0),
            "bz_push": (0.0, 3.0),
            "v_cap": (0.2, 5.0),
            "d_contact": (0.01, 0.15),
            "v_contact": (0.0, 0.05),
            "gate_sharp": (5.0, 200.0),
            "align_sharp": (5.0, 200.0),
            "grip_bias": (-2.0, 2.0),
            "log_std_app": (-5.0, 1.0),
            "log_std_con": (-5.0, 1.0),
            "log_std_push": (-5.0, 1.0),
        }
