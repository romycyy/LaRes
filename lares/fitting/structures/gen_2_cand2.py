# Frozen structure for the Phase 3 fitting comparison. Do not edit by hand:
# regenerate with scripts/build_structure_bank.py.
# provenance: gen_2 candidate 2 of the 2026-08-28 search; reported score 155.50
# ported: raw observation slices rewritten to named accessors; outputs verified
# identical to the original on random observations at two scales.

class GeneratedPolicy(SymbolicPolicy):
    """
    Hypothesis 3:
    Object-centric continuous vector field without explicit phases; blended tracking and push.
    """
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)

        # Geometry
        self.r_back = nn.Parameter(torch.tensor(0.07))
        self.h_push = nn.Parameter(torch.tensor(0.02))
        self.n_bias = nn.Parameter(torch.tensor(0.0))      # lateral bias magnitude

        # Tracking and push gains
        self.Kt = nn.Parameter(torch.tensor(6.0))
        self.Dt = nn.Parameter(torch.tensor(0.6))
        self.Kp_push = nn.Parameter(torch.tensor(2.0))
        self.K_lat = nn.Parameter(torch.tensor(3.0))
        self.Kz = nn.Parameter(torch.tensor(12.0))
        self.Dz = nn.Parameter(torch.tensor(0.7))

        # Push weighting gates
        self.r_s = nn.Parameter(torch.tensor(0.08))        # near setpoint threshold
        self.r_c = nn.Parameter(torch.tensor(0.06))        # near object threshold
        self.cos_thr = nn.Parameter(torch.tensor(0.5))     # alignment threshold (cosine)
        self.alpha_s = nn.Parameter(torch.tensor(60.0))
        self.alpha_c = nn.Parameter(torch.tensor(80.0))
        self.alpha_a = nn.Parameter(torch.tensor(60.0))

        # Goal lift
        self.r_goal = nn.Parameter(torch.tensor(0.05))
        self.alpha_g = nn.Parameter(torch.tensor(80.0))
        self.z_up = nn.Parameter(torch.tensor(0.02))

        # Speed cap
        self.v_cap = nn.Parameter(torch.tensor(2.0))

        # Gripper and noise
        self.grip_bias = nn.Parameter(torch.tensor(-0.6))
        self.log_std_vec = nn.Parameter(torch.tensor([-1.9, -1.9, -2.2, -2.2]))
        self.std_alpha = nn.Parameter(torch.tensor(2.0))   # std shrink sharpness with push weight

    def forward(self, obs):
        B = obs.shape[0]
        eps = 1e-8

        tcp = self.obs_field(obs, "tcp")
        obj = self.obs_field(obs, "obj")
        goal = self.obs_field(obs, "goal")
        prev_tcp = self.obs_field(obs, "prev_tcp")
        prev_obj = self.obs_field(obs, "prev_obj")

        v_tcp = tcp - prev_tcp
        v_tcp_xy = v_tcp[:, 0:2]
        v_tcp_z = v_tcp[:, 2:3]

        # Direction and normal in XY
        g_xy = (goal - obj)[:, 0:2]
        d_goal = torch.norm(g_xy, dim=-1, keepdim=True) + eps
        dir_xy = g_xy / d_goal
        n_xy = torch.stack([-dir_xy[:, 1], dir_xy[:, 0]], dim=1)

        # Setpoint behind object with lateral bias and push height
        s_xy = obj[:, 0:2] - self.r_back * dir_xy + self.n_bias * n_xy
        s_z = obj[:, 2:3] + self.h_push
        s = torch.cat([s_xy, s_z], dim=1)

        # Tracking control to setpoint
        e = s - tcp
        u_track_xy = self.Kt * e[:, 0:2] - self.Dt * v_tcp_xy
        u_track_z = self.Kz * e[:, 2:3] - self.Dz * v_tcp_z

        # Push drive along dir proportional to remaining distance (clipped)
        u_push_xy = self.Kp_push * dir_xy

        # Lateral correction toward anchor along normal (cross-track)
        anchor_xy = obj[:, 0:2] - self.r_back * dir_xy
        err_xy = tcp[:, 0:2] - anchor_xy
        along = torch.sum(err_xy * dir_xy, dim=-1, keepdim=True) * dir_xy
        side = err_xy - along
        lat_corr = -self.K_lat * side

        # Gates for push weighting
        d_set = torch.norm(tcp[:, 0:2] - s_xy, dim=-1, keepdim=True) + eps
        g_set = torch.sigmoid(self.alpha_s * (self.r_s - d_set))       # near setpoint
        d_obj = torch.norm(tcp[:, 0:2] - obj[:, 0:2], dim=-1, keepdim=True) + eps
        g_obj = torch.sigmoid(self.alpha_c * (self.r_c - d_obj))       # near object
        to_obj = (obj[:, 0:2] - tcp[:, 0:2])
        to_obj = to_obj / (torch.norm(to_obj, dim=-1, keepdim=True) + eps)
        align = torch.sum(to_obj * dir_xy, dim=-1, keepdim=True)
        g_align = torch.sigmoid(self.alpha_a * (align - self.cos_thr)) # aligned
        w_push = torch.clamp(g_set * g_obj * g_align, 0.0, 1.0)
        w_track = 1.0 - w_push

        # Combine planar controls
        move_xy = w_track * u_track_xy + w_push * (u_push_xy + lat_corr - self.Dt * v_tcp_xy)

        # Soft cap XY
        mag = torch.norm(move_xy, dim=-1, keepdim=True) + eps
        scaled = self.v_cap * torch.tanh(mag / (self.v_cap + eps))
        move_xy = move_xy * (scaled / mag)

        # Vertical: track to s_z; add lift near goal
        g_goal = torch.sigmoid(self.alpha_g * (self.r_goal - d_goal))
        z_target = s_z + self.z_up * g_goal
        dz = -self.Kz * (tcp[:, 2:3] - z_target) - self.Dz * v_tcp_z

        move3 = torch.cat([move_xy, dz], dim=1)

        # Gripper
        grip = self.grip_bias.unsqueeze(0).expand(B, 1)

        mean = torch.cat([move3, grip], dim=1)

        # Std shrinks as w_push increases
        base_log_std = self.log_std_vec.unsqueeze(0).expand(B, 4)
        shrink = torch.pow(1.0 - w_push, self.std_alpha) + 1e-3
        shrink = 0.3 + 0.7 * shrink
        std = torch.exp(base_log_std) * shrink

        return (mean, std)

    def get_param_ranges(self):
        return {
            "r_back": (0.02, 0.20),
            "h_push": (0.005, 0.06),
            "n_bias": (-0.05, 0.05),
            "Kt": (0.1, 20.0),
            "Dt": (0.0, 3.0),
            "Kp_push": (0.0, 8.0),
            "K_lat": (0.0, 20.0),
            "Kz": (0.5, 30.0),
            "Dz": (0.0, 3.0),
            "r_s": (0.02, 0.20),
            "r_c": (0.01, 0.15),
            "cos_thr": (-1.0, 1.0),
            "alpha_s": (5.0, 200.0),
            "alpha_c": (5.0, 200.0),
            "alpha_a": (5.0, 200.0),
            "r_goal": (0.01, 0.20),
            "alpha_g": (5.0, 200.0),
            "z_up": (0.0, 0.05),
            "v_cap": (0.2, 5.0),
            "grip_bias": (-2.0, 2.0),
            "log_std_vec": (-5.0, 0.0),
            "std_alpha": (0.5, 10.0),
        }
