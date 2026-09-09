# Frozen structure for the Phase 3 fitting comparison. Do not edit by hand:
# regenerate with scripts/build_structure_bank.py.
# provenance: gen_0 candidate 2 of the 2026-08-28 search; reported score 43.42
# ported: raw observation slices rewritten to named accessors; outputs verified
# identical to the original on random observations at two scales.

class GeneratedPolicy(SymbolicPolicy):
    """
    Hypothesis 5:
    Progress-scheduled pushing with learned speed and offset/height schedules and slip feedback.
    """
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)

        # Progress scaling and shaping
        self.dist_scale = nn.Parameter(torch.tensor(0.50))     # meters to scale distance for s
        self.s_mid = nn.Parameter(torch.tensor(0.5))           # mid-point in progress
        self.s_sharp = nn.Parameter(torch.tensor(10.0))        # sharpness of scheduler

        # Behind offset schedule: offset(s) = lerp(off0, off1, curve(s))
        self.off0 = nn.Parameter(torch.tensor(0.08))           # start standoff
        self.off1 = nn.Parameter(torch.tensor(0.01))           # end penetration

        # Speed profile along d_hat: k_along(s) = v0*(1-c) + v1*c with optional mid boost
        self.v0 = nn.Parameter(torch.tensor(1.5))
        self.v1 = nn.Parameter(torch.tensor(3.0))
        self.v_mid = nn.Parameter(torch.tensor(0.5))           # additional bump near mid
        self.mid_width = nn.Parameter(torch.tensor(0.2))       # width of mid bump

        # Tracking and side alignment
        self.k_track = nn.Parameter(torch.tensor(5.0))         # track C(s)
        self.k_side = nn.Parameter(torch.tensor(2.0))          # side alignment during push
        self.k_slip = nn.Parameter(torch.tensor(1.0))          # slip feedback along d_hat
        self.b_xy = nn.Parameter(torch.tensor(0.6))            # damping

        # Height schedule z(s) and slip-induced compression
        self.z_hi = nn.Parameter(torch.tensor(0.12))
        self.z_lo = nn.Parameter(torch.tensor(0.02))
        self.kz = nn.Parameter(torch.tensor(10.0))
        self.bz = nn.Parameter(torch.tensor(0.6))
        self.slip_down = nn.Parameter(torch.tensor(0.5))

        # Speed cap
        self.v_cap = nn.Parameter(torch.tensor(2.5))

        # Gripper
        self.grip_bias = nn.Parameter(torch.tensor(-0.5))

        # Std coefficients: log_std = base + slope * s
        self.log_std_base = nn.Parameter(torch.tensor([-0.9, -0.9, -0.8, -0.7]))
        self.log_std_slope = nn.Parameter(torch.tensor([-0.8, -0.8, -0.5, -0.5]))

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

        # Geometry
        g_vec = goal - obj
        d_goal = torch.norm(g_vec, dim=-1, keepdim=True) + eps
        d_hat = g_vec / d_goal

        # Progress s in [0,1) via exponential schedule with learnable scale
        L = torch.clamp(self.dist_scale, min=0.05, max=2.0)
        s_raw = 1.0 - torch.exp(-d_goal / L)
        # Smooth curve around s_mid
        s_curve = torch.sigmoid(self.s_sharp * (s_raw - self.s_mid))

        # Offset schedule and desired contact point C(s)
        offset = self.off0 * (1.0 - s_curve) + self.off1 * s_curve
        C = obj - offset * d_hat

        # Speed profile k_along(s) with a mid bump
        mid_bump = self.v_mid * torch.exp(-0.5 * ((s_raw - self.s_mid) / (self.mid_width + eps))**2)
        k_along = self.v0 * (1.0 - s_curve) + self.v1 * s_curve + mid_bump

        # Slip feedback along d_hat
        rel_along = torch.sum((v_tcp - v_obj) * d_hat, dim=-1, keepdim=True)

        # Lateral tracking to C(s) and along-track push
        err = C - tcp
        err_xy = err[:, 0:2]
        d_hat_xy = d_hat[:, 0:2] / (torch.norm(d_hat[:, 0:2], dim=-1, keepdim=True) + eps)

        # Side alignment: remove along component from (obj - tcp)
        obj_tcp = obj - tcp
        along_comp = torch.sum(obj_tcp * d_hat, dim=-1, keepdim=True) * d_hat
        side_vec = obj_tcp - along_comp
        side_xy = side_vec[:, 0:2]

        move_xy = self.k_track * err_xy + (k_along - self.k_slip * rel_along) * d_hat_xy + self.k_side * side_xy - self.b_xy * v_tcp[:, 0:2]

        # Soft speed cap
        def cap_xy(v_xy):
            mag = torch.norm(v_xy, dim=-1, keepdim=True) + eps
            cap = self.v_cap
            scaled_mag = cap * torch.tanh(mag / (cap + eps))
            return v_xy * (scaled_mag / mag)

        move_xy = cap_xy(move_xy)

        # Height schedule with slip-induced downward adjustment near s≈1
        z_des = self.z_hi * (1.0 - s_curve) + self.z_lo * s_curve
        tcp_z = tcp[:, 2:3]
        v_tcp_z = v_tcp[:, 2:3]
        dz = self.kz * (z_des - tcp_z) - self.bz * v_tcp_z - self.slip_down * torch.relu(rel_along)

        move3 = torch.cat([move_xy, dz], dim=1)

        # Gripper
        grip = self.grip_bias.unsqueeze(0).expand(B, 1)

        mean = torch.cat([move3, grip], dim=1)

        # Std schedule decreasing with progress
        log_std = self.log_std_base.unsqueeze(0).expand(B, 4) + s_raw * self.log_std_slope.unsqueeze(0).expand(B, 4)
        std = torch.exp(log_std)

        return (mean, std)

    def get_param_ranges(self):
        return {
            "dist_scale": (0.05, 2.0),
            "s_mid": (0.1, 0.9),
            "s_sharp": (1.0, 50.0),
            "off0": (0.02, 0.20),
            "off1": (0.0, 0.05),
            "v0": (0.0, 5.0),
            "v1": (0.0, 8.0),
            "v_mid": (0.0, 3.0),
            "mid_width": (0.05, 0.6),
            "k_track": (0.1, 15.0),
            "k_side": (0.0, 10.0),
            "k_slip": (0.0, 5.0),
            "b_xy": (0.0, 3.0),
            "z_hi": (0.05, 0.30),
            "z_lo": (0.0, 0.08),
            "kz": (0.5, 30.0),
            "bz": (0.0, 3.0),
            "slip_down": (0.0, 3.0),
            "v_cap": (0.2, 5.0),
            "grip_bias": (-2.0, 2.0),
            "log_std_base": (-5.0, 1.0),
            "log_std_slope": (-3.0, 3.0),
        }
