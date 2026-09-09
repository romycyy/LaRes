# Frozen structure for the Phase 3 fitting comparison. Do not edit by hand:
# regenerate with scripts/build_structure_bank.py.
# provenance: gen_0 candidate 4 of the 2026-08-28 search; reported score 10.77
# ported: raw observation slices rewritten to named accessors; outputs verified
# identical to the original on random observations at two scales.

class GeneratedPolicy(SymbolicPolicy):
    """
    Hypothesis 2:
    Orbit-to-back alignment field followed by regulated push using slip feedback, mixed via a softmax over orbit/align/push scores.
    """
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)

        # Swirl/orbit parameters
        self.k_swirl = nn.Parameter(torch.tensor(2.0))       # swirl strength
        self.r_orbit = nn.Parameter(torch.tensor(0.10))      # desired orbit radius
        self.decay_len = nn.Parameter(torch.tensor(0.20))    # distance decay for swirl

        # Approach/align/push gains
        self.k_approach = nn.Parameter(torch.tensor(4.0))    # approach to behind waypoint
        self.k_align = nn.Parameter(torch.tensor(3.0))       # alignment gain near object
        self.k_push = nn.Parameter(torch.tensor(3.0))        # push along d_hat
        self.k_slip = nn.Parameter(torch.tensor(1.0))        # slip feedback along d_hat
        self.k_side = nn.Parameter(torch.tensor(2.0))        # side alignment during push
        self.b_xy = nn.Parameter(torch.tensor(0.5))          # lateral damping

        # Waypoint and height schedule
        self.m_back = nn.Parameter(torch.tensor(0.06))       # behind offset
        self.z_orbit = nn.Parameter(torch.tensor(0.14))      # high during orbit
        self.z_contact = nn.Parameter(torch.tensor(0.02))    # contact

        # Speed cap
        self.v_cap = nn.Parameter(torch.tensor(2.0))

        # Softmax scoring parameters
        self.s_orbit_scale = nn.Parameter(torch.tensor(1.0))
        self.s_align_scale = nn.Parameter(torch.tensor(2.0))
        self.s_push_scale = nn.Parameter(torch.tensor(3.0))
        self.score_bias_orbit = nn.Parameter(torch.tensor(0.0))
        self.score_bias_align = nn.Parameter(torch.tensor(0.0))
        self.score_bias_push = nn.Parameter(torch.tensor(0.0))

        # Contact/proximity gates
        self.d_contact = nn.Parameter(torch.tensor(0.06))
        self.gate_sharp = nn.Parameter(torch.tensor(80.0))

        # Gripper closed bias
        self.grip_bias = nn.Parameter(torch.tensor(-0.5))

        # Per-phase log-stds
        self.log_std_orbit = nn.Parameter(torch.tensor([-0.6, -0.6, -0.7, -0.5]))
        self.log_std_align = nn.Parameter(torch.tensor([-1.0, -1.0, -0.9, -1.0]))
        self.log_std_push = nn.Parameter(torch.tensor([-1.5, -1.5, -1.1, -1.2]))

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
        to_obj = obj - tcp
        dist_to_obj = torch.norm(to_obj, dim=-1, keepdim=True) + eps
        u_hat = to_obj / dist_to_obj

        to_goal_from_obj = goal - obj
        dist_obj_goal = torch.norm(to_goal_from_obj, dim=-1, keepdim=True) + eps
        d_hat = to_goal_from_obj / dist_obj_goal

        # 2D helpers
        u_hat_xy = u_hat[:, 0:2]
        d_hat_xy = d_hat[:, 0:2]
        d_hat_xy = d_hat_xy / (torch.norm(d_hat_xy, dim=-1, keepdim=True) + eps)

        # Swirl direction: 90-deg rotation of u_hat_xy (clockwise)
        swirl_dir_xy = torch.stack([-u_hat_xy[:, 1], u_hat_xy[:, 0]], dim=1)
        swirl_dir_xy = swirl_dir_xy / (torch.norm(swirl_dir_xy, dim=-1, keepdim=True) + eps)

        # Orbit distance modulation toward desired radius
        r_err = (dist_to_obj - self.r_orbit).clamp(min=-1e3, max=1e3)  # (B,1)
        swirl_strength = self.k_swirl * torch.exp(-dist_to_obj / (self.decay_len + eps))
        move_orbit_xy = swirl_strength * swirl_dir_xy + self.k_approach * r_err * u_hat_xy - self.b_xy * (v_tcp[:, 0:2])

        # Align: go to behind waypoint
        W = obj - self.m_back * d_hat
        err_align_xy = (W - tcp)[:, 0:2]
        move_align_xy = self.k_align * err_align_xy - self.b_xy * (v_tcp[:, 0:2])

        # Push: along d_hat plus side alignment with slip feedback
        obj_tcp = obj - tcp
        along_comp = torch.sum(obj_tcp * d_hat, dim=-1, keepdim=True) * d_hat
        side_vec = obj_tcp - along_comp
        side_xy = side_vec[:, 0:2]
        rel_along = torch.sum((v_tcp - v_obj) * d_hat, dim=-1, keepdim=True)
        push_mag = self.k_push * torch.tanh(dist_obj_goal) - self.k_slip * rel_along
        move_push_xy = push_mag * d_hat_xy + self.k_side * side_xy - self.b_xy * (v_tcp[:, 0:2])

        # Soft speed cap
        def cap_xy(v_xy):
            mag = torch.norm(v_xy, dim=-1, keepdim=True) + eps
            cap = self.v_cap
            scaled_mag = cap * torch.tanh(mag / (cap + eps))
            return v_xy * (scaled_mag / mag)

        move_orbit_xy = cap_xy(move_orbit_xy)
        move_align_xy = cap_xy(move_align_xy)
        move_push_xy = cap_xy(move_push_xy)

        # Scores for phases
        misalign = 1.0 - torch.sum(u_hat_xy * (-d_hat_xy), dim=-1, keepdim=True)  # in [0,2]
        prox = torch.sigmoid(self.gate_sharp * (self.d_contact - dist_to_obj))    # near object
        contact_conf = prox

        s_orbit = self.s_orbit_scale * misalign + self.score_bias_orbit
        s_align = self.s_align_scale * (1.0 - misalign) + 0.5 * (1.0 - contact_conf) + self.score_bias_align
        s_push = self.s_push_scale * contact_conf + self.score_bias_push
        scores = torch.cat([s_orbit, s_align, s_push], dim=1)
        weights = torch.softmax(scores, dim=1)  # (B,3)

        w_orbit = weights[:, 0:1]
        w_align = weights[:, 1:2]
        w_push = weights[:, 2:3]

        # Z control: high during orbit, low at contact
        tcp_z = tcp[:, 2:3]
        v_tcp_z = v_tcp[:, 2:3]
        z_des = w_orbit * self.z_orbit + (w_align + w_push) * self.z_contact
        dz = 8.0 * (z_des - tcp_z) - 0.6 * v_tcp_z  # fixed impedance for simplicity

        # Blend XY
        move_xy = w_orbit * move_orbit_xy + w_align * move_align_xy + w_push * move_push_xy

        # 3D movement
        move3 = torch.cat([move_xy, dz], dim=1)

        # Gripper mean
        grip = self.grip_bias.unsqueeze(0).expand(B, 1)

        mean = torch.cat([move3, grip], dim=1)

        # Std: mix per-phase log-stds with the same softmax weights
        log_std_orbit = self.log_std_orbit.unsqueeze(0).expand(B, 4)
        log_std_align = self.log_std_align.unsqueeze(0).expand(B, 4)
        log_std_push = self.log_std_push.unsqueeze(0).expand(B, 4)
        log_std = w_orbit * log_std_orbit + w_align * log_std_align + w_push * log_std_push
        std = torch.exp(log_std)

        return (mean, std)

    def get_param_ranges(self):
        return {
            "k_swirl": (0.0, 10.0),
            "r_orbit": (0.03, 0.30),
            "decay_len": (0.05, 0.60),
            "k_approach": (0.1, 12.0),
            "k_align": (0.1, 12.0),
            "k_push": (0.1, 12.0),
            "k_slip": (0.0, 5.0),
            "k_side": (0.0, 10.0),
            "b_xy": (0.0, 3.0),
            "m_back": (0.01, 0.20),
            "z_orbit": (0.05, 0.30),
            "z_contact": (0.0, 0.08),
            "v_cap": (0.2, 5.0),
            "s_orbit_scale": (0.1, 10.0),
            "s_align_scale": (0.1, 10.0),
            "s_push_scale": (0.1, 10.0),
            "score_bias_orbit": (-3.0, 3.0),
            "score_bias_align": (-3.0, 3.0),
            "score_bias_push": (-3.0, 3.0),
            "d_contact": (0.01, 0.15),
            "gate_sharp": (5.0, 200.0),
            "grip_bias": (-2.0, 2.0),
            "log_std_orbit": (-5.0, 1.0),
            "log_std_align": (-5.0, 1.0),
            "log_std_push": (-5.0, 1.0),
        }
