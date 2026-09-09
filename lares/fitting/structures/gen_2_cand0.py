# Frozen structure for the Phase 3 fitting comparison. Do not edit by hand:
# regenerate with scripts/build_structure_bank.py.
# provenance: gen_2 candidate 0 of the 2026-08-28 search; reported score 702.76
# ported: raw observation slices rewritten to named accessors; outputs verified
# identical to the original on random observations at two scales.

class GeneratedPolicy(SymbolicPolicy):
    """
    Hypothesis 2:
    Impedance-based contact controller with data-driven contact gating and braking.
    """
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)

        # Desired contact geometry
        self.r_contact = nn.Parameter(torch.tensor(0.06))   # behind offset
        self.h_push = nn.Parameter(torch.tensor(0.02))      # push height above object/table

        # Impedance gains
        self.Kp = nn.Parameter(torch.tensor(6.0))
        self.Kd = nn.Parameter(torch.tensor(0.6))

        # Push drive and lateral correction
        self.Kf = nn.Parameter(torch.tensor(2.0))           # feedforward along dir
        self.K_lat = nn.Parameter(torch.tensor(3.0))        # keep centered behind puck

        # Contact likelihood gates
        self.alpha_r = nn.Parameter(torch.tensor(80.0))
        self.r_gate = nn.Parameter(torch.tensor(0.06))      # tcp-object proximity threshold
        self.alpha_v = nn.Parameter(torch.tensor(40.0))
        self.v_thr = nn.Parameter(torch.tensor(0.0))        # min obj vel along dir to confirm contact

        # Brake near goal
        self.alpha_b = nn.Parameter(torch.tensor(80.0))
        self.r_goal = nn.Parameter(torch.tensor(0.05))
        self.z_lift = nn.Parameter(torch.tensor(0.01))

        # Vertical gains
        self.Kz = nn.Parameter(torch.tensor(12.0))
        self.Dz = nn.Parameter(torch.tensor(0.7))

        # Speed cap
        self.v_cap = nn.Parameter(torch.tensor(2.0))

        # Gripper and noise
        self.grip_bias = nn.Parameter(torch.tensor(-0.6))
        self.log_std_vec = nn.Parameter(torch.tensor([-1.8, -1.8, -2.2, -2.2]))
        self.std_shrink = nn.Parameter(torch.tensor(0.6))   # fraction to shrink std during push/brake

    def forward(self, obs):
        B = obs.shape[0]
        eps = 1e-8

        # Parse obs
        tcp = self.obs_field(obs, "tcp")
        obj = self.obs_field(obs, "obj")
        goal = self.obs_field(obs, "goal")
        prev_tcp = self.obs_field(obs, "prev_tcp")
        prev_obj = self.obs_field(obs, "prev_obj")

        v_tcp = tcp - prev_tcp
        v_tcp_xy = v_tcp[:, 0:2]
        v_tcp_z = v_tcp[:, 2:3]
        v_obj = obj - prev_obj
        v_obj_xy = v_obj[:, 0:2]

        # Direction to goal (XY)
        g_xy = (goal - obj)[:, 0:2]
        d_goal = torch.norm(g_xy, dim=-1, keepdim=True) + eps
        dir_xy = g_xy / d_goal

        # Desired contact point behind object
        c_des_xy = obj[:, 0:2] - self.r_contact * dir_xy
        z_des = obj[:, 2:3] + self.h_push
        c_des = torch.cat([c_des_xy, z_des], dim=1)

        # Base impedance (3D)
        u_track = self.Kp * (c_des - tcp) - self.Kd * v_tcp
        u_track_xy = u_track[:, 0:2]
        u_track_z = u_track[:, 2:3]

        # Contact likelihood gate
        r_tcp_obj = torch.norm(tcp[:, 0:2] - obj[:, 0:2], dim=-1, keepdim=True) + eps
        g_r = torch.sigmoid(self.alpha_r * (self.r_gate - r_tcp_obj))  # high when close
        v_along = torch.sum(v_obj_xy * dir_xy, dim=-1, keepdim=True)
        g_v = torch.sigmoid(self.alpha_v * (v_along - self.v_thr))     # high when object moving forward
        g_contact = g_r * g_v

        # Brake gate near goal
        g_brake = torch.sigmoid(self.alpha_b * (self.r_goal - d_goal))

        # Lateral correction to stay centered behind object
        err_xy = tcp[:, 0:2] - c_des_xy
        along = torch.sum(err_xy * dir_xy, dim=-1, keepdim=True) * dir_xy
        side = err_xy - along
        lat_corr = -self.K_lat * side

        # Push drive along dir scaled by contact and reduced by brake
        push_xy = (self.Kf * g_contact * (1.0 - g_brake)) * dir_xy

        # Blend: reduce pure positioning as contact increases
        move_xy = (1.0 - g_contact) * u_track_xy + g_contact * (push_xy + lat_corr - self.Kd * v_tcp_xy)

        # Soft cap XY
        eps_speed = 1e-8
        mag = torch.norm(move_xy, dim=-1, keepdim=True) + eps_speed
        cap = self.v_cap
        scaled = cap * torch.tanh(mag / (cap + eps_speed))
        move_xy = move_xy * (scaled / mag)

        # Vertical: track to push height; lift slightly near goal
        z_target = z_des + self.z_lift * g_brake
        dz = -self.Kz * (tcp[:, 2:3] - z_target) - self.Dz * v_tcp_z

        move3 = torch.cat([move_xy, dz], dim=1)

        # Gripper
        grip = self.grip_bias.unsqueeze(0).expand(B, 1)

        mean = torch.cat([move3, grip], dim=1)

        # Std: shrink with contact/brake
        base_std = torch.exp(self.log_std_vec.unsqueeze(0).expand(B, 4))
        shrink_factor = 1.0 - self.std_shrink * torch.clamp(g_contact + g_brake, 0.0, 1.0)
        shrink_factor = torch.clamp(shrink_factor, 0.1, 1.0)
        std = base_std * shrink_factor

        return (mean, std)

    def get_param_ranges(self):
        return {
            "r_contact": (0.02, 0.20),
            "h_push": (0.005, 0.06),
            "Kp": (0.1, 20.0),
            "Kd": (0.0, 3.0),
            "Kf": (0.0, 8.0),
            "K_lat": (0.0, 20.0),
            "alpha_r": (5.0, 200.0),
            "r_gate": (0.01, 0.20),
            "alpha_v": (5.0, 200.0),
            "v_thr": (-0.5, 0.5),
            "alpha_b": (5.0, 200.0),
            "r_goal": (0.01, 0.20),
            "z_lift": (0.0, 0.05),
            "Kz": (0.5, 30.0),
            "Dz": (0.0, 3.0),
            "v_cap": (0.2, 5.0),
            "grip_bias": (-2.0, 2.0),
            "log_std_vec": (-5.0, 0.0),
            "std_shrink": (0.0, 0.95),
        }
