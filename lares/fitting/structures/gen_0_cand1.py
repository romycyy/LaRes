# Frozen structure for the Phase 3 fitting comparison. Do not edit by hand:
# regenerate with scripts/build_structure_bank.py.
# provenance: gen_0 candidate 1 of the 2026-08-28 search; reported score 51.46
# ported: raw observation slices rewritten to named accessors; outputs verified
# identical to the original on random observations at two scales.

class GeneratedPolicy(SymbolicPolicy):
    """
    Hypothesis 3:
    Impedance-style controller with a moving virtual target and contact-dependent stiffness and feedforward.
    """
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)

        # Offsets for virtual target
        self.off_back = nn.Parameter(torch.tensor(0.07))     # far/approach offset behind puck
        self.off_pen = nn.Parameter(torch.tensor(0.01))      # slight penetration on contact

        # Impedance gains (base)
        self.Kp_lat = nn.Parameter(torch.tensor(5.0))        # lateral P (perpendicular to d_hat)
        self.Kp_along = nn.Parameter(torch.tensor(3.0))      # along-track P (parallel to d_hat)
        self.Kd = nn.Parameter(torch.tensor(0.6))            # tcp velocity damping

        # Contact modulation factors
        self.contact_boost = nn.Parameter(torch.tensor(2.0)) # boost along stiffness when in contact
        self.lateral_relax = nn.Parameter(torch.tensor(0.5)) # reduce lateral stiffness at contact

        # Feedforward push along d_hat
        self.k_ff = nn.Parameter(torch.tensor(2.5))
        self.k_slip = nn.Parameter(torch.tensor(1.0))        # reduce ff by slip

        # Lateral restoring to line (object->goal)
        self.k_line = nn.Parameter(torch.tensor(2.0))

        # Heights and vertical impedance
        self.z_approach = nn.Parameter(torch.tensor(0.12))
        self.z_contact = nn.Parameter(torch.tensor(0.02))
        self.Kp_z = nn.Parameter(torch.tensor(10.0))
        self.Kd_z = nn.Parameter(torch.tensor(0.7))
        self.slip_down = nn.Parameter(torch.tensor(0.5))     # extra downward with slip

        # Contact gate params
        self.d_contact = nn.Parameter(torch.tensor(0.06))
        self.v_contact = nn.Parameter(torch.tensor(0.01))
        self.sharp = nn.Parameter(torch.tensor(80.0))

        # Gripper
        self.grip_bias = nn.Parameter(torch.tensor(-0.5))

        # Log-std base and modulation by contact and distance-to-go
        self.log_std_base = nn.Parameter(torch.tensor([-1.0, -1.0, -1.0, -1.0]))
        self.log_std_contact_slope = nn.Parameter(torch.tensor([-0.5, -0.5, -0.2, -0.3]))
        self.log_std_dist_slope = nn.Parameter(torch.tensor([0.2, 0.2, 0.1, 0.1]))

        # Speed cap
        self.v_cap = nn.Parameter(torch.tensor(2.5))

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
        dist_obj_goal = torch.norm(g_vec, dim=-1, keepdim=True) + eps
        d_hat = g_vec / dist_obj_goal

        # Contact confidence: proximity and low relative normal velocity
        to_obj = obj - tcp
        dist_to_obj = torch.norm(to_obj, dim=-1, keepdim=True) + eps
        rel_along = torch.sum((v_tcp - v_obj) * d_hat, dim=-1, keepdim=True)
        g_close = torch.sigmoid(self.sharp * (self.d_contact - dist_to_obj))
        g_slow = torch.sigmoid(self.sharp * (self.v_contact - torch.abs(rel_along)))
        c = g_close * g_slow  # contact confidence in [0,1]

        # Virtual target: behind -> slight penetration when in contact
        offset = (1.0 - c) * self.off_back + c * self.off_pen
        V = obj - offset * d_hat

        # Decompose error into along and lateral components
        err = tcp - V  # want tcp -> V so control is -Kp*err
        along_axis = torch.sum(err * d_hat, dim=-1, keepdim=True) * d_hat
        lat_err = err - along_axis

        # Contact-modulated stiffness
        Kp_along_eff = self.Kp_along * (1.0 + c * self.contact_boost)
        Kp_lat_eff = self.Kp_lat * (1.0 - c * (1.0 - self.lateral_relax)).clamp(min=0.0)

        # Feedforward along d_hat with slip reduction
        slip = rel_along
        ff = (self.k_ff * dist_obj_goal - self.k_slip * slip).clamp(min=-5.0, max=5.0)

        # Lateral restoring toward line from puck to goal
        # Side error: position of tcp relative to line through obj along d_hat
        obj_tcp = tcp - obj
        along_comp = torch.sum(obj_tcp * d_hat, dim=-1, keepdim=True) * d_hat
        side_vec = obj_tcp - along_comp

        # XY+Z impedance control
        u_imp = -Kp_lat_eff * lat_err - Kp_along_eff * along_axis - self.Kd * v_tcp + ff * d_hat - self.k_line * side_vec

        # Apply soft speed cap
        def cap_vec(v3):
            mag = torch.norm(v3[:, 0:2], dim=-1, keepdim=True) + eps
            cap = self.v_cap
            scaled_mag = cap * torch.tanh(mag / (cap + eps))
            scale = scaled_mag / mag
            vx = v3[:, 0:2] * scale
            return torch.cat([vx, v3[:, 2:3]], dim=1)

        u_imp = cap_vec(u_imp)

        # Vertical impedance with slip-induced extra downward force at contact
        tcp_z = tcp[:, 2:3]
        v_tcp_z = v_tcp[:, 2:3]
        z_des = (1.0 - c) * self.z_approach + c * self.z_contact
        dz = self.Kp_z * (z_des - tcp_z) - self.Kd_z * v_tcp_z - self.slip_down * torch.relu(slip)  # increase normal force if slipping forward

        # Compose final 3D move: use impedance XY from u_imp and dz for Z
        move3 = torch.cat([u_imp[:, 0:2], dz], dim=1)

        # Gripper
        grip = self.grip_bias.unsqueeze(0).expand(B, 1)

        mean = torch.cat([move3, grip], dim=1)

        # Std modulation: log_std = base + a*c + b*dist_obj_goal
        log_std = (self.log_std_base.unsqueeze(0).expand(B, 4) +
                   c * self.log_std_contact_slope.unsqueeze(0).expand(B, 4) +
                   dist_obj_goal * self.log_std_dist_slope.unsqueeze(0).expand(B, 4))
        std = torch.exp(log_std)

        return (mean, std)

    def get_param_ranges(self):
        return {
            "off_back": (0.02, 0.20),
            "off_pen": (0.0, 0.05),
            "Kp_lat": (0.1, 20.0),
            "Kp_along": (0.1, 20.0),
            "Kd": (0.0, 3.0),
            "contact_boost": (0.0, 5.0),
            "lateral_relax": (0.0, 1.0),
            "k_ff": (0.0, 10.0),
            "k_slip": (0.0, 5.0),
            "k_line": (0.0, 10.0),
            "z_approach": (0.05, 0.30),
            "z_contact": (0.0, 0.08),
            "Kp_z": (0.5, 30.0),
            "Kd_z": (0.0, 3.0),
            "slip_down": (0.0, 3.0),
            "d_contact": (0.01, 0.15),
            "v_contact": (0.0, 0.05),
            "sharp": (5.0, 200.0),
            "grip_bias": (-2.0, 2.0),
            "log_std_base": (-5.0, 1.0),
            "log_std_contact_slope": (-3.0, 3.0),
            "log_std_dist_slope": (-3.0, 3.0),
            "v_cap": (0.2, 5.0),
        }
