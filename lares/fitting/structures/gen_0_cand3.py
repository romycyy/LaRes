# Frozen structure for the Phase 3 fitting comparison. Do not edit by hand:
# regenerate with scripts/build_structure_bank.py.
# provenance: gen_0 candidate 3 of the 2026-08-28 search; reported score 14.59
# ported: raw observation slices rewritten to named accessors; outputs verified
# identical to the original on random observations at two scales.

class GeneratedPolicy(SymbolicPolicy):
    """
    Hypothesis 4:
    Potential-field composition with attraction to behind region, push funnel, and front-side barrier; pre/contact blending via proximity gate.
    """
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)

        # Length scales and weights
        self.setback = nn.Parameter(torch.tensor(0.06))       # behind setback
        self.lat_width = nn.Parameter(torch.tensor(0.08))     # lateral width scale
        self.k_back = nn.Parameter(torch.tensor(6.0))         # weight for behind attraction
        self.k_push = nn.Parameter(torch.tensor(3.0))         # push attraction
        self.k_bar = nn.Parameter(torch.tensor(4.0))          # repulsive front barrier
        self.bar_sharp = nn.Parameter(torch.tensor(40.0))     # barrier sharpness

        # Contact coupling gain once near object
        self.k_couple = nn.Parameter(torch.tensor(3.0))
        self.k_slip = nn.Parameter(torch.tensor(1.0))
        self.b_xy = nn.Parameter(torch.tensor(0.6))           # damping

        # Z potential parameters
        self.z_approach = nn.Parameter(torch.tensor(0.12))
        self.z_contact = nn.Parameter(torch.tensor(0.02))
        self.kz = nn.Parameter(torch.tensor(10.0))
        self.bz = nn.Parameter(torch.tensor(0.6))

        # Proximity gate
        self.d_contact = nn.Parameter(torch.tensor(0.06))
        self.gate_sharp = nn.Parameter(torch.tensor(80.0))

        # Speed cap
        self.v_cap = nn.Parameter(torch.tensor(2.0))

        # Gripper
        self.grip_bias = nn.Parameter(torch.tensor(-0.5))

        # Log-stds pre-contact and contact
        self.log_std_pre = nn.Parameter(torch.tensor([-0.8, -0.8, -0.7, -0.6]))
        self.log_std_con = nn.Parameter(torch.tensor([-1.4, -1.4, -1.0, -1.2]))

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
        dist_obj_goal = torch.norm(g_vec, dim=-1, keepdim=True) + eps
        d_hat = g_vec / dist_obj_goal

        # Pre-contact gate
        to_obj = obj - tcp
        dist_to_obj = torch.norm(to_obj, dim=-1, keepdim=True) + eps
        g_prox = torch.sigmoid(self.gate_sharp * (self.d_contact - dist_to_obj))

        # Potential gradients (XY)
        # Behind attraction: toward B = obj - setback*d_hat with lateral width shaping
        B_pt = obj - self.setback * d_hat
        err_back = (tcp - B_pt)
        # Lateral component relative to d_hat
        along = torch.sum(err_back * d_hat, dim=-1, keepdim=True) * d_hat
        lat = err_back - along
        grad_U_back = self.k_back * (along + (lat / (self.lat_width + eps)**2))  # gradient wrt tcp

        # Push attraction: encourage movement along +d_hat (negative gradient gives push -k_push*d_hat)
        grad_U_push = -self.k_push * d_hat

        # Front barrier: penalize being in front (tcp ahead of obj along d_hat)
        s_front = torch.sum((tcp - obj) * d_hat, dim=-1, keepdim=True)
        # Smooth barrier factor in (0,1) increasing with s_front
        barrier_factor = torch.sigmoid(self.bar_sharp * s_front)
        grad_U_front = self.k_bar * barrier_factor * d_hat  # push back along -d_hat when in front

        # Combined field pre-contact (negative gradient for movement)
        grad_total_pre = grad_U_back + grad_U_push + grad_U_front
        move_pre_xy = -(grad_total_pre)[:, 0:2] - self.b_xy * v_tcp[:, 0:2]

        # Contact coupling: stronger push proportional to (puck-goal) and couple tcp to object
        rel_along = torch.sum((v_tcp - v_obj) * d_hat, dim=-1, keepdim=True)
        couple = self.k_couple * dist_obj_goal * d_hat - self.k_slip * rel_along * d_hat
        # Also keep lateral alignment with object line
        obj_tcp = tcp - obj
        along_obj = torch.sum(obj_tcp * d_hat, dim=-1, keepdim=True) * d_hat
        side = obj_tcp - along_obj
        move_con_xy = (couple - self.b_xy * v_tcp - self.k_couple * side)[:, 0:2]

        # Soft speed cap
        def cap_xy(v_xy):
            mag = torch.norm(v_xy, dim=-1, keepdim=True) + eps
            cap = self.v_cap
            scaled_mag = cap * torch.tanh(mag / (cap + eps))
            return v_xy * (scaled_mag / mag)

        move_pre_xy = cap_xy(move_pre_xy)
        move_con_xy = cap_xy(move_con_xy)

        # Z control via bimodal potential selected by proximity
        tcp_z = tcp[:, 2:3]
        v_tcp_z = v_tcp[:, 2:3]
        z_des = (1.0 - g_prox) * self.z_approach + g_prox * self.z_contact
        dz = self.kz * (z_des - tcp_z) - self.bz * v_tcp_z

        # Blend pre-contact and contact lateral actions
        move_xy = (1.0 - g_prox) * move_pre_xy + g_prox * move_con_xy
        move3 = torch.cat([move_xy, dz], dim=1)

        # Gripper
        grip = self.grip_bias.unsqueeze(0).expand(B, 1)

        mean = torch.cat([move3, grip], dim=1)

        # Std blending
        log_std_pre = self.log_std_pre.unsqueeze(0).expand(B, 4)
        log_std_con = self.log_std_con.unsqueeze(0).expand(B, 4)
        log_std = (1.0 - g_prox) * log_std_pre + g_prox * log_std_con
        std = torch.exp(log_std)

        return (mean, std)

    def get_param_ranges(self):
        return {
            "setback": (0.01, 0.20),
            "lat_width": (0.02, 0.30),
            "k_back": (0.1, 20.0),
            "k_push": (0.1, 15.0),
            "k_bar": (0.0, 20.0),
            "bar_sharp": (5.0, 200.0),
            "k_couple": (0.0, 15.0),
            "k_slip": (0.0, 5.0),
            "b_xy": (0.0, 3.0),
            "z_approach": (0.05, 0.30),
            "z_contact": (0.0, 0.08),
            "kz": (0.5, 30.0),
            "bz": (0.0, 3.0),
            "d_contact": (0.01, 0.15),
            "gate_sharp": (5.0, 200.0),
            "v_cap": (0.2, 5.0),
            "grip_bias": (-2.0, 2.0),
            "log_std_pre": (-5.0, 1.0),
            "log_std_con": (-5.0, 1.0),
        }
