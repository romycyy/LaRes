# Frozen structure for the Phase 3 fitting comparison. Do not edit by hand:
# regenerate with scripts/build_structure_bank.py.
# provenance: gen_2 candidate 1 of the 2026-08-28 search; reported score 250.68
# ported: raw observation slices rewritten to named accessors; outputs verified
# identical to the original on random observations at two scales.

class GeneratedPolicy(SymbolicPolicy):
    """
    Hypothesis 5:
    Analytic potential-field with lateral funneling and near-goal disengage.
    """
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)

        # Geometry
        self.r_back = nn.Parameter(torch.tensor(0.07))
        self.h_push = nn.Parameter(torch.tensor(0.02))
        self.z_clear = nn.Parameter(torch.tensor(0.10))

        # Potential field gains
        self.K_att = nn.Parameter(torch.tensor(6.0))    # attraction to waypoint
        self.K_stream = nn.Parameter(torch.tensor(2.0)) # stream along dir
        self.K_lat = nn.Parameter(torch.tensor(3.0))    # lateral funnel on angle
        self.K_damp = nn.Parameter(torch.tensor(0.6))   # viscous damping

        # Gating and disengage
        self.r_near = nn.Parameter(torch.tensor(0.08))  # near waypoint for descent
        self.alpha_near = nn.Parameter(torch.tensor(60.0))
        self.r_goal = nn.Parameter(torch.tensor(0.05))
        self.alpha_goal = nn.Parameter(torch.tensor(80.0))
        self.v_side = nn.Parameter(torch.tensor(0.8))   # lateral disengage speed
        self.dz_up = nn.Parameter(torch.tensor(0.03))   # lift near goal

        # Vertical gains
        self.Kz = nn.Parameter(torch.tensor(12.0))
        self.Dz = nn.Parameter(torch.tensor(0.7))

        # Speed cap
        self.v_cap = nn.Parameter(torch.tensor(2.2))

        # Gripper and noise
        self.grip_bias = nn.Parameter(torch.tensor(-0.6))
        self.log_std_vec = nn.Parameter(torch.tensor([-1.8, -1.8, -2.2, -2.2]))
        self.std_shrink = nn.Parameter(torch.tensor(0.6))  # shrink noise when near and disengaging

    def forward(self, obs):
        B = obs.shape[0]
        eps = 1e-8

        tcp = self.obs_field(obs, "tcp")
        obj = self.obs_field(obs, "obj")
        goal = self.obs_field(obs, "goal")
        prev_tcp = self.obs_field(obs, "prev_tcp")

        v_tcp = tcp - prev_tcp
        v_tcp_xy = v_tcp[:, 0:2]
        v_tcp_z = v_tcp[:, 2:3]

        # Direction and waypoint
        g_xy = (goal - obj)[:, 0:2]
        d_goal = torch.norm(g_xy, dim=-1, keepdim=True) + eps
        dir_xy = g_xy / d_goal
        n_xy = torch.stack([-dir_xy[:, 1], dir_xy[:, 0]], dim=1)

        w_xy = obj[:, 0:2] - self.r_back * dir_xy
        z_way_above = obj[:, 2:3] + self.z_clear
        z_push = obj[:, 2:3] + self.h_push

        # Potential field components
        e_xy = tcp[:, 0:2] - w_xy

        # Attraction gradient: -K_att * (tcp_xy - w_xy)
        grad_att = self.K_att * e_xy

        # Stream term: -grad(-K_stream * (tcp_xy - obj_xy)·dir) = -(-K_stream*dir) = K_stream * dir
        grad_stream = -self.K_stream * dir_xy  # negative gradient of U gives control = -(grad U)
        # But we compute control as -grad(U) below; so keep grad of U:
        gradU_stream = -self.K_stream * dir_xy  # U = -K_stream*(tcp-obj)·dir -> gradU = -K_stream*dir

        # Lateral funnel based on angle error: U_lat = 0.5*K_lat*(cross)^2, cross = (tcp-obj) x dir
        a = tcp[:, 0:2] - obj[:, 0:2]
        cross = a[:, 0:1] * dir_xy[:, 1:2] - a[:, 1:2] * dir_xy[:, 0:1]  # scalar z-comp
        d_cross_da = torch.cat([dir_xy[:, 1:2], -dir_xy[:, 0:1]], dim=1) # grad wrt a
        gradU_lat = self.K_lat * cross * d_cross_da  # grad of 0.5*cross^2

        # Total control in XY: u_xy = -grad(U) - damping
        # gradU_total = grad_att + gradU_stream + gradU_lat
        gradU_total = grad_att + gradU_stream + gradU_lat
        move_xy = -gradU_total - self.K_damp * v_tcp_xy

        # Near-goal disengage: add lateral move, reduce stream via gate
        g_goal = torch.sigmoid(self.alpha_goal * (self.r_goal - d_goal))
        move_xy = (1.0 - g_goal) * move_xy + g_goal * (self.v_side * n_xy - self.K_damp * v_tcp_xy)

        # Soft cap XY
        mag = torch.norm(move_xy, dim=-1, keepdim=True) + eps
        cap = self.v_cap
        scaled = cap * torch.tanh(mag / (cap + eps))
        move_xy = move_xy * (scaled / mag)

        # Vertical: stay high until near waypoint, then descend to push height; lift near goal
        d_way = torch.norm(e_xy, dim=-1, keepdim=True) + eps
        g_near = torch.sigmoid(self.alpha_near * (self.r_near - d_way))
        z_des = (1.0 - g_near) * z_way_above + g_near * z_push + g_goal * self.dz_up
        dz = -self.Kz * (tcp[:, 2:3] - z_des) - self.Dz * v_tcp_z

        move3 = torch.cat([move_xy, dz], dim=1)

        # Gripper
        grip = self.grip_bias.unsqueeze(0).expand(B, 1)

        mean = torch.cat([move3, grip], dim=1)

        # Std: shrink when near waypoint or disengaging
        base_std = torch.exp(self.log_std_vec.unsqueeze(0).expand(B, 4))
        shrink_gate = torch.clamp(g_near + g_goal, 0.0, 1.0)
        shrink = 1.0 - self.std_shrink * shrink_gate
        shrink = torch.clamp(shrink, 0.1, 1.0)
        std = base_std * shrink

        return (mean, std)

    def get_param_ranges(self):
        return {
            "r_back": (0.02, 0.20),
            "h_push": (0.005, 0.06),
            "z_clear": (0.03, 0.25),
            "K_att": (0.1, 20.0),
            "K_stream": (0.0, 8.0),
            "K_lat": (0.0, 20.0),
            "K_damp": (0.0, 3.0),
            "r_near": (0.02, 0.30),
            "alpha_near": (5.0, 200.0),
            "r_goal": (0.01, 0.20),
            "alpha_goal": (5.0, 200.0),
            "v_side": (0.0, 3.0),
            "dz_up": (0.0, 0.06),
            "Kz": (0.5, 30.0),
            "Dz": (0.0, 3.0),
            "v_cap": (0.2, 5.0),
            "grip_bias": (-2.0, 2.0),
            "log_std_vec": (-5.0, 0.0),
            "std_shrink": (0.0, 0.95),
        }
