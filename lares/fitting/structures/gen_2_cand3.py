# Frozen structure for the Phase 3 fitting comparison. Do not edit by hand:
# regenerate with scripts/build_structure_bank.py.
# provenance: gen_2 candidate 3 of the 2026-08-28 search; reported score 126.29
# ported: raw observation slices rewritten to named accessors; outputs verified
# identical to the original on random observations at two scales.

class GeneratedPolicy(SymbolicPolicy):
    """
    Hypothesis 1:
    Three-phase, direction-aware push with smooth gating and cross-track correction.
    """
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)

        # Geometry
        self.r_back = nn.Parameter(torch.tensor(0.07))     # behind offset [m]
        self.z_clear = nn.Parameter(torch.tensor(0.10))    # clearance above object for approach
        self.h_push = nn.Parameter(torch.tensor(0.02))     # push height above table/object z

        # Gains
        self.K_app = nn.Parameter(torch.tensor(6.0))       # approach XY P gain
        self.D_app = nn.Parameter(torch.tensor(0.5))       # approach XY damping
        self.K_align = nn.Parameter(torch.tensor(2.0))     # descend XY small alignment
        self.K_long = nn.Parameter(torch.tensor(2.0))      # forward push drive (along dir)
        self.K_lat = nn.Parameter(torch.tensor(3.0))       # lateral funnel gain
        self.Kz = nn.Parameter(torch.tensor(12.0))         # vertical P gain
        self.Dz = nn.Parameter(torch.tensor(0.7))          # vertical D
        self.D_xy = nn.Parameter(torch.tensor(0.6))        # general XY damping

        # Gates and thresholds
        self.r_stage = nn.Parameter(torch.tensor(0.10))    # distance to stage to switch from approach
        self.r_contact = nn.Parameter(torch.tensor(0.06))  # tcp-object XY distance for contact
        self.align_thr = nn.Parameter(torch.tensor(0.4))   # dot threshold for push alignment (cosine)
        self.alpha_stage = nn.Parameter(torch.tensor(60.0))
        self.alpha_contact = nn.Parameter(torch.tensor(80.0))
        self.alpha_align = nn.Parameter(torch.tensor(60.0))

        # Speed cap
        self.v_cap = nn.Parameter(torch.tensor(2.0))

        # Gripper control (open/close biases blended by phase)
        self.grip_open = nn.Parameter(torch.tensor(0.2))
        self.grip_close = nn.Parameter(torch.tensor(-0.6))

        # Std control
        self.log_std_vec = nn.Parameter(torch.tensor([-1.8, -1.8, -2.2, -2.2]))
        self.alpha_std = nn.Parameter(torch.tensor(2.0))   # sharpness to shrink std during push

    def forward(self, obs):
        B = obs.shape[0]
        eps = 1e-8

        # Parse observations
        tcp = self.obs_field(obs, "tcp")
        obj = self.obs_field(obs, "obj")
        goal = self.obs_field(obs, "goal")
        prev_tcp = self.obs_field(obs, "prev_tcp")
        prev_obj = self.obs_field(obs, "prev_obj")

        # Velocities
        v_tcp = tcp - prev_tcp
        v_tcp_xy = v_tcp[:, 0:2]
        v_tcp_z = v_tcp[:, 2:3]

        # Geometry in XY
        g_vec = goal - obj
        g_xy = g_vec[:, 0:2]
        d_goal = torch.norm(g_xy, dim=-1, keepdim=True) + eps
        dir_xy = g_xy / d_goal

        # Stage point behind object with clearance
        anchor_xy = obj[:, 0:2] - self.r_back * dir_xy
        z_stage = obj[:, 2:3] + self.z_clear
        z_push = obj[:, 2:3] + self.h_push

        # Distances and alignments
        d_stage = torch.norm(tcp[:, 0:2] - anchor_xy, dim=-1, keepdim=True) + eps
        d_contact = torch.norm(tcp[:, 0:2] - obj[:, 0:2], dim=-1, keepdim=True) + eps
        to_obj_xy = (obj[:, 0:2] - tcp[:, 0:2])
        to_obj_xy = to_obj_xy / (torch.norm(to_obj_xy, dim=-1, keepdim=True) + eps)
        align = torch.sum(to_obj_xy * dir_xy, dim=-1, keepdim=True)

        # Gates
        g_close_stage = torch.sigmoid(self.alpha_stage * (self.r_stage - d_stage))   # near anchor
        g_contact = torch.sigmoid(self.alpha_contact * (self.r_contact - d_contact)) # near object
        g_align = torch.sigmoid(self.alpha_align * (align - self.align_thr))         # aligned to push

        # Phase raw weights
        w_push_raw = g_contact * g_align
        w_desc_raw = (1.0 - g_contact) * g_close_stage
        w_app_raw = 1.0 - g_close_stage

        # Normalize phase weights
        w_sum = w_push_raw + w_desc_raw + w_app_raw + eps
        w_push = w_push_raw / w_sum
        w_desc = w_desc_raw / w_sum
        w_app = w_app_raw / w_sum

        # Phase commands
        # Approach: move XY to anchor, keep z at z_stage
        e_app_xy = tcp[:, 0:2] - anchor_xy
        u_app_xy = -self.K_app * e_app_xy - self.D_app * v_tcp_xy
        u_app_z = -self.Kz * (tcp[:, 2:3] - z_stage) - self.Dz * v_tcp_z

        # Descend: small XY alignment to anchor, descend to z_push
        e_desc_xy = tcp[:, 0:2] - anchor_xy
        u_desc_xy = -self.K_align * e_desc_xy - self.D_app * v_tcp_xy
        u_desc_z = -self.Kz * (tcp[:, 2:3] - z_push) - self.Dz * v_tcp_z

        # Push: forward along dir with lateral funnel to anchor and damping
        err_xy = tcp[:, 0:2] - anchor_xy
        along = torch.sum(err_xy * dir_xy, dim=-1, keepdim=True) * dir_xy
        side = err_xy - along
        u_push_xy = self.K_long * dir_xy - self.K_lat * side - self.D_xy * v_tcp_xy
        u_push_z = -self.Kz * (tcp[:, 2:3] - z_push) - self.Dz * v_tcp_z

        # Blend phases
        move_xy = w_app * u_app_xy + w_desc * u_desc_xy + w_push * u_push_xy
        dz = w_app * u_app_z + w_desc * u_desc_z + w_push * u_push_z

        # Soft cap XY speed
        mag = torch.norm(move_xy, dim=-1, keepdim=True) + eps
        cap = self.v_cap
        scaled = cap * torch.tanh(mag / (cap + eps))
        move_xy = move_xy * (scaled / mag)

        move3 = torch.cat([move_xy, dz], dim=1)

        # Gripper: slightly open during approach, more closed during push
        grip_open = self.grip_open.unsqueeze(0).expand(B, 1)
        grip_close = self.grip_close.unsqueeze(0).expand(B, 1)
        grip = (w_app + w_desc) * grip_open + w_push * grip_close

        mean = torch.cat([move3, grip], dim=1)

        # Std: shrink during push
        base_log_std = self.log_std_vec.unsqueeze(0).expand(B, 4)
        shrink = torch.pow(1.0 - w_push, self.alpha_std) + 1e-3  # (B,1)
        shrink = 0.3 + 0.7 * shrink  # keep a floor
        std = torch.exp(base_log_std) * shrink

        return (mean, std)

    def get_param_ranges(self):
        return {
            "r_back": (0.02, 0.20),
            "z_clear": (0.03, 0.25),
            "h_push": (0.005, 0.06),
            "K_app": (0.1, 20.0),
            "D_app": (0.0, 3.0),
            "K_align": (0.0, 8.0),
            "K_long": (0.0, 8.0),
            "K_lat": (0.0, 20.0),
            "Kz": (0.5, 30.0),
            "Dz": (0.0, 3.0),
            "D_xy": (0.0, 3.0),
            "r_stage": (0.02, 0.30),
            "r_contact": (0.01, 0.15),
            "align_thr": (-1.0, 1.0),
            "alpha_stage": (5.0, 200.0),
            "alpha_contact": (5.0, 200.0),
            "alpha_align": (5.0, 200.0),
            "v_cap": (0.2, 5.0),
            "grip_open": (-1.0, 1.0),
            "grip_close": (-2.0, 0.5),
            "log_std_vec": (-5.0, 0.0),
            "alpha_std": (0.5, 10.0),
        }
