class GeneratedPolicy(SymbolicPolicy):
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        # Movement gains for different phases
        self.w_far = nn.Parameter(torch.tensor(3.0))     # strong push when far
        self.w_mid = nn.Parameter(torch.tensor(1.5))     # moderate push in mid band
        self.w_near = nn.Parameter(torch.tensor(0.8))    # gentle, distance-proportional push when close
        self.w_damp = nn.Parameter(torch.tensor(0.5))    # velocity damping near target

        # Distance scaling for mid-phase speed scheduling (tanh(dist / scale))
        self.dist_scale_mid = nn.Parameter(torch.tensor(0.10))

        # Phase thresholds (meters) and sharpness (sigmoid slopes)
        self.d_mid_low = nn.Parameter(torch.tensor(0.05))   # start of mid band
        self.d_mid_high = nn.Parameter(torch.tensor(0.20))  # end of mid band / start of far
        self.d_near = nn.Parameter(torch.tensor(0.03))      # near target region
        self.d_stab = nn.Parameter(torch.tensor(0.07))      # where to apply damping

        self.s_far = nn.Parameter(torch.tensor(20.0))    # sharpness for far gate
        self.s_mid = nn.Parameter(torch.tensor(20.0))    # sharpness for mid band edges
        self.s_near = nn.Parameter(torch.tensor(30.0))   # sharpness for near gate
        self.s_stab = nn.Parameter(torch.tensor(25.0))   # sharpness for damping gate

        # Gripper control (task doesn't need grasping; keep neutral with small optional near bias)
        self.grip_bias = nn.Parameter(torch.tensor(0.0))
        self.grip_near = nn.Parameter(torch.tensor(0.0))  # add small bias when very close, if helpful

        # Exploration noise (log std)
        self.log_std_move = nn.Parameter(torch.tensor(-1.0))
        self.log_std_grip = nn.Parameter(torch.tensor(-1.0))

    def forward(self, obs):
        # obs: (B, 39)
        B = obs.shape[0]
        eps = 1e-8

        # Parse observations
        tcp = obs[:, 0:3]          # (B, 3) current end-effector position
        prev_tcp = obs[:, 18:21]   # (B, 3) previous end-effector position
        target = obs[:, 36:39]     # (B, 3) target position (same as obs[4:7] for reach)

        # Core geometry
        diff = target - tcp                           # (B, 3)
        dist = torch.norm(diff, dim=-1, keepdim=True) + eps  # (B, 1) keepdim!
        direction = diff / dist                       # (B, 3) normalized direction

        # Estimated end-effector velocity (1-step finite difference)
        vel = tcp - prev_tcp                          # (B, 3)

        # Smooth phase gates (values in [0,1])
        # Far when distance > d_mid_high
        g_far = torch.sigmoid(self.s_far * (dist - self.d_mid_high))                 # (B, 1)
        # Mid band when d_mid_low < distance < d_mid_high (approximate with band-pass via product of sigmoids)
        g_mid_low = torch.sigmoid(self.s_mid * (dist - self.d_mid_low))              # (B, 1)
        g_mid_high = torch.sigmoid(self.s_mid * (self.d_mid_high - dist))            # (B, 1)
        g_mid = g_mid_low * g_mid_high                                               # (B, 1)
        # Near when distance < d_near
        g_near = torch.sigmoid(self.s_near * (self.d_near - dist))                   # (B, 1)
        # Stabilization (apply damping inside a small radius)
        g_stab = torch.sigmoid(self.s_stab * (self.d_stab - dist))                   # (B, 1)

        # Phase-wise movement proposals
        move_far = self.w_far * direction                                             # (B, 3)
        # In the mid band, schedule speed with tanh(dist / scale) to smoothly reduce speed as we get closer
        mid_scale = torch.tanh(dist / (self.dist_scale_mid + eps))                   # (B, 1)
        move_mid = self.w_mid * (mid_scale * direction)                              # (B, 3)
        # Near target: proportional to distance in the direction to avoid overshoot
        move_near = self.w_near * (dist * direction)                                 # (B, 3)
        # Damping to reduce oscillations near the target
        move_damp = -self.w_damp * vel                                               # (B, 3)

        # Blend phases with smooth gates
        move = g_far * move_far + g_mid * move_mid + g_near * move_near + g_stab * move_damp  # (B, 3)

        # Gripper: keep neutral with optional slight adjustment when very close (not required for reach)
        grip = self.grip_bias * torch.ones(B, 1, device=obs.device, dtype=obs.dtype) + self.grip_near * g_near

        # Concatenate action mean (pre-tanh): (B, 4)
        mean = torch.cat([move, grip], dim=1)

        # Isotropic std for movement, separate std for gripper
        std_move = torch.exp(self.log_std_move) * torch.ones(B, 3, device=obs.device, dtype=obs.dtype)
        std_grip = torch.exp(self.log_std_grip) * torch.ones(B, 1, device=obs.device, dtype=obs.dtype)
        std = torch.cat([std_move, std_grip], dim=1)  # (B, 4)

        return (mean, std)

    def get_param_ranges(self):
        return {
            "w_far": (0.1, 10.0),
            "w_mid": (0.0, 5.0),
            "w_near": (0.0, 3.0),
            "w_damp": (0.0, 3.0),

            "dist_scale_mid": (0.01, 0.5),

            "d_mid_low": (0.01, 0.30),
            "d_mid_high": (0.05, 0.50),
            "d_near": (0.005, 0.10),
            "d_stab": (0.02, 0.20),

            "s_far": (1.0, 200.0),
            "s_mid": (1.0, 200.0),
            "s_near": (1.0, 200.0),
            "s_stab": (1.0, 200.0),

            "grip_bias": (-1.0, 1.0),
            "grip_near": (-1.0, 1.0),

            "log_std_move": (-5.0, 0.0),
            "log_std_grip": (-5.0, 0.0),
        }