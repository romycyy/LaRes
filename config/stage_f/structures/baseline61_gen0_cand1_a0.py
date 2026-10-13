class GeneratedPolicy(SymbolicPolicy):
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)

        self.k_reach = nn.Parameter(torch.tensor(6.0))
        self.k_drive = nn.Parameter(torch.tensor(5.0))
        self.k_height = nn.Parameter(torch.tensor(8.0))
        self.backoff = nn.Parameter(torch.tensor(0.04))
        self.contact_offset = nn.Parameter(torch.tensor(0.005))
        self.lift = nn.Parameter(torch.tensor(0.075))
        self.capture_radius = nn.Parameter(torch.tensor(0.035))
        self.gate_width = nn.Parameter(torch.tensor(0.012))
        self.goal_radius = nn.Parameter(torch.tensor(0.025))

    def forward(self, obs):
        tcp = self.obs_field(obs, "tcp")
        obj = self.obs_field(obs, "obj")
        goal = self.obs_field(obs, "goal")

        # Component selection is from named fields, never from raw obs.
        tcp_xy = tcp[:, :2]
        tcp_z = tcp[:, 2:3]
        obj_xy = obj[:, :2]
        obj_z = obj[:, 2:3]
        goal_xy = goal[:, :2]

        eps = 1e-8

        # Normalized planar pushing direction.
        goal_delta = goal_xy - obj_xy
        goal_distance = torch.norm(goal_delta, dim=-1, keepdim=True)
        push_direction = goal_delta / (goal_distance + eps)

        rear_point = obj_xy - self.backoff * push_direction
        contact_height = obj_z + self.contact_offset

        rear_error = rear_point - tcp_xy
        rear_distance = torch.norm(rear_error, dim=-1, keepdim=True)
        rear_direction = rear_error / (rear_distance + eps)

        # Strong contact requires both rear alignment and correct height.
        gx = torch.sigmoid(
            (self.capture_radius - rear_distance) / self.gate_width
        )
        gx = self.record_gate("rear_alignment", gx)

        vertical_error = torch.abs(tcp_z - contact_height)
        gz = torch.sigmoid(
            (self.capture_radius - vertical_error) / self.gate_width
        )
        gz = self.record_gate("vertical_contact", gz)

        contact = self.record_gate("contact", gx * gz)

        distance_squared = goal_distance.square()
        q = distance_squared / (
            distance_squared + self.goal_radius.square() + 1e-12
        )
        q = self.record_gate("goal_drive_weight", q)

        acquisition = self.record_gate("soft_acquisition", 1.0 - contact)
        arrival = self.record_gate("soft_arrival", contact * (1.0 - q))

        # Normally contact - arrival == contact*q.
        # Clamping prevents reversed drive under forced gate diagnostics.
        transport = torch.clamp(contact - arrival, min=0.0, max=1.0)
        transport = self.record_gate("soft_transport", transport)

        # Restore distance after normalization to retain gains per metre.
        reach_mean = self.k_reach * (rear_distance + eps) * rear_direction
        drive_mean = self.k_drive * (goal_distance + eps) * push_direction
        planar_mean = acquisition * reach_mean + transport * drive_mean

        # Approach above the rear point, then lower continuously into contact.
        height_target = contact_height + self.lift * (1.0 - gx)
        height_mean = self.k_height * (height_target - tcp_z)

        # Closed gripper; this controller never grasps and lifts the puck.
        gripper_mean = -4.0 * torch.ones_like(tcp_z)

        mean = torch.cat(
            [planar_mean, height_mean, gripper_mean], dim=1
        )
        return mean, self.fixed_std(mean)

    def get_param_ranges(self):
        return {
            "k_reach": (1.0, 15.0),
            "k_drive": (0.5, 15.0),
            "k_height": (1.0, 20.0),
            "backoff": (0.02, 0.08),
            "contact_offset": (-0.02, 0.03),
            "lift": (0.03, 0.15),
            "capture_radius": (0.015, 0.07),
            "gate_width": (0.003, 0.035),
            "goal_radius": (0.005, 0.06),
        }