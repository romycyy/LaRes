class GeneratedPolicy(SymbolicPolicy):
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)

        self.k_radial = nn.Parameter(torch.tensor(6.0))
        self.k_orbit = nn.Parameter(torch.tensor(3.0))
        self.k_push = nn.Parameter(torch.tensor(5.0))
        self.k_z = nn.Parameter(torch.tensor(8.0))

        self.orbit_radius = nn.Parameter(torch.tensor(0.065))
        self.contact_backoff = nn.Parameter(torch.tensor(0.03))
        self.contact_offset = nn.Parameter(torch.tensor(0.005))
        self.angle_tol = nn.Parameter(torch.tensor(0.25))
        self.radial_tol = nn.Parameter(torch.tensor(0.012))
        self.goal_tol = nn.Parameter(torch.tensor(0.02))

    def forward(self, obs):
        tcp = self.obs_field(obs, "tcp")
        obj = self.obs_field(obs, "obj")
        goal = self.obs_field(obs, "goal")
        prev_tcp = self.obs_field(obs, "prev_tcp")
        prev_obj = self.obs_field(obs, "prev_obj")

        tcp_velocity = tcp - prev_tcp
        obj_velocity = obj - prev_obj
        relative_velocity = tcp_velocity - obj_velocity

        # Slice only named fields, never the raw observation.
        r = tcp[:, :2] - obj[:, :2]
        goal_error = goal[:, :2] - obj[:, :2]

        radius = torch.norm(r, dim=-1, keepdim=True)
        distance = torch.norm(goal_error, dim=-1, keepdim=True)

        ones = torch.ones_like(distance)
        zeros = torch.zeros_like(distance)
        goal_fallback = torch.cat([ones, zeros], dim=1)
        radial_fallback = torch.cat([-ones, zeros], dim=1)

        u = torch.where(
            distance > 1e-6,
            goal_error / torch.clamp(distance, min=1e-6),
            goal_fallback,
        )
        r_hat = torch.where(
            radius > 1e-6,
            r / torch.clamp(radius, min=1e-6),
            radial_fallback,
        )

        rear_direction = -u
        cross = (
            r_hat[:, 0:1] * rear_direction[:, 1:2]
            - r_hat[:, 1:2] * rear_direction[:, 0:1]
        )
        dot = torch.sum(
            r_hat * rear_direction, dim=-1, keepdim=True
        )
        theta = torch.atan2(cross, dot)
        j_r_hat = torch.cat(
            [-r_hat[:, 1:2], r_hat[:, 0:1]], dim=1
        )

        # Orbit: regulate the larger radius and circulate toward the rear.
        orbit_command = (
            self.k_radial * (self.orbit_radius - radius) * r_hat
            + self.k_orbit * self.orbit_radius * theta * j_r_hat
        )

        # Seat: acquire the smaller rear contact radius.
        seat_error = -self.contact_backoff * u - r
        seat_distance = torch.norm(
            seat_error, dim=-1, keepdim=True
        )
        seat_direction = seat_error / torch.clamp(
            seat_distance, min=1e-6
        )
        seat_command = (
            self.k_radial * seat_distance * seat_direction
        )

        # Drive: push along the goal direction and correct lateral offset.
        along_goal = torch.sum(r * u, dim=-1, keepdim=True)
        lateral_error = r - along_goal * u
        lateral_distance = torch.norm(
            lateral_error, dim=-1, keepdim=True
        )
        lateral_direction = lateral_error / torch.clamp(
            lateral_distance, min=1e-6
        )
        drive_command = (
            self.k_push * distance * u
            - self.k_radial * lateral_distance * lateral_direction
        )

        angular_score = torch.sigmoid(
            (torch.abs(theta) - self.angle_tol) / 0.04
        )
        radial_score = torch.sigmoid(
            (radius - self.contact_backoff - self.radial_tol)
            / 0.003
        )

        # Record FINAL phase weights, not nested switching factors.
        # Each returned weight directly multiplies its own command:
        # forcing seat is no longer masked by a saturated orbit gate.
        orbit = self.record_gate("orbit", angular_score)
        seat = self.record_gate(
            "seat", (1.0 - angular_score) * radial_score
        )
        drive = self.record_gate(
            "drive",
            (1.0 - angular_score) * (1.0 - radial_score),
        )

        goal_margin = distance - self.goal_tol
        moving_support = torch.clamp(
            goal_margin / 0.002, min=0.0, max=1.0
        )
        stop = self.record_gate(
            "stop",
            1.0
            - moving_support * torch.sigmoid(goal_margin / 0.002),
        )

        planar_command = (
            orbit * orbit_command
            + seat * seat_command
            + drive * drive_command
        )

        # Weak fixed damping; exactly zero planar mean inside goal_tol.
        planar_command = (1.0 - stop) * (
            planar_command
            - 0.5 * self.k_radial * relative_velocity[:, :2]
        )

        height_error = (
            obj[:, 2:3] + self.contact_offset - tcp[:, 2:3]
        )
        vertical_command = self.k_z * (
            height_error - 0.5 * relative_velocity[:, 2:3]
        )

        gripper_command = torch.full_like(vertical_command, -4.0)
        mean = torch.cat(
            [planar_command, vertical_command, gripper_command],
            dim=1,
        )
        return mean, self.fixed_std(mean)

    def get_param_ranges(self):
        return {
            "k_radial": (1.0, 15.0),
            "k_orbit": (0.5, 10.0),
            "k_push": (0.5, 15.0),
            "k_z": (1.0, 20.0),
            "orbit_radius": (0.04, 0.12),
            "contact_backoff": (0.015, 0.04),
            "contact_offset": (-0.02, 0.03),
            "angle_tol": (0.08, 0.65),
            "radial_tol": (0.004, 0.03),
            "goal_tol": (0.005, 0.05),
        }