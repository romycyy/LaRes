class GeneratedPolicy(SymbolicPolicy):
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)

        # Normalized position targets: a moderately curled retaining posture.
        self.grasp_pose = nn.Parameter(torch.tensor([
            0.555, 0.176, 0.0, 0.0, 0.0, -0.65, 0.15,
            -0.30, -0.30, -0.25, -0.20, -0.45,
            0.0, 0.0, 0.05, 0.0, 0.0, 0.05, -0.10, 0.35
        ]))

        self.tracking = nn.Parameter(torch.tensor(0.80))
        self.joint_damping = nn.Parameter(torch.tensor(0.012))
        self.capture_curl = nn.Parameter(torch.tensor(0.16))

        self.stroke_amp = nn.Parameter(torch.tensor(0.28))
        self.lift_amp = nn.Parameter(torch.tensor(0.16))
        self.spin_target = nn.Parameter(torch.tensor(8.0))
        self.speed_gain = nn.Parameter(torch.tensor(0.35))
        self.phase_lead = nn.Parameter(torch.tensor(0.65))

        self.center_gain = nn.Parameter(torch.tensor(0.22))
        self.center_length = nn.Parameter(torch.tensor(0.04))
        self.safe_radius = nn.Parameter(torch.tensor(0.055))
        self.position_sharpness = nn.Parameter(torch.tensor(120.0))

        self.contact_radius = nn.Parameter(torch.tensor(0.04))
        self.contact_sharpness = nn.Parameter(torch.tensor(120.0))
        self.stance_sharpness = nn.Parameter(torch.tensor(4.0))

        self.tilt_gain = nn.Parameter(torch.tensor(0.16))
        self.swing_damping = nn.Parameter(torch.tensor(0.10))

        # Fixed geometry and coordination coefficients, not learned layers.
        self.register_buffer("curl_mask", torch.tensor([[
            0.0, 0.0, 0.0, 0.0, 0.0, 0.20, 0.0,
            1.0, 1.0, 1.0, 1.0, 0.15,
            1.0, 1.0, 1.0, 1.0, 0.10, 1.0, 0.20, -0.60
        ]]))

        # FF, MF, RF, LF, TH phase offsets: 0, pi/2, pi, 3pi/2, pi.
        self.register_buffer(
            "phase_cos", torch.tensor([[1.0, 0.0, -1.0, 0.0, -1.0]])
        )
        self.register_buffer(
            "phase_sin", torch.tensor([[0.0, 1.0, 0.0, -1.0, 0.0]])
        )

    def forward(self, obs):
        q = self.obs_field(obs, "hand_dof_pos")
        qd = self.obs_field(obs, "hand_dof_vel")
        obj = self.obs_field(obs, "obj_pos")
        quat = self.obs_field(obs, "obj_rot")
        omega = 5.0 * self.obs_field(obs, "obj_angvel")
        home = self.obs_field(obs, "in_hand_pos")
        tips = self.obs_field(obs, "fingertip_pos").reshape(
            obs.shape[0], 5, 3
        )

        # Named hand fields have their own documented joint order.
        # Exclude passive distal joints and preserve the 20-action order.
        q_act = torch.cat(
            [q[:, :17], q[:, 20:22], q[:, 23:24]], dim=1
        )
        qd_act = 5.0 * torch.cat(
            [qd[:, :17], qd[:, 20:22], qd[:, 23:24]], dim=1
        )
        passive = torch.cat([q[:, 17:20], q[:, 22:23]], dim=1)
        passive_curl = torch.mean(
            0.5 * (passive + 1.0), dim=1, keepdim=True
        )

        # Quaternion -> world-frame pen long axis (local x).
        quat = quat / (
            torch.norm(quat, dim=1, keepdim=True) + 1e-8
        )
        w = quat[:, 0:1]
        x = quat[:, 1:2]
        y = quat[:, 2:3]
        z = quat[:, 3:4]

        axis = torch.cat([
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y)
        ], dim=1)
        axis = axis / (
            torch.norm(axis, dim=1, keepdim=True) + 1e-8
        )

        spin = torch.sum(omega * axis, dim=1, keepdim=True)
        transverse_omega = omega - spin * axis
        transverse_speed = torch.norm(
            transverse_omega, dim=1, keepdim=True
        )
        transverse_direction = transverse_omega / (
            transverse_speed + 1e-8
        )

        # Phase 1: retain/recenter. Roll is reduced outside the safe region.
        error = home - obj
        distance = torch.norm(error, dim=1, keepdim=True)
        center_direction = error / (distance + 1e-8)
        center = (
            self.center_gain
            * torch.tanh(distance / self.center_length)
            * center_direction
        )

        retained = torch.sigmoid(
            self.position_sharpness * (self.safe_radius - distance)
        )
        retained = self.record_gate("retained", retained)

        # Estimate fingertip proximity to a finite pen-axis segment.
        offset = tips - obj.unsqueeze(1)
        axis_3d = axis.unsqueeze(1)
        axial = torch.sum(
            offset * axis_3d, dim=2, keepdim=True
        )
        radial = offset - axial * axis_3d
        radial_distance = torch.norm(
            radial, dim=2, keepdim=True
        )
        end_excess = torch.clamp(torch.abs(axial) - 0.075, min=0.0)
        pen_distance = torch.sqrt(
            radial_distance * radial_distance
            + end_excess * end_excess
            + 1e-10
        )
        mean_tip_distance = torch.mean(
            pen_distance, dim=1, keepdim=False
        )  # (B, 1)

        contact = torch.sigmoid(
            self.contact_sharpness
            * (self.contact_radius - mean_tip_distance)
        )
        contact = self.record_gate("contact", contact)
        rolling = retained * contact

        # Unit tangents for positive rotation about the pen's own axis.
        radial_direction = radial / (radial_distance + 1e-8)
        tangent = torch.cross(
            axis_3d.expand_as(radial_direction),
            radial_direction,
            dim=2
        )
        tangent = tangent / (
            torch.norm(tangent, dim=2, keepdim=True) + 1e-8
        )

        # Axial twist phase from swing-twist decomposition about local x.
        # Squared quaternion expressions are invariant to quaternion sign.
        twist = torch.cat([w, x], dim=1)
        twist = twist / (
            torch.norm(twist, dim=1, keepdim=True) + 1e-8
        )
        tw = twist[:, 0:1]
        tx = twist[:, 1:2]
        cos_roll = tw * tw - tx * tx
        sin_roll = 2.0 * tw * tx

        sin_lead = torch.sin(self.phase_lead)
        cos_lead = torch.cos(self.phase_lead)
        sin_base = sin_roll * cos_lead + cos_roll * sin_lead
        cos_base = cos_roll * cos_lead - sin_roll * sin_lead

        sin_phase = (
            sin_base * self.phase_cos + cos_base * self.phase_sin
        )
        cos_phase = (
            cos_base * self.phase_cos - sin_base * self.phase_sin
        )

        # Phase 2: alternating loaded strokes and unloaded returns.
        stance = torch.sigmoid(self.stance_sharpness * cos_phase)
        stance = self.record_gate("stance", stance)

        # Positive startup drive; excessive spin can reverse the stroke.
        drive = 0.25 + 0.75 * torch.tanh(
            self.speed_gain * (self.spin_target - spin)
        )
        stroke = (
            self.stroke_amp
            * drive.unsqueeze(1)
            * sin_phase.unsqueeze(2)
            * tangent
        )
        load = self.lift_amp * (2.0 * stance - 1.0)

        sx = stroke[:, :, 0]
        sy = stroke[:, :, 1]
        sz = stroke[:, :, 2]

        # Explicit approximate palm-up joint/tip-direction mapping.
        flex = (-sy - 0.45 * sz) / (1.0 + 0.45**2)**0.5
        thumb_sweep = 0.8 * sx[:, 4:5] + 0.6 * sz[:, 4:5]
        thumb_flex = -0.6 * sy[:, 4:5] - 0.8 * sz[:, 4:5]
        zero = torch.zeros_like(spin)

        gait = torch.cat([
            zero,                                      # WRJ1
            zero,                                      # WRJ0
            0.45 * sx[:, 0:1],                          # FFJ3
            0.45 * sx[:, 1:2],                          # MFJ3
            0.45 * sx[:, 2:3],                          # RFJ3
            0.15 * load[:, 3:4],                        # LFJ4
            0.70 * thumb_sweep,                         # THJ4
            flex[:, 0:1],                              # FFJ2
            flex[:, 1:2],                              # MFJ2
            flex[:, 2:3],                              # RFJ2
            flex[:, 3:4],                              # LFJ3
            0.40 * thumb_flex,                          # THJ3
            0.35 * flex[:, 0:1] + load[:, 0:1],         # FFJ1
            0.35 * flex[:, 1:2] + load[:, 1:2],         # MFJ1
            0.35 * flex[:, 2:3] + load[:, 2:3],         # RFJ1
            0.35 * flex[:, 3:4] + load[:, 3:4],         # LFJ2
            0.50 * thumb_sweep,                         # THJ2
            0.35 * flex[:, 3:4] + load[:, 3:4],         # LFJ1
            0.50 * thumb_flex + 0.30 * load[:, 4:5],    # THJ1
            -0.60 * thumb_flex - 0.30 * load[:, 4:5]    # THJ0
        ], dim=1)

        # Stabilize the long axis in the palm plane without locking roll.
        ax = axis[:, 0:1]
        ay = axis[:, 1:2]
        az = axis[:, 2:3]
        tilt_correction = torch.cat(
            [-az * ay, az * ax, zero], dim=1
        )
        rotation_correction = (
            self.tilt_gain * tilt_correction
            - self.swing_damping
            * torch.tanh(transverse_speed / 4.0)
            * transverse_direction
        )

        cx = center[:, 0:1]
        cy = center[:, 1:2]
        cz = center[:, 2:3]
        rx = rotation_correction[:, 0:1]
        ry = rotation_correction[:, 1:2]
        rz = rotation_correction[:, 2:3]

        side_center = 0.6 * cx + 0.4 * rz
        finger_center = -0.5 * cy + 0.3 * cz
        middle_center = -0.25 * cy + 0.4 * cz

        stabilization = torch.cat([
            -cy + 0.25 * cz + rx,
            cx + ry,
            side_center,
            side_center,
            side_center,
            0.20 * cz,
            -0.50 * cx - 0.50 * rz,
            finger_center,
            finger_center,
            finger_center,
            finger_center,
            0.30 * cz,
            middle_center,
            middle_center,
            middle_center,
            middle_center,
            -0.40 * cx,
            middle_center,
            0.25 * cz,
            -0.50 * cz
        ], dim=1)

        # Increase retention when contact or centering deteriorates.
        extra_curl = 0.0 * self.capture_curl * (
            (1.0 - contact) + 0.75 * (1.0 - retained)
        )
        target = (
            self.grasp_pose.unsqueeze(0)
            + (extra_curl - 0.03 * passive_curl) * self.curl_mask
            + stabilization
            + self._stroke_gain(spin) * rolling * gait
        )

        mean = (
            q_act
            + self.tracking * (target - q_act)
            - self.joint_damping * qd_act
        )

        # The detailed environment specification executes clamped targets.
        mean = torch.clamp(mean, -1.0, 1.0)
        return mean, self.fixed_std(mean)

    def get_param_ranges(self):
        return {
            "grasp_pose": (-1.0, 1.0),
            "tracking": (0.40, 0.95),
            "joint_damping": (0.0, 0.05),
            "capture_curl": (0.02, 0.40),
            "stroke_amp": (0.03, 0.60),
            "lift_amp": (0.02, 0.35),
            "spin_target": (5.5, 14.0),
            "speed_gain": (0.05, 1.0),
            "phase_lead": (0.10, 1.40),
            "center_gain": (0.02, 0.60),
            "center_length": (0.015, 0.12),
            "safe_radius": (0.025, 0.075),
            "position_sharpness": (30.0, 200.0),
            "contact_radius": (0.018, 0.070),
            "contact_sharpness": (30.0, 200.0),
            "stance_sharpness": (1.0, 12.0),
            "tilt_gain": (0.0, 0.40),
            "swing_damping": (0.0, 0.40),
        }