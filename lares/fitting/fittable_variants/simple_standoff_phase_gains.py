# Controlled variant for the fittable-bank qualification study (predeclared 2026-09-14, before any
# evaluation of this file).
# provenance: hand-written variant of simple_standoff; one change: phase-specific gains.
# Identical to simple_standoff except the single servo gain is replaced by a far gain (initial 6.0,
# the parent's gain) and a near gain (initial 3.0), blended by a proximity gate on the distance to
# the standoff target with a fixed sharpness of 40 and a fittable near_radius (initial 0.05,
# bounds [0.01, 0.2]). The two gains start different so the gate is a lever from the first step.


class GeneratedPolicy(SymbolicPolicy):
    """Proportional servo toward a standoff point, with separate far and near gains."""

    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.standoff = nn.Parameter(torch.tensor(0.04))
        self.gain_far = nn.Parameter(torch.tensor(6.0))
        self.gain_near = nn.Parameter(torch.tensor(3.0))
        self.near_radius = nn.Parameter(torch.tensor(0.05))
        self.height_offset = nn.Parameter(torch.tensor(0.01))
        self.grip_bias = nn.Parameter(torch.tensor(-0.5))
        self.log_std = nn.Parameter(torch.tensor(-1.5))

    def forward(self, obs):
        tcp = self.obs_field(obs, "tcp")
        obj = self.obs_field(obs, "obj")
        goal = self.obs_field(obs, "goal")

        to_goal = goal - obj
        d_goal = torch.norm(to_goal, dim=-1, keepdim=True) + 1e-8
        direction = to_goal / d_goal

        target = obj - self.standoff * direction
        target = target + torch.cat(
            [
                torch.zeros_like(target[:, 0:2]),
                self.height_offset * torch.ones_like(target[:, 2:3]),
            ],
            dim=1,
        )
        error = target - tcp
        d_target = torch.norm(error, dim=-1, keepdim=True) + 1e-8
        near = torch.sigmoid(40.0 * (self.near_radius - d_target))
        near = self.record_gate("near", near)

        gain = (1.0 - near) * self.gain_far + near * self.gain_near
        move = gain * error
        grip = self.grip_bias.unsqueeze(0).expand(obs.shape[0], 1)
        mean = torch.cat([move, grip], dim=1)
        std = torch.exp(self.log_std) * torch.ones_like(mean)
        return mean, std

    def get_param_ranges(self):
        return {
            "standoff": (0.0, 0.15),
            "gain_far": (0.1, 20.0),
            "gain_near": (0.1, 20.0),
            "near_radius": (0.01, 0.20),
            "height_offset": (-0.05, 0.05),
            "grip_bias": (-2.0, 2.0),
            "log_std": (-5.0, 0.0),
        }
