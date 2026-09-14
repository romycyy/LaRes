# Controlled variant for the fittable-bank qualification study (predeclared 2026-09-14, before any
# evaluation of this file).
# provenance: hand-written variant of simple_standoff; one change: moderately parameterized geometry.
# Identical to simple_standoff except the standoff target gains two geometry parameters: a lateral
# offset along the planar perpendicular of the object-to-goal line (initial 0.0, bounds [-0.05, 0.05])
# and an approach height added while the arm is planar-far from the target, scaled linearly up to
# 0.1 m of planar distance (initial 0.03, bounds [0, 0.1]).


class GeneratedPolicy(SymbolicPolicy):
    """Proportional servo toward a standoff point with a lateral offset and an approach lift."""

    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.standoff = nn.Parameter(torch.tensor(0.04))
        self.gain = nn.Parameter(torch.tensor(6.0))
        self.height_offset = nn.Parameter(torch.tensor(0.01))
        self.lateral_offset = nn.Parameter(torch.tensor(0.0))
        self.approach_height = nn.Parameter(torch.tensor(0.03))
        self.grip_bias = nn.Parameter(torch.tensor(-0.5))
        self.log_std = nn.Parameter(torch.tensor(-1.5))

    def forward(self, obs):
        tcp = self.obs_field(obs, "tcp")
        obj = self.obs_field(obs, "obj")
        goal = self.obs_field(obs, "goal")

        to_goal = goal - obj
        d_goal = torch.norm(to_goal, dim=-1, keepdim=True) + 1e-8
        direction = to_goal / d_goal
        perpendicular = torch.cat(
            [-direction[:, 1:2], direction[:, 0:1], torch.zeros_like(direction[:, 2:3])], dim=1
        )
        perpendicular = perpendicular / (torch.norm(perpendicular, dim=-1, keepdim=True) + 1e-8)

        target = obj - self.standoff * direction + self.lateral_offset * perpendicular
        planar_distance = torch.norm(target[:, 0:2] - tcp[:, 0:2], dim=-1, keepdim=True)
        lift = self.approach_height * torch.clamp(planar_distance / 0.1, 0.0, 1.0)
        target = target + torch.cat(
            [torch.zeros_like(target[:, 0:2]), self.height_offset + lift], dim=1
        )
        move = self.gain * (target - tcp)
        grip = self.grip_bias.unsqueeze(0).expand(obs.shape[0], 1)
        mean = torch.cat([move, grip], dim=1)
        std = torch.exp(self.log_std) * torch.ones_like(mean)
        return mean, std

    def get_param_ranges(self):
        return {
            "standoff": (0.0, 0.15),
            "gain": (0.1, 20.0),
            "height_offset": (-0.05, 0.05),
            "lateral_offset": (-0.05, 0.05),
            "approach_height": (0.0, 0.1),
            "grip_bias": (-2.0, 2.0),
            "log_std": (-5.0, 0.0),
        }
