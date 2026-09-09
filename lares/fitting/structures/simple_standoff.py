# Reference structure for the Phase 3 complexity ablation (spec.md E3).
# provenance: hand-written simple family, 5 parameters, no phase gate.
# Servos to a standoff point behind the object on the object-to-goal line, so the
# push direction is set by geometry rather than by a learned switch.


class GeneratedPolicy(SymbolicPolicy):
    """Proportional servo toward a standoff point behind the object."""

    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.standoff = nn.Parameter(torch.tensor(0.04))
        self.gain = nn.Parameter(torch.tensor(6.0))
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
            "grip_bias": (-2.0, 2.0),
            "log_std": (-5.0, 0.0),
        }
