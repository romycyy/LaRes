# Reference structure for the Phase 3 complexity ablation (spec.md E3).
# provenance: hand-written simple family, 6 parameters, two phases, one gate.
# Kept deliberately minimal: reach the object, then push it at the goal.


class GeneratedPolicy(SymbolicPolicy):
    """Reach the object, then push it toward the goal, blended by one contact gate."""

    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.w_reach = nn.Parameter(torch.tensor(3.0))
        self.w_push = nn.Parameter(torch.tensor(2.0))
        self.sharpness = nn.Parameter(torch.tensor(40.0))
        self.contact_radius = nn.Parameter(torch.tensor(0.06))
        self.grip_bias = nn.Parameter(torch.tensor(-0.5))
        self.log_std = nn.Parameter(torch.tensor(-1.5))

    def forward(self, obs):
        tcp = self.obs_field(obs, "tcp")
        obj = self.obs_field(obs, "obj")
        goal = self.obs_field(obs, "goal")

        to_obj = obj - tcp
        d_obj = torch.norm(to_obj, dim=-1, keepdim=True) + 1e-8
        to_goal = goal - obj
        d_goal = torch.norm(to_goal, dim=-1, keepdim=True) + 1e-8

        contact = torch.sigmoid(self.sharpness * (self.contact_radius - d_obj))
        contact = self.record_gate("contact", contact)

        move = (1.0 - contact) * self.w_reach * (to_obj / d_obj) + contact * self.w_push * (
            to_goal / d_goal
        )
        grip = self.grip_bias.unsqueeze(0).expand(obs.shape[0], 1)
        mean = torch.cat([move, grip], dim=1)
        std = torch.exp(self.log_std) * torch.ones_like(mean)
        return mean, std

    def get_param_ranges(self):
        return {
            "w_reach": (0.1, 10.0),
            "w_push": (0.1, 10.0),
            "sharpness": (1.0, 200.0),
            "contact_radius": (0.01, 0.20),
            "grip_bias": (-2.0, 2.0),
            "log_std": (-5.0, 0.0),
        }
