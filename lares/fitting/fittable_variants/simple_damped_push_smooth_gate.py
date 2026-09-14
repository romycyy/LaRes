# Controlled variant for the fittable-bank qualification study (predeclared 2026-09-14, before any
# evaluation of this file).
# provenance: hand-written variant of simple_damped_push; one change: smoother gates.
# Identical to simple_damped_push except the shared gate sharpness initialises at 15 (parent 50) and
# is bounded to [1, 40] (parent [1, 200]), so both the contact and the brake gate blend gradually.


class GeneratedPolicy(SymbolicPolicy):
    """Two-phase push with velocity damping and smooth contact and brake gates."""

    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.w_reach = nn.Parameter(torch.tensor(4.0))
        self.w_push = nn.Parameter(torch.tensor(2.5))
        self.damping = nn.Parameter(torch.tensor(1.0))
        self.sharpness = nn.Parameter(torch.tensor(15.0))
        self.contact_radius = nn.Parameter(torch.tensor(0.05))
        self.brake_radius = nn.Parameter(torch.tensor(0.05))
        self.grip_bias = nn.Parameter(torch.tensor(-0.5))
        self.log_std = nn.Parameter(torch.tensor(-1.5))

    def forward(self, obs):
        tcp = self.obs_field(obs, "tcp")
        prev_tcp = self.obs_field(obs, "prev_tcp")
        obj = self.obs_field(obs, "obj")
        goal = self.obs_field(obs, "goal")

        velocity = tcp - prev_tcp
        to_obj = obj - tcp
        d_obj = torch.norm(to_obj, dim=-1, keepdim=True) + 1e-8
        to_goal = goal - obj
        d_goal = torch.norm(to_goal, dim=-1, keepdim=True) + 1e-8

        contact = torch.sigmoid(self.sharpness * (self.contact_radius - d_obj))
        brake = torch.sigmoid(self.sharpness * (self.brake_radius - d_goal))
        contact = self.record_gate("contact", contact)
        brake = self.record_gate("brake", brake)

        drive = (1.0 - contact) * self.w_reach * (to_obj / d_obj) + contact * (
            1.0 - brake
        ) * self.w_push * (to_goal / d_goal)
        move = drive - self.damping * velocity
        grip = self.grip_bias.unsqueeze(0).expand(obs.shape[0], 1)
        mean = torch.cat([move, grip], dim=1)
        std = torch.exp(self.log_std) * torch.ones_like(mean)
        return mean, std

    def get_param_ranges(self):
        return {
            "w_reach": (0.1, 10.0),
            "w_push": (0.1, 10.0),
            "damping": (0.0, 5.0),
            "sharpness": (1.0, 40.0),
            "contact_radius": (0.01, 0.20),
            "brake_radius": (0.01, 0.20),
            "grip_bias": (-2.0, 2.0),
            "log_std": (-5.0, 0.0),
        }
