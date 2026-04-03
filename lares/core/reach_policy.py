"""Hand-crafted symbolic policy for MetaWorld reach-v2 (demo / baseline)."""

import torch
import torch.nn as nn

from lares.core.symbolic_policy import SymbolicPolicy


class ReachPolicy(SymbolicPolicy):
    """Two-phase reach policy: approach target, then fine-adjust."""

    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.w_coarse = nn.Parameter(torch.tensor(4.0))
        self.w_fine = nn.Parameter(torch.tensor(1.0))
        self.threshold = nn.Parameter(torch.tensor(0.06))
        self.sharpness = nn.Parameter(torch.tensor(80.0))
        self.grip_bias = nn.Parameter(torch.tensor(0.0))
        self.log_std = nn.Parameter(torch.tensor(-1.0))

    def forward(self, obs):
        tcp = obs[:, 0:3]
        target = obs[:, 4:7]
        diff = target - tcp
        dist = torch.norm(diff, dim=-1, keepdim=True) + 1e-8
        direction = diff / dist

        phase = torch.sigmoid(self.sharpness * (self.threshold - dist))
        move = (1 - phase) * self.w_coarse * direction + phase * self.w_fine * direction
        grip = self.grip_bias * torch.ones(obs.shape[0], 1)
        mean = torch.cat([move, grip], dim=1)[:, : self.action_dim]
        std = torch.exp(self.log_std) * torch.ones_like(mean)
        return (mean, std)

    def get_param_ranges(self):
        return {
            "w_coarse": (0.5, 15.0),
            "w_fine": (0.1, 5.0),
            "threshold": (0.01, 0.3),
            "sharpness": (5.0, 200.0),
            "grip_bias": (-1.0, 1.0),
            "log_std": (-5.0, 0.0),
        }
