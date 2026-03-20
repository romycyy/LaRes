import torch
import torch.nn as nn


class SymbolicPolicy(nn.Module):
    """
    Base class for LLM-generated symbolic policies.

    Subclasses implement:
      forward(obs) -> (mean, std)   both (batch, action_dim)
      get_param_ranges() -> dict    param_name -> (lo, hi)

    Only explicit symbolic expressions are allowed.
    Neural-network layers (Linear, Conv, LSTM, …) are forbidden.
    """

    FORBIDDEN_MODULES = (
        nn.Linear,
        nn.Bilinear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
        nn.LSTM,
        nn.LSTMCell,
        nn.GRU,
        nn.GRUCell,
        nn.RNN,
        nn.RNNCell,
        nn.Transformer,
        nn.TransformerEncoder,
        nn.TransformerDecoder,
        nn.TransformerEncoderLayer,
        nn.TransformerDecoderLayer,
        nn.MultiheadAttention,
    )

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def forward(self, obs):
        """
        Map observations to a Gaussian action distribution.

        Args:
            obs: (batch_size, obs_dim) float tensor.
        Returns:
            mean: (batch_size, action_dim) — centre of the pre-tanh Gaussian.
            std:  (batch_size, action_dim) — scale (must be > 0).
        """
        raise NotImplementedError

    def get_param_ranges(self):
        """Return {parameter_name: (min_value, max_value)} for every Parameter."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    #  Utilities
    # ------------------------------------------------------------------

    def clip_params(self):
        """Project every parameter onto its declared [lo, hi] range."""
        ranges = self.get_param_ranges()
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in ranges:
                    lo, hi = ranges[name]
                    param.clamp_(lo, hi)

    def validate(self):
        """Raise TypeError if any forbidden NN module is found."""
        for name, module in self.named_modules():
            if name == "":
                continue
            if isinstance(module, self.FORBIDDEN_MODULES):
                raise TypeError(
                    f"SymbolicPolicy must not contain {type(module).__name__} "
                    f"(found at '{name}'). Use explicit symbolic expressions only."
                )
        return True

    def count_parameters(self):
        """Total number of scalar parameters."""
        return sum(p.numel() for p in self.parameters())
