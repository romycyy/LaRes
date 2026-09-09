import torch
import torch.nn as nn

from lares.core.obs_schema import get_active_schema


class SymbolicPolicy(nn.Module):
    """
    Base class for LLM-generated symbolic policies.

    Subclasses implement:
      forward(obs) -> (mean, std)   both (batch, action_dim)
      get_param_ranges() -> dict    param_name -> (lo, hi)

    Only explicit symbolic expressions are allowed.
    Neural-network layers (Linear, Conv, LSTM, …) are forbidden.

    Observations are read through named accessors, never raw indices::

        tcp  = self.obs_field(obs, "tcp")     # (batch, 3)
        obj  = self.obs_field(obs, "obj")     # (batch, 3)
        goal = self.obs_field(obs, "goal")    # (batch, 3)

    Optional telemetry: call ``self.record_gate(name, value)`` inside
    ``forward`` to expose a phase gate.  Recorded values are detached, so they
    cannot change the returned ``(mean, std)`` or leak into the gradient.
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
        # Captured at construction rather than looked up per call, so a policy
        # keeps the layout it was built and validated against.
        self.obs_schema = get_active_schema()
        self.last_gates: dict = {}
        #: Gate name to forced value. Empty in ordinary use; set only by a
        #: controlled substitution, whose results are labelled as interventions
        #: and can never enter fitness ranking.
        self.gate_overrides: dict = {}

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
    #  Named observation access
    # ------------------------------------------------------------------

    def obs_field(self, obs, name):
        """``(batch, size)`` view of a named observation field.

        Raises when no schema is bound rather than falling back to indices: an
        unchecked layout is exactly the failure this interface exists to stop.
        """
        if self.obs_schema is None:
            raise RuntimeError(
                "No observation schema is bound. Infrastructure must call "
                "lares.core.obs_schema.set_active_schema(...) before constructing "
                "a policy that reads named fields."
            )
        return self.obs_schema.slice(obs, name)

    def obs_fields(self, obs, *names):
        """Several named fields at once, in the order requested."""
        return tuple(self.obs_field(obs, n) for n in names)

    # ------------------------------------------------------------------
    #  Gate telemetry
    # ------------------------------------------------------------------

    def reset_diagnostics(self):
        """Clear telemetry between episodes.

        Separate from any policy state: diagnostics reset per episode without
        touching parameters (``spec.md`` AC-1). Gate overrides are *not* cleared:
        an intervention is set deliberately and must survive an episode boundary.
        """
        self.last_gates = {}

    def set_gate_override(self, name, value):
        """Force a gate to a fixed value, for controlled substitution (FR-9).

        Only takes effect where the generated code uses the value ``record_gate``
        returns, which is why the contract asks policies to write
        ``g = self.record_gate("g", g)``. An override on a gate whose return value
        is discarded changes the telemetry but not the action, and
        :func:`lares.repair.interventions.gate_override_is_effective` checks for
        exactly that rather than letting a null intervention look like evidence.
        """
        self.gate_overrides[str(name)] = value

    def clear_gate_overrides(self):
        self.gate_overrides = {}

    def record_gate(self, name, value):
        """Expose a phase gate for diagnostics, and allow it to be forced.

        The recorded value is detached, so telemetry can never change the action
        or the gradient. The *returned* value is normally the input unchanged, so
        ``g = self.record_gate("g", g)`` is a no-op in ordinary use. Under a gate
        override it returns the forced value instead, which is what makes a
        controlled substitution on a phase gate possible.
        """
        key = str(name)
        tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
        override = self.gate_overrides.get(key)
        if override is None:
            self.last_gates[key] = tensor.detach()
            return value
        forced = torch.full_like(tensor, float(override))
        self.last_gates[key] = forced.detach()
        return forced

    def gate_snapshot(self):
        """Detached copy of the gates recorded by the most recent ``forward``."""
        return {k: v.detach().clone() for k, v in self.last_gates.items()}

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
        """Raise TypeError if any forbidden NN module is found.

        This is the module blacklist only. For the full pre-rollout gate use
        :func:`lares.core.policy_validator.validate_policy`, which also checks
        parameter declaration, shapes, finiteness, batch independence and
        sensitivity to the observation fields the task requires.
        """
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
