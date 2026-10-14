"""Parameter and loss diagnostics shared by the fitter and the legacy BC path.

These measure a policy against expert transitions: per-axis error, which
parameters sit on a declared bound, and how the error splits by exposed gate.
They live here, not in the pipeline module, so fitting does not depend on the
orchestrator.
"""

from __future__ import annotations

import torch


def per_axis_mse(policy, obs_np, actions_np):
    """MSE between ``tanh(mean)`` and the expert action, per action axis."""
    with torch.no_grad():
        mean, _ = policy(torch.tensor(obs_np, dtype=torch.float32))
        target = torch.tensor(actions_np, dtype=torch.float32)
        per_axis = ((torch.tanh(mean) - target) ** 2).mean(dim=0)
    return [float(x) for x in per_axis]


def bound_activity(policy, atol=1e-6):
    """Fraction of each parameter's elements sitting on a declared bound.

    A parameter pinned at its bound is the optimiser saying the range is wrong,
    or that the structure needs a magnitude it is not allowed to have.
    """
    try:
        ranges = policy.get_param_ranges()
    except Exception:
        return None
    out = {}
    for name, param in policy.named_parameters():
        if name not in ranges:
            continue
        lo, hi = ranges[name]
        value = param.detach()
        at_low = (value <= lo + atol).float().mean().item()
        at_high = (value >= hi - atol).float().mean().item()
        out[name] = {
            "at_lower": float(at_low),
            "at_upper": float(at_high),
            "value": [float(x) for x in value.reshape(-1)],
            "range": [float(lo), float(hi)],
        }
    return out or None


def loss_by_phase(policy, obs_np, actions_np, threshold=0.5):
    """Split the fitting error by which exposed gate was open.

    Tells a structural failure from a fitting one: if error is concentrated in
    one phase, that phase's expression is the thing to repair.
    """
    reset_diagnostics = getattr(policy, "reset_diagnostics", None)
    if not callable(reset_diagnostics):
        return None
    reset_diagnostics()
    obs_t = torch.tensor(obs_np, dtype=torch.float32)
    target = torch.tensor(actions_np, dtype=torch.float32)
    with torch.no_grad():
        mean, _ = policy(obs_t)
        gates = dict(getattr(policy, "last_gates", {}) or {})
        if not gates:
            return None
        error = ((torch.tanh(mean) - target) ** 2).mean(dim=-1)
    out = {}
    for name, value in gates.items():
        flat = value.reshape(value.shape[0], -1).mean(dim=-1)
        if flat.shape[0] != error.shape[0]:
            continue
        open_mask = flat > threshold
        out[name] = {
            "loss_when_open": float(error[open_mask].mean()) if bool(open_mask.any()) else None,
            "loss_when_closed": (
                float(error[~open_mask].mean()) if bool((~open_mask).any()) else None
            ),
            "fraction_open": float(open_mask.float().mean()),
        }
    return out or None
