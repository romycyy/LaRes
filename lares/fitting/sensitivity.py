"""Action-versus-parameter sensitivity (``spec.md`` FR-6, AC-3).

Answers a question the loss curve cannot: which parameters actually move the
controller. A parameter the action is insensitive to cannot be tuned into a fix,
however long the optimizer runs, and a parameter pinned at a bound with high
sensitivity says the declared range is wrong rather than the structure.

Sensitivity is reported in units of the parameter's own declared range, so a
gain in ``[0.1, 10]`` and a threshold in ``[0.01, 0.3]`` are comparable.
"""

from __future__ import annotations

import numpy as np
import torch

#: Perturbation applied to each parameter, as a fraction of its declared range.
DEFAULT_PERTURBATION = 0.01


def action_sensitivity(
    policy, obs: torch.Tensor, perturbation: float = DEFAULT_PERTURBATION
) -> dict:
    """Mean absolute change in the executed action per parameter.

    Each parameter is nudged by ``perturbation`` of its own declared range and the
    change in ``tanh(mean)`` is measured. Central differences, so a parameter
    sitting at a bound is still measured rather than reading as dead.
    """
    ranges = policy.get_param_ranges()
    with torch.no_grad():
        baseline = torch.tanh(policy(obs)[0])

    out: dict[str, dict] = {}
    for name, param in policy.named_parameters():
        if name not in ranges:
            out[name] = {"unavailable": "no declared range"}
            continue
        lo, hi = float(ranges[name][0]), float(ranges[name][1])
        step = (hi - lo) * perturbation
        original = param.detach().clone()
        with torch.no_grad():
            param.add_(step)
            up = torch.tanh(policy(obs)[0])
            param.copy_(original)
            param.sub_(step)
            down = torch.tanh(policy(obs)[0])
            param.copy_(original)
        delta = (up - down).abs().mean().item() / 2.0
        out[name] = {
            "mean_abs_action_change": float(delta),
            "perturbation": float(step),
            "range_width": float(hi - lo),
            "relative_position": float(
                ((original.mean().item() - lo) / (hi - lo)) if hi > lo else float("nan")
            ),
            "dead": bool(delta < 1e-9),
        }
    # baseline is only needed to guarantee a forward pass happened first.
    del baseline
    return out


def gradient_sensitivity(policy, obs: torch.Tensor, actions: torch.Tensor) -> dict:
    """Per-parameter gradient magnitude of the fitting loss, scaled by range.

    Complements :func:`action_sensitivity`: a parameter can move the action a lot
    and still receive no gradient if the loss is flat in that direction.
    """
    from lares.fitting.optimizers import bc_objective

    ranges = policy.get_param_ranges()
    policy.zero_grad()
    loss = bc_objective(policy, obs, actions)
    loss.backward()
    out = {}
    for name, param in policy.named_parameters():
        grad = param.grad
        magnitude = float(grad.abs().mean()) if grad is not None else 0.0
        lo, hi = ranges.get(name, (0.0, 1.0))
        width = max(float(hi) - float(lo), 1e-12)
        out[name] = {
            "mean_abs_gradient": magnitude,
            "range_scaled_gradient": magnitude * width,
            "no_gradient": grad is None or magnitude == 0.0,
        }
    policy.zero_grad()
    return out


def summarise_sensitivity(action_stats: dict, gradient_stats: dict | None = None) -> str:
    """Table ordered by influence, most influential first."""
    rows = []
    for name, stats in action_stats.items():
        if "mean_abs_action_change" not in stats:
            continue
        grad = (gradient_stats or {}).get(name, {})
        rows.append(
            (
                name,
                stats["mean_abs_action_change"],
                stats["relative_position"],
                grad.get("range_scaled_gradient", float("nan")),
                stats["dead"],
            )
        )
    rows.sort(key=lambda r: r[1], reverse=True)
    lines = [
        f"{'parameter':<22}{'d|action|':>12}{'pos in range':>14}{'grad x range':>14}  note",
        "-" * 74,
    ]
    for name, delta, position, grad, dead in rows:
        note = "no influence" if dead else ""
        if not dead and (position < 0.02 or position > 0.98):
            note = "at a bound"
        lines.append(f"{name:<22}{delta:>12.6f}{position:>14.3f}{grad:>14.6f}  {note}")
    return "\n".join(lines)


def dead_parameters(action_stats: dict) -> list:
    """Parameters the executed action does not respond to at all."""
    return sorted(
        name for name, s in action_stats.items() if s.get("dead")
    )
