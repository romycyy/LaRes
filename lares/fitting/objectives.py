"""Fitting objectives, each bound to the execution it matches (FR-7, AC-5).

The rule this module enforces: an objective may only be called a likelihood if
the policy is executed by sampling from the distribution that objective assumes.
Every objective therefore carries an execution contract, and evaluation reads it
rather than defaulting to whatever the pipeline happened to do before.

What the audit in ``scripts/audit_expert_actions.py`` established:

* MetaWorld's expert is an unbounded proportional controller with gain 10. It
  does not clip. Our Stage 1 clips before storing, so 18.8% of stored targets sit
  exactly at an action endpoint.
* The expert is deterministic. ``p(a|s)`` is a point mass, so there is no
  aleatoric spread to fit and maximum likelihood drives any learned scale to zero.

Two consequences. A tanh-Gaussian likelihood is not the matching model: the
generator is not a tanh squash and ``atanh(+-1)`` is undefined on the censored
fraction. And a censored Gaussian *is* the matching model, but only with clipped
execution, which is a different contract from the tanh sampler the pipeline uses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
#  Execution contracts
# ---------------------------------------------------------------------------

EXEC_DETERMINISTIC_TANH = "deterministic_tanh"
EXEC_TANH_GAUSSIAN = "tanh_gaussian"
EXEC_CLIPPED_GAUSSIAN = "clipped_gaussian"
EXEC_CLIPPED_DETERMINISTIC = "clipped_deterministic"
EXECUTION_CONTRACTS = (
    EXEC_DETERMINISTIC_TANH,
    EXEC_TANH_GAUSSIAN,
    EXEC_CLIPPED_GAUSSIAN,
    EXEC_CLIPPED_DETERMINISTIC,
)

EXECUTION_DESCRIPTION = {
    EXEC_DETERMINISTIC_TANH: "a = tanh(mean); the scale is ignored",
    EXEC_TANH_GAUSSIAN: "a = tanh(mean + std * eps)",
    EXEC_CLIPPED_GAUSSIAN: "a = clip(mean + std * eps, -1, 1)",
    EXEC_CLIPPED_DETERMINISTIC: "a = clip(mean, -1, 1)",
}


def execute(mean, std, contract: str, generator=None):
    """Produce the action a contract prescribes, given ``(mean, std)``."""
    if contract == EXEC_DETERMINISTIC_TANH:
        return torch.tanh(mean)
    if contract == EXEC_CLIPPED_DETERMINISTIC:
        return torch.clamp(mean, -1.0, 1.0)
    eps = torch.randn(mean.shape, generator=generator)
    if contract == EXEC_TANH_GAUSSIAN:
        return torch.tanh(mean + std * eps)
    if contract == EXEC_CLIPPED_GAUSSIAN:
        return torch.clamp(mean + std * eps, -1.0, 1.0)
    raise ValueError(f"unknown execution contract {contract!r}; expected one of {EXECUTION_CONTRACTS}")


# ---------------------------------------------------------------------------
#  Objective definitions
# ---------------------------------------------------------------------------

DETERMINISTIC_MSE = "deterministic_mse"
LEGACY_MSE_STD_PENALTY = "legacy_mse_std_penalty"
TANH_GAUSSIAN_NLL = "tanh_gaussian_nll"
RESIDUAL_SCALE = "residual_scale"
CENSORED_GAUSSIAN = "censored_gaussian"

OBJECTIVES = (
    DETERMINISTIC_MSE,
    LEGACY_MSE_STD_PENALTY,
    TANH_GAUSSIAN_NLL,
    RESIDUAL_SCALE,
    CENSORED_GAUSSIAN,
)

#: Actions this close to an endpoint are treated as censored.
ENDPOINT_TOLERANCE = 1e-6
#: Smallest scale any objective is allowed to assume, keeping logs finite.
MIN_SCALE = 1e-4


@dataclass(frozen=True)
class ObjectiveSpec:
    """One objective plus the execution it is only valid under."""

    name: str
    execution: str
    is_likelihood: bool
    optimises_scale: bool
    #: Set when the objective cannot see part of the data, so a loss comparison
    #: against another objective is not like for like.
    excludes_censored: bool = False
    note: str = ""

    def describe(self) -> str:
        kind = "likelihood" if self.is_likelihood else "regression"
        return (
            f"{self.name}: {kind}, executed as {EXECUTION_DESCRIPTION[self.execution]}"
            + (f". {self.note}" if self.note else "")
        )


SPECS = {
    DETERMINISTIC_MSE: ObjectiveSpec(
        name=DETERMINISTIC_MSE,
        execution=EXEC_DETERMINISTIC_TANH,
        is_likelihood=False,
        optimises_scale=False,
        note=(
            "The primary baseline. Regresses the action that is actually executed. "
            "The scale is returned only to satisfy the (mean, std) interface and is "
            "excluded from optimisation, so it is not a fitted quantity and must not "
            "be read as uncertainty."
        ),
    ),
    LEGACY_MSE_STD_PENALTY: ObjectiveSpec(
        name=LEGACY_MSE_STD_PENALTY,
        execution=EXEC_DETERMINISTIC_TANH,
        is_likelihood=False,
        optimises_scale=True,
        note=(
            "What the pipeline used before. The scale term is a downward regulariser "
            "with no likelihood behind it: it pushes the scale to its lower bound "
            "whatever the data says. Kept only as a named comparison."
        ),
    ),
    TANH_GAUSSIAN_NLL: ObjectiveSpec(
        name=TANH_GAUSSIAN_NLL,
        execution=EXEC_TANH_GAUSSIAN,
        is_likelihood=True,
        optimises_scale=True,
        excludes_censored=True,
        note=(
            "Correct change of variables for a tanh-squashed Gaussian, but only "
            "defined on interior actions: atanh(+-1) diverges, so the 18.8% of "
            "targets our clipping put on the boundary are dropped. It also assumes a "
            "generator that squashes through tanh, which the expert does not."
        ),
    ),
    RESIDUAL_SCALE: ObjectiveSpec(
        name=RESIDUAL_SCALE,
        execution=EXEC_DETERMINISTIC_TANH,
        is_likelihood=False,
        optimises_scale=True,
        note=(
            "Two stages: fit the mean by regression, then fit the scale to the "
            "policy's own residual with the mean frozen. The scale then measures "
            "where this structure cannot express the expert, which is a useful "
            "diagnostic. It is residual scale, not aleatoric uncertainty: the expert "
            "has none."
        ),
    ),
    CENSORED_GAUSSIAN: ObjectiveSpec(
        name=CENSORED_GAUSSIAN,
        execution=EXEC_CLIPPED_GAUSSIAN,
        is_likelihood=True,
        optimises_scale=True,
        note=(
            "The model that matches how the targets were made: an unbounded Gaussian "
            "in action space, censored at the limits by our clip. Interior actions "
            "contribute a density, endpoint actions a tail probability. Requires "
            "clipped execution; it is not valid under the tanh sampler."
        ),
    ),
}


# ---------------------------------------------------------------------------
#  Scale parameters
# ---------------------------------------------------------------------------


def find_scale_parameters(policy, probe_obs) -> list:
    """Parameters that affect the scale but not the executed action.

    Found by measurement rather than by name. A parameter matched on a name like
    ``log_std`` would miss ``std_shrink`` or ``alpha_std``, and would wrongly
    catch a control parameter that happened to be named similarly.
    """
    from lares.fitting.sensitivity import action_sensitivity

    action_stats = action_sensitivity(policy, probe_obs)
    with torch.no_grad():
        base_std = policy(probe_obs)[1].clone()

    scale_params = []
    ranges = policy.get_param_ranges()
    for name, param in policy.named_parameters():
        if not action_stats.get(name, {}).get("dead"):
            continue
        lo, hi = ranges.get(name, (0.0, 1.0))
        step = (float(hi) - float(lo)) * 0.01
        original = param.detach().clone()
        with torch.no_grad():
            param.add_(step)
            moved = float((policy(probe_obs)[1] - base_std).abs().mean())
            param.copy_(original)
        if moved > 1e-9:
            scale_params.append(name)
    return scale_params


def freeze_scale_parameters(policy, probe_obs) -> list:
    """Exclude scale parameters from optimisation and report which were frozen."""
    names = find_scale_parameters(policy, probe_obs)
    lookup = dict(policy.named_parameters())
    for name in names:
        lookup[name].requires_grad_(False)
    return names


def trainable_parameters(policy):
    return [p for p in policy.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
#  Losses
# ---------------------------------------------------------------------------


def _split_censored(actions):
    """Boolean masks for interior and endpoint targets."""
    at_bound = actions.abs() >= (1.0 - ENDPOINT_TOLERANCE)
    return ~at_bound, at_bound


def deterministic_mse(mean, std, actions, **kwargs):
    """``MSE(tanh(mean), a)``. The scale plays no part."""
    return nn.functional.mse_loss(torch.tanh(mean), actions)


def legacy_mse_std_penalty(mean, std, actions, std_weight: float = 0.01, **kwargs):
    """The previous objective, kept only for comparison."""
    return nn.functional.mse_loss(torch.tanh(mean), actions) + std_weight * std.mean()


def tanh_gaussian_nll(mean, std, actions, **kwargs):
    """Negative log-likelihood of a tanh-squashed Gaussian, interior actions only.

    ``-log N(atanh(a); mean, std) + log(1 - a^2)``, the change of variables from
    Soft Actor-Critic appendix C. Endpoint targets are excluded because
    ``atanh`` diverges there; the excluded fraction is what
    :func:`censored_fraction` reports.
    """
    interior, _ = _split_censored(actions)
    if not bool(interior.any()):
        return torch.zeros((), device=mean.device, dtype=mean.dtype)
    a = actions[interior].clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    scale = std[interior].clamp_min(MIN_SCALE)
    location = mean[interior]
    pre_tanh = torch.atanh(a)
    log_density = -(
        0.5 * ((pre_tanh - location) / scale) ** 2
        + torch.log(scale)
        + 0.5 * math.log(2 * math.pi)
    )
    log_jacobian = torch.log1p(-a.pow(2))
    return -(log_density - log_jacobian).mean()


def residual_scale_loss(mean, std, actions, stage: str = "mean", **kwargs):
    """Two-stage heteroscedastic fit in action space.

    Stage ``mean`` is plain regression. Stage ``scale`` freezes the mean and fits
    the scale to the policy's own residual, so it ends up large where the
    structure cannot express the expert.
    """
    executed = torch.tanh(mean)
    if stage == "mean":
        return nn.functional.mse_loss(executed, actions)
    residual = (executed - actions).abs().detach()
    return nn.functional.mse_loss(std, residual.clamp_min(MIN_SCALE))


def censored_gaussian_nll(mean, std, actions, **kwargs):
    """Negative log-likelihood under clipped execution.

    ``mean`` is the location in *action* space, unbounded, matching the expert's
    unbounded proportional response. Interior targets contribute a density,
    endpoint targets the probability of the corresponding tail, which is exactly
    the event our clip records.
    """
    interior, at_bound = _split_censored(actions)
    scale = std.clamp_min(MIN_SCALE)
    total = torch.zeros((), device=mean.device, dtype=mean.dtype)
    count = 0

    if bool(interior.any()):
        z = (actions[interior] - mean[interior]) / scale[interior]
        log_density = -(
            0.5 * z.pow(2) + torch.log(scale[interior]) + 0.5 * math.log(2 * math.pi)
        )
        total = total - log_density.sum()
        count += int(interior.sum())

    if bool(at_bound.any()):
        # Standard normal tail beyond the limit the clip imposed.
        sign = torch.sign(actions[at_bound])
        limit = sign  # +1 or -1
        z = (limit - mean[at_bound]) / scale[at_bound]
        # log P(u >= limit) for the upper bound, log P(u <= limit) for the lower.
        tail = 0.5 * torch.erfc(sign * z / math.sqrt(2.0))
        total = total - torch.log(tail.clamp_min(1e-12)).sum()
        count += int(at_bound.sum())

    return total / max(count, 1)


LOSS_FUNCTIONS = {
    DETERMINISTIC_MSE: deterministic_mse,
    LEGACY_MSE_STD_PENALTY: legacy_mse_std_penalty,
    TANH_GAUSSIAN_NLL: tanh_gaussian_nll,
    RESIDUAL_SCALE: residual_scale_loss,
    CENSORED_GAUSSIAN: censored_gaussian_nll,
}


def objective_loss(name: str, mean, std, actions, **kwargs):
    """Evaluate a named objective."""
    if name not in LOSS_FUNCTIONS:
        raise ValueError(f"unknown objective {name!r}; expected one of {OBJECTIVES}")
    return LOSS_FUNCTIONS[name](mean, std, actions, **kwargs)


def spec(name: str) -> ObjectiveSpec:
    if name not in SPECS:
        raise ValueError(f"unknown objective {name!r}; expected one of {OBJECTIVES}")
    return SPECS[name]


def censored_fraction(actions) -> float:
    """Share of targets sitting on an action endpoint."""
    arr = actions if torch.is_tensor(actions) else torch.as_tensor(np.asarray(actions))
    _, at_bound = _split_censored(arr)
    return float(at_bound.any(dim=-1).float().mean())


def check_execution_matches(objective: str, execution: str) -> None:
    """Refuse to score an objective under an execution it does not model.

    Calling an action-space regression a likelihood, or scoring a censored model
    with a tanh sampler, is the mistake ``spec.md`` FR-7 singles out.
    """
    expected = spec(objective).execution
    if execution != expected:
        raise ValueError(
            f"objective {objective!r} assumes execution {expected!r} "
            f"({EXECUTION_DESCRIPTION[expected]}), but the policy would be executed as "
            f"{execution!r} ({EXECUTION_DESCRIPTION.get(execution, 'unknown')}). "
            f"An objective and its execution must describe the same distribution."
        )
