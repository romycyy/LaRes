"""Numerical fitting methods compared under a matched budget (FR-6, AC-3).

The structure is held fixed and only the optimizer changes, so a difference in
outcome is attributable to the fitting rather than to the controller. The point
of the comparison is stated in ``spec.md`` section 14: a promising structure must
not be rejected because one optimizer run from one initialisation failed.

Every method optimises the identical objective and is given the same number of
objective evaluations. Wall time and transitions processed are reported as well,
because equal evaluation counts do not mean equal compute and hiding that would
make the comparison look fairer than it is.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn

from lares.core.training_pipeline import (
    _bound_activity,
    _per_axis_mse,
    split_buffer_by_episode,
)

#: Names are stable: reports and the handoff table key on them.
ADAM_BASELINE = "adam_baseline"
ADAM_SCALED = "adam_scaled"
MULTISTART_ADAM = "multistart_adam"
DIFFERENTIAL_EVOLUTION = "differential_evolution"

METHODS = (ADAM_BASELINE, ADAM_SCALED, MULTISTART_ADAM, DIFFERENTIAL_EVOLUTION)

#: Checkpoint rules compared alongside the optimizers.
CHECKPOINT_FINAL = "final"
CHECKPOINT_BEST_VALIDATION = "best_validation"
#: Selects the parameters that scored best on *development rollouts* during
#: fitting. Needed because validation loss falls monotonically on this task while
#: development success peaks partway through and then collapses, so neither
#: "final" nor "best validation loss" can find the useful point.
CHECKPOINT_BEST_ROLLOUT = "best_rollout"
CHECKPOINT_RULES = (
    CHECKPOINT_FINAL,
    CHECKPOINT_BEST_VALIDATION,
    CHECKPOINT_BEST_ROLLOUT,
)


# ---------------------------------------------------------------------------
#  The shared objective
# ---------------------------------------------------------------------------


def bc_objective(policy, obs_t, act_t, std_weight: float = 0.01):
    """The live behavioural-cloning loss, identical for every method.

    ``MSE(tanh(mean), expert_action) + std_weight * mean(std)``. The mean term
    regresses the action that is actually executed. The scale term is the
    regulariser the current pipeline uses; ``spec.md`` FR-7 replaces it later,
    and keeping it here means the optimizer comparison is not entangled with an
    objective change.
    """
    mean, std = policy(obs_t)
    return nn.functional.mse_loss(torch.tanh(mean), act_t) + std_weight * std.mean()


@dataclass
class FittingData:
    """Train and validation tensors, split by complete episode (FR-6)."""

    train_obs: torch.Tensor
    train_actions: torch.Tensor
    val_obs: torch.Tensor | None
    val_actions: torch.Tensor | None
    train_indices: np.ndarray
    split_info: dict

    @property
    def has_validation(self) -> bool:
        return self.val_obs is not None

    def batch(self, batch_size: int, generator: np.random.Generator):
        idx = generator.integers(0, self.train_obs.shape[0], size=batch_size)
        return self.train_obs[idx], self.train_actions[idx]


def prepare_data(demo_buffer, val_fraction: float = 0.2, split_seed: int = 0) -> FittingData:
    """Materialise the tensors every method fits on, split by episode."""
    train_idx, val_idx, info = split_buffer_by_episode(
        demo_buffer, val_fraction=val_fraction, seed=split_seed
    )
    train_obs, train_actions = demo_buffer.take(train_idx)
    val_obs = val_actions = None
    if val_idx is not None and len(val_idx) > 0:
        v_obs, v_act = demo_buffer.take(val_idx)
        val_obs = torch.tensor(v_obs, dtype=torch.float32)
        val_actions = torch.tensor(v_act, dtype=torch.float32)
    return FittingData(
        train_obs=torch.tensor(train_obs, dtype=torch.float32),
        train_actions=torch.tensor(train_actions, dtype=torch.float32),
        val_obs=val_obs,
        val_actions=val_actions,
        train_indices=train_idx,
        split_info=info,
    )


def _evaluate(policy, obs, actions, std_weight=0.01) -> float:
    if obs is None:
        return float("nan")
    with torch.no_grad():
        return float(bc_objective(policy, obs, actions, std_weight))


# ---------------------------------------------------------------------------
#  Result
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    """What one fitting run produced, and what it cost."""

    method: str
    structure_id: str
    objective_evaluations: int
    transition_evaluations: int
    wall_time_seconds: float
    train_loss_final: float
    train_loss_best: float
    validation_loss_final: float
    validation_loss_best: float
    #: Evaluation index at which the training loss first came within 1% of its
    #: final value. ``None`` means it never settled inside the budget.
    converged_at: int | None
    restarts: int = 1
    final_state: dict = field(default_factory=dict, repr=False)
    best_state: dict = field(default_factory=dict, repr=False)
    best_rollout_state: dict = field(default_factory=dict, repr=False)
    best_rollout_score: float | None = None
    best_rollout_step: int | None = None
    rollout_history: list = field(default_factory=list, repr=False)
    history: list = field(default_factory=list, repr=False)
    gradient_norms: dict | None = None
    bound_activity: dict | None = None
    train_loss_by_axis: dict | None = None
    validation_loss_by_axis: dict | None = None
    notes: dict = field(default_factory=dict)

    def state_for(self, rule: str) -> dict:
        """Parameters selected by a checkpoint rule."""
        if rule not in CHECKPOINT_RULES:
            raise ValueError(f"checkpoint rule must be one of {CHECKPOINT_RULES}")
        if rule == CHECKPOINT_BEST_ROLLOUT and self.best_rollout_state:
            return self.best_rollout_state
        if rule == CHECKPOINT_BEST_VALIDATION and self.best_state:
            return self.best_state
        return self.final_state

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("final_state", None)
        d.pop("best_state", None)
        d.pop("best_rollout_state", None)
        d["history"] = self.history[-1:] if self.history else []
        return d


def _finish(
    method, structure, policy, data, history, start, objective_evals, transitions,
    best_state, best_val, restarts=1, gradient_norms=None, notes=None, std_weight=0.01,
):
    train_final = _evaluate(policy, data.train_obs, data.train_actions, std_weight)
    val_final = _evaluate(policy, data.val_obs, data.val_actions, std_weight)
    losses = [h[1] for h in history]
    converged_at = None
    if losses:
        target = losses[-1]
        tolerance = abs(target) * 0.01 + 1e-9
        for i, value in enumerate(losses):
            if abs(value - target) <= tolerance:
                converged_at = history[i][0]
                break
    axes = [f"axis_{i}" for i in range(policy.action_dim)]
    return FitResult(
        method=method,
        structure_id=structure.structure_id,
        objective_evaluations=objective_evals,
        transition_evaluations=transitions,
        wall_time_seconds=time.time() - start,
        train_loss_final=train_final,
        train_loss_best=min(losses) if losses else train_final,
        validation_loss_final=val_final,
        validation_loss_best=best_val if best_val is not None else val_final,
        converged_at=converged_at,
        restarts=restarts,
        final_state=copy.deepcopy(policy.state_dict()),
        best_state=best_state or copy.deepcopy(policy.state_dict()),
        history=history,
        gradient_norms=gradient_norms,
        bound_activity=_bound_activity(policy),
        train_loss_by_axis=dict(
            zip(axes, _per_axis_mse(policy, data.train_obs.numpy(), data.train_actions.numpy()))
        ),
        validation_loss_by_axis=(
            dict(zip(axes, _per_axis_mse(policy, data.val_obs.numpy(), data.val_actions.numpy())))
            if data.has_validation
            else None
        ),
        notes=notes or {},
    )


# ---------------------------------------------------------------------------
#  Gradient methods
# ---------------------------------------------------------------------------


def _run_adam(
    policy, data, budget, batch_size, lr, clip_grad_norm, seed, param_groups=None,
    scheduler_factory=None, val_every=0, std_weight=0.01,
):
    """Shared inner loop for the Adam variants. Returns bookkeeping, not a result."""
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(param_groups or policy.parameters(), lr=lr)
    scheduler = scheduler_factory(optimizer) if scheduler_factory else None
    history, grad_norms = [], []
    best_state, best_val = None, None

    policy.train()
    for step in range(budget):
        obs_t, act_t = data.batch(batch_size, rng)
        loss = bc_objective(policy, obs_t, act_t, std_weight)
        optimizer.zero_grad()
        loss.backward()
        max_norm = clip_grad_norm if clip_grad_norm > 0 else float("inf")
        grad_norms.append(float(torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm)))
        optimizer.step()
        policy.clip_params()
        if scheduler is not None:
            scheduler.step()
        history.append((step, float(loss)))

        if val_every and data.has_validation and (step + 1) % val_every == 0:
            value = _evaluate(policy, data.val_obs, data.val_actions, std_weight)
            if best_val is None or value < best_val:
                best_val, best_state = value, copy.deepcopy(policy.state_dict())
    return history, grad_norms, best_state, best_val


def _gradient_norm_summary(grad_norms, clip_grad_norm):
    if not grad_norms:
        return None
    arr = np.asarray(grad_norms, dtype=float)
    return {
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "final": float(arr[-1]),
        "fraction_clipped": float((arr > clip_grad_norm).mean()) if clip_grad_norm > 0 else 0.0,
    }


def fit_adam_baseline(
    structure, data, budget=2000, batch_size=256, lr=1e-3, clip_grad_norm=1.0,
    seed=0, schema=None, val_every=0, std_weight=0.01,
):
    """The current pipeline path, reproduced exactly as the named baseline.

    One Adam optimizer, one parameter group, learning rate 1e-3, no scheduler,
    no convergence check, no restarts, starting from the structure's own
    declared initial values (``spec.md`` FR-6).
    """
    start = time.time()
    policy = structure.build(schema)
    history, grad_norms, best_state, best_val = _run_adam(
        policy, data, budget, batch_size, lr, clip_grad_norm, seed,
        val_every=val_every, std_weight=std_weight,
    )
    return _finish(
        ADAM_BASELINE, structure, policy, data, history, start, budget,
        budget * batch_size, best_state, best_val,
        gradient_norms=_gradient_norm_summary(grad_norms, clip_grad_norm),
        notes={"lr": lr, "batch_size": batch_size, "scheduler": "none"},
        std_weight=std_weight,
    )


def fit_adam_scaled(
    structure, data, budget=2000, batch_size=256, lr=1e-3, clip_grad_norm=1.0,
    seed=0, schema=None, val_every=0, std_weight=0.01,
):
    """Adam with a per-parameter step size and a cosine schedule.

    A distance threshold bounded in ``[0.01, 0.3]`` and a sigmoid sharpness
    bounded in ``[1, 200]`` are three orders of magnitude apart, and the baseline
    gives both the same step. Here each parameter's learning rate is scaled by
    the width of its own declared range, so a step means the same fraction of the
    admissible interval whatever the units.
    """
    start = time.time()
    policy = structure.build(schema)
    ranges = policy.get_param_ranges()
    groups = []
    for name, param in policy.named_parameters():
        lo, hi = ranges.get(name, (0.0, 1.0))
        width = max(float(hi) - float(lo), 1e-8)
        groups.append({"params": [param], "lr": lr * width})
    history, grad_norms, best_state, best_val = _run_adam(
        policy, data, budget, batch_size, lr, clip_grad_norm, seed,
        param_groups=groups,
        scheduler_factory=lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, budget)
        ),
        val_every=val_every, std_weight=std_weight,
    )
    return _finish(
        ADAM_SCALED, structure, policy, data, history, start, budget,
        budget * batch_size, best_state, best_val,
        gradient_norms=_gradient_norm_summary(grad_norms, clip_grad_norm),
        notes={
            "lr_base": lr,
            "lr_scaling": "per-parameter, proportional to declared range width",
            "scheduler": "cosine",
            "batch_size": batch_size,
        },
        std_weight=std_weight,
    )


def perturb_within_ranges(policy, fraction: float, generator: torch.Generator):
    """Jitter every parameter by a fraction of its own declared range."""
    ranges = policy.get_param_ranges()
    with torch.no_grad():
        for name, param in policy.named_parameters():
            if name not in ranges:
                continue
            lo, hi = float(ranges[name][0]), float(ranges[name][1])
            span = (hi - lo) * fraction
            noise = (torch.rand(param.shape, generator=generator) * 2.0 - 1.0) * span
            param.add_(noise)
    policy.clip_params()


def fit_multistart_adam(
    structure, data, budget=2000, batch_size=256, lr=1e-3, clip_grad_norm=1.0,
    seed=0, schema=None, restarts=4, perturbation=0.25, val_every=0, std_weight=0.01,
):
    """Equal-budget multistart: the same total steps split across restarts.

    Each restart begins from the structure's declared values jittered within
    their own ranges. The best restart is chosen on *validation* loss, so the
    choice is not made on the data being fitted. This is the direct test of
    whether a structure was being rejected for its initialisation rather than
    its form.
    """
    start = time.time()
    restarts = max(1, int(restarts))
    # Spread the remainder so the total is exactly the declared budget. Integer
    # division alone silently under-spends by up to restarts-1 evaluations, which
    # would make this method look cheaper than the ones it is compared against.
    base, remainder = divmod(budget, restarts)
    allocation = [base + (1 if r < remainder else 0) for r in range(restarts)]
    if base == 0:
        allocation = [1] * restarts
    best_overall, best_score, best_policy = None, None, None
    all_history, all_grads = [], []
    torch_gen = torch.Generator().manual_seed(seed)
    spent = 0

    for r, steps in enumerate(allocation):
        policy = structure.build(schema)
        if r > 0:
            perturb_within_ranges(policy, perturbation, torch_gen)
        history, grads, _, _ = _run_adam(
            policy, data, steps, batch_size, lr, clip_grad_norm, seed + r,
            std_weight=std_weight,
        )
        offset = spent
        spent += steps
        all_history.extend([(offset + s, v) for s, v in history])
        all_grads.extend(grads)
        score = (
            _evaluate(policy, data.val_obs, data.val_actions, std_weight)
            if data.has_validation
            else _evaluate(policy, data.train_obs, data.train_actions, std_weight)
        )
        if best_score is None or score < best_score:
            best_score, best_policy = score, policy
            best_overall = copy.deepcopy(policy.state_dict())

    return _finish(
        MULTISTART_ADAM, structure, best_policy, data, all_history, start,
        spent, spent * batch_size,
        best_overall, best_score, restarts=restarts,
        gradient_norms=_gradient_norm_summary(all_grads, clip_grad_norm),
        notes={
            "restarts": restarts,
            "steps_per_restart": allocation,
            "perturbation_fraction_of_range": perturbation,
            "selection": "validation loss" if data.has_validation else "train loss",
            "batch_size": batch_size,
        },
        std_weight=std_weight,
    )


# ---------------------------------------------------------------------------
#  Derivative-free
# ---------------------------------------------------------------------------


def _flatten(policy):
    """Parameter vector, plus the bounds and shapes needed to rebuild it."""
    ranges = policy.get_param_ranges()
    vector, bounds, layout = [], [], []
    for name, param in policy.named_parameters():
        flat = param.detach().reshape(-1)
        lo, hi = ranges.get(name, (-10.0, 10.0))
        layout.append((name, param.shape, flat.numel()))
        vector.extend(float(x) for x in flat)
        bounds.extend([(float(lo), float(hi))] * flat.numel())
    return np.asarray(vector, dtype=float), bounds, layout


def _unflatten(policy, vector, layout):
    offset = 0
    with torch.no_grad():
        params = dict(policy.named_parameters())
        for name, shape, size in layout:
            chunk = np.asarray(vector[offset : offset + size], dtype=np.float32)
            params[name].copy_(torch.tensor(chunk).reshape(shape))
            offset += size


def fit_differential_evolution(
    structure, data, budget=2000, batch_size=256, seed=0, schema=None,
    popsize=12, std_weight=0.01, tol=0.0,
):
    """Derivative-free fitting with scipy's differential evolution.

    Chosen over CMA-ES as the first derivative-free integration because it ships
    with scipy, so the comparison adds no dependency (``spec.md`` open decision 4).

    The objective is a *fixed* batch drawn once from the training split. A
    resampled batch would make the objective stochastic, which population methods
    handle badly. The cost is that this method can overfit that batch, which is
    exactly what the reported validation loss will show.
    """
    from scipy.optimize import differential_evolution

    start = time.time()
    policy = structure.build(schema)
    x0, bounds, layout = _flatten(policy)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, data.train_obs.shape[0], size=min(batch_size, data.train_obs.shape[0]))
    obs_t, act_t = data.train_obs[idx], data.train_actions[idx]

    calls = {"n": 0}
    history = []

    def objective(vector):
        _unflatten(policy, vector, layout)
        with torch.no_grad():
            value = float(bc_objective(policy, obs_t, act_t, std_weight))
        calls["n"] += 1
        if calls["n"] % 50 == 0 or calls["n"] == 1:
            history.append((calls["n"], value))
        return value if math.isfinite(value) else 1e6

    n_params = len(x0)
    # scipy uses popsize * n_params individuals per generation, plus the initial
    # population. Solve for the generation count that lands on the budget.
    per_generation = max(1, popsize * n_params)
    maxiter = max(1, int(budget / per_generation) - 1)

    result = differential_evolution(
        objective,
        bounds,
        maxiter=maxiter,
        popsize=popsize,
        seed=seed,
        polish=False,
        tol=tol,
        init="sobol",
        x0=x0,
    )
    _unflatten(policy, result.x, layout)
    policy.clip_params()
    history.append((calls["n"], float(result.fun)))
    best_val = (
        _evaluate(policy, data.val_obs, data.val_actions, std_weight)
        if data.has_validation
        else None
    )
    return _finish(
        DIFFERENTIAL_EVOLUTION, structure, policy, data, history, start,
        calls["n"], calls["n"] * len(idx),
        copy.deepcopy(policy.state_dict()), best_val,
        notes={
            "popsize": popsize,
            "maxiter": maxiter,
            "n_parameters": n_params,
            "requested_budget": budget,
            "objective_batch": "fixed sample from the training split",
            "batch_size": int(len(idx)),
            "scipy_message": str(result.message),
        },
        std_weight=std_weight,
    )


FIT_FUNCTIONS = {
    ADAM_BASELINE: fit_adam_baseline,
    ADAM_SCALED: fit_adam_scaled,
    MULTISTART_ADAM: fit_multistart_adam,
    DIFFERENTIAL_EVOLUTION: fit_differential_evolution,
}


def fit(method: str, structure, data, **kwargs) -> FitResult:
    """Run one named fitting method on one frozen structure."""
    if method not in FIT_FUNCTIONS:
        raise ValueError(f"unknown fitting method {method!r}; expected one of {METHODS}")
    return FIT_FUNCTIONS[method](structure, data, **kwargs)


# ---------------------------------------------------------------------------
#  Objective-aware fitting (Phase 5)
# ---------------------------------------------------------------------------

SAMPLING_UNIFORM = "uniform"
SAMPLING_PHASE_BALANCED = "phase_balanced"
SAMPLING_STRATEGIES = (SAMPLING_UNIFORM, SAMPLING_PHASE_BALANCED)


def fit_with_objective(
    structure,
    data,
    objective: str,
    budget: int = 2000,
    batch_size: int = 256,
    lr: float = 1e-3,
    clip_grad_norm: float = 1.0,
    seed: int = 0,
    schema=None,
    sampling: str = SAMPLING_UNIFORM,
    phase_labels=None,
    val_every: int = 0,
    rollout_probe=None,
    rollout_every: int = 0,
):
    """Fit one structure under a named objective and sampling strategy.

    Adam throughout, so the objective and the sampling are the only things that
    vary. Scale parameters are frozen for objectives that declare they do not
    optimise the scale, found by measuring which parameters move the scale but
    not the executed action rather than by matching a name.

    ``residual_scale`` splits the budget in half: the mean is fitted first, then
    frozen while the scale is fitted to the residual.

    ``rollout_probe`` is a callable taking the policy and returning a development
    score. When supplied it is called every ``rollout_every`` steps and the best
    parameters are kept, which is the only checkpoint rule that can find the
    useful point on this task: development success peaks partway through fitting
    while validation loss keeps falling, so both "final" and "best validation
    loss" walk straight past it.
    """
    from lares.fitting.objectives import (
        RESIDUAL_SCALE,
        objective_loss,
        freeze_scale_parameters,
        spec,
        trainable_parameters,
    )
    from lares.fitting.phases import balanced_indices

    if sampling not in SAMPLING_STRATEGIES:
        raise ValueError(f"sampling must be one of {SAMPLING_STRATEGIES}, got {sampling!r}")
    if sampling == SAMPLING_PHASE_BALANCED and phase_labels is None:
        raise ValueError("phase-balanced sampling needs phase labels for the training split")

    start = time.time()
    objective_spec = spec(objective)
    policy = structure.build(schema)
    probe = data.train_obs[: min(64, data.train_obs.shape[0])]
    frozen = []
    if not objective_spec.optimises_scale:
        frozen = freeze_scale_parameters(policy, probe)

    rng = np.random.default_rng(seed)
    history, grad_norms, rollout_history = [], [], []
    best_state, best_val = None, None
    best_rollout_state, best_rollout_score, best_rollout_step = None, None, None

    def draw(size):
        if sampling == SAMPLING_PHASE_BALANCED:
            idx = balanced_indices(phase_labels, size, rng)
        else:
            idx = rng.integers(0, data.train_obs.shape[0], size=size)
        return data.train_obs[idx], data.train_actions[idx]

    def loss_for(step_index, obs_t, act_t):
        mean, std = policy(obs_t)
        if objective == RESIDUAL_SCALE:
            stage = "mean" if step_index < budget // 2 else "scale"
            return objective_loss(objective, mean, std, act_t, stage=stage)
        return objective_loss(objective, mean, std, act_t)

    optimizer = torch.optim.Adam(trainable_parameters(policy), lr=lr)
    switched = False
    policy.train()
    for step in range(budget):
        if objective == RESIDUAL_SCALE and not switched and step >= budget // 2:
            # Freeze the mean and hand the optimizer only the scale parameters,
            # so stage two cannot quietly undo stage one.
            scale_names = set(freeze_scale_parameters(policy, probe))
            for name, param in policy.named_parameters():
                param.requires_grad_(name in scale_names)
            optimizer = torch.optim.Adam(trainable_parameters(policy), lr=lr)
            switched = True
            frozen = [n for n, _ in policy.named_parameters() if n not in scale_names]
        obs_t, act_t = draw(batch_size)
        loss = loss_for(step, obs_t, act_t)
        optimizer.zero_grad()
        loss.backward()
        params = trainable_parameters(policy)
        if params:
            max_norm = clip_grad_norm if clip_grad_norm > 0 else float("inf")
            grad_norms.append(float(torch.nn.utils.clip_grad_norm_(params, max_norm)))
        optimizer.step()
        policy.clip_params()
        history.append((step, float(loss)))

        if val_every and data.has_validation and (step + 1) % val_every == 0:
            with torch.no_grad():
                mean, std = policy(data.val_obs)
                value = float(objective_loss(objective, mean, std, data.val_actions))
            if best_val is None or value < best_val:
                best_val, best_state = value, copy.deepcopy(policy.state_dict())

        if rollout_probe is not None and rollout_every and (step + 1) % rollout_every == 0:
            was_training = policy.training
            policy.eval()
            score = float(rollout_probe(policy))
            policy.train(was_training)
            rollout_history.append((step + 1, score))
            if best_rollout_score is None or score > best_rollout_score:
                best_rollout_score = score
                best_rollout_step = step + 1
                best_rollout_state = copy.deepcopy(policy.state_dict())

    for _, param in policy.named_parameters():
        param.requires_grad_(True)

    result = _finish(
        f"{objective}:{sampling}", structure, policy, data, history, start,
        budget, budget * batch_size, best_state, best_val,
        gradient_norms=_gradient_norm_summary(grad_norms, clip_grad_norm),
        notes={
            "objective": objective,
            "execution": objective_spec.execution,
            "is_likelihood": objective_spec.is_likelihood,
            "optimises_scale": objective_spec.optimises_scale,
            "excludes_censored": objective_spec.excludes_censored,
            "frozen_parameters": frozen,
            "sampling": sampling,
            "batch_size": batch_size,
            "lr": lr,
            "rollout_probe_every": rollout_every if rollout_probe is not None else 0,
        },
    )
    result.best_rollout_state = best_rollout_state or {}
    result.best_rollout_score = best_rollout_score
    result.best_rollout_step = best_rollout_step
    result.rollout_history = rollout_history
    return result
