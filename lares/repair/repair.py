"""Bounded repairs generated from a supported explanation (FR-9, AC-6).

A diagnosis names a component. A repair changes it, and nothing else. This
module makes "nothing else" checkable:

* The component is turned into a *scope*: which action axes it drives and which
  phases it acts in.
* The parameters inside that scope are found by measurement, not by name, so a
  repair cannot be aimed at a parameter that does not move the thing it is
  supposed to move.
* The refit sees only the scope's phases and only the scope's axes, and every
  parameter outside the scope is frozen. :func:`bounded_change` re-reads the
  policy afterwards and reports what actually moved.

Acceptance is decided by :func:`~lares.repair.hypotheses.evaluate_repair`, plus a
guard for the case this task keeps producing: a policy with no successes has an
empty protected set, and a regression threshold no case can fail is not a check.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from lares.repair.hypotheses import PROTECTED_REGRESSION_TOLERANCE, evaluate_repair

#: A parameter joins a repair's scope if it moves the scoped action at least this
#: fraction of the most influential parameter's effect. Declared here so a repair
#: cannot widen its own boundary.
SCOPE_SENSITIVITY_FRACTION = 0.05

#: Perturbation used when measuring scoped sensitivity, as a fraction of range.
SCOPE_PERTURBATION = 0.01

#: Protected cases may lose at most this much mean signed goal progress when no
#: case succeeded and the success threshold is therefore vacuous.
PROTECTED_PROGRESS_TOLERANCE = 0.01

ALL_AXES = (0, 1, 2, 3)

#: What each suspected component means in terms the fitter can act on. Keyed by
#: the ``suspected_component`` string the explanations declare.
COMPONENT_SCOPES = {
    "the approach term": {"axes": (0, 1, 2), "phases": ("hover", "descend")},
    "the vertical action dimension": {"axes": (2,), "phases": ()},
    "the push direction term": {"axes": (0, 1), "phases": ("push",)},
    "the planar action dimensions": {"axes": (0, 1), "phases": ()},
    "the push magnitude": {"axes": (0, 1, 2), "phases": ("push",)},
}


def scope_for(suspected_component: str) -> dict | None:
    """The axes and phases a component owns, or ``None`` if it owns none.

    A gate threshold has no fixed axes or phases: its parameters are found by
    measuring the gate itself, which :func:`gate_parameters` does.
    """
    if suspected_component in COMPONENT_SCOPES:
        return dict(COMPONENT_SCOPES[suspected_component])
    if suspected_component.startswith("the ") and suspected_component.endswith(" gate threshold"):
        return {"axes": ALL_AXES, "phases": (), "gate": suspected_component[4:-15]}
    return None


# ---------------------------------------------------------------------------
#  Measured parameter selection
# ---------------------------------------------------------------------------


def scoped_sensitivity(policy, obs, axes=ALL_AXES, perturbation: float = SCOPE_PERTURBATION) -> dict:
    """Mean absolute change in the executed action, restricted to ``axes``.

    :func:`lares.fitting.sensitivity.action_sensitivity` averages over every
    axis, which hides a parameter that moves only the axis under suspicion.
    """
    ranges = policy.get_param_ranges()
    columns = list(axes) if len(axes) else list(ALL_AXES)
    out = {}
    for name, param in policy.named_parameters():
        if name not in ranges:
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
        cols = [c for c in columns if c < up.shape[1]]
        delta = (up[:, cols] - down[:, cols]).abs().mean().item() / 2.0
        out[name] = float(delta)
    return out


def gate_parameters(policy, gate: str, obs, perturbation: float = SCOPE_PERTURBATION) -> list:
    """Parameters that move a named gate's recorded value."""
    ranges = policy.get_param_ranges()
    reset = getattr(policy, "reset_diagnostics", None)

    def gate_value():
        if callable(reset):
            reset()
        with torch.no_grad():
            policy(obs)
        recorded = (getattr(policy, "last_gates", {}) or {}).get(gate)
        return None if recorded is None else float(recorded.float().mean())

    base = gate_value()
    if base is None:
        return []
    chosen = []
    for name, param in policy.named_parameters():
        if name not in ranges:
            continue
        lo, hi = float(ranges[name][0]), float(ranges[name][1])
        original = param.detach().clone()
        with torch.no_grad():
            param.add_((hi - lo) * perturbation)
        moved = gate_value()
        with torch.no_grad():
            param.copy_(original)
        if moved is not None and abs(moved - base) > 1e-9:
            chosen.append(name)
    return sorted(chosen)


def select_parameters(policy, obs, axes=ALL_AXES, fraction: float = SCOPE_SENSITIVITY_FRACTION) -> list:
    """Parameters influential enough on the scoped axes to be worth repairing."""
    sensitivity = scoped_sensitivity(policy, obs, axes)
    if not sensitivity:
        return []
    strongest = max(sensitivity.values())
    if strongest <= 0.0:
        return []
    return sorted(n for n, v in sensitivity.items() if v >= fraction * strongest)


# ---------------------------------------------------------------------------
#  The repair itself
# ---------------------------------------------------------------------------


@dataclass
class BoundedRepair:
    """One edit, restricted to the parameters of one component."""

    explanation: str
    suspected_component: str
    parameters: list
    axes: tuple
    phases: tuple
    focused_metric: str
    predicted_direction: str
    kind: str = "scoped_refit"
    #: What actually changed once the refit ran.
    changed: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    note: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.parameters

    def to_dict(self) -> dict:
        d = asdict(self)
        d["axes"] = list(self.axes)
        d["phases"] = list(self.phases)
        return d


def propose_repair(policy, explanation, data, schema) -> BoundedRepair:
    """Turn a supported explanation into a bounded, parameter-level edit."""
    scope = scope_for(explanation.suspected_component)
    if scope is None:
        return BoundedRepair(
            explanation=explanation.name,
            suspected_component=explanation.suspected_component,
            parameters=[], axes=(), phases=(),
            focused_metric=explanation.focused_metric,
            predicted_direction=explanation.predicted_direction,
            note="no scope is defined for this component, so no bounded edit exists",
        )

    axes = tuple(scope["axes"])
    phases = tuple(scope["phases"])
    obs = phase_subset(data.train_obs, phases, schema)[0]
    probe = obs[: min(512, obs.shape[0])]
    if "gate" in scope:
        parameters = gate_parameters(policy, scope["gate"], probe)
        note = f"parameters selected by their effect on the {scope['gate']} gate"
    else:
        parameters = select_parameters(policy, probe, axes)
        note = "parameters selected by their measured effect on the scoped axes"
    return BoundedRepair(
        explanation=explanation.name,
        suspected_component=explanation.suspected_component,
        parameters=parameters,
        axes=axes,
        phases=phases,
        focused_metric=explanation.focused_metric,
        predicted_direction=explanation.predicted_direction,
        note=note,
    )


def phase_subset(obs, phases, schema):
    """Rows of ``obs`` in the named phases, and the indices that selected them."""
    if not phases:
        return obs, np.arange(obs.shape[0])
    from lares.fitting.phases import PHASE_NAMES, expert_phase_labels

    wanted = {PHASE_NAMES.index(p) for p in phases}
    labels = expert_phase_labels(obs.numpy(), schema)
    keep = np.array([i for i, l in enumerate(labels) if int(l) in wanted], dtype=np.int64)
    if keep.size == 0:
        return obs, np.arange(obs.shape[0])
    return obs[keep], keep


def apply_repair(
    policy,
    repair: BoundedRepair,
    data,
    schema,
    budget: int = 400,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
):
    """Refit only the repair's parameters, on its phases, against its axes.

    Returns a new policy. The original is left untouched so the two can be
    evaluated as a paired before/after.
    """
    if repair.is_empty:
        repair.cost = {"objective_evaluations": 0, "transition_evaluations": 0}
        return copy.deepcopy(policy)

    repaired = copy.deepcopy(policy)
    before = {n: p.detach().clone() for n, p in repaired.named_parameters()}

    _, keep = phase_subset(data.train_obs, repair.phases, schema)
    obs = data.train_obs[keep]
    actions = data.train_actions[keep]
    columns = [a for a in repair.axes if a < actions.shape[1]] or list(ALL_AXES)

    wanted = set(repair.parameters)
    for name, param in repaired.named_parameters():
        param.requires_grad_(name in wanted)
    trainable = [p for p in repaired.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)

    rng = np.random.default_rng(seed)
    repaired.train()
    for _ in range(budget):
        idx = rng.integers(0, obs.shape[0], size=min(batch_size, obs.shape[0]))
        mean, _ = repaired(obs[idx])
        loss = torch.nn.functional.mse_loss(
            torch.tanh(mean)[:, columns], actions[idx][:, columns]
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        repaired.clip_params()

    for _, param in repaired.named_parameters():
        param.requires_grad_(True)
    repaired.eval()

    repair.changed = bounded_change(before, repaired)
    repair.cost = {
        "objective_evaluations": budget,
        "transition_evaluations": budget * min(batch_size, int(obs.shape[0])),
        "transitions_available": int(obs.shape[0]),
    }
    return repaired


def learner_state_data(
    policy, env, cases, env_name: str, schema, max_steps: int = 150, base_buffer=None
):
    """Expert labels on the states the *learner* reaches, for a repair to fit.

    A repair aimed at an approach that never arrives cannot be fitted on the
    expert's own trajectories: those states are the ones the learner fails to
    reach. This collects the learner's states on training placements and labels
    them, so the repair is fitted where the failure happens.

    ``cases`` must be training placements. Development or final-test cases here
    would fit the repair on the cases it is later judged on.

    Returns ``(FittingData, AggregationStats)``.
    """
    from lares.fitting.dagger import aggregate_learner_states
    from lares.fitting.optimizers import prepare_data

    buffer, stats = aggregate_learner_states(
        policy, env, cases, env_name, schema, buffer=base_buffer, max_steps=max_steps
    )
    return prepare_data(buffer, val_fraction=0.0), stats


REPAIR_REFIT = "scoped_refit"
REPAIR_SEARCH = "scoped_search"


def scalar_parameters(policy, names) -> list:
    """The named parameters that hold a single number.

    The search moves one coordinate per parameter, so a vector parameter would
    be collapsed to a single value by :func:`_write_normalised`. Several bank
    structures declare a four-element ``log_std``. Dropping them is honest;
    silently flattening them is not.
    """
    wanted = set(names)
    return sorted(
        name for name, param in policy.named_parameters()
        if name in wanted and param.numel() == 1
    )


def _normalised(policy, names):
    ranges = policy.get_param_ranges()
    out = {}
    for name, param in policy.named_parameters():
        if name in names:
            lo, hi = float(ranges[name][0]), float(ranges[name][1])
            width = max(hi - lo, 1e-12)
            out[name] = (float(param.detach().mean()) - lo) / width
    return out


def _write_normalised(policy, values):
    ranges = policy.get_param_ranges()
    with torch.no_grad():
        for name, param in policy.named_parameters():
            if name in values:
                lo, hi = float(ranges[name][0]), float(ranges[name][1])
                param.fill_(lo + float(np.clip(values[name], 0.0, 1.0)) * (hi - lo))
    policy.clip_params()


def search_repair(
    policy,
    repair: BoundedRepair,
    manifest,
    env,
    focused_case_ids=None,
    budget_episodes: int = 300,
    population: int = 5,
    sigma: float = 0.25,
    decay: float = 0.8,
    seed: int = 0,
    max_steps: int = 150,
):
    """Search the in-scope parameters against the focused metric itself.

    The imitation refit in :func:`apply_repair` can only move a parameter to
    where the expert's actions put it. When the diagnosis says a *geometry* term
    is wrong, that is the wrong target: the expert data is exactly what produced
    the current value. This searches the same bounded set of parameters directly
    against the metric the explanation predicted, on the cases that failed.

    The budget is spent in whole evaluations of ``manifest``, so a candidate is
    always scored on the same ordered cases as every other. Cost is returned in
    episodes; nothing here is free.

    Returns ``(repaired_policy, trace)``.
    """
    from lares.eval.runner import PolicyActor, evaluate_manifest
    from lares.repair.hypotheses import episode_metric_mean

    repaired = copy.deepcopy(policy)
    before = {n: p.detach().clone() for n, p in repaired.named_parameters()}
    repair.kind = REPAIR_SEARCH
    searchable = scalar_parameters(repaired, repair.parameters)
    dropped = sorted(set(repair.parameters) - set(searchable))
    if dropped:
        repair.parameters = searchable
        repair.note = (
            f"{repair.note}; dropped {dropped} from the search because a vector "
            f"parameter cannot be moved along one coordinate"
        ).lstrip("; ")
    if repair.is_empty:
        repair.cost = {"rollout_episodes": 0, "evaluations": 0}
        return repaired, []

    per_evaluation = len(manifest)
    evaluations = max(1, budget_episodes // per_evaluation)
    sign = 1.0 if repair.predicted_direction == "increase" else -1.0
    rng = np.random.default_rng(seed)

    def score(candidate_policy):
        result = evaluate_manifest(
            PolicyActor(candidate_policy, deterministic=True, name="search"),
            manifest, env, max_steps=max_steps,
        )
        focused = episode_metric_mean(result, repair.focused_metric, focused_case_ids)
        value = float("-inf") if focused is None else sign * float(focused)
        return (result.success_rate, value), result

    current = _normalised(repaired, set(repair.parameters))
    best_score, _ = score(repaired)
    spent = 1
    best_values = dict(current)
    trace = [{"evaluation": 0, "values": dict(current), "success": best_score[0],
              "focused": sign * best_score[1]}]

    scale = sigma
    while spent < evaluations:
        # A short final generation still spends the budget. Stopping a whole
        # generation early would leave an arm quietly under-spent, and a
        # matched-cost comparison between arms that spend different amounts is
        # not matched.
        this_generation = min(population, evaluations - spent)
        candidates = []
        for _ in range(this_generation):
            proposal = {
                name: float(np.clip(value + rng.normal(0.0, scale), 0.0, 1.0))
                for name, value in best_values.items()
            }
            _write_normalised(repaired, proposal)
            candidate_score, _ = score(repaired)
            spent += 1
            candidates.append((candidate_score, proposal))
        candidates.sort(key=lambda c: c[0], reverse=True)
        if candidates[0][0] > best_score:
            best_score, best_values = candidates[0]
        trace.append({
            "evaluation": spent, "values": dict(best_values),
            "success": best_score[0], "focused": sign * best_score[1], "sigma": scale,
        })
        scale *= decay

    _write_normalised(repaired, best_values)
    repaired.eval()
    repair.changed = bounded_change(before, repaired)
    repair.cost = {
        "rollout_episodes": spent * per_evaluation,
        "evaluations": spent,
        "population": population,
        "sigma": sigma,
    }
    return repaired, trace


def bounded_change(before: dict, policy) -> dict:
    """Which parameters moved, and by how much. The boundary check."""
    out = {}
    for name, param in policy.named_parameters():
        original = before.get(name)
        if original is None:
            continue
        delta = float((param.detach() - original).abs().max())
        if delta > 0.0:
            out[name] = delta
    return out


def repair_stayed_in_scope(repair: BoundedRepair) -> bool:
    """Did the edit touch only the parameters it declared?"""
    return set(repair.changed) <= set(repair.parameters)


# ---------------------------------------------------------------------------
#  Acceptance
# ---------------------------------------------------------------------------


def protected_cases(before_result, cluster_case_ids) -> tuple:
    """The cases a repair must not break, and the basis for that judgement."""
    cluster = set(cluster_case_ids or [])
    succeeded = [e.case_id for e in before_result.episodes if e.success > 0]
    if succeeded:
        return succeeded, "success"
    outside = [e.case_id for e in before_result.episodes if e.case_id not in cluster]
    return outside, "no case succeeded, so the success threshold cannot fail"


def progress_regression(before_result, after_result, case_ids, metric="signed_goal_progress") -> dict:
    """Mean drop in a continuous metric over the protected cases."""
    from lares.repair.hypotheses import episode_metric_mean

    before = episode_metric_mean(before_result, metric, case_ids)
    after = episode_metric_mean(after_result, metric, case_ids)
    if before is None or after is None:
        return {"measurable": False, "metric": metric}
    drop = float(before - after)
    return {
        "measurable": True,
        "metric": metric,
        "before": float(before),
        "after": float(after),
        "drop": drop,
        "tolerance": PROTECTED_PROGRESS_TOLERANCE,
        "passed": drop <= PROTECTED_PROGRESS_TOLERANCE,
    }


def accept_repair(
    before_result,
    after_result,
    repair: BoundedRepair,
    cluster_case_ids,
    tolerance: float = PROTECTED_REGRESSION_TOLERANCE,
):
    """Focused improvement, protected-case regression, and the vacuity guard.

    When the policy succeeded nowhere, the success-based regression threshold is
    unfailable. The repair is then held to a continuous check on the cases
    outside the cluster instead, so acceptance still means something.
    """
    protected_ids, basis = protected_cases(before_result, cluster_case_ids)
    if not protected_ids:
        # Every case is in the cluster. Passing an empty list to
        # ``evaluate_repair`` would quietly protect *all* of them, including the
        # ones the repair is meant to change, so say plainly that there is no
        # protected set instead.
        decision = evaluate_repair(
            before_result, after_result, repair.focused_metric,
            repair.predicted_direction, cluster_case_ids, list(cluster_case_ids),
            tolerance=tolerance,
        )
        decision.protected_check = {
            "n_cases": 0,
            "basis": "every case is in the cluster; there is no protected set",
            "passed": True,
        }
        if decision.accepted:
            decision.reason = "focused metric moved; no case was outside the cluster to protect"
        return decision
    decision = evaluate_repair(
        before_result, after_result, repair.focused_metric, repair.predicted_direction,
        cluster_case_ids, protected_ids, tolerance=tolerance,
    )
    decision.protected_check["basis"] = basis
    if basis != "success":
        secondary = progress_regression(before_result, after_result, protected_ids)
        decision.protected_check["secondary"] = secondary
        if decision.accepted and secondary.get("measurable") and not secondary["passed"]:
            decision.accepted = False
            decision.reason = (
                f"protected cases lost {secondary['drop']:.4f} of mean "
                f"{secondary['metric']}, above the {secondary['tolerance']} tolerance"
            )
    if decision.accepted and not repair_stayed_in_scope(repair):
        outside = sorted(set(repair.changed) - set(repair.parameters))
        decision.accepted = False
        decision.reason = f"the edit moved parameters outside its scope: {outside}"
    return decision
