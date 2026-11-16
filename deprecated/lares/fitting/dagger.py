"""Learner-state imitation on training placements (``spec.md`` FR-8, AC-5).

Behavioural cloning fits the states the *expert* visits. The learner visits
different ones, and the error compounds: once it drifts off the expert's path it
has no labelled data for where it now is. Aggregating expert labels at
learner-visited states is the standard correction.

Three constraints this module enforces rather than assumes:

* Rollouts happen only on **training** placements. A development or final-test
  placement entering the imitation set would make every later score on it
  meaningless.
* Every expert query and every environment step is counted, because an
  aggregation round is not free and a comparison that hides its cost is not a
  comparison.
* The expert's answer at a learner-visited state is checked before it is stored.
  The expert was written for states on its own trajectory, and there is no
  guarantee it produces a sensible recovery from somewhere it never goes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

#: An expert label is rejected if its response is not finite, or if it asks for
#: essentially no motion while the object is still far from the goal, which is
#: the signature of a state the scripted controller has no plan for.
MIN_USEFUL_RESPONSE = 1e-3


@dataclass
class AggregationStats:
    """What one round of aggregation cost and produced."""

    rounds: int = 0
    episodes: int = 0
    environment_steps: int = 0
    expert_queries: int = 0
    labels_accepted: int = 0
    labels_rejected: int = 0
    rejection_reasons: dict = field(default_factory=dict)
    cases_used: list = field(default_factory=list)
    recovery_checked: int = 0
    recovery_improved: int = 0
    #: Change in goal distance the expert's own label produced on a branch.
    recovery_deltas: list = field(default_factory=list)
    #: Steps spent on the counterfactual branches, kept separate from the
    #: aggregation cost so the two are not conflated.
    counterfactual_steps: int = 0
    #: Recovery outcome split by expert phase; the approach phases and the push
    #: phase are trying to reduce different errors.
    recovery_by_phase: dict = field(default_factory=dict)

    @property
    def acceptance_rate(self) -> float:
        return self.labels_accepted / self.expert_queries if self.expert_queries else 0.0

    @property
    def recovery_rate(self) -> float | None:
        """Share of sampled labels that actually reduced the goal distance."""
        if not self.recovery_checked:
            return None
        return self.recovery_improved / self.recovery_checked

    def to_dict(self) -> dict:
        d = asdict(self)
        d["acceptance_rate"] = self.acceptance_rate
        d["recovery_rate"] = self.recovery_rate
        return d


def _snapshot(env):
    """Full simulator state, or ``None`` when the backend cannot branch.

    Restoring ``qpos`` and ``qvel`` alone is not enough: MetaWorld drives the arm
    through a mocap body whose pose lives outside the generalised coordinates, so
    a branch that ignored it would resume from a different setpoint.
    """
    from lares.utils.metaworld_env import _unwrap_through_wrappers

    inner = _unwrap_through_wrappers(getattr(env, "_env", env))
    if not hasattr(inner, "get_env_state"):
        return None
    try:
        qpos, qvel = inner.get_env_state()
        return {
            "inner": inner,
            "qpos": np.copy(qpos),
            "qvel": np.copy(qvel),
            "mocap_pos": np.copy(inner.data.mocap_pos),
            "mocap_quat": np.copy(inner.data.mocap_quat),
            "path_length": int(getattr(inner, "curr_path_length", 0)),
            "timesteps": int(getattr(env, "timesteps", 0)),
        }
    except Exception:
        return None


def _restore(env, snap) -> None:
    inner = snap["inner"]
    inner.set_env_state((snap["qpos"], snap["qvel"]))
    inner.data.mocap_pos[:] = snap["mocap_pos"]
    inner.data.mocap_quat[:] = snap["mocap_quat"]
    inner.curr_path_length = snap["path_length"]
    if hasattr(env, "timesteps"):
        env.timesteps = snap["timesteps"]


def _expert_error(obs, schema, expert):
    """Distance from the hand to the position the expert is steering it toward.

    This is the error the expert is definitionally reducing: its action is
    ``gain * (desired - hand)``. Any other yardstick fails a correct label by
    construction in some phase. Judging by object-to-goal fails both approach
    phases, since no action can move the puck before contact. Judging by
    hand-to-object fails the hover phase, where the expert deliberately lifts
    0.2 m *above* the puck and so increases that distance on purpose.
    """
    from lares.fitting.phases import PHASE_NAMES, expert_phase_labels

    phase = int(expert_phase_labels(obs[None, :], schema)[0])
    tcp = np.asarray(schema.slice(obs[None, :], "tcp"), dtype=np.float64)[0]
    desired = np.asarray(expert._desired_pos(expert._parse_obs(np.asarray(obs, dtype=np.float64))))
    return PHASE_NAMES[phase], float(np.linalg.norm(tcp - desired))


def _task_error(obs, schema):
    """Object-to-goal distance, the outcome the task is actually scored on."""
    obj = np.asarray(schema.slice(obs[None, :], "obj"), dtype=np.float64)[0]
    goal = np.asarray(schema.slice(obs[None, :], "goal"), dtype=np.float64)[0]
    return float(np.linalg.norm(obj - goal))


def counterfactual_recovery(env, expert_action, obs, schema, expert):
    """Does the expert's own label actually help from *this* state?

    Applies the expert action on a branch of the simulator, measures the distance
    to the position the expert is steering toward, then rewinds. Without the
    branch this would measure what the *learner's* action did, which says nothing
    about the label being stored.

    The task error is reported alongside but not used for the verdict: before
    contact no action can move the object, so a correct label would fail that test.

    Returns ``None`` when the backend cannot be branched.
    """
    snap = _snapshot(env)
    if snap is None:
        return None
    phase, before = _expert_error(obs, schema, expert)
    task_before = _task_error(obs, schema)
    try:
        after_obs, _, _, _ = env.step(
            env.action_space.high * np.clip(expert_action, -1.0, 1.0)
        )
        _, after = _expert_error(after_obs, schema, expert)
        task_after = _task_error(after_obs, schema)
    finally:
        _restore(env, snap)
    return {
        "phase": phase,
        "measure": "hand_to_expert_desired_position",
        "before": before,
        "after": after,
        "improved": after < before,
        "task_before": task_before,
        "task_after": task_after,
    }


def _reject(stats: AggregationStats, reason: str) -> None:
    stats.labels_rejected += 1
    stats.rejection_reasons[reason] = stats.rejection_reasons.get(reason, 0) + 1


def validate_expert_label(action, obs, schema, goal_distance=None) -> str | None:
    """Reason the expert's answer here is unusable, or ``None`` if it is fine."""
    arr = np.asarray(action, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return "non_finite"
    if np.abs(arr[:3]).max() < MIN_USEFUL_RESPONSE:
        # A near-zero motion command is only sensible once the task is done.
        if goal_distance is None or goal_distance > 0.05:
            return "no_motion_requested_while_unfinished"
    return None


def aggregate_learner_states(
    policy,
    env,
    cases,
    env_name: str,
    schema,
    buffer=None,
    max_steps: int = 150,
    execution: str = "deterministic_tanh",
    recovery_check_every: int = 25,
    stats: AggregationStats | None = None,
):
    """Roll out the learner on training placements and label what it visits.

    Args:
        policy: the current learner.
        env: an environment bound to the *training* pool.
        cases: training ``EpisodeCase`` list. Passing development or final-test
            cases here would contaminate the imitation set.
        env_name: task id, used to load the scripted expert.
        schema: observation schema, for the geometry the validity check needs.
        buffer: ``DemoBuffer`` to extend. A new one is created when omitted.
        execution: how the learner acts while collecting; the aggregation should
            reflect the states the deployed controller reaches.
        recovery_check_every: sample rate for the expensive recovery check.

    Returns:
        ``(buffer, stats)``.
    """
    import torch

    from lares.core.training_pipeline import DemoBuffer, get_expert_policy
    from lares.fitting.objectives import execute

    buffer = buffer if buffer is not None else DemoBuffer()
    stats = stats or AggregationStats()
    stats.rounds += 1
    expert = get_expert_policy(env_name)
    policy.eval()

    for case in cases:
        obs, _ = env.reset(case)
        stats.episodes += 1
        stats.cases_used.append(case.case_id)
        for step in range(max_steps):
            obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mean, std = policy(obs_t)
                action = execute(mean, std, execution).squeeze(0).numpy().astype(np.float64)

            expert_raw = np.asarray(expert.get_action(np.asarray(obs, dtype=np.float64)))
            stats.expert_queries += 1
            obj = np.asarray(schema.slice(obs[None, :], "obj"), dtype=np.float64)[0]
            goal = np.asarray(schema.slice(obs[None, :], "goal"), dtype=np.float64)[0]
            goal_distance = float(np.linalg.norm(obj - goal))
            reason = validate_expert_label(expert_raw, obs, schema, goal_distance)

            # The recovery check runs on a branch, before the learner's own step,
            # so it measures the label rather than the learner.
            check = None
            if reason is None and stats.expert_queries % recovery_check_every == 0:
                check = counterfactual_recovery(env, expert_raw, obs, schema, expert)
                if check is not None:
                    stats.recovery_checked += 1
                    stats.recovery_improved += int(check["improved"])
                    stats.recovery_deltas.append(check["after"] - check["before"])
                    stats.counterfactual_steps += 1
                    by_phase = stats.recovery_by_phase.setdefault(
                        check["phase"], {"checked": 0, "improved": 0}
                    )
                    by_phase["checked"] += 1
                    by_phase["improved"] += int(check["improved"])

            next_obs, reward, done, info = env.step(env.action_space.high * action)
            stats.environment_steps += 1

            if reason is None:
                label = np.clip(expert_raw, -1.0, 1.0)
                buffer.add(
                    obs, label, reward, next_obs, float(done),
                    episode_id=f"dagger-{case.case_id}",
                )
                stats.labels_accepted += 1
            else:
                _reject(stats, reason)

            obs = next_obs
            if done:
                break

    return buffer, stats


def summarise(stats: AggregationStats) -> str:
    lines = [
        f"  rounds                {stats.rounds}",
        f"  episodes              {stats.episodes}",
        f"  environment steps     {stats.environment_steps}",
        f"  expert queries        {stats.expert_queries}",
        f"  labels accepted       {stats.labels_accepted} "
        f"({stats.acceptance_rate:.3f} of queries)",
        f"  labels rejected       {stats.labels_rejected}",
    ]
    for reason, count in sorted(stats.rejection_reasons.items()):
        lines.append(f"    {reason:<38} {count}")
    if stats.recovery_rate is not None:
        lines.append(
            f"  expert label helps:   {stats.recovery_improved}/{stats.recovery_checked} "
            f"branches reduced the goal distance ({stats.recovery_rate:.3f})"
        )
        if stats.recovery_deltas:
            lines.append(
                f"    mean change in the phase's own error "
                f"{float(np.mean(stats.recovery_deltas)):+.5f} m"
            )
        for phase, counts in sorted(stats.recovery_by_phase.items()):
            rate = counts["improved"] / counts["checked"] if counts["checked"] else 0.0
            lines.append(
                f"    {phase:<10} {counts['improved']}/{counts['checked']} ({rate:.3f})"
            )
        lines.append(f"  counterfactual steps  {stats.counterfactual_steps}")
    return "\n".join(lines)
