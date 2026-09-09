"""Replay an :class:`~lares.eval.manifest.EvaluationManifest` and record every episode.

The runner owns its own environment instance, so a development evaluation never
shares mutable simulator or RNG state with dataset collection, GIF recording or
a debug rollout (``spec.md`` FR-1 / AC-0).

Deterministic and sampled action modes are separate actors and produce separate
results.  They are never merged: the fitness used by evolution executes
``tanh(mean)`` and ignores ``std``, while the sampled path executes
``tanh(Normal(mean, std))``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import numpy as np
import torch

from lares.eval.manifest import (
    SPLIT_FINAL_TEST,
    EpisodeCase,
    EvaluationManifest,
    TaskPool,
)

ACTION_MODE_DETERMINISTIC = "deterministic"
ACTION_MODE_SAMPLED = "sampled"


# ---------------------------------------------------------------------------
#  Environments
# ---------------------------------------------------------------------------


@dataclass
class PipelineEnvs:
    """One environment per pipeline stream.

    Sharing a single environment across dataset collection, development
    evaluation and GIF recording is what let earlier runs shift a candidate's
    cases by however much work happened first. Passing distinct instances is
    checked here rather than trusted.
    """

    development: Any
    rl: Any
    gif: Any
    allow_shared: bool = False

    def __post_init__(self):
        if self.allow_shared:
            return
        ids = [id(self.development), id(self.rl), id(self.gif)]
        if len(set(ids)) != len(ids):
            raise ValueError(
                "development, rl and gif must be separate env instances. Pass "
                "allow_shared=True only for mocks that hold no simulator state."
            )

    def close(self) -> None:
        for env in {id(e): e for e in (self.development, self.rl, self.gif)}.values():
            close = getattr(env, "close", None)
            if callable(close):
                close()


def make_manifest_env(manifest: EvaluationManifest, pool: TaskPool, episode_length: int = 200):
    """Build a fresh env bound to ``pool``, for the exclusive use of one stream.

    Callers that need two streams (say dataset collection and development
    evaluation) build two envs.  Sharing one is what let earlier runs shift a
    candidate's cases by however much work preceded it.
    """
    from lares.utils import env_wrapper, make_metaworld_env

    env_cfg = SimpleNamespace(
        env_name=manifest.environment.env_id,
        seed=manifest.episodes[0].reset_seed if manifest.episodes else 0,
        episode_length=int(episode_length),
        use_mt1=True,
    )
    raw = make_metaworld_env(env_cfg, env_cfg.seed, tasks=pool.task_index())
    return env_wrapper(raw, env_cfg, tasks=pool.task_index())


# ---------------------------------------------------------------------------
#  Actors
# ---------------------------------------------------------------------------


class Actor:
    """Maps observations to actions in ``[-1, 1]``, with per-case randomness reset."""

    name = "actor"
    action_mode = ACTION_MODE_DETERMINISTIC

    def reset(self, case: EpisodeCase) -> None:  # pragma: no cover - trivial
        pass

    def act(self, obs) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError


class PolicyActor(Actor):
    """Runs a :class:`~lares.core.symbolic_policy.SymbolicPolicy`.

    ``deterministic`` executes ``tanh(mean)``, matching the evolution fitness.
    Otherwise it executes ``tanh(mean + std * eps)`` with ``eps`` drawn from a
    generator seeded by the case, matching the RL rollout contract.
    """

    def __init__(self, policy, deterministic: bool = True, name: str = "policy"):
        self.policy = policy
        self.deterministic = bool(deterministic)
        self.name = name
        self.action_mode = (
            ACTION_MODE_DETERMINISTIC if deterministic else ACTION_MODE_SAMPLED
        )
        self._gen = None
        self.policy.eval()

    def reset(self, case: EpisodeCase) -> None:
        if not self.deterministic:
            self._gen = torch.Generator().manual_seed(int(case.policy_seed))
        # Diagnostic state resets per episode, independently of the parameters.
        reset_diagnostics = getattr(self.policy, "reset_diagnostics", None)
        if callable(reset_diagnostics):
            reset_diagnostics()

    def act(self, obs) -> np.ndarray:
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean, std = self.policy(obs_t)
            if self.deterministic:
                action = torch.tanh(mean)
            else:
                eps = torch.randn(mean.shape, generator=self._gen)
                action = torch.tanh(mean + std * eps)
        return action.squeeze(0).numpy().astype(np.float64)

    def gates(self):
        """Gates recorded by the most recent ``act``, or ``None`` if unexposed."""
        return getattr(self.policy, "last_gates", None) or None


class ExpertActor(Actor):
    """MetaWorld's scripted expert, clipped exactly as Stage 1 clips it."""

    action_mode = ACTION_MODE_DETERMINISTIC

    def __init__(self, env_name: str):
        from lares.core.training_pipeline import get_expert_policy

        self.expert = get_expert_policy(env_name)
        self.name = f"expert:{env_name}"

    def act(self, obs) -> np.ndarray:
        return np.clip(self.expert.get_action(np.asarray(obs, dtype=np.float64)), -1.0, 1.0)


class ConstantActor(Actor):
    """Fixed action.  The deterministic smoke policy Phase 0 re-evaluates."""

    def __init__(self, action, name: str = "constant"):
        self.action = np.asarray(action, dtype=np.float64)
        self.name = name

    def act(self, obs) -> np.ndarray:
        return self.action.copy()


# ---------------------------------------------------------------------------
#  Results
# ---------------------------------------------------------------------------


@dataclass
class EpisodeRecord:
    """One row of the per-episode table (``spec.md`` 7.3).

    Fields that could not be measured are ``None``, never zero: a gate that was
    never exposed and a gate that never fired are different findings.
    """

    case_id: str
    task_id: str
    success: float
    episode_return: float
    length: int
    terminal_obj_to_target: float | None
    initial_obj_to_target: float | None
    #: Fraction of steps within MetaWorld's 0.03 m ``near_object`` radius. This is
    #: proximity, not verified contact: the flag is a distance test.
    near_object_rate: float
    first_near_object_step: int | None
    grasp_success_rate: float
    action_saturation_rate: float
    #: True when the rollout ran out of horizon without the env terminating.
    #: MetaWorld never terminates early, so this is 1.0 there and carries no signal.
    timed_out: bool
    #: Per-gate statistics when the policy exposes ``last_gates``; ``None`` when
    #: it does not. Never zero-filled: an unexposed gate is unavailable, not off.
    gate_statistics: dict | None = None
    # -- geometry, from the named observation accessors ---------------------
    initial_goal_distance: float | None = None
    final_goal_distance: float | None = None
    signed_goal_progress: float | None = None
    lateral_drift: float | None = None
    object_displacement: float | None = None
    min_tcp_object_distance: float | None = None
    final_tcp_object_distance: float | None = None
    # -- per-axis action behaviour -----------------------------------------
    action_saturation_by_axis: list | None = None
    action_variation_by_axis: list | None = None
    action_mean_by_axis: list | None = None
    #: Coarse geometry-based cause of failure; ``None`` without a schema.
    failure_label: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _mean_se(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1) / math.sqrt(arr.size))


@dataclass
class ManifestResult:
    """Aggregates plus the per-episode table AC-2 will build on."""

    actor_name: str
    manifest_id: str
    action_mode: str
    episodes: list[EpisodeRecord] = field(default_factory=list)
    #: Observation schema the geometry fields were measured through, or ``None``
    #: when the task has none and those fields are unavailable.
    schema_id: str | None = None

    @property
    def success_rate(self) -> float:
        return _mean_se([e.success for e in self.episodes])[0]

    @property
    def success_se(self) -> float:
        return _mean_se([e.success for e in self.episodes])[1]

    @property
    def mean_reward(self) -> float:
        return _mean_se([e.episode_return for e in self.episodes])[0]

    @property
    def reward_se(self) -> float:
        return _mean_se([e.episode_return for e in self.episodes])[1]

    def summary(self) -> dict:
        d2t = [
            e.terminal_obj_to_target
            for e in self.episodes
            if e.terminal_obj_to_target is not None
        ]
        return {
            "actor_name": self.actor_name,
            "manifest_id": self.manifest_id,
            "action_mode": self.action_mode,
            "num_episodes": len(self.episodes),
            "success_rate": self.success_rate,
            "success_se": self.success_se,
            "mean_reward": self.mean_reward,
            "reward_se": self.reward_se,
            "mean_terminal_obj_to_target": _mean_se(d2t)[0] if d2t else None,
            "timeout_rate": _mean_se([float(e.timed_out) for e in self.episodes])[0],
        }

    def to_dict(self) -> dict:
        return {
            **self.summary(),
            "episodes": [e.to_dict() for e in self.episodes],
        }

    def fitness_dict(self) -> dict:
        """The two-scalar shape the legacy pipeline consumes."""
        return {"mean_reward": self.mean_reward, "success_rate": self.success_rate}


def paired_difference(a: ManifestResult, b: ManifestResult, metric: str = "success") -> dict:
    """Paired difference ``a - b`` over cases both results ran.

    Estimates uncertainty for the difference itself, never by comparing two
    independently computed intervals (``spec.md`` section 12, rule 5).
    """
    key = {"success": "success", "reward": "episode_return"}[metric]
    by_case_b = {e.case_id: e for e in b.episodes}
    diffs = [
        getattr(e, key) - getattr(by_case_b[e.case_id], key)
        for e in a.episodes
        if e.case_id in by_case_b
    ]
    if not diffs:
        raise ValueError("results share no cases; they are not paired")
    mean, se = _mean_se(diffs)
    return {
        "metric": metric,
        "n_paired": len(diffs),
        "mean_difference": mean,
        "se": se,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
    }


# ---------------------------------------------------------------------------
#  Rollout
# ---------------------------------------------------------------------------


def run_episode(
    actor: Actor,
    env,
    case: EpisodeCase,
    max_steps: int = 150,
    schema=None,
    success_radius: float = 0.05,
    observer=None,
) -> EpisodeRecord:
    """One episode on one named case.  No hidden state carries in or out.

    ``schema`` enables the geometry diagnostics: how close the arm got, how far
    the object moved, and whether it moved toward the goal. Without one those
    fields stay ``None`` rather than being guessed from raw indices.

    ``observer`` is called once per step with the state the action was chosen
    from, and once more after the loop with the terminal state. It exists so
    frame capture and other per-step recording can hang off the single rollout
    every actor already goes through, rather than becoming another copy of it.
    It cannot change the episode: it is called after the action is decided and
    its return value is discarded.
    """
    from lares.core.policy_validator import GateAggregator
    from lares.eval.diagnostics import RolloutGeometry

    actor.reset(case)
    obs, _ = env.reset(case)
    gate_reader = getattr(actor, "gates", None)
    gate_agg = GateAggregator() if callable(gate_reader) else None
    action_dim = int(np.prod(env.action_space.shape))
    geometry = RolloutGeometry(schema, action_dim, success_radius=success_radius)

    episode_return = 0.0
    success = 0.0
    near_steps = 0
    first_near = None
    grasp_steps = 0
    saturated = 0
    initial_d = None
    terminal_d = None
    steps = 0
    done = False

    for t in range(max_steps):
        action = np.asarray(actor.act(obs), dtype=np.float64)
        gates = gate_reader() if gate_agg is not None else None
        if gate_agg is not None:
            gate_agg.observe(gates)
        geometry.observe(obs, action)
        if observer is not None:
            observer({"timestep": t, "obs": obs, "action": action,
                      "gates": gates, "final": False})
        obs, reward, done, info = env.step(env.action_space.high * action)
        steps = t + 1
        episode_return += float(reward)
        if float(info.get("success", 0.0)) > 0:
            success = 1.0
        if float(info.get("near_object", 0.0)) > 0:
            near_steps += 1
            if first_near is None:
                first_near = t
        if float(info.get("grasp_success", 0.0)) > 0:
            grasp_steps += 1
        d = info.get("obj_to_target", None)
        if d is not None:
            terminal_d = float(d)
            if initial_d is None:
                initial_d = float(d)
        if np.any(np.abs(action) > 0.99):
            saturated += 1
        if done:
            break

    geometry.finish(obs)
    if observer is not None:
        observer({"timestep": steps, "obs": obs, "action": None,
                  "gates": None, "final": True})
    geo = geometry.geometry_statistics()
    acts = geometry.action_statistics()

    return EpisodeRecord(
        case_id=case.case_id,
        task_id=case.task_id,
        success=success,
        episode_return=episode_return,
        length=steps,
        terminal_obj_to_target=terminal_d,
        initial_obj_to_target=initial_d,
        near_object_rate=near_steps / steps if steps else 0.0,
        first_near_object_step=first_near,
        grasp_success_rate=grasp_steps / steps if steps else 0.0,
        action_saturation_rate=saturated / steps if steps else 0.0,
        timed_out=bool(not done and steps >= max_steps),
        gate_statistics=(
            gate_agg.statistics() if gate_agg is not None and gate_agg.names else None
        ),
        failure_label=geometry.failure_label(success),
        **geo,
        **acts,
    )


def evaluate_manifest(
    actor: Actor,
    manifest: EvaluationManifest,
    env,
    max_steps: int = 150,
    progress: Callable[[int, int], Any] | None = None,
    schema=None,
) -> ManifestResult:
    """Replay every episode of ``manifest`` in its recorded order.

    ``schema`` defaults to the layout for the manifest's task, so geometry
    diagnostics are on by default and absent only when the task has no schema.
    """
    if manifest.split == SPLIT_FINAL_TEST:
        # FR-4: winner selection and final estimation use different data. This is
        # the one choke point every actor goes through, so it is where the
        # final-test split is refused unless a session has deliberately opened it.
        from lares.eval.freeze import assert_final_test_allowed

        assert_final_test_allowed(manifest.manifest_id)
    if schema is None:
        from lares.core.obs_schema import OBS_SCHEMAS

        schema = OBS_SCHEMAS.get(manifest.environment.env_id)
    result = ManifestResult(
        actor_name=getattr(actor, "name", type(actor).__name__),
        manifest_id=manifest.manifest_id,
        action_mode=getattr(actor, "action_mode", ACTION_MODE_DETERMINISTIC),
        schema_id=schema.env_id if schema is not None else None,
    )
    total = len(manifest.episodes)
    for i, case in enumerate(manifest.episodes):
        result.episodes.append(
            run_episode(actor, env, case, max_steps=max_steps, schema=schema)
        )
        if progress is not None:
            progress(i + 1, total)
    return result
