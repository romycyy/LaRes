"""Evaluation contract for the pen-spin task, without Isaac Lab (R-038).

Everything here is numpy/yaml only, so manifests, reducers and the observation layout can be
built and tested on any machine; ``runner.py`` is the part that needs a launched Isaac app.

**Placements.** The base Shadow Hand config has no domain randomization (only
``ShadowHandOpenAIEnvCfg`` sets ``events``), so a reset is fully determined by the uniform(-1, 1)
draws ``InHandManipulationEnv._reset_idx`` and ``ShadowHandSpinEnv._reset_idx`` make: 3 for the
pen position, 2 for its rotation (scaled by 0.25 in the spin env) and one per hand joint
(``reset_dof_vel_noise`` is 0). A case's placement is that vector, drawn from
``numpy.random.default_rng(pool_seed)``, hashed like a MetaWorld ``rand_vec``, and *written*
into the simulator by the runner rather than reproduced by reseeding Isaac's global RNG.

**Episode rules.** The env scores success per physics step. Two reducers are declared here:
``any_hit`` (some step before termination satisfied the predicate) and ``sustained`` (the episode
ended by time-out, not by the pen falling, and every one of its final ``SUSTAIN_WINDOW`` steps
satisfied the predicate). R-038 preregisters ``sustained`` as the qualification rule.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from lares.eval.manifest import (
    SPLIT_DEVELOPMENT,
    EnvironmentSpec,
    EpisodeCase,
    EvaluationManifest,
    TaskPoolSpec,
    _hash_rand_vec,
)

BENCHMARK = "IsaacPenSpin"
NUM_HAND_DOFS = 24
NUM_ACTIONS = 20
PLACEMENT_DIM = 3 + 2 + NUM_HAND_DOFS
HORIZON = 360  # max_episode_length: 6 s at dt 1/120, decimation 2; time-out fires at step HORIZON - 1
SUSTAIN_WINDOW = 60  # one second of control
RESET_SEED_BASE = 1_000_000
POLICY_SEED_BASE = 9_000_000


@dataclass(frozen=True)
class StageSpec:
    stage: int
    task_id: str
    cfg_class: str
    success_spin_rate: float
    success_center_tolerance: float
    fall_dist: float


#: Thresholds copied from ``shadow_hand_spin_env_cfg``; ``tests/test_pen_spin`` checks them against the source.
STAGES = {
    1: StageSpec(1, "Isaac-ShadowHandSpin-Stage1-Direct-v0", "ShadowHandSpinStage1EnvCfg", 1.0, 0.10, 0.30),
    2: StageSpec(2, "Isaac-ShadowHandSpin-Stage2-Direct-v0", "ShadowHandSpinStage2EnvCfg", 3.0, 0.10, 0.30),
    3: StageSpec(3, "Isaac-ShadowHandSpin-Stage3-Direct-v0", "ShadowHandSpinStage3EnvCfg", 4.0, 0.08, 0.30),
}


# ---------------------------------------------------------------------------
#  Placements and manifests
# ---------------------------------------------------------------------------


def placements(pool_seed: int, n: int) -> np.ndarray:
    """``(n, PLACEMENT_DIM)`` uniform(-1, 1) reset draws for one pool seed."""
    rng = np.random.default_rng(int(pool_seed))
    return rng.uniform(-1.0, 1.0, size=(int(n), PLACEMENT_DIM))


def split_placement(vec) -> dict:
    vec = np.asarray(vec, dtype=np.float64)
    return {"pos": vec[0:3], "rot": vec[3:5], "dof_pos": vec[5:5 + NUM_HAND_DOFS]}


def _task_id(stage: int, pool_seed: int, i: int) -> str:
    return f"{BENCHMARK}:stage{stage}:s{pool_seed}:{i:03d}"


def success_rule_text(stage: int) -> str:
    s = STAGES[stage]
    return (f"per step: center_dist < {s.success_center_tolerance} and |spin_rate about pen long axis| > "
            f"{s.success_spin_rate} rad/s; episode: declared reducer (any_hit or sustained over the final "
            f"{SUSTAIN_WINDOW} steps, early termination = failure)")


def environment_spec(stage: int, version: str = "isaaclab d94504b + lares.envs.isaac_pen_spin") -> EnvironmentSpec:
    s = STAGES[stage]
    return EnvironmentSpec(
        package="isaaclab",
        version_or_commit=version,
        env_id=s.task_id,
        env_id_v3=s.task_id,
        reward_version="lares.envs.isaac_pen_spin.shadow_hand_spin_env.ShadowHandSpinEnv._get_rewards",
        wrappers=("lares.envs.isaac_pen_spin.runner.PenSpinBatchRunner",),
        horizon=HORIZON,
        action_scale=tuple(1.0 for _ in range(NUM_ACTIONS)),
        success_rule=success_rule_text(stage),
    )


def build_manifest(stage: int, pool_seed: int, n: int, *, split: str = SPLIT_DEVELOPMENT,
                   label: str = "probe") -> EvaluationManifest:
    """A manifest of ``n`` cases, a pure function of ``(stage, pool_seed, n)``."""
    vecs = placements(pool_seed, n)
    episodes = [
        EpisodeCase(
            case_id=f"{label}-{i:04d}",
            task_id=_task_id(stage, pool_seed, i),
            task_hash=_hash_rand_vec(vecs[i]),
            reset_seed=RESET_SEED_BASE + i,
            policy_seed=POLICY_SEED_BASE + i,
        )
        for i in range(int(n))
    ]
    return EvaluationManifest(
        manifest_id=f"pen-spin-stage{stage}:{label}:s{pool_seed}:n{int(n)}",
        environment=environment_spec(stage),
        pool=TaskPoolSpec(BENCHMARK, STAGES[stage].task_id, (int(pool_seed),), int(n)),
        split=split,
        episodes=episodes,
    )


def stage_of(manifest: EvaluationManifest) -> int:
    for k, s in STAGES.items():
        if s.task_id == manifest.environment.env_id:
            return k
    raise ValueError(f"{manifest.manifest_id} names no pen-spin stage ({manifest.environment.env_id!r})")


def resolve_placements(manifest: EvaluationManifest) -> np.ndarray:
    """Regenerate each case's placement and refuse any whose hash drifted.

    The pen-spin counterpart of ``EvaluationManifest.resolve_pool``: the YAML holds identities and
    hashes, and the vectors are rebuilt from the pool seed.
    """
    if manifest.pool.benchmark != BENCHMARK:
        raise ValueError(f"{manifest.manifest_id} is not a pen-spin manifest ({manifest.pool.benchmark!r})")
    if len(manifest.pool.seeds) != 1:
        raise ValueError(f"{manifest.manifest_id}: a pen-spin pool has exactly one seed")
    seed = manifest.pool.seeds[0]
    pool = placements(seed, manifest.pool.tasks_per_seed)
    out = []
    for case in manifest.episodes:
        i = int(case.task_id.rsplit(":", 1)[1])
        if case.task_id != _task_id(stage_of(manifest), seed, i):
            raise ValueError(f"{case.case_id}: task id {case.task_id!r} does not match the pool")
        vec = pool[i]
        if _hash_rand_vec(vec) != case.task_hash:
            raise ValueError(f"placement drift for {case.case_id}: expected {case.task_hash}")
        out.append(vec)
    return np.stack(out) if out else np.zeros((0, PLACEMENT_DIM))


def manifest_path(stage: int, label: str, pool_seed: int, root: str = "config/manifests") -> str:
    return os.path.join(root, f"pen_spin_stage{stage}_{label}_s{pool_seed}.yaml")


# ---------------------------------------------------------------------------
#  Observation layout ("full", 157 dims; InHandManipulationEnv.compute_full_observations)
# ---------------------------------------------------------------------------

#: (name, size, description). Velocities in the observation are multiplied by vel_obs_scale = 0.2.
OBS_LAYOUT = (
    ("hand_dof_pos", 24, "hand joint positions, unscaled to [-1, 1] by joint limits"),
    ("hand_dof_vel", 24, "hand joint velocities x 0.2"),
    ("obj_pos", 3, "pen centre position in the env frame (m)"),
    ("obj_rot", 4, "pen orientation quaternion (w, x, y, z); the long axis is local x"),
    ("obj_linvel", 3, "pen linear velocity (m/s)"),
    ("obj_angvel", 3, "pen angular velocity x 0.2 (world frame)"),
    ("in_hand_pos", 3, "nominal in-hand pen position (success centre)"),
    ("goal_rot", 4, "goal quaternion, fixed to identity in the spin task"),
    ("rel_rot", 4, "obj_rot * conj(goal_rot)"),
    ("fingertip_pos", 15, "five fingertip positions (ff, mf, rf, lf, th)"),
    ("fingertip_rot", 20, "five fingertip quaternions"),
    ("fingertip_vel", 30, "five fingertip linear+angular velocities"),
    ("last_action", 20, "previous action"),
)
VEL_OBS_SCALE = 0.2


def obs_slices() -> dict:
    out, start = {}, 0
    for name, size, _ in OBS_LAYOUT:
        out[name] = slice(start, start + size)
        start += size
    assert start == 157, start
    return out


# ---------------------------------------------------------------------------
#  Rotation helpers (scalar-first quaternions, as Isaac Lab)
# ---------------------------------------------------------------------------


def quat_apply(q, v):
    """Rotate vectors ``v`` (..., 3) by quaternions ``q`` (..., 4), w first."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w, xyz = q[..., :1], q[..., 1:]
    t = 2.0 * np.cross(xyz, v)
    return v + w * t + np.cross(xyz, t)


def pen_axis(q):
    """Unit long axis (local x) of the pen in the world frame."""
    a = quat_apply(q, np.broadcast_to([1.0, 0.0, 0.0], np.shape(q)[:-1] + (3,)))
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def spin_components(q, angvel) -> tuple[np.ndarray, np.ndarray]:
    """``(axial, transverse)`` angular speed: roll about the long axis, and end-over-end rotation.

    The env's ``spin_rate`` is the signed axial component; human pen spinning is the transverse one.
    """
    axis = pen_axis(q)
    angvel = np.asarray(angvel, dtype=np.float64)
    axial = np.sum(angvel * axis, axis=-1)
    transverse = np.linalg.norm(angvel - axial[..., None] * axis, axis=-1)
    return axial, transverse


# ---------------------------------------------------------------------------
#  Episode reducers
# ---------------------------------------------------------------------------


def any_hit(pred, length: int) -> bool:
    """Some step in ``pred[:length]`` satisfied the per-step predicate."""
    return bool(np.any(np.asarray(pred[: int(length)], dtype=bool)))


def sustained(pred, length: int, completed: bool, window: int = SUSTAIN_WINDOW) -> bool:
    """Ended by time-out (``completed``; a fall is failure), and each of the last ``window`` steps succeeded."""
    if not completed:
        return False
    tail = np.asarray(pred[: int(length)], dtype=bool)[-int(window):]
    return bool(tail.size == int(window) and tail.all())


REDUCERS = {"any_hit": any_hit, "sustained": sustained}
