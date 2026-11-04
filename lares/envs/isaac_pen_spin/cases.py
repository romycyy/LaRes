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

from lares.envs.isaac_pen_spin.layout import OBS_LAYOUT, VEL_OBS_SCALE  # noqa: F401  (no-import module)
from lares.eval.manifest import (
    SPLIT_DEVELOPMENT,
    EnvironmentSpec,
    EpisodeCase,
    EvaluationManifest,
    TaskPoolSpec,
    _hash_rand_vec,
    assert_disjoint,
)

BENCHMARK = "IsaacPenSpin"
NUM_HAND_DOFS = 24
NUM_ACTIONS = 20
OBS_DIM = sum(size for _, size, _ in OBS_LAYOUT)  # 157: the "full" observation
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


def freeze_manifest(path: str, manifest: EvaluationManifest) -> bool:
    """Save ``manifest`` at ``path`` unless an identical one is already there; refuse a different one.

    Returns True when the file was written. Every pen-spin protocol freezes its manifests this way,
    so a second ``--freeze-manifests`` (or a rerun) verifies rather than overwrites.
    """
    if os.path.exists(path):
        if EvaluationManifest.load(path).to_dict() != manifest.to_dict():
            raise ValueError(f"{path} exists and differs from the manifest this study builds")
        return False
    manifest.save(path)
    return True


def freeze_manifest_set(specs) -> dict:
    """Freeze ``(path, manifest, role)`` triples and refuse a placement shared between any two roles.

    Returns ``{role: path}`` in the given order, which is what each study's ``freeze_manifests`` prints.
    """
    specs = list(specs)
    for path, manifest, _ in specs:
        freeze_manifest(path, manifest)
    assert_disjoint([m for _, m, _ in specs])
    return {role: path for path, _, role in specs}


def role_manifests(manifest: EvaluationManifest, roles: dict, *, expected_cases: int | None = None,
                   stage: int | None = None) -> dict:
    """``{role: cases[a:b]}`` of a block manifest, each under an id naming the role.

    The id is ``"<block id>:<role>"`` rather than ``slice``'s ``#sliceA-B``, so a role's own ``head``/``slice``
    stays unambiguous; recorded results name manifests this way.
    """
    if expected_cases is not None and len(manifest) != expected_cases:
        raise ValueError(f"block manifest {manifest.manifest_id} has {len(manifest)} cases, not {expected_cases}")
    if stage is not None and stage_of(manifest) != stage:
        raise ValueError(f"block manifest {manifest.manifest_id} is not a stage-{stage} one")
    return {role: EvaluationManifest(manifest_id=f"{manifest.manifest_id}:{role}", environment=manifest.environment,
                                     pool=manifest.pool, split=manifest.split, episodes=list(manifest.episodes[a:b]))
            for role, (a, b) in roles.items()}


# ---------------------------------------------------------------------------
#  Observation layout ("full", 157 dims; InHandManipulationEnv.compute_full_observations)
# ---------------------------------------------------------------------------


def obs_slices() -> dict:
    out, start = {}, 0
    for name, size, _ in OBS_LAYOUT:
        out[name] = slice(start, start + size)
        start += size
    assert start == OBS_DIM, start
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


def reduce_episodes(trace: dict) -> list[dict]:
    """Per-episode rows: both reducers plus the geometry a reader needs to check them."""
    rows = []
    for i, cid in enumerate(trace["case_ids"]):
        L = int(trace["length"][i])
        p = trace["pred"][:L, i]
        spin = np.abs(trace["spin_rate"][:L, i])
        rows.append({
            "case_id": cid,
            "task_hash": trace["task_hashes"][i],
            "length": L,
            "completed": bool(trace["completed"][i]),
            "any_hit": any_hit(p, L),
            "sustained": sustained(p, L, bool(trace["completed"][i])),
            "hit_steps": int(p.sum()),
            "first_hit_step": int(np.argmax(p)) if p.any() else None,
            "tail_hit_fraction": float(p[-SUSTAIN_WINDOW:].mean()) if L else 0.0,
            "mean_abs_spin": float(spin.mean()) if L else None,
            "max_abs_spin": float(spin.max()) if L else None,
            "mean_transverse": float(np.mean(trace["transverse_rate"][:L, i])) if L else None,
            "final_center_dist": float(trace["center_dist"][L - 1, i]) if L else None,
            "return": float(np.nansum(trace["reward"][:L, i])),
        })
    return rows
