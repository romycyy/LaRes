"""Immutable, ordered evaluation manifests (``spec.md`` 7.1, FR-1, AC-0).

A manifest materialises the exact list of episodes a policy is scored on: which
MetaWorld task placement, which simulator reset seed, and which policy-sampling
seed.  Every policy in a comparison replays the same manifest in the same order,
so a score difference cannot come from having drawn easier placements.

Task placements come from ``metaworld.MT1(env_id, seed=S)``.  MT1 constructed
*without* a seed draws 50 fresh placements every time, so the pool seed is part
of the evaluation contract: without it, "episode 7" is a different puck and goal
on every run.  MT1 exposes no test tasks, so the train / development /
final-test splits are built from disjoint pool seeds and checked for overlap.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import yaml

SPLIT_TRAIN = "train"
SPLIT_DEVELOPMENT = "development"
SPLIT_FINAL_TEST = "final_test"
SPLITS = (SPLIT_TRAIN, SPLIT_DEVELOPMENT, SPLIT_FINAL_TEST)

#: Placements generated per MT1 seed.  Fixed by ``metaworld._N_GOALS``.
TASKS_PER_SEED = 50

#: Offsets keeping reset and policy seed streams from ever colliding.
_RESET_SEED_BASE = 1_000_000
_POLICY_SEED_BASE = 9_000_000

#: Marks a manifest that names no real placements. Such a manifest carries no
#: version lock, so ``resolve_pool`` refuses it and no reported number may use it.
SYNTHETIC_BENCHMARK = "none"


# ---------------------------------------------------------------------------
#  Task pool
# ---------------------------------------------------------------------------


def to_v3(env_id: str) -> str:
    """Map the ``-v2`` names used throughout this repo onto installed ``-v3`` ids."""
    return env_id.replace("-v2", "-v3").replace("-v1", "-v3")


def _hash_rand_vec(rand_vec) -> str:
    """Stable short digest of a placement vector, used to detect pool drift."""
    arr = np.asarray(rand_vec, dtype=np.float64)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


@dataclass(frozen=True)
class PoolEntry:
    """One MetaWorld placement plus the identity used to refer to it."""

    task_id: str
    task_hash: str
    rand_vec: tuple[float, ...]
    task: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class TaskPoolSpec:
    """Reproducible recipe for a set of placements."""

    benchmark: str
    env_id_v3: str
    seeds: tuple[int, ...]
    tasks_per_seed: int = TASKS_PER_SEED

    def to_dict(self) -> dict:
        d = asdict(self)
        d["seeds"] = list(self.seeds)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TaskPoolSpec":
        return cls(
            benchmark=str(d["benchmark"]),
            env_id_v3=str(d["env_id_v3"]),
            seeds=tuple(int(s) for s in d["seeds"]),
            tasks_per_seed=int(d.get("tasks_per_seed", TASKS_PER_SEED)),
        )


class TaskPool:
    """Ordered, addressable collection of :class:`PoolEntry` placements."""

    def __init__(self, spec: TaskPoolSpec, entries: Sequence[PoolEntry]):
        self.spec = spec
        self.entries = list(entries)
        self.by_id = {e.task_id: e for e in self.entries}
        if len(self.by_id) != len(self.entries):
            raise ValueError("duplicate task_id in task pool")

    def __len__(self) -> int:
        return len(self.entries)

    def entry(self, task_id: str) -> PoolEntry:
        try:
            return self.by_id[task_id]
        except KeyError:
            raise KeyError(
                f"task_id {task_id!r} is not in this pool "
                f"({len(self.entries)} entries from seeds {list(self.spec.seeds)})"
            ) from None

    def task_index(self) -> dict:
        """``{task_id: metaworld.Task}`` for handing to ``env_wrapper``."""
        return {e.task_id: e.task for e in self.entries}


def build_task_pool(
    env_id: str,
    seeds: Iterable[int],
    tasks_per_seed: int = TASKS_PER_SEED,
) -> TaskPool:
    """Construct placements from ``MT1(env_id_v3, seed=s)`` for each seed, in order.

    Raises if two seeds produce the same placement, so a split built from
    disjoint seeds is genuinely disjoint.
    """
    import metaworld as mw

    env_id_v3 = to_v3(env_id)
    seeds = tuple(int(s) for s in seeds)
    if not seeds:
        raise ValueError("build_task_pool needs at least one pool seed")

    entries: list[PoolEntry] = []
    seen: dict[str, str] = {}
    for seed in seeds:
        mt1 = mw.MT1(env_id_v3, seed=seed)
        tasks = list(mt1.train_tasks)
        if len(tasks) < tasks_per_seed:
            raise ValueError(
                f"MT1({env_id_v3!r}, seed={seed}) returned {len(tasks)} tasks, "
                f"expected at least {tasks_per_seed}"
            )
        for i, task in enumerate(tasks[:tasks_per_seed]):
            rand_vec = pickle.loads(task.data)["rand_vec"]
            task_hash = _hash_rand_vec(rand_vec)
            task_id = f"MT1:{env_id_v3}:s{seed}:{i:03d}"
            if task_hash in seen:
                raise ValueError(
                    f"placement collision: {task_id} duplicates {seen[task_hash]}"
                )
            seen[task_hash] = task_id
            entries.append(
                PoolEntry(
                    task_id=task_id,
                    task_hash=task_hash,
                    rand_vec=tuple(float(x) for x in np.asarray(rand_vec).ravel()),
                    task=task,
                )
            )
    return TaskPool(TaskPoolSpec("MT1", env_id_v3, seeds, tasks_per_seed), entries)


# ---------------------------------------------------------------------------
#  Manifest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeCase:
    """One immutable evaluation episode."""

    case_id: str
    task_id: str
    task_hash: str
    reset_seed: int
    policy_seed: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EpisodeCase":
        return cls(
            case_id=str(d["case_id"]),
            task_id=str(d["task_id"]),
            task_hash=str(d["task_hash"]),
            reset_seed=int(d["reset_seed"]),
            policy_seed=int(d["policy_seed"]),
        )


@dataclass(frozen=True)
class EnvironmentSpec:
    """Everything that has to be pinned for a score to mean the same thing twice."""

    package: str
    version_or_commit: str
    env_id: str
    env_id_v3: str
    reward_version: str
    wrappers: tuple[str, ...]
    horizon: int
    action_scale: tuple[float, ...]
    success_rule: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["wrappers"] = list(self.wrappers)
        d["action_scale"] = [float(x) for x in self.action_scale]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EnvironmentSpec":
        return cls(
            package=str(d["package"]),
            version_or_commit=str(d["version_or_commit"]),
            env_id=str(d["env_id"]),
            env_id_v3=str(d["env_id_v3"]),
            reward_version=str(d["reward_version"]),
            wrappers=tuple(str(w) for w in d["wrappers"]),
            horizon=int(d["horizon"]),
            action_scale=tuple(float(x) for x in d["action_scale"]),
            success_rule=str(d["success_rule"]),
        )


def _metaworld_version() -> str:
    try:
        from importlib.metadata import version

        return version("metaworld")
    except Exception:  # pragma: no cover - depends on install method
        import metaworld

        return str(getattr(metaworld, "__version__", "unknown"))


def environment_spec_for(env_id: str, horizon: int) -> EnvironmentSpec:
    """Read the pinned environment facts off the installed package."""
    import metaworld.env_dict as _env_dict

    env_id_v3 = to_v3(env_id)
    if env_id_v3 not in _env_dict.ALL_V3_ENVIRONMENTS:
        raise ValueError(f"{env_id!r} (as {env_id_v3!r}) is not an installed V3 env")
    env_cls = _env_dict.ALL_V3_ENVIRONMENTS[env_id_v3]
    radius = getattr(env_cls, "TARGET_RADIUS", None)
    success_rule = (
        f"info['success'] = float(obj_to_target <= TARGET_RADIUS={radius})"
        if radius is not None
        else "info['success'] > 0"
    )
    return EnvironmentSpec(
        package="metaworld",
        version_or_commit=_metaworld_version(),
        env_id=env_id,
        env_id_v3=env_id_v3,
        reward_version=f"{env_cls.__module__}.{env_cls.__name__}.compute_reward",
        wrappers=("NormalizedBoxEnv", "gym.TimeLimit", "lares.env_wrapper"),
        horizon=int(horizon),
        action_scale=(1.0, 1.0, 1.0, 1.0),
        success_rule=success_rule,
    )


@dataclass
class EvaluationManifest:
    """Ordered episode list plus the environment and pool it is only valid against."""

    manifest_id: str
    environment: EnvironmentSpec
    pool: TaskPoolSpec
    split: str
    episodes: list[EpisodeCase]

    def __post_init__(self):
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {self.split!r}")
        ids = {c.case_id for c in self.episodes}
        if len(ids) != len(self.episodes):
            raise ValueError("duplicate case_id in manifest")

    def __len__(self) -> int:
        return len(self.episodes)

    def head(self, n: int) -> "EvaluationManifest":
        """First ``n`` episodes, e.g. the 10-case screening subset of development.

        A prefix, never a resample, so screening cases stay nested inside the
        expanded evaluation and the two remain paired.
        """
        if n > len(self.episodes):
            raise ValueError(f"manifest has {len(self.episodes)} episodes, asked for {n}")
        # Re-heading an already-headed manifest replaces the suffix rather than
        # appending, so an id stays readable after two or three narrowings.
        base = self.manifest_id.split("#head")[0]
        return EvaluationManifest(
            manifest_id=f"{base}#head{n}",
            environment=self.environment,
            pool=self.pool,
            split=self.split,
            episodes=list(self.episodes[:n]),
        )

    def tail(self, n: int) -> "EvaluationManifest":
        """Last ``n`` episodes, disjoint from ``head(len - n)`` by construction.

        The held-out complement of a screening prefix: a loop may spend its
        budget on ``head``, and what it produced is then scored on ``tail``
        without any case appearing on both sides.
        """
        if n > len(self.episodes):
            raise ValueError(f"manifest has {len(self.episodes)} episodes, asked for {n}")
        base = self.manifest_id.split("#head")[0].split("#tail")[0]
        return EvaluationManifest(
            manifest_id=f"{base}#tail{n}",
            environment=self.environment,
            pool=self.pool,
            split=self.split,
            episodes=list(self.episodes[len(self.episodes) - n:]),
        )

    def task_hashes(self) -> set[str]:
        return {c.task_hash for c in self.episodes}

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "manifest_id": self.manifest_id,
            "environment": self.environment.to_dict(),
            "pool": self.pool.to_dict(),
            "split": self.split,
            "episodes": [c.to_dict() for c in self.episodes],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvaluationManifest":
        return cls(
            manifest_id=str(d["manifest_id"]),
            environment=EnvironmentSpec.from_dict(d["environment"]),
            pool=TaskPoolSpec.from_dict(d["pool"]),
            split=str(d["split"]),
            episodes=[EpisodeCase.from_dict(e) for e in d["episodes"]],
        )

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, default_flow_style=False)
        return path

    @classmethod
    def load(cls, path: str) -> "EvaluationManifest":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))

    # -- pool binding -----------------------------------------------------

    def resolve_pool(self) -> TaskPool:
        """Rebuild the placements this manifest names, and verify none have drifted.

        Regenerating from the recorded seeds is what makes the manifest portable:
        the YAML holds identities and hashes, never pickled simulator state.
        """
        if self.pool.benchmark == SYNTHETIC_BENCHMARK:
            raise ValueError(
                f"{self.manifest_id} is a synthetic manifest with no task pool. "
                f"It carries no version lock and must not be used for a reported result."
            )
        pool = build_task_pool(
            self.pool.env_id_v3, self.pool.seeds, self.pool.tasks_per_seed
        )
        for case in self.episodes:
            entry = pool.entry(case.task_id)
            if entry.task_hash != case.task_hash:
                raise ValueError(
                    f"placement drift for {case.case_id}: manifest expects "
                    f"{case.task_hash}, regenerated pool gives {entry.task_hash}. "
                    f"The installed metaworld version no longer reproduces this pool."
                )
        return pool


def build_manifest(
    env_id: str,
    split: str,
    pool_seeds: Iterable[int],
    *,
    horizon: int,
    num_episodes: int | None = None,
    tasks_per_seed: int = TASKS_PER_SEED,
    manifest_id: str | None = None,
) -> tuple[EvaluationManifest, TaskPool]:
    """Materialise one split's manifest and the pool it binds to.

    Reset and policy seeds are derived from the episode index, so the manifest
    is a pure function of ``(env_id, split, pool_seeds, num_episodes)``.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    pool = build_task_pool(env_id, pool_seeds, tasks_per_seed)
    entries = pool.entries if num_episodes is None else pool.entries[:num_episodes]
    if num_episodes is not None and len(entries) < num_episodes:
        raise ValueError(
            f"pool has {len(pool)} placements, manifest asked for {num_episodes}"
        )

    episodes = [
        EpisodeCase(
            case_id=f"{split}-{i:04d}",
            task_id=e.task_id,
            task_hash=e.task_hash,
            reset_seed=_RESET_SEED_BASE + i,
            policy_seed=_POLICY_SEED_BASE + i,
        )
        for i, e in enumerate(entries)
    ]
    seeds_tag = "-".join(str(s) for s in pool.spec.seeds)
    manifest = EvaluationManifest(
        manifest_id=manifest_id or f"{env_id}:{split}:s{seeds_tag}:n{len(episodes)}",
        environment=environment_spec_for(env_id, horizon),
        pool=pool.spec,
        split=split,
        episodes=episodes,
    )
    return manifest, pool


def synthetic_cases(
    n: int, prefix: str = "case", task_id: str = "default", base_seed: int = 0
) -> list[EpisodeCase]:
    """Cases for environments that have no MetaWorld task pool.

    The Isaac Lab stack and the unit-test mocks still have to name the episode
    they run, so the "every reset names its case" contract holds everywhere.
    These carry no placement, only reproducible seeds.
    """
    return [
        EpisodeCase(
            case_id=f"{prefix}-{i:04d}",
            task_id=task_id,
            task_hash="",
            reset_seed=base_seed + i,
            policy_seed=_POLICY_SEED_BASE + base_seed + i,
        )
        for i in range(int(n))
    ]


def synthetic_manifest(
    n: int,
    label: str = "mock",
    split: str = SPLIT_DEVELOPMENT,
    horizon: int = 150,
    action_dim: int = 4,
) -> EvaluationManifest:
    """Manifest for mocks and non-MetaWorld backends.

    Every field that would pin a real result is filled with a value that says
    it is not pinned, and :meth:`EvaluationManifest.resolve_pool` refuses it, so
    a synthetic manifest cannot be mistaken for a locked one.
    """
    return EvaluationManifest(
        manifest_id=f"synthetic:{label}:n{int(n)}",
        environment=EnvironmentSpec(
            package="synthetic",
            version_or_commit="not-locked",
            env_id=label,
            env_id_v3=label,
            reward_version="not-locked",
            wrappers=(),
            horizon=int(horizon),
            action_scale=tuple(1.0 for _ in range(action_dim)),
            success_rule="info['success'] > 0",
        ),
        pool=TaskPoolSpec(SYNTHETIC_BENCHMARK, label, (), 0),
        split=split,
        episodes=synthetic_cases(n, prefix=label),
    )


def assert_disjoint(manifests: Sequence[EvaluationManifest]) -> None:
    """Raise if any two manifests share a placement (AC-0 non-overlap check)."""
    for i, a in enumerate(manifests):
        for b in manifests[i + 1 :]:
            shared = a.task_hashes() & b.task_hashes()
            if shared:
                raise ValueError(
                    f"manifests {a.manifest_id} and {b.manifest_id} share "
                    f"{len(shared)} placements"
                )
