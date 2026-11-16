"""Replicate bundles that vary the data, not just minibatch order (decided 2026-09-14).

The first fittable-bank qualification used five fitting seeds that changed only minibatch order:
one demonstration buffer, one train/validation split, one probe manifest and one evaluation
manifest for every seed. Selected success barely moved between seeds, so the five "replicates"
measured insensitivity to minibatch order rather than robustness.

A :class:`ReplicateBundle` is everything data-dependent about one replicate, drawn independently
from that replicate's own seeds:

* a sample of demonstration episodes, without replacement, from the demonstration pool;
* a trajectory-level train/validation split of that sample (no episode on both sides);
* a probe manifest, which selects checkpoints;
* a qualification manifest, which scores the selected checkpoint, disjoint from the probe.

Every structure in a replicate is fitted and scored on the *same* bundle, so structures stay
paired within a replicate while replicates differ in data. :func:`run_replicate_qualification`
enforces that and applies the qualification rule to the per-replicate paired differences.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lares.utils.hashing import sha256_json


class ReplicateBundleError(ValueError):
    """A bundle that would not make its replicate independent, or a study that did not share one."""


def _digest(payload) -> str:
    return sha256_json(payload)


def _placements(manifest) -> set:
    """Placement identities of a manifest. Synthetic cases carry no task hash, so fall back."""
    return {c.task_hash if c.task_hash else f"{c.task_id}|{c.reset_seed}" for c in manifest.episodes}


@dataclass(frozen=True)
class ReplicateBundle:
    replicate: int
    data_seed: int
    demo_episode_ids: tuple
    train_episode_ids: tuple
    val_episode_ids: tuple
    probe_manifest: object = field(repr=False)
    qualification_manifest: object = field(repr=False)
    probe_pool_seed: int | None = None
    qualification_pool_seed: int | None = None

    def identity(self) -> dict:
        return {
            "replicate": self.replicate,
            "data_seed": self.data_seed,
            "demo_episode_ids": list(self.demo_episode_ids),
            "train_episode_ids": list(self.train_episode_ids),
            "val_episode_ids": list(self.val_episode_ids),
            "probe_manifest_id": self.probe_manifest.manifest_id,
            "probe_case_ids": [c.case_id for c in self.probe_manifest.episodes],
            "qualification_manifest_id": self.qualification_manifest.manifest_id,
            "qualification_case_ids": [c.case_id for c in self.qualification_manifest.episodes],
            "probe_pool_seed": self.probe_pool_seed,
            "qualification_pool_seed": self.qualification_pool_seed,
        }

    @property
    def bundle_id(self) -> str:
        return f"r{self.replicate}:{_digest(self.identity())[:16]}"

    def split_info(self) -> dict:
        return {
            "mode": "by_episode",
            "bundle_id": self.bundle_id,
            "seed": self.data_seed,
            "train_episodes": len(self.train_episode_ids),
            "val_episodes": len(self.val_episode_ids),
            "val_fraction": len(self.val_episode_ids) / max(1, len(self.demo_episode_ids)),
        }

    def fitting_data(self, demo_buffer):
        """The bundle's sampled episodes as fitting tensors, split exactly as the bundle says."""
        import torch

        from lares.fitting.optimizers import FittingData

        groups = demo_buffer.episode_index()
        missing = [e for e in self.demo_episode_ids if e not in groups]
        if missing:
            raise ReplicateBundleError(f"{self.bundle_id}: episodes {missing[:3]} are not in the buffer")
        train_idx = sorted(i for e in self.train_episode_ids for i in groups[e])
        val_idx = sorted(i for e in self.val_episode_ids for i in groups[e])
        train_obs, train_actions = demo_buffer.take(train_idx)
        val_obs, val_actions = demo_buffer.take(val_idx) if val_idx else (None, None)
        info = self.split_info()
        info.update(train_transitions=len(train_idx), val_transitions=len(val_idx))
        return FittingData(
            train_obs=torch.tensor(train_obs, dtype=torch.float32),
            train_actions=torch.tensor(train_actions, dtype=torch.float32),
            val_obs=torch.tensor(val_obs, dtype=torch.float32) if val_idx else None,
            val_actions=torch.tensor(val_actions, dtype=torch.float32) if val_idx else None,
            train_indices=np.asarray(train_idx),
            split_info=info,
        )

    def to_dict(self) -> dict:
        return {"bundle_id": self.bundle_id, **self.identity()}


def sample_demonstration_episodes(episode_ids, n: int, seed: int) -> tuple:
    """``n`` distinct episodes drawn without replacement, in a seed-determined draw, sorted."""
    ids = sorted(str(e) for e in episode_ids if e)
    if int(n) > len(ids):
        raise ReplicateBundleError(f"asked for {n} demonstration episodes, the pool has {len(ids)}")
    order = np.random.default_rng(int(seed)).permutation(len(ids))
    return tuple(sorted(ids[i] for i in order[: int(n)]))


def split_episodes(episode_ids, val_fraction: float, seed: int) -> tuple:
    """Trajectory-level split: whole episodes go to train or validation, never both."""
    ids = sorted(episode_ids)
    if len(ids) < 2:
        raise ReplicateBundleError("a trajectory-level split needs at least two episodes")
    order = np.random.default_rng(int(seed) + 1).permutation(len(ids))
    n_val = max(1, int(round(float(val_fraction) * len(ids))))
    val = {ids[i] for i in order[:n_val]}
    return tuple(e for e in ids if e not in val), tuple(sorted(val))


def build_replicate_bundles(demo_buffer, data_seeds, demos_per_replicate: int, val_fraction: float,
                            manifest_factory, probe_pool_seeds, qualification_pool_seeds,
                            probe_episodes: int, qualification_episodes: int) -> list:
    """One independent bundle per replicate.

    ``manifest_factory(role, pool_seed, episodes, replicate)`` returns an
    :class:`~lares.eval.manifest.EvaluationManifest`; ``role`` is ``"probe"`` or
    ``"qualification"``. In a live study it builds from ``MT1(env, seed=pool_seed)``; in tests it
    returns a synthetic manifest. Seeds must be distinct across replicates and roles, the probe and
    qualification manifests of a replicate must share no placement, and no two replicates may draw
    the same demonstration sample or reuse a manifest.
    """
    data_seeds = [int(s) for s in data_seeds]
    probe_pool_seeds = [int(s) for s in probe_pool_seeds]
    qualification_pool_seeds = [int(s) for s in qualification_pool_seeds]
    if not (len(data_seeds) == len(probe_pool_seeds) == len(qualification_pool_seeds)):
        raise ReplicateBundleError("each replicate needs a data seed, a probe pool seed and a qualification pool seed")
    for name, seeds in (("data", data_seeds), ("pool", probe_pool_seeds + qualification_pool_seeds)):
        if len(set(seeds)) != len(seeds):
            raise ReplicateBundleError(f"{name} seeds repeat: {seeds}; replicates would share data")
    pool = list(demo_buffer.episode_index())
    bundles = []
    for r, seed in enumerate(data_seeds):
        sample = sample_demonstration_episodes(pool, demos_per_replicate, seed)
        train, val = split_episodes(sample, val_fraction, seed)
        probe = manifest_factory("probe", probe_pool_seeds[r], int(probe_episodes), r)
        qualification = manifest_factory("qualification", qualification_pool_seeds[r], int(qualification_episodes), r)
        shared = _placements(probe) & _placements(qualification)
        if shared:
            raise ReplicateBundleError(
                f"replicate {r}: probe and qualification manifests share {len(shared)} placement(s); checkpoint "
                f"selection and qualification must use different cases"
            )
        bundles.append(ReplicateBundle(r, seed, sample, train, val, probe, qualification,
                                       probe_pool_seeds[r], qualification_pool_seeds[r]))
    samples = [b.demo_episode_ids for b in bundles]
    if len(set(samples)) != len(samples):
        raise ReplicateBundleError("two replicates drew the same demonstration sample")
    manifest_ids = [m.manifest_id for b in bundles for m in (b.probe_manifest, b.qualification_manifest)]
    if len(set(manifest_ids)) != len(manifest_ids):
        raise ReplicateBundleError("two replicates reuse a manifest")
    return bundles


def check_shared_bundles(results, bundles) -> list:
    """Problems when a structure was not scored on its replicate's one bundle.

    ``results`` rows carry ``structure_id``, ``replicate`` and ``bundle_id``.
    """
    expected = {b.replicate: b.bundle_id for b in bundles}
    problems = []
    structures = sorted({row["structure_id"] for row in results})
    for row in results:
        if expected.get(row["replicate"]) != row["bundle_id"]:
            problems.append(
                f"{row['structure_id']} replicate {row['replicate']} used bundle {row['bundle_id']}, "
                f"not the replicate's {expected.get(row['replicate'])}"
            )
    for replicate in expected:
        seen = {row["structure_id"] for row in results if row["replicate"] == replicate}
        if seen != set(structures):
            problems.append(f"replicate {replicate} scored {sorted(seen)}, not every structure {structures}")
    return problems


def qualification_checks(differences, rule) -> dict:
    """The seed-bank rule on one structure's per-replicate paired differences."""
    arr = np.asarray(differences, dtype=float)
    improvements = int((arr > 0).sum())
    worst = float(arr.min()) if arr.size else None
    mean = float(arr.mean()) if arr.size else None
    checks = {
        "improves_in_enough_seeds": improvements >= int(rule["improves_in_at_least"]),
        "no_seed_regresses_beyond_tolerance": worst is not None and worst >= float(rule["worst_seed_difference_at_least"]) - 1e-12,
        "mean_difference_positive": mean is not None and mean > float(rule["mean_difference_greater_than"]),
    }
    return {"improvements": improvements, "worst": worst, "mean": mean, "checks": checks,
            "qualifies": all(checks.values())}


def run_replicate_qualification(structure_ids, bundles, demo_buffer, fit_and_score, rule, controls=()) -> dict:
    """Fit and score every structure on every replicate's shared bundle, then qualify.

    ``fit_and_score(structure_id, bundle, data)`` returns ``{"zero_shot_success",
    "selected_success", "selected_step"}`` scored on ``bundle.qualification_manifest`` with the
    checkpoint chosen on ``bundle.probe_manifest``. A selected step of 0 is fitting rejected: its
    difference is exactly zero whatever the scores say. Controls are scored the same way and
    reported, but never qualify.
    """
    rows = []
    for bundle in bundles:
        data = bundle.fitting_data(demo_buffer)
        for sid in list(structure_ids) + list(controls):
            outcome = fit_and_score(sid, bundle, data)
            rejected = int(outcome["selected_step"]) == 0
            difference = 0.0 if rejected else float(outcome["selected_success"]) - float(outcome["zero_shot_success"])
            rows.append({
                "structure_id": sid, "replicate": bundle.replicate, "bundle_id": bundle.bundle_id,
                "data_split_id": bundle.split_info()["bundle_id"], "selected_step": int(outcome["selected_step"]),
                "fitting_rejected": rejected, "zero_shot_success": float(outcome["zero_shot_success"]),
                "selected_success": float(outcome["selected_success"]), "difference": difference,
            })
    problems = check_shared_bundles(rows, bundles)
    if problems:
        raise ReplicateBundleError("structures did not share their replicate's bundle:\n  " + "\n  ".join(problems))
    structures = {}
    for sid in list(structure_ids) + list(controls):
        mine = [r for r in rows if r["structure_id"] == sid]
        result = qualification_checks([r["difference"] for r in mine], rule)
        if sid in set(controls):
            result = {**result, "qualifies": False, "control": True}
        structures[sid] = {**result, "rows": mine}
    return {
        "bundles": [b.to_dict() for b in bundles],
        "structures": structures,
        "seed_bank": sorted(sid for sid, s in structures.items() if s["qualifies"]),
    }
