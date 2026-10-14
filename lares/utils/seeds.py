"""The experiment seed ledger: one reader, one conflict rule, one writer.

A seed number, once looked at, is not quietly reused under another label. A
request is refused if the seed appears in the ledger *or* as the pool seed of
any saved manifest under ``config/`` or ``logs/``.
"""

from __future__ import annotations

import glob
import json
import os

import yaml

from lares.utils.gitstate import PROJECT_ROOT

SEED_LEDGER = os.path.join(PROJECT_ROOT, "config", "experiment_seed_ledger.jsonl")
MANIFEST_SEED_GLOBS = (
    os.path.join(PROJECT_ROOT, "config", "**", "*.yaml"),
    os.path.join(PROJECT_ROOT, "logs", "**", "*.yaml"),
)


def read_ledger(path: str = SEED_LEDGER) -> list:
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def manifest_pool_seeds(manifest_globs=None) -> dict:
    """``{seed: relative manifest path}`` for every saved manifest's pool seeds.

    Reads the YAML directly rather than building an ``EvaluationManifest``: only
    ``pool.seeds`` is needed, and this keeps the ledger CLI free of torch.
    """
    out = {}
    for pattern in manifest_globs or MANIFEST_SEED_GLOBS:
        for path in sorted(glob.glob(pattern, recursive=True)):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    doc = yaml.safe_load(f)
                seeds = doc["pool"]["seeds"]
            except Exception:
                continue
            for s in seeds:
                out.setdefault(int(s), os.path.relpath(path, PROJECT_ROOT))
    return out


def spent_seeds(ledger_path: str = SEED_LEDGER, manifest_globs=None) -> dict:
    """``{seed: use}`` for every seed the ledger or a saved manifest has used."""
    used = {int(e["seed"]): f"{e['namespace']}: {e['use']}" for e in read_ledger(ledger_path)}
    for seed, path in manifest_pool_seeds(manifest_globs).items():
        used.setdefault(seed, f"manifest {path}")
    return used


def seed_conflicts(pool_seeds, fitting_seeds=(), ledger_path: str = SEED_LEDGER, manifest_globs=None) -> list:
    """Every recorded use of a requested seed, in either namespace."""
    requested = {int(s) for s in pool_seeds} | {int(s) for s in fitting_seeds}
    conflicts = [
        f"ledger: seed {e['seed']} ({e['namespace']}) used for {e['use']}"
        for e in read_ledger(ledger_path) if int(e["seed"]) in requested
    ]
    by_path: dict[str, set] = {}
    for seed, path in manifest_pool_seeds(manifest_globs).items():
        if seed in requested:
            by_path.setdefault(path, set()).add(seed)
    for path, seeds in by_path.items():
        conflicts.append(f"manifest {path} uses pool seed(s) {sorted(seeds)}")
    return conflicts


def reserve_seeds(entries, ledger_path: str = SEED_LEDGER) -> None:
    with open(ledger_path, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
