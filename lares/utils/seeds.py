"""The experiment seed ledger: one reader, one conflict rule, one writer.

A seed number, once looked at, is not quietly reused under another label. A
request is refused if the seed appears in the ledger *or* as the pool seed of
any saved manifest under ``config/`` or ``logs/``.
"""

from __future__ import annotations

import glob
import os

import yaml

from lares.utils.gitstate import PROJECT_ROOT
from lares.utils.jsonl import read_jsonl

_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

SEED_LEDGER = os.path.join(PROJECT_ROOT, "config", "experiment_seed_ledger.jsonl")
MANIFEST_SEED_GLOBS = (
    os.path.join(PROJECT_ROOT, "config", "**", "*.yaml"),
    os.path.join(PROJECT_ROOT, "logs", "**", "*.yaml"),
)


def read_ledger(path: str = SEED_LEDGER) -> list:
    return read_jsonl(path)


#: ``{glob patterns: {seed: path}}``: the manifest scan is a recursive glob plus a YAML parse of every file under
#: ``config/`` and ``logs/`` (over a thousand files), and one research CLI command asks for it several times
#: (briefing, state, one ``check_idea`` per idea). Nothing in those commands writes a manifest, so the first
#: scan in a process stands.
_MANIFEST_SEED_CACHE: dict[tuple, dict] = {}


def manifest_pool_seeds(manifest_globs=None) -> dict:
    """``{seed: relative manifest path}`` for every saved manifest's pool seeds.

    Reads the YAML directly rather than building an ``EvaluationManifest``: only
    ``pool.seeds`` is needed, and this keeps the ledger CLI free of torch. The
    result is cached per process and per set of globs; callers get a copy.
    """
    patterns = MANIFEST_SEED_GLOBS if manifest_globs is None else tuple(manifest_globs)
    cached = _MANIFEST_SEED_CACHE.get(patterns)
    if cached is None:
        cached = {}
        for pattern in patterns:
            for path in sorted(glob.glob(pattern, recursive=True)):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        doc = yaml.load(f, Loader=_YAML_LOADER)
                    seeds = doc["pool"]["seeds"]
                except Exception:
                    continue
                for s in seeds:
                    cached.setdefault(int(s), os.path.relpath(path, PROJECT_ROOT))
        _MANIFEST_SEED_CACHE[patterns] = cached
    return dict(cached)


def spent_seeds(ledger_path: str = SEED_LEDGER, manifest_globs=None) -> dict:
    """``{seed: use}`` for every seed the ledger or a saved manifest has used."""
    used = {int(e["seed"]): f"{e['namespace']}: {e['use']}" for e in read_ledger(ledger_path)}
    for seed, path in manifest_pool_seeds(manifest_globs).items():
        used.setdefault(seed, f"manifest {path}")
    return used
