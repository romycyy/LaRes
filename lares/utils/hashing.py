"""One set of digest helpers for every ledger, record and freeze.

Two conventions exist on disk and both are kept: full 64-character SHA-256 for
run records, manifests and ledgers, and a 16-character prefix for the code
hashes stored by :mod:`lares.search.schemas` and :mod:`lares.eval.freeze`.
The prefix length is a stored format; do not change it.
"""

from __future__ import annotations

import hashlib
import json

SHORT_HASH_LENGTH = 16


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(payload) -> str:
    """Digest of a JSON-serialisable payload with sorted keys."""
    return sha256_text(json.dumps(payload, sort_keys=True, default=str))


def short_hash_text(text: str) -> str:
    return sha256_text(text)[:SHORT_HASH_LENGTH]


def short_hash_file(path: str) -> str:
    return sha256_file(path)[:SHORT_HASH_LENGTH]
