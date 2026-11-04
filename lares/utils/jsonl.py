"""One JSON-lines reader and one appender for every ledger in the tree.

The seed ledger, the final-test ledger, the research decisions and the LLM call
ledgers are all append-only JSONL files; they share the blank-line rule and the
"missing file is an empty ledger" rule here.
"""

from __future__ import annotations

import json
import os
from typing import Iterable


def read_jsonl(path: str) -> list:
    """Every non-blank line of ``path`` parsed as JSON; a missing file is an empty list."""
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: str, entries: Iterable, *, ensure_ascii: bool = True) -> str:
    """Append each entry as one line, creating the parent directory; returns ``path``."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=ensure_ascii) + "\n")
    return path
