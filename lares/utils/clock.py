"""The two timestamp formats every record and run directory uses."""

from __future__ import annotations

import time

ISO_SECONDS = "%Y-%m-%dT%H:%M:%S"
RUN_STAMP = "%Y%m%d_%H%M%S"


def now_iso() -> str:
    """Local wall-clock time to the second, as stored in ledgers and records."""
    return time.strftime(ISO_SECONDS)


def run_stamp() -> str:
    """Local wall-clock time as a filesystem-safe run id (``20260914_032500``)."""
    return time.strftime(RUN_STAMP)
