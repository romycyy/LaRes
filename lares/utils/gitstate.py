"""Git identity for every recorded run: one SHA reader, one dirty-worktree rule.

Every protocol that records "which code ran" goes through here, so a research
run and a stage run agree on what "clean" means (no modified, staged or
untracked non-ignored file).
"""

from __future__ import annotations

import os
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def git(*args, cwd: str = PROJECT_ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def worktree_state(cwd: str = PROJECT_ROOT) -> dict:
    """``{"git_sha", "dirty", "dirty_files"}`` for the checkout at ``cwd``."""
    sha = git("rev-parse", "HEAD", cwd=cwd).strip()
    dirty = git("status", "--porcelain", "--untracked-files=all", cwd=cwd).strip()
    return {"git_sha": sha, "dirty": bool(dirty), "dirty_files": dirty.splitlines()}


def require_clean_worktree(purpose: str = "This run", cwd: str = PROJECT_ROOT) -> str:
    """Return HEAD's SHA, refusing any modified, staged or untracked (non-ignored) file."""
    state = worktree_state(cwd)
    if state["dirty"]:
        raise SystemExit(
            f"{purpose} needs a clean worktree so every result names exact code; commit or remove:\n"
            + "\n".join(state["dirty_files"])
        )
    return state["git_sha"]
