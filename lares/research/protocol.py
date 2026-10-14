"""Checks a research idea must pass before anything is spent on it.

``check_idea`` returns every problem rather than the first, in the style of the
policy validator. ``start_run`` is the only way an idea becomes ``running``: it
re-runs the checks, requires approval, and records the Git SHA so every result
names exact code.
"""

from __future__ import annotations

import json
import os

from lares.utils import seeds as seeds_module
from lares.utils.gitstate import git, worktree_state  # noqa: F401  (re-exported)
from lares.research.schemas import (
    STATUS_APPROVED,
    STATUS_RUNNING,
    ResearchIdea,
    ResearchSchemaError,
    allowed_metrics,
    load_ideas,
    now,
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEED_LEDGER = seeds_module.SEED_LEDGER
spent_seeds = seeds_module.spent_seeds


def reserved_seeds(root, exclude_id: str = "") -> dict:
    """Seeds other open ideas have declared, so two proposals cannot claim the same number."""
    out = {}
    for idea in load_ideas(root):
        if idea.id == exclude_id or idea.status in ("rejected", "decided"):
            continue
        for s in idea.protocol.seeds:
            out[int(s)] = idea.id
    return out


def check_idea(idea: ResearchIdea, root, project_root: str = _PROJECT_ROOT,
               ledger_path: str = SEED_LEDGER, metrics: set | None = None) -> list:
    """Every reason the idea may not be run. Empty means it may."""
    problems = []
    try:
        idea.validate(metrics if metrics is not None else allowed_metrics())
    except ResearchSchemaError as e:
        problems.append(str(e))
    proto = idea.protocol
    script = os.path.join(project_root, proto.script)
    if not os.path.isfile(script):
        problems.append(f"protocol script {proto.script!r} does not exist (a new script must be written first)")
    if proto.config and not os.path.isfile(os.path.join(project_root, proto.config)):
        problems.append(f"protocol config {proto.config!r} does not exist")
    for m in proto.manifests:
        if not os.path.isfile(os.path.join(project_root, m)):
            problems.append(f"manifest {m!r} does not exist")
    used = spent_seeds(ledger_path, manifest_globs=None if ledger_path == SEED_LEDGER else [])
    for s in proto.seeds:
        if int(s) in used:
            problems.append(f"seed {s} is spent ({used[int(s)]})")
    held = reserved_seeds(root, exclude_id=idea.id)
    for s in proto.seeds:
        if int(s) in held:
            problems.append(f"seed {s} is already declared by {held[int(s)]}")
    if proto.episodes is None and proto.needs_gpu:
        problems.append("protocol does not estimate its simulator episodes")
    if proto.needs_llm and proto.llm_calls is None:
        problems.append("protocol uses the LLM but does not estimate its calls")
    return problems


def start_run(idea: ResearchIdea, root, log_dir: str = "", require_clean: bool = False,
              project_root: str = _PROJECT_ROOT, ledger_path: str = SEED_LEDGER) -> ResearchIdea:
    """Mark an approved idea running, after re-checking it and recording the code it runs on."""
    if idea.status != STATUS_APPROVED:
        raise ResearchSchemaError(f"{idea.id} is {idea.status}; only an approved idea can start")
    problems = check_idea(idea, root, project_root, ledger_path)
    if problems:
        raise ResearchSchemaError(f"{idea.id} cannot start:\n  - " + "\n  - ".join(problems))
    state = worktree_state(project_root)
    if require_clean and state["dirty"]:
        raise ResearchSchemaError(f"{idea.id}: worktree is dirty, commit first:\n" + "\n".join(state["dirty_files"]))
    idea.run = {"started_at": now(), "git_sha": state["git_sha"], "dirty_worktree": state["dirty"], "log_dir": log_dir}
    idea.transition(STATUS_RUNNING, f"git {state['git_sha'][:10]}" + (" (dirty)" if state["dirty"] else ""))
    idea.save(root)
    return idea
