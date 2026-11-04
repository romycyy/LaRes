"""The packet the advisor model reads: built from the ledgers, hashed, and bounded in size.

Nothing in the packet is typed by hand at call time. Standing constraints come
from ``research/constraints.json`` (each with a source), spent seeds from the
seed ledger, open ideas and recent decisions from the research ledger, and the
list of scripts and configs from the tree. The packet is saved under
``research/briefings/`` and its hash is written into every idea it produced, so
"what did the advisor know when it said that" has an answer.
"""

from __future__ import annotations

import os

from lares.research.schemas import (
    BRIEFINGS_DIR,
    STATUS_DECIDED,
    STATUS_DEFERRED,
    STATUS_REJECTED,
    RESEARCH_METRICS,
    ResearchIdea,
    allowed_metrics,
    load_constraints,
    load_decisions,
    load_direction,
    load_ideas,
    now,
)
from lares.research.protocol import spent_seeds
from lares.utils.hashing import sha256_text

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CHAR_BUDGET = 60_000
HANDOFF = "RESEARCH_HANDOFF.md"


def _truncate(text: str, limit: int, label: str) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[... {label} truncated at {limit} characters ...]\n"


def constraints_block(root) -> str:
    lines = ["## Standing constraints (each with its source; do not propose anything that violates one)"]
    for c in load_constraints(root):
        lines.append(f"- {c['fact']}  _(source: {c['source']})_")
    used = spent_seeds()
    if used:
        lines.append(f"- Spent seeds (never reuse): {sorted(used)}")
    lines.append("- The final-test split is locked. No protocol may read or score it.")
    return "\n".join(lines)


def idea_summary(idea: ResearchIdea, with_result: bool = False) -> str:
    p = idea.prediction
    lines = [
        f"### {idea.id} [{idea.status}] {idea.title}",
        f"Hypothesis: {idea.hypothesis}",
        f"Prediction: {p.metric} {p.direction} by >= {p.threshold} vs {p.against or 'control'}",
        f"Falsified by: {idea.falsified_by}",
        f"Protocol: {idea.protocol.command or idea.protocol.script} (episodes={idea.protocol.episodes}, "
        f"llm_calls={idea.protocol.llm_calls}, seeds={idea.protocol.seeds})",
    ]
    if with_result and idea.result:
        lines.append(f"Result metrics: {idea.result.get('metrics')}")
        if idea.result.get("controls"):
            lines.append(f"Controls: {idea.result.get('controls')}")
        if idea.prediction_outcome:
            lines.append(f"Prediction outcome: {idea.prediction_outcome}")
        for v in idea.verdicts:
            lines.append(f"Verdict ({v['source']}): {v['verdict']} ({v['confidence']:.2f}) {v.get('reasoning', '')}")
    return "\n".join(lines)


def direction_block(root) -> str:
    d = load_direction(root)
    lines = ["## User's research direction"]
    if d.get("text"):
        lines.append(f"{d['text']}  _(set {d.get('set_at', '')[:10]})_")
    else:
        lines.append("(none set; the ledger's open questions and last decisions stand in for it)")
    for h in d.get("history", [])[-3:]:
        lines.append(f"- earlier direction ({h.get('set_at', '')[:10]} to {h.get('replaced_at', '')[:10]}): {h['text']}")
    return "\n".join(lines)


def triage_line(idea: ResearchIdea) -> str:
    rounds = idea.triage.get("rounds", [])
    if not rounds:
        return ""
    last = rounds[-1]
    imp = last.get("implementer") or {}
    return (f"Triage so far: {len(rounds)} round(s); last advisor {last['advisor']['decision']}, "
            f"implementer {imp.get('position', 'pending')}; {last['advisor']['reasoning'][:200]}")


def ledger_block(root, recent_decisions: int = 8, recent_rejected: int = 6) -> str:
    ideas = load_ideas(root)
    open_ideas = [i for i in ideas if i.status not in (STATUS_DECIDED, STATUS_REJECTED, STATUS_DEFERRED)]
    deferred = [i for i in ideas if i.status == STATUS_DEFERRED]
    rejected = [i for i in ideas if i.status == STATUS_REJECTED][-recent_rejected:]
    decided = [i for i in ideas if i.status == STATUS_DECIDED]
    parts = ["## Open ideas (already proposed; do not duplicate)"]
    for i in open_ideas:
        parts.append(idea_summary(i))
        t = triage_line(i)
        if t:
            parts.append(t)
    if not open_ideas:
        parts.append("(none)")
    parts.append("\n## Deferred ideas (judged right but not now; re-triaged when the direction fits)")
    parts += [f"- {i.id} {i.title}: {(i.triage.get('rounds') or [{}])[-1].get('advisor', {}).get('reasoning', '')[:200]}"
              for i in deferred] or ["(none)"]
    parts.append("\n## Recently rejected at triage (do not re-propose without saying what changed)")
    for i in rejected:
        why = (i.triage.get("rounds") or [{}])[-1].get("advisor", {}).get("reasoning", "") or (i.history[-1]["note"] if i.history else "")
        parts.append(f"- {i.id} {i.title}: {why[:200]}")
    if not rejected:
        parts.append("(none)")
    parts.append("\n## Decided ideas (most recent first)")
    by_id = {i.id: i for i in decided}
    for d in list(reversed(load_decisions(root)))[:recent_decisions]:
        idea = by_id.get(d["idea_id"])
        head = f"- {d['id']} on {d['idea_id']}: **{d['outcome']}**" + ("" if d["agreed"] else " (user ruling)")
        parts.append(head + (f" — {idea.title}" if idea else ""))
        parts.append(f"  {d['summary']}")
        if d.get("next_step"):
            parts.append(f"  Next step then proposed: {d['next_step']}")
    return "\n".join(parts)


def inventory_block(project_root: str = _PROJECT_ROOT) -> str:
    scripts = sorted(f for f in os.listdir(os.path.join(project_root, "scripts")) if f.endswith(".py"))
    configs = []
    for dirpath, _, files in os.walk(os.path.join(project_root, "config")):
        for f in files:
            if f.endswith((".yaml", ".json")) and "manifests" not in dirpath:
                configs.append(os.path.relpath(os.path.join(dirpath, f), project_root))
    return "\n".join([
        "## Available protocol scripts (reuse before writing new ones)",
        ", ".join(f"scripts/{s}" for s in scripts),
        "## Available configs",
        ", ".join(sorted(configs)),
        "## Metrics a prediction may name",
        ", ".join(sorted(allowed_metrics() or RESEARCH_METRICS)),
    ])


def handoff_block(project_root: str = _PROJECT_ROOT, limit: int = 20_000) -> str:
    """The 'Where this stands' section of the handoff plus the open-question list."""
    path = os.path.join(project_root, HANDOFF)
    if not os.path.isfile(path):
        return ""
    text = open(path, "r", encoding="utf-8").read()
    out = []
    for heading in ("## Where this stands", "## Questions requiring a research decision"):
        start = text.find(heading)
        if start < 0:
            continue
        nxt = text.find("\n## ", start + len(heading))
        out.append(text[start: nxt if nxt > 0 else len(text)])
    return _truncate("\n\n".join(out), limit, "handoff")


def build_briefing(root, purpose: str, extra: str = "", char_budget: int = DEFAULT_CHAR_BUDGET,
                   project_root: str = _PROJECT_ROOT) -> dict:
    """Assemble, bound, save and hash the packet. Returns ``{text, sha256, path}``."""
    sections = [
        f"# Research briefing ({purpose}) — generated {now()}",
        direction_block(root),
        constraints_block(root),
        ledger_block(root),
        inventory_block(project_root),
        handoff_block(project_root),
    ]
    if extra:
        sections.append(extra)
    text = "\n\n".join(s for s in sections if s)
    text = _truncate(text, char_budget, "briefing")
    digest = sha256_text(text)
    directory = os.path.join(root, BRIEFINGS_DIR)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{purpose}_{digest[:12]}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return {"text": text, "sha256": digest, "path": os.path.relpath(path, project_root)}
