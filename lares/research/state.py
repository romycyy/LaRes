"""Render ``research/STATE.md``: the one page both the implementer and the advisor read first.

Generated from the ledgers only. Hand edits are overwritten on the next render,
which is the point: the prose files drifted because they were the record.
"""

from __future__ import annotations

import os

from lares.research.schemas import (
    STATE_FILE,
    STATUS_ANALYSED,
    STATUS_APPROVED,
    STATUS_DECIDED,
    STATUS_ESCALATED,
    STATUS_PROPOSED,
    STATUS_REJECTED,
    STATUS_RUNNING,
    load_constraints,
    load_decisions,
    load_ideas,
    now,
)
from lares.research.protocol import spent_seeds

ORDER = (STATUS_ESCALATED, STATUS_ANALYSED, STATUS_RUNNING, STATUS_APPROVED, STATUS_PROPOSED)
NEXT_ACTION = {
    STATUS_ESCALATED: "USER must rule: `python scripts/research.py decide {id} --ruling <verdict> --note ...`",
    STATUS_ANALYSED: "implementer verdict then decide: `python scripts/research.py verdict {id} ...` / `decide {id}`",
    STATUS_RUNNING: "run the protocol, then `python scripts/research.py record {id} --results <json>`",
    STATUS_APPROVED: "`python scripts/research.py start {id}` then run its command",
    STATUS_PROPOSED: "USER decides: `python scripts/research.py approve {id}` or `reject {id} --reason ...`",
}


def render_state(root, recent: int = 5) -> str:
    ideas = load_ideas(root)
    decisions = load_decisions(root)
    by_status = {s: [i for i in ideas if i.status == s] for s in ORDER}
    lines = [
        "# Research state",
        "",
        f"_Generated {now()} by `python scripts/research.py state` from `research/ideas/` and "
        f"`research/decisions.jsonl`. Do not edit by hand._",
        "",
        "| status | count |",
        "|---|---|",
    ]
    for s in ORDER + (STATUS_DECIDED, STATUS_REJECTED):
        lines.append(f"| {s} | {sum(1 for i in ideas if i.status == s)} |")
    lines += ["", "## Needs attention", ""]
    any_open = False
    for s in ORDER:
        for i in by_status[s]:
            any_open = True
            lines.append(f"- **{i.id}** [{s}] {i.title}")
            lines.append(f"  - {NEXT_ACTION[s].format(id=i.id)}")
            if s == STATUS_ESCALATED:
                for v in i.verdicts:
                    lines.append(f"  - {v['source']}: {v['verdict']} ({v['confidence']:.2f}) {v.get('reasoning', '')[:200]}")
    if not any_open:
        lines.append("- Nothing open. Run `python scripts/research.py propose` to ask the advisor for ideas.")
    lines += ["", f"## Last {recent} decisions", ""]
    for d in list(reversed(decisions))[:recent] or []:
        tag = "" if d["agreed"] else " (user ruling)"
        lines.append(f"- {d['id']} · {d['idea_id']} · **{d['outcome']}**{tag} · {d['recorded_at'][:10]}")
        lines.append(f"  {d['summary']}")
        if d.get("next_step"):
            lines.append(f"  Next: {d['next_step']}")
    if not decisions:
        lines.append("- none yet")
    lines += ["", "## Standing constraints", ""]
    for c in load_constraints(root):
        lines.append(f"- {c['fact']} _({c['source']})_")
    used = spent_seeds()
    lines.append(f"- Spent seeds: {sorted(used)}")
    lines += ["", "## Pointers", "",
              "- Narrative findings: `RESEARCH_HANDOFF.md` (appended once per decision)",
              "- Frozen history: `IMPLEMENTATION_PROGRESS.md`, `DEBUG_PROGRESS.md`",
              "- Advisor calls: `research/llm_calls.jsonl`; briefings: `research/briefings/`",
              ""]
    return "\n".join(lines)


def write_state(root) -> str:
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, STATE_FILE)
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_state(root))
    return path
