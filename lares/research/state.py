"""Render ``research/STATE.md``: the one page both the implementer and the advisor read first.

Generated from the ledgers only. Hand edits are overwritten on the next render,
which is the point: the prose files drifted because they were the record.
"""

from __future__ import annotations

import os

from lares.research.schemas import (
    PROPOSALS_LOG,
    STATE_FILE,
    STATUS_ANALYSED,
    STATUS_APPROVED,
    STATUS_DECIDED,
    STATUS_DEFERRED,
    STATUS_ESCALATED,
    STATUS_PROPOSED,
    STATUS_REJECTED,
    STATUS_RUNNING,
    STAGE_TRIAGE,
    TRIAGE_LOG,
    load_constraints,
    load_decisions,
    load_direction,
    load_ideas,
    load_jsonl,
    now,
    open_triage_round,
)
from lares.research.protocol import spent_seeds, split_seed_uses

ORDER = (STATUS_ESCALATED, STATUS_ANALYSED, STATUS_RUNNING, STATUS_APPROVED, STATUS_PROPOSED)
NEXT_ACTION = {
    STATUS_ANALYSED: "implementer verdict, advisor analyse, then decide: `python scripts/research.py verdict {id} ...` / `analyse {id}` / `decide {id} --review --handoff`",
    STATUS_RUNNING: "run the protocol, then `python scripts/research.py record {id} --results <json>`",
    STATUS_APPROVED: "`python scripts/research.py start {id}` then run its command",
}


def _escalation_action(idea) -> str:
    if idea.escalation.get("stage") == STAGE_TRIAGE:
        return f"USER must rule: `python scripts/research.py rule {idea.id} --outcome <approve|reject|defer> --note ...`"
    return f"USER must rule: `python scripts/research.py decide {idea.id} --ruling <verdict> --note ...`"


def _proposed_action(idea) -> str:
    if open_triage_round(idea):
        rnd = idea.triage["rounds"][-1]
        return (f"advisor said **{rnd['advisor']['decision']}** in round {rnd['round']}; implementer answers: "
                f"`python scripts/research.py position {idea.id} --position <approve|revise|defer|reject> --reasoning ...`")
    rounds = len(idea.triage.get("rounds", []))
    if rounds:
        return f"round {rounds} closed without a final agreement; next round: `python scripts/research.py triage {idea.id}`"
    return f"triage: `python scripts/research.py triage {idea.id}`"


def loop_guard(root) -> dict:
    """How many recent propose calls ended with every idea rejected, so the loop can stop and ask for a direction."""
    proposals = load_jsonl(root, PROPOSALS_LOG)
    ideas = {i.id: i for i in load_ideas(root)}
    by_call: dict[str, list] = {}
    for p in proposals:
        by_call.setdefault(p["call_id"], []).append(p)
    streak = 0
    for call_id in reversed(list(by_call)):
        batch = by_call[call_id]
        statuses = [ideas[p["idea_id"]].status for p in batch if p.get("idea_id") in ideas]
        if statuses and all(s == STATUS_REJECTED for s in statuses):
            streak += 1
        else:
            break
    return {"propose_calls": len(by_call), "consecutive_all_rejected": streak}


def render_state(root, recent: int = 5) -> str:
    ideas = load_ideas(root)
    decisions = load_decisions(root)
    direction = load_direction(root)
    by_status = {s: [i for i in ideas if i.status == s] for s in ORDER}
    lines = [
        "# Research state",
        "",
        f"_Generated {now()} by `python scripts/research.py state` from `research/ideas/`, "
        f"`research/decisions.jsonl`, `research/direction.json`. Do not edit by hand._",
        "",
        "## Direction",
        "",
        f"{direction['text']}  _(set {direction.get('set_at', '')[:10]})_" if direction.get("text")
        else "- none set. The user sets one with `python scripts/research.py direct \"...\"`.",
        "",
        "| status | count |",
        "|---|---|",
    ]
    for s in ORDER + (STATUS_DEFERRED, STATUS_DECIDED, STATUS_REJECTED):
        lines.append(f"| {s} | {sum(1 for i in ideas if i.status == s)} |")
    guard = loop_guard(root)
    if guard["consecutive_all_rejected"]:
        lines += ["", f"**Loop guard:** the last {guard['consecutive_all_rejected']} propose call(s) were rejected in full at "
                      f"triage. After two in a row the loop stops and asks the user for a new direction."]
    lines += ["", "## Needs attention", ""]
    any_open = False
    for s in ORDER:
        for i in by_status[s]:
            any_open = True
            lines.append(f"- **{i.id}** [{s}] {i.title}")
            if s == STATUS_ESCALATED:
                lines.append(f"  - {_escalation_action(i)}")
                lines.append(f"  - {i.escalation.get('reason', '')}")
                for v in i.verdicts:
                    lines.append(f"  - {v['source']}: {v['verdict']} ({v['confidence']:.2f}) {v.get('reasoning', '')[:200]}")
                for r in i.triage.get("rounds", []):
                    imp = r.get("implementer") or {}
                    lines.append(f"  - triage round {r['round']}: advisor {r['advisor']['decision']} / implementer {imp.get('position', '-')}")
            elif s == STATUS_PROPOSED:
                lines.append(f"  - {_proposed_action(i)}")
            else:
                lines.append(f"  - {NEXT_ACTION[s].format(id=i.id)}")
    if not any_open:
        lines.append("- Nothing open. Next: `python scripts/research.py propose` (uses the direction above), then `triage`.")
    deferred = [i for i in ideas if i.status == STATUS_DEFERRED]
    if deferred:
        lines += ["", "## Deferred (right idea, not now; `triage R-xxx` re-opens one)", ""]
        for i in deferred:
            last = (i.triage.get("rounds") or [{}])[-1].get("advisor", {})
            lines.append(f"- {i.id} {i.title}: {last.get('reasoning', '')[:160]}")
    lines += ["", f"## Last {recent} decisions", ""]
    for d in list(reversed(decisions))[:recent] or []:
        tag = "" if d["agreed"] else " (user ruling)"
        lines.append(f"- {d['id']} · {d['idea_id']} · **{d['outcome']}**{tag} · {d['recorded_at'][:10]}")
        lines.append(f"  {d['summary']}")
        if d.get("next_step"):
            lines.append(f"  Next: {d['next_step']}")
    if not decisions:
        lines.append("- none yet")
    triage_log = load_jsonl(root, TRIAGE_LOG)
    if triage_log:
        lines += ["", f"## Last {recent} triage rounds (full log: `research/{TRIAGE_LOG}`)", ""]
        for t in list(reversed(triage_log))[:recent]:
            if t.get("outcome") == "user_ruling":
                lines.append(f"- {t['idea_id']} · user ruling → {t['status']}: {t['user_ruling']['note'][:120]}")
            else:
                imp = t.get("implementer") or {}
                lines.append(f"- {t['idea_id']} round {t['round']} · advisor {t['advisor']['decision']} / "
                             f"implementer {imp.get('position', '-')} → {t['outcome']} ({t['status']})")
    lines += ["", "## Standing constraints", ""]
    for c in load_constraints(root):
        lines.append(f"- {c['fact']} _({c['source']})_")
    consumed, reserved = split_seed_uses(spent_seeds(), load_ideas(root))
    lines.append(f"- Spent seeds: {sorted(consumed)}")
    if reserved:
        lines.append(f"- Reserved seeds (an open idea's own frozen manifest): {dict(sorted(reserved.items()))}")
    lines += ["", "## Pointers", "",
              "- Narrative findings: `RESEARCH_HANDOFF.md` (appended once per decision)",
              "- Frozen history: `IMPLEMENTATION_PROGRESS.md`, `DEBUG_PROGRESS.md`",
              f"- Every proposal: `research/{PROPOSALS_LOG}`; every triage round: `research/{TRIAGE_LOG}`",
              "- Advisor calls: `research/llm_calls.jsonl`; briefings: `research/briefings/`",
              ""]
    return "\n".join(lines)


def write_state(root) -> str:
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, STATE_FILE)
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_state(root))
    return path
