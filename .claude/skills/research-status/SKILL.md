---
name: research-status
description: Show where the LaRes research loop stands (direction, open ideas and their triage rounds, escalations waiting on the user, last decisions) from the research/ ledger. Use when the user asks "where are we", "what's pending", "research status", "what did gpt decide", or invokes /research-status.
---

# Research status

1. Run `python scripts/research.py state` from the project root. This regenerates `research/STATE.md`
   from the ideas, decisions, direction and triage log, and prints it.
2. Report to the user in plain prose, in this order:
   - the current direction (or that none is set);
   - anything **escalated** (the triage positions or verdicts of both sides, and the exact `rule` or
     `decide --ruling` command the user can run);
   - ideas in triage: for each, the advisor's latest decision, your latest position, round number;
   - anything approved / running / analysed that you can advance yourself;
   - the last decision and its proposed next step;
   - the loop guard, if the last propose calls were rejected in full.
3. If the user asks what the advisor decided or why, read `research/triage.jsonl` (one line per
   round: idea, advisor decision and reasoning, implementer position and reasoning, outcome) and
   `research/proposals.jsonl` (every proposal, parsed or not), and quote from them.
4. Do not call the advisor, change any idea, or rule. This skill reads only.
