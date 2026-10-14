---
name: research-status
description: Show where the LaRes research loop stands (open ideas, escalations waiting on the user, last decisions) from the research/ ledger. Use when the user asks "where are we", "what's pending", "research status", or invokes /research-status.
---

# Research status

1. Run `python scripts/research.py state` from the project root. This regenerates `research/STATE.md`
   from `research/ideas/` and `research/decisions.jsonl` and prints it.
2. Report to the user, in this order and in plain prose:
   - anything **escalated** (both verdicts, and the exact `decide --ruling` command they can run);
   - anything **proposed** awaiting their approval (title, prediction, cost, protocol-check result);
   - anything **approved / running / analysed** that you can advance yourself, and say you will if asked;
   - the last decision and its proposed next step.
3. Do not approve, reject or rule. Do not call the advisor. This skill reads only.
