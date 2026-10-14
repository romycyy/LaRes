---
name: research-cycle
description: Run one turn of the LaRes auto-research loop (propose -> approve -> implement -> run -> record -> analyse -> verdict -> decide) through scripts/research.py and the research/ ledger. Use when the user says "next research step", "run the research cycle", "what should we test next", "record the result of R-xxx", or invokes /research-cycle.
---

# Research cycle

The ledger under `research/` is the record. `RESEARCH_HANDOFF.md` is narrative only and is
appended by `research.py decide --handoff`. `IMPLEMENTATION_PROGRESS.md` and `DEBUG_PROGRESS.md`
are frozen history; never add to them.

Roles: the **advisor** (gpt-6.1-sol via `scripts/research.py propose|analyse`) proposes ideas and
judges numbers. **You** (the implementer) write every line of code, run every protocol, and record
your own verdict *before* reading the advisor's when possible. The **user** approves each idea
before any GPU or LLM spend and rules on escalations. Never approve, reject or rule yourself.

## Procedure

1. **Orient.** `python scripts/research.py state`. Read `research/STATE.md` top to bottom. Pick the
   first item under "Needs attention" in this order: escalated (stop and report to the user),
   analysed, running, approved, proposed (stop: the user must approve or reject).
2. **Nothing open?** `python scripts/research.py propose --n 3` (optionally `--focus "..."`; `--dry-run`
   first if you want to read the briefing). Report the saved ideas with their protocol-check
   result and **stop**: approval is the user's.
3. **Approved idea.** `python scripts/research.py show R-xxx`. If the check names code that does not
   exist, write it now: code first, unit tests second, `gpu test <targets>` third. Keep the change
   inside `protocol.code_changes`; if you must go outside it, say so in the result notes.
4. **Start.** `python scripts/research.py start R-xxx --log-dir <dir>` (add `--require-clean` when the
   protocol is a frozen-stage run). Then run the protocol's exact command on the GPU box:
   `gpu start`, `gpu run '<command>'` (use `nohup ... &` for long runs and poll with
   `gpu connect 'tail ...'`), `gpu get <log_dir> ./<log_dir>`, `gpu stop`. Always stop the box.
5. **Record.** Write `<log_dir>/research_result.json` with `metrics` (must include the prediction's
   metric), `controls` (the same metric for the control the prediction named), `artifacts` (paths the
   advisor may cite, copied locally), and `notes` (paired intervals, deviations, anything unmeasured).
   Then `python scripts/research.py record R-xxx --results <that file>`.
6. **Verdict, yours first.** Decide supported / refuted / inconclusive from the numbers and the
   predeclared threshold. Then
   `python scripts/research.py verdict R-xxx --verdict ... --confidence ... --reasoning "..." --next "..."`.
7. **Advisor verdict.** `python scripts/research.py analyse R-xxx`.
8. **Decide.** `python scripts/research.py decide R-xxx --summary "<one paragraph for the handoff>" --review --handoff`.
   Agreement writes `D-nnn` and appends the handoff. Disagreement escalates; report both verdicts to the
   user and stop. The user rules with `decide R-xxx --ruling <verdict> --note "..."`.
9. **Close.** `python scripts/research.py state`; commit `research/`, the code, and the handoff
   together with a message naming the idea id. Report: idea, result, decision, next step.

## Rules

- A protocol that names the final-test split, a spent seed, or an unmeasured metric is refused by
  `check_idea`; fix the idea file (and say so), never the check.
- Numbers go in `research_result.json`, not in prose. The advisor sees only what is recorded.
- Do not run `propose` while an idea is approved or running: finish it first.
- `research/STATE.md` is generated; never hand-edit it.
