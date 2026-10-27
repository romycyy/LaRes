---
name: research-cycle
description: Run the LaRes auto-research loop (propose -> triage with the advisor until agreement -> implement -> run -> record -> verdict/analyse until agreement -> decide -> propose again) through scripts/research.py and the research/ ledger. Use when the user gives a research direction, says "next research step", "run the research cycle", "keep going", "what should we test next", or invokes /research-cycle.
---

# Research cycle

The ledger under `research/` is the record. `RESEARCH_HANDOFF.md` is narrative only and is appended
by `research.py decide --handoff`. `IMPLEMENTATION_PROGRESS.md` and `DEBUG_PROGRESS.md` are frozen
history; never add to them. `research/STATE.md` is generated; never hand-edit it.

## Roles

- **User**: gives a research direction (`research.py direct "..."`) and nothing else. They do not
  review ideas. They rule only on an idea that exhausted its negotiation rounds (`escalated`).
- **Advisor** (gpt-6.1-sol, `research.py propose|triage|analyse|decide --review`): proposes ideas under
  the direction, decides approve / revise / defer / reject per idea, judges results from numbers.
- **You, the implementer**: state your *own* triage position and verdict, argue when you disagree,
  write every line of code, run every protocol, record every number. An idea moves only when you
  and the advisor state the same position. You never use `rule` or `decide --ruling`: those are the
  user's.

Every proposal is logged to `research/proposals.jsonl`, every triage round (advisor decision,
your position, both reasonings) to `research/triage.jsonl`, every advisor call to
`research/llm_calls.jsonl`. These exist so the user can inspect the loop later; write your
reasoning for them, not for the terminal.

## The loop

Run it until the user interrupts, the loop guard stops it, or something escalates. One pass:

1. **Orient.** `python scripts/research.py state`. If the user's message contains a direction, set
   it first: `python scripts/research.py direct "<their words>"`. If no direction is set and the user
   gave none, ask for one and stop. Take the first "Needs attention" item (escalated → report to the
   user and stop; analysed, running, approved, proposed → continue it).
2. **Propose** (nothing open): `python scripts/research.py propose --n 3`. Then `triage`.
3. **Triage** (proposed ideas): `python scripts/research.py triage`. For *each* idea, form your own
   position before reading the advisor's reasoning closely: is the mechanism new against the decided
   ideas, is the protocol runnable with existing scripts, is the cost proportionate, does it serve
   the direction? Then answer:
   `python scripts/research.py position R-xxx --position <approve|revise|defer|reject> --reasoning "<your argument>"`.
   - **Agreed approve** → go to step 4. **Agreed reject/defer** → idea closed; continue with the rest.
   - **Agreed revise** → write a patch file with the agreed edits and apply it:
     `python scripts/research.py revise R-xxx --patch <file.json> --note "..."`, then `triage R-xxx` again.
   - **Disagreed** → `python scripts/research.py triage R-xxx` for the next round. The advisor sees your
     objection; read its answer and either hold with a *new* argument or change your position. Never
     switch to a middle option to end it. After the round cap (default 5) the idea escalates: report
     both positions to the user and stop.
   - If every idea of a propose call is rejected twice in a row, `propose` refuses: tell the user the
     direction is exhausted and ask for a new one.
4. **Implement.** `python scripts/research.py show R-xxx`. Write the code its protocol names, unit
   tests, then `gpu test <targets>`. Stay inside `protocol.code_changes`; note any excursion in the
   result notes.
5. **Start and run.** `python scripts/research.py start R-xxx --log-dir <dir>` (`--require-clean` for a
   frozen-stage protocol), then the protocol's exact command on the GPU box (`gpu start` once per
   session; `gpu run 'nohup ... &'` for long runs; `gpu get <log_dir> ./<log_dir>`). Do **not**
   `gpu stop` between steps; stop the box only when the whole session ends.
6. **Record.** Write `<log_dir>/research_result.json` with `metrics` (must include the prediction's
   metric), `controls`, `artifacts` (paths copied locally), `notes` (intervals, deviations, anything
   unmeasured). `python scripts/research.py record R-xxx --results <file>`.
7. **Verdict, yours first.** `python scripts/research.py verdict R-xxx --verdict ... --confidence ... --reasoning "..." --next "..."`.
   Then `python scripts/research.py analyse R-xxx`.
8. **Decide.** `python scripts/research.py decide R-xxx --summary "<one paragraph for the handoff>" --review --handoff`.
   Agreement writes `D-nnn` and appends the handoff. Disagreement: read the advisor's review, restate
   with `verdict` (new argument or changed verdict) and run `decide --review --handoff` again. After the
   round cap it escalates: report both verdicts to the user and stop.
9. **Close the pass.** `python scripts/research.py state`; commit `research/`, code and handoff together,
   message naming the idea id. Then return to step 2.

## Rules

- A protocol that names the final-test split, a spent seed, or an unmeasured metric is refused by
  `check_idea`; fix the idea (and say so in the triage round), never the check.
- Numbers go in `research_result.json`, not in prose. The advisor sees only what is recorded.
- Do not `propose` while an idea is approved, running or analysed: finish it first.
- When you report to the user, lead with what was decided and what is running; the per-round
  arguments are in `research/triage.jsonl`, not in your message.
