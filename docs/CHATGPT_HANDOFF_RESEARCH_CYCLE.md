# The `/research-cycle` loop, summarised for ChatGPT (2026-11-22)

A summary of the Claude Code skill `.claude/skills/research-cycle/SKILL.md` (the source; read it for the exact
wording). It is how every idea from R-001 to R-102 was proposed, approved, run and decided. Companion to
`docs/CHATGPT_HANDOFF_2026-11-22.md`. Everything goes through `scripts/research.py` and the ledger under
`research/`, run from the project root.

## What is the record

- **The ledger in `research/` is the record**: `ideas/R-xxx.json` (one file per idea: hypothesis,
  prediction, protocol, triage rounds, result, verdicts), `decisions.jsonl` (the `D-nnn` decisions),
  `direction.json` (the user's direction and its history).
- Inspection logs, append-only: `proposals.jsonl` (every proposal, parsed or not), `triage.jsonl` (every
  triage round: advisor decision and reasoning, implementer position and reasoning, outcome),
  `llm_calls.jsonl` (every advisor call, with prompt/response files under `llm_calls/`), `briefings/`
  (the briefing each proposal was made from, by SHA).
- `research/STATE.md` is **generated** by `research.py state` (and by every command that changes an idea).
  Never hand-edit it.
- `RESEARCH_HANDOFF.md` is narrative only; `research.py decide --handoff` appends one decision block to it.
  `docs/deprecated/*_PROGRESS.md` are frozen history.

## Three roles

| Role | Does | Never does |
|---|---|---|
| **User** | Sets a research direction (`research.py direct "..."`). Rules on an idea only when it has `escalated` (`rule`, `decide --ruling`). | Review ideas one by one. |
| **Advisor** (gpt-6.1-sol, reasoning effort high) | `propose` ideas under the direction; `triage` each one (approve / revise / defer / reject, with reasoning); `analyse` results from the recorded numbers; `decide --review` checks the verdict. | Run code or see anything not recorded. |
| **Implementer** (Claude Code) | States its **own** triage position and verdict and argues when it disagrees; writes all code and tests; runs every protocol on the GPU box; records every number. | Use `rule` or `decide --ruling` (the user's). |

**An idea moves only when advisor and implementer state the same position.** A disagreement goes another
round with the implementer's argument in the advisor's prompt; after the round cap (5) the idea becomes
`escalated` and waits for the user.

## Idea lifecycle

`proposed → approved → running → analysed → decided`, plus `deferred` (right idea, not now; `triage R-xxx`
re-opens it), `rejected`, and `escalated` (round cap hit at `triage` or `verdict`).

## One pass of the loop

The loop runs until the user interrupts, the loop guard stops it, or something escalates.

1. **Orient.** `research.py state`. If the user gave a direction, `research.py direct "<their words>"`
   first; with no direction at all, ask and stop. Take the first "Needs attention" item: escalated →
   report and stop; analysed / running / approved / proposed → continue that idea.
2. **Propose** (only when nothing is open): `research.py propose --n 3`, then triage.
3. **Triage.** `research.py triage`. For each idea the implementer forms its own view first (new mechanism
   against decided ideas? runnable with existing scripts? proportionate cost? serves the direction?), then
   `research.py position R-xxx --position <approve|revise|defer|reject> --reasoning "..."`.
   - agreed approve → step 4; agreed reject / defer → closed;
   - agreed revise → write a JSON patch, `research.py revise R-xxx --patch <file> --note "..."`, triage again;
   - disagreed → `triage R-xxx` again; hold with a *new* argument or change position, never split the
     difference to end it. Round cap → escalate.
   - Loop guard: two `propose` calls in a row rejected in full (or returning zero ideas) and `propose`
     refuses: the direction is exhausted; ask the user for a new one.
4. **Implement.** `research.py show R-xxx`; write the code the protocol names (stay inside
   `protocol.code_changes`; note any excursion), unit tests, `gpu test <targets>`.
5. **Start and run.** `research.py start R-xxx --log-dir <dir>` (records the Git SHA and dirty flag;
   `--require-clean` for frozen-stage protocols), then the protocol's **exact** command on the box
   (`gpu start` once per session, `gpu run 'nohup ... &'` for long runs, `gpu get <log_dir> ./<log_dir>`).
   The box stays up across steps; `gpu stop` only at session end.
6. **Record.** Write `<log_dir>/research_result.json` with `metrics` (must contain the prediction's
   metric), `controls`, `artifacts` (local paths), `notes` (intervals, deviations, anything unmeasured);
   `research.py record R-xxx --results <file>`.
7. **Verdict, implementer first.** `research.py verdict R-xxx --verdict <supported|refuted|inconclusive>
   --confidence <0-1> --reasoning "..." --next "..."`, then `research.py analyse R-xxx` (the advisor's
   verdict).
8. **Decide.** `research.py decide R-xxx --summary "<one paragraph>" --review --handoff`. Agreement writes
   `D-nnn` and appends `RESEARCH_HANDOFF.md`. Disagreement: read the review, restate with `verdict`, run
   `decide` again; round cap → escalate.
9. **Close.** `research.py state`; commit `research/`, code and handoff together, message naming the idea
   id (the `git log` on `staging` shows the pattern: `R-102: supported (D-072): ...`). Back to step 2.

## Rules the tooling enforces or the skill requires

- `check_idea` refuses a protocol that names the **final-test split**, a **spent or doubly declared seed**,
  an **unmeasured metric** (the prediction must name a metric from `measurable_metrics()` or
  `RESEARCH_METRICS`, with direction and threshold), a missing script or config, or an unestimated cost.
  Fix the idea and say so in the triage round; never weaken the check.
- `start` requires the idea to be approved. Do not `propose` while an idea is approved, running or analysed.
- A frozen manifest reserves its seed for the idea that names it.
- Numbers go in `research_result.json`, not prose: the advisor sees only what is recorded.
- When reporting to the user, lead with what was decided and what is running; per-round arguments live in
  `research/triage.jsonl`.
- Stop for a destructive action, a missing requirement, or any spend no agreed triage covers.

## Command cheat sheet

```bash
python scripts/research.py direct "<direction>"      # user only; direct --show prints it
python scripts/research.py state | status | show R-xxx
python scripts/research.py propose --n 3 [--focus "..."] [--dry-run]
python scripts/research.py triage [R-xxx ...]
python scripts/research.py position R-xxx --position approve --reasoning "..."
python scripts/research.py revise R-xxx --patch patch.json --note "..."
python scripts/research.py consult R-xxx [--question "..."]   # advice only, changes no idea
python scripts/research.py start R-xxx --log-dir logs/...
python scripts/research.py record R-xxx --results <log_dir>/research_result.json
python scripts/research.py verdict R-xxx --verdict supported --confidence 0.8 --reasoning "..." --next "..."
python scripts/research.py analyse R-xxx
python scripts/research.py decide R-xxx --summary "..." --review --handoff
```

The advisor calls (`propose`, `triage`, `analyse`, `decide --review`, `consult`) need `OPENAI_API_KEY`. On
the Mac the ledger commands run under the conda env `lares-local`; protocols run on the GPU box.

## Where the loop stands now

Nothing open. The 2026-11-22 direction is exhausted (R-101 refuted, R-102 supported); the next pass starts at
step 1 with a new direction from the user. See `docs/CHATGPT_HANDOFF_2026-11-22.md`.
