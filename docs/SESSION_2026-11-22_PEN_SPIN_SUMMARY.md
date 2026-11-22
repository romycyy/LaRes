# Session summary, 2026-11-22: reward-trained symbolic construction for Stage-3 pen rolling

Direction: `docs/NEXT_RESEARCH_DIRECTION.md` (P1 to P4). Ideas R-103 to R-127, decisions D-073 to D-087.
The loop stopped under its own rule: the advisor returned zero ideas on two consecutive propose calls after D-087.

## Headline

Sustained success on Stage-3 Shadow Hand pen rolling rose from 0.37 (the saved incumbent) to about 0.79 with a
controller that is still fully symbolic: the incumbent's stroke, a 32-coefficient residual, and an 11-parameter
single-decision re-capture module. Nothing was installed as the incumbent; that is the user's decision.

Consolidated confirmation (R-124, D-086): six frozen controllers, the same 24 fresh manifests, 3,072 cases each.

| Controller | Sustained success | Drops | Persistent-slow failures |
|---|---|---|---|
| D-058 saved incumbent | 1,141 (0.371) | 140 | 1,610 |
| D-072 reference (incumbent + joint-motion residual) | 1,325 (0.431) | 140 | 1,446 |
| R-106 residual, ARS on shared placements | 1,684 (0.548) | 96 | 1,159 |
| R-116 learned re-capture module | 1,978 (0.644) | 127 | 670 |
| R-118 continued module | 2,372 (0.772) | 136 | 346 |
| R-121 JOINT (module and residual co-adapted) | 2,446 (0.796) | 136 | 281 |

- JOINT minus D-072: +0.365, 95% interval [+0.344, +0.386], positive in 24 of 24 manifests.
- JOINT minus the saved D-058 incumbent: +0.425 [+0.404, +0.446], with the same number of drops.
- Every adjacent step is positive after Holm and passes the drop bound. The smallest is JOINT over R-118,
  +0.024 [+0.008, +0.040].

Block transplant (R-126, D-087), 4,096 fresh pairs: JOINT's edge over R-118 is in its module, not its retrained
residual. The 11-parameter transplant (R-106 residual + JOINT module) scores 0.785 against JOINT's 0.792
(difference +0.007 [-0.006, +0.020]).

## Decisions waiting on the user

1. **Incumbent.** `config/pen_spin/d058_fixed_stroke_incumbent` is unchanged. Candidates:
   - `config/pen_spin/r121_joint_finalist` (43 coordinates, highest, confirmed four times);
   - `config/pen_spin/r126_transplant_finalist` (11 parameters on the unchanged R-106 residual; derived by block
     swap, not trained; one confirmation);
   - `config/pen_spin/r118_continue_finalist` (11 parameters, original ranges, 0.76 to 0.77).
2. **A new research direction.** `python scripts/research.py direct "..."`; `propose` refuses until then.

## Where the results are stored

| What | Where |
|---|---|
| Decision blocks D-073 to D-087 (prediction, protocol, result, decision, next) | `RESEARCH_HANDOFF.md` |
| One idea per file, with triage rounds, protocol, results and verdicts | `research/ideas/R-103.json` to `R-127.json` |
| Generated loop state (never hand-edited) | `research/STATE.md` |
| Every proposal, triage round and advisor call | `research/proposals.jsonl`, `research/triage.jsonl`, `research/llm_calls.jsonl` |
| Per-study summaries (committed; small JSON only) | `logs/pen_spin/r1XX_<study>/summary.json`, `research_result.json`, `frozen_config.json` |
| Per-case cells and per-step traces (`*_cell.json`, `*.npz`) | On the GPU box only, under `~/LaRes/logs/pen_spin/r1XX_<study>/`; not copied back, not in git |
| Frozen controller assets | `config/pen_spin/r106_success_ars_finalist`, `r116_recovery_finalist`, `r118_continue_finalist`, `r121_joint_finalist`, `r126_transplant_finalist`, `symbolic_reward_reference` |
| Evaluation manifests | `config/manifests/pen_spin_stage3_*_s<seed>.yaml` |
| Study code | `lares/envs/isaac_pen_spin/` (one module per study; list below) |
| One-line result per decision | `CLAUDE.md` (pen-spin bullets), `docs/PIPELINE_STATE.md` (table rows) |

Log directories by idea: `r103_symbolic_reward`, `r104_alignment_audit`, `r106_exact_success_ars`,
`r108_objective_attribution`, `r109_structure_search`, `r110_slow_mode_audit`, `r112_memory_study`,
`r114_recovery_authority`, `r116_recovery_module`, `r118_recovery_retry`, `r120_tail_audit`, `r121_recovery_joint`,
`r122_early_phase`, `r124_consolidated_ladder`, `r126_block_transplant`.

Everything is on branch `staging`, pushed; the last commit is `8f9281a`. The GPU box is stopped; its disk keeps the
per-case archives.

## Results by question

### P1. Can a symbolic residual be learned from reward?

| Idea | Decision | Result |
|---|---|---|
| R-103 | D-073, inconclusive (two of three blocks) | PPO, ARS and independent draws at 460,800 transitions per run: PPO minus draws +0.004 [-0.059, +0.067]; no arm beat the reference. |
| R-104 | D-074, refuted (archive only) | Reward misalignment did not limit R-103. A 64-case selection set cannot rank policies a few points apart. |
| R-106 | D-075, supported | ARS on the exact success count, 512 shared placements per signed candidate in one 2,048-environment batch, 17.6 M transitions: +0.109 [+0.079, +0.138] over the reference on 1,536 fresh pairs. |
| R-108 | D-076, refuted | Exact-success minus dense-reward ARS on identical placements and budget: -0.021 [-0.046, +0.005]. The gain is the instrument (shared placements, budget), not the objective. |

### P2. Does structural evolution add value?

| Idea | Decision | Result |
|---|---|---|
| R-109 | D-077, inconclusive | Fixed structure vs grammar edits vs LLM edits from the R-106 finalist: LLM minus grammar +0.009; no structural arm beat continued fitting. The selection gate was amended mid-run (outcome-informed; recorded). |

### P3. Does compact memory help?

| Idea | Decision | Result |
|---|---|---|
| R-110 | D-078, refuted (archive only) | 41% of R-106 episodes fail in a slow-rolling mode. Reset placement predicts it weakly (0.608); the first 60 steps predict it at 0.81. |
| R-112 | D-079, refuted | A slow-memory gate against the same coefficients on a single-filter gate: -0.001 [-0.029, +0.027]. |

### Recovery from the slow mode (follow-ups motivated by D-078)

| Idea | Decision | Result |
|---|---|---|
| R-114 | D-080, supported (diagnostic) | A scripted re-capture converts +0.107 of stuck pairs but drops 9% of treated episodes. Authority exists; the shape and trigger are the problem. |
| R-116 | D-081, supported | The re-capture as an 11-parameter learned module: +0.082 [+0.068, +0.096] over the R-106 root (130 wins, 4 losses). |
| R-118 | D-082, refuted on its contrast | A second attempt adds +0.004 over matched continuation. Continuation itself adds +0.104 [+0.086, +0.121]. |
| R-120 | D-083, supported (archive only) | Persistent-slow failures of 1,536 fall 699 -> 568 -> 333 -> 194 down the ladder. |
| R-121 | D-084, inconclusive on +0.05 | Co-adapting residual and module: +0.030 [+0.008, +0.052] over widened module-only continuation. |
| R-122 | D-085, refuted | Start-up offsets for the first 60 steps: +0.013 [-0.009, +0.035] over matched continuation; neither improves on JOINT. JOINT has plateaued under ARS continuation. |
| R-124 | D-086, supported | The consolidated ladder (table above). |
| R-126 | D-087, inconclusive | Block transplant: module block +0.023 [+0.016, +0.031]; residual block +0.002 [-0.012, +0.016]. |

### P4. Anchored component construction

Not run. Its opening condition (residual learning ineffective despite verified authority) was not met.

## What worked and what did not

Worked:
- ARS with every signed candidate scored on the same 512 placements inside one batch, at about 17.6 M transitions.
- A learned single-decision re-capture module, and its continued training.
- Widened module ranges (threshold [1, 6], ramp-out [3, 80] steps), which is where the JOINT module's gain came from.

Did not:
- PPO, ARS or random draws at 460,800 transitions.
- Changing the training objective.
- Grammar or LLM structural terms on the residual.
- A slow-memory gate; a second re-capture attempt; start-up offsets.
- Further ARS continuation from JOINT.
- Retraining the residual alongside the module (no measurable contribution once the module is in place).

## What remains

JOINT's failures on 3,072 cases (R-124): 281 persistent slow (9.1%), 94 early drops, 91 intermittent, 79 near
miss, 42 late drops, 39 off-centre. Early drops are largely the same cases for every trained policy.

Deferred ideas with seeds held: R-105, R-107, R-111, R-113, R-115, R-117, R-119, R-123, R-125, R-127. Two failed
their own preconditions: R-125 (only 1.0% of JOINT cases are slow through steps 30 to 60 and finish persistent slow,
against a required 7.5%) and R-127 (needs a positive residual contribution, which D-087 did not show).

## Caveats for the reviewer

- All results are one task, one simulator, one checkpoint lineage. Each trained controller is one ARS trajectory.
- R-103 ran two of its three registered blocks (it hit its episode ceiling; amendment agreed with the advisor).
- R-109's selection rule was changed mid-run after seeing outcomes; its result is labelled accordingly.
- R-104 was executed once before its ledger `start` succeeded (rerun was byte-identical).
- R-121's first estimate of the JOINT step (+0.059) was high; two later confirmations put it at +0.024 and +0.030.
- R-126's 80% power is nominal (pilot-estimated discordance, Holm adjustment). Its upper bound (0.02003) sits on the
  registered +0.02, so "inconclusive" there leans against the hypothesis.
- R-126 process slips, neither affecting a result: `research.py start` ran before the box-side test had passed, and
  24 stale manifest copies on the server were deleted by hand, with their lines removed from the server's sync record.
- No controller-generation LLM calls were made except R-109's agreed budget. The locked held-out split was not opened.

## Things that will bite the next session

- Size the instrument to the effect: two pen-spin policies disagree on 0.16 to 0.41 of cases, so +0.05 at 80% power
  needs roughly 500 to 1,270 pairs, and +0.02 needs about 4,000.
- Per-update mean candidate success is a poor progress indicator; finalists improved on fresh cases while it was flat.
- A time-out episode has 359 recorded steps, not 360.
- Action `i` drives the sorted joint order, not `env.cfg.actuated_joint_names[i]`.
- An untracked file renamed locally after a sync leaves its old copy on the box.
- Never chain `gpu test ... | tail && research.py start`; the pipe hides the test's exit code.

## Code added this session

`lares/envs/isaac_pen_spin/`: `symbolic_reward.py`, `symbolic_reward_child.py`, `symbolic_reward_audit.py`,
`success_ars.py`, `success_ars_child.py`, `structure_terms.py`, `structure_search.py`, `structure_search_child.py`,
`memory_study.py`, `slow_mode_audit.py`, `recovery_authority.py`, `recovery_module.py`, `recovery_retry.py`,
`recovery_tail_audit.py`, `recovery_joint.py`, `recovery_early.py`, `ladder_confirmation.py`, `block_transplant.py`.
Each has a `--study` entry in `scripts/run_shadowhand_spin_stages.py` (R-109 runs through
`scripts/run_full_evolution.py --study structure_search`) and a unittest file under `tests/`.
