# Session summary: plateau-triggered trace repair on Shadow Hand pen spinning (2026-10-27 to 2026-10-28)

This document summarises one research-loop session. The authoritative record is still the ledger:
`research/ideas/R-0xx.json`, `research/decisions.jsonl`, the decision blocks in `RESEARCH_HANDOFF.md`, and each run's
`logs/<run>/research_result.json`. Numbers here are copied from those files; if they disagree, the ledger wins.

## 1. Direction and what was asked

The user's direction was to continue the single-step exploration pipeline with plateau-triggered trace repair, extended
to pen spinning on the Shadow Hand (Isaac Lab `Isaac-ShadowHandSpin-Stage{1,2,3}-Direct-v0`).
- **Expert source.** Try a scripted spinning controller first, then search online for a trained policy. If neither
  qualified, train a PPO expert on the box.
- **Housekeeping.** Commit the triage refactor and vendor the env.

The loop ran under the advisor-triage workflow (`CLAUDE.md`, Development Workflow). gpt-6.1-sol was the advisor, with
reasoning_effort high. Claude never used `rule` or `decide --ruling`, the final-test split was never opened, and Qwen/E10
were not run.

The session ran 14 ideas to a decision: R-038 to R-049 except R-043, plus R-051, R-053 and R-054, giving D-034 to D-047. R-050 was
rejected, and R-043, R-052 and R-055 were deferred. The user asked specifically for R-046 to R-054 (D-041 to D-047); the earlier
part (R-038 to R-045) is included because it set up everything the later ideas stand on.

## 2. Bottom line

1. **A qualified expert exists.**
   - The scripted feedback gait did not qualify (D-035).
   - A Stage-3 PPO expert (`logs/pen_spin/r040_ppo_expert/train/model_8239.pt`) qualifies on all three stages: 29/30,
     30/30 and 30/30 sustained (D-036).
2. **The expert is a closed-loop, bang-bang controller with a narrow basin.**
   - It needs live feedback: a replay of its own action tape scores 0/40 (D-037).
   - It needs both pen state and hand state. Clamping pen state to its reset values drops it to 0/40, and clamping hand
     state drops it to 1/40 (D-040).
   - A four-step (67 ms) delay of either group is fatal: 40/40 falls to 0/40 (D-042).
   - 99.98% of its executed action components are exactly ±1 (D-043).
3. **Expert demonstrations have not helped the LLM search on this task.** These all left the incumbent unchanged
   (contrast 0.000), or showed no benefit:
   - Keyframe trace repair (D-038).
   - Dense expert-label fitting (D-039).
   - Gait-preserving repair (D-041, inconclusive).
   - Expert takeover from learner states (D-045, refuted).
   - Self-trace versus expert-trace repair (D-047, refuted): no promotions in either arm. Expert-trace edits broke
     the grasp (0/320 screen successes); self-trace edits kept some of it (27/320).
4. **The real bottleneck is search, not demonstrations.**
   - Cold LLM candidates mostly keep the pen in hand but fail on sustained axial spin and centring (D-044).
   - Whether a block reaches any non-zero success depends on seed and candidate budget (D-041, D-044).
   - The one non-trivial learner found, the R-046 block-3 incumbent at 10/30, spins open loop: its spin drive is
     saturated on 94.6% of in-play steps (D-046).
5. **New validated infrastructure: closed-loop prefix branching.** Re-running the same 32-case batch in fresh processes
   is bit-identical up to a controller switch (D-045 gates). This replaces action-tape replay, which cannot reproduce a
   feedback controller.

## 3. Ideas and decisions, in order

| Idea | Decision | Outcome | Question | Key number |
|---|---|---|---|---|
| R-038 | D-034 | supported | Is any-hit episode success degenerate? | Yes; sustained rule adopted (time-out without a fall, predicate on all of the final 60 steps) |
| R-039 | D-035 | supported (gait does not qualify) | Can a scripted feedback gait be the expert? | Beats the open-loop oscillator by +0.411 but qualifies on no stage |
| R-040 | D-036 | supported / inconclusive on Stage 1 | Does a PPO expert qualify? | 29/30, 30/30, 30/30; vs gait +0.267 / +0.567 / +0.700 |
| R-041 | D-037 | supported | Does the expert need feedback? | Closed loop 40/40 vs action-tape replay 0/40 |
| R-042 | D-038 | refuted | Event vs uniform keyframe trace repair (Stage 3) | 0.000 [0, 0]; trigger 4/4, all twelve arms kept the null incumbent |
| R-045 | D-039 | refuted | Dense expert-label fitting vs rollout search | 0.000 [0, 0]; MSE down 2-4x, no probe success |
| R-044 | D-040 | supported | Which feedback does the expert need? | Pen-state clamp -0.975 [-1.03, -0.92]; hand clamp 1/40 |
| R-046 | D-041 | inconclusive | Gait-preserving repair (177-d scaffold) | Trigger fired in only 1/4 blocks; cold prefix promoted in block 3 (+0.333 [0.15, 0.51]) |
| R-047 | D-042 | supported | Expert latency budget | 4-step delay of pen or hand state: 40/40 → 0/40 |
| R-048 | D-043 | inconclusive (closed before arms) | Local feedback map as evidence | Expert bang-bang (99.98% at ±1); switch coverage 2.8% |
| R-049 | D-044 | refuted | Require separate retention and spin channels (cold, Stages 1/2) | 0.000 [0, 0] over 8 blocks; 0/32 candidates per arm had a screen success |
| R-050 | — | rejected in triage | Resolve same-placement tape divergence before branching traces | Superseded by closed-loop re-execution |
| R-051 | D-045 | refuted | Expert takeover from learner states | +0.031 [-0.18, +0.25] (8/32 vs 7/32); rescued 6/21, broke 5/7 |
| R-052 | — | deferred | Repair with verified learner-state corrections, not expert-only traces | — |
| R-053 | D-046 | inconclusive | Does the learner use signed axial-spin feedback? | Sign reversal -0.031 [-0.18, +0.11]; drive saturated 94.6% |
| R-054 | D-047 | refuted | Self-trace vs expert-trace repair from the learner | 0.000 [0, 0]; no promotions. Candidate screens: self 27/320, expert 0/320 |
| R-055 | — | deferred | Train the expert's recovery basin on learner-prefix states | — |

## 4. The last two thirds of the session (R-046 to R-054) in more detail

**R-046 → D-041, inconclusive: gait-preserving repair.**
- *Setup.* A 177-d scaffold env (`Isaac-ShadowHandSpin-Stage3-GaitScaffold-v0`; `gait_scaffold.py`, `gait_repair.py`)
  exposes the frozen gait's own action. Repairs can then be bounded residuals on top of the gait instead of whole-
  controller edits.
- *Result.* The plateau trigger fired in only 1/4 blocks, because the cold 157-d prefix had probe successes in 3/4
  blocks and promoted in block 3. R-042's "Stage 3 is a floor" is therefore seed-dependent.
- *Block 1 only (not a powered contrast).*
  - whole_edit dropped the pen: 0/30.
  - Bounded residual matched the unchanged gait, 0.467 vs 0.433.
  - The unchanged gait alone promoted, at 0.500.
- *Artifact kept.* The block-3 promoted candidate (gen0_cand0, 10/30) was frozen as the common learner for R-051 to
  R-054: `config/pen_spin/r046_block3_incumbent/`, with source and state pinned by sha256.

**R-047 → D-042, supported: expert latency.**
- Feeding pen state or hand state four steps late drops the expert from 40/40 to 0/40.
- Pen-delay falls come later (median step 116) than the R-044 clamps (49).
- Only one delay was tested, so the threshold itself is unmeasured.

**R-048 → D-043, inconclusive: local feedback map.**
- The plan was to show the LLM the expert's local feedback map (finite differences) instead of isolated labels.
- It was closed before any arm ran. The expert is bang-bang, so clamped finite differences are zero almost everywhere,
  and executed switches occur in only 2.8% of state-coordinate-action triples within one demonstration range.
- Added `IsaacPenSpinStream.query_actions`, which queries an actor at given observations without running an episode.

**R-049 → D-044, refuted: two-channel cold prompt.**
- *Setup.* The cold prompt required separate retention and spin-feedback components (`two_channel.py`), on Stages 1/2,
  over 8 blocks with 4 slots per arm. A neutral pose served as a scored null incumbent.
- *Result.* 0.000 [0, 0]; neither arm had a screen success.
- *Finding.* Candidates already keep the pen: 13% of screen episodes dropped, against 97% for the neutral pose. They
  fail as `slow_spin` or `off_centre`.
- *Finding.* At 4 candidates per block, the looser stages scored 0/8 blocks, while R-046's 24-candidate Stage-3 prefix
  had probe successes in 3/4. **Candidate budget**, not stage difficulty, is the untested driver of the base rate.

**R-050: rejected.** Action-tape replay as a branching method was rejected in triage, because the expert cannot be
replayed (D-037) and closed-loop re-execution was the agreed replacement.

**R-051 → D-045, refuted: expert takeover from learner states.**
- *Setup* (`learner_recovery.py`). Run the frozen learner. At a predeclared state-triggered step (window 60-120), hand
  control to the PPO expert; prefixes are reproduced by closed-loop re-execution.
- *Gates.* Gate 1 (bit-identical repeat on 32/32) and Gate 2 (28/28 takeover prefixes match) both passed.
- *Result.* +0.031 [-0.18, +0.25], 8/32 vs 7/32. Takeover rescued 6 of 21 failing states but broke 5 of the learner's
  7 successes.
- *Mechanism.* The expert drops the pen in 11 switched cases where the learner dropped none. Its labels there are
  99.85% saturated, so its basin does not cover the learner's grasp.
- **Infrastructure result:** closed-loop prefix re-execution is a qualified branching mechanism.

**R-052: deferred.**

**R-053 → D-046, inconclusive: does the learner use axial-spin feedback?**
- *Setup* (`axial_sign.py`). Reflect the observed angular velocity across the plane normal to the pen axis, which
  negates only the axial component. An identity-transform arm is the gate.
- *Result.* Identity gate bit-identical on 32/32. Reversal effect -0.031 [-0.18, +0.11], with 27/32 ties.
- *Mechanism.* The source does read the signed rate. The frozen operating point (`spin_target` 12.16, `speed_gain`
  0.81) saturates `0.25 + 0.75 tanh(speed_gain (spin_target − spin))` on 94.6% of in-play steps, so in axial spin the
  learner is effectively an open-loop stroke.
- *Lesson.* A static source audit shows code paths, not active feedback; check the operating point.

**R-054 → D-047, refuted: self-trace vs expert-trace repair.**
- *Setup* (`self_trace.py`). Both arms repair the frozen learner and see the same parent code, failure evidence, prompt
  and budget. They differ only in the successful-demonstration packet, 32 keyframe rows each from two episodes:
  - The learner's own successes (`self.json`).
  - The PPO expert's successes (`expert.json`).
- *Budget.* 8 blocks (seeds 4500-4507). Each block runs 4 trace-repair slots per arm, each with one linked correction.
  It screens on 10 cases, promotes the best challenger against the parent on cases 10-39 (strict paired rule), and
  scores the final incumbent on 40 fresh cases.
- *Primary.* Self minus expert fresh-incumbent success was **0.000 [0, 0]** (8 ties), against the predicted +0.15.
  Neither arm promoted anything, and both kept the parent (fresh mean 0.225).
- *Secondary, not predeclared:*
  - **Self-trace candidates preserved the parent's behaviour; expert-trace candidates destroyed it.**
  - Candidate screen successes were 27/320 for self against 0/320 for expert. The block-level mean screen difference was
    +0.084 [-0.008, +0.177], with 7 wins, 0 losses and 1 tie.
  - Pen drops on candidate screens were 200/320 for expert, 97/320 for self and 14/80 for the parent.
  - Three self candidates beat the parent's screen. The best challenger's promotion score was 0.233, against the
    parent's 0.333 in the same block.
- *Reading.* The expert's evidence pulls edits toward a grasp the learner does not have (consistent with D-045). The
  learner's own successes are safe evidence but not informative enough to improve on the parent with 4 candidates.
- *Accounting.* 66 LLM calls, 2.03 M tokens, 1760 Isaac episodes (estimates were 128 calls and 2080 episodes).

## 5. Findings worth carrying forward

- **The expert is a poor teacher for symbolic code at this scale.** It is closed loop, bang-bang, fast-feedback
  (fatal at 67 ms delay) and needs both observation groups. A symbolic controller cannot imitate its labels, and its
  basin does not include the learner's grasp states.
- **Imitation loss does not track success** (D-039), as in MetaWorld.
- **Cold LLM candidates already solve retention.** The gap is sustained axial spin near the centre (D-044).
- **The base rate is seed- and budget-dependent.** Stage 3 is not a fixed floor (D-041). R-054's frozen parent scored
  0.1-0.3 on its screens.
- **Learners can look closed loop in source but run open loop.** Check the operating point, not the code (D-046).
- **Isaac reproducibility rules:**
  - One rollout per process (D-034).
  - A case's outcome depends on its batch (D-035, D-037), so every protocol scores one manifest per fresh process.
  - Within those rules, re-execution is bit-identical (D-045).
- **Generated code keeps only the class.** Module-level constants in an LLM reply are dropped.

## 6. Infrastructure added (all under `lares/envs/isaac_pen_spin/` unless stated)

| Module | What it does | Entry point |
|---|---|---|
| vendored env, `cases.py`, `layout.py`, `runner.py` | Manifests from pool seeds, 157-d layout, batched runner writing placements into the simulator, the sustained episode rule | `run_shadowhand_spin_stages.py` |
| `scripted_expert.py` | Feedback gait (not qualified) | R-039 |
| `ppo_expert.py` | PPO expert controller (qualified) | R-040 |
| `feedback_ablation.py`, `feedback_lesions.py`, `feedback_delay.py` | Tape replay, clamp lesions, delays | `--study feedback_ablation / feedback_lesions / feedback_delay` |
| `pipeline_backend.py`, `pipeline_child.py` | Isaac as a pipeline "env" for `evaluate_manifest`; `query_actions` | `run_full_evolution.py` |
| `lares/repair/pen_spin_traces.py` | Keyframe trace packets (uniform / event) | `--study keyframe_repair` |
| `dense_fit.py` | Dense expert-label fitting, rollout-selected | `--study dense_feedback_fit` |
| `gait_scaffold.py`, `gait_repair.py` | 177-d scaffold env, bounded residual repair | `--study gait_preserving_repair` |
| `local_feedback.py`, `scripts/isaac/audit_expert_switches.py` | Finite-difference / switch audit of the expert | R-048 |
| `two_channel.py` | Two-channel cold prompt, scored neutral null incumbent | `--study two_channel_cold` |
| `learner_recovery.py` | Frozen learner, state-triggered expert takeover, closed-loop prefix gates | `run_shadowhand_spin_stages.py --study learner_recovery` |
| `axial_sign.py` | Axial angular-velocity reflection, source audit, identity gate | `--study learner_axial_sign` |
| `self_trace.py` | Self vs expert evidence packets, 4-slot trace repair of a frozen parent, promotion and fresh scoring | `run_full_evolution.py --study self_trace_source` |

**Frozen assets.**
- Learner: `config/pen_spin/r046_block3_incumbent/` (`policy_source.py`, `state.pt`).
- R-054 evidence packets: `config/pen_spin/r054_evidence/{self,expert}.json`.
- Manifests: `config/manifests/pen_spin_stage3_*`.
- Seeds: recorded in `config/experiment_seed_ledger.jsonl`.

Mocked gates are in `tests/test_pen_spin.py` and `tests/test_pen_spin_pipeline.py`. All 100 tests passed on the box
before R-054.

## 7. Open threads (not proposed; for the user's next direction)

- **Deferred ideas, re-triaged with the next `propose` / `triage`:**
  - R-043: anatomical action coordinates for single-step exploration.
  - R-052: verified learner-state corrections.
  - R-055: train the expert's recovery basin on learner-prefix states.
- **Candidate-budget base-rate study.** Same seeds, 4 vs 24 cold candidates per block, to test D-044's claim that budget
  rather than stage drives the base rate. Nothing else works until the base rate is non-zero and stable.
- **Operating-point checks.** Retune or bound `spin_target` / `speed_gain` so the learner's spin feedback is actually
  active (D-046). Then re-ask whether closed-loop learners repair better.
- **Expert latency curve.** Only one delay (4 steps) was measured (D-042).
- **Final-test and E10/Qwen remain locked** (Phase 7 blocked; see `CLAUDE.md`).

## 8. How to resume

```bash
conda run -n lares-local python scripts/research.py state     # ledger state, "Needs attention"
gpu start && gpu test tests.test_pen_spin tests.test_pen_spin_pipeline
# set a new direction if wanted:  python scripts/research.py direct "..."
# then the /research-cycle skill: propose -> triage -> implement -> run -> record -> verdict -> analyse -> decide
```

- **Where things run.** Isaac-only studies run under Isaac's Python via `scripts/run_shadowhand_spin_stages.py --study
  <name>`. LLM studies run in `.venv-metaworld` via `scripts/run_full_evolution.py --config
  config/pen_spin_plateau_repair.yaml --study <name>`.
- **Results.** Fetch with `gpu get logs/<dir> <local>`.
- **Box state.** The GPU box was stopped at the end of this session (`gpu stop`); `gpu start` brings it back. The
  public hostname changes on every start, and `gpu` looks it up.
