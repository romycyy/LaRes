# Research direction V2: reproduce, stress-test and extend symbolic recovery

For `/research-cycle` in `romycyy/LaRes`, branch `staging`.
Starting evidence: session `2026-11-22`, R-103–R-127 / D-073–D-087, reviewed at commit `d7a7e97802dabd97d12f0f18888a762ce385bfae`.

The user approved the priorities in this document. Use it as the new direction when invoked. These are proposal requirements, not pre-approved experiment protocols. Refresh the live ledger and resolve newer results before implementing. Let the tooling allocate R-ids; Q1–Q4 below are local labels only.

## Direction

Continue Stage-3 Shadow Hand axial-rolling research from the frozen R-121 JOINT controller. Prioritize independent training reproducibility, then robustness of frozen controllers. Reproduce the successful residual-plus-recovery construction across independent training seeds before claiming a reliable learning algorithm. Evaluate longer horizons and predeclared physical/reset perturbations without fitting. Use archived failures to decide whether recovery hand-back deserves a state-dependent transition study. Then investigate whether LLM-guided construction can discover useful recovery-module structure against grammar search and the hand-designed template under matched budgets. Preserve the original six-second task and evaluation contract. Use JOINT as the experimental reference, preserve all existing assets, and leave default-incumbent installation separate. Run one protocol at a time through the existing advisor/implementer loop. Continue justified follow-ups within this direction; honor its escalation and exhaustion rules. No new global budget ceiling is imposed; each protocol must declare finite simulator, wall-time and model-call costs before agreement. No controller-generation calls for Q1–Q3. Q4 requires a separately agreed generation budget. Advisor calls for the research loop remain permitted throughout.

## What is established

R-124 evaluated six frozen controllers on the same 3,072 fresh cases:

| Controller | Sustained success | Drops |
|---|---:|---:|
| D-058 saved incumbent | 37.1% | 140 |
| D-072 qualified reference | 43.1% | 140 |
| R-106 learned residual | 54.8% | 96 |
| R-116 learned recovery | 64.4% | 127 |
| R-118 continued recovery | 77.2% | 136 |
| R-121 JOINT | 79.6% | 136 |

JOINT minus D-072: +0.365, 95% interval [0.344, 0.386], positive on all 24 manifests. These evaluations validate specific frozen controllers from one training lineage; they do not replace independent training runs.

Relevant assets:

- `config/pen_spin/r121_joint_finalist/`: main experimental reference.
- `config/pen_spin/r126_transplant_finalist/`: JOINT recovery module on the unchanged R-106 residual; useful secondary reference.
- `config/pen_spin/r106_success_ars_finalist/`, `r116_recovery_finalist/`, `r118_continue_finalist/`: frozen construction checkpoints.
- `config/pen_spin/symbolic_reward_reference/`: D-072 reference.
- `config/pen_spin/d058_fixed_stroke_incumbent/`: existing saved default, preserved.

R-126: JOINT 79.2%, transplant 78.5% on 4,096 fresh cases. JOINT minus transplant +0.007 [-0.006, 0.020]; no positive residual-coadaptation contribution is established, and equivalence is not established. Both execute a 32-coefficient residual plus an 11-parameter recovery module; “11 versus 43” describes which coordinates changed in the final adaptation, not complete runtime policy size.

## Questions already constrained by the record

- R-103 was an incomplete, low-budget PPO/ARS/draws pilot; it does not establish PPO inferiority at the successful ARS budget.
- R-108 shows the exact-success objective was not necessary for useful ARS learning. Shared placements and increased budget changed together relative to R-103; their separate causal effects remain unisolated.
- R-109 establishes no LLM structural advantage. Its outcome-informed selection amendment and overlapping grammar streams prevent strong algorithm-level claims.
- Extra additive memory (R-112), a second recovery attempt (R-118), startup offsets (R-122) and further JOINT continuation did not demonstrate useful improvement over their matched controls.
- Do not execute R-125 as written: its required early-latch failure prevalence was not met. Only about 1% of JOINT episodes matched that targeted failure condition.
- Do not execute R-127 as written: its required positive residual-coadaptation contribution was not established.
- R-117's support-termination mechanism may be reconsidered only from current JOINT traces; its old dependency on the R-116 endpoint is not automatically satisfied by later results.

## Q1 — Reproduce the learning procedure across independent training seeds

**Question:** does the successful recipe reliably construct a useful controller, rather than occasionally finding one good lineage?

Repeat the successful path from the same frozen D-072 reference: R-106 residual learning, R-116 single-attempt recovery learning, R-118 continuation, then R-121 JOINT adaptation. Copy the archived settings, parameter bounds, normalization procedures and inheritance semantics. Verify the historical schedules against saved configs before triage; do not re-run every failed side experiment or search for a better schedule during replication.

Use at least three independent full training lineages, preferably five when the prospective cost is justified. Freeze the number before viewing replication outcomes. Training placements and optimizer direction streams must be independent between lineages. Within-lineage antithetic comparisons retain the shared-placement instrument that worked. Repeating evaluation seeds for one trained controller is not training replication.

Freeze terminal checkpoints at the declared milestones. Keep the same stage schedule for every lineage; report failed or divergent runs, never replace them with extra seeds. Reconstruct and evaluate every endpoint, not just the best. Evaluate all lineages on the same unused confirmation manifests, with a separate D-072 baseline cell per manifest, so run differences can be assessed without changing test cases.

**Proposed registered prediction:** `paired_success_difference increase >= 0.30`, full learned recipe minus D-072, aggregated with equal weight across training lineages. Also report each run's absolute success and each construction milestone. An endpoint success around 0.75 is a useful planning target, not a criterion to tune against after the fact. Freeze support/refutation and uncertainty rules in triage; a positive interval does not by itself establish a minimum 0.30 gain.

**Required inference:** report all per-run results, mean/median/spread, and a two-level analysis accounting for training runs and evaluation manifests. Many cases do not create many independent training runs. With three runs, acknowledge limited precision at the algorithm level. For individual frozen-controller qualification, retain the paired success/drop criteria. Size confirmation prospectively from archived discordance and the actual contrast; do not reuse R-124/R-126 cases as new confirmation.

**Next:** if replication is unstable, diagnose training variance and data-layout dependence before adding structures. If it is consistent, preserve the recipe and move to Q2. A frozen policy can still enter Q2 if training replication is inconclusive, but report policy robustness separately from learning reliability.

## Q2 — Evaluate the frozen controllers beyond the original operating point

**Question:** does the controller maintain useful behavior over time and modest dynamics changes?

Evaluate frozen JOINT, the R-126 transplant and D-072. Include the qualified PPO expert as a competence control where its input/execution interface remains valid. Hash every controller before testing. No fitting, checkpoint selection, reward adjustment or controller revision during this study.

Predeclare a compact perturbation grid. Suggested starting levels, subject to simulator validation before outcomes:

| Factor | Proposed levels |
|---|---|
| Horizon | 6, 12, 30 seconds at the same 60 Hz |
| Pen mass | 0.8x, 1.0x, 1.2x |
| Friction | 0.8x, 1.0x, 1.2x |
| Radius and length, separately | 0.9x, 1.0x, 1.1x |
| Reset distribution | Fresh nominal cases plus a frozen, moderately expanded pose distribution |

Start with one-factor-at-a-time changes; declare a small combined-shift panel only if its cost is justified. Change geometry, collision shapes, inertia and relevant simulator metadata consistently. Record any observation fields whose meaning changes. Freeze case layouts and do not change batch size between paired controllers.

**Metrics:** original six-second sustained success, retention/survival, valid centered-rolling fraction, longest uninterrupted rolling interval, final-window success at each longer horizon, time-to-drop, spin-rate distribution and failure timing. Longer-horizon outcomes are new endpoints; do not silently replace or redefine the original benchmark or stretch the controller's time constants.

**Prediction for triage:** nominate one primary perturbation contrast and a defensible degradation margin from the proposed operating range; exact margins must be fixed before outcomes. Use the common metrics registry or add a genuinely computed metric through the normal implementation path. Never register an invented metric name that produces no data. Treat the rest of the grid as multiplicity-aware secondary analysis and report all cells.

**Next:** identify whether performance degrades because of capture, persistent slowness, recovery transitions or long-term retention. If it fails under shifts, propose a separate training-distribution/generalization study rather than retuning on this test panel.

## Q3 — Conditional recovery hand-back study

**First spend: archived-data diagnosis only.** Align JOINT/transplant traces to recovery onset, completion of re-capture blending, and the end of the temporary stroke boost. Separate triggered and untriggered cases. Label when intermittent rolling, off-centering and late drops begin. JOINT had 281 persistent-slow and 170 intermittent/near-miss failures among R-124's 3,072 cases; do not assume those failures share a cause.

Estimate how much failure mass is plausibly reachable by changing the support-ending rule, using observations available before that decision. Report transition-aligned associations as associations. Do not train an oracle gate using knowledge of an episode's eventual outcome.

**Proceed to simulation only if** the archive shows enough affected failures near support termination to justify the proposed effect and the advisor/implementer agree on a causal intervention. Otherwise close Q3 without a training sweep.

**Intervention:** compare a state-dependent hand-back with a matched timed hand-back from the same frozen parent. Keep onset, single-attempt semantics, re-capture waveform, action bounds and maximum support duration unchanged. Change only the rule ending post-recapture support. A candidate event rule may require centered rolling above a threshold for a consecutive duration, followed by a taper. Bound the support and force eventual return to the root controller.

Use the same optimizer, training placements and total budget. Match additional parameter counts where practical. Include the inherited controller and untrained transition variants to distinguish learning from scripted changes. If the event arm simply supplies more support, include a prospectively specified exposure/duration-matched timed control before crediting state dependence.

**Proposed registered prediction:** `paired_success_difference increase >= 0.03`, EVENT minus TIME, revised before spending if the archive shows insufficient reachable failure mass. Require positive fresh success intervals and the drop bound against both TIME and the frozen JOINT parent for useful adoption. A reduced failure category without a total-success gain is not success.

## Q4 — Test LLM construction of the useful recovery module

**Question:** can an LLM discover useful recovery structure with greater efficiency than grammar search, beyond a fixed hand-designed template?

Begin with a competent frozen root and the demonstrated recovery interface. Search recovery modules only; do not reopen cold whole-controller generation or merely rename a fixed module. Specify an explicit language of useful operations: observation-based triggers, bounded group-wise posture blending, ramps, finite support, state-dependent transitions and a small amount of resettable state. Declare which prior knowledge and example code each arm receives.

Compare three arms with one frozen effective inner learner:

1. The hand-designed module structure with parameter optimization.
2. Grammar/GP structural edits with that learner.
3. LLM-proposed structural edits with that learner.

Equalize total simulator transitions, candidate evaluations, inner-learning allocations, complexity/parameter caps and validation access. Let the fixed arm spend its matching allocation on prospectively specified restarts or continuation. Record LLM calls/tokens and wall time separately. Preserve parameters for unchanged components and measure executed authority of new components.

Use independent search blocks, preferably at least five when the cost is justified. Derive non-colliding seed streams from full tuples of block/arm/round/candidate/attempt; test uniqueness before any generation. Use the existing 16-worker pool for approved multi-block generation, while preserving one active protocol.

**Freeze search selection before any outcomes.** A declared best-validation ranking can select exploratory candidates within search; it need not require statistical promotion for every tiny mutation. Keep that search rule distinct from strict fresh-confirmation promotion. Apply the same rule to all arms. Do not change it mid-run because nothing moves. Any material outcome-informed amendment makes that run exploratory and requires a new prospective algorithm comparison for strong claims.

**Proposed registered prediction:** `paired_success_difference increase >= 0.05`, LLM search minus grammar search at a fixed total rollout budget, with independent search blocks as the primary replication units. Also compare both searches with fixed-template optimization and the frozen root. Report performance versus complexity and cost. If grammar succeeds without an LLM advantage, report that result; controller improvement alone does not prove an LLM contribution.

## Execution, records and safeguards against repeated mistakes

- Use the existing research-cycle workflow. Refresh `research/STATE.md`, the current handoff and relevant ledger records. Set `research.py direct` to the actual **Direction** text when the user invokes this file. Resume open work first.
- With nothing open, propose Q1 first. Supply the relevant section's actual contents through `propose --focus`; the CLI does not read a path passed as focus. Make the protocol self-contained because the advisor cannot see files merely read by the implementer.
- Ordinary ideas proceed through advisor/implementer agreement. Do not hand-edit approvals, generated state or historical decisions, bypass the loop guard, or start a second protocol while one is running. Respect escalation and exhaustion without inventing repeated confirmations.
- Every protocol freezes hypotheses, estimands, controls, seeds, costs, sample-size rules, failure handling and entry conditions before spending. Resolve reserved/spent seeds from the live ledger. Include failed smoke runs, forced resets, validation and confirmation in accounting; report simulator transitions and episode starts separately.
- Keep the original Stage-3 success predicate: no fall and centered axial rolling above 4 rad/s on all final 60 steps. Use the established scorer; a time-out trace has 359 recorded steps in the current implementation. Check actual action joint order rather than assuming config order.
- Use one fixed whole evaluation manifest per fresh Isaac process and preserve case layout. Validate checkpoint reconstruction, disabled-module identity, state reset and meaningful action authority. Training resets do not replace the independent evaluator.
- Keep training, search selection and prospective confirmation disjoint. Retain strict positive paired-success promotion, multiplicity handling and a one-sided 95% upper bound <= 0.05 on challenger-minus-parent drop rate. Report manifest uncertainty and run-level uncertainty where appropriate; do not declare equivalence from a nonsignificant difference.
- Record metrics, controls, per-run/per-case outcomes, full configuration, source/state hashes, optimizer history, runtime complexity, simulation costs and model calls by role in reconstructable artifacts. Preserve frozen controls and intermediate milestones.
- Use the repository's current compute backend and wrapper; do not assume a historical host is current. Run required tests and check their actual exit status before `start`; do not hide failures behind a pipe. Verify remote source/manifests against the intended code, including stale copies left by renames.
- When necessary fixes are found, log what changed and whether any outcome was seen; never quietly select a favorable rerun. Keep unavailable quantities explicit.
- After every agreed decision, regenerate state, append the handoff through tooling, commit and push code, ledger and small summaries together. Report decided results, running work, next dependency and the commit SHA. Preserve all existing controller assets; default-incumbent installation is not required to conduct this research.

## Prompt to start

```text
/research-cycle Read docs/NEXT_RESEARCH_DIRECTION_V2.md and adopt its Direction as my new research direction. Resume open work first; otherwise start with Q1, followed by Q2. Pass the relevant proposal text to the advisor through --focus. Run agreed protocols, use the stated gates for Q3 and Q4, record results, and commit and push each completed idea. Continue under the existing loop rules.
```

## Source pointers

- `docs/SESSION_2026-11-22_PEN_SPIN_SUMMARY.md`: session results and known limitations.
- `RESEARCH_HANDOFF.md`: decision narratives D-073–D-087.
- `research/ideas/R-103.json` through `R-127.json`: authoritative hypotheses, protocols, amendments and verdicts.
- `logs/pen_spin/r124_consolidated_ladder/`: consolidated frozen-policy confirmation.
- `logs/pen_spin/r126_block_transplant/`: module/residual attribution.
- `.claude/skills/research-cycle/SKILL.md`: execution workflow; this document supplies research content rather than replacing it.
