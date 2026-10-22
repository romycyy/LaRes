# Research briefing (analyse_R-036) — generated 2026-10-22T11:27:59

## Standing constraints (each with its source; do not propose anything that violates one)
- LLM model for every pipeline call is gpt-6.1-sol with reasoning_effort high and temperature 1 (the only temperature the model accepts).  _(source: CLAUDE.md, decided 2026-10-13)_
- Validation (imitation) loss correlates with development success at +0.19, the wrong sign; selecting on loss cannot work. Behavioural cloning is non-monotonic in gradient steps: success peaks near 1000 steps and is zero by 3000.  _(source: RESEARCH_HANDOFF.md, Phase 5)_
- Imitation fitting is disabled by default (bc_steps 0). If fitting is turned on it must be rollout-selected on a probe manifest disjoint from the screen; fixed-budget fitting is harmful.  _(source: Stage E, logs/stage_e/20260914_032500; config/run_full_evolution.yaml)_
- Promotion requires a strictly positive paired success difference whose paired t interval lower bound is above zero on the promotion set (development cases 10-39). Tied success never promotes.  _(source: lares/eval/promotion.py, decided 2026-09-14)_
- The saved incumbent scores 0.000 success on the development manifest, not 0.30.  _(source: RESEARCH_HANDOFF.md, 'The incumbent does not reproduce')_
- Only the five fittable_seed structures (simple_reach_push_sharp_gate, simple_damped_push, simple_reach_push, simple_reach_push_damped, simple_reach_push_shared_gain) may be fittable-lane parents; the standoff family is a non-fittable control or retired.  _(source: config/fittable_bank/structure_roles.yaml)_
- Stage F: no correction-LLM repair arm beat no_repair; search arms led, LLM arms were flat.  _(source: RESEARCH_HANDOFF.md 'Stage F: correction-LLM repair does not beat no repair (2026-10-13)')_
- The gpt-6.1-sol baseline rerun (nine unseeded zero-shot candidates) reached zero success; the gpt-5.6-sol run of the same config reached 0.333 on the promotion set.  _(source: logs/baseline_gpt61_sol_high, logs/baseline_gpt56_sol_high; RESEARCH_HANDOFF.md 2026-10-13)_
- The AC-6 repair ablation did not support the repair loop: optimizer-only 0.650 vs full loop 0.517 held-out success over six replicates.  _(source: RESEARCH_HANDOFF.md, 'The repair ablation (AC-6, E6, E9)')_
- The final-test split (config/manifests/push-v2_final_test.yaml) is locked until Phase 7, which needs the user's approval and a FreezeRecord. Qwen/E10 is also blocked.  _(source: lares/eval/freeze.py; CLAUDE.md)_
- Seeds 2400-2424 are held for config/fittable_bank/qualification_v2.yaml (predeclared, not run). New protocols pick seeds from 2600 upward.  _(source: config/experiment_seed_ledger.jsonl; CLAUDE.md)_
- Every protocol runs on the remote GPU box via `gpu run`, which bills by the hour; the implementer stops it afterwards. Live LLM or GPU spend needs the user's per-idea approval.  _(source: CLAUDE.md, 'Remote GPU server')_
- Spent seeds (never reuse): [0, 1, 2, 1000, 1001, 1002, 2000, 2100, 2200, 2201, 2202, 2203, 2204, 2300, 2301, 2302, 2303, 2304, 2400, 2401, 2402, 2403, 2404, 2410, 2411, 2412, 2413, 2414, 2420, 2421, 2422, 2423, 2424, 2500, 2501, 2502, 2600, 2601, 2602, 2603, 2604, 2605, 2606, 2607, 2610, 2611, 2612, 2613, 2614, 2615, 2616, 2617, 2618, 2619, 2620, 2621, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 2700, 2701, 2702, 2703, 2704, 2705, 2710, 2800, 2801, 2802, 2803, 2804, 2805, 2810, 2811, 2812, 2813, 2820, 2822, 2823, 2824, 2825, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2910, 2911, 2912, 2913, 2920, 2921, 2922, 2923, 2940, 3000, 3001, 3010, 3011, 3012, 3013, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3030, 3031, 3032, 3033, 3040, 3041, 3042, 3043, 3044, 3045, 3046, 3047, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3130, 3131, 3132, 3133, 3134, 3135, 3200, 3201, 3202, 3205, 3210, 3211, 3212, 3213, 3214, 3215, 3220, 3221, 3222, 3223, 3300, 3301, 3302, 3303, 3304, 3305, 3310, 3311, 3312, 3313, 3314, 3315, 3320, 3321, 3330, 3331, 3332, 3333, 3334, 3335, 3400, 3410, 3420, 3421, 3422, 3500, 3501, 3502, 3503, 3505, 3506, 3507, 3520, 3521, 3522, 3523, 3524, 3525, 3526, 3527, 3528, 3529, 3540, 3541, 3544, 3600, 3601, 3602, 3603, 3604, 3605, 3606, 3607, 3608, 3609, 3710, 3711, 3712, 3713, 3720, 3721, 3722, 3723, 3724, 3725, 3726, 3800, 3801, 3802, 3803, 3804, 3805, 3806, 3807, 3810, 3811, 3812, 3813, 3814, 3815, 3816, 3817, 3818, 3820, 3821, 3822, 3823, 3824, 3825]
- The final-test split is locked. No protocol may read or score it.

## Open ideas (already proposed; do not duplicate)
### R-036 [analysed] Plateau-triggered trace repair versus exploration restart
Hypothesis: Expert trajectories supply missing phase geometry at a plateau, enabling a repair that survives the pipeline's selection and promotion gates.
Prediction: paired_success_difference increase by >= 0.3 vs the budget-matched no-trace fresh-generation continuation from the same checkpoint; final incumbents compared on development cases 10-39
Falsified by: A paired upper bound below +0.30 refutes the effect-size prediction; an interval containing zero is inconclusive about a benefit. No trigger leaves the integration question inconclusive.
Protocol: python scripts/compare_generation_modes.py --assembly-study plateau-repair-pipeline --pool-seed 3820 --out-dir logs/research/two_lane_plateau_repair/block_0 (episodes=485, llm_calls=6, seeds=[3820, 3821, 3822, 3823, 3824, 3825])

## Decided ideas (most recent first)
- D-031 on R-035: **refuted** — Find another recoverable terminal-phase plateau
  R-035 scanned shelf-place, basketball and soccer for a second recoverable plateau. All three experts qualified (1.00, 1.00, 0.90), and one single-phase structure plus 24-candidate search already succeeded on every task's probe (22, 20 and 23 of 24 candidates), so no task is a plateau: refuted. Across all scans, assembly-v2 is the only MetaWorld task where single-phase generation plus search plateaus, and expert-trace editing solves it (D-027, D-028, D-030).
  Next step then proposed: Integrate trace-guided repair into the two-lane pipeline (R-036).
- D-030 on R-034: **supported** — Transport expert-trace edits across assembly parent structures
  R-034 applied expert-trace whole-controller editing (ten expert-only demonstrations, two traces, one validity correction) to three assembly-v2 parents that all score 0.00: the D-020 structure whose grasps the expert could not recover, and two fresh generations. All three trace edits scored 1.00 and all three summary-only edits 0.00 on 30 fresh cases: +1.000 per parent, supported. The trace effect does not need a recoverable parent; edits rewrite grasping as well as placement when shown the expert trajectory.
  Next step then proposed: Integrate trace-guided repair into the two-lane pipeline (R-036) and look for a second plateau task (R-035).
- D-029 on R-033: **inconclusive** — Transfer trace-guided structural repair to a gated hammer plateau
  R-033 tried to test trace-guided editing on a fresh hammer-v2 plateau parent, but the plateau gate failed: 24-candidate search of the fresh structure already produced probe successes, so no handoff or edit stage ran (inconclusive, 120 episodes, 1 call). Hammer is not a plateau for single-phase generation plus search; assembly-v2 remains the only plateau found.
  Next step then proposed: Scan shelf-place, basketball and soccer for another plateau, or test from a fresh assembly parent.
- D-028 on R-032: **supported** — Can expert-only placement traces replace learner handoffs?
  R-032 replaced D-027's learner handoff traces with expert-only demonstrations (10/10 successful; rows from when the nut first rose 3 cm). Edits of the recoverable assembly-v2 parent that saw two such traces scored 1.00 on 30 fresh cases at their declared values (no search) in both blocks; summary-only edits scored 0.00: +1.000 [1.0, 1.0], supported. Learner-state alignment is not needed: plain expert demonstrations carry the placement geometry the edit lacks.
  Next step then proposed: Test transfer to hammer (R-033) and from a fresh generated parent; integrate trace-conditioned editing as a repair step.
- D-027 on R-031: **supported** — Trace-conditioned assembly edits with one validity correction
  R-031 repeated R-029 with the pipeline's standard single validity correction and ten learner-to-expert handoff demonstrations (6/10 successful continuations). In both blocks the edit of the recoverable D-020 assembly-v2 parent that saw two successful expert placement traces scored 1.00 on 30 fresh promotion cases, while the summary-only edit scored 0.00: +1.000 [1.0, 1.0], 30/0/0, supported. The trace edits already succeed untrained (probe 1.0 at step 0). Three of four edits needed their one correction, which is why R-029 never tested the traces. The code reads observations through the schema; constants are centimetre-scale offsets. Narrow: one parent, one demonstration set, two blocks.
  Next step then proposed: Generalise trace-conditioned editing: another parent or fresh structure, another failing task, and as a repair step in the two-lane pipeline where an expert exists.
- D-026 on R-030: **inconclusive** — 3. Test insertion specificity with an expert-calibrated hard-task scan
  R-030 planned an expert-calibrated contrast of learner success on impact tasks (hammer, soccer) against insertion (peg-insert-side). MetaWorld's scripted expert scored 1.00 on hammer, 0.90 on soccer and only 0.63 on peg-insert-side on the frozen screen cases, failing the predeclared >= 0.90 gate, so no structure was generated: inconclusive, 90 episodes, 0 LLM calls.
  Next step then proposed: Replace the insertion control with a task whose expert is reliable, or scan hammer and soccer as further testbeds.
- D-025 on R-029: **refuted** — 2. Give assembly edits successful expert placement traces
  R-029 compared assembly-v2 placement edits of the recoverable D-020 parent given the handoff summary alone or the summary plus a successful expert continuation trace (only 1 of 5 handoffs succeeded), in two blocks under identical 24-candidate search. Both summary-only edits were valid and scored 0.00; both trace edits failed validation on gate_not_a_lever (unused post-lift gates) and, with no correction allowed, score 0.00. Trace minus summary-only: +0.000 [0, 0]; refuted for the procedure as specified, while the trace mechanism itself was never evaluated.
  Next step then proposed: Rerun with one validity correction and more successful continuations if trace evidence is still worth testing; otherwise move to the remaining hard tasks.
- D-024 on R-028: **refuted** — 1. Trade assembly parameter-search depth for structural breadth
  R-028 compared structural breadth (6 search candidates from each of four fresh gpt-6.1-sol assembly-v2 structures) with depth (24 candidates of one), at an equal 24-candidate budget. No candidate of any structure succeeded on the probe; both winners scored 0.00 on 30 promotion cases: +0.000 [0, 0], refuted. Three of the four new structures plateau at the same ~0.24 progress as the D-020 pair: the plain generator reproducibly writes assembly controllers that lift but cannot seat the nut.
  Next step then proposed: R-029: does trajectory evidence (successful expert placement traces) let an edit seat the nut?

## Available protocol scripts (reuse before writing new ones)
scripts/ablate_repair.py, scripts/analyse_failures.py, scripts/audit_expert_actions.py, scripts/benchmark_fitting.py, scripts/build_structure_bank.py, scripts/compare_child_vs_parent.py, scripts/compare_evolution_runs.py, scripts/compare_generation_modes.py, scripts/compare_objectives.py, scripts/diagnose_repair.py, scripts/evaluate_on_fresh_manifest.py, scripts/final_report.py, scripts/freeze_method.py, scripts/lock_baseline.py, scripts/plot_training_dynamics.py, scripts/qualify_fittable_bank.py, scripts/recover_provenance.py, scripts/research.py, scripts/run_full_evolution.py, scripts/run_shadowhand_spin_stages.py, scripts/stage_e.py, scripts/stage_f.py, scripts/summarise_experiments.py, scripts/visualize_expert_policy.py
## Available configs
config/debug_smoke_two_generation.yaml, config/environment.yaml, config/fittable_bank/manifest_provenance.json, config/fittable_bank/push-v2_development_confirmation_s2300.yaml, config/fittable_bank/qualification.yaml, config/fittable_bank/qualification_v2.yaml, config/fittable_bank/seed_bank.json, config/fittable_bank/structure_roles.yaml, config/fittable_bank/variants_frozen.json, config/run_full_evolution.yaml, config/shadowhand_spin_stages.yaml, config/stage_e/manifest_provenance.json, config/stage_e/push-v2_development_confirmation_s2200.yaml, config/stage_e/selection.json, config/stage_e/stage_e.yaml, config/stage_f/manifest_provenance.json, config/stage_f/push-v2_development_heldout_s2500.yaml, config/stage_f/selection.json, config/stage_f/stage_f.yaml, config/stage_f_vision/manifest_provenance.json, config/stage_f_vision/probes/push-v2_probe_s2620.yaml, config/stage_f_vision/probes/push-v2_probe_s2621.yaml, config/stage_f_vision/probes/push-v2_probe_s2622.yaml, config/stage_f_vision/probes/push-v2_probe_s2623.yaml, config/stage_f_vision/probes/push-v2_probe_s2624.yaml, config/stage_f_vision/probes/push-v2_probe_s2625.yaml, config/stage_f_vision/probes/push-v2_probe_s2626.yaml, config/stage_f_vision/probes/push-v2_probe_s2627.yaml, config/stage_f_vision/probes/push-v2_probe_s2628.yaml, config/stage_f_vision/push-v2_development_heldout_s2629.yaml, config/stage_f_vision/stage_f_vision.yaml, config/two_lane_search.yaml, config/two_lane_search_v2.yaml
## Metrics a prediction may name
action_saturation_by_axis, action_variation_by_axis, failure_label, failure_labels, final_goal_distance, fitting.bound_activity, fitting.checkpoint_step, fitting.gradient_norms, fitting.loss_by_phase, fitting.num_steps, fitting.objective, fitting.parameters_sha256, fitting.split, fitting.train_loss_by_axis, fitting.train_loss_end, fitting.train_loss_start, fitting.validation_loss_by_axis, fitting.validation_loss_end, gate_statistics, initial_goal_distance, lateral_drift, llm_calls, llm_total_tokens, mean_return, min_tcp_object_distance, num_episodes, object_displacement, paired_interval_lower, paired_success_difference, phase_occupancy, phase_transitions, promotion_success_rate, proximity_rate, return_se, screen_success_rate, signed_goal_progress, simulator_steps, slots_valid_within_one_correction, success_rate, success_se, timeout_rate, wall_time_seconds

## Where this stands

Phases 0 through 6 are implemented and measured. FR-4 and FR-11 are implemented.
Phase 7's machinery is built but its experiment has not run.

**Status corrected by `debug_spec.md` (2026-09-13).** A review found that the live evolutionary path
did not carry what the phase statuses claimed: generation-1 records had empty lineage, the live
fitter still trained the scale under the legacy objective, a selected checkpoint could be reported
with another checkpoint's diagnostics, and gradient steps were stored as `environment_steps`. The
corrected statuses are below. Stages A–C of the debug spec are implemented and pass a mocked
two-generation run; see `DEBUG_PROGRESS.md`.

| Phase | State |
|---|---|
| 0 measurement lock | complete, 7/7 AC-0 checks |
| 1 trusted interface | complete, 6/6 AC-1 items |
| 2 evaluation and reporting | 8/9 AC-2 items; the ninth is E10, which is blocked |
| 3 fitting comparison | complete, 6/6 AC-3 items |
| 4 search evidence and archive | **reopened**; lineage and bounded-edit semantics integrated, live smoke test passed; edit generations evaluated in Stage F |
| 5 objectives and data coverage | machinery complete; live integration done in the debug cycle; Stage E confirmation not run |
| 6 intervention-guided repair | **pilot**; the ablation does not test correction-LLM repair and its budgets were unequal |
| 7 freeze and final validation | **blocked** through Stage F of `debug_spec.md` |

Three results decide what is worth doing next, and all three run against the
plan's assumptions:

1. **Fitting the expert better does not build a better controller.** Validation
   loss correlates with development success at +0.19, the wrong sign, and
   development success collapses to zero well before the loss minimum.
2. **In a pilot, the repair loop did not beat spending the same budget on the
   optimizer.** Across the six structure/seed pairs `optimizer_only - full` was
   +0.35, +0.20, +0.05, +0.25, -0.05 and 0.00, mean +0.133. The interval first
   reported, [+0.008, +0.259], used 1.96 x SE; the small-sample t interval (5
   degrees of freedom) is about [-0.031, +0.298] and includes zero. This supports
   a directional optimizer advantage, not a significance claim, and the arms did
   not share preprocessing or a correction-LLM path (`debug_spec.md` D10).
   The controlled substitution changed the chosen component in two of six
   replicates and the outcome in one.
3. **The biggest lever on this task is the fitting schedule**, not the structure
   search or the repair. Recovering the pre-collapse checkpoint takes one
   structure from 0.000 to 1.000 held-out success.

Two things are blocked on a decision rather than on implementation:

- **Phase 7's experiment.** Five independent searches plus at least 100
  final-test episodes per finalist. The largest single spend in the plan.
- **E10.** It needs a Qwen checkpoint chosen *and* a correction-LLM repair path,
  because the current repair operators read no prompt and so cannot be affected
  by an evidence block. That same path is what the AC-6 result argues for:
  explanations generated rather than drawn from a fixed table.

The full list of open decisions is at the end of this file.


## Questions requiring a research decision

-2. **Raised by the bank study.** (a) Four of the five qualifying references are reach-push variants, so the
   seeded lane starts from essentially one design family plus `simple_damped_push`. Should the six seeded
   mutations be spread across references by a fixed allocation (for example, damped push guaranteed at least
   one slot) rather than ranked? (b) Should future qualification vary the data split as well as the minibatch
   seed, since the current seeds do not produce independent replicates? (c) The standoff family never succeeds;
   is there a reason to keep it in any bank?


-1. **Raised by Stage E.** (a) The frozen rule pooled both families, and the default it produced removes the
   route to the strongest policies yet found. Should fitting stay disabled for generated structures while
   remaining available, rollout-selected only, for structures that are fitting-dependent, and if so, how is
   that decided before seeing a structure's fitted score? A predeclared per-structure probe (fit only when a
   step-500 or step-1000 checkpoint beats step 0 on the probe cases) is one option; it is what
   rollout selection already does, at the cost of the probe episodes. (b) The generation prompts now reward
   untrained behaviour. That produced fitting-resistant structures capped near 0.40, while two hand-written
   six-parameter structures reach 0.9 after fitting. Should the next search ask for fittable structures, or
   seed the search with the bank references? (c) Evidence about fitting for fitting-dependent structures
   rests on three bank structures; more are needed before any decision specific to them.


0. **Answered on 2026-09-14** (details in `DEBUG_PROGRESS.md`, "Decisions of 2026-09-14"): fitting is an
   experimental treatment decided by Stage E's four conditions; promotion requires a paired interval
   above zero on an independent promotion manifest and never promotes on a tie; the next search runs one
   budget-matched initial generation with no edit generations; seed 2200 and fitting seeds 2200-2204 are
   reserved for Stage E. The original questions follow for the record.
   **New, from the debug-cycle runs.**
   (a) *Should behavioural cloning stay in the loop?* For the structures the current prompts produce, every
   promoted candidate selected step 0 and 2000 steps drive the incumbent to 0.000. Options: keep fitting
   behind the step-0 probe (current: correct, but it spends 30,000 gradient steps and 112,500 probe simulator
   steps per 15-candidate run for nothing), cut the budget and schedule, or replace imitation with direct
   search on rollouts. Stage E's predeclared arms (fixed 2000, rollout-selected, phase-balanced) will
   quantify this; its prediction for the fixed-2000 arm is now close to zero.
   (b) *Should promotion require the paired interval to exclude zero, or at least forbid promotion on tied
   success?* The current rule promoted twice on differences held-out data does not support.
   (c) *Is the edit loop worth its cost?* Generations 1–2 cost roughly two thirds of the run's LLM spend and
   did not improve held-out success. The feedback may need to show per-case differences between parent and
   child, since several repairs changed code without changing any outcome.
   (d) MT1 pool seed 2100 is spent on this comparison. Stage E needs another unused seed.

1. **Answered by the debug cycle.** Section 2.3's reliability target asks that 95% of candidates
   compile and pass validation after at most one repair. Generation now allows exactly one linked
   correction per slot (`max_repair_per_candidate=1`) and no outer retry loop, and the summary
   reports "slots valid within one correction" against the 95% target. What remains open is whether
   the target is reachable: in two Tier 3 runs gpt-4o-mini produced schema-valid structured ideas,
   but its one-slot implementation failed validation after its correction both times (an unknown
   observation field, then a policy that ignored `goal`). The previous loop hid this by retrying up
   to 50 calls. Measure the rate on the live smoke test before deciding whether prompts, the model,
   or the target should change.
2. FR-11 needs a small Qwen vision model. `transformers` is not installed, and the T4 is Turing, so
   fp16 rather than bf16. Which model and quantisation should be frozen? Everything around the model
   is now built and tested, so this decision plus an install is all that stands between here and a
   first real analysis. `QwenAnalyst` raises rather than defaulting, so nothing is frozen by accident.
3. The development split holds 50 placements and the search loop will see them repeatedly. Section 14
   warns about overfitting development through repeated feedback. Should development rotate on a
   predeclared schedule, and if so how?
4. **Imitation loss does not predict task success here, and following it destroys the controller.**
   Phase 5 answered the first half of this: the checkpoint is now chosen on development rollouts
   rather than on loss, which is the only rule that can find the peak. What remains open is whether
   behavioural cloning should keep its current role at all. A structure that scores 1.000 at 1000
   steps and 0.000 at 3000 is being *found* by the rollout probe, not by the fitting, so the fitting
   is closer to a random search over an unhelpfully parameterised path. Options: keep it as an
   initialiser and select on rollouts, which is what is implemented now; replace it with direct
   search on rollout success, which is cheap for six-parameter structures and would not need a
   reward; or keep both and treat the loss purely as a diagnostic.
5. The Phase 3 evidence rests on ten development cases per configuration and success rates near
   0.05. Should the comparison be re-run at 30 cases before the optimizer default is frozen? That
   costs roughly three times the 1120 seconds the current run took. The Phase 5 finding makes this
   more urgent, because every Phase 3 number was measured at the fixed 2000-step budget that sits
   past the peak, so the optimizer comparison may have been comparing points on the wrong side of it.
6. **The scripted expert is not a reliable recovery controller** (open decision 6, answered). About
   a third of its labels at learner-visited states fail to reduce even its own tracking error, worst
   in the descend phase. Should learner-state aggregation filter on the counterfactual check rather
   than only on the cheap validity test? That would raise label quality at the cost of one extra
   simulator step per accepted label.
7. `simple_reach_push` reaches 1.000 on all 50 development cases. It is hand-written, not
   LLM-generated, so it says nothing about the search. Should it become a reference target the
   search is measured against, and should the final-test evaluation of it wait until Phase 7 as the
   spec requires? The final-test manifest has not been touched.

8. **The protected-case regression tolerance (open decision 3) is answered at 0.05 paired success,
   but that number cannot be validated on a policy that scores 0.000.** Nothing can fail it, so the
   fallback progress check is what is actually doing the work today. Once a candidate scores above
   zero the success tolerance starts to bind, and 0.05 on ten screening cases means a single lost
   case is inside tolerance while two are outside. Should the tolerance be expressed in cases rather
   than in rate, so it does not change meaning with the screen size?
9. **The competing explanations are a fixed table keyed on the failure label.** They cover all four
   labels and always offer at least two accounts blaming different components, which is what makes a
   substitution informative. But the table is hand-written, so the search cannot propose an
   explanation the table does not contain. Should the correction LLM propose the competing
   explanations, with the table as a fallback? That is the natural place for FR-11's visual evidence
   to enter as well.
10. **The ablation's `full` arm tests only the two label-based explanations, not the gate one.**
   Holding the diagnostic cost fixed across structures made the arms easier to match, but it means a
   structure whose real fault is a gate that never opens cannot have that found. Worth a follow-up
   run that passes the available gates to `competing_explanations`.
