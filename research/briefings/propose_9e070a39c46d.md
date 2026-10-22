# Research briefing (propose) — generated 2026-10-22T04:29:45

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
- Spent seeds (never reuse): [0, 1, 2, 1000, 1001, 1002, 2000, 2100, 2200, 2201, 2202, 2203, 2204, 2300, 2301, 2302, 2303, 2304, 2400, 2401, 2402, 2403, 2404, 2410, 2411, 2412, 2413, 2414, 2420, 2421, 2422, 2423, 2424, 2500, 2501, 2502, 2610, 2611, 2620, 2621, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 2700, 2701, 2702, 2703, 2704, 2705, 2710, 2800, 2801, 2802, 2803, 2804, 2805, 2810, 2811, 2812, 2813, 3000, 3001]
- The final-test split is locked. No protocol may read or score it.

## Open ideas (already proposed; do not duplicate)
### R-003 [proposed] Rotating the development screen removes selection on fixed cases
Hypothesis: Drawing each generation's ten screening cases from a predeclared rotation of the development manifest, instead of always cases 0-9, raises promotion-set success because candidates stop being selected on ten fixed placements.
Prediction: promotion_success_rate increase by >= 0.1 vs the same config with the fixed screen (logs/baseline_gpt61_sol_high)
Falsified by: promotion success no higher than the fixed-screen run, or a rotation schedule that overlaps the promotion slice (refused by check_manifest_isolation)
Protocol: python scripts/run_full_evolution.py --config config/screen_rotation.yaml (episodes=130, llm_calls=12, seeds=[])
### R-009 [proposed] Test LLM bounds-only edits against matched random bounds
Hypothesis: LLM-chosen parameter bounds constrain imitation updates to successful control regimes while preserving the parent's executable structure and initialization.
Prediction: paired_success_difference increase by >= 0.05 vs Matched random-bound edits with identical fitting and rollout selection
Falsified by: An upper paired 95% confidence bound below +0.05 against random bounds refutes the prediction. No positive gain over the identically fitted unedited parent refutes the survive-refit claim; zero-spanning intervals are inconclusive.
Protocol: gpu run -- python scripts/stage_f.py --config config/stage_f/stage_f.yaml --research-protocol bounds-only --parents simple_damped_push,simple_reach_push_shared_gain --probe-seed 2820 --screen-seed 2821 --heldout-seed 2822 --fit-seeds 2823,2824 --range-seed 2825 --llm-model gpt-6.1-sol --reasoning-effort high --temperature 1 --output logs/research/bounds_only_s2820 (episodes=1080, llm_calls=4, seeds=[2820, 2821, 2822, 2823, 2824, 2825])

## Decided ideas (most recent first)
- D-005 on R-008: **inconclusive** — Replace imitation with direct search on weak evolved structures
  R-008 compared direct rollout search (original + 3 local + 4 global draws) against rollout-selected imitation (8 checkpoints to 1000 steps), each with 80 probe episodes, from the frozen original parameters of the two weak evolved Stage F structures (seed 2812, split 2813, probe s2810, confirmation s2811). Search minus imitation: +0.080 [+0.002, +0.158]; the predicted +0.10 is neither supported nor refuted. The average hides a complete split: on baseline61_gen0_cand2_a1 one global draw reached 0.52 while imitation never left step 0 (0.00); on baseline61_gen0_cand1_a0 imitation reached 0.36 at step 125 while no search draw succeeded. The two optimisers fail on different structures; the probe agreed with confirmation in both.
  Next step then proposed: Test pooling search candidates and imitation checkpoints under one rollout probe at the combined budget, over more weak structures.
- D-004 on R-007: **refuted** — Measure false rejection from a ten-case screen
  R-007 refitted the frozen geometry-scope child gen0_cand5_a0 and its parent simple_damped_push identically (fit seed 2802, probe s2804), scored both on a fresh 30-case screen (s2800) and a 100-case confirmation set (s2801+s2805), and replayed Stage F's acceptance rule on 1000 ten-case subsets. A 30-case screen selects a better controller than a 10-case screen by only +0.0044 [+0.0019, +0.0069]: the ten-case screen keeps the child 96% of the time, rejecting it only on tied success. The predicted +0.02 is refuted; screen size is not what blocks good edits. The refitted child beats its parent by +0.110 [+0.048, +0.172], 11/0/89, replicating R-005's +0.14: this one LLM geometry mutation is a real improvement.
  Next step then proposed: Consider admitting gen0_cand5_a0 to the fittable seed bank via the qualification rule; investigate why geometry-scope mutation works where correction-LLM repair does not.
- D-003 on R-006: **refuted** — Vision evidence from a local Qwen VLM makes correction-LLM repair beat numeric-only repair (development-only E10 pilot)
  R-006 added a local VLM evidence arm to correction-LLM repair: Qwen2.5-VL-7B-Instruct (pinned cc59489, NF4 language weights, fp16 compute, 7.55 GiB on the T4) analysed the parent's worst failed screen episodes (five rule-selected, upright frames each) and its validated, geometry-merged analysis was appended to the Stage F numeric packet. Nine independent cells (three frozen Stage F structures x bundles 2620-2628), held-out pool 2629, arms no_repair / llm_numeric / llm_numeric_vision at matched repair-episode caps. Vision minus numeric held-out success: +0.000 [+0.000, +0.000], 0/0/9; all arms 0.440. Every packet carried valid analyses (23 returned, 0 rejected, 3 stage conflicts), but all 18 LLM children scored at or below their refitted parent on the screen and were rejected, so neither evidence level changed any outcome. Predicted +0.05 refuted; localisation benefit untested because no edit survived refit. With Stage F and R-005, the correction LLM's structural edits, not its evidence, are the bottleneck.
  Next step then proposed: Do not add more evidence to correction-LLM repair; test whether any LLM edit can survive refit, or redirect effort to parameter search.
- D-002 on R-004: **refuted** — Replicate bundles that vary the data give independent fitting replicates
  R-004 fitted the five seed-bank structures under five data-varying bundles (own demo sample, trajectory split, probe manifest and fitting seed; seeds 2400-2404, 2410-2414) and under a matched minibatch-only control (fixed sample/split 2700, probe 2710, fitting seeds 2701-2705), replicate r of both arms scored on the same qualification manifest (pool seed 2420+r). Between-replicate SE of bank-mean success: 0.0075 vs 0.0056, difference +0.0019, paired bootstrap 95% [-0.0034, +0.0054]; the predicted +0.01 is refuted and the direction is inconclusive. Bank-mean success 0.896 vs 0.911. Minibatch-only qualification did not materially understate bank-level uncertainty. One exception: simple_reach_push_shared_gain rejected fitting (step 0 selected) under 2 of 5 data draws, so its fitted success is data-sensitive (sd 0.088 vs 0.037).
  Next step then proposed: Keep the seed bank as is; treat simple_reach_push_shared_gain as data-sensitive when reading single-run results.
- D-001 on R-005: **refuted** — A bounded LLM mutation beats its own fitted parent at matched fitting (absorbs R-001)
  R-005 refitted each of the six fittable-lane children of logs/two_lane_search and its seed-bank parent identically (fit seed 2611, same demos, deterministic_mse, budget 1000, checkpoints {0,500,1000} probe-selected on development 40-49) and scored all on 50 fresh cases (pool seed 2610). Child minus parent, averaged over pairs per placement: -0.020, t95 [-0.046, +0.006]; the predicted +0.05 is refuted, and the direction is inconclusive. Per pair: the geometry-scope mutation of simple_damped_push gained +0.14 (7/0/43), cand0 lost -0.16, two children tied their parents on every case. All 11 fits selected step 1000. The two-lane winner beats the zero-success gpt-6.1-sol baseline by +0.98 [+0.94, +1.02] (R-001), but it ties its own fitted parent, so that gain is the seed bank plus rollout-selected fitting, not the LLM mutation.
  Next step then proposed: Treat seed bank + rollout-selected fitting as the operative method; test the LLM mutation step only where it showed a signal (geometry scope), or redirect LLM budget to repair with richer evidence.

## Available protocol scripts (reuse before writing new ones)
scripts/ablate_repair.py, scripts/analyse_failures.py, scripts/audit_expert_actions.py, scripts/benchmark_fitting.py, scripts/build_structure_bank.py, scripts/compare_child_vs_parent.py, scripts/compare_evolution_runs.py, scripts/compare_generation_modes.py, scripts/compare_objectives.py, scripts/diagnose_repair.py, scripts/evaluate_on_fresh_manifest.py, scripts/final_report.py, scripts/freeze_method.py, scripts/lock_baseline.py, scripts/plot_training_dynamics.py, scripts/qualify_fittable_bank.py, scripts/recover_provenance.py, scripts/research.py, scripts/run_full_evolution.py, scripts/run_shadowhand_spin_stages.py, scripts/stage_e.py, scripts/stage_f.py, scripts/summarise_experiments.py, scripts/visualize_expert_policy.py
## Available configs
config/debug_smoke_two_generation.yaml, config/environment.yaml, config/fittable_bank/manifest_provenance.json, config/fittable_bank/push-v2_development_confirmation_s2300.yaml, config/fittable_bank/qualification.yaml, config/fittable_bank/qualification_v2.yaml, config/fittable_bank/seed_bank.json, config/fittable_bank/structure_roles.yaml, config/fittable_bank/variants_frozen.json, config/run_full_evolution.yaml, config/shadowhand_spin_stages.yaml, config/stage_e/manifest_provenance.json, config/stage_e/push-v2_development_confirmation_s2200.yaml, config/stage_e/selection.json, config/stage_e/stage_e.yaml, config/stage_f/manifest_provenance.json, config/stage_f/push-v2_development_heldout_s2500.yaml, config/stage_f/selection.json, config/stage_f/stage_f.yaml, config/stage_f_vision/manifest_provenance.json, config/stage_f_vision/probes/push-v2_probe_s2620.yaml, config/stage_f_vision/probes/push-v2_probe_s2621.yaml, config/stage_f_vision/probes/push-v2_probe_s2622.yaml, config/stage_f_vision/probes/push-v2_probe_s2623.yaml, config/stage_f_vision/probes/push-v2_probe_s2624.yaml, config/stage_f_vision/probes/push-v2_probe_s2625.yaml, config/stage_f_vision/probes/push-v2_probe_s2626.yaml, config/stage_f_vision/probes/push-v2_probe_s2627.yaml, config/stage_f_vision/probes/push-v2_probe_s2628.yaml, config/stage_f_vision/push-v2_development_heldout_s2629.yaml, config/stage_f_vision/stage_f_vision.yaml, config/two_lane_search.yaml
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


User focus: New positive leads: (1) D-004: the LLM geometry-scope mutation gen0_cand5_a0 of simple_damped_push beats its identically refitted parent by +0.11 [+0.05, +0.17] on 100 fresh cases (and +0.14 in R-005); it is the only LLM edit that has ever helped. (2) D-005: rollout search and rollout-selected imitation fail on different weak structures (+0.52 vs -0.36); a pooled selector over both candidate sets is untested. Correction-LLM repair (D-003) and generic bounded mutation (D-001) do not help. Propose development-only experiments that build on these: e.g. qualify gen0_cand5_a0 for the fittable seed bank, test whether geometry-scope mutations of other bank members also help (a mutation-scope effect), or a pooled search+imitation selector over more structures. Prefer cheap, decisive protocols; LLM calls allowed where the hypothesis needs them.