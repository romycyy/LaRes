# Implementation Progress

## Current phase and task

**Phase 0 — Lock measurement and reproduce baselines** (FR-1, AC-0): **complete**, 7/7 checks.
**Phase 1 — Trusted interface and validator** (FR-2, AC-1): **complete**, all six AC-1 items closed.
**Phase 2 — Structured evaluation and reporting** (FR-3, AC-2): **eight of nine items closed**. The
two VLM-discipline items are now enforced and tested (`lares/vision/`). The ninth needs a Qwen
checkpoint and an LLM repair path before E10 can decide anything.
**Phase 3 — Structure search separated from parameter fitting** (FR-6, AC-3): **complete**, all six
AC-3 items closed, with the comparison run over the full bank.
**Phase 4 — LLM proposal, feedback and archive** (FR-5, FR-10, AC-4): **complete**, all eight AC-4
items closed including the generation-mode comparison.
**Phase 5 — Objectives and data coverage** (FR-7, FR-8, AC-5): **complete**, all seven AC-5 items
closed, with E7 and E8 run on the simple family.
**Phase 6 — Intervention-guided repair** (FR-9, AC-6): **implemented and measured; four of five AC-6
items closed, and the fifth is answered in the negative.** The full repair loop does not beat the
optimizer-only ablation at matched cost. The machinery works, an accepted repair is on record, and
the experiment says the hypothesis does not hold at this budget. See the ablation table below.
**FR-4** is now fully enforced: the final-test split is locked and every look at it is ledgered.
**Phase 7 — Freeze and validate** (AC-7): machinery built (freeze record, drift check, aggregate
report). The five independent searches and the 100-episode final test are an expensive run awaiting
approval.

**FR-11 — VLM-assisted failure analysis**: **built and tested without a model.** The contract, the
fixed media rule, capture against the real simulator, the merge and conflict marking are all in
place. What is missing is a checkpoint and a correction-LLM repair path for E10 to compare.

Next: either accept AC-6's negative result and proceed to Phase 7, or build the correction-LLM repair
path, which the ablation and FR-11 now both point at. Generated rather than tabulated explanations is
the change most likely to move E6, and it is the same path E10 needs.

## Completed requirements and acceptance criteria

### FR-1 / AC-0 — measurement and baseline lock

| AC-0 item | Status | Where it is enforced |
|---|---|---|
| Package/commit, env id, reward version, wrappers, horizon, action scale, success rule, seeds recorded | done | `EnvironmentSpec` in every manifest; `version_lock()` in `scripts/lock_baseline.py` |
| Every compared candidate receives the identical ordered manifest | done | `EvolutionOrchestrator` takes one `dev_manifest`; `evaluate_manifest` replays it in order |
| Deterministic rerun produces identical per-episode outcomes | done, exactly | `lock_baseline` AC-0.3; `test_manifest_runner.test_expert_replay_is_bit_identical` |
| Collection, GIF and debug rollouts cannot change a later evaluation case | done | `lock_baseline` AC-0.4/0.5; `PipelineEnvs`; `test_intervening_work_does_not_shift_later_cases` |
| Development and final evaluation share no mutable env or RNG state | done | one env per stream, checked by `PipelineEnvs.__post_init__` |
| Train / development / final-test manifests non-overlapping with documented provenance | done | disjoint MT1 pool seeds; `assert_disjoint`; `test_splits_share_no_placement` |
| Expert and saved incumbent reproduced, discrepancies explained | done | see `RESEARCH_HANDOFF.md`; incumbent reproduces at 0.000 success, not the reported 0.30 |

Verification command:

```bash
python scripts/lock_baseline.py            # exits non-zero if any AC-0 check fails
```

### FR-2 / AC-1 — trusted policy interface

| AC-1 item | Status | Where it is enforced |
|---|---|---|
| Generated bodies contain no raw observation indices, only validated named accessors | done | AST check in `check_source`; `SymbolicPolicy.obs_field`; prompts rewritten |
| Missing schemas, wrong shapes, NaN/Inf, range mismatches, batch coupling, forbidden layers rejected before rollout | done | `validate_policy`, run by the subprocess harness and again in-process |
| Every parameter declared, registered, initialised, bounded, serialized | done | `_check_parameters`, `_check_serialization` |
| Policy and diagnostic state reset independently per episode | done | `reset_diagnostics` called by `PolicyActor.reset`, `_collect_trajectories`, `record_episode_gif` |
| `last_gates` telemetry detached, batch-safe, reset per episode, action-neutral | done | `record_gate` detaches; `_check_gate_telemetry` |
| Unit tests trigger every validator failure class | done | `tests/test_policy_interface.py`, one policy per defect |

The observation layout is machine-checked rather than asserted. `ObsSchema.check_against_env`
compares `tcp`, `obj`, `obj_quat` and `goal` against the simulator's own accessors, and a
deliberately wrong slice is caught in test. Fields the simulator exposes no accessor for are
reported by `unverifiable_names()` rather than silently passed.

### FR-3 / AC-2 — structured evaluation and reporting

| AC-2 item | Status | Where it is enforced |
|---|---|---|
| Every valid candidate produces an `EvaluationReport` and a per-episode table | done | orchestrator builds three or four reports per candidate into `gen_<N>/reports/` |
| Success primary; return and distance/progress secondary | done | `RolloutSection`; promotion decides on success and uses return only to break a tie |
| Phase and gate occupancy appear when exposed; unavailable fields explicit | done | `unavailable_fields`; gate fields are `None`, never zero-filled |
| Each exposed gate reports mean, min, max, occupancy above 0.5, first crossing, transitions | done | `GateAggregator`; `aggregate_gate_statistics` keeps "never crossed" distinct from "crossed late" |
| Comparisons report paired success and return differences with intervals | done | `compare_to_incumbent` over `paired_difference` |
| Zero-shot, intermediate, fitted, deterministic, sampled results not conflated | done | `checkpoint` is validated against a closed set; action mode is on every report |
| Qwen outputs schema-valid with frame citations | deferred | needs a model decision and `transformers` |
| Qwen evidence never used alone to promote or reject | deferred | structural; nothing consumes visual evidence yet |
| Qwen enabled only if E10 justifies it | deferred | experiment not yet runnable |

The diagnostics earn their place immediately. On the development manifest the saved incumbent
splits into two distinct failures rather than one flat zero:

| Measure | Incumbent | Scripted expert |
|---|---:|---:|
| Never reached the object | 17 of 30 | 0 of 30 |
| Reached it, pushed away from the goal | 13 of 30 | 0 of 30 |
| Mean signed goal progress (m) | -0.051 | +0.193 |
| Mean closest approach (m) | 0.054 | 0.033 |
| Action saturation, every axis | 0.00 | 0.07 on two axes |

The incumbent moves the object about 9 cm on average and, on balance, moves it *further from the
goal*. It is not railed against its action limits, so the problem is the control law rather than the
output scale.

### FR-6 / AC-3 — numerical fitting

| AC-3 item | Status | Where it is enforced |
|---|---|---|
| At least four fitting configurations compared on the same frozen structures and data | done | 18 structures under `lares/fitting/structures/`, four methods, one shared `FittingData` |
| The current one-shot Adam path reproduced exactly as a named baseline | done | `fit_adam_baseline`: one group, learning rate 1e-3, no scheduler, no restarts, no convergence check |
| Objective-evaluation budgets matched; wall time and convergence reported | done | every method reports objective evaluations, transitions and seconds; multistart spends an uneven budget exactly |
| Train and validation split by complete trajectory | done | `prepare_data` over `split_buffer_by_episode` |
| Gradient, sensitivity and bound-activity evidence per structure | done | `gradient_sensitivity`, `action_sensitivity`, `bound_activity` on every row |
| Default optimizer and checkpoint rule chosen on predeclared development metrics | done | `SELECTION_RULE` is a module constant, fixed before the comparison ran; development split only |

Full run: 18 structures, four methods, 2000 objective evaluations each, ten development cases per
fitted result under both checkpoint rules. 1120 seconds total.

| Method | Validation loss | Development success | Parameters at a bound | Seconds |
|---|---:|---:|---:|---:|
| Adam baseline | 0.0594 | 0.044 | 0.234 | 8.5 |
| Adam, range-scaled with cosine decay | 0.0512 | 0.044 | 0.258 | 9.6 |
| Multistart Adam, four restarts | 0.0824 | 0.072 | 0.174 | 8.6 |
| Differential evolution | 0.0730 | 0.044 | 0.000 | 4.1 |

Paired by structure against the reproduced baseline, only one difference has an interval excluding
zero in each direction, and they point opposite ways:

| Comparison | Success difference | 95% interval |
|---|---:|---|
| Multistart minus baseline | +0.028 | [+0.001, +0.054] |
| Multistart minus baseline, validation loss | +0.023 | [+0.014, +0.033] |
| Range-scaled minus baseline, validation loss | -0.008 | [-0.011, -0.005] |

The method that fits the expert *worst* is the one with the better controller, and the method that
fits *best* gains nothing. Across all 144 configurations the correlation between validation loss and
development success is positive, which is the wrong sign, at Pearson 0.19 with p equal to 0.023.
See `RESEARCH_HANDOFF.md` for what follows from that.

### FR-5 / FR-10 / AC-4 — LLM search loop and archive

| AC-4 item | Status | Where it is enforced |
|---|---|---|
| Every proposal satisfies the idea and mutation schemas | done | `PolicyIdea.validate`, `MutationProposal.validate` |
| Every non-exploratory edit states a hypothesis, a measurable prediction and protected behaviours | done | `MutationProposal.validate` refuses a prediction naming a metric no report produces |
| All population diagnostics and errors retained; two informative parents comparable | done | `build_feedback`; `choose_informative_parents` picks the leader plus the most behaviourally distant candidate |
| At least 95% of candidates reach rollout validation after at most one repair | done | `max_repair_per_candidate` is now 1; `summarise_experiments.py` reports the rate and flags any candidate that exceeded it |
| The archive retains robust, simple and behaviourally distinct candidates | done | `Archive` with three slots, diversity measured on the failure profile |
| Every attempted candidate has one schema-valid record, including invalid ones | done | `_record_generation_failures` and `_record_candidate` write to `<log_dir>/experiments/` |
| Every evaluation traceable to code hash, checkpoint, manifest, action mode, prompt config and lineage | done | `ExperimentRecord.validate` refuses a record missing any of them |
| Expert-assisted results labelled and ineligible for ranking | done | `ExperimentRecord.eligible_for_ranking`, `Archive.rankable` |
| Per-idea versus batched generation compared under a matched budget | done | `scripts/compare_generation_modes.py`; both returned 12 of 12 valid, per-idea cost 0.73x the tokens per usable candidate, so the default changed to `per_idea` |

The feedback the generator sees changed from two floats to a structured block: a ranked population
table, full code for two parents chosen to differ behaviourally, per-episode geometry and failure
labels, gate occupancy, dead parameters, the archive, and the errors from candidates that never
compiled. It carries an explicit caveat that fitting loss does not predict success on this task.

### FR-7 / FR-8 / AC-5 — objectives and data coverage

| AC-5 item | Status | Where it is enforced |
|---|---|---|
| Primary baseline is `MSE(tanh(mean), a)` with a fixed positive scale that is not optimised | done | `deterministic_mse`; `freeze_scale_parameters` finds the scale parameters by measuring which ones move the scale but not the action |
| The stale NLL docstring is corrected | done | rewritten in Phase 2 |
| Endpoint frequency and expert clipping semantics measured before choosing a likelihood | done | `scripts/audit_expert_actions.py` |
| Each distributional objective matches its executed sampling contract | done | every objective carries an execution contract; `check_execution_matches` refuses a mismatch |
| Deterministic MSE remains as a matched baseline | done | it is one of the five compared objectives |
| Learner-state data uses training placements only, with costs reported | done | `aggregate_learner_states` takes training cases; `AggregationStats` counts every query and step |
| Uniform against phase-balanced, and fixed-buffer against learner-state, at matched budgets | done | `scripts/compare_objectives.py` |

**The expert audit settles the likelihood question.** MetaWorld's scripted expert is an unbounded
proportional controller with gain 10 that performs no clipping of its own. It warns and returns the
raw value; the environment clips, and Stage 1 clips before storing. So the endpoint mass is produced
by *our* clip of an unbounded signal, not by anything the expert does.

| Axis | Raw range | Fraction censored |
|---|---|---:|
| dx | -1.70 to 1.62 | 0.017 |
| dy | -0.09 to 2.67 | 0.112 |
| dz | -1.69 to 0.25 | 0.073 |
| gripper | 0.00 to 0.60 | 0.000 |

18.8% of transitions are censored on at least one axis, and the largest raw magnitude is only three
times the action limit. Two consequences. A tanh-Gaussian likelihood is not the matching model,
since the generator is not a tanh squash and `atanh(±1)` is undefined on the censored fraction. A
censored Gaussian is the matching model, and it requires clipped execution rather than the tanh
sampler, so it is implemented with that contract attached.

**The expert also has three explicit control branches**, so the phase label is ground truth rather
than something inferred from a learned gate. The labels are checked against the branch the expert
actually takes, and agree on 200 of 200 random states.

| Phase | Share of expert steps |
|---|---:|
| hover | 0.085 |
| descend | 0.159 |
| push | 0.757 |

**The headline result: minimising the imitation loss destroys the controller.** Development success
is non-monotonic in the number of gradient steps while validation loss falls monotonically. Two
structures, three seeds, all 50 development cases.

| Steps | `simple_reach_push` | `simple_damped_push` | Validation loss |
|---:|---:|---:|---:|
| 250 | 0.000 | 0.553 | 0.268 / 0.200 |
| 750 | 0.820 | 0.633 | 0.135 / 0.091 |
| **1000** | **1.000** | **0.927** | 0.095 / 0.072 |
| 1500 | 0.700 | 0.613 | 0.071 / 0.060 |
| 2000 | 0.053 | 0.520 | 0.066 / 0.056 |
| 3000 | 0.000 | 0.000 | 0.060 / 0.043 |

At the loss minimum both score zero. The pipeline's fixed 2000-step budget sits well past the peak.
`behavioral_cloning` now takes a rollout probe and keeps the best-scoring parameters, and the
orchestrator supplies one that scores the screening cases four times during fitting. Selecting on
loss cannot work, because loss improves through the collapse.

`simple_reach_push` at 1000 steps reaches 1.000 on all 50 development cases across three seeds, mean
return 1138 against the scripted expert's 1063, paired difference against the expert +0.000 with a
95% interval of [0.000, 0.000]. That is a development result on a hand-written structure; the
final-test manifest has not been touched.

### FR-9 / AC-6 — intervention-guided repair

| AC-6 item | Status | Where it is enforced |
|---|---|---|
| Each targeted failure cluster has at least two documented competing hypotheses | done | `competing_explanations` returns at least two accounts per supported label, each blaming a different component; a cluster with fewer is recorded with a note and never repaired |
| A controlled substitution isolates the suspected dimension, phase, gate or geometry term on matched cases | done | four substitutions in `interventions.py`, run through the same manifest runner as the control, so the pair shares its ordered cases |
| An accepted repair moves the predicted focused metric in the predicted direction | done | `evaluate_repair` refuses on any other outcome; the direction is declared by the explanation before the substitution runs |
| An accepted repair passes a predeclared protected-case regression threshold | done | `PROTECTED_REGRESSION_TOLERANCE` and `PROTECTED_PROGRESS_TOLERANCE` are module constants, not call-site arguments |
| The full loop beats evaluation-only, diagnostics-only and optimizer-only at matched cost | **not met** | `scripts/ablate_repair.py`; the run and the numbers are below |

**A gate had to become a lever before it could be tested.** `record_gate` detached its argument and
returned nothing, so a policy could report a gate it did not use, and forcing such a gate would
change nothing. A null result there reads as evidence that the gate is not at fault, which is the
worst possible failure for a diagnostic. `record_gate` now returns the forced value when an override
is set, the two bank structures with gates use the returned value, and `gate_override_is_effective`
checks the lever bites before an explanation is built on it.

**The first diagnosis discriminated between two accounts.** `simple_standoff`, fitted with
deterministic MSE at a 2000-evaluation budget, scores 0.000 on the ten screening cases and fails
eight of them as `never_reached_object`, closest approach 0.063 m against a 0.05 m success radius.

| Explanation | Substitution | `min_tcp_object_distance` | Verdict |
|---|---|---|---|
| `approach_control` | expert drives the first 40 steps | 0.0633 -> 0.0285 | supported |
| `vertical_axis` | expert drives `dz` | 0.0633 -> 0.0643 | rejected |

The planar approach is the fault; the vertical axis is not. Both were run on the same ten cases, and
both are stored.

**Repairing the expert's own target does not work; repairing the metric does.** The first operator
refits the in-scope parameters toward the expert. It moved `gain` and `height_offset` and left the
closest approach where it was, which is the correct outcome for the wrong tool: the current values
are exactly what fitting to the expert produces, so more of the same cannot change them. The second
operator searches the same bounded parameter set against the focused metric itself.

The full cycle on `simple_standoff`, from the study through acceptance, is the Phase 6 artifact:

| Step | Result |
|---|---|
| Parameters in scope, found by measurement | `gain`, `height_offset`, `standoff` |
| Parameters the edit actually moved | `gain` alone, inside its declared range |
| Focused metric, the eight cluster cases | 0.0633 -> 0.0603, the predicted direction |
| Protected cases, no case had succeeded | mean signed goal progress rose 0.039, from -0.092 to -0.053 |
| Screening success, ten cases | 0.000 -> 0.100 |
| Held-out success, twenty cases scored after the decision | 0.000 -> 0.050 |
| Decision | accepted |

The repair cost 200 rollout episodes of search. Of the 290 episodes the whole run spent, 20 were
expert-assisted and are excluded from ranking; the other 270 ran the policy unaided.

**The ablation does not support AC-6's last claim.** Five arms, three structures, two seeds, 200
development-screen episodes each, scored on twenty held-out cases no arm saw. Every arm starts from
the identical fitted policy.

| Arm | Held-out success | Paired against `full` | 95% interval | w/l/t |
|---|---:|---:|---|---:|
| `optimizer_only` | 0.650 | +0.133 | [+0.008, +0.259] | 4/1/1 |
| `full` | 0.517 | | | |
| `diagnostics_only` | 0.500 | -0.017 | [-0.095, +0.062] | 1/1/4 |
| `evaluation_only` | 0.433 | -0.083 | [-0.173, +0.007] | 0/3/3 |
| `global_search` | 0.367 | -0.150 | [-0.312, +0.012] | 1/4/1 |

Read as the spec requires, on paired differences rather than on five separately computed means:

1. **Optimizer-only wins, and its interval excludes zero.** The claim AC-6 asks to be true is false at
   this budget. It also spends 2000 Adam steps the search arms do not, which favours it further.
2. **The controlled substitution buys almost nothing over untested diagnostics.** It changed the
   chosen component in two of six replicates and the held-out outcome in one, for 10% of the budget.
   That is E6's answer and it is negative here.
3. **Localisation does appear to help.** Naming a component beats searching every parameter at the
   same cost, by 0.150, and beats a component chosen without evidence by 0.083. Both intervals
   include zero across six runs, so this is a direction, not a result.

**Why optimizer-only wins is the Phase 5 finding, not a repair failure.** Its rollout probe finds the
checkpoint before the behavioural-cloning collapse. On `simple_reach_push` that takes held-out
success from 0.000 to 1.000 in both seeds, and on `simple_damped_push` from 0.450 to 0.950. Every arm
starts from the 2000-step policy, which sits past the peak, so the largest gain available on this
task is recovering a checkpoint rather than repairing a structure. On `simple_standoff`, where no
checkpoint helps, only the two diagnostic arms move at all.

**The success-regression check is vacuous on a policy that never succeeds.** With zero successes no
repair can fail a success-drop threshold. `accept_repair` detects this, names the cases outside the
cluster as the protected set, and holds them to a mean signed-goal-progress check instead.

### FR-4 — fair candidate selection

Four of FR-4's five requirements were already enforced by `lares/eval/promotion.py`: one ordered
manifest for every candidate, thresholds declared before a generation runs, paired
candidate-versus-incumbent differences, and no inference from two separately computed intervals.

The fifth, **winner selection and final estimation use different data**, was enforced nowhere. Any
call could evaluate a candidate on `push-v2_final_test.yaml`, read the number and continue; the
manifest would then be development data under another name.

| Mechanism | What it does |
|---|---|
| `evaluate_manifest` guard | refuses any manifest whose split is `final_test` outside a session |
| `FreezeRecord` | digests models, prompts, source, manifests, optimizer, selection rule and budgets; refuses to save over itself |
| `verify()` | recomputes the digests and names the file that drifted, rather than returning a boolean |
| `final_test_session` | the only way to unlock the split; refuses to nest; closes on an exception |
| `final_test_ledger.jsonl` | one line per look at the held-out data, surviving the session |
| re-score refusal | the same code hash cannot be scored twice unless the caller says so, and the admission is recorded |

The lock is at `evaluate_manifest` because that is the single choke point every actor already goes
through. `scripts/lock_baseline.py` still passes 7/7 with it in place, which is the check that it
does not block ordinary work.

### FR-11 / AC-2 — VLM-assisted failure analysis

The three AC-2 items left open since Phase 2 were deferred on a model decision. Two of them are
about discipline rather than the model, and those are now enforced and tested. The third needs a
checkpoint and cannot be closed here.

| AC-2 item | Status |
|---|---|
| VLM outputs are schema-valid and every claimed visual event cites a frame, clip or timestep | enforced; `VisualFailureAnalysis.validate` refuses a claim citing a frame that was never shown or a timestep outside its span |
| VLM evidence is never used alone, and final-test media never returns to the search loop | enforced; an analysis must recommend a measurable check, the merge runs it, and media capture refuses the final-test split outright |
| The VLM is enabled in the final method only if E10 justifies it | **blocked**; E10 needs a checkpoint and a correction-LLM path that consumes the evidence block |

**The contract is the point.** A vision model will describe things it did not see, so the analysis
carries the frames it was shown and every claim must land inside one of them. A recommendation must
name a metric the reports measure, for the same reason a `MutationProposal` may not predict something
unmeasurable: an analysis that cannot be refuted is not evidence. An analysis failing either check is
dropped with its reason recorded, never edited into validity.

**The media rule is declared before any analysis runs**: episode start, closest approach, the first
gate crossing of 0.5, the largest single-step object movement, and termination. A moment the episode
cannot supply is named rather than replaced. Capture hangs off a new `observer` hook on `run_episode`
rather than becoming a sixth copy of the rollout loop, and the observer is called after the action is
chosen with its return value discarded, so it cannot change the episode it is recording.

Run against the simulator on ten development cases: `simple_standoff` produced four frames per
episode and reported the gate transition unavailable, which is correct because it has no gate.
`simple_reach_push` produced five on the episode where its contact gate crossed and four on the two
where it never did, which is itself a diagnosis.

### AC-7 groundwork — freeze, lock and aggregate report

Phase 7 itself needs five independent LLM searches and at least 100 final-test episodes, which is an
expensive run awaiting approval. Three of its seven items are machinery rather than experiment, and
those are now built and tested.

| AC-7 item | Status |
|---|---|
| Prompts, models, code, optimizer, manifests, selection rules and budgets frozen before final testing | machinery done: `scripts/freeze_method.py`, drift check with `--verify` |
| At least five independent search runs | not run; needs approval |
| Each finalist on at least 100 isolated final-test episodes | the split is locked until a session opens it; the ledger counts every look |
| The paired 95% interval excludes zero | `paired_difference` exists; no final run to apply it to |
| MVP and stretch targets reported separately | not run |
| The report includes the full quantity list | assembled from artifacts; fourteen of fifteen quantities are absent today and each names where it would come from |
| One documented command regenerates the report | done: `python scripts/final_report.py` |

**The report will not fill a gap with a zero.** It carries `unavailable` in the same spirit as
`EvaluationReport.unavailable_fields`. Running it today shows what Phase 7 instrumentation is
missing: LLM call and token counts, wall time, inference latency, expert-label totals, sampled-action
performance and per-episode final distance are recorded nowhere, so the report says so rather than
printing zeros that would read as measurements.

## Files changed

**New**

| File | Role |
|---|---|
| `lares/eval/__init__.py` | public surface of the evaluation contract |
| `lares/eval/manifest.py` | `EvaluationManifest`, `EpisodeCase`, `EnvironmentSpec`, `TaskPool`, `build_manifest`, `assert_disjoint`, `synthetic_cases`, `synthetic_manifest` |
| `lares/eval/runner.py` | `run_episode`, `evaluate_manifest`, `ManifestResult`, `EpisodeRecord`, `PolicyActor`, `ExpertActor`, `ConstantActor`, `PipelineEnvs`, `paired_difference` |
| `scripts/lock_baseline.py` | builds the manifests, reproduces the baselines, runs every AC-0 check |
| `config/manifests/push-v2_{train,development,final_test}.yaml` | the committed evaluation contract |
| `tests/test_evaluation_manifest.py` | 22 tests: schema, pool reproducibility, drift detection, committed splits |
| `tests/test_manifest_runner.py` | 12 tests: replay order, isolation, sampled vs deterministic, paired comparison |
| `lares/core/obs_schema.py` | `ObsSchema`, `ObsField`, `get_obs_schema`, `check_against_env`, active-schema binding |
| `lares/core/policy_validator.py` | 17 error categories, `validate_policy`, `check_source`, `GateAggregator` |
| `tests/test_policy_interface.py` | 39 tests: schema behaviour, one policy per validator failure class, gate telemetry, layout verified against the simulator |
| `lares/eval/diagnostics.py` | `RolloutGeometry`, failure labels, per-axis action statistics, gate aggregation |
| `lares/eval/report.py` | `EvaluationReport` with validity, fitting and rollout sections, checkpoint constants, `build_report`, `episode_table` |
| `lares/eval/promotion.py` | `PromotionPolicy`, screening decisions, `compare_to_incumbent`, `PromotionDecision` |
| `tests/test_evaluation_report.py` | 41 tests: geometry, failure labels, report assembly, unavailable fields, fitting diagnostics, screening and paired promotion |
| `lares/fitting/structure_bank.py` | frozen bank, `port_raw_indices`, `check_port_equivalence`, family classification |
| `lares/fitting/optimizers.py` | four fitting methods behind one `fit()` call, `FitResult`, checkpoint rules |
| `lares/fitting/sensitivity.py` | action-versus-parameter and gradient sensitivity, dead-parameter detection |
| `lares/fitting/benchmark.py` | matched-budget comparison, aggregation, paired method comparison, the predeclared selection rule |
| `lares/fitting/structures/*.py` | 18 frozen structures: 15 ported from the 2026-08-28 search, 3 hand-written simple-family references |
| `scripts/build_structure_bank.py` | ports, verifies and writes the bank |
| `scripts/benchmark_fitting.py` | runs the comparison and writes a report |
| `tests/test_fitting.py` | 42 tests: porting, bank integrity, budget matching, checkpoint rules, sensitivity, aggregation and selection |
| `lares/search/schemas.py` | `PolicyIdea`, `MutationProposal`, `ExperimentRecord`, `measurable_metrics`, `source_hash` |
| `lares/search/archive.py` | `Archive` with robust, simplest and behaviourally distinct slots; `behavior_descriptor` |
| `lares/search/feedback.py` | `build_feedback`, `choose_informative_parents`, the population table and per-candidate diagnostics |
| `scripts/summarise_experiments.py` | regenerates the search summary from saved records alone |
| `tests/test_search.py` | 56 tests: schema refusals, record traceability, archive slots, parent choice, feedback content |
| `lares/fitting/objectives.py` | five objectives, four execution contracts, `check_execution_matches`, scale-parameter freezing |
| `lares/fitting/phases.py` | ground-truth expert phases, `balanced_indices`, `verify_against_expert` |
| `lares/fitting/dagger.py` | learner-state aggregation, cost accounting, counterfactual recovery check |
| `scripts/audit_expert_actions.py` | measures expert generation and clipping semantics before any likelihood is chosen |
| `scripts/compare_objectives.py` | E7 objectives, uniform against phase-balanced, E8 fixed buffer against learner state |
| `tests/test_objectives.py` | 48 tests: contracts, objective behaviour, scale freezing, phases, label validity, simulator branching, rollout checkpoint selection |
| `lares/eval/freeze.py` | `FreezeRecord`, drift detection, the final-test lock, the look-ledger |
| `scripts/freeze_method.py` | writes the freeze record; `--verify` reports drift and exits non-zero |
| `tests/test_freeze.py` | 25 tests: record completeness, write-once, drift naming, the lock, session nesting and exception safety, ledger accounting and re-score refusal |
| `lares/eval/final_report.py` | assembles the aggregate report from saved artifacts alone; names every AC-7 quantity no artifact carries |
| `scripts/final_report.py` | the one documented command that regenerates it |
| `tests/test_final_report.py` | 27 tests: section arithmetic, intervention exclusion, budget separation, determinism, and that no gap is filled with zero |
| `lares/vision/schema.py` | `VisualFailureAnalysis`, `MediaItem`, `Evidence`, citation and measurability validation |
| `lares/vision/media.py` | the fixed selection rule, the `run_episode` observer that records frames, final-test refusal |
| `lares/vision/merge.py` | runs the recommended checks, marks conflicts, builds the repair-prompt block |
| `lares/vision/analyst.py` | the analyst interface, the scripted double, and `QwenAnalyst` which refuses to pick a checkpoint |
| `lares/vision/pipeline.py` | `analyse_failures`, the one entry point, with cost and audit accounting |
| `scripts/analyse_failures.py` | captures media and merges evidence; useful with no model |
| `tests/test_vision.py` | 65 tests: the citation contract, selection determinism, unavailable moments, final-test refusal, conflict marking, and that the observer cannot change the episode |
| `lares/repair/interventions.py` | four controlled substitutions, `InterventionActor`, gate-override effectiveness check |
| `lares/repair/hypotheses.py` | failure clusters, competing explanations, ranking, `evaluate_repair` |
| `lares/repair/repair.py` | component scopes, measured parameter selection, the two repair operators, boundary check, acceptance with the vacuity guard |
| `lares/repair/study.py` | `run_study` (diagnosis), `run_repair` (one bounded repair, validated), `records_from_study` (FR-10 traceability), summary and JSON record |
| `scripts/diagnose_repair.py` | fits a structure, diagnoses it by substitution, generates and validates one repair |
| `scripts/ablate_repair.py` | AC-6 / E9: five arms at a matched development-episode budget, scored on a held-out slice |
| `tests/test_repair.py` | 94 tests: substitutions, gate overrides, clusters, competing explanations, ranking, scopes, parameter selection, bounded edits, search budget accounting, acceptance, the repair cycle end to end, and intervention traceability |

**Modified**

| File | Change |
|---|---|
| `lares/utils/metaworld_env.py` | `_mt1_rng` removed. `reset(case)` is now required and installs the placement plus reset seed. `make_metaworld_env` requires an explicit task pool when `use_mt1`. Non-MT1 path sets `seeded_rand_vec = True`. |
| `lares/core/training_pipeline.py` | `generate_dataset(env, env_name, cases, ...)`, `_collect_trajectories(policy, env, cases, ...)`, `rl_finetune(policy, env, train_cases, ...)`, `record_episode_gif(..., case, ...)`. `evaluate_policy(policy, env, manifest, ...)` returns a `ManifestResult`. `EvolutionOrchestrator` takes `dev_manifest`, `train_cases`, `gif_case`, and `run(client, envs, ...)`. `DemoBuffer` records `episode_ids`. |
| `scripts/run_full_evolution.py` | loads the committed manifests, builds one env per stream, scores the expert through the same runner as the candidates. New config keys `manifest_dir` and `eval_max_steps`. |
| `scripts/visualize_expert_policy.py` | replays named development cases so a GIF can be matched to a reported episode. |
| `scripts/run_shadowhand_spin_stages.py` | uses `synthetic_cases`; local `evaluate_stage_policy` replaces the shared `evaluate_policy`, which now needs a MetaWorld manifest. |
| `lares/envs/isaac_lab_adapter.py` | `reset(case=None)` for interface uniformity; seeds from the case when the backend supports it. |
| `lares/core/policy_generation.py` | validation subprocess uses `sys.executable`; `env_name` threaded through and now required; in-process instantiation re-validates; `TASK_NOTES` added. |
| `lares/core/symbolic_policy.py` | named accessors `obs_field` / `obs_fields`, gate telemetry `record_gate` / `reset_diagnostics` / `gate_snapshot`, schema captured at construction. |
| `lares/core/symbolic_policy.py` (Phase 6) | `set_gate_override` / `clear_gate_overrides`; `record_gate` returns the forced value when one is set, so a gate is a lever and not only telemetry. `reset_diagnostics` deliberately leaves an override in place. |
| `lares/fitting/structures/simple_reach_push.py`, `simple_damped_push.py` | use the value `record_gate` returns, so forcing the contact gate actually changes the action. |
| `lares/eval/manifest.py` | `tail(n)`, the disjoint complement of a screening prefix, for scoring on cases a loop never saw. |
| `lares/eval/runner.py` (FR-4) | `evaluate_manifest` refuses the `final_test` split outside a `final_test_session`. |
| `lares/eval/runner.py` (FR-11) | `run_episode` takes an optional `observer`, called after the action is chosen and with its return value discarded, so frame capture is not another rollout copy. |
| `lares/core/training_pipeline.py` | `load_policy_prompt_assets` builds the layout from the verified schema instead of a hand-written table; diagnostics reset per episode in the rollout sites. |
| `lares/eval/runner.py` | per-episode `gate_statistics`, geometry and per-axis action fields; `schema_id` on the result; `PolicyActor` resets diagnostics and exposes gates. |
| `lares/core/training_pipeline.py` (Phase 2) | `split_buffer_by_episode`; `DemoBuffer.sample(indices=)` and `take`; BC reports per-axis train and validation loss, gradient norms, bound activity and loss by phase, and calls a halfway `checkpoint_callback`; the orchestrator evaluates three checkpoints, screens, expands and compares against the incumbent. |
| `scripts/lock_baseline.py` (Phase 2) | writes a report per baseline and gains `--collect-demos` for an expert buffer with episode ids. |
| `lares/utils/policy_prompts/*.txt` | rewritten for named accessors, plus the known code-generation traps and the zero-shot-behaviour request. |
| `tests/test_generate_policy.py` | harness delegates to `validate_policy`, binds the schema from `--env_name`, recovers the generated source from its own file. |
| `config/run_full_evolution.yaml` | manifest keys, and comments explaining why the pool seed is part of the contract. |
| `CLAUDE.md` | evaluation-contract and policy-contract sections, corrected MT1 and rollout-duplication notes, incumbent warning. |
| `tests/test_training_pipeline.py`, `tests/test_record_episode_gif.py`, `tests/test_env_action_response.py`, `tests/test_phase1_phase2.py`, `tests/test_two_phase_policy_generation.py` | migrated to the cased-reset, manifest and named-accessor APIs. |
| `docs/SYMBOLIC_POLICY_IMPL.md`, `docs/PIPELINE_REFINEMENT_SPEC.md`, `docs/TRAINING_PIPELINE.md`, `docs/Codespace_explanation.md`, `docs/run_full_evolution_overview.md` | status notes and corrected API descriptions. |

## Tests run and exact results

| Suite | Result |
|---|---|
| `python tests/test_phase1_phase2.py` | 41/41 passed |
| `python tests/test_training_pipeline.py` | 139/139 passed, 0 skipped (Tier 1, 2 and 3) |
| `python -m unittest ...` (all thirteen unittest modules) | 490 passed (478 in the eleven contract suites, 12 in the two generation suites) |
| `python scripts/analyse_failures.py --structures simple_standoff` | 12 frames captured at the selected moments; gate transition correctly reported unavailable |
| `python scripts/analyse_failures.py --structures simple_reach_push` | 13 frames; the gate crossing found on the one episode where the contact gate opened |
| `python scripts/lock_baseline.py --build-manifests --episodes 30` | 7/7 AC-0 checks passed |
| `python scripts/run_full_evolution.py --config <smoke>` | end-to-end run completed; the generated policy used named accessors and passed all nine checks first try; GIF written for a named case |
| `python scripts/lock_baseline.py --episodes 30 --collect-demos 150` | 7/7 AC-0 checks; 22,500-transition buffer with episode ids collected on the train manifest |
| `python scripts/build_structure_bank.py` | 15 structures ported, every port verified output-identical and validator-clean |
| `python scripts/benchmark_fitting.py --budget 2000` | 18 structures x 4 methods x 2 checkpoint rules, 1120 s |
| `python scripts/run_full_evolution.py --config <2-generation smoke>` | records, archive slots and the evidence block written per generation |
| `python scripts/summarise_experiments.py` | 4 records, 100% reached validation, no traceability gaps |
| `python scripts/audit_expert_actions.py --episodes 30` | 4500 transitions; 18.8% censored on at least one axis |
| `python scripts/compare_objectives.py --budget 2000 --family simple` | E7, sampling and E8 complete |
| budget sweep, 2 structures x 8 budgets x 3 seeds on 50 development cases | success peaks at 1000 steps, collapses to 0.000 by 3000 |
| `python scripts/diagnose_repair.py --structures simple_standoff` | two competing explanations tested, one supported and one rejected; one bounded repair generated, validated and accepted |
| `python scripts/ablate_repair.py --budget 200 --seeds 0 1` | 5 arms x 3 structures x 2 seeds, 6000 screen episodes, 2350 s; AC-6's last item not met |
| `python scripts/final_report.py` | aggregate report regenerated from saved artifacts; 14 of 15 AC-7 quantities named absent |
| `python scripts/lock_baseline.py --episodes 10` | 7/7 AC-0 checks, re-run with the final-test lock in place |

## Existing failures versus newly introduced failures

No failing tests. Two failures present before this work are now fixed:

- `BC: has log_prob key` asserted a key from the pre-MSE objective. Replaced with the two keys the
  live objective actually produces.
- Tier 3 crashed with `FileNotFoundError: 'python'` because the validation subprocess spawned a bare
  `python`, which is absent unless the venv is on `PATH`. Now `sys.executable`.

Four failures were introduced by Phase 1 and fixed in the same phase. Every one was a test fixture
written against the old contract: policies using raw `obs[:, 0:3]` indexing, and a
`get_symbolic_policies` call that did not name its task so no schema was bound. Fixtures were
migrated to named accessors, and `get_symbolic_policies` now refuses an empty `env_name` rather than
returning an empty population.

## Decisions and deviations from spec.md

1. **The incumbent is 0.00, not 0.13.** Confirmed with the user: adopt the reproduced value as the
   formal incumbent and keep the section 2.3 targets as written, so minimum credible improvement
   means reaching 0.20 final-test success.
2. **Splits come from disjoint MT1 pool seeds**, not from slicing one pool. Provenance is one line
   per split and a split grows by adding a seed. MT1 exposes no test tasks, so a holdout had to be
   generated either way.
3. **Clean break on `_mt1_rng`**, confirmed with the user. All five rollout sites were migrated in
   one change rather than keeping the legacy path behind a flag.
4. **`head(n)` is a prefix, never a resample.** The 10-case screen is nested inside the 30-case
   expansion, so a promoted candidate is re-scored on the cases that promoted it plus more.
5. **`synthetic_manifest` exists but refuses `resolve_pool()`.** Mocks and the Isaac Lab backend
   need to name a case, but must not be able to stand in for a locked manifest in a reported result.
6. **`DemoBuffer` now records `episode_ids`.** MetaWorld never terminates early, so `dones` is all
   zeros and cannot mark episode boundaries. FR-6 asks for splits by complete trajectory, which the
   saved buffer could not support. Buffers written earlier load with blank ids.
7. **Per-episode diagnostics are partial.** `EpisodeRecord` captures what the `info` dict provides
   plus gate statistics. The geometry fields of `EvaluationReport` (section 7.3) are Phase 2 work,
   now that named accessors exist to compute them without hard-coding indices.

### Phase 1

8. **Named accessors rather than an observation view object.** `forward(obs)` keeps its tensor
   argument and the base class provides `self.obs_field(obs, name)`. Passing a view object instead
   would have changed the signature that behavioural cloning, the RL collector, the manifest runner
   and the subprocess harness all call. The ban on raw indices is enforced by an AST check on the
   generated source, which is a precise rule rather than a type change.
9. **Probes run at several observation scales.** A single standard-normal scale drives a 0.06 m
   distance gate to exactly zero, so a gated term contributes nothing and both the batch-coupling
   and field-sensitivity checks silently measure nothing. Coupling fails if it appears at any scale;
   sensitivity passes if it appears at any scale, because a term behind a contact gate is a real
   dependence even though it is inert far from contact.
10. **The subprocess harness now delegates to the validator.** CLAUDE.md previously warned that the
    harness kept its own forbidden-module list which had drifted from the base class. There is now
    one list. The harness still runs in a subprocess, which is what makes untrusted source safe.
11. **`get_symbolic_policies` requires `env_name`.** Without it no schema is bound, every generated
    policy raises on its first `obs_field` call, and the population comes back empty with no
    explanation. It now fails loudly instead.
12. **The saved incumbent predates the interface.** It uses raw indices and would be rejected by
    today's validator. `scripts/lock_baseline.py` loads it with the module blacklist only, since it
    is a historical artifact being reproduced, not a candidate being generated.

### Phase 2

13. **The screening rule has a fallback, and it has to.** Promoting only on success would advance
    nobody while every candidate scores zero, and the search would get no signal at all. The rule
    promotes on any success, and otherwise advances the top two by mean signed goal progress. Both
    thresholds are stored with the results, so a promotion can be read back against the rule that
    justified it.
14. **`eval_episodes` became the expanded budget.** Configs now carry `screen_episodes` and
    `expanded_episodes`; `eval_episodes` is kept as the expanded alias so older configs keep their
    meaning rather than silently shrinking the evaluation.
15. **Geometry is measured, not inferred from reward.** Progress, drift, displacement and closest
    approach come from the named accessors. `signed_goal_progress` is the measure that separates
    "pushed the wrong way" from "did not push far enough", and it is the fallback ranking key.
16. **Failure labels are heuristic and say so.** The thresholds they use are written into every
    report's `measurement_notes`, so the word is auditable rather than authoritative.
17. **`timeout_rate` is reported but carries no signal here.** MetaWorld never terminates early, so
    every episode reaches the horizon. The note is in the report rather than left for a reader to
    rediscover.

### Phase 5

24. **The audit decided the likelihood, not the other way round.** MetaWorld's expert does no
    clipping: `move()` returns `10 * error` and only warns when it leaves the limits. The endpoint
    mass in the targets is ours. That makes the censored Gaussian the model matching the generating
    process, and makes the tanh-Gaussian likelihood wrong twice over.
25. **Execution contracts are first-class.** Every objective declares the execution it assumes, and
    `check_execution_matches` refuses to score one under another. `spec.md` FR-7 warns against
    calling an action-space regression a likelihood; this makes that structurally impossible rather
    than a matter of discipline.
26. **Scale parameters are found by measurement.** Matching on a name like `log_std` would miss
    `std_shrink` or `alpha_std` and could catch a control parameter that happened to be named
    similarly. The parameters frozen for the deterministic baseline are the ones measured to move
    the scale but not the executed action.
27. **The phase labels come from the expert's own branches**, and are checked against the branch it
    actually takes on 200 random states. Deriving phases from a learned gate would be circular: the
    gate is part of what is being evaluated.
28. **The recovery check needed two corrections before it measured anything.** First it measured the
    *learner's* step rather than the expert's label; the fix was to branch the simulator, apply the
    label, measure, and rewind. Then it measured object-to-goal distance, which fails both approach
    phases by construction since no action can move the object before contact, and then
    hand-to-object distance, which fails the hover phase because the expert deliberately lifts 0.2 m
    above the puck. The criterion is now the distance to the expert's own desired position, which is
    definitionally what its action reduces.

### Phase 3

18. **The bank is ported, not rewritten.** The fifteen historical candidates keep their exact
    arithmetic; only the observation reads change, and `check_port_equivalence` proves the outputs
    are identical before any structure enters the bank. A comparison run on structures chosen or
    rewritten for the occasion would not say anything about the pipeline's own output.
19. **Differential evolution rather than CMA-ES**, answering `spec.md` open decision 4. It ships
    with scipy, so the comparison adds no dependency. Its objective is a fixed batch drawn once,
    because a resampled batch makes the objective stochastic and population methods handle that
    badly; the cost is a risk of overfitting that batch, which the reported validation loss exposes.
20. **The budget unit is objective evaluations**, as AC-3 asks. Transitions processed and wall time
    are reported alongside, because equal evaluation counts are not equal compute and hiding that
    would make the comparison look fairer than it is. Differential evolution can only land near the
    budget, since scipy evaluates whole generations; both the request and the actual count are
    recorded.
21. **Multistart splits the same budget, it does not add to it.** Four restarts of 500 steps against
    one run of 2000. Its worse fit is expected; its better controller is the finding.
22. **The two checkpoint rules coincide on this data.** Validation loss improves monotonically
    within the budget, so the best check lands at the end. The rules are still evaluated separately,
    because that will stop being true once fitting runs long enough to overfit.
23. **Every structure carries dead parameters.** The scale parameters cannot move the executed
    action, since deterministic execution is `tanh(mean)` and ignores `std`. Two structures also have
    dead control parameters. This is reported per structure rather than aggregated away.

### Phase 6

24. **Open decision 3 is answered, and the answer exposed a hole.** The protected-case regression
    tolerance is 0.05 of paired success, declared as a module constant so a repair cannot be waved
    through by loosening it at the call site. On this task that threshold is unfailable: the
    incumbent scores 0.000, so no case can regress. `accept_repair` detects the vacuous case and
    falls back to a mean signed-goal-progress tolerance of 0.01 on the cases outside the cluster.
25. **A component is turned into a parameter set by measurement, never by name.** Matching a name
    like `gain` would be one rename away from silently repairing nothing. The parameters in scope
    are the ones a perturbation shows move the scoped action axes, or the named gate.
26. **The edit's boundary is checked, not asserted.** `bounded_change` re-reads every parameter
    after the repair and `accept_repair` refuses a decision whose edit touched anything outside the
    scope it declared.
27. **Two repair operators, because the diagnoses differ in kind.** Refitting the in-scope
    parameters toward the expert cannot correct a geometry term, since the current values are what
    fitting to the expert produces. Searching the same bounded set against the focused metric can.
    Both are kept and the record names which ran.
28. **Intervention runs enter the experiment record stream marked as interventions.** They were
    otherwise visible only in a study file, so a summary regenerated from records alone would not
    know the rollouts had happened. The repair itself is recorded as an ordinary rankable candidate.

## Blockers

One AC-2 item is blocked: whether the VLM earns its place in the final method, which E10 decides.
That needs a Qwen checkpoint (`transformers` is not installed, and the model and quantisation are
unchosen) and a correction-LLM repair path, because the current repair operators read no prompt and
so cannot be affected by an evidence block. The other two VLM items are now enforced.

Phase 7's experiment needs approval before it runs. Five independent LLM searches plus at least 100
final-test episodes per finalist is the largest single expenditure in the plan, and the final-test
manifest can only be opened once per candidate without the ledger recording a re-score.

AC-6's fifth item is answered rather than blocked. The full repair loop lost to optimizer-only with
a paired interval that excludes zero. Deciding what follows from that is a research call, not an
implementation one.

## Exact next action

Phase 6 is measured and Phase 7's machinery is built. The next action is a decision, not a keystroke:

1. **Accept the AC-6 result and freeze.** Run `python scripts/freeze_method.py --freeze-id v1`, then
   the five independent searches and the final test, with approval. The freeze must be written before
   the first final-test episode, and `--verify` must be clean at the end.
2. **Or attack the reason the substitution does not pay.** The competing explanations are a fixed
   table keyed on the failure label, and for the commonest cluster its first entry is already the
   supported one, so the substitution confirms rather than discriminates. Generating the explanations
   instead of tabulating them is the change most likely to move E6, and it is where FR-11's visual
   evidence would enter.

Whichever is chosen, the fitting schedule is the larger lever on this task and the ablation says so
plainly: recovering the pre-collapse checkpoint is worth more than any repair measured here.
