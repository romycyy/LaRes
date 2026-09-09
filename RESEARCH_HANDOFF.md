# Research Handoff

Evidence log for the symbolic-policy evolution pipeline. Updated as phases land.

## Where this stands

Phases 0 through 6 are implemented and measured. FR-4 and FR-11 are implemented.
Phase 7's machinery is built but its experiment has not run.

| Phase | State |
|---|---|
| 0 measurement lock | complete, 7/7 AC-0 checks |
| 1 trusted interface | complete, 6/6 AC-1 items |
| 2 evaluation and reporting | 8/9 AC-2 items; the ninth is E10, which is blocked |
| 3 fitting comparison | complete, 6/6 AC-3 items |
| 4 search evidence and archive | complete, 8/8 AC-4 items |
| 5 objectives and data coverage | complete, 7/7 AC-5 items |
| 6 intervention-guided repair | 4/5 AC-6 items; the fifth is answered in the negative |
| 7 freeze and final validation | machinery built, experiment awaiting approval |

Three results decide what is worth doing next, and all three run against the
plan's assumptions:

1. **Fitting the expert better does not build a better controller.** Validation
   loss correlates with development success at +0.19, the wrong sign, and
   development success collapses to zero well before the loss minimum.
2. **The repair loop does not beat spending the same budget on the optimizer.**
   Paired difference +0.133 in optimizer-only's favour, interval excluding zero.
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

## Repository commit and working-tree status

| Field | Value |
|---|---|
| Commit at Phase 0 start | `619f0cf` "doc setup for remote R&D" |
| Branch | `staging`, pushed to `origin/staging` |
| Head after Phases 0-6, FR-4, FR-11 | `bb1d94f` "Docs: record what each phase measured, and what it did not" |
| Working tree | clean; all work committed as ten commits, one per phase |
| Not independently runnable | the intermediate commits. Three files span phases and could not be split: `lares/eval/runner.py` (Phase 0, carrying the FR-4 lock and the FR-11 observer), `lares/core/symbolic_policy.py` (Phase 1, carrying the Phase 6 gate overrides), `lares/core/training_pipeline.py` (Phase 2, carrying the Phase 5 rollout probe). Only the tip passes its tests. |

## Environment and package versions

Recorded automatically by `scripts/lock_baseline.py` into
`logs/baseline_lock/baseline_lock_<timestamp>.json`. At the Phase 0 lock:

| Component | Version |
|---|---|
| Python | 3.10.20 |
| torch | 2.7.0+cu128 |
| numpy | 1.26.4 |
| metaworld | 3.0.0 |
| mujoco | 3.8.0 |
| gymnasium | 1.3.0 |
| gym | 0.26.2 |
| openai | 2.33.0 |
| Platform | Linux 6.17.0-1012-aws, Tesla T4, CUDA 12.8 |
| `MUJOCO_GL` | `egl` (no `DISPLAY` on this box) |

The working interpreter is `.venv-metaworld/bin/python`, not the conda spec in
`config/environment.yaml`. It is not checked in.

## Implemented pipeline behaviour

Phase 0 replaced the hidden task stream with an explicit evaluation contract.

- `EvaluationManifest` is an ordered, immutable list of `EpisodeCase`s. Each case names a task
  placement, a reset seed and a policy seed. Manifests are committed under `config/manifests/`.
- Placements come from `MT1(env_id, seed=S)`. The seed is part of the contract.
- `env_wrapper.reset(case)` installs the placement and reseeds the simulator. `reset()` with no
  case raises. There is no `_mt1_rng`.
- Each pipeline stream gets its own environment. `PipelineEnvs` refuses a shared instance unless
  a caller explicitly opts in for mocks.
- Deterministic evaluation executes `tanh(mean)`. Sampled evaluation executes
  `tanh(mean + std * eps)` with `eps` from a generator seeded by the case. Results are never merged.
- `paired_difference` estimates uncertainty for the difference itself, over cases both results ran.

Phase 1 added the trusted policy interface.

- Generated policies read the observation through `self.obs_field(obs, "<name>")`. Raw slicing is
  rejected by an abstract-syntax-tree check on the source before the policy is ever run.
- `ObsSchema.check_against_env` compares the declared layout against the simulator's own accessors
  for the end-effector, object, object orientation and goal. A deliberately wrong slice is caught.
- `validate_policy` collects every defect in one report with stable category strings, covering raw
  indexing, missing schema, forbidden modules, undeclared and phantom parameters, inverted ranges,
  out-of-range initial values, wrong shapes, non-finite output, non-positive scale, batch coupling,
  insensitivity to a required field, gate-telemetry misuse and serialization.
- Policies may expose phase gates through `record_gate`. Values are detached, so telemetry cannot
  change the action. Per-episode statistics report mean, minimum, maximum, occupancy above 0.5,
  first crossing and transition count.

Phase 2 added structured evaluation.

- Every candidate produces an `EvaluationReport` per checkpoint, holding validity, fitting, rollout
  aggregates and the full per-episode table. Anything not measured is listed in
  `unavailable_fields` and left null.
- Each episode is measured through the observation schema: goal progress, lateral drift, object
  displacement, closest approach, per-axis saturation and variation, and a coarse failure label.
- Candidates are screened on ten cases and only re-scored on thirty if they clear a rule declared
  before the generation ran. The screen is a prefix of the expanded set, so the two stay paired.
- Promotion is decided on the paired success difference against the incumbent, with return breaking
  a tie only when success is equal.
- Behavioural cloning splits validation off by complete episode and reports per-axis train and
  validation loss, gradient norms with a clipping fraction, per-parameter bound activity, and the
  fitting error split by which exposed gate was open.

Phase 3 separated structure search from parameter fitting.

- A frozen bank of 18 structures lives in `lares/fitting/structures/`: 15 ported from the candidates
  an earlier search produced, plus 3 hand-written simple-family references. The ported ones keep
  their exact arithmetic and only their observation reads changed; equivalence is proved before a
  structure enters the bank.
- Four fitting methods run behind one call: the reproduced Adam baseline, range-scaled Adam with a
  cosine schedule, equal-budget multistart Adam, and scipy differential evolution.
- All four optimise the identical objective and receive the same objective-evaluation count.
  Transitions processed and wall time are reported too.
- Sensitivity measures how much each parameter moves the executed action, in units of that
  parameter's own declared range, which surfaces parameters that cannot affect the controller at all.

Phase 5 fixed the objective contracts and the checkpoint rule.

- Five objectives are available, each carrying the execution contract it assumes.
  `check_execution_matches` refuses to score one under another, so an action-space regression cannot
  be reported as a likelihood.
- The primary baseline is `MSE(tanh(mean), a)` with the scale frozen. The scale parameters are found
  by measuring which ones move the scale but not the executed action, not by matching a name.
- `behavioral_cloning` accepts a rollout probe and keeps the best-scoring parameters. This replaces
  a fixed budget with a measured stopping point, which the budget sweep below shows is necessary.
- Learner-state aggregation runs on training placements only, counts every expert query and
  environment step, and validates each label by branching the simulator.

Phase 4 replaced the feedback the generator receives and made every attempt auditable.

- Three schemas are enforced. An idea must declare a role, type, units, initial value and range for
  every parameter, and must state what the untrained policy should already do. A mutation proposal
  is refused before code generation if its predicted metric names nothing the reports measure. An
  experiment record refuses to validate if an evaluated candidate lacks a code hash, a manifest id
  or an action mode.
- One record is written per attempted candidate, including the ones that never compiled, under
  `<log_dir>/experiments/`.
- The archive keeps three slots: most robust, simplest competitive, and most behaviourally distinct.
  Distance is measured on the failure profile and geometry, not on source text, so two programs that
  read differently and fail identically are not counted as diverse.
- Intervention results are recorded and are structurally barred from occupying a slot or entering
  fitness ranking.
- The generator now receives a ranked population table, full code for two parents chosen to differ
  behaviourally, per-episode geometry and failure labels, gate occupancy, dead parameters, the
  archive, and the errors from candidates that never compiled.
- Repairs are capped at one attempt per candidate, as AC-4 requires.

## Reproduction commands

```bash
export MUJOCO_GL=egl
export PATH="$PWD/.venv-metaworld/bin:$PATH"     # policy_generation spawns a subprocess

python scripts/lock_baseline.py                  # AC-0 checks against committed manifests
python scripts/lock_baseline.py --build-manifests # regenerate the manifests first
python -m unittest tests.test_evaluation_manifest tests.test_manifest_runner \
  tests.test_policy_interface tests.test_evaluation_report tests.test_fitting
python scripts/lock_baseline.py --collect-demos 150   # expert buffer with episode ids
python scripts/build_structure_bank.py                # freeze the structure bank
python scripts/benchmark_fitting.py --budget 2000     # the fitting comparison
python scripts/summarise_experiments.py               # search summary from records alone
python scripts/audit_expert_actions.py                # expert generation and clipping semantics
python scripts/compare_objectives.py --budget 2000 --family simple   # E7, sampling, E8

python scripts/diagnose_repair.py --structures simple_standoff       # Phase 6: study + one repair
python scripts/ablate_repair.py --budget 200 --seeds 0 1             # AC-6 / E9, five arms, ~40 min
python scripts/ablate_repair.py --summarise logs/repair_ablation/<run>.json   # re-print, no rollouts
python scripts/analyse_failures.py --structures simple_standoff      # FR-11: media, no model needed
python scripts/freeze_method.py --freeze-id v1                       # Phase 7: freeze before final test
python scripts/freeze_method.py --verify                             # non-zero exit if anything moved
python scripts/final_report.py                                       # AC-7: report from saved files

# The full contract suite, eleven modules.
python -m unittest tests.test_evaluation_manifest tests.test_manifest_runner \
  tests.test_policy_interface tests.test_evaluation_report tests.test_fitting \
  tests.test_search tests.test_objectives tests.test_repair tests.test_freeze \
  tests.test_final_report tests.test_vision
python tests/test_phase1_phase2.py && python tests/test_training_pipeline.py
```

Current totals: 478 tests across the eleven contract suites, 139 in the
training-pipeline tiers, 41 in the phase-1/2 script, 12 in the two generation
suites. All passing at `bb1d94f`.

## Manifest and reproducibility checks

| Split | Cases | MT1 pool seeds |
|---|---:|---|
| train | 150 | 1000, 1001, 1002 |
| development | 50 | 2000 |
| final_test | 100 | 3000, 3001 |

Disjoint by placement hash, asserted on every lock run. Development case 0 through 9 is the
screening subset and 0 through 29 the expanded set, as prefixes rather than resamples.

Three findings established the contract:

1. **`MT1('push-v3')` without a seed is not reproducible.** Two constructions in the same process
   gave different placements. The previous `make_metaworld_env` used exactly that call, so episode
   index 7 named a different puck and goal on every run. Seeded construction is reproducible and
   costs 1.2 s per 50 placements.
2. **A fixed placement gives bit-exact episodes.** Maximum observation difference over 150 steps
   across repeats was exactly 0. AC-0 asks for identical outcomes or documented nondeterminism; the
   strict version holds, so `lock_baseline` asserts equality rather than a tolerance.
3. **Nothing else moves a case.** Running dataset collection and a debug rollout between two
   evaluations leaves every episode return unchanged.

## Baseline and paired-evaluation results

Development manifest, first 30 cases, deterministic actions, 150-step horizon.

| Actor | Mean reward | Success | Terminal object-to-target |
|---|---:|---:|---:|
| Scripted expert | 1068.92 ± 10.98 | 1.000 | 0.028 |
| Saved incumbent | 105.13 ± 4.72 | 0.000 | 0.272 |
| Zero action | 7.80 ± 0.03 | 0.000 | 0.221 |

Paired, incumbent minus expert:

| Metric | Difference | 95% CI |
|---|---:|---|
| Success | -1.000 | [-1.000, -1.000] |
| Reward | -963.79 | [-985.97, -941.61] |

### The incumbent does not reproduce

`logs/evolution/best_policy_code.py` is byte-identical to generation 2 candidate 0 of run
`20260828_190507`, the candidate its own run reported at 402.76 reward and 0.30 success. Re-scored
with its saved weights it reaches 0.000 success on 30 matched placements. Success needs
object-to-target at or below 0.05; the incumbent finishes at 0.272, so it is failing rather than
near-missing.

Zero successes in 30 episodes is very unlikely under a true rate of 0.30 and unlikely under the
0.13 that `spec.md` section 3 carried as a planning reference. The most economical explanation is
the one `docs/PIPELINE_REFINEMENT_SPEC.md` section 4 predicted: each candidate was scored on ten
placements nobody else saw, so the reported winner of each generation was substantially a lottery.
The same document records the saved policy re-scoring at 175.79, 293.04, 197.24 and 179.54 over
four 20-episode batches, against the 402.76 its run reported.

**Decision (confirmed with the user).** Adopt 0.000 as the formal incumbent. Record 0.30 and 0.13
as small-sample artifacts of unmatched evaluation. Keep the `spec.md` section 2.3 targets as
written, so minimum credible improvement now means reaching 0.20 final-test success.

### What the incumbent actually does wrong

The Phase 2 diagnostics split that flat zero into two distinct failures, measured over the same
thirty development placements.

| Measure | Incumbent | Scripted expert |
|---|---:|---:|
| Never reached the object | 17 of 30 | 0 of 30 |
| Reached it, then pushed away from the goal | 13 of 30 | 0 of 30 |
| Mean signed goal progress (m) | -0.051 | +0.193 |
| Mean lateral drift (m) | 0.027 | 0.027 |
| Mean closest approach (m) | 0.054 | 0.033 |
| Mean object displacement (m) | 0.093 | see progress |
| Action saturation, all four axes | 0.00 | 0.07 on two axes |

Three things follow. The controller does move the object, about 9 cm on average, so it is not inert.
It moves it *further from the goal* on balance, which is a control-law fault rather than a
range-of-motion one. And it never saturates any action axis, so the output gain is not the binding
constraint; that rules out the output-scale hypothesis in `docs/PIPELINE_REFINEMENT_SPEC.md`
section 3.5 for this candidate.

Its lateral drift matches the expert's almost exactly, which says the failure is along the goal
direction rather than sideways. Reports for each baseline are written to
`logs/baseline_lock/report_<actor>.json`.

## Distribution and action endpoint statistics

Measured on the saved expert buffer, `logs/evolution/demo_push-v2.pkl`, 22,500 transitions,
collected before the evaluation contract existed.

| Action dim | Mean | Std | Fraction \|a\| > 0.99 | Fraction exactly ±1 |
|---|---:|---:|---:|---:|
| 0 (x) | 0.005 | 0.209 | 0.0104 | 0.0100 |
| 1 (y) | 0.188 | 0.353 | 0.1042 | 0.1030 |
| 2 (z) | -0.446 | 0.262 | 0.0711 | 0.0711 |
| 3 (gripper) | 0.495 | 0.228 | 0.0000 | 0.0000 |

17.44% of transitions have at least one action exactly at an endpoint, and 17.57% exceed 0.99.
This answers `spec.md` open decision 8. A tanh-Gaussian likelihood on raw targets is undefined for
that 17.44%, since `atanh(±1)` diverges, which is why FR-7 keeps deterministic MSE as the primary
baseline. Endpoint mass, not merely near-saturation, is what a censored likelihood would have to
model.

Re-collected on the locked training manifest, 150 placements and 22,500 transitions, the figure is
**18.46%**. Path: `logs/baseline_lock/demo_push-v2.pkl`, rebuilt with

```bash
python scripts/lock_baseline.py --collect-demos 150
```

This buffer records an episode id per transition, so behavioural cloning can split validation off by
complete trajectory. The older buffer cannot, and `split_buffer_by_episode` refuses rather than
falling back to a transition split.

## Parameter-fitting comparison

18 frozen structures, four methods, 2000 objective evaluations each, ten development cases per
fitted result, both checkpoint rules. Total wall time 1120 seconds. Report at
`logs/fitting_benchmark/fitting_benchmark_20260909_095102.json`.

| Method | Validation loss | Development success | Parameters at a bound | Seconds |
|---|---:|---:|---:|---:|
| Adam baseline, reproduced | 0.0594 | 0.044 | 0.234 | 8.5 |
| Adam, range-scaled, cosine decay | 0.0512 | 0.044 | 0.258 | 9.6 |
| Multistart Adam, four restarts | 0.0824 | 0.072 | 0.174 | 8.6 |
| Differential evolution | 0.0730 | 0.044 | 0.000 | 4.1 |

Paired by structure against the reproduced baseline, since every method saw the same 18 structures:

| Comparison | Difference | 95% interval | Distinguishable |
|---|---:|---|---|
| Multistart, success | +0.0278 | [+0.0012, +0.0543] | yes, barely |
| Range-scaled, success | +0.0000 | [-0.0709, +0.0709] | no |
| Differential evolution, success | +0.0000 | [-0.0634, +0.0634] | no |
| Multistart, validation loss | +0.0231 | [+0.0136, +0.0325] | yes |
| Range-scaled, validation loss | -0.0082 | [-0.0110, -0.0054] | yes |
| Differential evolution, validation loss | +0.0136 | [+0.0073, +0.0199] | yes |

Multistart splits the same budget into four 500-step runs rather than one of 2000, so its worse fit
is expected. Its better controller is not.

### The headline result: fitting the expert better does not build a better controller

Across all 144 configurations the correlation between validation loss and development success is
**positive**, which is the wrong sign.

| Statistic | Value | p |
|---|---:|---:|
| Pearson | +0.189 | 0.023 |
| Spearman | +0.167 | 0.045 |

The paired comparisons say the same thing more sharply. Range-scaled Adam fits significantly better
than the baseline and gains exactly nothing in success. Multistart fits significantly worse and is
the only method whose success improvement excludes zero.

Three consequences for the pipeline.

1. **Behavioural-cloning loss is not a usable proxy for candidate quality on this task.** The inner
   loop currently fits on it and then ranks on success. Telling the LLM that a candidate "fitted
   well" is close to uninformative, and could be actively misleading.
2. **The optimizer is not the binding constraint.** The spread across four methods is smaller than
   the spread across structures, so effort spent on better fitting buys less than effort spent on
   better structures. That is the opposite of what `spec.md` section 14's risk "a promising structure
   is rejected after poor fitting" anticipated, and it lowers the priority of that mitigation.
3. **The success rates are all tiny**, 0.02 to 0.20 on ten cases. The multistart interval barely
   excludes zero. This is weak evidence and should be re-run at a larger development budget before
   any of it is treated as settled.

### Complexity ablation (E3)

| Family | Structures | Best method | Development success |
|---|---:|---|---:|
| Simple, at most 10 parameters | 3 | Multistart Adam | 0.200 |
| Complex, 22 to 36 parameters | 15 | Range-scaled Adam | 0.053 |

Simple structures score roughly four times higher despite fitting no better. With three simple
structures this is suggestive rather than settled, but it points the same way as the parameter-budget
proposal in `docs/PIPELINE_REFINEMENT_SPEC.md` section 2.2.

### Dead parameters

Every structure in the bank carries parameters the executed action cannot respond to. The scale
parameters are dead by construction, since deterministic execution is `tanh(mean)` and ignores
`std`. Two structures also carry dead *control* parameters: `gen_1_cand1` never uses `k_slip` or
`s_thr`. That is a concrete defect the generator could be told about.

### Bound activity

Adam leaves roughly a quarter of parameters resting on a declared bound: 0.234 for the baseline and
0.258 for the range-scaled variant. Multistart leaves 0.174 and differential evolution leaves none,
since it samples inside the box. A parameter pinned at a bound is the optimizer saying either that
the range is wrong or that the structure needs a magnitude it is not allowed to have.

## Generation-mode comparison (E4)

Per-idea versus batched implementation, both asking for the same four policies from the same
ideation prompt, three replicates each, gpt-4o-mini. Token counts are read from the pipeline's own
LLM transcript, not estimated. Report at `logs/generation_modes/20260909_103227/comparison.json`.

| Mode | Valid | Calls | Prompt tokens | Completion tokens | Tokens per usable candidate | Seconds |
|---|---:|---:|---:|---:|---:|---:|
| Batched | 12 of 12 | 14 | 69,756 | 14,997 | 7,063 | 146 |
| Per-idea | 12 of 12 | 16 | 44,723 | 16,975 | 5,142 | 145 |

Both modes are perfectly reliable at this size, so the comparison turns on cost, and per-idea costs
0.73 times as much per usable candidate. That is the opposite of the intuition that fewer calls
means fewer tokens: a batched request repeats every hypothesis inside one long prompt, so its input
grows with the population while a per-idea prompt carries one hypothesis each.

The default in `config/run_full_evolution.yaml` changed to `per_idea` on this evidence. Caveats: one
model, four policies per generation, three replicates, and a 100% validity rate that leaves no room
to detect a reliability difference. The comparison should be repeated at the population size and
model a real search uses before the choice is frozen for the final method.

## Expert action generation and clipping (AC-5)

`scripts/audit_expert_actions.py`, 30 training placements, 4500 transitions. Report at
`logs/expert_audit/`.

MetaWorld's scripted expert is an unbounded proportional controller. Its motion command is
`10 * (desired_position - hand_position)` and its grab effort is either 0.0 or 0.6. It performs no
clipping of its own: it warns and returns the raw value. The environment clips, and Stage 1 clips
before storing.

| Axis | Raw range | Fraction censored |
|---|---|---:|
| dx | -1.70 to 1.62 | 0.017 |
| dy | -0.09 to 2.67 | 0.112 |
| dz | -1.69 to 0.25 | 0.073 |
| gripper | 0.00 to 0.60 | 0.000 |

18.8% of transitions are censored on at least one axis, and the largest raw magnitude is only three
times the action limit.

**This settles the likelihood question.** The endpoint mass is a censored observation of an
unbounded deterministic signal, produced by our clip. A tanh-Gaussian likelihood is wrong twice
over: the generator is not a tanh squash, and `atanh(±1)` is undefined on the censored fraction. A
censored Gaussian is the matching model, and it requires clipped execution rather than the tanh
sampler, so it is implemented with that contract attached and is not the default.

The expert also has three explicit control branches, which makes the phase label ground truth. The
labels agree with the branch the expert actually takes on 200 of 200 random states.

| Phase | Share of expert steps |
|---|---:|
| hover | 0.085 |
| descend | 0.159 |
| push | 0.757 |

## The headline result: minimising the imitation loss destroys the controller

Development success is strongly non-monotonic in the number of behavioural-cloning gradient steps,
while validation loss falls monotonically the whole way. Two structures, three seeds each, scored on
all 50 development cases. Standard deviation across seeds is in brackets.

| Steps | `simple_reach_push` success | `simple_damped_push` success | Validation loss |
|---:|---:|---:|---:|
| 250 | 0.000 (0.000) | 0.553 (0.025) | 0.268 / 0.200 |
| 500 | 0.007 (0.009) | 0.553 (0.019) | 0.191 / 0.131 |
| 750 | 0.820 (0.016) | 0.633 (0.025) | 0.135 / 0.091 |
| **1000** | **1.000 (0.000)** | **0.927 (0.009)** | 0.095 / 0.072 |
| 1250 | 0.820 (0.000) | 0.727 (0.034) | 0.077 / 0.064 |
| 1500 | 0.700 (0.000) | 0.613 (0.019) | 0.071 / 0.060 |
| 2000 | 0.053 (0.034) | 0.520 (0.016) | 0.066 / 0.056 |
| 3000 | 0.000 (0.000) | 0.000 (0.000) | 0.060 / 0.043 |

At the loss minimum both structures score **zero**. At roughly 2.4 times that loss one of them
matches the scripted expert on every case. The pipeline's fixed 2000-step budget sits well past the
peak, which is a large part of why the search has produced nothing that works.

`simple_reach_push` at 1000 steps reaches 1.000 success on all 50 development cases across three
seeds, with mean return 1138 against the scripted expert's 1063. Paired against the expert on the
same cases the success difference is +0.000 with a 95% interval of [0.000, 0.000]. This is a
development result on a hand-written structure; the final-test manifest has not been touched.

### The residual-scale result was a budget artifact

The objective comparison initially showed residual-scale fitting reaching 0.633 mean success against
the deterministic baseline's 0.167. A control run settles what caused it: plain regression at 1000
steps reproduces the residual-scale numbers **exactly**, to every reported digit, on both
structures. Residual scale spends half its budget fitting the mean and half fitting the scale, and
deterministic execution ignores the scale, so its stage two cannot change the executed action at
all. The entire apparent gain was the halved mean-fitting budget.

Recorded because the first reading was wrong and the correction is the finding: the objective did
nothing, the budget did everything.

### What was changed in response

`behavioral_cloning` now accepts a rollout probe and keeps the best-scoring parameters, and the
orchestrator supplies one that scores the screening cases four times during fitting. Selecting on
loss cannot work here, because loss improves monotonically through the collapse. The probe ranks on
success first and mean signed goal progress second, matching how the promotion policy ranks
candidates, and prefers the later checkpoint on an exact tie so a generation where nothing succeeds
does not throw away fitting for nothing.

## Objective comparison (E7)

Five objectives, 2000 gradient steps each, three simple-family structures, ten development cases.

| Objective | Likelihood | Own execution | Deterministic | Validation MSE |
|---|---|---:|---:|---:|
| deterministic_mse | no | 0.167 | 0.167 | 0.0702 |
| legacy_mse_std_penalty | no | 0.167 | 0.167 | 0.0685 |
| tanh_gaussian_nll | yes | 0.133 | 0.133 | 0.0697 |
| residual_scale | no | 0.633 | 0.633 | 0.0888 |
| censored_gaussian | yes | 0.200 | 0.167 | 0.0694 |

Read with the control above: the residual-scale row is a budget effect, not an objective effect.
Nothing here shows a distributional objective beating the deterministic baseline at a matched
mean-fitting budget. The censored Gaussian is the only one that scores differently under its own
execution than under the deterministic one, which is what having a genuinely different contract
means.

## Sampling and learner-state imitation (E8)

| Comparison | Mean success | Validation loss |
|---|---:|---:|
| Uniform minibatches | 0.167 | 0.0702 |
| Phase-balanced minibatches | 0.300 | 0.0847 |
| Fixed expert buffer | 0.167 | |
| Learner-state aggregation | 0.333 | |

Learner-state aggregation cost 6750 expert queries and 6750 environment steps across three
structures. Both interventions raise success while raising the loss, consistent with everything
above.

### Is the scripted expert a valid recovery controller? (open decision 6)

No, not reliably. Measured by branching the simulator at a learner-visited state, applying the
expert's own label, measuring the distance to the position the expert is steering toward, and
rewinding.

| Structure | Labels that helped | hover | descend |
|---|---:|---:|---:|
| simple_damped_push | 65 of 90 | 0.850 | 0.407 |
| simple_reach_push | 60 of 90 | 0.778 | 0.407 |
| simple_standoff | 55 of 90 | 0.583 | 0.643 |

About a third of the expert's labels fail to reduce even the error the expert itself is trying to
reduce, and the descend phase is the worst. Aggregated data is therefore partly mislabelled, which
bounds how much learner-state imitation can be expected to deliver.

This measurement needed three corrections before it meant anything, recorded under unexpected
findings.

## Diagnostic intervention and repair (Phase 6, E6)

### A recorded gate was not a lever

`record_gate` detached its argument and discarded the return value, so a policy could report a gate
it never used. Forcing such a gate would change nothing, and that null result reads as evidence the
gate is not at fault. The gate is now a real intervention point: `record_gate` returns the forced
value when an override is set, and `gate_override_is_effective` verifies on the policy at hand that
forcing it changes the executed action before any explanation is built on it.

### The first diagnosis separated two accounts of the same failure

`simple_standoff` fitted with deterministic MSE at 2000 evaluations scores 0.000 on the ten
screening cases, with eight failing as `never_reached_object` at a closest approach of 0.063 m
against a 0.05 m success radius.

| Explanation | Substitution | `min_tcp_object_distance` | Verdict |
|---|---|---|---|
| `approach_control` | expert drives the first 40 steps | 0.0633 -> 0.0285 | supported |
| `vertical_axis` | expert drives `dz` only | 0.0633 -> 0.0643 | rejected |

Both ran on the same ten cases through the same runner as the control, so the comparison is paired.
Both are stored; a record of only the supported hypothesis would make the search look better
informed than it was.

### Fitting the suspected component harder is the wrong repair

The first repair operator refits the in-scope parameters toward the expert on the phases the
component acts in. On `simple_standoff` it moved `gain` and `height_offset` and left the closest
approach unchanged, on both the fixed expert buffer and on learner-state data collected by
aggregation.

That is the correct outcome for the wrong tool. The current parameter values *are* what fitting to
the expert produces, so more of the same cannot move them. The second operator searches the same
bounded parameter set against the metric the explanation predicted.

| Quantity | Before | After |
|---|---:|---:|
| `min_tcp_object_distance` on the eight cluster cases | 0.0633 | 0.0603 |
| Screening success, ten cases | 0.000 | 0.100 |
| Held-out success, twenty cases | 0.000 | 0.050 |
| Mean signed goal progress, the two protected cases | -0.0923 | -0.0531 |

Three of the five parameters were in scope and only `gain` moved. The held-out cases were scored
after the decision and took no part in it. Both operators are kept: which one applies depends on
whether the diagnosis blames a fitting failure or a geometry term.

### A regression threshold nothing can fail is not a check

Protected-case regression is defined on success. A policy scoring 0.000 has no successful case, so
no repair can fail the threshold and every repair passes it. `accept_repair` now detects the vacuous
case, names the cases outside the cluster as the protected set, and holds them to a mean
signed-goal-progress check instead. This will keep mattering while the incumbent scores 0.000.

## The repair ablation (AC-6, E6, E9)

Five arms from the identical fitted policy, 200 development-screen episodes each, three structures
(`simple_reach_push`, `simple_damped_push`, `simple_standoff`) at two seeds. Scored on twenty
held-out development cases that took no part in any arm's decisions. The arms differ only in what
they spend the budget on.

| Arm | Held-out success | Paired against `full` | 95% interval | w/l/t |
|---|---:|---:|---|---:|
| `optimizer_only` | 0.650 | +0.133 | [+0.008, +0.259] | 4/1/1 |
| `full` | 0.517 | | | |
| `diagnostics_only` | 0.500 | -0.017 | [-0.095, +0.062] | 1/1/4 |
| `evaluation_only` | 0.433 | -0.083 | [-0.173, +0.007] | 0/3/3 |
| `global_search` | 0.367 | -0.150 | [-0.312, +0.012] | 1/4/1 |

**The central research hypothesis is not supported at this budget.** AC-6 asks that the full repair
loop beat evaluation-only, diagnostics-only and optimizer-only at matched cost. It beats neither
optimizer-only nor diagnostics-only, and its advantage over the two blind arms has an interval that
includes zero across six runs.

**The controlled substitution is the part that does not pay.** It costs 20 of the 200 episodes and
changed the chosen component in only two of six replicates. In one of those two it helped
(`simple_reach_push` seed 0, 0.650 against 0.450, where the substitution rejected `push_geometry` in
favour of `planar_axes`); in the other the arms tied. In the remaining four replicates both arms
chose `approach_control`, so the substitution budget bought a confirmation.

That is a real finding rather than a null one. The competing explanations are a fixed table keyed on
the failure label, and for `never_reached_object` the first entry is already the one the evidence
supports. A diagnostic that only confirms the obvious guess is not worth 10% of a budget. It would
become worth it if the explanations were generated rather than tabulated, or if the first-listed
account were wrong more often.

**Localisation, separately, does look real.** Naming a component beats searching every parameter at
the same cost by 0.150, and beats a component chosen at random by 0.083. Both intervals include
zero over six runs, so this needs more replicates before it is a claim.

**Optimizer-only wins for a reason that has nothing to do with repair.** Its rollout probe recovers
the checkpoint before the behavioural-cloning collapse: `simple_reach_push` goes from 0.000 to 1.000
at both seeds, `simple_damped_push` from 0.450 to 0.950. Every arm begins from the 2000-step policy,
which sits well past the success peak, so the largest gain available on this task is a checkpoint,
not a structure. On `simple_standoff`, where no checkpoint helps, optimizer-only gains nothing and
only the two diagnostic arms move at all.

The comparison is therefore honest but unflattering to its own premise: the biggest lever on this
task is the fitting schedule, and any repair loop measured against it starts a long way behind.

## Visual failure analysis is built; the model decision is what remains (FR-11)

Everything in FR-11 that does not need a checkpoint is implemented and tested: the
`VisualFailureAnalysis` contract, the fixed media-selection rule, capture against the real
simulator, the merge with numerical diagnostics, conflict marking, and the audit metadata. Sixty-five
tests, none of which load a model.

**What the contract refuses.** A claim citing a frame the model was never shown, a claim citing a
timestep outside the frame it names, an analysis with no evidence, and a recommended follow-up naming
a quantity the reports do not measure. The last one matters most: it is what stops a visual account
from being unfalsifiable. An analysis failing any check is dropped and the reason kept, so a model
that hallucinates frames leaves a trace instead of a plausible paragraph.

**What capture found on the real simulator**, ten development cases, three failures each:

| Structure | Frames per episode | Note |
|---|---:|---|
| `simple_standoff` | 4 | no gate, so the gate-transition moment is reported unavailable |
| `simple_reach_push` | 5, 4, 4 | the contact gate crossed on one episode of three |

That the contact gate never opens on two of three failures is a diagnosis in itself, and it arrived
from the media rule rather than from anything visual.

**The decision that is still open (question 2).** Which small Qwen checkpoint and which quantization.
`QwenAnalyst` raises rather than defaulting, because a default chosen here would be the one frozen
into E10 without anyone deciding it. The box is a Turing T4, so fp16 and not bf16.

**E10 needs a second thing, not just a model.** The comparison is repair quality with and without
visual analysis at matched budgets. Today the repair operators are a scoped refit and a scoped metric
search, neither of which reads a prompt. Until a correction-LLM repair path exists, there is nothing
for the evidence block to change, so E10 would compare a model against itself. Building that path is
the prerequisite, and it is the same path the ablation finding argues for: generated rather than
tabulated explanations.

## The final-test split is now locked (FR-4)

FR-4 asks that winner selection and final estimation use different data. Four of its five
requirements were enforced; this one was not enforced at all, and the failure mode it guards against
is silent. Reading a final-test number during a search and then continuing does not raise anything,
leave a trace, or change any output. It only makes the eventual interval meaningless.

`evaluate_manifest` now refuses the `final_test` split outside a `final_test_session`, which needs a
complete `FreezeRecord`. Every look inside a session is appended to a ledger that survives the
session, and the same code hash cannot be scored twice unless the caller passes `allow_rescore=True`,
which is itself written into the entry.

The ledger is the point. The number a selection-effect argument needs is how many times the held-out
data was consulted, and nobody records it. Now it is a file.

The final-test manifest has still never been evaluated. The ledger is empty, which is the correct
state before Phase 7.

## Resource usage

| Operation | Wall time |
|---|---|
| Build one 50-placement MT1 pool | 1.2 s |
| 30 expert episodes, 150 steps | 7.5 s |
| 30 incumbent episodes, 150 steps | 11.1 s |
| Full `lock_baseline` run, 30 cases, three actors, reruns | about 90 s |
| Collecting 150 expert demonstration episodes | about 60 s |
| Full fitting comparison, 18 structures, 4 methods | 1120 s |
| Generation-mode comparison, 6 generations of 4 policies | 291 s |
| One diagnosis, two substitutions and one searched repair | 290 episodes, 350 s |
| Repair ablation, 5 arms x 3 structures x 2 seeds | 6000 episodes, 2350 s |

Phase 2 costs three checkpoint evaluations per candidate on the screening subset, plus one expanded
evaluation for each candidate that clears screening. At ten screening cases, thirty expanded cases
and a population of five, that is roughly 200 episodes per generation.

## Unexpected findings

1. **`DemoBuffer` records no episode boundaries.** Every `done` in the saved buffer is zero, because
   MetaWorld does not terminate early and the 150-step rollout never reaches the 200-step cap. FR-6
   asks for train/validation splits by complete trajectory, which the saved buffer cannot support.
   `DemoBuffer` now records an `episode_id` per transition; buffers written earlier load with blanks.
2. **`timeout_rate` is uninformative here.** MetaWorld's push environment never terminates, so every
   episode hits the horizon and the rate is trivially 1.0. It is reported for schema completeness,
   not as a signal.
3. **The validation subprocess spawned a bare `python`.** That works only when the venv is activated.
   Invoking the interpreter by absolute path made every generated candidate fail with `ENOENT`, which
   presented as a code-generation failure rather than an environment one. Now `sys.executable`.
4. **`action_space.high` is all ones.** After `NormalizedBoxEnv` the action space is exactly
   `[-1, 1]^4`, so the `action_space.high *` scaling that four rollout sites apply is an identity
   multiply. It is kept for explicitness, but it is not why the Isaac path differs.
5. **Validator probes had to be rescaled.** Drawing probe observations from a standard normal puts
   the end-effector and object about 2.4 metres apart. A contact gate with a 0.06 metre threshold
   and a sharpness of 40 then evaluates to exactly zero, so any term behind that gate contributes
   nothing and both the batch-coupling and field-sensitivity checks pass a policy they should
   reject. Probes now run at three scales, and the smallest one opens the gate. This is worth
   remembering for any future check that perturbs an observation: unit scale is far outside the
   state distribution these controllers were designed for.
6. **The observation layout was correct all along, but nothing had checked it.** Comparing the
   declared slices against the simulator's own accessors passes on the first try for `push-v2`.
   The value of the check is that a future edit, or a new task, cannot silently drift.
7. **Imitation loss and task success are positively correlated on this task, which is the wrong
   sign.** See the parameter-fitting comparison above. This is the most consequential finding so far:
   the inner loop optimises a quantity that does not predict the outcome it is judged on. It also
   explains why the earlier search could report large loss improvements alongside a controller that
   pushes the object away from the goal.
8. **Differential evolution reaches a much lower loss than Adam at small budgets and a worse
   controller.** At 200 evaluations on a six-parameter structure it reached 0.096 against the
   baseline's 0.276, and scored 0.067 success against 0.400. At 2000 evaluations on the full bank the
   loss advantage disappears. Small-budget loss comparisons are not informative here.
9. **The same measurement was wrong three times before it meant anything.** The expert recovery
   check first measured the *learner's* step rather than the expert's label; the fix was to branch
   the simulator, apply the label, measure, and rewind. It then measured object-to-goal distance,
   which fails both approach phases by construction since no action can move the object before
   contact. It then measured hand-to-object distance, which fails the hover phase because the expert
   deliberately lifts 0.2 m above the puck. Only the distance to the expert's own desired position
   is definitionally what its action reduces. Each wrong version produced a confident number.
10. **An apparent objective effect was a budget artifact.** Residual-scale fitting appeared to beat
   the deterministic baseline nearly fourfold. A control showed plain regression at half the budget
   reproducing its numbers exactly: stage two fits only the scale, and deterministic execution
   ignores the scale, so it cannot change the action. Worth remembering that a two-stage method
   silently halves the budget of stage one.
11. **Sensitivity probes need real observations.** Probing dead parameters on an all-zero batch put
   the end-effector, object and goal at the same point, which made every direction term read as
   dead and would have told the generator to delete the controller. The probe now samples the
   demonstration buffer. This is the same class of mistake as the unit-scale probe in Phase 1: an
   artificial observation is outside the state distribution these controllers were designed for.

## Questions requiring a research decision

1. Section 2.3's reliability target asks that 95% of candidates compile and pass validation after at
   most one repair. The current pipeline allows three repairs per candidate
   (`max_repair_per_candidate=3`). Tighten the code to one, or restate the target?
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

## Paths to detailed artifacts

| Artifact | Path |
|---|---|
| Committed manifests | `config/manifests/push-v2_{train,development,final_test}.yaml` |
| Baseline lock reports | `logs/baseline_lock/baseline_lock_<timestamp>.json` |
| Per-baseline evaluation reports | `logs/baseline_lock/report_<actor>.json` |
| Expert buffer with episode ids | `logs/baseline_lock/demo_push-v2.pkl` |
| Per-candidate reports from a search run | `<log_dir>/gen_<N>/reports/<candidate>_<checkpoint>.json` |
| Frozen structure bank | `lares/fitting/structures/*.py` |
| Fitting comparison reports | `logs/fitting_benchmark/fitting_benchmark_<timestamp>.json` |
| Generation-mode comparison | `logs/generation_modes/<timestamp>/comparison.json` |
| Experiment records from a search run | `<log_dir>/experiments/*.json` |
| Evidence block shown to the generator | `<log_dir>/gen_<N>/feedback.md` |
| Screening decisions from a search run | `<log_dir>/gen_<N>/screening.json` |
| Saved incumbent source and weights | `logs/evolution/best_policy_code.py`, `logs/evolution/best_policy.pt` |
| Expert demo buffer | `logs/evolution/demo_push-v2.pkl` |
| Historical generation results | `logs/evolution/gen_{0,1,2}/results.pkl` |
| Historical training dynamics | `logs/evolution/*.jsonl` |
| Repair studies, with rejected explanations | `logs/repair_studies/<structure>_<timestamp>.json` |
| Experiment records from a repair study | `logs/repair_studies/experiments/*.json` |
| The AC-6 ablation | `logs/repair_ablation/ablation_20260909_195923.json` (the 200-episode run; the two 60-episode files are smoke runs and are not pooled with it) |
| Selected frames and merged visual evidence | `logs/vision/<structure>/{media,analyses,evidence.md,run.json}` |
| Freeze record and the final-test look-ledger | `logs/final/freeze.json`, `logs/final/final_test_ledger.jsonl` (neither exists yet, which is correct) |

Everything under `logs/` is gitignored and lives only on this box. `config/manifests/`
is the exception and is committed, because it defines the evaluation contract.
