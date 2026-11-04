# Research Handoff

Evidence log for the symbolic-policy evolution pipeline. Updated as phases land.

> **From 2026-10-13** new findings arrive here only as decision blocks appended by
> `python scripts/research.py decide --handoff`. The structured record of every idea, its predeclared
> protocol, result and both verdicts is `research/ideas/*.json` plus `research/decisions.jsonl`;
> `research/STATE.md` is the generated one-page summary. The open-question list at the end of this
> file was backfilled into ideas R-001 to R-004 and is not extended further.

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

> **Pilot, not the AC-6 test.** Relabelled by `debug_spec.md` (D10). The arms mixed unequal fitting
> and search budgets and none exercised a correction-LLM repair. The `optimizer_only` row's interval
> below is a normal 1.96 x SE interval; over six paired runs the t interval is about
> [-0.031, +0.298], which includes zero. Read this table as direction, not significance. The clean
> ablation, with common preprocessing and matched budgets, is Stage F of `debug_spec.md`.

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

## Debug-cycle runs: the improvement is real, and it is not fitting (2026-09-13)

After `debug_spec.md` Stages A–C, a live two-generation smoke test passed every Stage D check
(`logs/debug_smoke_two_generation`), and a full run with `config/run_full_evolution.yaml` (gpt-5, 3 x 5,
deterministic MSE, fixed scale, checkpoints including step 0) completed with complete provenance
(`logs/evolution_debug_full`). Details and every number are in `DEBUG_PROGRESS.md`.

**Held-out, on 50 fresh placements (MT1 seed 2100, disjoint from all committed splits):** the new incumbent
scores 0.340 against the pre-debug best's 0.120, paired +0.220, t95 [+0.045, +0.395]. The development figures
(0.567 against 0.267) were optimistic in both runs, as expected from selecting on those cases.

**Why it improved.** Every promoted candidate kept its untrained initial values (selected step 0). The new
incumbent scores 0.340 untrained and 0.000 after 2000 behavioural-cloning steps; the pre-debug best scores
0.000 untrained and 0.120 fitted. The structured idea prompt, which asks for a stated zero-shot behaviour
within a 6–10 parameter budget, produced controllers that work before fitting, and a checkpoint rule that
probes step 0 let the run keep them. This extends the Phase 5 headline: fitting to the scripted expert does
not merely fail to help these controllers, it destroys them.

**What did not help.** Generations 1–2 added nothing held-out: the first-generation best scores 0.400 against
the final incumbent's 0.340 (paired -0.060, t95 [-0.194, +0.074]). The development "improvement" from 0.500
to 0.567 came from one promotion on a tie in success with higher return and one on a +0.067 difference whose
interval included zero. Bounded repairs frequently produced different code with identical per-case outcomes.

## Stage E: fitting is disabled by default, and the family split is the real finding (2026-09-14)

Run `logs/stage_e/20260914_032500` at commit `f96f060`, frozen protocol `config/stage_e/stage_e.yaml`, 50
confirmation placements from MT1 seed 2200, checkpoints selected on development cases 40-49, 8 structures x 5
seeds. Full tables in `DEBUG_PROGRESS.md`.

Against zero-shot, structure-level: fixed 2000 steps -0.267 [-0.524, -0.010]; rollout-selected MSE +0.146
[-0.098, +0.389]; phase-balanced +0.147 [-0.105, +0.399]. No interval lies above zero, so by the frozen rule
fitting is disabled by default.

The pooled number conceals two opposite families. On the five evolved structures (all descendants of
`gen0_cand0`) rollout selection always returns step 0 and fixed fitting drives success from about 0.40 to 0.000.
On the hand-written bank structures rollout-selected fitting takes `simple_damped_push` from 0.120 to 0.908
(0.952 phase-balanced) and `simple_reach_push` from 0.580 to 0.956: the best held-out policies in the project,
more than double the best evolved policy's 0.400. `simple_standoff` stays at zero whatever is done.

## Next: a fittable structure bank before any further search (decided 2026-09-14)

Stage E's family split led to three decisions, recorded in full in `DEBUG_PROGRESS.md`: rollout-selected
fitting is allowed only in a fittable search lane (checkpoints {0, 500, 1000}, step 0 always eligible, selection
on a probe set, evaluation on a separate screen, promotion on independent paired evidence); step-0 strength and
fittability are classified separately, with separate `performance_incumbent` and `fittable_incumbent`; and the
hand-written bank is expanded from three to nine with six predeclared single-change variants before the next
search. The qualification study (`scripts/qualify_fittable_bank.py`, `config/fittable_bank/qualification.yaml`)
admits a structure to the fittable seed bank if fitting improves held-out success in at least three of five
seeds, never regresses by more than 0.05, and has a positive mean paired difference.

### Result: five structures qualify for the fittable seed bank

Run `logs/fittable_bank/20260914_040245` at `707322a`, evaluation on MT1 seed 2300: `simple_reach_push_sharp_gate`
(0.56 zero-shot to 0.98 fitted), `simple_damped_push` (0.12 to 0.87), `simple_reach_push` (0.60 to 0.92),
`simple_reach_push_damped` (0.62 to 0.93) and `simple_reach_push_shared_gain` (0.62 to 0.82) improved in 5/5
seeds. All three standoff structures score zero untrained and fitted, and smoothing the damped-push gates broke
that structure outright. The five fitting seeds vary only minibatch order and produced near-identical results, so
the qualification speaks to insensitivity to minibatch order, not to robustness across placements or data splits.

## LLM model: gpt-5.6-sol, reasoning effort high (decided 2026-09-14)

Every LLM call now uses gpt-5.6-sol with `reasoning_effort: high` (configs, the generation-mode comparison
and Tier 3). All generation results recorded above were produced with gpt-5 or gpt-4o-mini, so they are
evidence about those models. In particular the next search's token cap was matched to the gpt-5
debug-cycle run, and reasoning tokens count against it: a comparison with that run would confound the
model change with the search change. Whether to re-match the budget by calls only, or to rerun a baseline
under the new model, is open.

## Two-lane search: built and mock-tested, not run (2026-09-14)

The two-lane search exists (`lares/search/lanes.py`, `config/two_lane_search.yaml`) and has been exercised only
against a scripted LLM in a kinematic push world. Nothing about its value is known yet. Order of work: run the
nine-candidate zero-shot baseline under gpt-5.6-sol high (`config/run_full_evolution.yaml`), match the two-lane
token budget to its ledger, then run the two-lane search on the same manifests. Held-out comparison should use a
fresh MT1 pool seed, as for the debug-cycle run.

Open design questions this build had to answer provisionally: the scope boundaries (`gate_control`, `geometry`,
`single_bounded_change`), the bound of three parameter changes and 50% kept, and whether a fittable incumbent
must also beat the previous fittable incumbent (implemented: yes). The standoff family and the smooth-gate
damped-push variant are now controls or retired (`structure_roles.yaml`), and future qualification replicates
vary the data (`qualification_v2.yaml`, not run).

## Baseline under gpt-5.6-sol high, and Stage F built (2026-09-23)

The nine-candidate zero-shot baseline ran: every slot valid on its first attempt, 10 calls, 100,786 tokens. Its
best candidate scores 0.333 on the 30-case promotion set (paired against the 0.00 null, t95 [+0.154, +0.512]),
still failing mostly by never reaching the object. It has not been compared with the gpt-5 run on shared
held-out cases; that needs a fresh MT1 pool seed. Stage F's correction-LLM repair ablation is implemented and
mock-tested (`lares/repair/correction.py`, `scripts/stage_f.py`) and waits on the choice of structures, seeds
and held-out pool seed. Details in `DEBUG_PROGRESS.md`.

## Two-lane search, first live run (2026-10-13, gpt-6.1-sol)

All six reference-seeded bounded mutations fitted to step 1000 and screened at 0.6-1.0; the promoted child of
`simple_reach_push_damped` (damping on a blended hand/object velocity) scored 30/30 on the promotion set and beat
its own step 0 with an interval above zero. Unseeded explorations reached at most 0.40. Whether any mutation beats
its *fitted parent* is untested, and is the question that decides whether the fittable lane's LLM step adds
anything over fitting the seed bank. Details in `DEBUG_PROGRESS.md`.

## Baseline rerun under gpt-6.1-sol: zero success (2026-10-13)

Nine unseeded zero-shot candidates under gpt-6.1-sol all scored 0.000 on the screen and on the promotion set
(gpt-5.6-sol's best was 0.333). One run per model cannot separate model from sampling noise, but together with the
two-lane run (seeded fittable children 0.6-1.0, unseeded explorations at most 0.40) it says the pipeline's strength
on push-v2 comes from reference-seeded fitting, not from unseeded zero-shot ideas.

## Stage F protocol decided (2026-10-13)

Three structures (`simple_damped_push` and the top two gpt-6.1-sol baseline candidates) x three seeds (2500-2502),
all six arms, 60 repair-evaluation episodes per arm, held-out cases from MT1 pool seed 2500. Frozen in
`config/stage_f/stage_f.yaml`; details in `DEBUG_PROGRESS.md`.

## Stage F: correction-LLM repair does not beat no repair (2026-10-13)

Nine replicates, six arms at matched repair-evaluation episodes, held-out pool seed 2500. No arm beats no repair
with an interval excluding zero. Parameter search leads (+0.042, local and global) at no LLM cost; the LLM arms
are flat (-0.004, +0.000, +0.011). Most LLM children were worse than their parent on the screen after the same
refit, and the acceptance rule kept the parent. The intervention evidence produced the only LLM wins and the best
prediction rate (5/9), too few to call. This is the clean version of the pilot's finding: on push-v2 the repair
loop's structural edits do not pay for themselves. Details in `DEBUG_PROGRESS.md`.

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

## A bounded LLM mutation beats its own fitted parent at matched fitting (absorbs R-001) (D-001, 2026-10-22)

**Idea R-005.** Averaged over the six frozen fittable-lane children of logs/two_lane_search, a child fitted and rollout-selected exactly as its parent wins more fresh held-out episodes than the parent, because its one structural change adds control that fitting the parent's own parameters cannot reach.
**Prediction.** paired_success_difference increase by at least 0.05 against each child's own parent, refitted with identical demonstrations, objective (deterministic_mse), budget 1000, checkpoints {0,500,1000}, probe cases (development 40-49) and fitting seed 2611; checkpoints selected by probe rollouts, never by loss.
**Protocol.** `python scripts/compare_child_vs_parent.py --run logs/two_lane_search --baseline logs/baseline_gpt61_sol_high --pool-seed 2610 --fit-seed 2611 --out-dir logs/child_vs_parent_2610` at git `f7e9f74e92`.
**Result.** {"paired_success_difference": -0.02, "paired_success_difference_t95_low": -0.04645036412582304, "paired_success_difference_t95_high": 0.006450364125823036, "paired_success_difference_p_value": 0.1350617842971326, "mean_child_success": 0.89, "mean_parent_success": 0.9100000000000001, "per_pair_success_difference": {"gen0_cand0_a0-simple_reach_push": -0.16, "gen0_cand1_a0-simple_reach_push_damped": 0.0, "gen0_cand2_a0-simple_reach_push_shared_gain": -0.08, "gen0_cand3_a0-simple_reach_push_sharp_gate": 0.0, "gen0_cand4_a0-simple_damped_push": -0.02, "gen0_cand5_a0-simple_damped_push": 0.14}, "per_pair_wins_losses_ties": {"gen0_cand0_a0-simple_reach_push": [0, 8, 42], "gen0_cand1_a0-simple_reach_push_damped": [0, 0, 50], "gen0_cand2_a0-simple_reach_push_shared_gain": [0, 4, 46], "gen0_cand3_a0-simple_reach_push_sharp_gate": [0, 0, 50], "gen0_cand4_a0-simple_damped_push": [0, 1, 49], "gen0_cand5_a0-simple_damped_push": [7, 0, 43]}, "r001_paired_success_difference": 0.98, "r001_t95": [0.9398084952574152, 1.0201915047425847], "r001_wins_losses_ties": [49, 0, 1], "success_by_policy": {"gen0_cand0_a0": 0.82, "simple_reach_push": 0.98, "simple_reach_push_damped": 0.98, "gen0_cand3_a0": 1.0, "gen0_cand1_a0": 0.98, "gen0_cand2_a0": 0.7, "gen0_cand4_a0": 0.84, "gen0_cand5_a0": 1.0, "baseline:gen0_cand2_a1": 0.0, "simple_reach_push_shared_gain": 0.78, "simple_reach_push_sharp_gate": 1.0, "simple_damped_push": 0.86}, "selected_steps": {"gen0_cand0_a0": 1000, "simple_reach_push": 1000, "simple_reach_push_damped": 1000, "gen0_cand3_a0": 1000, "gen0_cand1_a0": 1000, "gen0_cand2_a0": 1000, "gen0_cand4_a0": 1000, "gen0_cand5_a0": 1000, "baseline:gen0_cand2_a1": null, "simple_reach_push_shared_gain": 1000, "simple_reach_push_sharp_gate": 1000, "simple_damped_push": 1000}, "resources": {"probe_episodes": 330, "evaluation_episodes": 600, "bc_gradient_steps": 11000}}; controls {"paired_success_difference": 0.0, "note": "control is each child's own parent refitted identically; the metric is already child minus parent, so the parent-vs-itself control is 0 by construction", "parent_success": {"simple_reach_push": 0.98, "simple_reach_push_damped": 0.98, "simple_reach_push_shared_gain": 0.78, "simple_reach_push_sharp_gate": 1.0, "simple_damped_push": 0.86}}.
**Decision: refuted**. R-005 refitted each of the six fittable-lane children of logs/two_lane_search and its seed-bank parent identically (fit seed 2611, same demos, deterministic_mse, budget 1000, checkpoints {0,500,1000} probe-selected on development 40-49) and scored all on 50 fresh cases (pool seed 2610). Child minus parent, averaged over pairs per placement: -0.020, t95 [-0.046, +0.006]; the predicted +0.05 is refuted, and the direction is inconclusive. Per pair: the geometry-scope mutation of simple_damped_push gained +0.14 (7/0/43), cand0 lost -0.16, two children tied their parents on every case. All 11 fits selected step 1000. The two-lane winner beats the zero-success gpt-6.1-sol baseline by +0.98 [+0.94, +1.02] (R-001), but it ties its own fitted parent, so that gain is the seed bank plus rollout-selected fitting, not the LLM mutation.
**Next.** Treat seed bank + rollout-selected fitting as the operative method; test the LLM mutation step only where it showed a signal (geometry scope), or redirect LLM budget to repair with richer evidence.

## Replicate bundles that vary the data give independent fitting replicates (D-002, 2026-10-22)

**Idea R-004.** Varying the demonstrations, the trajectory-level split, the probe manifest and the fitting seed together (qualification_v2 bundles) increases between-replicate uncertainty in the five seed-bank structures' fitted success relative to varying only the minibatch seed on one fixed bundle, so minibatch-only intervals understate uncertainty.
**Prediction.** success_se increase by at least 0.01 against matched minibatch-only control arm: one fixed demo sample/split (seed 2700) and probe manifest (pool seed 2710), fitting seeds 2701-2705; replicate r of both arms is scored on the same qualification manifest (pool seed 2420+r).
**Protocol.** `python scripts/qualify_fittable_bank.py --config config/fittable_bank/qualification_v2.yaml replication --out-dir logs/fittable_bank_v2_replication` at git `984acef550`.
**Result.** {"success_se": 0.007483314773547877, "success_se_difference": 0.001911959462674244, "success_se_difference_t95_bootstrap": [-0.0034355632888638586, 0.005421620490562363], "bank_mean_success_data_varying": 0.8959999999999999, "bank_mean_by_replicate": {"data_varying": [0.9199999999999999, 0.8879999999999999, 0.876, 0.9039999999999999, 0.8919999999999998], "minibatch_only": [0.9, 0.9279999999999999, 0.9199999999999999, 0.9079999999999998, 0.9], "step0": [0.536, 0.488, 0.44800000000000006, 0.532, 0.5840000000000001]}, "per_structure": {"simple_damped_push": {"sd_data_varying": 0.036331804249169895, "sd_minibatch_only": 0.03033150177620617, "mean_data_varying": 0.892, "mean_minibatch_only": 0.908, "mean_step0": 0.10800000000000001, "selected_steps": {"data_varying": [1000, 1000, 1000, 1000, 1000], "minibatch_only": [1000, 1000, 1000, 1000, 1000]}}, "simple_reach_push": {"sd_data_varying": 0.02607680962081058, "sd_minibatch_only": 0.016733200530681527, "mean_data_varying": 0.9560000000000001, "mean_minibatch_only": 0.9559999999999998, "mean_step0": 0.604, "selected_steps": {"data_varying": [1000, 1000, 1000, 1000, 1000], "minibatch_only": [1000, 1000, 1000, 1000, 1000]}}, "simple_reach_push_damped": {"sd_data_varying": 0.020000000000000018, "sd_minibatch_only": 0.014142135623730963, "mean_data_varying": 0.96, "mean_minibatch_only": 0.96, "mean_step0": 0.632, "selected_steps": {"data_varying": [1000, 1000, 1000, 1000, 1000], "minibatch_only": [1000, 1000, 1000, 1000, 1000]}}, "simple_reach_push_shared_gain": {"sd_data_varying": 0.08831760866327845, "sd_minibatch_only": 0.0374165738677394, "mean_data_varying": 0.6799999999999999, "mean_minibatch_only": 0.74, "mean_step0": 0.632, "selected_steps": {"data_varying": [1000, 0, 0, 1000, 1000], "minibatch_only": [1000, 1000, 1000, 1000, 1000]}}, "simple_reach_push_sharp_gate": {"sd_data_varying": 0.010954451150103331, "sd_minibatch_only": 0.010954451150103331, "mean_data_varying": 0.992, "mean_minibatch_only": 0.992, "mean_step0": 0.612, "selected_steps": {"data_varying": [1000, 1000, 1000, 1000, 1000], "minibatch_only": [1000, 1000, 1000, 1000, 1000]}}}, "resources": {"probe_episodes": 1500, "scoring_episodes": 3750}}; controls {"success_se": 0.005571355310873633, "bank_mean_success_minibatch_only": 0.9112}.
**Decision: refuted**. R-004 fitted the five seed-bank structures under five data-varying bundles (own demo sample, trajectory split, probe manifest and fitting seed; seeds 2400-2404, 2410-2414) and under a matched minibatch-only control (fixed sample/split 2700, probe 2710, fitting seeds 2701-2705), replicate r of both arms scored on the same qualification manifest (pool seed 2420+r). Between-replicate SE of bank-mean success: 0.0075 vs 0.0056, difference +0.0019, paired bootstrap 95% [-0.0034, +0.0054]; the predicted +0.01 is refuted and the direction is inconclusive. Bank-mean success 0.896 vs 0.911. Minibatch-only qualification did not materially understate bank-level uncertainty. One exception: simple_reach_push_shared_gain rejected fitting (step 0 selected) under 2 of 5 data draws, so its fitted success is data-sensitive (sd 0.088 vs 0.037).
**Next.** Keep the seed bank as is; treat simple_reach_push_shared_gain as data-sensitive when reading single-run results.

## Vision evidence from a local Qwen VLM makes correction-LLM repair beat numeric-only repair (development-only E10 pilot) (D-003, 2026-10-22)

**Idea R-006.** Adding a validated visual failure analysis (Qwen VL on the five rule-selected frames of the parent's worst <=3 failed screen episodes, merged with the numbers by lares.vision.merge) to the Stage F numeric evidence packet lets the correction LLM localise the failing component, so its repairs win more held-out episodes than the identical numeric-only arm at the same repair-episode cap.
**Prediction.** paired_success_difference increase by at least 0.05 against arm llm_numeric in the same cell: same structure, same independent data/split/probe bundle and fitting seed, same common preprocessing, acceptance rule, repair-episode cap and held-out cases (pool seed 2629); no_repair reported as a second control.
**Protocol.** `python scripts/stage_f.py run --config config/stage_f_vision/stage_f_vision.yaml` at git `a76cc5464c`.
**Result.** {"paired_success_difference": 0.0, "paired_success_difference_t95": [0.0, 0.0], "wins_losses_ties": [0, 0, 9], "held_out_case_bootstrap": [0.0, 0.0], "vision_minus_no_repair": 0.0, "held_out_success": {"no_repair": 0.44000000000000006, "llm_numeric": 0.44000000000000006, "llm_numeric_vision": 0.44000000000000006}, "outcomes": {"no_repair": {"kept_parent": 9}, "llm_numeric": {"kept_parent": 9}, "llm_numeric_vision": {"kept_parent": 9}}, "mean_cost": {"no_repair": {"repair_episodes": 0.0, "probe_episodes": 0.0, "expert_assisted_episodes": 0.0, "bc_gradient_steps": 0.0, "llm_calls": 0.0, "llm_total_tokens": 0.0}, "llm_numeric": {"repair_episodes": 10.0, "probe_episodes": 30.0, "expert_assisted_episodes": 0.0, "bc_gradient_steps": 1000.0, "llm_calls": 2.0, "llm_total_tokens": 13279.111111111111}, "llm_numeric_vision": {"repair_episodes": 12.555555555555555, "probe_episodes": 30.0, "expert_assisted_episodes": 0.0, "bc_gradient_steps": 1000.0, "llm_calls": 2.0, "llm_total_tokens": 13992.666666666666}}, "vision_packets": {"cells": 9, "with_valid_analysis": 9, "analyses_returned": 23, "analyses_rejected": 0, "conflicts": 3, "episodes_rerun": 23}, "child_screen_success_mean": {"llm_numeric": 0.3111, "llm_numeric_vision": 0.3222}, "child_screen_success_paired_vision_minus_numeric": 0.0111}; controls {"paired_success_difference": 0.0, "note": "control is arm llm_numeric in the same cell; the metric is already vision minus numeric"}.
**Decision: refuted**. R-006 added a local VLM evidence arm to correction-LLM repair: Qwen2.5-VL-7B-Instruct (pinned cc59489, NF4 language weights, fp16 compute, 7.55 GiB on the T4) analysed the parent's worst failed screen episodes (five rule-selected, upright frames each) and its validated, geometry-merged analysis was appended to the Stage F numeric packet. Nine independent cells (three frozen Stage F structures x bundles 2620-2628), held-out pool 2629, arms no_repair / llm_numeric / llm_numeric_vision at matched repair-episode caps. Vision minus numeric held-out success: +0.000 [+0.000, +0.000], 0/0/9; all arms 0.440. Every packet carried valid analyses (23 returned, 0 rejected, 3 stage conflicts), but all 18 LLM children scored at or below their refitted parent on the screen and were rejected, so neither evidence level changed any outcome. Predicted +0.05 refuted; localisation benefit untested because no edit survived refit. With Stage F and R-005, the correction LLM's structural edits, not its evidence, are the bottleneck.
**Next.** Do not add more evidence to correction-LLM repair; test whether any LLM edit can survive refit, or redirect effort to parameter search.

## Measure false rejection from a ten-case screen (D-004, 2026-10-22)

**Idea R-007.** A thirty-case screen preserves beneficial frozen edits that a ten-case screen rejects through sampling ties.
**Prediction.** paired_success_difference increase by at least 0.02 against Ten-case screening of the same frozen, identically refitted parent-child pair.
**Protocol.** `python scripts/compare_child_vs_parent.py --run logs/two_lane_search --research-protocol nested-screen --child gen0_cand5_a0 --screen-seed 2800 --heldout-seed 2801 2805 --fit-seed 2802 --replay-seed 2803 --probe-seed 2804 --out-dir logs/research/nested_screen_s2800` at git `401ec7c33b`.
**Result.** {"paired_success_difference": 0.004400000000000004, "paired_success_difference_t95": [0.0019041246290080298, 0.006895875370991978], "ten_case_acceptance_rate": 0.96, "ten_case_branches": {"accepted": 960, "child_lower_success": 0, "tied_success_lower_or_equal_progress": 40}, "full_screen_decision": "accepted", "refitted_child_minus_parent_confirmation": 0.11, "refitted_child_minus_parent_t95": [0.04760311572520071, 0.1723968842747993], "refitted_child_minus_parent_wlt": [11, 0, 89], "screen_success": {"child": 1.0, "parent": 0.8666666666666667}, "confirmation_success": {"child": 1.0, "parent": 0.89}, "selected_steps": {"child": 1000, "parent": 1000}, "resources": {"probe_episodes": 60, "screen_episodes": 60, "confirmation_episodes": 200}}; controls {"paired_success_difference": 0.0, "note": "control is ten-case screening of the same refitted pair; the metric is already full-screen minus ten-case selection"}.
**Decision: refuted**. R-007 refitted the frozen geometry-scope child gen0_cand5_a0 and its parent simple_damped_push identically (fit seed 2802, probe s2804), scored both on a fresh 30-case screen (s2800) and a 100-case confirmation set (s2801+s2805), and replayed Stage F's acceptance rule on 1000 ten-case subsets. A 30-case screen selects a better controller than a 10-case screen by only +0.0044 [+0.0019, +0.0069]: the ten-case screen keeps the child 96% of the time, rejecting it only on tied success. The predicted +0.02 is refuted; screen size is not what blocks good edits. The refitted child beats its parent by +0.110 [+0.048, +0.172], 11/0/89, replicating R-005's +0.14: this one LLM geometry mutation is a real improvement.
**Next.** Consider admitting gen0_cand5_a0 to the fittable seed bank via the qualification rule; investigate why geometry-scope mutation works where correction-LLM repair does not.

## Replace imitation with direct search on weak evolved structures (D-005, 2026-10-22)

**Idea R-008.** Direct rollout search escapes poor imitation trajectories in weak evolved structures by optimizing their existing parameters for task success.
**Prediction.** paired_success_difference increase by at least 0.1 against Rollout-selected imitation fitting alone at an equal selection-episode budget.
**Protocol.** `python scripts/stage_f.py search-vs-fit --config config/stage_f/stage_f.yaml --probe-seed 2810 --heldout-seed 2811 --optimizer-seed 2812 --split-seed 2813 --out-dir logs/research/search_replaces_fitting_s2810` at git `fde22ef24d`.
**Result.** {"paired_success_difference": 0.08, "paired_success_difference_t95": [0.0021165578049845007, 0.1578834421950155], "wins_losses_ties_by_placement": [12, 4, 34], "per_structure": {"baseline61_gen0_cand2_a1": {"search": 0.52, "fitting": 0.0, "difference": 0.52, "search_selected": "global", "fitting_selected_step": 0}, "baseline61_gen0_cand1_a0": {"search": 0.0, "fitting": 0.36, "difference": -0.36, "search_selected": "global", "fitting_selected_step": 125}}, "episodes": {"search_selection": 160, "fitting_selection": 160, "confirmation": 200}}; controls {"paired_success_difference": 0.0, "note": "control is rollout-selected imitation fitting on the same structure, probe and confirmation cases; the metric is already search minus fitting"}.
**Decision: inconclusive**. R-008 compared direct rollout search (original + 3 local + 4 global draws) against rollout-selected imitation (8 checkpoints to 1000 steps), each with 80 probe episodes, from the frozen original parameters of the two weak evolved Stage F structures (seed 2812, split 2813, probe s2810, confirmation s2811). Search minus imitation: +0.080 [+0.002, +0.158]; the predicted +0.10 is neither supported nor refuted. The average hides a complete split: on baseline61_gen0_cand2_a1 one global draw reached 0.52 while imitation never left step 0 (0.00); on baseline61_gen0_cand1_a0 imitation reached 0.36 at step 125 while no search draw succeeded. The two optimisers fail on different structures; the probe agreed with confirmation in both.
**Next.** Test pooling search candidates and imitation checkpoints under one rollout probe at the combined budget, over more weak structures.

## Qualify the geometry child without admitting it as a parent (D-006, 2026-10-22)

**Idea R-010.** The geometry child changes the controller's fitting basin enough to retain its advantage across fitting seeds, making it a credible qualification candidate rather than a single-fit exception.
**Prediction.** paired_success_difference increase by at least 0.05 against simple_damped_push, identically refitted in every replicate.
**Protocol.** `python scripts/compare_child_vs_parent.py --run logs/two_lane_search --research-protocol seed-robustness --child gen0_cand5_a0 --fit-seeds 2900 2901 2902 2903 2904 --probe-seed 2905 --heldout-seed 2906 --out-dir logs/research/geometry_qualification_s2900` at git `a88eb26ef6`.
**Result.** {"paired_success_difference": 0.10400000000000001, "placement_level_t95": [0.02260625387986538, 0.18539374612013465], "seed_level_t95": [0.08322298732632857, 0.12477701267367139], "seed_level_by_seed": {"2900": 0.07999999999999996, "2901": 0.12, "2902": 0.09999999999999998, "2903": 0.09999999999999998, "2904": 0.12}, "seed_level_wlt": [5, 0, 0], "success_by_seed": {"child": {"2900": 1.0, "2901": 1.0, "2902": 1.0, "2903": 1.0, "2904": 1.0}, "parent": {"2900": 0.92, "2901": 0.88, "2902": 0.9, "2903": 0.9, "2904": 0.88}}, "step0_success": {"child": 0.0, "parent": 0.1}, "child_qualification_report_only": {"fitted_minus_step0_by_seed": [1.0, 1.0, 1.0, 1.0, 1.0], "improvements": 5, "worst": 1.0, "mean": 1.0, "checks": {"improves_in_enough_seeds": true, "no_seed_regresses_beyond_tolerance": true, "mean_difference_positive": true}, "qualifies": true}, "selected_steps": {"child_None": 0, "child_2900": 1000, "child_2901": 1000, "child_2902": 1000, "child_2903": 1000, "child_2904": 1000, "parent_None": 0, "parent_2900": 1000, "parent_2901": 1000, "parent_2902": 1000, "parent_2903": 1000, "parent_2904": 1000}, "resources": {"probe_episodes": 300, "confirmation_episodes": 600}}; controls {"paired_success_difference": 0.0, "note": "control is simple_damped_push refitted identically under each of the five seeds; the metric is already child minus parent"}.
**Decision: supported**. R-010 refitted the LLM geometry-scope child gen0_cand5_a0 and its parent simple_damped_push under five fitting seeds (2900-2904; probe s2905) and scored both on 50 fresh cases (s2906). The child reached 1.00 under every seed, the parent 0.88-0.92: child minus parent +0.104, placement-level 95% [+0.023, +0.185], seed-level [+0.083, +0.125], 5/0/0. The advantage does not depend on one fit. Report only: the child passes the unchanged qualification rule (0.00 untrained, 1.00 fitted), but the promotion-set (development 10-39) comparison was not run, so bank admission is not yet justified.
**Next.** Run the development 10-39 comparison (or a qualification run) before admitting gen0_cand5_a0 as a fittable_seed; test geometry-scope transfer to other parents (R-012).

## Budget-matched pooling of search and imitation candidates (D-007, 2026-10-22)

**Idea R-011.** A shared rollout selector exploits complementary search and imitation trajectories across structures better than spending the same probe budget entirely on either optimizer.
**Prediction.** paired_success_difference increase by at least 0.05 against Both equal-budget search-only and imitation-only selectors on the same frozen structures.
**Protocol.** `python scripts/stage_f.py pooled-optimizer --config config/stage_f/stage_f.yaml --additional-source logs/baseline_gpt61_sol_high --search-seed 2910 --fit-seed 2911 --probe-seed 2912 --heldout-seed 2913 --out-dir logs/research/pooled_optimizer_s2910` at git `1481a72218`.
**Result.** {"paired_success_difference": -0.055, "pooled_minus_search_only": -0.055, "pooled_minus_search_only_t975": [-0.11049798707520114, 0.0004979870752011492], "pooled_minus_search_only_holm_p": 0.05252841084879786, "pooled_minus_imitation_only": 0.0, "pooled_minus_imitation_only_t975": [-0.03303393091143082, 0.03303393091143082], "pooled_minus_imitation_only_holm_p": 1.0, "by_group": {"additional": {"pooled": 0.02, "search_only": 0.03, "imitation_only": 0.0}, "anchor": {"pooled": 0.18000000000000002, "search_only": 0.28, "imitation_only": 0.2}}, "by_structure": {"baseline61_gen0_cand2_a1": {"selected": {"pooled": "G4", "search_only": "G6", "imitation_only": "original"}, "success": {"pooled": 0.02, "search_only": 0.56, "imitation_only": 0.02}}, "baseline61_gen0_cand1_a0": {"selected": {"pooled": "fit250", "search_only": "G6", "imitation_only": "fit312"}, "success": {"pooled": 0.34, "search_only": 0.0, "imitation_only": 0.38}}, "baseline61_gen0_cand0_a0": {"selected": {"pooled": "G3", "search_only": "G5", "imitation_only": "original"}, "success": {"pooled": 0.0, "search_only": 0.02, "imitation_only": 0.0}}, "baseline61_gen0_cand3_a0": {"selected": {"pooled": "G4", "search_only": "G4", "imitation_only": "original"}, "success": {"pooled": 0.04, "search_only": 0.04, "imitation_only": 0.0}}}, "episodes": {"probe": 1160, "confirmation": 550}}; controls {"paired_success_difference": 0.0, "note": "controls are the equal-budget search-only and imitation-only selectors on the same subjects, probe and confirmation cases; the metric is already pooled minus the weaker comparison"}.
**Decision: refuted**. R-011 pooled 15 candidates per structure (original, 3 local and 4 global search draws, 7 imitation checkpoints) under one rollout probe and compared the pick against equal-budget search-only (15 draws) and imitation-only (15 checkpoints) selectors on four frozen weak evolved structures (probe s2912, confirmation s2913). Pooled minus search-only -0.055, 97.5% [-0.110, +0.000]; pooled minus imitation-only +0.000 [-0.033, +0.033]: the predicted +0.05 against both is refuted. Search's lead rests on one lucky global draw on cand2_a1 (0.56) outside the pooled subset; imitation wins cand1_a0; the two additional structures stay near zero under every selector. Parameter optimisation does not rescue weak zero-shot evolved structures at this budget.
**Next.** Prefer structural changes known to work (geometry-scope mutation, D-006) over optimiser mixing; test their transfer to other parents (R-012).

## Transfer geometry-only mutation beyond damped push (D-008, 2026-10-22)

**Idea R-012.** Restricting mutations to object–goal-relative contact geometry transfers useful approach bias to other bank structures more reliably than equally constrained gain-and-damping edits.
**Prediction.** paired_success_difference increase by at least 0.05 against Matched gain-and-damping-only LLM mutations of the same parents.
**Protocol.** `python scripts/stage_f.py mutation-scope-transfer --parents simple_reach_push_sharp_gate simple_reach_push_shared_gain --children-per-parent-scope 2 --fit-seeds 2920 2921 --probe-seed 2922 --heldout-seed 2923 --out-dir logs/research/geometry_transfer_s2920` at git `503c9322dc`.
**Result.** {"paired_success_difference": -0.2375, "geometry_minus_control_t95": [-0.3148824935102108, -0.16011750648978915], "geometry_minus_control_wlt": [3, 36, 11], "geometry_minus_parent": -0.1925, "geometry_minus_parent_t95": [-0.23386001398937997, -0.15113998601062004], "control_minus_parent": 0.045, "control_minus_parent_t95": [-0.019497724716260503, 0.1094977247162605], "by_parent": {"simple_reach_push_shared_gain": {"geometry_minus_control": {"mean": -0.335, "t95": [-0.43711614315769565, -0.2328838568423044]}, "geometry_minus_parent": {"mean": -0.175, "t95": [-0.25394759860150584, -0.09605240139849415]}, "control_minus_parent": {"mean": 0.16, "t95": [0.04996847595607269, 0.27003152404392733]}}, "simple_reach_push_sharp_gate": {"geometry_minus_control": {"mean": -0.14, "t95": [-0.22982507250064604, -0.05017492749935398]}, "geometry_minus_parent": {"mean": -0.21, "t95": [-0.2864415259686171, -0.13355847403138288]}, "control_minus_parent": {"mean": -0.07, "t95": [-0.12566729942282318, -0.014332700577176824]}}}, "by_child": {"gen0_cand0_a0": {"parent": "simple_reach_push_sharp_gate", "scope": "contact_geometry", "draw": 0, "valid": true, "success_by_seed": {"2920": 0.54, "2921": 0.54}, "selected_step_by_seed": {"2920": 0, "2921": 0}}, "gen0_cand1_a0": {"parent": "simple_reach_push_sharp_gate", "scope": "contact_geometry", "draw": 1, "valid": true, "success_by_seed": {"2920": 1.0, "2921": 1.0}, "selected_step_by_seed": {"2920": 1000, "2921": 1000}}, "gen0_cand2_a0": {"parent": "simple_reach_push_sharp_gate", "scope": "gain_damping", "draw": 0, "valid": true, "success_by_seed": {"2920": 0.96, "2921": 0.96}, "selected_step_by_seed": {"2920": 1000, "2921": 1000}}, "gen0_cand3_a0": {"parent": "simple_reach_push_sharp_gate", "scope": "gain_damping", "draw": 1, "valid": true, "success_by_seed": {"2920": 0.88, "2921": 0.84}, "selected_step_by_seed": {"2920": 1000, "2921": 1000}}, "gen0_cand4_a0": {"parent": "simple_reach_push_shared_gain", "scope": "contact_geometry", "draw": 0, "valid": true, "success_by_seed": {"2920": 0.62, "2921": 0.62}, "selected_step_by_seed": {"2920": 0, "2921": 0}}, "gen0_cand5_a0": {"parent": "simple_reach_push_shared_gain", "scope": "contact_geometry", "draw": 1, "valid": true, "success_by_seed": {"2920": 0.28, "2921": 0.26}, "selected_step_by_seed": {"2920": 1000, "2921": 1000}}, "gen0_cand6_a0": {"parent": "simple_reach_push_shared_gain", "scope": "gain_damping", "draw": 0, "valid": true, "success_by_seed": {"2920": 0.8, "2921": 0.84}, "selected_step_by_seed": {"2920": 1000, "2921": 1000}}, "gen0_cand7_a0": {"parent": "simple_reach_push_shared_gain", "scope": "gain_damping", "draw": 1, "valid": true, "success_by_seed": {"2920": 0.74, "2921": 0.74}, "selected_step_by_seed": {"2920": 1000, "2921": 1000}}}, "parent_success_by_seed": {"simple_reach_push_shared_gain": {"2920": 0.62, "2921": 0.62}, "simple_reach_push_sharp_gate": {"2920": 0.98, "2921": 0.98}}, "validity": {"valid": 8, "total": 8, "invalid": []}, "episodes": {"probe_episodes": 600, "confirmation_episodes": 1000}, "llm_calls": 9}; controls {"paired_success_difference": 0.0, "note": "control is the matched gain-and-damping-only child (same parent, draw index and fit seed); the metric is already geometry minus control"}.
**Decision: refuted**. R-012 asked the LLM (gpt-6.1-sol, 9 calls) for code-only mutations of two reach-push seed-bank parents, two geometry-only and two gain/damping-only per parent, all valid, and fitted each child and parent identically under two seeds (probe s2922, confirmation s2923). Geometry minus gain/damping: -0.237 [-0.315, -0.160]; geometry minus parent: -0.193 [-0.234, -0.151]; gain/damping minus parent: +0.045 [-0.019, +0.109]. Three of four geometry children damaged a working controller (e.g. sharp_gate 0.98 -> 0.54). The damped-push geometry win (D-004, D-006) does not transfer as a scope rule; caveat: these edits were code-only, while the winning child added parameters.
**Next.** Treat gen0_cand5_a0 as a single validated structure; run its cheap bank-admission check; do not spend further on scope rules.

## Bank-admission check: the geometry child beats its parent on the promotion set (development 10-39) (D-009, 2026-10-22)

**Idea R-013.** The geometry child gen0_cand5_a0, at the fitted state predeclared from R-010 (fit seed 2900), wins more promotion-set episodes than simple_damped_push at its identically fitted seed-2900 state, so it meets the remaining bank-admission requirement.
**Prediction.** paired_success_difference increase by at least 0.03333333333333333 against simple_damped_push at its R-010 fit-seed-2900 selected state, on development cases 10-39.
**Protocol.** `python scripts/compare_child_vs_parent.py --run logs/two_lane_search --research-protocol promotion-check --child gen0_cand5_a0 --fit-seed 2900 --from-run logs/research/geometry_qualification_s2900 --eval-seed 2940 --out-dir logs/research/geometry_admission_s2940` at git `8b4b64f6f5`.
**Result.** {"paired_success_difference": 0.03333333333333333, "paired_t95": [-0.03484098807109009, 0.10150765473775675], "wins_losses_ties": [1, 0, 29], "child_success": 1.0, "parent_success": 0.9666666666666667, "hash_verified": {"child": true, "parent": true}, "episodes": 60}; controls {"paired_success_difference": 0.0, "note": "control is simple_damped_push at its R-010 fit-seed-2900 state; the metric is already child minus parent"}.
**Decision: inconclusive**. R-013 rebuilt the predeclared seed-2900 fitted states of the geometry child gen0_cand5_a0 and its parent simple_damped_push by deterministic refit (no rollouts; both parameter hashes match R-010's records) and scored both on the promotion set (development 10-39). Child 1.000, parent 0.967: +0.033 [-0.035, +0.102], 1/0/29. The interval spans zero, so the bank-admission requirement (strictly positive lower bound) is not met, and the refutation condition is not met either. The parent fails only one promotion case, a ceiling that 30 cases cannot resolve; the child's advantage stands on the fresh-case evidence of D-004 and D-006.
**Next.** Bank admission of gen0_cand5_a0 needs a ruling on whether fresh-case evidence may substitute for the promotion-set rule; R-009 (LLM bounds-only edits) is approved and next.

## Test LLM bounds-only edits against matched random bounds (D-010, 2026-10-22)

**Idea R-009.** LLM-chosen parameter bounds constrain imitation updates to successful control regimes while preserving the parent's executable structure and initialization.
**Prediction.** paired_success_difference increase by at least 0.05 against Matched random-bound edits with identical fitting and rollout selection.
**Protocol.** `python scripts/stage_f.py bounds-only --parents simple_damped_push simple_reach_push_shared_gain --fit-seeds 2823 2824 --probe-seed 2820 --heldout-seed 2822 --range-seed 2825 --out-dir logs/research/bounds_only_s2820` at git `d43be120c8`.
**Result.** {"paired_success_difference": 0.0, "llm_minus_random_t95": [0.0, 0.0], "llm_minus_unedited": 0.01, "llm_minus_unedited_t95": [-0.004064096944854701, 0.0240640969448547], "random_minus_unedited": 0.01, "by_cell": [{"parent": "simple_damped_push", "fit_seed": 2823, "edits": {}, "random": {}, "edit_problems": [], "selected_steps": {"unedited": 1000, "llm": 1000, "random": 1000}, "success": {"unedited": 0.88, "llm": 0.88, "random": 0.88}}, {"parent": "simple_damped_push", "fit_seed": 2824, "edits": {}, "random": {}, "edit_problems": [], "selected_steps": {"unedited": 1000, "llm": 1000, "random": 1000}, "success": {"unedited": 0.86, "llm": 0.86, "random": 0.86}}, {"parent": "simple_reach_push_shared_gain", "fit_seed": 2823, "edits": {"contact_radius": [0.05, 0.2]}, "random": {"contact_radius": [0.039861465880129295, 0.1898614658801293]}, "edit_problems": [], "selected_steps": {"unedited": 0, "llm": 0, "random": 0}, "success": {"unedited": 0.62, "llm": 0.62, "random": 0.62}}, {"parent": "simple_reach_push_shared_gain", "fit_seed": 2824, "edits": {"contact_radius": [0.03, 0.2]}, "random": {"contact_radius": [0.022802267764058623, 0.19280226776405862]}, "edit_problems": [], "selected_steps": {"unedited": 1000, "llm": 1000, "random": 1000}, "success": {"unedited": 0.64, "llm": 0.68, "random": 0.68}}], "episodes": {"probe": 180, "confirmation": 600}, "llm_calls": 4}; controls {"paired_success_difference": 0.0, "note": "control is the width- and parameter-matched random-bound arm in the same cell; the metric is already LLM minus random"}.
**Decision: refuted**. R-009 let gpt-6.1-sol narrow at most two parameter ranges (no code change) of two seed-bank parents, given the unedited fit's probe trajectory, and compared against width-matched random ranges and the unedited fit, all fitted identically (fit seeds 2823/2824, probe s2820, confirmation s2822). The LLM made no edit for simple_damped_push and raised the contact_radius floor for shared_gain; LLM minus random +0.000 [+0.000, +0.000], 0/0/50, and both edited arms equal the unedited fit within noise (+0.010 [-0.004, +0.024]). The predicted +0.05 is refuted; the bounds rarely bind under rollout-selected fitting.
**Next.** LLM interventions on the fitted seed bank (bounded mutation, repair, scope edits, bounds) are null or negative except one structure; prefer a ruling on admitting gen0_cand5_a0 and changes to the search pipeline itself.

## Ablate the geometry child's added trainable dimensions (D-011, 2026-10-22)

**Idea R-014.** The geometry child's advantage depends on fitting its added parameters, rather than only on its changed forward geometry.
**Prediction.** paired_success_difference increase by at least 0.05 against gen0_cand5_a0 with all child-only parameters clamped at declared initial values.
**Protocol.** `python scripts/compare_child_vs_parent.py --run logs/two_lane_search --research-protocol added-parameter-ablation --child gen0_cand5_a0 --fit-seeds 3010 3011 --probe-seed 3012 --heldout-seed 3013 --out-dir logs/research/added_parameter_ablation_s3010` at git `99be396f59`.
**Result.** {"paired_success_difference": 0.0, "placement_t95": [0.0, 0.0], "wlt": [0, 0, 50], "seed_level": {"3010": 0.0, "3011": 0.0}, "success_by_seed": {"trainable": {"3010": 1.0, "3011": 1.0}, "clamped": {"3010": 1.0, "3011": 1.0}}, "selected_steps": {"trainable_fit3010": 1000, "trainable_fit3011": 1000, "clamped_fit3010": 1000, "clamped_fit3011": 1000}, "resources": {"probe_episodes": 240, "confirmation_episodes": 200}}; controls {"paired_success_difference": 0.0, "note": "control is the same child with its child-only parameter (approach_standoff) clamped at its initial value 0.02; the metric is already trainable minus clamped"}.
**Decision: refuted**. R-014 clamped the geometry child's only added parameter (approach_standoff, fixed at its declared 0.02) and compared it with the fully trainable child under identical fitting (seeds 3010/3011, probe s3012, confirmation s3013). Both reach 1.00 under both seeds: +0.000 [0, 0], 0/0/50. The predicted +0.05 is refuted: the child's advantage over simple_damped_push (D-006) comes from its changed forward geometry with a fixed 0.02 standoff, not from fitting the added dimension.
**Next.** A hand-written simple_damped_push variant with a fixed 0.02 standoff would reproduce the gain without an LLM; R-016 (adaptive search) is next.

## Adapt parameter proposals using progress before success appears (D-012, 2026-10-22)

**Idea R-016.** Sequential parameter proposals guided by task progress can exploit weak structures more efficiently than independent local and global draws.
**Prediction.** paired_success_difference increase by at least 0.08 against budget-matched independent parameter search on baseline61_gen0_cand1_a0 and baseline61_gen0_cand2_a1.
**Protocol.** `python scripts/stage_f.py adaptive-search --config config/stage_f/stage_f.yaml --search-seeds 3030 3031 --probe-seed 3032 --heldout-seed 3033 --probe-cases 5 --out-dir logs/research/adaptive_search_s3030` at git `4a3cb11856`.
**Result.** {"paired_success_difference": 0.07, "placement_t95": [0.03777510558450309, 0.10222489441549693], "placement_wlt": [14, 0, 36], "cell_level": {"by_cell": {"baseline61_gen0_cand2_a1_s3030": 0.17999999999999994, "baseline61_gen0_cand2_a1_s3031": 0.1, "baseline61_gen0_cand1_a0_s3030": 0.0, "baseline61_gen0_cand1_a0_s3031": 0.0}, "mean": 0.06999999999999998, "t95": [-0.06871961837978494, 0.2087196183797849]}, "by_cell": {"baseline61_gen0_cand2_a1_s3030": {"success": {"adaptive": 0.94, "independent": 0.76}, "finalist": {"adaptive": 20, "independent": 6}}, "baseline61_gen0_cand2_a1_s3031": {"success": {"adaptive": 0.1, "independent": 0.0}, "finalist": {"adaptive": 9, "independent": 4}}, "baseline61_gen0_cand1_a0_s3030": {"success": {"adaptive": 0.68, "independent": 0.68}, "finalist": {"adaptive": 7, "independent": 7}}, "baseline61_gen0_cand1_a0_s3031": {"success": {"adaptive": 0.42, "independent": 0.42}, "finalist": {"adaptive": 2, "independent": 2}}}, "episodes": {"probe": 800, "confirmation": 300}}; controls {"paired_success_difference": 0.0, "note": "control is budget-matched independent search (shared first 8 + 16 independent local/global draws); the metric is already adaptive minus independent"}.
**Decision: inconclusive**. R-016 compared progress-guided adaptive parameter search (two batches of 8 from a Gaussian over the top-3 candidates) with budget-matched independent local/global draws, 24 candidates per arm with the first 8 shared, on two weak evolved structures x two search seeds (5-case probe s3032, confirmation s3033). Adaptive minus independent: +0.070 placement-level [+0.038, +0.102], 14/0/36, but cell-level [-0.069, +0.209]; the predicted +0.08 is neither supported nor refuted. All gains are on baseline61_gen0_cand2_a1 (0.94 vs 0.76, 0.10 vs 0.00); gen0_cand1_a0 tied on a shared-batch pick. Notably, 24-candidate rollout search lifted these 0.00-zero-shot structures to 0.42-0.94 in three of four cells.
**Next.** Confirm on more weak structures with a larger probe before changing the exploration lane; 24-candidate rollout search itself looks like a strong rescue for zero-shot structures.

## Generate fit-ready structures without supplying a parent (D-013, 2026-10-22)

**Idea R-015.** Bounded, phase-scaled parameters and smooth transitions make parameter perturbations behaviorally useful, increasing the success attainable by budget-matched direct rollout search on seedless generated structures.
**Prediction.** paired_success_difference increase by at least 0.1 against the same single-phase generation prompt without the contract, under identical 24-candidate independent rollout search.
**Protocol.** `python scripts/compare_generation_modes.py --research-mode seedless-searchability --generation-seeds 3020 3021 3022 --search-seeds 3023 3024 --probe-seed 3025 --heldout-seed 3026 --out-dir logs/research/seedless_searchability_s3020` at git `da232834de`.
**Result.** {"paired_success_difference": -0.29333333333333333, "block_t95": [-0.655313484206806, 0.06864681754013935], "by_block": {"3020": -0.13999999999999996, "3021": -0.31, "3022": -0.43000000000000005}, "placement_t95": [-0.3653013735065549, -0.22136529316011166], "placement_wlt": [3, 39, 8], "step0_contract_minus_baseline": -0.24, "step0_by_block": {"3020": -0.07999999999999996, "3021": -0.16, "3022": -0.48}, "by_slot": {"b3020_baseline": {"block": 3020, "arm": "baseline", "valid": true, "searched": {"3023": 1.0, "3024": 0.98}, "original": 0.88}, "b3020_contract": {"block": 3020, "arm": "contract", "valid": true, "searched": {"3023": 0.8, "3024": 0.9}, "original": 0.8}, "b3021_baseline": {"block": 3021, "arm": "baseline", "valid": true, "searched": {"3023": 0.84, "3024": 0.94}, "original": 0.16}, "b3021_contract": {"block": 3021, "arm": "contract", "valid": true, "searched": {"3023": 0.58, "3024": 0.58}, "original": 0.0}, "b3022_baseline": {"block": 3022, "arm": "baseline", "valid": true, "searched": {"3023": 0.8, "3024": 0.38}, "original": 0.48}, "b3022_contract": {"block": 3022, "arm": "contract", "valid": true, "searched": {"3023": 0.1, "3024": 0.22}, "original": 0.0}}, "episodes": {"probe": 2880, "confirmation": 850}, "llm_calls": 6}; controls {"paired_success_difference": 0.0, "note": "control is the same single-phase prompt without the contract, generated in the same block and searched identically; the metric is already contract minus baseline"}.
**Decision: refuted**. R-015 appended a parameterization contract (smooth sigmoid gates, unit-direction drives scaled by bounded gains, 6-10 bounded parameters) to the single-phase seedless generation prompt and compared it with the same prompt without the contract: three blocks, one gpt-6.1-sol structure per prompt (all valid first time), each given 24-candidate independent rollout search under two seeds (probe s3025, confirmation s3026). Contract minus baseline: -0.293, block-level [-0.655, +0.069], placement-level [-0.365, -0.221]; worse in every block, untrained and searched. The predicted +0.10 is refuted. Side observation: the plain single-phase prompt's structures scored 0.16-0.88 untrained and 0.80-1.00 after search, against 0.00 for the two-phase gpt-6.1-sol baseline run.
**Next.** Test single-phase versus two-phase seedless generation under rollout search with more blocks; if it holds, switch the exploration lane's prompt and add rollout search.

## Single-phase seedless generation beats the two-phase ideas prompt under rollout search (D-014, 2026-10-22)

**Idea R-017.** The two-phase ideas prompt (designs first, then per-idea implementation, as in the gpt-6.1-sol baseline and the exploration lane) produces structures that rollout search rescues less often than the single-phase prompt, so the exploration lane should switch prompts.
**Prediction.** paired_success_difference increase by at least 0.15 against two-phase per-idea seedless generation (ideas_system/ideas_user then one implementation) in the same block, under identical 24-candidate independent rollout search.
**Protocol.** `python scripts/compare_generation_modes.py --research-mode phase-searchability --generation-seeds 3040 3041 3042 3043 --search-seeds 3044 3045 --probe-seed 3046 --heldout-seed 3047 --out-dir logs/research/phase_searchability_s3040` at git `bfc31a5183`.
**Result.** {"paired_success_difference": 0.6975, "block_t95": [0.3949234328828564, 1.0000765671171437], "by_block": {"3040": 0.44, "3041": 0.79, "3042": 0.6799999999999999, "3043": 0.88}, "placement_t95": [0.658130732625354, 0.736869267374646], "placement_wlt": [50, 0, 0], "step0_single_minus_two": 0.25, "step0_by_block": {"3040": 0.0, "3041": 0.0, "3042": 0.1, "3043": 0.9}, "by_slot": {"b3040_two_phase": {"block": 3040, "arm": "two_phase", "valid": true, "searched": {"3044": 0.0, "3045": 0.0}, "original": 0.0}, "b3040_single_phase": {"block": 3040, "arm": "single_phase", "valid": true, "searched": {"3044": 0.38, "3045": 0.5}, "original": 0.0}, "b3041_two_phase": {"block": 3041, "arm": "two_phase", "valid": true, "searched": {"3044": 0.04, "3045": 0.0}, "original": 0.0}, "b3041_single_phase": {"block": 3041, "arm": "single_phase", "valid": true, "searched": {"3044": 0.82, "3045": 0.8}, "original": 0.0}, "b3042_two_phase": {"block": 3042, "arm": "two_phase", "valid": true, "searched": {"3044": 0.64, "3045": 0.0}, "original": 0.0}, "b3042_single_phase": {"block": 3042, "arm": "single_phase", "valid": true, "searched": {"3044": 1.0, "3045": 1.0}, "original": 0.1}, "b3043_two_phase": {"block": 3043, "arm": "two_phase", "valid": true, "searched": {"3044": 0.06, "3045": 0.02}, "original": 0.0}, "b3043_single_phase": {"block": 3043, "arm": "single_phase", "valid": true, "searched": {"3044": 0.88, "3045": 0.96}, "original": 0.9}}, "episodes": {"probe": 3840, "confirmation": 1200}, "llm_calls": 13}; controls {"paired_success_difference": 0.0, "note": "control is the two-phase per-idea generator (ideas prompt, then one implementation) in the same block, searched identically; the metric is already single-phase minus two-phase"}.
**Decision: supported**. R-017 compared seedless generation by the single-phase prompt with the two-phase ideas-then-implementation generator the exploration lane and the gpt-6.1-sol baseline use, four blocks, one structure per generator per block (13 gpt-6.1-sol calls), each given 24-candidate independent rollout search under two seeds (probe s3046, confirmation s3047). Single-phase minus two-phase: +0.698, block-level [+0.395, +1.000]; single-phase won every block (0.44-1.00 vs 0.00-0.32). Untrained, two-phase scored 0.00 everywhere (as in the baseline run) and single-phase 0.00-0.90. Supported: the generator bundle (prompt wording plus design/implementation split), not phasing alone, is what the result isolates; part of the gain exists before search (+0.25).
**Next.** Switch the exploration lane to single-phase generation plus 24-candidate rollout search and rerun the end-to-end search against the current configuration on a fresh confirmation set.

## End-to-end: a single-phase, rollout-searched exploration lane raises the search's best held-out success (D-015, 2026-10-22)

**Idea R-018.** Single-phase generation plus rollout search supplies stronger exploration candidates that survive selection against the unchanged fittable lane and improve the run's selected incumbent.
**Prediction.** paired_success_difference increase by at least 0.05 against the current config/two_lane_search.yaml run (fresh rerun, same model and manifests) — each run's performance incumbent scored on the same fresh confirmation cases.
**Protocol.** `for each block b in 0..5: python scripts/run_full_evolution.py --config config/two_lane_search.yaml (run seed 3100+2b) and --config config/two_lane_search_v2.yaml (run seed 3101+2b); then python scripts/evaluate_on_fresh_manifest.py --pool-seed 313b on both incumbents` at git `04fd32e6f2`.
**Result.** {"paired_success_difference": -0.006666666666666665, "run_level_t95": [-0.03534270755748655, 0.022009374224153222], "block_differences": [0.0, 0.02, 0.0, -0.06, 0.0, 0.0], "wins": 1, "losses": 1, "ties": 4, "incumbent_lane": "fittable in all 12 runs (promotion-set success 1.00 in all)", "exploration_screen_success_by_run": {"control_s3100": [0.0, 0.0, 0.0], "control_s3102": [0.0, 0.0, 0.2], "control_s3104": [0.0, 0.0, 0.1], "control_s3106": [0.0, 0.0, 0.2], "control_s3108": [0.0, 0.0, 0.1], "control_s3110": [0.0, 0.0, 0.0], "v2_s3101": [0.9, 1.0, 1.0], "v2_s3103": [0.0, 0.6, 0.9], "v2_s3105": [0.8, 0.8, 0.9], "v2_s3107": [0.4, 0.9, 1.0], "v2_s3109": [0.5, 1.0, 1.0], "v2_s3111": [0.4, 0.9, 1.0]}, "exploration_best_screen_mean": {"control": 0.09999999999999999, "v2": 0.9666666666666667}, "exploration_mean_screen": {"control": 0.03333333333333333, "v2": 0.7777777777777778}, "llm_calls": {"control_runs": [11, 11, 11, 11, 12, 11], "v2_runs": [10, 10, 10, 10, 10, 10]}, "episodes": {"control_run": 450, "v2_run": 1200, "confirmation": 600}}; controls {"paired_success_difference": 0.0, "note": "control is a fresh rerun of config/two_lane_search.yaml per block; the metric is already v2 minus control incumbent success on the block's fresh cases"}.
**Decision: refuted**. R-018 added an exploration-lane option (config/two_lane_search_v2.yaml): single-phase generation with code-derived idea records, treated by 24-candidate independent rollout search on the probe. Six paired runs per configuration (seeds 3100-3111), incumbents scored on per-block fresh manifests (pool seeds 3130-3135). v2 minus control incumbent success: -0.007, run-level [-0.035, +0.022]; the predicted +0.05 is refuted. Every incumbent in all twelve runs came from the unchanged fittable lane at 1.00 promotion success, so the task is saturated and exploration candidates could only tie. The new lane itself works: exploration screen success rose from 0.03 to 0.78 on average (best per run 0.10 to 0.97) with fewer LLM calls (10 vs 11-12).
**Next.** Keep v2 as an option, not the default, on push-v2; evaluate the v2 exploration lane where no seed bank saturates the task (a harder MetaWorld task, or the fittable lane disabled), and confirm its screen winners on fresh cases.

## 2. Test whether search needs grasp-and-lift-aware selection (D-016, 2026-10-22)

**Idea R-020.** When pick-place success is sparse, ranking successful grasp-and-lift behaviour ahead of geometric goal progress selects more transferable controllers from the same rollout candidates.
**Prediction.** paired_success_difference increase by at least 0.1 against baseline.
**Protocol.** `python scripts/compare_generation_modes.py --research-mode pick-place-suite --env-name pick-place-v2 --probe-seed 3214 --heldout-seed 3215 --out-dir logs/pick_place_selector_ablation` at git `dd60a340d0`.
**Result.** {"paired_success_difference": 0.0, "block_t95": [0.0, 0.0], "by_block": [0.0, 0.0, 0.0, 0.0], "placement_t95": [0.0, 0.0], "by_slot": {"control_b0_s3210": {"arm": "control", "valid": true, "picks": {"original": 7, "lift": 7}, "success": {"original": 1.0, "lift": 1.0}, "best_probe": 1.0, "probe_lift_max": 1.0}, "control_b1_s3211": {"arm": "control", "valid": true, "picks": {"original": 23, "lift": 23}, "success": {"original": 1.0, "lift": 1.0}, "best_probe": 1.0, "probe_lift_max": 1.0}, "control_b2_s3212": {"arm": "control", "valid": true, "picks": {"original": 13, "lift": 13}, "success": {"original": 1.0, "lift": 1.0}, "best_probe": 1.0, "probe_lift_max": 1.0}, "control_b3_s3213": {"arm": "control", "valid": true, "picks": {"original": 14, "lift": 14}, "success": {"original": 1.0, "lift": 1.0}, "best_probe": 1.0, "probe_lift_max": 1.0}, "grounded_b0_s3220": {"arm": "grounded", "valid": true, "picks": {"original": 21}, "success": {"original": 1.0}, "best_probe": 1.0, "probe_lift_max": 1.0}, "grounded_b1_s3221": {"arm": "grounded", "valid": true, "picks": {"original": 15}, "success": {"original": 1.0}, "best_probe": 1.0, "probe_lift_max": 1.0}, "grounded_b2_s3222": {"arm": "grounded", "valid": true, "picks": {"original": 3}, "success": {"original": 1.0}, "best_probe": 1.0, "probe_lift_max": 1.0}, "grounded_b3_s3223": {"arm": "grounded", "valid": true, "picks": {"original": 1}, "success": {"original": 1.0}, "best_probe": 1.0, "probe_lift_max": 1.0}, "two_phase_b0_s3200": {"arm": "two_phase", "valid": true, "picks": {"original": 10}, "success": {"original": 0.0}, "best_probe": 0.0, "probe_lift_max": 1.0}, "two_phase_b1_s3201": {"arm": "two_phase", "valid": true, "picks": {"original": 22}, "success": {"original": 0.9}, "best_probe": 1.0, "probe_lift_max": 1.0}, "two_phase_b2_s3202": {"arm": "two_phase", "valid": true, "picks": {"original": 17}, "success": {"original": 1.0}, "best_probe": 1.0, "probe_lift_max": 1.0}, "two_phase_b3_s3205": {"arm": "two_phase", "valid": true, "picks": {"original": 18}, "success": {"original": 0.9666666666666667}, "best_probe": 1.0, "probe_lift_max": 1.0}}, "candidates": {"control_b0_s3210": {"original_probe_success": 1.0, "candidates_with_probe_success": 18}, "control_b1_s3211": {"original_probe_success": 1.0, "candidates_with_probe_success": 16}, "control_b2_s3212": {"original_probe_success": 1.0, "candidates_with_probe_success": 16}, "control_b3_s3213": {"original_probe_success": 1.0, "candidates_with_probe_success": 20}, "grounded_b0_s3220": {"original_probe_success": 1.0, "candidates_with_probe_success": 13}, "grounded_b1_s3221": {"original_probe_success": 1.0, "candidates_with_probe_success": 17}, "grounded_b2_s3222": {"original_probe_success": 1.0, "candidates_with_probe_success": 16}, "grounded_b3_s3223": {"original_probe_success": 0.0, "candidates_with_probe_success": 9}, "two_phase_b0_s3200": {"original_probe_success": 0.0, "candidates_with_probe_success": 0}, "two_phase_b1_s3201": {"original_probe_success": 1.0, "candidates_with_probe_success": 17}, "two_phase_b2_s3202": {"original_probe_success": 1.0, "candidates_with_probe_success": 9}, "two_phase_b3_s3205": {"original_probe_success": 0.8, "candidates_with_probe_success": 13}}, "episodes": {"probe": 1440, "confirmation": 360}, "llm_calls": 16}; controls {"paired_success_difference": 0.0, "note": "control is the control pools, original (success, progress) selector; the metric is already treatment minus control"}.
**Decision: refuted**. R-020 (pick-place-v2 shared-control suite) re-ranked the same 24-candidate search pools of four single-phase structures by success, then grasp-and-lift rate, then progress, against the original success-then-progress selector. Lift-aware minus original: +0.000 [0, 0]; the predicted +0.10 is refuted. Ceiling: every pool already held fully successful candidates and both selectors picked the same one, so the lift key was never consulted.
**Next.** Selection is not the bottleneck when pools contain successes; revisit only where successes are sparse.

## 3. Test explicit observation and gripper-action grounding (D-017, 2026-10-22)

**Idea R-021.** An explicit environment-derived observation/action contract prevents gripper-sign and coordinate-interpretation errors that otherwise make single-phase pick-place structures unsearchable.
**Prediction.** paired_success_difference increase by at least 0.1 against baseline.
**Protocol.** `python scripts/compare_generation_modes.py --research-mode pick-place-suite --env-name pick-place-v2 --probe-seed 3214 --heldout-seed 3215 --out-dir logs/pick_place_selector_ablation` at git `dd60a340d0`.
**Result.** {"paired_success_difference": 0.0, "block_t95": [0.0, 0.0], "by_block": [0.0, 0.0, 0.0, 0.0], "placement_t95": [0.0, 0.0], "by_slot": {"control_b0_s3210": {"arm": "control", "valid": true, "picks": {"original": 7, "lift": 7}, "success": {"original": 1.0, "lift": 1.0}, "best_probe": 1.0, "probe_lift_max": 1.0}, "control_b1_s3211": {"arm": "control", "valid": true, "picks": {"original": 23, "lift": 23}, "success": {"original": 1.0, "lift": 1.0}, "best_probe": 1.0, "probe_lift_max": 1.0}, "control_b2_s3212": {"arm": "control", "valid": true, "picks": {"original": 13, "lift": 13}, "success": {"original": 1.0, "lift": 1.0}, "best_probe": 1.0, "probe_lift_max": 1.0}, "control_b3_s3213": {"arm": "control", "valid": true, "picks": {"original": 14, "lift": 14}, "success": {"original": 1.0, "lift": 1.0}, "best_probe": 1.0, "probe_lift_max": 1.0}, "grounded_b0_s3220": {"arm": "grounded", "valid": true, "picks": {"original": 21}, "success": {"original": 1.0}, "best_probe": 1.0, "probe_lift_max": 1.0}, "grounded_b1_s3221": {"arm": "grounded", "valid": true, "picks": {"original": 15}, "success": {"original": 1.0}, "best_probe": 1.0, "probe_lift_max": 1.0}, "grounded_b2_s3222": {"arm": "grounded", "valid": true, "picks": {"original": 3}, "success": {"original": 1.0}, "best_probe": 1.0, "probe_lift_max": 1.0}, "grounded_b3_s3223": {"arm": "grounded", "valid": true, "picks": {"original": 1}, "success": {"original": 1.0}, "best_probe": 1.0, "probe_lift_max": 1.0}, "two_phase_b0_s3200": {"arm": "two_phase", "valid": true, "picks": {"original": 10}, "success": {"original": 0.0}, "best_probe": 0.0, "probe_lift_max": 1.0}, "two_phase_b1_s3201": {"arm": "two_phase", "valid": true, "picks": {"original": 22}, "success": {"original": 0.9}, "best_probe": 1.0, "probe_lift_max": 1.0}, "two_phase_b2_s3202": {"arm": "two_phase", "valid": true, "picks": {"original": 17}, "success": {"original": 1.0}, "best_probe": 1.0, "probe_lift_max": 1.0}, "two_phase_b3_s3205": {"arm": "two_phase", "valid": true, "picks": {"original": 18}, "success": {"original": 0.9666666666666667}, "best_probe": 1.0, "probe_lift_max": 1.0}}, "candidates": {"control_b0_s3210": {"original_probe_success": 1.0, "candidates_with_probe_success": 18}, "control_b1_s3211": {"original_probe_success": 1.0, "candidates_with_probe_success": 16}, "control_b2_s3212": {"original_probe_success": 1.0, "candidates_with_probe_success": 16}, "control_b3_s3213": {"original_probe_success": 1.0, "candidates_with_probe_success": 20}, "grounded_b0_s3220": {"original_probe_success": 1.0, "candidates_with_probe_success": 13}, "grounded_b1_s3221": {"original_probe_success": 1.0, "candidates_with_probe_success": 17}, "grounded_b2_s3222": {"original_probe_success": 1.0, "candidates_with_probe_success": 16}, "grounded_b3_s3223": {"original_probe_success": 0.0, "candidates_with_probe_success": 9}, "two_phase_b0_s3200": {"original_probe_success": 0.0, "candidates_with_probe_success": 0}, "two_phase_b1_s3201": {"original_probe_success": 1.0, "candidates_with_probe_success": 17}, "two_phase_b2_s3202": {"original_probe_success": 1.0, "candidates_with_probe_success": 9}, "two_phase_b3_s3205": {"original_probe_success": 0.8, "candidates_with_probe_success": 13}}, "episodes": {"probe": 1440, "confirmation": 360}, "llm_calls": 16}; controls {"paired_success_difference": 0.0, "note": "control is the single-phase prompt without the interface table, same block index; the metric is already treatment minus control"}.
**Decision: refuted**. R-021 appended a source-verified interface table (action scaling, gripper polarity, observation meanings, read from MetaWorld 3 source) to the single-phase pick-place-v2 prompt and compared it with the unchanged prompt under identical 24-candidate search, four paired blocks. Grounded minus control: +0.000 [0, 0]; the predicted +0.10 is refuted at a ceiling: both arms reach 1.00 after search, and untrained the plain prompt was if anything better (4/4 vs 3/4 structures succeeding at step 0).
**Next.** Keep the schema-only prompt; do not add the interface table.

## 1. Test seedless generator transfer to grasp-and-transport (D-018, 2026-10-22)

**Idea R-019.** Single-phase generation preserves grasp-to-transport coordination better than ideas-then-implementation generation, giving rollout search better pick-place-v2 structures without a seed bank.
**Prediction.** paired_success_difference increase by at least 0.15 against baseline.
**Protocol.** `python scripts/compare_generation_modes.py --research-mode pick-place-suite --env-name pick-place-v2 --probe-seed 3214 --heldout-seed 3215 --out-dir logs/pick_place_selector_ablation` at git `dd60a340d0`.
**Result.** {"paired_success_difference": 0.2833333333333333, "block_t95": [-0.4797904822732848, 1.0464571489399515], "by_block": [1.0, 0.09999999999999998, 0.0, 0.033333333333333326], "placement_t95": [0.24280266571320142, 0.32386400095346524], "by_slot": {"control_b0_s3210": {"arm": "control", "valid": true, "picks": {"original": 7, "lift": 7}, "success": {"original": 1.0, "lift": 1.0}, "best_probe": 1.0, "probe_lift_max": 1.0}, "control_b1_s3211": {"arm": "control", "valid": true, "picks": {"original": 23, "lift": 23}, "success": {"original": 1.0, "lift": 1.0}, "best_probe": 1.0, "probe_lift_max": 1.0}, "control_b2_s3212": {"arm": "control", "valid": true, "picks": {"original": 13, "lift": 13}, "success": {"original": 1.0, "lift": 1.0}, "best_probe": 1.0, "probe_lift_max": 1.0}, "control_b3_s3213": {"arm": "control", "valid": true, "picks": {"original": 14, "lift": 14}, "success": {"original": 1.0, "lift": 1.0}, "best_probe": 1.0, "probe_lift_max": 1.0}, "grounded_b0_s3220": {"arm": "grounded", "valid": true, "picks": {"original": 21}, "success": {"original": 1.0}, "best_probe": 1.0, "probe_lift_max": 1.0}, "grounded_b1_s3221": {"arm": "grounded", "valid": true, "picks": {"original": 15}, "success": {"original": 1.0}, "best_probe": 1.0, "probe_lift_max": 1.0}, "grounded_b2_s3222": {"arm": "grounded", "valid": true, "picks": {"original": 3}, "success": {"original": 1.0}, "best_probe": 1.0, "probe_lift_max": 1.0}, "grounded_b3_s3223": {"arm": "grounded", "valid": true, "picks": {"original": 1}, "success": {"original": 1.0}, "best_probe": 1.0, "probe_lift_max": 1.0}, "two_phase_b0_s3200": {"arm": "two_phase", "valid": true, "picks": {"original": 10}, "success": {"original": 0.0}, "best_probe": 0.0, "probe_lift_max": 1.0}, "two_phase_b1_s3201": {"arm": "two_phase", "valid": true, "picks": {"original": 22}, "success": {"original": 0.9}, "best_probe": 1.0, "probe_lift_max": 1.0}, "two_phase_b2_s3202": {"arm": "two_phase", "valid": true, "picks": {"original": 17}, "success": {"original": 1.0}, "best_probe": 1.0, "probe_lift_max": 1.0}, "two_phase_b3_s3205": {"arm": "two_phase", "valid": true, "picks": {"original": 18}, "success": {"original": 0.9666666666666667}, "best_probe": 1.0, "probe_lift_max": 1.0}}, "candidates": {"control_b0_s3210": {"original_probe_success": 1.0, "candidates_with_probe_success": 18}, "control_b1_s3211": {"original_probe_success": 1.0, "candidates_with_probe_success": 16}, "control_b2_s3212": {"original_probe_success": 1.0, "candidates_with_probe_success": 16}, "control_b3_s3213": {"original_probe_success": 1.0, "candidates_with_probe_success": 20}, "grounded_b0_s3220": {"original_probe_success": 1.0, "candidates_with_probe_success": 13}, "grounded_b1_s3221": {"original_probe_success": 1.0, "candidates_with_probe_success": 17}, "grounded_b2_s3222": {"original_probe_success": 1.0, "candidates_with_probe_success": 16}, "grounded_b3_s3223": {"original_probe_success": 0.0, "candidates_with_probe_success": 9}, "two_phase_b0_s3200": {"original_probe_success": 0.0, "candidates_with_probe_success": 0}, "two_phase_b1_s3201": {"original_probe_success": 1.0, "candidates_with_probe_success": 17}, "two_phase_b2_s3202": {"original_probe_success": 1.0, "candidates_with_probe_success": 9}, "two_phase_b3_s3205": {"original_probe_success": 0.8, "candidates_with_probe_success": 13}}, "episodes": {"probe": 1440, "confirmation": 360}, "llm_calls": 16}; controls {"paired_success_difference": 0.0, "note": "control is the two-phase ideas-then-implementation generator, same block index; the metric is already treatment minus control"}.
**Decision: inconclusive**. R-019 compared single-phase and two-phase seedless generation on pick-place-v2 under identical 24-candidate rollout search (four blocks, shared controls with R-020/R-021). Single minus two-phase: +0.283 block-level [-0.480, +1.046], inconclusive: one two-phase structure found no successful candidate (1.00 vs 0.00) while the other blocks differ by at most 0.10. The suite's main finding: with no seed bank, single-phase gpt-6.1-sol structures plus 24-candidate search solve pick-place-v2 (8/8 at 1.00 on 30 fresh cases; 7/8 succeed untrained).
**Next.** Find a task where single-phase zero-shot fails before testing generator or selector variants further; consider the D-014 recipe as the seedless default.

## 1. Scan eight tasks for failures of seedless generation plus search (D-019, 2026-10-22)

**Idea R-022.** Task-specific contact and articulation requirements will expose failures of the unchanged seedless controller generator that are absent on reach-v2.
**Prediction.** promotion_success_rate decrease by at least 0.2 against baseline: reach-v2 arm within this scan.
**Protocol.** `python scripts/compare_generation_modes.py --research-mode task-scan --out-dir logs/research/task_scan_s2600` at git `0f40b0592c`.
**Result.** {"promotion_success_rate": 1.0, "reference_reach_success": 1.0, "largest_drop_below_reach": 0.0, "tasks_at_least_0.20_below": [], "by_task": {"reach-v2": {"valid": true, "attempts": 1, "confirmation_success": 1.0, "untrained_probe_success": 1.0, "candidates_with_probe_success": 24, "best_probe_success": 1.0}, "door-open-v2": {"valid": true, "attempts": 1, "confirmation_success": 1.0, "untrained_probe_success": 0.8, "candidates_with_probe_success": 15, "best_probe_success": 1.0}, "door-close-v2": {"valid": true, "attempts": 1, "confirmation_success": 1.0, "untrained_probe_success": 1.0, "candidates_with_probe_success": 21, "best_probe_success": 1.0}, "drawer-open-v2": {"valid": true, "attempts": 1, "confirmation_success": 1.0, "untrained_probe_success": 1.0, "candidates_with_probe_success": 21, "best_probe_success": 1.0}, "drawer-close-v2": {"valid": true, "attempts": 1, "confirmation_success": 1.0, "untrained_probe_success": 1.0, "candidates_with_probe_success": 17, "best_probe_success": 1.0}, "window-open-v2": {"valid": true, "attempts": 1, "confirmation_success": 1.0, "untrained_probe_success": 1.0, "candidates_with_probe_success": 17, "best_probe_success": 1.0}, "window-close-v2": {"valid": true, "attempts": 1, "confirmation_success": 1.0, "untrained_probe_success": 1.0, "candidates_with_probe_success": 20, "best_probe_success": 1.0}, "button-press-v2": {"valid": true, "attempts": 1, "confirmation_success": 1.0, "untrained_probe_success": 0.2, "candidates_with_probe_success": 11, "best_probe_success": 1.0}}, "episodes": {"probe": 960, "confirmation": 240}, "llm_calls": 8}; controls {"promotion_success_rate": 1.0, "note": "control is the reach-v2 arm of this scan; the prediction asks whether any other task's confirmation success is at least 0.20 lower"}.
**Decision: refuted**. R-022 scanned eight schema-supported MetaWorld tasks (reach, door-open/close, drawer-open/close, window-open/close, button-press) with one gpt-6.1-sol single-phase structure each, 24-candidate independent rollout search on a 5-case probe and the winner on 30 fresh cases. Every task reached 1.00; six structures already succeeded untrained, and search lifted button-press (0.2) and door-open (0.8) to 1.0. No task fell 0.20 below reach: refuted. With push-v2 and pick-place-v2, the single-phase + search recipe now saturates every task this codebase supports.
**Next.** Method comparisons need harder tasks (new observation schemas and prompt assets for assembly, peg-insert, multi-object tasks) or a stricter success criterion; R-023 should not run under its own predeclared rule.

## Harder-task scan: where does single-phase generation plus rollout search fail? (D-020, 2026-10-22)

**Idea R-024.** Tasks needing tool use, insertion or multi-stage contact (assembly, hammer, shelf-place, basketball, sweep, soccer) expose failures of the single-phase + 24-candidate search recipe that the eight scanned tasks (D-019) did not, giving a task on which method variants can be compared.
**Prediction.** promotion_success_rate decrease by at least 0.2 against the reach-v2 arm of R-022 (1.00 on its fresh confirmation cases).
**Protocol.** `python scripts/compare_generation_modes.py --research-mode task-scan --scan-set hard --out-dir logs/research/hard_task_scan_s3300` at git `090501bf79`.
**Result.** {"promotion_success_rate": 0.0, "assembly_first": 0.0, "assembly_repeat": 0.0, "tasks_at_least_0.20_below_reach": ["assembly-v2"], "stopped_early": true, "rows": [{"task": "assembly-v2", "label": "assembly-v2", "generation_seed": 3300, "pool_seed": 3310, "valid": true, "attempts": 1, "stage": "passed", "errors": [], "source_path": "logs/research/hard_task_scan_s3300/structures/assembly-v2.py", "confirmation_success": 0.0, "untrained_probe_success": 0.0, "candidates_with_probe_success": 0, "best_probe_success": 0.0, "episodes": {"probe": 120, "confirmation": 30}}], "repeat": {"task": "assembly-v2", "label": "assembly-v2_repeat", "generation_seed": 3320, "pool_seed": 3321, "valid": true, "attempts": 1, "stage": "passed", "errors": [], "source_path": "logs/research/hard_task_scan_s3300/structures/assembly-v2_repeat.py", "confirmation_success": 0.0, "untrained_probe_success": 0.0, "candidates_with_probe_success": 0, "best_probe_success": 0.0, "episodes": {"probe": 120, "confirmation": 30}}, "schema_audit": [{"task": "assembly-v2", "problems": [], "objects_in_source": 1}, {"task": "peg-insert-side-v2", "problems": [], "objects_in_source": 1}, {"task": "hammer-v2", "problems": [], "objects_in_source": 2}, {"task": "shelf-place-v2", "problems": [], "objects_in_source": 1}, {"task": "basketball-v2", "problems": [], "objects_in_source": 1}, {"task": "soccer-v2", "problems": [], "objects_in_source": 1}], "episodes": {"probe": 240, "confirmation": 60}, "schema_audit_episodes": 6, "llm_calls": 2}; controls {"promotion_success_rate": 1.0, "note": "control is R-022's reach-v2 arm (1.00 on fresh confirmation, D-019), historical"}.
**Decision: supported**. R-024 registered six harder MetaWorld tasks (assembly, peg-insert-side, hammer with a verified second object, shelf-place, basketball, soccer; schemas audited against the simulator before any generation) and scanned them in order with single-phase generation plus 24-candidate search. assembly-v2 scored 0.00 on 30 fresh cases and the predeclared independent repeat also 0.00, so the scan stopped early: supported. Both structures grasp and lift the nut and carry it partway (progress ~0.24) but never seat it on the peg. assembly-v2 is the first task on which the recipe fails reproducibly.
**Next.** Use assembly-v2 as the testbed for structure breadth, fitted lanes (needs an expert demonstration buffer) and placement-phase repair.

## Localize the bottleneck with reciprocal post-lift handoffs (D-021, 2026-10-22)

**Idea R-025.** Assembly fails because of post-lift control rather than irreversible grasp errors, so an expert suffix can complete learner-produced grasps.
**Prediction.** paired_success_difference increase by at least 0.3 against baseline: uninterrupted execution of the two frozen D-020 assembly policies.
**Protocol.** `python scripts/compare_generation_modes.py --assembly-study reciprocal-handoff --pool-seed 3400 --out-dir logs/assembly_handoff_s3400` at git `6b257e9c59`.
**Result.** {"paired_success_difference": 0.3333333333333333, "t95": [0.2438160637708626, 0.422850602895804], "wlt": [20, 0, 10], "full_expert_success": 1.0, "localization_valid": true, "by_structure": {"assembly_first": {"success": {"uninterrupted": 0.0, "learner_then_expert": 0.0, "expert_then_learner": 1.0}, "learner_then_expert_minus_uninterrupted": {"n": 30, "mean": 0.0, "se": 0.0, "low": 0.0, "high": 0.0, "p_value": 1.0, "median": 0.0, "level": 0.95, "wins": 0, "losses": 0, "ties": 30}, "handoff_reach_rate": {"learner_then_expert": 1.0, "expert_then_learner": 0.0}}, "assembly_repeat": {"success": {"uninterrupted": 0.0, "learner_then_expert": 0.6666666666666666, "expert_then_learner": 1.0}, "learner_then_expert_minus_uninterrupted": {"n": 30, "mean": 0.6666666666666666, "se": 0.0875376219064817, "low": 0.4876321275417252, "high": 0.845701205791608, "p_value": 2.1380892374201875e-08, "median": 1.0, "level": 0.95, "wins": 20, "losses": 0, "ties": 10}, "handoff_reach_rate": {"learner_then_expert": 1.0, "expert_then_learner": 0.0}}}, "expert_then_learner_minus_uninterrupted": 1.0, "trigger": {"lift_m": 0.03, "near_m": 0.04, "consecutive_steps": 3}, "policies": {"assembly_first": {"source": "logs/research/hard_task_scan_s3300/structures/assembly-v2.py", "search_seed": 3300, "candidate_index": 17, "parameters_sha256": "19d8f975dd2db319b981b5922e4e5486159fc9f53d01e2c1ec55fba0ad80f24b"}, "assembly_repeat": {"source": "logs/research/hard_task_scan_s3300/structures/assembly-v2_repeat.py", "search_seed": 3320, "candidate_index": 1, "parameters_sha256": "c604a51e44d7af0dcf6d1d7b7870c6067a8294e6f55d55a852996c5372b4d254"}}, "episodes": 210}; controls {"paired_success_difference": 0.0, "note": "control is uninterrupted execution of the same frozen D-020 policy on the same cases; the metric is already learner-then-expert minus uninterrupted"}.
**Decision: supported**. R-025 handed control from each frozen D-020 assembly-v2 policy to the scripted expert once the nut was lifted 3 cm and held for three steps (expert alone: 1.00). Learner-then-expert minus uninterrupted: +0.333 [+0.244, +0.423], averaged over the two fixed structures: supported. The handoff fired in every episode, but rescue split by structure: 0.00 -> 0.67 for one (a post-lift placement bottleneck) and 0.00 -> 0.00 for the other (its grasp configuration is unrecoverable). The reciprocal expert-then-learner arm never triggered and is uninformative.
**Next.** Target the placement phase for recoverable grasps and the grasp configuration otherwise; R-026 (expert-gradient fitting vs numerical search on the same structures) is approved and next.

## Test expert gradients against matched numerical proposals (D-022, 2026-10-22)

**Idea R-026.** Assembly expert gradients can coordinate existing placement parameters more efficiently than independent numerical proposals when checkpoints are selected by rollout success.
**Prediction.** paired_success_difference increase by at least 0.1 against baseline: five-candidate numerical search from each frozen D-020 assembly incumbent.
**Protocol.** `python scripts/compare_generation_modes.py --assembly-study expert-fit-vs-numerical --pool-seed 3410 --out-dir logs/assembly_fitting_s3410` at git `fb3415626f`.
**Result.** {"paired_success_difference": 0.0, "t95": [0.0, 0.0], "wlt": [0, 0, 30], "audit": {"demo_successes": 10, "demos": 10}, "per_structure": {"assembly_first": {"meta": {"source": "logs/research/hard_task_scan_s3300/structures/assembly-v2.py", "search_seed": 3300, "candidate_index": 17, "parameters_sha256": "19d8f975dd2db319b981b5922e4e5486159fc9f53d01e2c1ec55fba0ad80f24b"}, "feasibility": {"ok": true, "parameters_with_gradient": ["align_radius", "closed_threshold", "contact_radius", "grasp_height", "grip_strength", "hover_height", "lift_clearance", "release_radius", "servo_radius", "sharpness", "w_approach", "w_insert", "w_retreat", "w_transport"], "scale_parameters_frozen": []}, "fitted_selected_step": 0, "fitted_trace": [{"step": 0, "success": 0.0, "progress": 0.2306823638843684}, {"step": 100, "success": 0.0, "progress": 0.1450073076361336}, {"step": 300, "success": 0.0, "progress": 0.1330340293889555}, {"step": 600, "success": 0.0, "progress": 0.13224718694742593}, {"step": 1000, "success": 0.0, "progress": 0.12677493535966383}], "numerical_trace": [{"index": 0, "kind": "original", "success": 0.0, "progress": 0.2306823638843684}, {"index": 1, "kind": "local", "success": 0.0, "progress": 0.23399439412979278}, {"index": 2, "kind": "global", "success": 0.0, "progress": -0.2552598007938905}, {"index": 3, "kind": "local", "success": 0.0, "progress": 0.05122326038026977}, {"index": 4, "kind": "global", "success": 0.0, "progress": 0.021336838370337386}], "numerical_pick": 1, "fitted_parameters_sha256": "19d8f975dd2db319b981b5922e4e5486159fc9f53d01e2c1ec55fba0ad80f24b", "success": {"fitted": 0.0, "numerical": 0.0}, "paired": {"n": 30, "mean": 0.0, "se": 0.0, "low": 0.0, "high": 0.0, "p_value": 1.0, "median": 0.0, "level": 0.95, "wins": 0, "losses": 0, "ties": 30}}, "assembly_repeat": {"meta": {"source": "logs/research/hard_task_scan_s3300/structures/assembly-v2_repeat.py", "search_seed": 3320, "candidate_index": 1, "parameters_sha256": "c604a51e44d7af0dcf6d1d7b7870c6067a8294e6f55d55a852996c5372b4d254"}, "feasibility": {"ok": true, "parameters_with_gradient": ["align_radius", "approach_radius", "carry_clearance", "closed_threshold", "damping", "gate_sharpness", "goal_bias", "grasp_height", "grasp_radius", "grip_strength", "hold_radius", "hover_height", "position_gain", "seat_radius", "w_approach", "w_carry", "w_insert"], "scale_parameters_frozen": []}, "fitted_selected_step": 0, "fitted_trace": [{"step": 0, "success": 0.0, "progress": 0.2311429756959144}, {"step": 100, "success": 0.0, "progress": 0.15330594411525614}, {"step": 300, "success": 0.0, "progress": 0.1539720545791206}, {"step": 600, "success": 0.0, "progress": 0.17036976587707248}, {"step": 1000, "success": 0.0, "progress": 0.1762014519281452}], "numerical_trace": [{"index": 0, "kind": "original", "success": 0.0, "progress": 0.2311429756959144}, {"index": 1, "kind": "local", "success": 0.0, "progress": 0.22551679736515823}, {"index": 2, "kind": "global", "success": 0.0, "progress": -0.0003296741902038602}, {"index": 3, "kind": "local", "success": 0.0, "progress": 0.1883136573371853}, {"index": 4, "kind": "global", "success": 0.0, "progress": -0.0003296741902038602}], "numerical_pick": 0, "fitted_parameters_sha256": "c604a51e44d7af0dcf6d1d7b7870c6067a8294e6f55d55a852996c5372b4d254", "success": {"fitted": 0.0, "numerical": 0.0}, "paired": {"n": 30, "mean": 0.0, "se": 0.0, "low": 0.0, "high": 0.0, "p_value": 1.0, "median": 0.0, "level": 0.95, "wins": 0, "losses": 0, "ties": 30}}}, "episodes": {"demo": 10, "probe": 100, "screen": 120}}; controls {"paired_success_difference": 0.0, "note": "control is the five-candidate numerical search from the same frozen starting vector, rollout-selected on the same probe; the metric is already fitted minus numerical"}.
**Decision: refuted**. R-026 fitted the two frozen D-020 assembly-v2 structures to ten successful expert demonstrations (deterministic MSE, checkpoints 0-1000 selected by rollout on 5 probe cases) and compared the winner with a five-candidate numerical search from the same starting vector, on 30 screen cases. Both arms scored 0.00: +0.000 [0, 0], refuted. Imitation never produced a probe success and lowered goal progress at every checkpoint (0.23 to 0.13-0.18), so step 0 was kept; gradients did reach the placement parameters.
**Next.** Parameter-level methods cannot seat the nut with these structures; try structural changes aimed at placement, or more independent structures (R-027).

## A handoff-guided placement edit lets the recoverable assembly structure seat the nut (D-023, 2026-10-22)

**Idea R-027.** Missing post-lift placement logic prevents seating despite a recoverable grasp: one LLM structural edit of the recoverable D-020 parent, given the handoff evidence and keeping its pre-lift logic, raises autonomous success under identical 24-candidate search.
**Prediction.** paired_success_difference increase by at least 0.1 against 24-candidate numerical search of the unchanged recoverable D-020 parent (assembly_repeat).
**Protocol.** `python scripts/compare_generation_modes.py --assembly-study targeted-placement-edit --pool-seed 3420 --out-dir logs/assembly_targeted_edit_s3420` at git `45cddef24e`.
**Result.** {"paired_success_difference": 0.0, "t95": [0.0, 0.0], "wlt": [0, 0, 30], "success": {"parent": 0.0, "edited": 0.0}, "untrained_probe": {"parent": {"index": 0, "kind": "original", "success": 0.0, "progress": 0.21980384871227762}, "edited": {"index": 0, "kind": "original", "success": 0.0, "progress": 0.2253856175408544}}, "best_probe_success": {"parent": 0.0, "edited": 0.0}, "max_probe_progress": {"parent": 0.2227489572776201, "edited": 0.22869763891106293}, "picks": {"parent": 16, "edited": 13}, "episodes": {"probe": 240, "promotion": 60}, "llm_calls": 1}; controls {"paired_success_difference": 0.0, "note": "control is 24-candidate search of the unchanged recoverable parent (assembly_repeat); the metric is already edited minus parent"}.
**Decision: refuted**. R-027 asked gpt-6.1-sol for one structural edit of the recoverable D-020 assembly-v2 parent, given the D-021 handoff evidence (expert completes 0.67 of its lifted grasps), keeping pre-lift logic and rewriting post-lift placement. The edit was valid first time, but under identical 24-candidate search neither the edit nor the parent ever succeeded: +0.000 [0, 0] on 30 promotion cases, refuted. Goal progress barely moved (0.220 to 0.225).
**Next.** Assembly placement is not produced by parameter methods (D-022) or one targeted edit (D-023); consider breadth of independent edits, expert placement traces in the prompt, or the five hard tasks left unscanned.

## 1. Trade assembly parameter-search depth for structural breadth (D-024, 2026-10-22)

**Idea R-028.** Independent assembly structures escape shared placement-topology errors more efficiently than deeper parameter search within one structure.
**Prediction.** paired_success_difference increase by at least 0.2 against logs/next_assembly_breadth_s3500/depth.
**Protocol.** `python scripts/compare_generation_modes.py --assembly-study breadth-vs-depth --pool-seed 3500 --out-dir logs/next_assembly_breadth_s3500` at git `0328164098`.
**Result.** {"paired_success_difference": 0.0, "t95": [0.0, 0.0], "wlt": [0, 0, 30], "success": {"depth": 0.0, "breadth": 0.0}, "picks": {"depth": [0, 23], "breadth": [3, 0]}, "best_probe_by_structure": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.0}, "max_probe_progress_by_structure": {"0": 0.2461492584355592, "1": 0.2409032134716953, "2": -0.0003176347988724848, "3": 0.24875213661142972}, "valid_structures": [0, 1, 2, 3], "episodes": {"probe": 210, "promotion": 60}, "llm_calls": 4}; controls {"paired_success_difference": 0.0, "note": "control is 24-candidate search of structure 0 (seed 3500); the metric is already breadth minus depth"}.
**Decision: refuted**. R-028 compared structural breadth (6 search candidates from each of four fresh gpt-6.1-sol assembly-v2 structures) with depth (24 candidates of one), at an equal 24-candidate budget. No candidate of any structure succeeded on the probe; both winners scored 0.00 on 30 promotion cases: +0.000 [0, 0], refuted. Three of the four new structures plateau at the same ~0.24 progress as the D-020 pair: the plain generator reproducibly writes assembly controllers that lift but cannot seat the nut.
**Next.** R-029: does trajectory evidence (successful expert placement traces) let an edit seat the nut?

## 2. Give assembly edits successful expert placement traces (D-025, 2026-10-22)

**Idea R-029.** Successful expert state-action sequences expose nut-to-peg placement geometry that handoff summaries and parameter-level imitation do not communicate to the symbolic editor.
**Prediction.** paired_success_difference increase by at least 0.2 against logs/next_assembly_traces_s3520/summary_only.
**Protocol.** `python scripts/compare_generation_modes.py --assembly-study trace-edit-suite --pool-seed 3520 --out-dir logs/next_assembly_traces_s3520` at git `f89ad2df66`.
**Result.** {"paired_success_difference": 0.0, "t95": [0.0, 0.0], "per_block": {"0": {"n": 30, "mean": 0.0, "se": 0.0, "low": 0.0, "high": 0.0, "p_value": 1.0, "median": 0.0, "level": 0.95, "wins": 0, "losses": 0, "ties": 30}, "1": {"n": 30, "mean": 0.0, "se": 0.0, "low": 0.0, "high": 0.0, "p_value": 1.0, "median": 0.0, "level": 0.95, "wins": 0, "losses": 0, "ties": 30}}, "success": {"b0_summary_only": 0.0, "b0_trace": 0.0, "b1_summary_only": 0.0, "b1_trace": 0.0}, "edit_validity": {"b0_summary_only": true, "b0_trace": false, "b1_summary_only": true, "b1_trace": false}, "trace_edit_errors": {"b0_trace": ["Error: policy failed validation.\n  [gate_not_a_lever] gate 'post_opened' is recorded, but forcing it to 0 or 1 never changes the action on any diagnostic state. Assign the returned value and use it: post_opened = self.record_gate('post_opened', post_opened)"], "b1_trace": ["Error: policy failed validation.\n  [gate_not_a_lever] gate 'open_enough' is recorded, but forcing it to 0 or 1 never changes the action on any diagnostic state. Assign the returned value and use it: open_enough = self.record_gate('open_enough', open_enough)\n  [gate_not_a_lever] gate 'placement_near' is recorded, but forcing it to 0 or 1 never changes the action on any diagnostic state. Assign the returned value and use it: placement_near = self.record_gate('placement_near', placement_near)\n  [gate_not_a_lever] gate 'settled' is recorded, but forcing it to 0 or 1 never changes the action on any diagnostic state. Assign the returned value and use it: settled = self.record_gate('settled', settled)"]}, "handoff_demo_successes": 1, "episodes": {"demo": 5, "probe": 240, "promotion": 60}, "llm_calls": 4}; controls {"paired_success_difference": 0.0, "note": "control is the summary-only edit in the same block (same handoff summary, no traces), searched identically; the metric is already trace minus summary-only"}.
**Decision: refuted**. R-029 compared assembly-v2 placement edits of the recoverable D-020 parent given the handoff summary alone or the summary plus a successful expert continuation trace (only 1 of 5 handoffs succeeded), in two blocks under identical 24-candidate search. Both summary-only edits were valid and scored 0.00; both trace edits failed validation on gate_not_a_lever (unused post-lift gates) and, with no correction allowed, score 0.00. Trace minus summary-only: +0.000 [0, 0]; refuted for the procedure as specified, while the trace mechanism itself was never evaluated.
**Next.** Rerun with one validity correction and more successful continuations if trace evidence is still worth testing; otherwise move to the remaining hard tasks.

## 3. Test insertion specificity with an expert-calibrated hard-task scan (D-026, 2026-10-22)

**Idea R-030.** Assembly exposes a precision-insertion limitation rather than generic manipulation difficulty, so searched structures will solve impact tasks more often than side insertion.
**Prediction.** screen_success_rate increase by at least 0.5 against peg-insert-side arm in logs/next_contact_regimes_s3540.
**Protocol.** `python scripts/compare_generation_modes.py --research-mode contact-regime-scan --out-dir logs/next_contact_regimes_s3540` at git `2c0eecc51a`.
**Result.** {"expert_calibration": {"peg-insert-side-v2": 0.6333333333333333, "hammer-v2": 1.0, "soccer-v2": 0.9}, "episodes": {"expert": 90, "probe": 0, "screen": 0}, "llm_calls": 0, "outcome": "inconclusive: expert below 0.9 on {'peg-insert-side-v2': 0.6333333333333333}; no generation call made"}; controls {"note": "no learner arm was run; the predeclared calibration gate (expert >= 0.90 on every contrast task) failed"}.
**Decision: inconclusive**. R-030 planned an expert-calibrated contrast of learner success on impact tasks (hammer, soccer) against insertion (peg-insert-side). MetaWorld's scripted expert scored 1.00 on hammer, 0.90 on soccer and only 0.63 on peg-insert-side on the frozen screen cases, failing the predeclared >= 0.90 gate, so no structure was generated: inconclusive, 90 episodes, 0 LLM calls.
**Next.** Replace the insertion control with a task whose expert is reliable, or scan hammer and soccer as further testbeds.

## Trace-conditioned assembly edits with one validity correction (D-027, 2026-10-22)

**Idea R-031.** Successful expert placement traces let an edit of the recoverable assembly parent seat the nut, once the edit is allowed the pipeline's standard single validity correction (D-025's trace edits failed only on gate_not_a_lever).
**Prediction.** paired_success_difference increase by at least 0.2 against summary-only edits of the same parent in the same blocks, also allowed one validity correction, under identical 24-candidate search.
**Protocol.** `python scripts/compare_generation_modes.py --assembly-study trace-edit-suite --pool-seed 3600 --demo-cases 10 --min-continuations 2 --allow-correction --out-dir logs/next_assembly_traces_s3600` at git `d4e32d0dea`.
**Result.** {"paired_success_difference": 1.0, "t95": [1.0, 1.0], "wlt": [30, 0, 0], "per_block": {"0": {"n": 30, "mean": 1.0, "se": 0.0, "low": 1.0, "high": 1.0, "p_value": 0.0, "median": 1.0, "level": 0.95, "wins": 30, "losses": 0, "ties": 0}, "1": {"n": 30, "mean": 1.0, "se": 0.0, "low": 1.0, "high": 1.0, "p_value": 0.0, "median": 1.0, "level": 0.95, "wins": 30, "losses": 0, "ties": 0}}, "success": {"b0_summary_only": 0.0, "b0_trace": 1.0, "b1_summary_only": 0.0, "b1_trace": 1.0}, "untrained_probe": {"b0_summary_only": {"index": 0, "kind": "original", "success": 0.0, "progress": 0.23443630674964747}, "b0_trace": {"index": 0, "kind": "original", "success": 1.0, "progress": 0.10174682827477448}, "b1_summary_only": {"index": 0, "kind": "original", "success": 0.0, "progress": 0.234686413473227}, "b1_trace": {"index": 0, "kind": "original", "success": 1.0, "progress": 0.10204343822059543}}, "probe_candidates_succeeding": {"b0_summary_only": 0, "b0_trace": 13, "b1_summary_only": 0, "b1_trace": 10}, "edits": {"b0_summary_only": {"valid": true, "corrected": true, "first_attempt_passed": false, "llm_calls": 2}, "b0_trace": {"valid": true, "corrected": true, "first_attempt_passed": false, "llm_calls": 2}, "b1_summary_only": {"valid": true, "corrected": false, "first_attempt_passed": true, "llm_calls": 1}, "b1_trace": {"valid": true, "corrected": true, "first_attempt_passed": false, "llm_calls": 2}}, "handoff_demo_successes": 6, "episodes": {"demo": 10, "probe": 480, "promotion": 120}, "llm_calls": 7}; controls {"paired_success_difference": 0.0, "note": "control is the summary-only edit in the same block with the same correction allowance and search; the metric is already trace minus summary-only"}.
**Decision: supported**. R-031 repeated R-029 with the pipeline's standard single validity correction and ten learner-to-expert handoff demonstrations (6/10 successful continuations). In both blocks the edit of the recoverable D-020 assembly-v2 parent that saw two successful expert placement traces scored 1.00 on 30 fresh promotion cases, while the summary-only edit scored 0.00: +1.000 [1.0, 1.0], 30/0/0, supported. The trace edits already succeed untrained (probe 1.0 at step 0). Three of four edits needed their one correction, which is why R-029 never tested the traces. The code reads observations through the schema; constants are centimetre-scale offsets. Narrow: one parent, one demonstration set, two blocks.
**Next.** Generalise trace-conditioned editing: another parent or fresh structure, another failing task, and as a repair step in the two-lane pipeline where an expert exists.

## Can expert-only placement traces replace learner handoffs? (D-028, 2026-10-22)

**Idea R-032.** Expert-only post-lift traces encode sufficient placement geometry to improve assembly edits without showing expert recovery from learner-visited states.
**Prediction.** paired_success_difference increase by at least 0.5 against matched expert-demonstration-summary-only edits of D-020:recoverable.
**Protocol.** `python scripts/compare_generation_modes.py --assembly-study expert-only-traces --pool-seed 3710 --out-dir logs/trace_generalisation/expert_only_s3710` at git `ea89867353`.
**Result.** {"paired_success_difference": 1.0, "t95": [1.0, 1.0], "wlt": [30, 0, 0], "per_block": {"0": {"n": 30, "mean": 1.0, "se": 0.0, "low": 1.0, "high": 1.0, "p_value": 0.0, "median": 1.0, "level": 0.95, "wins": 30, "losses": 0, "ties": 0}, "1": {"n": 30, "mean": 1.0, "se": 0.0, "low": 1.0, "high": 1.0, "p_value": 0.0, "median": 1.0, "level": 0.95, "wins": 30, "losses": 0, "ties": 0}}, "success": {"b0_summary_only": 0.0, "b0_expert_only_trace": 1.0, "b1_summary_only": 0.0, "b1_expert_only_trace": 1.0}, "edits": {"b0_summary_only": {"valid": true, "corrected": false, "llm_calls": 1}, "b0_expert_only_trace": {"valid": true, "corrected": true, "llm_calls": 2}, "b1_summary_only": {"valid": true, "corrected": false, "llm_calls": 1}, "b1_expert_only_trace": {"valid": true, "corrected": false, "llm_calls": 1}}, "demo_successes": 10, "episodes": {"demo": 10, "eval": 120}, "llm_calls": 5}; controls {"paired_success_difference": 0.0, "note": "control is the summary-only edit (same expert-demonstration summary, no trace rows) in the same block; the metric is already trace minus summary-only"}.
**Decision: supported**. R-032 replaced D-027's learner handoff traces with expert-only demonstrations (10/10 successful; rows from when the nut first rose 3 cm). Edits of the recoverable assembly-v2 parent that saw two such traces scored 1.00 on 30 fresh cases at their declared values (no search) in both blocks; summary-only edits scored 0.00: +1.000 [1.0, 1.0], supported. Learner-state alignment is not needed: plain expert demonstrations carry the placement geometry the edit lacks.
**Next.** Test transfer to hammer (R-033) and from a fresh generated parent; integrate trace-conditioned editing as a repair step.

## Transfer trace-guided structural repair to a gated hammer plateau (D-029, 2026-10-22)

**Idea R-033.** Successful expert hammer continuations expose missing strike geometry and timing that structural edits can incorporate when parameter search cannot produce success.
**Prediction.** paired_success_difference increase by at least 0.3 against matched summary-only edits of the freshly generated hammer plateau parent.
**Protocol.** `python scripts/compare_generation_modes.py --assembly-study hammer-transfer --pool-seed 3720 --out-dir logs/trace_generalisation/hammer_s3720` at git `585e002286`.
**Result.** {"gates": {"probe_candidates_succeeding": 23}, "parent": {"valid": true, "attempts": 1}, "episodes": {"probe": 120, "gate": 0, "demo": 0, "eval": 0}, "llm_calls": 1, "outcome": "inconclusive: no generator plateau (a search candidate succeeded on the probe)"}; controls {"note": "no edit arm ran: the predeclared plateau gate failed"}.
**Decision: inconclusive**. R-033 tried to test trace-guided editing on a fresh hammer-v2 plateau parent, but the plateau gate failed: 24-candidate search of the fresh structure already produced probe successes, so no handoff or edit stage ran (inconclusive, 120 episodes, 1 call). Hammer is not a plateau for single-phase generation plus search; assembly-v2 remains the only plateau found.
**Next.** Scan shelf-place, basketball and soccer for another plateau, or test from a fresh assembly parent.

## Transport expert-trace edits across assembly parent structures (D-030, 2026-10-22)

**Idea R-034.** Whole-controller edits conditioned on successful expert placement traces can repair upstream phase alignment as well as seating, removing dependence on a recoverable assembly parent.
**Prediction.** paired_success_difference increase by at least 0.5 against Matched summary-only whole-controller edits of the same three parents.
**Protocol.** `python scripts/compare_generation_modes.py --assembly-study parent-transfer --pool-seed 3800 --out-dir logs/research/assembly_parent_transport` at git `4f2d8113e8`.
**Result.** {"paired_success_difference": 1.0, "parent_level_t95": [1.0, 1.0], "placement_level_t95": [1.0, 1.0], "placement_wlt": [30, 0, 0], "per_parent": {"archived_unrecoverable": {"trace": 1.0, "summary": 0.0, "mean": 1.0, "wins": 30, "losses": 0, "ties": 0}, "fresh1": {"trace": 1.0, "summary": 0.0, "mean": 1.0, "wins": 30, "losses": 0, "ties": 0}, "fresh2": {"trace": 1.0, "summary": 0.0, "mean": 1.0, "wins": 30, "losses": 0, "ties": 0}}, "parent_probe_success": {"archived_unrecoverable": 0.0, "fresh1": 0.0, "fresh2": 0.0}, "edits": {"archived_unrecoverable_summary": {"valid": true, "corrected": true, "llm_calls": 2}, "archived_unrecoverable_expert_trace": {"valid": true, "corrected": false, "llm_calls": 1}, "fresh1_summary": {"valid": true, "corrected": true, "llm_calls": 2}, "fresh1_expert_trace": {"valid": true, "corrected": false, "llm_calls": 1}, "fresh2_summary": {"valid": true, "corrected": false, "llm_calls": 1}, "fresh2_expert_trace": {"valid": true, "corrected": false, "llm_calls": 1}}, "demo_successes": 10, "episodes": {"demo": 10, "probe": 15, "eval": 180}, "llm_calls": 10}; controls {"paired_success_difference": 0.0, "note": "control is the summary-only whole-controller edit of the same parent; the metric is already trace minus summary"}.
**Decision: supported**. R-034 applied expert-trace whole-controller editing (ten expert-only demonstrations, two traces, one validity correction) to three assembly-v2 parents that all score 0.00: the D-020 structure whose grasps the expert could not recover, and two fresh generations. All three trace edits scored 1.00 and all three summary-only edits 0.00 on 30 fresh cases: +1.000 per parent, supported. The trace effect does not need a recoverable parent; edits rewrite grasping as well as placement when shown the expert trajectory.
**Next.** Integrate trace-guided repair into the two-lane pipeline (R-036) and look for a second plateau task (R-035).

## Find another recoverable terminal-phase plateau (D-031, 2026-10-22)

**Idea R-035.** At least one unscanned task has a terminal-geometry bottleneck that expert takeover can complete from learner states but fixed-budget parameter search cannot.
**Prediction.** promotion_success_rate decrease by at least 0.9 against Task-matched scripted expert on the same 30 confirmation cases.
**Protocol.** `python scripts/compare_generation_modes.py --research-mode recoverable-plateau-scan --out-dir logs/research/recoverable_plateau_scan` at git `04e8b873fa`.
**Result.** {"promotion_success_rate": 1.0, "note_on_metric": "no task reached confirmation; every task's search already succeeded on the probe", "rows": {"shelf-place-v2": {"expert_success": 1.0, "probe_candidates_succeeding": 22, "status": "not a plateau (probe success)", "attempts": 1}, "basketball-v2": {"expert_success": 1.0, "probe_candidates_succeeding": 20, "status": "not a plateau (probe success)", "attempts": 1}, "soccer-v2": {"expert_success": 0.9, "probe_candidates_succeeding": 23, "status": "not a plateau (probe success)", "attempts": 1}}, "recoverable_plateau": null, "expert_qualified": ["shelf-place-v2", "basketball-v2", "soccer-v2"], "episodes": {"expert": 90, "probe": 360, "confirmation": 0, "takeover": 0}, "llm_calls": 3}; controls {"note": "control is each task's scripted expert on its 30 confirmation cases (all >= 0.90)"}.
**Decision: refuted**. R-035 scanned shelf-place, basketball and soccer for a second recoverable plateau. All three experts qualified (1.00, 1.00, 0.90), and one single-phase structure plus 24-candidate search already succeeded on every task's probe (22, 20 and 23 of 24 candidates), so no task is a plateau: refuted. Across all scans, assembly-v2 is the only MetaWorld task where single-phase generation plus search plateaus, and expert-trace editing solves it (D-027, D-028, D-030).
**Next.** Integrate trace-guided repair into the two-lane pipeline (R-036).

## Plateau-triggered trace repair versus exploration restart (D-032, 2026-10-22)

**Idea R-036.** Expert trajectories supply missing phase geometry at a plateau, enabling a repair that survives the pipeline's selection and promotion gates.
**Prediction.** paired_success_difference increase by at least 0.3 against the budget-matched no-trace fresh-generation continuation from the same checkpoint; final incumbents compared on development cases 10-39.
**Protocol.** `python scripts/compare_generation_modes.py --assembly-study plateau-repair-pipeline --pool-seed 3820 --out-dir logs/research/two_lane_plateau_repair/block_0` at git `b50af881bd`.
**Result.** {"paired_success_difference": 1.0, "t95": [1.0, 1.0], "wlt": [30, 0, 0], "final_incumbent_success": {"trace_repair": 1.0, "fresh_generation": 0.0}, "finals": {"trace_repair": {"incumbent": "trace_repair", "promotion": {"promoted": true, "mean_difference": 1.0, "low": 1.0, "high": 1.0}, "declared_value_probe": {"success": 1.0, "progress": 0.08253241136780043}, "selected_probe": {"success": 1.0, "progress": 0.09226703115400925}, "finalist_promotion_success": 1.0}, "fresh_generation": {"incumbent": "null_incumbent", "promotion": {"promoted": false, "mean_difference": 0.0, "low": 0.0, "high": 0.0}, "declared_value_probe": {"success": 0.0, "progress": 0.22121548515214995}, "selected_probe": {"success": 0.0, "progress": 0.222914983416111}, "finalist_promotion_success": 0.0}}, "prefix": {"challenger": "exploration", "promotion": {"promoted": false, "mean_difference": 0.0, "low": 0.0, "high": 0.0}, "incumbent": "null_incumbent"}, "exploration_probe_successes": 0, "screen": {"bank:simple_damped_push": {"success": 0.0, "progress": 0.03990953532338189}, "bank:simple_reach_push": {"success": 0.0, "progress": -0.06194647986836192}, "bank:simple_reach_push_damped": {"success": 0.0, "progress": -0.11410919578280854}, "bank:simple_reach_push_shared_gain": {"success": 0.0, "progress": -0.09840057396764475}, "bank:simple_reach_push_sharp_gate": {"success": 0.0, "progress": -0.07570856604671901}, "exploration": {"success": 0.0, "progress": 0.22303692036055806}}, "demo_successes": 10, "episodes": {"bank_probe": 25, "exploration_probe": 120, "incumbent": 30, "demo": 10, "branch_probe": 240, "finalist": 60}, "llm_calls": 3}; controls {"paired_success_difference": 0.0, "note": "control is the fresh no-trace generation continuation from the same checkpoint; the metric is already the trace-repair final incumbent minus the fresh-branch final incumbent"}.
**Decision: supported**. R-036 piloted plateau-triggered trace repair inside the pipeline's selection and promotion on assembly-v2 (one block). The exploration lane plateaued (0/24 probe successes) and no prefix candidate passed promotion, so the incumbent stayed at 0.00. Forking at that checkpoint, the trace-repair branch (whole-controller edit with two expert-only traces) produced a controller at 1.00 that passed the strict paired promotion rule; the fresh no-trace generation stayed at 0.00. Final incumbents: +1.000 [1.0, 1.0], supported as a pilot. The pipeline steps were reproduced in a study mode, not run_full_evolution.py.
**Next.** Build trace repair into the orchestrator with a plateau trigger and expert-demonstration source, generalise run_full_evolution.py beyond push-v2, and confirm with an independent block.

## Trace repair inside the real orchestrator: an end-to-end assembly-v2 run of run_full_evolution.py (D-033, 2026-10-22)

**Idea R-037.** Building plateau-triggered expert-trace repair into EvolutionOrchestrator (with task-generic manifests and an expert-demonstration source) makes run_full_evolution.py itself solve assembly-v2, reproducing D-032 without the study-mode emulation.
**Prediction.** paired_success_difference increase by at least 0.3 against the same orchestrator run with the trace-repair step disabled (exploration lane v2 only), same seeds and manifests; final incumbents on fresh development cases.
**Protocol.** `python scripts/run_full_evolution.py --config config/assembly_two_lane_repair.yaml --seed <3900|3910> --log-dir logs/r037/block<A|B>_repair_on [--dev-pool-seed 3912 --demo-pool-seed 3913] and the same with --seed <3901|3911> --repair disabled; then scripts/evaluate_on_fresh_manifest.py --env-name assembly-v2 --pool-seed <3904|3914> --episodes 30 --frozen <manifest>` at git `d526eefb54`.
**Result.** {"paired_success_difference": 1.0, "across_block_interval_low": 1.0, "across_block_interval_high": 1.0, "block_A_paired_success_difference": 1.0, "block_B_paired_success_difference": 1.0, "repair_on_fresh_success_A": 1.0, "repair_on_fresh_success_B": 1.0, "plateau_triggers": 4, "episodes_total": 1440, "llm_calls_total": 8}; controls {"paired_success_difference": 0.0, "repair_off_fresh_success_A": 0.0, "repair_off_fresh_success_B": 0.0}.
**Decision: supported**. R-037 integrated plateau-triggered expert-trace repair into EvolutionOrchestrator and made run_full_evolution.py task-generic (manifests from pool seeds, expert demonstrations via EXPERT_POLICY_MAP), passing a 21-test mocked gate before any spend. Live on assembly-v2, two blocks of paired repair-on/repair-off production runs: all four runs hit the plateau trigger; with repair on, demonstrations qualified 10/10 and the trace repair became the final incumbent through the unchanged strict promotion rule (30/30 on promotion cases), scoring 30/30 on frozen fresh cases in both blocks, while every repair-off candidate scored 0/30 there (the repair-off incumbent stayed null). Paired fresh difference +1.000 in each block (n=2 blocks, degenerate interval), 1560 episodes and 8 calls, every record traceable. The real pipeline reproduces D-032 without the study-mode emulation.
**Next.** Make plateau-triggered trace repair a standard pipeline option; check the trigger stays silent and costless on non-plateau tasks; look for a second plateau task.

## 1. Audit transient hits versus sustained axial-roll success (D-034, 2026-10-27)

**Direction.** Continue researching the single-step exploration pipeline with plateau-triggered trace repair, extended to pen spinning on the Shadow Hand (Isaac Lab Isaac-ShadowHandSpin-Stage{1,2,3}-Direct-v0, vendored in lares/envs/isaac_pen_spin; 157-d full obs, 20-d action, 60 Hz, 360-step episodes). Expert demonstrations: first try a hand-written scripted spinning controller; also search online for an existing trained pen-spinning policy; if neither qualifies, train an RL expert (rsl_rl PPO on the box). No budget ceiling on GPU hours or LLM calls. Known gaps to turn into questions: the env's spin_rate is roll about the pen's own long axis (not end-over-end twirl), success is per step with no episode rule, and the pipeline (manifests, obs schema, runner, records) is MetaWorld-only today. Debug issues, improve the pipeline, and turn unexpected findings into concrete research questions; document results and limitations.
**Idea R-038.** Transient axial-spin bursts inflate any-hit episode success because a pen can briefly satisfy the native predicate before losing stable in-hand rotation.
**Prediction.** success_rate decrease by at least 0.2 against baseline.
**Protocol.** `gpu run 'source ~/IsaacLab/source_env.sh && python scripts/run_shadowhand_spin_stages.py --config config/shadowhand_spin_stages.yaml --research-protocol episode_rule_audit'` at git `d8cbfa35d4`.
**Result.** {"success_rate": 0.0, "success_rate_any_hit": 0.9833333333333333, "success_rate_sustained": 0.0, "paired_difference_sustained_minus_any_hit": -0.9833333333333333, "paired_difference_ci95": [-1.0166832563014712, -0.9499834103651955], "episodes": 60, "llm_calls": 0, "per_stage_controller": {"stage1/neutral_pose": {"any_hit_rate": 1.0, "sustained_rate": 0.0, "completed_rate": 0.0, "mean_length": 67.1, "mean_tail_hit_fraction": 0.6078331328679392, "mean_abs_spin": 11.339570045471191, "mean_transverse": 1.7387796968221665}, "stage1/open_loop_digit_oscillation": {"any_hit_rate": 1.0, "sustained_rate": 0.0, "completed_rate": 1.0, "mean_length": 359.0, "mean_tail_hit_fraction": 0.8400000000000001, "mean_abs_spin": 3.872316098213196, "mean_transverse": 0.44259398579597475}, "stage2/neutral_pose": {"any_hit_rate": 1.0, "sustained_rate": 0.0, "completed_rate": 0.0, "mean_length": 87.2, "mean_tail_hit_fraction": 0.6109787798408488, "mean_abs_spin": 10.12760033607483, "mean_transverse": 1.6471677124500275}, "stage2/open_loop_digit_oscillation": {"any_hit_rate": 1.0, "sustained_rate": 0.0, "completed_rate": 1.0, "mean_length": 359.0, "mean_tail_hit_fraction": 0.5633333333333334, "mean_abs_spin": 4.306938505172729, "mean_transverse": 0.3596875831484795}, "stage3/neutral_pose": {"any_hit_rate": 0.9, "sustained_rate": 0.0, "completed_rate": 0.1, "mean_length": 108.9, "mean_tail_hit_fraction": 0.39016822560379727, "mean_abs_spin": 9.136602059006691, "mean_transverse": 1.1046076148748398}, "stage3/open_loop_digit_oscillation": {"any_hit_rate": 1.0, "sustained_rate": 0.0, "completed_rate": 0.9, "mean_length": 328.1, "mean_tail_hit_fraction": 0.38666666666666666, "mean_abs_spin": 4.160055983066559, "mean_transverse": 0.7201119571924209}}, "neutral_pose_first_hit_step_median": 7}; controls {"baseline": "any_hit reducer on identical trajectories (same 60 episodes, offline)", "controllers": "neutral_pose (default joint targets) and open_loop_digit_oscillation (amp 0.35, 1.5 Hz, staggered phases), frozen in lares/envs/isaac_pen_spin/controllers.py before the probe cases ran", "manifests": ["config/manifests/pen_spin_stage1_probe_s4000.yaml", "config/manifests/pen_spin_stage2_probe_s4001.yaml", "config/manifests/pen_spin_stage3_probe_s4002.yaml"], "reproducibility": "one rollout per process; a first rollout is bit-identical across fresh processes (debug seed 4099), a second rollout in the same process is not", "success_rate": 0.9833333333333333}.
**Decision: supported**. R-038 built the first manifest-driven evaluation path for the Shadow Hand pen-spin task (frozen reset placements written into the simulator, a named 157-d observation layout, per-step traces) and audited the episode success rule the env leaves undefined. On 60 identical trajectories (3 stages x 10 probe cases x neutral hand and open-loop finger oscillator), any-hit success was 0.983 and sustained success (time-out without a fall, predicate true on all of the final 60 steps) 0.000: paired -0.983 [-1.017, -0.950], supported. Any-hit is degenerate: a neutral hand scores 1.0 because the dropped pen rolls on landing at steps 5-8, while it drops the pen in 29/30 episodes. The oscillator keeps the pen in hand to time-out in 29/30 episodes and passes 84/56/39% of tail steps (stage 1/2/3) at about 4 rad/s of roll, but never all 60, so whether 'all steps' is too strict for a periodic gait is open. Two infrastructure findings: the previous run's action leaked into the next run's observation, and a second rollout in one Isaac process is not reproducible (the first is bit-identical across processes), so protocols run one process per cell. The task rewards roll about the pen's own axis; end-over-end rotation stays below 0.8 rad/s.
**Next.** Re-triage R-039 (scripted feedback gait) with the sustained rule frozen as the qualification reducer, and report tail-hit fraction beside it so a periodic gait's near-misses are visible. Open question to propose: is 'all 60 steps' too strict for any periodic gait, i.e. should the rule be a tail-fraction or mean-rate criterion? Decide it before any expert is qualified, not after.

## 2. Can feedback-controlled finger gaiting qualify a scripted expert? (D-035, 2026-10-27)

**Direction.** Continue researching the single-step exploration pipeline with plateau-triggered trace repair, extended to pen spinning on the Shadow Hand (Isaac Lab Isaac-ShadowHandSpin-Stage{1,2,3}-Direct-v0, vendored in lares/envs/isaac_pen_spin; 157-d full obs, 20-d action, 60 Hz, 360-step episodes). Expert demonstrations: first try a hand-written scripted spinning controller; also search online for an existing trained pen-spinning policy; if neither qualifies, train an RL expert (rsl_rl PPO on the box). No budget ceiling on GPU hours or LLM calls. Known gaps to turn into questions: the env's spin_rate is roll about the pen's own long axis (not end-over-end twirl), success is per step with no episode rule, and the pipeline (manifests, obs schema, runner, records) is MetaWorld-only today. Debug issues, improve the pipeline, and turn unexpected findings into concrete research questions; document results and limitations.
**Idea R-039.** Feedback that alternates finger contacts while correcting pen-centre drift sustains axial rolling better than an open-loop finger oscillator.
**Prediction.** paired_success_difference increase by at least 0.2 against baseline.
**Protocol.** `gpu run 'source ~/IsaacLab/source_env.sh && python scripts/run_shadowhand_spin_stages.py --config config/shadowhand_spin_stages.yaml --research-protocol scripted_expert_qualification'` at git `627ea272e1`.
**Result.** {"paired_success_difference": 0.4111111111111111, "paired_success_difference_ci95": [0.30747896415779924, 0.5147432580644229], "pooled_promotion_pairs": 90, "feedback_gait_promotion_success_pooled": 0.41111111111111115, "oscillator_promotion_success_pooled": 0.0, "per_stage_promotion": {"stage1": {"feedback_gait_successes_of_30": 14, "feedback_gait_success": 0.4666666666666667, "oscillator_success": 0.0, "paired_difference": 0.4666666666666667, "paired_ci95": [0.27719432002040323, 0.6561390133129301], "feedback_gait_tail_hit_fraction": 0.8283333333333334, "oscillator_tail_hit_fraction": 0.8038888888888888, "feedback_gait_completed": 0.8666666666666667, "feedback_gait_mean_abs_spin": 24.115314372380574, "feedback_gait_mean_transverse": 3.560781545440356}, "stage2": {"feedback_gait_successes_of_30": 16, "feedback_gait_success": 0.5333333333333333, "oscillator_success": 0.0, "paired_difference": 0.5333333333333333, "paired_ci95": [0.3438609866870699, 0.7228056799795968], "feedback_gait_tail_hit_fraction": 0.84, "oscillator_tail_hit_fraction": 0.5011111111111112, "feedback_gait_completed": 0.9333333333333333, "feedback_gait_mean_abs_spin": 25.83052158355713, "feedback_gait_mean_transverse": 4.759572259585062}, "stage3": {"feedback_gait_successes_of_30": 7, "feedback_gait_success": 0.23333333333333334, "oscillator_success": 0.0, "paired_difference": 0.23333333333333334, "paired_ci95": [0.07270033612932628, 0.3939663305373404], "feedback_gait_tail_hit_fraction": 0.5072222222222221, "oscillator_tail_hit_fraction": 0.3753046594982079, "feedback_gait_completed": 0.8666666666666667, "feedback_gait_mean_abs_spin": 21.40856667757034, "feedback_gait_mean_transverse": 3.486862709124883}}, "per_stage_screen_feedback_gait_success": {"stage1": 0.4, "stage2": 0.4, "stage3": 0.5}, "qualifies": {"stage1": {"open_loop_digit_oscillation": false, "feedback_gait": false}, "stage2": {"open_loop_digit_oscillation": false, "feedback_gait": false}, "stage3": {"open_loop_digit_oscillation": false, "feedback_gait": false}}, "promotes_per_stage": {"stage1": true, "stage2": true, "stage3": true}, "episodes": {"evaluation": 240, "reproducibility_control": 480, "tuning_debug": 1720}, "llm_calls": 0}; controls {"paired_success_difference": 0.0, "baseline": "frozen R-038 open-loop oscillator (D-034) on the same 40-case development manifests, own process per stage", "reproducibility_control": {"stage1/open_loop_digit_oscillation": {"bit_identical": false, "spin_rate_diverged_envs": 1, "center_dist_diverged_envs": 1, "union_diverged_envs": 1, "sustained_outcome_flips": 0, "passes": true}, "stage1/feedback_gait": {"bit_identical": false, "spin_rate_diverged_envs": 1, "center_dist_diverged_envs": 1, "union_diverged_envs": 1, "sustained_outcome_flips": 0, "passes": true}, "stage2/open_loop_digit_oscillation": {"bit_identical": false, "spin_rate_diverged_envs": 2, "center_dist_diverged_envs": 2, "union_diverged_envs": 2, "sustained_outcome_flips": 0, "passes": true}, "stage2/feedback_gait": {"bit_identical": true, "spin_rate_diverged_envs": 0, "center_dist_diverged_envs": 0, "union_diverged_envs": 0, "sustained_outcome_flips": 0, "passes": true}, "stage3/open_loop_digit_oscillation": {"bit_identical": false, "spin_rate_diverged_envs": 2, "center_dist_diverged_envs": 2, "union_diverged_envs": 2, "sustained_outcome_flips": 0, "passes": true}, "stage3/feedback_gait": {"bit_identical": false, "spin_rate_diverged_envs": 1, "center_dist_diverged_envs": 1, "union_diverged_envs": 1, "sustained_outcome_flips": 0, "passes": true}}}.
**Decision: supported**. R-039 tested whether a scripted feedback finger gait could qualify as the pen-spin expert. Tuned on a debug seed only and frozen before the development cases, the gait regulates roll rate through the long fingers' flexion (the fingers act as a variable slope), which turns the open-loop oscillator's rocking into fast one-directional spin (21-26 rad/s mean). On 90 pooled promotion pairs it beat the frozen R-038 oscillator by +0.411 [0.307, 0.515] sustained success, promoting on every stage: supported. It does not qualify as an expert on any stage: 14/30, 16/30 and 7/30 sustained successes against the 27/30 rule, with failures from drops and tail dips, not speed (stage 3's 0.08 m tolerance costs most); it also tumbles the pen end over end (3.5-4.8 rad/s). The frozen bit-identity reproducibility control failed (GPU PhysX at batch 40 is not bit-deterministic across processes: 1-2/40 environments diverge after step 78) and was amended before any development case ran to zero outcome flips and at most 4/40 diverged environments, which passed (0 flips in 240 pairs). The user's fallback, a PPO expert, now applies to all three stages.
**Next.** Re-triage R-040 (PPO expert) for all three stages, recalibrated as argued in its round-1 triage: budget in env steps with many parallel envs, success and roll logged separately from return, a check that the reward (centring 3.0 vs spin 0.15 at stage 1) does not teach holding still, and the D-034 reducer for qualification. Separate idea: should the frozen gait's 47-53% sustained success seed the policy-search pipeline as a non-expert prior, or the tail-fraction near-misses inform a less brittle reducer?

## 3. Conditional PPO fallback: does reward optimization yield a usable expert? (D-036, 2026-10-27)

**Direction.** Continue researching the single-step exploration pipeline with plateau-triggered trace repair, extended to pen spinning on the Shadow Hand (Isaac Lab Isaac-ShadowHandSpin-Stage{1,2,3}-Direct-v0, vendored in lares/envs/isaac_pen_spin; 157-d full obs, 20-d action, 60 Hz, 360-step episodes). Expert demonstrations: first try a hand-written scripted spinning controller; also search online for an existing trained pen-spinning policy; if neither qualifies, train an RL expert (rsl_rl PPO on the box). No budget ceiling on GPU hours or LLM calls. Known gaps to turn into questions: the env's spin_rate is roll about the pen's own long axis (not end-over-end twirl), success is per step with no episode rule, and the pipeline (manifests, obs schema, runner, records) is MetaWorld-only today. Debug issues, improve the pipeline, and turn unexpected findings into concrete research questions; document results and limitations.
**Idea R-040.** On-policy PPO learns state-dependent multi-finger coordination that sustains axial rolling on stages where the frozen scripted controllers cannot qualify.
**Prediction.** paired_success_difference increase by at least 0.1 against baseline.
**Protocol.** `gpu run 'source ~/IsaacLab/source_env.sh && python scripts/isaac/train_pen_spin_expert.py --config config/shadowhand_spin_stages.yaml --research-protocol ppo_expert'` at git `2543253760`.
**Result.** {"paired_success_difference": 0.5111111111111111, "paired_success_difference_ci95": [0.4058274562526499, 0.6163947659695723], "pooled_promotion_pairs": 90, "per_stage_promotion": {"stage1": {"ppo_successes_of_30": 29, "gait_successes_of_30": 21, "ppo_success": 0.9666666666666667, "gait_success": 0.7, "paired_difference": 0.26666666666666666, "paired_ci95": [0.0987173818937967, 0.43461595143953663], "qualifies": true, "promotes": true, "ppo_mean_abs_spin": 16.860679259517177, "ppo_mean_transverse": 2.7692530611652737, "ppo_completed": 1.0, "gait_mean_transverse": 3.0936859051386514, "ppo_failures": [{"case_id": "promotion-0021", "completed": true, "hit_steps": 351, "tail_hit_fraction": 0.9833333333333333, "final_center_dist": 0.0066612218506634235}]}, "stage2": {"ppo_successes_of_30": 30, "gait_successes_of_30": 13, "ppo_success": 1.0, "gait_success": 0.43333333333333335, "paired_difference": 0.5666666666666667, "paired_ci95": [0.3784673849181791, 0.7548659484151542], "qualifies": true, "promotes": true, "ppo_mean_abs_spin": 17.013145601207057, "ppo_mean_transverse": 2.7728778580547817, "ppo_completed": 1.0, "gait_mean_transverse": 3.3676134342948596, "ppo_failures": []}, "stage3": {"ppo_successes_of_30": 30, "gait_successes_of_30": 9, "ppo_success": 1.0, "gait_success": 0.3, "paired_difference": 0.7, "paired_ci95": [0.5259585363808446, 0.8740414636191554], "qualifies": true, "promotes": true, "ppo_mean_abs_spin": 16.977541782729805, "ppo_mean_transverse": 2.7862051706128135, "ppo_completed": 1.0, "gait_mean_transverse": 3.6596095169583958, "ppo_failures": []}}, "qualifies": {"stage1": true, "stage2": true, "stage3": true}, "promotes_per_stage": {"stage1": true, "stage2": true, "stage3": true}, "checkpoint_diagnostics_debug_s4097_stage3": [{"iteration": 0, "completed": 0.4, "mean_length": 174.85, "sustained_s1": 0.0, "tail_s1": 0.19778239689281335, "sustained_s2": 0.0, "tail_s2": 0.18713750990576045, "sustained_s3": 0.0, "tail_s3": 0.14452317199710601, "mean_abs_spin": 3.8591973924077783, "mean_transverse": 0.8587338952360952}, {"iteration": 824, "completed": 0.975, "mean_length": 353.175, "sustained_s1": 0.9, "tail_s1": 0.99375, "sustained_s2": 0.875, "tail_s2": 0.9920833333333334, "sustained_s3": 0.85, "tail_s3": 0.9875, "mean_abs_spin": 17.687590695122815, "mean_transverse": 3.00523211315212}, {"iteration": 1648, "completed": 1.0, "mean_length": 359.0, "sustained_s1": 1.0, "tail_s1": 1.0, "sustained_s2": 1.0, "tail_s2": 1.0, "sustained_s3": 1.0, "tail_s3": 1.0, "mean_abs_spin": 16.275774721448467, "mean_transverse": 3.10665694637883}, {"iteration": 2472, "completed": 1.0, "mean_length": 359.0, "sustained_s1": 1.0, "tail_s1": 1.0, "sustained_s2": 1.0, "tail_s2": 1.0, "sustained_s3": 1.0, "tail_s3": 1.0, "mean_abs_spin": 16.685099669220055, "mean_transverse": 2.8303947597493035}, {"iteration": 3296, "completed": 0.975, "mean_length": 351.325, "sustained_s1": 0.95, "tail_s1": 0.9822435897435898, "sustained_s2": 0.95, "tail_s2": 0.9715705128205128, "sustained_s3": 0.95, "tail_s3": 0.9639423076923077, "mean_abs_spin": 16.504485252259304, "mean_transverse": 2.7464876271970398}, {"iteration": 4120, "completed": 1.0, "mean_length": 359.0, "sustained_s1": 1.0, "tail_s1": 1.0, "sustained_s2": 1.0, "tail_s2": 1.0, "sustained_s3": 1.0, "tail_s3": 1.0, "mean_abs_spin": 16.866534644846798, "mean_transverse": 2.751165346448468}, {"iteration": 4944, "completed": 1.0, "mean_length": 359.0, "sustained_s1": 1.0, "tail_s1": 1.0, "sustained_s2": 1.0, "tail_s2": 1.0, "sustained_s3": 1.0, "tail_s3": 1.0, "mean_abs_spin": 16.91627567896936, "mean_transverse": 2.748587656685237}, {"iteration": 5768, "completed": 1.0, "mean_length": 359.0, "sustained_s1": 1.0, "tail_s1": 1.0, "sustained_s2": 1.0, "tail_s2": 1.0, "sustained_s3": 1.0, "tail_s3": 1.0, "mean_abs_spin": 16.86594054665738, "mean_transverse": 2.745105218488858}, {"iteration": 6592, "completed": 1.0, "mean_length": 359.0, "sustained_s1": 1.0, "tail_s1": 1.0, "sustained_s2": 1.0, "tail_s2": 1.0, "sustained_s3": 1.0, "tail_s3": 1.0, "mean_abs_spin": 16.95558844011142, "mean_transverse": 2.7458048615947077}, {"iteration": 7416, "completed": 0.975, "mean_length": 351.65, "sustained_s1": 0.975, "tail_s1": 0.99375, "sustained_s2": 0.975, "tail_s2": 0.9904166666666667, "sustained_s3": 0.975, "tail_s3": 0.9854166666666668, "mean_abs_spin": 16.99894581792976, "mean_transverse": 2.7765481146914546}, {"iteration": 8239, "completed": 1.0, "mean_length": 359.0, "sustained_s1": 1.0, "tail_s1": 1.0, "sustained_s2": 1.0, "tail_s2": 1.0, "sustained_s3": 1.0, "tail_s3": 1.0, "mean_abs_spin": 16.985603455779945, "mean_transverse": 2.7429886076775767}], "training": {"num_envs": 8192, "iterations": 8240, "transitions": 1080033280, "steps_per_second_gate": 38874.7, "final_train_mean_episode_length": 359.0}, "episodes": {"training_estimate_full_horizon": 3008449, "evaluation": 180, "reproducibility_control": 240, "checkpoint_diagnostics": 440}, "llm_calls": 0}; controls {"paired_success_difference": 0.0, "baseline": "frozen R-039 feedback gait (D-035) on the same fresh 30-case promotion manifests (seeds 4030-4032), own process per (stage, controller)", "reproducibility_control": {"stage1/ppo_expert": {"union_diverged_envs": 0, "sustained_outcome_flips": 0, "passes": true}, "stage2/ppo_expert": {"union_diverged_envs": 0, "sustained_outcome_flips": 0, "passes": true}, "stage3/ppo_expert": {"union_diverged_envs": 0, "sustained_outcome_flips": 0, "passes": true}}}.
**Decision: inconclusive**. PPO expert (one Stage-3 policy, seed 4020, 1.08e9 transitions, final checkpoint only) qualifies on every stage: 29/30, 30/30, 30/30 sustained vs the frozen gait's 21/13/9 on fresh 30-case manifests (seeds 4030-4032). Paired gains +0.267 [0.099, 0.435], +0.567 [0.378, 0.755], +0.700 [0.526, 0.874]; the predeclared >= +0.10 per-stage prediction is established on stages 2-3 but not stage 1 (interval contains +0.10), so the prediction is inconclusive while the expert qualifies and promotes everywhere. Reproducibility control: zero diverged envs. The only failure is a completed, centred episode missing 1 of 60 tail steps. Spin is axial roll ~17 rad/s, transverse ~2.8 rad/s.
**Next.** Use model_8239 as the pen-spin trace-repair expert: export traces on a disjoint demonstration manifest and build the pen-spin plateau-repair pipeline.

## Does the qualified expert need feedback, or only a replayed gait? (D-037, 2026-10-27)

**Direction.** Continue researching the single-step exploration pipeline with plateau-triggered trace repair, extended to pen spinning on the Shadow Hand (Isaac Lab Isaac-ShadowHandSpin-Stage{1,2,3}-Direct-v0, vendored in lares/envs/isaac_pen_spin; 157-d full obs, 20-d action, 60 Hz, 360-step episodes). Expert demonstrations: first try a hand-written scripted spinning controller; also search online for an existing trained pen-spinning policy; if neither qualifies, train an RL expert (rsl_rl PPO on the box). No budget ceiling on GPU hours or LLM calls. Known gaps to turn into questions: the env's spin_rate is roll about the pen's own long axis (not end-over-end twirl), success is per step with no episode rule, and the pipeline (manifests, obs schema, runner, records) is MetaWorld-only today. Debug issues, improve the pipeline, and turn unexpected findings into concrete research questions; document results and limitations.
**Idea R-041.** The PPO expert maintains sustained axial roll through state-dependent corrections that a clock-only replay of its successful actions cannot reproduce across reset placements.
**Prediction.** paired_success_difference increase by at least 0.2 against clock-only replay of one frozen successful PPO demonstration.
**Protocol.** `gpu run 'bash -lc "source ~/IsaacLab/source_env.sh && python scripts/run_shadowhand_spin_stages.py --config config/shadowhand_spin_stages.yaml --study feedback_ablation"'` at git `1eea3e5e06`.
**Result.** {"paired_success_difference": 1.0, "paired_success_difference_ci95": [1.0, 1.0], "pairs": 40, "closed_loop": {"n": 40, "successes": 40, "success": 1.0, "completed": 1.0, "tail_hit_fraction": 1.0, "hit_fraction": 0.9793175487465181, "mean_length": 359.0, "mean_abs_spin": 16.93339321613312, "mean_transverse": 2.769303226470947}, "action_tape": {"n": 40, "successes": 0, "success": 0.0, "completed": 0.25, "tail_hit_fraction": 0.34536343610883524, "hit_fraction": 0.4023492391946445, "mean_length": 159.475, "mean_abs_spin": 11.570241242647171, "mean_transverse": 4.274598932266235}, "demonstration": {"n": 40, "successes": 40, "success": 1.0, "completed": 1.0, "tail_hit_fraction": 1.0, "hit_fraction": 0.9743036211699163, "mean_length": 359.0, "mean_abs_spin": 16.866925525665284, "mean_transverse": 2.7512913644313812}, "action_tape_falls": 30, "action_tape_fall_step_median": 49, "action_tape_fall_step_range": [17, 355], "episodes": {"demonstration": 40, "closed_loop": 80, "action_tape": 40, "total": 160}, "llm_calls": 0}; controls {"paired_success_difference": 0.0, "baseline": "clock-only replay of the first sustained-success PPO demonstration (demo-0000 of seed 4100), 359 actions, on the same 40 diagnostic cases (seed 4101)", "reproducibility_control": {"union_diverged_envs": 0, "sustained_outcome_flips": 0, "max_diverged_envs": 4, "passes": true}}.
**Decision: supported**. R-041 asked whether the qualified PPO pen-spin expert (D-036) needs state feedback or only a replayable gait. On 40 fresh Stage-3 cases the closed-loop expert succeeded 40/40 while a clock-only replay of its first successful demonstration (359 actions) succeeded 0/40: paired +1.000, supported. The replay dropped the pen in 30/40 episodes (steps 17-355) and the survivors passed 35% of tail steps; end-over-end rate rose from 2.8 to 4.3 rad/s. A pre-launch smoke test showed that even on its own placement the tape tracks the recording only through step 16, then drifts to failure once the batch's other environments differ. The expert's success depends on corrections; a single tape is not a sufficient demonstration. Limitation: one tape; the corrective mechanism is not identified. A smoke test also caught and fixed a tape-slicing bug (the runner stores actions case-major).
**Next.** Revise R-042: Isaac backend for run_full_evolution and plateau-triggered trace repair on pen spinning, with a no-trace arm beside uniform and event-selected keyframe arms; traces must expose state-action corrections, not only action timelines.

## Plateau-triggered trace repair on pen spinning: event-selected vs uniform keyframes, with a no-trace arm (D-038, 2026-10-27)

**Direction.** Continue researching the single-step exploration pipeline with plateau-triggered trace repair, extended to pen spinning on the Shadow Hand (Isaac Lab Isaac-ShadowHandSpin-Stage{1,2,3}-Direct-v0, vendored in lares/envs/isaac_pen_spin; 157-d full obs, 20-d action, 60 Hz, 360-step episodes). Expert demonstrations: first try a hand-written scripted spinning controller; also search online for an existing trained pen-spinning policy; if neither qualifies, train an RL expert (rsl_rl PPO on the box). No budget ceiling on GPU hours or LLM calls. Known gaps to turn into questions: the env's spin_rate is roll about the pen's own long axis (not end-over-end twirl), success is per step with no episode rule, and the pipeline (manifests, obs schema, runner, records) is MetaWorld-only today. Debug issues, improve the pipeline, and turn unexpected findings into concrete research questions; document results and limitations.
**Idea R-042.** On Stage-3 pen spinning, where the expert succeeds only through state feedback (D-037), keyframes concentrated around changes in expert actions and pen angular velocity teach a whole-controller edit corrective structure better than uniformly spaced keyframes of the same traces at equal context; both trace arms are also scored against a no-trace continuation so the keyframe contrast is interpretable.
**Prediction.** paired_success_difference increase by at least 0.2 against uniform-keyframe plateau-triggered repair (same prefix, traces, fields and keyframe count).
**Protocol.** `gpu run 'source ~/.profile && .venv-metaworld/bin/python -u scripts/run_full_evolution.py --config config/pen_spin_plateau_repair.yaml --study keyframe_repair'` at git `d3eac64808`.
**Result.** {"paired_success_difference": 0.0, "paired_success_difference_ci95": [0.0, 0.0], "blocks": 4, "eligible_blocks": 4, "triggers_fired": 4, "event_minus_uniform": {"per_block": [0.0, 0.0, 0.0, 0.0], "n": 4, "mean": 0.0, "se": 0.0, "low": 0.0, "high": 0.0, "p_value": 1.0}, "uniform_minus_no_trace": {"per_block": [0.0, 0.0, 0.0, 0.0], "n": 4, "mean": 0.0, "se": 0.0, "low": 0.0, "high": 0.0, "p_value": 1.0}, "event_minus_no_trace": {"per_block": [0.0, 0.0, 0.0, 0.0], "n": 4, "mean": 0.0, "se": 0.0, "low": 0.0, "high": 0.0, "p_value": 1.0}, "fresh_success": {"no_trace": {"successes": 0, "cases": 120, "episodes_run": 0, "final_incumbents": [null, null, null, null]}, "uniform": {"successes": 0, "cases": 120, "episodes_run": 0, "final_incumbents": [null, null, null, null]}, "event": {"successes": 0, "cases": 120, "episodes_run": 0, "final_incumbents": [null, null, null, null]}}, "per_arm_development": {"no_trace": {"promotion_successes": 2, "promotion_cases": 120, "screen_successes": 1, "screen_cases": 40, "promotion_tail_hit_mean_per_block": [0.214, 0.194, 0.182, 0.191], "promotion_tail_hit_mean": 0.195, "promotion_failure_labels": {"dropped": 66, "null": 2, "off_centre": 32, "slow_spin": 20}, "promoted_blocks": 0, "llm_calls_per_block": [2, 2, 3, 3], "llm_tokens_per_block": [28342, 22196, 40280, 43149]}, "uniform": {"promotion_successes": 0, "promotion_cases": 120, "screen_successes": 0, "screen_cases": 40, "promotion_tail_hit_mean_per_block": [0.4, 0.335, 0.137, 0.223], "promotion_tail_hit_mean": 0.274, "promotion_failure_labels": {"dropped": 92, "slow_spin": 9, "off_centre": 19}, "promoted_blocks": 0, "llm_calls_per_block": [2, 2, 2, 3], "llm_tokens_per_block": [47553, 38649, 50509, 82787]}, "event": {"promotion_successes": 1, "promotion_cases": 120, "screen_successes": 1, "screen_cases": 40, "promotion_tail_hit_mean_per_block": [0.376, 0.217, 0.213, 0.102], "promotion_tail_hit_mean": 0.227, "promotion_failure_labels": {"dropped": 69, "off_centre": 26, "slow_spin": 24, "null": 1}, "promoted_blocks": 0, "llm_calls_per_block": [2, 2, 3, 2], "llm_tokens_per_block": [51377, 41155, 74630, 45004]}, "prefix": {"promotion_successes": 0, "promotion_cases": 120, "screen_successes": 0, "screen_cases": 40, "promotion_tail_hit_mean_per_block": [0.293, 0.095, 0.29, 0.107], "promotion_tail_hit_mean": 0.196, "promotion_failure_labels": {"dropped": 65, "off_centre": 32, "slow_spin": 23}, "promoted_blocks": 0, "llm_calls_per_block": [1, 1, 1, 1], "llm_tokens_per_block": [17774, 11041, 11612, 14043]}}, "exploratory_tail_hit_contrasts_promotion_set": {"event_minus_uniform": {"per_block": [-0.024, -0.118, 0.076, -0.121], "mean": -0.0468, "se": 0.0467, "ci95": [-0.1954, 0.1019]}, "uniform_minus_no_trace": {"per_block": [0.186, 0.141, -0.045, 0.032], "mean": 0.0785, "se": 0.0523, "ci95": [-0.0881, 0.2451]}, "event_minus_no_trace": {"per_block": [0.162, 0.023, 0.031, -0.089], "mean": 0.0318, "se": 0.0513, "ci95": [-0.1316, 0.1951]}}, "demonstrations": {"successes": 10, "episodes": 10, "qualified": true, "traces_case_ids": ["demo-0000", "demo-0001"]}, "parity": {"batch4": {"batch": 4, "sustained_outcome_flips": 0, "diverged_envs": 0, "max_diverged_envs": 0, "passes": true, "successes": [4, 4]}, "batch10": {"batch": 10, "sustained_outcome_flips": 0, "diverged_envs": 0, "max_diverged_envs": 1, "passes": true, "successes": [10, 10]}, "batch30": {"batch": 30, "sustained_outcome_flips": 0, "diverged_envs": 0, "max_diverged_envs": 3, "passes": true, "successes": [30, 30]}, "passes": true, "episodes": 88}, "accounting": {"llm_calls": 20, "llm_tokens": 456691, "note": "unique calls/tokens: the four prefix calls (tokens 17774, 11041, 11612, 14043) are copied into each arm's ledger and counted once here; per-arm lists in per_arm_development include the copied prefix call", "episodes_per_arm_block": {"no_trace": 146, "uniform": 156, "event": 156}, "parity_episodes": 88, "demonstration_episodes": 10, "fresh_episodes": 0}}; controls {"paired_success_difference": 0.0, "baseline": "uniform-keyframe plateau-triggered repair arm (same prefix copy, same two expert traces demo-0000/demo-0001, same fields and 16 keyframes per trace), fresh 30-case sustained success per block; secondary control the no-trace arm (one more searched single-phase exploration)", "parity_gate": {"passes": true, "sustained_outcome_flips": 0, "diverged_envs": 0}}.
**Decision: refuted**. The predeclared primary, event minus uniform fresh sustained success paired over four blocks, is exactly 0.000 with a degenerate [0,0] interval; the protocol's conclusiveness rule (>=3 eligible blocks) is met with 4/4 triggers fired, and the interval excludes the predicted +0.20, so the prediction as stated is refuted. The caveat is a floor: all twelve continuation arms ended with the null incumbent, so the study cannot rank keyframe rules at any effect size it could detect and does not show trace repair is useless in general. Development evidence agrees there is no trace benefit: promotion-set successes no_trace 2/120, uniform 0/120, event 1/120; exploratory tail-hit contrasts event-uniform -0.047 [-0.195,+0.102], uniform-no_trace +0.079 [-0.088,+0.245]. Trace-arm candidates in blocks 0-1 spin while held (highest tail-hit 0.40, 0.335) but drop the pen in 28-30 of 30 promotion cases, consistent with D-037: the expert's competence is its feedback map, which 32 sparse state-action keyframes from 2 episodes do not convey to a 6-10 parameter controller. Demonstrations 10/10 and parity 0 flips, so the null is not an integration fault.
**Next.** Stage 3 is at the floor for single-step symbolic exploration; next test a dense, fitted use of the expert (e.g. parameter fitting of the edited controller to expert state-action pairs, rollout-selected) or move the repair study to Stage 1/2 where generated controllers have non-zero base rates, so a trace effect is measurable.

## Fit dense expert labels into the R-042 prefix controllers, selecting only by rollouts (D-039, 2026-10-27)

**Direction.** Continue researching the single-step exploration pipeline with plateau-triggered trace repair, extended to pen spinning on the Shadow Hand (Isaac Lab Isaac-ShadowHandSpin-Stage{1,2,3}-Direct-v0, vendored in lares/envs/isaac_pen_spin; 157-d full obs, 20-d action, 60 Hz, 360-step episodes). Expert demonstrations: first try a hand-written scripted spinning controller; also search online for an existing trained pen-spinning policy; if neither qualifies, train an RL expert (rsl_rl PPO on the box). No budget ceiling on GPU hours or LLM calls. Known gaps to turn into questions: the env's spin_rate is roll about the pen's own long axis (not end-over-end twirl), success is per step with no episode rule, and the pipeline (manifests, obs schema, runner, records) is MetaWorld-only today. Debug issues, improve the pipeline, and turn unexpected findings into concrete research questions; document results and limitations.
**Idea R-045.** Dense expert state-action labels move a frozen symbolic pen-spin controller's bounded parameters toward sustained success more effectively than an equally budgeted rollout-only parameter search over the same structure, bounds and probe cases.
**Prediction.** paired_success_difference increase by at least 0.15 against rollout_search arm: the same frozen R-042 prefix controller per block, its selected parameters plus three predeclared bounded draws (rollout_search.perturb: local, global, local) scored on the same 10 probe cases.
**Protocol.** `gpu run 'source ~/.profile >/dev/null 2>&1; nohup .venv-metaworld/bin/python -u scripts/run_full_evolution.py --config config/pen_spin_plateau_repair.yaml --study dense_feedback_fit > logs_r045.log 2>&1 &'` at git `d5200616ef`.
**Result.** {"paired_success_difference": 0.0, "paired_success_difference_ci95": [0.0, 0.0], "blocks": 4, "dense_fit_minus_rollout_search_fresh_incumbent": {"per_block": [0.0, 0.0, 0.0, 0.0], "n": 4, "mean": 0.0, "se": 0.0, "low": 0.0, "high": 0.0, "p_value": 1.0}, "dense_fit_minus_rollout_search_fresh_selected": {"per_block": [0.0, 0.0, 0.0, -0.1], "n": 4, "mean": -0.025, "se": 0.025, "low": -0.10456115763210658, "high": 0.05456115763210658, "p_value": 0.39100221895577053}, "promoted": {"rollout_search": 0, "dense_fit": 0}, "dense_fit_selected_steps": ["fit_step250", "fit_step500", "fit_step0", "fit_step1000"], "per_arm": {"rollout_search": [{"block": 0, "selected": "search0_original", "probe": [{"label": "search0_original", "success": 0.0, "tail_hit": 0.252}, {"label": "search1_local", "success": 0.0, "tail_hit": 0.249}, {"label": "search2_global", "success": 0.0, "tail_hit": 0.051}, {"label": "search3_local", "success": 0.0, "tail_hit": 0.179}], "screen_success": 0.0, "promotion_success": 0.0, "promoted": false, "promotion_p_value": 1.0, "fresh_selected_success": 0.0, "fresh_selected_tail_hit": 0.254, "fresh_failure_labels": {"dropped": 22, "slow_spin": 6, "off_centre": 2}}, {"block": 1, "selected": "search3_local", "probe": [{"label": "search0_original", "success": 0.0, "tail_hit": 0.06}, {"label": "search1_local", "success": 0.0, "tail_hit": 0.053}, {"label": "search2_global", "success": 0.0, "tail_hit": 0.0}, {"label": "search3_local", "success": 0.0, "tail_hit": 0.062}], "screen_success": 0.0, "promotion_success": 0.0, "promoted": false, "promotion_p_value": 1.0, "fresh_selected_success": 0.0, "fresh_selected_tail_hit": 0.068, "fresh_failure_labels": {"off_centre": 10, "slow_spin": 12, "dropped": 8}}, {"block": 2, "selected": "search0_original", "probe": [{"label": "search0_original", "success": 0.0, "tail_hit": 0.336}, {"label": "search1_local", "success": 0.0, "tail_hit": 0.281}, {"label": "search2_global", "success": 0.0, "tail_hit": 0.092}, {"label": "search3_local", "success": 0.0, "tail_hit": 0.131}], "screen_success": 0.0, "promotion_success": 0.0, "promoted": false, "promotion_p_value": 1.0, "fresh_selected_success": 0.0, "fresh_selected_tail_hit": 0.15, "fresh_failure_labels": {"dropped": 13, "slow_spin": 13, "off_centre": 4}}, {"block": 3, "selected": "search3_local", "probe": [{"label": "search0_original", "success": 0.0, "tail_hit": 0.08}, {"label": "search1_local", "success": 0.0, "tail_hit": 0.077}, {"label": "search2_global", "success": 0.0, "tail_hit": 0.0}, {"label": "search3_local", "success": 0.2, "tail_hit": 0.2}], "screen_success": 0.0, "promotion_success": 0.1, "promoted": false, "promotion_p_value": 0.08307466551442909, "fresh_selected_success": 0.1, "fresh_selected_tail_hit": 0.149, "fresh_failure_labels": {"slow_spin": 16, "dropped": 2, "off_centre": 9, "None": 3}}], "dense_fit": [{"block": 0, "selected": "fit_step250", "probe": [{"label": "fit_step0", "success": 0.0, "tail_hit": 0.252}, {"label": "fit_step250", "success": 0.0, "tail_hit": 0.256}, {"label": "fit_step500", "success": 0.0, "tail_hit": 0.23}, {"label": "fit_step1000", "success": 0.0, "tail_hit": 0.138}], "screen_success": 0.0, "promotion_success": 0.0, "promoted": false, "promotion_p_value": 1.0, "fresh_selected_success": 0.0, "fresh_selected_tail_hit": 0.246, "fresh_failure_labels": {"dropped": 23, "slow_spin": 3, "off_centre": 4}, "fit_mse": [0.9178, 0.3482, 0.3376, 0.336]}, {"block": 1, "selected": "fit_step500", "probe": [{"label": "fit_step0", "success": 0.0, "tail_hit": 0.06}, {"label": "fit_step250", "success": 0.0, "tail_hit": 0.062}, {"label": "fit_step500", "success": 0.0, "tail_hit": 0.101}, {"label": "fit_step1000", "success": 0.0, "tail_hit": 0.097}], "screen_success": 0.0, "promotion_success": 0.0, "promoted": false, "promotion_p_value": 1.0, "fresh_selected_success": 0.0, "fresh_selected_tail_hit": 0.065, "fresh_failure_labels": {"slow_spin": 12, "off_centre": 13, "dropped": 5}, "fit_mse": [0.683, 0.3305, 0.3281, 0.328]}, {"block": 2, "selected": "fit_step0", "probe": [{"label": "fit_step0", "success": 0.0, "tail_hit": 0.336}, {"label": "fit_step250", "success": 0.0, "tail_hit": 0.035}, {"label": "fit_step500", "success": 0.0, "tail_hit": 0.058}, {"label": "fit_step1000", "success": 0.0, "tail_hit": 0.028}], "screen_success": 0.0, "promotion_success": 0.0, "promoted": false, "promotion_p_value": 1.0, "fresh_selected_success": 0.0, "fresh_selected_tail_hit": 0.15, "fresh_failure_labels": {"dropped": 13, "slow_spin": 13, "off_centre": 4}, "fit_mse": [1.2953, 0.3484, 0.3306, 0.3292]}, {"block": 3, "selected": "fit_step1000", "probe": [{"label": "fit_step0", "success": 0.0, "tail_hit": 0.08}, {"label": "fit_step250", "success": 0.0, "tail_hit": 0.075}, {"label": "fit_step500", "success": 0.0, "tail_hit": 0.027}, {"label": "fit_step1000", "success": 0.0, "tail_hit": 0.153}], "screen_success": 0.0, "promotion_success": 0.0, "promoted": false, "promotion_p_value": 1.0, "fresh_selected_success": 0.0, "fresh_selected_tail_hit": 0.214, "fresh_failure_labels": {"off_centre": 12, "dropped": 17, "slow_spin": 1}, "fit_mse": [1.3736, 0.4164, 0.3959, 0.3643]}]}, "expert": {"successes": 48, "episodes": 48, "pairs_per_block": 4308}, "parity": {"batch": 40, "sustained_outcome_flips": 0, "diverged_envs": 0, "max_diverged_envs": 4, "passes": true, "successes": [40, 40], "episodes": 80}, "episodes_total": 1008, "llm_calls": 0}; controls {"paired_success_difference": 0.0, "baseline": "rollout_search arm: the same frozen R-042 prefix controller per block, its selected parameters plus three predeclared draws (local, global, local) on the same 10 probe cases; fresh 30-case sustained success of its final incumbent", "parity_gate": {"passes": true, "sustained_outcome_flips": 0, "diverged_envs": 0}}.
**Decision: refuted**. Primary dense_fit minus rollout_search final-incumbent fresh success is 0.000 [0,0] over four blocks (neither arm promoted anywhere); the interval lies below the predicted +0.15, so the prediction is refuted under this protocol. As predeclared, an all-zero result cannot separate insufficient structure from ineffective fitting, so this does not show dense labels are useless in general. Mechanistic evidence: fitting succeeded as optimisation (MSE 2-4x lower, about 0.33 floor per action dimension) and rollouts chose fitted checkpoints in 3/4 blocks, yet no fitted setting scored a single probe success, and in block 2 every fitted checkpoint collapsed tail-hit (0.336 to 0.03-0.06). Lower imitation loss did not track rollout quality, matching the MetaWorld Stage E finding. The only success came from rollout search (block 3 local draw: 3/30 promotion, 3/30 fresh, not promoted). Expert 48/48 and parity 0 flips rule out integration faults. Secondary selected-setting contrast -0.025 [-0.105,+0.055].
**Next.** Stage 3 symbolic controllers sit at the floor under prompting, sparse traces and dense fitting alike; measure base rates on Stage 1/2 (R-044's lesion study can tell which feedback a structure must contain) before another Stage-3 repair comparison.

## 1. Identify which state feedback the qualified expert needs (D-040, 2026-10-27)

**Direction.** Continue researching the single-step exploration pipeline with plateau-triggered trace repair, extended to pen spinning on the Shadow Hand (Isaac Lab Isaac-ShadowHandSpin-Stage{1,2,3}-Direct-v0, vendored in lares/envs/isaac_pen_spin; 157-d full obs, 20-d action, 60 Hz, 360-step episodes). Expert demonstrations: first try a hand-written scripted spinning controller; also search online for an existing trained pen-spinning policy; if neither qualifies, train an RL expert (rsl_rl PPO on the box). No budget ceiling on GPU hours or LLM calls. Known gaps to turn into questions: the env's spin_rate is roll about the pen's own long axis (not end-over-end twirl), success is per step with no episode rule, and the pipeline (manifests, obs schema, runner, records) is MetaWorld-only today. Debug issues, improve the pipeline, and turn unexpected findings into concrete research questions; document results and limitations.
**Idea R-044.** The qualified PPO checkpoint relies on changing pen-state observations to maintain sustained axial-roll success when hand proprioception and action history remain available.
**Prediction.** paired_success_difference decrease by at least 0.3 against the unmodified qualified D-036 PPO expert on the same 40 cases (seed 4140), its own process.
**Protocol.** `gpu run 'source ~/IsaacLab/source_env.sh && python scripts/run_shadowhand_spin_stages.py --config config/shadowhand_spin_stages.yaml --study feedback_lesions'` at git `a9f4c20d83`.
**Result.** {"paired_success_difference": -0.975, "paired_success_difference_ci95": [-1.0255672730009189, -0.924432726999081], "pen_state_clamped_minus_expert": {"n": 40, "mean": -0.975, "se": 0.024999999999999998, "low": -1.0255672730009189, "high": -0.924432726999081, "p_value": 7.390618436886587e-33}, "hand_state_clamped_minus_expert": {"n": 40, "mean": -0.95, "se": 0.03489912202260564, "low": -1.0205901372323793, "high": -0.8794098627676206, "p_value": 5.547810049097881e-27}, "arms": {"expert": {"n": 40, "successes": 39, "success": 0.975, "completed": 1.0, "tail_hit_fraction": 0.9995833333333334, "hit_fraction": 0.9814066852367688, "mean_length": 359.0, "mean_abs_spin": 17.0460946559906, "mean_transverse": 2.774606728553772, "drops": 0, "tail_spin_failures": 1, "median_fall_step": null}, "pen_state_clamped": {"n": 40, "successes": 0, "success": 0.0, "completed": 0.025, "tail_hit_fraction": 0.43459674885493416, "hit_fraction": 0.4428737996084623, "mean_length": 68.025, "mean_abs_spin": 8.189444118738175, "mean_transverse": 4.572741687297821, "drops": 39, "tail_spin_failures": 1, "median_fall_step": 49.0}, "hand_state_clamped": {"n": 40, "successes": 1, "success": 0.025, "completed": 0.45, "tail_hit_fraction": 0.19392903161876593, "hit_fraction": 0.24812765805158726, "mean_length": 204.35, "mean_abs_spin": 9.096962953358888, "mean_transverse": 3.362861725315452, "drops": 22, "tail_spin_failures": 17, "median_fall_step": 52.5}, "expert_repeat": {"n": 40, "successes": 39, "success": 0.975, "completed": 1.0, "tail_hit_fraction": 0.9995833333333334, "hit_fraction": 0.9814066852367688, "mean_length": 359.0, "mean_abs_spin": 17.0460946559906, "mean_transverse": 2.774606728553772, "drops": 0, "tail_spin_failures": 1, "median_fall_step": null}}, "lesions": {"pen_state_clamped": ["obj_pos", "obj_rot", "obj_linvel", "obj_angvel", "rel_rot"], "hand_state_clamped": ["hand_dof_pos", "hand_dof_vel", "fingertip_pos", "fingertip_rot", "fingertip_vel"]}, "episodes": 160, "llm_calls": 0}; controls {"paired_success_difference": 0.0, "baseline": "unmodified D-036 PPO expert on the same 40 cases (seed 4140), own process; expert_repeat reproduces it exactly", "reproducibility_control": {"union_diverged_envs": 0, "sustained_outcome_flips": 0, "max_diverged_envs": 4, "passes": true}}.
**Decision: supported**. Pen-state clamped minus the unmodified expert is -0.975 [-1.026,-0.924] over 40 paired cases, below the predicted -0.30 with the interval far from it; the reproducibility control is exact (0 flips, 0 diverged). Within the narrowed hypothesis (this checkpoint, under clamping, with hand proprioception and action history live) the prediction is supported. Two limits: the hand-state clamp is nearly as destructive (-0.950 [-1.021,-0.879]), so pen state is not uniquely required; and clamping shifts the input distribution, so this is dependence, not necessity. The failure split is informative: pen clamps produce early drops (39/40, median step 49, matching D-037's replay falls), hand clamps split into drops (22) and completed-but-slow spins (17).
**Next.** A symbolic controller needs both pen-state retention feedback and proprioceptive spin feedback; next, measure whether generated controllers on Stage 1/2 (non-zero base rate) improve when the prompt states this two-channel requirement.

## 3. Preserve the scripted gait while repairing retention feedback (D-041, 2026-10-27)

**Direction.** Continue researching the single-step exploration pipeline with plateau-triggered trace repair, extended to pen spinning on the Shadow Hand (Isaac Lab Isaac-ShadowHandSpin-Stage{1,2,3}-Direct-v0, vendored in lares/envs/isaac_pen_spin; 157-d full obs, 20-d action, 60 Hz, 360-step episodes). Expert demonstrations: first try a hand-written scripted spinning controller; also search online for an existing trained pen-spinning policy; if neither qualifies, train an RL expert (rsl_rl PPO on the box). No budget ceiling on GPU hours or LLM calls. Known gaps to turn into questions: the env's spin_rate is roll about the pen's own long axis (not end-over-end twirl), success is per step with no episode rule, and the pipeline (manifests, obs schema, runner, records) is MetaWorld-only today. Debug issues, improve the pipeline, and turn unexpected findings into concrete research questions; document results and limitations.
**Idea R-046.** An immutable scripted gait preserves contact coordination while a bounded symbolic residual repairs retention failures that whole-controller trace edits otherwise introduce.
**Prediction.** paired_success_difference increase by at least 0.2 against whole_edit arm: an unrestricted whole-controller edit given the same frozen gait source and the same 32 uniform expert keyframes, same parameter-search budget.
**Protocol.** `gpu run 'source ~/.profile >/dev/null 2>&1; nohup .venv-metaworld/bin/python -u scripts/run_full_evolution.py --config config/pen_spin_plateau_repair.yaml --study gait_preserving_repair > logs_r046.log 2>&1 &'` at git `9bec484193`.
**Result.** {"paired_success_difference": null, "conclusive": false, "eligible_blocks": 1, "min_eligible_blocks": 3, "bounded_residual_minus_whole_edit_fresh_incumbent": {"per_block": [0.4667], "n": 1, "block": 1, "wins": 14, "losses": 0, "ties": 16}, "bounded_residual_minus_unchanged_gait_fresh_incumbent": {"per_block": [0.0333], "n": 1, "block": 1, "wins": 9, "losses": 8, "ties": 13}, "fresh_success_block1": {"unchanged_gait": 0.43333333333333335, "bounded_residual": 0.4666666666666667, "whole_edit": 0.0}, "unchanged_gait_promotion_success_block1": 0.5, "prefix_blocks": [{"block": 0, "seed": 4190, "trigger_fired": false, "reason": "1 candidate(s) succeeded on the probe", "probe_successes_of_96": 1, "prefix_promoted": false, "challenger_screen_success": 0.0, "promotion_mean_difference": 0.03333333333333333, "promotion_ci95": [-0.0348409880710901, 0.10150765473775678], "episodes": 146, "llm_calls": 1, "llm_tokens": 11905}, {"block": 1, "seed": 4191, "trigger_fired": true, "reason": "plateau: no probe success and no promotion", "probe_successes_of_96": 0, "prefix_promoted": false, "challenger_screen_success": 0.0, "promotion_mean_difference": 0.0, "promotion_ci95": [0.0, 0.0], "episodes": 146, "llm_calls": 2, "llm_tokens": 28032}, {"block": 2, "seed": 4192, "trigger_fired": false, "reason": "1 candidate(s) succeeded on the probe", "probe_successes_of_96": 1, "prefix_promoted": false, "challenger_screen_success": 0.0, "promotion_mean_difference": 0.0, "promotion_ci95": [0.0, 0.0], "episodes": 146, "llm_calls": 2, "llm_tokens": 30127}, {"block": 3, "seed": 4193, "trigger_fired": false, "reason": "a prefix candidate was promoted", "probe_successes_of_96": 2, "prefix_promoted": true, "challenger_screen_success": 0.2, "promotion_mean_difference": 0.3333333333333333, "promotion_ci95": [0.15429879420839188, 0.5123678724582748], "episodes": 146, "llm_calls": 1, "llm_tokens": 14374}], "prefix_probe_success_blocks": 3, "prefix_promoted_blocks": 1, "per_arm_block1": {"unchanged_gait": {"selected": "frozen_gait", "probe": [], "screen_success": 0.2, "promotion_success": 0.5, "promoted": true, "promotion_ci95": [0.31010519497721845, 0.6898948050227816], "fresh_incumbent_success": 0.43333333333333335, "fresh_selected_success": 0.43333333333333335, "fresh_failure_labels": {"None": 13, "slow_spin": 12, "off_centre": 2, "dropped": 3}, "isaac_episodes": 70, "llm_calls": 0, "llm_tokens": 0}, "bounded_residual": {"selected": "search4_global", "probe": [{"label": "search0_original", "success": 0.125}, {"label": "search1_local", "success": 0.625}, {"label": "search2_global", "success": 0.625}, {"label": "search3_local", "success": 0.5}, {"label": "search4_global", "success": 0.625}, {"label": "search5_local", "success": 0.625}], "screen_success": 0.1, "promotion_success": 0.4, "promoted": true, "promotion_ci95": [0.21394184915555403, 0.586058150844446], "fresh_incumbent_success": 0.4666666666666667, "fresh_selected_success": 0.4666666666666667, "fresh_failure_labels": {"None": 14, "off_centre": 5, "slow_spin": 11}, "isaac_episodes": 118, "llm_calls": 1, "llm_tokens": 24319}, "whole_edit": {"selected": "search2_global", "probe": [{"label": "search0_original", "success": 0.0}, {"label": "search1_local", "success": 0.0}, {"label": "search2_global", "success": 0.0}, {"label": "search3_local", "success": 0.0}, {"label": "search4_global", "success": 0.0}, {"label": "search5_local", "success": 0.0}], "screen_success": 0.0, "promotion_success": 0.0, "promoted": false, "promotion_ci95": [0.0, 0.0], "fresh_incumbent_success": 0.0, "fresh_selected_success": 0.0, "fresh_failure_labels": {"off_centre": 6, "dropped": 24}, "isaac_episodes": 118, "llm_calls": 1, "llm_tokens": 37722}}, "expert_block1": {"successes": 12, "episodes": 12, "qualify_successes": 10, "qualify_cases": 10, "qualified": true, "traced_case_ids": ["gait-0062", "gait-0063"]}}; controls {"parity": {"version": 2, "rule": "every environment identical through step 150; at most 4 of 40 diverge overall; sustained flips reported, not gating", "first_divergence_step": {"2": 340, "6": 185, "20": 260}, "diverged_envs": 3, "diverged_before_match_step": 0, "sustained_outcome_flips": 1, "flipped_envs": [20], "passes": true, "amended_by": "advisor consult call_00095", "v1": {"batch": 40, "sustained_outcome_flips": 1, "diverged_envs": 3, "max_diverged_envs": 4, "passes": false, "successes": [21, 22], "episodes": 80}}}.
**Decision: inconclusive**. The predeclared rule needs >=3 eligible blocks; the plateau trigger fired in only 1 of 4 (blocks 0 and 2 had a probe success, block 3 promoted a cold prefix candidate at +0.333 [0.154, 0.512]), so the primary contrast is undefined and the trigger was not weakened. Block 1 is suggestive only: whole_edit replaced the gait (delta = desired - gait) and dropped the pen in 24/30 fresh cases (0/30), while bounded_residual matched the unchanged gait (0.467 vs 0.433; 9 wins 8 losses 13 ties), i.e. preservation, not improvement. The unexpected finding is the cold prefix's non-zero base rate on Stage 3 (3/4 blocks with probe successes, 1 promotion) after R-042 found 0/4 under the same search; whether Stage 3 is a floor depends on seeds, which a base-rate study should measure before any further plateau-gated comparison.
**Next.** Measure the cold single-step search's Stage 3 base rate (probe-success and promotion rates) over many prefix blocks, and run the gait-residual comparison ungated (every block) instead of behind the plateau trigger.

## Measure the expert's retention-feedback latency budget (D-042, 2026-10-27)

**Direction.** Continue researching the single-step exploration pipeline with plateau-triggered trace repair, extended to pen spinning on the Shadow Hand (Isaac Lab Isaac-ShadowHandSpin-Stage{1,2,3}-Direct-v0, vendored in lares/envs/isaac_pen_spin; 157-d full obs, 20-d action, 60 Hz, 360-step episodes). Expert demonstrations: first try a hand-written scripted spinning controller; also search online for an existing trained pen-spinning policy; if neither qualifies, train an RL expert (rsl_rl PPO on the box). No budget ceiling on GPU hours or LLM calls. Known gaps to turn into questions: the env's spin_rate is roll about the pen's own long axis (not end-over-end twirl), success is per step with no episode rule, and the pipeline (manifests, obs schema, runner, records) is MetaWorld-only today. Debug issues, improve the pipeline, and turn unexpected findings into concrete research questions; document results and limitations.
**Idea R-047.** Delaying pen-state feedback by four control steps makes the qualified expert's retention corrections arrive too late, reducing sustained Stage-3 success despite live proprioception and action history.
**Prediction.** paired_success_difference decrease by at least 0.25 against baseline.
**Protocol.** `gpu run 'bash -lc "source ~/IsaacLab/source_env.sh && python scripts/run_shadowhand_spin_stages.py --config config/shadowhand_spin_stages.yaml --study feedback_delay --manifest-seed 4200 --output-dir logs/pen_spin_feedback_delay_s4200"'` at git `08cfe506aa`.
**Result.** {"paired_success_difference": -1.0, "paired_success_difference_ci95": [-1.0, -1.0], "pen_state_delayed4_minus_expert": {"n": 40, "mean": -1.0, "se": 0.0, "low": -1.0, "high": -1.0, "p_value": 0.0}, "hand_state_delayed4_minus_expert": {"n": 40, "mean": -1.0, "se": 0.0, "low": -1.0, "high": -1.0, "p_value": 0.0}, "delay_steps": 4, "delay_seconds": 0.0667, "arms": {"expert": {"n": 40, "successes": 40, "success": 1.0, "completed": 1.0, "tail_hit_fraction": 1.0, "hit_fraction": 0.9806406685236769, "mean_length": 359.0, "mean_abs_spin": 17.015426921844483, "mean_transverse": 2.7613835632801056, "drops": 0, "tail_spin_failures": 0, "median_fall_step": null}, "pen_state_delayed4": {"n": 40, "successes": 0, "success": 0.0, "completed": 0.175, "tail_hit_fraction": 0.45469632768361573, "hit_fraction": 0.5358633198655072, "mean_length": 176.9, "mean_abs_spin": 11.14821549654007, "mean_transverse": 3.999337711930275, "drops": 33, "tail_spin_failures": 7, "median_fall_step": 116.0}, "hand_state_delayed4": {"n": 40, "successes": 0, "success": 0.0, "completed": 0.025, "tail_hit_fraction": 0.37356608092615884, "hit_fraction": 0.44487004809531017, "mean_length": 113.825, "mean_abs_spin": 14.558973002433778, "mean_transverse": 5.673943960666657, "drops": 39, "tail_spin_failures": 1, "median_fall_step": 83.0}, "expert_repeat": {"n": 40, "successes": 40, "success": 1.0, "completed": 1.0, "tail_hit_fraction": 1.0, "hit_fraction": 0.9806406685236769, "mean_length": 359.0, "mean_abs_spin": 17.015426921844483, "mean_transverse": 2.7613835632801056, "drops": 0, "tail_spin_failures": 0, "median_fall_step": null}}, "fall_step_quartiles": {"pen_state_delayed4": [80.0, 116.0, 183.0], "hand_state_delayed4": [53.0, 83.0, 122.5]}, "r044_comparison": {"pen_state_clamped": {"successes": 0, "n": 40, "drops": 39, "median_fall_step": 49}, "hand_state_clamped": {"successes": 1, "n": 40, "drops": 22, "tail_spin_failures": 17}}}; controls {"reproducibility": {"union_diverged_envs": 0, "sustained_outcome_flips": 0, "max_diverged_envs": 4, "passes": true}, "expert": "40/40 twice, bit-identical traces (0 diverged environments, 0 flips)"}.
**Decision: supported**. Predicted a paired decrease of at least 0.25; observed -1.000 (40/40 -> 0/40) for a four-step pen-state delay, with the reproducibility control bit-identical (0 diverged, 0 flips). The hand-state delay is equally fatal (0/40, 39 drops). The claim is limited to this checkpoint under a temporal joint-distribution shift; with one delay value there is no dose-response, so the failure threshold lies at or below 67 ms but is otherwise unmeasured.
**Next.** If latency matters for the pipeline, a dose-response (1, 2 steps) would locate the threshold; for generated controllers, the implication is that retention needs current-step pen and hand feedback, which R-049's two-channel requirement tests directly.

## Expose the expert's local feedback map instead of isolated labels (D-043, 2026-10-27)

**Direction.** Continue researching the single-step exploration pipeline with plateau-triggered trace repair, extended to pen spinning on the Shadow Hand (Isaac Lab Isaac-ShadowHandSpin-Stage{1,2,3}-Direct-v0, vendored in lares/envs/isaac_pen_spin; 157-d full obs, 20-d action, 60 Hz, 360-step episodes). Expert demonstrations: first try a hand-written scripted spinning controller; also search online for an existing trained pen-spinning policy; if neither qualifies, train an RL expert (rsl_rl PPO on the box). No budget ceiling on GPU hours or LLM calls. Known gaps to turn into questions: the env's spin_rate is roll about the pen's own long axis (not end-over-end twirl), success is per step with no episode rule, and the pipeline (manifests, obs schema, runner, records) is MetaWorld-only today. Debug issues, improve the pipeline, and turn unexpected findings into concrete research questions; document results and limitations.
**Idea R-048.** Local expert action sensitivities reveal corrective feedback that isolated state-action labels obscure, allowing gait-preserving symbolic residuals to improve retention.
**Prediction.** paired_success_difference increase by at least 0.15 against baseline.
**Protocol.** `gpu run 'source ~/.profile >/dev/null 2>&1; nohup .venv-metaworld/bin/python -u scripts/run_full_evolution.py --config config/pen_spin_plateau_repair.yaml --study local_feedback_evidence --block-seeds 4210,4211,4212,4213,4214,4215 --output-dir logs/pen_spin_local_feedback_evidence > logs/r048.log 2>&1 &'` at git `b222600a99`.
**Result.** {"paired_success_difference": null, "conclusive": false, "closure_reason": "insufficient executed-action switch coverage within the searched coordinate grid (advisor call_00101)", "expert_saturation": {"demo_executed_at_plus_minus_1": 0.9998, "raw_beyond_1_at_gait_states": 0.996875, "raw_beyond_1_all_queries": 0.9967830882352942}, "query_states_demo_distance_h_units": {"median": 82.12324505810719, "max": 128.58593040646448}, "switch_audit": {"coverage": 0.028125, "coverage_by_coordinate": {"offset_x": 0.034375, "offset_y": 0.028125, "offset_z": 0.0359375, "axial_rate": 0.0234375, "ff_flex": 0.021875, "mf_flex": 0.0296875, "rf_flex": 0.021875, "lf_flex": 0.0296875}, "states_with_any_switch": 0.6875, "multiple_crossings": 0.0001953125, "median_abs_distance": 0.7, "radius_ranges": 1.0, "states": 32, "queries": 5152, "episodes": 0, "llm_calls": 0, "demo_ranges": {"offset_x": 0.009569996595382688, "offset_y": 0.007310938835144037, "offset_z": 0.031898716092109675, "axial_rate": 8.646951880831624, "ff_flex": 0.4534321452180544, "mf_flex": 0.45176070195933177, "rf_flex": 0.22962127228577922, "lf_flex": 0.3945484248300393}}, "block0_parent_gait": {"screen": 0.6, "probe": 0.4, "promotion": 0.43333333333333335, "fresh": 0.5666666666666667}, "block0_parent_promotion_vs_null_ci95": [0.24513405158484583, 0.6215326150818209]}; controls {"query_path_check": "scripts/isaac/check_expert_query.py: the expert queried at 10 demonstration states reproduced its executed (clamped) actions exactly (max error 0), 0 episodes"}.
**Decision: inconclusive**. No paired comparison ran, so the +0.15 prediction was not evaluated. The PPO expert is bang-bang (99.98% of executed components at +/-1), which made clamped finite differences zero almost everywhere; the executed-switch audit found switches in only 2.8% of state-coordinate-action triples within one demonstration range (median distance 0.7 ranges), so the evidence representation would be impoverished and the comparison would test it rather than the feedback-map hypothesis. Closed on the advisor's recommendation (call_00101) as insufficient demonstrated local signal on the sampled grid, not falsification.
**Next.** Run R-049 (two-channel cold exploration on Stages 1/2). A future feedback-map idea needs a smooth expert (e.g. pre-saturation logits from a policy trained with an action penalty) or switch maps at on-distribution states.

## Require separate retention and spin feedback in cold exploration (D-044, 2026-10-27)

**Direction.** Continue researching the single-step exploration pipeline with plateau-triggered trace repair, extended to pen spinning on the Shadow Hand (Isaac Lab Isaac-ShadowHandSpin-Stage{1,2,3}-Direct-v0, vendored in lares/envs/isaac_pen_spin; 157-d full obs, 20-d action, 60 Hz, 360-step episodes). Expert demonstrations: first try a hand-written scripted spinning controller; also search online for an existing trained pen-spinning policy; if neither qualifies, train an RL expert (rsl_rl PPO on the box). No budget ceiling on GPU hours or LLM calls. Known gaps to turn into questions: the env's spin_rate is roll about the pen's own long axis (not end-over-end twirl), success is per step with no episode rule, and the pipeline (manifests, obs schema, runner, records) is MetaWorld-only today. Debug issues, improve the pipeline, and turn unexpected findings into concrete research questions; document results and limitations.
**Idea R-049.** Explicitly separating pen-state retention feedback from proprioceptive spin feedback prevents generated controllers from omitting a necessary control channel.
**Prediction.** paired_success_difference increase by at least 0.15 against baseline.
**Protocol.** `gpu run 'source ~/.profile >/dev/null 2>&1; nohup .venv-metaworld/bin/python -u scripts/run_full_evolution.py --config config/pen_spin_plateau_repair.yaml --study two_channel_cold --block-seeds 4220,4221,4222,4223,4224,4225,4226,4227 --output-dir logs/pen_spin_two_channel_cold > logs/r049.log 2>&1 &'` at git `d6241fe50b`.
**Result.** {"paired_success_difference": 0.0, "paired_success_difference_ci95": [0.0, 0.0], "two_channel_minus_baseline_fresh_incumbent": {"per_block": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "n": 8, "mean": 0.0, "se": 0.0, "low": 0.0, "high": 0.0, "p_value": 1.0}, "two_channel_minus_baseline_fresh_selected": {"per_block": [0.0, 0.06666666666666667, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "n": 8, "mean": 0.008333333333333333, "se": 0.008333333333333333, "low": -0.011371868763273204, "high": 0.028038535429939872, "p_value": 0.3506166628202075}, "by_stage": {"stage1": {"blocks": 4, "null_fresh": 0.0, "baseline": {"fresh_incumbent": 0.0, "promoted_blocks": 0, "plateau_blocks": 4, "candidates_with_screen_success": 0, "valid_candidates": 16}, "two_channel": {"fresh_incumbent": 0.0, "promoted_blocks": 0, "plateau_blocks": 4, "candidates_with_screen_success": 0, "valid_candidates": 16}, "two_channel_minus_baseline": {"per_block": [0.0, 0.0, 0.0, 0.0], "mean": 0.0}}, "stage2": {"blocks": 4, "null_fresh": 0.0, "baseline": {"fresh_incumbent": 0.0, "promoted_blocks": 0, "plateau_blocks": 4, "candidates_with_screen_success": 0, "valid_candidates": 16}, "two_channel": {"fresh_incumbent": 0.0, "promoted_blocks": 0, "plateau_blocks": 4, "candidates_with_screen_success": 0, "valid_candidates": 16}, "two_channel_minus_baseline": {"per_block": [0.0, 0.0, 0.0, 0.0], "mean": 0.0}}}, "per_block": [{"block": 0, "stage": 1, "seed": 4220, "null_fresh": 0.0, "baseline": {"valid": 4, "selected_slot": 2, "probe": 0.0, "promotion": 0.0, "promoted": false, "fresh_selected": 0.0, "fresh_incumbent": 0.0}, "two_channel": {"valid": 4, "selected_slot": 2, "probe": 0.0, "promotion": 0.0, "promoted": false, "fresh_selected": 0.0, "fresh_incumbent": 0.0}}, {"block": 1, "stage": 1, "seed": 4221, "null_fresh": 0.0, "baseline": {"valid": 4, "selected_slot": 3, "probe": 0.0, "promotion": 0.0, "promoted": false, "fresh_selected": 0.0, "fresh_incumbent": 0.0}, "two_channel": {"valid": 4, "selected_slot": 1, "probe": 0.0, "promotion": 0.06666666666666667, "promoted": false, "fresh_selected": 0.06666666666666667, "fresh_incumbent": 0.0}}, {"block": 2, "stage": 1, "seed": 4222, "null_fresh": 0.0, "baseline": {"valid": 4, "selected_slot": 3, "probe": 0.0, "promotion": 0.0, "promoted": false, "fresh_selected": 0.0, "fresh_incumbent": 0.0}, "two_channel": {"valid": 4, "selected_slot": 1, "probe": 0.0, "promotion": 0.0, "promoted": false, "fresh_selected": 0.0, "fresh_incumbent": 0.0}}, {"block": 3, "stage": 1, "seed": 4223, "null_fresh": 0.0, "baseline": {"valid": 4, "selected_slot": 0, "probe": 0.0, "promotion": 0.0, "promoted": false, "fresh_selected": 0.0, "fresh_incumbent": 0.0}, "two_channel": {"valid": 4, "selected_slot": 3, "probe": 0.0, "promotion": 0.0, "promoted": false, "fresh_selected": 0.0, "fresh_incumbent": 0.0}}, {"block": 4, "stage": 2, "seed": 4224, "null_fresh": 0.0, "baseline": {"valid": 4, "selected_slot": 3, "probe": 0.0, "promotion": 0.0, "promoted": false, "fresh_selected": 0.0, "fresh_incumbent": 0.0}, "two_channel": {"valid": 4, "selected_slot": 0, "probe": 0.0, "promotion": 0.0, "promoted": false, "fresh_selected": 0.0, "fresh_incumbent": 0.0}}, {"block": 5, "stage": 2, "seed": 4225, "null_fresh": 0.0, "baseline": {"valid": 4, "selected_slot": 3, "probe": 0.0, "promotion": 0.0, "promoted": false, "fresh_selected": 0.0, "fresh_incumbent": 0.0}, "two_channel": {"valid": 4, "selected_slot": 0, "probe": 0.0, "promotion": 0.0, "promoted": false, "fresh_selected": 0.0, "fresh_incumbent": 0.0}}, {"block": 6, "stage": 2, "seed": 4226, "null_fresh": 0.0, "baseline": {"valid": 4, "selected_slot": 0, "probe": 0.0, "promotion": 0.0, "promoted": false, "fresh_selected": 0.0, "fresh_incumbent": 0.0}, "two_channel": {"valid": 4, "selected_slot": 2, "probe": 0.0, "promotion": 0.0, "promoted": false, "fresh_selected": 0.0, "fresh_incumbent": 0.0}}, {"block": 7, "stage": 2, "seed": 4227, "null_fresh": 0.0, "baseline": {"valid": 4, "selected_slot": 2, "probe": 0.0, "promotion": 0.0, "promoted": false, "fresh_selected": 0.0, "fresh_incumbent": 0.0}, "two_channel": {"valid": 4, "selected_slot": 0, "probe": 0.0, "promotion": 0.0, "promoted": false, "fresh_selected": 0.0, "fresh_incumbent": 0.0}}], "candidates_with_screen_success": {"baseline": 0, "two_channel": 0}, "promoted_blocks": {"baseline": 0, "two_channel": 0}, "plateau_blocks": {"baseline": 8, "two_channel": 8}, "screen_failure_labels": {"baseline": {"off_centre": 134, "slow_spin": 145, "dropped": 41}, "two_channel": {"slow_spin": 145, "off_centre": 132, "dropped": 43}}, "null_failure_labels": {"slow_spin": 18, "dropped": 622}, "screen_drop_fraction": {"baseline": 0.128125, "two_channel": 0.134375}, "null_drop_fraction": 0.971875}; controls {"null_incumbent": "neutral pose scored on all 80 cases per block in its own processes: 0/640 sustained, but 'dropped' in 622/640", "baseline_arm": "contemporaneous cold search with the unchanged prompt, identical manifests, slots and selection"}.
**Decision: refuted**. Predicted two_channel minus baseline fresh-incumbent success of at least +0.15 with the interval establishing it; observed 0.000 [0.000, 0.000] over 8 blocks. Both arms kept the neutral-pose null incumbent everywhere: 0 of 32 candidates per arm had a screen success (0/320 screen episodes each), 0 promotions; the only non-zero cell is two_channel block 1 (2/30 promotion, not promoted). Confidence is 0.7 because both arms sit at the floor and 4 candidates per block gives little resolution. Diagnostics: generated candidates already retain the pen (13% of screen episodes dropped vs 97% for the neutral pose) and fail on slow_spin/off_centre, so the bottleneck on Stages 1/2 is sustained axial spin near the centre, not retention; and these per-step-looser stages scored 0/8 blocks while R-046's 24-candidate Stage 3 prefix had probe successes in 3/4, so candidates per block, not stage difficulty, is the untested candidate driver of the base rate.
**Next.** Measure the cold-search per-candidate success rate with a larger candidate budget on Stage 1 and Stage 3 under one protocol, and target spin generation (axial angular-velocity feedback) rather than retention.

## 2. Can the expert recover from actual symbolic-controller states? (D-045, 2026-10-27)

**Direction.** Continue researching the single-step exploration pipeline with plateau-triggered trace repair, extended to pen spinning on the Shadow Hand (Isaac Lab Isaac-ShadowHandSpin-Stage{1,2,3}-Direct-v0, vendored in lares/envs/isaac_pen_spin; 157-d full obs, 20-d action, 60 Hz, 360-step episodes). Expert demonstrations: first try a hand-written scripted spinning controller; also search online for an existing trained pen-spinning policy; if neither qualifies, train an RL expert (rsl_rl PPO on the box). No budget ceiling on GPU hours or LLM calls. Known gaps to turn into questions: the env's spin_rate is roll about the pen's own long axis (not end-over-end twirl), success is per step with no episode rule, and the pipeline (manifests, obs schema, runner, records) is MetaWorld-only today. Debug issues, improve the pipeline, and turn unexpected findings into concrete research questions; document results and limitations.
**Idea R-051.** The qualified PPO expert can recover retained-but-slow learner states, making its on-policy corrections usable repair evidence rather than unsupported labels.
**Prediction.** paired_success_difference increase by at least 0.25 against baseline.
**Protocol.** `gpu run 'bash -lc "source ~/IsaacLab/source_env.sh && python scripts/run_shadowhand_spin_stages.py --config config/shadowhand_spin_stages.yaml --study learner_recovery --manifest-seed 4310 --output-dir logs/pen_spin_learner_recovery_s4310"'` at git `7a800c309b`.
**Result.** {"paired_success_difference": 0.03125, "paired_success_difference_ci95": [-0.18321148355818018, 0.24571148355818018], "takeover_minus_learner": {"n": 32, "mean": 0.03125, "se": 0.10515325796802646, "low": -0.18321148355818018, "high": 0.24571148355818018, "p_value": 0.7683074113079962}, "arms": {"learner": {"n": 32, "successes": 7, "success": 0.21875, "completed": 0.875, "tail_hit_fraction": 0.25265877016129035, "hit_fraction": 0.26876253060697275, "mean_length": 319.21875, "mean_abs_spin": 4.836397275328636, "mean_transverse": 1.5364455340895802, "drops": 4, "tail_spin_failures": 21, "median_fall_step": 36.0}, "takeover": {"n": 32, "successes": 8, "success": 0.25, "completed": 0.53125, "tail_hit_fraction": 0.4214087701612903, "hit_fraction": 0.423415567945806, "mean_length": 248.59375, "mean_abs_spin": 10.333789110183716, "mean_transverse": 2.8369073942303658, "drops": 15, "tail_spin_failures": 9, "median_fall_step": 114.0}}, "switched_cases": 28, "never_switched_cases": [7, 10, 13, 20], "switched_only": {"n": 28, "learner_successes": 7, "takeover_successes": 8, "difference": {"n": 28, "mean": 0.03571428571428571, "se": 0.12042849924994854, "low": -0.21138458410068162, "high": 0.282813155529253, "p_value": 0.7690717158092564}, "takeover_drops_after_switch": 11, "learner_drops_after_switch": 0}, "by_branch_reason": {"learner_failing_in_window (switch 61-62)": {"cases": [0, 1, 2, 4, 5, 8, 9, 11, 12, 14, 15, 17, 19, 21, 22, 23, 24, 25, 26, 29, 30], "n": 21, "learner_successes": 0, "takeover_successes": 6, "takeover_drops": 6, "takeover_completed_tail_fail": 9}, "learner_meeting_predicate_through_120 (switch 120)": {"cases": [3, 6, 16, 18, 27, 28, 31], "n": 7, "learner_successes": 7, "takeover_successes": 2, "takeover_drops": 5, "takeover_completed_tail_fail": 0}}, "flips": {"rescued": 6, "broken": 5, "both_success": 2, "both_fail": 15}, "transitions": {"rows": 840, "expert_saturated_fraction": 0.9985119047619048, "learner_saturated_fraction": 0.06964285714285715}}; controls {"gate1_identity": {"cases": 32, "observation_mismatch_cases": [], "outcome_mismatch_cases": [], "passes": true}, "gate2_prefix": {"switched_cases": 28, "mismatches": [], "passes": true}, "interpretable": true}.
**Decision: refuted**. Both gates passed (repeat bit-identical on 32/32; 28/28 takeover prefixes match), so the contrast is interpretable. Predicted takeover minus learner >= +0.25; observed +0.031 [-0.183, +0.246] (8/32 vs 7/32). The upper bound is below +0.25, which the predeclared rule names as refutation; confidence 0.75 rather than higher because the bound clears the threshold by only 0.004 with 32 cases. The aggregate hides opposing effects: on 21 already-failing learner states takeover rescued 6 but dropped 6 the learner kept; on the 7 states where the learner was succeeding it broke 5 (all drops 38-126 steps after the switch). Takeover doubles spin (10.3 vs 4.8 rad/s) but drops the pen in 11 switched cases vs 0 for the learner, so the expert's recovery basin covers some slow-spin learner states but not the learner's own grasp; expert labels at these states are 99.85% saturated.
**Next.** Do not collect expert takeover transitions as repair evidence for this learner as-is (R-052 should stay deferred or be re-scoped to the 6 rescued cases only). Closed-loop prefix re-execution is qualified for branching. A useful next question is whether the learner's own successful episodes (not the expert's) can serve as within-learner demonstrations for its failing cases.

## Does the symbolic learner actually use signed axial-spin feedback? (D-046, 2026-10-27)

**Direction.** Continue researching the single-step exploration pipeline with plateau-triggered trace repair, extended to pen spinning on the Shadow Hand (Isaac Lab Isaac-ShadowHandSpin-Stage{1,2,3}-Direct-v0, vendored in lares/envs/isaac_pen_spin; 157-d full obs, 20-d action, 60 Hz, 360-step episodes). Expert demonstrations: first try a hand-written scripted spinning controller; also search online for an existing trained pen-spinning policy; if neither qualifies, train an RL expert (rsl_rl PPO on the box). No budget ceiling on GPU hours or LLM calls. Known gaps to turn into questions: the env's spin_rate is roll about the pen's own long axis (not end-over-end twirl), success is per step with no episode rule, and the pipeline (manifests, obs schema, runner, records) is MetaWorld-only today. Debug issues, improve the pipeline, and turn unexpected findings into concrete research questions; document results and limitations.
**Idea R-053.** The frozen R-051 learner sustains spin through signed axial-angular-velocity feedback rather than an effectively open-loop gait.
**Prediction.** paired_success_difference decrease by at least 0.15 against baseline.
**Protocol.** `gpu run 'bash -lc "source ~/IsaacLab/source_env.sh && python scripts/run_shadowhand_spin_stages.py --config config/shadowhand_spin_stages.yaml --study learner_axial_sign --manifest-seed 4400 --output-dir logs/pen_spin_learner_axial_sign_s4400"'` at git `5c47f3302b`.
**Result.** {"paired_success_difference": -0.03125, "paired_success_difference_ci95": [-0.17559252888006782, 0.11309252888006782], "reversed_minus_learner": {"n": 32, "mean": -0.03125, "se": 0.07077302144543589, "low": -0.17559252888006782, "high": 0.11309252888006782, "p_value": 0.6618778576080805}, "win_loss_tie": {"wins": 2, "losses": 3, "ties": 27}, "arms": {"learner": {"n": 32, "successes": 5, "success": 0.15625, "completed": 0.75, "tail_hit_fraction": 0.2635460480149701, "hit_fraction": 0.2905263753130203, "mean_length": 279.34375, "mean_abs_spin": 6.891378875821829, "mean_transverse": 2.159853901481256, "drops": 8, "tail_spin_failures": 19, "median_fall_step": 39.0}, "axial_reversed": {"n": 32, "successes": 4, "success": 0.125, "completed": 0.71875, "tail_hit_fraction": 0.21691423867130752, "hit_fraction": 0.25035783198607076, "mean_length": 268.5625, "mean_abs_spin": 7.391530886292458, "mean_transverse": 2.499750124523416, "drops": 9, "tail_spin_failures": 19, "median_fall_step": 38.0}}, "post_hoc_drive_saturation": {"steps": 8939, "drive_true_mean": 0.9855383634567261, "drive_reversed_mean": 0.9241534471511841, "frac_abs_diff_gt_0.05": 0.10437409106164, "frac_true_drive_gt_0.99": 0.9457433717418056, "frac_abs_spin_gt_9": 0.11265242197113771, "selected_parameters": {"spin_target": 12.156, "speed_gain": 0.809}, "drive_law": "0.25 + 0.75 tanh(speed_gain (spin_target - spin))", "how": "evaluated on the unchanged learner's in-play axial spin (env spin_rate, rad/s, equal to the learner's 5*obs projection); post hoc, not predeclared"}}; controls {"identity_gate": {"cases": 32, "observation_mismatch_cases": [], "outcome_mismatch_cases": [], "passes": true}, "source_audit": {"fields_read": ["fingertip_pos", "hand_dof_pos", "hand_dof_vel", "in_hand_pos", "obj_angvel", "obj_pos", "obj_rot"], "reads_obj_angvel": true, "axial_projection": true, "axial_drive": true, "lines": ["omega = 5.0 * self.obs_field(obs, \"obj_angvel\")", "spin = torch.sum(omega * axis, dim=1, keepdim=True)", "transverse_omega = omega - spin * axis", "# Positive startup drive; excessive spin can reverse the stroke.", "self.speed_gain * (self.spin_target - spin)", "zero = torch.zeros_like(spin)"]}}.
**Decision: inconclusive**. Identity gate passed (bit-identical on 32/32). Predicted a decrease of at least 0.15; observed -0.031 [-0.176, +0.113] (4/32 vs 5/32, 27 ties). The predeclared refutation needs the interval entirely above -0.15 and its lower bound is -0.176, so neither the prediction nor its refutation is established; the interval includes zero. Post hoc, the reason the lever is weak is visible: the frozen parameters (spin_target 12.16, speed_gain 0.81) saturate the drive 0.25 + 0.75 tanh(gain (target - spin)) above 0.99 on 94.6% of the learner's in-play steps, and reversing the axial sign changes it by more than 0.05 on only 10.4% of steps. The source audit found a code path, but at its operating point the learner is effectively an open-loop fixed-amplitude stroke in axial spin; reversal also did not slow the pen (|spin| 7.39 vs 6.89).
**Next.** Treat the learner's spin as open loop when designing repair evidence (R-054). A sharper dependence test would set the operating point inside the unsaturated range (e.g. spin_target at its declared 8.0) or replace the measured axial rate with the target (zero error) rather than reversing it.

## Repair with successful traces from the learner's own grasp basin (D-047, 2026-10-28)

**Direction.** Continue researching the single-step exploration pipeline with plateau-triggered trace repair, extended to pen spinning on the Shadow Hand (Isaac Lab Isaac-ShadowHandSpin-Stage{1,2,3}-Direct-v0, vendored in lares/envs/isaac_pen_spin; 157-d full obs, 20-d action, 60 Hz, 360-step episodes). Expert demonstrations: first try a hand-written scripted spinning controller; also search online for an existing trained pen-spinning policy; if neither qualifies, train an RL expert (rsl_rl PPO on the box). No budget ceiling on GPU hours or LLM calls. Known gaps to turn into questions: the env's spin_rate is roll about the pen's own long axis (not end-over-end twirl), success is per step with no episode rule, and the pipeline (manifests, obs schema, runner, records) is MetaWorld-only today. Debug issues, improve the pipeline, and turn unexpected findings into concrete research questions; document results and limitations.
**Idea R-054.** Successful traces from the learner's own grasp distribution produce more transferable symbolic repairs than saturated PPO demonstrations from a different grasp basin.
**Prediction.** paired_success_difference increase by at least 0.15 against baseline.
**Protocol.** `gpu run 'source ~/.profile >/dev/null 2>&1; nohup .venv-metaworld/bin/python -u scripts/run_full_evolution.py --config config/pen_spin_plateau_repair.yaml --study self_trace_source --block-seeds 4500,4501,4502,4503,4504,4505,4506,4507 --output-dir logs/pen_spin_self_trace_source > logs/r054.log 2>&1 &'` at git `52776e2d94`.
**Result.** {"paired_success_difference": 0.0, "paired_success_difference_ci95": [0.0, 0.0], "self_minus_expert_fresh_incumbent": {"per_block": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "n": 8, "mean": 0.0, "se": 0.0, "low": 0.0, "high": 0.0, "p_value": 1.0, "wins": 0, "losses": 0, "ties": 8}, "promoted_blocks": {"self": 0, "expert": 0}, "fresh_incumbent_success_per_block": [0.25, 0.225, 0.275, 0.25, 0.2, 0.175, 0.175, 0.25], "per_arm": {"self": {"candidate_screen_successes": 27.0, "candidate_screen_episodes": 320, "candidate_screen_success": 0.084375, "candidate_screen_labels": {"off_centre": 99, "dropped": 97, "success": 27, "slow_spin": 97}, "mean_candidate_screen_per_block": [0.025, 0.0, 0.075, 0.025, 0.07500000000000001, 0.05, 0.07500000000000001, 0.35], "challenger_promotions": [{"block": 0, "challenger": "self:slot0", "challenger_promotion_success": 0.1, "parent_promotion_success": 0.26666666666666666, "reason": "lower_paired_success"}, {"block": 1, "challenger": "self:slot1", "challenger_promotion_success": 0.1, "parent_promotion_success": 0.2, "reason": "lower_paired_success"}, {"block": 2, "challenger": "self:slot1", "challenger_promotion_success": 0.23333333333333334, "parent_promotion_success": 0.3333333333333333, "reason": "lower_paired_success"}, {"block": 3, "challenger": "self:slot2", "challenger_promotion_success": 0.03333333333333333, "parent_promotion_success": 0.23333333333333334, "reason": "lower_paired_success"}, {"block": 4, "challenger": "self:slot2", "challenger_promotion_success": 0.16666666666666666, "parent_promotion_success": 0.1, "reason": "paired_success_interval_includes_zero"}, {"block": 5, "challenger": "self:slot3", "challenger_promotion_success": 0.0, "parent_promotion_success": 0.2, "reason": "lower_paired_success"}, {"block": 6, "challenger": "self:slot1", "challenger_promotion_success": 0.06666666666666667, "parent_promotion_success": 0.1, "reason": "lower_paired_success"}, {"block": 7, "challenger": "self:slot3", "challenger_promotion_success": 0.23333333333333334, "parent_promotion_success": 0.3333333333333333, "reason": "lower_paired_success"}], "isaac_episodes": 560, "isaac_processes": 40, "promoted_blocks": 0, "valid_candidates": 32, "candidates_beating_parent_screen": 3, "mean_fresh_incumbent": 0.225, "mean_best_screen": 0.1875, "llm_calls": 32, "llm_tokens": 942712}, "expert": {"candidate_screen_successes": 0.0, "candidate_screen_episodes": 320, "candidate_screen_success": 0.0, "candidate_screen_labels": {"dropped": 200, "slow_spin": 52, "off_centre": 68}, "mean_candidate_screen_per_block": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "challenger_promotions": [{"block": 0, "challenger": "expert:slot0", "challenger_promotion_success": 0.0, "parent_promotion_success": 0.26666666666666666, "reason": "lower_paired_success"}, {"block": 1, "challenger": "expert:slot2", "challenger_promotion_success": 0.0, "parent_promotion_success": 0.2, "reason": "lower_paired_success"}, {"block": 2, "challenger": "expert:slot3", "challenger_promotion_success": 0.0, "parent_promotion_success": 0.3333333333333333, "reason": "lower_paired_success"}, {"block": 3, "challenger": "expert:slot3", "challenger_promotion_success": 0.0, "parent_promotion_success": 0.23333333333333334, "reason": "lower_paired_success"}, {"block": 4, "challenger": "expert:slot1", "challenger_promotion_success": 0.0, "parent_promotion_success": 0.1, "reason": "lower_paired_success"}, {"block": 5, "challenger": "expert:slot3", "challenger_promotion_success": 0.0, "parent_promotion_success": 0.2, "reason": "lower_paired_success"}, {"block": 6, "challenger": "expert:slot0", "challenger_promotion_success": 0.0, "parent_promotion_success": 0.1, "reason": "lower_paired_success"}, {"block": 7, "challenger": "expert:slot3", "challenger_promotion_success": 0.0, "parent_promotion_success": 0.3333333333333333, "reason": "lower_paired_success"}], "isaac_episodes": 560, "isaac_processes": 40, "promoted_blocks": 0, "valid_candidates": 32, "candidates_beating_parent_screen": 0, "mean_fresh_incumbent": 0.225, "mean_best_screen": 0.0, "llm_calls": 34, "llm_tokens": 1085343}}, "parent": {"mean_screen": 0.25, "mean_promotion": 0.22083333333333333, "screen_labels": {"success": 20, "off_centre": 16, "dropped": 14, "slow_spin": 30}}, "secondary_post_hoc": {"self_minus_expert_mean_candidate_screen_per_block": {"n": 8, "mean": 0.084375, "se": 0.03922733080573871, "low": -0.008382897748502469, "high": 0.1771328977485025, "p_value": 0.06852009452351054}, "win_loss_tie": {"wins": 7, "losses": 0, "ties": 1}, "note": "not predeclared as primary; candidate screens are 10 cases, 4 candidates per arm per block"}}; controls {"parent": "frozen R-051 learner (config/pen_spin/r046_block3_incumbent), shared by both arms; fresh score shared when no promotion", "evidence": {"self": {"source": "logs/pen_spin_learner_recovery_s4310/donor.json", "cases": ["recovery-0003", "recovery-0006"], "rows_shown": 32, "packet_sha256": "5351187bd0b2a915017be1a77e943d5aaa19b55a24933693300db5cfb0dd26b8"}, "expert": {"source": "logs/pen_spin/r046_gait_scaffold/block1/expert.json", "cases": ["gait-0062", "gait-0063"], "rows_shown": 32, "packet_sha256": "3a2b053756538407d62a4d1917ed3a3749cdfed10e9c0f767290a3347e77451c"}}, "promotion_rule": "strict paired difference > 0 and t lower bound > 0 vs the parent on cases 10-39"}.
**Decision: refuted**. Predicted self minus expert final-incumbent fresh success of at least +0.15, paired over eight blocks. Observed 0.000 with a degenerate [0, 0] interval (8 ties): no candidate in either arm passed the strict promotion rule against the frozen parent, so both arms kept the parent (fresh mean 0.225) in every block. The interval's upper bound is below +0.15, which refutes the prediction under this protocol (4 candidates per arm per block, one linked correction). Secondary and not predeclared: self-trace candidates kept some parent behaviour (27/320 candidate screen successes, 3 candidates beat the parent screen, block-level mean candidate screen self minus expert +0.084 [-0.008, +0.177], 7 wins 0 losses 1 tie) while expert-trace candidates scored 0/320 and dropped the pen in 200/320 screen episodes versus 97/320 for self and 14/80 for the parent. So the evidence source changes how much an edit damages the grasp, but neither source produced an improvement over the parent; the claim that self traces give more transferable repairs is not established at the final-incumbent level.
**Next.** none

## R-046 scalar tuning: budget-matched CMA-ES versus random search (D-048, 2026-11-04)

**Direction.** Resume Shadow Hand pen spinning from docs/SESSION_2026-10-27_PEN_SPIN_SUMMARY.md. First compare CMA-ES and random search at equal rollout budgets on the frozen R-046 controller's spin and centering parameters (config/pen_spin/r046_block3_incumbent). Compare both winners with the parent on fresh paired cases. Report sustained success, drops, and saturation. Keep final test locked. If tuning improves success without more drops, use the winner for repair. If feedback activates without improving success, prioritize R-043. Defer cold search and R-055. Require verified learner-state recovery before imitation. Use llm_pool with 16 workers and high reasoning effort. Run one protocol at a time.
**Idea R-058.** CMA-ES can coordinate spin-drive and centering scalars to reduce destabilizing strokes, yielding more sustained successes than independent random draws at the same rollout budget.
**Prediction.** paired_success_difference increase by at least 0.1 against Equal-budget random-search screen winners on identical fresh cases.
**Protocol.** `gpu run 'bash -lc "source ~/IsaacLab/source_env.sh && python scripts/run_shadowhand_spin_stages.py --config config/shadowhand_spin_stages.yaml --study r058_scalar_cma_vs_random --log-dir logs/pen_spin/r058_scalar_cma_vs_random"'` at git `ad01a0833d`.
**Result.** {"paired_success_difference": 0.003124999999999996, "paired_success_difference_ci95": [-0.0762164089490264, 0.08246640894902638], "cma_minus_random_fresh": {"per_block": [0.175, -0.04999999999999999, -0.12499999999999997, -0.04999999999999999, -0.025000000000000022, 0.09999999999999998, 0.024999999999999994, -0.025000000000000022], "n": 8, "mean": 0.003124999999999996, "se": 0.03355349540020276, "low": -0.0762164089490264, "high": 0.08246640894902638, "p_value": 0.9284060163077056}, "cma_minus_random_fresh_drops": {"per_block": [2.0, 1.0, -3.0, 1.0, 0.0, 2.0, -1.0, -1.0], "n": 8, "mean": 0.125, "se": 0.6105471784742413, "low": -1.3187146649617392, "high": 1.5687146649617392, "p_value": 0.8436054350779514}, "cma_winner_minus_parent_fresh": {"per_block": [0.04999999999999999, 0.0, 0.024999999999999994, -0.024999999999999994, -0.07500000000000001, 0.025000000000000022, 0.0, -0.05000000000000002], "n": 8, "mean": -0.006250000000000002, "se": 0.014752421108802058, "low": -0.041133932723582664, "high": 0.028633932723582656, "p_value": 0.6845283355605283}, "cma_winner_minus_parent_fresh_drops": {"per_block": [1.0, -1.0, -2.0, 1.0, 0.0, 0.0, 1.0, -1.0], "n": 8, "mean": -0.125, "se": 0.3980981573144277, "low": -1.0663525573000951, "high": 0.8163525573000951, "p_value": 0.7626770646286338}, "cma_winner_minus_parent_promotion": {"per_block": [0.03333333333333333, 0.0, 0.03333333333333333, -0.06666666666666667, -0.1, 0.13333333333333333, 0.0, 0.06666666666666667], "n": 8, "mean": 0.012499999999999999, "se": 0.02592533540521734, "low": -0.048803676829853974, "high": 0.07380367682985398, "p_value": 0.6444071578961849}, "random_winner_minus_parent_fresh": {"per_block": [-0.125, 0.04999999999999999, 0.14999999999999997, 0.024999999999999994, -0.04999999999999999, -0.07499999999999996, -0.024999999999999994, -0.024999999999999994], "n": 8, "mean": -0.009374999999999998, "se": 0.029857361798582074, "low": -0.07997644179750713, "high": 0.061226441797507136, "p_value": 0.7626770646286338}, "random_winner_minus_parent_fresh_drops": {"per_block": [-1.0, -2.0, 1.0, 0.0, 0.0, -2.0, 2.0, 0.0], "n": 8, "mean": -0.25, "se": 0.4909902530309828, "low": -1.4110074596127395, "high": 0.9110074596127395, "p_value": 0.6262832660486098}, "random_winner_minus_parent_promotion": {"per_block": [-0.03333333333333333, 0.06666666666666667, -0.03333333333333333, 0.03333333333333333, -0.1, 0.1, 0.06666666666666667, 0.03333333333333333], "n": 8, "mean": 0.016666666666666666, "se": 0.023570226039551584, "low": -0.03906806144198076, "high": 0.07240139477531408, "p_value": 0.502354017079918}, "promotions": {"cma_winner": [4515], "random_winner": []}, "qualifying_repair_parents": {"cma_winner": [], "random_winner": []}, "feedback_activated_blocks": {"cma_winner": [], "random_winner": []}, "cma_winner_feedback_unsaturated_gain": {"per_block": [0.13915172353760447, 0.09440957250607344, -0.03835619777158774, 0.0032033426183843946, 0.07512239385498438, 0.07671600216189248, 0.01142061281337047, 0.0025790901099778858], "n": 8, "mean": 0.04553081747883747, "se": 0.02106079095753649, "low": -0.004270039577079332, "high": 0.09533167453475427, "p_value": 0.0674209271239758}, "random_winner_feedback_unsaturated_gain": {"per_block": [-0.008565459610027849, 0.013583202961040949, 0.039986420612813364, 0.026740947075208917, 0.01019759749303622, 0.009376169292811713, -0.016103760445682465, 0.00289157917291622], "n": 8, "mean": 0.009763337069014635, "se": 0.006365190569154866, "low": -0.00528794691681864, "high": 0.02481462105484791, "p_value": 0.16893274293859625}, "controllers": {"cma_winner": {"promotion": {"success_mean": 0.2333333333333333, "success_per_block": [0.23333333333333334, 0.26666666666666666, 0.3, 0.03333333333333333, 0.16666666666666666, 0.3333333333333333, 0.26666666666666666, 0.26666666666666666], "successes_total": 56, "drops_total": 35, "drive_saturated_fraction_per_block": [0.8650802314157843, 0.9630471297302639, 0.9976776100496147, 0.9333333333333333, 0.7791824694479562, 0.9853956976514703, 0.9534807713258525, 0.993386916268186], "action_saturated_fraction_per_block": [0.05976421788014409, 0.058650330994960975, 0.060878285653963894, 0.06833004602235372, 0.07341445427728614, 0.07608052101835405, 0.06371609107745942, 0.07115169396683284]}, "fresh": {"success_mean": 0.21875, "success_per_block": [0.25, 0.25, 0.225, 0.125, 0.175, 0.375, 0.175, 0.175], "successes_total": 70, "drops_total": 46, "drive_saturated_fraction_per_block": [0.8244599694852646, 0.8729342927176406, 0.9812848521885265, 0.9630880579010856, 0.8510990928122819, 0.9036755386565273, 0.9281182234358686, 0.9925826560404009], "action_saturated_fraction_per_block": [0.06501244679996789, 0.06084842255080256, 0.06505614544343442, 0.06417772416566143, 0.07397505233775296, 0.07024155669872512, 0.06638422616657297, 0.06455456482285173]}}, "random_winner": {"promotion": {"success_mean": 0.2375, "success_per_block": [0.16666666666666666, 0.3333333333333333, 0.23333333333333334, 0.13333333333333333, 0.16666666666666666, 0.3, 0.3333333333333333, 0.23333333333333334], "successes_total": 57, "drops_total": 34, "drive_saturated_fraction_per_block": [0.9967351237493418, 0.9661033698982113, 0.9951483696265373, 0.9095313204146012, 0.9416873449131513, 0.9943997556257, 0.9911411094705759, 0.9497967479674797], "action_saturated_fraction_per_block": [0.06034755134281201, 0.06151793655499555, 0.061446462822971906, 0.05408404686795854, 0.06760658696142567, 0.07283372365339578, 0.057076566125290024, 0.060823170731707316]}, "fresh": {"success_mean": 0.21562499999999998, "success_per_block": [0.075, 0.3, 0.35, 0.175, 0.2, 0.275, 0.15, 0.2], "successes_total": 69, "drops_total": 45, "drive_saturated_fraction_per_block": [0.9942621069543264, 0.9615776679996864, 0.8871499176276771, 0.9374901960784314, 0.9324901875272569, 0.9768485539250605, 0.9652983844378503, 0.9870183222413935], "action_saturated_fraction_per_block": [0.06895034809884477, 0.05847643691680389, 0.06207990115321252, 0.06586274509803922, 0.059058002616659395, 0.07439093888018236, 0.06819155951203429, 0.06014707090625257]}}, "parent": {"promotion": {"success_mean": 0.22083333333333333, "success_per_block": [0.2, 0.26666666666666666, 0.26666666666666666, 0.1, 0.26666666666666666, 0.2, 0.26666666666666666, 0.2], "successes_total": 53, "drops_total": 38, "drive_saturated_fraction_per_block": [0.9882468168462292, 0.9596042027950628, 0.9922105263157894, 0.976921274601687, 0.9546889089269612, 0.9877280625543006, 0.9491340812547653, 0.9599308451133937], "action_saturated_fraction_per_block": [0.058140167591685714, 0.05535550341732123, 0.061905263157894734, 0.05931935332708529, 0.06315374211000901, 0.06706125108601216, 0.05944886177976255, 0.06398352486524966]}, "fresh": {"success_mean": 0.225, "success_per_block": [0.2, 0.25, 0.2, 0.15, 0.25, 0.35, 0.175, 0.225], "successes_total": 72, "drops_total": 47, "drive_saturated_fraction_per_block": [0.9844912665465654, 0.9838126540673788, 0.9369497977635023, 0.9676230793352149, 0.9434209377455689, 0.9861617439178633, 0.9426891415131321, 0.990034590676989], "action_saturated_fraction_per_block": [0.06711443565442156, 0.05415776499589154, 0.06555634863986042, 0.06714487300094073, 0.06317122151401379, 0.06411353321925452, 0.06618972951783614, 0.05860237193213639]}}}, "screen_dynamics_pooled": {"cma": {"mean_candidate_successes_of_10_gen0": 1.796875, "mean_candidate_successes_of_10_gen7": 2.53125, "mean_candidate_successes_of_10_gens0_3": 2.03125, "mean_candidate_successes_of_10_gens4_7": 2.3671875, "screen_drops_per_80_episode_generation_gens0_3": 14.09375, "screen_drops_per_80_episode_generation_gens4_7": 12.6875, "winner_screen_success_mean": 0.4375, "winner_promotion_success_mean": 0.23333333333333334, "duplicate_candidates_total": 0}, "random": {"mean_candidate_successes_of_10_gen0": 2.03125, "mean_candidate_successes_of_10_gen7": 2.015625, "mean_candidate_successes_of_10_gens0_3": 1.93359375, "mean_candidate_successes_of_10_gens4_7": 2.02734375, "screen_drops_per_80_episode_generation_gens0_3": 15.03125, "screen_drops_per_80_episode_generation_gens4_7": 14.9375, "winner_screen_success_mean": 0.4, "winner_promotion_success_mean": 0.2375, "duplicate_candidates_total": 0}, "parent_screen_success_mean": 0.225}, "blocks": 8, "episodes": 12000, "llm_calls": 0, "pycma_version": "4.5.0"}; controls {"parent": "the unchanged frozen R-046 block-3 learner, scored in its own process on every promotion and fresh manifest (and on each block's 10 screen cases)", "random_arm": "frozen copy of the CMA-ES initial sampling law: a fresh pycma instance per generation (identical options and BoundTransform bounds), asked once, never told", "matched_batches": "every generation of either arm ran in one fresh Isaac process of 8 candidates x 10 screen cases = 80 environments; winners and the parent in their own processes per manifest", "frozen_before_promotion": "each arm's winner.json and the block's winners.json were written before any promotion or fresh episode; fresh cases selected nothing", "decision_artifact": {"rule": {"repair_parent": "strict promotion on cases 10-39, fresh success interval lower bound above zero, and no more fresh drops than the parent, in the same block", "feedback_activated": "unsaturated in-play drive fraction up by at least 0.25 versus the parent on the fresh cases with the paired interval above zero"}, "qualifying_repair_parents": [], "repair_parent": null, "feedback_activated_blocks": {"cma_winner": [], "random_winner": []}, "route": "neither follow-up is enabled by this result"}}.
**Decision: refuted**. Predicted that budget-matched CMA-ES would beat random search by at least +0.10 fresh sustained success when tuning the frozen R-046 learner's twelve spin and centering scalars inside their declared bounds (8 blocks, 64 candidates per arm per block on 10 screen cases, winners and parent scored on 30 promotion and 40 fresh cases each, 12,000 episodes, no LLM). Observed +0.003 [-0.076, +0.082] over the eight block contrasts: the upper bound is below +0.10, the predeclared refutation. Neither winner differs from the parent on fresh cases (CMA -0.006 [-0.041, +0.029]; random -0.009 [-0.080, +0.061]) or on promotion pooled over blocks; one strict promotion in sixteen (CMA, block 4515, 10/30 vs 6/30) did not confirm on fresh cases. Drops 46/45/47 of 320 with every interval through zero. The spin drive stays saturated above 0.99 on most recorded in-play steps for every controller (parent 94-99%, CMA winners 82-99%; unsaturated gain +0.046 [-0.004, +0.095]), far from the +0.25 activation margin, so tuning did not activate spin feedback. Not predeclared: CMA-ES did adapt on the screen (mean candidate screen successes 1.8 to 2.5 of 10 across generations, screen drops per generation 14.1 to 12.7; random flat at 2.0 and 15.0) and the screen winners scored 0.44/0.40 against the parent's 0.225 on the same ten cases, yet 0.233/0.237 against 0.221 on promotion: at this success rate selection on ten screen cases is selection noise, and the twelve scalars do not contain a better controller that ten cases can identify. No (block, arm) qualifies as a repair parent and no block activates feedback, so R-056 and R-057 stay deferred.
**Next.** Neither follow-up is enabled: no tuned winner qualifies as a repair parent (R-056 stays deferred) and no block activates feedback (R-057 stays deferred). Scalar tuning of this learner at this screen size is exhausted; a transferable gain would need a screen large enough to resolve 0.05 differences (about 40+ cases per candidate) or a structural change rather than a scalar one.
