# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment setup

Two environment specs exist and have diverged — pick by pipeline:

- `config/environment.yaml` — conda env `Metaworld-v2`, what the README installs (MetaWorld path).
- `environment.yml` — the root spec used on the Isaac Lab GPU box.

On the GPU box the working environment is a pre-built venv (`.venv-metaworld`, Python 3.10,
torch 2.7+cu128, metaworld 3.0.0, mujoco 3.8, openai 2.x), not the conda env; it is not checked in.

```bash
export MUJOCO_GL=egl                    # required on a headless machine with no DISPLAY
export OPENAI_API_KEY=...               # required for any LLM/Stage-4 path
```

There is no `setup.py`/`pyproject.toml`. The `lares` package is importable only because every
script/test does `sys.path.insert(0, _PROJECT_ROOT)`. **All commands must be run from the project root.**

## Common commands

```bash
# Phase 0 evaluation contract: build the manifests, reproduce the baselines, check AC-0.
# Exits non-zero if any AC-0 check fails. Run this before trusting any other number.
python scripts/lock_baseline.py                      # reuse the committed manifests
python scripts/lock_baseline.py --build-manifests    # regenerate config/manifests/*.yaml
python scripts/lock_baseline.py --collect-demos 150  # expert buffer with episode ids

# Phase 3 fitting comparison: four optimizers on the frozen structure bank, matched budgets.
python scripts/build_structure_bank.py               # regenerate lares/fitting/structures/
python scripts/benchmark_fitting.py                  # full bank, 2000-evaluation budget
python scripts/benchmark_fitting.py --family simple --budget 500 --no-rollouts

# Regenerate the search summary from saved experiment records alone.
python scripts/summarise_experiments.py --log-dir ./logs/evolution

# Phase 5: expert semantics, objectives, sampling and learner-state imitation.
python scripts/audit_expert_actions.py                              # clipping semantics, phases
python scripts/compare_objectives.py --budget 2000 --family simple  # E7, sampling, E8

# Phase 6: diagnose a policy by controlled substitution, then repair and validate it.
python scripts/diagnose_repair.py --structures simple_standoff      # study + one bounded repair
python scripts/ablate_repair.py --budget 200 --seeds 0 1            # AC-6 / E9, five arms
python scripts/ablate_repair.py --summarise logs/repair_ablation/<run>.json  # re-print, no rollouts

# Phase 7: freeze the method before any final-test episode runs, and check for drift.
python scripts/freeze_method.py --freeze-id v1
python scripts/freeze_method.py --verify             # non-zero exit if anything moved
python scripts/final_report.py                       # aggregate report, from saved files only

# FR-11: capture the selected frames for a candidate's worst failures, and merge with the numbers.
python scripts/analyse_failures.py --structures simple_standoff      # no model: media + numbers only

# Symbolic-policy evolution (primary entry point; config-driven, no CLI flags besides --config)
python scripts/run_full_evolution.py
python scripts/run_full_evolution.py --config path/to/custom.yaml   # see config/run_full_evolution.yaml

# Isaac Lab ShadowHandSpin 3-stage smoke run (needs IsaacLab at config's isaaclab_root, CUDA)
python scripts/run_shadowhand_spin_stages.py --config config/shadowhand_spin_stages.yaml

# Plots and expert-policy inspection
python scripts/plot_training_dynamics.py --log-path ./logs/evolution/<run>/training_dynamics.jsonl
python scripts/visualize_expert_policy.py --config config/run_full_evolution.yaml --episodes 10
```

### Tests

Two incompatible styles coexist:

```bash
# Self-running scripts (print PASS/FAIL counts, sys.exit(1) on failure) — NOT pytest-collectable
python tests/test_phase1_phase2.py          # no API key needed
python tests/test_training_pipeline.py      # self-tiering; Tier 1 mocks always run
python tests/test_env_action_response.py    # needs a working env from config/run_full_evolution.yaml

# unittest style — pytest is NOT installed in .venv-metaworld, use unittest
python -m unittest tests.test_record_episode_gif tests.test_two_phase_policy_generation
python -m unittest tests.test_evaluation_manifest tests.test_manifest_runner \
  tests.test_policy_interface tests.test_evaluation_report tests.test_fitting \
  tests.test_search tests.test_objectives tests.test_repair tests.test_freeze \
  tests.test_final_report tests.test_vision
python -m unittest tests.test_record_episode_gif.TestRecordEpisodeGif.test_saves_gif_and_returns_metadata
```

`tests/test_training_pipeline.py` self-tiers: Tier 1 mocks always; Tier 2 needs `metaworld`; Tier 3
needs `OPENAI_API_KEY`. It imports `EvolutionOrchestrator`, the only orchestrator in the tree.

`tests/test_generate_policy.py` is **not** a test suite — it is a validation harness *appended to
LLM-generated source* and run as a subprocess by the generation pipeline (`policy_generation.py`).
Never run it directly; changing it changes what the LLM's output must satisfy. It now delegates to
`lares/core/policy_validator.validate_policy`, so there is one list of rules rather than two that
drift. It recovers the generated source by splitting its own file on the harness banner, and takes
`--env_name` to bind the observation schema.

## Architecture

The original LaRes reward-search pipeline (SAC + LLM reward evolution: `scripts/LaRes_*.py`,
`lares/rl/`, `lares/utils/utils.py`, the `prompts/`/`no_init_prompts/` template dirs) was **deleted**
in the codespace cleanup. Only symbolic-policy search remains. Look for it in git history, not on disk.

**Symbolic policy search.** `scripts/run_full_evolution.py` →
`lares/core/training_pipeline.py`. The LLM writes *policies*, not rewards. Four stages:

| Stage | Function | What it does |
|---|---|---|
| 1 | `generate_dataset()` | MetaWorld expert policy (`EXPERT_POLICY_MAP`) over the *train* manifest → `DemoBuffer` |
| 2 | `behavioral_cloning()` | `MSE(tanh(mean), expert action) + 0.01 * std.mean()`; `clip_params()` after each step. Splits validation off **by episode** (`split_buffer_by_episode`), reports per-axis train and validation loss, gradient norms, bound activity and loss by phase. Takes a `rollout_probe` and keeps the best-scoring parameters, because success is not monotonic in gradient steps |
| 3 | `rl_finetune()` | GRPO: advantages relative to the *group* mean return, + entropy bonus, + L2 KL toward the BC init |
| 4 | `EvolutionOrchestrator.run()` / `llm_evolution()` | LLM proposes structures; each runs BC→RL→eval; the next generation gets a structured evidence block, not two scalars |

**The evaluation contract (`lares/eval/`).** Nothing scores a policy without a manifest.
`EvaluationManifest` is an ordered, immutable list of `EpisodeCase`s, each naming a task
placement, a reset seed and a policy seed. `build_task_pool()` generates placements from
`MT1(env_id, seed=S)`; the pool seed is part of the contract because unseeded MT1 draws
different goals every construction. The committed splits live in `config/manifests/`:
train (150 cases, seeds 1000-1002), development (50, seed 2000), final_test (100, seeds
3000-3001), disjoint by placement hash. `evaluate_manifest()` replays a manifest and returns
a `ManifestResult` with a per-episode table; `paired_difference()` compares two results over
shared cases. `PipelineEnvs` holds one env per stream and refuses a shared instance.

**Diagnostics and reports.** `lares/eval/diagnostics.py` measures each episode through the
observation schema: goal progress, lateral drift, object displacement, closest approach, per-axis
saturation and variation, plus a coarse `failure_label` (`never_reached_object`,
`object_not_moved`, `pushed_away_from_goal`, `stopped_short_of_goal`). `lares/eval/report.py`
assembles those into an `EvaluationReport` per candidate per checkpoint, with `validity` from the
validator, `fitting` from BC, `rollout` aggregates and the full per-episode table. Anything not
measured is listed in `unavailable_fields` and left `None`; nothing is zero-filled. Reports land in
`<log_dir>/gen_<N>/reports/`.

**Staged development budget (`lares/eval/promotion.py`).** Every candidate is screened on
`screen_episodes` (10), and only those clearing a rule declared before the generation ran get the
`expanded_episodes` (30) evaluation. The screen is a *prefix* of the expanded set, so the two stay
paired. The rule promotes on any success, and when no candidate succeeds it promotes the top
`fallback_top_k` by mean signed goal progress, so a generation of all-zero candidates still teaches
the search something. `compare_to_incumbent()` decides promotion on the *paired* success difference,
never on two separately computed intervals. Decisions are written to `gen_<N>/screening.json`.

**Checkpoints.** The orchestrator evaluates `zero_shot` (before BC), `intermediate` (BC halfway, via
`behavioral_cloning(checkpoint_callback=...)`) and `fitted`. They are separate reports and are never
merged: a structure that is decent untrained and worse after fitting is a fitting failure, which an
aggregate hides.

**Search evidence and archive (`lares/search/`).** `schemas.py` holds three contracts. `PolicyIdea`
requires every parameter to declare a role, type, units, initial value and range, and requires the
idea to state what the *untrained* policy should do, since the zero-shot checkpoint is evaluated.
`MutationProposal` is rejected before code generation if `predicted_metric` names something the
reports do not measure, because a prediction that cannot be wrong teaches nothing; exploration mode
is exempt. `ExperimentRecord` is written for every attempted candidate including the ones that never
compiled, and refuses to validate if an evaluated record lacks a code hash, manifest ids or an action
mode. `archive.py` keeps three slots: most robust, simplest competitive, and most behaviourally
distinct, where distance is measured on the failure profile and geometry rather than source text.
Intervention results are recorded but can never occupy a slot. `feedback.py` builds the block the
generator sees: full code for two parents chosen to *differ*, a ranked population table, per-episode
geometry, dead parameters, and errors from candidates that never compiled. It always carries the
caveat that validation loss correlates with success at +0.19 on this task, the wrong sign.

Records land in `<log_dir>/experiments/`, the evidence block in `<log_dir>/gen_<N>/feedback.md`.
`scripts/summarise_experiments.py` regenerates the search summary from records alone.

**Objectives and execution contracts (`lares/fitting/objectives.py`).** Five objectives, each
carrying the execution it assumes: `deterministic_mse` (the primary baseline, scale frozen),
`legacy_mse_std_penalty` (what the pipeline used before, kept only for comparison),
`tanh_gaussian_nll`, `residual_scale`, and `censored_gaussian`. `check_execution_matches` refuses to
score an objective under an execution it does not model, so an action-space regression cannot be
reported as a likelihood. Scale parameters are found by *measuring* which parameters move the scale
but not the executed action, not by matching a name like `log_std`.

The expert audit (`scripts/audit_expert_actions.py`) settles which likelihood is right. MetaWorld's
expert is an unbounded proportional controller with gain 10 that does no clipping; the endpoint mass
in the targets is ours. So the censored Gaussian is the matching model and needs clipped execution,
and a tanh-Gaussian likelihood is wrong twice over.

**Behavioural cloning is non-monotonic in gradient steps.** Development success peaks around 1000
steps and collapses to zero by 3000, while validation loss falls the whole way. Measured on two
structures over three seeds; at the loss minimum both score zero. `behavioral_cloning` therefore
takes a `rollout_probe` and keeps the best-scoring parameters, and `EvolutionOrchestrator` supplies
one that scores the screening cases `rollout_checkpoint_probes` times (default 4) during fitting.
Selecting on loss cannot work here. The probe ranks on success then mean signed goal progress, and
prefers the later checkpoint on an exact tie. Set `rollout_checkpoint_probes=0` to fit to the end as
the pipeline used to.

**Ground-truth phases (`lares/fitting/phases.py`).** The expert's `_desired_pos` has three explicit
branches, so `expert_phase_labels` is ground truth rather than inferred from a learned gate;
`verify_against_expert` checks it against the branch the expert actually takes. `balanced_indices`
gives each occupied phase an equal share of a minibatch, which matters because the expert spends
75% of its steps in the push phase.

**Learner-state imitation (`lares/fitting/dagger.py`).** Rolls out on training placements only,
counts every expert query and environment step, and validates each label by branching the simulator,
applying the label, measuring the distance to the expert's own desired position, and rewinding.
About a third of labels fail that test, so the scripted expert is not a reliable recovery controller.

**Fitting comparison (`lares/fitting/`).** `structure_bank.py` owns a frozen bank under
`lares/fitting/structures/`: fifteen candidates ported from the 2026-08-28 search plus three
hand-written simple-family references. The ported ones had their raw observation slices rewritten to
named accessors by `port_raw_indices`, with `check_port_equivalence` proving the outputs are
identical. Rebuild with `scripts/build_structure_bank.py`. `optimizers.py` implements four methods
behind one `fit(method, structure, data)` call: `adam_baseline` reproduces the live path exactly,
`adam_scaled` gives each parameter a step proportional to its declared range plus a cosine schedule,
`multistart_adam` splits the same budget across perturbed restarts and picks on validation loss, and
`differential_evolution` is derivative-free via scipy. All four optimise the identical objective and
receive the same objective-evaluation count; wall time and transitions are reported too, because
equal evaluation counts are not equal compute. `sensitivity.py` measures how much each parameter
moves the executed action, in units of that parameter's own range. Run the comparison with
`scripts/benchmark_fitting.py`; the selection rule is declared in `benchmark.py` before any run.

**The final-test split is locked (`lares/eval/freeze.py`).** FR-4 requires that winner selection
and final estimation use different data, and nothing enforced it: any call could score a candidate
on `push-v2_final_test.yaml` and carry on. `evaluate_manifest` now refuses a manifest whose split is
`final_test` unless a `final_test_session` is open, which needs a complete `FreezeRecord`. That
record digests the models, every prompt template, the frozen source files, the committed manifests,
the optimizer settings, the selection rule and the budgets; `verify()` recomputes them and names
what drifted rather than returning a boolean. Write one with `scripts/freeze_method.py`, check for
drift with `--verify`, which exits non-zero if anything moved.

Inside a session every evaluation is appended to `final_test_ledger.jsonl`. Scoring the same code
hash twice is refused unless the caller passes `allow_rescore=True`, and that admission is written
into the entry. The ledger is the record of how many times the held-out data was actually looked at,
which is what a selection-effect argument needs.

**VLM-assisted failure analysis (`lares/vision/`).** Optional throughout, and built so the contract,
the media rule and the merge are all testable without a model. `schema.py` holds
`VisualFailureAnalysis`: every claim must cite a frame that was actually shown and a timestep inside
that frame's span, and every recommended follow-up must name a metric the reports measure, the same
rule `MutationProposal` applies to predictions. An analysis that fails either check is dropped and the
reason recorded, never repaired into shape.

`media.py` owns the fixed selection rule, declared before any analysis runs: episode start, closest
approach, first gate crossing of 0.5, largest single-step object movement, termination. A moment the
episode cannot supply is named in `unavailable` rather than replaced by a nearby frame. Capture hangs
off a new `observer` hook on `run_episode`, so it is not a sixth copy of the rollout loop, and it
refuses the final-test split outright because FR-11 says that media must never reach the search loop.

`merge.py` runs the recommended checks against the episode record and marks disagreement between the
model's reported stage and the geometry's failure label. It never picks a winner. `analyst.py` puts
the model behind an interface: `ScriptedAnalyst` is the test double and the numeric-only arm of E10,
and `QwenAnalyst` refuses to invent a checkpoint, because which Qwen and which quantization are open
decisions that would otherwise get frozen into that comparison by default.

Run it with `scripts/analyse_failures.py`. Without `--model` it still captures and saves the selected
frames and reports which moments were unavailable, which is the half of FR-11 that needs no download.

**The aggregate report (`lares/eval/final_report.py`).** AC-7 asks that one documented command
regenerate the report from saved artifacts. `scripts/final_report.py` reads four streams and runs
nothing: experiment records, repair studies, ablation runs and the freeze directory. Ablation runs at
different budgets are never pooled, because arms are comparable only at a matched cost. Quantities
AC-7 requires that no artifact carries are listed in `unavailable` with the field each would come
from, so the gap names its own fix; today that is fourteen of fifteen, most of them LLM and timing
instrumentation the pipeline does not yet record.

**Intervention-guided repair (`lares/repair/`).** A failure label says what went wrong, not which
part is responsible. `interventions.py` answers that by replacing exactly one part with a known-good
source and re-running the same cases: one or more action axes from the scripted expert, the whole
action during a named phase, an expert prefix, or a phase gate clamped open. `InterventionActor`
goes through the same manifest runner as everything else, so an intervention pairs case-for-case
with the unmodified control. `SymbolicPolicy.set_gate_override` makes the gate lever real, and
`gate_override_is_effective` checks it bites: a policy that calls `record_gate` without using the
returned value has telemetry but no lever, and forcing such a gate would produce a null result that
reads as evidence.

`hypotheses.py` finds recurrent failure clusters (a label needs `MIN_CLUSTER_SIZE` = 3 episodes),
gives each at least two competing explanations that blame *different* components, and marks one
supported only when its substitution moves the predicted metric in the predicted direction on the
same cases. Rejected explanations are stored beside accepted ones.

`repair.py` turns a supported explanation into a bounded edit. The component becomes a scope of
action axes and expert phases; the parameters inside it are found by *measuring* which ones move the
scoped axes (or the named gate), never by matching a name; and `bounded_change` re-reads the policy
afterwards, so "only this component changed" is checked rather than asserted. Two operators:
`apply_repair` refits the in-scope parameters against the expert on the scope's phases, and
`search_repair` searches them directly against the focused metric, which is the one that can move a
geometry term the expert data itself produced. `accept_repair` adds the guard the success threshold
needs on this task: when no case succeeded, no repair can fail a success-regression check, so the
protected cases are held to a continuous progress check instead.

**The AC-6 ablation does not support the repair loop.** Five arms, six replicates, 200
development-screen episodes each: optimizer-only reaches 0.650 held-out success against the full
loop's 0.517, a paired +0.133 whose interval excludes zero. The controlled substitution changed the
chosen component in two of six replicates and the outcome in one, for 10% of the budget. Localisation
does look worth something, beating an unlocalised search by 0.150 and a randomly chosen component by
0.083, but both intervals include zero over six runs. Optimizer-only wins because its rollout probe
recovers the pre-collapse checkpoint, which is the Phase 5 finding rather than anything about repair.
Read `RESEARCH_HANDOFF.md` before building on this module.

Run one diagnosis and repair with `scripts/diagnose_repair.py`, and the AC-6 ablation with
`scripts/ablate_repair.py`. Every arm of that ablation gets the identical budget of
development-screen episodes and is scored on `manifest.tail(...)`, which no arm sees. `records_from_study`
writes one `ExperimentRecord` per substitution, each marked `intervention=True`, so the rollouts are
visible to `summarise_experiments.py` and can never be ranked as fitness; the repair itself is
recorded as an ordinary rankable candidate, because no expert helped it.

Prompts in `lares/utils/policy_prompts/` (six templates). Generation lives in
`lares/core/policy_generation.py`.

**Isaac Lab ShadowHandSpin.** A second, parallel stack merged in from the Isaac branch:
`scripts/run_shadowhand_spin_stages.py` → `lares/envs/isaac_lab_adapter.py`
(`IsaacLabSingleEnvAdapter`, `launch_isaac_app`), configured by `config/shadowhand_spin_stages.yaml`.
The adapter turns a batched Isaac env into the same 4-tuple `step` / `(obs, info)` `reset` /
`info["success"]` shape the MetaWorld `env_wrapper` provides, but the stage orchestration is
*copied*, not shared: that script has its own GIF recorder, dataset collector and action-squashing
helper duplicating `training_pipeline.record_episode_gif` / `generate_dataset`. Changes to the
action or rollout convention must be made in both. Design notes: `docs/PIPELINE_REFINEMENT_SPEC.md`.

### The `SymbolicPolicy` contract

Everything depends on `lares/core/symbolic_policy.py`. A generated policy must:

- be a class named exactly `GeneratedPolicy(SymbolicPolicy)`, constructed as `(obs_dim, action_dim)`
- `forward(obs) -> (mean, std)`, both `(batch, action_dim)`, `std > 0`, differentiable to `nn.Parameter`s
- read the observation **only** through `self.obs_field(obs, "<name>")`. Raw slicing such as
  `obs[:, 0:3]`, `obs.narrow(...)` or `torch.split(obs, ...)` is rejected by an AST check.
  `obs.shape[0]` is fine.
- contain **no** NN modules (`Linear`/`Conv`/`LSTM`/…) — `validate()` rejects them
- declare `get_param_ranges() -> {param_name: (lo, hi)}` for every parameter, with initial values
  inside their own ranges; training calls `clip_params()` after every gradient step
- keep row `i` of the output dependent only on row `i` of the input, and actually respond to every
  field the schema marks `required` (for MetaWorld: `tcp`, `obj`, `goal`)
- optionally call `self.record_gate("name", tensor)` to expose a phase gate. The value is detached,
  so recording cannot change the action or the gradient.

**Observation schema (`lares/core/obs_schema.py`).** `get_obs_schema(env_id)` returns the layout and
raises for an unknown task rather than handing back a blank one. `ObsSchema.check_against_env`
compares `tcp`, `obj`, `obj_quat` and `goal` against what the simulator reports, which is the check
that would have caught the original `push-v2` failure. Fields with no simulator accessor are listed
by `unverifiable_names()` rather than silently passed. Infrastructure binds the schema with
`set_active_schema` or the `active_schema` context manager before constructing a policy; the policy
captures it at construction.

**Validator (`lares/core/policy_validator.py`).** `validate_policy(...)` returns a
`ValidationReport` collecting every defect, each with a stable category string, rather than raising
at the first. Categories cover raw indexing, missing schema, forbidden modules, undeclared and
phantom parameters, inverted ranges, out-of-range initial values, wrong shapes, non-finite output,
non-positive `std`, batch coupling, insensitivity to a required field, gate-telemetry misuse and
serialization. Probes run at several observation scales because a distance gate with a 0.06 m
threshold is fully closed at unit scale, which would make both the coupling and sensitivity checks
measure nothing. `GateAggregator` turns per-step gates into the statistics AC-2 asks for.

Generation flow in `get_symbolic_policies()`: format prompts → LLM call → regex-extract code block →
find `class GeneratedPolicy` → write temp file (`imports + generated_code + tests/test_generate_policy.py`)
→ run as subprocess and require `"Success!"` → only then `exec()` in-process and `validate()`.
Failures are fed back to the LLM as repair feedback. Two-phase mode (`policy_gen_two_phase: true`)
first asks for JSON design ideas (`ideas_*.txt`), then implementations (`policy_impl_mode:
batched | per_idea`).

## Development Workflow

1. Read `spec.md` and `IMPLEMENTATION_PROGRESS.md` before starting.
2. Continue from the first unfinished requirement.
3. Implement working code, not only a plan.
4. Run focused tests after each change.
5. Fix failures before continuing.
6. Update `IMPLEMENTATION_PROGRESS.md` with changes, tests, and remaining work.
7. Record experiment results and design questions in `RESEARCH_HANDOFF.md`.
8. Continue through safe tasks without asking for confirmation.
9. Stop only for a genuine blocker, destructive action, missing requirement, or expensive experiment requiring approval.
10. Do not run full training or large VLM experiments without approval.
11. Preserve existing user changes and avoid unrelated refactoring.

## Things that will bite you

- **MetaWorld 3.0 is installed, but all task names in this codebase are `-v2`.**
  `make_metaworld_env()` rewrites `-v2` → `-v3` and looks up `ALL_V3_ENVIRONMENTS`. Keep using the
  `-v2` spelling in configs and dicts; do not "fix" it to `-v3`.
- **`use_mt1: true`** (in `run_full_evolution.yaml`) makes `push-v3` fully observable with valid
  `rand_vec`s, and now *requires* an explicit task pool: `make_metaworld_env(cfg, seed, tasks=...)`
  raises without one. `env_wrapper.reset(case)` installs `case.task_id` and reseeds with
  `case.reset_seed`. There is no `_mt1_rng` and no hidden task stream. The non-MT1 path sets
  `seeded_rand_vec = True` so its placement draw also follows the case seed.
- **Every reset names its case.** `env_wrapper.reset()` with no argument raises. That is deliberate:
  it is what makes an episode a function of its case rather than of how much work preceded it.
  Mocks and the Isaac Lab adapter accept `case=None`; use `lares.eval.manifest.synthetic_cases()`
  for backends with no placements to name.
- **`get_dict()`**: env state is read through `env._env.get_dict()`, which MetaWorld V3 does not
  provide. `_patch_get_dict()` in `lares/utils/metaworld_env.py` attaches a picklable `_GetDict`
  shim whose aliases (`tcp`/`hand_pos`/`gripper` are one array, `obj`/`handle`/`current_pos`
  another) exist for prompt vocabulary. Its original consumer, the reward-search pipeline, is gone —
  keep it only for LLM-generated policies. New tasks need their variables exposed there.
- **Rendering**: MuJoCo needs `render_mode="rgb_array"` at *construction* and `MUJOCO_GL` set *before*
  the env is built. Call `ensure_mujoco_headless_gl()` early. `record_episode_gif()` silently logs
  "No frames captured" rather than raising.
- **`lares/core/training_pipeline.py` has no `__main__`** — it is a library; drive it through
  `scripts/run_full_evolution.py`.
- **Rollout logic is still duplicated, now four ways** — `generate_dataset`,
  `_collect_trajectories` and `record_episode_gif` in `training_pipeline.py`, plus
  `lares/eval/runner.run_episode`, which `evaluate_policy` and
  `run_full_evolution.evaluate_expert_policy` both delegate to. The Isaac stack adds a fifth,
  `run_shadowhand_spin_stages.evaluate_stage_policy`, whose `_policy_action` omits the
  `action_space.high *` scaling the others apply. Change one, check all five.
- **`best_policy_code.py`, `best_policy.pt`, `data.pkl`** are *runtime outputs* written into
  `log_dir`, not repo files. `logs/` and `*.pkl` are gitignored. `config/manifests/*.yaml` is the
  exception: manifests are committed because they define the evaluation contract.
- **The saved incumbent scores 0.00, not 0.30.** `logs/evolution/best_policy_code.py` is generation
  2 candidate 0, which its run reported at 402.76 reward and 0.30 success on 10 unmatched episodes.
  Re-scored on the development manifest it gets ~105 reward and 0.000 success. Treat 0.00 as the
  incumbent; see `RESEARCH_HANDOFF.md`.
- **`config/shadowhand_spin_stages.yaml` has dead keys** — `stage1_task`/`stage2_task` are never
  read, and all three "stages" run against the one env named by `pipeline_task`.

## Logging

`lares/core/training_logger.py` defines every metric key as a constant (`BC_*`, `RL_*`, `EVO_*`) —
use those constants, never string literals, so `plot_training_dynamics.py` keeps working.
`TrainingLogger` appends JSONL to `<log_dir>/<run_id>.jsonl` (`run_id` defaults to a `%Y%m%d_%H%M%S`
timestamp). `run_full_evolution.py` writes flat into `log_dir` (default `./logs/evolution`):
`demo_<env>.pkl`, `best_policy_code.py`, `best_policy.pt`, per-generation LLM responses and code, and
optional demo GIFs under `gen_<N>/`. `run_shadowhand_spin_stages.py` instead creates a timestamped run
dir holding `config.yaml`, `summary.json`, `training_dynamics.jsonl`, the demo pickle, and `*_policy.pt`.
