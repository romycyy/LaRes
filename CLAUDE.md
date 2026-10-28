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

**LLM model (decided 2026-10-13): `gpt-6.1-sol` with `reasoning_effort: high` for every LLM call** —
ideation, implementation and correction in `run_full_evolution.py`, the two-generation smoke config,
`scripts/compare_generation_modes.py` and Tier 3 of `tests/test_training_pipeline.py`. The defaults live in
`lares/core/policy_generation.py` (`DEFAULT_LLM_MODEL`, `DEFAULT_REASONING_EFFORT`); configs name both
explicitly, and every ledger entry records the effort in `decoding`. The model accepts only temperature 1
(checked against the API on 2026-10-13, as are `reasoning_effort` and `n>1`). From 2026-09-14 to 2026-10-13
the model was gpt-5.6-sol (`logs/baseline_gpt56_sol_high`). No token cap on LLM runs for now.
Runs completed before 2026-09-14 (`logs/evolution_debug_full`, `logs/debug_smoke_two_generation`, Tier 3
notes) used gpt-5 or gpt-4o-mini, so their token caps and results do not transfer. The vision analyst
(`lares/vision/`) is a local Qwen by design and is not an OpenAI call; it stays blocked with E10.

`pyproject.toml` exists (setuptools, packages `lares*`) so `pip install -e .` works, but no environment
has it installed yet: every script/test still does `sys.path.insert(0, _PROJECT_ROOT)`. **All commands
must be run from the project root.**

**Shared helpers (added 2026-10-14; one copy each, every older name re-exports them).** `lares/utils/hashing.py`
(`sha256_bytes/text/file/json`, and the 16-character `short_hash_*` used by `freeze.py` and `source_hash`; the
prefix length is a stored format), `lares/utils/gitstate.py` (`git`, `worktree_state`, `require_clean_worktree`),
`lares/utils/seeds.py` (ledger reader, `spent_seeds`, `seed_conflicts`, `reserve_seeds`; scans `config/**` *and*
`logs/**` manifests for every caller, where the research ledger used to scan `config/` only),
`lares/utils/env_ids.py` (`resolve_registry_env_id`, the v2→v3 rule), `lares/eval/stats.py` (`t_summary`,
`t_interval`, `t_bounds`, `win_loss_tie`: the one Student-t rule behind promotion, Stage E/F reports and the
comparison scripts), `lares/fitting/diagnostics.py` (`per_axis_mse`, `bound_activity`, `loss_by_phase`, so
`fitting` no longer imports them from the pipeline module) and `structure_bank.load_structures(ids)`.
`lares/utils/__init__.py` resolves the MetaWorld builders lazily so these import without MetaWorld.

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

# Regenerate the search summary from saved experiment records alone. Traceability is "complete"
# only if every record's lineage, feedback, prompts, usage, states, manifests, resources and an
# exact reconstruction verify; --no-reconstruct skips the rebuild (and reports it incomplete).
python scripts/summarise_experiments.py --log-dir ./logs/evolution
# Versioned provenance sidecar for a run recorded before lineage existed (raw records untouched).
python scripts/recover_provenance.py --log-dir logs/evolution_review_20260913
# Pair runs' best candidates on shared expanded development cases (selection-biased; see docstring).
python scripts/compare_evolution_runs.py --baseline logs/evolution_review_20260913 --run logs/evolution_debug_full
# Score saved policies, paired, on a fresh MT1 pool seed disjoint from every committed manifest.
# A seed used here has been looked at; record it and pick a new one next time (2100 is spent).
python scripts/evaluate_on_fresh_manifest.py --pool-seed <unused seed> --out-dir <dir> \
  --policy label=<policy_source.py>:<state.pt> --expert --reference label

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

# Stage E (debug_spec.md): four fitting conditions on a frozen confirmation manifest.
# Selection and manifest are frozen and committed before the run; `run` needs a clean worktree.
python scripts/stage_e.py select            # choose and freeze the evolved structures
python scripts/stage_e.py build-manifest    # check the seed ledger, reserve seeds, build the manifest
python scripts/stage_e.py run               # records the Git SHA in every result
python scripts/stage_e.py report --run <logs/stage_e/run_dir>

# Fittable-bank qualification (decided 2026-09-14): 3 references + 6 predeclared variants, 5 seeds each.
# Variants are frozen (structural checks only) and committed before the evaluation manifest exists.
python scripts/qualify_fittable_bank.py validate
python scripts/qualify_fittable_bank.py build-manifest
python scripts/qualify_fittable_bank.py run          # clean worktree; records the Git SHA
python scripts/qualify_fittable_bank.py report --run <logs/fittable_bank/run_dir>

# Stage F (debug_spec.md, D10): correction-LLM repair vs matched-budget arms (lares/repair/correction.py).
# Protocol frozen in config/stage_f/stage_f.yaml (decided 2026-10-13; held-out pool seed 2500, replicate seeds
# 2500-2502). select and build-manifest run before the commit; run --config needs a clean worktree.
python scripts/stage_f.py select
python scripts/stage_f.py build-manifest
python scripts/stage_f.py run --config config/stage_f/stage_f.yaml
python scripts/stage_f.py run --structures <ids> --seeds 0 1 2 --held-out <fresh manifest.yaml>   # ad hoc
python scripts/stage_f.py run ... --arms no_repair local_search global_search    # no LLM calls
python scripts/stage_f.py report --run logs/stage_f/<run_dir>

# Two-lane search (decided 2026-09-14; lares/search/lanes.py). Implemented and tested against mocks only.
# Run the new-model baseline first. Both are live gpt-6.1-sol runs and need approval.
python scripts/run_full_evolution.py                                      # baseline: 9 unseeded zero-shot candidates
python scripts/run_full_evolution.py --config config/two_lane_search.yaml # 6 fittable mutations + 3 explorations

# R-037: plateau-triggered trace repair through the real entry point (assembly-v2; two structure slots).
# Freeze a block's manifests first (no LLM call, no episode), then the paired repair-on / repair-off runs.
python scripts/run_full_evolution.py --config config/assembly_two_lane_repair.yaml --freeze-manifests
python scripts/run_full_evolution.py --config config/assembly_two_lane_repair.yaml --seed 3900 --log-dir <dir>
python scripts/run_full_evolution.py --config config/assembly_two_lane_repair.yaml --seed 3901 --repair disabled --log-dir <dir>

# Symbolic-policy evolution (primary entry point; config-driven)
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
  tests.test_final_report tests.test_vision tests.test_fitting_provenance tests.test_provenance \
  tests.test_two_generation_mock tests.test_stage_e tests.test_fittable_bank tests.test_two_lane_search \
  tests.test_replicate_bundles tests.test_stage_f tests.test_research tests.test_plateau_repair_mock \
  tests.test_pen_spin tests.test_pen_spin_pipeline
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
| 2 | `fit_policy_with_objective()` (`lares/fitting/optimizers.py`) | The orchestrator's fitter: the named objective (`deterministic_mse` by default), scale-only parameters frozen, `clip_params()` after each step, validation split **by episode**. Probes step 0 and each scheduled checkpoint, selects by the tie rule in `lares/fitting/checkpoints.py`, keeps `initial_state`/`final_state`/`selected_state` distinct and recomputes diagnostics after loading the selected one. `behavioral_cloning()` is the legacy `MSE + 0.01 * std.mean()` path, kept for the Isaac stack and old tests |
| 3 | `rl_finetune()` | GRPO: advantages relative to the *group* mean return, + entropy bonus, + L2 KL toward the BC init. **Refused by `EvolutionOrchestrator`** (`rl_iterations` must be 0): RL after selection would break the record's selected-state link |
| 4 | `EvolutionOrchestrator.run()` / `llm_evolution()` | Two-phase generation returns `GeneratedCandidate`s with lineage; each valid one is fitted, probed, screened and recorded; the next generation gets a hashed `FeedbackBundle` and proposes bounded edits of named parents |

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

**Screening and promotion (`lares/eval/promotion.py`; rule decided 2026-09-14).** Every candidate is
screened on the first `screen_episodes` (10) development cases; the screening rule (any success, else
the top `fallback_top_k` by signed goal progress) only decides which candidates *may* challenge. The
best screened challenger alone is scored on the **promotion set**, the next `expanded_episodes` (30)
cases, a `manifest.slice` disjoint from both the screen and the probe tail
(`check_manifest_isolation` refuses overlap). It replaces the incumbent only if the paired success
difference is strictly positive **and** the lower bound of its paired t interval is above zero
(`compare_to_incumbent`, refusing evidence from a screening or probe manifest). Tied success never
promotes; return and progress may rank or admit to the archive but never replace the incumbent. With
`promotion_challengers > 1`, `holm_promotion` applies Holm's step-down correction and promotes at most
one. Before any promotion the incumbent is the 0.00-success baseline (`NULL_INCUMBENT_ID`).
Decisions are written to `gen_<N>/screening.json`. Resource keys and report names still say
`expanded`; they mean the promotion set.

**Checkpoints.** The orchestrator evaluates `zero_shot` (before fitting), `intermediate` (halfway, via
`fit_policy_with_objective(checkpoint_callback=...)`) and `fitted` (the *selected* checkpoint). They are separate reports and are never
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

**Imitation fitting is disabled by default (Stage E, 2026-09-14).** `config/run_full_evolution.yaml` sets
`bc_steps: 0` and no checkpoint probes, so candidates are evaluated at their declared initial values; the
records still carry a (zero-step) fit, selected step 0 and identical initial and selected states. Fitting is
an experimental treatment: no fitted condition beat zero-shot with a 95% interval above zero across 8
structures (`config/stage_e/stage_e.yaml`, `logs/stage_e/20260914_032500`). Fixed-budget fitting is harmful;
rollout-selected fitting helps the hand-written bank structures a great deal and the evolved structures not at
all. If you turn fitting on, always rollout-select. Seed 2200 and fitting seeds 2200-2204 are spent
(`config/experiment_seed_ledger.jsonl`).

**Fittable lane and seed bank (decided 2026-09-14).** Rollout-selected fitting is allowed only in a
fittable lane: checkpoints {0, 500, 1000}, selected on a probe manifest (step 0 always eligible and recorded as
"fitting rejected"), evaluated on a separate manifest. Probe scores select parameters and are never promotion
evidence. Step-0 strength and fittability are classified separately (`performance_incumbent` versus
`fittable_incumbent`). `lares/fitting/fittable_variants/` holds six predeclared single-change variants of the
three references, qualified by `scripts/qualify_fittable_bank.py`; the frozen Phase 3 bank is unchanged. The
fittable seed bank (`config/fittable_bank/seed_bank.json`, run `logs/fittable_bank/20260914_040245`) is
`simple_reach_push_sharp_gate`, `simple_damped_push`, `simple_reach_push`, `simple_reach_push_damped` and
`simple_reach_push_shared_gain`. Seeds 2300-2304 are spent.

**Structure roles (`config/fittable_bank/structure_roles.yaml`).** Parent selection reads this file, not the
seed bank's scores. Only `fittable_seed` structures (the five above) may be a fittable-lane parent. The standoff
family and `simple_damped_push_smooth_gate` are out: `simple_standoff` and the smooth-gate variant are
`non_fittable_control`s (scored beside the bank, never mutated, never admitted), and the other two standoff
variants are `retired`. No file or result was deleted; `seed_bank.json` and the Phase 3 bank are unchanged.

**Two-lane search (`lares/search/lanes.py`, `generate_two_lane_candidates`; decided 2026-09-14).** Implemented
and tested against mocks only (`tests/test_two_lane_search.py`); never run live. One generation of exactly nine
candidates, configured by the `two_lane_search` block of `config/two_lane_search.yaml` and refused on any other
shape:
- *Fittable lane, 6 slots, fixed 4:2 allocation* (declared in config, never ranked): one slot for each
  reach-push seed, two for `simple_damped_push` with the predeclared scopes `gate_control` and `geometry`. One
  design call returns a `fittable_mutation` per slot (`parse_fittable_mutations_payload`): the slot's parent and
  scope echoed, one structural change, declared adds/removes/changes inside the scope (at most 3, at least 50%
  of parent parameters kept), a predicted `fitting.*` diagnostic, protected behaviours, no action scale. The
  implemented child must differ from its parent exactly as declared (`check_bounded_mutation`), or it fails at
  `mutation_contract` and gets its one correction. Only this lane fits: budget 1000, checkpoints {0, 500, 1000}
  on the probe manifest; selecting step 0 is recorded as `fitting_rejected`.
- *Exploration lane, 3 slots*: the ordinary idea request, labelled `unseeded_exploration`, no parent, no
  proposal. Never fitted or probed; its zero-shot screen is its screen (no duplicate "fitted" screen).
- *Incumbents*: `performance_incumbent` is the existing cross-lane challenge. `fittable_incumbent` needs a
  fittable-lane candidate with `selected_step > 0` whose fitted state beats both its own step-0 state and the
  current fittable incumbent on the promotion manifest, each by the strict paired rule
  (`decide_fittable_incumbent`). Written to `incumbents.json`; per-lane resources to `lane_resources.json`.
- The orchestrator (`lane_plan=`) requires one generation, `pop_size` 9, global `bc_steps: 0`, one challenger,
  and a probe manifest disjoint from the screen. Records carry `lineage.lane` and `lineage.lane_metadata`, and
  `ExperimentRecord.validate` enforces the lane rules.

**Plateau-triggered trace repair (`PlateauRepairPlan` in `lares/search/lanes.py`; R-037, 2026-10-22).** A
`plateau_repair` config block replaces the two-lane plan with two structure slots generated one at a time
(`PlateauSlotGenerator`). Slot 0 is a searched single-phase exploration; after it is screened and challenged,
`plateau_trigger` fires only if all 24 search candidates scored zero probe successes and nothing was promoted. With
repair enabled the expert runs alone on a disjoint demonstration manifest (`lares/repair/traces.py`; at least 9/10
successes or the run stops inconclusive without spending slot 1) and slot 1 is one whole-controller edit of the best
failed prefix candidate (lane `trace_repair`, one linked correction, parent = the edited candidate, searched like an
exploration). Otherwise slot 1 is another searched exploration and no demonstration is ever collected. `task_manifests`
builds development (screen/promotion/probe) and demonstration manifests for any task from pool seeds and freezes them;
`demo_buffer` is `None` because nothing is fitted. The run log is `gen_0/plateau_repair.json` (trigger, edited candidate,
demonstration hashes, continuation, final incumbent, episode and call accounting). Mocked gate:
`tests/test_plateau_repair_mock.py`.

**Replicate bundles (`lares/fitting/replicates.py`).** Future qualification varies data, not minibatch order:
each replicate draws its own demonstration sample, trajectory-level split, probe manifest and qualification
manifest, and every structure in a replicate shares that bundle (`run_replicate_qualification` refuses
otherwise). Protocol predeclared, not run: `config/fittable_bank/qualification_v2.yaml` (seeds 2400-2424, unused
and unreserved on 2026-09-14).

**Baseline under the new model (`config/run_full_evolution.yaml`).** One generation of nine unseeded zero-shot
candidates on the same manifests, fitting disabled, no token cap. Ran under gpt-5.6-sol on 2026-09-23
(`logs/baseline_gpt56_sol_high`: 10 calls, 100,786 tokens, best 0.333 on the promotion set); the config's log
dir is now `logs/baseline_gpt61_sol_high` for a gpt-6.1-sol rerun. No token cap on LLM runs for now (decided
2026-09-23); tokens are recorded in the ledger, not capped.

**Stage F repair ablation (`lares/repair/correction.py`, `scripts/stage_f.py`).** Six arms per replicate from
one common preprocessing (rollout-selected fit, then one screen run): `no_repair`, three correction-LLM arms
differing only in the evidence packet (none / numeric / numeric + substitution study), `local_search`,
`global_search`. An LLM repair is one `repair`-mode proposal tied to the parent and the packet's feedback id,
one implementation plus at most one correction, then the parent's preprocessing again. Every arm is capped at
the same repair-evaluation episodes (`StageFBudget.repair_episode_cap`) and replaces the parent only if strictly
better on the screen by the checkpoint tie rule. Chains land in `<run>/<structure>_s<seed>/<arm>/chain.json`;
`verify_chain` rechecks them. Manifests are compared by placement hash (`task_hash`), not case id: case ids are
per-manifest indices and collide across manifests.

**Behavioural cloning is non-monotonic in gradient steps.** Development success peaks around 1000
steps and collapses to zero by 3000, while validation loss falls the whole way. Measured on two
structures over three seeds; at the loss minimum both score zero. `behavioral_cloning` therefore
takes a `rollout_probe` and keeps the best-scoring parameters, and `EvolutionOrchestrator` supplies
one to `fit_policy_with_objective` that scores the probe cases at every step in `checkpoint_steps`
(default step 0 plus `rollout_checkpoint_probes`=4 evenly spaced steps).
Selecting on loss cannot work here. The probe returns a `RolloutScore(success, progress)` and the
declared tie rule is higher success, then higher progress, then the **earlier** checkpoint; step 0 is
probed and can win. The probe runs on its own env (`PipelineEnvs(probe=...)`) and on
`probe_manifest`, the development tail, disjoint from screening. Set `rollout_checkpoint_probes=0` (or
`checkpoint_steps=()`) to fit to the end.

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

Prompts in `lares/utils/policy_prompts/` (seven templates; `ideas_user.txt` and `proposals_user.txt`
demand strict JSON). Generation lives in `lares/core/policy_generation.py`.

**Generation provenance (`lares/search/generation.py`, `resources.py`, `provenance.py`;
`debug_spec.md`, progress in `DEBUG_PROGRESS.md`).** `generate_candidates` returns one
`GeneratedCandidate` per *attempt*; slot and attempt ids (`gen<g>_cand<s>_a<k>`) are allocated before
any call. Generation 0 parses `PolicyIdea`s strictly (no padding: a wrong count raises, a bad entry fails
its slot); generation 1+ parses `MutationProposal` + idea, with repair/simplification = one parent,
crossover ≥ 2, exploration may have none only with `no_parent_rationale`, and every parent drawn from
the `FeedbackBundle`'s `context_parent_ids`. One linked correction per slot, never an outer retry. Every
LLM call (model and `reasoning_effort` from the config) is one `LLMCallLedger` entry in `<log_dir>/llm_calls.jsonl` with hashed prompt and response
files; shared calls list several `attempt_ids` and are counted once. Code must declare exactly the
idea's parameters, return `self.fixed_std(mean)` (scale parameters are rejected), and use the gate
`record_gate` returns (`GATE_NOT_A_LEVER` otherwise). Records (`origin="generated"`) must carry lineage,
call references, and for evaluated candidates the objective config, hashed state and report artifacts,
a `ResourceLedger` (`environment_steps` is forbidden) and a verified reconstruction; a changed record
keeps its old version under `experiments/history/`. The archive keeps `screen_metrics` and
`confirmed_metrics` and ranks on confirmed ones.

**Research-idea ledger (`lares/research/`, `scripts/research.py`, `research/`; added 2026-10-13, triage 2026-10-26).**
One level above `ExperimentRecord`: a `ResearchIdea` is a hypothesis about the *method* with a `Prediction` that must
name a metric from `measurable_metrics()` or `RESEARCH_METRICS` with a direction and threshold, a `Protocol` declared
before any result (script, config, exact command, seeds, manifests, episode and call estimates; anything naming the
final-test split is refused), and a status that moves `proposed → approved → running → analysed → decided`, with
`deferred` (right idea, not now; re-triaged later), `rejected`, and `escalated` (negotiation rounds exhausted at the
`triage` or `verdict` stage; `idea.escalation.stage` says which). `check_idea` refuses spent or doubly declared seeds,
missing scripts and configs, and unestimated cost; `start_run` requires approval and records the Git SHA.
*Triage* (`record_triage_advice` / `record_triage_position`): the advisor's `triage` call returns approve / revise /
defer / reject per idea with reasoning; the implementer answers with its own position; equal positions resolve the
round (approve → `approved` with `approval.by = advisor+implementer` and the call id; revise → stays proposed until
`apply_revision` and the next round), unequal ones open another round whose prompt carries every earlier round, and
the round cap (`DEFAULT_MAX_ROUNDS` = 5) escalates to the user's `rule_triage`. `decide` applies the same cap to
verdicts (`verdict_rounds`) before escalating. The user's input is `research/direction.json` (`set_direction`,
history kept), shown in every briefing and recorded on each idea. `Advisor` makes five calls (`propose`, `triage`,
`analyse`, `review`, `consult`; prompts in `lares/utils/research_prompts/`), each an `LLMCallLedger` entry in
`research/llm_calls.jsonl`; each idea carries the SHA of the briefing it came from (`research/briefings/`), and the
briefing is generated from `research/constraints.json`, the seed ledger, open / deferred / recently rejected / decided
ideas, the script inventory and the handoff's status section. Append-only inspection logs: `research/proposals.jsonl`
and `research/triage.jsonl`. `loop_guard` counts propose calls rejected in full; two in a row stop `propose`.
`research/STATE.md` is regenerated by every mutating command and is never hand-edited. Tests: `tests/test_research.py`,
scripted advisor, no API key.

**Shadow Hand pen spinning (`lares/envs/isaac_pen_spin/`; R-038 to R-042, 2026-10-27).** The env is vendored from
the box's Isaac Lab checkout. `spin_rate` is roll about the pen's own long axis, and success is per step, so the
episode rule is ours: `cases.sustained` (time-out without a fall, and the predicate on all of the final 60 steps;
D-034). `cases.py` and `layout.py` are numpy-only: manifests from pool seeds, the 157-d layout, reducers.
`runner.PenSpinBatchRunner` writes each case's placement into the simulator.
- **One rollout per process.** A second rollout in one Isaac process is not reproducible (D-034). A case's outcome
  also depends on what the rest of its batch runs (D-035, D-037), so every protocol scores one manifest per fresh
  process.
- **Expert (D-036).** `logs/pen_spin/r040_ppo_expert/train/model_8239.pt` (PPO, `ppo_expert.PPOController`)
  qualifies on all three stages. The scripted `scripted_expert.FeedbackGait` does not (D-035). The expert needs
  feedback: a replay of its own actions scores 0/40 (D-037).
- **Pipeline (R-042).** `pipeline_backend.IsaacPenSpinStream` is a pipeline "env". `evaluate_manifest` hands it a
  whole manifest, and it runs `pipeline_child` under Isaac's environment. It rebuilds the generated policy from
  `policy._generated_source`, runs it on the CPU, and executes `clamp(mean, -1, 1)`, not `tanh`.
- **Orchestration.** The orchestrator runs in `.venv-metaworld`, which has openai; Isaac's Python does not.
  `run_full_evolution.py --config config/pen_spin_plateau_repair.yaml --study keyframe_repair` drives a parity gate,
  then per block a prefix run that stops at the plateau trigger (`plateau_fork={"phase": "prefix"}`). Three
  continuation arms then run in copies of it (`no_trace`, `uniform`, `event`; keyframes in
  `lares/repair/pen_spin_traces.py`), followed by fresh evaluation. Mocked gate: `tests/test_pen_spin_pipeline.py`.
- **R-042 result (D-038, refuted).** Stage 3 is a floor for this pipeline: the trigger fired in 4/4 blocks and all twelve
  continuation arms (no_trace, uniform, event) kept the null incumbent, so every contrast is exactly 0. Trace-arm
  candidates spin while held but mostly drop the pen. A trace study here needs a non-zero base rate first.
- **R-045 result (D-039, refuted).** Dense fitting of the R-042 prefix controllers to 4308 expert state-action pairs
  (`dense_fit.py`, `--study dense_feedback_fit`) cut MSE 2-4x but never scored a probe success, and can collapse
  rollouts (block 2). Imitation loss does not track pen-spin success, as in MetaWorld. Both arms stayed null.
- **R-044 result (D-040, supported).** Clamping the expert's pen-state observations to their reset values drops it
  from 39/40 to 0/40 (early drops, median step 49); clamping hand state is nearly as bad (1/40). The expert depends
  on both channels (`feedback_lesions.py`, `run_shadowhand_spin_stages.py --study feedback_lesions`).
- **R-046 result (D-041, inconclusive).** Gait-scaffold continuation (`gait_scaffold.py`, `gait_repair.py`, 177-d
  `Isaac-ShadowHandSpin-Stage3-GaitScaffold-v0`, `--study gait_preserving_repair`). The plateau trigger fired in only
  1/4 blocks: the cold 157-d prefix had probe successes in 3/4 and promoted in block 3 (+0.333 [0.15, 0.51]), so
  R-042's "floor" is seed-dependent. In block 1, whole_edit replaced the gait and dropped the pen (0/30); bounded
  residual matched the unchanged gait (0.467 vs 0.433), which alone promoted at 0.500.
- **R-047 result (D-042, supported).** Feeding either group (pen state or hand state) four steps late (67 ms) drops the
  expert from 40/40 to 0/40 (`feedback_delay.py`, `--study feedback_delay`); pen-delay falls come later (median step
  116) than R-044's clamps (49). One delay only, so the latency threshold is unmeasured; this checkpoint only.
- **R-048 result (D-043, inconclusive, closed before any arm ran).** The PPO expert is **bang-bang**: 99.98% of its
  executed action components are exactly +/-1, so clamped finite differences of it are zero almost everywhere, and
  its executed action switches sign in only 2.8% of state-coordinate-action triples within one demonstration range
  at the gait's states (`local_feedback.py`, `scripts/isaac/audit_expert_switches.py`). Expert-label evidence shows
  only saturated values. `IsaacPenSpinStream.query_actions` queries an actor at given observations with no episode.
- **R-049 result (D-044, refuted).** Requiring separate retention and spin-feedback components in the cold prompt
  (`two_channel.py`, `--study two_channel_cold`, 4 slots per arm, Stages 1 and 2, neutral pose as a *scored* null
  incumbent) gave 0.000 [0, 0] over 8 blocks: neither arm had a screen success or a promotion. Candidates already keep
  the pen (13% of screen episodes dropped vs 97% for the neutral pose) and fail on `slow_spin` / `off_centre`, so on
  Stages 1/2 the bottleneck is sustained axial spin near the centre, not retention. These looser stages scored 0/8
  blocks at 4 candidates while R-046's 24-candidate Stage 3 prefix had probe successes in 3/4: candidate budget, not
  stage difficulty, is the untested driver of the base rate.
- **R-051 result (D-045, refuted).** Expert takeover from the R-046 block-3 learner's own states (`learner_recovery.py`,
  `--study learner_recovery`; learner frozen in `config/pen_spin/r046_block3_incumbent/`) scored +0.031 [-0.18, +0.25]
  (8/32 vs 7/32). It rescued 6 of 21 already-failing states but broke 5 of the learner's 7 successes; it drops the
  pen in 11 switched cases where the learner dropped none. The expert's basin does not cover the learner's grasp, and
  its labels there are 99.85% saturated. **Closed-loop prefix re-execution is a qualified branching mechanism**: three
  fresh processes on one 32-case batch were bit-identical up to each switch (Gates 1 and 2).
- **R-053 result (D-046, inconclusive).** Reversing the sign of the R-051 learner's observed axial spin
  (`axial_sign.py`, `--study learner_axial_sign`) gave -0.031 [-0.18, +0.11] (27/32 ties). The code reads the signed
  rate, but its frozen operating point (`spin_target` 12.16, `speed_gain` 0.81) saturates the spin drive on 94.6% of
  in-play steps, so the learner is effectively an open-loop stroke in axial spin. A static source audit finds code
  paths, not active feedback; check the operating point.
- **R-054 result (D-047, refuted).** Trace repair of the frozen R-051 learner with its own successful traces versus PPO
  expert traces (`self_trace.py`, `run_full_evolution.py --study self_trace_source`, packets in
  `config/pen_spin/r054_evidence/`, 8 blocks, 4 slots per arm): 0.000 [0, 0], no promotion in either arm; both kept the
  parent (fresh 0.225). Not predeclared: self-trace candidates kept some of the parent's behaviour (27/320 screen
  successes), while expert-trace candidates scored 0/320 and dropped the pen in 200/320 screens. Expert evidence breaks the
  learner's grasp; self evidence does not, but it did not improve on the parent either. Session summary:
  `docs/SESSION_2026-10-27_PEN_SPIN_SUMMARY.md`.
- **Code extraction.** Generated code keeps only the class, so module-level constants in a reply are dropped.

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
- return `std = self.fixed_std(mean)`; the action scale is fixed (`fixed_action_scale`, default 0.1)
  and a parameter that only moves the scale is rejected in generation
- optionally expose a phase gate as `g = self.record_gate("name", g)` and *use* `g`. The recorded
  value is detached; the returned value is what an override forces, and a gate whose forced value
  never changes the action fails validation (`gate_not_a_lever`).

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

The research loop is driven by the ledger under `research/`, not by the prose files. Skills:
`/research-cycle` runs the loop; `/research-status` reads the state. Roles (changed 2026-10-26): the **user**
sets a research direction (`python scripts/research.py direct "..."`) and does not review ideas one by one;
the **advisor** (gpt-6.1-sol through `research.py propose|triage|analyse|decide --review`) proposes ideas,
decides approve / revise / defer / reject per idea and judges numbers; **Claude** states its own triage position
and verdict, argues disagreements, writes all code, runs all protocols. An idea moves only when advisor and
Claude state the same position; a disagreement goes another round with Claude's argument in the prompt, and
only after the round cap (5) is the idea `escalated` for the user to `rule` (triage) or `decide --ruling`
(verdict). Claude never uses `rule` or `--ruling`.

1. `python scripts/research.py state`; set the direction if the user gave one. Take the first "Needs attention" item.
2. Nothing open: `research.py propose --n 3`, then `research.py triage`; answer each idea with
   `research.py position <id> --position ... --reasoning "..."`. Agreed revise → `revise --patch`, then triage again.
3. Approved idea: write the code its protocol names, run focused tests, `gpu test`, `research.py start`, run the
   exact protocol command on the GPU box (box stays up for the session; `gpu stop` only at the end).
4. Write `research_result.json` (`metrics`, `controls`, `artifacts`, `notes`), `research.py record`, your own
   `verdict` **before** `analyse`, then `decide --review --handoff`; on disagreement restate and review again.
5. Commit the ledger, code and handoff together, naming the idea id. Then propose again. The loop stops when
   two propose calls in a row are rejected in full (`propose` refuses; ask the user for a new direction), when
   something escalates, or when the user interrupts.
6. Preserve existing user changes and avoid unrelated refactoring. Stop for a destructive action, a missing
   requirement, or any spend no agreed triage covers.

Inspection logs: `research/proposals.jsonl` (every proposal, parsed or not), `research/triage.jsonl` (every
round: advisor decision and reasoning, Claude's position and reasoning, outcome), `research/llm_calls.jsonl`.
`IMPLEMENTATION_PROGRESS.md` and `DEBUG_PROGRESS.md` are frozen history (2026-10-13); do not add to them.
`RESEARCH_HANDOFF.md` receives only the decision blocks `research.py` appends.

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
- **Phase 7 is blocked until Stages A–F of `debug_spec.md` are complete**, not merely the Stage D smoke
  test. Stage F tests the main repair hypothesis, so a final evaluation before it would freeze an
  incomplete method. Do not open the final-test split or run Qwen/E10 before then.
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

## Remote GPU server

Code is written and edited **locally**; tests and experiments **run on the server**. Everything goes
through `scripts/gpu.sh`, symlinked as `gpu` on the PATH (`~/.local/bin/gpu`). The instance is EC2
`i-00844bdd35e7d6f96` (`LLM-symbolic-backup`, g4dn.2xlarge, Tesla T4, us-east-1), reached with AWS
profile `ec2-control` and key `~/.ssh/sybolicRL.pem` as `ubuntu`. The public hostname changes on
every start, so never hard-code it or use the `sybolicRL` entry in `~/.ssh/config`; `gpu` looks it up.

```bash
gpu status                 # stopped|running + current hostname
gpu start                  # start, wait until sshd answers
gpu test                   # sync, then every test file (unittest ones + self-running ones)
gpu test tests.test_search # sync, then specific unittest targets
gpu run python scripts/lock_baseline.py   # sync, then any command in ~/LaRes, venv + MUJOCO_GL=egl set
gpu get logs/evolution ./remote_logs      # copy results back
gpu connect ['cmd']        # plain ssh (no sync)
gpu stop                   # ALWAYS stop when done: the instance bills by the hour while running
```

- `sync` copies the local working tree (tracked + untracked-not-ignored files, uncommitted edits
  included) into `~/LaRes` and deletes files removed locally. No commit or push is needed to test.
- The server's git HEAD is left where it is; its `git status` shows the synced local edits. That is
  expected. Commits are made locally and pushed to GitHub from the Mac.
- **Guard:** sync refuses if the server's HEAD is not in the local repo, or if any server file changed
  since the last sync. Do not edit code on the server; if it happened, `gpu pull` fetches server
  commits into `gpu/<branch>` (then `git merge --ff-only gpu/<branch>`). Uncommitted server edits
  must be committed on the server first.
- Long runs: `gpu run 'nohup python ... > run.log 2>&1 &'`, then `gpu connect 'tail run.log'`.
  Do not start full training or large experiments without the user's approval (see workflow rules).
- `OPENAI_API_KEY` is not set in the non-interactive ssh shell, so LLM tiers of the tests skip.
