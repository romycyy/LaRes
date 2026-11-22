# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Start with `docs/PIPELINE_STATE.md`**: the current pipeline structure, what is verified, and where the research
stands (written 2026-11-09). Superseded plans and progress logs are in `docs/deprecated/`.

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
`scripts/research.py` (the advisor) and Tier 3 of `tests/test_training_pipeline.py`. The defaults live in
`lares/core/policy_generation.py` (`DEFAULT_LLM_MODEL`, `DEFAULT_REASONING_EFFORT`); configs name both
explicitly, and every ledger entry records the effort in `decoding`. The model accepts only temperature 1
(checked against the API on 2026-10-13, as are `reasoning_effort` and `n>1`). From 2026-09-14 to 2026-10-13
the model was gpt-5.6-sol (`logs/baseline_gpt56_sol_high`). No token cap on LLM runs for now.
Runs completed before 2026-09-14 (`logs/evolution_debug_full`, `logs/debug_smoke_two_generation`, Tier 3
notes) used gpt-5 or gpt-4o-mini, so their token caps and results do not transfer. The archived vision analyst
(`deprecated/lares/vision/`) was a local Qwen by design, not an OpenAI call.

`pyproject.toml` exists (setuptools, packages `lares*`) so `pip install -e .` works, but no environment
has it installed yet: every script/test still does `sys.path.insert(0, _PROJECT_ROOT)`. **All commands
must be run from the project root.**

**Shared helpers (added 2026-10-14; one copy each, every older name re-exports them).** `lares/utils/hashing.py`
(`sha256_bytes/text/file`, and the 16-character `short_hash_*` used by `freeze.py` and `source_hash`; the
prefix length is a stored format), `lares/utils/gitstate.py` (`git`, `worktree_state`, `require_clean_worktree`),
`lares/utils/seeds.py` (ledger reader, `spent_seeds`, `manifest_pool_seeds`; scans `config/**` *and*
`logs/**` manifests for every caller, where the research ledger used to scan `config/` only),
`lares/utils/env_ids.py` (`resolve_registry_env_id`, the v2→v3 rule), `lares/eval/stats.py` (`t_summary`,
`t_interval`, `t_bounds`, `win_loss_tie`: the one Student-t rule behind promotion, Stage E/F reports and the
comparison scripts), `lares/fitting/diagnostics.py` (`per_axis_mse`, `bound_activity`, `loss_by_phase`, so
`fitting` no longer imports them from the pipeline module) and `structure_bank.load_structures(ids)`.
`lares/utils/__init__.py` resolves the MetaWorld builders lazily so these import without MetaWorld.
Added 2026-11-03 (repo-wide simplification pass): `lares/utils/jsonl.py` (`read_jsonl`, `append_jsonl`: every ledger
reader/appender delegates here), `lares/utils/clock.py` (`now_iso`, `run_stamp`: the two timestamp formats), `lares/utils/cli.py`
(`separator`, `load_yaml`, `require_api_key`, `openai_client`: what every `scripts/*.py` used to carry its own copy of),
`gitstate.rel` (path relative to the project root), `lares.eval.manifest.unique_manifests`, `lares.eval.stats.block_contrast`
(the per-block paired contrast every pen-spin study summarises), and in `lares/envs/isaac_pen_spin/`: `cases.OBS_DIM`
(derived from the layout; no more literal 157), `cases.freeze_manifest` / `freeze_manifest_set` / `role_manifests` (the
build-or-load-and-refuse-if-different idiom, placement disjointness through `assert_disjoint`, role slices keeping the
`<id>:<role>` manifest id), `protocols.child_cell`, `protocols.build_single_manifest`, `protocols.aggregate_vs_expert`.
Scripts import helpers from `lares`, never from another script.
`lares.fitting.objectives.EXECUTIONS` is the execution-contract registry (`execute()` is a lookup), and
`lares.core.policy_generation.DEFAULT_POLICY_IMPL_MODE` / `normalise_impl_mode` is the one `policy_impl_mode` rule
(`load_config` calls it; the config loader no longer has its own spelling).
Added 2026-11-09 (second simplification pass, `/simplify`): `lares/utils/cli.self_argv` (the interpreter + `-u` + script +
`--config` prefix every one-rollout-per-process child is spawned with), `policy_generation._correct_until_valid` (the one
linked-correction loop behind the per-idea implementer, the single-phase exploration and the trace-repair slot),
`policy_generation.validation_batch_path` (the slot validator writes the harness's `data.pkl` itself when none is given;
nothing else writes one), `training_pipeline.llm_evolution(plan=...)` (the one generation front-end; a `TwoLanePlan`
selects the two-lane generator), `EvolutionOrchestrator._objective_config` / `_unfitted_fit` (the recorded fitting
config and the fit-shaped result of an unfitted or searched candidate), `run_full_evolution.orchestrator_from_config`
(the one orchestrator construction; callers pass overrides), `training_pipeline.DEFAULT_BC_BATCH_SIZE` / `DEFAULT_BC_LR`
/ `FIT_VAL_FRACTION` / `FIT_SPLIT_SEED` (the fitter's constants, which `load_config` and `freeze_method.py` read),
`lares.utils.seeds` memoises the manifest scan per process, `ObsSchema.field` is a
dict lookup and `RolloutGeometry` slices numpy directly, and `record_episode_gif` runs the shared
`lares.eval.runner.run_episode` with a frame-capturing observer. Pen-spin studies share their prologue:
`protocols.roles` / `manifest_path` / `disjoint_from` / `cached_json` / `freeze_json` (was `scalar_tuning._freeze_json`)
/ `learner_drive_params` / `load_cell` (was `learner_recovery._load`) / `isaac_shell` (the one
`bash -c "source <isaac_env> && ..."` launcher) / `study_cell` (every module's `_child` wrapper; the study name comes
from `protocol_cfg["study"]`, which `run_shadowhand_spin_stages.run_study` sets) / `diverged_envs`;
`cases.require_frozen` / `load_frozen` / `manifest_record` / `freeze_roles`; and `block_study.py`
(`make_slot_generator`, the one `PlateauSlotGenerator` construction for the LLM block studies, plus `parent_like`,
`arm_needs_generation`, `started`, `screen_best`, `attempts_record`). `run_shadowhand_spin_stages.PEN_SPIN_STUDIES` is a
module-level name-to-module dict resolved lazily by `study_module()`, so `--help` imports no study, and `isaac_app(cfg)`
is its one launch/close context manager. Removed as dead: the legacy single-phase entry points
(`get_symbolic_policies`, `_generate_single_phase`, `valid_policies`, `_extract_n_policy_codes`, `obs_description_dict`),
`fit_with_objective`, `MutationProposal.bypasses_llm` / `check_outcome`, `SymbolicPolicy.clear_gate_overrides`,
`sha256_json`, `freeze.hash_text`, `ALL_CATEGORIES`, `TrainingLogger.advance_step` and its W&B/TensorBoard mirrors, the
orchestrator's `rl_iterations` / `rl_episodes_per_iter` / `train_cases` / `gif_case` / `sampling` / `val_fraction` /
`split_seed` / `demonstration_actor` arguments, the `lares.fitting` / `lares.research` package re-exports (import
the submodules), `cases.REDUCERS`, `gait_repair.role_manifest` and `clamp_qualification._study_name` (the
``"r080" in out_path`` sniff). Archived code under `deprecated/` that used these names is listed in `deprecated/README.md`.
Added 2026-11-14 (third pass, findings from six Opus reviewers after the OpenAI credit ran out): `_call_llm` retries
only transient errors (rate limit, connection, 5xx; `_LLM_MAX_ATTEMPTS` = 8 with exponential backoff) and raises
`RuntimeError` on anything else, so a spent credit balance or a bad key stops the run at once instead of looping
1000 times and exiting 0; the validation harness subprocess has a wall-clock cap (`VALIDATION_TIMEOUT_S` = 600, a
timeout is a harness failure that gets the slot's one correction); the validator reports a NaN output as `NON_FINITE`
only (`_has_nan`), not also as batch coupling or irreproducible telemetry; `manifest_pool_seeds([])` scans nothing
(it used to fall back to the whole repo, so research tests read real manifests); `training_pipeline._rank_cleared`,
`EvolutionOrchestrator._score_on_promotion_set` and `_finish_plateau_log` are the one copy of the challenger ranking,
the promotion-set scoring and the plateau log's closing; `obs_schema._register_scaffold_schemas` builds both 177-d
layouts; `run_full_evolution._save_best` writes the best policy for both entry paths;
`dense_fit.build_policy(obs_dim=, action_dim=)` is the one pen-spin policy rebuild (the Isaac child, a saved incumbent,
a reloaded generation, the residual); `block_study.make_slot_generator` no longer writes its own `data.pkl`; pen-spin
manifests are checked for disjointness before they are written. Removed as dead: `seeds.seed_conflicts` /
`reserve_seeds` / `clear_manifest_cache`, `plateau_trigger(already_fired=)`, `PlateauRepairPlan.exploration_slots` /
`reference_ids` / `parent_sources`, the orchestrator's `plateau_log` and `run_manifest_ref`,
`pipeline_backend.make_temp_stream`, `cases.split_placement`, `protocols._child`.
Archived 2026-11-15: the BC + GRPO smoke stages of `run_shadowhand_spin_stages.py` and `IsaacLabSingleEnvAdapter` (see
"Isaac Lab ShadowHandSpin smoke stages" below). `behavioral_cloning`, `rl_finetune` and `_collect_trajectories` stay in
`training_pipeline.py` because live tests exercise them; no entry point calls them.
GPT review 2026-11-19 (gpt-6.1-sol, six packets): `protocols.load_cell` returns a plain dict of arrays, not a lazy
`NpzFile`; `generation.write_artifact` refuses an outside-root path before writing; `run_cell` / `pipeline_child` attach
the controller inside the runner's `try`; `gpu sync` resolves the host once, survives a deleted last path and writes its
guard manifest atomically. Two sync guard gaps are open (server-side untracked additions are not detected; a synced
untracked file deleted locally stays on the server); see `docs/PIPELINE_STATE.md`, "GPT review on 2026-11-19".

## Common commands

```bash
# Phase 0 evaluation contract: build the manifests, reproduce the baselines, check AC-0.
# Exits non-zero if any AC-0 check fails. Run this before trusting any other number.
python scripts/lock_baseline.py                      # reuse the committed manifests
python scripts/lock_baseline.py --build-manifests    # regenerate config/manifests/*.yaml
python scripts/lock_baseline.py --collect-demos 150  # expert buffer with episode ids

# Regenerate the search summary from saved experiment records alone. Traceability is "complete"
# only if every record's lineage, feedback, prompts, usage, states, manifests, resources and an
# exact reconstruction verify; --no-reconstruct skips the rebuild (and reports it incomplete).
python scripts/summarise_experiments.py --log-dir ./logs/evolution
# Score saved policies, paired, on a fresh MT1 pool seed disjoint from every committed manifest.
# A seed used here has been looked at; record it and pick a new one next time (2100 is spent).
python scripts/evaluate_on_fresh_manifest.py --pool-seed <unused seed> --out-dir <dir> \
  --policy label=<policy_source.py>:<state.pt> --expert --reference label

# Phase 7: freeze the method before any final-test episode runs, and check for drift.
python scripts/freeze_method.py --freeze-id v1
python scripts/freeze_method.py --verify             # non-zero exit if anything moved
python scripts/final_report.py                       # aggregate report, from saved files only

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

# Isaac-only pen-spin studies (needs IsaacLab at config's isaaclab_root, CUDA); a bare call with no --study is refused
python scripts/run_shadowhand_spin_stages.py --study <name>      # names: PEN_SPIN_STUDIES in that script

# Plots
python scripts/plot_training_dynamics.py --log-path ./logs/evolution/<run>/training_dynamics.jsonl

# Pen-spin incumbent: verify its hashes; render GIFs of it (numpy 1.x overlay, see "Things that will bite you")
python scripts/isaac/adopt_fixed_stroke.py
python scripts/isaac/record_pen_spin_gif.py --controller incumbent --manifest <frozen.yaml> --env-index 0 1 2 --out-dir <dir>
```

Finished studies' scripts (Phase 3/5/6, Stage E/F, fittable-bank qualification, R-005 to R-036 comparisons, FR-11
failure analysis, one-off pen-spin checks) are in `deprecated/scripts/`; see `deprecated/README.md`.

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
  tests.test_search tests.test_objectives tests.test_freeze \
  tests.test_final_report tests.test_fitting_provenance tests.test_provenance \
  tests.test_two_generation_mock tests.test_two_lane_search \
  tests.test_research tests.test_plateau_repair_mock \
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
| 2 | `fit_policy_with_objective()` (`lares/fitting/optimizers.py`) | The orchestrator's fitter: the named objective (`deterministic_mse` by default), scale-only parameters frozen, `clip_params()` after each step, validation split **by episode**. Probes step 0 and each scheduled checkpoint, selects by the tie rule in `lares/fitting/checkpoints.py`, keeps `initial_state`/`final_state`/`selected_state` distinct and recomputes diagnostics after loading the selected one. `behavioral_cloning()` is the legacy `MSE + 0.01 * std.mean()` path; since 2026-11-15 only tests call it |
| 3 | `rl_finetune()` | GRPO: advantages relative to the *group* mean return, + entropy bonus, + L2 KL toward the BC init. **Not an orchestrator stage** (the orchestrator has no `rl_iterations`; `load_config` refuses a nonzero one): RL after selection would break the record's selected-state link. Since the Isaac smoke stages were archived (2026-11-15) only `tests/test_training_pipeline.py` calls it |
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

The expert audit (`deprecated/scripts/audit_expert_actions.py`) settles which likelihood is right. MetaWorld's
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
three references, qualified by `deprecated/scripts/qualify_fittable_bank.py`; the frozen Phase 3 bank is unchanged. The
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

**Baseline under the new model (`config/run_full_evolution.yaml`).** One generation of nine unseeded zero-shot
candidates on the same manifests, fitting disabled, no token cap. Ran under gpt-5.6-sol on 2026-09-23
(`logs/baseline_gpt56_sol_high`: 10 calls, 100,786 tokens, best 0.333 on the promotion set); the config's log
dir is now `logs/baseline_gpt61_sol_high` for a gpt-6.1-sol rerun. No token cap on LLM runs for now (decided
2026-09-23); tokens are recorded in the ledger, not capped.

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

**Fitting comparison (`lares/fitting/`).** `structure_bank.py` owns a frozen bank under
`lares/fitting/structures/`: fifteen candidates ported from the 2026-08-28 search plus three
hand-written simple-family references. The ported ones had their raw observation slices rewritten to
named accessors by `port_raw_indices`, with `check_port_equivalence` proving the outputs are
identical. It was built by `deprecated/scripts/build_structure_bank.py` and is frozen. `optimizers.py` implements four methods
behind one `fit(method, structure, data)` call: `adam_baseline` reproduces the live path exactly,
`adam_scaled` gives each parameter a step proportional to its declared range plus a cosine schedule,
`multistart_adam` splits the same budget across perturbed restarts and picks on validation loss, and
`differential_evolution` is derivative-free via scipy. All four optimise the identical objective and
receive the same objective-evaluation count; wall time and transitions are reported too, because
equal evaluation counts are not equal compute. `sensitivity.py` measures how much each parameter
moves the executed action, in units of that parameter's own range. The comparison ran from
`deprecated/scripts/benchmark_fitting.py`; the selection rule declared in `benchmark.py` stays live because
`freeze_method.py` digests it.

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

**The aggregate report (`lares/eval/final_report.py`).** AC-7 asks that one documented command
regenerate the report from saved artifacts. `scripts/final_report.py` reads four streams and runs
nothing: experiment records, repair studies, ablation runs and the freeze directory. Ablation runs at
different budgets are never pooled, because arms are comparable only at a matched cost. Quantities
AC-7 requires that no artifact carries are listed in `unavailable` with the field each would come
from, so the gap names its own fix; today that is fourteen of fifteen, most of them LLM and timing
instrumentation the pipeline does not yet record.

**Archived side studies (`deprecated/`, 2026-11-09).** The intervention-guided repair loop (`hypotheses`,
`interventions`, `repair`, `study`; the AC-6 ablation found optimizer-only 0.650 against the full loop's 0.517), the
Stage F correction-LLM (`correction.py`), the Qwen failure analyst (`lares/vision/`; D-003 found it added nothing), DAgger and
replicate bundles moved to `deprecated/`, with the scripts and tests that drove them. `deprecated/README.md` lists every
moved file and keeps what this file used to say about them. `lares/repair/` now holds only the two trace modules.

Prompts in `lares/utils/policy_prompts/` (seven templates; `ideas_user.txt` and `proposals_user.txt`
demand strict JSON). Generation lives in `lares/core/policy_generation.py`.

**Generation provenance (`lares/search/generation.py`, `resources.py`, `provenance.py`;
`docs/deprecated/debug_spec.md`, progress in `docs/deprecated/DEBUG_PROGRESS.md`).** `generate_candidates` returns one
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
  at the gait's states (`local_feedback.py`, `deprecated/scripts/isaac/audit_expert_switches.py`). Expert-label evidence shows
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
- **Concurrent generation (`llm_pool.py`, 2026-11-03).** In R-054, LLM calls took 4.73 h of 5.32 h (avg 258 s per
  call, all in sequence), and Isaac took 0.52 h (about 18 s per process: about 8 s of startup, about 10 s of
  rollouts). `two_channel` and `self_trace` now run each (block, arm) generation on a worker thread
  (`llm_workers` in the study block, default 16; `1` is the old serial order). Isaac evaluation stays on the main
  thread, one process at a time. Each cell's slots are saved to `generated.json` before any of its episodes run, so a
  rerun rebuilds them without an LLM call. `active_schema` is now per thread (`obs_schema._LOCAL`). New multi-block
  studies should use the same pattern.
- **2026-11-04 session (R-058 to R-089, D-048 to D-063; `docs/SESSION_2026-11-04_PEN_SPIN_SUMMARY.md`).** Scalar tuning of
  the frozen R-046 learner is closed (D-048 CMA-ES vs random +0.003 [-0.08, +0.08]; D-049 re-ranking -0.002). Its
  reactive capture curl was the defect: the exact zero-curl clamp (`channel_lesions.curl_clamp`) qualified on a 90-pair
  instrument (D-057, +0.167 [+0.04, +0.29], drops 4 vs 10) after four 30-pair panels each missed the strict rule by
  0.0004-0.02. The incumbent is now the clamp with its stroke scaled by a fixed 1.408 (D-058/D-059,
  `config/pen_spin/d058_fixed_stroke_incumbent/`, verified by `scripts/isaac/adopt_fixed_stroke.py`): about 0.35-0.44
  fresh success, few drops, slow spin as the failure mode. Feedback on the stroke's amplitude (D-058) and phase lead
  (D-061), anonymous-coordinate LLM residuals (D-060, the R-043 branch) and a within-cycle warp (D-062/D-063) did not
  qualify. The learner's roll direction differs across cases, so signed-rate targets are ill-posed. Isaac-only studies
  are `--study` entries of `scripts/run_shadowhand_spin_stages.py` (one module each: `scalar_tuning`,
  `archive_rescreen`, `stroke_lesion`, `channel_lesions`, `curl_confirmation`, `constant_preload`, `curl_bandwidth`,
  `clamp_qualification` (30- and 90-pair instruments), `stroke_feedback`, `lead_feedback`, `phase_warp`,
  `warp_qualification`); the LLM study `coordinate_repair` runs through `run_full_evolution.py` on the 177-d
  `ParentScaffold` schema with the `parent_residual` actor kind. pycma 4.5.0 is installed in the box venvs and
  `lares-local`, not checked in. Seed bookkeeping: a frozen manifest *reserves* its seed for the idea that names it
  (`protocol.check_idea`, briefing and state distinguish reserved from consumed); the loop guard counts a zero-idea
  advisor reply as a propose call rejected in full.
- **R-093 result (D-064; contact-frame decoder gate, 2026-11-20).** `contact_control.py` (no Isaac: capsule frames
  `[normal, circumferential, axial]` at the fingertip link origins, a damped least-squares velocity decoder, the gate's
  statistics) and `contact_gate.py` (`ContactFrameController`, `--study contact_decoder_calibration` / `contact_decoder_gate`
  in `run_shadowhand_spin_stages.py`; constants in a hashed `frozen` block of the config). The registered retention
  prediction held (zero-request hold 0.775 timeout rate vs 0.000 for the neutral pose) but **the gate failed**: sign
  agreement 0.6825 < 0.80, because at steps 10-40 the pen is still settling and the fingers are closing (a common
  relative motion of about 0.17 m/s against a 0.03 m/s pulse). Post hoc, the paired positive-minus-negative contrast
  agrees in sign on 0.938 of pairs (gain 0.54). The decoder is not qualified; R-094 (LLM contact rules) stays deferred.
  **Things this found that will bite you:** PhysX Jacobians (`root_physx_view.get_jacobians`) and the observation's
  fingertip *linear velocity* are for each distal link's centre of mass, but the observed fingertip *position* is the
  link origin, 15-16 mm away (`ControllerContext.fingertip_jacobians` and `contact_control.tip_origin_linvel` correct
  both; uncorrected, a displacement audit is 20% off). The fixed-base hand's Jacobian has 25 link rows (body `b` is row
  `b - 1`) and 24 joint columns; the four passive `J0` joints follow their `J1` at roughly 1:1; `act_moving_average` is
  1.0; joint PD stiffness is 1.0 and damping 0.1. A trusted controller's `extra_trace()` is saved beside its cell as
  `.extra.npz` (`protocols.run_cell`), and `protocols.isaac_shell` now times a child out after 30 minutes.
- **R-092 result (D-065, refuted; modular vs monolithic cold generation, 2026-11-20).** `lares/core/module_contract.py`
  (`MODULAR_REQUIREMENT`, `check_module_contract`: a literal class-level `MODULE_SPEC`, every `obs_field` read inside a
  declared module whose `inputs` match, `forward` only composes module outputs) is an opt-in source check
  (`PlateauSlotGenerator(extra_source_check=)`, `block_study.make_slot_generator(extra_source_check=)`, validation stage
  `module_contract`, one linked correction like any defect). `modular_cold.py` (`run_full_evolution.py --study
  modular_cold`) is the block-study template **with** the 24-candidate probe search (`probe_search`, one process per
  candidate). 8 blocks: +0.0083 [-0.011, +0.028] fresh final-incumbent success against a predicted +0.05; searched
  winner +0.025 [-0.011, +0.061]; 0 of 64 candidates had a screen success; 32/32 slots valid in both arms. Cold
  Stage-3 search is at its floor for both representations (one promotion in 16 arms). The contract is syntactic: a
  controller with one real module and a trivial second one passes.
- **R-095 / R-096 / R-094 results (D-066, D-067, D-068; 2026-11-20).** `contact_settled_gate.py` pulses the R-093
  controller at steps 150-180, after the pen has come to rest in the closed reference posture, with pen-following on
  the commanded fingertip, and scores the **paired contrast** of the `+` and `-` pulse per case (`paired_contrast`,
  `paired_verdict`). R-095 (40 cases) passed every tracking check but was inconclusive on coverage (16 and 19 of 20
  pairs in two contrasts); R-096, one pre-declared 80-case replication of the same frozen block (hash `7219ec6d...`),
  **passed**: paired sign agreement 1.000 over 901 pairs, median gain 0.615. That decoder then served R-094
  (`contact_rules.py`, `run_full_evolution.py --study contact_rules`, 182-d schema
  `Isaac-ShadowHandSpin-Stage3-ContactFrame-v0` = the 157-d layout + `contact_distance` / `contact_axial_coord` /
  `contact_rel_vel`; policy attribute `_contact_actor` -> spec key `contact_actor`, kinds `contact_features` (20 joint
  targets) and `contact_frame` (15 requests through `ContactFrameController(request_fn=)`); `query_actions` refuses a
  `contact_frame` actor; the study refuses to run without a gate summary whose verdict is `pass` and whose hashes
  match). **R-094 refuted:** contact rules minus direct joint targets 0.000 [0, 0] fresh final-incumbent success over 8
  blocks (no promotion in either arm; searched winner -0.0125 [-0.054, +0.029]). In rollouts the decoder is far outside
  the regime the gate qualified: residual 0.108 m/s against a 0.05 m/s request cap, a bound active on 97% of steps;
  the hand-authored contact controller scored 0.02. A gate on small pulses with the pen at rest does not qualify a
  decoder for rolling.
- **R-097 result (D-069, refuted; no-LLM direction, 2026-11-21).** `rolling_decoder_gate.py` (three `--study` names of
  `run_shadowhand_spin_stages.py`, run in order: `rolling_baseline`, `rolling_decoder_calibration`,
  `rolling_decoder_gate`). `RollingPulseController` runs the D-058 incumbent unchanged and, inside a pulse window, adds
  a decoded joint increment that **integrates** (`delta += dt * qdot`, capped per joint): the incumbent's action is
  `q + 0.8 (target - q)`, so a non-accumulating offset is a position step, not a velocity. It reports requested,
  feasible (after bounds, through the Jacobian) and achieved motion separately as paired +/- contrasts, with a
  case-clustered bootstrap interval (`clustered_interval`) because the 15 contrasts share cases. S1: the incumbent
  reproduces on a fresh manifest (28/80, Wilson [0.255, 0.459]; 42 slow-spin failures rolling steadily at about a third
  of the successes' rate). S2: **the gate fails** (sign agreement 0.724 [0.710, 0.739] vs 0.80, feasible/requested
  0.445, bounds active on 0.44 of steps). Forefinger and ring finger track almost perfectly; the middle finger, thumb
  and little finger do not. **Things this found that will bite you:** the incumbent's own action is saturated at +/-1
  on `MFJ2` (79% of steps) and `LFJ1` (64%), so an additive residual is clipped there; when comparing two cells'
  prefixes use in-play steps only (`in_play_prefix`), since a case's recorded steps after it ends are the next
  episode's auto-reset and differ between processes.
- **R-099 result (D-070, inconclusive; 2026-11-21).** `incumbent_residual.py` (`--study incumbent_residual`, the
  24-draw probe search, which stops after hashing its winner; then `--study incumbent_residual_fresh`, which refuses to
  run unless the config's `selected` block matches `selected_sha256` and the search's record). `JointMotionResidual`
  adds `clip(g sign(v_f) max(|v_f| - v0, 0), -b, b)` (filtered joint velocities only, no geometry, no roll sign) to the
  incumbent's action. Only this anonymous arm ran, because R-097 failed. Fresh: +0.056 [-0.041, +0.152] (41 vs 36 of
  90), drops 1 vs 1: no improvement established, not adopted; unsigned roll rose 27% with a 0.0066 rad mean residual.
  A 90-pair instrument resolves about +/-0.10. **Bites:** a driver's own record files must not share a name with a
  cell (`<controller>.json`), or `study_cell` skips that cell as already run.
- **R-101 result (D-071, refuted; follow-up direction, 2026-11-22).** `transfer_audit.py` (`--study transfer_audit`;
  `CoherentPulseController` = R-097's pulse controller commanding several fingertips at once, with every bound recorded
  by type through the base class's `_observe` hook; `RollingPulseController(directions=)` now also takes a list of
  triples per case). Coherent forefinger + ring circumferential pulses on the incumbent, 40 fresh cases, three cells:
  authority is adequate (0.69 and 0.82 of the full fingertip displacement) but roll progress pro minus anti is
  -0.054 rad [-0.147, +0.038] against a predicted +0.10. **Bite (post hoc): under the incumbent only the middle
  fingertip and thumb are at the pen surface; the forefinger, ring and little fingertips sit about 58 mm away**, so
  D-069's "tracking" fingers were moving in free space. Check `capsule_frames(...)["distance"]` before reading a
  fingertip's tracking as contact control. Launch a `--study` driver on the box under Isaac's environment
  (`source ~/IsaacLab/source_env.sh`), not through `gpu run`'s venv: its children use the driver's interpreter.
- **Adding a study (2026-11-03).** `scripts/run_full_evolution.py` dispatches `--study` through the `PEN_SPIN_STUDIES`
  registry (`PenSpinStudy`: name, config key, module, `run`, `freeze_extra`); the argparse choices, the
  `--freeze-manifests` loop and `main()` all read it, and `_study_block` holds the shared preamble (config block, override
  check, model/effort, client, stream factory). `scripts/run_shadowhand_spin_stages.py` does the same with a module dict:
  every feedback-study module exposes `run(protocol_cfg, config_path, out_dir)` and `make_controller(name, protocol_cfg,
  out_dir)`. A new study is one module plus one registry entry; the `--block/--phase/--arm/--gait-block` child argv
  contracts are unchanged.

**Isaac Lab ShadowHandSpin smoke stages (archived 2026-11-15).** `scripts/run_shadowhand_spin_stages.py` used to run
a three-stage smoke pipeline when called with no `--study`: a scripted dataset, legacy BC and GRPO on a toy policy,
through `IsaacLabSingleEnvAdapter`. Nothing used it after the pen-spin stack (`lares/envs/isaac_pen_spin/`) replaced
it. The stages are now `deprecated/scripts/run_shadowhand_spin_stages.py`, the adapter class is
`deprecated/lares/envs/isaac_lab_adapter.py`, and the smoke config keys are `deprecated/config/shadowhand_spin_smoke.yaml`.
The live script only dispatches `--study` / `--research-protocol` (a bare call exits with an argparse error), the live
`lares/envs/isaac_lab_adapter.py` holds only `launch_isaac_app` / `close_isaac_app` / `_ensure_isaaclab_on_path`, and
`config/shadowhand_spin_stages.yaml` keeps `isaaclab_root`, `device` and the unchanged `research_protocols` block.

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

Generation flow in `generate_candidates()` (two-phase, the only path the orchestrator runs; the single-phase
exploration and trace-repair slots share its validator and correction loop): format prompts → LLM call →
regex-extract code block → find `class GeneratedPolicy` → write temp file
(`imports + generated_code + tests/test_generate_policy.py`) → run as subprocess and require `"Success!"` → only
then `exec()` in-process and `validate()`. Failures are fed back to the LLM as one linked correction per slot
(`_correct_until_valid`). `policy_gen_two_phase: true` is required: it first asks for JSON design ideas
(`ideas_*.txt`), then implementations (`policy_impl_mode: batched | per_idea`).

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
`docs/deprecated/IMPLEMENTATION_PROGRESS.md` and `docs/deprecated/DEBUG_PROGRESS.md` are frozen history (2026-10-13); do not add to them.
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
  Mocks accept `case=None`; use `lares.eval.manifest.synthetic_cases()`
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
- **Rollout logic is still duplicated, three ways** — `generate_dataset` and `_collect_trajectories` in
  `training_pipeline.py`, plus `lares/eval/runner.run_episode`, which `evaluate_policy`,
  `run_full_evolution.evaluate_expert_policy`, the expert-trace collector and (since 2026-11-09)
  `record_episode_gif` all go through. Change one, check all three. (The fourth copy, the Isaac smoke stages'
  `evaluate_stage_policy`, was archived on 2026-11-15.)
- **`best_policy_code.py`, `best_policy.pt`, `gen_<N>/data.pkl`** are *runtime outputs* written into
  `log_dir`, not repo files. `logs/` and `*.pkl` are gitignored. `config/manifests/*.yaml` is the
  exception: manifests are committed because they define the evaluation contract.
- **Phase 7 is blocked until Stages A–F of `docs/deprecated/debug_spec.md` are complete**, not merely the Stage D smoke
  test. Stage F tests the main repair hypothesis, so a final evaluation before it would freeze an
  incomplete method. Do not open the final-test split or run Qwen/E10 before then.
- **The saved incumbent scores 0.00, not 0.30.** `logs/evolution/best_policy_code.py` is generation
  2 candidate 0, which its run reported at 402.76 reward and 0.30 success on 10 unmatched episodes.
  Re-scored on the development manifest it gets ~105 reward and 0.000 success. Treat 0.00 as the
  incumbent; see `RESEARCH_HANDOFF.md`.
- **Rendering pen spin needs numpy 1.x.** The box's Isaac venv has numpy 2.4.3, and Isaac Sim 5.1's camera annotators
  fail under it ("Unable to write from unknown dtype"). `scripts/isaac/record_pen_spin_gif.py` runs with
  `PYTHONPATH=/home/ubuntu/np126_overlay` (numpy 1.26.4 installed with `pip install --target`); the venv is unchanged.
  Protocols never render.
- **`config/shadowhand_spin_stages.yaml` is a protocol file now.** Its top level is `isaaclab_root`, `device` and
  `research_protocols`; the smoke keys (`pipeline_task`, `bc_steps`, `rl_iterations`, the GIF keys) went to
  `deprecated/config/shadowhand_spin_smoke.yaml` on 2026-11-15.

## Logging

`lares/core/training_logger.py` defines every metric key as a constant (`BC_*`, `RL_*`, `EVO_*`) —
use those constants, never string literals, so `plot_training_dynamics.py` keeps working.
`TrainingLogger` appends JSONL to `<log_dir>/<run_id>.jsonl` (`run_id` defaults to a `%Y%m%d_%H%M%S`
timestamp). `run_full_evolution.py` writes flat into `log_dir` (default `./logs/evolution`):
`demo_<env>.pkl`, `best_policy_code.py`, `best_policy.pt`, per-generation LLM responses and code, and
optional demo GIFs under `gen_<N>/`. `run_shadowhand_spin_stages.py` studies write under their protocol's `log_dir`
(`logs/pen_spin/<study>/`).

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
