# Deprecated code (archived 2026-11-09)

Code that the pipeline described in `docs/PIPELINE_STATE.md` no longer runs. Each file was moved with `git mv`, so
`git log --follow` still shows its history. Paths mirror the live tree: `deprecated/scripts/stage_f.py` used to be
`scripts/stage_f.py`. Documents that were retired earlier are in `docs/deprecated/`.

**Nothing here is imported, run or tested.** The files still use their original import paths (`lares.vision`,
`lares.repair.study`, ...). To run one again, `git mv` it and its dependencies back, then restore whatever they need:

- the eager imports removed from `lares/repair/__init__.py` (the old file is `deprecated/lares/repair/__init__.py`)
  and from `lares/fitting/__init__.py`, which since the same day's `/simplify` pass is a docstring only (import the
  submodules; `AggregationStats`, `aggregate_learner_states`, `validate_expert_label` came from `dagger`);
- the `lares/repair/repair.py` and `lares/repair/hypotheses.py` entries in `scripts/freeze_method.py`'s
  `FROZEN_MODULES`, which now lists `lares/repair/traces.py` in their place;
- live symbols the `/simplify` pass of 2026-11-09 deleted as dead, which only archived code still names (take them
  from commit `613fa84`, the last before the pass): `policy_generation.get_symbolic_policies` / `valid_policies`
  (`compare_generation_modes.py`), `optimizers.fit_with_objective` (`compare_objectives.py`, `diagnose_repair.py`,
  `analyse_failures.py`, `ablate_repair.py`), `SymbolicPolicy.clear_gate_overrides` (`repair/interventions.py`,
  `repair/study.py`, `tests/test_repair.py`), `MutationProposal.check_outcome` (`repair/correction.py`) and
  `hashing.sha256_json` (`fitting/replicates.py`);
- live symbols the 2026-11-14 pass deleted for the same reason: `seeds.seed_conflicts` / `reserve_seeds`
  (`stage_e.py`, `stage_f.py`, `qualify_fittable_bank.py`, `compare_child_vs_parent.py`, `tests/test_stage_e.py`).

Configs, frozen manifests and seed-ledger entries were **not** moved. `lares/utils/seeds.py` scans `config/**` to
find spent seeds, and the decided ideas in `research/ideas/` still name these configs.

## What stays live, and why

The rule was: keep a file if a live entry point reaches it (`run_full_evolution.py`, `run_shadowhand_spin_stages.py`,
`research.py`, the evaluation contract and Phase 7 scripts, the pen-spin incumbent tools) or if the planned final
test needs it. Whole files only on 2026-11-09. GRPO `rl_finetune` and legacy `behavioral_cloning` stay in
`training_pipeline.py` because live tests call them. The BC+GRPO smoke path of `run_shadowhand_spin_stages.py` was cut
out on 2026-11-15 (see "Cut out of live files" below). All pen-spin `--study` modules stay,
because the incumbent's code depends on several of them and they are the re-runnable protocols behind D-034 to D-063.

| Kept | Why |
|---|---|
| `lares/fitting/benchmark.py` | `freeze_method.py` digests its selection rule |
| `lares/fitting/structures/`, `fittable_variants/`, `structure_bank.py` | The two-lane plan and `evaluate_on_fresh_manifest.py` load them; `freeze_method.py` hashes the bank |
| `lares/fitting/phases.py`, `sensitivity.py`, `optimizers.py`, `objectives.py` | The orchestrator's fitter (`fit_policy_with_objective`) |
| `lares/repair/traces.py`, `pen_spin_traces.py` | Plateau-triggered trace repair (D-032/D-033) and the pen-spin trace studies |
| `scripts/plot_training_dynamics.py` | Plots any `TrainingLogger` JSONL; documented in `CLAUDE.md` |
| `scripts/isaac/{adopt_fixed_stroke,record_pen_spin_gif,train_pen_spin_expert,probe_pen_spin_cfg,debug_pen_spin_runner,smoke_pen_spin_backend}.py` | Incumbent verification and rendering, the expert's training recipe, and the backend debug tools |

## Index

### Scripts (`deprecated/scripts/`)

| File | What it was | Decided by |
|---|---|---|
| `ablate_repair.py`, `diagnose_repair.py` | Phase 6 controlled-substitution diagnosis and the AC-6 repair ablation | AC-6: optimizer-only 0.650 vs full loop 0.517 |
| `analyse_failures.py` | FR-11 frame capture and VLM failure analysis | D-003: vision added +0.000 |
| `audit_expert_actions.py` | Phase 5 audit of the MetaWorld expert's clipping semantics | Phase 5 (finding kept in `lares/fitting/objectives.py`) |
| `benchmark_fitting.py`, `build_structure_bank.py` | Phase 3 optimizer comparison; builder of the now-frozen structure bank | Phase 3: the optimizer is not the bottleneck |
| `compare_objectives.py` | Phase 5 objectives, sampling and DAgger comparison (E7, E8) | Phase 5 |
| `compare_child_vs_parent.py` | Bounded mutation vs its own parent | R-005, R-007, R-010, R-013, R-014 (D-001, D-004, D-006, D-009, D-011) |
| `compare_generation_modes.py` | Single- vs two-phase generation, 24-candidate search, hard-task and trace-edit studies | R-015 to R-036 (D-013 to D-032). The recipe now runs in the orchestrator |
| `compare_evolution_runs.py`, `recover_provenance.py` | One-off comparisons and provenance sidecars for September runs | — |
| `qualify_fittable_bank.py` | Fittable-bank qualification (seed bank in `config/fittable_bank/seed_bank.json`) | R-004 (D-002) |
| `stage_e.py` | Stage E: imitation fitting vs zero-shot | Stage E: fitting off by default |
| `stage_f.py` | Stage F: correction-LLM repair vs matched-budget arms, including the vision arm | R-006, R-008, R-009, R-011, R-012, R-016 (D-003, D-005, D-010, D-007, D-008, D-012) |
| `visualize_expert_policy.py` | GIFs of the MetaWorld scripted expert | — |
| `isaac/audit_expert_switches.py`, `isaac/check_expert_query.py` | R-048 switch audit and its pre-launch check | D-043 |
| `isaac/axial_reversals.py` | R-071 axial-direction reversal rates from archived traces | R-071 (D-053) |
| `isaac/check_gait_scaffold.py` | R-046 pre-launch check of the gait-scaffold actors | D-041 |
| `isaac/tune_pen_spin_gait.py` | Tuning of the R-039 scripted gait on the debug seed | D-035: the scripted gait does not qualify |

### Cut out of live files (2026-11-15)

These three are not `git mv`s: each holds the part of a live file that was removed, so `git log --follow` on them
shows nothing. The history is under the live path.

| File | What it was | Cut from |
|---|---|---|
| `scripts/run_shadowhand_spin_stages.py` | Three-stage smoke run on Isaac Lab ShadowHandSpin: scripted dataset, legacy BC, GRPO on a toy `ShadowHandSpinSymbolicPolicy`, its own GIF recorder and `evaluate_stage_policy` | `scripts/run_shadowhand_spin_stages.py`, which keeps the `--study` / `--research-protocol` dispatch |
| `lares/envs/isaac_lab_adapter.py` | `IsaacLabSingleEnvAdapter`, the single-env view of a batched Isaac env. The smoke run was its only caller | `lares/envs/isaac_lab_adapter.py`, which keeps `launch_isaac_app` / `close_isaac_app` |
| `config/shadowhand_spin_smoke.yaml` | The smoke run's config keys | The head of `config/shadowhand_spin_stages.yaml` |

To run the smoke stages again, append the class back to `lares/envs/isaac_lab_adapter.py` (it needs `numpy` there) and
run the archived script from the project root; it reads the archived config by default.

### Library modules (`deprecated/lares/`)

| File | What it was | Only callers |
|---|---|---|
| `vision/` (whole package) | Qwen VLM failure analysis: frame selection, schema, merge, analyst | `analyse_failures.py`, `stage_f.py`, `repair/correction.py`. `training_pipeline.py` used one constant, `FLIP_RENDER_VERTICALLY`, which it now defines itself |
| `repair/hypotheses.py`, `interventions.py`, `repair.py`, `study.py` | Intervention-guided repair (FR-9, AC-6) | `ablate_repair.py`, `diagnose_repair.py`, `compare_generation_modes.py` |
| `repair/correction.py` | Stage F correction-LLM repair | `stage_f.py` |
| `repair/__init__.py` | The old package `__init__`, which re-exported the four modules above | — |
| `fitting/dagger.py` | Learner-state imitation with simulator-branched label validation | `compare_objectives.py` |
| `fitting/replicates.py` | Data-varying replicate bundles for qualification | `qualify_fittable_bank.py` (D-002) |
| `envs/isaac_pen_spin/axial_reversals.py` | R-071 reversal statistics | `isaac/axial_reversals.py` |

### Tests (`deprecated/tests/`)

`test_stage_e.py`, `test_stage_f.py`, `test_vision.py`, `test_repair.py`, `test_replicate_bundles.py`,
`test_fittable_bank.py` and `test_child_vs_parent.py` moved whole. Three files hold tests cut out of live test
files: `test_dagger.py` (from `tests/test_objectives.py`), `test_axial_reversals.py` (from `tests/test_pen_spin.py`)
and `test_r036_trace_request_parity.py` (from `tests/test_plateau_repair_mock.py`). The parity test checked that the
live plateau-repair request was byte-identical to R-036's. It passed on 2026-11-09, just before the move.
`scripts/gpu.sh test` only collects `tests/test_*.py`, so none of these run.

## Notes moved from CLAUDE.md

These paragraphs described the archived modules in `CLAUDE.md` until 2026-11-09. They are reproduced unchanged.

**Replicate bundles (`lares/fitting/replicates.py`).** Future qualification varies data, not minibatch order:
each replicate draws its own demonstration sample, trajectory-level split, probe manifest and qualification
manifest, and every structure in a replicate shares that bundle (`run_replicate_qualification` refuses
otherwise). Protocol predeclared, not run: `config/fittable_bank/qualification_v2.yaml` (seeds 2400-2424, unused
and unreserved on 2026-09-14).

**Stage F repair ablation (`lares/repair/correction.py`, `scripts/stage_f.py`).** Six arms per replicate from
one common preprocessing (rollout-selected fit, then one screen run): `no_repair`, three correction-LLM arms
differing only in the evidence packet (none / numeric / numeric + substitution study), `local_search`,
`global_search`. An LLM repair is one `repair`-mode proposal tied to the parent and the packet's feedback id,
one implementation plus at most one correction, then the parent's preprocessing again. Every arm is capped at
the same repair-evaluation episodes (`StageFBudget.repair_episode_cap`) and replaces the parent only if strictly
better on the screen by the checkpoint tie rule. Chains land in `<run>/<structure>_s<seed>/<arm>/chain.json`;
`verify_chain` rechecks them. Manifests are compared by placement hash (`task_hash`), not case id: case ids are
per-manifest indices and collide across manifests.

**Learner-state imitation (`lares/fitting/dagger.py`).** Rolls out on training placements only,
counts every expert query and environment step, and validates each label by branching the simulator,
applying the label, measuring the distance to the expert's own desired position, and rewinding.
About a third of labels fail that test, so the scripted expert is not a reliable recovery controller.

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
