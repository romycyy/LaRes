# Debug Progress

Tracks the implementation of `debug_spec.md`. Started 2026-09-13 on branch `staging` at `350409e`.

## Where this stands

| `debug_spec.md` step | State |
|---|---|
| 1. End-to-end lineage and LLM-call provenance (Stage A) | **done**, unit and mocked-integration tested |
| 2. Objective-aware fitter in the live path; consistent checkpoint reports (Stage B) | **done**, tested |
| 3. Records, resources, archive, reconstruction, gate levers, summaries (Stage C) | **done**, tested |
| 4. Two-generation smoke test (Stage D) | **done**: mocked run and live gpt-5 run pass every check, including a live correction |
| — Full run on the updated `config/run_full_evolution.yaml` | **done**: held-out success 0.340 against the pre-debug best's 0.120, paired +0.220, t95 [+0.045, +0.395] |
| 5. Fresh development-confirmation experiment (Stage E) | **done** at `f96f060`: no fitted condition beat zero-shot with its 95% interval above zero, so fitting is disabled by default |
| 6. Correction-LLM repair and clean ablation (Stage F) | not started; now unblocked. The old ablation is relabelled a pilot |

Phase 7 stays blocked. Nothing in this cycle opened the final-test split or ran Qwen.

## Defects

| ID | Defect | State | Where it is fixed | Test |
|---|---|---|---|---|
| D1 | Generation-1 records had empty `parent_ids`, `idea_id`, `proposal_id` | fixed for new runs; historical run gets a sidecar | `GeneratedCandidate.lineage_dict`, `ExperimentRecord._validate_generated`, `scripts/recover_provenance.py` | `test_two_generation_mock.test_generation_one_candidates_have_proposal_feedback_and_valid_parents` |
| D2 | Live generation returned parallel lists and discarded lineage and usage | fixed | `generate_candidates` returns `list[GeneratedCandidate]`; `LLMCallLedger` | `test_two_phase_policy_generation.TestGenerationZero` |
| D3 | Live fitter used `MSE + 0.01 * std.mean()` and trained the scale | fixed | `fit_policy_with_objective` (`deterministic_mse`, scale frozen), `SymbolicPolicy.fixed_std`, prompts | `test_fitting_provenance.TestScaleIsNotFitted` |
| D4 | Selected checkpoint paired with another checkpoint's diagnostics | fixed | `fit_policy_with_objective` recomputes diagnostics after loading the selected state; orchestrator asserts the policy holds it; the fitted report carries `checkpoint_step` and `parameters_sha256` | `test_fitting_provenance.TestCheckpointSelection`, `test_two_generation_mock.test_selected_parameters_diagnostics_and_report_agree` |
| D5 | Records could not reconstruct a result | fixed | hashed `policy_source.py`, `initial/final/selected_state.pt`, probe batch and expected actions under `gen_<N>/candidates/<attempt>/`; `provenance.reconstruct_record` | `test_provenance.TestReconstruction`, `test_two_generation_mock.test_candidates_from_both_generations_reconstruct_exactly` |
| D6 | `environment_steps` held gradient steps; simulator work uncounted | fixed | `lares/search/resources.py`; the key is refused in generated records; `final_report` counts only stream counters | `test_provenance.TestLedgerAndBundle.test_resource_ledger_keeps_streams_apart`, `test_two_generation_mock.test_simulator_and_gradient_work_are_separate` |
| D7 | Archive built from screen results, never updated | fixed | `Archive.add(scope=...)`, `Archive.confirm`; slots use confirmed metrics | `test_provenance.TestArchiveUpsert`, `test_two_generation_mock.test_expanded_results_update_the_archive_and_the_records` |
| D8 | A recorded gate could be forced without changing the action | fixed | `policy_validator._check_gate_levers` (`gate_not_a_lever`); prompts show `g = self.record_gate("g", g)` | `test_provenance.TestGateLevers` |
| D9 | Missing ideas silently padded | fixed | `_parse_ideas_response` and its padding deleted; `parse_ideas_payload` / `parse_proposals_payload` | `test_two_phase_policy_generation.TestParseIdeas` |
| D10 | Repair ablation had unequal budgets and no correction-LLM path | **open** (Stage F); relabelled a pilot in `RESEARCH_HANDOFF.md` and `IMPLEMENTATION_PROGRESS.md` | — | — |
| D11 | Summary claimed traceability from three fields | fixed | `provenance.traceability_summary`; `scripts/summarise_experiments.py` | `test_two_generation_mock.test_the_summary_reports_complete_traceability_only_when_it_is`, `test_missing_evidence_breaks_traceability_and_validation` |
| D12 | Candidate and failure ids not one-to-one with attempts | fixed | `CandidateAllocator`; record id is the attempt id `gen<g>_cand<s>_a<k>` | `test_two_generation_mock.test_every_attempt_has_a_unique_record` |

## What changed

### Stage A — generation and lineage

- **New `lares/search/generation.py`**, importing nothing from the pipeline or the generator:
  `FeedbackBundle` and `write_feedback_bundle`, `GeneratedCandidate`, `CandidateAllocator`,
  `LLMCallRecord` and `LLMCallLedger` (prompt and response written to `llm_calls/` and hashed, one
  JSONL line per call, `attempt_ids` marking shared calls), strict `parse_ideas_payload` and
  `parse_proposals_payload`, the parent-count rules (`check_semantic_parents`), the complexity budget
  (`check_complexity`, 6–10 parameters, more needs `complexity_justification`), `env_identity`.
- **`lares/core/policy_generation.py`** rebuilt after the prompt tables. `generate_candidates` asks for
  structured ideas (generation 0) or proposals plus ideas (generation 1+), retries a bad design payload
  once, then implements each slot: per-idea attaches each call to its idea; batched maps fences by a
  `# idea: <id>` label and fills any missing fence with a per-idea call. Each slot gets one attempt and
  at most one linked correction; the old unbounded outer loop is gone. Validation is the subprocess
  harness, then in-process validation with scale parameters forbidden, then the idea contract
  (`idea.check_against_policy`). Generation 1+ shows the implementer the parents' source and tells a
  repair or simplification to change only the suspected component. The feedback text must hash to the
  bundle. `get_symbolic_policies` keeps its signature but returns candidates; `valid_policies` and
  `final_attempts` recover the old shape. The single-phase path survives for scripts, maps sample `k`
  to slot `k`, and can never produce complete lineage.
- **`lares/search/schemas.py`**: `PolicyIdea.complexity_justification`,
  `MutationProposal.no_parent_rationale`, simplification now needs exactly one parent.
- **Prompts**: `ideas_system.txt` and `ideas_user.txt` demand the `PolicyIdea` JSON schema; new
  `proposals_user.txt` demands proposals plus ideas, lists eligible parents and measurable metrics;
  `code_feedback.txt` asks for bounded edits and proposals rather than code.

### Stage B — objective and checkpoints

- **New `lares/fitting/checkpoints.py`**: `RolloutScore(success, progress)`, `CHECKPOINT_TIE_RULE`
  (higher success, higher progress, earlier checkpoint), `is_better_checkpoint`,
  `default_checkpoint_steps` (step 0 plus evenly spaced steps).
- **`lares/fitting/optimizers.py`**: `fit_policy_with_objective(policy, data, objective, ...)` fits an
  existing policy; `fit_with_objective(structure, ...)` is now a thin wrapper. Probes step 0 and each
  `checkpoint_steps` entry, keeps `initial_state`, `final_state` and `selected_state` distinct,
  computes `final_diagnostics` on the final state and `selected_diagnostics` after loading the selected
  one (objective losses, per-axis losses, bounds, parameters, parameter hash, loss by phase). Reported
  final losses use the named objective, not the legacy `bc_objective`. `objective_config` stores the
  full configuration. `state_parameters_sha256` hashes tensors independently of pickling.
- **`lares/core/symbolic_policy.py`**: `DEFAULT_ACTION_SCALE = 0.1`, `fixed_action_scale` (a float,
  not a parameter), `fixed_std(mean)`. Prompts now return `self.fixed_std(mean)` and declare no scale.
- **`lares/core/training_pipeline.py`**: `EvolutionOrchestrator` fits through
  `fit_policy_with_objective` with `deterministic_mse`, uses a dedicated `PipelineEnvs.probe` env and a
  `probe_manifest`, and the fitted report's fitting section comes from the selected state only.
  `behavioral_cloning` now applies the same tie rule.
- **`lares/eval/report.py`**: `FittingSection` gains `checkpoint_step`, `objective`, `parameters_sha256`.
- **`lares/eval/runner.py`**: `PipelineEnvs` gains an optional `probe` stream, included in the
  distinct-instance check.

### Stage C — records, archive, gates, summaries

- **`ExperimentRecord`**: new `origin` (`generated`, `repair_study`, `legacy`; files written before it
  existed load as `legacy`), `lineage`, `llm_call_ids`, `reconstruction`, `record_version`. Generated
  records must carry slot, attempt, lineage status, validation result, code hash and embedded call
  hashes whatever their status; past ideation an idea; from generation 1 a proposal, feedback id and
  parents that obey the mode rule and lie in the context. Evaluated ones also need environment
  identity, hashed reports, objective config, fixed scale, selected step, three hashed states,
  checkpoint history, both diagnostics, a complete `ResourceLedger` and reconstruction metadata.
  `environment_steps` is refused. `save` is versioned: changed content moves the old file to
  `experiments/history/<id>.v<N>.json`.
- **New `lares/search/resources.py`**: `ResourceLedger` with the counters of spec 4.5 and wall time by
  generation, fitting, evaluation and repair.
- **New `lares/search/provenance.py`**: `lineage_status` (honours a sidecar), `traceability_gaps` and
  `traceability_summary` over the eight evidence fields, `rebuild_policy`, `reconstruct_actions`,
  `reconstruct_record`, `reconcile_resources` (tokens over unique call ids, and exclusive plus shared
  once; screen and expanded episodes against the saved reports).
- **`lares/search/archive.py`**: entries keep `screen_metrics` and `confirmed_metrics`; `add` upserts;
  `confirm` applies the expanded report; slots use confirmed metrics; an upsert cannot change the
  source hash.
- **`lares/core/policy_validator.py`**: `gate_not_a_lever` (forces every recorded gate to 0 and 1 on
  random and near-contact states), `scale_parameter` (an error when `forbid_scale_parameters`, else a
  warning). Test fixtures that discarded `record_gate`'s return value were corrected.
- **The orchestrator** writes, per run: `run_manifest.json` (environment identity, objective config,
  fixed scale, schedule, tie rule, probe isolation, case ids of every manifest), `llm_calls.jsonl`,
  `reconstruction_probe_batch.pt`, `archive.json`, `resource_totals.json` with the reconciliation;
  per generation `gen_<N>/feedback.md` plus `feedback_bundle.json` and `Iter_<N>_designs.json`; per
  attempt a record. Context parents are every evaluated candidate in the evidence plus the archive
  slot occupants.
- **`scripts/summarise_experiments.py`**: lineage-status counts, attempts versus slots, validated
  counts from `validation.passed`, slots valid within one correction, corrections attempted and
  succeeded with their token cost, per-field traceability that is complete only with no gap on any
  record, and resource reconciliation.
- **`lares/eval/final_report.py`**: simulator steps come from the stream counters only.
- **`lares/repair/study.py`**: its records are `origin="repair_study"`.

### Historical provenance (spec section 6)

`scripts/recover_provenance.py --log-dir logs/evolution_review_20260913` wrote
`provenance_sidecar.v1.json` and left the raw records untouched. All six records are
`partial_recovery`. Recovered: the gen-1 feedback context `[gen0_cand1, gen0_cand2]` from the full-code
parent headings (gen0_cand0's diagnostics-only appearance is listed separately), and the positional
mapping `gen<g>_cand<k> -> idea<k>` for both generations, justified by the transcript's
`policy_implement_0..2` sequence. Semantic `parent_ids`, `proposal_id`, structured ideas, call ids,
usage, states and simulator steps are listed as unknown. The summary now reports that run as six
`partial_recovery` records with every traceability field incomplete.

### Other changes

- `scripts/run_full_evolution.py` and `config/run_full_evolution.yaml`: `objective`,
  `fixed_action_scale`, `checkpoint_steps`, `probe_episodes` (the development tail, checked disjoint
  from the evaluated head), `max_repair_per_candidate`, `max_llm_calls_per_generation`, `param_budget`;
  a probe env; `rl_iterations` must be 0 and `policy_gen_two_phase` true.
- `scripts/compare_generation_modes.py` consumes candidates.
- `config/debug_smoke_two_generation.yaml`: the Stage D live smoke configuration.
- `CLAUDE.md`, `IMPLEMENTATION_PROGRESS.md` (status corrections from spec 2.2), `RESEARCH_HANDOFF.md`
  (pilot relabel with the t interval, open question 1 answered).

## Tests

| Suite | Result |
|---|---|
| `python -m unittest` over all sixteen unittest modules (thirteen existing, three new) | **587 passed** |
| `tests.test_two_phase_policy_generation` (rewritten) | 36 passed |
| `tests.test_fitting_provenance` (new) | 19 passed |
| `tests.test_provenance` (new) | 30 passed |
| `tests.test_two_generation_mock` (new, mocked Stage D) | 18 passed |
| `python tests/test_phase1_phase2.py` | 41/41 |
| `python tests/test_training_pipeline.py` | 123 passed, 0 failed, 2 skipped (Tier 3, see below) |
| `python scripts/summarise_experiments.py --log-dir logs/evolution_review_20260913` | 6 `partial_recovery`, every traceability field reported incomplete |

Baseline before any change: 490 unittest, 139/139 pipeline, 41/41 phase checks, all passing.

**Tests changed, and why.** `test_objectives.test_a_tie_prefers_the_later_checkpoint` became
`test_an_exact_tie_prefers_the_earlier_checkpoint` (the rule the spec declares). `test_search`'s
record fixture is marked `origin="legacy"` because it tests the legacy contract. `test_final_report`'s
fixture uses a stream counter. Two gate fixtures now use `record_gate`'s return value. Two prompt
checks in `test_phase1_phase2.py` and `test_training_pipeline.py` follow the new templates.

**Tier 3 now skips where it used to pass.** The pipeline suite's Tier 3 makes real gpt-4o-mini calls.
It used to pass because the old generator retried up to 50 calls until one policy validated. With one
linked correction per slot, gpt-4o-mini produced no valid policy in either run: single-phase code read
a non-existent `target` field and then mis-shaped a tensor, and in two-phase mode its structured ideas
parsed cleanly but the implementation ignored `goal` twice. Tier 3 therefore asserts the contract
(typed attempts, at most one correction, a stage and error on every failure, every record valid,
tokens reconciling) and skips the policy checks with the recorded reason. The pipeline suite's check
count fell from 139 to 125 because the old retry-dependent checks were replaced by contract checks.

## Stage D checklist against the mocked run

Every item is asserted in `tests/test_two_generation_mock.py` against saved files, on a kinematic push
world in the push-v2 schema (2 generations, 3 slots each, deterministic MSE, fixed scale, checkpoints
{0, 10, 20, 30, 40}, isolated probe/screen/expanded manifests, one successful and one failed
correction). The live run must repeat them.

- [x] unique candidate and attempt ids — `test_every_attempt_has_a_unique_record`
- [x] every generation-0 candidate has a valid idea — `test_generation_zero_candidates_have_a_valid_idea_and_no_parent`
- [x] generation-1 proposal, idea, feedback id, context and valid parents — `test_generation_one_candidates_have_proposal_feedback_and_valid_parents`
- [x] every prompt and response stored and hashed — `test_every_prompt_and_response_is_stored_and_hashed`
- [x] tokens reconcile on unique call ids — `test_token_totals_reconcile_on_unique_call_ids`
- [x] no scale-only parameter changes — `test_fitting_uses_the_named_objective_and_a_fixed_scale`
- [x] step 0 and scheduled probes recorded — `test_step_zero_and_every_scheduled_checkpoint_is_probed`
- [x] selected parameters and diagnostics agree — `test_selected_parameters_diagnostics_and_report_agree`
- [x] simulator and gradient steps separate — `test_simulator_and_gradient_work_are_separate`
- [x] expanded results update the archive — `test_expanded_results_update_the_archive_and_the_records`
- [x] candidates from both generations reconstruct exactly — `test_candidates_from_both_generations_reconstruct_exactly`
- [x] no false "complete" — `test_the_summary_reports_complete_traceability_only_when_it_is`, `test_missing_evidence_breaks_traceability_and_validation`, `test_a_deleted_state_file_is_not_reported_complete`
- [x] no final-test split or Qwen — `test_probe_screen_and_expanded_cases_are_isolated` checks every manifest's split; nothing imports the vision analyst

## Stage D — live smoke test (2026-09-13)

`scripts/run_full_evolution.py --config config/debug_smoke_two_generation.yaml`, gpt-5, per-idea,
2 generations x 3 slots, deterministic MSE, fixed scale 0.1, checkpoints {0, 500, 1000, 1500, 2000},
probe on `development#tail10`, screen `#head10`, expanded `#head30`. Exit 0 in 778 s. Artifacts in
`logs/debug_smoke_two_generation/`.

Every Stage D check passed, verified from the saved files alone (unique attempt ids; all records
validate; generation-0 ideas with no parents; generation-1 proposals, ideas, feedback id, context and
mode-valid parents; ledger hashes verify; tokens reconcile on unique call ids; no frozen or fitted scale
parameter; step 0 and every scheduled checkpoint probed; selected state, selected diagnostics, probe
entry and fitted report share one parameter hash; simulator and gradient steps separate and screen and
expanded episodes reconcile with the reports; the archive holds confirmed metrics for every expanded
candidate; all six candidates from both generations reconstruct with identical actions; probe isolated;
development split only; `push-v2` resolves to `push-v3`). `summarise_experiments.py` reports all six
lineages complete and all eight traceability fields complete.

| attempt | mode | parents | params | selected step | screen | expanded | status |
|---|---|---|---:|---:|---:|---:|---|
| gen0_cand0_a0 | idea | — | 10 | 1000 | 0.000 | — | rejected |
| gen0_cand1_a0 | idea | — | 10 | 1000 | 0.100 | 0.067 | promoted |
| gen0_cand2_a0 | idea | — | 10 | 2000 | 0.000 | — | rejected |
| gen1_cand0_a0 | repair | gen0_cand1 | 10 | 1000 | 0.100 | 0.033 | rejected |
| gen1_cand1_a0 | simplification | gen0_cand2 | 8 | 2000 | 0.000 | — | rejected |
| gen1_cand2_a0 | crossover | gen0_cand1, gen0_cand2 | 10 | 2000 | 0.100 | 0.100 | promoted |

Cost: 9 LLM calls, 98,984 tokens; 300 probe, 180 screen and 90 expanded episodes (85,500 simulator
steps); 12,000 gradient steps.

Observations:

- All six implementations passed validation first time, so no correction ran naturally. The spec's
  controlled fixture was therefore run live (`logs/debug_smoke_two_generation/correction_fixture/`):
  real gpt-5 ideation and implementation, with the implementation's `tcp` read replaced by
  `obs[:, 0:3]` before validation. Attempt `gen0_cand0_a0` failed at the harness with
  `raw_obs_indexing`; the real correction call produced `gen0_cand0_a1`, valid, with
  `previous_attempt_id = gen0_cand0_a0`, call labels `[ideation, correction]`, complete lineage, and a
  ledger of three calls whose hashes verify. The first
  ideation reply was rejected because one idea declared 11 parameters without a justification; the
  single retry fixed it. A retry resends the whole request (about 12k tokens); retrying only the failing
  slot would be cheaper.
- Every generated idea sat at the 10-parameter budget (one simplification at 8), against 15–19 in the
  pre-debug run.
- Most candidates never reach the object (`never_reached_object` dominates), and most probes score
  zero success, so checkpoints are chosen on progress.
- Performance is not better than the pre-debug run's best (0.267 expanded), but six candidates cannot
  show that either way. The full run is the comparison.

## Full run and held-out confirmation (2026-09-13)

**Config.** `config/run_full_evolution.yaml`, updated first: every Stage B key, the episode-id demo buffer
`logs/baseline_lock/demo_push-v2.pkl`, and a fresh `log_dir` (`logs/evolution_debug_full`) so the historical
incumbent in `logs/evolution` is untouched. gpt-5, per-idea, 3 generations x 5 slots, deterministic MSE, fixed
scale 0.1, checkpoints {0, 500, 1000, 1500, 2000}. Exit 0 in 1706 s.

**Run health.** 17 attempts over 15 slots; two first attempts failed validation and both corrections passed,
so 15/15 slots were valid within one correction. Lineage complete on all 15 evaluated candidates, every
traceability field complete, reconstruction verified. 20 LLM calls, 242,642 tokens (34,353 on the two
corrections), reconciled on unique call ids. 750 probe, 450 screen and 330 expanded episodes (229,500
simulator steps); 30,000 gradient steps.

**Development result** (`scripts/compare_evolution_runs.py`, 30 shared expanded cases, `#head30`). Best
candidate `gen2_cand0_a1`, 9 parameters, expanded success 0.567, against the pre-debug run's best
`gen0_cand1`, 15 parameters, at 0.267: paired +0.300, t95 [+0.057, +0.543], 12 wins, 3 losses, 15 ties.
Both bests were selected on these cases, from 15 candidates here against 6 there, so this number is
optimistic.

**Held-out result** (`scripts/evaluate_on_fresh_manifest.py`, MT1 pool seed 2100, 50 placements,
disjoint by placement hash from train, development and final-test; split `development`; outputs under
`logs/evolution_debug_full/confirmation/`). No selection happened on these cases.

| policy | success | mean return | dominant outcome |
|---|---:|---:|---|
| scripted expert | 1.000 | 1061.3 | success 50 |
| new run, first-generation best `gen0_cand0` at its selected step 0 | 0.400 | 358.7 | stopped short 25, success 20 |
| new run, final incumbent `gen2_cand0_a1` at its selected step 0 | 0.340 | 318.6 | stopped short 28, success 17 |
| pre-debug best `gen0_cand1`, fitted 2000 steps (its saved state) | 0.120 | 212.7 | stopped short 44, success 6 |
| pre-debug best `gen0_cand1`, untrained initial values | 0.000 | 15.3 | object not moved 43 |
| new final incumbent after 2000 fitting steps (its `final_state`) | 0.000 | 30.3 | object not moved 49 |

| paired comparison (50 cases) | success difference | t95 | w/l/t |
|---|---:|---|---:|
| new incumbent (step 0) - pre-debug best (fitted) | **+0.220** | [+0.045, +0.395] | 16/5/29 |
| new incumbent (step 0) - pre-debug best (untrained) | +0.340 | [+0.204, +0.476] | 17/0/33 |
| pre-debug best: fitted - untrained | +0.120 | [+0.027, +0.213] | 6/0/44 |
| new incumbent: fitted 2000 - pre-debug best fitted | -0.120 | [-0.213, -0.027] | 0/6/44 |
| new final incumbent - new first-generation best | -0.060 | [-0.194, +0.074] | 4/7/39 |

**What this says.**

1. **Performance improved, and the improvement survives on unseen placements**: +0.220 held-out success over
   the pre-debug best, with an interval above zero. The development estimates shrank on held-out cases in
   both runs (0.567 to 0.340, 0.267 to 0.120), which is the selection effect the caveat predicted.
2. **The gain is untrained structure plus the ability to keep it, not fitting.** Every promoted candidate in
   the full run selected step 0. The final incumbent scores 0.340 untrained and 0.000 after the 2000 fitting
   steps the pre-debug pipeline would have used. The pre-debug best is the reverse: 0.000 untrained, 0.120
   fitted. The structured idea prompt (stated untrained behaviour, a 6–10 parameter budget, a fixed scale)
   produced controllers that work before fitting, and the step-0 checkpoint let the run keep them. Neither
   change alone explains the result: the old checkpoint rule never probed step 0.
3. **Generations 1 and 2 added nothing measurable.** The first-generation best is at least as good held-out as
   the final incumbent. On development the incumbent moved from 0.500 to 0.567 through two promotions, one on
   tied success and higher return, one on +0.067 success with an interval containing zero. Several repairs had
   different source but identical per-case development success to their parent, and every step-0 candidate
   that kept its parent's parameter names and initial values shares one parameter hash.
4. **Behavioural cloning spent 30,000 gradient steps and 112,500 probe simulator steps for no benefit to any
   promoted candidate.** For these structures it moves the controller away from success.

Seed 2100 has now been looked at. Stage E must use a different, unused pool seed.

## Decisions of 2026-09-14 (after the full run)

Recorded as given; each is implemented or scheduled as noted.

1. **Imitation fitting is an experimental treatment.** Stage E reports four conditions on one immutable
   confirmation manifest: zero-shot (step 0), deterministic MSE fixed at step 2000, deterministic MSE
   rollout-selected from {0, 500, 1000, 1500, 2000}, and phase-balanced sampling rollout-selected from the
   same checkpoints. Checkpoints are selected on a development slice, never on the confirmation cases.
   Fitting is kept only if at least one fitted condition beats zero-shot on held-out paired success with
   its 95% interval above zero; otherwise it is disabled by default. Frozen in
   `config/stage_e/stage_e.yaml`, driven by `scripts/stage_e.py`.
2. **Both structure families.** `simple_reach_push`, `simple_damped_push`, `simple_standoff`, and the top
   five reconstructable evolved policies from `logs/evolution_debug_full`, selected and deduplicated by
   code and behaviour before any Stage E case is generated (minimum three).
3. **Promotion requires real evidence** (implemented, `lares/eval/promotion.py`). A challenger replaces the
   incumbent only when its paired success difference is strictly positive and the lower bound of its
   paired t interval is above zero, on a promotion manifest used for neither screening nor checkpoint
   selection. Tied success never promotes; return and progress only rank or admit to the archive. Only the
   best screened challenger is tested; several are judged with Holm's correction. The development
   manifest is now split into a screen (cases 0-9), a disjoint promotion set (10-39) and the probe tail
   (40-49), and `EvolutionOrchestrator.check_manifest_isolation` refuses overlap. Before any promotion the
   incumbent is the documented 0.00-success baseline.
4. **No ordinary edit generations in the next search** (implemented in `config/run_full_evolution.yaml`).
   One initial generation receives the former three-generation budget, matched by LLM calls and tokens: a
   new `max_llm_tokens_per_generation` cap (242,642) alongside the call cap (20). The measured per-slot cost
   buys about 15 slots rather than the nine estimated; `pop_size` is 15 and the caps enforce the match.
   The two-generation smoke config still exercises edit plumbing; meaningful edit evaluation moves to
   Stage F, whose feedback must include paired parent-versus-child case differences.
5. **Seeds.** MT1 pool seed 2200 for the confirmation manifest and fitting seeds 2200-2204, verified unused
   against `config/experiment_seed_ledger.jsonl` (new, backfilled from every recorded use) and every saved
   manifest before the manifest is built, then reserved in the ledger.
6. **Commits.** The debug implementation, prompts, tests, scripts and documentation are committed after all
   tests pass; the frozen Stage E configuration, selection and manifest are committed separately; Stage E
   refuses a dirty worktree and records the Git SHA in every result.
7. **Phase 7 stays blocked through Stage F** (`CLAUDE.md` corrected).

## Stage E result (2026-09-14)

**Run.** `scripts/stage_e.py run`, run `logs/stage_e/20260914_032500`, 697 s on 8 workers, at commit `f96f060`
with a clean worktree (every structure result records the SHA). Frozen inputs verified by hash: protocol
`config/stage_e/stage_e.yaml`, selection `config/stage_e/selection.json`, confirmation manifest
`push-v2:development_confirmation:s2200:n50` (sha256 `37645f74…`), demo buffer, seed ledger. Checkpoints were
selected on development cases 40-49 (`#slice40-50`); every condition was scored on the 50 confirmation
placements. 8 structures x 5 fitting seeds (2200-2204). Cost: 160,000 gradient steps, 4,000 probe episodes
(600,000 simulator steps), 3,900 confirmation episodes (585,000 steps). The final-test split was not opened.

**Structures.** Bank: `simple_reach_push`, `simple_damped_push`, `simple_standoff`. Evolved (selected before the
manifest existed, all 15 candidates reconstructed, `gen2_cand1_a0` dropped as behaviourally identical to
`gen0_cand0_a0`): `gen2_cand0_a1`, `gen1_cand2_a0`, `gen1_cand1_a0`, `gen1_cand0_a0`, `gen0_cand0_a0`. All five
descend from `gen0_cand0`, so the evolved family is narrow.

**Confirmation success, mean over seeds** (selected checkpoint steps in brackets):

| structure | family | zero-shot | MSE fixed 2000 | MSE rollout-selected | phase-balanced rollout-selected |
|---|---|---:|---:|---:|---:|
| evolved_gen0_cand0_a0 | evolved | 0.400 | 0.000 | 0.400 [0] | 0.400 [0] |
| evolved_gen1_cand0_a0 | evolved | 0.400 | 0.000 | 0.400 [0] | 0.400 [0] |
| evolved_gen1_cand1_a0 | evolved | 0.400 | 0.000 | 0.400 [0] | 0.400 [0] |
| evolved_gen1_cand2_a0 | evolved | 0.380 | 0.000 | 0.380 [0] | 0.380 [0] |
| evolved_gen2_cand0_a1 | evolved | 0.380 | 0.000 | 0.380 [0] | 0.380 [0] |
| simple_damped_push | bank | 0.120 | 0.504 | 0.908 [1000] | 0.952 [1500, 2000] |
| simple_reach_push | bank | 0.580 | 0.020 | 0.956 [1000] | 0.924 [1000] |
| simple_standoff | bank | 0.000 | 0.000 | 0.000 [500] | 0.000 [500] |

**Predeclared analysis** (structure is the unit; difference from zero-shot averaged over seeds; t interval
over the 8 structures):

| condition | mean | 95% t interval | median | bank mean [t95, n=3] | evolved mean [t95, n=5] | Holm |
|---|---:|---|---:|---|---|---|
| MSE fixed 2000 | -0.267 | [-0.524, -0.010] | -0.390 | -0.059 [-1.238, +1.121] | -0.392 [-0.406, -0.378] | fail |
| MSE rollout-selected | +0.146 | [-0.098, +0.389] | +0.000 | +0.388 [-0.591, +1.367] | +0.000 [0, 0] | fail |
| phase-balanced rollout-selected | +0.147 | [-0.105, +0.399] | +0.000 | +0.392 [-0.647, +1.431] | +0.000 [0, 0] | fail |

**Decision, as frozen: fitting is disabled by default.** No fitted condition beat zero-shot with its 95%
interval above zero. `config/run_full_evolution.yaml` now sets `bc_steps: 0` and no checkpoint probes; fitting
remains an experimental treatment. A mocked run with fitting disabled was added
(`test_two_generation_mock.TestFittingDisabled`): selected step 0, no probes, initial and selected states
identical, records valid, reconstruction exact, traceability complete.

**What the pooled rule hides.**

1. **The effect is family-specific, not small.** Rollout-selected fitting lifts two of three bank structures
   from 0.12 and 0.58 to 0.91–0.96, the highest held-out success of any policy in this project, and does
   nothing for the evolved structures, whose probes always pick step 0. The pooled interval crosses zero
   because five evolved structures contribute exactly +0.000, not because fitting is weakly positive.
2. **Fixed-budget fitting is harmful**, significantly so overall, and destroys every evolved structure
   (0.38–0.40 to 0.000). If fitting is used at all, it must be rollout-selected.
3. **The evolved structures are fitting-resistant, and the bank structures are fitting-dependent.** The
   debug-cycle prompts produced controllers that work untrained and that imitation moves away from success;
   the hand-written references are poor untrained and excellent at the pre-collapse checkpoint.
4. **Phase-balanced sampling adds nothing measurable over uniform** once the checkpoint is rollout-selected
   (+0.147 against +0.146).
5. With the family split this stark, the family-level intervals are too wide to decide anything on three bank
   structures. A decision about fitting for hand-written or fitting-dependent structures would need more of
   them.

## Decisions and deviations

1. **Paths.** The spec names `lares/fit/objectives.py` and `lares/fit/fitter.py`; the package is
   `lares/fitting/`, and the new fitter sits beside `fit_with_objective` in `optimizers.py`.
2. **RL is refused by the orchestrator.** Fine-tuning after checkpoint selection would change the
   parameters after they were selected and recorded. `rl_iterations` is 0 in the config already.
3. **The orchestrator requires two-phase generation.** Single-phase candidates have no idea, so their
   lineage could never be complete.
4. **A probe env is required** unless `PipelineEnvs(allow_shared=True)`, in which case the run
   manifest records that the GIF stream was shared.
5. **`origin` defaults to `generated`** for new records, so the strict contract is the default; files
   without the field load as `legacy`.
6. **Scale parameters are an error only in generation.** The frozen bank structures declare `log_std`
   and are still validated elsewhere, where it is a warning.
7. **Record ids are attempt ids**; report files keep the candidate id, since only a slot's final valid
   attempt is evaluated.
8. **The tie rule changed for `behavioral_cloning` too**, which the Isaac Lab script still uses. An
   exact tie now keeps the earlier checkpoint there as well.
9. **Losses reported by `fit_with_objective` now use the named objective.** Phase 5 numbers produced
   before this change included the legacy scale term in `train_loss_final`; rerun rather than compare.
10. **Probe episodes have no saved report**, so they are reconciled from records only, and the
    reconciliation says so. Probe wall time is counted inside fitting.
11. **Prompt templates changed.** No freeze record existed, so no drift check is affected; the first
    freeze will digest the new templates.

## Remaining work

1. **Done: live Stage D and a full run.** Reproduce with:

   ```bash
   export MUJOCO_GL=egl
   .venv-metaworld/bin/python scripts/run_full_evolution.py --config config/debug_smoke_two_generation.yaml
   .venv-metaworld/bin/python scripts/run_full_evolution.py --config config/run_full_evolution.yaml
   .venv-metaworld/bin/python scripts/summarise_experiments.py --log-dir logs/evolution_debug_full
   .venv-metaworld/bin/python scripts/compare_evolution_runs.py --baseline logs/evolution_review_20260913 --run logs/evolution_debug_full
   ```
2. **Stage E: done** (see "Stage E result"). Fitting disabled by default. Seed 2200 and fitting seeds
   2200-2204 are spent.
3. **Research decisions raised by the full run** (`RESEARCH_HANDOFF.md`, question 0): whether imitation
   fitting stays in the loop, whether promotion should require a paired interval above zero, and whether the
   edit generations earn their cost, since they did not improve held-out success.
3. **Stage F (D10)**: numeric evidence packet, correction-LLM `MutationProposal` tied to parent and
   feedback ids, one generation or patch, common preprocessing, matched budgets across the six arms,
   the whole chain recorded with its cost.
4. Not yet on the new contract: `scripts/run_shadowhand_spin_stages.py` still fits with
   `behavioral_cloning`, and `scripts/compare_objectives.py` and `scripts/ablate_repair.py` still
   report through their own JSON rather than `ExperimentRecord`s with resource ledgers. Both
   matter for Stages E and F.
