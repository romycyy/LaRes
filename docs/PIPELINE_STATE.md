# LaRes pipeline: current structure and what has been established (2026-11-09; code notes updated 2026-11-15)

This is the one document to read first. It describes the pipeline as it exists on `staging` at commit `613fa84`, and
separates what has been **verified** from what is only **built**, **assumed** or **open**. It covers 91 research ideas
(R-001 to R-091) and 63 decisions (D-001 to D-063).

Authoritative sources, in order (if this file disagrees with them, they win):

1. The research ledger: `research/ideas/R-*.json`, `research/decisions.jsonl`, `research/triage.jsonl`.
2. Each run's `logs/<run>/research_result.json` and `summary.json` (logs are gitignored; they live locally and on the box).
3. The decision blocks appended to `RESEARCH_HANDOFF.md`.
4. `CLAUDE.md`, the operating manual: commands, contracts and the things that will bite you.
5. Session summaries: `docs/SESSION_2026-10-27_PEN_SPIN_SUMMARY.md`, `docs/SESSION_2026-11-04_PEN_SPIN_SUMMARY.md`.

Older plans and progress files are in `docs/deprecated/` (index below). Code docstrings still cite `spec.md` (FR-*, AC-*)
and `debug_spec.md` (Stages A to F, D1 to D11). Both files are now in `docs/deprecated/`.

---

## 1. The idea in one paragraph

An LLM (gpt-6.1-sol, `reasoning_effort: high`, temperature 1) writes **symbolic policies**: small PyTorch classes with
named, range-bounded parameters and no neural-network modules. They read the observation only through named fields.
Every policy is scored on frozen **evaluation manifests**, never on ad hoc episodes. A candidate replaces the
incumbent only by a **strict paired promotion rule**. Every attempt, including code that never compiled, is recorded with
lineage, LLM-call ledger entries, hashes and resources. Above this sits a **research ledger**. The research loop is run
by an advisor LLM and Claude, who must agree on each step. The user sets the direction. The pipeline runs on two task
families:

| Family | Simulator | Status |
|---|---|---|
| MetaWorld manipulation (push, pick-place, 8 easy tasks, 6 hard tasks) | MuJoCo, MetaWorld 3.0 (task names stay `-v2`) | **Solved for this recipe** except where noted (§5) |
| Shadow Hand pen spinning (axial roll, Stage 1/2/3) | Isaac Lab 2.x / Isaac Sim 5.1, vendored env | **Stuck** at about 0.35-0.44 sustained success (§6) |

---

## 2. Core contracts (verified by tests; unchanged since October)

### 2.1 The `SymbolicPolicy` contract (`lares/core/symbolic_policy.py`)

A generated policy must:

- be `class GeneratedPolicy(SymbolicPolicy)`, constructed as `(obs_dim, action_dim)`;
- return `(mean, std)` from `forward(obs)`, both `(batch, action_dim)`, with `std = self.fixed_std(mean)`. The action
  scale is fixed (`fixed_action_scale`, default 0.1), and a parameter that only moves the scale is rejected;
- read the observation only with `self.obs_field(obs, "<name>")`. An AST check rejects raw slicing,
  `narrow` and `split`;
- contain no `nn.Linear`/`Conv`/`LSTM`/... modules;
- declare `get_param_ranges()` for every parameter, with each initial value inside its range. `clip_params()` runs
  after every gradient step;
- keep rows independent (no batch coupling) and respond to every `required` schema field;
- if it has a phase gate, write `g = self.record_gate("name", g)` and use the returned `g`. A gate whose forced value
  never changes the action fails as `GATE_NOT_A_LEVER`.

**Observation schemas** (`lares/core/obs_schema.py`): `get_obs_schema(env_id)` raises for unknown tasks.
`check_against_env` compares positions against the simulator. Schemas exist for the MetaWorld tasks in §5, the 157-d
pen-spin layout, the 177-d `GaitScaffold` and the `ParentScaffold` layouts. The active schema is thread-local
(`obs_schema._LOCAL`).

**Validator** (`lares/core/policy_validator.py`): `validate_policy` collects every defect with a stable category.
Categories include raw indexing, forbidden modules, phantom or undeclared parameters, inverted ranges, out-of-range
initial values, shapes, non-finite output, `std <= 0`, batch coupling, insensitivity, gate misuse and serialisation.
It probes at several observation scales. `tests/test_generate_policy.py` is the harness appended to generated source
and run as a subprocess. Do not run it directly.

### 2.2 Generation (`lares/core/policy_generation.py`, `lares/search/generation.py`)

- **Flow:** format prompt → LLM call → extract code block → find `class GeneratedPolicy` → subprocess harness must print
  `Success!` → `exec` in-process → `validate()`. Failures go back to the LLM as **one linked correction per slot**,
  never an outer retry (`_correct_until_valid`, shared by the per-idea implementer, the single-phase exploration and
  the trace-repair slot). The slot validator writes the harness's one-row `data.pkl` itself. Only the class is kept,
  so module-level constants in a reply are dropped. `generate_candidates` is the only entry point; the legacy
  `get_symbolic_policies` single-phase path was removed on 2026-11-09.
- **Two generators:**
  - *Single-phase*: write code directly. **Established better:** +0.698 [+0.395, +1.000] over two-phase on push
    (D-014). Since then it is the generator every exploration study uses.
  - *Two-phase*: JSON `PolicyIdea`s, then implementations. It scored 0.00 untrained on every block.
- **Provenance:** slot and attempt ids (`gen<g>_cand<s>_a<k>`) are allocated before any call. Every call is one
  `LLMCallLedger` entry in `<log_dir>/llm_calls.jsonl`, with hashed prompt and response files.
- **Concurrency:** `lares/envs/isaac_pen_spin/llm_pool.py` runs per-(block, arm) generation on worker threads
  (`llm_workers`, default 16). Slots are saved to `generated.json` before any episode runs, so a rerun rebuilds them
  without LLM calls.

### 2.3 The evaluation contract (`lares/eval/`)

- **Manifests.** An `EvaluationManifest` is an ordered, immutable list of `EpisodeCase`s: task placement, reset seed
  and policy seed. Pools come from `MT1(env_id, seed=S)` (MetaWorld) or from pool seeds through `cases.py` (pen spin).
  Every reset must name its case: `env_wrapper.reset()` with no argument raises.
- **Committed splits** (`config/manifests/`): push-v2 train (150 cases, seeds 1000-1002), development (50, seed 2000)
  and **final_test** (100, seeds 3000-3001), disjoint by placement hash. Pen-spin manifests follow the pattern
  `pen_spin_stage<k>_<role>_s<seed>.yaml`.
- **Roles inside a development manifest.** Screen: cases 0-9. Promotion set: cases 10-39 (`manifest.slice`). Probe:
  the tail. `check_manifest_isolation` refuses any overlap.
- **Promotion rule** (`lares/eval/promotion.py`, `lares/eval/stats.py`): a challenger replaces the incumbent only if
  its paired success difference on the promotion set is **> 0 and the lower bound of the paired Student-t 95% interval
  is > 0**. Tied success never promotes. Several challengers are Holm-corrected. The starting incumbent is the null
  policy at 0.00.
- **Final-test lock** (`lares/eval/freeze.py`): `evaluate_manifest` refuses the `final_test` split unless a
  `final_test_session` is open on a complete `FreezeRecord`. **The final-test split has never been opened.** Phase 7
  (freeze plus final test) is still blocked.
- **Seed ledger** (`config/experiment_seed_ledger.jsonl`, `lares/utils/seeds.py`). A seed that has been looked at is
  spent. A frozen manifest *reserves* its seed for the idea that names it. New protocols take seeds above the highest
  spent one (4702 is the latest).

### 2.4 Records, archive and feedback (`lares/search/`)

- `ExperimentRecord` is written for every attempt (`<log_dir>/experiments/`). Evaluated records must carry a code
  hash, manifest ids, action mode, objective config, hashed state, a `ResourceLedger` and a verified reconstruction.
- The archive keeps three slots: most robust, simplest competitive and most behaviourally distinct. Intervention
  results can never fill a slot.
- `feedback.py` builds the generator's evidence block: two differing parents, the population table, per-episode
  geometry and errors. `scripts/summarise_experiments.py` rebuilds the search summary from records alone.

---

## 3. The orchestrator (`scripts/run_full_evolution.py` → `lares/core/training_pipeline.py`)

`EvolutionOrchestrator.run()` is the only orchestrator. Per generation:

```
generate candidates (single- or two-phase, lanes)        -> GeneratedCandidate with lineage
  for each valid candidate:
     zero_shot report (declared initial values)           <- always evaluated
     [optional] fit (fit_policy_with_objective)           <- OFF by default (bc_steps: 0)
     screen on cases 0-9 (success, else top-k by progress)
  best screened challenger -> promotion set (10-39) -> strict paired rule -> incumbent or not
  records, screening.json, incumbents.json, feedback for next generation
```

Configured plans that change this skeleton:

| Plan | Config | Status |
|---|---|---|
| Baseline: 1 generation × 9 unseeded zero-shot candidates | `config/run_full_evolution.yaml` | gpt-5.6-sol 0.333; gpt-6.1-sol **0.000** (two-phase generator) |
| Two-lane: 6 fittable mutations of seed-bank parents + 3 explorations | `config/two_lane_search.yaml`, `_v2` | Live. Saturates push at 1.00 (D-015); children ≈ parents (D-001) |
| Plateau-triggered trace repair: two slots; if all 24 search candidates score 0 probe successes, edit the best failed one with expert traces | `config/assembly_two_lane_repair.yaml`, `config/pen_spin_plateau_repair.yaml` | **Works on assembly-v2** (D-032, D-033). **Fails on pen spin** (D-038, D-047) |
| Pen-spin studies (`--study keyframe_repair`, `two_channel_cold`, `self_trace_source`, `coordinate_repair`, ...) | `PEN_SPIN_STUDIES` registry in `run_full_evolution.py` | See §6 |

**"24-candidate search"** means one generated structure plus independent rollout search over its declared parameter
ranges: 24 parameter draws, each scored on the probe, with the winner taken to promotion. From R-017 on this has been
the standard treatment for a generated structure.

Built but **off the main path** (kept, tested, used only by specific plans):

- **Fitting** (`lares/fitting/`): five objectives with execution contracts, four optimizers, a frozen 18-structure bank
  and ground-truth expert phases. The orchestrator's fitter and the two-lane plan use them.
- **RL fine-tuning** (`rl_finetune`, GRPO): not an orchestrator stage at all (the orchestrator has no `rl_iterations`;
  `load_config` refuses a nonzero one and the key may be omitted). No entry point calls it, or the legacy
  `behavioral_cloning`, since the BC+GRPO smoke stages of `run_shadowhand_spin_stages.py` were archived on 2026-11-15;
  both stay in `training_pipeline.py` because `tests/test_training_pipeline.py` and two other test files exercise them.

**Simplified on 2026-11-09** (`/simplify` pass over the live tree; behaviour, recorded formats and episode order
unchanged; every live test passes): one generation front-end (`llm_evolution(plan=...)`) and one orchestrator
construction (`run_full_evolution.orchestrator_from_config`); the orchestrator lost eight constructor arguments nothing
passed; `record_episode_gif` runs the shared `run_episode`; one correction loop in generation; the research seed scan is
memoised per process (a `research.py` command went from about 23 s to about 1 s locally); `ObsSchema.field` is a dict
lookup; child processes are spawned with `cli.self_argv`; every ledger goes through `lares/utils/jsonl.py`; dead entry
points and package re-exports were deleted (list in `CLAUDE.md`, "Added 2026-11-09"). The pen-spin study modules share
their prologue through `protocols.py` / `cases.py` / `block_study.py` (see §9).

**Refined on 2026-11-14** (third pass; findings from six Opus reviewers, because the OpenAI account had no credit and
the planned gpt-6.1-sol review never ran; behaviour, recorded formats, prompts, child argv, episode order and RNG use
unchanged; an Isaac replay of the incumbent on the R-081 fresh manifest was bit-identical, 80 cases, 35 successes):

- *Defects fixed.* `_call_llm` used to retry any error up to 1000 times and then call `exit()`, so a spent credit
  balance looped for hours and ended with status 0. It now retries only rate-limit, connection and 5xx errors
  (`_LLM_MAX_ATTEMPTS` = 8, exponential backoff capped at 60 s) and raises `RuntimeError` on anything else. The
  validation-harness subprocess has a 600 s cap (`VALIDATION_TIMEOUT_S`); a timeout is a harness failure and gets the
  slot's one correction. A NaN output is reported once as `NON_FINITE`, not also as batch coupling or irreproducible
  telemetry. `manifest_pool_seeds([])` scans nothing instead of the whole repo. Pen-spin manifests are checked for
  disjointness before they are written, not after. `gpu sync` deletes files that were removed or archived locally (45 stale files, including
  archived tests, were still on the box and ran under `gpu test`).
- *One copy each.* `training_pipeline._rank_cleared` (challenger ranking), `EvolutionOrchestrator._score_on_promotion_set`
  and `_finish_plateau_log`; `obs_schema._register_scaffold_schemas` (both 177-d layouts);
  `run_full_evolution._save_best`; `dense_fit.build_policy(obs_dim=, action_dim=)` (the one pen-spin policy rebuild: the
  Isaac child, a saved incumbent, a reloaded generation, the residual); `protocols.in_play` / `actions_match` (the seven
  studies' offline identity gates); `learner_recovery.make_learner(protocol_cfg, transform=None)` / `learner_source` (the
  frozen learner, used by nine study modules); `block_study.started` / `arm_needs_generation` / `parent_like` /
  `screen_best` in the four LLM block studies. `block_study.make_slot_generator` no longer writes a second `data.pkl`.
- *Removed as dead.* `seeds.seed_conflicts` / `reserve_seeds` / `clear_manifest_cache`, `plateau_trigger(already_fired=)`,
  three unused `PlateauRepairPlan` fields, the orchestrator's `plateau_log` and `run_manifest_ref`,
  `pipeline_backend.make_temp_stream`, `cases.split_placement`, `protocols._child`, unused `max_steps` parameters and
  aliases in the study modules.
- *Considered and left alone:* the repeated fitted screen when the fitted state equals the initial state (it is part of
  the recorded resource accounting), dead `behavioral_cloning` arguments, and merging the generation front-end's two
  prompt-setup blocks.

**Archived on 2026-11-15:** the BC + GRPO smoke stages that `scripts/run_shadowhand_spin_stages.py` ran when called
with no `--study` (scripted dataset, legacy BC, GRPO, a toy policy, its own GIF recorder and evaluator) and their only
dependency, `IsaacLabSingleEnvAdapter`. They are `deprecated/scripts/run_shadowhand_spin_stages.py`,
`deprecated/lares/envs/isaac_lab_adapter.py` and `deprecated/config/shadowhand_spin_smoke.yaml`. The live script is now
only the `--study` / `--research-protocol` dispatcher (599 → 165 lines) and refuses a bare call;
`lares/envs/isaac_lab_adapter.py` keeps the app launch and close helpers every Isaac child uses;
`config/shadowhand_spin_stages.yaml` keeps `isaaclab_root`, `device` and an unchanged `research_protocols` block. This
also removes the fourth copy of the rollout loop. Checked on the box: all live tests pass, and the R-044 expert cell
replayed through the slimmed script is identical to the archived one (40 cases, 39 successes).

**Archived on 2026-11-09** (`deprecated/`, index in `deprecated/README.md`; moved with `git mv`, not run or tested):
intervention-guided repair and the Stage F correction-LLM (`lares/repair/` except the two trace modules), VLM failure
analysis (`lares/vision/`, D-003), DAgger and replicate bundles, and the drivers of finished studies (Phase 3/5/6,
Stage E/F, fittable-bank qualification, the R-005 to R-036 comparison scripts, and one-off pen-spin checks). Configs and
manifests stayed in `config/` so seed bookkeeping is unchanged.

---

## 4. The research loop (`lares/research/`, `scripts/research.py`, `research/`)

```
direction (user) -> propose (advisor) -> triage: advisor + Claude agree (<= 5 rounds, else escalate to user)
   -> approved -> Claude implements, runs on the GPU box -> record research_result.json
   -> Claude's verdict -> analyse (advisor) -> decide (agree, <= 5 rounds) -> RESEARCH_HANDOFF.md block -> commit
```

- An idea must predict a measured metric with a direction and threshold. Its protocol (script, config, exact command,
  seeds, manifests, episode and call estimates) is fixed before any result. Final-test references are refused.
- **Loop guard:** two propose calls in a row rejected in full, or a zero-idea advisor reply, stop `propose` until the
  user sets a new direction. **This is the current state.** The advisor abstained twice on 2026-11-04 (calls
  00178/00180).
- Status counts: 63 decided, 19 deferred, 9 rejected, 0 open.

---

## 5. MetaWorld: what is established

Every number below is a paired success difference with a t95 interval, or a success rate on fresh manifests, from the
named decision.

### 5.1 Imitation, fitting and repair (Phases 3-6, Stages E-F, D-001 to D-013)

| Finding | Evidence | Status |
|---|---|---|
| Imitation loss is the wrong selection signal: validation loss correlates with success at **+0.19**. BC success peaks near 1000 steps and is 0 by 3000 | Phase 5, `RESEARCH_HANDOFF.md` | Verified (2 structures × 3 seeds) |
| Fixed-budget fitting is harmful (−0.267 vs zero-shot). Rollout-selected fitting helps hand-written bank structures (0.91-0.96) but not evolved ones | Stage E | Verified. Fitting off by default |
| The optimizer is not the bottleneck. Structures differ more than the four optimizers | Phase 3 benchmark | Verified |
| The repair loop does not pay. AC-6: optimizer-only 0.650 vs full loop 0.517. Stage F: no correction-LLM arm beats no-repair. Vision adds +0.000 (D-003) | AC-6, Stage F, D-003 | Verified |
| Data-varying replicate bundles are no noisier than minibatch-only ones (D-002) | D-002 | Refuted prediction |
| LLM mutations of fitted seed-bank parents ≈ parents (−0.020 [−0.046, +0.006]). One geometry child of `simple_damped_push` beat its parent robustly (+0.104, 5/5 seeds, D-006), but geometry edits do not transfer as a rule (−0.237, D-008) | D-001, D-006, D-008 | Verified |
| Search vs imitation from the same parameters: +0.080 [+0.002, +0.158], split by structure | D-005 | Inconclusive |

### 5.2 Generation and search (D-014 to D-019)

| Finding | Evidence |
|---|---|
| **Single-phase generation + 24-candidate rollout search** is the working recipe. It beats two-phase generation by +0.698 | D-014 |
| A parameterisation contract appended to the prompt hurt (D-013). A source-verified interface table did not help (D-017) | D-013, D-017 |
| The recipe **saturates** push, pick-place and 8 easy tasks (reach, door, drawer, window, button) at 1.00 | D-015, D-016, D-019 |

### 5.3 Hard tasks and trace repair (D-020 to D-033)

| Finding | Evidence |
|---|---|
| Of 6 hard tasks, **assembly-v2** is the only reproducible plateau (0.00, and 0.00 again on an independent repeat). Policies lift the nut but cannot seat it (progress ≈ 0.24). Hammer, shelf-place, basketball and soccer are solved by search; peg-insert-side's expert fails the 0.90 gate (0.63) | D-020, D-026, D-029, D-031 |
| Structural breadth vs depth at a 24-candidate budget: both 0.00 | D-024 |
| Fitting to expert demonstrations does not fix assembly (D-022). An edit from the evidence summary alone does not either (D-023) | D-022, D-023 |
| **Expert-trace whole-controller editing works on assembly-v2.** Two expert-only traces in the edit prompt give **1.00 vs 0.00** for summary-only edits. This replicated across 3 parents and 2 blocks | D-027, D-028, D-030 |
| Integrated into the orchestrator as plateau-triggered repair: the trace edit became the incumbent through the strict rule and scored **30/30 fresh** in both blocks, with repair-off at 0 | D-032, D-033 |

**MetaWorld bottom line.** Single-phase generation plus rollout search solves every supported task except assembly.
On assembly, plateau-triggered expert-trace editing solves it. Imitation fitting and correction-LLM repair do not help.

---

## 6. Shadow Hand pen spinning: what is established

### 6.1 The task as implemented (`lares/envs/isaac_pen_spin/`)

| Item | Value |
|---|---|
| Env | Vendored `ShadowHandSpinEnv` (Isaac Lab in-hand manipulation). Stages 1/2/3 share dynamics; they differ in success thresholds and reward weights (target spin 2/5/7 rad/s) |
| Observation | 157-d `full` layout (`layout.py`, `cases.OBS_DIM`): 24 joint positions and 24 joint velocities; pen position, rotation, linear and angular velocity; in-hand position; goal and relative rotation; 5 fingertip positions, rotations and velocities; last action |
| Action | 20-d joint targets in [-1, 1], 60 Hz control (sim dt 1/120, decimation 2) |
| Episode | 360 steps (6 s). The pen falls at 0.30 m from the in-hand position |
| Per-step predicate | Centre distance < 0.10/0.10/0.08 m and `|spin_rate|` > 1/3/4 rad/s (Stage 1/2/3). `spin_rate` is **axial roll** about the pen's long axis, not end-over-end twirl |
| **Episode success ("sustained", D-034)** | The episode times out without a fall **and** the predicate holds on all of the final 60 steps. Any-hit success was shown to be degenerate |
| Execution | `clamp(mean, -1, 1)`, not `tanh`. Policies run on the CPU |
| Process rule | **One rollout per process (D-034).** A second rollout in one Isaac process is not reproducible, and a case's outcome depends on its batch (D-035, D-037). So every protocol scores one whole manifest per fresh process. The first rollout of a fixed batch *is* bit-identical across processes. That gives closed-loop prefix branching (D-045) and makes the GIFs in §8 exact replays |
| Interpreters | The orchestrator and LLM calls run in `.venv-metaworld` (it has openai). Isaac runs under `~/IsaacLab/source_env.sh` (Python 3.11). `pipeline_backend.IsaacPenSpinStream` bridges them: each manifest becomes a `pipeline_child` subprocess |

### 6.2 Experts (D-035 to D-043)

- The scripted feedback gait (`scripted_expert.FeedbackGait`) does **not** qualify (D-035).
- **PPO expert** `logs/pen_spin/r040_ppo_expert/train/model_8239.pt` qualifies on all stages: 29/30, 30/30, 30/30
  sustained (D-036). It is the only competent controller.
- It needs closed-loop feedback: a replay of its own actions scores 0/40 (D-037). It needs **both** pen-state and
  hand-state feedback: clamping either gives 0/40 or 1/40 (D-040). A 67 ms delay of either is fatal: 40/40 → 0/40
  (D-042).
- It is **bang-bang**: 99.98% of executed components are exactly ±1. Local finite differences of it are zero almost
  everywhere, so its labels carry little gradient information (D-043).

### 6.3 LLM search and repair on pen spin (D-038 to D-047)

| Attempt | Result |
|---|---|
| Keyframe trace repair at the plateau (event vs uniform vs none) | 0.000 [0, 0]. All 12 arms kept the null incumbent (D-038) |
| Dense fitting to 4308 expert state-action pairs | MSE cut 2-4×, never a probe success (D-039). Imitation loss again does not track success |
| Gait-scaffold continuation (177-d) | The trigger fired 1/4. **The cold 157-d prefix itself promoted in block 3 at +0.333 [0.15, 0.51] → the "R-046 learner"** (D-041) |
| Two-channel cold prompt (separate retention and spin), Stages 1/2, 4 slots | 0.000 [0, 0]. Candidates keep the pen (13% drops) but spin slowly or off-centre (D-044) |
| Expert takeover from learner states | +0.031 [−0.18, +0.25]. Rescued 6 cases, broke 5 of 7 successes (D-045) |
| Reversing the learner's axial sign | −0.031. The learner's spin drive is saturated 94.6% of the time, so it is open loop in spin (D-046) |
| Self-trace vs expert-trace repair of the learner | 0.000. Expert traces break the grasp (0/320 screen successes); self traces keep some (27/320) but do not improve on the parent (D-047) |

**Conclusion:** expert evidence in any form (keyframes, dense labels, traces, takeover) has not helped the LLM on this
task. Expert evidence is what solved assembly. The expert's basin does not cover the learner's grasp, and its labels
are saturated.

### 6.4 Hand-tuning the one learner we have (D-048 to D-063)

All of these started from the frozen **R-046 block-3 learner** (`config/pen_spin/r046_block3_incumbent/`, about
0.22-0.33 sustained):

| Step | Result | Decision |
|---|---|---|
| CMA-ES vs random search on 12 scalars (12,000 episodes) | +0.003; neither beats the parent | D-048 (refuted) |
| Re-rank the archive's top candidates on bigger screens | −0.002. Ten-case screen winners were selection noise | D-049 |
| Lesion the periodic stroke | −0.088, but the pen still spins at the same mean rate | D-050 |
| Lesion panel: **zero the reactive capture curl** | Drops 13 → 5, off-centre 18 → 3 | D-051 |
| Confirm the zero-curl clamp on fresh cases | **+0.150 [+0.052, +0.248]** | D-052 |
| Constant preload / low-pass curl instead | 0.000 / −0.094 | D-054, D-055 |
| 30-pair promotion panels | Four landed 0.0004-0.02 short of the strict rule | D-052, D-054-D-056 |
| **90-pair instrument** (three 30-pair panels, one paired interval) | Clamp qualifies: **+0.167 [+0.040, +0.293]**, drops 4 vs 10 | D-057 |
| Spin feedback on stroke amplitude vs **constant gain 1.408** | Feedback −0.163. The constant gain beats the clamp: +0.133 promotion, +0.150 fresh | D-058 |
| **Adopt the fixed-stroke controller as the incumbent** | Frozen with hashes | D-059 |
| LLM residuals in anonymous coordinates (R-043 branch) | −0.025. Residuals used about a tenth of their authority | D-060 |
| Phase-lead feedback / constant lead offsets | Flat across 12 settings. Signed-rate targets are ill-posed (the roll direction differs by case) | D-061 |
| Within-cycle phase warp of the stroke | +0.0875 [+0.015, +0.160] fresh, then **−0.022 [−0.105, +0.061]** on the 90-pair instrument | D-062, D-063 |

### 6.5 The current pen-spin incumbent

- **Asset:** `config/pen_spin/d058_fixed_stroke_incumbent/`. It holds the parent source and state, the evaluated
  substituted source, `constant_gain.json` (1.4081491479357957) and `incumbent.json` with sha256 hashes. Verify with
  `python scripts/isaac/adopt_fixed_stroke.py`.
- **What it is:** the LLM-written R-046 learner, then the capture curl clamped to 0, then the periodic stroke
  (`rolling * gait`) scaled by 1.408. It is open loop in spin: the stroke phase is locked to the measured roll angle
  and the drive is saturated.
- **Performance (Stage 3, sustained):** 35/80 fresh in R-081 and 16/30 on promotion, about **0.35-0.44**, with few drops
  (about 4/80).
- **Failure mode:** **slow spin**. Most failures keep the pen centred in hand for the full 6 s but spin below 4 rad/s.
  The GIFs in §8 show it.

### 6.6 Why we are stuck

1. **Cold LLM search rarely reaches a non-zero base rate on pen spin.** It reached one in 1 of 8 Stage-1/2 blocks at
   4 candidates and in 3 of 4 Stage-3 blocks at 24 candidates (D-041, D-044). Candidate budget is the untested driver.
2. **Expert evidence does not transfer.** The PPO expert is a narrow-basin, bang-bang, feedback-critical controller.
   Its traces break the learner's grasp (D-045, D-047).
3. **The one learner is an open-loop stroke.** Every feedback mechanism added to it (amplitude, phase lead, residuals,
   warp) failed to qualify. Only static changes moved it: curl removal and a stroke gain.
4. **The measurement floor.** At a 0.3-0.4 base rate, 30-pair panels cannot resolve effects of 0.1-0.2. The 90-pair
   instrument can, at 180 episodes per test.
5. The research loop has **no open idea and no direction**. 19 deferred ideas wait (R-043, R-052, R-055 to R-088). Most
   are conditional on a mechanism that did not appear.

---

## 7. Verified vs. not

| Claim | Status |
|---|---|
| Manifest-driven evaluation is reproducible (MetaWorld; pen spin: first rollout per process, bit-identical) | **Verified**. Pen spin re-verified 2026-11-09 by the GIF script: 80/80 outcomes and every spin trace identical to R-081 |
| Strict promotion rule, isolation checks, final-test lock, seed ledger | **Verified** by tests and in use |
| Single-phase + search solves supported MetaWorld tasks except assembly | **Verified** (D-014-D-019, D-031) |
| Plateau-triggered expert-trace repair solves assembly-v2 | **Verified** on one task, 2 blocks (D-033) |
| It generalises to tasks other than assembly | **Untested**. No second recoverable plateau was found on MetaWorld; pen spin failed |
| LLM repair / VLM evidence / imitation fitting improve search | **Refuted** repeatedly |
| Pen-spin incumbent ≈ 0.35-0.44 Stage-3 sustained | **Verified** on frozen panels |
| A larger cold candidate budget raises the pen-spin base rate | **Open** (suggested by D-041 vs D-044, never tested directly) |
| Final-test performance of anything | **Never measured**. The split is locked |

---

## 8. Looking at the pen-spin incumbent

`scripts/isaac/record_pen_spin_gif.py` replays a frozen manifest with the viewport camera following chosen
environments. It overlays the step, axial spin rate, centre distance, per-step success and a pen-roll dial (a uniform
capsule rolling about its own axis looks still on camera). The pen is tinted orange; this is visual only. With
`--archived` it checks that the rendered rollout reproduces the protocol's archived cell.

```bash
gpu run 'bash -lc "source ~/IsaacLab/source_env.sh && PYTHONPATH=/home/ubuntu/np126_overlay \
  python scripts/isaac/record_pen_spin_gif.py --controller incumbent \
  --manifest logs/pen_spin/r081_stroke_feedback/fresh.yaml \
  --archived logs/pen_spin/r081_stroke_feedback/constant_gain_fresh.json \
  --env-index 0 1 2 --out-dir logs/pen_spin/gifs/incumbent"'
```

**Rendered on 2026-11-09** (`logs/pen_spin/gifs/`, gitignored). These are R-081 fresh cases 40-42, the incumbent's
exact evaluated batch. The rendered batch matched the archive on 80/80 outcomes, with identical spin traces. The
overview is `overview_2x2.gif`:

| Case | Incumbent | PPO expert (same case, same batch) |
|---|---|---|
| r081_block-0040 | **Sustained success**, mean \|spin\| 6.3 rad/s | Success, 16.8 rad/s |
| r081_block-0041 | **Slow spin**: pen held and centred for 6 s, mean \|spin\| 0.9 rad/s | Success, 17.5 rad/s |
| r081_block-0042 | **Dropped** at step 43 after a 20 rad/s burst | Success, 17.3 rad/s |

The expert spins about three times faster than the incumbent and centres tighter (0.026 m vs 0.061 m). The
incumbent's typical failure is the 0041 kind: a stable grasp with too little roll.

**Rendering needs numpy 1.x.** The box's Isaac venv has numpy 2.4.3. Isaac Sim 5.1's OmniGraph bindings cannot read
numpy 2 arrays, so every camera annotator fails with "Unable to write from unknown dtype". The script refuses to run
under numpy 2. A numpy 1.26.4 overlay was installed outside the venv (`pip install --target ~/np126_overlay`) and is
put first on `PYTHONPATH` for this one process. The research venv was not changed. Why the venv has numpy 2 (pycma?)
is not known. Protocols never render, so their results are unaffected.

---

## 9. Infrastructure quick reference

- **GPU box:** EC2 g4dn.2xlarge (T4), driven only through `gpu` (`scripts/gpu.sh`): `start`, `test`, `run`, `get`,
  `connect`, `stop`. It bills hourly, so stop it at the end of a session. Long runs: launch with `run_in_background`;
  the launch command's completion means the run finished.
- **Local:** conda env `lares-local` runs `research.py` and the mock tests.
- **Tests:** `python -m unittest tests.test_pen_spin tests.test_pen_spin_pipeline tests.test_research ...` (full list in
  `CLAUDE.md`), plus the self-running `tests/test_phase1_phase2.py` and `tests/test_training_pipeline.py`.
- **Adding a pen-spin study:** one module in `lares/envs/isaac_pen_spin/` that exposes `run(protocol_cfg, config_path,
  out_dir)` and `make_controller(...)`, plus a registry entry. Isaac-only studies go in the
  `run_shadowhand_spin_stages.PEN_SPIN_STUDIES` dict (name → module, imported lazily), LLM studies in
  `run_full_evolution.PEN_SPIN_STUDIES`. The shared prologue lives in `protocols.py` (`roles`, `manifest_path`,
  `build_single_manifest`, `cached_json`, `freeze_json`, `study_cell`, `isaac_shell`, `load_cell`), `cases.py`
  (`load_frozen`, `manifest_record`, `freeze_roles`) and, for LLM block studies, `block_study.py`
  (`make_slot_generator`, `screen_best`); `run_study` puts the `--study` name in `protocol_cfg["study"]`, so a module
  never names itself.

---

## 10. Documentation map

| File | Role |
|---|---|
| `CLAUDE.md` | Operating manual (commands, contracts, gotchas). Current |
| `docs/PIPELINE_STATE.md` | This file |
| `docs/SESSION_2026-10-27_PEN_SPIN_SUMMARY.md`, `docs/SESSION_2026-11-04_PEN_SPIN_SUMMARY.md` | Pen-spin session narratives. Current |
| `RESEARCH_HANDOFF.md` | Append-only decision narrative (read by `research.py`; stays at the root) |
| `research/STATE.md` | Generated by `research.py state`. Never edited by hand |
| `README.md` | Install and usage. Its overview is outdated and points here |
| `deprecated/README.md` | Index of archived code (scripts, modules, tests) and why each moved |
| `docs/deprecated/spec.md` | Original requirements plan (FR-1 to FR-11, AC-0 to AC-7, Phases 0-7). Still the source of FR/AC labels in docstrings |
| `docs/deprecated/debug_spec.md` | Debug-cycle spec (Stages A-F, D1-D11). Still cited by docstrings and tests |
| `docs/deprecated/IMPLEMENTATION_PROGRESS.md`, `DEBUG_PROGRESS.md` | Progress logs, frozen 2026-10-13 |
| `docs/deprecated/TRAINING_PIPELINE.md`, `SYMBOLIC_POLICY_IMPL.md`, `run_full_evolution_overview.md`, `Codespace_explanation.md` | September descriptions of the BC + GRPO pipeline. Superseded |
| `docs/deprecated/PIPELINE_REFINEMENT_SPEC.md`, `INSTRUMENTATION_PLAN.md` | September proposals, implemented or superseded |
| `docs/deprecated/summary.md` | Local, git-excluded snapshot from about 2026-10-13 |
