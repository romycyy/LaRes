# LaRes: LLM-Based Symbolic Policy Evolution — Code Overview

## Purpose

`scripts/run_full_evolution.py` is the top-level entry point for the LaRes project.
It searches for the best **symbolic robotic-arm policy** for a MetaWorld manipulation task
by using an LLM to propose candidate policy _structures_, then training each candidate
with a BC → RL inner loop, and iterating the process across multiple evolutionary generations.

---

## High-Level Architecture

```
run_full_evolution.py
│
├── Stage 1  Expert dataset generation     (training_pipeline.py → DemoBuffer)
│
└── Stages 2–4  EvolutionOrchestrator.run()
      │
      ├── [each generation]
      │     ├── llm_evolution()            (policy_generation.py → LLM API)
      │     │     └── get_symbolic_policies()
      │     │           ├── ideation call  (two-phase mode only)
      │     │           └── implementation calls → validate → repair
      │     │
      │     └── [each candidate]
      │           ├── Stage 2  behavioral_cloning()
      │           ├── Stage 3  rl_finetune()
      │           └── evaluate_policy()
      │
      └── best policy saved to log_dir
```

---

## Configuration (`config/run_full_evolution.yaml`)

All runtime behaviour is controlled by a single YAML file.  The script validates it on load
and exposes every key as a `SimpleNamespace` attribute.

| Key | Default | Purpose |
|-----|---------|---------|
| `env_name` | `push-v2` | MetaWorld task |
| `use_mt1` | `true` | Build the env via the Farama MT1 benchmark (full observability, valid `rand_vec`s) instead of plain construction |
| `seed` | `42` | NumPy + PyTorch seed |
| `dataset_episodes` | `150` | Expert episodes to collect |
| `bc_steps` | `2000` | Gradient steps for BC |
| `rl_iterations` | `0` | GRPO outer iterations per candidate |
| `rl_episodes` | `30` | Rollout episodes per RL iteration |
| `eval_episodes` | `10` | Episodes used to score each candidate |
| `model` | `gpt-5` | OpenAI model for policy generation |
| `policy_gen_two_phase` | `true` | Use ideation + implementation flow |
| `policy_impl_mode` | `batched` | `batched` or `per_idea` implementation |
| `num_generations` | `3` | Evolution generations |
| `pop_size` | `5` | Candidates per generation |
| `elite_num` | `2` | Top-k fed back as context |
| `log_dir` | `./logs/evolution` | Artefact output directory |
| `record_demo_gif` | `true` | Save GIF of best per generation |
| `demo_buffer_path` | _(unset)_ | Path to a cached demo buffer (skips Stage 1) |

---

## Stage 1 — Expert Dataset Generation

**File:** `lares/core/training_pipeline.py` — `generate_dataset()`

- Loads the MetaWorld built-in expert policy for the task via `EXPERT_POLICY_MAP`
  (e.g. `push-v2` → `SawyerPushV3Policy`).
- Runs one rollout per training-manifest case (up to 150 steps each), storing every
  `(obs, action, reward, next_obs, done, episode_id)` tuple into a `DemoBuffer`.
- `DemoBuffer` is a plain Python list-backed structure with `add()`, `sample(batch_size)`,
  `save(path)` / `load(path)` (pickle).
- The buffer is saved to `log_dir/demo_<env_name>.pkl` and can be reloaded in future runs
  via `demo_buffer_path` to skip this stage entirely.

---

## Stage 2 — Behavioral Cloning (BC)

**File:** `lares/core/training_pipeline.py` — `behavioral_cloning()`

- Minimises `MSE(tanh(mean), expert_action) + 0.01 * std.mean()`.
- The policy outputs `(mean, std)` as pre-tanh Gaussian parameters, so the loss compares the
  squashed mean against the expert action; the std penalty pushes toward near-deterministic
  behaviour. (An earlier NLL-on-inverse-tanh formulation with a Jacobian correction was tried
  and reverted — see `docs/PIPELINE_REFINEMENT_SPEC.md`.)
- Uses Adam, gradient clipping, and calls `policy.clip_params()` after each step to enforce
  the parameter bounds declared in `get_param_ranges()`.
- Logs BC metrics (`bc/train_loss`, grad norms, etc.) to `TrainingLogger`.

---

## Stage 3 — RL Fine-Tuning (GRPO)

**File:** `lares/core/training_pipeline.py` — `rl_finetune()`

- Implements **GRPO** (Group Relative Policy Optimisation): advantages are computed relative
  to the group of trajectories collected in the same iteration — no value function.
- Each iteration:
  1. Collect `episodes_per_iter` on-policy trajectories with the current policy.
  2. Normalise returns to advantages: `A = (R - mean(R)) / std(R)`.
  3. Compute policy gradient loss weighted by advantages.
  4. Add entropy bonus (encourages exploration) and L2 KL penalty toward BC initialisation
     (prevents catastrophic forgetting).
  5. Clip gradients, apply Adam step, `clip_params()`.
- Checkpoints the best parameters by success rate and restores them at the end.
- RL can be disabled entirely by setting `rl_iterations: 0` in the config.

---

## Stage 4 — LLM Structure Search

### Symbolic Policy Contract

**File:** `lares/core/symbolic_policy.py`

`SymbolicPolicy` is the abstract base class every LLM-generated policy must subclass.

- Implements `nn.Module`.
- `forward(obs) -> (mean, std)`: maps a batch of observations to a Gaussian distribution.
- `get_param_ranges() -> dict`: declares `{param_name: (lo, hi)}` bounds for `clip_params()`.
- **Forbidden modules** (enforced by `validate()`): any `nn.Linear`, `Conv*`, `LSTM`, `GRU`,
  `Transformer`, etc.  Policies must use **explicit symbolic expressions** only
  (arithmetic on observation slices, `nn.Parameter` scalars, `torch.tanh/sigmoid`, etc.).

### Policy Generation Pipeline

**File:** `lares/core/policy_generation.py` — `get_symbolic_policies()`

Two modes, selected by `policy_gen_two_phase`:

#### Single-shot mode (`policy_gen_two_phase: false`)
1. Build a prompt from `initial_system.txt` + `new_initial_user.txt` + task/obs description.
2. Call the LLM with `n=pop_size*2` completions.
3. For each response: extract the `GeneratedPolicy` class with regex, validate via subprocess,
   repair up to `max_repair_per_candidate` times if validation fails (error fed back to LLM).

#### Two-phase mode (`policy_gen_two_phase: true`)
1. **Ideation call** (`ideas_system.txt` / `ideas_user.txt`): ask the LLM for `n` JSON design
   hypotheses (no code), optionally seeded with previous-generation feedback.
2. **Implementation call(s)**:
   - `batched`: one LLM response containing `n` fenced `GeneratedPolicy` classes.
   - `per_idea`: one LLM call per hypothesis.
3. Each code block is validated (subprocess) and repaired as needed.

### Validation

- Generated code + a fixed test harness (`tests/test_generate_policy.py`) is written to a
  temp file and executed as a subprocess.
- Validation passes when `"Success!"` appears in stdout.
- On failure, the error output and failing code are fed back to the LLM as a repair prompt
  (up to `max_repair_per_candidate` attempts, with a hard cap of `max_total_attempts=50`
  total LLM calls per generation).
- All LLM exchanges are appended to `log_dir/llm_evolution_transcript.log`.

### Evolution Loop — `EvolutionOrchestrator`

**File:** `lares/core/training_pipeline.py`

```
for gen in range(num_generations):
    policy_pop = llm_evolution(...)          # LLM proposes pop_size candidates
    for candidate in policy_pop:
        behavioral_cloning(candidate, ...)   # Stage 2
        rl_finetune(candidate, ...)          # Stage 3
        score = evaluate_policy(candidate, dev_env, dev_manifest)   # same cases for all candidates
    sort candidates by score
    update global best
    record GIF of best candidate
    feed top-elite_num results as context to next generation
```

Fitness score: `success_rate × 1000 + mean_reward` — success rate dominates, reward breaks ties.

---

## Logging — `TrainingLogger`

**File:** `lares/core/training_logger.py`

- Writes append-only **JSONL** to `log_dir/<run_id>.jsonl`.
- Every record contains: `stage`, `task_name`, `global_step`, `update`, `wall_clock_time`,
  `metric_name`, `metric_value`.
- Optionally mirrors to **Weights & Biases** or **TensorBoard** if a run/writer is injected.
- Canonical metric names are module-level constants (`BC_TRAIN_LOSS`, `RL_POLICY_LOSS`,
  `EVO_FITNESS_BEST`, etc.) shared between the pipeline and any downstream plotting code.

---

## Output Artefacts

| Path | Content |
|------|---------|
| `log_dir/demo_<env>.pkl` | Expert demo buffer (Stage 1) |
| `log_dir/<run_id>.jsonl` | Full structured metric log |
| `log_dir/llm_evolution_transcript.log` | All LLM request/response exchanges |
| `log_dir/gen_<N>/` | Per-generation directory |
| `log_dir/gen_<N>/results.pkl` | Candidate codes, evals, scores for generation N |
| `log_dir/gen_<N>/gen_<N>_best.gif` | Demo GIF of best candidate in generation N |
| `log_dir/gen_<N>/Iter_<N>_ideas.json` | Ideation response (two-phase mode) |
| `log_dir/gen_<N>/Iter_<N>_Policy_Code_<k>.py` | Validated generated policy source |
| `log_dir/best_policy_code.py` | Best policy class source across all generations |
| `log_dir/best_policy.pt` | Best policy `state_dict` |

---

## Key Data Flow Summary

```
YAML config
    │
    ▼
MetaWorld env (MuJoCo / MetaWorld v2 + Farama wrapper)
    │
    ├──► expert policy ──► DemoBuffer (transitions)
    │                              │
    │                              ▼
    │                      behavioral_cloning()
    │                              │
    │                              ▼
    │                      rl_finetune()  ◄── on-policy rollouts
    │                              │
    │                              ▼
    │                      evaluate_policy() ──► score
    │                              │
    └──► LLM (OpenAI API) ◄── elite scores & code
              │
              ▼
         new candidate policies (SymbolicPolicy subclasses)
```
