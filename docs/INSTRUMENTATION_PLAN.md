# Training Dynamics Instrumentation & Visualization Plan

This document maps the codebase for adding per-stage loss logging, RL gradient norm logging, and multi-task plotting across MetaWorld tasks.

**Run from project root:** `python scripts/run_demo.py`, `python scripts/plot_training_dynamics.py --log-path ./logs/...`

---

## 1. Map of Relevant Files / Functions / Classes

### Core Training Pipeline (`lares/core/training_pipeline.py`)

| Location | Type | Role |
|----------|------|------|
| `EXPERT_POLICY_MAP` | dict | Maps env_name → MetaWorld expert policy class (25+ tasks) |
| `TASK_DESCRIPTIONS` | dict | Human-readable task descriptions |
| `generate_dataset()` | fn | **Stage 1**: Expert demo collection |
| `behavioral_cloning()` | fn | **Stage 2**: Supervised imitation (BC) |
| `rl_finetune()` | fn | **Stage 3**: GRPO-style RL fine-tuning |
| `_collect_trajectories()` | fn | Helper for Stage 3 trajectory collection |
| `_compute_grpo_advantages()` | fn | Helper for Stage 3 advantage computation |
| `evaluate_policy()` | fn | Evaluation helper (used after BC, RL, and in Stage 4) |
| `llm_evolution()` | fn | **Stage 4**: LLM structure search with BC+RL inner loop |
| `SymbolicPolicyPipeline` | class | Orchestrator for all 4 stages |
| `DemoBuffer` | class | Stage 1 output; Stage 2 input |

### Supporting Files

| File | Role |
|------|------|
| `lares/core/policy_generation.py` | `get_symbolic_policies()`, `obs_description_dict`, `input_dict_for_policy` — used by Stage 4 |
| `lares/core/symbolic_policy.py` | `SymbolicPolicy` base class, `clip_params()` |
| `lares/utils/utils.py` | `make_metaworld_env()`, `env_wrapper` — env creation |
| `scripts/run_demo.py` | Single-task demo (reach-v2); calls stages 1–4 directly |

---

## 2. Stage Boundaries

| Stage | Start | End | Main loop / entry point |
|-------|-------|-----|-------------------------|
| **1** | Line 184: `generate_dataset()` first `for ep in range(num_episodes)` | Line 233: `return buffer, summary` | `for ep in range(num_episodes)` → inner `for step in range(max_steps)` |
| **2** | Line 272: `for step in range(num_steps)` | Line 306: `return stats` | `behavioral_cloning()` step loop |
| **3** | Line 420: `for iteration in range(num_iterations)` | Line 387: `return stats` | `rl_finetune()` iteration loop |
| **4** | Line 461: `for gen in range(num_generations)` | Line 523: `return best_overall` | Outer gen loop; inner per-candidate BC+RL at 651–668 |

---

## 3. Loss Computation Points

| Stage | Location | Loss | Variable / Key |
|-------|----------|------|----------------|
| **2 (BC)** | 279–281 | `mean_loss = MSE(mean, actions_t)`, `std_loss = 0.01 * std.mean()`, `loss = mean_loss + std_loss` | `stats["bc_loss"]`, `stats["mean_loss"]`, `stats["std_loss"]` |
| **3 (RL)** | 456, 350, 353–356, 360 | `policy_loss`, `entropy`, `kl_loss`, `total_loss` | `stats["policy_loss"]`, `stats["entropy"]`, `stats["kl"]` |

Stage 1 has no trainable loss; it collects episode_rewards, episode_successes, episode_lengths.

---

## 4. Optimizer Step Points

| Stage | Location | Context |
|-------|----------|---------|
| **2** | Line 287 | `optimizer.step()` after `loss.backward()` and optional gradient clipping |
| **3** | Line 471 | `optimizer.step()` after `total_loss.backward()` and optional gradient clipping |

---

## 5. Gradient Clipping Points (RL Stage)

| Stage | Location | Code |
|-------|----------|------|
| **2** | Lines 285–286 | `if clip_grad_norm > 0: torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_grad_norm)` |
| **3** | Lines 469–470 | Same pattern |

**Note:** `torch.nn.utils.clip_grad_norm_()` returns the total gradient norm **before** clipping. Capture it with:
```python
grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_grad_norm)
```

---

## 6. MetaWorld Task Specification and Looping

### Current behavior
- **Single-task only**: Each run uses one `env_name` (e.g. `reach-v2`, `window-close-v2`).
- `run_demo.py`: hardcodes `ENV_NAME = "reach-v2"`.
- `training_pipeline.py` CLI: `--env-name` defaults to `window-close-v2`.
- No multi-task loop exists; each call to `generate_dataset`, `behavioral_cloning`, `rl_finetune`, `llm_evolution` operates on one task.

### Task sources
- `EXPERT_POLICY_MAP` keys: 25+ task names with expert policies.
- `obs_description_dict` / `input_dict_for_policy` in `policy_generation.py`: 6 tasks (window-close, window-open, button-press, door-close, drawer-open, reach).
- `run_demo.py` and `SymbolicPolicyPipeline`: pass `env_name` through to env creation and LLM prompts.

### Task difficulty
- MetaWorld docs describe MT1 < MT10 < MT50 but **do not** provide an explicit per-task difficulty ordering.
- Need a **curated difficulty ordering** for plotting (easy → hard). Suggested heuristic based on MT benchmarks and common practice:
  - Easy: `reach-v2`, `push-v2`, `window-close-v2`, `window-open-v2`
  - Medium: `button-press-v2`, `drawer-open-v2`, `door-close-v2`, `door-open-v2`, `lever-pull-v2`, `faucet-open-v2`, `faucet-close-v2`
  - Hard: `pick-place-v2`, `assembly-v2`, `hammer-v2`, `peg-insert-side-v2`, `peg-unplug-side-v2`, etc.

---

## 7. Exact Logging Points to Instrument

### Per-stage loss logging

| Stage | Function | Insert after | What to log |
|-------|----------|--------------|-------------|
| **1** | `generate_dataset` | Line 225 (end of episode loop), Line 234 | `episode_rewards`, `episode_successes`, `episode_lengths`; final `mean_reward`, `mean_success` |
| **2** | `behavioral_cloning` | Line 294 (already in stats) | `bc_loss`, `mean_loss`, `std_loss` per step — **centralize to logger** |
| **3** | `rl_finetune` | Line 373 | `policy_loss`, `entropy`, `kl` per iteration — **centralize to logger** |
| **4** | `llm_evolution` | Line 664, 668 | Per-candidate `bc_stats`, `rl_stats`, `eval` — **aggregate and log** |

### RL gradient norm logging (before clipping)

| Stage | Function | Insert at | Code change |
|-------|----------|-----------|-------------|
| **3** | `rl_finetune` | Line 469 | Replace `torch.nn.utils.clip_grad_norm_(...)` with `grad_norm = torch.nn.utils.clip_grad_norm_(...)` and log `grad_norm` |
| **(optional) 2** | `behavioral_cloning` | Line 286 | Same pattern if BC gradient norms are desired |

### Centralization
- Introduce a `TrainingLogger` (or use `wandb` / `tensorboard`) that:
  - Accepts `stage`, `step`/`iteration`/`episode`, `env_name`, `metrics`.
  - Writes to a structured log dir (e.g. `./logs/training_dynamics/{run_id}/`).
  - Persists as JSON/CSV for offline plotting.

---

## 8. Multi-Task Plotting (Easy → Hard)

### New infrastructure needed

1. **Task ordering**
   - Add `TASK_DIFFICULTY_ORDER` (or similar) in `training_pipeline.py` or a new `task_ordering.py`:
     - List of `(env_name, difficulty_rank)` for tasks in `EXPERT_POLICY_MAP` with known ordering.
   - Document source (benchmark, heuristic, or manual).

2. **Multi-task runner**
   - New script or entry point, e.g. `run_multi_task.py`:
     - Loop over `TASK_DIFFICULTY_ORDER`.
     - For each task: Stage 1 → Stage 2 → Stage 3 (optionally skip Stage 4 for speed).
     - Log all metrics with `env_name` and `difficulty_rank`.
     - Save per-task stats to `./logs/multi_task/{run_id}/{env_name}.json`.

3. **Visualization script**
   - New script, e.g. `plot_training_dynamics.py`:
     - Load logs from `./logs/multi_task/` or `./logs/training_dynamics/`.
     - Produce:
       - Per-stage loss curves (BC: step vs loss; RL: iteration vs policy_loss, entropy, kl).
       - RL gradient norm curves (iteration vs `grad_norm`).
       - Multi-task summary: x-axis = task (ordered easy → hard), y-axis = success_rate / mean_reward, with optional error bars from multiple seeds.

---

## 9. Ambiguities and Missing Infrastructure

### Ambiguities
1. **Task difficulty ordering**: No official MetaWorld easy→hard list. Need to adopt or define a convention.
2. **Stage 4 logging granularity**: Log per-candidate, per-generation, or both? Recommend both for debugging and summary.
3. **Run ID / experiment naming**: No current convention; suggest timestamp + `env_name` or `multi_task`.

### Missing infrastructure
1. **Logger abstraction**: Only `print()` and returned `stats` dicts; no shared logger.
2. **Log directory layout**: `./logs/` exists; need `./logs/training_dynamics/`, `./logs/multi_task/` and conventions.
3. **WandB / TensorBoard**: Not used by `training_pipeline.py`; optional integration for live monitoring.
4. **Multi-task loop**: No script that runs multiple tasks in sequence; must be added.
5. **Plotting code**: No matplotlib/plotly scripts for training dynamics or multi-task comparison.

---

## 10. Implementation Checklist

- [ ] Add `TrainingLogger` (or integrate wandb/tensorboard) with structured logging
- [ ] Instrument Stage 2: log `bc_loss`, `mean_loss`, `std_loss` per step via logger
- [ ] Instrument Stage 3: log `policy_loss`, `entropy`, `kl` per iteration via logger
- [ ] Instrument Stage 3: capture and log `grad_norm` before clipping
- [ ] (Optional) Instrument Stage 2: capture and log BC `grad_norm`
- [ ] Instrument Stage 1: log per-episode and summary stats via logger
- [ ] Instrument Stage 4: log per-candidate and per-generation aggregates
- [ ] Add `TASK_DIFFICULTY_ORDER` for supported tasks
- [ ] Add `run_multi_task.py` to run pipeline across tasks
- [ ] Add `plot_training_dynamics.py` for loss curves and gradient norms
- [ ] Add multi-task comparison plot (task vs success/reward)
