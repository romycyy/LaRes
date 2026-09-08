LaRes Codebase Overview

This repository implements **LLM-based symbolic policy evolution** for MetaWorld robotic manipulation tasks. The LLM proposes learnable symbolic policy structures; each candidate is trained with BC and GRPO-style RL, evaluated, and evolved across generations.

## Core Entry Points

Two entry points exist, one per environment backend. They share `lares/core/` but each carries its
own stage orchestration.

**`scripts/run_full_evolution.py`** — MetaWorld, the primary pipeline

- Loads YAML config from `config/run_full_evolution.yaml`
- Creates a MetaWorld environment via `lares.utils.metaworld_env`
- Collects or loads an expert demo buffer (Stage 1)
- Runs `EvolutionOrchestrator` for LLM structure search with BC+RL inner loop (Stages 2–4)
- Saves best policy code and weights under `log_dir`

See [`docs/run_full_evolution_overview.md`](run_full_evolution_overview.md) for the full pipeline diagram.

**`scripts/run_shadowhand_spin_stages.py`** — Isaac Lab `ShadowHandSpin` (needs CUDA + Isaac Lab)

- Loads YAML config from `config/shadowhand_spin_stages.yaml`
- Builds the env via `lares.envs.isaac_lab_adapter`
- Runs Stage 1 (scripted-demo collection) → Stage 2 (BC) → Stage 3 (RL) against one env, writing a
  timestamped run dir with `config.yaml`, `summary.json`, `training_dynamics.jsonl`, the demo
  pickle and `*_policy.pt`
- Note: it reimplements its own GIF recorder, dataset collector and action-squashing helper rather
  than calling the `training_pipeline` equivalents — keep the two in sync

Design notes and open issues: [`docs/PIPELINE_REFINEMENT_SPEC.md`](PIPELINE_REFINEMENT_SPEC.md).

## Core Library

### `lares/core/training_pipeline.py`

| Component | Role |
|-----------|------|
| `DemoBuffer` | Expert demonstration storage |
| `generate_dataset()` | Stage 1: collect expert trajectories |
| `behavioral_cloning()` | Stage 2: supervised imitation |
| `rl_finetune()` | Stage 3: GRPO-style policy gradient |
| `llm_evolution()` | Stage 4: LLM population generation |
| `EvolutionOrchestrator` | Orchestrates multi-generation evolution |
| `record_episode_gif()` | Demo GIF recording |

### `lares/core/policy_generation.py`

LLM policy generation, subprocess validation, two-phase ideation/implementation.

### `lares/core/symbolic_policy.py`

Base class for LLM-generated policies (`forward` → `(mean, std)`, parameter bounds).

### `lares/core/training_logger.py`

JSONL metrics for BC, RL, and evolution stages.

## Environment Setup

Configuration lives in `config/`: `run_full_evolution.yaml` (MetaWorld),
`shadowhand_spin_stages.yaml` (Isaac Lab), and `environment.yaml` (conda spec; the root
`environment.yml` is a separate, diverged spec used on the Isaac Lab box).

### `lares/utils/metaworld_env.py`

- `make_metaworld_env(cfg, seed)` — creates MetaWorld V3 env with `NormalizedBoxEnv` + `TimeLimit`
- `env_wrapper` — episode length cap, MT1 task sampling on reset

### `lares/envs/isaac_lab_adapter.py`

- `launch_isaac_app(...)` — one-shot Isaac SimulationApp launcher (process-global; a second launch
  with different options raises)
- `IsaacLabSingleEnvAdapter` — exposes a batched Isaac vector env as a single env with the same
  contract the MetaWorld wrapper provides: `reset() -> (obs, info)`, 4-tuple `step()`,
  `info["success"]`, `Box(-1, 1)` action space. Rows beyond the first are simulated and discarded.

### `lares/envs/rlkit/`

- `NormalizedBoxEnv` — action normalization to `[-1, 1]`
- `ProxyEnv` — transparent gym proxy base

## LLM Prompts

`lares/utils/policy_prompts/` — six text templates loaded by `load_policy_prompt_assets()`.

## Validation

**`tests/test_generate_policy.py`** — appended to generated policy temp files and run as a subprocess to validate instantiation and forward pass.

## Companion Scripts

| Script | Purpose |
|--------|---------|
| `scripts/plot_training_dynamics.py` | Plot metrics from `TrainingLogger` JSONL |
| `scripts/visualize_expert_policy.py` | Expert-policy GIF rollouts |
| `scripts/run_shadowhand_spin_stages.py` | Isaac Lab ShadowHandSpin 3-stage run |

## Tests

| File | Scope |
|------|-------|
| `tests/test_training_pipeline.py` | Mock / MetaWorld / LLM pipeline checks |
| `tests/test_generate_policy.py` | Policy validation harness |
| `tests/test_two_phase_policy_generation.py` | Two-phase LLM generation |
| `tests/test_phase1_phase2.py` | Policy generation phases |
| `tests/test_record_episode_gif.py` | GIF recording |
| `tests/test_env_action_response.py` | Env action scaling |

## Pipeline Flow

1. **Setup:** Parse config, seed RNGs, create env, evaluate MetaWorld expert baseline.
2. **Stage 1:** Collect expert demos into `DemoBuffer` (or load cached pickle).
3. **Stages 2–4:** For each generation, LLM proposes `pop_size` policies → BC → RL → evaluate → keep elites.
4. **Output:** Best policy code (`.py`) and weights (`.pt`) saved to `log_dir`.
