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
python -m unittest tests.test_record_episode_gif.TestRecordEpisodeGif.test_saves_gif_and_returns_metadata
```

`tests/test_training_pipeline.py` self-tiers: Tier 1 mocks always; Tier 2 needs `metaworld`; Tier 3
needs `OPENAI_API_KEY`. It imports `EvolutionOrchestrator`, the only orchestrator in the tree.

`tests/test_generate_policy.py` is **not** a test suite — it is a validation harness *appended to
LLM-generated source* and run as a subprocess by the generation pipeline (`policy_generation.py`).
Never run it directly; changing it changes what the LLM's output must satisfy. Its `_FORBIDDEN`
tuple partly duplicates `SymbolicPolicy.FORBIDDEN_MODULES` on purpose (the harness runs against
untrusted source) — the two lists have drifted, so update both when tightening either.

## Architecture

The original LaRes reward-search pipeline (SAC + LLM reward evolution: `scripts/LaRes_*.py`,
`lares/rl/`, `lares/utils/utils.py`, the `prompts/`/`no_init_prompts/` template dirs) was **deleted**
in the codespace cleanup. Only symbolic-policy search remains. Look for it in git history, not on disk.

**Symbolic policy search.** `scripts/run_full_evolution.py` →
`lares/core/training_pipeline.py`. The LLM writes *policies*, not rewards. Four stages:

| Stage | Function | What it does |
|---|---|---|
| 1 | `generate_dataset()` | MetaWorld expert policy (`EXPERT_POLICY_MAP`) → `DemoBuffer` |
| 2 | `behavioral_cloning()` | `MSE(tanh(mean), expert action) + 0.01 * std.mean()`; `clip_params()` after each step |
| 3 | `rl_finetune()` | GRPO: advantages relative to the *group* mean return, + entropy bonus, + L2 KL toward the BC init |
| 4 | `EvolutionOrchestrator.run()` / `llm_evolution()` | LLM proposes structures; each runs BC→RL→eval; top-`elite_num` fed back as feedback |

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
- contain **no** NN modules (`Linear`/`Conv`/`LSTM`/…) — `validate()` rejects them
- declare `get_param_ranges() -> {param_name: (lo, hi)}` for every parameter; training calls
  `clip_params()` after every gradient step

Generation flow in `get_symbolic_policies()`: format prompts → LLM call → regex-extract code block →
find `class GeneratedPolicy` → write temp file (`imports + generated_code + tests/test_generate_policy.py`)
→ run as subprocess and require `"Success!"` → only then `exec()` in-process and `validate()`.
Failures are fed back to the LLM as repair feedback. Two-phase mode (`policy_gen_two_phase: true`)
first asks for JSON design ideas (`ideas_*.txt`), then implementations (`policy_impl_mode:
batched | per_idea`).

## Things that will bite you

- **MetaWorld 3.0 is installed, but all task names in this codebase are `-v2`.**
  `make_metaworld_env()` rewrites `-v2` → `-v3` and looks up `ALL_V3_ENVIRONMENTS`. Keep using the
  `-v2` spelling in configs and dicts; do not "fix" it to `-v3`.
- **`use_mt1: true`** (in `run_full_evolution.yaml`) switches env construction to the Farama MT1
  benchmark so `push-v3` gets full observability and valid `rand_vec`s. `env_wrapper.reset()` then
  resamples a task from `mt1_train_tasks` each episode. Plain construction sets `_freeze_rand_vec=False`
  instead. These two paths behave differently — check which one a bug is on.
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
- **Rollout logic is duplicated five ways** — `generate_dataset`, `_collect_trajectories`,
  `evaluate_policy` and `record_episode_gif` in `training_pipeline.py`, plus
  `run_full_evolution.evaluate_expert_policy`. They already disagree about action squashing
  (`run_shadowhand_spin_stages._policy_action` omits the `action_space.high *` scaling the others
  apply). Change one, check all five.
- **`best_policy_code.py`, `best_policy.pt`, `data.pkl`** are *runtime outputs* written into
  `log_dir`, not repo files. `logs/` and `*.pkl` are gitignored.
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
