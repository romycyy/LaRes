# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment setup

The working environment is the pre-built venv in the repo, **not** the conda env described in the README:

```bash
cd /home/ubuntu/LaRes
source .venv-metaworld/bin/activate     # Python 3.10, torch 2.7+cu128, metaworld 3.0.0, mujoco 3.8, openai 2.x
export MUJOCO_GL=egl                    # required: headless machine, no DISPLAY
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

# Original LaRes reward-search training (SAC + LLM reward evolution)
python scripts/LaRes_from_scratch.py --env-name='window-close-v2' [many flags — see lares/rl/arguments.py]
python scripts/LaRes_with_init.py    --env-name='coffee-pull-v2'  [...]
bash config/run.sh                    # batch launcher; edit to uncomment the runs you want

# Plots and expert-policy inspection
python scripts/plot_training_dynamics.py --log-path ./logs/evolution/<run>/training_dynamics.jsonl
python scripts/visualize_expert_policy.py --config config/run_full_evolution.yaml --episodes 10
```

### Tests

Two incompatible styles coexist:

```bash
# Self-running scripts (print PASS/FAIL counts, sys.exit(1) on failure) — NOT pytest-collectable
python tests/test_phase1_phase2.py          # 41 checks, no API key needed
python tests/test_training_pipeline.py      # BROKEN: imports SymbolicPolicyPipeline, which no longer exists
python tests/test_env_action_response.py    # needs a working env from config/run_full_evolution.yaml

# unittest style — pytest is NOT installed in .venv-metaworld, use unittest
python -m unittest tests.test_record_episode_gif tests.test_two_phase_policy_generation
python -m unittest tests.test_record_episode_gif.TestRecordEpisodeGif.test_saves_gif_and_returns_metadata
```

`tests/test_training_pipeline.py` self-tiers (Tier 1 mocks always; Tier 2 needs `metaworld`; Tier 3
needs `OPENAI_API_KEY`) but currently dies at import: it expects `SymbolicPolicyPipeline`, and
`training_pipeline.py` only defines `EvolutionOrchestrator`. Fix the import before trusting it.

`tests/test_generate_code.py` and `tests/test_generate_policy.py` are **not** test suites — they are
validation harnesses *appended to LLM-generated source* and run as a subprocess by the generation
pipeline. Never run them directly; changing them changes what the LLM's output must satisfy.

## Two pipelines live in this repo

The repo contains the published LaRes method *and* a newer rewrite; they share almost nothing but
the `lares.utils` env helpers.

**1. Reward search (original NeurIPS paper).** `scripts/LaRes_*.py` → `lares/rl/sac.py`.
An LLM writes a *population of reward functions*; SAC actors train against them from a shared
`replay_buffer` that stores one reward *per population member* per transition, so replacing a reward
function relabels history instead of recollecting it. Thompson sampling picks which policy interacts.
Multi-process via `lares.utils.utils.Worker` (each worker owns a local SAC agent + buffer copy).
Prompts in `lares/utils/prompts/` (with-init) and `lares/utils/no_init_prompts/` (from-scratch).

**2. Symbolic policy search (current work).** `scripts/run_full_evolution.py` →
`lares/core/training_pipeline.py`. The LLM writes *policies*, not rewards. Four stages:

| Stage | Function | What it does |
|---|---|---|
| 1 | `generate_dataset()` | MetaWorld expert policy (`EXPERT_POLICY_MAP`) → `DemoBuffer` |
| 2 | `behavioral_cloning()` | MSE(mean, expert action) + std penalty; `clip_params()` after each step |
| 3 | `rl_finetune()` | GRPO: advantages relative to the *group* mean return, + entropy bonus, + L2 KL toward the BC init |
| 4 | `EvolutionOrchestrator.run()` / `llm_evolution()` | LLM proposes structures; each runs BC→RL→eval; top-`elite_num` fed back as feedback |

Prompts in `lares/utils/policy_prompts/`. Generation lives in `lares/core/policy_generation.py`.

### The `SymbolicPolicy` contract

Everything in pipeline 2 depends on `lares/core/symbolic_policy.py`. A generated policy must:

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
- **`get_dict()`**: reward functions read env state through `env._env.get_dict()`, which MetaWorld V3
  does not provide. `_patch_get_dict()` in `lares/utils/utils.py` attaches a picklable `_GetDict`
  shim. New tasks need their variables exposed there.
- **Rendering**: MuJoCo needs `render_mode="rgb_array"` at *construction* and `MUJOCO_GL` set *before*
  the env is built. Call `ensure_mujoco_headless_gl()` early. `record_episode_gif()` silently logs
  "No frames captured" rather than raising.
- **`config/run_demo.yaml` does not exist**, so `python scripts/run_demo.py` fails on a missing config
  unless you pass `--config`. Prefer `run_full_evolution.py`.
- **`lares/core/training_pipeline.py` has no `__main__`** — the `python lares/core/training_pipeline.py
  --stage all` line in `STRUCTURE_REORGANIZATION.md` is stale, as is that file's `sync.sh`/`rlkit`
  symlink description and parts of the README's project-structure tree.
- **`lares/utils/utils.py` (2.6k lines) defines `_gripper_caging_reward` and `compute_reward` many
  times over** — these are per-task *code strings and templates* for prompts, not a single live API.
  Grep by task name, not by function name.
- Root-level `best_policy_code.py`, `expert_policy.py`, `reward_push_v3.py`, `data.pkl` are scratch
  reference artifacts (a saved LLM policy, a copied MetaWorld expert, a copied reward fn), not imports.

## Logging

`lares/core/training_logger.py` defines every metric key as a constant (`BC_*`, `RL_*`, `EVO_*`) —
use those constants, never string literals, so `plot_training_dynamics.py` keeps working.
`TrainingLogger` appends JSONL to `<log_dir>/<run_id>.jsonl` (`run_id` defaults to a `%Y%m%d_%H%M%S`
timestamp). `run_full_evolution.py` writes flat into `log_dir` (default `./logs/evolution`):
`demo_<env>.pkl`, `best_policy_code.py`, `best_policy.pt`, per-generation LLM responses and code, and
optional demo GIFs under `gen_<N>/`. `run_shadowhand_spin_stages.py` instead creates a timestamped run
dir holding `config.yaml`, `summary.json`, `training_dynamics.jsonl`, the demo pickle, and `*_policy.pt`.
