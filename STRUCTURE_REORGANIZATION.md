# LaRes Codebase Structure Reorganization

This document describes the proposed folder structure so that all files belong to logical folders. **No code or imports have been changed yet** — this is the structural layout only.

## Proposed Structure

```
LaRes/
├── lares/                          # Main Python package
│   ├── __init__.py
│   ├── core/                       # Policy, training, models
│   │   ├── __init__.py
│   │   ├── symbolic_policy.py      # SymbolicPolicy base class
│   │   ├── training_pipeline.py    # 4-stage pipeline (dataset, BC, RL, evolution)
│   │   ├── training_logger.py      # Training metrics logging
│   │   ├── policy_generation.py    # LLM policy generation
│   │   ├── models.py               # NN models (flatten_mlp, tanh_gaussian_actor)
│   │   ├── replay_buffer.py        # Replay buffer for LaRes / SAC
│   │   └── reward_utils.py         # Tolerance, hamacher_product, etc.
│   ├── rl/                         # RL algorithms
│   │   ├── __init__.py
│   │   ├── sac.py                  # SAC agent
│   │   └── arguments.py            # Argument parsing for LaRes
│   ├── envs/                       # Environment wrappers
│   │   └── rlkit/                  # rlkit envs (normalized_box_env, proxy_env)
│   │       ├── __init__.py
│   │       └── envs/
│   │           ├── __init__.py
│   │           ├── proxy_env.py
│   │           └── wrappers/
│   │               ├── __init__.py
│   │               └── normalized_box_env.py
│   └── utils/                      # Utilities (merged with root utils)
│       ├── __init__.py
│       ├── utils.py                # Main utils (env, reward functions, etc.)
│       ├── misc.py
│       ├── file_utils.py
│       ├── extract_task_code.py
│       ├── create_task.py
│       ├── prune_env.py
│       ├── prune_env_dexterity.py
│       ├── prune_env_isaac.py
│       ├── prompts/                # Reward function prompts
│       ├── no_init_prompts/
│       └── policy_prompts/
│
├── scripts/                        # Runnable entry points
│   ├── run_demo.py                 # 4-stage symbolic policy demo
│   ├── plot_training_dynamics.py   # Plot training curves
│   ├── LaRes_from_scratch.py       # LaRes without init
│   └── LaRes_with_init.py          # LaRes with init
│
├── tests/                          # Test files
│   ├── test_phase1_phase2.py
│   ├── test_training_pipeline.py
│   ├── test_generate_policy.py
│   └── test_generate_code.py
│
├── docs/                           # Documentation
│   ├── INSTRUMENTATION_PLAN.md
│   ├── TRAINING_PIPELINE.md
│   ├── SYMBOLIC_POLICY_IMPL.md
│   └── Codespace_explanation.md
│
├── config/                         # Config & shell scripts
│   ├── environment.yaml
│   ├── run.sh
│   └── sync.sh
│
├── README.md
├── LICENSE
├── .gitignore
└── STRUCTURE_REORGANIZATION.md     # This file
```

## Current Layout (after reorganization)

```
LaRes/
├── lares/                    # Main package
│   ├── core/                 # symbolic_policy, training_*, policy_generation, models, replay_buffer, reward_utils
│   ├── rl/                   # sac, arguments
│   ├── envs/rlkit/           # rlkit (normalized_box_env, proxy_env)
│   └── utils/                # utils.py + misc, file_utils, extract_task_code, create_task, prune_env*, prompts
├── scripts/                  # run_demo, plot_training_dynamics, LaRes_from_scratch, LaRes_with_init
├── tests/                    # test_phase1_phase2, test_training_pipeline, test_generate_policy, test_generate_code
├── docs/                     # INSTRUMENTATION_PLAN, TRAINING_PIPELINE, SYMBOLIC_POLICY_IMPL, Codespace_explanation
├── config/                   # environment.yaml, run.sh, sync.sh
├── rlkit → lares/envs/rlkit  # Symlink for backward-compatible imports (from rlkit.envs.wrappers)
├── README.md, LICENSE, .gitignore
└── data.pkl, LaRes.png, logs/  # Runtime data / assets (left at root)
```

## File Mapping (before → after)

| Current Location | New Location |
|------------------|--------------|
| `arguments.py` | `lares/rl/arguments.py` |
| `models.py` | `lares/core/models.py` |
| `policy_generation.py` | `lares/core/policy_generation.py` |
| `replay_buffer.py` | `lares/core/replay_buffer.py` |
| `reward_utils.py` | `lares/core/reward_utils.py` |
| `sac.py` | `lares/rl/sac.py` |
| `symbolic_policy.py` | `lares/core/symbolic_policy.py` |
| `training_logger.py` | `lares/core/training_logger.py` |
| `training_pipeline.py` | `lares/core/training_pipeline.py` |
| `utils.py` | `lares/utils/utils.py` |
| `utils/*.py` | `lares/utils/*.py` |
| `utils/prompts/` | `lares/utils/prompts/` |
| `utils/no_init_prompts/` | `lares/utils/no_init_prompts/` |
| `utils/policy_prompts/` | `lares/utils/policy_prompts/` |
| `rlkit/` | `lares/envs/rlkit/` |
| `run_demo.py` | `scripts/run_demo.py` |
| `plot_training_dynamics.py` | `scripts/plot_training_dynamics.py` |
| `LaRes_from_scratch.py` | `scripts/LaRes_from_scratch.py` |
| `LaRes_with_init.py` | `scripts/LaRes_with_init.py` |
| `test_*.py` | `tests/test_*.py` |
| `INSTRUMENTATION_PLAN.md` | `docs/INSTRUMENTATION_PLAN.md` |
| `TRAINING_PIPELINE.md` | `docs/TRAINING_PIPELINE.md` |
| `SYMBOLIC_POLICY_IMPL.md` | `docs/SYMBOLIC_POLICY_IMPL.md` |
| `Codespace_explanation.md` | `docs/Codespace_explanation.md` |
| `environment.yaml` | `config/environment.yaml` |
| `run.sh` | `config/run.sh` |
| `sync.sh` | `config/sync.sh` |

## How to Run Scripts

**All commands must be run from the LaRes project root.**

```bash
# Symbolic policy pipeline demo (stages 1–3, no API key)
python scripts/run_demo.py

# Full demo with LLM evolution (needs OPENAI_API_KEY)
python scripts/run_demo.py --run-stage4

# LaRes training (no init)
python scripts/LaRes_from_scratch.py --env-name='window-close-v2' [options]

# LaRes training (with init)
python scripts/LaRes_with_init.py --env-name='coffee-pull-v2' [options]

# Plot training dynamics from JSONL logs
python scripts/plot_training_dynamics.py --log-path ./logs/training_dynamics/run.jsonl

# Run training pipeline directly (alternative to run_demo)
python lares/core/training_pipeline.py --env-name reach-v2 --stage all

# Run tests
python tests/test_training_pipeline.py
python tests/test_phase1_phase2.py

# Batch training (uncomment desired commands in config/run.sh first)
bash config/run.sh
```

## Import Changes (completed)

After moving files, these imports will need to be updated:

- **`scripts/*`** and **`tests/*`**: Add project root to `sys.path` or use `pip install -e .`, then:
  - `from symbolic_policy` → `from lares.core.symbolic_policy`
  - `from training_pipeline` → `from lares.core.training_pipeline`
  - `from training_logger` → `from lares.core.training_logger`
  - `from policy_generation` → `from lares.core.policy_generation`
  - `from replay_buffer` → `from lares.core.replay_buffer`
  - `from arguments` → `from lares.rl.arguments`
  - `from sac` → `from lares.rl.sac`
  - `from models` → `from lares.core.models`
  - `import utils` → `from lares import utils` or `from lares.utils import utils`
  - `from reward_utils` → `from lares.core.reward_utils`
  - `from rlkit.envs.wrappers` → `from lares.envs.rlkit.envs.wrappers`

- **Internal package imports** (e.g. `utils` importing `utils.extract_task_code`): paths may need `lares.utils.` prefix.

- **run.sh / config**: Update `python LaRes_with_init.py` → `python scripts/LaRes_with_init.py` (or run from project root with module: `python -m scripts.LaRes_with_init`).

## Alternative: Simpler Flattened Layout

If you prefer minimal nesting and fewer import changes:

```
LaRes/
├── core/              # symbolic_policy, training_*, policy_generation, models, replay_buffer, reward_utils
├── rl/                # sac, arguments
├── envs/              # rlkit (move as-is)
├── utils/             # utils.py + existing utils/*
├── scripts/           # run_demo, plot_*, LaRes_*
├── tests/
├── docs/
└── config/
```

This keeps modules at top-level package names (e.g. `from core.symbolic_policy`) and requires fewer path/package changes.
