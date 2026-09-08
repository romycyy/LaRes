<div align="center">

<img src="LaRes.png" alt="LaRes Logo" width="30%">

# LaRes: LLM-Based Symbolic Policy Evolution

[![Paper](https://img.shields.io/badge/Paper-OpenReview-blue)](https://openreview.net/pdf?id=jRjvcqtdtA)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-green)](https://github.com/yeshenpy/LaRes)

</div>

This repository runs a **symbolic policy evolution** pipeline for MetaWorld manipulation tasks. An LLM proposes candidate policy structures; each candidate is trained with behavioral cloning (BC) and GRPO-style RL fine-tuning, then ranked across evolutionary generations.

The main entry point is **`scripts/run_full_evolution.py`**. See [`docs/run_full_evolution_overview.md`](docs/run_full_evolution_overview.md) for architecture details.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Companion Scripts](#companion-scripts)
- [Project Structure](#project-structure)
- [Citation](#citation)

## Overview

```
Stage 1   Expert dataset collection  →  DemoBuffer
Stages 2–4  EvolutionOrchestrator
              ├── LLM proposes pop_size policy structures per generation
              ├── BC → RL → evaluate per candidate
              └── elites feed back as context to the next generation
```

Configuration is driven by [`config/run_full_evolution.yaml`](config/run_full_evolution.yaml).

## Installation

### 1. Clone and create the environment

```bash
git clone https://github.com/yeshenpy/LaRes.git
cd LaRes
conda env create -f config/environment.yaml
conda activate Metaworld-v2
```

### 2. MetaWorld

Install MetaWorld following the [Farama MetaWorld](https://github.com/Farama-Foundation/Metaworld) instructions, or use the git pin in `config/environment.yaml`.

### 3. API key

```bash
export OPENAI_API_KEY="your-key-here"
```

## Usage

From the project root:

```bash
python scripts/run_full_evolution.py
python scripts/run_full_evolution.py --config path/to/custom.yaml
```

Key config fields (see YAML for defaults):

| Key | Purpose |
|-----|---------|
| `env_name` | MetaWorld task (e.g. `push-v2`) |
| `dataset_episodes` | Expert demos for Stage 1 |
| `bc_steps`, `rl_iterations`, `rl_episodes` | Inner training loop per candidate |
| `num_generations`, `pop_size`, `elite_num` | Evolution outer loop |
| `model` | OpenAI model for policy generation |
| `log_dir` | Output directory for logs, checkpoints, GIFs |

Outputs under `log_dir` include `demo_<task>.pkl`, JSONL training logs, `best_policy_code.py`, and `best_policy.pt`.

## Companion Scripts

| Script | Purpose |
|--------|---------|
| [`scripts/plot_training_dynamics.py`](scripts/plot_training_dynamics.py) | Plot BC/RL/evolution metrics from JSONL logs |
| [`scripts/visualize_expert_policy.py`](scripts/visualize_expert_policy.py) | Record expert-policy GIF rollouts |
| [`scripts/run_shadowhand_spin_stages.py`](scripts/run_shadowhand_spin_stages.py) | Isaac Lab ShadowHandSpin 3-stage run (`--config config/shadowhand_spin_stages.yaml`) |

## Isaac Lab pipeline

A second environment backend runs the same Stage 1→3 recipe on Isaac Lab's `ShadowHandSpin`
instead of MetaWorld. It needs a CUDA machine with Isaac Lab installed at the config's
`isaaclab_root`:

```bash
python scripts/run_shadowhand_spin_stages.py --config config/shadowhand_spin_stages.yaml
```

[`lares/envs/isaac_lab_adapter.py`](lares/envs/isaac_lab_adapter.py) (`IsaacLabSingleEnvAdapter`)
presents Isaac's batched vector env with the same `reset() -> (obs, info)` / 4-tuple `step()` /
`info["success"]` contract the MetaWorld wrapper provides. Design notes and open issues:
[`docs/PIPELINE_REFINEMENT_SPEC.md`](docs/PIPELINE_REFINEMENT_SPEC.md).

## Project Structure

```
LaRes/
├── scripts/
│   ├── run_full_evolution.py       # Main entry (MetaWorld)
│   ├── run_shadowhand_spin_stages.py  # Isaac Lab ShadowHandSpin entry
│   ├── plot_training_dynamics.py
│   └── visualize_expert_policy.py
├── config/
│   ├── run_full_evolution.yaml
│   ├── shadowhand_spin_stages.yaml
│   └── environment.yaml
├── lares/
│   ├── core/                       # training_pipeline, policy_generation,
│   │                               #   symbolic_policy, training_logger
│   ├── utils/
│   │   ├── metaworld_env.py        # MetaWorld env factory + wrapper
│   │   └── policy_prompts/         # LLM prompt templates
│   └── envs/
│       ├── isaac_lab_adapter.py    # Isaac Lab single-env adapter
│       └── rlkit/                  # Gym wrappers
├── tests/
└── docs/
```

## Citation

If you use this work, please cite the original LaRes paper:

```bibtex
@inproceedings{
li2025lares,
title={LaRes: Evolutionary Reinforcement Learning with {LLM}-based Adaptive Reward Search},
author={Pengyi Li and Hongyao Tang and Jinbin Qiao and YAN ZHENG and Jianye Hao},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/pdf?id=jRjvcqtdtA}
}
```

## License

MIT License — see [LICENSE](LICENSE).
