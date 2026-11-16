<div align="center">

<img src="LaRes.png" alt="LaRes Logo" width="30%">

# LaRes: LLM-Based Symbolic Policy Evolution

[![Paper](https://img.shields.io/badge/Paper-OpenReview-blue)](https://openreview.net/pdf?id=jRjvcqtdtA)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-green)](https://github.com/yeshenpy/LaRes)

</div>

This repository runs a **symbolic policy evolution** pipeline for MetaWorld manipulation tasks. An LLM proposes candidate policy structures; each candidate is trained with behavioral cloning (BC) and GRPO-style RL fine-tuning, then ranked across evolutionary generations.

The main entry point is **`scripts/run_full_evolution.py`**. **This README predates most of the current pipeline (it still describes BC + GRPO training). For what the pipeline is and what has been established, read [`docs/PIPELINE_STATE.md`](docs/PIPELINE_STATE.md); older design documents are in [`docs/deprecated/`](docs/deprecated/).**

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
| `bc_steps` | Imitation-fitting budget per candidate (0 by default: candidates are evaluated zero-shot; there is no RL stage) |
| `num_generations`, `pop_size`, `elite_num` | Evolution outer loop |
| `model` | OpenAI model for policy generation |
| `log_dir` | Output directory for logs, checkpoints, GIFs |

Outputs under `log_dir` include `demo_<task>.pkl`, JSONL training logs, `best_policy_code.py`, and `best_policy.pt`.

## Companion Scripts

| Script | Purpose |
|--------|---------|
| [`scripts/plot_training_dynamics.py`](scripts/plot_training_dynamics.py) | Plot BC/RL/evolution metrics from JSONL logs |
| [`scripts/isaac/record_pen_spin_gif.py`](scripts/isaac/record_pen_spin_gif.py) | Render pen-spin rollouts of a controller to GIFs |
| [`scripts/run_shadowhand_spin_stages.py`](scripts/run_shadowhand_spin_stages.py) | Isaac-only pen-spin studies and protocols (`--study <name>`; config `config/shadowhand_spin_stages.yaml`) |

Finished studies' scripts are archived in [`deprecated/`](deprecated/README.md).

## Isaac Lab pipeline

The Shadow Hand pen-spin stack (`lares/envs/isaac_pen_spin/`) runs on Isaac Lab. It needs a CUDA machine with
Isaac Lab installed at the config's `isaaclab_root`. Isaac-only studies run through:

```bash
python scripts/run_shadowhand_spin_stages.py --study <name>     # names: PEN_SPIN_STUDIES in the script
```

LLM studies on pen spin run through `scripts/run_full_evolution.py --config config/pen_spin_plateau_repair.yaml --study <name>`.
The old three-stage BC + GRPO smoke run and `IsaacLabSingleEnvAdapter` were archived on 2026-11-15
(see [`deprecated/README.md`](deprecated/README.md)); [`lares/envs/isaac_lab_adapter.py`](lares/envs/isaac_lab_adapter.py)
now only launches and closes the Isaac app.

## Project Structure

```
LaRes/
├── scripts/
│   ├── run_full_evolution.py       # Main entry (MetaWorld)
│   ├── run_shadowhand_spin_stages.py  # Isaac-only pen-spin studies
│   ├── plot_training_dynamics.py
│   └── isaac/                      # pen-spin incumbent tools (adopt_fixed_stroke, record_pen_spin_gif)
├── config/
│   ├── run_full_evolution.yaml
│   ├── shadowhand_spin_stages.yaml
│   └── environment.yaml
├── lares/
│   ├── core/                       # training_pipeline, policy_generation,
│   │                               #   symbolic_policy, policy_validator, obs_schema
│   ├── eval/                       # manifests, runner, promotion, freeze
│   ├── search/                     # generation provenance, lanes, records, archive
│   ├── fitting/                    # objectives, optimizers, structure bank
│   ├── repair/                     # expert-trace evidence for plateau repair
│   ├── research/                   # research-idea ledger and advisor
│   ├── utils/
│   │   ├── metaworld_env.py        # MetaWorld env factory + wrapper
│   │   └── policy_prompts/         # LLM prompt templates
│   └── envs/
│       ├── isaac_lab_adapter.py    # Isaac app launch/close      
│       ├── isaac_pen_spin/         # Shadow Hand pen-spin env, protocols, studies
│       └── rlkit/                  # Gym wrappers
├── tests/
├── deprecated/                     # archived scripts, modules and tests (see its README)
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
