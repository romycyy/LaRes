#!/usr/bin/env python
"""
Record GIF rollouts of the MetaWorld built-in expert policy.

This script reuses the same YAML config format as ``scripts/run_full_evolution.py``.

Usage (from project root):
    python scripts/visualize_expert_policy.py
    python scripts/visualize_expert_policy.py --config config/run_full_evolution.yaml
    python scripts/visualize_expert_policy.py --episodes 3 --fps 15
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from scripts.run_full_evolution import (
    DEFAULT_CONFIG_PATH,
    load_config,
    make_env,
)  # noqa: E402
from lares.core.training_pipeline import (  # noqa: E402
    TASK_DESCRIPTIONS,
    ensure_mujoco_headless_gl,
    record_episode_gif,
)

from expert_policy import SawyerPushV3Policy


class ExpertPolicyAdapter(nn.Module):
    """Wrap MetaWorld expert policy to match ``record_episode_gif`` model interface."""

    def __init__(self, env_name: str, std: float = 0.000000000001):
        super().__init__()
        self.expert = SawyerPushV3Policy()
        self.std = float(std)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs_np = obs.detach().cpu().numpy()
        # print(obs_np.shape)
        for one_obs in obs_np:
            act = np.clip(self.expert.get_action(one_obs), -1.0, 1.0)
        print(f"Action: {act}")
        mean = torch.tensor(act, dtype=obs.dtype, device=obs.device)
        std = torch.ones_like(mean) * self.std
        print(f"Mean: {mean}, Std: {std}")
        return mean, std


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize MetaWorld expert policy with GIF rollouts"
    )
    p.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"YAML config file from run_full_evolution (default: {DEFAULT_CONFIG_PATH})",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of expert episodes to record",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=150,
        help="Max steps per rollout episode",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=20,
        help="GIF frame rate",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for GIF outputs (default: <log_dir>/expert_viz)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(os.path.abspath(args.config))

    if args.episodes <= 0:
        raise ValueError("--episodes must be > 0")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be > 0")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    ensure_mujoco_headless_gl()
    env = make_env(cfg.env_name, cfg.seed)
    expert_model = ExpertPolicyAdapter(cfg.env_name)

    out_dir = args.output_dir.strip() or os.path.join(cfg.log_dir, "expert_viz")
    os.makedirs(out_dir, exist_ok=True)

    print("\nExpert Policy Visualization")
    print(
        f"Task    : {cfg.env_name} — {TASK_DESCRIPTIONS.get(cfg.env_name, cfg.env_name)}"
    )
    print(f"Seed    : {cfg.seed}")
    print(f"Config  : {os.path.abspath(args.config)}")
    print(f"Output  : {os.path.abspath(out_dir)}")
    print(f"Episodes: {args.episodes}, max_steps={args.max_steps}, fps={args.fps}")

    rewards = []
    successes = []
    saved_count = 0
    try:
        for ep in range(args.episodes):
            gif_path = os.path.join(out_dir, f"expert_ep_{ep + 3:03d}.gif")
            result = record_episode_gif(
                policy=expert_model,
                env=env,
                path=gif_path,
                max_steps=args.max_steps,
                fps=args.fps,
                verbose=True,
            )
            rewards.append(float(result["episode_reward"]))
            successes.append(float(result["success"]))
            saved_count += int(bool(result["saved"]))
            print(
                f"  [{ep + 1}/{args.episodes}] reward={result['episode_reward']:.2f}, "
                f"success={int(result['success'])}, frames={result['num_frames']}, "
                f"saved={result['saved']}, path={gif_path}"
            )
    finally:
        env.close()

    if rewards:
        print("\nRollout summary")
        print(f"  mean_reward : {np.mean(rewards):.2f}")
        print(f"  success_rate: {np.mean(successes):.2f}")
        print(f"  gifs_saved  : {saved_count}/{args.episodes}")


if __name__ == "__main__":
    main()
