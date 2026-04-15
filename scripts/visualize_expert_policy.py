#!/usr/bin/env python
"""
Record GIF rollouts of the MetaWorld built-in expert policy.

Uses the same YAML config, ``load_config`` / ``make_env``, and expert rollout
logic as ``scripts/run_full_evolution.py`` Stage 1 (``get_expert_policy`` +
deterministic clipped actions, no Gaussian sampling around the mean).

Usage (from project root):
    python scripts/visualize_expert_policy.py
    python scripts/visualize_expert_policy.py --config config/run_full_evolution.yaml
    python scripts/visualize_expert_policy.py --episodes 50 --failure-gifs 5
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

from scripts.run_full_evolution import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_config,
    make_env,
)
from lares.core.training_pipeline import (  # noqa: E402
    TASK_DESCRIPTIONS,
    ensure_mujoco_headless_gl,
    get_expert_policy,
    record_episode_gif,
)


class ExpertPolicyAdapter(nn.Module):
    """Wrap MetaWorld expert policy to match ``record_episode_gif`` model interface."""

    def __init__(self, env_name: str, std: float = 1e-12):
        super().__init__()
        self.expert = get_expert_policy(env_name)
        self.std = float(std)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs_np = obs.detach().cpu().numpy()
        row = obs_np[0] if obs_np.ndim == 2 else obs_np
        act = np.clip(self.expert.get_action(row.astype(np.float64)), -1.0, 1.0)
        mean = torch.tensor(act, dtype=obs.dtype, device=obs.device)
        std = torch.ones_like(mean) * self.std
        return mean, std


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize MetaWorld expert policy with GIF rollouts "
        "(matches run_full_evolution Stage 1 expert + env)."
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
        default=10,
        help="Number of expert rollouts (GIF + stats per episode)",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=150,
        help="Max steps per rollout (same default as generate_dataset)",
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
    p.add_argument(
        "--failure-gifs",
        type=int,
        default=0,
        help="If >0, keep at most this many GIFs from failed episodes only "
        "(success rollouts delete their GIF file). 0 = save one GIF per episode.",
    )
    p.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Print rolling Stage-1-style stats every N episodes (0 to disable)",
    )
    p.add_argument(
        "--verbose-steps",
        action="store_true",
        help="Per-step prints inside record_episode_gif (very noisy)",
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
    if args.failure_gifs < 0:
        raise ValueError("--failure-gifs must be >= 0")

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    ensure_mujoco_headless_gl()
    env = make_env(cfg)
    expert_model = ExpertPolicyAdapter(cfg.env_name)

    out_dir = args.output_dir.strip() or os.path.join(cfg.log_dir, "expert_viz")
    os.makedirs(out_dir, exist_ok=True)

    print("\nExpert Policy Visualization (aligned with run_full_evolution Stage 1)")
    print(
        f"Task    : {cfg.env_name} — {TASK_DESCRIPTIONS.get(cfg.env_name, cfg.env_name)}"
    )
    print(f"Seed    : {cfg.seed}")
    print(f"Config  : {os.path.abspath(args.config)}")
    print(f"Output  : {os.path.abspath(out_dir)}")
    print(
        f"Episodes: {args.episodes}, max_steps={args.max_steps}, fps={args.fps}, "
        f"deterministic expert actions (no policy sampling)"
    )
    print(
        f"Env     : use_mt1={getattr(cfg, 'use_mt1', False)}, "
        f"episode_length={cfg.episode_length}"
    )
    if args.failure_gifs > 0:
        print(f"Failure GIFs: keep up to {args.failure_gifs} failed-episode GIFs only")

    rewards: list[float] = []
    successes: list[float] = []
    saved_count = 0
    fail_saved = 0
    ep_rewards_buf: list[float] = []
    ep_success_buf: list[float] = []

    try:
        for ep in range(args.episodes):
            tmp_path = os.path.join(out_dir, f"_expert_rollout_{ep:04d}.gif")
            result = record_episode_gif(
                policy=expert_model,
                env=env,
                path=tmp_path,
                max_steps=args.max_steps,
                fps=args.fps,
                verbose=args.verbose_steps,
                deterministic=True,
            )
            rewards.append(float(result["episode_reward"]))
            succ = float(result["success"])
            successes.append(succ)
            ep_rewards_buf.append(float(result["episode_reward"]))
            ep_success_buf.append(succ)

            if args.failure_gifs > 0:
                if result["success"]:
                    if os.path.isfile(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass
                else:
                    if fail_saved < args.failure_gifs and result["saved"]:
                        dest = os.path.join(
                            out_dir, f"expert_FAIL_{fail_saved + 1:03d}_ep{ep + 1:04d}.gif"
                        )
                        if os.path.isfile(tmp_path):
                            os.replace(tmp_path, dest)
                            fail_saved += 1
                            saved_count += 1
                    elif os.path.isfile(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass
            else:
                final_path = os.path.join(out_dir, f"expert_ep_{ep + 1:03d}.gif")
                if os.path.isfile(tmp_path):
                    os.replace(tmp_path, final_path)
                    if result["saved"]:
                        saved_count += 1

            if args.log_interval > 0 and (ep + 1) % args.log_interval == 0:
                w = min(args.log_interval, len(ep_rewards_buf))
                tail_r = ep_rewards_buf[-w:]
                tail_s = ep_success_buf[-w:]
                print(
                    f"  [expert viz] {ep + 1}/{args.episodes} eps, "
                    f"avg_reward={np.mean(tail_r):.2f}, "
                    f"success_rate={np.mean(tail_s):.2f}"
                )

            print(
                f"  ep {ep + 1}/{args.episodes}  reward={result['episode_reward']:.2f}  "
                f"success={int(result['success'])}  frames={result['num_frames']}  "
                f"saved={result['saved']}"
            )
    finally:
        env.close()

    if rewards:
        print("\nRollout summary")
        print(f"  mean_reward : {np.mean(rewards):.2f}")
        print(f"  success_rate: {np.mean(successes):.2f}")
        if args.failure_gifs > 0:
            print(f"  failure gifs saved: {fail_saved} (cap {args.failure_gifs})")
        else:
            print(f"  gifs_saved  : {saved_count}/{args.episodes}")


if __name__ == "__main__":
    main()
