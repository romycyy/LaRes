#!/usr/bin/env python
"""
End-to-end demo of the 4-stage symbolic policy pipeline on reach-v2.

reach-v2 is the simplest MetaWorld task: move the robotic arm to a target
position. No grasping or object manipulation, so it converges fast and
is ideal for validating the full pipeline.

Covers:
  - Environment creation & expert baseline
  - Stage 1: Expert dataset generation
  - Stage 2: Behavioral cloning (hand-crafted + LLM-generated policy)
  - Stage 3: GRPO RL fine-tuning
  - Stage 4: LLM-based evolution / structure search
  - Final evaluation & comparison table

Usage:
    python run_demo.py                        # Stages 1-3 only (no API key needed)
    python run_demo.py --run-stage4           # All 4 stages (needs OPENAI_API_KEY)
    python run_demo.py --run-stage4 --model gpt-4o   # Use a specific model
"""

import os
import sys
import time
import argparse
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from symbolic_policy import SymbolicPolicy
from training_pipeline import (
    generate_dataset,
    behavioral_cloning,
    rl_finetune,
    evaluate_policy,
    get_expert_policy,
    llm_evolution,
    TASK_DESCRIPTIONS,
)

# ===================================================================
#  Configuration
# ===================================================================

ENV_NAME = "reach-v2"
OBS_DIM = 39
ACT_DIM = 4
SEED = 42


def parse_args():
    p = argparse.ArgumentParser(
        description="Symbolic Policy Pipeline Demo (reach-v2)"
    )
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--dataset-episodes", type=int, default=30)
    p.add_argument("--bc-steps", type=int, default=1500)
    p.add_argument("--rl-iterations", type=int, default=15)
    p.add_argument("--rl-episodes", type=int, default=10)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--run-stage4", action="store_true")
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    p.add_argument("--evo-generations", type=int, default=1)
    p.add_argument("--evo-pop-size", type=int, default=2)
    return p.parse_args()


# ===================================================================
#  Hand-crafted symbolic policy for reach-v2
# ===================================================================

class ReachPolicy(SymbolicPolicy):
    """Two-phase reach policy: approach target, then fine-adjust."""

    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.w_coarse = nn.Parameter(torch.tensor(4.0))
        self.w_fine = nn.Parameter(torch.tensor(1.0))
        self.threshold = nn.Parameter(torch.tensor(0.06))
        self.sharpness = nn.Parameter(torch.tensor(80.0))
        self.grip_bias = nn.Parameter(torch.tensor(0.0))
        self.log_std = nn.Parameter(torch.tensor(-1.0))

    def forward(self, obs):
        tcp = obs[:, 0:3]
        target = obs[:, 4:7]
        diff = target - tcp
        dist = torch.norm(diff, dim=-1, keepdim=True) + 1e-8
        direction = diff / dist

        phase = torch.sigmoid(self.sharpness * (self.threshold - dist))
        move = (1 - phase) * self.w_coarse * direction + phase * self.w_fine * direction
        grip = self.grip_bias * torch.ones(obs.shape[0], 1)
        mean = torch.cat([move, grip], dim=1)[:, :self.action_dim]
        std = torch.exp(self.log_std) * torch.ones_like(mean)
        return (mean, std)

    def get_param_ranges(self):
        return {
            "w_coarse": (0.5, 15.0),
            "w_fine": (0.1, 5.0),
            "threshold": (0.01, 0.3),
            "sharpness": (5.0, 200.0),
            "grip_bias": (-1.0, 1.0),
            "log_std": (-5.0, 0.0),
        }


# ===================================================================
#  Utilities
# ===================================================================

def separator(title):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def print_eval(label, result):
    print(f"  {label:30s}  reward={result['mean_reward']:8.2f}  "
          f"success={result['success_rate']:.2f}")


def make_env(seed):
    """Create the MetaWorld reach-v2 environment with proper wrappers."""
    from utils import make_metaworld_env, env_wrapper
    cfg = SimpleNamespace(env_name=ENV_NAME, episode_length=150)
    raw_env = make_metaworld_env(cfg, seed)
    return env_wrapper(raw_env, cfg)


def evaluate_expert(env, env_name, num_episodes=10):
    """Run the MetaWorld built-in expert and return eval stats."""
    expert = get_expert_policy(env_name)
    total_reward, total_success = 0.0, 0.0

    for _ in range(num_episodes):
        obs, _ = env.reset()
        ep_reward, success = 0.0, False
        for _ in range(150):
            action = np.clip(expert.get_action(obs), -1.0, 1.0)
            next_obs, reward, done, info = env.step(env.action_space.high * action)
            ep_reward += reward
            if info.get("success", 0) > 0:
                success = True
            obs = next_obs
            if done:
                break
        total_reward += ep_reward
        total_success += float(success)

    return {
        "mean_reward": total_reward / num_episodes,
        "success_rate": total_success / num_episodes,
    }


# ===================================================================
#  Main
# ===================================================================

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    results = {}
    t0 = time.time()

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("\nSymbolic Policy Pipeline Demo")
    print(f"Task: {ENV_NAME} — {TASK_DESCRIPTIONS[ENV_NAME]}")
    print(f"Seed: {args.seed}")
    print(f"GPU:  {gpu}")

    # ---------------------------------------------------------------
    #  Environment + Expert Baseline
    # ---------------------------------------------------------------
    separator("Stage 0: Environment & Expert Baseline")

    env = make_env(args.seed)
    obs, _ = env.reset()
    print(f"  obs_dim={obs.shape[0]}, action_dim={env.action_space.shape[0]}")

    expert_result = evaluate_expert(env, ENV_NAME, num_episodes=args.eval_episodes)
    results["expert"] = expert_result
    print_eval("Expert (built-in)", expert_result)

    # ---------------------------------------------------------------
    #  Stage 1: Dataset Generation
    # ---------------------------------------------------------------
    separator("Stage 1: Expert Dataset Generation")

    demo_buf, dataset_stats = generate_dataset(
        env, ENV_NAME,
        num_episodes=args.dataset_episodes,
        max_steps=150,
    )
    print(f"  Collected {len(demo_buf)} transitions from {args.dataset_episodes} episodes")
    print(f"  Expert demo reward={dataset_stats['mean_reward']:.2f}, "
          f"success={dataset_stats['mean_success']:.2f}")

    obs_all, act_all, _, _, _ = demo_buf.get_all()
    print(f"  Action range: [{act_all.min():.3f}, {act_all.max():.3f}]")
    assert not np.isnan(obs_all).any(), "NaN in observations!"
    assert not np.isnan(act_all).any(), "NaN in actions!"
    print("  Data integrity check: PASSED")

    # ---------------------------------------------------------------
    #  Stage 2: Behavioral Cloning
    # ---------------------------------------------------------------
    separator("Stage 2: Behavioral Cloning")

    policy = ReachPolicy(OBS_DIM, ACT_DIM)
    policy.validate()
    print(f"  Policy: ReachPolicy ({policy.count_parameters()} parameters)")
    print(f"  Param ranges: {policy.get_param_ranges()}")

    eval_pre_bc = evaluate_policy(policy, env, num_episodes=args.eval_episodes)
    results["pre_bc"] = eval_pre_bc
    print_eval("Before BC (random init)", eval_pre_bc)

    bc_stats = behavioral_cloning(
        policy, demo_buf,
        num_steps=args.bc_steps,
        batch_size=min(128, len(demo_buf)),
        lr=1e-3,
        log_interval=args.bc_steps // 3,
    )

    early_loss = np.mean(bc_stats["bc_loss"][:50])
    final_loss = bc_stats["final_loss"]
    print(f"  BC loss: {early_loss:.6f} -> {final_loss:.6f} "
          f"({(1 - final_loss / early_loss) * 100:.1f}% reduction)")

    eval_post_bc = evaluate_policy(policy, env, num_episodes=args.eval_episodes)
    results["post_bc"] = eval_post_bc
    print_eval("After BC", eval_post_bc)

    # ---------------------------------------------------------------
    #  Stage 3: RL Fine-tuning (GRPO)
    # ---------------------------------------------------------------
    separator("Stage 3: RL Fine-tuning (GRPO)")

    rl_stats = rl_finetune(
        policy, env,
        num_iterations=args.rl_iterations,
        episodes_per_iter=args.rl_episodes,
        lr=3e-4,
        gamma=0.99,
        max_steps=150,
        log_interval=max(1, args.rl_iterations // 3),
    )

    print(f"  Best success rate during RL: {rl_stats['best_success_rate']:.2f}")
    print(f"  Final mean return: {rl_stats['final_mean_return']:.2f}")
    print(f"  Policy loss trend: {rl_stats['policy_loss'][0]:.4f} -> "
          f"{rl_stats['policy_loss'][-1]:.4f}")

    eval_post_rl = evaluate_policy(policy, env, num_episodes=args.eval_episodes)
    results["post_rl"] = eval_post_rl
    print_eval("After RL", eval_post_rl)

    # ---------------------------------------------------------------
    #  Parameter inspection
    # ---------------------------------------------------------------
    separator("Learned Parameters")

    for name, param in policy.named_parameters():
        lo, hi = policy.get_param_ranges()[name]
        val = param.detach()
        if val.numel() == 1:
            print(f"  {name:15s} = {val.item():8.4f}  (range: [{lo}, {hi}])")
        else:
            print(f"  {name:15s} = {val.numpy()}  (range: [{lo}, {hi}])")

    # ---------------------------------------------------------------
    #  Stage 4: LLM Evolution (optional)
    # ---------------------------------------------------------------
    if args.run_stage4:
        separator("Stage 4: LLM-Based Evolution")

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key or len(api_key) < 10:
            print("  ERROR: OPENAI_API_KEY not set. Skipping Stage 4.")
        else:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            llm_args = SimpleNamespace(model=args.model)
            log_dir = os.path.join(ROOT, "logs", "demo_evolution")

            print(f"  LLM model: {args.model}")
            print(f"  Generations: {args.evo_generations}, "
                  f"Pop size: {args.evo_pop_size}")

            evo_result = llm_evolution(
                client=client,
                env=env,
                env_name=ENV_NAME,
                demo_buffer=demo_buf,
                args=llm_args,
                obs_dim=OBS_DIM,
                action_dim=ACT_DIM,
                num_generations=args.evo_generations,
                pop_size=args.evo_pop_size,
                elite_num=1,
                bc_steps=args.bc_steps,
                rl_iterations=args.rl_iterations,
                rl_episodes_per_iter=args.rl_episodes,
                log_dir=log_dir,
            )

            if evo_result["policy"] is not None:
                eval_evo = evaluate_policy(
                    evo_result["policy"], env, num_episodes=args.eval_episodes
                )
                results["llm_evo"] = eval_evo
                print_eval("LLM-evolved policy", eval_evo)
                print(f"  LLM policy parameters: "
                      f"{evo_result['policy'].count_parameters()}")

                code_path = os.path.join(log_dir, "best_policy_code.py")
                with open(code_path, "w") as f:
                    f.write(evo_result["code"])
                print(f"  Best policy code saved to: {code_path}")
            else:
                print("  WARNING: LLM evolution produced no valid policies.")
    else:
        print("\n  [Stage 4 skipped — pass --run-stage4 to enable]")

    # ---------------------------------------------------------------
    #  Save demo buffer for reuse
    # ---------------------------------------------------------------
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    buf_path = os.path.join(ROOT, "logs", f"demo_{ENV_NAME}.pkl")
    demo_buf.save(buf_path)
    print(f"\n  Demo buffer saved to: {buf_path}")

    model_path = os.path.join(ROOT, "logs", f"demo_{ENV_NAME}_policy.pt")
    torch.save(policy.state_dict(), model_path)
    print(f"  Trained policy saved to: {model_path}")

    # ---------------------------------------------------------------
    #  Summary Table
    # ---------------------------------------------------------------
    elapsed = time.time() - t0
    separator(f"Results Summary  (total time: {elapsed:.1f}s)")

    print(f"\n  {'Stage':<30s} {'Reward':>10s} {'Success':>10s}")
    print(f"  {'-' * 50}")
    for label, key in [
        ("Expert (built-in)", "expert"),
        ("Before BC (random init)", "pre_bc"),
        ("After BC (Stage 2)", "post_bc"),
        ("After RL (Stage 3)", "post_rl"),
        ("LLM-evolved (Stage 4)", "llm_evo"),
    ]:
        if key in results:
            r = results[key]
            print(f"  {label:<30s} {r['mean_reward']:>10.2f} {r['success_rate']:>10.2f}")

    print(f"\n  Pipeline validated on {ENV_NAME}.")

    reward_improved = results["post_rl"]["mean_reward"] > results["pre_bc"]["mean_reward"]
    print(f"  Reward improved from init: {'YES' if reward_improved else 'NO'}")

    if results["post_rl"]["success_rate"] > 0:
        print("  Policy achieves non-zero success rate after training.")

    print()


if __name__ == "__main__":
    main()
