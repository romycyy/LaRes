"""
Four-stage training pipeline for symbolic policies in modified LaRes.

Stage 1 — Dataset Generation:  use MetaWorld expert policies to collect demos
Stage 2 — Behavioral Cloning:  supervised imitation of expert actions
Stage 3 — RL Fine-tuning:      GRPO-style policy gradient improvement
Stage 4 — LLM Evolution:       structure search with BC+RL inner loop

Each stage is usable independently or as part of the full pipeline via
``SymbolicPolicyPipeline``.
"""

import inspect
import os
import copy
import pickle
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from lares.core.training_logger import (
    BC_GRAD_NORM_POST_CLIP,
    BC_GRAD_NORM_PRE_CLIP,
    BC_MEAN_LOSS,
    BC_STD_LOSS,
    BC_TRAIN_LOSS,
    EVO_FITNESS_BEST,
    EVO_FITNESS_ELITE_MEAN,
    EVO_FITNESS_MEAN,
    EVO_FITNESS_MEDIAN,
    EVO_FITNESS_WORST,
    RL_ADVANTAGE_MEAN,
    RL_ADVANTAGE_STD,
    RL_ENTROPY,
    RL_ENTROPY_BONUS,
    RL_GRAD_NORM_POST_CLIP,
    RL_GRAD_NORM_PRE_CLIP,
    RL_KL,
    RL_KL_PENALTY,
    RL_LEARNING_RATE,
    RL_POLICY_LOSS,
    RL_RETURN_MEAN,
    RL_RETURN_STD,
    RL_REWARD_MEAN,
    RL_REWARD_STD,
    RL_SUCCESS_RATE,
    RL_TOTAL_LOSS,
)

# ---------------------------------------------------------------------------
#  Expert policy mapping for MetaWorld tasks
# ---------------------------------------------------------------------------

EXPERT_POLICY_MAP = {
    "window-close-v2": "SawyerWindowCloseV3Policy",
    "window-open-v2": "SawyerWindowOpenV3Policy",
    "button-press-v2": "SawyerButtonPressV3Policy",
    "button-press-topdown-v2": "SawyerButtonPressTopdownV3Policy",
    "door-close-v2": "SawyerDoorCloseV3Policy",
    "door-open-v2": "SawyerDoorOpenV3Policy",
    "drawer-open-v2": "SawyerDrawerOpenV3Policy",
    "drawer-close-v2": "SawyerDrawerCloseV3Policy",
    "faucet-open-v2": "SawyerFaucetOpenV3Policy",
    "faucet-close-v2": "SawyerFaucetCloseV3Policy",
    "handle-press-v2": "SawyerHandlePressV3Policy",
    "handle-pull-v2": "SawyerHandlePullV3Policy",
    "lever-pull-v2": "SawyerLeverPullV3Policy",
    "reach-v2": "SawyerReachV3Policy",
    "push-v2": "SawyerPushV3Policy",
    "pick-place-v2": "SawyerPickPlaceV3Policy",
    "assembly-v2": "SawyerAssemblyV3Policy",
    "basketball-v2": "SawyerBasketballV3Policy",
    "coffee-button-v2": "SawyerCoffeeButtonV3Policy",
    "coffee-pull-v2": "SawyerCoffeePullV3Policy",
    "coffee-push-v2": "SawyerCoffeePushV3Policy",
    "dial-turn-v2": "SawyerDialTurnV3Policy",
    "hammer-v2": "SawyerHammerV3Policy",
    "sweep-v2": "SawyerSweepV3Policy",
    "soccer-v2": "SawyerSoccerV3Policy",
    "shelf-place-v2": "SawyerShelfPlaceV3Policy",
}

TASK_DESCRIPTIONS = {
    "window-close-v2": "Control the robotic arm to close the window",
    "window-open-v2": "Control the robotic arm to open the window",
    "button-press-v2": "Control the robotic arm to press the button",
    "door-close-v2": "Control the robotic arm to close the open door",
    "drawer-open-v2": "Control the robotic arm to open the drawer",
    "door-open-v2": "Control the robotic arm to open the door",
    "drawer-close-v2": "Control the robotic arm to close the drawer",
    "faucet-open-v2": "Control the robotic arm to open the faucet",
    "faucet-close-v2": "Control the robotic arm to close the faucet",
    "reach-v2": "Control the robotic arm to reach the target position",
    "push-v2": "Control the robotic arm to push the object to the target",
    "pick-place-v2": "Control the robotic arm to pick and place the object",
}


def get_expert_policy(env_name):
    """Load the MetaWorld semi-optimal expert policy for a task."""
    policy_class_name = EXPERT_POLICY_MAP.get(env_name)
    if policy_class_name is None:
        raise ValueError(
            f"No expert policy mapping for '{env_name}'. "
            f"Available: {list(EXPERT_POLICY_MAP.keys())}"
        )
    try:
        import metaworld.policies as mw_policies

        policy_class = getattr(mw_policies, policy_class_name)
        return policy_class()
    except ImportError:
        raise ImportError(
            "MetaWorld is required for expert policy loading. "
            "Install with: pip install metaworld"
        )
    except AttributeError:
        raise AttributeError(
            f"Policy class '{policy_class_name}' not found in metaworld.policies"
        )


# ---------------------------------------------------------------------------
#  DemoBuffer — lightweight storage for expert demonstrations
# ---------------------------------------------------------------------------


class DemoBuffer:
    """Stores (obs, action, reward, next_obs, done) demonstration tuples.

    Supports batch sampling, save/load, and conversion to numpy arrays.
    """

    def __init__(self):
        self.obs = []
        self.actions = []
        self.rewards = []
        self.next_obs = []
        self.dones = []

    def add(self, obs, action, reward, next_obs, done):
        self.obs.append(np.array(obs, dtype=np.float32))
        self.actions.append(np.array(action, dtype=np.float32))
        self.rewards.append(float(reward))
        self.next_obs.append(np.array(next_obs, dtype=np.float32))
        self.dones.append(float(done))

    def __len__(self):
        return len(self.obs)

    def get_all(self):
        """Return all data as numpy arrays."""
        return (
            np.array(self.obs),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.next_obs),
            np.array(self.dones),
        )

    def sample(self, batch_size):
        """Sample a random mini-batch."""
        idxes = np.random.randint(0, len(self), size=batch_size)
        return (
            np.array([self.obs[i] for i in idxes]),
            np.array([self.actions[i] for i in idxes]),
            np.array([self.rewards[i] for i in idxes]),
            np.array([self.next_obs[i] for i in idxes]),
            np.array([self.dones[i] for i in idxes]),
        )

    def save(self, path):
        data = {
            "obs": np.array(self.obs),
            "actions": np.array(self.actions),
            "rewards": np.array(self.rewards),
            "next_obs": np.array(self.next_obs),
            "dones": np.array(self.dones),
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        buf = cls()
        buf.obs = list(data["obs"])
        buf.actions = list(data["actions"])
        buf.rewards = list(data["rewards"])
        buf.next_obs = list(data["next_obs"])
        buf.dones = list(data["dones"])
        return buf


# ===========================================================================
#  Stage 1 — Dataset Generation
# ===========================================================================


def generate_dataset(env, env_name, num_episodes=100, max_steps=150):
    """Collect expert demonstrations using MetaWorld's built-in policies.

    Args:
        env: MetaWorld environment (wrapped with env_wrapper).
        env_name: Task identifier (e.g. 'window-close-v2').
        num_episodes: Number of episodes to collect.
        max_steps: Maximum steps per episode.

    Returns:
        (DemoBuffer, stats_dict)
    """
    expert = get_expert_policy(env_name)
    buffer = DemoBuffer()
    stats = defaultdict(list)

    for ep in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        success = False

        for step in range(max_steps):
            action = expert.get_action(obs)
            action_clipped = np.clip(action, -1.0, 1.0)

            next_obs, reward, done, info = env.step(
                env.action_space.high * action_clipped
            )

            buffer.add(obs, action_clipped, reward, next_obs, float(done))
            episode_reward += reward
            if info.get("success", 0) > 0:
                success = True

            obs = next_obs
            if done:
                break

        stats["episode_rewards"].append(episode_reward)
        stats["episode_successes"].append(float(success))
        stats["episode_lengths"].append(step + 1)

        if (ep + 1) % 10 == 0:
            print(
                f"  [Stage 1] {ep + 1}/{num_episodes} eps, "
                f"avg_reward={np.mean(stats['episode_rewards'][-10:]):.2f}, "
                f"success_rate={np.mean(stats['episode_successes'][-10:]):.2f}"
            )

    summary = {
        "mean_reward": float(np.mean(stats["episode_rewards"])),
        "mean_success": float(np.mean(stats["episode_successes"])),
        "num_transitions": len(buffer),
        "num_episodes": num_episodes,
    }
    print(
        f"  [Stage 1] Done: {len(buffer)} transitions, "
        f"reward={summary['mean_reward']:.2f}, "
        f"success={summary['mean_success']:.2f}"
    )
    return buffer, summary


# ===========================================================================
#  Stage 2 — Behavioral Cloning
# ===========================================================================


def behavioral_cloning(
    policy,
    demo_buffer,
    num_steps=5000,
    batch_size=256,
    lr=1e-3,
    clip_grad_norm=1.0,
    log_interval=500,
    logger=None,
    task_name=None,
    log_every_n_steps=1,
):
    """Train a symbolic policy to imitate expert actions via supervised learning.

    Uses MSE loss between ``policy.forward(obs)[0]`` (mean) and the expert
    action, plus a small penalty on std to encourage determinism near the
    expert trajectory.

    Args:
        policy: SymbolicPolicy instance with uninitialised or random params.
        demo_buffer: DemoBuffer from Stage 1.
        num_steps: Gradient steps.
        batch_size: Mini-batch size.
        lr: Adam learning rate.
        clip_grad_norm: Max gradient norm (0 to disable).
        log_interval: Print every N steps.
        logger: Optional TrainingLogger for structured metrics.
        task_name: Task identifier for logging (e.g. env_name). Required for
            multi-task runs so plots can be split by MetaWorld task.
        log_every_n_steps: Log metrics every N steps (1 = every step). Use
            >1 to reduce log volume for long runs.

    Returns:
        dict with training statistics.

    Note:
        No validation split is used; bc/val_loss is not logged. To add
        validation, provide a separate val_buffer and compute val loss
        periodically outside the optimization loop.
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    # stats = {"bc_loss": [], "mean_loss": [], "std_loss": []} old code
    stats = {"bc_loss": [], "log_prob": []}

    policy.train()
    for step in range(num_steps):
        obs_np, actions_np, _, _, _ = demo_buffer.sample(batch_size)
        obs_t = torch.tensor(obs_np, dtype=torch.float32)
        actions_t = torch.tensor(actions_np, dtype=torch.float32)

        mean, std = policy(obs_t)

        # old code
        # mean_loss = nn.functional.mse_loss(mean, actions_t)
        # std_loss = 0.01 * std.mean()
        # loss = mean_loss + std_loss


        #--- new code start
        mean, std = policy(obs_t)

        eps = 1e-6
        actions_clamped = torch.clamp(actions_t, -1 + eps, 1 - eps)
        pretanh_actions = 0.5 * torch.log((1 + actions_clamped) / (1 - actions_clamped))

        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(pretanh_actions) - torch.log(1 - actions_clamped.pow(2) + eps)
        log_probs = log_probs.sum(dim=-1)

        loss = -log_probs.mean()
        #--- new code end

        optimizer.zero_grad()
        loss.backward()

        max_norm = clip_grad_norm if clip_grad_norm > 0 else float("inf")
        grad_norm_pre = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm)
        grad_norm_pre = float(grad_norm_pre)
        grad_norm_post = (
            min(grad_norm_pre, clip_grad_norm) if clip_grad_norm > 0 else grad_norm_pre
        )

        optimizer.step()
        policy.clip_params()

        # old code
        # stats["bc_loss"].append(loss.item())
        # stats["mean_loss"].append(mean_loss.item())
        # stats["std_loss"].append(std_loss.item())


        # new code
        stats["bc_loss"].append(loss.item())
        stats["log_prob"].append(log_probs.mean().item())

        # Structured logging for training dynamics (task_name enables per-task plots)
        # if logger is not None and step % log_every_n_steps == 0:
        #     logger.log_metrics(
        #         stage="bc",
        #         update=step,
        #         metrics={
        #             BC_TRAIN_LOSS: loss.item(),
        #             BC_MEAN_LOSS: mean_loss.item(),
        #             BC_STD_LOSS: std_loss.item(),
        #             BC_GRAD_NORM_PRE_CLIP: float(grad_norm_pre),
        #             BC_GRAD_NORM_POST_CLIP: float(grad_norm_post),
        #         },
        #         task_name=task_name,
        #     )
        if logger is not None and step % log_every_n_steps == 0:
            logger.log_metrics(
                stage="bc",
                update=step,
                metrics={
                    BC_TRAIN_LOSS: loss.item(),
                    BC_MEAN_LOSS: (-log_probs.mean()).item(),
                    BC_STD_LOSS: std.mean().item(),
                    BC_GRAD_NORM_PRE_CLIP: float(grad_norm_pre),
                    BC_GRAD_NORM_POST_CLIP: float(grad_norm_post),
                },
                task_name=task_name,
            )

        if log_interval > 0 and (step + 1) % log_interval == 0:
            recent = stats["bc_loss"][-log_interval:]
            # old code
            # print(
            #     f"  [Stage 2] step {step + 1}/{num_steps}: "
            #     f"loss={np.mean(recent):.6f}, "
            #     f"mean_loss={np.mean(stats['mean_loss'][-log_interval:]):.6f}, "
            #     f"std_loss={np.mean(stats['std_loss'][-log_interval:]):.6f}"
            # )
            print(
                f"  [Stage 2] step {step + 1}/{num_steps}: "
                f"loss={np.mean(recent):.6f}, "
                f"log_prob={np.mean(stats['log_prob'][-log_interval:]):.6f}"
            )

    stats["final_loss"] = float(np.mean(stats["bc_loss"][-min(100, num_steps) :]))
    return stats


# ===========================================================================
#  Stage 3 — RL Fine-tuning (GRPO-style)
# ===========================================================================


def _collect_trajectories(policy, env, num_episodes, max_steps=150):
    """Collect on-policy trajectories from the current symbolic policy."""
    trajectories = []
    policy.eval()

    for _ in range(num_episodes):
        obs, _ = env.reset()
        traj = {
            "obs": [],
            "pretanh": [],
            "rewards": [],
            "dones": [],
        }

        for step in range(max_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mean, std = policy(obs_t)

            dist = torch.distributions.Normal(mean, std)
            pretanh_action = dist.sample()
            action = torch.tanh(pretanh_action)
            action_np = action.squeeze(0).numpy()

            next_obs, reward, done, info = env.step(env.action_space.high * action_np)

            traj["obs"].append(obs)
            traj["pretanh"].append(pretanh_action.squeeze(0).numpy())
            traj["rewards"].append(reward)
            traj["dones"].append(float(done))

            obs = next_obs
            if done:
                break

        traj["return"] = sum(traj["rewards"])
        traj["success"] = float(info.get("success", 0))
        traj["length"] = len(traj["rewards"])
        trajectories.append(traj)

    return trajectories


def _compute_grpo_advantages(trajectories, gamma=0.99):
    """GRPO-style: advantages are relative to the group's mean return."""
    returns = np.array([t["return"] for t in trajectories])
    mean_return = np.mean(returns)
    std_return = np.std(returns) + 1e-8

    for traj in trajectories:
        T = len(traj["rewards"])
        traj_advantage = (traj["return"] - mean_return) / std_return
        traj["advantages"] = traj_advantage * np.ones(T)
    return trajectories


def _grad_norm(params):
    """Global L2 norm of gradients over parameters that have grads. Skips params with None grad."""
    total_sq = 0.0
    for p in params:
        if p.grad is not None:
            total_sq += p.grad.data.norm(2).item() ** 2
    return total_sq**0.5


def _rl_metrics_from_trajectories(trajectories, all_advantages):
    """Extract reward/return/advantage stats for RL logging.

    Returns dict with default 0.0 for empty data. Safe for logging.
    """
    iter_returns = [t["return"] for t in trajectories]
    all_rewards = [r for t in trajectories for r in t["rewards"]]
    adv_arr = np.array(all_advantages) if all_advantages else np.array([0.0])

    return {
        "return_mean": float(np.mean(iter_returns)) if iter_returns else 0.0,
        "return_std": float(np.std(iter_returns)) if len(iter_returns) > 1 else 0.0,
        "reward_mean": float(np.mean(all_rewards)) if all_rewards else 0.0,
        "reward_std": float(np.std(all_rewards)) if len(all_rewards) > 1 else 0.0,
        "advantage_mean": float(np.mean(adv_arr)),
        "advantage_std": float(np.std(adv_arr)) if len(adv_arr) > 1 else 0.0,
    }


def rl_finetune(
    policy,
    env,
    num_iterations=50,
    episodes_per_iter=20,
    lr=3e-4,
    gamma=0.99,
    clip_grad_norm=1.0,
    max_steps=150,
    log_interval=5,
    kl_coeff=0.01,
    entropy_coeff=0.01,
    logger=None,
    task_name=None,
):
    """Fine-tune a BC-initialised symbolic policy with GRPO-style RL.

    Each iteration:
      1. Collect a *group* of trajectories with the current policy.
      2. Compute advantages relative to the group mean (GRPO).
      3. Update the policy via policy gradient weighted by advantage.
      4. Apply entropy bonus and L2 KL penalty toward BC initialisation.

    Args:
        policy: SymbolicPolicy (should be BC-initialised from Stage 2).
        env: MetaWorld environment.
        num_iterations: Outer RL iterations.
        episodes_per_iter: Trajectories per iteration (the "group").
        lr: Adam learning rate.
        gamma: Discount factor.
        clip_grad_norm: Max gradient norm.
        max_steps: Max steps per episode.
        log_interval: Print every N iterations.
        kl_coeff: L2 KL penalty toward BC params.
        entropy_coeff: Entropy bonus weight.
        logger: Optional TrainingLogger for structured metrics.
        task_name: Task identifier for logging (e.g. env_name).

    Returns:
        dict with training statistics.

    Note:
        GRPO has no value function; rl/value_loss is not logged.
        Policy ratio (π_new/π_old) is not computed; only L2 KL to BC params.
        No gradient accumulation: one backward and one optimizer step per
        iteration; logged grad norms are per backward pass (same as per step).
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    bc_params = {
        name: param.detach().clone() for name, param in policy.named_parameters()
    }

    stats = {
        "returns": [],
        "successes": [],
        "policy_loss": [],
        "entropy": [],
        "kl": [],
    }
    best_success_rate = -1.0
    best_params = copy.deepcopy(policy.state_dict())

    for iteration in range(num_iterations):
        trajectories = _collect_trajectories(policy, env, episodes_per_iter, max_steps)
        trajectories = _compute_grpo_advantages(trajectories, gamma)

        iter_returns = [t["return"] for t in trajectories]
        iter_successes = [t["success"] for t in trajectories]
        stats["returns"].extend(iter_returns)
        stats["successes"].extend(iter_successes)

        all_obs, all_pretanh, all_advantages = [], [], []
        for traj in trajectories:
            all_obs.extend(traj["obs"])
            all_pretanh.extend(traj["pretanh"])
            all_advantages.extend(traj["advantages"])

        obs_t = torch.tensor(np.array(all_obs), dtype=torch.float32)
        pretanh_t = torch.tensor(np.array(all_pretanh), dtype=torch.float32)
        advantages_t = torch.tensor(np.array(all_advantages), dtype=torch.float32)
        actions_t = torch.tanh(pretanh_t)

        policy.train()
        mean, std = policy(obs_t)
        dist = torch.distributions.Normal(mean, std)

        log_probs = dist.log_prob(pretanh_t) - torch.log(1 - actions_t.pow(2) + 1e-6)
        log_probs = log_probs.sum(dim=-1)

        policy_loss = -(log_probs * advantages_t).mean()

        entropy = dist.entropy().mean()

        kl_loss = sum(
            ((param - bc_params[name]) ** 2).sum()
            for name, param in policy.named_parameters()
        )

        total_loss = policy_loss - entropy_coeff * entropy + kl_coeff * kl_loss
        entropy_bonus = entropy_coeff * entropy.item()
        kl_penalty = kl_coeff * kl_loss.item()

        optimizer.zero_grad()
        total_loss.backward()

        # Measure global gradient norm before any clipping (policy.parameters() matches
        # the optimizer param set). Used to diagnose GRPO stability; large norms may
        # indicate instability. Skips params with no gradients.
        rl_params = list(policy.parameters())
        grad_norm_pre = _grad_norm(rl_params)

        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(rl_params, clip_grad_norm)
            grad_norm_post = min(grad_norm_pre, clip_grad_norm)
        else:
            grad_norm_post = grad_norm_pre

        optimizer.step()
        policy.clip_params()

        stats["policy_loss"].append(policy_loss.item())
        stats["entropy"].append(entropy.item())
        stats["kl"].append(kl_loss.item())

        # Structured RL logging: losses, GRPO terms, and optimization diagnostics
        if logger is not None:
            traj_metrics = _rl_metrics_from_trajectories(trajectories, all_advantages)
            metrics = {
                RL_TOTAL_LOSS: total_loss.item(),
                RL_POLICY_LOSS: policy_loss.item(),
                RL_ENTROPY: entropy.item(),
                RL_KL: kl_loss.item(),
                RL_ENTROPY_BONUS: entropy_bonus,
                RL_KL_PENALTY: kl_penalty,
                RL_GRAD_NORM_PRE_CLIP: grad_norm_pre,
                RL_GRAD_NORM_POST_CLIP: grad_norm_post,
                RL_LEARNING_RATE: lr,
                RL_RETURN_MEAN: traj_metrics.get("return_mean", 0.0),
                RL_RETURN_STD: traj_metrics.get("return_std", 0.0),
                RL_REWARD_MEAN: traj_metrics.get("reward_mean", 0.0),
                RL_REWARD_STD: traj_metrics.get("reward_std", 0.0),
                RL_ADVANTAGE_MEAN: traj_metrics.get("advantage_mean", 0.0),
                RL_ADVANTAGE_STD: traj_metrics.get("advantage_std", 0.0),
                RL_SUCCESS_RATE: float(np.mean(iter_successes)),
            }
            logger.log_metrics(
                stage="rl",
                update=iteration,
                metrics=metrics,
                task_name=task_name,
            )

        success_rate = np.mean(iter_successes)
        if success_rate >= best_success_rate:
            best_success_rate = success_rate
            best_params = copy.deepcopy(policy.state_dict())

        if log_interval > 0 and (iteration + 1) % log_interval == 0:
            recent_r = stats["returns"][-episodes_per_iter * log_interval :]
            recent_s = stats["successes"][-episodes_per_iter * log_interval :]
            print(
                f"  [Stage 3] iter {iteration + 1}/{num_iterations}: "
                f"return={np.mean(recent_r):.2f}, "
                f"success={np.mean(recent_s):.2f}, "
                f"loss={np.mean(stats['policy_loss'][-log_interval:]):.4f}"
            )

    policy.load_state_dict(best_params)
    stats["best_success_rate"] = float(best_success_rate)
    stats["final_mean_return"] = float(np.mean(stats["returns"][-episodes_per_iter:]))
    return stats


# ===========================================================================
#  Evaluation helper
# ===========================================================================


def evaluate_policy(policy, env, num_episodes=10, max_steps=150):
    """Evaluate a symbolic policy and return mean reward / success rate."""
    total_reward = 0.0
    total_success = 0.0

    policy.eval()
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        success = False

        for step in range(max_steps):
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mean, std = policy(obs_t)
            action = torch.tanh(mean).squeeze(0).numpy()

            next_obs, reward, done, info = env.step(env.action_space.high * action)
            episode_reward += reward
            if info.get("success", 0) > 0:
                success = True
            obs = next_obs
            if done:
                break

        total_reward += episode_reward
        total_success += float(success)

    return {
        "mean_reward": total_reward / num_episodes,
        "success_rate": total_success / num_episodes,
    }


def ensure_mujoco_headless_gl():
    """Use EGL for MuJoCo offscreen rendering when there is no X display (SSH / batch).

    Set ``MUJOCO_GL`` before constructing MetaWorld / ``MujocoEnv`` or before the first
    ``rgb_array`` render; otherwise GLFW looks for ``DISPLAY`` and OpenGL init fails.
    Safe to call repeatedly; does not override an existing ``MUJOCO_GL``.
    """
    if os.environ.get("DISPLAY"):
        return
    if os.environ.get("MUJOCO_GL"):
        return
    os.environ["MUJOCO_GL"] = "egl"


def _coerce_rgb_hwc_uint8(frame):
    """Return ``(H, W, 3)`` uint8 array or ``None`` if ``frame`` is not a valid RGB image."""
    if isinstance(frame, np.ndarray) and frame.ndim == 3:
        return frame.astype(np.uint8, copy=False)
    return None


def _render_kw_attempts(render_fn):
    """Build an ordered list of keyword dicts to pass to ``render`` (``{}`` means no kwargs).

    Avoids guessing the API via exceptions. MetaWorld (Farama) follows Gymnasium: set
    ``render_mode='rgb_array'`` when constructing the env, then call ``render()`` with
    no arguments; pixels are an ``(H, W, 3)`` uint8 array. See Gymnasium ``Env.render``:
    https://gymnasium.farama.org/api/env/

    Classic OpenAI Gym used ``render(mode='rgb_array')`` instead. Wrappers such as
    rlkit ``ProxyEnv`` expose ``render(*args, **kwargs)`` and forward to the inner env.

    Do **not** pass ``mode='rgb_array'`` through ``*args, **kwargs`` forwarders: OpenAI
    Gym's ``gym.Wrapper.render`` forwards ``mode`` as a *positional* argument to the
    child (``env.render(mode, ...)``). Gymnasium ``MujocoEnv.render`` only accepts
    ``self``, so that call raises ``TypeError``. MetaWorld + ``render_mode='rgb_array'``
    only needs a no-arg ``render()`` through the stack.
    """
    try:
        sig = inspect.signature(render_fn)
    except (TypeError, ValueError):
        # Built-in or C extension: only a no-arg call is well-defined.
        return [{}]

    params = sig.parameters
    has_varkw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

    def _explicit(name):
        if name not in params:
            return False
        k = params[name].kind
        return k in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )

    has_mode = _explicit("mode")
    has_offscreen = _explicit("offscreen")

    # Gymnasium ``Env.render(self)`` — no mode/offscreen parameters on the method.
    if not has_mode and not has_offscreen and not has_varkw:
        return [{}]

    # Classic gym ``render(self, mode='human', ...)`` without ``**kwargs``.
    if has_mode and not has_varkw:
        attempts = [{"mode": "rgb_array"}]
        if has_offscreen:
            attempts.append({"offscreen": True})
        return attempts

    # MuJoCo-py style without a ``mode`` keyword on the signature.
    if has_offscreen and not has_mode and not has_varkw:
        return [{"offscreen": True}, {}]

    # ``*args, **kwargs`` forwarders: only no-arg ``render()``. Do not pass
    # ``offscreen`` here: the forwarder accepts it, but Gymnasium leaf envs like
    # ``MujocoEnv.render(self)`` do not, so forwarding raises ``TypeError``.
    # Leaf envs that need ``offscreen`` should declare it (handled above).
    return [{}]


def _unwrap_env_leaf(env):
    """Return the innermost env along common wrapper links.

    OpenAI Gym ``gym.Wrapper.render`` passes ``mode`` positionally to the child
    (default ``'human'``), which breaks Gymnasium ``MujocoEnv.render(self)``.
    GIF capture must call ``render()`` on the real task env, not on
    ``TimeLimit`` / ``Wrapper`` in between.
    """
    cur = env
    for _ in range(256):
        nxt = None
        u = getattr(cur, "unwrapped", None)
        if u is not None and u is not cur:
            cur = u
            continue
        for attr in ("_env", "env", "_wrapped_env"):
            child = getattr(cur, attr, None)
            if child is not None and child is not cur:
                nxt = child
                break
        if nxt is None:
            return cur
        cur = nxt
    return cur


def _grab_frame_from_env(env_leaf):
    """Return RGB uint8 array from ``env_leaf`` or ``None`` if no valid frame."""
    leaf = _unwrap_env_leaf(env_leaf)
    render_fn = getattr(leaf, "render", None)
    if not callable(render_fn):
        return None
    for kwargs in _render_kw_attempts(render_fn):
        frame = render_fn(**kwargs)
        arr = _coerce_rgb_hwc_uint8(frame)
        if arr is not None:
            return arr
    return None


def record_episode_gif(
    policy,
    env,
    path,
    max_steps=150,
    fps=20,
    verbose=True,
    deterministic=False,
):
    """Run one deterministic episode and save it as a GIF.

    Unwraps to the innermost env before calling ``render``, so OpenAI Gym
    ``Wrapper`` layers (e.g. ``TimeLimit``) are skipped — their ``render`` passes
    ``mode`` positionally and breaks Gymnasium MetaWorld envs.

    Uses signature-based kwargs (Gymnasium: ``render()`` with ``render_mode`` set at
    construction; classic Gym: ``mode='rgb_array'``). If every attempt returns a
    non-array or ``None``, no GIF is written.

    Calls :func:`ensure_mujoco_headless_gl` at entry so SSH runs pick EGL before the
    first frame. MuJoCo / OpenGL errors from ``render`` propagate to the caller unless
    a caller (e.g. :class:`EvolutionOrchestrator`) catches them.

    Args:
        policy: Trained policy (will be placed in eval mode).
        env: MetaWorld environment (raw or wrapped).
        path: Destination ``.gif`` file path.
        max_steps: Maximum steps per episode.
        fps: Playback frame rate of the saved GIF (converted to per-frame
            ``duration`` in ms for imageio/Pillow; must be > 0).
        verbose: If True, print capture/write diagnostics (set False in unit tests).
        deterministic: If True, use the policy mean as the action (clipped to
            ``[-1, 1]``) with no Gaussian sampling. Matches Stage 1 expert rollout
            when the policy forward returns expert actions in that range.

    Returns:
        dict with keys ``episode_reward`` (float), ``success`` (bool),
        ``num_frames`` (int), ``saved`` (bool).  ``saved`` is ``False`` when
        no frames could be captured or ``imageio`` is unavailable.
    """
    import imageio

    ensure_mujoco_headless_gl()

    frames = []
    policy.eval()

    try:
        obs, _ = env.reset()
    except TypeError:
        obs = env.reset()

    episode_reward = 0.0
    success = False

    for _ in range(max_steps):
        frame = _grab_frame_from_env(env)
        if frame is not None:
            frames.append(frame)

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean, std = policy(obs_t)
            if deterministic:
                action = (
                    mean.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
                )
                action = np.clip(action, -1.0, 1.0)
            else:
                dist = torch.distributions.Normal(mean, std)
                pretanh_action = dist.sample()
                action = (
                    torch.clamp(pretanh_action, -0.999999, 0.999999)
                    .squeeze(0)
                    .numpy()
                )
        try:
            next_obs, reward, done, info = env.step(env.action_space.high * action)
        except Exception as e:
            if verbose:
                print(f"  [record_gif] step failed: {e!r}; action={action!r}")
            break

        episode_reward += reward
        if info.get("success", 0) > 0:
            success = True
        obs = next_obs
        if done:
            break

    saved = False
    if frames:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        try:
            # Pillow plugin deprecated ``fps``; duration is ms per frame.
            duration_ms = 1000.0 / float(fps)
            imageio.mimsave(path, frames, duration=duration_ms)
            saved = True
        except Exception as exc:
            if verbose:
                print(f"  [record_gif] Failed to write {path}: {exc}")
    else:
        if verbose:
            print("  [record_gif] No frames captured — check render mode support.")

    return {
        "episode_reward": episode_reward,
        "success": success,
        "num_frames": len(frames),
        "saved": saved,
    }


# ===========================================================================
#  Stage 4 — LLM-Based Evolution / Structure Search
# ===========================================================================


def load_policy_prompt_assets(env_name):
    """Load policy prompt files and per-task strings for LLM policy generation."""
    from lares.core.policy_generation import (
        obs_description_dict,
        input_dict_for_policy,
    )

    root_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    prompt_dir = os.path.join(root_dir, "lares", "utils", "policy_prompts")

    def _read(filename):
        with open(os.path.join(prompt_dir, filename), "r", encoding="utf-8") as f:
            return f.read()

    return {
        "initial_system": _read("initial_system.txt"),
        "initial_user": _read("new_initial_user.txt"),
        "code_output_tip": _read("new_code_output_tip.txt"),
        "code_feedback_tmpl": _read("code_feedback.txt"),
        "ideas_system": _read("ideas_system.txt"),
        "ideas_user": _read("ideas_user.txt"),
        "task_description": TASK_DESCRIPTIONS.get(env_name, env_name),
        "obs_description": obs_description_dict.get(env_name, ""),
        "input_dict_string": input_dict_for_policy.get(env_name, ""),
    }


def bootstrap_symbolic_policy_from_llm(
    client,
    env_name,
    obs_dim,
    action_dim,
    args,
    log_dir,
    llm_transcript_path=None,
):
    """Generate one validated symbolic policy via the LLM (no hand-crafted policy).

    If ``args.policy_gen_two_phase`` is true, uses ideation then implementation
    (see :func:`~lares.core.policy_generation.get_symbolic_policies`).

    Returns:
        (policy, code, response_text) or (None, None, None) if generation failed.
    """
    from lares.core.policy_generation import get_symbolic_policies

    os.makedirs(log_dir, exist_ok=True)
    data_pkl_path = os.path.join(log_dir, "data.pkl")
    with open(data_pkl_path, "wb") as f:
        pickle.dump([{"obs": np.zeros(obs_dim)}], f)

    prompts = load_policy_prompt_assets(env_name)
    policy_pop, code_pop, resp_list = get_symbolic_policies(
        client=client,
        dir_path=log_dir,
        llm_iter=0,
        args=args,
        obs_dim=obs_dim,
        action_dim=action_dim,
        initial_system=prompts["initial_system"],
        initial_user=prompts["initial_user"],
        task_description=prompts["task_description"],
        obs_description=prompts["obs_description"],
        input_dict_string=prompts["input_dict_string"],
        code_output_tip=prompts["code_output_tip"],
        data_pkl_path=data_pkl_path,
        real_num=1,
        llm_transcript_path=llm_transcript_path,
        ideas_system=prompts.get("ideas_system"),
        ideas_user=prompts.get("ideas_user"),
    )
    if not policy_pop:
        return None, None, None
    return policy_pop[0], code_pop[0], resp_list[0]


def llm_evolution(
    client,
    env_name,
    obs_dim,
    action_dim,
    args,
    previous_results=None,
    pop_size=5,
    generation=0,
    log_dir="./logs/evolution",
    logger=None,
    llm_transcript_path=None,
):
    """Propose a new population of symbolic policy structures via the LLM.

    Takes performance data from previously trained models as feedback and
    generates ``pop_size`` new candidate policy structures.  This function is
    responsible only for LLM-based structure search; BC and RL training are
    the caller's responsibility (see :class:`EvolutionOrchestrator`).

    Args:
        client: OpenAI client.
        env_name: MetaWorld task identifier.
        obs_dim: Observation dimensionality.
        action_dim: Action dimensionality.
        args: Namespace with at least a ``model`` attribute.  Optional:
            ``policy_gen_two_phase`` (bool), ``policy_impl_mode`` (``"batched"``
            or ``"per_idea"``) forwarded to :func:`~lares.core.policy_generation.get_symbolic_policies`.
        previous_results: List of dicts, each with keys ``code`` (str),
            ``eval`` (dict with ``mean_reward`` and ``success_rate``),
            ``score`` (float), ``response`` (str), and optionally
            ``train_steps`` (int).  Pass ``None`` or an empty list for the
            first generation to skip performance feedback.
        pop_size: Number of candidate policies to generate.
        generation: Current generation index (used for artefact naming).
        log_dir: Directory for generation artefacts.
        logger: Optional TrainingLogger.
        llm_transcript_path: Optional path to append LLM call transcript.

    Returns:
        Tuple ``(policy_pop, code_pop, response_pop)`` of untrained candidate
        policies, their code strings, and raw LLM response texts.
    """
    from lares.core.policy_generation import get_symbolic_policies

    os.makedirs(log_dir, exist_ok=True)
    prompts = load_policy_prompt_assets(env_name)

    # Build LLM feedback from the best previous result (if any)
    elite_response = None
    code_feedback = None
    if previous_results:
        best_prev = max(previous_results, key=lambda r: r["score"])
        elite_response = best_prev["response"]
        code_feedback = prompts["code_feedback_tmpl"].format(
            train_steps=best_prev.get("train_steps", 0),
            win_rate=best_prev["eval"]["success_rate"],
            mean_reward=best_prev["eval"]["mean_reward"],
            current_output=str(best_prev["eval"]),
        )

    data_pkl_path = os.path.join(log_dir, "data.pkl")
    with open(data_pkl_path, "wb") as f:
        pickle.dump([{"obs": np.zeros(obs_dim)}], f)

    gen_dir = os.path.join(log_dir, f"gen_{generation}")
    os.makedirs(gen_dir, exist_ok=True)

    policy_pop, code_pop, response_pop = get_symbolic_policies(
        client=client,
        dir_path=gen_dir,
        llm_iter=generation,
        args=args,
        obs_dim=obs_dim,
        action_dim=action_dim,
        initial_system=prompts["initial_system"],
        initial_user=prompts["initial_user"],
        task_description=prompts["task_description"],
        obs_description=prompts["obs_description"],
        input_dict_string=prompts["input_dict_string"],
        code_output_tip=prompts["code_output_tip"],
        data_pkl_path=data_pkl_path,
        provided_response=elite_response,
        code_feedback=code_feedback,
        real_num=pop_size,
        llm_transcript_path=llm_transcript_path,
        ideas_system=prompts.get("ideas_system"),
        ideas_user=prompts.get("ideas_user"),
    )

    if logger is not None:
        logger.log_metrics(
            stage="evolutionary",
            update=generation,
            metrics={"evo/candidates_generated": float(len(policy_pop))},
            task_name=env_name,
        )

    return policy_pop, code_pop, response_pop


# ===========================================================================
#  EvolutionOrchestrator — full generation → train → evaluate loop
# ===========================================================================


class EvolutionOrchestrator:
    """Full LLM evolution loop: generation → train (BC + RL) → evaluate → repeat.

    Separates LLM structure search (:func:`llm_evolution`) from the inner
    training loop so each concern is independently configurable and testable.

    Each generation:
      1. Call :func:`llm_evolution` with performance history → new candidates.
      2. Train each candidate through BC → RL.
      3. Evaluate and rank candidates.
      4. Feed top-``elite_num`` results back to next generation.

    Usage::

        orchestrator = EvolutionOrchestrator(
            env_name='reach-v2',
            obs_dim=39,
            action_dim=4,
            num_generations=5,
            pop_size=5,
            elite_num=2,
            bc_steps=3000,
            rl_iterations=30,
            rl_episodes_per_iter=10,
            log_dir='./logs/evolution',
            record_demo_gif=True,
        )
        best = orchestrator.run(client, env, demo_buffer, args)
    """

    def __init__(
        self,
        env_name,
        obs_dim=39,
        action_dim=4,
        num_generations=5,
        pop_size=5,
        elite_num=2,
        bc_steps=3000,
        rl_iterations=30,
        rl_episodes_per_iter=10,
        log_dir="./logs/evolution",
        record_demo_gif=True,
    ):
        self.env_name = env_name
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_generations = num_generations
        self.pop_size = pop_size
        self.elite_num = elite_num
        self.bc_steps = bc_steps
        self.rl_iterations = rl_iterations
        self.rl_episodes_per_iter = rl_episodes_per_iter
        self.log_dir = log_dir
        self.record_demo_gif = record_demo_gif

        # Populated by run()
        self.history: list = []
        self.best_overall: dict | None = None

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    def _train_candidate(self, policy, demo_buffer, env, logger=None):
        """Run the BC → RL inner loop on a single policy candidate."""
        bc_stats = behavioral_cloning(
            policy,
            demo_buffer,
            num_steps=self.bc_steps,
            batch_size=min(256, len(demo_buffer)),
            log_interval=self.bc_steps,
            logger=logger,
            task_name=self.env_name,
        )
        rl_stats = rl_finetune(
            policy,
            env,
            num_iterations=self.rl_iterations,
            episodes_per_iter=self.rl_episodes_per_iter,
            log_interval=self.rl_iterations,
            logger=logger,
            task_name=self.env_name,
        )
        return bc_stats, rl_stats

    @staticmethod
    def _score(eval_result):
        """Composite fitness: success rate dominates, mean reward breaks ties."""
        return eval_result["success_rate"] * 1000 + eval_result["mean_reward"]

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def run(self, client, env, demo_buffer, args, logger=None, eval_episodes=10):
        """Execute the full evolution loop.

        Args:
            client: OpenAI client.
            env: MetaWorld environment.
            demo_buffer: :class:`DemoBuffer` with expert demonstrations.
            args: Namespace with ``model`` attribute for the LLM.
            logger: Optional :class:`~lares.core.training_logger.TrainingLogger`.
            eval_episodes: Number of evaluation episodes per candidate.

        Returns:
            Dict with ``policy``, ``code``, ``score``, ``eval``, ``response``
            for the best candidate found across all generations.  Returns
            ``None`` if no valid candidate was ever generated.
        """
        if logger is None:
            from lares.core.training_logger import TrainingLogger

            logger = TrainingLogger(log_dir=self.log_dir, task_name=self.env_name)

        os.makedirs(self.log_dir, exist_ok=True)
        llm_transcript_path = os.path.join(self.log_dir, "llm_evolution_transcript.log")
        with open(llm_transcript_path, "w", encoding="utf-8") as _tf:
            _tf.write(
                f"LLM evolution transcript\n"
                f"task={self.env_name}\n"
                f"model={getattr(args, 'model', '')}\n"
                f"policy_gen_two_phase={getattr(args, 'policy_gen_two_phase', False)}\n"
                f"policy_impl_mode={getattr(args, 'policy_impl_mode', 'batched')}\n"
                f"---\n"
            )
        print(f"  LLM message log: {llm_transcript_path}")

        ensure_mujoco_headless_gl()

        previous_results: list = []

        for gen in range(self.num_generations):
            print(f"\n{'=' * 60}")
            print(f"[Evolution] Generation {gen + 1}/{self.num_generations}")
            print(f"{'=' * 60}")

            gen_dir = os.path.join(self.log_dir, f"gen_{gen}")

            # --- Structure search: LLM proposes new untrained candidates ---
            policy_pop, code_pop, response_pop = llm_evolution(
                client=client,
                env_name=self.env_name,
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                args=args,
                previous_results=previous_results,
                pop_size=self.pop_size,
                generation=gen,
                log_dir=self.log_dir,
                logger=logger,
                llm_transcript_path=llm_transcript_path,
            )

            # --- Inner loop: train and evaluate every candidate ---
            gen_results = []
            for i, (policy, code, response) in enumerate(
                zip(policy_pop, code_pop, response_pop)
            ):
                print(f"\n  --- Candidate {i + 1}/{len(policy_pop)} ---")
                bc_stats, rl_stats = self._train_candidate(
                    policy, demo_buffer, env, logger=logger
                )
                eval_result = evaluate_policy(policy, env, num_episodes=eval_episodes)
                score = self._score(eval_result)
                train_steps = (
                    self.bc_steps + self.rl_iterations * self.rl_episodes_per_iter * 150
                )
                result = {
                    "policy": policy,
                    "code": code,
                    "response": response,
                    "eval": eval_result,
                    "score": score,
                    "bc_stats": bc_stats,
                    "rl_stats": rl_stats,
                    "train_steps": train_steps,
                    "generation": gen,
                    "candidate_idx": i,
                }
                gen_results.append(result)
                self.history.append(result)
                print(
                    f"    reward={eval_result['mean_reward']:.2f}, "
                    f"success={eval_result['success_rate']:.2f}"
                )

            if not gen_results:
                print(f"  WARNING: Generation {gen + 1} produced no valid candidates.")
                continue

            gen_results.sort(key=lambda r: r["score"], reverse=True)

            # Log generation-level fitness metrics
            scores = [r["score"] for r in gen_results]
            elite_scores = (
                [r["score"] for r in gen_results[: self.elite_num]]
                if self.elite_num > 0
                else scores
            )
            logger.log_metrics(
                stage="evolutionary",
                update=gen,
                metrics={
                    EVO_FITNESS_MEAN: float(np.mean(scores)),
                    EVO_FITNESS_BEST: float(max(scores)),
                    EVO_FITNESS_MEDIAN: float(np.median(scores)),
                    EVO_FITNESS_WORST: float(min(scores)),
                    EVO_FITNESS_ELITE_MEAN: float(np.mean(elite_scores)),
                },
                task_name=self.env_name,
            )

            print(f"\n  Generation {gen + 1} ranking:")
            for rank, r in enumerate(gen_results):
                tag = " (elite)" if rank < self.elite_num else ""
                print(
                    f"    [{rank + 1}] reward={r['eval']['mean_reward']:.2f}, "
                    f"success={r['eval']['success_rate']:.2f}{tag}"
                )

            # Track global best
            if (
                self.best_overall is None
                or gen_results[0]["score"] > self.best_overall["score"]
            ):
                self.best_overall = {
                    "policy": copy.deepcopy(gen_results[0]["policy"]),
                    "code": gen_results[0]["code"],
                    "score": gen_results[0]["score"],
                    "eval": gen_results[0]["eval"],
                    "response": gen_results[0]["response"],
                }

            # Record a demo GIF of the best candidate from this generation
            if self.record_demo_gif:
                gif_path = os.path.join(gen_dir, f"gen_{gen}_best.gif")
                print(f"\n  Recording demo GIF → {gif_path}")
                try:
                    gif_info = record_episode_gif(
                        gen_results[0]["policy"], env, path=gif_path
                    )
                except Exception as exc:
                    print(
                        f"  [record_gif] Skipped ({type(exc).__name__}: {exc}). "
                        "Set MUJOCO_GL=egl (or osmesa) before creating the env on headless nodes."
                    )
                    gif_info = {
                        "saved": False,
                        "num_frames": 0,
                        "episode_reward": 0.0,
                        "success": False,
                    }
                if gif_info["saved"]:
                    print(
                        f"  GIF saved ({gif_info['num_frames']} frames, "
                        f"reward={gif_info['episode_reward']:.2f}, "
                        f"success={gif_info['success']})"
                    )

            # Feed elite results back to the next generation
            previous_results = [
                {
                    "code": r["code"],
                    "eval": r["eval"],
                    "score": r["score"],
                    "response": r["response"],
                    "train_steps": r["train_steps"],
                }
                for r in gen_results[: self.elite_num]
            ]

            # Persist generation artefacts
            with open(os.path.join(gen_dir, "results.pkl"), "wb") as f:
                pickle.dump(
                    {
                        "generation": gen,
                        "candidates": [
                            {"code": r["code"], "eval": r["eval"], "score": r["score"]}
                            for r in gen_results
                        ],
                    },
                    f,
                )

        return self.best_overall

# dummy ti escapde testfile error---

class SymbolicPolicyPipeline:
    """Backward-compatible wrapper for older tests/code paths."""

    def __init__(self, env_name, obs_dim=39, action_dim=4):
        self.env_name = env_name
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.demo_buffer = None
        self.best_policy = None
        self.best_code = None

    def stage1_generate_dataset(self, env, num_episodes=100, **kwargs):
        self.demo_buffer, stats = generate_dataset(
            env, self.env_name, num_episodes=num_episodes, **kwargs
        )
        return self.demo_buffer, stats

    def stage2_behavioral_cloning(self, policy, demo_buffer=None, **kwargs):
        buf = demo_buffer if demo_buffer is not None else self.demo_buffer
        if buf is None:
            raise ValueError("No demo buffer available. Run stage1 first or pass demo_buffer.")
        return behavioral_cloning(policy, buf, **kwargs)

    def stage3_rl_finetune(self, policy, env, **kwargs):
        return rl_finetune(policy, env, **kwargs)

    def run_single_policy(
        self,
        policy,
        env,
        demo_buffer=None,
        bc_steps=5000,
        rl_iterations=50,
        rl_episodes=20,
    ):
        bc_stats = self.stage2_behavioral_cloning(
            policy,
            demo_buffer=demo_buffer,
            num_steps=bc_steps,
        )
        rl_stats = self.stage3_rl_finetune(
            policy,
            env,
            num_iterations=rl_iterations,
            episodes_per_iter=rl_episodes,
        )
        eval_result = evaluate_policy(policy, env)
        return {
            "bc_stats": bc_stats,
            "rl_stats": rl_stats,
            "eval": eval_result,
        }
