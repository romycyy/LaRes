"""
Four-stage training pipeline for symbolic policies in modified LaRes.

Stage 1 — Dataset Generation:  use MetaWorld expert policies to collect demos
Stage 2 — Behavioral Cloning:  supervised imitation of expert actions
Stage 3 — RL Fine-tuning:      GRPO-style policy gradient improvement
Stage 4 — LLM Evolution:       structure search with BC+RL inner loop

Each stage is usable independently or orchestrated via
:class:`EvolutionOrchestrator`.
"""

import inspect
import json
import os
import copy
import pickle
import time
from collections import defaultdict
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn

from lares.utils.hashing import sha256_file, sha256_text

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
        # Which episode each transition came from. MetaWorld never terminates
        # early, so ``dones`` is all zeros and cannot mark episode boundaries;
        # without this a train/validation split can only be by transition,
        # which leaks neighbouring frames across the split (spec.md FR-6).
        self.episode_ids = []

    def add(self, obs, action, reward, next_obs, done, episode_id=""):
        self.obs.append(np.array(obs, dtype=np.float32))
        self.actions.append(np.array(action, dtype=np.float32))
        self.rewards.append(float(reward))
        self.next_obs.append(np.array(next_obs, dtype=np.float32))
        self.dones.append(float(done))
        self.episode_ids.append(str(episode_id))

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

    def sample(self, batch_size, indices=None):
        """Sample a random mini-batch, optionally restricted to ``indices``.

        ``indices`` is how a training split stays a training split: without it a
        validation episode's transitions leak back into every mini-batch.
        """
        if indices is None:
            idxes = np.random.randint(0, len(self), size=batch_size)
        else:
            pool = np.asarray(indices)
            idxes = pool[np.random.randint(0, len(pool), size=batch_size)]
        return (
            np.array([self.obs[i] for i in idxes]),
            np.array([self.actions[i] for i in idxes]),
            np.array([self.rewards[i] for i in idxes]),
            np.array([self.next_obs[i] for i in idxes]),
            np.array([self.dones[i] for i in idxes]),
        )

    def take(self, indices):
        """``(obs, actions)`` arrays for the given transition indices."""
        idx = np.asarray(indices)
        return (
            np.array([self.obs[i] for i in idx]),
            np.array([self.actions[i] for i in idx]),
        )

    def episode_index(self):
        """``{episode_id: [transition indices]}`` in insertion order."""
        groups = defaultdict(list)
        for i, ep in enumerate(self.episode_ids):
            groups[ep].append(i)
        return dict(groups)

    def save(self, path):
        data = {
            "obs": np.array(self.obs),
            "actions": np.array(self.actions),
            "rewards": np.array(self.rewards),
            "next_obs": np.array(self.next_obs),
            "dones": np.array(self.dones),
            "episode_ids": np.array(self.episode_ids, dtype=object),
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
        # Buffers written before episode ids were recorded load with blanks, so
        # a trajectory-level split of them is refused rather than faked.
        buf.episode_ids = [str(e) for e in data.get("episode_ids", [""] * len(buf.obs))]
        return buf


# ===========================================================================
#  Stage 1 — Dataset Generation
# ===========================================================================


def generate_dataset(env, env_name, cases, max_steps=150):
    """Collect expert demonstrations on an explicit list of training placements.

    Args:
        env: MetaWorld environment (wrapped with env_wrapper), dedicated to
            dataset collection so it shares no state with evaluation.
        env_name: Task identifier (e.g. 'window-close-v2').
        cases: ordered ``EpisodeCase`` sequence, normally the training manifest.
            Collection must never touch development or final-test placements.
        max_steps: Maximum steps per episode.

    Returns:
        (DemoBuffer, stats_dict)
    """
    expert = get_expert_policy(env_name)
    buffer = DemoBuffer()
    stats = defaultdict(list)
    cases = list(cases)
    num_episodes = len(cases)
    if num_episodes == 0:
        raise ValueError("generate_dataset needs at least one case")

    for ep, case in enumerate(cases):
        obs, _ = env.reset(case)
        episode_reward = 0
        success = False

        for step in range(max_steps):
            action = expert.get_action(obs)
            action_clipped = np.clip(action, -1.0, 1.0)

            next_obs, reward, done, info = env.step(
                env.action_space.high * action_clipped
            )

            buffer.add(
                obs,
                action_clipped,
                reward,
                next_obs,
                float(done),
                episode_id=case.case_id,
            )
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


def split_buffer_by_episode(demo_buffer, val_fraction=0.2, seed=0):
    """Split transitions into train and validation by *complete episode*.

    Splitting by transition would put neighbouring frames of the same rollout on
    both sides, so validation loss would measure interpolation within a
    trajectory rather than generalisation to a new placement (``spec.md`` FR-6).

    Returns ``(train_idx, val_idx, info)``. ``val_idx`` is ``None`` when the
    buffer carries no episode ids, and ``info`` says why rather than silently
    falling back to a transition split.
    """
    groups = demo_buffer.episode_index()
    unlabelled = [k for k in groups if not k]
    if unlabelled or len(groups) < 2:
        return (
            np.arange(len(demo_buffer)),
            None,
            {
                "mode": "none",
                "reason": (
                    "buffer has no episode ids, so a trajectory-level split is not "
                    "possible; it was collected before episode ids were recorded"
                    if unlabelled
                    else "buffer holds fewer than two episodes"
                ),
            },
        )

    episode_ids = sorted(groups)
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(episode_ids))
    n_val = max(1, int(round(val_fraction * len(episode_ids))))
    val_ids = {episode_ids[i] for i in order[:n_val]}

    train_idx, val_idx = [], []
    for ep, idx in groups.items():
        (val_idx if ep in val_ids else train_idx).extend(idx)
    return (
        np.asarray(sorted(train_idx)),
        np.asarray(sorted(val_idx)),
        {
            "mode": "by_episode",
            "seed": int(seed),
            "val_fraction": float(val_fraction),
            "train_episodes": len(episode_ids) - len(val_ids),
            "val_episodes": len(val_ids),
            "train_transitions": len(train_idx),
            "val_transitions": len(val_idx),
        },
    )


def _per_axis_mse(policy, obs_np, actions_np):
    from lares.fitting.diagnostics import per_axis_mse

    return per_axis_mse(policy, obs_np, actions_np)


def _bound_activity(policy, atol=1e-6):
    from lares.fitting.diagnostics import bound_activity

    return bound_activity(policy, atol)


def _loss_by_phase(policy, obs_np, actions_np, threshold=0.5):
    from lares.fitting.diagnostics import loss_by_phase

    return loss_by_phase(policy, obs_np, actions_np, threshold)


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
    val_fraction=0.2,
    split_seed=0,
    val_interval=0,
    checkpoint_callback=None,
    rollout_probe=None,
    rollout_interval=0,
):
    """Train a symbolic policy to imitate expert actions via supervised learning.

    The live objective is ``MSE(tanh(mean), expert_action) + 0.01 * mean(std)``.
    The mean term regresses the action that is actually executed, since every
    rollout squashes through ``tanh``; regressing the pre-tanh target instead
    sends saturated expert actions to ``atanh(+-1)``, outside any declared
    parameter range. The scale term is a downward regulariser, not a likelihood.
    ``spec.md`` FR-7 replaces it with a fixed positive scale in a later phase.

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
        val_fraction: Share of *episodes* held out for validation. Splitting by
            transition would leak neighbouring frames across the split.
        split_seed: Seed for the episode split, so it is reproducible.
        val_interval: Steps between validation evaluations; 0 evaluates only at
            the start and the end.
        checkpoint_callback: Optional ``fn(step, policy)`` called at the halfway
            point, used to capture the intermediate checkpoint AC-2 requires.
        rollout_probe: Optional ``fn(policy) -> comparable`` returning a
            development score, called every ``rollout_interval`` steps. The
            best-scoring parameters are kept in ``stats["best_rollout_state"]``.
            Returning a tuple gives a lexicographic tie-break; the orchestrator
            returns ``(success_rate, signed_goal_progress)`` so a generation where
            nothing succeeds is still ordered by something meaningful.

            This exists because development success is *not* monotonic in the
            number of gradient steps on this task, while validation loss is.
            Measured on two structures over three seeds, success peaks around
            1000 steps and then collapses to zero by 3000 while the loss keeps
            improving. Selecting on loss, or simply stopping at a fixed budget,
            walks past the useful parameters.

    Returns:
        dict with training statistics and the fitting diagnostics that
        ``EvaluationReport`` reports.
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    stats = {
        "bc_loss": [],
        "mean_loss": [],
        "std_loss": [],
        "val_loss": [],
        "rollout_history": [],
        "best_rollout_state": None,
        "best_rollout_score": None,
        "best_rollout_step": None,
    }

    train_idx, val_idx, split_info = split_buffer_by_episode(
        demo_buffer, val_fraction=val_fraction, seed=split_seed
    )
    stats["split"] = split_info
    stats["num_steps"] = int(num_steps)
    val_obs = val_actions = None
    if val_idx is not None and len(val_idx) > 0:
        val_obs, val_actions = demo_buffer.take(val_idx)
    train_obs, train_actions = demo_buffer.take(train_idx)
    halfway = max(1, num_steps // 2)

    policy.train()
    for step in range(num_steps):
        obs_np, actions_np, _, _, _ = demo_buffer.sample(batch_size, indices=train_idx)
        obs_t = torch.tensor(obs_np, dtype=torch.float32)
        actions_t = torch.tensor(actions_np, dtype=torch.float32)

        mean, std = policy(obs_t)

        # Fit the action that is actually executed: evaluate_policy / _collect_trajectories
        # squash through tanh, so BC must regress tanh(mean) onto the expert action.
        # Regressing the pre-tanh target instead sends saturated expert actions to
        # atanh(+-1) = +-7.25, far outside get_param_ranges() / clip_params().
        mean_loss = nn.functional.mse_loss(torch.tanh(mean), actions_t)
        std_loss = 0.01 * std.mean()
        loss = mean_loss + std_loss

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


        stats["bc_loss"].append(loss.item())
        stats["mean_loss"].append(mean_loss.item())
        stats["std_loss"].append(std_loss.item())
        stats.setdefault("grad_norms", []).append(grad_norm_pre)

        # Structured logging for training dynamics (task_name enables per-task plots)
        if logger is not None and step % log_every_n_steps == 0:
            logger.log_metrics(
                stage="bc",
                update=step,
                metrics={
                    BC_TRAIN_LOSS: loss.item(),
                    BC_MEAN_LOSS: mean_loss.item(),
                    BC_STD_LOSS: std_loss.item(),
                    BC_GRAD_NORM_PRE_CLIP: float(grad_norm_pre),
                    BC_GRAD_NORM_POST_CLIP: float(grad_norm_post),
                },
                task_name=task_name,
            )

        if val_obs is not None and val_interval > 0 and (step + 1) % val_interval == 0:
            policy.eval()
            with torch.no_grad():
                v_mean, _ = policy(torch.tensor(val_obs, dtype=torch.float32))
                v_loss = nn.functional.mse_loss(
                    torch.tanh(v_mean), torch.tensor(val_actions, dtype=torch.float32)
                )
            stats["val_loss"].append((step, float(v_loss)))
            policy.train()

        if checkpoint_callback is not None and step + 1 == halfway:
            checkpoint_callback(step + 1, policy)

        if rollout_probe is not None and rollout_interval and (step + 1) % rollout_interval == 0:
            from lares.fitting.checkpoints import is_better_checkpoint

            policy.eval()
            score = rollout_probe(policy)
            policy.train()
            stats["rollout_history"].append((step + 1, score))
            # The declared tie rule (lares.fitting.checkpoints): higher success,
            # then higher progress, then the *earlier* checkpoint.
            if is_better_checkpoint(score, stats["best_rollout_score"]):
                stats["best_rollout_score"] = score
                stats["best_rollout_step"] = step + 1
                stats["best_rollout_state"] = copy.deepcopy(policy.state_dict())

        if log_interval > 0 and (step + 1) % log_interval == 0:
            recent = stats["bc_loss"][-log_interval:]
            print(
                f"  [Stage 2] step {step + 1}/{num_steps}: "
                f"loss={np.mean(recent):.6f}, "
                f"mean_loss={np.mean(stats['mean_loss'][-log_interval:]):.6f}"
            )

    stats["final_loss"] = float(np.mean(stats["bc_loss"][-min(100, num_steps) :]))
    stats["train_loss_start"] = (
        float(np.mean(stats["bc_loss"][: min(100, num_steps)])) if stats["bc_loss"] else None
    )

    # Fitting diagnostics. Training loss alone cannot separate "this structure
    # cannot express the expert" from "these constants were badly tuned".
    policy.eval()
    axes = [f"axis_{i}" for i in range(policy.action_dim)]
    stats["train_loss_by_axis"] = dict(
        zip(axes, _per_axis_mse(policy, train_obs, train_actions))
    )
    if val_obs is not None:
        val_by_axis = _per_axis_mse(policy, val_obs, val_actions)
        stats["validation_loss_by_axis"] = dict(zip(axes, val_by_axis))
        stats["validation_loss_end"] = float(np.mean(val_by_axis))
    else:
        stats["validation_loss_by_axis"] = None
        stats["validation_loss_end"] = None
    grad_norms = stats.get("grad_norms", [])
    stats["gradient_norms"] = (
        {
            "mean": float(np.mean(grad_norms)),
            "max": float(np.max(grad_norms)),
            "final": float(grad_norms[-1]),
            "fraction_clipped": float(np.mean(np.asarray(grad_norms) > clip_grad_norm))
            if clip_grad_norm > 0
            else 0.0,
        }
        if grad_norms
        else None
    )
    stats["bound_activity"] = _bound_activity(policy)
    stats["loss_by_phase"] = _loss_by_phase(policy, train_obs, train_actions)
    policy.train()
    return stats


# ===========================================================================
#  Stage 3 — RL Fine-tuning (GRPO-style)
# ===========================================================================


def _collect_trajectories(policy, env, cases, max_steps=150):
    """Collect on-policy trajectories on an explicit list of training placements.

    ``cases`` names the placements, so a policy-gradient iteration cannot draw
    from, or advance, the stream any evaluation later uses.
    """
    trajectories = []
    policy.eval()

    for case in cases:
        obs, _ = env.reset(case)
        # Diagnostic state resets per episode, independently of the parameters.
        reset_diagnostics = getattr(policy, "reset_diagnostics", None)
        if callable(reset_diagnostics):
            reset_diagnostics()
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
        traj["advantages"] = np.full(T, traj_advantage)
    return trajectories


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
    train_cases,
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
        env: MetaWorld environment dedicated to trajectory collection.
        train_cases: ordered ``EpisodeCase`` sequence of *training* placements.
            Iteration ``i`` takes the next ``episodes_per_iter`` cases, cycling,
            so every candidate sees the same groups in the same order.
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

    train_cases = list(train_cases)
    if not train_cases:
        raise ValueError("rl_finetune needs a non-empty train_cases sequence")

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
        start = (iteration * episodes_per_iter) % len(train_cases)
        iter_cases = [
            train_cases[(start + j) % len(train_cases)]
            for j in range(episodes_per_iter)
        ]
        trajectories = _collect_trajectories(policy, env, iter_cases, max_steps)
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
        max_norm = clip_grad_norm if clip_grad_norm > 0 else float("inf")
        grad_norm_pre = float(torch.nn.utils.clip_grad_norm_(rl_params, max_norm))
        grad_norm_post = (
            min(grad_norm_pre, clip_grad_norm) if clip_grad_norm > 0 else grad_norm_pre
        )

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


def evaluate_policy(policy, env, manifest, max_steps=150, deterministic=True):
    """Score a symbolic policy on a manifest and return the full episode record.

    Every candidate in a comparison is handed the same manifest, so a score
    difference reflects the policy rather than which placements it drew
    (``spec.md`` FR-1, FR-4).

    Args:
        policy: SymbolicPolicy to score.
        env: environment bound to the manifest's task pool.
        manifest: ``EvaluationManifest``; its episode order is replayed exactly.
        max_steps: rollout horizon.
        deterministic: execute ``tanh(mean)`` (the evolution fitness) rather
            than sampling. Deterministic and sampled results stay separate.

    Returns:
        :class:`~lares.eval.runner.ManifestResult`. Call ``.fitness_dict()``
        for the ``{mean_reward, success_rate}`` pair the LLM feedback uses.
    """
    from lares.eval.runner import PolicyActor, evaluate_manifest

    actor = PolicyActor(policy, deterministic=deterministic, name="candidate")
    return evaluate_manifest(actor, manifest, env, max_steps=max_steps)


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
    case,
    max_steps=150,
    fps=20,
    verbose=True,
    deterministic=False,
):
    """Run one named episode and save it as a GIF.

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
        env: MetaWorld environment (raw or wrapped), dedicated to recording so
            capturing frames cannot shift a later evaluation case.
        path: Destination ``.gif`` file path.
        case: ``EpisodeCase`` naming the placement to record.
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
    reset_diagnostics = getattr(policy, "reset_diagnostics", None)
    if callable(reset_diagnostics):
        reset_diagnostics()

    obs, _ = env.reset(case)

    episode_reward = 0.0
    success = False

    for _ in range(max_steps):
        frame = _grab_frame_from_env(env)
        if frame is not None:
            frames.append(frame)

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean, std = policy(obs_t)
            # Must match evaluate_policy / _collect_trajectories: the policy's
            # output is a pre-tanh Gaussian, so the executed action is tanh(...).
            # Clipping the pre-tanh value instead records a different controller
            # than the one whose score is reported alongside the GIF.
            if deterministic:
                action = (
                    torch.tanh(mean)
                    .squeeze(0)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
            else:
                dist = torch.distributions.Normal(mean, std)
                pretanh_action = dist.sample()
                action = torch.tanh(pretanh_action).squeeze(0).numpy()
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


@lru_cache(maxsize=1)
def _read_policy_prompt_templates():
    """Read the prompt templates from disk. Cached: called once per generation."""
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
        "proposals_user": _read("proposals_user.txt"),
        "fittable_mutations_user": _read("fittable_mutations_user.txt"),
    }


def load_policy_prompt_assets(env_name):
    """Load prompt templates plus the machine-checked observation layout for a task.

    The layout comes from :mod:`lares.core.obs_schema`, not from a hand-written
    table. A missing table entry once sent a blank layout to the LLM, which then
    invented indices; ``get_obs_schema`` raises instead.
    """
    from lares.core.obs_schema import get_obs_schema
    from lares.core.policy_generation import TASK_NOTES, input_dict_for_policy

    schema = get_obs_schema(env_name)
    notes = TASK_NOTES.get(env_name, "")
    obs_description = schema.describe()
    if notes:
        obs_description = f"{obs_description}\n\n{notes}"

    if env_name not in input_dict_for_policy:
        print(
            f"  NOTE: no semantic-key table for '{env_name}'; the prompt carries the "
            f"verified observation schema only."
        )

    return {
        **_read_policy_prompt_templates(),
        "task_description": TASK_DESCRIPTIONS.get(env_name, env_name),
        "obs_description": obs_description,
        "input_dict_string": input_dict_for_policy.get(env_name, ""),
        "obs_schema": schema,
    }


def _impl_mode(args):
    from lares.core.policy_generation import POLICY_IMPL_BATCHED, POLICY_IMPL_PER_IDEA

    raw = getattr(args, "policy_impl_mode", POLICY_IMPL_PER_IDEA)
    raw = str(raw).strip().lower().replace("-", "_")
    return POLICY_IMPL_BATCHED if raw == POLICY_IMPL_BATCHED else POLICY_IMPL_PER_IDEA


def llm_evolution(
    client,
    env_name,
    obs_dim,
    action_dim,
    args,
    pop_size=5,
    generation=0,
    log_dir="./logs/evolution",
    logger=None,
    llm_transcript_path=None,
    ledger=None,
    feedback_bundle=None,
    parent_sources=None,
    max_total_calls=50,
    max_repair_per_candidate=1,
    param_budget=None,
    max_total_tokens=None,
):
    """Propose one population of candidate structures, with lineage, via the LLM.

    Structured two-phase generation only: generation 0 proposes ``PolicyIdea``
    objects and generation 1 and later proposes ``MutationProposal`` edits of
    parents drawn from ``feedback_bundle``. The feedback text is read back from
    the bundle's hashed file, so the generator sees exactly what the records say
    it saw. Training and evaluation are the caller's job.

    Returns:
        ``list[GeneratedCandidate]``, one per attempt, failures included.
    """
    from lares.core.policy_generation import generate_candidates
    from lares.search.generation import LLMCallLedger

    if not bool(getattr(args, "policy_gen_two_phase", False)):
        raise ValueError(
            "llm_evolution needs policy_gen_two_phase: true. Single-phase generation has no "
            "idea or proposal, so its candidates can never carry complete lineage."
        )
    os.makedirs(log_dir, exist_ok=True)
    prompts = load_policy_prompt_assets(env_name)
    ledger = ledger if ledger is not None else LLMCallLedger(log_dir)
    feedback_text = feedback_bundle.read_text(log_dir) if feedback_bundle is not None else None

    data_pkl_path = os.path.join(log_dir, "data.pkl")
    with open(data_pkl_path, "wb") as f:
        pickle.dump([{"obs": np.zeros(obs_dim)}], f)

    gen_dir = os.path.join(log_dir, f"gen_{generation}")
    os.makedirs(gen_dir, exist_ok=True)

    candidates = generate_candidates(
        client,
        args,
        gen_dir,
        generation,
        obs_dim,
        action_dim,
        env_name,
        prompts,
        pop_size,
        data_pkl_path,
        ledger,
        feedback_bundle=feedback_bundle,
        feedback_text=feedback_text,
        parent_sources=parent_sources,
        llm_transcript_path=llm_transcript_path,
        impl_mode=_impl_mode(args),
        max_total_calls=max_total_calls,
        max_repair_per_candidate=max_repair_per_candidate,
        param_budget=param_budget,
        max_total_tokens=max_total_tokens,
    )

    if logger is not None:
        from lares.core.policy_generation import final_attempts

        logger.log_metrics(
            stage="evolutionary",
            update=generation,
            metrics={
                "evo/candidates_generated": float(sum(1 for c in final_attempts(candidates) if c.valid))
            },
            task_name=env_name,
        )
    return candidates


def two_lane_generation(
    client,
    env_name,
    obs_dim,
    action_dim,
    args,
    plan,
    log_dir="./logs/evolution",
    logger=None,
    llm_transcript_path=None,
    ledger=None,
    max_total_calls=50,
    max_repair_per_candidate=1,
    param_budget=None,
    max_total_tokens=None,
):
    """Generation 0 of the two-lane search (:mod:`lares.search.lanes`), with lineage.

    Returns ``list[GeneratedCandidate]``, one per attempt, failures included; each carries its lane.
    """
    from lares.core.policy_generation import final_attempts, generate_two_lane_candidates
    from lares.search.generation import LLMCallLedger

    if not bool(getattr(args, "policy_gen_two_phase", False)):
        raise ValueError("two_lane_generation needs policy_gen_two_phase: true so candidates carry lineage")
    os.makedirs(log_dir, exist_ok=True)
    prompts = load_policy_prompt_assets(env_name)
    ledger = ledger if ledger is not None else LLMCallLedger(log_dir)
    data_pkl_path = os.path.join(log_dir, "data.pkl")
    with open(data_pkl_path, "wb") as f:
        pickle.dump([{"obs": np.zeros(obs_dim)}], f)
    candidates = generate_two_lane_candidates(
        client, args, os.path.join(log_dir, "gen_0"), obs_dim, action_dim, env_name, prompts, plan,
        data_pkl_path, ledger, llm_transcript_path=llm_transcript_path, impl_mode=_impl_mode(args),
        max_total_calls=max_total_calls, max_repair_per_candidate=max_repair_per_candidate,
        param_budget=param_budget, max_total_tokens=max_total_tokens,
    )
    if logger is not None:
        logger.log_metrics(
            stage="evolutionary", update=0,
            metrics={"evo/candidates_generated": float(sum(1 for c in final_attempts(candidates) if c.valid))},
            task_name=env_name,
        )
    return candidates


# ===========================================================================
#  EvolutionOrchestrator — full generation → fit → evaluate loop
# ===========================================================================


class EvolutionOrchestrator:
    """Full LLM evolution loop: generation → fit → evaluate → feedback → repeat.

    Each generation:
      1. :func:`llm_evolution` proposes typed candidates with lineage.
      2. Every attempt that failed is recorded; each valid candidate is fitted
         with the named objective and a fixed action scale, its checkpoint is
         chosen on an isolated probe stream, and it is screened.
      3. Screened contenders get the expanded evaluation, which updates the
         archive and the records.
      4. The evidence for the next generation is written as a hashed
         :class:`~lares.search.generation.FeedbackBundle`.

    Usage::

        orchestrator = EvolutionOrchestrator(
            env_name='push-v2', obs_dim=39, action_dim=4,
            num_generations=2, pop_size=3, bc_steps=2000,
            log_dir='./logs/evolution', dev_manifest=dev.head(30),
            probe_manifest=dev.tail(10),
        )
        best = orchestrator.run(client, envs, demo_buffer, args)
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
        rl_iterations=0,
        rl_episodes_per_iter=10,
        log_dir="./logs/evolution",
        record_demo_gif=True,
        dev_manifest=None,
        train_cases=(),
        gif_case=None,
        eval_max_steps=150,
        promotion_policy=None,
        rollout_checkpoint_probes=4,
        objective=None,
        checkpoint_steps=None,
        fixed_action_scale=None,
        probe_manifest=None,
        bc_batch_size=256,
        bc_lr=1e-3,
        fit_seed=0,
        val_fraction=0.2,
        split_seed=0,
        sampling="uniform",
        max_repair_per_candidate=1,
        max_llm_calls_per_generation=50,
        param_budget=None,
        max_llm_tokens_per_generation=None,
        promotion_challengers=1,
        lane_plan=None,
    ):
        from lares.core.symbolic_policy import SymbolicPolicy
        from lares.eval.promotion import PromotionPolicy
        from lares.fitting.checkpoints import default_checkpoint_steps
        from lares.fitting.objectives import DETERMINISTIC_MSE, spec
        from lares.search.generation import PARAMETER_BUDGET

        if int(rl_iterations) > 0:
            raise ValueError(
                "rl_iterations > 0 is not supported by the provenance-tracked fitter: RL would "
                "change the parameters after the checkpoint was selected, so the selected state, "
                "its diagnostics and its record would no longer describe the evaluated policy."
            )
        self.dev_manifest = dev_manifest
        self.train_cases = list(train_cases)
        self.gif_case = gif_case
        self.eval_max_steps = int(eval_max_steps)
        # Declared before any generation runs, and stored with the results, so a
        # promotion can be read back against the threshold that justified it.
        self.promotion_policy = promotion_policy or PromotionPolicy()
        # Success is not monotonic in gradient steps on this task while validation
        # loss is, so the checkpoint is chosen on rollouts. Set to 0 to fit to the end.
        self.rollout_checkpoint_probes = int(rollout_checkpoint_probes)
        self.objective = objective or DETERMINISTIC_MSE
        spec(self.objective)
        self.checkpoint_steps = (
            tuple(sorted({int(s) for s in checkpoint_steps}))
            if checkpoint_steps is not None
            else default_checkpoint_steps(bc_steps, self.rollout_checkpoint_probes)
        )
        self.fixed_action_scale = float(
            fixed_action_scale if fixed_action_scale is not None else SymbolicPolicy.DEFAULT_ACTION_SCALE
        )
        self._probe_manifest = probe_manifest
        self.bc_batch_size = int(bc_batch_size)
        self.bc_lr = float(bc_lr)
        self.fit_seed = int(fit_seed)
        self.val_fraction = float(val_fraction)
        self.split_seed = int(split_seed)
        self.sampling = sampling
        self.max_repair_per_candidate = int(max_repair_per_candidate)
        self.max_llm_calls_per_generation = int(max_llm_calls_per_generation)
        self.max_llm_tokens_per_generation = (
            int(max_llm_tokens_per_generation) if max_llm_tokens_per_generation else None
        )
        if int(promotion_challengers) < 1:
            raise ValueError("promotion_challengers must be at least 1")
        #: Screened challengers tested on the promotion manifest per generation.
        #: One by default; more are judged with Holm's correction.
        self.promotion_challengers = int(promotion_challengers)
        self.param_budget = tuple(param_budget or PARAMETER_BUDGET)
        self.env_name = env_name
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_generations = num_generations
        self.pop_size = pop_size
        self.elite_num = elite_num
        self.bc_steps = bc_steps
        self.rl_iterations = 0
        self.rl_episodes_per_iter = rl_episodes_per_iter
        self.log_dir = log_dir
        self.record_demo_gif = record_demo_gif
        #: A :class:`~lares.search.lanes.TwoLanePlan`, or ``None`` for the single-lane search.
        self.lane_plan = lane_plan
        if lane_plan is not None:
            if int(num_generations) != 1:
                raise ValueError("a two-lane search runs exactly one generation; edit generations belong to Stage F")
            if int(pop_size) != lane_plan.n:
                raise ValueError(f"pop_size {pop_size} does not match the two-lane plan's {lane_plan.n} candidates")
            if int(bc_steps) != 0:
                raise ValueError(
                    "with a two-lane plan the global bc_steps must be 0: explorations are zero-shot, and the "
                    "fittable lane's budget and checkpoints come from the plan"
                )
            if int(promotion_challengers) != 1:
                raise ValueError("the two-lane search tests one challenger per incumbent")
        #: The fittable incumbent, kept apart from the performance incumbent (``best_overall``).
        self.fittable_incumbent = None
        self._fittable_incumbent_result = None
        self.fittable_decisions: list = []

        # Populated by run()
        self.history: list = []
        self.best_overall: dict | None = None
        self.promotion_log: list = []
        self._incumbent_expanded = None
        self._incumbent_id = None
        from lares.search import Archive
        from lares.search.resources import ResourceLedger

        self.archive = Archive()
        self.records: list = []
        self.records_dir = os.path.join(self.log_dir, "experiments")
        self.feedback_bundles: list = []
        self.generation_history: list = []
        self.run_resources = ResourceLedger()
        self.ledger = None
        self.data = None
        self.probe_batch = None
        self.probe_batch_ref = None

    # ------------------------------------------------------------------
    #  Manifests
    # ------------------------------------------------------------------

    @property
    def screen_manifest(self):
        """The fixed screening subset: the first ``screen_episodes`` development cases."""
        n = min(self.promotion_policy.screen_episodes, len(self.dev_manifest))
        return self.dev_manifest.head(n)

    @property
    def promotion_manifest(self):
        """The promotion set: the ``expanded_episodes`` cases after the screen, disjoint from it.

        Promotion evidence must come from cases that took no part in screening or
        checkpoint selection, so this is a slice after the screen, never a prefix
        containing it.
        """
        start = min(self.promotion_policy.screen_episodes, len(self.dev_manifest))
        stop = start + int(self.promotion_policy.expanded_episodes)
        if stop > len(self.dev_manifest):
            raise ValueError(
                f"dev_manifest has {len(self.dev_manifest)} cases; a {start}-case screen plus a "
                f"{self.promotion_policy.expanded_episodes}-case promotion set needs {stop}"
            )
        return self.dev_manifest.slice(start, stop)

    #: Kept for callers that still say "expanded": the expanded evaluation *is* the promotion set.
    expanded_manifest = promotion_manifest

    @property
    def probe_manifest(self):
        """Cases the mid-fitting probe scores; the screen when none was supplied."""
        return self._probe_manifest if self._probe_manifest is not None else self.screen_manifest

    @property
    def probe_is_isolated(self) -> bool:
        """True when no probe case is also a screen or promotion case."""
        if self._probe_manifest is None:
            return False
        probe = {c.case_id for c in self._probe_manifest.episodes}
        used = {c.case_id for c in self.promotion_manifest.episodes} | {c.case_id for c in self.screen_manifest.episodes}
        return not (probe & used)

    def check_manifest_isolation(self):
        """Refuse a run whose promotion evidence overlaps screening or checkpoint selection."""
        promotion = {c.case_id for c in self.promotion_manifest.episodes}
        for name, manifest in (("screen", self.screen_manifest), ("probe", self.probe_manifest)):
            overlap = promotion & {c.case_id for c in manifest.episodes}
            if overlap:
                raise ValueError(
                    f"the promotion manifest shares {len(overlap)} case(s) with the {name} manifest "
                    f"({sorted(overlap)[:3]}...); promotion evidence must be independent"
                )
        if self.lane_plan is not None:
            overlap = {c.case_id for c in self.probe_manifest.episodes} & {c.case_id for c in self.screen_manifest.episodes}
            if self._probe_manifest is None or overlap:
                raise ValueError(
                    "a two-lane search needs its own probe manifest, disjoint from the screen: checkpoint "
                    "selection, screening and promotion must use separate cases"
                )

    # ------------------------------------------------------------------
    #  Run-level artifacts
    # ------------------------------------------------------------------

    def _artifact(self, rel_path, data):
        from lares.search.generation import write_artifact

        return write_artifact(self.log_dir, rel_path, data)

    def _tensor_artifact(self, rel_path, value):
        import io

        buffer = io.BytesIO()
        torch.save(value, buffer)
        return self._artifact(rel_path, buffer.getvalue())

    def _probe_env(self, envs):
        probe = getattr(envs, "probe", None)
        if probe is not None:
            return probe, "probe"
        if getattr(envs, "allow_shared", False):
            return envs.gif, "gif (shared; allow_shared=True)"
        raise ValueError(
            "EvolutionOrchestrator needs PipelineEnvs(probe=...): the checkpoint probe gets its "
            "own environment rather than borrowing the GIF stream."
        )

    def _write_probe_batch(self, demo_buffer):
        """A fixed observation batch every reconstruction check replays."""
        n = min(32, len(demo_buffer))
        obs, _ = demo_buffer.take(list(range(n)))
        self.probe_batch = torch.tensor(obs, dtype=torch.float32)
        self.probe_batch_ref = self._tensor_artifact("reconstruction_probe_batch.pt", self.probe_batch)

    def _write_run_manifest(self, args, probe_stream):
        from lares.fitting.checkpoints import CHECKPOINT_TIE_RULE
        from lares.fitting.optimizers import objective_config
        from lares.search.generation import env_identity

        manifests = {
            "probe": self.probe_manifest,
            "screen": self.screen_manifest,
            "promotion": self.promotion_manifest,
        }
        payload = {
            "environment": env_identity(self.env_name),
            "model": getattr(args, "model", ""),
            "reasoning_effort": getattr(args, "reasoning_effort", None),
            "policy_impl_mode": _impl_mode(args),
            "objective_config": objective_config(
                self.objective, self.sampling, self.bc_steps, self.bc_batch_size, self.bc_lr, 1.0,
                self.fit_seed, self.checkpoint_steps, self.fixed_action_scale,
                val_fraction=self.val_fraction, split_seed=self.split_seed,
            ),
            "fixed_action_scale": self.fixed_action_scale,
            "action_mode": "deterministic",
            "checkpoint_steps": list(self.checkpoint_steps),
            "checkpoint_tie_rule": CHECKPOINT_TIE_RULE,
            "probe_stream": probe_stream,
            "probe_is_isolated": self.probe_is_isolated,
            "manifests": {
                name: {"manifest_id": m.manifest_id, "split": m.split, "case_ids": [c.case_id for c in m.episodes]}
                for name, m in manifests.items()
            },
            "promotion_policy": self.promotion_policy.to_dict(),
            "promotion_rule": __import__("lares.eval.promotion", fromlist=["PROMOTION_RULE"]).PROMOTION_RULE,
            "promotion_challengers": self.promotion_challengers,
            "max_llm_tokens_per_generation": self.max_llm_tokens_per_generation,
            "param_budget": list(self.param_budget),
            "max_repair_per_candidate": self.max_repair_per_candidate,
            "max_llm_calls_per_generation": self.max_llm_calls_per_generation,
            "split": self.data.split_info if self.data is not None else None,
            "probe_batch": self.probe_batch_ref,
            "two_lane_plan": self.lane_plan.to_dict() if self.lane_plan is not None else None,
        }
        self.run_manifest_ref = self._artifact("run_manifest.json", json.dumps(payload, indent=2, default=str))
        return payload

    def _write_run_totals(self):
        from lares.search.provenance import reconcile_resources

        totals = {
            "run_wall_time_seconds": self.run_resources.wall_time_seconds,
            "reconciliation": reconcile_resources(self.records, self.ledger, self.log_dir),
        }
        if self.lane_plan is not None:
            from lares.search.lanes import lane_resource_totals

            totals["lanes"] = lane_resource_totals(self.records, self.ledger)
            self._artifact("lane_resources.json", json.dumps(totals["lanes"], indent=2, default=str))
        self._artifact("resource_totals.json", json.dumps(totals, indent=2, default=str))
        return totals

    # ------------------------------------------------------------------
    #  Experiment records
    # ------------------------------------------------------------------

    def _save_record(self, record):
        """Validate and persist one record; a changed record keeps its old version."""
        record.validate()
        record.save(self.records_dir)
        self.records = [r for r in self.records if r.experiment_id != record.experiment_id] + [record]
        return record

    def _llm_evidence(self, candidate, args):
        """Embedded call references and this attempt's exclusive LLM usage."""
        calls = [self.ledger.get(c) for c in candidate.llm_call_ids]
        exclusive = [c.call_id for c in calls if c.attempt_ids == [candidate.attempt_id]]
        block = {
            "model_id": getattr(args, "model", ""),
            "reasoning_effort": getattr(args, "reasoning_effort", None),
            "two_phase": True,
            "impl_mode": _impl_mode(args),
            "calls": [c.reference() for c in calls],
            "exclusive_call_ids": exclusive,
            "shared_call_ids": [c.call_id for c in calls if c.call_id not in exclusive],
            "response_sha256": (
                sha256_text(candidate.response)
                if candidate.response is not None
                else None
            ),
        }
        return block, self.ledger.totals(exclusive), sum(c.wall_time_seconds for c in calls if c.call_id in exclusive)

    def _record_failed_attempt(self, candidate, args, final_by_slot, feedback_bundle):
        """One record for an attempt that never reached fitting, corrected or not."""
        from lares.search import STATUS_INVALID, ExperimentRecord, source_hash
        from lares.search.resources import ResourceLedger

        llm, exclusive_totals, exclusive_wall = self._llm_evidence(candidate, args)
        resources = ResourceLedger()
        resources.add_llm_totals(exclusive_totals)
        resources.add_wall_time("generation", exclusive_wall)
        final = final_by_slot[candidate.candidate_id]
        stage = candidate.validation.get("stage", "unknown")
        if final is candidate:
            disposition = f"slot failed at {stage} after {candidate.attempt} correction(s)"
        else:
            disposition = f"failed at {stage}; superseded by a later attempt of {candidate.candidate_id}"
        record = ExperimentRecord(
            experiment_id=candidate.attempt_id,
            candidate_id=candidate.candidate_id,
            generation=candidate.generation,
            status=STATUS_INVALID,
            parent_ids=candidate.parent_ids,
            idea_id=candidate.idea_id,
            proposal_id=candidate.proposal_id,
            code={
                "source_hash": source_hash(candidate.code or ""),
                "source": candidate.code,
                "empty": candidate.code is None,
            },
            llm=llm,
            llm_call_ids=list(candidate.llm_call_ids),
            validation=dict(candidate.validation),
            lineage=candidate.lineage_dict(feedback_bundle),
            resources=resources.to_dict(),
            timestamps={"recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
            final_disposition=disposition,
        )
        return self._save_record(record)

    # ------------------------------------------------------------------
    #  Evidence
    # ------------------------------------------------------------------

    def _build_evidence(self, generation, gen_results, failures, manifest, demo_buffer=None):
        """Assemble the feedback text and the ids of every policy it shows.

        Sensitivity is probed on real observations from the demonstration buffer.
        An all-zero or standard-normal batch puts the end-effector, object and
        goal on top of each other or metres apart, and every direction term then
        reads as dead, which would tell the generator to delete the controller.
        """
        from lares.search import CandidateEvidence, build_feedback

        probe = None
        if demo_buffer is not None and len(demo_buffer) > 0:
            obs_np, _ = demo_buffer.take(list(range(min(64, len(demo_buffer)))))
            probe = torch.tensor(obs_np, dtype=torch.float32)

        evidence = []
        for result in gen_results:
            sensitivity = []
            if probe is not None:
                try:
                    from lares.fitting.sensitivity import action_sensitivity, dead_parameters

                    sensitivity = dead_parameters(action_sensitivity(result["policy"], probe))
                except Exception:
                    sensitivity = []
            evidence.append(
                CandidateEvidence(
                    candidate_id=result["candidate_id"],
                    code=result["code"],
                    report=result.get("expanded_report") or result["screen_report"],
                    zero_shot_report=result.get("zero_shot_report"),
                    num_parameters=result["policy"].count_parameters(),
                    dead_parameters=sensitivity,
                )
            )
        for failure in failures:
            errors = failure.validation.get("errors") or [""]
            evidence.append(
                CandidateEvidence(
                    candidate_id=failure.candidate_id,
                    code=failure.code or "",
                    report=None,
                    error=f"[{failure.validation.get('stage', 'unknown')}] {str(errors[0])[:300]}",
                )
            )
        best = max((e.success for e in evidence if e.report is not None), default=0.0)
        self.generation_history.append(
            f"generation {generation}: {len(gen_results)} valid, "
            f"{len(failures)} invalid, best success {best:.3f}"
        )
        text = build_feedback(
            evidence,
            generation=generation,
            manifest_id=manifest.manifest_id,
            history=self.generation_history,
            archive=self.archive,
        )
        context = [r["candidate_id"] for r in gen_results]
        if len(self.archive):
            for entry in self.archive.occupied_slots().values():
                if entry is not None and entry.candidate_id not in context:
                    context.append(entry.candidate_id)
        return text, context

    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------

    def _evaluate_checkpoint(
        self, policy, env, manifest, candidate_id, checkpoint, bc_stats=None
    ):
        """Score one checkpoint and build its report.

        Zero-shot, intermediate and fitted results are separate reports and are
        never merged: a structure that is decent before fitting and worse after
        is a fitting failure, which an aggregate would hide.
        """
        from lares.eval.report import build_report

        result = evaluate_policy(policy, env, manifest, max_steps=self.eval_max_steps)
        report = build_report(
            result,
            candidate_id=candidate_id,
            checkpoint=checkpoint,
            bc_stats=bc_stats,
            action_dim=self.action_dim,
        )
        return result, report

    def _rollout_probe(self, env, manifest, resources):
        """Score a policy mid-fitting on the probe cases, counting every episode."""
        from lares.eval.runner import PolicyActor, evaluate_manifest
        from lares.fitting.checkpoints import RolloutScore

        def probe(policy):
            result = evaluate_manifest(
                PolicyActor(policy, deterministic=True, name="probe"),
                manifest,
                env,
                max_steps=self.eval_max_steps,
            )
            resources.add_episodes("probe", result)
            progress = [
                e.signed_goal_progress
                for e in result.episodes
                if e.signed_goal_progress is not None
            ]
            return RolloutScore(result.success_rate, float(np.mean(progress)) if progress else 0.0)

        return probe

    def _save_report(self, report, rel_name, gen):
        from lares.search.generation import run_relative

        path = os.path.join(self.log_dir, f"gen_{gen}", "reports", rel_name)
        report.save(path)
        return {"path": run_relative(path, self.log_dir), "sha256": sha256_file(path)}

    def _report_fitting_stats(self, fit):
        """The fitting section of the fitted report, taken from the selected state only."""
        diag = fit.selected_diagnostics
        losses = [h[1] for h in fit.history[: min(100, len(fit.history))]]
        return {
            "train_loss_by_axis": diag["train_loss_by_axis"],
            "validation_loss_by_axis": diag["validation_loss_by_axis"],
            "loss_by_phase": diag["loss_by_phase"],
            "gradient_norms": fit.gradient_norms,
            "bound_activity": diag["bound_activity"],
            "train_loss_start": float(np.mean(losses)) if losses else None,
            "final_loss": diag["train_loss"],
            "validation_loss_end": diag["validation_loss"],
            "num_steps": fit.selected_step,
            "split": self.data.split_info,
            "checkpoint_step": fit.selected_step,
            "objective": self.objective,
            "parameters_sha256": diag["parameters_sha256"],
        }

    def _verify_reconstruction(self, candidate, policy, source_ref, selected_ref, cand_dir):
        """Rebuild the policy from its record's artifacts and compare actions exactly."""
        from lares.core.policy_generation import GENERATED_IMPORTS
        from lares.search.provenance import reconstruct_actions
        from lares.search.schemas import RECONSTRUCTION_FAILED, RECONSTRUCTION_VERIFIED

        policy.eval()
        with torch.no_grad():
            live = torch.tanh(policy(self.probe_batch)[0]).detach().clone()
        expected_ref = self._tensor_artifact(os.path.join(cand_dir, "expected_actions.pt"), live)
        recon = {
            "source": source_ref,
            "selected_state": selected_ref,
            "probe_batch": self.probe_batch_ref,
            "expected_actions": expected_ref,
            "imports_sha256": sha256_text(GENERATED_IMPORTS),
            "env_id": self.env_name,
            "obs_dim": int(self.obs_dim),
            "action_dim": int(self.action_dim),
            "fixed_action_scale": self.fixed_action_scale,
            "comparison": "torch.equal on tanh(mean) over the fixed probe batch",
        }
        try:
            rebuilt = reconstruct_actions(self.log_dir, recon)
            recon["max_abs_diff"] = float((rebuilt - live).abs().max()) if rebuilt.numel() else 0.0
            recon["status"] = RECONSTRUCTION_VERIFIED if torch.equal(rebuilt, live) else RECONSTRUCTION_FAILED
        except Exception as exc:
            recon["error"] = f"{type(exc).__name__}: {exc}"
            recon["status"] = RECONSTRUCTION_FAILED
        return recon

    def _fitting_settings(self, candidate):
        """``(budget, checkpoint_steps, lane)`` for one candidate.

        Without a lane plan every candidate uses the orchestrator's settings. With one, only the
        fittable lane fits, with the plan's budget and checkpoints; explorations are never fitted.
        """
        from lares.search.schemas import LANE_FITTABLE

        if self.lane_plan is None:
            return self.bc_steps, self.checkpoint_steps, ""
        if candidate.lane == LANE_FITTABLE:
            return int(self.lane_plan.fittable_budget), tuple(self.lane_plan.checkpoint_steps), candidate.lane
        return 0, (), candidate.lane

    def _zero_shot_fit(self, policy):
        """The fitting block of a candidate that is never fitted: no data, no gradient, no probe."""
        from types import SimpleNamespace

        from lares.fitting.optimizers import objective_config, state_parameters_sha256

        state = copy.deepcopy(policy.state_dict())
        diagnostics = {
            "computed": False,
            "reason": "exploration lane: evaluated at its initial values; no fitting data was read",
            "parameters_sha256": state_parameters_sha256(state),
            "train_loss": None,
            "validation_loss": None,
        }
        return SimpleNamespace(
            initial_state=state, final_state=state, selected_state=state, selected_step=0,
            checkpoint_history=[], bc_gradient_steps=0, objective_evaluations=0, history=[],
            gradient_norms=None, frozen_parameters=[], final_diagnostics=diagnostics,
            selected_diagnostics=dict(diagnostics),
            objective_config=objective_config(
                self.objective, self.sampling, 0, self.bc_batch_size, self.bc_lr, 1.0, self.fit_seed, [],
                self.fixed_action_scale, val_fraction=self.val_fraction, split_seed=self.split_seed,
            ),
        )

    def _fit_and_screen(self, candidate, envs, probe_env, args, feedback_bundle):
        """Fit one valid candidate, choose its checkpoint, screen it and record it."""
        from lares.eval.report import CHECKPOINT_FITTED, CHECKPOINT_INTERMEDIATE, CHECKPOINT_ZERO_SHOT
        from lares.fitting.checkpoints import CHECKPOINT_TIE_RULE
        from lares.fitting.optimizers import fit_policy_with_objective, state_parameters_sha256
        from lares.search import STATUS_EVALUATED, ExperimentRecord, source_hash
        from lares.search.generation import env_identity
        from lares.search.resources import ResourceLedger
        from lares.search.schemas import FITTING_REJECTED, LANE_EXPLORATION, LANE_FITTABLE

        gen = candidate.generation
        cid = candidate.candidate_id
        policy = candidate.policy
        policy.fixed_action_scale = self.fixed_action_scale
        cand_dir = os.path.join(f"gen_{gen}", "candidates", candidate.attempt_id)
        screen = self.screen_manifest
        budget, steps, lane = self._fitting_settings(candidate)
        zero_shot_only = lane == LANE_EXPLORATION
        resources = ResourceLedger()
        llm, exclusive_totals, exclusive_wall = self._llm_evidence(candidate, args)
        resources.add_llm_totals(exclusive_totals)
        resources.add_wall_time("generation", exclusive_wall)
        source_ref = self._artifact(os.path.join(cand_dir, "policy_source.py"), candidate.code)

        # Zero-shot: what the LLM's own initial values do untrained. If a
        # structure is decent here and worse after fitting, the fault is in the
        # fitting, not the structure.
        started = time.time()
        zero_result, zero_report = self._evaluate_checkpoint(
            policy, envs.development, screen, cid, CHECKPOINT_ZERO_SHOT
        )
        resources.add_episodes("screen", zero_result)
        resources.add_wall_time("evaluation", time.time() - started)

        halfway = {}

        def capture(step, pol, _store=halfway):
            _store["step"] = step
            _store["state"] = copy.deepcopy(pol.state_dict())

        probe = None
        if zero_shot_only:
            # An unseeded exploration never reads the demonstrations, fits or probes.
            fit = self._zero_shot_fit(policy)
        else:
            probe = self._rollout_probe(probe_env, self.probe_manifest, resources) if steps else None
            started = time.time()
            fit = fit_policy_with_objective(
                policy,
                self.data,
                self.objective,
                budget=budget,
                batch_size=min(self.bc_batch_size, int(self.data.train_obs.shape[0])),
                lr=self.bc_lr,
                seed=self.fit_seed,
                sampling=self.sampling,
                rollout_probe=probe,
                checkpoint_steps=steps if probe is not None else None,
                checkpoint_callback=capture,
                structure_id=cid,
            )
            resources.add_wall_time("fitting", time.time() - started)
            resources.add("bc_gradient_steps", fit.bc_gradient_steps)
            resources.add("bc_objective_evaluations", fit.objective_evaluations)
            if state_parameters_sha256(policy.state_dict()) != fit.selected_diagnostics["parameters_sha256"]:
                raise RuntimeError(f"{cid}: the policy does not hold the selected checkpoint after fitting")
            rejected = " (fitting rejected: step 0 kept)" if lane == LANE_FITTABLE and fit.selected_step == 0 else ""
            print(
                f"    selected checkpoint: step {fit.selected_step}/{budget}{rejected} "
                f"({CHECKPOINT_TIE_RULE}); probes {[(h['step'], round(h['success'], 3)) for h in fit.checkpoint_history]}"
            )
        state_refs = {
            name: self._tensor_artifact(os.path.join(cand_dir, f"{name}.pt"), state)
            for name, state in (
                ("initial_state", fit.initial_state),
                ("final_state", fit.final_state),
                ("selected_state", fit.selected_state),
            )
        }

        started = time.time()
        intermediate_report, intermediate_result = None, None
        if zero_shot_only:
            # The selected state is the initial state, so the zero-shot screen is the screen.
            bc_stats = None
            screen_result, screen_report = zero_result, zero_report
        else:
            if halfway:
                mid = copy.deepcopy(policy)
                mid.load_state_dict(halfway["state"])
                intermediate_result, intermediate_report = self._evaluate_checkpoint(
                    mid, envs.development, screen, cid, CHECKPOINT_INTERMEDIATE
                )
                resources.add_episodes("screen", intermediate_result)
            bc_stats = self._report_fitting_stats(fit)
            screen_result, screen_report = self._evaluate_checkpoint(
                policy, envs.development, screen, cid, CHECKPOINT_FITTED, bc_stats=bc_stats
            )
            resources.add_episodes("screen", screen_result)
        resources.add_wall_time("evaluation", time.time() - started)

        report_refs = {"zero_shot": self._save_report(zero_report, f"{cid}_{CHECKPOINT_ZERO_SHOT}.json", gen)}
        if intermediate_report is not None:
            report_refs["intermediate"] = self._save_report(
                intermediate_report, f"{cid}_{CHECKPOINT_INTERMEDIATE}.json", gen
            )
        if not zero_shot_only:
            report_refs["fitted_screen"] = self._save_report(screen_report, f"{cid}_{CHECKPOINT_FITTED}.json", gen)

        reconstruction = self._verify_reconstruction(
            candidate, policy, source_ref, state_refs["selected_state"], cand_dir
        )

        fitting_rejected = (int(fit.selected_step) == 0) if lane == LANE_FITTABLE else None
        result = {
            "candidate_id": cid,
            "attempt_id": candidate.attempt_id,
            "candidate": candidate,
            "policy": policy,
            "code": candidate.code,
            "response": candidate.response,
            "eval": screen_result.fitness_dict(),
            "eval_record": screen_result,
            "screen_report": screen_report,
            "zero_shot_report": zero_report,
            "intermediate_report": intermediate_report,
            "expanded_record": None,
            "expanded_report": None,
            "score": self._score(screen_result.fitness_dict()),
            "fit": fit,
            "bc_stats": bc_stats,
            "resources": resources,
            "report_refs": report_refs,
            "generation": gen,
            "candidate_idx": candidate.slot,
            "lane": lane,
            "fitting_rejected": fitting_rejected,
        }

        fitting = {
            "optimizer": None if zero_shot_only else "adam",
            "objective": self.objective,
            "objective_config": fit.objective_config,
            "action_mode": fit.objective_config["execution"],
            "fixed_action_scale": self.fixed_action_scale,
            "steps": int(budget),
            "split": self.data.split_info,
            "frozen_parameters": fit.frozen_parameters,
            "checkpoint_steps": list(steps) if probe is not None else [],
            "checkpoint_tie_rule": CHECKPOINT_TIE_RULE,
            "probe_manifest_id": self.probe_manifest.manifest_id if probe is not None else None,
            "probe_is_isolated": self.probe_is_isolated,
            "checkpoint_history": fit.checkpoint_history,
            "selected_step": int(fit.selected_step),
            "initial_state": state_refs["initial_state"],
            "final_state": state_refs["final_state"],
            "selected_state": state_refs["selected_state"],
            "final_diagnostics": fit.final_diagnostics,
            "selected_diagnostics": fit.selected_diagnostics,
            "gradient_norms": fit.gradient_norms,
            "train_loss_end": fit.selected_diagnostics["train_loss"],
            "validation_loss_end": fit.selected_diagnostics["validation_loss"],
        }
        if lane:
            fitting["lane"] = lane
            fitting["fitting_rejected"] = fitting_rejected
            if fitting_rejected:
                fitting["selection_outcome"] = FITTING_REJECTED
            if zero_shot_only:
                fitting["lane_fitting"] = "disabled: unseeded exploration, evaluated at its initial values"

        record = ExperimentRecord(
            experiment_id=candidate.attempt_id,
            candidate_id=cid,
            generation=gen,
            status=STATUS_EVALUATED,
            parent_ids=candidate.parent_ids,
            idea_id=candidate.idea_id,
            proposal_id=candidate.proposal_id,
            code={
                "source_hash": source_hash(candidate.code),
                "source_path": source_ref["path"],
                "source_sha256": source_ref["sha256"],
                "source": candidate.code,
                "num_parameters": policy.count_parameters(),
                "num_parameter_tensors": len(list(policy.parameters())),
            },
            llm=llm,
            llm_call_ids=list(candidate.llm_call_ids),
            validation=dict(candidate.validation),
            lineage=candidate.lineage_dict(feedback_bundle),
            fitting=fitting,
            evaluation={
                "manifest_ids": [screen.manifest_id],
                "probe_manifest_id": self.probe_manifest.manifest_id if probe is not None else None,
                "action_mode": screen_report.action_mode,
                "environment": env_identity(self.env_name),
                "reports": report_refs,
                "checkpoints": {
                    "zero_shot": zero_report.rollout.success_rate,
                    "intermediate": (
                        intermediate_report.rollout.success_rate if intermediate_report is not None else None
                    ),
                    "fitted": screen_report.rollout.success_rate,
                },
                "success_rate": screen_report.rollout.success_rate,
                "mean_return": screen_report.rollout.mean_return,
                "failure_labels": screen_report.rollout.failure_labels,
            },
            resources=resources.to_dict(),
            reconstruction=reconstruction,
            timestamps={"recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
            final_disposition="screened",
        )
        self._save_record(record)
        self.archive.add(
            candidate_id=cid,
            generation=gen,
            report=screen_report,
            num_parameters=policy.count_parameters(),
            source_hash=record.code["source_hash"],
        )
        result["record"] = record
        return result

    def _fittable_promotion(self, gen_results, envs, expanded, screen, gen):
        """Challenge for the fittable incumbent. Returns ``{candidate_id: decision}`` for every candidate.

        Only fittable-lane candidates whose selected step is above 0 are eligible. They are
        screened with the same rule and on the same screen as the performance challenge; the best
        cleared one alone is scored on the promotion set at its fitted state (reused when it was
        also the performance challenger) and at its own step-0 state, and
        :func:`~lares.search.lanes.decide_fittable_incumbent` applies both strict paired tests.
        """
        from lares.eval.report import CHECKPOINT_FITTED, CHECKPOINT_ZERO_SHOT
        from lares.search.lanes import NOT_BEST_FITTABLE_CHALLENGER, decide_fittable_incumbent, fittable_eligibility

        decisions, eligible = {}, []
        for r in gen_results:
            ok, reason = fittable_eligibility(r["lane"], r["fit"].selected_step)
            if ok:
                eligible.append(r)
            else:
                decisions[r["candidate_id"]] = {
                    "candidate_id": r["candidate_id"], "lane": r["lane"], "selected_step": int(r["fit"].selected_step),
                    "eligible": False, "promoted": False, "reason": reason,
                    "fitting_gain": None, "incumbent_comparison": None,
                }
        if not eligible:
            print("\n  Fittable incumbent: no fittable-lane candidate kept a fitted checkpoint (all rejected fitting).")
            return decisions
        screened = self.promotion_policy.screen(
            [{"candidate_id": r["candidate_id"], "report": r["screen_report"], "valid": True} for r in eligible]
        )
        cleared = sorted((d for d in screened if d["promoted"]),
                         key=lambda d: (d["success_rate"], d["progress"], d["mean_return"]), reverse=True)
        challenger = cleared[0] if cleared else None
        by_id = {r["candidate_id"]: r for r in eligible}
        for d in screened:
            if challenger is None or d["candidate_id"] != challenger["candidate_id"]:
                r = by_id[d["candidate_id"]]
                decisions[d["candidate_id"]] = {
                    "candidate_id": d["candidate_id"], "lane": r["lane"], "selected_step": int(r["fit"].selected_step),
                    "eligible": True, "promoted": False,
                    "reason": NOT_BEST_FITTABLE_CHALLENGER if d["promoted"] else f"screening: {d['reason']}",
                    "fitting_gain": None, "incumbent_comparison": None, "fittable_screening": d,
                }
        if challenger is None:
            return decisions

        cid = challenger["candidate_id"]
        r = by_id[cid]
        started = time.time()
        if r["expanded_record"] is None:
            exp_result, exp_report = self._evaluate_checkpoint(
                r["policy"], envs.development, expanded, cid, CHECKPOINT_FITTED, bc_stats=r["bc_stats"]
            )
            r["resources"].add_episodes("expanded", exp_result)
            r["report_refs"]["fitted_expanded"] = self._save_report(exp_report, f"{cid}_expanded.json", gen)
            self.archive.confirm(cid, exp_report)
            r["expanded_record"], r["expanded_report"] = exp_result, exp_report
            r["eval"] = exp_result.fitness_dict()
            r["score"] = self._score(r["eval"])
        step0 = copy.deepcopy(r["policy"])
        step0.load_state_dict(r["fit"].initial_state)
        step0_result, step0_report = self._evaluate_checkpoint(step0, envs.development, expanded, cid, CHECKPOINT_ZERO_SHOT)
        r["resources"].add_episodes("expanded", step0_result)
        r["report_refs"]["zero_shot_expanded"] = self._save_report(step0_report, f"{cid}_zero_shot_expanded.json", gen)
        r["step0_expanded_record"] = step0_result
        elapsed = time.time() - started
        r["resources"].add_wall_time("evaluation", elapsed)
        self.run_resources.add_wall_time("evaluation", elapsed)

        incumbent_id = self.fittable_incumbent["candidate_id"] if self.fittable_incumbent else None
        decision = decide_fittable_incumbent(
            cid, r["lane"], r["fit"].selected_step, r["expanded_record"], step0_result,
            self._fittable_incumbent_result, incumbent_id,
            excluded_manifest_ids=(screen.manifest_id, self.probe_manifest.manifest_id),
        )
        decision.update(fittable_screening=challenger, generation=gen)
        gain = decision["fitting_gain"]["paired_success"]
        print(
            f"\n  Fittable challenger {cid} (step {r['fit'].selected_step}): fitted {r['expanded_record'].success_rate:.3f} "
            f"vs its step 0 {step0_result.success_rate:.3f} on {expanded.manifest_id}, paired {gain['mean_difference']:+.3f} "
            f"-> {decision['reason']}"
        )
        if decision["promoted"]:
            self._fittable_incumbent_result = r["expanded_record"]
            self.fittable_incumbent = {
                "candidate_id": cid, "attempt_id": r["attempt_id"], "lane": r["lane"],
                "selected_step": int(r["fit"].selected_step), "code": r["code"],
                "policy": copy.deepcopy(r["policy"]), "promotion_manifest_id": expanded.manifest_id,
            }
        decisions[cid] = decision
        self.fittable_decisions.append(decision)
        return decisions

    def _write_incumbents(self, gen, expanded):
        """Both incumbents and the evidence behind each, as one hashed artifact."""
        from lares.eval.promotion import NULL_INCUMBENT_ID, PROMOTION_RULE

        lanes = {r["candidate_id"]: r.get("lane") for r in self.history}
        performance = None
        if self.best_overall is not None:
            cid = self.best_overall["candidate_id"]
            performance = {"candidate_id": cid, "lane": lanes.get(cid), "attempt_id": self.best_overall["attempt_id"],
                           "promotion_success_rate": self._incumbent_expanded.success_rate}
        fittable = None
        if self.fittable_incumbent is not None:
            fittable = {k: v for k, v in self.fittable_incumbent.items() if k not in ("policy", "code")}
            fittable["promotion_success_rate"] = self._fittable_incumbent_result.success_rate
        payload = {
            "generation": gen,
            "promotion_manifest_id": expanded.manifest_id,
            "excluded_manifest_ids": [self.screen_manifest.manifest_id, self.probe_manifest.manifest_id],
            "null_incumbent_id": NULL_INCUMBENT_ID,
            "performance_incumbent": performance,
            "fittable_incumbent": fittable,
            "rules": {
                "performance_incumbent": "either lane; " + PROMOTION_RULE,
                "fittable_incumbent": (
                    "fittable-lane candidate with selected_step > 0 whose fitted state beats its own step-0 state "
                    "and the current fittable incumbent, both on the promotion manifest under the same strict rule"
                ),
            },
            "performance_comparisons": [d.get("incumbent_comparison") for d in self.promotion_log if d.get("incumbent_comparison")],
            "fittable_decisions": self.fittable_decisions,
        }
        return self._artifact("incumbents.json", json.dumps(payload, indent=2, default=str))

    def _update_records_after_promotion(self, decisions, by_id, expanded, fittable_decisions=None):
        """Fold screening, expansion and the archive's confirmed metrics into each record."""
        from lares.search import STATUS_PROMOTED, STATUS_REJECTED
        from lares.search.lanes import FITTABLE_INCUMBENT, PERFORMANCE_INCUMBENT

        fittable_decisions = fittable_decisions or {}
        for decision in decisions:
            result = by_id.get(decision["candidate_id"])
            if result is None or result.get("record") is None:
                continue
            record = result["record"]
            comparison = decision.get("incumbent_comparison") or {}
            promoted = bool(comparison.get("promoted"))
            fittable = fittable_decisions.get(decision["candidate_id"])
            fittable_promoted = bool(fittable and fittable.get("promoted"))
            record.status = STATUS_PROMOTED if promoted or fittable_promoted else STATUS_REJECTED
            record.final_disposition = (
                f"screening: {decision['reason']}; promotion: {comparison['reason']}"
                if comparison
                else f"screening: {decision['reason']}"
                + (f"; promotion: {decision['promotion_evaluation']}" if decision.get("promotion_evaluation") else "")
            )
            if self.lane_plan is not None:
                record.evaluation["incumbents"] = {PERFORMANCE_INCUMBENT: promoted, FITTABLE_INCUMBENT: fittable_promoted}
                if fittable is not None:
                    record.evaluation[FITTABLE_INCUMBENT] = fittable
                    record.final_disposition += f"; fittable incumbent: {fittable['reason']}"
                if result.get("step0_expanded_record") is not None:
                    record.evaluation["promotion_step0_success_rate"] = result["step0_expanded_record"].success_rate
            record.evaluation["screening"] = {
                k: v for k, v in decision.items() if k != "incumbent_comparison"
            }
            if comparison:
                record.evaluation["incumbent_comparison"] = comparison
            expanded_report = result.get("expanded_report")
            if expanded_report is not None:
                record.evaluation["manifest_ids"] = sorted(
                    set(record.evaluation["manifest_ids"]) | {expanded.manifest_id}
                )
                record.evaluation["promotion_manifest_id"] = expanded.manifest_id
                record.evaluation["expanded_success_rate"] = expanded_report.rollout.success_rate
                record.evaluation["expanded_mean_return"] = expanded_report.rollout.mean_return
                record.evaluation["reports"] = dict(result["report_refs"])
            entry = self.archive.get(result["candidate_id"])
            if entry is not None:
                record.evaluation["archive"] = {
                    "metrics_source": entry.metrics_source,
                    "screen_metrics": entry.screen_metrics,
                    "confirmed_metrics": entry.confirmed_metrics,
                }
            record.resources = result["resources"].to_dict()
            self._save_record(record)

    @staticmethod
    def _score(eval_result):
        """Composite fitness: success rate dominates, mean reward breaks ties."""
        return eval_result["success_rate"] * 1000 + eval_result["mean_reward"]

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def run(self, client, envs, demo_buffer, args, logger=None):
        """Execute the full evolution loop.

        Args:
            client: OpenAI client.
            envs: :class:`~lares.eval.runner.PipelineEnvs` holding one env per
                stream: development evaluation, checkpoint probe, and GIF
                recording, so none can shift another's cases.
            demo_buffer: :class:`DemoBuffer` with expert demonstrations and
                episode ids.
            args: Namespace with ``model``, ``policy_gen_two_phase`` (must be
                true) and ``policy_impl_mode``.
            logger: Optional :class:`~lares.core.training_logger.TrainingLogger`.

        Returns:
            Dict with ``policy``, ``code``, ``score``, ``eval``, ``response``
            for the incumbent found across all generations, or ``None`` if no
            candidate was ever promoted.
        """
        from lares.core.policy_generation import final_attempts
        from lares.eval.promotion import (
            NOT_BEST_SCREENED_CHALLENGER,
            holm_promotion,
            summarise_decisions,
        )
        from lares.eval.report import CHECKPOINT_FITTED
        from lares.fitting.optimizers import prepare_data
        from lares.search.generation import LLMCallLedger, write_feedback_bundle

        if self.dev_manifest is None:
            raise ValueError(
                "EvolutionOrchestrator needs a dev_manifest: every candidate must be "
                "scored on the same ordered cases (spec.md FR-1/FR-4)."
            )
        if not bool(getattr(args, "policy_gen_two_phase", False)):
            raise ValueError(
                "EvolutionOrchestrator needs policy_gen_two_phase: true, so every candidate "
                "carries an idea (and from generation 1 a proposal) by construction."
            )
        probe_env, probe_stream = self._probe_env(envs)
        if logger is None:
            from lares.core.training_logger import TrainingLogger

            logger = TrainingLogger(log_dir=self.log_dir, task_name=self.env_name)

        os.makedirs(self.log_dir, exist_ok=True)
        llm_transcript_path = os.path.join(self.log_dir, "llm_evolution_transcript.log")
        with open(llm_transcript_path, "w", encoding="utf-8") as _tf:
            _tf.write(
                f"LLM evolution transcript\n"
                f"task={self.env_name}\n"
                f"model={getattr(args, 'model', '')} reasoning_effort={getattr(args, 'reasoning_effort', None)}\n"
                f"policy_gen_two_phase={getattr(args, 'policy_gen_two_phase', False)}\n"
                f"policy_impl_mode={_impl_mode(args)}\n"
                f"---\n"
            )
        print(f"  LLM message log: {llm_transcript_path}")

        ensure_mujoco_headless_gl()
        self.ledger = LLMCallLedger(self.log_dir)
        self.data = prepare_data(demo_buffer, val_fraction=self.val_fraction, split_seed=self.split_seed)
        self._write_probe_batch(demo_buffer)
        self._write_run_manifest(args, probe_stream)
        self.check_manifest_isolation()
        if not self.probe_is_isolated:
            print("  WARNING: the checkpoint probe shares cases with screening; pass probe_manifest.")

        feedback_bundle = None
        parent_sources: dict = {}

        for gen in range(self.num_generations):
            print(f"\n{'=' * 60}")
            print(f"[Evolution] Generation {gen + 1}/{self.num_generations}")
            print(f"{'=' * 60}")
            gen_dir = os.path.join(self.log_dir, f"gen_{gen}")
            os.makedirs(gen_dir, exist_ok=True)

            # --- Structure search: typed candidates with lineage -------------
            started = time.time()
            if self.lane_plan is not None:
                candidates = two_lane_generation(
                    client=client, env_name=self.env_name, obs_dim=self.obs_dim, action_dim=self.action_dim,
                    args=args, plan=self.lane_plan, log_dir=self.log_dir, logger=logger,
                    llm_transcript_path=llm_transcript_path, ledger=self.ledger,
                    max_total_calls=self.max_llm_calls_per_generation,
                    max_repair_per_candidate=self.max_repair_per_candidate, param_budget=self.param_budget,
                    max_total_tokens=self.max_llm_tokens_per_generation,
                )
            else:
              candidates = llm_evolution(
                client=client,
                env_name=self.env_name,
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                args=args,
                pop_size=self.pop_size,
                generation=gen,
                log_dir=self.log_dir,
                logger=logger,
                llm_transcript_path=llm_transcript_path,
                ledger=self.ledger,
                feedback_bundle=feedback_bundle,
                parent_sources=parent_sources,
                max_total_calls=self.max_llm_calls_per_generation,
                max_repair_per_candidate=self.max_repair_per_candidate,
                param_budget=self.param_budget,
                max_total_tokens=self.max_llm_tokens_per_generation,
              )
            self.run_resources.add_wall_time("generation", time.time() - started)

            final_by_slot = {c.candidate_id: c for c in final_attempts(candidates)}
            failures = [c for c in final_by_slot.values() if not c.valid]
            for c in candidates:
                if not (c.valid and final_by_slot[c.candidate_id] is c):
                    self._record_failed_attempt(c, args, final_by_slot, feedback_bundle)

            # --- Inner loop: fit, select, screen ------------------------------
            screen = self.screen_manifest
            expanded = self.promotion_manifest
            gen_results = []
            for c in final_by_slot.values():
                if not c.valid:
                    continue
                lane_note = f"{c.lane} lane; " if c.lane else ""
                print(f"\n  --- {c.attempt_id} ({lane_note}{c.mode or 'idea'}; parents {c.parent_ids}) ---")
                result = self._fit_and_screen(c, envs, probe_env, args, feedback_bundle)
                self.run_resources.merge(
                    __import__("lares.search.resources", fromlist=["ResourceLedger"]).ResourceLedger(
                        wall_time_seconds=dict(result["resources"].wall_time_seconds)
                    )
                )
                gen_results.append(result)
                self.history.append(result)
                parent_sources[c.candidate_id] = c.code
                print(f"    zero-shot   {result['zero_shot_report'].headline()}")
                if result["intermediate_report"] is not None:
                    print(f"    intermediate {result['intermediate_report'].headline()}")
                print(f"    fitted      {result['screen_report'].headline()}")

            decisions = []
            if not gen_results:
                print(f"  WARNING: Generation {gen + 1} produced no valid candidates.")
            else:
                # --- Screening: picks the challenger; it never replaces the incumbent ---
                decisions = self.promotion_policy.screen(
                    [
                        {"candidate_id": r["candidate_id"], "report": r["screen_report"], "valid": True}
                        for r in gen_results
                    ]
                )
                print(f"\n  Screening on {len(screen)} cases (rule declared before the generation ran):")
                print("    " + summarise_decisions(decisions).replace("\n", "\n    "))
                by_id = {r["candidate_id"]: r for r in gen_results}
                cleared = sorted(
                    (d for d in decisions if d["promoted"]),
                    key=lambda d: (d["success_rate"], d["progress"], d["mean_return"]),
                    reverse=True,
                )
                challengers = cleared[: self.promotion_challengers]
                for d in decisions:
                    d["generation"] = gen
                    d["screen_cleared"] = bool(d["promoted"])
                    d["promotion_tested"] = any(d is c for c in challengers)
                    if d["promoted"] and not d["promotion_tested"]:
                        d["promotion_evaluation"] = NOT_BEST_SCREENED_CHALLENGER

                # --- Promotion: best screened challenger(s) on the promotion set -----
                tested = []
                for d in challengers:
                    r = by_id[d["candidate_id"]]
                    started = time.time()
                    exp_result, exp_report = self._evaluate_checkpoint(
                        r["policy"], envs.development, expanded, r["candidate_id"],
                        CHECKPOINT_FITTED, bc_stats=r["bc_stats"],
                    )
                    r["resources"].add_episodes("expanded", exp_result)
                    elapsed = time.time() - started
                    r["resources"].add_wall_time("evaluation", elapsed)
                    self.run_resources.add_wall_time("evaluation", elapsed)
                    r["report_refs"]["fitted_expanded"] = self._save_report(
                        exp_report, f"{r['candidate_id']}_expanded.json", gen
                    )
                    self.archive.confirm(r["candidate_id"], exp_report)
                    r["expanded_record"] = exp_result
                    r["expanded_report"] = exp_report
                    r["eval"] = exp_result.fitness_dict()
                    r["score"] = self._score(r["eval"])
                    print(f"    promotion set {exp_report.headline()}")
                    tested.append((d, r, exp_result, exp_report))

                comparisons = holm_promotion(
                    [(r["candidate_id"], exp_result) for _, r, exp_result, _ in tested],
                    self._incumbent_expanded,
                    self._incumbent_id,
                    excluded_manifest_ids=(screen.manifest_id, self.probe_manifest.manifest_id),
                )
                for (d, r, exp_result, exp_report), comparison in zip(tested, comparisons):
                    d["incumbent_comparison"] = comparison.to_dict()
                    ps = comparison.paired_success
                    low = "n/a" if ps["ci_low"] is None else f"{ps['ci_low']:+.3f}"
                    high = "n/a" if ps["ci_high"] is None else f"{ps['ci_high']:+.3f}"
                    print(
                        f"      paired vs {comparison.incumbent_id}: success {ps['mean_difference']:+.3f} "
                        f"{comparison.interval_level:.1%} t interval [{low}, {high}] "
                        f"n={ps['n_paired']}  -> {comparison.reason}"
                    )
                    if comparison.promoted:
                        self._incumbent_expanded = exp_result
                        self._incumbent_id = r["candidate_id"]
                        self.best_overall = {
                            "policy": copy.deepcopy(r["policy"]),
                            "code": r["code"],
                            "score": r["score"],
                            "eval": r["eval"],
                            "response": r["response"],
                            "candidate_id": r["candidate_id"],
                            "attempt_id": r["attempt_id"],
                            "report": exp_report,
                        }
                fittable_decisions = {}
                if self.lane_plan is not None:
                    fittable_decisions = self._fittable_promotion(gen_results, envs, expanded, screen, gen)
                self.promotion_log.extend(decisions)
                self._update_records_after_promotion(decisions, by_id, expanded, fittable_decisions)
                if self.lane_plan is not None:
                    self._write_incumbents(gen, expanded)

            # --- Evidence for the next generation, hashed as a bundle ---------
            text, context = self._build_evidence(gen, gen_results, failures, screen, demo_buffer=demo_buffer)
            feedback_bundle = write_feedback_bundle(self.log_dir, gen, screen.manifest_id, context, text)
            self.feedback_bundles.append(feedback_bundle)
            print(f"\n  Evidence for the next generation: {feedback_bundle.text_path} "
                  f"({feedback_bundle.feedback_id}; context {context})")
            print("\n  Archive:")
            print("    " + self.archive.summary().replace("\n", "\n    "))

            if gen_results:
                gen_results.sort(key=lambda r: r["score"], reverse=True)
                scores = [r["score"] for r in gen_results]
                elite_scores = (
                    [r["score"] for r in gen_results[: self.elite_num]] if self.elite_num > 0 else scores
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
                    scope = "expanded" if r["expanded_record"] is not None else "screen"
                    print(
                        f"    [{rank + 1}] {r['candidate_id']} reward={r['eval']['mean_reward']:.2f}, "
                        f"success={r['eval']['success_rate']:.2f} on {scope}"
                    )

                if self.record_demo_gif:
                    gif_case = self.gif_case or self.dev_manifest.episodes[0]
                    gif_path = os.path.join(gen_dir, f"gen_{gen}_best.gif")
                    print(f"\n  Recording demo GIF ({gif_case.case_id}) → {gif_path}")
                    try:
                        gif_info = record_episode_gif(
                            gen_results[0]["policy"], envs.gif, path=gif_path, case=gif_case,
                            max_steps=self.eval_max_steps, deterministic=True,
                        )
                    except Exception as exc:
                        print(
                            f"  [record_gif] Skipped ({type(exc).__name__}: {exc}). "
                            "Set MUJOCO_GL=egl (or osmesa) before creating the env on headless nodes."
                        )
                        gif_info = {"saved": False, "num_frames": 0, "episode_reward": 0.0, "success": False}
                    if gif_info["saved"]:
                        print(
                            f"  GIF saved ({gif_info['num_frames']} frames, "
                            f"reward={gif_info['episode_reward']:.2f}, success={gif_info['success']})"
                        )

            with open(os.path.join(gen_dir, "results.pkl"), "wb") as f:
                pickle.dump(
                    {
                        "generation": gen,
                        "screen_manifest_id": screen.manifest_id,
                        "expanded_manifest_id": expanded.manifest_id,
                        "probe_manifest_id": self.probe_manifest.manifest_id,
                        "promotion_policy": self.promotion_policy.to_dict(),
                        "screening_decisions": decisions,
                        "feedback_id": feedback_bundle.feedback_id,
                        "attempts": [
                            {"attempt_id": c.attempt_id, "valid": c.valid, "stage": c.validation.get("stage")}
                            for c in candidates
                        ],
                        "candidates": [
                            {
                                "candidate_id": r["candidate_id"],
                                "attempt_id": r["attempt_id"],
                                "code": r["code"],
                                "eval": r["eval"],
                                "score": r["score"],
                                "selected_step": r["fit"].selected_step,
                                "lane": r.get("lane", ""),
                                "fitting_rejected": r.get("fitting_rejected"),
                                "screen_report": r["screen_report"].to_dict(),
                                "zero_shot_report": r["zero_shot_report"].to_dict(),
                                "intermediate_report": (
                                    r["intermediate_report"].to_dict()
                                    if r["intermediate_report"] is not None
                                    else None
                                ),
                                "expanded_report": (
                                    r["expanded_report"].to_dict() if r["expanded_report"] is not None else None
                                ),
                            }
                            for r in gen_results
                        ],
                    },
                    f,
                )
            with open(os.path.join(gen_dir, "screening.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "generation": gen,
                        "promotion_policy": self.promotion_policy.to_dict(),
                        "screen_manifest_id": screen.manifest_id,
                        "expanded_manifest_id": expanded.manifest_id,
                        "decisions": decisions,
                    },
                    f,
                    indent=2,
                    default=str,
                )
            with open(os.path.join(self.log_dir, "archive.json"), "w", encoding="utf-8") as f:
                json.dump(self.archive.to_dict(), f, indent=2, default=str)
            self._write_run_totals()

        return self.best_overall
