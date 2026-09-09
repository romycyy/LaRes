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
import os
import copy
import pickle
import time
from collections import defaultdict
from functools import lru_cache

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
    """MSE between ``tanh(mean)`` and the expert action, per action axis."""
    with torch.no_grad():
        mean, _ = policy(torch.tensor(obs_np, dtype=torch.float32))
        target = torch.tensor(actions_np, dtype=torch.float32)
        per_axis = ((torch.tanh(mean) - target) ** 2).mean(dim=0)
    return [float(x) for x in per_axis]


def _bound_activity(policy, atol=1e-6):
    """Fraction of each parameter's elements sitting on a declared bound.

    A parameter pinned at its bound is the optimiser saying the range is wrong,
    or that the structure needs a magnitude it is not allowed to have.
    """
    try:
        ranges = policy.get_param_ranges()
    except Exception:
        return None
    out = {}
    for name, param in policy.named_parameters():
        if name not in ranges:
            continue
        lo, hi = ranges[name]
        value = param.detach()
        at_low = (value <= lo + atol).float().mean().item()
        at_high = (value >= hi - atol).float().mean().item()
        out[name] = {
            "at_lower": float(at_low),
            "at_upper": float(at_high),
            "value": [float(x) for x in value.reshape(-1)],
            "range": [float(lo), float(hi)],
        }
    return out or None


def _loss_by_phase(policy, obs_np, actions_np, threshold=0.5):
    """Split the fitting error by which exposed gate was open.

    Tells a structural failure from a fitting one: if error is concentrated in
    one phase, that phase's expression is the thing to repair.
    """
    reset_diagnostics = getattr(policy, "reset_diagnostics", None)
    if not callable(reset_diagnostics):
        return None
    reset_diagnostics()
    obs_t = torch.tensor(obs_np, dtype=torch.float32)
    target = torch.tensor(actions_np, dtype=torch.float32)
    with torch.no_grad():
        mean, _ = policy(obs_t)
        gates = dict(getattr(policy, "last_gates", {}) or {})
        if not gates:
            return None
        error = ((torch.tanh(mean) - target) ** 2).mean(dim=-1)
    out = {}
    for name, value in gates.items():
        flat = value.reshape(value.shape[0], -1).mean(dim=-1)
        if flat.shape[0] != error.shape[0]:
            continue
        open_mask = flat > threshold
        out[name] = {
            "loss_when_open": float(error[open_mask].mean()) if bool(open_mask.any()) else None,
            "loss_when_closed": (
                float(error[~open_mask].mean()) if bool((~open_mask).any()) else None
            ),
            "fraction_open": float(open_mask.float().mean()),
        }
    return out or None


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
            policy.eval()
            score = rollout_probe(policy)
            policy.train()
            stats["rollout_history"].append((step + 1, score))
            # ``>=`` rather than ``>``: on an exact tie the later checkpoint wins.
            # When every probe scores zero the earlier ones carry no information,
            # and reverting to the first would throw away fitting for nothing.
            if stats["best_rollout_score"] is None or score >= stats["best_rollout_score"]:
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
    """Read the six prompt templates from disk. Cached: called once per generation."""
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
    evidence_block=None,
    failure_log=None,
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
        evidence_block: Optional structured feedback from
            :func:`lares.search.feedback.build_feedback`. When present it replaces
            the two-scalar summary entirely.
        failure_log: Optional list; candidates that never reached evaluation are
            appended to it rather than dropped.
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
    # The generator sees structured evidence when the caller assembled it, and
    # falls back to the guidance template only when it did not. The old path sent
    # two floats, which cannot separate a structural mistake from a tuning one.
    elite_response = None
    code_feedback = None
    if evidence_block:
        code_feedback = evidence_block + "\n\n" + prompts["code_feedback_tmpl"]
        if previous_results:
            elite_response = max(previous_results, key=lambda r: r["score"])["response"]
    elif previous_results:
        best_prev = max(previous_results, key=lambda r: r["score"])
        elite_response = best_prev["response"]
        code_feedback = (
            f"## Previous generation\n"
            f"Best candidate: success {best_prev['eval']['success_rate']:.3f}, "
            f"mean return {best_prev['eval']['mean_reward']:.2f} after "
            f"{best_prev.get('train_steps', 0)} behavioural-cloning gradient steps.\n"
            f"No structured diagnostics were supplied for this generation.\n\n"
            + prompts["code_feedback_tmpl"]
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
        env_name=env_name,
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
        failure_log=failure_log,
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
        dev_manifest=None,
        train_cases=(),
        gif_case=None,
        eval_max_steps=150,
        promotion_policy=None,
        rollout_checkpoint_probes=4,
    ):
        from lares.eval.promotion import PromotionPolicy

        self.dev_manifest = dev_manifest
        self.train_cases = list(train_cases)
        self.gif_case = gif_case
        self.eval_max_steps = int(eval_max_steps)
        # Declared before any generation runs, and stored with the results, so a
        # promotion can be read back against the threshold that justified it.
        self.promotion_policy = promotion_policy or PromotionPolicy()
        # How many times to score development rollouts during fitting. Success is
        # not monotonic in gradient steps on this task while validation loss is,
        # so a fixed budget or a loss-based checkpoint lands past the useful
        # parameters. Set to 0 to fit to the end as the pipeline used to.
        self.rollout_checkpoint_probes = int(rollout_checkpoint_probes)
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
        self.reports: list = []
        self.promotion_log: list = []
        self._incumbent_expanded = None
        self._incumbent_id = None
        # Phase 4: one record per attempted candidate, an archive that keeps a
        # simple and a behaviourally distinct policy alongside the best one, and
        # the evidence block the next generation is shown.
        from lares.search import Archive

        self.archive = Archive()
        self.records: list = []
        self.records_dir = os.path.join(self.log_dir, "experiments")
        self.evidence_block: str | None = None
        self.generation_history: list = []

    # ------------------------------------------------------------------
    #  Manifests
    # ------------------------------------------------------------------

    @property
    def screen_manifest(self):
        """The fixed screening subset: a prefix, so it nests inside the expanded set."""
        n = min(self.promotion_policy.screen_episodes, len(self.dev_manifest))
        return self.dev_manifest.head(n)

    @property
    def expanded_manifest(self):
        """The larger development subset a screened contender is re-scored on."""
        n = min(self.promotion_policy.expanded_episodes, len(self.dev_manifest))
        return self.dev_manifest.head(n)

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    #  Experiment records
    # ------------------------------------------------------------------

    def _save_record(self, record):
        """Validate and persist one experiment record, then keep it in memory."""
        record.validate()
        record.save(self.records_dir)
        self.records.append(record)
        return record

    def _record_generation_failures(self, generation, failure_log, args):
        """One record per candidate that never reached evaluation.

        Dropping these is how a search loses the evidence that would explain why
        it is not improving (``spec.md`` FR-10, AC-4).
        """
        from lares.search import STATUS_INVALID, ExperimentRecord, source_hash

        for i, failure in enumerate(failure_log or []):
            code = failure.get("code", "")
            self._save_record(
                ExperimentRecord(
                    experiment_id=f"gen{generation}_invalid{i}",
                    candidate_id=failure.get("candidate_id", f"gen{generation}_invalid{i}"),
                    generation=generation,
                    status=STATUS_INVALID,
                    code={
                        "source_hash": source_hash(code) if code else "empty",
                        "source": code,
                    },
                    llm={
                        "model_id": getattr(args, "model", ""),
                        "two_phase": bool(getattr(args, "policy_gen_two_phase", False)),
                        "impl_mode": getattr(args, "policy_impl_mode", "batched"),
                    },
                    validation={
                        "passed": False,
                        "stage": failure.get("stage", "unknown"),
                        "repair_attempts": failure.get("repair_attempts", 0),
                        "errors": [failure.get("error", "")],
                    },
                    timestamps={"recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                    final_disposition="never reached rollout validation",
                )
            )

    def _record_candidate(self, result, args, manifest, envs):
        """Record one evaluated candidate and place it in the archive."""
        from lares.search import STATUS_EVALUATED, ExperimentRecord, source_hash

        report = result["screen_report"]
        bc_stats = result.get("bc_stats") or {}
        record = ExperimentRecord(
            experiment_id=f"gen{result['generation']}_cand{result['candidate_idx']}",
            candidate_id=result["candidate_id"],
            generation=result["generation"],
            status=STATUS_EVALUATED,
            code={
                "source_hash": source_hash(result["code"]),
                "source": result["code"],
                "num_parameters": result["policy"].count_parameters(),
            },
            llm={
                "model_id": getattr(args, "model", ""),
                "two_phase": bool(getattr(args, "policy_gen_two_phase", False)),
                "impl_mode": getattr(args, "policy_impl_mode", "batched"),
                "response": result.get("response", ""),
            },
            validation={"passed": True, "errors": []},
            fitting={
                "optimizer": "adam_baseline",
                "steps": self.bc_steps,
                "split": bc_stats.get("split"),
                "train_loss_end": bc_stats.get("final_loss"),
                "validation_loss_end": bc_stats.get("validation_loss_end"),
                "bound_activity": bc_stats.get("bound_activity"),
                "gradient_norms": bc_stats.get("gradient_norms"),
            },
            evaluation={
                "manifest_ids": [manifest.manifest_id],
                "action_mode": report.action_mode,
                "checkpoints": {
                    "zero_shot": result["zero_shot_report"].rollout.success_rate,
                    "fitted": report.rollout.success_rate,
                },
                "success_rate": report.rollout.success_rate,
                "mean_return": report.rollout.mean_return,
                "failure_labels": report.rollout.failure_labels,
            },
            resources={
                "environment_steps": result.get("train_steps", 0),
                "bc_gradient_steps": self.bc_steps,
            },
            timestamps={"recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
            final_disposition="screened",
        )
        self._save_record(record)
        self.archive.add(
            candidate_id=result["candidate_id"],
            generation=result["generation"],
            report=report,
            num_parameters=result["policy"].count_parameters(),
            source_hash=record.code["source_hash"],
        )
        return record

    def _update_records_after_promotion(self, decisions, by_id, expanded):
        """Fold the screening outcome back into each candidate's record."""
        from lares.search import STATUS_PROMOTED, STATUS_REJECTED

        by_candidate = {r.candidate_id: r for r in self.records}
        for decision in decisions:
            record = by_candidate.get(decision["candidate_id"])
            if record is None:
                continue
            result = by_id.get(decision["candidate_id"], {})
            comparison = decision.get("incumbent_comparison") or {}
            promoted = bool(comparison.get("promoted"))
            record.status = STATUS_PROMOTED if promoted else STATUS_REJECTED
            record.final_disposition = (
                f"screening: {decision['reason']}; incumbent: {comparison['reason']}"
                if comparison
                else f"screening: {decision['reason']}"
            )
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
                record.evaluation["expanded_success_rate"] = (
                    expanded_report.rollout.success_rate
                )
            record.validate()
            record.save(self.records_dir)

    def _build_evidence(self, generation, gen_results, failure_log, manifest, demo_buffer=None):
        """Assemble the structured feedback block for the next generation.

        Sensitivity is probed on real observations from the demonstration buffer.
        An all-zero or standard-normal batch puts the end-effector, object and
        goal on top of each other or metres apart, and every direction term then
        reads as dead, which would tell the generator to delete the controller.
        """
        from lares.search import CandidateEvidence, build_feedback

        probe = None
        if demo_buffer is not None and len(demo_buffer) > 0:
            obs_np, _, _, _, _ = demo_buffer.sample(min(64, len(demo_buffer)))
            probe = torch.tensor(obs_np, dtype=torch.float32)

        evidence = []
        for result in gen_results:
            sensitivity = []
            if probe is not None:
                try:
                    from lares.fitting.sensitivity import action_sensitivity, dead_parameters

                    sensitivity = dead_parameters(
                        action_sensitivity(result["policy"], probe)
                    )
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
        for failure in failure_log or []:
            evidence.append(
                CandidateEvidence(
                    candidate_id=failure.get("candidate_id", "unknown"),
                    code=failure.get("code", ""),
                    report=None,
                    error=failure.get("error", ""),
                )
            )
        best = max((e.success for e in evidence if e.report is not None), default=0.0)
        self.generation_history.append(
            f"generation {generation}: {len(gen_results)} valid, "
            f"{len(failure_log or [])} invalid, best success {best:.3f}"
        )
        return build_feedback(
            evidence,
            generation=generation,
            manifest_id=manifest.manifest_id,
            history=self.generation_history,
            archive=self.archive,
        )

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
        self.reports.append(report)
        return result, report

    def _rollout_probe(self, env, manifest):
        """Score a policy mid-fitting on the screening cases."""
        from lares.eval.runner import PolicyActor, evaluate_manifest

        def probe(policy):
            result = evaluate_manifest(
                PolicyActor(policy, deterministic=True, name="probe"),
                manifest,
                env,
                max_steps=self.eval_max_steps,
            )
            # Success first, then goal progress, matching how the promotion policy
            # ranks candidates. Without the secondary key every checkpoint of a
            # failing candidate ties and the choice is arbitrary.
            progress = [
                e.signed_goal_progress
                for e in result.episodes
                if e.signed_goal_progress is not None
            ]
            return (
                result.success_rate,
                float(np.mean(progress)) if progress else 0.0,
            )

        return probe

    def _train_candidate(
        self, policy, demo_buffer, env, logger=None, checkpoint_callback=None,
        rollout_probe=None,
    ):
        """Run the BC → RL inner loop on a single policy candidate."""
        interval = (
            max(1, self.bc_steps // self.rollout_checkpoint_probes)
            if rollout_probe is not None and self.rollout_checkpoint_probes > 0
            else 0
        )
        bc_stats = behavioral_cloning(
            policy,
            demo_buffer,
            num_steps=self.bc_steps,
            batch_size=min(256, len(demo_buffer)),
            log_interval=self.bc_steps,
            logger=logger,
            task_name=self.env_name,
            checkpoint_callback=checkpoint_callback,
            rollout_probe=rollout_probe,
            rollout_interval=interval,
        )
        if self.rl_iterations <= 0:
            return bc_stats, {"returns": [], "successes": [], "skipped": True}
        if not self.train_cases:
            raise ValueError(
                "rl_iterations > 0 requires train_cases; RL rollouts must name the "
                "training placements they run on."
            )
        rl_stats = rl_finetune(
            policy,
            env,
            self.train_cases,
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

    def run(self, client, envs, demo_buffer, args, logger=None):
        """Execute the full evolution loop.

        Args:
            client: OpenAI client.
            envs: :class:`~lares.eval.runner.PipelineEnvs` holding one env per
                stream. Development evaluation, RL collection and GIF recording
                each get their own, so none can shift another's cases.
            demo_buffer: :class:`DemoBuffer` with expert demonstrations.
            args: Namespace with ``model`` attribute for the LLM.
            logger: Optional :class:`~lares.core.training_logger.TrainingLogger`.

        Returns:
            Dict with ``policy``, ``code``, ``score``, ``eval``, ``response``
            for the best candidate found across all generations.  Returns
            ``None`` if no valid candidate was ever generated.
        """
        if self.dev_manifest is None:
            raise ValueError(
                "EvolutionOrchestrator needs a dev_manifest: every candidate must be "
                "scored on the same ordered cases (spec.md FR-1/FR-4)."
            )
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
            failure_log: list = []
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
                evidence_block=self.evidence_block,
                failure_log=failure_log,
            )
            self._record_generation_failures(gen, failure_log, args)

            # --- Inner loop: train and evaluate every candidate ---
            from lares.eval.report import (
                CHECKPOINT_FITTED,
                CHECKPOINT_INTERMEDIATE,
                CHECKPOINT_ZERO_SHOT,
            )
            from lares.eval.promotion import compare_to_incumbent, summarise_decisions

            screen = self.screen_manifest
            expanded = self.expanded_manifest
            reports_dir = os.path.join(gen_dir, "reports")
            gen_results = []
            for i, (policy, code, response) in enumerate(
                zip(policy_pop, code_pop, response_pop)
            ):
                candidate_id = f"gen{gen}_cand{i}"
                print(f"\n  --- Candidate {i + 1}/{len(policy_pop)}  ({candidate_id}) ---")

                # Zero-shot: what the LLM's own initial values do untrained. If a
                # structure is decent here and worse after fitting, the fault is
                # in the fitting, not the structure.
                _, zero_report = self._evaluate_checkpoint(
                    policy, envs.development, screen, candidate_id, CHECKPOINT_ZERO_SHOT
                )

                halfway_state = {}

                def _capture(step, pol, _store=halfway_state):
                    _store["step"] = step
                    _store["state"] = copy.deepcopy(pol.state_dict())

                # The probe runs on its own environment, so scoring mid-fitting
                # cannot disturb the case stream the final evaluation replays.
                probe = (
                    self._rollout_probe(envs.gif, screen)
                    if self.rollout_checkpoint_probes > 0
                    else None
                )
                bc_stats, rl_stats = self._train_candidate(
                    policy, demo_buffer, envs.rl, logger=logger,
                    checkpoint_callback=_capture, rollout_probe=probe,
                )
                if bc_stats.get("best_rollout_state") is not None:
                    success, progress = bc_stats["best_rollout_score"]
                    print(
                        f"    best mid-fitting checkpoint at step "
                        f"{bc_stats['best_rollout_step']}/{self.bc_steps} "
                        f"(screen success {success:.3f}, progress {progress:+.4f}); "
                        f"fitting to the end would have used step {self.bc_steps}"
                    )
                    policy.load_state_dict(bc_stats["best_rollout_state"])

                intermediate_report = None
                if halfway_state:
                    mid = copy.deepcopy(policy)
                    mid.load_state_dict(halfway_state["state"])
                    _, intermediate_report = self._evaluate_checkpoint(
                        mid, envs.development, screen, candidate_id, CHECKPOINT_INTERMEDIATE
                    )

                screen_result, screen_report = self._evaluate_checkpoint(
                    policy,
                    envs.development,
                    screen,
                    candidate_id,
                    CHECKPOINT_FITTED,
                    bc_stats=bc_stats,
                )
                for report in (zero_report, intermediate_report, screen_report):
                    if report is not None:
                        report.save(
                            os.path.join(
                                reports_dir, f"{candidate_id}_{report.checkpoint}.json"
                            )
                        )

                train_steps = (
                    self.bc_steps + self.rl_iterations * self.rl_episodes_per_iter * 150
                )
                gen_results.append(
                    {
                        "candidate_id": candidate_id,
                        "policy": policy,
                        "code": code,
                        "response": response,
                        "eval": screen_result.fitness_dict(),
                        "eval_record": screen_result,
                        "screen_report": screen_report,
                        "zero_shot_report": zero_report,
                        "intermediate_report": intermediate_report,
                        "expanded_record": None,
                        "expanded_report": None,
                        "score": self._score(screen_result.fitness_dict()),
                        "bc_stats": bc_stats,
                        "rl_stats": rl_stats,
                        "train_steps": train_steps,
                        "generation": gen,
                        "candidate_idx": i,
                    }
                )
                self.history.append(gen_results[-1])
                self._record_candidate(gen_results[-1], args, screen, envs)
                print(f"    zero-shot   {zero_report.headline()}")
                if intermediate_report is not None:
                    print(f"    intermediate {intermediate_report.headline()}")
                print(f"    fitted      {screen_report.headline()}")

            if not gen_results:
                print(f"  WARNING: Generation {gen + 1} produced no valid candidates.")
                continue

            # --- Screening: only contenders earn the expanded evaluation ---
            decisions = self.promotion_policy.screen(
                [
                    {"candidate_id": r["candidate_id"], "report": r["screen_report"], "valid": True}
                    for r in gen_results
                ]
            )
            print(f"\n  Screening on {len(screen)} cases "
                  f"(rule declared before the generation ran):")
            print("    " + summarise_decisions(decisions).replace("\n", "\n    "))
            by_id = {r["candidate_id"]: r for r in gen_results}
            for d in decisions:
                d["generation"] = gen
                if not d["promoted"]:
                    continue
                r = by_id[d["candidate_id"]]
                exp_result, exp_report = self._evaluate_checkpoint(
                    r["policy"],
                    envs.development,
                    expanded,
                    r["candidate_id"],
                    CHECKPOINT_FITTED,
                    bc_stats=r["bc_stats"],
                )
                exp_report.save(
                    os.path.join(reports_dir, f"{r['candidate_id']}_expanded.json")
                )
                r["expanded_record"] = exp_result
                r["expanded_report"] = exp_report
                r["eval"] = exp_result.fitness_dict()
                r["score"] = self._score(r["eval"])
                comparison = compare_to_incumbent(
                    exp_result, self._incumbent_expanded, r["candidate_id"], self._incumbent_id
                )
                d["incumbent_comparison"] = comparison.to_dict()
                print(f"    expanded {exp_report.headline()}")
                if comparison.paired_success is not None:
                    ps = comparison.paired_success
                    print(
                        f"      paired vs {self._incumbent_id}: success "
                        f"{ps['mean_difference']:+.3f} "
                        f"95% CI [{ps['ci95_low']:+.3f}, {ps['ci95_high']:+.3f}] "
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
                        "report": exp_report,
                    }
            self.promotion_log.extend(decisions)
            self._update_records_after_promotion(decisions, by_id, expanded)

            # Evidence for the next generation, assembled before the population is
            # discarded. This replaces the two scalars the loop used to send.
            self.evidence_block = self._build_evidence(
                gen, gen_results, failure_log, screen, demo_buffer=demo_buffer
            )
            evidence_path = os.path.join(gen_dir, "feedback.md")
            with open(evidence_path, "w", encoding="utf-8") as f:
                f.write(self.evidence_block)
            print(f"\n  Evidence for the next generation: {evidence_path}")
            print("\n  Archive:")
            print("    " + self.archive.summary().replace("\n", "\n    "))

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
                scope = "expanded" if r["expanded_record"] is not None else "screen"
                print(
                    f"    [{rank + 1}] {r['candidate_id']} reward={r['eval']['mean_reward']:.2f}, "
                    f"success={r['eval']['success_rate']:.2f} on {scope}{tag}"
                )

            # Record a demo GIF of the best candidate from this generation
            if self.record_demo_gif:
                gif_case = self.gif_case or self.dev_manifest.episodes[0]
                gif_path = os.path.join(gen_dir, f"gen_{gen}_best.gif")
                print(f"\n  Recording demo GIF ({gif_case.case_id}) → {gif_path}")
                try:
                    gif_info = record_episode_gif(
                        gen_results[0]["policy"],
                        envs.gif,
                        path=gif_path,
                        case=gif_case,
                        max_steps=self.eval_max_steps,
                        deterministic=True,
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
                        "screen_manifest_id": screen.manifest_id,
                        "expanded_manifest_id": expanded.manifest_id,
                        "promotion_policy": self.promotion_policy.to_dict(),
                        "screening_decisions": decisions,
                        "candidates": [
                            {
                                "candidate_id": r["candidate_id"],
                                "code": r["code"],
                                "eval": r["eval"],
                                "score": r["score"],
                                "screen_report": r["screen_report"].to_dict(),
                                "zero_shot_report": r["zero_shot_report"].to_dict(),
                                "intermediate_report": (
                                    r["intermediate_report"].to_dict()
                                    if r["intermediate_report"] is not None
                                    else None
                                ),
                                "expanded_report": (
                                    r["expanded_report"].to_dict()
                                    if r["expanded_report"] is not None
                                    else None
                                ),
                            }
                            for r in gen_results
                        ],
                    },
                    f,
                )
            with open(os.path.join(gen_dir, "screening.json"), "w", encoding="utf-8") as f:
                import json as _json

                _json.dump(
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

        return self.best_overall
        
