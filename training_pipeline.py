"""
Four-stage training pipeline for symbolic policies in modified LaRes.

Stage 1 — Dataset Generation:  use MetaWorld expert policies to collect demos
Stage 2 — Behavioral Cloning:  supervised imitation of expert actions
Stage 3 — RL Fine-tuning:      GRPO-style policy gradient improvement
Stage 4 — LLM Evolution:       structure search with BC+RL inner loop

Each stage is usable independently or as part of the full pipeline via
``SymbolicPolicyPipeline``.
"""

import os
import copy
import pickle
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

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

    Returns:
        dict with training statistics.
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    stats = {"bc_loss": [], "mean_loss": [], "std_loss": []}

    policy.train()
    for step in range(num_steps):
        obs_np, actions_np, _, _, _ = demo_buffer.sample(batch_size)
        obs_t = torch.tensor(obs_np, dtype=torch.float32)
        actions_t = torch.tensor(actions_np, dtype=torch.float32)

        mean, std = policy(obs_t)

        mean_loss = nn.functional.mse_loss(mean, actions_t)
        std_loss = 0.01 * std.mean()
        loss = mean_loss + std_loss

        optimizer.zero_grad()
        loss.backward()
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_grad_norm)
        optimizer.step()
        policy.clip_params()

        stats["bc_loss"].append(loss.item())
        stats["mean_loss"].append(mean_loss.item())
        stats["std_loss"].append(std_loss.item())

        if log_interval > 0 and (step + 1) % log_interval == 0:
            recent = stats["bc_loss"][-log_interval:]
            print(
                f"  [Stage 2] step {step + 1}/{num_steps}: "
                f"loss={np.mean(recent):.6f}, "
                f"mean_loss={np.mean(stats['mean_loss'][-log_interval:]):.6f}, "
                f"std_loss={np.mean(stats['std_loss'][-log_interval:]):.6f}"
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

            next_obs, reward, done, info = env.step(
                env.action_space.high * action_np
            )

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

    Returns:
        dict with training statistics.
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
        trajectories = _collect_trajectories(
            policy, env, episodes_per_iter, max_steps
        )
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
        advantages_t = torch.tensor(
            np.array(all_advantages), dtype=torch.float32
        )
        actions_t = torch.tanh(pretanh_t)

        policy.train()
        mean, std = policy(obs_t)
        dist = torch.distributions.Normal(mean, std)

        log_probs = dist.log_prob(pretanh_t) - torch.log(
            1 - actions_t.pow(2) + 1e-6
        )
        log_probs = log_probs.sum(dim=-1)

        policy_loss = -(log_probs * advantages_t).mean()

        entropy = dist.entropy().mean()

        kl_loss = sum(
            ((param - bc_params[name]) ** 2).sum()
            for name, param in policy.named_parameters()
        )

        total_loss = policy_loss - entropy_coeff * entropy + kl_coeff * kl_loss

        optimizer.zero_grad()
        total_loss.backward()
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), clip_grad_norm)
        optimizer.step()
        policy.clip_params()

        stats["policy_loss"].append(policy_loss.item())
        stats["entropy"].append(entropy.item())
        stats["kl"].append(kl_loss.item())

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
    stats["final_mean_return"] = float(
        np.mean(stats["returns"][-episodes_per_iter:])
    )
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

            next_obs, reward, done, info = env.step(
                env.action_space.high * action
            )
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


# ===========================================================================
#  Stage 4 — LLM-Based Evolution / Structure Search
# ===========================================================================


def llm_evolution(
    client,
    env,
    env_name,
    demo_buffer,
    args,
    obs_dim,
    action_dim,
    num_generations=5,
    pop_size=5,
    elite_num=2,
    bc_steps=3000,
    rl_iterations=30,
    rl_episodes_per_iter=10,
    log_dir="./logs/evolution",
):
    """LLM-based evolution of symbolic policy structures.

    Each generation:
      1. ``get_symbolic_policies()`` proposes ``pop_size`` structures via the LLM.
      2. Each candidate is trained through the BC -> RL inner loop.
      3. Top ``elite_num`` are kept; performance is fed back to the LLM.

    Reuses the existing ``policy_generation.py`` pipeline — no separate
    code-gen infrastructure needed.

    Returns:
        dict with ``policy``, ``code``, ``score``, ``stats``, ``response``.
    """
    from policy_generation import (
        get_symbolic_policies,
        obs_description_dict,
        input_dict_for_policy,
    )

    os.makedirs(log_dir, exist_ok=True)

    root_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_dir = os.path.join(root_dir, "utils", "policy_prompts")

    def _read(filename):
        with open(os.path.join(prompt_dir, filename), "r") as f:
            return f.read()

    initial_system = _read("initial_system.txt")
    initial_user = _read("new_initial_user.txt")
    code_output_tip = _read("new_code_output_tip.txt")
    code_feedback_tmpl = _read("code_feedback.txt")

    obs_description = obs_description_dict.get(env_name, "")
    input_dict_string = input_dict_for_policy.get(env_name, "")
    task_description = TASK_DESCRIPTIONS.get(env_name, env_name)

    data_pkl_path = os.path.join(log_dir, "data.pkl")
    dummy_data = [{"obs": np.zeros(obs_dim)}]
    with open(data_pkl_path, "wb") as f:
        pickle.dump(dummy_data, f)

    best_overall = {
        "policy": None,
        "code": None,
        "score": -float("inf"),
        "stats": None,
        "response": None,
    }

    elite_response = None
    elite_feedback = None

    for gen in range(num_generations):
        print(f"\n{'=' * 60}")
        print(f"[Stage 4] Generation {gen + 1}/{num_generations}")
        print(f"{'=' * 60}")

        gen_dir = os.path.join(log_dir, f"gen_{gen}")
        os.makedirs(gen_dir, exist_ok=True)

        policy_pop, code_pop, response_list = get_symbolic_policies(
            client=client,
            dir_path=gen_dir,
            llm_iter=gen,
            args=args,
            obs_dim=obs_dim,
            action_dim=action_dim,
            initial_system=initial_system,
            initial_user=initial_user,
            task_description=task_description,
            obs_description=obs_description,
            input_dict_string=input_dict_string,
            code_output_tip=code_output_tip,
            data_pkl_path=data_pkl_path,
            provided_response=elite_response,
            code_feedback=elite_feedback,
            real_num=pop_size,
        )

        candidates = []
        for i, (policy, code, response) in enumerate(
            zip(policy_pop, code_pop, response_list)
        ):
            print(f"\n  --- Candidate {i + 1}/{len(policy_pop)} ---")

            bc_stats = behavioral_cloning(
                policy,
                demo_buffer,
                num_steps=bc_steps,
                batch_size=min(256, len(demo_buffer)),
                log_interval=bc_steps,
            )

            rl_stats = rl_finetune(
                policy,
                env,
                num_iterations=rl_iterations,
                episodes_per_iter=rl_episodes_per_iter,
                log_interval=rl_iterations,
            )

            eval_result = evaluate_policy(policy, env)
            score = eval_result["success_rate"] * 1000 + eval_result["mean_reward"]

            candidates.append(
                {
                    "policy": policy,
                    "code": code,
                    "response": response,
                    "eval": eval_result,
                    "score": score,
                    "bc_stats": bc_stats,
                    "rl_stats": rl_stats,
                }
            )
            print(
                f"    reward={eval_result['mean_reward']:.2f}, "
                f"success={eval_result['success_rate']:.2f}"
            )

        candidates.sort(key=lambda c: c["score"], reverse=True)

        print(f"\n  Generation {gen + 1} ranking:")
        for rank, c in enumerate(candidates):
            tag = " (elite)" if rank < elite_num else ""
            print(
                f"    [{rank + 1}] reward={c['eval']['mean_reward']:.2f}, "
                f"success={c['eval']['success_rate']:.2f}{tag}"
            )

        if candidates[0]["score"] > best_overall["score"]:
            best_overall = {
                "policy": copy.deepcopy(candidates[0]["policy"]),
                "code": candidates[0]["code"],
                "score": candidates[0]["score"],
                "stats": candidates[0]["eval"],
                "response": candidates[0]["response"],
            }

        best_c = candidates[0]
        elite_response = best_c["response"]
        elite_feedback = code_feedback_tmpl.format(
            train_steps=bc_steps + rl_iterations * rl_episodes_per_iter * 150,
            win_rate=best_c["eval"]["success_rate"],
            mean_reward=best_c["eval"]["mean_reward"],
            current_output=str(best_c["eval"]),
        )

        gen_results = {
            "generation": gen,
            "candidates": [
                {"code": c["code"], "eval": c["eval"], "score": c["score"]}
                for c in candidates
            ],
        }
        with open(os.path.join(gen_dir, "results.pkl"), "wb") as f:
            pickle.dump(gen_results, f)

    return best_overall


# ===========================================================================
#  Full Pipeline Orchestrator
# ===========================================================================


class SymbolicPolicyPipeline:
    """Complete four-stage training pipeline for symbolic policies.

    Usage::

        pipeline = SymbolicPolicyPipeline('window-close-v2')

        # Stage 1
        demo_buffer, stats = pipeline.stage1_generate_dataset(env)

        # Stages 2 + 3 on a single hand-crafted policy
        result = pipeline.run_single_policy(my_policy, env)

        # Full Stage 4 with LLM
        best = pipeline.stage4_llm_evolution(client, env, args)
    """

    def __init__(self, env_name, obs_dim=39, action_dim=4):
        self.env_name = env_name
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.demo_buffer = None
        self.best_policy = None
        self.best_code = None

    def stage1_generate_dataset(self, env, num_episodes=100, **kwargs):
        """Stage 1: collect expert demonstrations."""
        print("\n" + "=" * 60)
        print("STAGE 1: Dataset Generation")
        print("=" * 60)
        self.demo_buffer, stats = generate_dataset(
            env, self.env_name, num_episodes, **kwargs
        )
        return self.demo_buffer, stats

    def stage2_behavioral_cloning(self, policy, demo_buffer=None, **kwargs):
        """Stage 2: supervised imitation of expert actions."""
        print("\n" + "=" * 60)
        print("STAGE 2: Behavioral Cloning")
        print("=" * 60)
        buf = demo_buffer if demo_buffer is not None else self.demo_buffer
        if buf is None:
            raise ValueError(
                "No demo buffer available. Run stage1 first or pass demo_buffer."
            )
        return behavioral_cloning(policy, buf, **kwargs)

    def stage3_rl_finetune(self, policy, env, **kwargs):
        """Stage 3: GRPO-style RL fine-tuning."""
        print("\n" + "=" * 60)
        print("STAGE 3: RL Fine-tuning (GRPO)")
        print("=" * 60)
        return rl_finetune(policy, env, **kwargs)

    def stage4_llm_evolution(self, client, env, args, **kwargs):
        """Stage 4: LLM-based structure evolution with BC+RL inner loop."""
        print("\n" + "=" * 60)
        print("STAGE 4: LLM Evolution")
        print("=" * 60)
        if self.demo_buffer is None:
            raise ValueError("No demo buffer. Run stage1 first.")
        result = llm_evolution(
            client=client,
            env=env,
            env_name=self.env_name,
            demo_buffer=self.demo_buffer,
            args=args,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            **kwargs,
        )
        self.best_policy = result["policy"]
        self.best_code = result["code"]
        return result

    def run_single_policy(
        self,
        policy,
        env,
        demo_buffer=None,
        bc_steps=5000,
        rl_iterations=50,
        rl_episodes=20,
    ):
        """Run Stages 2-3 on a single policy (no LLM needed)."""
        bc_stats = self.stage2_behavioral_cloning(
            policy, demo_buffer, num_steps=bc_steps
        )
        rl_stats = self.stage3_rl_finetune(
            policy, env, num_iterations=rl_iterations, episodes_per_iter=rl_episodes
        )
        eval_result = evaluate_policy(policy, env)
        return {"bc_stats": bc_stats, "rl_stats": rl_stats, "eval": eval_result}


# ===========================================================================
#  CLI entry point
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Symbolic Policy Training Pipeline")
    parser.add_argument(
        "--env-name",
        type=str,
        default="window-close-v2",
        help="MetaWorld task name",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-episodes", type=int, default=100)
    parser.add_argument("--bc-steps", type=int, default=5000)
    parser.add_argument("--rl-iterations", type=int, default=50)
    parser.add_argument("--rl-episodes", type=int, default=20)
    parser.add_argument("--stage", type=str, default="all", help="1|2|3|4|all")
    cli_args = parser.parse_args()

    np.random.seed(cli_args.seed)
    torch.manual_seed(cli_args.seed)

    print(f"Environment: {cli_args.env_name}")
    print(f"Seed: {cli_args.seed}")

    import utils

    env = utils.make_metaworld_env(cli_args, cli_args.seed)

    pipeline = SymbolicPolicyPipeline(cli_args.env_name)

    if cli_args.stage in ("1", "all"):
        demo_buffer, stats = pipeline.stage1_generate_dataset(
            env, num_episodes=cli_args.dataset_episodes
        )
        demo_buffer.save(f"./logs/demo_{cli_args.env_name}.pkl")
        print(f"Dataset saved: {stats}")
