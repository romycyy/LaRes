"""The R-040 PPO expert behind the evaluator's controller interface. Needs a launched Isaac app.

``make_agent_cfg`` is the vendored ``ShadowHandSpinPPORunnerCfg`` with ``clip_actions = 1.0``, so the
``last_action`` the policy observes in training matches the clamped action the evaluator applies.
D-036: ``logs/pen_spin/r040_ppo_expert/train/model_8239.pt`` (training seed 4020) qualifies on every stage.
"""

from __future__ import annotations


def make_agent_cfg(seed: int, max_iterations: int, save_interval: int, experiment_name: str):
    from importlib.metadata import version

    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    from lares.envs.isaac_pen_spin.rsl_rl_ppo_cfg import ShadowHandSpinPPORunnerCfg

    agent = ShadowHandSpinPPORunnerCfg()
    agent.seed = int(seed)
    agent.max_iterations = int(max_iterations)
    agent.save_interval = int(save_interval)
    agent.experiment_name = experiment_name
    agent.clip_actions = 1.0
    return handle_deprecated_rsl_rl_cfg(agent, version("rsl-rl-lib"))


def build_runner(env, agent_cfg, log_dir, device):
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    return OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=log_dir, device=device), wrapped


class PPOController:
    """A trained checkpoint behind the evaluator's controller interface."""

    name = "ppo_expert"

    def __init__(self, checkpoint: str, seed: int):
        self.checkpoint = checkpoint
        self.seed = seed
        self.params = {"checkpoint": checkpoint}

    def attach(self, env):
        agent = make_agent_cfg(self.seed, 1, 1, "eval")
        runner, _ = build_runner(env, agent, None, str(env.device))
        runner.load(self.checkpoint)
        self.policy = runner.get_inference_policy(device=str(env.device))

    def reset(self, ctx):
        pass

    def __call__(self, obs, t, ctx):
        import torch
        from tensordict import TensorDict

        with torch.inference_mode():
            return self.policy(TensorDict({"policy": obs}, batch_size=[obs.shape[0]])).clone()
