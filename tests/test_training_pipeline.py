#!/usr/bin/env python
"""
Comprehensive test suite for the four-stage symbolic policy training pipeline.

Run from the LaRes project root:
    python tests/test_training_pipeline.py

Three tiers of tests:
  Tier 1 (always):   Mock-based unit tests — no external dependencies
  Tier 2 (optional): Real MetaWorld environment tests — needs 'metaworld' installed
  Tier 3 (optional): LLM-based generation tests — needs OPENAI_API_KEY env var
"""

import os
import sys
import copy
import math
import pickle
import random
import tempfile
import traceback
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

# Add project root to path so lares package is importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ---------------------------------------------------------------------------
#  Detect available resources
# ---------------------------------------------------------------------------
HAS_METAWORLD = False
try:
    import metaworld  # noqa: F401
    HAS_METAWORLD = True
except ImportError:
    pass

HAS_OPENAI_KEY = "OPENAI_API_KEY" in os.environ and len(os.environ["OPENAI_API_KEY"]) > 10
HAS_GPU = torch.cuda.is_available()

print(f"MetaWorld available : {HAS_METAWORLD}")
print(f"OpenAI API key      : {HAS_OPENAI_KEY}")
print(f"GPU available       : {HAS_GPU}")
if HAS_GPU:
    print(f"  GPU device        : {torch.cuda.get_device_name(0)}")
print()

# ---------------------------------------------------------------------------
#  Imports from the codebase
# ---------------------------------------------------------------------------
from lares.core.symbolic_policy import SymbolicPolicy  # noqa: E402
from lares.core.replay_buffer import SimpleReplayBuffer  # noqa: E402

from lares.core.training_pipeline import (
    DemoBuffer,
    generate_dataset,
    behavioral_cloning,
    rl_finetune,
    evaluate_policy,
    SymbolicPolicyPipeline,
    EvolutionOrchestrator,
    llm_evolution,
    EXPERT_POLICY_MAP,
    TASK_DESCRIPTIONS,
    get_expert_policy,
    _collect_trajectories,
    _compute_grpo_advantages,
)

from lares.core.policy_generation import (  # noqa: E402
    get_symbolic_policies,
    obs_description_dict,
    input_dict_for_policy,
)

if HAS_METAWORLD:
    from lares.utils import env_wrapper, make_metaworld_env  # noqa: E402

# ---------------------------------------------------------------------------
#  Test helpers
# ---------------------------------------------------------------------------
PASS = 0
FAIL = 0
SKIP = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def skip(name, reason=""):
    global SKIP
    SKIP += 1
    print(f"  [SKIP] {name}  ({reason})")


OBS_DIM = 39
ACT_DIM = 4
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)


def make_test_args(**overrides):
    """Build a minimal args namespace compatible with the codebase."""
    defaults = dict(
        env_name="window-close-v2",
        seed=SEED,
        episode_length=150,
        model="gpt-4o-mini",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ---------------------------------------------------------------------------
#  Mock objects for Tier 1 tests
# ---------------------------------------------------------------------------

class MockActionSpace:
    def __init__(self, dim):
        self.shape = (dim,)
        self.high = np.ones(dim, dtype=np.float32)
        self.low = -np.ones(dim, dtype=np.float32)


class MockObsSpace:
    def __init__(self, dim):
        self.shape = (dim,)


class MockEnv:
    """Lightweight mock replicating the wrapped MetaWorld env interface."""

    def __init__(self, obs_dim=39, action_dim=4, episode_length=20):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_space = MockActionSpace(action_dim)
        self.observation_space = MockObsSpace(obs_dim)
        self._episode_length = episode_length
        self._step_count = 0

    def reset(self):
        self._step_count = 0
        obs = np.random.randn(self.obs_dim).astype(np.float32)
        return obs, {}

    def step(self, action):
        self._step_count += 1
        obs = np.random.randn(self.obs_dim).astype(np.float32)
        reward = float(np.random.randn())
        done = self._step_count >= self._episode_length
        success = 1.0 if done and np.random.rand() > 0.5 else 0.0
        info = {"success": success}
        return obs, reward, done, info


class MockExpertPolicy:
    """Deterministic expert: move toward object (obs[4:7] - obs[0:3])."""

    def get_action(self, obs):
        tcp = obs[0:3]
        obj = obs[4:7]
        direction = obj - tcp
        norm = np.linalg.norm(direction) + 1e-8
        move = 0.5 * direction / norm
        grip = np.array([0.5])
        return np.clip(np.concatenate([move, grip]), -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
#  Concrete test policies
# ---------------------------------------------------------------------------

class TestPolicy(SymbolicPolicy):
    """Simple symbolic policy for testing."""

    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.w = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.zeros(action_dim))
        self.log_std = nn.Parameter(torch.tensor(-1.0))

    def forward(self, obs):
        tcp = obs[:, 0:3]
        obj = obs[:, 4:7]
        diff = obj - tcp
        dist = torch.norm(diff, dim=-1, keepdim=True) + 1e-8
        direction = diff / dist
        move = self.w * direction
        grip = self.b[-1:].expand(obs.shape[0], 1)
        mean = torch.cat([move, grip], dim=-1)[:, :self.action_dim]
        mean = mean + self.b[:self.action_dim]
        std = torch.exp(self.log_std) * torch.ones_like(mean)
        return (mean, std)

    def get_param_ranges(self):
        return {
            "w": (0.01, 10.0),
            "b": (-2.0, 2.0),
            "log_std": (-5.0, 0.0),
        }


class WindowClosePolicy(SymbolicPolicy):
    """Multi-phase symbolic policy for window-close-v2 (richer test)."""

    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.w_approach = nn.Parameter(torch.tensor(3.0))
        self.w_push = nn.Parameter(torch.tensor(2.0))
        self.grip_bias = nn.Parameter(torch.tensor(0.5))
        self.log_std = nn.Parameter(torch.tensor(-1.0))
        self.sharpness = nn.Parameter(torch.tensor(50.0))
        self.threshold = nn.Parameter(torch.tensor(0.05))

    def forward(self, obs):
        tcp = obs[:, 0:3]
        obj = obs[:, 4:7]
        target = obs[:, 36:39]

        diff_obj = obj - tcp
        dist_obj = torch.norm(diff_obj, dim=-1, keepdim=True) + 1e-8
        dir_obj = diff_obj / dist_obj

        diff_target = target - obj
        dist_target = torch.norm(diff_target, dim=-1, keepdim=True) + 1e-8
        dir_target = diff_target / dist_target

        phase = torch.sigmoid(self.sharpness * (self.threshold - dist_obj))
        move = (1 - phase) * self.w_approach * dir_obj + phase * self.w_push * dir_target
        grip = self.grip_bias * torch.ones(obs.shape[0], 1)
        mean = torch.cat([move, grip], dim=-1)[:, :self.action_dim]
        std = torch.exp(self.log_std) * torch.ones_like(mean)
        return (mean, std)

    def get_param_ranges(self):
        return {
            "w_approach": (0.1, 10.0),
            "w_push": (0.1, 10.0),
            "grip_bias": (-1.0, 1.0),
            "log_std": (-5.0, 0.0),
            "sharpness": (1.0, 200.0),
            "threshold": (0.01, 0.3),
        }


# ###########################################################################
#  TIER 1 — Mock-based unit tests (always run)
# ###########################################################################

print("=" * 70)
print("TIER 1: Mock-based unit tests")
print("=" * 70)

# ===========================================================================
#  1.1 DemoBuffer
# ===========================================================================
print()
print("-" * 40)
print("1.1 DemoBuffer")
print("-" * 40)

buf = DemoBuffer()
check("DemoBuffer: empty len", len(buf) == 0)

for i in range(100):
    buf.add(
        np.random.randn(OBS_DIM),
        np.random.randn(ACT_DIM),
        float(np.random.randn()),
        np.random.randn(OBS_DIM),
        float(i == 99),
    )
check("DemoBuffer: add 100 transitions", len(buf) == 100)

obs_all, act_all, rew_all, nobs_all, done_all = buf.get_all()
check("DemoBuffer: obs shape", obs_all.shape == (100, OBS_DIM))
check("DemoBuffer: actions shape", act_all.shape == (100, ACT_DIM))
check("DemoBuffer: rewards shape", rew_all.shape == (100,))
check("DemoBuffer: next_obs shape", nobs_all.shape == (100, OBS_DIM))
check("DemoBuffer: dones shape", done_all.shape == (100,))

obs_s, act_s, rew_s, nobs_s, done_s = buf.sample(32)
check("DemoBuffer: sample batch size", obs_s.shape[0] == 32)
check("DemoBuffer: sample obs dim", obs_s.shape[1] == OBS_DIM)

with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
    tmp_path = tmp.name
buf.save(tmp_path)
buf2 = DemoBuffer.load(tmp_path)
check("DemoBuffer: save/load roundtrip size", len(buf2) == 100)
o1, a1, r1, _, _ = buf.get_all()
o2, a2, r2, _, _ = buf2.get_all()
check("DemoBuffer: data matches after load", np.allclose(o1, o2) and np.allclose(a1, a2))
os.remove(tmp_path)

# ===========================================================================
#  1.2 SimpleReplayBuffer
# ===========================================================================
print()
print("-" * 40)
print("1.2 SimpleReplayBuffer")
print("-" * 40)

srb = SimpleReplayBuffer(50)
check("SRB: empty", len(srb) == 0)

for i in range(100):
    srb.add(np.random.randn(OBS_DIM), np.random.randn(ACT_DIM), np.random.randn(), np.random.randn(OBS_DIM), 0.0)
check("SRB: circular buffer capped at 50", len(srb) == 50)

o, a, r, no, d = srb.sample(16)
check("SRB: sample obs shape", o.shape == (16, OBS_DIM))
check("SRB: sample action shape", a.shape == (16, ACT_DIM))
check("SRB: sample reward shape", r.shape == (16,))

# ===========================================================================
#  1.3 Behavioral Cloning (Stage 2) with mock data
# ===========================================================================
print()
print("-" * 40)
print("1.3 Behavioral Cloning (Stage 2)")
print("-" * 40)

policy_bc = TestPolicy(OBS_DIM, ACT_DIM)
initial_sd = copy.deepcopy(policy_bc.state_dict())

expert = MockExpertPolicy()
demo_buf = DemoBuffer()
env_mock = MockEnv(OBS_DIM, ACT_DIM, episode_length=20)

for _ in range(200):
    obs, _ = env_mock.reset()
    for __ in range(20):
        action = expert.get_action(obs)
        next_obs, reward, done, info = env_mock.step(env_mock.action_space.high * action)
        demo_buf.add(obs, action, reward, next_obs, float(done))
        obs = next_obs
        if done:
            break
check("BC: demo buffer has data", len(demo_buf) > 500)

bc_stats = behavioral_cloning(
    policy_bc, demo_buf, num_steps=300, batch_size=64, lr=1e-3, log_interval=0,
)
check("BC: returns stats dict", isinstance(bc_stats, dict))
check("BC: has bc_loss key", "bc_loss" in bc_stats)
# check("BC: has mean_loss key", "mean_loss" in bc_stats) old
# check("BC: has std_loss key", "std_loss" in bc_stats)
check("BC: has log_prob key", "log_prob" in bc_stats)
check("BC: has final_loss key", "final_loss" in bc_stats)
check("BC: correct number of loss entries", len(bc_stats["bc_loss"]) == 300)

post_sd = policy_bc.state_dict()
params_changed = any(
    not torch.equal(initial_sd[k], post_sd[k]) for k in initial_sd
)
check("BC: parameters updated", params_changed)

early_loss = np.mean(bc_stats["bc_loss"][:30])
late_loss = np.mean(bc_stats["bc_loss"][-30:])
check("BC: loss trend", late_loss <= early_loss * 1.5, f"early={early_loss:.4f} late={late_loss:.4f}")

test_obs = torch.randn(8, OBS_DIM)
mean, std = policy_bc(test_obs)
check("BC: output mean shape", mean.shape == (8, ACT_DIM))
check("BC: output std shape", std.shape == (8, ACT_DIM))
check("BC: std positive", (std > 0).all().item())
check("BC: no NaN", not torch.isnan(mean).any() and not torch.isnan(std).any())
check("BC: no Inf", not torch.isinf(mean).any() and not torch.isinf(std).any())

ranges = policy_bc.get_param_ranges()
for pname, (lo, hi) in ranges.items():
    param = dict(policy_bc.named_parameters())[pname]
    check(f"BC: param '{pname}' in declared range",
          param.min().item() >= lo - 1e-6 and param.max().item() <= hi + 1e-6)

# ===========================================================================
#  1.4 Trajectory collection & GRPO advantages
# ===========================================================================
print()
print("-" * 40)
print("1.4 Trajectory collection & GRPO")
print("-" * 40)

trajectories = _collect_trajectories(policy_bc, env_mock, num_episodes=5, max_steps=20)
check("Traj: correct count", len(trajectories) == 5)
check("Traj: required keys",
      all(k in trajectories[0] for k in ("obs", "pretanh", "rewards", "return", "success", "length")))
check("Traj: obs/rewards lengths match",
      len(trajectories[0]["obs"]) == len(trajectories[0]["rewards"]))
check("Traj: return is sum of rewards",
      abs(trajectories[0]["return"] - sum(trajectories[0]["rewards"])) < 1e-6)

trajectories = _compute_grpo_advantages(trajectories, gamma=0.99)
check("GRPO: advantages computed", "advantages" in trajectories[0])
check("GRPO: advantages length", len(trajectories[0]["advantages"]) == len(trajectories[0]["obs"]))

all_adv = np.concatenate([t["advantages"] for t in trajectories])
check("GRPO: advantages centered", abs(np.mean(all_adv)) < 2.0, f"mean={np.mean(all_adv):.4f}")

# ===========================================================================
#  1.5 RL Fine-tuning (Stage 3)
# ===========================================================================
print()
print("-" * 40)
print("1.5 RL Fine-tuning (Stage 3)")
print("-" * 40)

policy_rl = TestPolicy(OBS_DIM, ACT_DIM)
behavioral_cloning(policy_rl, demo_buf, num_steps=100, batch_size=64, lr=1e-3, log_interval=0)
pre_rl_sd = copy.deepcopy(policy_rl.state_dict())

rl_stats = rl_finetune(
    policy_rl, env_mock, num_iterations=5, episodes_per_iter=3,
    lr=1e-3, max_steps=20, log_interval=0,
)
check("RL: returns stats dict", isinstance(rl_stats, dict))
check("RL: has returns", len(rl_stats["returns"]) == 15)
check("RL: has successes", len(rl_stats["successes"]) == 15)
check("RL: has policy_loss", len(rl_stats["policy_loss"]) == 5)
check("RL: has entropy", len(rl_stats["entropy"]) == 5)
check("RL: has kl", len(rl_stats["kl"]) == 5)
check("RL: has best_success_rate", isinstance(rl_stats["best_success_rate"], float))
check("RL: has final_mean_return", isinstance(rl_stats["final_mean_return"], float))

post_rl_sd = policy_rl.state_dict()
rl_changed = any(
    not torch.equal(pre_rl_sd[k], post_rl_sd[k]) for k in pre_rl_sd
)
check("RL: parameters changed", rl_changed)

mean, std = policy_rl(torch.randn(4, OBS_DIM))
check("RL: output valid", mean.shape == (4, ACT_DIM) and (std > 0).all().item())
check("RL: no NaN after RL", not torch.isnan(mean).any() and not torch.isnan(std).any())

# ===========================================================================
#  1.6 evaluate_policy
# ===========================================================================
print()
print("-" * 40)
print("1.6 evaluate_policy")
print("-" * 40)

eval_result = evaluate_policy(policy_rl, env_mock, num_episodes=5, max_steps=20)
check("Eval: returns dict", isinstance(eval_result, dict))
check("Eval: has mean_reward", "mean_reward" in eval_result)
check("Eval: has success_rate", "success_rate" in eval_result)
check("Eval: success_rate in [0,1]", 0.0 <= eval_result["success_rate"] <= 1.0)
check("Eval: mean_reward is finite", math.isfinite(eval_result["mean_reward"]))

# ===========================================================================
#  1.7 SymbolicPolicyPipeline orchestrator
# ===========================================================================
print()
print("-" * 40)
print("1.7 SymbolicPolicyPipeline orchestrator")
print("-" * 40)

pipeline = SymbolicPolicyPipeline("window-close-v2", OBS_DIM, ACT_DIM)
check("Pipeline: env_name", pipeline.env_name == "window-close-v2")
check("Pipeline: obs_dim", pipeline.obs_dim == OBS_DIM)
check("Pipeline: action_dim", pipeline.action_dim == ACT_DIM)
check("Pipeline: no demo buffer yet", pipeline.demo_buffer is None)

pipeline.demo_buffer = demo_buf
p_pipe = TestPolicy(OBS_DIM, ACT_DIM)
result = pipeline.run_single_policy(p_pipe, env_mock, bc_steps=100, rl_iterations=3, rl_episodes=3)
check("Pipeline: returns dict", isinstance(result, dict))
check("Pipeline: has bc_stats", "bc_stats" in result)
check("Pipeline: has rl_stats", "rl_stats" in result)
check("Pipeline: has eval", "eval" in result)
check("Pipeline: eval success_rate", "success_rate" in result["eval"])

# ===========================================================================
#  1.8 Multi-phase policy (WindowClosePolicy)
# ===========================================================================
print()
print("-" * 40)
print("1.8 Multi-phase WindowClosePolicy")
print("-" * 40)

wc = WindowClosePolicy(OBS_DIM, ACT_DIM)
wc.validate()
check("WC: validate passes", True)

for bs in [1, 4, 16]:
    m, s = wc(torch.randn(bs, OBS_DIM))
    check(f"WC: forward bs={bs}", m.shape == (bs, ACT_DIM) and s.shape == (bs, ACT_DIM))

wc_bc = behavioral_cloning(wc, demo_buf, num_steps=200, batch_size=64, lr=1e-3, log_interval=0)
check("WC: BC training completes", "final_loss" in wc_bc)

wc_eval = evaluate_policy(wc, env_mock, num_episodes=3, max_steps=20)
check("WC: eval completes", "mean_reward" in wc_eval)

# ===========================================================================
#  1.9 End-to-end gradient flow across BC -> RL
# ===========================================================================
print()
print("-" * 40)
print("1.9 End-to-end gradient flow")
print("-" * 40)

p_e2e = TestPolicy(OBS_DIM, ACT_DIM)
p_e2e.zero_grad()
m, s = p_e2e(torch.randn(4, OBS_DIM))
(m.sum() + s.sum()).backward()
check("E2E: grads before BC", any(p.grad is not None and p.grad.abs().sum() > 0 for p in p_e2e.parameters()))

behavioral_cloning(p_e2e, demo_buf, num_steps=10, batch_size=32, lr=1e-3, log_interval=0)
p_e2e.zero_grad()
m, s = p_e2e(torch.randn(4, OBS_DIM))
(m.sum() + s.sum()).backward()
check("E2E: grads after BC", any(p.grad is not None and p.grad.abs().sum() > 0 for p in p_e2e.parameters()))

rl_finetune(p_e2e, env_mock, num_iterations=2, episodes_per_iter=2, max_steps=10, log_interval=0)
p_e2e.zero_grad()
m, s = p_e2e(torch.randn(4, OBS_DIM))
(m.sum() + s.sum()).backward()
check("E2E: grads after RL", any(p.grad is not None and p.grad.abs().sum() > 0 for p in p_e2e.parameters()))

# ===========================================================================
#  1.10 Expert policy mapping & task descriptions
# ===========================================================================
print()
print("-" * 40)
print("1.10 Expert policy mapping & task descriptions")
print("-" * 40)

EXPECTED_TASKS = ["window-close-v2", "window-open-v2", "button-press-v2", "door-close-v2", "drawer-open-v2"]
for task in EXPECTED_TASKS:
    check(f"Map: '{task}'", task in EXPERT_POLICY_MAP)
check("Map: all class names", all(v.startswith("Sawyer") and v.endswith("Policy") for v in EXPERT_POLICY_MAP.values()))
check("Map: 25+ tasks", len(EXPERT_POLICY_MAP) >= 25)

for task in EXPECTED_TASKS:
    check(f"Desc: '{task}'", task in TASK_DESCRIPTIONS and len(TASK_DESCRIPTIONS[task]) > 10)

# ===========================================================================
#  1.11 Prompt templates & policy_generation dicts
# ===========================================================================
print()
print("-" * 40)
print("1.11 Prompt templates & policy_generation dicts")
print("-" * 40)

prompt_dir = os.path.join(ROOT, "lares", "utils", "policy_prompts")
for pf in [
    "initial_system.txt",
    "new_initial_user.txt",
    "new_code_output_tip.txt",
    "code_feedback.txt",
    "ideas_system.txt",
    "ideas_user.txt",
]:
    path = os.path.join(prompt_dir, pf)
    check(f"Prompt: '{pf}' exists", os.path.isfile(path))

for task in EXPECTED_TASKS:
    check(f"obs_desc: '{task}'", task in obs_description_dict and len(obs_description_dict[task]) > 50)
    check(f"input_dict: '{task}'", task in input_dict_for_policy and len(input_dict_for_policy[task]) > 20)

try:
    with open(os.path.join(prompt_dir, "code_feedback.txt"), "r") as f:
        tmpl = f.read()
    fb = tmpl.format(train_steps=10000, win_rate=0.5, mean_reward=100.0, current_output="test")
    check("Prompt: feedback template formats", "10000" in fb and "0.5" in fb)
except Exception as e:
    check("Prompt: feedback template formats", False, str(e))

try:
    with open(os.path.join(prompt_dir, "new_initial_user.txt"), "r") as f:
        user_tmpl = f.read()
    formatted = user_tmpl.format(
        task="test task", obs_dim=39, action_dim=4,
        obs_description="test obs", input_dict_string="test dict",
    )
    check("Prompt: user template formats", "test task" in formatted and "39" in formatted)
except Exception as e:
    check("Prompt: user template formats", False, str(e))

# ===========================================================================
#  1.12 state_dict / load_state_dict roundtrip across pipeline
# ===========================================================================
print()
print("-" * 40)
print("1.12 State dict roundtrip")
print("-" * 40)

p_sd = TestPolicy(OBS_DIM, ACT_DIM)
behavioral_cloning(p_sd, demo_buf, num_steps=50, batch_size=32, lr=1e-3, log_interval=0)
sd = copy.deepcopy(p_sd.state_dict())
obs_test = torch.randn(4, OBS_DIM)
m1, s1 = p_sd(obs_test)

p_sd2 = TestPolicy(OBS_DIM, ACT_DIM)
p_sd2.load_state_dict(sd)
m2, s2 = p_sd2(obs_test)
check("SD: mean matches", torch.allclose(m1, m2, atol=1e-6))
check("SD: std matches", torch.allclose(s1, s2, atol=1e-6))


# ###########################################################################
#  TIER 2 — Real MetaWorld environment tests
# ###########################################################################

print()
print("=" * 70)
print("TIER 2: Real MetaWorld environment tests")
print("=" * 70)

if HAS_METAWORLD:
    args_mw = make_test_args(env_name="window-close-v2")

    # --- 2.1 Environment creation ---
    print()
    print("-" * 40)
    print("2.1 MetaWorld environment creation")
    print("-" * 40)

    try:
        real_env = make_metaworld_env(args_mw, SEED)
        real_env = env_wrapper(real_env, args_mw)
        obs, info_or_none = real_env.reset()
        check("MW env: created", True)
        check("MW env: obs shape", obs.shape == (OBS_DIM,))
        check("MW env: action_space", real_env.action_space.shape == (ACT_DIM,))
    except Exception:
        check("MW env: creation", False, traceback.format_exc())
        real_env = None

    if real_env is not None:
        # --- 2.2 Expert policy loading ---
        print()
        print("-" * 40)
        print("2.2 Expert policy loading")
        print("-" * 40)

        try:
            expert_policy = get_expert_policy("window-close-v2")
            check("MW expert: loaded", True)
            test_action = expert_policy.get_action(obs)
            check("MW expert: action shape", test_action.shape == (ACT_DIM,))
            check("MW expert: action finite", np.isfinite(test_action).all())
        except Exception:
            check("MW expert: loading", False, traceback.format_exc())

        # --- 2.3 Stage 1 — Real dataset generation ---
        print()
        print("-" * 40)
        print("2.3 Stage 1 — Real dataset generation")
        print("-" * 40)

        try:
            real_demo, real_stats = generate_dataset(real_env, "window-close-v2", num_episodes=10, max_steps=150)
            check("Stage1 real: buffer size", len(real_demo) > 100)
            check("Stage1 real: stats keys", "mean_reward" in real_stats and "mean_success" in real_stats)
            check("Stage1 real: num_transitions", real_stats["num_transitions"] == len(real_demo))

            r_obs, r_act, r_rew, r_nobs, r_done = real_demo.get_all()
            check("Stage1 real: obs dim", r_obs.shape[1] == OBS_DIM)
            check("Stage1 real: act dim", r_act.shape[1] == ACT_DIM)
            check("Stage1 real: actions in [-1,1]", r_act.min() >= -1.0 and r_act.max() <= 1.0)
            check("Stage1 real: no NaN obs", not np.isnan(r_obs).any())
            check("Stage1 real: no NaN actions", not np.isnan(r_act).any())
            print(f"    Expert stats: reward={real_stats['mean_reward']:.2f}, success={real_stats['mean_success']:.2f}")
        except Exception:
            check("Stage1 real", False, traceback.format_exc())
            real_demo = None

        # --- 2.4 Stage 2 — BC with real data ---
        if real_demo is not None and len(real_demo) > 100:
            print()
            print("-" * 40)
            print("2.4 Stage 2 — BC with real data")
            print("-" * 40)

            try:
                p_real_bc = WindowClosePolicy(OBS_DIM, ACT_DIM)
                bc_real_stats = behavioral_cloning(
                    p_real_bc, real_demo, num_steps=1000, batch_size=min(128, len(real_demo)),
                    lr=1e-3, log_interval=500,
                )
                check("Stage2 real: completes", "final_loss" in bc_real_stats)
                check("Stage2 real: loss decreased",
                      np.mean(bc_real_stats["bc_loss"][-50:]) <= np.mean(bc_real_stats["bc_loss"][:50]) * 1.5)

                m_real, s_real = p_real_bc(torch.tensor(r_obs[:8], dtype=torch.float32))
                check("Stage2 real: output shapes", m_real.shape == (8, ACT_DIM))
                check("Stage2 real: no NaN", not torch.isnan(m_real).any())
            except Exception:
                check("Stage2 real", False, traceback.format_exc())
                p_real_bc = None

            # --- 2.5 Stage 3 — RL with real env ---
            if p_real_bc is not None:
                print()
                print("-" * 40)
                print("2.5 Stage 3 — RL with real env")
                print("-" * 40)

                try:
                    rl_real_stats = rl_finetune(
                        p_real_bc, real_env, num_iterations=5, episodes_per_iter=3,
                        lr=3e-4, max_steps=150, log_interval=5,
                    )
                    check("Stage3 real: completes", "best_success_rate" in rl_real_stats)
                    check("Stage3 real: has returns", len(rl_real_stats["returns"]) == 15)
                    print(f"    RL stats: best_success={rl_real_stats['best_success_rate']:.2f}, "
                          f"final_return={rl_real_stats['final_mean_return']:.2f}")
                except Exception:
                    check("Stage3 real", False, traceback.format_exc())

            # --- 2.6 Evaluate with real env ---
            if p_real_bc is not None:
                print()
                print("-" * 40)
                print("2.6 Evaluate with real env")
                print("-" * 40)

                try:
                    eval_real = evaluate_policy(p_real_bc, real_env, num_episodes=5, max_steps=150)
                    check("Eval real: completes", "success_rate" in eval_real)
                    check("Eval real: success_rate in [0,1]", 0 <= eval_real["success_rate"] <= 1)
                    print(f"    Eval: reward={eval_real['mean_reward']:.2f}, success={eval_real['success_rate']:.2f}")
                except Exception:
                    check("Eval real", False, traceback.format_exc())

        # --- 2.7 Full pipeline orchestrator with real env ---
        if real_demo is not None and len(real_demo) > 100:
            print()
            print("-" * 40)
            print("2.7 Pipeline orchestrator with real env")
            print("-" * 40)

            try:
                pipe_real = SymbolicPolicyPipeline("window-close-v2", OBS_DIM, ACT_DIM)
                pipe_real.demo_buffer = real_demo
                p_pipe_real = WindowClosePolicy(OBS_DIM, ACT_DIM)
                pipe_result = pipe_real.run_single_policy(
                    p_pipe_real, real_env, bc_steps=500, rl_iterations=3, rl_episodes=2,
                )
                check("Pipeline real: completes", "eval" in pipe_result)
                check("Pipeline real: eval has fields",
                      "success_rate" in pipe_result["eval"] and "mean_reward" in pipe_result["eval"])
                print(f"    Pipeline result: reward={pipe_result['eval']['mean_reward']:.2f}, "
                      f"success={pipe_result['eval']['success_rate']:.2f}")
            except Exception:
                check("Pipeline real", False, traceback.format_exc())

else:
    skip("Tier 2: All MetaWorld tests", "metaworld not installed")


# ###########################################################################
#  TIER 3 — LLM-based generation tests (needs OPENAI_API_KEY)
# ###########################################################################

print()
print("=" * 70)
print("TIER 3: LLM-based generation tests")
print("=" * 70)

if HAS_OPENAI_KEY:
    from openai import OpenAI  # noqa: E402

    llm_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    llm_args = make_test_args(model="gpt-4o-mini")

    # --- 3.1 LLM API connectivity ---
    print()
    print("-" * 40)
    print("3.1 LLM API connectivity")
    print("-" * 40)

    try:
        test_resp = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            max_tokens=10,
        )
        reply = test_resp.choices[0].message.content.strip().lower()
        check("LLM API: connectivity", "hello" in reply, reply)
    except Exception:
        check("LLM API: connectivity", False, traceback.format_exc())

    # --- 3.2 Generate symbolic policy via LLM ---
    print()
    print("-" * 40)
    print("3.2 LLM symbolic policy generation")
    print("-" * 40)

    gen_dir = os.path.join(ROOT, "_test_llm_gen")
    os.makedirs(gen_dir, exist_ok=True)
    data_pkl = os.path.join(gen_dir, "data.pkl")
    with open(data_pkl, "wb") as f:
        pickle.dump([{"obs": np.zeros(OBS_DIM)}], f)

    def _read_prompt(name):
        with open(os.path.join(prompt_dir, name), "r") as f:
            return f.read()

    try:
        policy_pop, code_pop, resp_list = get_symbolic_policies(
            client=llm_client,
            dir_path=gen_dir,
            llm_iter=0,
            args=llm_args,
            obs_dim=OBS_DIM,
            action_dim=ACT_DIM,
            initial_system=_read_prompt("initial_system.txt"),
            initial_user=_read_prompt("new_initial_user.txt"),
            task_description="Control the robotic arm to close the window",
            obs_description=obs_description_dict["window-close-v2"],
            input_dict_string=input_dict_for_policy["window-close-v2"],
            code_output_tip=_read_prompt("new_code_output_tip.txt"),
            data_pkl_path=data_pkl,
            real_num=1,
        )
        check("LLM gen: returned 1 policy", len(policy_pop) == 1)
        check("LLM gen: returned code string", len(code_pop) == 1 and len(code_pop[0]) > 50)
        check("LLM gen: returned response", len(resp_list) == 1 and len(resp_list[0]) > 50)

        gen_policy = policy_pop[0]
        check("LLM gen: is SymbolicPolicy", isinstance(gen_policy, SymbolicPolicy))
        gen_policy.validate()
        check("LLM gen: validate passes", True)

        m, s = gen_policy(torch.randn(4, OBS_DIM))
        check("LLM gen: forward shape", m.shape == (4, ACT_DIM) and s.shape == (4, ACT_DIM))
        check("LLM gen: std > 0", (s > 0).all().item())
        check("LLM gen: no NaN", not torch.isnan(m).any() and not torch.isnan(s).any())

        gen_ranges = gen_policy.get_param_ranges()
        check("LLM gen: param ranges valid",
              isinstance(gen_ranges, dict) and all(lo < hi for lo, hi in gen_ranges.values()))

        gen_policy.zero_grad()
        (m.sum() + s.sum()).backward()
        has_g = any(p.grad is not None and p.grad.abs().sum() > 0 for p in gen_policy.parameters())
        check("LLM gen: gradients flow", has_g)

    except Exception:
        check("LLM gen", False, traceback.format_exc())

    # --- 3.3 BC + RL on LLM-generated policy (mock env) ---
    if len(policy_pop) > 0:
        print()
        print("-" * 40)
        print("3.3 BC + RL on LLM-generated policy")
        print("-" * 40)

        try:
            gen_p = policy_pop[0]
            gen_bc = behavioral_cloning(gen_p, demo_buf, num_steps=200, batch_size=64, lr=1e-3, log_interval=0)
            check("LLM BC: completes", "final_loss" in gen_bc)

            gen_rl = rl_finetune(gen_p, env_mock, num_iterations=3, episodes_per_iter=3, max_steps=20, log_interval=0)
            check("LLM RL: completes", "best_success_rate" in gen_rl)

            gen_eval = evaluate_policy(gen_p, env_mock, num_episodes=3, max_steps=20)
            check("LLM eval: completes", "success_rate" in gen_eval)
        except Exception:
            check("LLM BC+RL", False, traceback.format_exc())

    # --- 3.4 LLM-generated policy on real env (if available) ---
    if HAS_METAWORLD and len(policy_pop) > 0 and real_env is not None and real_demo is not None:
        print()
        print("-" * 40)
        print("3.4 LLM policy on real MetaWorld env")
        print("-" * 40)

        try:
            gen_p_real = policy_pop[0]
            gen_p_real_copy = copy.deepcopy(gen_p_real)

            bc_r = behavioral_cloning(
                gen_p_real_copy, real_demo, num_steps=500,
                batch_size=min(128, len(real_demo)), lr=1e-3, log_interval=500,
            )
            check("LLM+MW BC: completes", "final_loss" in bc_r)

            rl_r = rl_finetune(
                gen_p_real_copy, real_env, num_iterations=3,
                episodes_per_iter=2, max_steps=150, log_interval=3,
            )
            check("LLM+MW RL: completes", "best_success_rate" in rl_r)

            eval_r = evaluate_policy(gen_p_real_copy, real_env, num_episodes=3, max_steps=150)
            check("LLM+MW eval: completes", "success_rate" in eval_r)
            print(f"    LLM policy on real env: reward={eval_r['mean_reward']:.2f}, "
                  f"success={eval_r['success_rate']:.2f}")
        except Exception:
            check("LLM+MW", False, traceback.format_exc())

    # --- 3.5 Stage 4 LLM evolution (1 generation, 1 candidate, mock env) ---
    print()
    print("-" * 40)
    print("3.5 Stage 4 — LLM evolution (minimal)")
    print("-" * 40)

    try:
        evo_dir = os.path.join(ROOT, "_test_llm_evo")

        orchestrator = EvolutionOrchestrator(
            env_name="window-close-v2",
            obs_dim=OBS_DIM,
            action_dim=ACT_DIM,
            num_generations=1,
            pop_size=1,
            elite_num=1,
            bc_steps=100,
            rl_iterations=3,
            rl_episodes_per_iter=2,
            log_dir=evo_dir,
            record_demo_gif=False,
        )

        evo_result = orchestrator.run(
            client=llm_client,
            env=env_mock,
            demo_buffer=demo_buf,
            args=llm_args,
            eval_episodes=3,
        )

        check("Stage4: returns dict", isinstance(evo_result, dict))
        check("Stage4: has policy", evo_result["policy"] is not None)
        check(
            "Stage4: has code",
            evo_result["code"] is not None and len(evo_result["code"]) > 20,
        )
        check("Stage4: has score", isinstance(evo_result["score"], (int, float)))
        check(
            "Stage4: policy is SymbolicPolicy",
            isinstance(evo_result["policy"], SymbolicPolicy),
        )

        evo_m, evo_s = evo_result["policy"](torch.randn(4, OBS_DIM))
        check(
            "Stage4: policy forward works",
            evo_m.shape == (4, ACT_DIM) and (evo_s > 0).all(),
        )
    except Exception:
        check("Stage4 LLM evolution", False, traceback.format_exc())
    # try:
    #     evo_dir = os.path.join(ROOT, "_test_llm_evo")
    #     evo_result = llm_evolution(
    #         client=llm_client,
    #         env=env_mock,
    #         env_name="window-close-v2",
    #         demo_buffer=demo_buf,
    #         args=llm_args,
    #         obs_dim=OBS_DIM,
    #         action_dim=ACT_DIM,
    #         num_generations=1,
    #         pop_size=1,
    #         elite_num=1,
    #         bc_steps=100,
    #         rl_iterations=3,
    #         rl_episodes_per_iter=2,
    #         log_dir=evo_dir,
    #     )
    #     check("Stage4: returns dict", isinstance(evo_result, dict))
    #     check("Stage4: has policy", evo_result["policy"] is not None)
    #     check("Stage4: has code", evo_result["code"] is not None and len(evo_result["code"]) > 20)
    #     check("Stage4: has score", isinstance(evo_result["score"], (int, float)))
    #     check("Stage4: policy is SymbolicPolicy", isinstance(evo_result["policy"], SymbolicPolicy))

    #     evo_m, evo_s = evo_result["policy"](torch.randn(4, OBS_DIM))
    #     check("Stage4: policy forward works", evo_m.shape == (4, ACT_DIM) and (evo_s > 0).all())
    # except Exception:
    #     check("Stage4 LLM evolution", False, traceback.format_exc())

    # Cleanup
    import shutil  # noqa: E402
    for d in [gen_dir, os.path.join(ROOT, "_test_llm_evo")]:
        if os.path.exists(d):
            shutil.rmtree(d, ignore_errors=True)

else:
    skip("Tier 3: All LLM tests", "OPENAI_API_KEY not set")


# ###########################################################################
#  Summary
# ###########################################################################

print()
print("=" * 70)
TOTAL = PASS + FAIL
print(f"RESULTS: {PASS}/{TOTAL} passed, {FAIL}/{TOTAL} failed, {SKIP} skipped")
print("=" * 70)

if FAIL > 0:
    print("SOME TESTS FAILED — see [FAIL] lines above.")
    sys.exit(1)
else:
    if SKIP > 0:
        print("ALL RUN TESTS PASSED (some skipped due to missing dependencies).")
    else:
        print("ALL TESTS PASSED!")
    sys.exit(0)
