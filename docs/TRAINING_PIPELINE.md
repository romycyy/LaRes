# Symbolic Policy Training Pipeline

## Overview

A four-stage training pipeline that replaces the original LaRes reward-search loop with a **symbolic policy search** approach. Instead of the LLM generating reward functions for SAC to train neural-network actors, it generates *learnable symbolic policies* that are trained through behavioral cloning, GRPO RL fine-tuning, and evolutionary structure search.

```
Stage 1: Dataset Generation
    MetaWorld expert policies  -->  DemoBuffer(obs, action, reward, next_obs, done)

Stage 2: Behavioral Cloning
    DemoBuffer + SymbolicPolicy  -->  BC-initialised policy (supervised MSE)

Stage 3: RL Fine-tuning (GRPO)
    BC policy + environment  -->  RL-improved policy (policy gradient)

Stage 4: LLM Evolution
    LLM proposes structures  -->  [BC -> RL -> eval] per candidate  -->  best policy
```

---

## Architecture

```
NEW symbolic policy pipeline:
  Expert demos -> BC -> RL fine-tune -> evaluate
        ^                                    |
        |                                    v
  LLM -> GeneratedPolicy(SymbolicPolicy) -> keep best candidates
         ├── forward(obs) -> (mean, std)     Gaussian action distribution
         ├── get_param_ranges()              declares bounds on learnable params
         └── clip_params()                   enforces bounds after gradient steps
```

The symbolic policy outputs `(mean, std)` defining a Gaussian distribution over pre-tanh actions:
- `mean`: (batch, action_dim) — centre of the Gaussian
- `std`: (batch, action_dim) — scale (always positive)
- Actions are sampled, squashed through tanh, and scaled to the environment's action range
- Gradients flow through symbolic expressions to `nn.Parameter`s
- The population in Stage 4 consists of different symbolic policy *structures*

---

## Files

| File | Role | Status |
|------|------|--------|
| `lares/core/training_pipeline.py` | Main pipeline: all 4 stages + `SymbolicPolicyPipeline` orchestrator | New |
| `tests/test_training_pipeline.py` | 50+ checks across 3 tiers (mock / MetaWorld / LLM) | New |
| `lares/core/symbolic_policy.py` | Base class for symbolic policies | Existing |
| `lares/core/policy_generation.py` | LLM generation pipeline | Existing |
| `tests/test_generate_policy.py` | Subprocess validation harness | Existing |
| `lares/core/replay_buffer.py` | Added `SimpleReplayBuffer` for single-reward storage | Modified |
| `lares/utils/policy_prompts/` | Prompt templates for LLM policy generation | Existing |
| `scripts/run_demo.py` | End-to-end demo on reach-v2 | New |

**Run commands** (from project root): `python scripts/run_demo.py`, `python tests/test_training_pipeline.py`

---

## Stage 1 — Dataset Generation

**Purpose:** Collect expert demonstration trajectories from MetaWorld's built-in semi-optimal policies.

**Function:** `generate_dataset(env, env_name, num_episodes=100)`

**How it works:**
1. Loads the MetaWorld expert policy via `get_expert_policy(env_name)` (maps e.g. `window-close-v2` → `SawyerWindowCloseV3Policy`)
2. Runs the expert in the environment for `num_episodes` episodes
3. Clips expert actions to `[-1, 1]` for compatibility with the normalised action space
4. Stores `(obs, action, reward, next_obs, done)` tuples in a `DemoBuffer`

**Output:** `DemoBuffer` with thousands of transitions + statistics dict.

**Supported tasks:** All MetaWorld V2 tasks with policies in `metaworld.policies` — 25+ tasks mapped in `EXPERT_POLICY_MAP`.

```python
from lares.core.training_pipeline import generate_dataset
buffer, stats = generate_dataset(env, "window-close-v2", num_episodes=100)
buffer.save("./logs/demo_window-close-v2.pkl")
```

---

## Stage 2 — Behavioral Cloning

**Purpose:** Initialise the symbolic policy close to expert behaviour via supervised learning.

**Function:** `behavioral_cloning(policy, demo_buffer, num_steps=5000)`

**How it works:**
1. Samples mini-batches `(obs, expert_action)` from the `DemoBuffer`
2. Computes `mean, std = policy(obs)`
3. Optimises `MSE(mean, expert_action) + 0.01 * mean(std)` — the std penalty encourages near-deterministic behaviour
4. Clips parameters to declared ranges after each gradient step
5. Uses Adam optimiser with gradient clipping

**Key design choice:** BC alone is insufficient for optimal performance (the expert is semi-optimal and the policy structure is constrained), but it provides a strong initialisation that dramatically speeds up RL convergence.

```python
from lares.core.training_pipeline import behavioral_cloning
stats = behavioral_cloning(policy, demo_buffer, num_steps=5000, lr=1e-3)
```

---

## Stage 3 — RL Fine-tuning (GRPO)

**Purpose:** Improve the BC-initialised policy using environment interaction with a GRPO-style policy gradient.

**Function:** `rl_finetune(policy, env, num_iterations=50, episodes_per_iter=20)`

**GRPO (Group Relative Policy Optimization) approach:**
1. Each iteration, collect a **group** of trajectories from the current policy
2. Compute advantages *relative to the group mean return* — this is the core GRPO idea
3. Update policy to increase probability of above-average trajectories
4. Apply entropy bonus (exploration) and L2 KL penalty toward BC initialisation (stability)

**Loss function:**
```
L = -E[log π(a|s) · A_grpo] - β_ent · H(π) + β_kl · ||θ - θ_bc||²
```

Where `A_grpo = (R_traj - mean(R_group)) / std(R_group)` — advantages normalised within the group.

**Why GRPO over PPO/SAC:**
- Symbolic policies have few parameters (10–100 vs. millions for NNs) — on-policy methods work better
- Group-relative advantages reduce variance without a learned baseline
- Simple and effective for the low-dimensional parameter spaces of symbolic policies

```python
from lares.core.training_pipeline import rl_finetune
stats = rl_finetune(policy, env, num_iterations=50, episodes_per_iter=20, lr=3e-4)
```

---

## Stage 4 — LLM Evolution / Structure Search

**Purpose:** Use the LLM to propose and evolve symbolic policy *structures*, training each through the BC→RL inner loop.

**Function:** `llm_evolution(client, env, env_name, demo_buffer, args, ...)`

**How it works:**
1. **Generate:** Call `get_symbolic_policies()` (from `policy_generation.py`) to produce `pop_size` validated symbolic policy structures
2. **Train:** For each candidate:
   - Initialise parameters (from the `__init__` method)
   - Run **behavioral cloning** (Stage 2)
   - Run **RL fine-tuning** (Stage 3)
   - Evaluate performance
3. **Select:** Rank candidates by `success_rate × 1000 + mean_reward`; keep top `elite_num`
4. **Evolve:** Feed the best candidate's performance back to the LLM via `code_feedback.txt`; repeat

This reuses the **entire existing LLM pipeline** (`policy_generation.py`, prompt templates, subprocess validation) — no separate infrastructure needed.

```python
from training_pipeline import llm_evolution
result = llm_evolution(
    client=openai_client,
    env=env,
    env_name="window-close-v2",
    demo_buffer=demo_buffer,
    args=args,
    obs_dim=39,
    action_dim=4,
    num_generations=5,
    pop_size=5,
)
best_policy = result["policy"]
best_code = result["code"]
```

---

## Orchestrator

`SymbolicPolicyPipeline` wraps all four stages:

```python
from training_pipeline import SymbolicPolicyPipeline

pipeline = SymbolicPolicyPipeline("window-close-v2")

# Stage 1
demo_buffer, stats = pipeline.stage1_generate_dataset(env, num_episodes=100)

# Stages 2+3 on a single policy (no LLM)
result = pipeline.run_single_policy(my_policy, env, bc_steps=5000, rl_iterations=50)

# Full Stage 4 with LLM
best = pipeline.stage4_llm_evolution(client, env, args, num_generations=5)
```

---

## Replay Buffer Changes

Added `SimpleReplayBuffer` to `replay_buffer.py`:
- Stores `(obs, action, reward, next_obs, done)` — single reward, no `org_info`
- Circular buffer with configurable size
- Used internally by the training pipeline; the original `replay_buffer` class is unchanged for backward compatibility

---

## Testing

```bash
python test_training_pipeline.py
```

50+ checks across three tiers:

**Tier 1 (mock-based, always runs):**
- `DemoBuffer` operations (add, sample, save/load roundtrip)
- `SimpleReplayBuffer` (capacity, sampling)
- Behavioral cloning (loss decrease, parameter update, output validity)
- Trajectory collection and GRPO advantage computation
- RL fine-tuning (parameter change, stats, output validity)
- `evaluate_policy` (return format, value ranges)
- `SymbolicPolicyPipeline` orchestrator
- Expert policy mapping completeness
- End-to-end gradient flow (BC → RL → backward)
- Prompt templates and policy generation dicts

**Tier 2 (requires MetaWorld):**
- Real environment creation and expert policy loading
- Stages 1–3 with real MetaWorld data and environments
- Full pipeline orchestration on real environments

**Tier 3 (requires OPENAI_API_KEY):**
- LLM API connectivity
- Policy generation, validation, and training
- Stage 4 LLM evolution (minimal run)

All Tier 1 tests use a `MockEnv` and `MockExpertPolicy` — **no MetaWorld or OpenAI API key needed**.

---

## Minimal Changes to Existing Code

| Module | Change | Reason |
|--------|--------|--------|
| `replay_buffer.py` | Added `SimpleReplayBuffer` class | Single-reward storage for the pipeline |
| Everything else | **No changes** | Pipeline is additive — all new functionality lives in `training_pipeline.py` |

The pipeline is designed to be **non-invasive**: it imports from the existing codebase (`symbolic_policy.py`, `policy_generation.py`) but does not modify any existing functions or classes.
