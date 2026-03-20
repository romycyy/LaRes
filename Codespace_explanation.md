LaRes Codebase Overview
LaRes (LLM-based Adaptive Reward Search) is a NeurIPS 2025 framework that combines Evolutionary Algorithms with Reinforcement Learning. It uses LLMs (e.g., GPT-4o-mini) to generate and evolve a population of reward functions, which then guide SAC-based policy training on MetaWorld robotic manipulation tasks.
Core Training Scripts
LaRes_from_scratch.py (~1400 lines)
Purpose: Main training loop for the "no human reward initialization" setting. This is the primary entry point.
Key functions:
evluate() -- runs one episode in the environment, collects transitions, computes rewards from all reward functions in the population, stores data in the shared replay buffer with reward relabeling.
main() -- orchestrates the full pipeline: initializes a population of SAC agents with parallel Worker processes, calls the OpenAI API to generate/evolve reward functions via LLM prompts, implements Thompson Sampling for selecting which policy to interact with, handles reward scaling and parameter constraints, and periodically triggers LLM-based reward evolution.
Key dictionaries (defined inline): task_description_dict, reward_function_format_dict, criteria_code_dict, input_dict -- these provide per-task metadata used to construct LLM prompts.
Interacts with: sac.py, utils.py, replay_buffer.py, arguments.py, models.py, OpenAI API, WandB.
LaRes_with_init.py (~850 lines)
Purpose: Main training loop for the "with human reward initialization" setting. Structurally very similar to LaRes_from_scratch.py.
Key difference: Instead of generating reward functions purely from scratch, it initializes the reward population with human-designed reward functions imported from utils.py (reward_function_dict, parents_function_dict).
Interacts with: Same modules as above, plus imports task-specific data from utils.py.
RL Components
sac.py (747 lines)
Purpose: Implements the SAC (Soft Actor-Critic) reinforcement learning agent.
Key classes:
tanh_normal -- Tanh-squashed Normal distribution for the reparameterization trick.
reward_recorder -- Tracks rolling average of episode rewards/success.
get_action_info -- Wraps policy output into a distribution, handles action selection and log-probability computation.
sac_agent -- The core SAC agent. Manages:
A population of actor networks (self.pop) plus a dedicated RL actor (self.actor_net).
Twin Q-networks (qf1, qf2) with target networks.
_update_newtork() -- performs one SAC gradient step with optional L2 parameter constraint (KL-like penalty) toward an elite network, and reward relabeling through indexed buffer sampling.
_evaluate_agent() -- evaluates a policy over multiple episodes, optionally computing custom reward function values.
_initial_exploration() / _initial_exploration_and_store() -- collects initial trajectories for warm-starting the buffer.
rl_to_evo() -- copies RL actor parameters into an evolutionary population slot.
Interacts with: models.py (networks), replay_buffer.py (experience storage), rlkit/ (environment wrapper).
models.py (166 lines)
Purpose: Defines the neural network architectures used by SAC.
Key classes:
flatten_mlp -- A 3-hidden-layer MLP used for Q-value networks. Takes (obs, action) as input, outputs a scalar Q-value. Has a get_fau() method to compute fraction of active units (a plasticity metric).
tanh_gaussian_actor -- A 3-hidden-layer MLP actor that outputs (mean, std) for a Gaussian policy. Includes methods for evolutionary operations: set_params(), get_params(), get_size(), inject_parameters(), extract_parameters(), extract_grad(), count_parameters() -- enabling flat parameter-vector manipulation for CEM/ES integration.
Interacts with: Used by sac.py to construct all actor and critic networks.
replay_buffer.py (66 lines)
Purpose: Implements a shared experience replay buffer that supports multi-reward storage and relabeling.
Key class: replay_buffer
Stores tuples of (org_info, obs, action, reward_list, obs_, done) where reward_list is a list of reward values from all reward functions in the population.
sample(batch_size, index) -- samples a batch, selecting the reward at index from each transition's reward list. This enables reward relabeling: when a new reward function replaces an old one, historical data can be relabeled without recollection.
sample_with_elite(batch_size, index, elite_index, dynamic_weight) -- samples with a weighted blend of the agent's own reward and the elite's reward.
sample_data_for_fau() -- samples data for computing active unit fraction.
Interacts with: Used by sac.py and both main training scripts.
Task Configuration & Utilities
utils.py (~2540 lines)
Purpose: The central hub for task definitions and multi-process training infrastructure. This is the largest and most complex utility file.
Key components:
Task dictionaries (for LaRes_with_init.py):
criteria_code_dict -- per-task success criteria code strings.
task_description_dict -- natural language task descriptions for LLM prompts.
input_dict -- per-task lists of available environment variables.
reward_function_dict -- human-designed reward function code (Python strings) for each task.
parents_function_dict -- gripper caging reward helper functions for each task.
Classes:
Worker(mp.Process) -- a multiprocessing worker that owns a local SAC agent and a copy of the replay buffer. Receives training instructions from the main process via shared parameters dict, performs gradient updates, relabels rewards, and sends back updated network weights via a queue.
_GetDict / _patch_get_dict() -- picklable callable that implements the get_dict() API for MetaWorld V3 environments, extracting state information (TCP position, object position, target position, etc.) needed by reward functions.
make_metaworld_env() -- factory function to create and wrap a MetaWorld environment with NormalizedBoxEnv and TimeLimit.
tanh_normal, get_action_info, reward_recorder, env_wrapper -- duplicated from sac.py for use in the multiprocessing context (necessary because spawned processes need their own copies).
Interacts with: Imported by both LaRes_from_scratch.py and LaRes_with_init.py.
arguments.py (61 lines)
Purpose: Defines all command-line arguments via argparse.
Key function: get_args() -- returns parsed arguments including environment name, hyperparameters (learning rates, batch size, gamma, tau), population size, EA parameters (elite_num, EA_tau, damp, sigma_init), LLM settings (model, LLM_freq), Thompson sampling parameters (windows_length, ucb_type, c), and reward scaling options.
Interacts with: Imported by both main training scripts.
reward_utils.py (225 lines)
Purpose: Reward shaping utilities adapted from DeepMind's dm_control.
Key functions:
tolerance(x, bounds, margin, sigmoid) -- returns 1 when x is inside bounds, smoothly decays outside using configurable sigmoid functions (gaussian, hyperbolic, long_tail, reciprocal, cosine, linear, quadratic, tanh_squared).
inverse_tolerance() -- the complement of tolerance.
rect_prism_tolerance() -- computes a reward based on whether a 3D point is inside a rectangular prism.
hamacher_product(a, b) -- T-norm product used to combine reward components multiplicatively (like fuzzy AND).
Interacts with: Used extensively inside the human-designed reward functions stored in utils.py dictionaries, and also available to LLM-generated reward functions.
test_generate_code.py (54 lines)
Purpose: Validation script to test that an LLM-generated compute_reward function executes correctly and returns a scalar reward.
Key behavior: Loads a pickle file of environment state dicts, parses the function signature of compute_reward from the generated code (via AST), calls it with matching inputs, and checks the return type. Prints "Success!" if valid.
Interacts with: Called as a subprocess by the main training scripts to validate generated reward code before using it.
Environment Wrappers
rlkit/envs/proxy_env.py (49 lines)
Purpose: A transparent gym.Env proxy that delegates all calls to a wrapped environment. Provides __getattr__ pass-through for attribute access.
Interacts with: Base class for NormalizedBoxEnv.
rlkit/envs/wrappers/normalized_box_env.py (63 lines)
Purpose: Normalizes action space to [-1, 1], optionally normalizes observations, and scales rewards.
Key class: NormalizedBoxEnv -- wraps MetaWorld environments so SAC agents operate in a normalized action space. The step() method rescales actions from [-1, 1] back to the environment's native range, and handles the Gymnasium 5-tuple return (next_obs, reward, terminated, truncated, info).
Interacts with: Used by make_metaworld_env() in utils.py.
Helper Utilities (utils/ directory)
utils/create_task.py
Creates YAML config files for IsaacGym tasks. Not directly used in the MetaWorld pipeline.
utils/extract_task_code.py
Parses Python source files to extract task code and reward function code. Includes get_function_signature() to extract function names and parameter lists via AST.
utils/file_utils.py
File searching, TensorBoard log loading, and dynamic module importing (import_class_from_file).
utils/misc.py
GPU selection utilities (get_freest_gpu), traceback filtering, and training status monitoring (block_until_training).
utils/prune_env.py, utils/prune_env_dexterity.py, utils/prune_env_isaac.py
Scripts for pruning and modifying IsaacGym environment files to create simplified versions for LLM prompting. These are from a related project (Eureka-style) and not directly used in the MetaWorld pipeline.
Overall Pipeline Flow
Initialization: Parse args, create MetaWorld environment, initialize SAC agent with a population of actors, create shared replay buffer, spawn Worker processes.
LLM Reward Generation: At intervals (LLM_freq), construct prompts with task description, available variables, success criteria, and performance feedback. Call OpenAI API to generate/mutate reward function code.
Validation: Test generated code with test_generate_code.py using saved environment states.
Reward Relabeling: When new reward functions arrive, relabel all existing buffer data by re-computing rewards from stored org_info state dicts.
Reward Scaling: Normalize new reward functions to match the statistical properties (mean, scale) of the elite reward, ensuring stable training.
Training: Each Worker runs SAC updates using rewards indexed from the shared buffer. Thompson Sampling selects which policy collects new data.
Parameter Constraints: Non-elite agents are regularized toward the elite via L2 distance penalties on network parameters.
Evaluation & Logging: Periodically evaluate all agents, log metrics to WandB, save best model checkpoints.