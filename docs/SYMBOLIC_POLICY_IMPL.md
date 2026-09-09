# Symbolic Policy Implementation Summary

> **Status: partly superseded.** The base class and harness described below still exist, but the
> contract has grown. Generated policies now read the observation through named accessors rather
> than raw indices, every parameter must initialise inside its declared range, batch coupling and
> insensitivity to a required field are rejected, and the subprocess harness delegates to
> `lares/core/policy_validator.py` instead of carrying its own checks. See the `SymbolicPolicy`
> contract section of `CLAUDE.md` for the current rules.

## Objective

Modify the LaRes system so the LLM proposes **learnable symbolic policies** instead of reward functions. The generated policies must be purely symbolic (no neural networks), contain learnable parameters with defined weight ranges, output action probability distributions, and conform to a default policy class interface.

## Implementation Status

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 1 | Done | Foundation: base class, validation, prompts |
| Phase 2 | Done | Policy generation pipeline |

---

## Phase 1 & 2: Files Created

### `symbolic_policy.py` — Base Class

The `SymbolicPolicy(nn.Module)` base class provides:

- **Interface**: `forward(obs) -> (mean, std)` — outputs `(batch, action_dim)` mean and std for a Gaussian action distribution.
- **`obs_field(obs, name)`**: named view of one observation field. The only sanctioned way to read the
  observation; raw slicing is rejected by a static check.
- **`record_gate(name, value)` / `reset_diagnostics()`**: optional detached phase-gate telemetry.
- **`get_param_ranges()`**: Abstract method returning `{param_name: (min, max)}` for every `nn.Parameter`.
- **`validate()`**: Checks that no forbidden NN modules (Linear, Conv, LSTM, Transformer, etc.) are present in the policy.
- **`clip_params()`**: Projects parameters onto their declared ranges after gradient steps.
- **`count_parameters()`**: Returns total number of scalar learnable parameters.
- Inherits `state_dict()`/`load_state_dict()` from `nn.Module` for serialisation.

### `test_generate_policy.py` — Subprocess Validation Harness

Appended to LLM-generated code and run as a subprocess. It binds the observation schema named by
`--env_name`, recovers the generated source by splitting its own file on the harness banner, and
delegates every rule to `lares.core.policy_validator.validate_policy`, which checks:

1. no raw observation indexing in the source
2. `GeneratedPolicy` instantiates with `(obs_dim, action_dim)`
3. no forbidden NN modules
4. every parameter declared, bounded, and initialised inside its own range
5. correct `(mean, std)` shapes at batch sizes 1, 4, 16, with `std > 0` and no NaN/Inf
6. row `i` of a batched forward equals that row evaluated alone
7. the action responds to every field the schema marks required
8. gradients reach the parameters
9. gate telemetry is detached, batch-shaped and inert
10. `state_dict` round-trips

All local variables use `_` prefix to avoid name collisions with generated code.

### `lares/utils/policy_prompts/` — Prompt Templates

| File | Purpose | Format Placeholders |
|------|---------|-------------------|
| `initial_system.txt` | System role definition | None |
| `new_initial_user.txt` | Task + interface specification | `{task}`, `{obs_dim}`, `{action_dim}`, `{obs_description}`, `{input_dict_string}` |
| `new_code_output_tip.txt` | 10 design guidelines | None |
| `code_feedback.txt` | Performance feedback + improvement tips | `{train_steps}`, `{win_rate}`, `{mean_reward}`, `{current_output}` |
| `ideas_system.txt` | Two-phase ideation: design hypotheses only (no code) | None |
| `ideas_user.txt` | Ideation task + JSON schema for `n` ideas | `{task}`, `{obs_dim}`, `{action_dim}`, `{obs_description}`, `{input_dict_string}`, `{n}` |

The canonical path is `lares/utils/policy_prompts/` (see `load_policy_prompt_assets` in `lares/core/training_pipeline.py`).

Key design choices:
- The user prompt includes the full `SymbolicPolicy` interface so the LLM knows exactly what to implement.
- Guidelines emphasize phase decomposition, smooth transitions (sigmoid), direction normalization, and differentiability.
- The LLM must name its class `GeneratedPolicy` for automated extraction.

### `policy_generation.py` — Generation Pipeline

Contains:
- **`obs_description_dict`**: Per-task documentation of MetaWorld V2 observation indices (what obs[0:3], obs[4:7], obs[36:39] mean for each task).
- **`input_dict_for_policy`**: Per-task named state variable descriptions.
- **`_call_llm()`**: OpenAI API wrapper with retry/backoff.
- **`get_symbolic_policies()`**: Main entry point. Returns `(policy_pop, code_string_pop, response_list)`.

Generation flow:
```
1. Format prompts with task/env info
2. Build message list (initial or evolution feedback). Optional two-phase: ideation
   (`ideas_system.txt`, `ideas_user.txt`) then implementation (`policy_impl_mode`
   batched or per-idea); controlled by `args.policy_gen_two_phase` / YAML in
   `run_full_evolution.yaml`.
3. Call LLM, extract code block from response (regex)
4. Find "class GeneratedPolicy" definition
5. Write temp file: path_setup + imports + generated_code + test_harness
6. Run as subprocess, check for "Success!"
7. On pass: exec() in-process, instantiate, validate()
8. Save response text and code to log directory
```

### `test_phase1_phase2.py` — Test Suite

Comprehensive test covering 25+ checks across Phase 1 and Phase 2. Run with:
```bash
python tests/test_phase1_phase2.py
```

---

## Architecture: How It Connects

```
NEW (symbolic policy search):
  LLM -> GeneratedPolicy(SymbolicPolicy) -> BC + RL optimise nn.Parameters -> actions
         ├── forward(obs) -> (mean, std)     Gaussian action distribution
         ├── get_param_ranges()              declares bounds on learnable params
         └── clip_params()                   enforces bounds after gradient steps
```
