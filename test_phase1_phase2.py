#!/usr/bin/env python
"""
Comprehensive test suite for Phase 1 & Phase 2 implementation.

Run from the LaRes root directory:
    python test_phase1_phase2.py

No OpenAI API key is required. Tests cover:
  Phase 1: SymbolicPolicy base class, validation, subprocess harness
  Phase 2: Prompt templates, obs descriptions, code extraction, end-to-end subprocess
"""

import os
import sys
import re
import pickle
import subprocess

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
PASS = 0
FAIL = 0
ROOT = os.path.dirname(os.path.abspath(__file__))


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def file_to_string(filename):
    with open(filename, "r") as f:
        return f.read()


# ---------------------------------------------------------------------------
#  Import Phase-1 / Phase-2 modules
# ---------------------------------------------------------------------------
print("=" * 70)
print("IMPORTING MODULES")
print("=" * 70)

try:
    from symbolic_policy import SymbolicPolicy

    check("import symbolic_policy", True)
except Exception as e:
    check("import symbolic_policy", False, str(e))
    sys.exit(1)

try:
    from policy_generation import (
        obs_description_dict,
        input_dict_for_policy,
        get_symbolic_policies,
    )

    check("import policy_generation", True)
except Exception as e:
    check("import policy_generation", False, str(e))
    sys.exit(1)


# ===================================================================
#  Define two concrete test policies: one valid, one invalid (has NN)
# ===================================================================
class ValidPolicy(SymbolicPolicy):
    """A multi-parameter symbolic policy suitable for window-close-v2."""

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
        mean = torch.cat([move, grip], dim=-1)[:, : self.action_dim]
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


class InvalidPolicy(SymbolicPolicy):
    """Contains nn.Linear — must be rejected by validate()."""

    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.fc = nn.Linear(obs_dim, action_dim)
        self.log_std = nn.Parameter(torch.tensor(-1.0))

    def forward(self, obs):
        mean = self.fc(obs)
        std = torch.exp(self.log_std) * torch.ones_like(mean)
        return (mean, std)

    def get_param_ranges(self):
        return {"log_std": (-5.0, 0.0)}


# ===================================================================
#  Phase 1 Tests
# ===================================================================
print()
print("=" * 70)
print("PHASE 1: SymbolicPolicy base class")
print("=" * 70)

OBS_DIM = 39
ACT_DIM = 4

# --- 1.1 instantiation ---
p = ValidPolicy(OBS_DIM, ACT_DIM)
check("1.1  Instantiation", p.obs_dim == OBS_DIM and p.action_dim == ACT_DIM)

# --- 1.2 validate() on valid policy ---
try:
    result = p.validate()
    check("1.2  validate() passes for valid policy", result is True)
except Exception as e:
    check("1.2  validate() passes for valid policy", False, str(e))

# --- 1.3 validate() on invalid policy ---
ip = InvalidPolicy(OBS_DIM, ACT_DIM)
try:
    ip.validate()
    check("1.3  validate() rejects NN policy", False, "should have raised TypeError")
except TypeError:
    check("1.3  validate() rejects NN policy", True)
except Exception as e:
    check("1.3  validate() rejects NN policy", False, f"wrong exception: {e}")

# --- 1.4 forward() shapes ---
for bs in [1, 8, 32]:
    obs = torch.randn(bs, OBS_DIM)
    mean, std = p(obs)
    ok = mean.shape == (bs, ACT_DIM) and std.shape == (bs, ACT_DIM)
    check(f"1.4  forward() shape bs={bs}", ok, f"mean={mean.shape} std={std.shape}")

# --- 1.5 std positivity ---
obs = torch.randn(16, OBS_DIM)
_, std = p(obs)
check("1.5  std > 0", (std > 0).all().item())

# --- 1.6 no NaN/Inf ---
mean, std = p(obs)
ok = (
    not torch.isnan(mean).any()
    and not torch.isinf(mean).any()
    and not torch.isnan(std).any()
    and not torch.isinf(std).any()
)
check("1.6  No NaN/Inf in output", ok)

# --- 1.7 gradient flow ---
p.zero_grad()
mean, std = p(torch.randn(4, OBS_DIM))
loss = mean.sum() + std.sum()
loss.backward()
grads_exist = any(
    param.grad is not None and param.grad.abs().sum() > 0
    for param in p.parameters()
)
check("1.7  Gradients flow to parameters", grads_exist)

# --- 1.8 clip_params ---
p_clip = ValidPolicy(OBS_DIM, ACT_DIM)
with torch.no_grad():
    p_clip.w_approach.fill_(999.0)
    p_clip.log_std.fill_(-999.0)
p_clip.clip_params()
check(
    "1.8  clip_params() enforces ranges",
    p_clip.w_approach.item() <= 10.0 and p_clip.log_std.item() >= -5.0,
    f"w_approach={p_clip.w_approach.item()}, log_std={p_clip.log_std.item()}",
)

# --- 1.9 get_param_ranges() ---
ranges = p.get_param_ranges()
check(
    "1.9  get_param_ranges() returns dict with all params",
    isinstance(ranges, dict) and len(ranges) == 6,
    f"len={len(ranges)}",
)
all_valid = all(lo < hi for lo, hi in ranges.values())
check("1.9b All ranges have lo < hi", all_valid)

# --- 1.10 count_parameters ---
num_params = sum(param.numel() for param in p.parameters())
check("1.10 count_parameters()", p.count_parameters() == num_params)

# --- 1.11 state_dict / load_state_dict ---
sd = p.state_dict()
p2 = ValidPolicy(OBS_DIM, ACT_DIM)
p2.load_state_dict(sd)
obs_test = torch.randn(2, OBS_DIM)
m1, s1 = p(obs_test)
m2, s2 = p2(obs_test)
check("1.11 state_dict roundtrip", torch.allclose(m1, m2) and torch.allclose(s1, s2))


# ===================================================================
#  Phase 1: test_generate_policy.py subprocess
# ===================================================================
print()
print("=" * 70)
print("PHASE 1: test_generate_policy.py subprocess validation")
print("=" * 70)

dummy_pkl_path = os.path.join(ROOT, "_test_dummy_data.pkl")
with open(dummy_pkl_path, "wb") as f:
    pickle.dump([{"some_key": 1.0}], f)

VALID_POLICY_CODE = '''
import torch
import torch.nn as nn
from symbolic_policy import SymbolicPolicy

class GeneratedPolicy(SymbolicPolicy):
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.w = nn.Parameter(torch.tensor(2.0))
        self.log_std = nn.Parameter(torch.tensor(-1.0))

    def forward(self, obs):
        tcp = obs[:, 0:3]
        obj = obs[:, 4:7]
        diff = obj - tcp
        dist = torch.norm(diff, dim=-1, keepdim=True) + 1e-8
        direction = diff / dist
        move = self.w * direction
        grip = torch.zeros(obs.shape[0], 1)
        mean = torch.cat([move, grip], dim=-1)[:, :self.action_dim]
        std = torch.exp(self.log_std) * torch.ones_like(mean)
        return (mean, std)

    def get_param_ranges(self):
        return {'w': (0.1, 10.0), 'log_std': (-5.0, 0.0)}
'''

INVALID_NN_POLICY_CODE = '''
import torch
import torch.nn as nn
from symbolic_policy import SymbolicPolicy

class GeneratedPolicy(SymbolicPolicy):
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.fc = nn.Linear(obs_dim, action_dim)
        self.log_std = nn.Parameter(torch.tensor(-1.0))

    def forward(self, obs):
        mean = self.fc(obs)
        std = torch.exp(self.log_std) * torch.ones_like(mean)
        return (mean, std)

    def get_param_ranges(self):
        return {'log_std': (-5.0, 0.0)}
'''

BAD_SHAPE_POLICY_CODE = '''
import torch
import torch.nn as nn
from symbolic_policy import SymbolicPolicy

class GeneratedPolicy(SymbolicPolicy):
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.w = nn.Parameter(torch.tensor(1.0))
        self.log_std = nn.Parameter(torch.tensor(-1.0))

    def forward(self, obs):
        # Wrong: returns (batch, 3) instead of (batch, action_dim=4)
        mean = obs[:, 0:3] * self.w
        std = torch.exp(self.log_std) * torch.ones_like(mean)
        return (mean, std)

    def get_param_ranges(self):
        return {'w': (0.1, 10.0), 'log_std': (-5.0, 0.0)}
'''


def run_validation_subprocess(policy_code, label):
    """Write policy code + test harness to a temp file and run it."""
    head = (
        "import os, sys\n"
        "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n"
        f"sys.path.insert(0, {repr(ROOT)})\n"
    )
    test_code = file_to_string(os.path.join(ROOT, "test_generate_policy.py"))
    full = head + policy_code + "\n\n" + test_code

    tmp_path = os.path.join(ROOT, f"_test_temp_{label}.py")
    with open(tmp_path, "w") as f:
        f.write(full)
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-u",
                tmp_path,
                dummy_pkl_path,
                "--obs_dim",
                "39",
                "--action_dim",
                "4",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout + result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "TIMEOUT", -1
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# Test valid policy subprocess
out, rc = run_validation_subprocess(VALID_POLICY_CODE, "valid")
check("1.12 Subprocess: valid policy -> Success!", "Success!" in out, out[:200])

# Test NN policy rejected
out, rc = run_validation_subprocess(INVALID_NN_POLICY_CODE, "nn")
check(
    "1.13 Subprocess: NN policy rejected",
    "Success!" not in out and "forbidden" in out.lower(),
    out[:200],
)

# Test bad shape rejected
out, rc = run_validation_subprocess(BAD_SHAPE_POLICY_CODE, "shape")
check(
    "1.14 Subprocess: bad shape rejected",
    "Success!" not in out and "Error" in out,
    out[:200],
)


# ===================================================================
#  Phase 2 Tests
# ===================================================================
print()
print("=" * 70)
print("PHASE 2: Prompt templates and policy_generation module")
print("=" * 70)

TASKS = [
    "window-close-v2",
    "window-open-v2",
    "button-press-v2",
    "door-close-v2",
    "drawer-open-v2",
]

# --- 2.1 obs_description_dict coverage ---
for task in TASKS:
    check(
        f"2.1  obs_description_dict['{task}'] exists",
        task in obs_description_dict and len(obs_description_dict[task]) > 50,
    )

# --- 2.2 input_dict_for_policy coverage ---
for task in TASKS:
    check(
        f"2.2  input_dict_for_policy['{task}'] exists",
        task in input_dict_for_policy and len(input_dict_for_policy[task]) > 20,
    )

# --- 2.3 Prompt templates load and format correctly ---
prompt_dir = os.path.join(ROOT, "utils", "policy_prompts")

initial_system = file_to_string(os.path.join(prompt_dir, "initial_system.txt"))
check("2.3a initial_system.txt loads", len(initial_system) > 50)

initial_user = file_to_string(os.path.join(prompt_dir, "new_initial_user.txt"))
check("2.3b new_initial_user.txt loads", len(initial_user) > 200)

code_output_tip = file_to_string(os.path.join(prompt_dir, "new_code_output_tip.txt"))
check("2.3c new_code_output_tip.txt loads", len(code_output_tip) > 100)

code_feedback_tmpl = file_to_string(os.path.join(prompt_dir, "code_feedback.txt"))
check("2.3d code_feedback.txt loads", len(code_feedback_tmpl) > 100)

# --- 2.4 initial_user template formatting ---
try:
    formatted = initial_user.format(
        task="Control the robotic arm to close the window",
        obs_dim=39,
        action_dim=4,
        obs_description=obs_description_dict["window-close-v2"],
        input_dict_string=input_dict_for_policy["window-close-v2"],
    )
    ok = (
        "close the window" in formatted
        and "39" in formatted
        and "4" in formatted
        and "obs[0:3]" in formatted
        and "GeneratedPolicy" in formatted
    )
    check("2.4  initial_user formats correctly", ok)
except Exception as e:
    check("2.4  initial_user formats correctly", False, str(e))

# --- 2.5 code_feedback template formatting ---
try:
    fb = code_feedback_tmpl.format(
        train_steps=200000,
        win_rate=0.35,
        mean_reward=123.4,
        current_output="[{'success': 1.0}, {'success': 0.0}]",
    )
    ok = "200000" in fb and "0.35" in fb and "123.4" in fb
    check("2.5  code_feedback formats correctly", ok)
except Exception as e:
    check("2.5  code_feedback formats correctly", False, str(e))

# --- 2.6 Escaped braces render as dict literal in template ---
try:
    formatted = initial_user.format(
        task="test",
        obs_dim=39,
        action_dim=4,
        obs_description="test",
        input_dict_string="test",
    )
    ok = "{'w1': (0.1, 10.0), 'log_std': (-5.0, 0.0)}" in formatted
    check("2.6  Escaped braces render as dict literal", ok)
except Exception as e:
    check("2.6  Escaped braces render as dict literal", False, str(e))

# --- 2.7 Code extraction regex (same patterns as policy_generation.py) ---
print()
print("-" * 40)
print("Phase 2: Code extraction logic")
print("-" * 40)

patterns = [
    r"```python(.*?)```",
    r"```(.*?)```",
    r'"""(.*?)"""',
    r'""(.*?)""',
    r'"(.*?)"',
]

test_response = '''Here is the policy:

```python
import torch
import torch.nn as nn
from symbolic_policy import SymbolicPolicy

class GeneratedPolicy(SymbolicPolicy):
    def __init__(self, obs_dim, action_dim):
        super().__init__(obs_dim, action_dim)
        self.w = nn.Parameter(torch.tensor(2.0))
        self.log_std = nn.Parameter(torch.tensor(-1.0))

    def forward(self, obs):
        mean = obs[:, :self.action_dim] * self.w
        std = torch.exp(self.log_std) * torch.ones_like(mean)
        return (mean, std)

    def get_param_ranges(self):
        return {'w': (0.1, 10.0), 'log_std': (-5.0, 0.0)}
```

This policy simply scales the first action_dim observations.
'''

extracted = None
for pattern in patterns:
    match = re.search(pattern, test_response, re.DOTALL)
    if match is not None:
        extracted = match.group(1).strip()
        break
check("2.7a Code block extraction", extracted is not None and "GeneratedPolicy" in extracted)

if extracted:
    lines = extracted.split("\n")
    class_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("class ") and "GeneratedPolicy" in line:
            class_start = i
            break
    check("2.7b Class definition found", class_start is not None)
    if class_start is not None:
        class_code = "\n".join(lines[class_start:])
        check("2.7c Class code starts with 'class'", class_code.startswith("class"))

# --- 2.8 get_symbolic_policies callable ---
check(
    "2.8  get_symbolic_policies is callable",
    callable(get_symbolic_policies),
)

# --- 2.9 End-to-end: exec a valid policy in-process ---
print()
print("-" * 40)
print("Phase 2: In-process exec of extracted policy")
print("-" * 40)

exec_imports = (
    "import torch\n"
    "import torch.nn as nn\n"
    "import numpy as np\n"
    "from symbolic_policy import SymbolicPolicy\n"
)
exec_code = exec_imports + VALID_POLICY_CODE.strip()
namespace = {}
try:
    exec(exec_code, namespace)
    ep = namespace["GeneratedPolicy"](39, 4)
    ep.validate()
    m, s = ep(torch.randn(4, 39))
    ok = m.shape == (4, 4) and s.shape == (4, 4) and (s > 0).all()
    check("2.9  In-process exec + validate + forward", ok)
except Exception as e:
    check("2.9  In-process exec + validate + forward", False, str(e))


# ===================================================================
#  Cleanup
# ===================================================================
if os.path.exists(dummy_pkl_path):
    os.remove(dummy_pkl_path)

# ===================================================================
#  Summary
# ===================================================================
print()
print("=" * 70)
TOTAL = PASS + FAIL
print(f"RESULTS: {PASS}/{TOTAL} passed, {FAIL}/{TOTAL} failed")
print("=" * 70)

if FAIL > 0:
    print("SOME TESTS FAILED — see [FAIL] lines above for details.")
    sys.exit(1)
else:
    print("ALL TESTS PASSED!")
    sys.exit(0)
