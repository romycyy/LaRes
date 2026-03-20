"""
Policy generation pipeline for LLM-generated symbolic policies.

Generates, validates, and instantiates SymbolicPolicy subclasses from LLM
output.  Follows the same pattern as reward-function generation in
LaRes_from_scratch.py but produces policy *structures* instead of reward
functions.
"""

import os
import re
import time
import subprocess
from datetime import datetime

# ---------------------------------------------------------------------------
#  Per-task observation descriptions for MetaWorld V2
# ---------------------------------------------------------------------------

obs_description_dict = {
    "window-close-v2": """The 39-dimensional observation vector contains:
  obs[0:3]   - End-effector (gripper/TCP) position (x, y, z)
  obs[3]     - Normalized gripper opening distance
  obs[4:7]   - Window handle position (x, y, z) — the object to manipulate
  obs[7:11]  - Window handle quaternion orientation (4 values)
  obs[18:21] - Previous end-effector position (x, y, z)
  obs[36:39] - Goal/target position (x, y, z) — where the window should move to
Actions are 4-dimensional: (dx, dy, dz, gripper) controlling movement and gripper.
The task requires moving the gripper to the window handle and pushing it to close.""",
    "window-open-v2": """The 39-dimensional observation vector contains:
  obs[0:3]   - End-effector (gripper/TCP) position (x, y, z)
  obs[3]     - Normalized gripper opening distance
  obs[4:7]   - Window handle position (x, y, z) — the object to manipulate
  obs[7:11]  - Window handle quaternion orientation (4 values)
  obs[18:21] - Previous end-effector position (x, y, z)
  obs[36:39] - Goal/target position (x, y, z) — where the window should move to
Actions are 4-dimensional: (dx, dy, dz, gripper) controlling movement and gripper.
The task requires moving the gripper to the window handle and pulling it to open.""",
    "button-press-v2": """The 39-dimensional observation vector contains:
  obs[0:3]   - End-effector (gripper/TCP) position (x, y, z)
  obs[3]     - Normalized gripper opening distance
  obs[4:7]   - Button position (x, y, z) — the current button location
  obs[7:11]  - Button quaternion orientation (4 values)
  obs[36:39] - Goal/target position (x, y, z) — where the button should be pressed to
Actions are 4-dimensional: (dx, dy, dz, gripper) controlling movement and gripper.
The task requires moving the gripper above the button and pressing downward.""",
    "door-close-v2": """The 39-dimensional observation vector contains:
  obs[0:3]   - End-effector (gripper/TCP) position (x, y, z)
  obs[3]     - Normalized gripper opening distance
  obs[4:7]   - Door handle position (x, y, z)
  obs[7:11]  - Door quaternion orientation (4 values)
  obs[36:39] - Goal/target position (x, y, z) — target door position (closed)
Actions are 4-dimensional: (dx, dy, dz, gripper) controlling movement and gripper.
The task requires moving the gripper to the door and pushing it closed.""",
    "drawer-open-v2": """The 39-dimensional observation vector contains:
  obs[0:3]   - End-effector (gripper/TCP) position (x, y, z)
  obs[3]     - Normalized gripper opening distance
  obs[4:7]   - Drawer handle position (x, y, z)
  obs[7:11]  - Drawer handle quaternion orientation (4 values)
  obs[36:39] - Goal/target position (x, y, z) — target drawer position (open)
Actions are 4-dimensional: (dx, dy, dz, gripper) controlling movement and gripper.
The task requires grasping the drawer handle and pulling it open.""",
    "reach-v2": """The 39-dimensional observation vector contains:
  obs[0:3]   - End-effector (gripper/TCP) position (x, y, z)
  obs[3]     - Normalized gripper opening distance
  obs[4:7]   - Target/goal marker position (x, y, z) — the point to reach
  obs[7:11]  - Target marker quaternion orientation (4 values, unused for reach)
  obs[18:21] - Previous end-effector position (x, y, z)
  obs[36:39] - Goal/target position (x, y, z) — same as obs[4:7] for reach
Actions are 4-dimensional: (dx, dy, dz, gripper) controlling movement and gripper.
The task is simply to move the end-effector to the target position. No grasping needed.""",
}

# Reuse the existing input_dict strings so the LLM understands variable names
input_dict_for_policy = {
    "window-close-v2": """{
  "tcp": "3D position of the robotic arm end-effector",
  "obj": "3D position of the window handle",
  "target": "3D target position the window should move to",
  "actions": "4D action vector (dx, dy, dz, gripper)"}""",
    "window-open-v2": """{
  "tcp": "3D position of the robotic arm end-effector",
  "obj": "3D position of the window handle",
  "target": "3D target position the window should move to",
  "actions": "4D action vector (dx, dy, dz, gripper)"}""",
    "button-press-v2": """{
  "tcp": "3D position of the robotic arm end-effector",
  "init_tcp": "3D initial position of the robotic arm",
  "target_pos": "3D target position of the button",
  "current_pos": "3D current position of the button",
  "action": "4D action vector (dx, dy, dz, gripper)"}""",
    "door-close-v2": """{
  "tcp": "3D position of the robotic arm end-effector",
  "target": "3D target position (closed door)",
  "obj": "3D current position of the door",
  "obj_init_pos": "3D initial position of the door",
  "hand_init_pos": "3D initial position of the door handle",
  "actions": "4D action vector (dx, dy, dz, gripper)"}""",
    "drawer-open-v2": """{
  "handle": "3D position of the drawer handle",
  "_target_pos": "3D drawer target position (open)",
  "gripper": "3D position of the gripper",
  "init_tcp": "3D initial position of the robotic arm",
  "actions": "4D action vector (dx, dy, dz, gripper)"}""",
    "reach-v2": """{
  "tcp": "3D position of the robotic arm end-effector",
  "target": "3D target position the arm should reach",
  "actions": "4D action vector (dx, dy, dz, gripper)"}""",
}


# ---------------------------------------------------------------------------
#  LLM helper
# ---------------------------------------------------------------------------


def _call_llm(client, sample_size, model, messages, temperature):
    """Call the OpenAI chat completion API.  Same retry logic as
    ``get_LLM_reward_function`` in ``LaRes_from_scratch.py``."""
    total_samples = 0
    total_token = 0
    total_completion_token = 0
    prompt_tokens = 0
    chunk_size = sample_size
    responses = []

    while total_samples < sample_size:
        response_cur = None
        for attempt in range(1000):
            try:
                response_cur = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    n=chunk_size,
                )
                total_samples += chunk_size
                break
            except Exception as e:
                if attempt >= 10:
                    chunk_size = max(int(chunk_size / 2), 1)
                    print("Current Chunk Size", chunk_size)
                print(f"Attempt {attempt + 1} failed with error: {e}")
                time.sleep(1)
        if response_cur is None:
            print("Code terminated due to too many failed attempts!")
            exit()

        responses.extend(response_cur.choices)
        prompt_tokens += response_cur.usage.prompt_tokens
        total_completion_token += response_cur.usage.completion_tokens
        total_token += response_cur.usage.total_tokens

    return responses, prompt_tokens, total_completion_token, total_token


# ---------------------------------------------------------------------------
#  Main entry point
# ---------------------------------------------------------------------------


def _extract_code_string(response_text):
    """Extract the policy class code from an LLM response."""
    patterns = [
        r"```python(.*?)```",
        r"```(.*?)```",
        r'"""(.*?)"""',
        r'""(.*?)""',
        r'"(.*?)"',
    ]
    code_string = None
    for pattern in patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match is not None:
            code_string = match.group(1).strip()
            break
    if code_string is None:
        code_string = response_text

    lines = code_string.split("\n")
    class_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("class ") and "GeneratedPolicy" in stripped:
            class_start = i
            break
    if class_start is None:
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("class ") and "SymbolicPolicy" in stripped:
                class_start = i
                break
    if class_start is not None:
        code_string = "\n".join(lines[class_start:])

    return code_string


def _build_error_feedback(code_string, error_output, obs_dim, action_dim):
    """Build a concise error message to feed back to the LLM for self-repair."""
    error_lines = error_output.strip().split("\n")
    short_error = "\n".join(error_lines[:20])
    return (
        f"Your previous GeneratedPolicy failed validation with this error:\n"
        f"```\n{short_error}\n```\n\n"
        f"The failed code was:\n"
        f"```python\n{code_string}\n```\n\n"
        f"Please fix the code and return the corrected GeneratedPolicy class."
    )


def get_symbolic_policies(
    client,
    dir_path,
    llm_iter,
    args,
    obs_dim,
    action_dim,
    initial_system,
    initial_user,
    task_description,
    obs_description,
    input_dict_string,
    code_output_tip,
    data_pkl_path,
    provided_response=None,
    code_feedback=None,
    real_num=5,
    max_total_attempts=50,
    max_repair_per_candidate=3,
):
    """Generate and validate symbolic policies via the LLM.

    When a candidate fails validation, the error message and failed code are
    fed back to the LLM so it can self-repair (up to ``max_repair_per_candidate``
    times per candidate).  Falls back to fresh generation if repair fails.

    Parameters
    ----------
    max_total_attempts : int
        Hard limit on total LLM calls to prevent infinite loops.
    max_repair_per_candidate : int
        Max error-feedback repair attempts per failed candidate before
        moving on to the next candidate.

    Returns
    -------
    policy_pop : list[SymbolicPolicy]
        Instantiated symbolic policy objects.
    code_string_pop : list[str]
        Raw policy class code strings.
    response_list : list[str]
        Full LLM response texts (for feedback in later iterations).
    """
    policy_pop = []
    code_string_pop = []
    response_list = []
    get_res_try_num = 0
    success_num = 0

    # ---- build the base LLM message list ----
    user_content = initial_user.format(
        task=task_description,
        obs_dim=obs_dim,
        action_dim=action_dim,
        obs_description=obs_description,
        input_dict_string=input_dict_string,
    )

    if provided_response is not None and code_feedback is not None:
        base_messages = [
            {"role": "system", "content": initial_system},
            {"role": "user", "content": user_content + "\n" + code_output_tip},
            {"role": "assistant", "content": provided_response},
            {"role": "user", "content": code_feedback + "\n" + code_output_tip},
        ]
    elif code_feedback is not None:
        base_messages = [
            {"role": "system", "content": initial_system},
            {
                "role": "user",
                "content": user_content + "\n" + code_feedback + "\n" + code_output_tip,
            },
        ]
    else:
        base_messages = [
            {"role": "system", "content": initial_system},
            {"role": "user", "content": user_content + "\n" + code_output_tip},
        ]

    # ---- helpers ----
    try_num = 1
    total_llm_calls = 0
    root_dir = os.path.dirname(os.path.abspath(__file__))

    imports = (
        "import torch\n"
        "import torch.nn as nn\n"
        "import numpy as np\n"
        "from symbolic_policy import SymbolicPolicy\n"
    )
    head = "import os\n" "import sys\n" f"sys.path.insert(0, {repr(root_dir)})\n"
    test_code_path = os.path.join(root_dir, "test_generate_policy.py")
    with open(test_code_path, "r") as f:
        test_code = f.read()

    def _validate_subprocess(code_str):
        """Write code to file, run validation subprocess, return (ok, output)."""
        full_code = head + imports + code_str
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
        temp_file_path = os.path.join(dir_path, f"{timestamp}_generated_policy.py")
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write(full_code)
            f.write("\n\n# --- Validation code ---\n")
            f.write(test_code)

        filter_filepath = os.path.join(
            dir_path, f"Policy_{get_res_try_num}_response.txt"
        )
        with open(filter_filepath, "w") as f:
            process = subprocess.Popen(
                [
                    "python",
                    "-u",
                    temp_file_path,
                    data_pkl_path,
                    "--obs_dim",
                    str(obs_dim),
                    "--action_dim",
                    str(action_dim),
                ],
                stdout=f,
                stderr=f,
            )
        process.communicate()

        with open(filter_filepath, "r") as f:
            stdout_str = f.read()
        return "Success!" in stdout_str, stdout_str, full_code

    def _try_instantiate(code_str):
        """exec the code in-process and instantiate the policy."""
        namespace = {}
        exec_code = imports + code_str
        exec(exec_code, namespace)
        policy = namespace["GeneratedPolicy"](obs_dim, action_dim)
        policy.validate()
        return policy

    # ---- generation loop ----
    while success_num < real_num and total_llm_calls < max_total_attempts:
        responses, prompt_tokens, total_completion_token, total_token = _call_llm(
            client, real_num * 2, args.model, base_messages, 1.0
        )
        total_llm_calls += 1

        for resp_idx in range(len(responses)):
            if success_num >= real_num:
                break
            if total_llm_calls >= max_total_attempts:
                break

            response_cur = responses[resp_idx].message.content
            print(
                f"Policy gen try {try_num}",
                prompt_tokens,
                total_completion_token,
                total_token,
            )
            try_num += 1

            code_string = _extract_code_string(response_cur)

            # ---- validate, with error-feedback repair loop ----
            ok, stdout_str, full_code = _validate_subprocess(code_string)
            repair_attempt = 0

            while not ok and repair_attempt < max_repair_per_candidate:
                repair_attempt += 1
                print(
                    f"Policy validation failed (attempt {resp_idx}, "
                    f"repair {repair_attempt}/{max_repair_per_candidate}): "
                    f"{stdout_str[:200]}"
                )

                error_fb = _build_error_feedback(
                    code_string, stdout_str, obs_dim, action_dim
                )
                repair_messages = base_messages + [
                    {"role": "assistant", "content": response_cur},
                    {"role": "user", "content": error_fb + "\n" + code_output_tip},
                ]

                repair_responses, _, _, _ = _call_llm(
                    client, 1, args.model, repair_messages, 0.7
                )
                total_llm_calls += 1
                response_cur = repair_responses[0].message.content
                code_string = _extract_code_string(response_cur)
                print(f"Policy gen try {try_num} (repair)")
                try_num += 1

                ok, stdout_str, full_code = _validate_subprocess(code_string)

            if not ok:
                print(
                    f"Policy validation failed after {repair_attempt} repairs "
                    f"(attempt {resp_idx}): {stdout_str[:200]}"
                )
                continue

            print(stdout_str)

            # ---- validation passed — save artefacts ----
            response_txt_path = os.path.join(
                dir_path,
                f"Iter_{llm_iter}_Policy_Response_{success_num}.txt",
            )
            with open(response_txt_path, "w", encoding="utf-8") as f:
                f.write(response_cur)

            saved_code_path = os.path.join(
                dir_path,
                f"Iter_{llm_iter}_Policy_Code_{get_res_try_num}.py",
            )
            with open(saved_code_path, "w", encoding="utf-8") as f:
                f.write(full_code)

            try:
                policy = _try_instantiate(code_string)
            except Exception as e:
                print(f"Error instantiating policy after validation: {e}")
                continue

            policy_pop.append(policy)
            code_string_pop.append(code_string)
            response_list.append(response_cur)
            get_res_try_num += 1
            success_num += 1

    if success_num < real_num:
        print(
            f"Warning: only generated {success_num}/{real_num} valid policies "
            f"after {total_llm_calls} LLM calls (limit: {max_total_attempts})"
        )

    return policy_pop, code_string_pop, response_list
