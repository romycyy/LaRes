"""
Policy generation pipeline for LLM-generated symbolic policies.

Generates, validates, and instantiates SymbolicPolicy subclasses from LLM
output.  Follows the same pattern as reward-function generation in
LaRes_from_scratch.py but produces policy *structures* instead of reward
functions.
"""

import json
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

# OpenAI chat.completions `n` is capped per model (often 8 for recent models).
_OPENAI_CHAT_N_MAX = 8


def _usage_to_dict(usage):
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def _append_llm_transcript(transcript_path, call_seq, label, messages, response_obj):
    """Append one request/response exchange to the evolution LLM log file."""
    if not transcript_path:
        return
    parts = [
        f"\n{'=' * 72}\n",
        f"call {call_seq} — {label} — {datetime.now().isoformat()}\n",
        "--- REQUEST (messages) ---\n",
        json.dumps(messages, ensure_ascii=False, indent=2),
        "\n--- RESPONSE ---\n",
    ]
    for i, ch in enumerate(response_obj.choices):
        content = ""
        if ch.message is not None and ch.message.content is not None:
            content = ch.message.content
        parts.append(f"\n### choice {i}\n{content}\n")
    parts.append("\n--- usage ---\n")
    parts.append(json.dumps(_usage_to_dict(response_obj.usage), indent=2))
    parts.append("\n")
    with open(transcript_path, "a", encoding="utf-8") as f:
        f.write("".join(parts))


def _call_llm(
    client,
    sample_size,
    model,
    messages,
    temperature,
    transcript_path=None,
    transcript_counter=None,
    transcript_label="completion",
):
    """Call the OpenAI chat completion API.  Same retry logic as
    ``get_LLM_reward_function`` in ``LaRes_from_scratch.py``."""
    total_samples = 0
    total_token = 0
    total_completion_token = 0
    prompt_tokens = 0
    chunk_size = min(sample_size, _OPENAI_CHAT_N_MAX)
    responses = []

    while total_samples < sample_size:
        need = sample_size - total_samples
        response_cur = None
        for attempt in range(1000):
            this_n = min(chunk_size, need, _OPENAI_CHAT_N_MAX)
            try:
                response_cur = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    n=this_n,
                )
                total_samples += this_n
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

        if transcript_path and transcript_counter is not None:
            transcript_counter[0] += 1
            _append_llm_transcript(
                transcript_path,
                transcript_counter[0],
                transcript_label,
                messages,
                response_cur,
            )

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
    code_string = None
    for pattern in (
        r"```python(.*?)```",
        r"```(.*?)```",
    ):
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
        if match is not None:
            code_string = match.group(1).strip()
            break
    if code_string is None:
        # Plain Python (e.g. body of a ```python fence): do not use r'"(.*?)"' — it
        # grabs JSON keys like "w_move" inside get_param_ranges() dicts.
        if "GeneratedPolicy" in response_text and "class " in response_text:
            code_string = response_text.strip()
        else:
            for pattern in (
                r'"""(.*?)"""',
                r'""(.*?)""',
                r'"(.*?)"',
            ):
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


GENERATION_MODE_SINGLE_SHOT = "single_shot"
GENERATION_MODE_TWO_PHASE = "two_phase"
POLICY_IMPL_BATCHED = "batched"
POLICY_IMPL_PER_IDEA = "per_idea"


def _run_policy_validation_subprocess(
    code_str,
    dir_path,
    response_log_idx,
    head,
    imports,
    test_code,
    data_pkl_path,
    obs_dim,
    action_dim,
):
    """Write policy code plus harness, run subprocess validation. Returns (ok, stdout, full_code)."""
    full_code = head + imports + code_str
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    temp_file_path = os.path.join(dir_path, f"{timestamp}_generated_policy.py")
    with open(temp_file_path, "w", encoding="utf-8") as f:
        f.write(full_code)
        f.write("\n\n# --- Validation code ---\n")
        f.write(test_code)

    filter_filepath = os.path.join(dir_path, f"Policy_{response_log_idx}_response.txt")
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


def _try_instantiate_policy(imports, code_str, obs_dim, action_dim):
    """exec generated code and instantiate ``GeneratedPolicy``."""
    namespace = {}
    exec_code = imports + code_str
    exec(exec_code, namespace)
    policy = namespace["GeneratedPolicy"](obs_dim, action_dim)
    policy.validate()
    return policy


def _parse_ideas_response(text, expected_n):
    """Parse LLM output into ``expected_n`` idea strings (JSON preferred)."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t)
    ideas = None
    try:
        data = json.loads(t)
        if isinstance(data, dict) and isinstance(data.get("ideas"), list):
            ideas = [str(x).strip() for x in data["ideas"] if str(x).strip()]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    if ideas is None or len(ideas) < expected_n:
        alt = []
        for m in re.finditer(r"(?m)^\s*\d+[\.)]\s+(.+)$", text):
            line = m.group(1).strip()
            if line:
                alt.append(line)
        if len(alt) >= expected_n:
            ideas = alt[:expected_n]
    if ideas is None:
        ideas = []
    if len(ideas) >= expected_n:
        return ideas[:expected_n]
    pad = (
        "Use smooth phase gating toward the task-relevant geometry in the observation, "
        "normalize direction vectors before scaling, and learn magnitudes with nn.Parameter."
    )
    while len(ideas) < expected_n:
        ideas.append(pad)
    return ideas[:expected_n]


def _extract_n_policy_codes(response_text, n):
    """Extract up to ``n`` ``GeneratedPolicy`` code strings from ordered fenced blocks."""
    blocks = re.findall(r"```python\s*(.*?)```", response_text, re.DOTALL | re.IGNORECASE)
    if len(blocks) < n:
        for m in re.finditer(r"```\s*(\w*)\s*(.*?)```", response_text, re.DOTALL):
            inner = m.group(2).strip()
            if inner and inner not in blocks:
                blocks.append(inner)
            if len(blocks) >= n * 2:
                break
    out = []
    for raw in blocks:
        code = _extract_code_string(raw.strip())
        if code and "GeneratedPolicy" in code:
            out.append(code)
        if len(out) >= n:
            break
    return out[:n]


def _format_batched_implementation_request(ideas):
    """User-facing instructions: one response, ``n`` fenced Python policies in hypothesis order."""
    lines = [
        "## Design hypotheses to implement (in order; one policy each)",
        "",
    ]
    for i, idea in enumerate(ideas):
        lines.append(f"### Hypothesis {i + 1}")
        lines.append(idea)
        lines.append("")
    lines.extend(
        [
            "## Required output",
            f"Output exactly {len(ideas)} separate Markdown code fences of the form ```python ... ```.",
            "Each fence must contain exactly one complete class named `GeneratedPolicy` subclassing `SymbolicPolicy`.",
            "The k-th code fence (k starting at 1) must implement hypothesis k only — do not merge designs.",
            "Do not add extra text inside a fence except valid Python for that class.",
        ]
    )
    return "\n".join(lines)


def get_policy_ideas(
    client,
    args,
    n,
    ideas_system,
    ideas_user_template,
    task_description,
    obs_dim,
    action_dim,
    obs_description,
    input_dict_string,
    code_feedback=None,
    llm_transcript_path=None,
    transcript_counter=None,
):
    """One LLM call returning ``n`` parsed design hypotheses (no policy code).

    Returns
    -------
    ideas : list[str]
        Length ``n``.
    raw_response : str
        Assistant message content.
    llm_calls : int
        Number of chat completion calls (1 or 2 if a parse retry was needed).
    """
    ideas_user = ideas_user_template.format(
        task=task_description,
        obs_dim=obs_dim,
        action_dim=action_dim,
        obs_description=obs_description,
        input_dict_string=input_dict_string,
        n=n,
    )
    user_parts = [ideas_user]
    if code_feedback:
        user_parts.append(
            "\n\nPerformance context from the previous generation "
            "(for brainstorming only; do not write code):\n"
            + code_feedback
        )
    messages = [
        {"role": "system", "content": ideas_system},
        {"role": "user", "content": "\n".join(user_parts)},
    ]
    responses, _, _, _ = _call_llm(
        client,
        1,
        args.model,
        messages,
        1.0,
        transcript_path=llm_transcript_path,
        transcript_counter=transcript_counter,
        transcript_label="policy_ideas",
    )
    raw = responses[0].message.content
    ideas = _parse_ideas_response(raw, n)
    llm_calls = 1
    non_trivial = sum(1 for s in ideas if len(s) > 30)
    if non_trivial < max(1, n // 2):
        responses2, _, _, _ = _call_llm(
            client,
            1,
            args.model,
            messages
            + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        f"Your previous reply did not yield {n} clear distinct hypotheses. "
                        "Reply again with ONLY valid JSON: "
                        '{"ideas": ["...", ...]} with exactly '
                        f"{n} non-empty strings."
                    ),
                },
            ],
            1.0,
            transcript_path=llm_transcript_path,
            transcript_counter=transcript_counter,
            transcript_label="policy_ideas_retry",
        )
        raw = responses2[0].message.content
        ideas = _parse_ideas_response(raw, n)
        llm_calls = 2
    return ideas, raw, llm_calls


def _build_impl_messages(
    initial_system,
    user_content,
    code_output_tip,
    provided_response,
    code_feedback,
    extra_user_suffix="",
):
    """Same conversation shape as single-shot policy generation (optional elite chain)."""
    tip_block = (extra_user_suffix + "\n\n" if extra_user_suffix else "") + code_output_tip
    if provided_response is not None and code_feedback is not None:
        return [
            {"role": "system", "content": initial_system},
            {"role": "user", "content": user_content + "\n" + code_output_tip},
            {"role": "assistant", "content": provided_response},
            {"role": "user", "content": code_feedback + "\n" + tip_block},
        ]
    if code_feedback is not None:
        return [
            {"role": "system", "content": initial_system},
            {
                "role": "user",
                "content": user_content + "\n" + code_feedback + "\n" + tip_block,
            },
        ]
    return [
        {"role": "system", "content": initial_system},
        {"role": "user", "content": user_content + "\n" + tip_block},
    ]


def _get_symbolic_policies_two_phase(
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
    ideas_system,
    ideas_user,
    policy_impl_mode,
    provided_response=None,
    code_feedback=None,
    real_num=5,
    max_total_attempts=50,
    max_repair_per_candidate=3,
    llm_transcript_path=None,
):
    """Ideation LLM call then batched or per-idea implementation calls."""
    policy_pop = []
    code_string_pop = []
    response_list = []
    get_res_try_num = 0
    success_num = 0
    try_num = 1
    total_llm_calls = 0
    transcript_counter = [0]

    user_content = initial_user.format(
        task=task_description,
        obs_dim=obs_dim,
        action_dim=action_dim,
        obs_description=obs_description,
        input_dict_string=input_dict_string,
    )

    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    imports = (
        "import torch\n"
        "import torch.nn as nn\n"
        "import numpy as np\n"
        "from lares.core.symbolic_policy import SymbolicPolicy\n"
    )
    head = "import os\nimport sys\n" f"sys.path.insert(0, {repr(_project_root)})\n"
    test_code_path = os.path.join(_project_root, "tests", "test_generate_policy.py")
    with open(test_code_path, "r") as f:
        test_code = f.read()

    ideas, ideas_raw, idea_calls = get_policy_ideas(
        client=client,
        args=args,
        n=real_num,
        ideas_system=ideas_system,
        ideas_user_template=ideas_user,
        task_description=task_description,
        obs_dim=obs_dim,
        action_dim=action_dim,
        obs_description=obs_description,
        input_dict_string=input_dict_string,
        code_feedback=code_feedback,
        llm_transcript_path=llm_transcript_path,
        transcript_counter=transcript_counter,
    )
    total_llm_calls += idea_calls

    ideas_path = os.path.join(dir_path, f"Iter_{llm_iter}_ideas.json")
    with open(ideas_path, "w", encoding="utf-8") as f:
        json.dump(
            {"ideas": ideas, "raw_response": ideas_raw},
            f,
            ensure_ascii=False,
            indent=2,
        )
    ideas_txt_path = os.path.join(dir_path, f"Iter_{llm_iter}_ideas_response.txt")
    with open(ideas_txt_path, "w", encoding="utf-8") as f:
        f.write(ideas_raw)

    def _validate(code_str):
        return _run_policy_validation_subprocess(
            code_str,
            dir_path,
            get_res_try_num,
            head,
            imports,
            test_code,
            data_pkl_path,
            obs_dim,
            action_dim,
        )

    # --- process one slot: validate + repair; returns (policy, code, response) or None ---
    def _finalize_slot(code_string, response_cur, stdout_str, full_code):
        nonlocal get_res_try_num, success_num
        print(stdout_str)
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
            policy = _try_instantiate_policy(imports, code_string, obs_dim, action_dim)
        except Exception as e:
            print(f"Error instantiating policy after validation: {e}")
            return None
        get_res_try_num += 1
        success_num += 1
        return policy

    def _slot_validate_and_repair(idea_text, initial_response, initial_code):
        nonlocal total_llm_calls, try_num
        response_cur = initial_response
        code_string = initial_code
        ok, stdout_str, full_code = _validate(code_string)
        repair_attempt = 0
        while not ok and repair_attempt < max_repair_per_candidate:
            if total_llm_calls >= max_total_attempts:
                break
            repair_attempt += 1
            print(
                f"Policy validation failed (two_phase repair {repair_attempt}/"
                f"{max_repair_per_candidate}): {stdout_str[:200]}"
            )
            fb = _build_error_feedback(code_string, stdout_str, obs_dim, action_dim)
            repair_messages = [
                {"role": "system", "content": initial_system},
                {
                    "role": "user",
                    "content": user_content
                    + "\n\nImplement this design hypothesis only as `GeneratedPolicy`:\n"
                    + idea_text
                    + "\n\n"
                    + code_output_tip,
                },
                {"role": "assistant", "content": response_cur},
                {"role": "user", "content": fb + "\n" + code_output_tip},
            ]
            repair_responses, _, _, _ = _call_llm(
                client,
                1,
                args.model,
                repair_messages,
                1.0,
                transcript_path=llm_transcript_path,
                transcript_counter=transcript_counter,
                transcript_label="policy_repair_two_phase",
            )
            total_llm_calls += 1
            response_cur = repair_responses[0].message.content
            code_string = _extract_code_string(response_cur)
            print(f"Policy gen try {try_num} (two_phase repair)")
            try_num += 1
            ok, stdout_str, full_code = _validate(code_string)
        if not ok:
            print(f"Policy validation failed after repairs: {stdout_str[:200]}")
            return None
        return _finalize_slot(code_string, response_cur, stdout_str, full_code)

    # --- Batched implementation: one LLM response, n fences ---
    if policy_impl_mode == POLICY_IMPL_BATCHED:
        batch_extra = _format_batched_implementation_request(ideas)
        impl_messages = _build_impl_messages(
            initial_system,
            user_content,
            code_output_tip,
            provided_response,
            code_feedback,
            extra_user_suffix=batch_extra,
        )
        batch_response_text = ""
        codes = []
        max_batch_retries = 3
        for batch_try in range(max_batch_retries):
            if total_llm_calls >= max_total_attempts:
                break
            responses, pt, tc, tt = _call_llm(
                client,
                1,
                args.model,
                impl_messages,
                1.0,
                transcript_path=llm_transcript_path,
                transcript_counter=transcript_counter,
                transcript_label="policy_implement_batched",
            )
            total_llm_calls += 1
            batch_response_text = responses[0].message.content
            print(f"Policy batched impl try {try_num} (round {batch_try + 1})", pt, tc, tt)
            try_num += 1
            codes = _extract_n_policy_codes(batch_response_text, real_num)
            if len(codes) > 0:
                break
            print("Batched response contained no parseable policies; retrying.")

        if len(codes) < real_num:
            print(
                f"Warning: batched parse got {len(codes)}/{real_num} code blocks; "
                "missing slots will use per-idea fill calls if budget allows."
            )

        for slot in range(real_num):
            if success_num >= real_num or total_llm_calls >= max_total_attempts:
                break
            response_for_slot = batch_response_text
            if slot < len(codes):
                code_string = codes[slot]
            else:
                if total_llm_calls >= max_total_attempts:
                    break
                idea_one = ideas[slot]
                fill_messages = [
                    {"role": "system", "content": initial_system},
                    {
                        "role": "user",
                        "content": user_content
                        + "\n\nImplement this design hypothesis only as `GeneratedPolicy`:\n"
                        + idea_one
                        + "\n\n"
                        + code_output_tip,
                    },
                ]
                fill_r, _, _, _ = _call_llm(
                    client,
                    1,
                    args.model,
                    fill_messages,
                    1.0,
                    transcript_path=llm_transcript_path,
                    transcript_counter=transcript_counter,
                    transcript_label=f"policy_implement_fill_{slot}",
                )
                total_llm_calls += 1
                response_for_slot = fill_r[0].message.content
                code_string = _extract_code_string(response_for_slot)
            pol = _slot_validate_and_repair(ideas[slot], response_for_slot, code_string)
            if pol is not None:
                policy_pop.append(pol)
                code_string_pop.append(code_string)
                response_list.append(response_for_slot)

    # --- Per-idea implementation ---
    elif policy_impl_mode == POLICY_IMPL_PER_IDEA:
        for slot in range(real_num):
            if success_num >= real_num or total_llm_calls >= max_total_attempts:
                break
            idea_one = ideas[slot]
            slot_messages = _build_impl_messages(
                initial_system,
                user_content,
                code_output_tip,
                provided_response,
                code_feedback,
                extra_user_suffix=(
                    "\n\nImplement this design hypothesis only as `GeneratedPolicy`:\n" + idea_one
                ),
            )
            slot_done = False
            while not slot_done and total_llm_calls < max_total_attempts:
                responses, pt, tc, tt = _call_llm(
                    client,
                    1,
                    args.model,
                    slot_messages,
                    1.0,
                    transcript_path=llm_transcript_path,
                    transcript_counter=transcript_counter,
                    transcript_label=f"policy_implement_{slot}",
                )
                total_llm_calls += 1
                response_cur = responses[0].message.content
                print(f"Policy per_idea gen try {try_num}", pt, tc, tt)
                try_num += 1
                code_string = _extract_code_string(response_cur)
                pol = _slot_validate_and_repair(idea_one, response_cur, code_string)
                if pol is not None:
                    policy_pop.append(pol)
                    code_string_pop.append(code_string)
                    response_list.append(response_cur)
                    slot_done = True
    else:
        raise ValueError(
            f"Unknown policy_impl_mode {policy_impl_mode!r}; "
            f"use {POLICY_IMPL_BATCHED!r} or {POLICY_IMPL_PER_IDEA!r}"
        )

    if success_num < real_num:
        print(
            f"Warning: only generated {success_num}/{real_num} valid policies "
            f"after {total_llm_calls} LLM calls (limit: {max_total_attempts})"
        )

    return policy_pop, code_string_pop, response_list


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
    llm_transcript_path=None,
    ideas_system=None,
    ideas_user=None,
):
    """Generate and validate symbolic policies via the LLM.

    When a candidate fails validation, the error message and failed code are
    fed back to the LLM so it can self-repair (up to ``max_repair_per_candidate``
    times per candidate).  Falls back to fresh generation if repair fails.

    If ``args.policy_gen_two_phase`` is true, runs ideation (``ideas_system`` /
    ``ideas_user`` prompts) then implementation.  Implementation style is
    ``args.policy_impl_mode`` (default ``"batched"``): one LLM response with
    ``n`` fenced policies, or ``"per_idea"`` for ``n`` separate calls.

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
    if bool(getattr(args, "policy_gen_two_phase", False)):
        raw_impl = getattr(args, "policy_impl_mode", POLICY_IMPL_BATCHED)
        if isinstance(raw_impl, str):
            raw_impl = raw_impl.strip().lower().replace("-", "_")
        if raw_impl in ("per_idea",):
            impl_mode = POLICY_IMPL_PER_IDEA
        else:
            impl_mode = POLICY_IMPL_BATCHED
        if ideas_system is None or ideas_user is None:
            raise ValueError(
                "args.policy_gen_two_phase requires ideas_system and ideas_user "
                "(from load_policy_prompt_assets)."
            )
        return _get_symbolic_policies_two_phase(
            client=client,
            dir_path=dir_path,
            llm_iter=llm_iter,
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
            ideas_system=ideas_system,
            ideas_user=ideas_user,
            policy_impl_mode=impl_mode,
            provided_response=provided_response,
            code_feedback=code_feedback,
            real_num=real_num,
            max_total_attempts=max_total_attempts,
            max_repair_per_candidate=max_repair_per_candidate,
            llm_transcript_path=llm_transcript_path,
        )

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
    transcript_counter = [0]
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _project_root = os.path.dirname(os.path.dirname(_this_dir))

    imports = (
        "import torch\n"
        "import torch.nn as nn\n"
        "import numpy as np\n"
        "from lares.core.symbolic_policy import SymbolicPolicy\n"
    )
    head = "import os\nimport sys\n" f"sys.path.insert(0, {repr(_project_root)})\n"
    test_code_path = os.path.join(_project_root, "tests", "test_generate_policy.py")
    with open(test_code_path, "r") as f:
        test_code = f.read()

    def _validate_subprocess(code_str):
        return _run_policy_validation_subprocess(
            code_str,
            dir_path,
            get_res_try_num,
            head,
            imports,
            test_code,
            data_pkl_path,
            obs_dim,
            action_dim,
        )

    # ---- generation loop ----
    while success_num < real_num and total_llm_calls < max_total_attempts:
        responses, prompt_tokens, total_completion_token, total_token = _call_llm(
            client,
            real_num * 2,
            args.model,
            base_messages,
            1.0,
            transcript_path=llm_transcript_path,
            transcript_counter=transcript_counter,
            transcript_label="policy_generation",
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
                    client,
                    1,
                    args.model,
                    repair_messages,
                    1.0,
                    transcript_path=llm_transcript_path,
                    transcript_counter=transcript_counter,
                    transcript_label="policy_repair",
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
                policy = _try_instantiate_policy(imports, code_string, obs_dim, action_dim)
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
