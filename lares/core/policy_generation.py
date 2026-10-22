"""
Policy generation pipeline for LLM-generated symbolic policies.

Generates, validates, and instantiates SymbolicPolicy subclasses from LLM
output using prompt templates, subprocess validation, and optional two-phase
ideation/implementation.
"""

import json
import os
import re
import sys
import time
import subprocess
from datetime import datetime

# ---------------------------------------------------------------------------
#  Per-task observation descriptions for MetaWorld V2
# ---------------------------------------------------------------------------

# Task narrative that the machine-checked observation schema cannot carry: what the
# task is, and what the manipulation actually requires. The layout itself comes from
# ``lares.core.obs_schema``, which is verified against the simulator.
TASK_NOTES = {
    "push-v2": (
        "Actions are 4-dimensional: (dx, dy, dz, gripper).\n"
        "The task requires reaching the puck and pushing it along the table to the goal.\n"
        "The gripper stays closed on the puck; this is a planar push, not a pick-and-place.\n"
        "Finite-difference 'tcp' against 'prev_tcp', and 'obj' against 'prev_obj', for velocity."
    ),
    "reach-v2": (
        "Actions are 4-dimensional: (dx, dy, dz, gripper).\n"
        "Move the end-effector to the goal position. No grasping is needed."
    ),
    "pick-place-v2": (
        "Actions are 4-dimensional: (dx, dy, dz, gripper).\n"
        "Reach the object, close the gripper on it, lift, and carry it to the goal."
    ),
    "window-close-v2": (
        "Actions are 4-dimensional: (dx, dy, dz, gripper).\n"
        "'obj' is the window handle. Move the gripper to it and push it closed."
    ),
    "window-open-v2": (
        "Actions are 4-dimensional: (dx, dy, dz, gripper).\n"
        "'obj' is the window handle. Move the gripper to it and pull it open."
    ),
    "button-press-v2": (
        "Actions are 4-dimensional: (dx, dy, dz, gripper).\n"
        "'obj' is the button. Move above it and press downward."
    ),
    "door-close-v2": (
        "Actions are 4-dimensional: (dx, dy, dz, gripper).\n"
        "'obj' is the door handle. Push the door to the closed goal position."
    ),
    "door-open-v2": (
        "Actions are 4-dimensional: (dx, dy, dz, gripper).\n"
        "'obj' is the door handle. Pull the door to the open goal position."
    ),
    "drawer-open-v2": (
        "Actions are 4-dimensional: (dx, dy, dz, gripper).\n"
        "'obj' is the drawer handle. Grasp it and pull the drawer open."
    ),
    "drawer-close-v2": (
        "Actions are 4-dimensional: (dx, dy, dz, gripper).\n"
        "'obj' is the drawer handle. Push the drawer closed."
    ),
}

# Legacy index-based layout text. Superseded by ObsSchema.describe(); kept because
# older artefacts and tests reference it, and it documents what the LLM used to see.
obs_description_dict = {
    "push-v2": """The 39-dimensional observation vector contains:
  obs[0:3]   - End-effector (gripper/TCP) position (x, y, z)
  obs[3]     - Normalized gripper opening (1.0 = fully open)
  obs[4:7]   - Puck position (x, y, z) - the object to push
  obs[7:11]  - Puck quaternion orientation (4 values)
  obs[11:18] - Second object slot (unused for push; always zeros)
  obs[18:36] - Previous timestep's obs[0:18] (so obs[18:21] is the previous
               end-effector position and obs[22:25] the previous puck position;
               finite-difference these against obs[0:3] / obs[4:7] for velocity)
  obs[36:39] - Goal position (x, y, z) - where the puck must end up
Actions are 4-dimensional: (dx, dy, dz, gripper) controlling movement and gripper.
The task requires reaching the puck and pushing it along the table to the goal.
The gripper stays closed on the puck; this is a planar push, not a pick-and-place.""",
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
    "push-v2": """{
  "tcp": "3D position of the robotic arm end-effector",
  "obj": "3D position of the puck to be pushed",
  "target": "3D goal position the puck must be pushed to",
  "actions": "4D action vector (dx, dy, dz, gripper)"}""",
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

# The pipeline's LLM, for every generation call (gpt-5.6-sol from 2026-09-14, gpt-6.1-sol from
# 2026-10-13). gpt-6.1-sol accepts only the
# default temperature (1) and takes `reasoning_effort`; configs name both, and every ledger entry
# records them in `decoding`.
DEFAULT_LLM_MODEL = "gpt-6.1-sol"
DEFAULT_REASONING_EFFORT = "high"
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")


def check_reasoning_effort(value):
    """Return ``value`` if it is a known effort (or ``None``), else raise."""
    if value is None:
        return None
    value = str(value).strip().lower()
    if value not in REASONING_EFFORTS:
        raise ValueError(f"reasoning_effort must be one of {REASONING_EFFORTS}, got {value!r}")
    return value


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
    reasoning_effort=None,
):
    """Call the OpenAI chat completion API with retry logic.

    ``reasoning_effort`` is sent only when set, so models without the parameter still work.
    """
    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
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
                    **extra,
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


# ---------------------------------------------------------------------------
#  Candidate generation with lineage (``debug_spec.md`` Stage A)
# ---------------------------------------------------------------------------
#
# The generator used to return three parallel lists: policies, code strings and
# responses. Everything else a later reader needs, which idea a policy
# implemented, which parents it edited, what feedback it saw, which calls made
# it and whether it was corrected, was discarded on the way out. It now returns
# ``GeneratedCandidate`` objects, one per *attempt*, with the slot and attempt ids
# allocated before any LLM or validation call so a failure keeps its identity.

POLICY_IMPL_BATCHED = "batched"
POLICY_IMPL_PER_IDEA = "per_idea"

#: Imports every generated policy is executed with. Reconstruction uses the same
#: prefix, so a rebuilt policy is the same program that was validated.
GENERATED_IMPORTS = (
    "import torch\n"
    "import torch.nn as nn\n"
    "import numpy as np\n"
    "from lares.core.symbolic_policy import SymbolicPolicy\n"
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PROMPT_DIR = os.path.join(_PROJECT_ROOT, "lares", "utils", "policy_prompts")

STAGE_PASSED = "passed"
STAGE_EXTRACTION = "extraction"
STAGE_HARNESS = "harness"
STAGE_INSTANTIATION = "instantiation"
STAGE_IDEA_MISMATCH = "idea_mismatch"
STAGE_IDEATION = "ideation"
STAGE_PROPOSAL = "proposal"
STAGE_BUDGET = "budget_exhausted"
#: A fittable-lane child that validated but does not differ from its parent exactly as declared.
STAGE_MUTATION_CONTRACT = "mutation_contract"

#: How a single-phase sample was mapped to its slot. Such candidates have no
#: idea, so their lineage is ``legacy_missing`` by construction.
MAPPING_SINGLE_PHASE = "single_phase_sample"


def _read_template(name):
    with open(os.path.join(_PROMPT_DIR, name), "r", encoding="utf-8") as f:
        return f.read()


def _build_error_feedback(code_string, error_output):
    """Build a concise error message to feed back to the LLM for self-repair."""
    error_lines = (error_output or "").strip().split("\n")
    short_error = "\n".join(error_lines[:20])
    return (
        f"Your previous GeneratedPolicy failed validation with this error:\n"
        f"```\n{short_error}\n```\n\n"
        f"The failed code was:\n"
        f"```python\n{code_string}\n```\n\n"
        f"Please fix the code and return the corrected GeneratedPolicy class."
    )


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
    env_name="",
):
    """Write policy code plus harness, run subprocess validation. Returns (ok, stdout, full_code).

    ``env_name`` selects the observation schema the harness binds, so the generated
    source is checked against the layout it claims to read.
    """
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
                # sys.executable, not "python": the venv is usually invoked by
                # absolute path rather than activated, so bare "python" is not
                # on PATH and every candidate fails validation with ENOENT.
                sys.executable,
                "-u",
                temp_file_path,
                data_pkl_path,
                "--obs_dim",
                str(obs_dim),
                "--action_dim",
                str(action_dim),
                "--env_name",
                str(env_name),
            ],
            stdout=f,
            stderr=f,
        )
    process.communicate()

    with open(filter_filepath, "r") as f:
        stdout_str = f.read()
    return "Success!" in stdout_str, stdout_str, full_code


def instantiate_generated_policy(
    code_str, obs_dim, action_dim, env_name="", forbid_scale_parameters=True, imports=GENERATED_IMPORTS
):
    """exec generated code, construct ``GeneratedPolicy`` and validate it in-process.

    Returns ``(policy, ValidationReport)``. Raises only if the source cannot be
    executed or constructed; a policy that constructs but fails a check comes back
    with the report saying why.
    """
    from lares.core.obs_schema import active_schema, get_obs_schema
    from lares.core.policy_validator import validate_policy

    schema = get_obs_schema(env_name) if env_name else None
    namespace = {}
    exec(imports + code_str, namespace)
    if "GeneratedPolicy" not in namespace:
        raise ValueError("source defines no class named GeneratedPolicy")
    with active_schema(schema):
        policy = namespace["GeneratedPolicy"](obs_dim, action_dim)
    report = validate_policy(
        policy,
        obs_dim,
        action_dim,
        source=code_str,
        schema=schema,
        require_schema=bool(env_name),
        forbid_scale_parameters=forbid_scale_parameters,
    )
    return policy, report


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


_DESIGN_LABEL = re.compile(r"^\s*#\s*idea:\s*(\S+)", re.MULTILINE)


def _extract_labelled_policy_codes(response_text, idea_ids):
    """Map fenced policies onto ideas by the ``# idea: <id>`` label each must carry.

    Positional extraction silently shifts every later policy onto the wrong idea
    when one fence is malformed. A label cannot shift. Only when no fence is
    labelled *and* the fence count is exactly right is position trusted, and the
    mapping says so.

    Returns ``({idea_id: code}, mapping_mode, fences_found)``.
    """
    blocks = re.findall(r"```(?:python)?[ \t]*\n?(.*?)```", response_text, re.DOTALL | re.IGNORECASE)
    wanted = set(idea_ids)
    by_id, unlabelled = {}, []
    for raw in blocks:
        label = _DESIGN_LABEL.search(raw)
        code = _extract_code_string(raw.strip())
        if not code or "GeneratedPolicy" not in code:
            continue
        if label is not None:
            if label.group(1) in wanted and label.group(1) not in by_id:
                by_id[label.group(1)] = code
        else:
            unlabelled.append(code)
    mode = "labelled"
    if not by_id and len(unlabelled) == len(idea_ids):
        by_id = dict(zip(idea_ids, unlabelled))
        mode = "positional_exact"
    return by_id, mode, len(blocks)


def _format_batched_implementation_request(items):
    """One request for several designs. ``items`` is ``[(idea_id, brief), ...]``."""
    lines = ["## Designs to implement (one policy each)", ""]
    for idea_id, brief in items:
        lines.append(f"### Design {idea_id}")
        lines.append(brief)
        lines.append("")
    lines.extend(
        [
            "## Required output",
            f"Output exactly {len(items)} separate Markdown code fences of the form ```python ... ```.",
            "The first line inside each fence must be the comment `# idea: <design id>`, naming the "
            "design that fence implements, for example `# idea: " + (items[0][0] if items else "gen0_idea0") + "`.",
            "Each fence must contain exactly one complete class named `GeneratedPolicy` subclassing `SymbolicPolicy`.",
            "Implement each design separately; do not merge designs.",
        ]
    )
    return "\n".join(lines)


def implementation_brief(idea, proposal=None, parent_sources=None, lane_metadata=None):
    """What the implementer is shown for one slot: the proposal, its parents, the idea."""
    from lares.search.schemas import MODE_CROSSOVER, MODE_EXPLORATION, MODE_FITTABLE_MUTATION

    lines = []
    if proposal is not None:
        lines += [
            f"## Mutation proposal {proposal.proposal_id} ({proposal.mode})",
            f"Observation: {proposal.observation}",
            f"Hypothesis: {proposal.hypothesis}",
            f"Suspected component: {proposal.suspected_component}",
            f"Intervention: {proposal.intervention}",
            f"Prediction: {proposal.predicted_metric} will {proposal.predicted_direction}",
            f"Protected behaviours: {'; '.join(proposal.protected_behaviors)}",
            "",
        ]
        for pid in proposal.parent_ids:
            source = (parent_sources or {}).get(pid)
            lines += [
                f"### Parent {pid} source",
                "```python",
                source.strip() if source else "# source unavailable",
                "```",
                "",
            ]
        if proposal.mode == MODE_FITTABLE_MUTATION:
            meta = lane_metadata or {}
            mutation = meta.get("mutation") or {}
            scope = meta.get("mutation_scope") or {}
            listed = lambda key: ", ".join(mutation.get(key) or []) or "none"  # noqa: E731
            lines += [
                f"This is a bounded fittable mutation of the parent above, in scope {scope.get('id')!r}: "
                f"{scope.get('description', '')}",
                f"The one structural change: {mutation.get('structural_change', '')}",
                f"Parameters added: {listed('adds')}; removed: {listed('removes')}; "
                f"changed (initial value or range): {listed('changes')}.",
                "Start from the parent source above and make only this change. Every other expression, and "
                "every other parameter's name, initial value and range, stays exactly as in the parent.",
                "Remove the parent's action-scale parameter (log_std) and its range entry, and return "
                "`self.fixed_std(mean)` as std. Declare no other scale parameter.",
            ]
        elif proposal.mode == MODE_CROSSOVER:
            lines.append("Combine the named components of the parents above into one policy.")
        elif proposal.mode == MODE_EXPLORATION:
            lines.append("This is a new design, not an edit of a parent.")
            if proposal.no_parent_rationale:
                lines.append(f"Why no parent: {proposal.no_parent_rationale}")
        else:
            lines.append(
                "Start from the parent source above. Change only the suspected component; keep "
                "every other expression, and every parameter's name, initial value and range, as "
                "in the parent unless the idea below changes it."
            )
        lines.append("")
    lines.append("## Idea to implement")
    lines.append(idea.implementation_brief())
    return "\n".join(lines)


class _LLMSession:
    """Makes calls against a budget, writing each one to the ledger and transcript."""

    def __init__(self, client, model, ledger, transcript_path, generation, max_calls, temperature=1.0,
                 max_tokens=None, reasoning_effort=None):
        self.client = client
        self.model = model
        self.reasoning_effort = check_reasoning_effort(reasoning_effort)
        self.ledger = ledger
        self.transcript_path = transcript_path
        self.generation = generation
        self.max_calls = int(max_calls)
        self.temperature = float(temperature)
        self.calls_made = 0
        #: Total-token budget. Checked before each call, so the last call can
        #: overshoot it by at most one call's usage.
        self.max_tokens = int(max_tokens) if max_tokens else None
        self.tokens_used = 0
        self.transcript_counter = [0]

    @property
    def exhausted(self):
        if self.max_tokens is not None and self.tokens_used >= self.max_tokens:
            return True
        return self.calls_made >= self.max_calls

    def call(self, label, messages, purpose, attempt_ids, transcript_label, n=1):
        started = time.time()
        responses, prompt_tokens, completion_tokens, total_tokens = _call_llm(
            self.client,
            n,
            self.model,
            messages,
            self.temperature,
            transcript_path=self.transcript_path,
            transcript_counter=self.transcript_counter,
            transcript_label=transcript_label,
            reasoning_effort=self.reasoning_effort,
        )
        self.calls_made += 1
        self.tokens_used += int(total_tokens or 0)
        texts = []
        for choice in responses:
            message = getattr(choice, "message", None)
            texts.append(getattr(message, "content", None) or "")
        record = self.ledger.record(
            label=label,
            model=self.model,
            temperature=self.temperature,
            messages=messages,
            responses=texts,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            wall_time_seconds=time.time() - started,
            decoding={"n": int(n), "temperature": self.temperature, "reasoning_effort": self.reasoning_effort},
            purpose=purpose,
            generation=self.generation,
            attempt_ids=attempt_ids,
        )
        return texts, record


class _SlotValidator:
    """Subprocess harness, in-process validation and the idea contract, in that order."""

    def __init__(self, dir_path, data_pkl_path, obs_dim, action_dim, env_name, forbid_scale_parameters):
        self.dir_path = dir_path
        self.data_pkl_path = data_pkl_path
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.env_name = env_name
        self.forbid_scale_parameters = forbid_scale_parameters
        self.head = "import os\nimport sys\n" f"sys.path.insert(0, {repr(_PROJECT_ROOT)})\n"
        with open(os.path.join(_PROJECT_ROOT, "tests", "test_generate_policy.py"), "r") as f:
            self.test_code = f.read()

    def __call__(self, code, idea, attempt_id, extra_check=None):
        if not code or "GeneratedPolicy" not in code:
            return None, {"passed": False, "stage": STAGE_EXTRACTION,
                          "errors": ["the response contains no class GeneratedPolicy"]}
        ok, stdout_str, _ = _run_policy_validation_subprocess(
            code, self.dir_path, attempt_id, self.head, GENERATED_IMPORTS, self.test_code,
            self.data_pkl_path, self.obs_dim, self.action_dim, env_name=self.env_name,
        )
        if not ok:
            tail = [ln for ln in (stdout_str or "").strip().splitlines() if ln.strip()][-12:]
            return None, {"passed": False, "stage": STAGE_HARNESS, "errors": ["\n".join(tail)]}
        try:
            policy, report = instantiate_generated_policy(
                code, self.obs_dim, self.action_dim, env_name=self.env_name,
                forbid_scale_parameters=self.forbid_scale_parameters,
            )
        except Exception as exc:
            return None, {"passed": False, "stage": STAGE_INSTANTIATION,
                          "errors": [f"{type(exc).__name__}: {exc}"]}
        if not report.ok:
            return None, {"passed": False, "stage": STAGE_INSTANTIATION,
                          "errors": [str(e) for e in report.errors], "validator": report.to_dict()}
        if idea is not None:
            problems = idea.check_against_policy(policy)
            if problems:
                return None, {"passed": False, "stage": STAGE_IDEA_MISMATCH, "errors": problems,
                              "validator": report.to_dict()}
        if extra_check is not None:
            problems = extra_check(policy)
            if problems:
                return None, {"passed": False, "stage": STAGE_MUTATION_CONTRACT, "errors": list(problems),
                              "validator": report.to_dict()}
        return policy, {"passed": True, "stage": STAGE_PASSED, "errors": [], "validator": report.to_dict()}


def _save_attempt(dir_path, candidate):
    attempts_dir = os.path.join(dir_path, "attempts")
    os.makedirs(attempts_dir, exist_ok=True)
    if candidate.code is not None:
        with open(os.path.join(attempts_dir, f"{candidate.attempt_id}.py"), "w", encoding="utf-8") as f:
            f.write(candidate.code)
    if candidate.response is not None:
        with open(os.path.join(attempts_dir, f"{candidate.attempt_id}.response.txt"), "w", encoding="utf-8") as f:
            f.write(candidate.response)


def _design_messages(generation, n, prompts, fmt, feedback_text, context, param_budget):
    from lares.search.schemas import measurable_metrics

    if generation == 0:
        user = prompts["ideas_user"].format(n=n, param_min=param_budget[0], param_max=param_budget[1], **fmt)
    else:
        feedback = (feedback_text or "").rstrip() + "\n\n" + prompts["code_feedback_tmpl"]
        user = prompts["proposals_user"].format(
            n=n,
            param_min=param_budget[0],
            param_max=param_budget[1],
            feedback=feedback,
            eligible_parents=json.dumps(list(context)),
            metrics=", ".join(sorted(measurable_metrics())),
            **fmt,
        )
    return [{"role": "system", "content": prompts["ideas_system"]}, {"role": "user", "content": user}]


def _request_designs(session, generation, n, prompts, fmt, schema, feedback_text, context,
                     attempt_ids, param_budget):
    """Ideas (generation 0) or proposals (later), with at most one retry on bad output.

    Returns ``(slot_parses | None, call_ids, payload_problems, raw_texts)``. A
    payload that cannot be mapped onto slots after the retry fails every slot;
    one malformed entry fails only its own slot.
    """
    from lares.search.generation import parse_ideas_payload, parse_proposals_payload

    kind = "ideas" if generation == 0 else "proposals"
    messages = _design_messages(generation, n, prompts, fmt, feedback_text, context, param_budget)

    def parse(text):
        if generation == 0:
            return parse_ideas_payload(text, n, generation, schema=schema, budget=param_budget)
        return parse_proposals_payload(text, n, generation, context, schema=schema, budget=param_budget)

    return _request_structured(session, messages, parse, n, kind, attempt_ids, f"policy_{kind}")


def _request_structured(session, messages, parse, n, kind, attempt_ids, transcript_label):
    """One structured design request with at most one retry on output that cannot be parsed."""
    from lares.search.generation import LABEL_IDEATION, StructuredOutputError

    call_ids, texts = [], []
    parses, problems = None, [f"call budget exhausted before any {kind} request"]
    for round_index in range(2):
        if session.exhausted:
            break
        request = messages
        if round_index == 1:
            request = messages + [
                {"role": "assistant", "content": texts[-1]},
                {
                    "role": "user",
                    "content": (
                        f"Your previous reply was rejected for these reasons:\n- "
                        + "\n- ".join(problems)
                        + f"\n\nReply again with ONLY the JSON object, with exactly {n} complete "
                        f"entries that satisfy every rule. Nothing will be filled in for you."
                    ),
                },
            ]
        replies, record = session.call(
            LABEL_IDEATION, request, purpose=kind if round_index == 0 else f"{kind}_retry",
            attempt_ids=attempt_ids, transcript_label=transcript_label + ("_retry" if round_index else ""),
        )
        call_ids.append(record.call_id)
        texts.append(replies[0] if replies else "")
        try:
            parses = parse(texts[-1])
            problems = [p for sp in parses for p in sp.problems]
        except StructuredOutputError as exc:
            parses, problems = None, list(exc.problems)
        if parses is not None and not problems:
            break
    if parses is not None:
        return parses, call_ids, [], texts
    return None, call_ids, problems, texts


class _SlotImplementer:
    """One implementation call per parsed slot and at most ``max_repair`` linked corrections."""

    def __init__(self, session, validate, prompts, user_content, allocator, make, parent_sources,
                 max_repair_per_candidate, max_total_calls, max_total_tokens):
        self.session = session
        self.validate = validate
        self.prompts = prompts
        self.user_content = user_content
        self.allocator = allocator
        self.make = make
        self.parent_sources = parent_sources
        self.max_repair = int(max_repair_per_candidate)
        self.max_total_calls = max_total_calls
        self.max_total_tokens = max_total_tokens

    def _brief(self, sp, make_kw):
        return implementation_brief(sp.idea, sp.proposal, self.parent_sources, make_kw.get("lane_metadata"))

    def implement(self, cid, slot, sp, response, code, design_calls, call_ids, mapping, extra_check=None, **make_kw):
        from lares.search.generation import LABEL_CORRECTION, attempt_id_for

        brief = self._brief(sp, make_kw)
        current = self.make(cid, slot, 0, code=code, response=response, idea=sp.idea, proposal=sp.proposal,
                            llm_call_ids=list(design_calls) + list(call_ids), mapping=mapping, **make_kw)
        current.policy, current.validation = self.validate(code, sp.idea, current.attempt_id, extra_check)
        attempts = [current]
        corrections = 0
        while not current.valid and corrections < self.max_repair and not self.session.exhausted:
            corrections += 1
            attempt = self.allocator.next_attempt(cid)
            aid = attempt_id_for(cid, attempt)
            messages = [
                {"role": "system", "content": self.prompts["initial_system"]},
                {"role": "user", "content": self.user_content + "\n\n" + brief + "\n\n" + self.prompts["code_output_tip"]},
                {"role": "assistant", "content": f"```python\n{current.code or ''}\n```"},
                {
                    "role": "user",
                    "content": _build_error_feedback(current.code or "", "\n".join(current.validation.get("errors", [])))
                    + "\n" + self.prompts["code_output_tip"],
                },
            ]
            replies, record = self.session.call(
                LABEL_CORRECTION, messages, purpose="validation_repair", attempt_ids=[aid],
                transcript_label=f"policy_correction_{slot}",
            )
            text = replies[0] if replies else ""
            corrected = self.make(
                cid, slot, attempt, code=_extract_code_string(text), response=text, idea=sp.idea,
                proposal=sp.proposal, llm_call_ids=list(design_calls) + [record.call_id],
                previous_attempt_id=current.attempt_id,
                mapping={"mode": "correction", "call_id": record.call_id, "corrects": current.attempt_id},
                **make_kw,
            )
            corrected.policy, corrected.validation = self.validate(corrected.code, sp.idea, aid, extra_check)
            attempts.append(corrected)
            current = corrected
        return attempts

    def fill(self, cid, slot, sp, reason, design_calls, extra_check=None, **make_kw):
        from lares.search.generation import LABEL_GENERATION, attempt_id_for

        if self.session.exhausted:
            return [self.make(cid, slot, 0, idea=sp.idea, proposal=sp.proposal, llm_call_ids=list(design_calls),
                              validation={"passed": False, "stage": STAGE_BUDGET,
                                          "errors": [f"LLM budget exhausted: {self.session.calls_made}/{self.max_total_calls} calls, "
                                                     f"{self.session.tokens_used}/{self.max_total_tokens or 'unlimited'} tokens ({reason})"]},
                              **make_kw)]
        brief = self._brief(sp, make_kw)
        messages = [
            {"role": "system", "content": self.prompts["initial_system"]},
            {"role": "user", "content": self.user_content + "\n\n" + brief + "\n\n" + self.prompts["code_output_tip"]},
        ]
        purpose = "per_idea" if reason == "per_idea" else "fill"
        replies, record = self.session.call(
            LABEL_GENERATION, messages, purpose=purpose, attempt_ids=[attempt_id_for(cid, 0)],
            transcript_label=f"policy_implement_{slot}",
        )
        text = replies[0] if replies else ""
        mapping = {"mode": purpose, "call_id": record.call_id, "idea_id": sp.idea.idea_id}
        if reason != "per_idea":
            mapping["reason"] = reason
        return self.implement(cid, slot, sp, text, _extract_code_string(text), design_calls, [record.call_id],
                              mapping, extra_check, **make_kw)


def generate_candidates(
    client,
    args,
    dir_path,
    generation,
    obs_dim,
    action_dim,
    env_name,
    prompts,
    n,
    data_pkl_path,
    ledger,
    feedback_bundle=None,
    feedback_text=None,
    parent_sources=None,
    llm_transcript_path=None,
    impl_mode=POLICY_IMPL_PER_IDEA,
    max_total_calls=50,
    max_repair_per_candidate=1,
    param_budget=None,
    forbid_scale_parameters=True,
    max_total_tokens=None,
):
    """Structured two-phase generation: designs, then one implementation per slot.

    Generation 0 asks for ``PolicyIdea`` objects. Generation 1 and later asks for
    a ``MutationProposal`` plus the resulting idea per slot, with semantic parents
    drawn from the feedback bundle's context. Each slot gets one implementation
    attempt and at most ``max_repair_per_candidate`` linked correction attempts;
    there is no unbounded outer retry.

    Returns every attempt as a :class:`~lares.search.generation.GeneratedCandidate`,
    failures included, in slot order then attempt order.
    """
    from lares.core.obs_schema import get_obs_schema
    from lares.search.generation import (
        LABEL_CORRECTION,
        LABEL_GENERATION,
        PARAMETER_BUDGET,
        CandidateAllocator,
        GeneratedCandidate,
        attempt_id_for,
        sha256_text,
    )

    if not env_name:
        raise ValueError("generate_candidates requires env_name to bind the observation schema")
    if impl_mode not in (POLICY_IMPL_BATCHED, POLICY_IMPL_PER_IDEA):
        raise ValueError(
            f"Unknown policy_impl_mode {impl_mode!r}; use {POLICY_IMPL_BATCHED!r} or {POLICY_IMPL_PER_IDEA!r}"
        )
    param_budget = tuple(param_budget or PARAMETER_BUDGET)
    if generation > 0:
        if feedback_bundle is None:
            raise ValueError(
                f"generation {generation} needs a feedback bundle: its proposals must name parents "
                f"from evidence that was actually shown"
            )
        if feedback_text is None or sha256_text(feedback_text) != feedback_bundle.text_sha256:
            raise ValueError(
                "feedback_text does not match the bundle's hashed text; the generator must be shown "
                "exactly the evidence the record says it was shown"
            )
    context = list(feedback_bundle.context_parent_ids) if feedback_bundle is not None else []
    feedback_id = feedback_bundle.feedback_id if feedback_bundle is not None else None
    schema = get_obs_schema(env_name)
    os.makedirs(dir_path, exist_ok=True)

    allocator = CandidateAllocator(generation)
    slots = allocator.allocate_slots(n)
    for cid, _ in slots:
        allocator.next_attempt(cid)
    first_attempts = [attempt_id_for(cid, 0) for cid, _ in slots]

    session = _LLMSession(client, args.model, ledger, llm_transcript_path, generation, max_total_calls,
                          max_tokens=max_total_tokens,
                          reasoning_effort=getattr(args, "reasoning_effort", None))
    validate = _SlotValidator(dir_path, data_pkl_path, obs_dim, action_dim, env_name, forbid_scale_parameters)
    fmt = dict(
        task=prompts["task_description"],
        obs_dim=obs_dim,
        action_dim=action_dim,
        obs_description=prompts["obs_description"],
        input_dict_string=prompts["input_dict_string"],
    )
    user_content = prompts["initial_user"].format(**fmt)

    def make(cid, slot, attempt, **kw):
        return GeneratedCandidate(
            candidate_id=cid, generation=generation, slot=slot, attempt=attempt,
            feedback_id=feedback_id, context_parent_ids=list(context), **kw,
        )

    # -- designs --------------------------------------------------------------
    design_stage = STAGE_IDEATION if generation == 0 else STAGE_PROPOSAL
    parses, design_calls, payload_problems, design_texts = _request_designs(
        session, generation, n, prompts, fmt, schema, feedback_text, context, first_attempts, param_budget
    )
    with open(os.path.join(dir_path, f"Iter_{generation}_designs.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "generation": generation,
                "kind": "ideas" if generation == 0 else "proposals",
                "call_ids": design_calls,
                "payload_problems": payload_problems,
                "slots": [
                    {
                        "slot": sp.slot,
                        "idea": sp.idea.to_dict() if sp.idea else None,
                        "proposal": sp.proposal.to_dict() if sp.proposal else None,
                        "problems": sp.problems,
                    }
                    for sp in (parses or [])
                ],
                "raw_responses": design_texts,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    if parses is None:
        failed = [
            make(cid, slot, 0, llm_call_ids=list(design_calls),
                 validation={"passed": False, "stage": design_stage, "errors": list(payload_problems)})
            for cid, slot in slots
        ]
        for c in failed:
            _save_attempt(dir_path, c)
        return failed

    candidates, ready = [], []
    for (cid, slot), sp in zip(slots, parses):
        if sp.ok:
            ready.append((cid, slot, sp))
        else:
            candidates.append(
                make(cid, slot, 0, idea=sp.idea, proposal=sp.proposal, llm_call_ids=list(design_calls),
                     validation={"passed": False, "stage": design_stage, "errors": list(sp.problems)})
            )

    # -- implementation -------------------------------------------------------
    implementer = _SlotImplementer(session, validate, prompts, user_content, allocator, make, parent_sources,
                                   max_repair_per_candidate, max_total_calls, max_total_tokens)
    if impl_mode == POLICY_IMPL_BATCHED and ready:
        items = [(sp.idea.idea_id, implementation_brief(sp.idea, sp.proposal, parent_sources)) for _, _, sp in ready]
        codes, batch_mode, fences, batch_text, batch_call = {}, "none", 0, "", None
        if not session.exhausted:
            messages = [
                {"role": "system", "content": prompts["initial_system"]},
                {"role": "user", "content": user_content + "\n\n" + _format_batched_implementation_request(items)
                 + "\n\n" + prompts["code_output_tip"]},
            ]
            replies, batch_call = session.call(
                LABEL_GENERATION, messages, purpose="batched",
                attempt_ids=[attempt_id_for(cid, 0) for cid, _, _ in ready],
                transcript_label="policy_implement_batched",
            )
            batch_text = replies[0] if replies else ""
            codes, batch_mode, fences = _extract_labelled_policy_codes(batch_text, [i for i, _ in items])
        for cid, slot, sp in ready:
            code = codes.get(sp.idea.idea_id)
            if code is not None:
                mapping = {"mode": "batched", "call_id": batch_call.call_id, "idea_id": sp.idea.idea_id,
                           "match": batch_mode, "fences_found": fences}
                candidates.extend(implementer.implement(cid, slot, sp, batch_text, code, design_calls,
                                                        [batch_call.call_id], mapping))
            else:
                candidates.extend(implementer.fill(cid, slot, sp, "no fence labelled for this idea in the batched reply",
                                                   design_calls))
    else:
        for cid, slot, sp in ready:
            candidates.extend(implementer.fill(cid, slot, sp, "per_idea", design_calls))

    candidates.sort(key=lambda c: (c.slot, c.attempt))
    for c in candidates:
        _save_attempt(dir_path, c)
    return candidates


def generate_two_lane_candidates(
    client,
    args,
    dir_path,
    obs_dim,
    action_dim,
    env_name,
    prompts,
    plan,
    data_pkl_path,
    ledger,
    llm_transcript_path=None,
    impl_mode=POLICY_IMPL_PER_IDEA,
    max_total_calls=50,
    max_repair_per_candidate=1,
    param_budget=None,
    max_total_tokens=None,
):
    """Generation 0 of the two-lane search: seeded fittable mutations and unseeded explorations.

    Slots, attempts, lanes, references and scopes are all allocated from ``plan`` before any call.
    Two design calls follow, one per lane: the fittable lane's request names each slot's fixed
    parent and scope and returns bounded mutations (``parse_fittable_mutations_payload``); the
    exploration lane's request is the ordinary idea request for the remaining slots, with no parent
    and no feedback. Each parsed slot then gets one implementation and at most
    ``max_repair_per_candidate`` linked corrections. A fittable child must also differ from its
    parent exactly as declared (:meth:`TwoLanePlan.child_problems`), or it fails at
    ``mutation_contract``. Returns every attempt, failures included, in slot then attempt order.
    """
    from lares.core.obs_schema import get_obs_schema
    from lares.search.generation import (
        PARAMETER_BUDGET,
        CandidateAllocator,
        GeneratedCandidate,
        attempt_id_for,
        parse_fittable_mutations_payload,
        parse_ideas_payload,
    )
    from lares.search.lanes import slots_prompt_block
    from lares.search.schemas import LANE_FITTABLE, fitting_metrics

    if impl_mode != POLICY_IMPL_PER_IDEA:
        raise ValueError("the two-lane search implements each slot separately; use policy_impl_mode: per_idea")
    if not env_name:
        raise ValueError("generate_two_lane_candidates requires env_name to bind the observation schema")
    param_budget = tuple(param_budget or PARAMETER_BUDGET)
    schema = get_obs_schema(env_name)
    os.makedirs(dir_path, exist_ok=True)

    allocator = CandidateAllocator(0)
    slots = allocator.allocate_slots(plan.n)
    for cid, _ in slots:
        allocator.next_attempt(cid)
    if [slot for _, slot in slots] != [spec.slot for spec in plan.slots]:
        raise ValueError("the plan's slot numbers do not match the allocator's")
    ids = {slot: cid for cid, slot in slots}
    context = plan.reference_ids()
    metadata = {spec.slot: spec.lane_metadata(plan.seed_bank_sha256) for spec in plan.slots}
    lane_of = {spec.slot: spec.lane for spec in plan.slots}

    session = _LLMSession(client, args.model, ledger, llm_transcript_path, 0, max_total_calls,
                          max_tokens=max_total_tokens, reasoning_effort=getattr(args, "reasoning_effort", None))
    validate = _SlotValidator(dir_path, data_pkl_path, obs_dim, action_dim, env_name, forbid_scale_parameters=True)
    fmt = dict(
        task=prompts["task_description"], obs_dim=obs_dim, action_dim=action_dim,
        obs_description=prompts["obs_description"], input_dict_string=prompts["input_dict_string"],
    )
    user_content = prompts["initial_user"].format(**fmt)

    def make(cid, slot, attempt, lane_metadata=None, **kw):
        lane = lane_of[slot]
        return GeneratedCandidate(
            candidate_id=cid, generation=0, slot=slot, attempt=attempt, feedback_id=None,
            context_parent_ids=list(context) if lane == LANE_FITTABLE else [], lane=lane,
            lane_metadata=dict(lane_metadata if lane_metadata is not None else metadata[slot]), **kw,
        )

    fittable, exploration = plan.fittable_slots, plan.exploration_slots
    system = {"role": "system", "content": prompts["ideas_system"]}

    # -- designs, one request per lane -----------------------------------------
    fittable_user = prompts["fittable_mutations_user"].format(
        n=len(fittable), slots=slots_prompt_block(plan, schema),
        fitting_metrics=", ".join(sorted(fitting_metrics())), max_changes=plan.max_parameter_changes,
        min_kept_percent=int(round(100 * plan.min_parent_parameters_kept)),
        param_min=param_budget[0], param_max=param_budget[1], **fmt,
    )
    parse_slots = [spec.as_parse_slot() for spec in fittable]
    f_parses, f_calls, f_problems, f_texts = _request_structured(
        session, [system, {"role": "user", "content": fittable_user}],
        lambda text: parse_fittable_mutations_payload(
            text, parse_slots, 0, schema=schema, budget=param_budget,
            declaration_check=lambda slot, declaration, idea: plan.declaration_problems(slot, declaration, idea, schema),
        ),
        len(fittable), "mutations", [attempt_id_for(ids[spec.slot], 0) for spec in fittable],
        "policy_fittable_mutations",
    )
    # A plan with no exploration slots (a research protocol, never the two-lane search) makes no exploration call.
    # A single-phase exploration lane (R-018) writes its code below, one call per slot, with no design call.
    single_phase_exploration = plan.exploration_mode.get("generator") == "single_phase"
    e_parses, e_calls, e_problems, e_texts = [], [], [], []
    if exploration and not single_phase_exploration:
        exploration_user = prompts["ideas_user"].format(
            n=len(exploration), param_min=param_budget[0], param_max=param_budget[1], **fmt
        )
        e_parses, e_calls, e_problems, e_texts = _request_structured(
            session, [system, {"role": "user", "content": exploration_user}],
            lambda text: parse_ideas_payload(text, len(exploration), 0, schema=schema, budget=param_budget,
                                             slot_offset=exploration[0].slot),
            len(exploration), "ideas", [attempt_id_for(ids[spec.slot], 0) for spec in exploration], "policy_ideas",
        )

    lanes = (
        (fittable, f_parses, f_calls, f_problems, f_texts, STAGE_PROPOSAL),
        (exploration if not single_phase_exploration else [], e_parses, e_calls, e_problems, e_texts, STAGE_IDEATION),
    )
    with open(os.path.join(dir_path, "Iter_0_designs.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "generation": 0,
                "kind": "two_lane",
                "plan": plan.to_dict(),
                "lanes": {
                    specs[0].lane: {
                        "call_ids": calls,
                        "payload_problems": problems,
                        "slots": [
                            {"slot": sp.slot, "idea": sp.idea.to_dict() if sp.idea else None,
                             "proposal": sp.proposal.to_dict() if sp.proposal else None,
                             "mutation": sp.mutation, "problems": sp.problems}
                            for sp in (parses or [])
                        ],
                        "raw_responses": texts,
                    }
                    for specs, parses, calls, problems, texts, _ in lanes if specs
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    candidates, ready = [], []
    for specs, parses, calls, problems, _, stage in lanes:
        if parses is None:
            candidates.extend(
                make(ids[spec.slot], spec.slot, 0, llm_call_ids=list(calls),
                     validation={"passed": False, "stage": stage, "errors": list(problems)})
                for spec in specs
            )
            continue
        for spec, sp in zip(specs, parses):
            meta = spec.lane_metadata(plan.seed_bank_sha256, mutation=sp.mutation) if spec.lane == LANE_FITTABLE else None
            if sp.ok:
                ready.append((spec, sp, calls, meta))
            else:
                candidates.append(make(ids[spec.slot], spec.slot, 0, lane_metadata=meta, idea=sp.idea,
                                       proposal=sp.proposal, llm_call_ids=list(calls),
                                       validation={"passed": False, "stage": stage, "errors": list(sp.problems)}))

    # -- implementation -------------------------------------------------------
    implementer = _SlotImplementer(session, validate, prompts, user_content, allocator, make, plan.parent_sources(),
                                   max_repair_per_candidate, max_total_calls, max_total_tokens)
    for spec, sp, calls, meta in ready:
        extra = None
        if spec.lane == LANE_FITTABLE:
            def extra(policy, _slot=spec.slot, _declaration=sp.mutation):
                return plan.child_problems(_slot, policy, _declaration, schema)
        candidates.extend(implementer.fill(ids[spec.slot], spec.slot, sp, "per_idea", calls, extra_check=extra,
                                           lane_metadata=meta))

    if single_phase_exploration:
        candidates.extend(_single_phase_exploration(
            session, prompts, user_content, allocator, make, exploration, ids, dir_path, data_pkl_path, obs_dim,
            action_dim, env_name, max_repair_per_candidate))

    candidates.sort(key=lambda c: (c.slot, c.attempt))
    for c in candidates:
        _save_attempt(dir_path, c)
    return candidates


def _single_phase_exploration(session, prompts, user_content, allocator, make, specs, ids, dir_path, data_pkl_path,
                              obs_dim, action_dim, env_name, max_repair_per_candidate) -> list:
    """R-018's exploration lane: one single-phase code request per slot and at most one linked correction.

    The generator states no design, so a valid policy gets a code-derived idea (:func:`idea_from_code`). Scale
    parameters are allowed here, as in the protocol D-014 tested: this lane is never fitted, and actions are
    deterministic, so a scale parameter cannot change what is executed.
    """
    from lares.search.generation import LABEL_CORRECTION, LABEL_GENERATION, attempt_id_for
    from lares.search.schemas import idea_from_code

    validate = _SlotValidator(dir_path, data_pkl_path, obs_dim, action_dim, env_name, forbid_scale_parameters=False)
    base_user = user_content + "\n" + prompts["code_output_tip"]
    messages = [{"role": "system", "content": prompts["initial_system"]}, {"role": "user", "content": base_user}]
    out = []

    def attach(candidate):
        candidate.policy, candidate.validation = validate(candidate.code, None, candidate.attempt_id)
        if candidate.valid:
            candidate.idea = idea_from_code(candidate.policy, candidate.code, f"gen0_idea{candidate.slot}",
                                            f"single-phase exploration {candidate.slot}")
        return candidate

    for spec in specs:
        cid = ids[spec.slot]
        if session.exhausted:
            out.append(make(cid, spec.slot, 0, validation={"passed": False, "stage": STAGE_BUDGET,
                                                           "errors": ["the LLM call budget is spent"]}))
            continue
        replies, record = session.call(LABEL_GENERATION, messages, purpose="single_phase_exploration",
                                       attempt_ids=[attempt_id_for(cid, 0)], transcript_label="policy_generation")
        text = replies[0] if replies else ""
        current = attach(make(cid, spec.slot, 0, code=_extract_code_string(text), response=text,
                              llm_call_ids=[record.call_id],
                              mapping={"mode": MAPPING_SINGLE_PHASE, "call_id": record.call_id, "choice_index": 0}))
        out.append(current)
        corrections = 0
        while not current.valid and current.code and corrections < max_repair_per_candidate and not session.exhausted:
            corrections += 1
            attempt = allocator.next_attempt(cid)
            aid = attempt_id_for(cid, attempt)
            repair = messages + [
                {"role": "assistant", "content": f"```python\n{current.code}\n```"},
                {"role": "user", "content": _build_error_feedback(current.code, "\n".join(current.validation.get("errors", [])))
                 + "\n" + prompts["code_output_tip"]},
            ]
            fixes, fix_record = session.call(LABEL_CORRECTION, repair, purpose="validation_repair", attempt_ids=[aid],
                                             transcript_label="policy_repair")
            text = fixes[0] if fixes else ""
            current = attach(make(cid, spec.slot, attempt, code=_extract_code_string(text), response=text,
                                  llm_call_ids=[fix_record.call_id], previous_attempt_id=current.attempt_id,
                                  mapping={"mode": MAPPING_SINGLE_PHASE, "call_id": fix_record.call_id,
                                           "corrects": current.attempt_id}))
            out.append(current)
    return out


class PlateauSlotGenerator:
    """R-037's two structure slots, generated one at a time so the second can depend on the first's outcome.

    Both slot ids and their first attempts are allocated at construction, before any call. Slot 0 is always a
    searched single-phase exploration. Slot 1 is decided by the orchestrator after slot 0 has been searched, screened
    and challenged: :meth:`exploration` again, or :meth:`trace_repair`, one whole-controller edit of a named prefix
    candidate written from expert traces. Every slot gets at most ``max_repair_per_candidate`` linked corrections, and
    every call goes through one session, so the run's call budget covers both slots.
    """

    def __init__(self, client, args, dir_path, obs_dim, action_dim, env_name, prompts, plan, data_pkl_path, ledger,
                 llm_transcript_path=None, max_total_calls=50, max_repair_per_candidate=1, max_total_tokens=None):
        from lares.search.generation import CandidateAllocator

        if not env_name:
            raise ValueError("PlateauSlotGenerator requires env_name to bind the observation schema")
        os.makedirs(dir_path, exist_ok=True)
        self.dir_path, self.obs_dim, self.action_dim, self.env_name = dir_path, obs_dim, action_dim, env_name
        self.prompts, self.plan, self.data_pkl_path = prompts, plan, data_pkl_path
        self.max_repair_per_candidate = int(max_repair_per_candidate)
        self.allocator = CandidateAllocator(0)
        self.ids = {slot: cid for cid, slot in self.allocator.allocate_slots(plan.n)}
        for cid in self.ids.values():
            self.allocator.next_attempt(cid)
        self.session = _LLMSession(client, args.model, ledger, llm_transcript_path, 0, max_total_calls,
                                   max_tokens=max_total_tokens, reasoning_effort=getattr(args, "reasoning_effort", None))
        fmt = dict(task=prompts["task_description"], obs_dim=obs_dim, action_dim=action_dim,
                   obs_description=prompts["obs_description"], input_dict_string=prompts["input_dict_string"])
        self.user_content = prompts["initial_user"].format(**fmt)

    def _make(self, lane, metadata, parents=()):
        from lares.search.generation import GeneratedCandidate

        def make(cid, slot, attempt, lane_metadata=None, **kw):
            return GeneratedCandidate(candidate_id=cid, generation=0, slot=slot, attempt=attempt, feedback_id=None,
                                      context_parent_ids=list(parents), lane=lane,
                                      lane_metadata=dict(lane_metadata if lane_metadata is not None else metadata),
                                      operation_parent_ids=list(parents), **kw)
        return make

    def _finish(self, out):
        for c in out:
            _save_attempt(self.dir_path, c)
        return out

    def exploration(self, slot: int) -> list:
        """A searched single-phase exploration in ``slot``; every attempt, failures included."""
        from lares.search.schemas import LANE_EXPLORATION

        spec = self.plan.slot(slot)
        out = _single_phase_exploration(
            self.session, self.prompts, self.user_content, self.allocator, self._make(LANE_EXPLORATION, spec.lane_metadata()),
            [spec], self.ids, self.dir_path, self.data_pkl_path, self.obs_dim, self.action_dim, self.env_name,
            self.max_repair_per_candidate)
        return self._finish(out)

    def trace_repair(self, slot: int, edited: dict, request: str, lane_metadata: dict) -> list:
        """One whole-controller edit of ``edited`` from ``request`` (parent code and traces), plus linked corrections."""
        from lares.search.generation import LABEL_CORRECTION, LABEL_REPAIR, attempt_id_for
        from lares.search.schemas import LANE_TRACE_REPAIR, idea_from_code

        cid, prompts = self.ids[slot], self.prompts
        make = self._make(LANE_TRACE_REPAIR, lane_metadata, parents=[edited["candidate_id"]])
        validate = _SlotValidator(self.dir_path, self.data_pkl_path, self.obs_dim, self.action_dim, self.env_name,
                                  forbid_scale_parameters=False)

        def attach(candidate):
            candidate.policy, candidate.validation = validate(candidate.code, None, candidate.attempt_id)
            if candidate.valid:
                candidate.idea = idea_from_code(candidate.policy, candidate.code, f"gen0_idea{slot}",
                                                f"trace repair of {edited['candidate_id']}")
                candidate.idea.summary = ("Code-derived idea: a whole-controller rewrite of "
                                          f"{edited['candidate_id']} written from expert traces, without a stated design.")
            return candidate

        if self.session.exhausted:
            return self._finish([make(cid, slot, 0, validation={"passed": False, "stage": STAGE_BUDGET,
                                                                 "errors": ["the LLM call budget is spent"]})])
        user = self.user_content + "\n\n" + request + "\n" + prompts["code_output_tip"]
        messages = [{"role": "system", "content": prompts["initial_system"]}, {"role": "user", "content": user}]
        replies, record = self.session.call(LABEL_REPAIR, messages, purpose="trace_repair",
                                            attempt_ids=[attempt_id_for(cid, 0)], transcript_label="trace_repair")
        text = replies[0] if replies else ""
        current = attach(make(cid, slot, 0, code=_extract_code_string(text), response=text, llm_call_ids=[record.call_id],
                              mapping={"mode": "trace_repair", "call_id": record.call_id, "choice_index": 0}))
        out, corrections = [current], 0
        while (not current.valid and current.code and corrections < self.max_repair_per_candidate
               and not self.session.exhausted):
            corrections += 1
            attempt = self.allocator.next_attempt(cid)
            aid = attempt_id_for(cid, attempt)
            repair = messages + [
                {"role": "assistant", "content": f"```python\n{current.code}\n```"},
                {"role": "user", "content": _build_error_feedback(current.code, "\n".join(current.validation.get("errors", [])))
                 + "\n" + prompts["code_output_tip"]},
            ]
            fixes, fix_record = self.session.call(LABEL_CORRECTION, repair, purpose="validation_repair", attempt_ids=[aid],
                                                  transcript_label="trace_repair_correction")
            text = fixes[0] if fixes else ""
            current = attach(make(cid, slot, attempt, code=_extract_code_string(text), response=text,
                                  llm_call_ids=[fix_record.call_id], previous_attempt_id=current.attempt_id,
                                  mapping={"mode": "trace_repair", "call_id": fix_record.call_id,
                                           "corrects": current.attempt_id}))
            out.append(current)
        return self._finish(out)


def _generate_single_phase(client, args, dir_path, generation, obs_dim, action_dim, env_name, prompts, n,
                           data_pkl_path, ledger, code_feedback, llm_transcript_path, max_total_calls,
                           max_repair_per_candidate):
    """Legacy one-shot generation: ``n`` samples of one request, sample ``k`` to slot ``k``.

    No idea and no proposal exist, so these candidates can never carry complete
    lineage. The orchestrator refuses this path; it remains for scripts and tests.
    """
    from lares.search.generation import (
        LABEL_CORRECTION,
        LABEL_GENERATION,
        CandidateAllocator,
        GeneratedCandidate,
        attempt_id_for,
    )

    allocator = CandidateAllocator(generation)
    slots = allocator.allocate_slots(n)
    for cid, _ in slots:
        allocator.next_attempt(cid)
    session = _LLMSession(client, args.model, ledger, llm_transcript_path, generation, max_total_calls,
                          reasoning_effort=getattr(args, "reasoning_effort", None))
    validate = _SlotValidator(dir_path, data_pkl_path, obs_dim, action_dim, env_name, forbid_scale_parameters=False)
    fmt = dict(
        task=prompts["task_description"], obs_dim=obs_dim, action_dim=action_dim,
        obs_description=prompts["obs_description"], input_dict_string=prompts["input_dict_string"],
    )
    user_content = prompts["initial_user"].format(**fmt)
    base_user = user_content + ("\n" + code_feedback if code_feedback else "") + "\n" + prompts["code_output_tip"]
    messages = [{"role": "system", "content": prompts["initial_system"]}, {"role": "user", "content": base_user}]
    replies, record = session.call(
        LABEL_GENERATION, messages, purpose="single_phase", attempt_ids=[attempt_id_for(c, 0) for c, _ in slots],
        transcript_label="policy_generation", n=n,
    )
    candidates = []
    for cid, slot in slots:
        text = replies[slot] if slot < len(replies) else None
        current = GeneratedCandidate(
            candidate_id=cid, generation=generation, slot=slot, attempt=0,
            code=_extract_code_string(text) if text is not None else None, response=text,
            llm_call_ids=[record.call_id],
            mapping={"mode": MAPPING_SINGLE_PHASE, "call_id": record.call_id, "choice_index": slot},
        )
        current.policy, current.validation = validate(current.code, None, current.attempt_id)
        candidates.append(current)
        corrections = 0
        while not current.valid and current.code and corrections < max_repair_per_candidate and not session.exhausted:
            corrections += 1
            attempt = allocator.next_attempt(cid)
            aid = attempt_id_for(cid, attempt)
            repair_messages = messages + [
                {"role": "assistant", "content": f"```python\n{current.code}\n```"},
                {"role": "user", "content": _build_error_feedback(current.code, "\n".join(current.validation.get("errors", [])))
                 + "\n" + prompts["code_output_tip"]},
            ]
            fixes, fix_record = session.call(
                LABEL_CORRECTION, repair_messages, purpose="validation_repair", attempt_ids=[aid],
                transcript_label="policy_repair",
            )
            text = fixes[0] if fixes else ""
            corrected = GeneratedCandidate(
                candidate_id=cid, generation=generation, slot=slot, attempt=attempt,
                code=_extract_code_string(text), response=text, llm_call_ids=[fix_record.call_id],
                previous_attempt_id=current.attempt_id,
                mapping={"mode": MAPPING_SINGLE_PHASE, "call_id": fix_record.call_id, "corrects": current.attempt_id},
            )
            corrected.policy, corrected.validation = validate(corrected.code, None, aid)
            candidates.append(corrected)
            current = corrected
    for c in candidates:
        _save_attempt(dir_path, c)
    return candidates


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
    env_name="",
    provided_response=None,
    code_feedback=None,
    real_num=5,
    max_total_attempts=50,
    max_repair_per_candidate=1,
    llm_transcript_path=None,
    ideas_system=None,
    ideas_user=None,
    proposals_user=None,
    code_feedback_tmpl=None,
    ledger=None,
    feedback_bundle=None,
    parent_sources=None,
):
    """Generate and validate symbolic policies via the LLM.

    Returns a list of :class:`~lares.search.generation.GeneratedCandidate`, one
    per attempt and failures included. Use :func:`valid_policies` for the policies
    that passed.

    If ``args.policy_gen_two_phase`` is true, runs structured design generation
    then implementation (``args.policy_impl_mode``: ``"per_idea"`` or
    ``"batched"``). ``llm_iter`` is the generation index; generation 1 and later
    need ``feedback_bundle`` and take ``code_feedback`` as its exact text.

    Otherwise runs the legacy single-phase path, whose candidates carry no idea
    and so can never have complete lineage.

    ``max_total_attempts`` caps the number of LLM calls. ``provided_response`` is
    accepted for compatibility and ignored: a bounded edit is now shown its
    parent's source, not the previous transcript.
    """
    from lares.search.generation import LLMCallLedger

    if not env_name:
        raise ValueError(
            "get_symbolic_policies requires env_name: it selects the observation schema "
            "the generated policy reads through. Without it every candidate raises on its "
            "first self.obs_field(...) call and the population comes back empty."
        )
    ledger = ledger if ledger is not None else LLMCallLedger(dir_path)
    prompts = {
        "initial_system": initial_system,
        "initial_user": initial_user,
        "code_output_tip": code_output_tip,
        "task_description": task_description,
        "obs_description": obs_description,
        "input_dict_string": input_dict_string,
    }

    if bool(getattr(args, "policy_gen_two_phase", False)):
        if ideas_system is None or ideas_user is None:
            raise ValueError(
                "args.policy_gen_two_phase requires ideas_system and ideas_user "
                "(from load_policy_prompt_assets)."
            )
        raw_impl = getattr(args, "policy_impl_mode", POLICY_IMPL_BATCHED)
        if isinstance(raw_impl, str):
            raw_impl = raw_impl.strip().lower().replace("-", "_")
        impl_mode = POLICY_IMPL_PER_IDEA if raw_impl == POLICY_IMPL_PER_IDEA else POLICY_IMPL_BATCHED
        prompts.update(
            ideas_system=ideas_system,
            ideas_user=ideas_user,
            proposals_user=proposals_user if proposals_user is not None else _read_template("proposals_user.txt"),
            code_feedback_tmpl=code_feedback_tmpl if code_feedback_tmpl is not None else _read_template("code_feedback.txt"),
        )
        return generate_candidates(
            client, args, dir_path, int(llm_iter), obs_dim, action_dim, env_name, prompts, real_num,
            data_pkl_path, ledger, feedback_bundle=feedback_bundle, feedback_text=code_feedback,
            parent_sources=parent_sources, llm_transcript_path=llm_transcript_path, impl_mode=impl_mode,
            max_total_calls=max_total_attempts, max_repair_per_candidate=max_repair_per_candidate,
        )

    return _generate_single_phase(
        client, args, dir_path, int(llm_iter), obs_dim, action_dim, env_name, prompts, real_num,
        data_pkl_path, ledger, code_feedback, llm_transcript_path, max_total_attempts, max_repair_per_candidate,
    )


def final_attempts(candidates):
    """The last attempt of every slot, in slot order."""
    last = {}
    for c in candidates:
        if c.candidate_id not in last or c.attempt > last[c.candidate_id].attempt:
            last[c.candidate_id] = c
    return [last[k] for k in sorted(last, key=lambda cid: last[cid].slot)]


def valid_policies(candidates):
    """``(policies, codes, responses)`` for slots whose final attempt passed."""
    ok = [c for c in final_attempts(candidates) if c.valid]
    return [c.policy for c in ok], [c.code for c in ok], [c.response for c in ok]
