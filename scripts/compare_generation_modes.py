#!/usr/bin/env python
"""Per-idea versus batched code generation, under a matched budget (FR-5, AC-4).

Both modes ask for the same number of policies from the same ideas prompt. The
question is which one returns more *usable* candidates per token spent, since a
candidate that fails validation costs a population slot.

Batched asks for all N policies in one response, which is cheaper but plausibly
degrades over a long generation. Per-idea makes N separate calls, each with the
full prompt, which costs more input tokens. Token counts come from the LLM
transcript the pipeline already writes, so nothing is estimated.

Usage (from the project root)::

    python scripts/compare_generation_modes.py
    python scripts/compare_generation_modes.py --replicates 3 --pop-size 5
"""

import argparse
import json
import os
import re
import sys
import time
from types import SimpleNamespace

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402

from lares.core.policy_generation import (  # noqa: E402
    POLICY_IMPL_BATCHED,
    POLICY_IMPL_PER_IDEA,
    get_symbolic_policies,
)
from lares.core.training_pipeline import load_policy_prompt_assets  # noqa: E402

_CALL = re.compile(r"^call (\d+) — (\S+)", re.MULTILINE)
_USAGE_MARKER = "--- usage ---"


def _json_objects_after(text, marker):
    """Yield each JSON object following ``marker``, matching braces.

    A non-greedy regex stops at the first closing brace, which for the OpenAI
    usage payload is the nested token-details object, so every block fails to
    parse and the totals silently come out zero.
    """
    index = text.find(marker)
    while index != -1:
        start = text.find("{", index)
        if start == -1:
            return
        depth, i = 0, start
        while i < len(text):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        try:
            yield json.loads(text[start : i + 1])
        except json.JSONDecodeError:
            pass
        index = text.find(marker, i)


def parse_transcript(path):
    """Calls and token counts, read from the transcript rather than estimated."""
    if not os.path.isfile(path):
        return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for usage in _json_objects_after(text, _USAGE_MARKER):
        for key in totals:
            value = usage.get(key)
            if isinstance(value, (int, float)):
                totals[key] += int(value)
    labels = [m.group(2) for m in _CALL.finditer(text)]
    totals["calls"] = len(labels)
    totals["call_labels"] = labels
    return totals


def run_one(mode, replicate, args, prompts, out_dir):
    """One generation in one mode. Returns the row for the comparison table."""
    from openai import OpenAI

    run_dir = os.path.join(out_dir, f"{mode}_r{replicate}")
    os.makedirs(run_dir, exist_ok=True)
    transcript = os.path.join(run_dir, "transcript.log")
    open(transcript, "w").close()

    data_pkl = os.path.join(run_dir, "data.pkl")
    import pickle

    with open(data_pkl, "wb") as f:
        pickle.dump([{"obs": np.zeros(args.obs_dim)}], f)

    llm_args = SimpleNamespace(
        model=args.model, policy_gen_two_phase=True, policy_impl_mode=mode
    )
    failures: list = []
    t0 = time.time()
    policies, codes, _ = get_symbolic_policies(
        client=OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
        dir_path=run_dir,
        llm_iter=replicate,
        args=llm_args,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        initial_system=prompts["initial_system"],
        initial_user=prompts["initial_user"],
        task_description=prompts["task_description"],
        obs_description=prompts["obs_description"],
        input_dict_string=prompts["input_dict_string"],
        code_output_tip=prompts["code_output_tip"],
        data_pkl_path=data_pkl,
        env_name=args.env_name,
        real_num=args.pop_size,
        max_total_attempts=args.max_calls,
        max_repair_per_candidate=1,
        llm_transcript_path=transcript,
        ideas_system=prompts["ideas_system"],
        ideas_user=prompts["ideas_user"],
        failure_log=failures,
    )
    usage = parse_transcript(transcript)
    valid = len(policies)
    return {
        "mode": mode,
        "replicate": replicate,
        "requested": args.pop_size,
        "valid": valid,
        "valid_rate": valid / args.pop_size,
        "failures": len(failures),
        "failure_stages": [f.get("stage") for f in failures],
        "mean_parameters": (
            float(np.mean([p.count_parameters() for p in policies])) if policies else None
        ),
        "mean_code_lines": (
            float(np.mean([len(c.splitlines()) for c in codes])) if codes else None
        ),
        "wall_time_seconds": time.time() - t0,
        **{k: v for k, v in usage.items() if k != "call_labels"},
        "tokens_per_valid": (usage["total_tokens"] / valid) if valid else None,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env-name", default="push-v2")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--pop-size", type=int, default=4)
    p.add_argument("--replicates", type=int, default=2)
    p.add_argument("--obs-dim", type=int, default=39)
    p.add_argument("--action-dim", type=int, default=4)
    p.add_argument("--max-calls", type=int, default=30)
    p.add_argument(
        "--out-dir", default=os.path.join(_PROJECT_ROOT, "logs", "generation_modes")
    )
    return p.parse_args()


def separator(title):
    print(f"\n{'=' * 82}\n  {title}\n{'=' * 82}")


def main():
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set; this comparison makes live calls.")
        return 1
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, stamp)
    os.makedirs(out_dir, exist_ok=True)
    prompts = load_policy_prompt_assets(args.env_name)

    separator("Per-idea versus batched implementation")
    print(f"  model {args.model}, {args.pop_size} policies per generation, "
          f"{args.replicates} replicates per mode")

    rows = []
    for mode in (POLICY_IMPL_BATCHED, POLICY_IMPL_PER_IDEA):
        for replicate in range(args.replicates):
            print(f"\n  --- {mode}, replicate {replicate} ---")
            rows.append(run_one(mode, replicate, args, prompts, out_dir))
            r = rows[-1]
            print(
                f"  {r['valid']}/{r['requested']} valid, {r['calls']} calls, "
                f"{r['total_tokens']} tokens, {r['wall_time_seconds']:.1f}s"
            )

    separator("Results")
    header = (
        f"{'mode':<14}{'valid/asked':>13}{'calls':>7}{'prompt tok':>12}"
        f"{'completion':>12}{'tok/valid':>11}{'seconds':>9}"
    )
    print(header)
    print("-" * len(header))
    summary = {}
    for mode in (POLICY_IMPL_BATCHED, POLICY_IMPL_PER_IDEA):
        group = [r for r in rows if r["mode"] == mode]
        valid = sum(r["valid"] for r in group)
        asked = sum(r["requested"] for r in group)
        prompt_tokens = sum(r["prompt_tokens"] for r in group)
        completion = sum(r["completion_tokens"] for r in group)
        total = sum(r["total_tokens"] for r in group)
        calls = sum(r["calls"] for r in group)
        seconds = sum(r["wall_time_seconds"] for r in group)
        summary[mode] = {
            "valid": valid,
            "requested": asked,
            "valid_rate": valid / asked if asked else 0.0,
            "calls": calls,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion,
            "total_tokens": total,
            "tokens_per_valid": total / valid if valid else None,
            "wall_time_seconds": seconds,
        }
        per_valid = f"{total / valid:>11.0f}" if valid else f"{'n/a':>11}"
        print(
            f"{mode:<14}{f'{valid}/{asked}':>13}{calls:>7}{prompt_tokens:>12}"
            f"{completion:>12}{per_valid}{seconds:>9.1f}"
        )

    separator("Verdict")
    batched, per_idea = summary[POLICY_IMPL_BATCHED], summary[POLICY_IMPL_PER_IDEA]
    print(f"  valid-candidate rate: batched {batched['valid_rate']:.2f}, "
          f"per-idea {per_idea['valid_rate']:.2f}")
    if batched["tokens_per_valid"] and per_idea["tokens_per_valid"]:
        ratio = per_idea["tokens_per_valid"] / batched["tokens_per_valid"]
        print(f"  per-idea costs {ratio:.2f}x the tokens per usable candidate")
    if per_idea["valid_rate"] > batched["valid_rate"]:
        verdict = "per_idea returns more usable candidates"
    elif per_idea["valid_rate"] < batched["valid_rate"]:
        verdict = "batched returns more usable candidates"
    else:
        verdict = "the two modes are indistinguishable on validity at this sample size"
    print(f"  {verdict}")
    print(f"  Sample size is {args.replicates} replicates per mode; treat a small "
          f"difference in validity as noise.")

    payload = {
        "config": vars(args),
        "rows": rows,
        "summary": summary,
        "verdict": verdict,
    }
    path = os.path.join(out_dir, "comparison.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n  report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
