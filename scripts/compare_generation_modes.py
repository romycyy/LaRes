#!/usr/bin/env python
"""Per-idea versus batched code generation, under a matched budget (FR-5, AC-4).

Both modes ask for the same number of policies from the same ideas prompt. The
question is which one returns more *usable* candidates per token spent, since a
candidate that fails validation costs a population slot.

Batched asks for all N policies in one response, which is cheaper but plausibly
degrades over a long generation. Per-idea makes N separate calls, each with the
full prompt, which costs more input tokens. Token counts come from the LLM
transcript the pipeline already writes, so nothing is estimated.

``--research-mode seedless-searchability`` (R-015) asks a different question: does appending a parameterization
contract (bounded, phase-scaled parameters, smooth gates) to the single-phase generation prompt produce structures
that 24-candidate rollout search can make succeed? Per generation block, one structure per prompt; each is searched
under two search seeds with the same independent sampler as R-016's control, selected on a 10-case probe and
confirmed on 50 fresh cases together with its original parameters.

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
    DEFAULT_LLM_MODEL,
    DEFAULT_REASONING_EFFORT,
    POLICY_IMPL_BATCHED,
    POLICY_IMPL_PER_IDEA,
    REASONING_EFFORTS,
    final_attempts,
    get_symbolic_policies,
    valid_policies,
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
        model=args.model, reasoning_effort=args.reasoning_effort, policy_gen_two_phase=True,
        policy_impl_mode=mode,
    )
    t0 = time.time()
    candidates = get_symbolic_policies(
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
    )
    policies, codes, _ = valid_policies(candidates)
    failures = [c for c in final_attempts(candidates) if not c.valid]
    usage = parse_transcript(transcript)
    valid = len(policies)
    return {
        "mode": mode,
        "replicate": replicate,
        "requested": args.pop_size,
        "valid": valid,
        "valid_rate": valid / args.pop_size,
        "failures": len(failures),
        "failure_stages": [f.validation.get("stage") for f in failures],
        "corrections": sum(1 for c in candidates if c.previous_attempt_id),
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


# ---------------------------------------------------------------------------
#  R-015: a parameterization contract in the seedless generation prompt, judged under rollout search
# ---------------------------------------------------------------------------

#: Appended to the single-phase prompt for the treatment arm only. No braces: the template is ``str.format``-ed.
SEARCHABILITY_CONTRACT = """
Parameterization contract (follow it exactly):
- Write the controller as phases blended by smooth gates: each gate is torch.sigmoid of a sharpness parameter times
  (a radius parameter minus a distance in metres). Declare sharpness with a range of about 5 to 100 and radii with
  ranges of about 0.01 to 0.2 m. Never use hard thresholds, torch.where on distances, or integer phase indices.
- Each phase drives along a unit direction (a difference vector divided by its norm plus 1e-8) scaled by its own
  gain parameter with a range of about 0.1 to 10, so a gain means the same thing at every distance.
- Every parameter belongs to one phase or one gate, has a bounded declared range, and its initial value already
  produces the whole reach-then-push behaviour, so that a random change to one parameter changes behaviour smoothly.
- Use between 6 and 10 parameters."""


#: Arms of each research mode: arm name -> (generator, contract appended?). The first arm is the control.
RESEARCH_ARMS = {
    "seedless-searchability": (("baseline", "single_phase", False), ("contract", "single_phase", True)),
    "phase-searchability": (("two_phase", "two_phase", False), ("single_phase", "single_phase", False)),
}


def generate_one(prompts, contract: bool, run_dir, args, generator="single_phase"):
    """One structure with at most one linked correction. Returns ``(final candidate, attempts)``.

    ``single_phase`` writes code from the plain prompt in one call; ``two_phase`` is the exploration lane's
    generator: one ideas call, then one per-idea implementation."""
    import pickle

    from openai import OpenAI

    os.makedirs(run_dir, exist_ok=True)
    data_pkl = os.path.join(run_dir, "data.pkl")
    with open(data_pkl, "wb") as f:
        pickle.dump([{"obs": np.zeros(args.obs_dim)}], f)
    two_phase = generator == "two_phase"
    candidates = get_symbolic_policies(
        client=OpenAI(api_key=os.environ["OPENAI_API_KEY"]), dir_path=run_dir, llm_iter=0,
        args=SimpleNamespace(model=args.model, reasoning_effort=args.reasoning_effort, policy_gen_two_phase=two_phase,
                             policy_impl_mode=POLICY_IMPL_PER_IDEA),
        obs_dim=args.obs_dim, action_dim=args.action_dim, initial_system=prompts["initial_system"],
        initial_user=prompts["initial_user"] + (SEARCHABILITY_CONTRACT if contract else ""),
        task_description=prompts["task_description"], obs_description=prompts["obs_description"],
        input_dict_string=prompts["input_dict_string"], code_output_tip=prompts["code_output_tip"],
        data_pkl_path=data_pkl, env_name=args.env_name, real_num=1, max_total_attempts=3 if two_phase else 2,
        max_repair_per_candidate=1, llm_transcript_path=os.path.join(run_dir, "transcript.log"),
        ideas_system=prompts["ideas_system"] if two_phase else None, ideas_user=prompts["ideas_user"] if two_phase else None,
    )
    final = final_attempts(candidates)[0]
    return final, len(candidates)


def searchability_job(job) -> dict:
    """Independent 24-candidate rollout search of one structure under one search seed; confirm finalist (and original)."""
    import copy

    import torch

    torch.set_num_threads(1)
    import stage_f
    from lares.core.obs_schema import get_obs_schema
    from lares.core.training_pipeline import ensure_mujoco_headless_gl
    from lares.eval import EvaluationManifest, make_manifest_env
    from lares.eval.runner import PolicyActor, evaluate_manifest
    from lares.fitting.structure_bank import instantiate_source

    ensure_mujoco_headless_gl()
    original = instantiate_source(job["source"], 39, 4, schema=get_obs_schema("push-v2"))
    original.fixed_action_scale = 0.1
    probe = EvaluationManifest.load(job["probe_manifest"])
    probe_env = make_manifest_env(probe, probe.resolve_pool(), 200)
    episodes = {"probe": 0, "confirmation": 0}

    def score(policy):
        result = evaluate_manifest(PolicyActor(policy, deterministic=True, name="probe"), probe, probe_env, max_steps=150)
        episodes["probe"] += len(result.episodes)
        return stage_f.sf.score_of(result)

    shared_rng, control_rng, _ = (np.random.default_rng(x) for x in np.random.SeedSequence(int(job["search_seed"])).spawn(3))
    rows = [(pol, score(pol)) for _, pol in stage_f.search_candidates(original, shared_rng)]
    for kind in stage_f.ADAPTIVE_CONTROL_REST:
        cand = copy.deepcopy(original)
        stage_f.sf._propose(cand, stage_f.sf.ARM_LOCAL_SEARCH if kind == "local" else stage_f.sf.ARM_GLOBAL_SEARCH,
                            control_rng)
        rows.append((cand, score(cand)))
    pick = min(range(len(rows)), key=lambda i: (-rows[i][1].success, -rows[i][1].progress, i))
    confirmation = EvaluationManifest.load(job["confirmation_manifest"])
    conf_env = make_manifest_env(confirmation, confirmation.resolve_pool(), 200)

    def confirm(policy, name):
        result = evaluate_manifest(PolicyActor(policy, deterministic=True, name=name), confirmation, conf_env, max_steps=150)
        episodes["confirmation"] += len(result.episodes)
        return {e.case_id: float(e.success) for e in result.episodes}

    out = {"slot": job["slot"], "search_seed": job["search_seed"], "finalist_index": pick,
           "probe_trace": [r[1].to_dict() for r in rows], "searched": confirm(rows[pick][0], "searched")}
    if job["confirm_original"]:
        out["original"] = out["searched"] if pick == 0 else confirm(original, "original")
    out["episodes"] = episodes
    return out


def searchability_summary(slots, results, search_seeds, arms=("baseline", "contract")) -> dict:
    """Treatment minus control (``arms`` = (control, treatment)) per generation block, mean over both search seeds,
    plus placement-level pairing. An invalid slot scores zero on every case."""
    from lares.eval.stats import t_interval, win_loss_tie

    cases = sorted(next(iter(results.values()))["searched"])
    zero = {c: 0.0 for c in cases}

    def searched(slot_id, seed):
        r = results.get((slot_id, seed))
        return r["searched"] if r else zero

    def original(slot_id):
        r = results.get((slot_id, search_seeds[0]))
        return r["original"] if r else zero

    rate = lambda d: float(np.mean(list(d.values())))  # noqa: E731
    blocks = sorted({s["block"] for s in slots})
    by = {(s["block"], s["arm"]): s["slot_id"] for s in slots}
    block_diffs, per_case = {}, {c: [] for c in cases}
    for b in blocks:
        t, c = by[(b, arms[1])], by[(b, arms[0])]
        block_diffs[b] = float(np.mean([rate(searched(t, k)) - rate(searched(c, k)) for k in search_seeds]))
        for x in cases:
            per_case[x].append(float(np.mean([searched(t, k)[x] - searched(c, k)[x] for k in search_seeds])))
    placement = [float(np.mean(v)) for v in per_case.values()]
    step0 = [rate(original(by[(b, arms[1])])) - rate(original(by[(b, arms[0])])) for b in blocks]
    return {"paired_success_difference": float(np.mean(list(block_diffs.values()))),
            "block_level": {"by_block": block_diffs, **t_interval(list(block_diffs.values()))},
            "placement_level": {"n": len(cases), **t_interval(placement), **win_loss_tie(placement)},
            "step0_block_level": {"by_block": dict(zip(blocks, step0)), **t_interval(step0)},
            "by_slot": {s["slot_id"]: {"block": s["block"], "arm": s["arm"], "valid": s["valid"],
                                       "searched": {k: rate(searched(s["slot_id"], k)) for k in search_seeds},
                                       "original": rate(original(s["slot_id"]))} for s in slots}}


def run_seedless_searchability(args) -> int:
    import multiprocessing as mp

    sys.path.insert(0, _SCRIPT_DIR)
    import compare_child_vs_parent as cvp

    os.makedirs(args.out_dir, exist_ok=True)
    arms = RESEARCH_ARMS[args.research_mode]
    tag = args.research_mode
    status = cvp.ensure_entries_reserved(
        [(k, "generation_block", f"{tag}: generation block label") for k in args.generation_seeds]
        + [(k, "fitting", f"{tag}: search RNG seed") for k in args.search_seeds]
        + [(args.probe_seed, "mt1_pool", f"{tag}: 10-case selection probe"),
           (args.heldout_seed, "mt1_pool", f"{tag}: 50-case confirmation manifest")], args.out_dir)
    print(f"  blocks {args.generation_seeds} x {[a[0] for a in arms]}; search seeds {args.search_seeds}; seeds: {status}")
    if args.reserve_only:
        return 0
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("R-015 needs OPENAI_API_KEY")
    from lares.core.training_pipeline import ensure_mujoco_headless_gl

    ensure_mujoco_headless_gl()
    prompts = load_policy_prompt_assets(args.env_name)
    slots_path = os.path.join(args.out_dir, "slots.json")
    if os.path.exists(slots_path):                      # a rerun never calls the LLM again
        with open(slots_path, encoding="utf-8") as f:
            slots = json.load(f)
    else:
        slots, calls = [], 0
        for block in args.generation_seeds:
            for arm, generator, contract in arms:
                slot_id = f"b{block}_{arm}"
                final, n = generate_one(prompts, contract, os.path.join(args.out_dir, "generation", slot_id), args,
                                        generator=generator)
                calls += n
                entry = {"slot_id": slot_id, "block": block, "arm": arm, "valid": bool(final.valid), "attempts": n,
                         "stage": (final.validation or {}).get("stage"), "errors": (final.validation or {}).get("errors")}
                if final.valid:
                    path = os.path.join(args.out_dir, "structures", f"{slot_id}.py")
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(final.code)
                    entry["source_path"] = path
                slots.append(entry)
                print(f"  {slot_id}: valid={final.valid} after {n} call(s)", flush=True)
        with open(slots_path, "w", encoding="utf-8") as f:
            json.dump(slots, f, indent=2)
    probe, probe_path = cvp.fresh_manifest(args.env_name, args.probe_seed, 10, 150, args.out_dir, role="probe")
    conf, conf_path = cvp.fresh_manifest(args.env_name, args.heldout_seed, 50, 150, args.out_dir, role="confirmation",
                                         also_disjoint_from=[probe])
    jobs = []
    for s in slots:
        if not s["valid"]:
            continue
        with open(s["source_path"], encoding="utf-8") as f:
            source = f.read()
        jobs += [{"slot": s["slot_id"], "source": source, "search_seed": k, "confirm_original": i == 0,
                  "probe_manifest": os.path.abspath(probe_path), "confirmation_manifest": os.path.abspath(conf_path)}
                 for i, k in enumerate(args.search_seeds)]
    with mp.get_context("spawn").Pool(processes=min(8, max(1, len(jobs)))) as pool:
        out = pool.map(searchability_job, jobs)
    results = {(r["slot"], r["search_seed"]): r for r in out}
    summary = searchability_summary(slots, results, args.search_seeds, arms=(arms[0][0], arms[1][0]))
    summary["episodes"] = {k: int(sum(r["episodes"][k] for r in out)) for k in ("probe", "confirmation")}
    summary["llm_attempts"] = int(sum(s["attempts"] for s in slots))
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump({"argv": sys.argv, "contract": SEARCHABILITY_CONTRACT, "model": args.model,
                   "reasoning_effort": args.reasoning_effort,
                   "manifests": {"probe": probe.manifest_id, "confirmation": conf.manifest_id}}, f, indent=2)
    for sid, row in summary["by_slot"].items():
        print(f"  {sid:<16} valid={row['valid']} original {row['original']:.2f} searched {row['searched']}")
    bl, pl = summary["block_level"], summary["placement_level"]
    print(f"  {arms[1][0]} - {arms[0][0]}: block {bl['mean']:+.3f} [{bl['low']}, {bl['high']}]; placement {pl['mean']:+.3f} "
          f"[{pl['low']:+.3f}, {pl['high']:+.3f}]; episodes {summary['episodes']}; LLM attempts {summary['llm_attempts']}")
    return 0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env-name", default="push-v2")
    p.add_argument("--model", default=DEFAULT_LLM_MODEL)
    p.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT, choices=REASONING_EFFORTS)
    p.add_argument("--pop-size", type=int, default=4)
    p.add_argument("--replicates", type=int, default=2)
    p.add_argument("--obs-dim", type=int, default=39)
    p.add_argument("--action-dim", type=int, default=4)
    p.add_argument("--max-calls", type=int, default=30)
    p.add_argument(
        "--out-dir", default=os.path.join(_PROJECT_ROOT, "logs", "generation_modes")
    )
    p.add_argument("--research-mode", choices=sorted(RESEARCH_ARMS), default="")
    p.add_argument("--generation-seeds", type=int, nargs="+", default=[])
    p.add_argument("--search-seeds", type=int, nargs="+", default=[])
    p.add_argument("--probe-seed", type=int)
    p.add_argument("--heldout-seed", type=int)
    p.add_argument("--reserve-only", action="store_true")
    return p.parse_args()


def separator(title):
    print(f"\n{'=' * 82}\n  {title}\n{'=' * 82}")


def main():
    args = parse_args()
    if args.research_mode:
        return run_seedless_searchability(args)
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set; this comparison makes live calls.")
        return 1
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, stamp)
    os.makedirs(out_dir, exist_ok=True)
    prompts = load_policy_prompt_assets(args.env_name)

    separator("Per-idea versus batched implementation")
    print(f"  model {args.model} (reasoning effort {args.reasoning_effort}), {args.pop_size} policies per generation, "
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
