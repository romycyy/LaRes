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


def generate_one(prompts, contract, run_dir, args, generator="single_phase"):
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
        initial_user=prompts["initial_user"] + (contract if isinstance(contract, str) else
                                                (SEARCHABILITY_CONTRACT if contract else "")),
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


# ---------------------------------------------------------------------------
#  R-019/R-020/R-021: one shared-control suite on pick-place-v2 (advisor consult call_00031)
# ---------------------------------------------------------------------------

#: R-021's treatment, appended to the single-phase prompt. Every line was read from MetaWorld 3 source on 2026-10-22:
#: sawyer_xyz_env.step / set_xyz_action / _get_curr_obs_combined_no_goal and SawyerPickPlaceV3Policy. No braces.
GROUNDING_TABLE = """
Verified interface for this task (read from the simulator source; use it exactly):
- The executed action is tanh(mean) for each of the 4 action dimensions.
- Action 0, 1, 2: end-effector displacement in x, y, z. Each is clipped to [-1, 1] and multiplied by 0.01 m, so 1.0
  moves the end effector 1 cm along that axis in one step.
- Action 3: gripper command. It drives the two fingers symmetrically; positive values close the fingers on the
  object, and 0 or negative values leave them open. The scripted expert sends +1.0 to grasp and 0.0 otherwise.
- obs field gripper: finger separation divided by 0.1 m, clipped to [0, 1]; 1.0 is fully open and it falls as the
  fingers close on the puck.
- obs fields tcp, obj, goal: end-effector, puck and target positions in metres (x, y, z); the target can be above
  the table, so the puck must be lifted and carried, not pushed.
- prev_tcp, prev_gripper, prev_obj: the same quantities one step earlier."""

#: R-020's alternative selector: a probe episode "lifts" when the puck rises at least this far above its start.
LIFT_THRESHOLD_M = 0.02
SUITE_ARMS = {   # arm -> (generator, appendix, block seeds paired by index)
    "control": ("single_phase", "", (3210, 3211, 3212, 3213)),
    "grounded": ("single_phase", GROUNDING_TABLE, (3220, 3221, 3222, 3223)),
    "two_phase": ("two_phase", "", (3200, 3201, 3202, 3205)),
}


def lift_probe(actor_policy, manifest, env, schema, horizon=150):
    """Probe every case once: ``(success rate, lift rate, mean signed progress)``. Object z is obs index 6."""
    from lares.eval.runner import PolicyActor, run_episode

    successes, lifts, progress = [], [], []
    for case in manifest.episodes:
        heights = []
        ep = run_episode(PolicyActor(actor_policy, deterministic=True, name="probe"), env, case, max_steps=horizon,
                         schema=schema, observer=lambda step: heights.append(float(np.asarray(step["obs"])[6])))
        successes.append(float(ep.success))
        lifts.append(float(heights and max(heights) - heights[0] >= LIFT_THRESHOLD_M))
        progress.append(float(ep.signed_goal_progress) if ep.signed_goal_progress is not None else 0.0)
    return float(np.mean(successes)), float(np.mean(lifts)), float(np.mean(progress))


def suite_select(trace, selector: str) -> int:
    """``original``: success, then progress. ``lift``: success, then lift rate, then progress. Ties: earlier index."""
    key = (lambda t: (-t["success"], -t["progress"])) if selector == "original" else \
          (lambda t: (-t["success"], -t["lift"], -t["progress"]))
    return min(range(len(trace)), key=lambda i: (*key(trace[i]), i))


def suite_job(job) -> dict:
    import torch

    torch.set_num_threads(1)
    from lares.core.obs_schema import get_obs_schema
    from lares.core.training_pipeline import ensure_mujoco_headless_gl
    from lares.eval import EvaluationManifest, make_manifest_env
    from lares.eval.runner import PolicyActor, evaluate_manifest
    from lares.fitting.structure_bank import instantiate_source
    from lares.search import rollout_search

    ensure_mujoco_headless_gl()
    schema = get_obs_schema(job["env_name"])
    original = instantiate_source(job["source"], 39, 4, schema=schema)
    original.fixed_action_scale = 0.1
    probe = EvaluationManifest.load(job["probe_manifest"])
    probe_env = make_manifest_env(probe, probe.resolve_pool(), 200)
    cands = rollout_search.independent_candidates(original, job["search_seed"])
    trace = []
    for i, (kind, pol) in enumerate(cands):
        s, l, g = lift_probe(pol, probe, probe_env, schema)
        trace.append({"index": i, "kind": kind, "success": s, "lift": l, "progress": g})
    picks = {sel: suite_select(trace, sel) for sel in job["selectors"]}
    conf = EvaluationManifest.load(job["confirmation_manifest"])
    conf_env = make_manifest_env(conf, conf.resolve_pool(), 200)
    confirmed = {}
    for i in sorted(set(picks.values())):
        r = evaluate_manifest(PolicyActor(cands[i][1], deterministic=True, name=f"cand{i}"), conf, conf_env, max_steps=150)
        confirmed[i] = {e.case_id: float(e.success) for e in r.episodes}
    return {"slot": job["slot"], "trace": trace, "picks": picks, "confirmed": {str(k): v for k, v in confirmed.items()},
            "episodes": {"probe": len(cands) * len(probe), "confirmation": len(confirmed) * len(conf)}}


def suite_summary(slots, results, cases) -> dict:
    """Three block-paired contrasts (t over the four blocks) and placement-level intervals. Invalid slots score 0."""
    from lares.eval.stats import t_interval, win_loss_tie

    zero = {c: 0.0 for c in cases}

    def outcome(arm, block_index, selector="original"):
        slot = next(s for s in slots if s["arm"] == arm and s["block_index"] == block_index)
        r = results.get(slot["slot_id"])
        return r["confirmed"][str(r["picks"][selector])] if r else zero

    rate = lambda d: float(np.mean(list(d.values())))  # noqa: E731
    contrasts = {"R-020_lift_minus_original_selector": (("control", "lift"), ("control", "original")),
                 "R-021_grounded_minus_control": (("grounded", "original"), ("control", "original")),
                 "R-019_single_minus_two_phase": (("control", "original"), ("two_phase", "original"))}
    out = {}
    for name, ((a_arm, a_sel), (b_arm, b_sel)) in contrasts.items():
        blocks = [rate(outcome(a_arm, b, a_sel)) - rate(outcome(b_arm, b, b_sel)) for b in range(4)]
        placement = [float(np.mean([outcome(a_arm, b, a_sel)[c] - outcome(b_arm, b, b_sel)[c] for b in range(4)]))
                     for c in cases]
        out[name] = {"paired_success_difference": float(np.mean(blocks)), "block_level": {"by_block": blocks,
                     **t_interval(blocks), **win_loss_tie(blocks)}, "placement_level": t_interval(placement)}
    out["by_slot"] = {s["slot_id"]: {"arm": s["arm"], "valid": s["valid"],
                                      **({"picks": results[s["slot_id"]]["picks"],
                                          "success": {sel: rate(outcome(s["arm"], s["block_index"], sel))
                                                      for sel in results[s["slot_id"]]["picks"]},
                                          "best_probe": max(t["success"] for t in results[s["slot_id"]]["trace"]),
                                          "probe_lift_max": max(t["lift"] for t in results[s["slot_id"]]["trace"])}
                                         if s["slot_id"] in results else {})} for s in slots}
    return out


def run_pick_place_suite(args) -> int:
    import multiprocessing as mp

    sys.path.insert(0, _SCRIPT_DIR)
    import compare_child_vs_parent as cvp

    os.makedirs(args.out_dir, exist_ok=True)
    entries = [(seed, "generation_block", f"pick-place suite: {arm} block {i}")
               for arm, (_, _, seeds) in SUITE_ARMS.items() for i, seed in enumerate(seeds)]
    entries += [(args.probe_seed, "mt1_pool", "pick-place suite: 5-case selection probe"),
                (args.heldout_seed, "mt1_pool", "pick-place suite: 30-case confirmation manifest")]
    print(f"  pick-place suite: seeds {cvp.ensure_entries_reserved(entries, args.out_dir)}")
    if args.reserve_only:
        return 0
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("the suite needs OPENAI_API_KEY")
    from lares.core.training_pipeline import ensure_mujoco_headless_gl

    ensure_mujoco_headless_gl()
    manifest_dir = os.path.join(args.out_dir, "manifests")
    probe, probe_path = cvp.fresh_manifest(args.env_name, args.probe_seed, 5, 150, manifest_dir, role="probe")
    conf, conf_path = cvp.fresh_manifest(args.env_name, args.heldout_seed, 30, 150, manifest_dir, role="confirmation",
                                         also_disjoint_from=[probe])
    prompts = load_policy_prompt_assets(args.env_name)
    slots_path = os.path.join(args.out_dir, "slots.json")
    if os.path.exists(slots_path):
        with open(slots_path, encoding="utf-8") as f:
            slots = json.load(f)
    else:
        slots = []
        for arm, (generator, appendix, seeds) in SUITE_ARMS.items():
            for i, seed in enumerate(seeds):
                slot_id = f"{arm}_b{i}_s{seed}"
                final, n = generate_one(prompts, appendix, os.path.join(args.out_dir, "generation", slot_id), args,
                                        generator=generator)
                entry = {"slot_id": slot_id, "arm": arm, "block_index": i, "seed": seed, "valid": bool(final.valid),
                         "attempts": n, "stage": (final.validation or {}).get("stage"),
                         "errors": (final.validation or {}).get("errors")}
                if final.valid:
                    path = os.path.join(args.out_dir, "structures", f"{slot_id}.py")
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(final.code)
                    entry["source_path"] = path
                slots.append(entry)
                print(f"  {slot_id}: valid={final.valid} after {n} attempt(s)", flush=True)
        with open(slots_path, "w", encoding="utf-8") as f:
            json.dump(slots, f, indent=2)
    jobs = []
    for s in slots:
        if s["valid"]:
            with open(s["source_path"], encoding="utf-8") as f:
                jobs.append({"slot": s["slot_id"], "source": f.read(), "search_seed": s["seed"], "env_name": args.env_name,
                             "selectors": ["original", "lift"] if s["arm"] == "control" else ["original"],
                             "probe_manifest": os.path.abspath(probe_path), "confirmation_manifest": os.path.abspath(conf_path)})
    with mp.get_context("spawn").Pool(processes=min(8, max(1, len(jobs)))) as pool:
        out = pool.map(suite_job, jobs)
    results = {r["slot"]: r for r in out}
    summary = suite_summary(slots, results, sorted(c.case_id for c in conf.episodes))
    summary["episodes"] = {k: int(sum(r["episodes"][k] for r in out)) for k in ("probe", "confirmation")}
    summary["llm_attempts"] = int(sum(s["attempts"] for s in slots))
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump({"argv": sys.argv, "arms": {k: [v[0], bool(v[1]), list(v[2])] for k, v in SUITE_ARMS.items()},
                   "grounding_table": GROUNDING_TABLE, "lift_threshold_m": LIFT_THRESHOLD_M,
                   "manifests": {"probe": probe.manifest_id, "confirmation": conf.manifest_id}}, f, indent=2)
    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    for sid, row in summary["by_slot"].items():
        print(f"  {sid:<22} {row}")
    for name in ("R-020_lift_minus_original_selector", "R-021_grounded_minus_control", "R-019_single_minus_two_phase"):
        b = summary[name]["block_level"]
        print(f"  {name}: {b['mean']:+.3f} [{b['low']}, {b['high']}] blocks {b['by_block']}")
    print(f"  episodes {summary['episodes']}; LLM attempts {summary['llm_attempts']}")
    return 0


# ---------------------------------------------------------------------------
#  R-022: where does the D-014 recipe (single-phase + 24-candidate search) fail?
# ---------------------------------------------------------------------------

#: task -> (generation block seed, MT1 pool seed). Probe = first 5 placements, confirmation = next 30 (disjoint).
SCAN_TASKS = {
    "reach-v2": (2600, 2612), "door-open-v2": (2601, 2613), "door-close-v2": (2602, 2614),
    "drawer-open-v2": (2603, 2615), "drawer-close-v2": (2604, 2616), "window-open-v2": (2605, 2617),
    "window-close-v2": (2606, 2618), "button-press-v2": (2607, 2619),
}
SCAN_PROBE, SCAN_CONFIRMATION = 5, 30
#: R-024 (advisor call_00040), run in this order: task -> (generation seed, pool seed, schema-audit pool seed).
HARD_TASKS = {
    "assembly-v2": (3300, 3310, 3330), "peg-insert-side-v2": (3301, 3311, 3331), "hammer-v2": (3302, 3312, 3332),
    "shelf-place-v2": (3303, 3313, 3333), "basketball-v2": (3304, 3314, 3334), "soccer-v2": (3305, 3315, 3335),
}
HARD_REPEAT = (3320, 3321)      # generation and pool seeds of the one independent repeat
WEAK_AT_OR_BELOW = 0.80


def audit_schema(task: str, pool_seed: int) -> dict:
    """Reset one placement and take one zero step; compare every checkable schema field with the simulator both times."""
    from lares.core.obs_schema import get_obs_schema
    from lares.eval import make_manifest_env
    from lares.eval.manifest import SPLIT_DEVELOPMENT, build_manifest
    from lares.utils.metaworld_env import _unwrap_through_wrappers

    schema = get_obs_schema(task)
    manifest, pool = build_manifest(task, SPLIT_DEVELOPMENT, [pool_seed], horizon=150, num_episodes=1,
                                    manifest_id=f"{task}:schema_audit:s{pool_seed}")
    env = make_manifest_env(manifest, pool)
    problems = [f"reset: {p}" for p in schema.check_against_env(env, manifest.episodes[0])]
    obs, *_ = env.step(np.zeros(4))
    inner = _unwrap_through_wrappers(getattr(env, "_env", env))
    obs = np.asarray(obs, dtype=np.float64)
    pos = np.asarray(inner._get_pos_objects()).ravel()
    for name, expected in (("tcp", inner.get_endeff_pos()), ("obj", pos[:3]), ("goal", inner._target_pos),
                           *((("obj2", pos[3:6]),) if "obj2" in schema.names else ())):
        f = schema.field(name)
        if not np.allclose(obs[f.start:f.stop], np.asarray(expected, dtype=np.float64), atol=1e-6):
            problems.append(f"step: {name} [{f.start}:{f.stop}] = {obs[f.start:f.stop]}, simulator {expected}")
    env.close()
    return {"task": task, "pool_seed": pool_seed, "fields": list(schema.names), "objects_in_source": int(pos.size // 3),
            "problems": problems, "episodes": 1}


def scan_summary(rows, reference="reach-v2", margin=0.20) -> dict:
    """Per task: confirmation success of the searched winner and the untrained probe score; which tasks fall at
    least ``margin`` below the reference task (an invalid structure scores 0)."""
    ref = rows[reference]["confirmation_success"]
    below = sorted(t for t, r in rows.items() if t != reference and ref - r["confirmation_success"] >= margin)
    worst = min((t for t in rows if t != reference), key=lambda t: (rows[t]["confirmation_success"], t))
    return {"reference": reference, "reference_success": ref, "tasks_at_least_margin_below": below,
            "largest_drop": ref - rows[worst]["confirmation_success"], "weakest_task": worst, "by_task": rows}


def _scan_one(task, gen_seed, pool_seed, args, cvp, label):
    """Generate one single-phase structure for ``task``, search it, confirm the winner. Returns a scan row."""
    from lares.core.training_pipeline import load_policy_prompt_assets as assets

    manifest_dir = os.path.join(args.out_dir, "manifests")
    full, _ = cvp.fresh_manifest(task, pool_seed, SCAN_PROBE + SCAN_CONFIRMATION, 150, manifest_dir, role="scan")
    paths = {}
    for role, part in (("probe", full.head(SCAN_PROBE)), ("confirmation", full.slice(SCAN_PROBE, SCAN_PROBE + SCAN_CONFIRMATION))):
        paths[role] = os.path.join(manifest_dir, f"{task}_{role}_s{pool_seed}.yaml")
        if not os.path.exists(paths[role]):
            part.save(paths[role])
    task_args = SimpleNamespace(**{**vars(args), "env_name": task})
    final, n = generate_one(assets(task), "", os.path.join(args.out_dir, "generation", label), task_args)
    row = {"task": task, "label": label, "generation_seed": gen_seed, "pool_seed": pool_seed, "valid": bool(final.valid),
           "attempts": n, "stage": (final.validation or {}).get("stage"), "errors": (final.validation or {}).get("errors")}
    print(f"  {label}: valid={final.valid} after {n} attempt(s)", flush=True)
    if not final.valid:
        return {**row, "confirmation_success": 0.0, "episodes": {"probe": 0, "confirmation": 0}}
    path = os.path.join(args.out_dir, "structures", f"{label}.py")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(final.code)
    r = suite_job({"slot": label, "source": final.code, "search_seed": gen_seed, "env_name": task, "selectors": ["original"],
                   "probe_manifest": os.path.abspath(paths["probe"]), "confirmation_manifest": os.path.abspath(paths["confirmation"])})
    with open(os.path.join(args.out_dir, f"result_{label}.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2)
    return {**row, "source_path": path, "trace": r["trace"],
            "confirmation_success": float(np.mean(list(r["confirmed"][str(r["picks"]["original"])].values()))),
            "untrained_probe_success": r["trace"][0]["success"],
            "candidates_with_probe_success": sum(t["success"] > 0 for t in r["trace"]),
            "best_probe_success": max(t["success"] for t in r["trace"]), "episodes": r["episodes"]}


def run_hard_scan(args) -> int:
    """R-024: audit every schema, then scan tasks in order; the first weak winner gets one independent repeat."""
    sys.path.insert(0, _SCRIPT_DIR)
    import compare_child_vs_parent as cvp

    os.makedirs(args.out_dir, exist_ok=True)
    entries = []
    for task, (g, q, a) in HARD_TASKS.items():
        entries += [(g, "generation_block", f"R-024 {task}: generation and search seed"),
                    (q, "mt1_pool", f"R-024 {task}: probe and confirmation pool"),
                    (a, "mt1_pool", f"R-024 {task}: schema audit placement")]
    entries += [(HARD_REPEAT[0], "generation_block", "R-024 repeat: generation and search seed"),
                (HARD_REPEAT[1], "mt1_pool", "R-024 repeat: probe and confirmation pool")]
    print(f"  hard scan over {list(HARD_TASKS)}; seeds: {cvp.ensure_entries_reserved(entries, args.out_dir)}")
    if args.reserve_only:
        return 0
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("the hard scan needs OPENAI_API_KEY")
    from lares.core.training_pipeline import ensure_mujoco_headless_gl

    ensure_mujoco_headless_gl()
    audits = [audit_schema(task, a) for task, (_, _, a) in HARD_TASKS.items()]
    with open(os.path.join(args.out_dir, "schema_audit.json"), "w", encoding="utf-8") as f:
        json.dump(audits, f, indent=2, default=str)
    bad = {a["task"]: a["problems"] for a in audits if a["problems"]}
    print(f"  schema audit: {'all fields verified' if not bad else bad}", flush=True)
    if bad:
        raise SystemExit("schema audit failed; nothing generated")
    rows, repeat, stopped_early = [], None, False
    for task, (g, q, _) in HARD_TASKS.items():
        row = _scan_one(task, g, q, args, cvp, task)
        rows.append(row)
        print(f"  {task}: confirmation {row['confirmation_success']:.2f}", flush=True)
        if repeat is None and row["confirmation_success"] <= WEAK_AT_OR_BELOW:
            repeat = _scan_one(task, HARD_REPEAT[0], HARD_REPEAT[1], args, cvp, f"{task}_repeat")
            print(f"  {task} repeat: confirmation {repeat['confirmation_success']:.2f}", flush=True)
            if repeat["confirmation_success"] <= WEAK_AT_OR_BELOW:
                stopped_early = True
                break
    reach = 1.0   # R-022's reach-v2 arm (historical control, D-019)
    summary = {"reference_reach_success": reach, "rows": [{k: v for k, v in r.items() if k != "trace"} for r in rows],
               "repeat": {k: v for k, v in repeat.items() if k != "trace"} if repeat else None, "stopped_early": stopped_early,
               "tasks_at_least_0.20_below_reach": [r["task"] for r in rows if reach - r["confirmation_success"] >= 0.20],
               "min_confirmation_success": min(r["confirmation_success"] for r in rows),
               "episodes": {k: int(sum(r["episodes"][k] for r in rows + ([repeat] if repeat else []))) for k in ("probe", "confirmation")},
               "schema_audit_episodes": len(audits),
               "llm_attempts": int(sum(r["attempts"] for r in rows + ([repeat] if repeat else [])))}
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  below reach by >= 0.20: {summary['tasks_at_least_0.20_below_reach']}; min {summary['min_confirmation_success']:.2f}; "
          f"repeat {summary['repeat'] and summary['repeat']['confirmation_success']}; stopped early {stopped_early}; "
          f"episodes {summary['episodes']}; LLM attempts {summary['llm_attempts']}", flush=True)
    return 0


def run_task_scan(args) -> int:
    import multiprocessing as mp

    sys.path.insert(0, _SCRIPT_DIR)
    import compare_child_vs_parent as cvp
    from lares.eval import EvaluationManifest

    os.makedirs(args.out_dir, exist_ok=True)
    entries = [(g, "generation_block", f"R-022 task scan: {task} generation block") for task, (g, _) in SCAN_TASKS.items()]
    entries += [(q, "mt1_pool", f"R-022 task scan: {task} probe and confirmation pool") for task, (_, q) in SCAN_TASKS.items()]
    print(f"  task scan over {list(SCAN_TASKS)}; seeds: {cvp.ensure_entries_reserved(entries, args.out_dir)}")
    if args.reserve_only:
        return 0
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("the task scan needs OPENAI_API_KEY")
    from lares.core.training_pipeline import ensure_mujoco_headless_gl

    ensure_mujoco_headless_gl()
    manifest_dir = os.path.join(args.out_dir, "manifests")
    paths = {}
    for task, (_, pool) in SCAN_TASKS.items():     # frozen before any generation
        full, _ = cvp.fresh_manifest(task, pool, SCAN_PROBE + SCAN_CONFIRMATION, 150, manifest_dir, role="scan")
        for role, part in (("probe", full.head(SCAN_PROBE)), ("confirmation", full.slice(SCAN_PROBE, SCAN_PROBE + SCAN_CONFIRMATION))):
            path = os.path.join(manifest_dir, f"{task}_{role}_s{pool}.yaml")
            if not os.path.exists(path):
                part.save(path)
            paths[(task, role)] = path
    slots_path = os.path.join(args.out_dir, "slots.json")
    if os.path.exists(slots_path):
        with open(slots_path, encoding="utf-8") as f:
            slots = json.load(f)
    else:
        slots = []
        for task, (block, _) in SCAN_TASKS.items():
            task_args = SimpleNamespace(**{**vars(args), "env_name": task})
            final, n = generate_one(load_policy_prompt_assets(task), "", os.path.join(args.out_dir, "generation", task),
                                    task_args, generator="single_phase")
            entry = {"task": task, "seed": block, "valid": bool(final.valid), "attempts": n,
                     "stage": (final.validation or {}).get("stage"), "errors": (final.validation or {}).get("errors")}
            if final.valid:
                path = os.path.join(args.out_dir, "structures", f"{task}.py")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(final.code)
                entry["source_path"] = path
            slots.append(entry)
            print(f"  {task}: valid={final.valid} after {n} attempt(s)", flush=True)
        with open(slots_path, "w", encoding="utf-8") as f:
            json.dump(slots, f, indent=2)
    jobs = []
    for s in slots:
        if s["valid"]:
            with open(s["source_path"], encoding="utf-8") as f:
                jobs.append({"slot": s["task"], "source": f.read(), "search_seed": s["seed"], "env_name": s["task"],
                             "selectors": ["original"], "probe_manifest": os.path.abspath(paths[(s["task"], "probe")]),
                             "confirmation_manifest": os.path.abspath(paths[(s["task"], "confirmation")])})
    with mp.get_context("spawn").Pool(processes=min(8, max(1, len(jobs)))) as pool:
        out = pool.map(suite_job, jobs)
    results = {r["slot"]: r for r in out}
    rows = {}
    for s in slots:
        r = results.get(s["task"])
        rows[s["task"]] = {
            "valid": s["valid"], "attempts": s["attempts"],
            "confirmation_success": float(np.mean(list(r["confirmed"][str(r["picks"]["original"])].values()))) if r else 0.0,
            "untrained_probe_success": r["trace"][0]["success"] if r else 0.0,
            "candidates_with_probe_success": sum(t["success"] > 0 for t in r["trace"]) if r else 0,
            "best_probe_success": max(t["success"] for t in r["trace"]) if r else 0.0,
        }
    summary = scan_summary(rows)
    summary["episodes"] = {k: int(sum(r["episodes"][k] for r in out)) for k in ("probe", "confirmation")}
    summary["llm_attempts"] = int(sum(s["attempts"] for s in slots))
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    for task, row in rows.items():
        print(f"  {task:<18} {row}")
    print(f"  below reach by >= 0.20: {summary['tasks_at_least_margin_below']}; weakest {summary['weakest_task']} "
          f"(drop {summary['largest_drop']:+.3f}); episodes {summary['episodes']}; LLM attempts {summary['llm_attempts']}")
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
    p.add_argument("--research-mode", choices=sorted(RESEARCH_ARMS) + ["pick-place-suite", "task-scan"], default="")
    p.add_argument("--scan-set", choices=["default", "hard"], default="default")
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
    if args.research_mode == "task-scan":
        return run_hard_scan(args) if args.scan_set == "hard" else run_task_scan(args)
    if args.research_mode == "pick-place-suite":
        return run_pick_place_suite(args)
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
