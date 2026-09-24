#!/usr/bin/env python
"""Stage F (``debug_spec.md``, D10): correction-LLM repair against matched-budget arms.

Six arms per replicate (structure x seed), all from one common preprocessing of
the structure; see :mod:`lares.repair.correction`. Screen and probe cases come
from the committed development manifest; the held-out manifest must be named
explicitly, because which fresh pool seed to spend on it is a decision, not a
default. The three LLM arms make live gpt-5.6-sol calls: run only with approval.

Usage (from the project root)::

    python scripts/stage_f.py run --structures simple_reach_push simple_damped_push \\
        --seeds 0 1 2 --held-out <fresh manifest.yaml>
    python scripts/stage_f.py run ... --arms no_repair local_search global_search   # no LLM calls
    python scripts/stage_f.py report --run logs/stage_f/<run_dir>
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from lares.repair import correction as sf  # noqa: E402

MANIFESTS = os.path.join(_PROJECT_ROOT, "config", "manifests")
DEFAULT_BUFFER = os.path.join(_PROJECT_ROOT, "logs", "baseline_lock", "demo_push-v2.pkl")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--env-name", default="push-v2")
    run.add_argument("--structures", nargs="+", required=True,
                     help="ids from the frozen bank or lares/fitting/fittable_variants/")
    run.add_argument("--seeds", nargs="+", type=int, required=True)
    run.add_argument("--held-out", required=True, help="manifest disjoint from the development screen and probe")
    run.add_argument("--arms", nargs="+", default=list(sf.ARMS))
    run.add_argument("--buffer", default=DEFAULT_BUFFER)
    run.add_argument("--screen", type=int, default=10, help="development cases 0..N-1")
    run.add_argument("--probe", type=int, default=10, help="development tail")
    run.add_argument("--fit-budget", type=int, default=1000)
    run.add_argument("--checkpoints", nargs="+", type=int, default=[0, 500, 1000])
    run.add_argument("--max-substitutions", type=int, default=2)
    run.add_argument("--max-steps", type=int, default=150)
    run.add_argument("--model", default=None, help="defaults to DEFAULT_LLM_MODEL")
    run.add_argument("--reasoning-effort", default=None, help="defaults to DEFAULT_REASONING_EFFORT")
    run.add_argument("--out-dir", default=os.path.join(_PROJECT_ROOT, "logs", "stage_f"))
    report = sub.add_parser("report")
    report.add_argument("--run", required=True)
    return p.parse_args()


def load_structures(ids, schema):
    from lares.fitting.structure_bank import load_bank, load_structure

    found = {s.structure_id: s for s in load_bank()}
    for path in glob.glob(os.path.join(_PROJECT_ROOT, "lares", "fitting", "fittable_variants", "*.py")):
        name = os.path.splitext(os.path.basename(path))[0]
        if name in ids and name not in found:
            found[name] = load_structure(path, schema=schema)
    missing = [i for i in ids if i not in found]
    if missing:
        raise SystemExit(f"unknown structures: {missing}")
    return [found[i] for i in ids]


def git_state() -> dict:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=_PROJECT_ROOT, capture_output=True,
                                text=True).stdout.strip())
    return {"git_sha": sha, "worktree_dirty": dirty}


def run(args) -> int:
    from lares.core.obs_schema import get_obs_schema
    from lares.core.policy_generation import DEFAULT_LLM_MODEL, DEFAULT_REASONING_EFFORT
    from lares.core.training_pipeline import DemoBuffer, ensure_mujoco_headless_gl, load_policy_prompt_assets
    from lares.eval import EvaluationManifest, make_manifest_env
    from lares.fitting import prepare_data

    unknown = [a for a in args.arms if a not in sf.ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {sf.ARMS}")
    ensure_mujoco_headless_gl()
    schema = get_obs_schema(args.env_name)
    development = EvaluationManifest.load(os.path.join(MANIFESTS, f"{args.env_name}_development.yaml"))
    screen, probe = development.head(args.screen), development.tail(args.probe)
    held_out = EvaluationManifest.load(args.held_out)
    sf.check_isolation(screen, probe, held_out)
    budget = sf.StageFBudget(screen_episodes=len(screen), probe_episodes=len(probe), fit_budget=args.fit_budget,
                             checkpoint_steps=tuple(args.checkpoints) if args.fit_budget else (),
                             max_substitutions=args.max_substitutions)

    uses_llm = any(a in sf.LLM_ARM_EVIDENCE for a in args.arms)
    client = None
    llm_args = SimpleNamespace(model=args.model or DEFAULT_LLM_MODEL,
                               reasoning_effort=args.reasoning_effort or DEFAULT_REASONING_EFFORT)
    if uses_llm:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("the LLM arms need OPENAI_API_KEY")
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

    data = prepare_data(DemoBuffer.load(args.buffer), split_seed=0)
    structures = load_structures(args.structures, schema)
    env = make_manifest_env(development, development.resolve_pool(), 200)
    held_out_env = make_manifest_env(held_out, held_out.resolve_pool(), 200)
    prompts = load_policy_prompt_assets(args.env_name)

    run_dir = os.path.join(args.out_dir, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    meta = {"arms": list(args.arms), "structures": list(args.structures), "seeds": list(args.seeds),
            "budget": budget.to_dict(), "acceptance_rule": sf.ACCEPTANCE_RULE,
            "llm": {"model": llm_args.model, "reasoning_effort": llm_args.reasoning_effort} if uses_llm else None,
            "manifests": {"screen": screen.manifest_id, "probe": probe.manifest_id, "held_out": held_out.manifest_id},
            "buffer": os.path.relpath(args.buffer, _PROJECT_ROOT), **git_state()}
    with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Stage F -> {run_dir}\n  cap {budget.repair_episode_cap} repair-evaluation episodes per arm; "
          f"search arms get {budget.search_candidates} candidates")

    replicates = []
    for structure in structures:
        for seed in args.seeds:
            ctx = sf.ArmContext(env=env, env_name=args.env_name, schema=schema, data=data, screen=screen,
                                probe=probe, budget=budget, seed=seed, max_steps=args.max_steps, client=client,
                                llm_args=llm_args, prompts=prompts)
            rep = sf.run_replicate(structure.structure_id, structure.source, lambda s=structure: s.build(schema),
                                   ctx, held_out, os.path.join(run_dir, f"{structure.structure_id}_s{seed}"),
                                   arms=args.arms, held_out_env=held_out_env)
            replicates.append(rep)
            line = "  ".join(f"{r['arm']} {r['held_out']['success']:.2f}" for r in rep["rows"])
            print(f"  {structure.structure_id} seed {seed}: {line}")
    env.close()
    held_out_env.close()
    return report(run_dir)


def report(run_dir: str) -> int:
    replicates = []
    for path in sorted(glob.glob(os.path.join(run_dir, "*", "replicate.json"))):
        with open(path, encoding="utf-8") as f:
            replicates.append(json.load(f))
    if not replicates:
        raise SystemExit(f"no replicate.json under {run_dir}")
    problems = {}
    for rep in replicates:
        rep_dir = os.path.join(run_dir, f"{rep['structure_id']}_s{rep['seed']}")
        for row in rep["rows"]:
            found = sf.verify_chain(row, os.path.join(rep_dir, row["arm"]))
            if found:
                problems[f"{rep['structure_id']}_s{rep['seed']}/{row['arm']}"] = found
    summary = sf.summarise_replicates(replicates)
    summary["chain_problems"] = problems
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'arm':<26}{'held-out':>9}{'repair ep':>10}{'expert ep':>10}{'grad':>7}{'calls':>6}{'tokens':>9}  outcomes")
    for arm, a in summary["arms"].items():
        c = a["mean_cost"]
        print(f"{arm:<26}{a['held_out_success']:>9.3f}{c['repair_episodes'] + c['probe_episodes']:>10.0f}"
              f"{c['expert_assisted_episodes']:>10.0f}{c['bc_gradient_steps']:>7.0f}{c['llm_calls']:>6.1f}"
              f"{c['llm_total_tokens']:>9.0f}  {a['outcomes']}")
    print(f"\nPaired against {summary['reference']} over replicates (t interval):")
    for arm, p in summary["paired"].items():
        interval = (f"[{p['ci95_low']:+.3f}, {p['ci95_high']:+.3f}]" if p["ci95_low"] is not None else "n/a")
        print(f"  {arm:<26}{p['mean_difference']:+.3f}  {interval}  w/l/t {p['wins']}/{p['losses']}/{p['ties']}")
    print(f"\nchain problems: {problems or 'none'}")
    return 1 if problems else 0


def main() -> int:
    args = parse_args()
    return run(args) if args.command == "run" else report(args.run)


if __name__ == "__main__":
    sys.exit(main())
