#!/usr/bin/env python
"""Stage F (``debug_spec.md``, D10): correction-LLM repair against matched-budget arms.

Six arms per replicate (structure x seed), all from one common preprocessing of
the structure; see :mod:`lares.repair.correction`. Screen and probe cases come
from the committed development manifest; the held-out manifest must be named
explicitly, because which fresh pool seed to spend on it is a decision, not a
default. The three LLM arms make live gpt-6.1-sol calls: run only with approval.

Usage (from the project root)::

    # The frozen protocol (config/stage_f/stage_f.yaml): select, build the manifest, commit, run.
    python scripts/stage_f.py select
    python scripts/stage_f.py build-manifest
    python scripts/stage_f.py run --config config/stage_f/stage_f.yaml     # clean worktree; records the SHA

    # Ad hoc runs:
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
CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "stage_f", "stage_f.yaml")
DEFAULT_BUFFER = os.path.join(_PROJECT_ROOT, "logs", "baseline_lock", "demo_push-v2.pkl")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("select", "build-manifest"):
        sub.add_parser(name).add_argument("--config", default=CONFIG_PATH)
    run = sub.add_parser("run")
    run.add_argument("--config", default=None, help="run the frozen protocol; every other run flag is ignored")
    run.add_argument("--env-name", default="push-v2")
    run.add_argument("--structures", nargs="+",
                     help="ids from the frozen bank or lares/fitting/fittable_variants/")
    run.add_argument("--seeds", nargs="+", type=int)
    run.add_argument("--held-out", help="manifest disjoint from the development screen and probe")
    run.add_argument("--arms", nargs="+", default=list(sf.ARMS))
    run.add_argument("--buffer", default=DEFAULT_BUFFER)
    run.add_argument("--screen", type=int, default=10, help="development cases 0..N-1")
    run.add_argument("--probe", type=int, default=10, help="development tail")
    run.add_argument("--fit-budget", type=int, default=1000)
    run.add_argument("--checkpoints", nargs="+", type=int, default=[0, 500, 1000])
    run.add_argument("--max-substitutions", type=int, default=2)
    run.add_argument("--max-llm-calls", type=int, default=4)
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


def run(args, structures=None, extra_meta=None) -> int:
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
                             max_substitutions=args.max_substitutions, max_llm_calls=args.max_llm_calls)

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
    if structures is None:
        if not (args.structures and args.seeds and args.held_out):
            raise SystemExit("an ad hoc run needs --structures, --seeds and --held-out (or use --config)")
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
            "buffer": os.path.relpath(args.buffer, _PROJECT_ROOT), **git_state(), **(extra_meta or {})}
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


# ---------------------------------------------------------------------------
#  The frozen protocol
# ---------------------------------------------------------------------------


def load_config(path):
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def rel(path):
    return os.path.relpath(path, _PROJECT_ROOT)


def evolved_rank_key(record, progress):
    """Promotion-set success (absent ranks lowest), then screen success, then screen progress."""
    expanded = record.evaluation.get("expanded_success_rate")
    screen = record.evaluation.get("success_rate")
    return (1 if expanded is not None else 0, float(expanded) if expanded is not None else -1.0,
            float(screen) if screen is not None else -1.0, float(progress) if progress is not None else -1e9)


def screen_progress(run_dir, record):
    """Mean signed goal progress in the record's saved screen report (the selected state's)."""
    ref = (record.evaluation.get("reports") or {}).get("fitted_screen")
    if not ref:
        return None
    with open(os.path.join(run_dir, ref["path"]), encoding="utf-8") as f:
        dist = (json.load(f).get("rollout") or {}).get("signed_goal_progress")
    return dist.get("mean") if dist else None


def cmd_select(args) -> int:
    from lares.search import STATUS_INVALID, load_records
    from lares.search.generation import sha256_text
    from lares.search.provenance import reconstruct_record

    cfg = load_config(args.config)
    ev = cfg["structures"]["evolved"]
    selection_path = os.path.join(_PROJECT_ROOT, ev["selection_file"])
    manifest_path = os.path.join(_PROJECT_ROOT, cfg["manifests"]["held_out"]["path"])
    if os.path.exists(manifest_path):
        raise SystemExit(f"{rel(manifest_path)} exists: structures must be selected before Stage F cases exist")
    if os.path.exists(selection_path):
        raise SystemExit(f"{rel(selection_path)} exists; the selection is frozen")
    run_dir = os.path.join(_PROJECT_ROOT, ev["source_run"])
    ranked, considered = [], []
    for record in load_records(os.path.join(run_dir, "experiments")):
        if record.status == STATUS_INVALID or record.intervention:
            continue
        outcome = reconstruct_record(record, run_dir)
        entry = {"id": record.experiment_id, "candidate_id": record.candidate_id,
                 "expanded_success_rate": record.evaluation.get("expanded_success_rate"),
                 "screen_success_rate": record.evaluation.get("success_rate"),
                 "reconstructs": bool(outcome["identical"])}
        considered.append(entry)
        if not outcome["identical"]:
            entry["excluded"] = f"reconstruction failed: {outcome['problems']}"
            continue
        with open(os.path.join(run_dir, record.reconstruction["source"]["path"]), encoding="utf-8") as f:
            source = f.read()
        entry["screen_progress"] = screen_progress(run_dir, record)
        entry["rank_key"] = list(evolved_rank_key(record, entry["screen_progress"]))
        ranked.append((entry, source))
    ranked.sort(key=lambda es: tuple(es[0]["rank_key"]), reverse=True)
    selected, seen = [], {}
    for entry, source in ranked:
        digest = sha256_text(source)
        if digest in seen:
            entry["excluded"] = f"identical source to {seen[digest]}"
            continue
        seen[digest] = entry["id"]
        if len(selected) < int(ev["count"]):
            sid = f"baseline61_{entry['id']}"
            path = os.path.join(_PROJECT_ROOT, ev["structures_dir"], f"{sid}.py")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "x", encoding="utf-8") as f:
                f.write(source)
            selected.append({"structure_id": sid, "record": entry["id"], "source_path": rel(path),
                             "source_sha256": digest, "rank_key": entry["rank_key"],
                             "expanded_success_rate": entry["expanded_success_rate"],
                             "screen_success_rate": entry["screen_success_rate"]})
    import stage_e

    payload = {"selected_at": stage_e.now(), "selected_before_held_out_manifest": True, "rule": ev["rule"],
               "source_run": ev["source_run"], "selected": selected, "considered": considered}
    with open(selection_path, "x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  selected {[s['structure_id'] for s in selected]} -> {rel(selection_path)}")
    return 0


def cmd_build_manifest(args) -> int:
    import stage_e
    from lares.eval.manifest import SPLIT_DEVELOPMENT, EvaluationManifest, assert_disjoint, build_manifest

    cfg = load_config(args.config)
    if not os.path.isfile(os.path.join(_PROJECT_ROOT, cfg["structures"]["evolved"]["selection_file"])):
        raise SystemExit("run `select` first: structures are chosen before Stage F cases exist")
    held = cfg["manifests"]["held_out"]
    out_path = os.path.join(_PROJECT_ROOT, held["path"])
    if os.path.exists(out_path):
        raise SystemExit(f"{rel(out_path)} exists and is immutable")
    pool_seed = int(cfg["seeds"]["held_out_pool_seed"])
    replicate_seeds = [int(s) for s in cfg["seeds"]["replicate_seeds"]]
    conflicts = stage_e.seed_conflicts([pool_seed], replicate_seeds)
    if conflicts:
        raise SystemExit("seeds already used:\n  " + "\n  ".join(conflicts))
    manifest, _ = build_manifest(cfg["env_name"], SPLIT_DEVELOPMENT, [pool_seed], horizon=int(cfg["horizon"]),
                                 num_episodes=int(held["episodes"]), manifest_id=held["manifest_id"])
    known = stage_e.unique_manifests(
        [EvaluationManifest.load(p) for p in sorted(glob.glob(os.path.join(_PROJECT_ROOT, "config", "**", "*.yaml"),
                                                               recursive=True)) if _is_manifest(p)])
    assert_disjoint(known + [manifest])
    manifest.save(out_path)
    stamp = stage_e.now()
    stage_e.reserve_seeds(
        [{"seed": pool_seed, "namespace": "mt1_pool", "use": "Stage F held-out manifest", "artifact": rel(out_path),
          "recorded_at": stamp, "backfilled": False}]
        + [{"seed": s, "namespace": "fitting", "use": "Stage F replicate seeds", "artifact": rel(args.config),
            "recorded_at": stamp, "backfilled": False} for s in replicate_seeds])
    provenance = {"manifest_id": manifest.manifest_id, "manifest_path": rel(out_path),
                  "manifest_sha256": stage_e.sha256_file(out_path), "pool_seed": pool_seed,
                  "episodes": len(manifest), "built_at": stamp, "disjoint_from": [m.manifest_id for m in known],
                  "selection_sha256": stage_e.sha256_file(os.path.join(_PROJECT_ROOT,
                                                                       cfg["structures"]["evolved"]["selection_file"]))}
    with open(os.path.join(os.path.dirname(out_path), "manifest_provenance.json"), "x", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    print(f"  manifest {manifest.manifest_id}: {len(manifest)} cases -> {rel(out_path)}; seeds reserved")
    return 0


def _is_manifest(path) -> bool:
    try:
        from lares.eval.manifest import EvaluationManifest

        EvaluationManifest.load(path)
        return True
    except Exception:  # noqa: BLE001 - configs that are not manifests
        return False


class _Spec:
    def __init__(self, structure_id, source, schema):
        from lares.fitting.structure_bank import instantiate_source

        self.structure_id, self.source = structure_id, source
        self.build = lambda _schema=None: instantiate_source(source, 39, 4, schema=schema)


def run_from_config(path) -> int:
    import stage_e
    from lares.core.obs_schema import get_obs_schema

    cfg = load_config(path)
    sha = stage_e.require_clean_worktree()
    schema = get_obs_schema(cfg["env_name"])
    held = cfg["manifests"]["held_out"]
    prov_path = os.path.join(_PROJECT_ROOT, os.path.dirname(held["path"]), "manifest_provenance.json")
    with open(prov_path, encoding="utf-8") as f:
        prov = json.load(f)
    if stage_e.sha256_file(os.path.join(_PROJECT_ROOT, held["path"])) != prov["manifest_sha256"]:
        raise SystemExit(f"{held['path']} no longer matches its frozen hash")
    structures = [_Spec(s.structure_id, s.source, schema) for s in load_structures(cfg["structures"]["bank"], schema)]
    with open(os.path.join(_PROJECT_ROOT, cfg["structures"]["evolved"]["selection_file"]), encoding="utf-8") as f:
        selection = json.load(f)
    for s in selection["selected"]:
        with open(os.path.join(_PROJECT_ROOT, s["source_path"]), encoding="utf-8") as f:
            source = f.read()
        from lares.search.generation import sha256_text

        if sha256_text(source) != s["source_sha256"]:
            raise SystemExit(f"{s['source_path']} no longer matches its frozen hash")
        structures.append(_Spec(s["structure_id"], source, schema))
    m, fit, b, llm = cfg["manifests"], cfg["fitting"], cfg["budget"], cfg["llm"]
    args = SimpleNamespace(
        env_name=cfg["env_name"], arms=list(cfg["arms"]), seeds=list(cfg["seeds"]["replicate_seeds"]),
        held_out=os.path.join(_PROJECT_ROOT, held["path"]), structures=[s.structure_id for s in structures],
        buffer=os.path.join(_PROJECT_ROOT, fit["demo_buffer"]), screen=int(m["screen_episodes"]),
        probe=int(m["probe_episodes"]), fit_budget=int(fit["budget"]), checkpoints=list(fit["checkpoint_steps"]),
        max_substitutions=int(b["max_substitutions"]), max_llm_calls=int(b["max_llm_calls"]),
        max_steps=int(cfg["horizon"]), model=llm["model"], reasoning_effort=llm["reasoning_effort"],
        out_dir=os.path.join(_PROJECT_ROOT, "logs", "stage_f"))
    extra = {"config": rel(path), "config_sha256": stage_e.sha256_file(path), "frozen_git_sha": sha,
             "selection_sha256": prov["selection_sha256"], "held_out_manifest_sha256": prov["manifest_sha256"]}
    return run(args, structures=structures, extra_meta=extra)


def main() -> int:
    args = parse_args()
    if args.command == "select":
        return cmd_select(args)
    if args.command == "build-manifest":
        return cmd_build_manifest(args)
    if args.command == "run":
        return run_from_config(args.config) if args.config else run(args)
    return report(args.run)


if __name__ == "__main__":
    sys.exit(main())
