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
import sys
import time
from types import SimpleNamespace

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from lares.repair import correction as sf  # noqa: E402
from lares.utils.gitstate import worktree_state  # noqa: E402

MANIFESTS = os.path.join(_PROJECT_ROOT, "config", "manifests")
CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "stage_f", "stage_f.yaml")
DEFAULT_BUFFER = os.path.join(_PROJECT_ROOT, "logs", "baseline_lock", "demo_push-v2.pkl")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("select", "build-manifest", "reserve", "vlm-smoke"):
        sub.add_parser(name).add_argument("--config", default=CONFIG_PATH)
    run = sub.add_parser("run")
    run.add_argument("--config", default=None, help="run the frozen protocol; every other run flag is ignored")
    run.add_argument("--env-name", default="push-v2")
    run.add_argument("--structures", nargs="+",
                     help="ids from the frozen bank or lares/fitting/fittable_variants/")
    run.add_argument("--seeds", nargs="+", type=int)
    run.add_argument("--held-out", help="manifest disjoint from the development screen and probe")
    run.add_argument("--arms", nargs="+", default=list(sf.ARMS), help=f"subset of {list(sf.ALL_ARMS)}")
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
    svf = sub.add_parser("search-vs-fit", help="R-008: rollout search vs rollout-selected imitation on weak structures")
    svf.add_argument("--config", default=CONFIG_PATH)
    svf.add_argument("--probe-seed", type=int, required=True)
    svf.add_argument("--heldout-seed", type=int, required=True)
    svf.add_argument("--optimizer-seed", type=int, required=True)
    svf.add_argument("--split-seed", type=int, required=True)
    svf.add_argument("--out-dir", required=True)
    svf.add_argument("--reserve-only", action="store_true")
    return p.parse_args()


def load_structures(ids, schema=None):
    from lares.fitting.structure_bank import load_structures as _load

    try:
        return _load(ids)
    except KeyError as exc:
        raise SystemExit(str(exc)) from exc


def git_state() -> dict:
    state = worktree_state(_PROJECT_ROOT)
    return {"git_sha": state["git_sha"], "worktree_dirty": state["dirty"]}


def run(args, structures=None, extra_meta=None) -> int:
    from lares.core.obs_schema import get_obs_schema
    from lares.core.policy_generation import DEFAULT_LLM_MODEL, DEFAULT_REASONING_EFFORT
    from lares.core.training_pipeline import DemoBuffer, ensure_mujoco_headless_gl, load_policy_prompt_assets
    from lares.eval import EvaluationManifest, make_manifest_env
    from lares.fitting import prepare_data

    unknown = [a for a in args.arms if a not in sf.ALL_ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {sf.ALL_ARMS}")
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
    if any(row["arm"] == sf.ARM_LLM_VISION for rep in replicates for row in rep["rows"]):
        summary["vision"] = vision_summary(replicates)
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
    if "vision" in summary:
        v = summary["vision"]
        p = v["paired_vs_llm_numeric"]
        print(f"\nR-006 primary: {sf.ARM_LLM_VISION} - {sf.ARM_LLM_NUMERIC} over {p['n_replicates']} cells: "
              f"{p['mean_difference']:+.3f}  [{p['ci95_low']:+.3f}, {p['ci95_high']:+.3f}]  "
              f"w/l/t {p['wins']}/{p['losses']}/{p['ties']}")
        cb = v["held_out_case_bootstrap"]
        print(f"  held-out case resampling: [{cb['low']:+.3f}, {cb['high']:+.3f}]; packets {v['packets']}")
    print(f"\nchain problems: {problems or 'none'}")
    return 1 if problems else 0


def vision_summary(replicates, resamples: int = 10_000, seed: int = 0) -> dict:
    """R-006: vision minus numeric, paired per cell, plus held-out-case resampling and packet audits.

    The case bootstrap resamples held-out cases (shared by every cell) and recomputes the mean over cells of
    the paired difference; it shows how much the cell-level result leans on particular placements.
    """
    import numpy as np

    rows = {(rep["structure_id"], rep["seed"]): {r["arm"]: r for r in rep["rows"]} for rep in replicates}
    cells = sorted(k for k, arms in rows.items() if sf.ARM_LLM_VISION in arms and sf.ARM_LLM_NUMERIC in arms)
    diffs = [rows[k][sf.ARM_LLM_VISION]["held_out"]["success"] - rows[k][sf.ARM_LLM_NUMERIC]["held_out"]["success"]
             for k in cells]
    cases = sorted(rows[cells[0]][sf.ARM_LLM_VISION]["held_out"]["per_case"]) if cells else []
    per_case = np.array([[rows[k][sf.ARM_LLM_VISION]["held_out"]["per_case"][c]
                          - rows[k][sf.ARM_LLM_NUMERIC]["held_out"]["per_case"][c] for c in cases] for k in cells])
    low = high = None
    if per_case.size:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(cases), size=(resamples, len(cases)))
        means = per_case[:, idx].mean(axis=2).mean(axis=0)
        low, high = float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
    packets = {"cells": len(cells), "with_valid_analysis": 0, "analyses_returned": 0, "analyses_rejected": 0,
               "conflicts": 0, "episodes_rerun": 0}
    for k in cells:
        v = rows[k][sf.ARM_LLM_VISION].get("vision") or {}
        packets["with_valid_analysis"] += int((v.get("analyses_returned") or 0) > 0)
        for key in ("analyses_returned", "analyses_rejected", "conflicts", "episodes_rerun"):
            packets[key] += int(v.get(key) or 0)
    vs_none = [rows[k][sf.ARM_LLM_VISION]["held_out"]["success"] - rows[k][sf.ARM_NO_REPAIR]["held_out"]["success"]
               for k in cells if sf.ARM_NO_REPAIR in rows[k]]
    return {"cells": [list(k) for k in cells], "paired_vs_llm_numeric": sf.replicate_interval(diffs),
            "paired_vs_no_repair": sf.replicate_interval(vs_none) if vs_none else None,
            "held_out_case_bootstrap": {"low": low, "high": high, "resamples": resamples, "seed": seed},
            "packets": packets}


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


def selected_structures(cfg, schema) -> list:
    """The frozen evolved selection, hash-checked."""
    from lares.search.generation import sha256_text

    with open(os.path.join(_PROJECT_ROOT, cfg["structures"]["evolved"]["selection_file"]), encoding="utf-8") as f:
        selection = json.load(f)
    out = []
    for s in selection["selected"]:
        with open(os.path.join(_PROJECT_ROOT, s["source_path"]), encoding="utf-8") as f:
            source = f.read()
        if sha256_text(source) != s["source_sha256"]:
            raise SystemExit(f"{s['source_path']} no longer matches its frozen hash")
        out.append(_Spec(s["structure_id"], source, schema))
    return out


def run_from_config(path) -> int:
    import stage_e
    from lares.core.obs_schema import get_obs_schema

    cfg = load_config(path)
    if "cells" in cfg:
        schema = get_obs_schema(cfg["env_name"])
        bank = [_Spec(s.structure_id, s.source, schema) for s in load_structures(cfg["structures"]["bank"], schema)]
        return run_cells(cfg, path, bank + selected_structures(cfg, schema))
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


# ---------------------------------------------------------------------------
#  Cell protocols (R-006): one independent data/split/probe bundle per (structure, replicate) cell
# ---------------------------------------------------------------------------


def cell_seeds(cfg) -> dict:
    cells = [int(c["seed"]) for c in cfg["cells"]]
    held = int(cfg["seeds"]["held_out_pool_seed"])
    if len(set(cells + [held])) != len(cells) + 1:
        raise SystemExit("cell and held-out seeds must be distinct")
    return {"cells": cells, "held_out": held}


def cmd_reserve(args) -> int:
    """Reserve a cell protocol's seeds in the ledger (run locally, before the GPU box touches it)."""
    import stage_e

    cfg = load_config(args.config)
    seeds, artifact = cell_seeds(cfg), rel(os.path.abspath(args.config))
    mine = {(int(e["seed"]), e["namespace"]) for e in stage_e.read_ledger() if e.get("artifact") == artifact}
    wanted = {(s, ns) for s in seeds["cells"] for ns in ("fitting", "mt1_pool")} | {(seeds["held_out"], "mt1_pool")}
    if mine == wanted:
        print(f"  seeds already reserved for {artifact}")
        return 0
    if mine:
        raise SystemExit(f"the ledger holds a partial reservation for {artifact}")
    conflicts = stage_e.seed_conflicts(seeds["cells"] + [seeds["held_out"]], seeds["cells"])
    if conflicts:
        raise SystemExit("seeds already used:\n  " + "\n  ".join(conflicts))
    stamp, use = stage_e.now(), cfg.get("study", "cell protocol")
    entries = [{"seed": s, "namespace": "fitting", "use": f"{use}: cell demo sample, split and fitting seed",
                "artifact": artifact, "recorded_at": stamp, "backfilled": False} for s in seeds["cells"]]
    entries += [{"seed": s, "namespace": "mt1_pool", "use": f"{use}: cell probe manifest", "artifact": artifact,
                 "recorded_at": stamp, "backfilled": False} for s in seeds["cells"]]
    entries.append({"seed": seeds["held_out"], "namespace": "mt1_pool", "use": f"{use}: held-out manifest",
                    "artifact": artifact, "recorded_at": stamp, "backfilled": False})
    stage_e.reserve_seeds(entries)
    print(f"  reserved {len(entries)} seed uses for {artifact}")
    return 0


def probe_path(cfg, seed) -> str:
    return os.path.join(_PROJECT_ROOT, cfg["manifests"]["probe_dir"], f"{cfg['env_name']}_probe_s{seed}.yaml")


def build_cell_manifests(cfg, config_path) -> int:
    """Held-out and per-cell probe manifests; refuses unless the seeds were reserved for this config."""
    import stage_e
    from lares.eval.manifest import SPLIT_DEVELOPMENT, EvaluationManifest, assert_disjoint, build_manifest

    seeds, artifact = cell_seeds(cfg), rel(os.path.abspath(config_path))
    reserved = {int(e["seed"]) for e in stage_e.read_ledger() if e.get("artifact") == artifact}
    missing = sorted(set(seeds["cells"] + [seeds["held_out"]]) - reserved)
    if missing:
        raise SystemExit(f"seeds {missing} are not reserved for {artifact}; run `stage_f.py reserve` locally first")
    held = cfg["manifests"]["held_out"]
    out_path = os.path.join(_PROJECT_ROOT, held["path"])
    if os.path.exists(out_path):
        raise SystemExit(f"{rel(out_path)} exists and is immutable")
    horizon, env_name = int(cfg["horizon"]), cfg["env_name"]
    held_out, _ = build_manifest(env_name, SPLIT_DEVELOPMENT, [seeds["held_out"]], horizon=horizon,
                                 num_episodes=int(held["episodes"]), manifest_id=held["manifest_id"])
    probes = {s: build_manifest(env_name, SPLIT_DEVELOPMENT, [s], horizon=horizon,
                                num_episodes=int(cfg["bundle"]["probe_episodes"]),
                                manifest_id=f"{env_name}:cell_probe:s{s}:n{cfg['bundle']['probe_episodes']}")[0]
              for s in seeds["cells"]}
    known = stage_e.unique_manifests(
        [EvaluationManifest.load(p) for p in sorted(glob.glob(os.path.join(_PROJECT_ROOT, "config", "**", "*.yaml"),
                                                               recursive=True)) if _is_manifest(p)])
    assert_disjoint(known + [held_out] + list(probes.values()))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    held_out.save(out_path)
    for s, m in probes.items():
        os.makedirs(os.path.dirname(probe_path(cfg, s)), exist_ok=True)
        m.save(probe_path(cfg, s))
    provenance = {"manifest_id": held_out.manifest_id, "manifest_path": rel(out_path),
                  "manifest_sha256": stage_e.sha256_file(out_path), "pool_seed": seeds["held_out"],
                  "episodes": len(held_out), "built_at": stage_e.now(),
                  "probes": {str(s): {"manifest_id": m.manifest_id, "path": rel(probe_path(cfg, s)),
                                      "sha256": stage_e.sha256_file(probe_path(cfg, s))} for s, m in probes.items()},
                  "disjoint_from": [m.manifest_id for m in known],
                  "selection_sha256": stage_e.sha256_file(os.path.join(_PROJECT_ROOT,
                                                                       cfg["structures"]["evolved"]["selection_file"]))}
    with open(os.path.join(os.path.dirname(out_path), "manifest_provenance.json"), "x", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    print(f"  {held_out.manifest_id} and {len(probes)} probe manifests built; disjoint from {len(known)} saved manifests")
    return 0


def make_analyst(cfg):
    from lares.vision import QwenAnalyst

    v = cfg["vision"]
    return QwenAnalyst(model_id=v["model"], quantization=v["quantization"], revision=v["revision"],
                       max_pixels=v.get("max_pixels"), max_new_tokens=int(v.get("max_new_tokens", 512)))


def cmd_vlm_smoke(args) -> int:
    """Load the pinned VLM on the GPU and analyse one rendered screen failure, before any protocol spend.

    Spends one development screen episode (case 0, the first evolved structure at its declared values), its media
    rerun and one VLM call;
    reports peak GPU memory, latency and whether the reply passes the analysis contract.
    """
    import torch

    from lares.core.obs_schema import get_obs_schema
    from lares.core.training_pipeline import ensure_mujoco_headless_gl
    from lares.eval import EvaluationManifest, make_manifest_env
    from lares.eval.runner import PolicyActor, evaluate_manifest
    from lares.vision import analyse_failures

    cfg = load_config(args.config)
    ensure_mujoco_headless_gl()
    schema = get_obs_schema(cfg["env_name"])
    development = EvaluationManifest.load(os.path.join(_PROJECT_ROOT, cfg["manifests"]["development"]))
    case = development.head(1)
    # An evolved structure: they score 0 untrained, so the case is a failure the analyst must read.
    structure = selected_structures(cfg, schema)[0]
    policy = structure.build(schema)
    env = make_manifest_env(development, development.resolve_pool(), 200)
    actor = PolicyActor(policy, deterministic=True, name=structure.structure_id)
    result = evaluate_manifest(actor, case, env, max_steps=int(cfg["horizon"]))
    if result.success_rate > 0:
        print("  smoke case succeeded; no failure to analyse (memory still measured below)")
    out = os.path.join(_PROJECT_ROOT, "logs", "stage_f_vision_smoke")
    os.makedirs(out, exist_ok=True)
    analyst = make_analyst(cfg)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    analyses, merged, run_info = analyse_failures(actor, env, case, result, analyst, schema,
                                                  candidate_id=structure.structure_id, max_steps=int(cfg["horizon"]),
                                                  limit=1, media_dir=os.path.join(out, "media"))
    report = {"model": cfg["vision"], "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
              "total_gpu_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
              "seconds": time.time() - started, "run": run_info.to_dict(),
              "analyses": [a.to_dict() if hasattr(a, "to_dict") else str(a) for a in analyses],
              "merged": [m.to_dict() for m in merged], "episodes": {"screen": 1, "media_reruns": run_info.episodes_rerun}}
    with open(os.path.join(out, "smoke.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  peak GPU memory {report['peak_gpu_memory_gib']:.2f} / {report['total_gpu_memory_gib']:.2f} GiB, "
          f"{report['seconds']:.0f} s; analyses returned {run_info.analyses_returned}, "
          f"rejected {run_info.analyses_rejected} {run_info.rejection_reasons}")
    env.close()
    return 0


def run_cells(cfg, config_path, structures) -> int:
    """Run a cell protocol: each cell fits and probes on its own bundle; screen and held-out are shared."""
    import stage_e
    from lares.core.obs_schema import get_obs_schema
    from lares.core.training_pipeline import DemoBuffer, ensure_mujoco_headless_gl, load_policy_prompt_assets
    from lares.eval import EvaluationManifest, make_manifest_env
    from lares.fitting.replicates import ReplicateBundle, sample_demonstration_episodes, split_episodes

    sha = stage_e.require_clean_worktree()
    ensure_mujoco_headless_gl()
    env_name, horizon = cfg["env_name"], int(cfg["horizon"])
    schema = get_obs_schema(env_name)
    held = cfg["manifests"]["held_out"]
    with open(os.path.join(_PROJECT_ROOT, os.path.dirname(held["path"]), "manifest_provenance.json"),
              encoding="utf-8") as f:
        prov = json.load(f)
    if stage_e.sha256_file(os.path.join(_PROJECT_ROOT, held["path"])) != prov["manifest_sha256"]:
        raise SystemExit(f"{held['path']} no longer matches its frozen hash")
    for s, p in prov["probes"].items():
        if stage_e.sha256_file(os.path.join(_PROJECT_ROOT, p["path"])) != p["sha256"]:
            raise SystemExit(f"{p['path']} no longer matches its frozen hash")
    b = cfg["bundle"]
    demo_path = os.path.join(_PROJECT_ROOT, b["demonstrations"])
    if stage_e.sha256_file(demo_path) != b["demonstrations_sha256"]:
        raise SystemExit("the demonstration buffer does not match its declared hash")
    buffer = DemoBuffer.load(demo_path)
    arms = list(cfg["arms"])
    unknown = [a for a in arms if a not in sf.ALL_ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}")
    development = EvaluationManifest.load(os.path.join(_PROJECT_ROOT, cfg["manifests"]["development"]))
    screen = development.head(int(cfg["manifests"]["screen_episodes"]))
    held_out = EvaluationManifest.load(os.path.join(_PROJECT_ROOT, held["path"]))
    fit, budget_cfg, llm = cfg["fitting"], cfg["budget"], cfg["llm"]
    budget = sf.StageFBudget(screen_episodes=len(screen), probe_episodes=int(b["probe_episodes"]),
                             fit_budget=int(fit["budget"]), checkpoint_steps=tuple(fit["checkpoint_steps"]),
                             max_substitutions=int(budget_cfg["max_substitutions"]),
                             max_llm_calls=int(budget_cfg["max_llm_calls"]))
    client = None
    llm_args = SimpleNamespace(model=llm["model"], reasoning_effort=llm["reasoning_effort"])
    if any(a in sf.LLM_ARM_EVIDENCE for a in arms):
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("the LLM arms need OPENAI_API_KEY")
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    vision_fn = None
    if sf.ARM_LLM_VISION in arms:
        vision_fn = sf.vision_evidence(make_analyst(cfg), schema, limit=int(cfg["vision"]["episodes_per_packet"]))
    by_id = {s.structure_id: s for s in structures}
    env = make_manifest_env(development, development.resolve_pool(), 200)
    held_out_env = make_manifest_env(held_out, held_out.resolve_pool(), 200)
    prompts = load_policy_prompt_assets(env_name)
    run_dir = os.path.join(_PROJECT_ROOT, cfg["out_dir"], time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    meta = {"study": cfg.get("study"), "arms": arms, "cells": cfg["cells"], "budget": budget.to_dict(),
            "acceptance_rule": sf.ACCEPTANCE_RULE, "llm": llm, "vision": cfg.get("vision"),
            "manifests": {"screen": screen.manifest_id, "held_out": held_out.manifest_id,
                          "probes": {k: v["manifest_id"] for k, v in prov["probes"].items()}},
            "config": rel(os.path.abspath(config_path)), "config_sha256": stage_e.sha256_file(config_path),
            "frozen_git_sha": sha, **git_state(), "held_out_manifest_sha256": prov["manifest_sha256"],
            "selection_sha256": prov["selection_sha256"]}
    with open(os.path.join(run_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"cell run -> {run_dir}\n  cap {budget.repair_episode_cap} repair-evaluation episodes per arm", flush=True)
    for cell in cfg["cells"]:
        seed, structure = int(cell["seed"]), by_id[cell["structure"]]
        probe = EvaluationManifest.load(probe_path(cfg, seed))
        probe_env = make_manifest_env(probe, probe.resolve_pool(), 200)
        sample = sample_demonstration_episodes(list(buffer.episode_index()), int(b["episodes_per_cell"]), seed)
        train, val = split_episodes(sample, float(b["val_fraction"]), seed)
        bundle = ReplicateBundle(seed, seed, sample, train, val, probe, held_out, seed,
                                 int(cfg["seeds"]["held_out_pool_seed"]))
        ctx = sf.ArmContext(env=env, env_name=env_name, schema=schema, data=bundle.fitting_data(buffer),
                            screen=screen, probe=probe, budget=budget, seed=seed, max_steps=horizon, client=client,
                            llm_args=llm_args, prompts=prompts, vision_fn=vision_fn, probe_env=probe_env)
        rep_dir = os.path.join(run_dir, f"{structure.structure_id}_s{seed}")
        rep = sf.run_replicate(structure.structure_id, structure.source, lambda s=structure: s.build(schema), ctx,
                               held_out, rep_dir, arms=arms, held_out_env=held_out_env)
        with open(os.path.join(rep_dir, "bundle.json"), "w", encoding="utf-8") as f:
            json.dump(bundle.to_dict(), f, indent=2)
        probe_env.close()
        print(f"  {structure.structure_id} s{seed}: " + "  ".join(
            f"{r['arm']} {r['held_out']['success']:.2f}" for r in rep["rows"]), flush=True)
    env.close()
    held_out_env.close()
    return report(run_dir)


# ---------------------------------------------------------------------------
#  R-008: direct rollout search versus rollout-selected imitation, from the frozen original vector
# ---------------------------------------------------------------------------

#: Fixed before scoring (advisor consult call_00009): the original vector first, then local and global draws
#: interleaved. Every local draw is centred on the original, never on a running winner.
SEARCH_ORDER = ("original", "local", "global", "local", "global", "local", "global", "global")
#: Eight imitation checkpoints; step 0 is the original vector, so both arms can keep it.
FIT_CHECKPOINTS = (0, 125, 250, 375, 500, 625, 750, 1000)


def search_candidates(original, rng):
    """The eight search candidates in :data:`SEARCH_ORDER`, each a copy of ``original`` (perturbed except the first)."""
    import copy

    out = []
    for kind in SEARCH_ORDER:
        candidate = copy.deepcopy(original)
        if kind != "original":
            sf._propose(candidate, sf.ARM_LOCAL_SEARCH if kind == "local" else sf.ARM_GLOBAL_SEARCH, rng)
        out.append((kind, candidate))
    return out


def search_vs_fit_job(job) -> dict:
    import numpy as np
    import torch

    torch.set_num_threads(1)
    from lares.core.obs_schema import get_obs_schema
    from lares.core.training_pipeline import DemoBuffer, ensure_mujoco_headless_gl
    from lares.eval import EvaluationManifest, make_manifest_env
    from lares.eval.runner import PolicyActor, evaluate_manifest
    from lares.fitting.checkpoints import is_better_checkpoint
    from lares.fitting.objectives import DETERMINISTIC_MSE
    from lares.fitting.optimizers import fit_policy_with_objective, prepare_data, state_parameters_sha256
    from lares.fitting.structure_bank import instantiate_source

    ensure_mujoco_headless_gl()
    started, horizon = time.time(), int(job["horizon"])
    schema = get_obs_schema(job["env_name"])

    def build():
        policy = instantiate_source(job["source"], 39, 4, schema=schema)
        policy.fixed_action_scale = float(job["fixed_action_scale"])
        return policy

    probe = EvaluationManifest.load(job["probe_manifest"])
    probe_env = make_manifest_env(probe, probe.resolve_pool(), 200)
    episodes = {"search_selection": 0, "fitting_selection": 0, "confirmation": 0}

    def probe_score(policy, arm):
        result = evaluate_manifest(PolicyActor(policy, deterministic=True, name=f"{arm}:probe"), probe, probe_env,
                                   max_steps=horizon)
        episodes[f"{arm}_selection"] += len(result.episodes)
        return sf.score_of(result)

    # Search arm: its own RNG instance, so the fitting arm cannot shift its draws.
    search_rng = np.random.default_rng(int(job["optimizer_seed"]))
    best, best_score, trace = None, None, []
    for index, (kind, candidate) in enumerate(search_candidates(build(), search_rng)):
        score = probe_score(candidate, "search")
        trace.append({"index": index, "kind": kind, **score.to_dict()})
        if is_better_checkpoint(score, best_score):
            best, best_score, best_index = candidate, score, index

    # Fitting arm: rollout-selected imitation over eight checkpoints on the same probe.
    data = prepare_data(DemoBuffer.load(job["demo_buffer"]), split_seed=int(job["split_seed"]))
    fitted = build()
    fit = fit_policy_with_objective(fitted, data, DETERMINISTIC_MSE, budget=int(FIT_CHECKPOINTS[-1]),
                                    seed=int(job["optimizer_seed"]), rollout_probe=lambda p: probe_score(p, "fitting"),
                                    checkpoint_steps=FIT_CHECKPOINTS, structure_id=job["structure_id"])
    fitted.load_state_dict(fit.selected_state)

    confirmation = EvaluationManifest.load(job["confirmation_manifest"])
    conf_env = make_manifest_env(confirmation, confirmation.resolve_pool(), 200)
    scored = {}
    for arm, policy in (("search", best), ("fitting", fitted)):
        result = evaluate_manifest(PolicyActor(policy, deterministic=True, name=arm), confirmation, conf_env,
                                   max_steps=horizon)
        episodes["confirmation"] += len(result.episodes)
        scored[arm] = {"success_rate": result.success_rate,
                       "per_case": {e.case_id: float(e.success) for e in result.episodes},
                       "parameters_sha256": state_parameters_sha256(policy.state_dict())}
    out = {"structure_id": job["structure_id"], "search": {**scored["search"], "selected_index": best_index,
                                                           "selected_kind": SEARCH_ORDER[best_index], "trace": trace},
           "fitting": {**scored["fitting"], "selected_step": int(fit.selected_step),
                       "checkpoint_history": fit.checkpoint_history},
           "episodes": episodes, "wall_time_seconds": time.time() - started}
    path = os.path.join(job["out_dir"], "structures", f"{job['structure_id']}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    return out


def search_vs_fit_summary(results) -> dict:
    """Search minus fitting confirmation success, averaged over structures within each placement, t over placements."""
    import numpy as np

    from lares.eval.stats import t_interval, win_loss_tie

    cases = sorted(results[0]["search"]["per_case"])
    per_case = [float(np.mean([r["search"]["per_case"][c] - r["fitting"]["per_case"][c] for r in results]))
                for c in cases]
    per_structure = {r["structure_id"]: {
        "search": r["search"]["success_rate"], "fitting": r["fitting"]["success_rate"],
        "difference": r["search"]["success_rate"] - r["fitting"]["success_rate"],
        "search_selected": r["search"]["selected_kind"], "fitting_selected_step": r["fitting"]["selected_step"],
    } for r in results}
    return {"paired_success_difference": float(np.mean(per_case)),
            "primary": {"n_placements": len(cases), "n_structures": len(results), **t_interval(per_case),
                        **win_loss_tie(per_case)},
            "per_structure": per_structure,
            "episodes": {k: int(sum(r["episodes"][k] for r in results)) for k in results[0]["episodes"]}}


def cmd_search_vs_fit(args) -> int:
    import multiprocessing as mp

    import compare_child_vs_parent as cvp
    from lares.core.obs_schema import get_obs_schema

    cfg = load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)
    status = cvp.ensure_entries_reserved([
        (args.probe_seed, "mt1_pool", "R-008 search vs fitting: 10-case selection probe"),
        (args.heldout_seed, "mt1_pool", "R-008 search vs fitting: 50-case confirmation manifest"),
        (args.optimizer_seed, "fitting", "R-008 search vs fitting: fitting seed and search RNG seed"),
        (args.split_seed, "fitting", "R-008 search vs fitting: demonstration split seed"),
    ], args.out_dir)
    schema = get_obs_schema(cfg["env_name"])
    structures = selected_structures(cfg, schema)
    if len(structures) < 2:
        raise SystemExit("R-008 needs two evolved diagnostic structures in the frozen selection")
    print(f"  structures {[s.structure_id for s in structures]}; seeds: {status}")
    if args.reserve_only:
        return 0
    from lares.core.training_pipeline import ensure_mujoco_headless_gl

    ensure_mujoco_headless_gl()
    horizon = int(cfg["horizon"])
    probe, probe_path = cvp.fresh_manifest(cfg["env_name"], args.probe_seed, 10, horizon, args.out_dir, role="probe")
    conf, conf_path = cvp.fresh_manifest(cfg["env_name"], args.heldout_seed, 50, horizon, args.out_dir,
                                         role="confirmation", also_disjoint_from=[probe])
    common = {"env_name": cfg["env_name"], "horizon": horizon, "fixed_action_scale": 0.1,
              "optimizer_seed": args.optimizer_seed, "split_seed": args.split_seed,
              "demo_buffer": os.path.join(_PROJECT_ROOT, cfg["fitting"]["demo_buffer"]),
              "probe_manifest": os.path.abspath(probe_path), "confirmation_manifest": os.path.abspath(conf_path),
              "out_dir": os.path.abspath(args.out_dir)}
    jobs = [{**common, "structure_id": s.structure_id, "source": s.source} for s in structures[:2]]
    with open(os.path.join(args.out_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump({"argv": sys.argv, **git_state(), "search_order": SEARCH_ORDER, "fit_checkpoints": FIT_CHECKPOINTS,
                   "local_sigma_fraction": sf.LOCAL_SEARCH_SIGMA, "structures": [j["structure_id"] for j in jobs],
                   "manifests": {"probe": probe.manifest_id, "confirmation": conf.manifest_id},
                   "seeds": {"probe": args.probe_seed, "confirmation": args.heldout_seed,
                             "optimizer": args.optimizer_seed, "split": args.split_seed}}, f, indent=2)
    with mp.get_context("spawn").Pool(processes=len(jobs)) as pool:
        results = pool.map(search_vs_fit_job, jobs)
    summary = search_vs_fit_summary(results)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    pr = summary["primary"]
    for sid, row in summary["per_structure"].items():
        print(f"  {sid:<28} search {row['search']:.2f} ({row['search_selected']})  fitting {row['fitting']:.2f} "
              f"(step {row['fitting_selected_step']})  {row['difference']:+.2f}")
    print(f"  search - fitting, placement-averaged over {pr['n_placements']} cases: {pr['mean']:+.3f} "
          f"[{pr['low']:+.3f}, {pr['high']:+.3f}]; episodes {summary['episodes']}")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "search-vs-fit":
        return cmd_search_vs_fit(args)
    if args.command == "reserve":
        return cmd_reserve(args)
    if args.command == "vlm-smoke":
        return cmd_vlm_smoke(args)
    if args.command == "select":
        return cmd_select(args)
    if args.command == "build-manifest":
        cfg = load_config(args.config)
        return build_cell_manifests(cfg, args.config) if "cells" in cfg else cmd_build_manifest(args)
    if args.command == "run":
        return run_from_config(args.config) if args.config else run(args)
    return report(args.run)


if __name__ == "__main__":
    sys.exit(main())
