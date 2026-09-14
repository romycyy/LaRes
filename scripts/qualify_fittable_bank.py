#!/usr/bin/env python
"""Fittable-bank qualification: which hand-written structures should seed the fittable lane?

Decided 2026-09-14, after Stage E showed rollout-selected fitting transforms some hand-written
structures and does nothing for the evolved ones. Nine structures (the three references and six
predeclared single-change variants) are each fitted with five seeds. The checkpoint is chosen from
{0, 500, 1000} on a fixed probe manifest and scored on a separate evaluation manifest. A selected
step of 0 is "fitting rejected". A structure enters the fittable seed bank only if fitting improves
held-out success in at least three of five seeds, never regresses by more than 0.05 in any seed,
and its mean paired difference is positive. The protocol lives in
``config/fittable_bank/qualification.yaml``.

Subcommands, in order (from the project root)::

    python scripts/qualify_fittable_bank.py validate        # structural checks only, no rollouts
    git add lares/fitting/fittable_variants config/fittable_bank && git commit
    python scripts/qualify_fittable_bank.py build-manifest  # ledger check, reservation, manifest
    git add config && git commit
    python scripts/qualify_fittable_bank.py run             # clean worktree; records the SHA
    python scripts/qualify_fittable_bank.py report --run logs/fittable_bank/<run_id>
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import json
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _SCRIPT_DIR)

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from stage_e import (  # noqa: E402
    LEDGER_PATH,
    _outcome,
    now,
    rel,
    require_clean_worktree,
    reserve_seeds,
    seed_conflicts,
    sha256_file,
    t_interval,
    unique_manifests,
)

CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "fittable_bank", "qualification.yaml")
FITTING_REJECTED = "fitting rejected"


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
#  Structures and their declared differences
# ---------------------------------------------------------------------------


def reference_structures(cfg) -> dict:
    from lares.fitting.structure_bank import load_bank

    bank = {s.structure_id: s for s in load_bank()}
    return {sid: bank[sid] for sid in cfg["structures"]["references"]}


def variant_structures(cfg) -> dict:
    from lares.fitting.structure_bank import load_structure
    from lares.core.obs_schema import get_obs_schema

    schema = get_obs_schema(cfg["env_name"])
    directory = os.path.join(_PROJECT_ROOT, cfg["structures"]["variants_dir"])
    return {
        v["id"]: load_structure(os.path.join(directory, f"{v['id']}.py"), schema=schema)
        for v in cfg["structures"]["variants"]
    }


def declared_difference_problems(parent_policy, variant_policy, declaration) -> list:
    """Differences between parent and variant that the declaration does not name."""
    parent_ranges, variant_ranges = parent_policy.get_param_ranges(), variant_policy.get_param_ranges()
    parent_values = {n: p.detach().reshape(-1).tolist() for n, p in parent_policy.named_parameters()}
    variant_values = {n: p.detach().reshape(-1).tolist() for n, p in variant_policy.named_parameters()}
    problems = []
    added, removed = set(variant_values) - set(parent_values), set(parent_values) - set(variant_values)
    if added != set(declaration.get("adds", [])):
        problems.append(f"adds {sorted(added)}, declared {sorted(declaration.get('adds', []))}")
    if removed != set(declaration.get("removes", [])):
        problems.append(f"removes {sorted(removed)}, declared {sorted(declaration.get('removes', []))}")
    changed = {
        n for n in set(parent_values) & set(variant_values)
        if parent_values[n] != variant_values[n] or tuple(parent_ranges[n]) != tuple(variant_ranges[n])
    }
    if changed != set(declaration.get("changes", [])):
        problems.append(f"changes {sorted(changed)}, declared {sorted(declaration.get('changes', []))}")
    return problems


def cmd_validate(args) -> int:
    """Structural checks only. Nothing here runs a rollout or reads an evaluation case."""
    from lares.core.obs_schema import get_obs_schema
    from lares.core.policy_validator import validate_policy
    from lares.fitting.structure_bank import SIMPLE_FAMILY_MAX_PARAMS

    cfg = load_config(args.config)
    frozen_path = os.path.join(_PROJECT_ROOT, cfg["structures"]["frozen_file"])
    if os.path.exists(frozen_path):
        raise SystemExit(f"{rel(frozen_path)} exists; the variants are frozen")
    if os.path.exists(os.path.join(_PROJECT_ROOT, cfg["evaluation_manifest"]["path"])):
        raise SystemExit("the evaluation manifest already exists; variants must be frozen before it")
    schema = get_obs_schema(cfg["env_name"])
    references, variants = reference_structures(cfg), variant_structures(cfg)
    entries, failures = [], []
    for declaration in cfg["structures"]["variants"]:
        structure = variants[declaration["id"]]
        parent = references[declaration["parent"]]
        report = validate_policy(structure.build(schema), 39, 4, source=structure.source, schema=schema)
        problems = declared_difference_problems(parent.build(schema), structure.build(schema), declaration)
        if not structure.provenance:
            problems.append("no provenance line")
        if structure.num_parameters > SIMPLE_FAMILY_MAX_PARAMS:
            problems.append(f"{structure.num_parameters} parameters, above the simple family")
        if not report.ok:
            problems.extend(str(e) for e in report.errors)
        entry = {
            "id": declaration["id"], "parent": declaration["parent"], "axis": declaration["axis"],
            "path": rel(structure.path), "sha256": sha256_file(structure.path),
            "num_parameters": structure.num_parameters, "validator_checks": report.checked,
            "validator_warnings": [str(w) for w in report.warnings], "problems": problems,
        }
        entries.append(entry)
        print(f"  {declaration['id']:<36} {structure.num_parameters:>2} params  "
              f"{'ok' if not problems else 'PROBLEMS: ' + '; '.join(problems)}")
        if problems:
            failures.append(declaration["id"])
    if failures:
        raise SystemExit(f"variants failed validation: {failures}")
    payload = {
        "frozen_at": now(), "rollouts_run_before_freezing": 0,
        "references": {sid: {"path": rel(s.path), "sha256": sha256_file(s.path)} for sid, s in references.items()},
        "variants": entries,
    }
    os.makedirs(os.path.dirname(frozen_path), exist_ok=True)
    with open(frozen_path, "x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  written: {rel(frozen_path)}")
    return 0


def verify_frozen_structures(cfg) -> dict:
    with open(os.path.join(_PROJECT_ROOT, cfg["structures"]["frozen_file"]), encoding="utf-8") as f:
        frozen = json.load(f)
    for sid, ref in frozen["references"].items():
        if sha256_file(os.path.join(_PROJECT_ROOT, ref["path"])) != ref["sha256"]:
            raise SystemExit(f"reference {sid} changed after the variants were frozen")
    for v in frozen["variants"]:
        if sha256_file(os.path.join(_PROJECT_ROOT, v["path"])) != v["sha256"]:
            raise SystemExit(f"variant {v['id']} changed after it was frozen")
    return frozen


# ---------------------------------------------------------------------------
#  Evaluation manifest
# ---------------------------------------------------------------------------


def cmd_build_manifest(args) -> int:
    from lares.eval.manifest import SPLIT_DEVELOPMENT, EvaluationManifest, assert_disjoint, build_manifest

    cfg = load_config(args.config)
    verify_frozen_structures(cfg)
    out_path = os.path.join(_PROJECT_ROOT, cfg["evaluation_manifest"]["path"])
    if os.path.exists(out_path):
        raise SystemExit(f"{rel(out_path)} exists and is immutable")
    pool_seed = int(cfg["seeds"]["evaluation_pool_seed"])
    fitting_seeds = [int(s) for s in cfg["seeds"]["fitting_seeds"]]
    conflicts = seed_conflicts([pool_seed], fitting_seeds)
    if conflicts:
        raise SystemExit("seeds already used:\n  " + "\n  ".join(conflicts))
    print(f"  seeds {pool_seed} (MT1 pool) and {fitting_seeds} (fitting) are unused in the ledger and every saved manifest")
    manifest, _ = build_manifest(
        cfg["env_name"], SPLIT_DEVELOPMENT, [pool_seed], horizon=int(cfg["horizon"]),
        num_episodes=int(cfg["evaluation_manifest"]["episodes"]), manifest_id=cfg["evaluation_manifest"]["manifest_id"],
    )
    known = unique_manifests(
        [EvaluationManifest.load(p) for p in sorted(glob.glob(os.path.join(_PROJECT_ROOT, "config", "manifests", "*.yaml")))]
        + [EvaluationManifest.load(p) for p in sorted(glob.glob(os.path.join(_PROJECT_ROOT, "config", "stage_e", "*confirmation*.yaml")))]
        + [EvaluationManifest.load(p) for p in sorted(glob.glob(os.path.join(_PROJECT_ROOT, "logs", "**", "*confirmation*.yaml"), recursive=True))]
    )
    assert_disjoint(known + [manifest])
    manifest.save(out_path)
    stamp = now()
    reserve_seeds(
        [{"seed": pool_seed, "namespace": "mt1_pool", "use": "fittable-bank qualification evaluation manifest",
          "artifact": rel(out_path), "recorded_at": stamp, "backfilled": False}]
        + [{"seed": s, "namespace": "fitting", "use": "fittable-bank qualification fitting seeds",
            "artifact": rel(args.config), "recorded_at": stamp, "backfilled": False} for s in fitting_seeds]
    )
    provenance = {
        "manifest_id": manifest.manifest_id, "manifest_path": rel(out_path), "manifest_sha256": sha256_file(out_path),
        "pool_seed": pool_seed, "episodes": len(manifest), "built_at": stamp,
        "disjoint_from": [m.manifest_id for m in known],
        "variants_frozen_sha256": sha256_file(os.path.join(_PROJECT_ROOT, cfg["structures"]["frozen_file"])),
        "ledger_sha256_after_reservation": sha256_file(LEDGER_PATH),
    }
    with open(os.path.join(os.path.dirname(out_path), "manifest_provenance.json"), "x", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    print(f"  manifest {manifest.manifest_id}: {len(manifest)} cases -> {rel(out_path)}")
    return 0


# ---------------------------------------------------------------------------
#  Run
# ---------------------------------------------------------------------------


def run_structure(job) -> dict:
    """Zero-shot once, then one rollout-selected fit per seed, each scored on the evaluation manifest."""
    import torch

    torch.set_num_threads(1)
    from lares.core.obs_schema import get_obs_schema
    from lares.core.training_pipeline import DemoBuffer, ensure_mujoco_headless_gl
    from lares.eval import EvaluationManifest, make_manifest_env
    from lares.eval.runner import PolicyActor, evaluate_manifest
    from lares.fitting.checkpoints import RolloutScore
    from lares.fitting.optimizers import fit_policy_with_objective, prepare_data, state_parameters_sha256
    from lares.fitting.structure_bank import instantiate_source

    ensure_mujoco_headless_gl()
    cfg, spec, out_dir, git_sha = job["cfg"], job["spec"], job["out_dir"], job["git_sha"]
    fit_cfg = cfg["fitting"]
    schema = get_obs_schema(cfg["env_name"])
    started = time.time()
    evaluation = EvaluationManifest.load(os.path.join(_PROJECT_ROOT, cfg["evaluation_manifest"]["path"]))
    development = EvaluationManifest.load(os.path.join(_PROJECT_ROOT, cfg["checkpoint_selection_manifest"]["path"]))
    start, stop = cfg["checkpoint_selection_manifest"]["slice"]
    selection = development.slice(int(start), int(stop))
    evaluation_env = make_manifest_env(evaluation, evaluation.resolve_pool())
    probe_env = make_manifest_env(selection, development.resolve_pool())
    horizon = int(cfg["horizon"])
    data = prepare_data(DemoBuffer.load(os.path.join(_PROJECT_ROOT, fit_cfg["demo_buffer"])),
                        val_fraction=float(fit_cfg["val_fraction"]), split_seed=int(fit_cfg["split_seed"]))
    counters = {"probe_episodes": 0, "probe_environment_steps": 0, "evaluation_episodes": 0,
                "evaluation_environment_steps": 0, "bc_gradient_steps": 0}
    cache = {}

    def build():
        policy = instantiate_source(spec["source"], 39, 4, schema)
        policy.fixed_action_scale = float(fit_cfg["fixed_action_scale"])
        return policy

    def evaluate(policy):
        digest = state_parameters_sha256(policy.state_dict())
        if digest not in cache:
            result = evaluate_manifest(PolicyActor(policy, deterministic=True, name=spec["structure_id"]),
                                       evaluation, evaluation_env, max_steps=horizon)
            counters["evaluation_episodes"] += len(result.episodes)
            counters["evaluation_environment_steps"] += int(sum(e.length for e in result.episodes))
            cache[digest] = _outcome(result)
        return digest, cache[digest]

    def probe(policy):
        result = evaluate_manifest(PolicyActor(policy, deterministic=True, name="probe"), selection, probe_env,
                                   max_steps=horizon)
        counters["probe_episodes"] += len(result.episodes)
        counters["probe_environment_steps"] += int(sum(e.length for e in result.episodes))
        progress = [e.signed_goal_progress for e in result.episodes if e.signed_goal_progress is not None]
        return RolloutScore(result.success_rate, float(np.mean(progress)) if progress else 0.0)

    zero_digest, zero = evaluate(build())
    seeds = []
    for seed in [int(s) for s in cfg["seeds"]["fitting_seeds"]]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        policy = build()
        fit = fit_policy_with_objective(
            policy, data, fit_cfg["objective"], budget=int(fit_cfg["budget"]), batch_size=int(fit_cfg["batch_size"]),
            lr=float(fit_cfg["lr"]), clip_grad_norm=float(fit_cfg["clip_grad_norm"]), seed=seed,
            sampling=fit_cfg["sampling"], rollout_probe=probe,
            checkpoint_steps=[int(s) for s in fit_cfg["checkpoint_steps"]], structure_id=spec["structure_id"],
        )
        counters["bc_gradient_steps"] += int(fit.bc_gradient_steps)
        selected = build()
        selected.load_state_dict(fit.selected_state)
        digest, outcome = evaluate(selected)
        seeds.append({
            "seed": seed, "selected_step": int(fit.selected_step),
            "fitting_rejected": int(fit.selected_step) == 0,
            "selected_parameters_sha256": digest, "zero_shot_parameters_sha256": zero_digest,
            "zero_shot_success": zero["success_rate"], "selected_success": outcome["success_rate"],
            "difference": outcome["success_rate"] - zero["success_rate"],
            "checkpoint_history": fit.checkpoint_history, "objective_config": fit.objective_config,
            "frozen_parameters": fit.frozen_parameters,
        })
    result = {
        "study": cfg["study"], "git_sha": git_sha, "structure_id": spec["structure_id"], "role": spec["role"],
        "parent": spec.get("parent"), "axis": spec.get("axis"), "source_path": spec["source_path"],
        "source_sha256": spec["source_sha256"], "evaluation_manifest_id": evaluation.manifest_id,
        "checkpoint_selection_manifest_id": selection.manifest_id, "outcomes_by_parameters": cache,
        "seeds": seeds, "resources": counters, "wall_time_seconds": time.time() - started, "finished_at": now(),
    }
    path = os.path.join(out_dir, "structures", f"{spec['structure_id']}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    return {"structure_id": spec["structure_id"], "wall_time_seconds": result["wall_time_seconds"]}


def cmd_run(args) -> int:
    import multiprocessing as mp

    git_sha = require_clean_worktree()
    cfg = load_config(args.config)
    frozen = verify_frozen_structures(cfg)
    manifest_path = os.path.join(_PROJECT_ROOT, cfg["evaluation_manifest"]["path"])
    with open(os.path.join(os.path.dirname(manifest_path), "manifest_provenance.json"), encoding="utf-8") as f:
        provenance = json.load(f)
    if sha256_file(manifest_path) != provenance["manifest_sha256"]:
        raise SystemExit("the evaluation manifest changed after it was built")
    if sha256_file(os.path.join(_PROJECT_ROOT, cfg["fitting"]["demo_buffer"])) != cfg["fitting"]["demo_buffer_sha256"]:
        raise SystemExit("the demo buffer does not match its frozen hash")
    references, variants = reference_structures(cfg), variant_structures(cfg)
    specs = [{"structure_id": sid, "role": "reference", "source": s.source, "source_path": rel(s.path),
              "source_sha256": sha256_file(s.path)} for sid, s in references.items()]
    for v in cfg["structures"]["variants"]:
        s = variants[v["id"]]
        specs.append({"structure_id": v["id"], "role": "variant", "parent": v["parent"], "axis": v["axis"],
                      "source": s.source, "source_path": rel(s.path), "source_sha256": sha256_file(s.path)})
    run_id = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(_PROJECT_ROOT, "logs", "fittable_bank", run_id)
    os.makedirs(out_dir)
    with open(os.path.join(out_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump({"git_sha": git_sha, "started_at": now(), "config": cfg, "config_sha256": sha256_file(args.config),
                   "variants_frozen": frozen, "manifest_provenance": provenance,
                   "structures": [{k: v for k, v in s.items() if k != "source"} for s in specs]}, f, indent=2)
    print(f"  qualification run {run_id} at {git_sha[:12]}: {len(specs)} structures x "
          f"{len(cfg['seeds']['fitting_seeds'])} seeds -> {rel(out_dir)}", flush=True)
    jobs = [{"cfg": cfg, "spec": s, "out_dir": out_dir, "git_sha": git_sha} for s in specs]
    with mp.get_context("spawn").Pool(processes=min(int(args.workers), len(jobs))) as pool:
        for done in pool.imap_unordered(run_structure, jobs):
            print(f"  finished {done['structure_id']} in {done['wall_time_seconds']:.0f} s", flush=True)
    return report(out_dir, cfg)


# ---------------------------------------------------------------------------
#  Qualification
# ---------------------------------------------------------------------------


def qualify(differences, rule) -> dict:
    """Apply the seed-bank rule to one structure's per-seed paired differences."""
    arr = np.asarray(differences, dtype=float)
    improvements = int((arr > 0).sum())
    worst = float(arr.min()) if arr.size else None
    mean = float(arr.mean()) if arr.size else None
    checks = {
        "improves_in_enough_seeds": improvements >= int(rule["improves_in_at_least"]),
        "no_seed_regresses_beyond_tolerance": worst is not None and worst >= float(rule["worst_seed_difference_at_least"]) - 1e-12,
        "mean_difference_positive": mean is not None and mean > float(rule["mean_difference_greater_than"]),
    }
    return {"improvements": improvements, "worst": worst, "mean": mean, "checks": checks,
            "qualifies": all(checks.values()), "interval": t_interval(arr.tolist())}


def report(run_dir, cfg=None) -> int:
    cfg = cfg or load_config()
    rule = cfg["qualification_rule"]["enters_fittable_seed_bank_if"]
    results = []
    for path in sorted(glob.glob(os.path.join(run_dir, "structures", "*.json"))):
        with open(path, encoding="utf-8") as f:
            results.append(json.load(f))
    if not results:
        raise SystemExit(f"no structure results under {run_dir}")
    shas = {r["git_sha"] for r in results}
    if len(shas) != 1:
        raise SystemExit(f"structure results come from different commits: {shas}")
    rows = {}
    for r in results:
        diffs = [s["difference"] for s in r["seeds"]]
        rows[r["structure_id"]] = {
            "role": r["role"], "parent": r.get("parent"), "axis": r.get("axis"),
            "zero_shot_success": r["seeds"][0]["zero_shot_success"],
            "selected_success_by_seed": [s["selected_success"] for s in r["seeds"]],
            "difference_by_seed": diffs,
            "selected_step_by_seed": [s["selected_step"] for s in r["seeds"]],
            "fitting_rejected_seeds": sum(1 for s in r["seeds"] if s["fitting_rejected"]),
            "median_difference": float(np.median(diffs)),
            **qualify(diffs, rule),
        }
    seed_bank = sorted(sid for sid, row in rows.items() if row["qualifies"])
    out = {
        "study": cfg["study"], "git_sha": shas.pop(), "rule": rule,
        "note": "seed-bank qualification rule, not a publication claim; probe scores selected parameters only",
        "structures": rows, "fittable_seed_bank": seed_bank,
        "resources": {k: sum(int(r["resources"].get(k, 0)) for r in results)
                      for k in ("probe_episodes", "probe_environment_steps", "evaluation_episodes",
                                "evaluation_environment_steps", "bc_gradient_steps")},
    }
    with open(os.path.join(run_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  qualification report at {out['git_sha'][:12]}")
    print(f"  {'structure':<36}{'role':<10}{'zero':>6}  {'selected success by seed':<32}{'steps':<22}"
          f"{'improve':>8}{'worst':>8}{'mean':>8}  t95              qualifies")
    for sid, row in sorted(rows.items(), key=lambda kv: (kv[1]["role"] != "reference", kv[0])):
        ci = row["interval"]
        ci_text = f"[{ci['low']:+.3f}, {ci['high']:+.3f}]" if ci["low"] is not None else "n/a"
        print(f"  {sid:<36}{row['role']:<10}{row['zero_shot_success']:>6.2f}  "
              f"{' '.join(f'{x:.2f}' for x in row['selected_success_by_seed']):<32}"
              f"{str(row['selected_step_by_seed']):<22}{row['improvements']:>6}/5{row['worst']:>+8.2f}"
              f"{row['mean']:>+8.3f}  {ci_text:<17}{'YES' if row['qualifies'] else 'no'}")
    print(f"\n  fittable seed bank: {seed_bank}")
    print(f"  written: {rel(os.path.join(run_dir, 'report.json'))}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=CONFIG_PATH)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    sub.add_parser("build-manifest")
    run = sub.add_parser("run")
    run.add_argument("--workers", type=int, default=8)
    rep = sub.add_parser("report")
    rep.add_argument("--run", required=True)
    args = p.parse_args(argv)
    if args.command == "validate":
        return cmd_validate(args)
    if args.command == "build-manifest":
        return cmd_build_manifest(args)
    if args.command == "run":
        return cmd_run(args)
    return report(args.run, load_config(args.config))


if __name__ == "__main__":
    sys.exit(main())
