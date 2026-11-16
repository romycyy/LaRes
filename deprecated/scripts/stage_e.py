#!/usr/bin/env python
"""Stage E of ``debug_spec.md``: does imitation fitting beat zero-shot on held-out success?

Imitation fitting is an experimental treatment (decided 2026-09-14). Four conditions
are scored on one immutable development-confirmation manifest shared by every arm:

``zero_shot``                        the structure's own initial values, step 0
``mse_fixed_2000``                   deterministic MSE, the step-2000 parameters
``mse_rollout_selected``             deterministic MSE, best of {0, 500, 1000, 1500, 2000}
``phase_balanced_rollout_selected``  phase-balanced sampling, best of the same checkpoints

Checkpoints are selected on a development slice; the comparison uses the
confirmation manifest, so selection and final comparison never share a case.
Fitting stays in the default pipeline only if at least one fitted condition beats
zero-shot on held-out paired success with its 95% interval above zero; otherwise it
is disabled by default. The rule is written into ``config/stage_e/stage_e.yaml``
and committed before any Stage E case exists.

Subcommands, in order (from the project root)::

    python scripts/stage_e.py select            # freeze the evolved structures first
    python scripts/stage_e.py build-manifest    # ledger check, seed reservation, manifest
    git add config/stage_e config/experiment_seed_ledger.jsonl && git commit
    python scripts/stage_e.py run               # clean worktree required; records the SHA
    python scripts/stage_e.py report --run logs/stage_e/<run_id>
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402

CONFIG_PATH = os.path.join(_PROJECT_ROOT, "config", "stage_e", "stage_e.yaml")
from lares.eval.manifest import unique_manifests as _unique_manifests  # noqa: E402
from lares.eval.stats import t_interval  # noqa: E402
from lares.utils.cli import load_yaml  # noqa: E402
from lares.utils.clock import now_iso as now, run_stamp  # noqa: E402
from lares.utils.gitstate import rel  # noqa: E402
from lares.utils.gitstate import require_clean_worktree as _require_clean_worktree  # noqa: E402
from lares.utils.hashing import sha256_file, sha256_text  # noqa: E402
from lares.utils.seeds import SEED_LEDGER as LEDGER_PATH  # noqa: E402
from lares.utils.seeds import read_ledger, reserve_seeds, seed_conflicts  # noqa: E402,F401  (read_ledger used by tests)

ZERO_SHOT = "zero_shot"
MSE_FIXED = "mse_fixed_2000"
MSE_ROLLOUT = "mse_rollout_selected"
PHASE_ROLLOUT = "phase_balanced_rollout_selected"
FITTED_CONDITIONS = (MSE_FIXED, MSE_ROLLOUT, PHASE_ROLLOUT)
CONDITIONS = (ZERO_SHOT,) + FITTED_CONDITIONS

FAMILY_BANK = "bank"
FAMILY_EVOLVED = "evolved"


# ---------------------------------------------------------------------------
#  Small utilities
# ---------------------------------------------------------------------------


def load_config(path: str = CONFIG_PATH) -> dict:
    return load_yaml(path)


def require_clean_worktree(cwd=_PROJECT_ROOT) -> str:
    """Return HEAD's SHA, refusing any modified, staged or untracked (non-ignored) file."""
    return _require_clean_worktree("Stage E", cwd=cwd)


# ---------------------------------------------------------------------------
#  Structure selection
# ---------------------------------------------------------------------------


def rank_key(record):
    """Predeclared ranking: promotion-set success, then screen success, then screen progress."""
    expanded = record.evaluation.get("expanded_success_rate")
    screen = record.evaluation.get("success_rate")
    progress = record.evaluation.get("_screen_progress")
    return (
        1 if expanded is not None else 0,
        float(expanded) if expanded is not None else -1.0,
        float(screen) if screen is not None else -1.0,
        float(progress) if progress is not None else -math.inf,
    )


def development_outcomes(run_dir, record) -> dict:
    """``{report: {case_id: (success, return)}}`` for every saved development report."""
    out = {}
    for name, ref in (record.evaluation.get("reports") or {}).items():
        path = os.path.join(run_dir, ref["path"])
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            report = json.load(f)
        out[name] = {e["case_id"]: (float(e["success"]), float(e["episode_return"])) for e in report["episodes"]}
    return out


def same_development_behaviour(a: dict, b: dict, atol: float = 1e-6) -> bool:
    """True when every shared report case has identical success and return."""
    shared = [(name, case) for name in a if name in b for case in a[name] if case in b[name]]
    if not shared:
        return False
    return all(
        a[n][c][0] == b[n][c][0] and abs(a[n][c][1] - b[n][c][1]) <= atol for n, c in shared
    )


def deduplicate(candidates, actions_equal, behaviour_equal) -> tuple:
    """Keep candidates in rank order, dropping identical code or identical behaviour.

    ``candidates`` are dicts with ``id`` and ``source_sha256``; ``actions_equal`` and
    ``behaviour_equal`` are ``fn(a, b) -> bool``. Identical behaviour needs *both*
    identical actions on the behaviour batch and identical development outcomes,
    because either alone can miss a difference in states the other never visits.
    """
    kept, dropped = [], []
    for c in candidates:
        duplicate_of = None
        for k in kept:
            if c["source_sha256"] == k["source_sha256"]:
                duplicate_of, why = k["id"], "identical source"
                break
            if actions_equal(c, k) and behaviour_equal(c, k):
                duplicate_of, why = k["id"], "identical actions on the behaviour batch and identical development outcomes"
                break
        if duplicate_of is None:
            kept.append(c)
        else:
            dropped.append({"id": c["id"], "duplicate_of": duplicate_of, "reason": why})
    return kept, dropped


def behaviour_batch(cfg, n=2048, seed=0):
    """Training-buffer observations for the behaviour check; no Stage E case is involved."""
    import torch

    from lares.core.training_pipeline import DemoBuffer

    buffer = DemoBuffer.load(os.path.join(_PROJECT_ROOT, cfg["fitting"]["demo_buffer"]))
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(buffer), size=min(n, len(buffer)), replace=False))
    obs, _ = buffer.take(idx)
    return torch.tensor(obs, dtype=torch.float32)


def cmd_select(args) -> int:
    import torch

    from lares.core.obs_schema import get_obs_schema
    from lares.fitting.optimizers import state_parameters_sha256
    from lares.fitting.structure_bank import instantiate_source
    from lares.search import STATUS_INVALID, load_records
    from lares.search.provenance import load_tensor_artifact, reconstruct_record

    cfg = load_config(args.config)
    ev = cfg["structures"]["evolved"]
    out_dir = os.path.join(_PROJECT_ROOT, os.path.dirname(ev["selection_file"]))
    selection_path = os.path.join(_PROJECT_ROOT, ev["selection_file"])
    manifest_path = os.path.join(_PROJECT_ROOT, cfg["confirmation_manifest"]["path"])
    if os.path.exists(manifest_path):
        raise SystemExit(f"{rel(manifest_path)} exists: evolved policies must be selected before Stage E cases exist")
    if os.path.exists(selection_path):
        raise SystemExit(f"{rel(selection_path)} exists; the selection is frozen")

    run_dir = os.path.join(_PROJECT_ROOT, ev["source_run"])
    schema = get_obs_schema(cfg["env_name"])
    batch = behaviour_batch(cfg)
    considered, pool = [], []
    for record in load_records(os.path.join(run_dir, "experiments")):
        if record.status == STATUS_INVALID or record.intervention:
            continue
        entry = {"id": record.experiment_id, "candidate_id": record.candidate_id,
                 "expanded_success_rate": record.evaluation.get("expanded_success_rate"),
                 "screen_success_rate": record.evaluation.get("success_rate")}
        outcome = reconstruct_record(record, run_dir)
        entry["reconstructs"] = bool(outcome["identical"])
        if not outcome["identical"]:
            entry["excluded"] = f"reconstruction failed: {outcome['problems']}"
            considered.append(entry)
            continue
        source = open(os.path.join(run_dir, record.reconstruction["source"]["path"]), encoding="utf-8").read()
        policy = instantiate_source(source, 39, 4, schema)
        initial = load_tensor_artifact(run_dir, record.fitting["initial_state"])
        if state_parameters_sha256(policy.state_dict()) != state_parameters_sha256(initial):
            entry["excluded"] = "the source's initial values differ from the recorded initial state"
            considered.append(entry)
            continue
        with torch.no_grad():
            actions = torch.tanh(policy(batch)[0])
        report_path = os.path.join(run_dir, record.evaluation["reports"]["fitted_screen"]["path"])
        with open(report_path, encoding="utf-8") as f:
            screen_report = json.load(f)
        values = [e["signed_goal_progress"] for e in screen_report["episodes"] if e.get("signed_goal_progress") is not None]
        record.evaluation["_screen_progress"] = float(np.mean(values)) if values else None
        entry.update(
            source=source, source_sha256=sha256_text(source), actions=actions,
            outcomes=development_outcomes(run_dir, record),
            initial_parameters_sha256=state_parameters_sha256(initial),
            selected_step=record.fitting.get("selected_step"),
            num_parameters=record.code.get("num_parameters"), rank_key=list(rank_key(record)),
        )
        pool.append(entry)
        considered.append(entry)

    pool.sort(key=lambda e: tuple(e["rank_key"]), reverse=True)
    kept, dropped = deduplicate(
        pool,
        actions_equal=lambda a, b: bool(torch.allclose(a["actions"], b["actions"], atol=1e-6, rtol=0.0)),
        behaviour_equal=lambda a, b: same_development_behaviour(a["outcomes"], b["outcomes"]),
    )
    chosen = kept[: int(ev["top_k"])]
    if len(chosen) < int(ev["min_k"]):
        raise SystemExit(f"only {len(chosen)} reconstructable, distinct evolved policies; Stage E needs {ev['min_k']}")

    structures_dir = os.path.join(out_dir, "structures")
    os.makedirs(structures_dir, exist_ok=True)
    selected = []
    for rank, c in enumerate(chosen, start=1):
        path = os.path.join(structures_dir, f"evolved_{c['id']}.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(c["source"])
        selected.append({
            "structure_id": f"evolved_{c['id']}", "rank": rank, "record": c["id"], "source_run": ev["source_run"],
            "source_path": rel(path), "source_sha256": sha256_file(path),
            "initial_parameters_sha256": c["initial_parameters_sha256"], "num_parameters": c["num_parameters"],
            "expanded_success_rate": c["expanded_success_rate"], "screen_success_rate": c["screen_success_rate"],
            "selected_step_in_source_run": c["selected_step"], "rank_key": c["rank_key"],
        })
    payload = {
        "selected_at": now(),
        "selected_before_confirmation_manifest": True,
        "rule": ev["rule"],
        "source_run": ev["source_run"],
        "selected": selected,
        "deduplicated": dropped,
        "considered": [{k: v for k, v in e.items() if k not in ("source", "actions", "outcomes")} for e in considered],
        "behaviour_batch": {"source": cfg["fitting"]["demo_buffer"], "size": int(batch.shape[0]), "seed": 0},
    }
    with open(selection_path, "x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  selected {len(selected)} evolved structures: {[s['record'] for s in selected]}")
    for d in dropped:
        print(f"  deduplicated {d['id']} -> {d['duplicate_of']} ({d['reason']})")
    print(f"  written: {rel(selection_path)}")
    return 0


# ---------------------------------------------------------------------------
#  Confirmation manifest
# ---------------------------------------------------------------------------


def unique_manifests(manifests) -> list:
    """``lares.eval.manifest.unique_manifests`` with the clash reported as a clean exit."""
    try:
        return _unique_manifests(manifests)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def cmd_build_manifest(args) -> int:
    from lares.eval.manifest import SPLIT_DEVELOPMENT, EvaluationManifest, assert_disjoint, build_manifest

    cfg = load_config(args.config)
    selection_path = os.path.join(_PROJECT_ROOT, cfg["structures"]["evolved"]["selection_file"])
    if not os.path.isfile(selection_path):
        raise SystemExit("run `select` first: evolved policies are chosen before Stage E cases exist")
    out_path = os.path.join(_PROJECT_ROOT, cfg["confirmation_manifest"]["path"])
    if os.path.exists(out_path):
        raise SystemExit(f"{rel(out_path)} exists and is immutable")

    pool_seed = int(cfg["seeds"]["confirmation_pool_seed"])
    fitting_seeds = [int(s) for s in cfg["seeds"]["fitting_seeds"]]
    conflicts = seed_conflicts([pool_seed], fitting_seeds)
    if conflicts:
        raise SystemExit("seeds already used:\n  " + "\n  ".join(conflicts))
    print(f"  seeds {pool_seed} (MT1 pool) and {fitting_seeds} (fitting) are unused in the ledger and every saved manifest")

    manifest, _ = build_manifest(
        cfg["env_name"], SPLIT_DEVELOPMENT, [pool_seed], horizon=int(cfg["horizon"]),
        num_episodes=int(cfg["confirmation_manifest"]["episodes"]),
        manifest_id=cfg["confirmation_manifest"]["manifest_id"],
    )
    committed = [EvaluationManifest.load(p) for p in sorted(glob.glob(os.path.join(_PROJECT_ROOT, "config", "manifests", "*.yaml")))]
    spent = [EvaluationManifest.load(p) for p in sorted(glob.glob(os.path.join(_PROJECT_ROOT, "logs", "**", "*confirmation*.yaml"), recursive=True))]
    known = unique_manifests(committed + spent)
    assert_disjoint(known + [manifest])
    committed, spent = known[: len(committed)], known[len(committed):]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    manifest.save(out_path)
    stamp = now()
    reserve_seeds(
        [{"seed": pool_seed, "namespace": "mt1_pool", "use": "Stage E development-confirmation manifest",
          "artifact": rel(out_path), "recorded_at": stamp, "backfilled": False}]
        + [{"seed": s, "namespace": "fitting", "use": "Stage E fitting seeds", "artifact": rel(args.config),
            "recorded_at": stamp, "backfilled": False} for s in fitting_seeds]
    )
    provenance = {
        "manifest_id": manifest.manifest_id, "manifest_path": rel(out_path), "manifest_sha256": sha256_file(out_path),
        "pool_seed": pool_seed, "episodes": len(manifest), "built_at": stamp,
        "disjoint_from": [m.manifest_id for m in committed + spent],
        "selection_sha256": sha256_file(selection_path), "ledger_sha256_after_reservation": sha256_file(LEDGER_PATH),
    }
    with open(os.path.join(os.path.dirname(out_path), "manifest_provenance.json"), "x", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)
    print(f"  manifest {manifest.manifest_id}: {len(manifest)} cases -> {rel(out_path)}")
    print("  seeds reserved in config/experiment_seed_ledger.jsonl")
    return 0


# ---------------------------------------------------------------------------
#  Run
# ---------------------------------------------------------------------------


def structure_specs(cfg) -> list:
    from lares.fitting.structure_bank import load_bank

    bank = {s.structure_id: s for s in load_bank()}
    specs = []
    for sid in cfg["structures"]["bank"]:
        s = bank[sid]
        specs.append({"structure_id": sid, "family": FAMILY_BANK, "source": s.source,
                      "source_path": rel(s.path), "source_sha256": sha256_text(s.source)})
    with open(os.path.join(_PROJECT_ROOT, cfg["structures"]["evolved"]["selection_file"]), encoding="utf-8") as f:
        selection = json.load(f)
    for s in selection["selected"]:
        path = os.path.join(_PROJECT_ROOT, s["source_path"])
        if sha256_file(path) != s["source_sha256"]:
            raise SystemExit(f"{s['source_path']} no longer matches its frozen hash")
        specs.append({"structure_id": s["structure_id"], "family": FAMILY_EVOLVED,
                      "source": open(path, encoding="utf-8").read(), "source_path": s["source_path"],
                      "source_sha256": s["source_sha256"]})
    return specs


def _outcome(result) -> dict:
    progress = [e.signed_goal_progress for e in result.episodes if e.signed_goal_progress is not None]
    labels = {}
    for e in result.episodes:
        labels[e.failure_label] = labels.get(e.failure_label, 0) + 1
    return {
        "manifest_id": result.manifest_id,
        "success_rate": result.success_rate,
        "mean_return": result.mean_reward,
        "mean_signed_goal_progress": float(np.mean(progress)) if progress else None,
        "failure_labels": labels,
        "episodes": len(result.episodes),
        "environment_steps": int(sum(e.length for e in result.episodes)),
        "per_case": {e.case_id: {"success": e.success, "return": e.episode_return,
                                 "signed_goal_progress": e.signed_goal_progress} for e in result.episodes},
    }


def run_structure(job) -> dict:
    """One structure: zero-shot once, then two fits per seed, every condition scored on confirmation."""
    import torch

    torch.set_num_threads(1)
    from lares.core.obs_schema import get_obs_schema
    from lares.core.training_pipeline import DemoBuffer, ensure_mujoco_headless_gl
    from lares.eval import EvaluationManifest, make_manifest_env
    from lares.eval.runner import PolicyActor, evaluate_manifest
    from lares.fitting.checkpoints import RolloutScore
    from lares.fitting.optimizers import (
        SAMPLING_PHASE_BALANCED,
        SAMPLING_UNIFORM,
        fit_policy_with_objective,
        prepare_data,
        state_parameters_sha256,
    )
    from lares.fitting.phases import expert_phase_labels
    from lares.fitting.structure_bank import instantiate_source

    ensure_mujoco_headless_gl()
    cfg, spec, out_dir, git_sha = job["cfg"], job["spec"], job["out_dir"], job["git_sha"]
    fit_cfg = cfg["fitting"]
    schema = get_obs_schema(cfg["env_name"])
    started = time.time()

    confirmation = EvaluationManifest.load(os.path.join(_PROJECT_ROOT, cfg["confirmation_manifest"]["path"]))
    development = EvaluationManifest.load(os.path.join(_PROJECT_ROOT, cfg["checkpoint_selection_manifest"]["path"]))
    start, stop = cfg["checkpoint_selection_manifest"]["slice"]
    selection_manifest = development.slice(int(start), int(stop))
    confirmation_env = make_manifest_env(confirmation, confirmation.resolve_pool())
    probe_env = make_manifest_env(selection_manifest, development.resolve_pool())
    horizon = int(cfg["horizon"])

    buffer = DemoBuffer.load(os.path.join(_PROJECT_ROOT, fit_cfg["demo_buffer"]))
    data = prepare_data(buffer, val_fraction=float(fit_cfg["val_fraction"]), split_seed=int(fit_cfg["split_seed"]))
    labels = expert_phase_labels(data.train_obs.numpy(), schema)
    counters = {"probe_episodes": 0, "probe_environment_steps": 0, "confirmation_episodes": 0,
                "confirmation_environment_steps": 0, "bc_gradient_steps": 0}

    def build():
        policy = instantiate_source(spec["source"], 39, 4, schema)
        policy.fixed_action_scale = float(fit_cfg["fixed_action_scale"])
        return policy

    cache = {}

    def confirm(policy):
        digest = state_parameters_sha256(policy.state_dict())
        if digest not in cache:
            result = evaluate_manifest(PolicyActor(policy, deterministic=True, name=spec["structure_id"]),
                                       confirmation, confirmation_env, max_steps=horizon)
            counters["confirmation_episodes"] += len(result.episodes)
            counters["confirmation_environment_steps"] += int(sum(e.length for e in result.episodes))
            cache[digest] = _outcome(result)
        return digest, cache[digest]

    def probe(policy):
        result = evaluate_manifest(PolicyActor(policy, deterministic=True, name="probe"),
                                   selection_manifest, probe_env, max_steps=horizon)
        counters["probe_episodes"] += len(result.episodes)
        counters["probe_environment_steps"] += int(sum(e.length for e in result.episodes))
        progress = [e.signed_goal_progress for e in result.episodes if e.signed_goal_progress is not None]
        return RolloutScore(result.success_rate, float(np.mean(progress)) if progress else 0.0)

    zero = build()
    zero_digest, zero_outcome = confirm(zero)
    seeds_out = []
    for seed in [int(s) for s in cfg["seeds"]["fitting_seeds"]]:
        per_condition = {ZERO_SHOT: {"parameters_sha256": zero_digest, "selected_step": 0, "shared_across_seeds": True}}
        fits = {}
        for name, sampling in (("mse", SAMPLING_UNIFORM), ("phase_balanced", SAMPLING_PHASE_BALANCED)):
            torch.manual_seed(seed)
            np.random.seed(seed)
            policy = build()
            fit = fit_policy_with_objective(
                policy, data, fit_cfg["objective"], budget=int(fit_cfg["budget"]),
                batch_size=int(fit_cfg["batch_size"]), lr=float(fit_cfg["lr"]),
                clip_grad_norm=float(fit_cfg["clip_grad_norm"]), seed=seed, sampling=sampling,
                phase_labels=labels if sampling == SAMPLING_PHASE_BALANCED else None,
                rollout_probe=probe, checkpoint_steps=[int(s) for s in fit_cfg["checkpoint_steps"]],
                structure_id=spec["structure_id"],
            )
            counters["bc_gradient_steps"] += int(fit.bc_gradient_steps)
            fits[name] = fit
        mse, pb = fits["mse"], fits["phase_balanced"]
        for condition, fit, state, step in (
            (MSE_FIXED, mse, mse.final_state, int(fit_cfg["budget"])),
            (MSE_ROLLOUT, mse, mse.selected_state, int(mse.selected_step)),
            (PHASE_ROLLOUT, pb, pb.selected_state, int(pb.selected_step)),
        ):
            policy = build()
            policy.load_state_dict(state)
            digest, _ = confirm(policy)
            per_condition[condition] = {
                "parameters_sha256": digest, "selected_step": step,
                "checkpoint_history": fit.checkpoint_history, "objective_config": fit.objective_config,
                "frozen_parameters": fit.frozen_parameters,
            }
        seeds_out.append({"seed": seed, "conditions": per_condition})

    result = {
        "stage": "E", "git_sha": git_sha, "structure_id": spec["structure_id"], "family": spec["family"],
        "source_path": spec["source_path"], "source_sha256": spec["source_sha256"],
        "confirmation_manifest_id": confirmation.manifest_id,
        "checkpoint_selection_manifest_id": selection_manifest.manifest_id,
        "outcomes_by_parameters": cache, "seeds": seeds_out, "resources": counters,
        "wall_time_seconds": time.time() - started, "finished_at": now(),
    }
    path = os.path.join(out_dir, "structures", f"{spec['structure_id']}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    return {"structure_id": spec["structure_id"], "path": path, "wall_time_seconds": result["wall_time_seconds"]}


def verify_frozen_inputs(cfg) -> dict:
    """Hashes of everything Stage E reads, checked against what was frozen."""
    manifest_path = os.path.join(_PROJECT_ROOT, cfg["confirmation_manifest"]["path"])
    provenance_path = os.path.join(os.path.dirname(manifest_path), "manifest_provenance.json")
    with open(provenance_path, encoding="utf-8") as f:
        provenance = json.load(f)
    if sha256_file(manifest_path) != provenance["manifest_sha256"]:
        raise SystemExit("the confirmation manifest changed after it was built")
    selection_path = os.path.join(_PROJECT_ROOT, cfg["structures"]["evolved"]["selection_file"])
    if sha256_file(selection_path) != provenance["selection_sha256"]:
        raise SystemExit("the structure selection changed after the manifest was built")
    buffer_path = os.path.join(_PROJECT_ROOT, cfg["fitting"]["demo_buffer"])
    if sha256_file(buffer_path) != cfg["fitting"]["demo_buffer_sha256"]:
        raise SystemExit(f"{cfg['fitting']['demo_buffer']} does not match its frozen hash")
    return {
        "config_sha256": sha256_file(CONFIG_PATH), "manifest_sha256": provenance["manifest_sha256"],
        "selection_sha256": provenance["selection_sha256"], "demo_buffer_sha256": cfg["fitting"]["demo_buffer_sha256"],
        "ledger_sha256": sha256_file(LEDGER_PATH),
    }


def cmd_run(args) -> int:
    import multiprocessing as mp

    git_sha = require_clean_worktree()
    cfg = load_config(args.config)
    frozen = verify_frozen_inputs(cfg)
    specs = structure_specs(cfg)
    run_id = run_stamp()
    out_dir = os.path.join(_PROJECT_ROOT, "logs", "stage_e", run_id)
    os.makedirs(out_dir)
    with open(os.path.join(out_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump({"git_sha": git_sha, "started_at": now(), "frozen_inputs": frozen,
                   "structures": [{k: v for k, v in s.items() if k != "source"} for s in specs],
                   "config": cfg}, f, indent=2)
    print(f"  Stage E run {run_id} at {git_sha[:12]}: {len(specs)} structures x "
          f"{len(cfg['seeds']['fitting_seeds'])} seeds, output {rel(out_dir)}")
    jobs = [{"cfg": cfg, "spec": s, "out_dir": out_dir, "git_sha": git_sha} for s in specs]
    started = time.time()
    with mp.get_context("spawn").Pool(processes=min(int(args.workers), len(jobs))) as pool:
        for done in pool.imap_unordered(run_structure, jobs):
            print(f"  finished {done['structure_id']} in {done['wall_time_seconds']:.0f} s", flush=True)
    with open(os.path.join(out_dir, "run.json"), encoding="utf-8") as f:
        run = json.load(f)
    run.update(finished_at=now(), wall_time_seconds=time.time() - started)
    with open(os.path.join(out_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)
    return report(out_dir)


# ---------------------------------------------------------------------------
#  Report and decision
# ---------------------------------------------------------------------------


def success_of(structure_result, seed_entry, condition) -> float:
    digest = seed_entry["conditions"][condition]["parameters_sha256"]
    return float(structure_result["outcomes_by_parameters"][digest]["success_rate"])


def analyse(structure_results, alpha: float = 0.05) -> dict:
    """The predeclared analysis: structure-level paired differences against zero-shot.

    For each structure and fitted condition, the per-seed difference is that seed's
    confirmation success minus the structure's zero-shot success; the structure's
    difference is the mean over seeds. The interval is a t interval over structures,
    which treats the structure, not the seed or the case, as the unit that varies.
    """
    out = {"conditions": {}, "families": {}, "alpha": alpha}
    per_structure = {c: {} for c in FITTED_CONDITIONS}
    per_pair = {c: [] for c in FITTED_CONDITIONS}
    for r in structure_results:
        for c in FITTED_CONDITIONS:
            diffs = [success_of(r, s, c) - success_of(r, s, ZERO_SHOT) for s in r["seeds"]]
            per_structure[c][r["structure_id"]] = {"family": r["family"], "mean_over_seeds": float(np.mean(diffs)),
                                                   "per_seed": diffs}
            per_pair[c].extend(diffs)
    for c in FITTED_CONDITIONS:
        values = [v["mean_over_seeds"] for v in per_structure[c].values()]
        out["conditions"][c] = {
            "structure_level": t_interval(values, 1 - alpha),
            "per_structure": per_structure[c],
            "structure_seed_pairs": {"n": len(per_pair[c]), "mean": float(np.mean(per_pair[c])),
                                     "median": float(np.median(per_pair[c]))},
        }
        for family in (FAMILY_BANK, FAMILY_EVOLVED):
            fam = [v["mean_over_seeds"] for v in per_structure[c].values() if v["family"] == family]
            out["families"].setdefault(family, {})[c] = t_interval(fam, 1 - alpha)

    p_values = {c: out["conditions"][c]["structure_level"]["p_value"] for c in FITTED_CONDITIONS}
    order = sorted(FITTED_CONDITIONS, key=lambda c: (p_values[c] if p_values[c] is not None else 1.0))
    still = True
    holm = {}
    m = len(order)
    for rank, c in enumerate(order, start=1):
        level = 1 - alpha / (m - rank + 1)
        values = [v["mean_over_seeds"] for v in per_structure[c].values()]
        interval = t_interval(values, level)
        passes = still and interval["mean"] is not None and interval["mean"] > 0 and interval["low"] is not None and interval["low"] > 0
        if not passes:
            still = False
        holm[c] = {"rank": rank, "level": level, "low": interval["low"], "high": interval["high"], "passes": bool(passes)}
    out["holm"] = holm

    beats = [
        c for c in FITTED_CONDITIONS
        if (lambda s: s["mean"] is not None and s["mean"] > 0 and s["low"] is not None and s["low"] > 0)(
            out["conditions"][c]["structure_level"])
    ]
    out["decision"] = {
        "rule": "keep fitting only if at least one fitted condition beats zero-shot with its 95% interval above zero",
        "fitted_conditions_beating_zero_shot": beats,
        "keep_fitting": bool(beats),
        "default_after_stage_e": "fitting enabled" if beats else "fitting disabled by default (bc_steps: 0)",
        "holm_robust": any(holm[c]["passes"] for c in beats),
    }
    return out


def report(run_dir) -> int:
    results = []
    for path in sorted(glob.glob(os.path.join(run_dir, "structures", "*.json"))):
        with open(path, encoding="utf-8") as f:
            results.append(json.load(f))
    if not results:
        raise SystemExit(f"no structure results under {run_dir}")
    shas = {r["git_sha"] for r in results}
    if len(shas) != 1:
        raise SystemExit(f"structure results come from different commits: {shas}")
    analysis = analyse(results)
    analysis["git_sha"] = shas.pop()
    analysis["structures"] = {
        r["structure_id"]: {
            "family": r["family"],
            "zero_shot_success": success_of(r, r["seeds"][0], ZERO_SHOT),
            "by_seed": {s["seed"]: {c: {"success": success_of(r, s, c),
                                        "selected_step": s["conditions"][c]["selected_step"]} for c in CONDITIONS}
                        for s in r["seeds"]},
        }
        for r in results
    }
    analysis["resources"] = {k: sum(int(r["resources"].get(k, 0)) for r in results)
                             for k in ("probe_episodes", "probe_environment_steps", "confirmation_episodes",
                                       "confirmation_environment_steps", "bc_gradient_steps")}
    with open(os.path.join(run_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"\n  Stage E report at {analysis['git_sha'][:12]} ({len(results)} structures)")
    print(f"  {'structure':<28}{'family':<9}{'zero':>6}" + "".join(f"{c:>34}" for c in FITTED_CONDITIONS))
    for sid, s in analysis["structures"].items():
        cells = []
        for c in FITTED_CONDITIONS:
            vals = [v[c]["success"] for v in s["by_seed"].values()]
            steps = sorted({v[c]["selected_step"] for v in s["by_seed"].values()})
            cells.append(f"{np.mean(vals):.3f} (steps {steps})")
        print(f"  {sid:<28}{s['family']:<9}{s['zero_shot_success']:>6.3f}" + "".join(f"{x:>34}" for x in cells))
    for c in FITTED_CONDITIONS:
        sl = analysis["conditions"][c]["structure_level"]
        low = "n/a" if sl["low"] is None else f"{sl['low']:+.3f}"
        high = "n/a" if sl["high"] is None else f"{sl['high']:+.3f}"
        fam = "  ".join(
            f"{f}: {analysis['families'][f][c]['mean']:+.3f}" for f in (FAMILY_BANK, FAMILY_EVOLVED)
            if analysis["families"][f][c]["mean"] is not None
        )
        print(f"  {c:<34} vs zero-shot {sl['mean']:+.3f} t95 [{low}, {high}] median {sl['median']:+.3f}   {fam}   "
              f"Holm {'pass' if analysis['holm'][c]['passes'] else 'fail'}")
    d = analysis["decision"]
    print(f"\n  decision: {'KEEP' if d['keep_fitting'] else 'DISABLE'} fitting by default; "
          f"beating zero-shot: {d['fitted_conditions_beating_zero_shot']}; Holm-robust: {d['holm_robust']}")
    print(f"  written: {rel(os.path.join(run_dir, 'report.json'))}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", default=CONFIG_PATH)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("select")
    sub.add_parser("build-manifest")
    run = sub.add_parser("run")
    run.add_argument("--workers", type=int, default=8)
    rep = sub.add_parser("report")
    rep.add_argument("--run", required=True)
    args = p.parse_args(argv)
    if args.command == "select":
        return cmd_select(args)
    if args.command == "build-manifest":
        return cmd_build_manifest(args)
    if args.command == "run":
        return cmd_run(args)
    return report(args.run)


if __name__ == "__main__":
    sys.exit(main())
