#!/usr/bin/env python
"""R-005: does a bounded LLM mutation beat its own parent when both are fitted the same way?

The two-lane run (``logs/two_lane_search``) fitted each fittable-lane child but never its parent
under the same conditions, so "the child is good" and "the parent fitted is just as good" were
never told apart. This refits every child (from its saved source, at its declared initial values)
and every unique parent (the seed-bank source the child's lineage names) with identical
demonstrations, objective, budget, checkpoints, probe cases and fitting seed. Each checkpoint is
selected by probe rollouts (development 40-49), never by loss. Every selected policy is then scored
on one fresh MT1 manifest from an unused pool seed.

The primary statistic is the paired t interval over placements: for each fresh case, the mean over
the six child-minus-parent success differences. The six pairs share placements, so the 300 episode
differences are not independent; the 50 placement averages are. Per-pair differences are descriptive.

Secondary (R-001, merged): the two-lane winner (``performance_incumbent`` in ``incumbents.json``,
here refitted with the R-005 fitting seed like every child) against the baseline run's best
candidate (the one sent to the promotion test, scored at its saved state) on the same cases.

Seeds are checked against the seed ledger and every saved manifest, then reserved with this
out-dir as the artifact. ``--reserve-only`` does just that, so the reservation can be committed
from a machine without MetaWorld before the run. A rerun into the same out-dir reuses its manifest.

``--research-protocol nested-screen`` (R-007) asks a narrower question of one frozen pair: would a 30-case
screen keep a beneficial child that a 10-case screen rejects? Child and parent are refitted identically (checkpoint
chosen on a fresh 10-case probe), both are scored on a fresh 30-case screen and a fresh 100-case confirmation
manifest, and Stage F's acceptance rule is replayed on 1000 seeded 10-case subsets of the screen and on the whole
screen. The statistic is the confirmation success of the controller each screen size selects, paired over the 100
confirmation placements (replays are averaged within a placement; they are not replicates).

Usage (from the project root)::

    python scripts/compare_child_vs_parent.py --run logs/two_lane_search \\
        --baseline logs/baseline_gpt61_sol_high --pool-seed 2610 --fit-seed 2611 \\
        --out-dir logs/child_vs_parent_2610
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402

from lares.eval.stats import t_interval, win_loss_tie  # noqa: E402
from lares.utils.cli import load_yaml  # noqa: E402
from lares.utils.clock import now_iso as now  # noqa: E402
from lares.utils.gitstate import rel  # noqa: E402
from lares.utils.hashing import sha256_file  # noqa: E402
from lares.utils.seeds import SEED_LEDGER, read_ledger, reserve_seeds, seed_conflicts  # noqa: E402

LANE_FITTABLE = "fittable"
FIT_KEYS = ("objective", "sampling", "budget_gradient_steps", "batch_size", "lr", "clip_grad_norm",
            "checkpoint_steps", "fixed_action_scale", "val_fraction", "split_seed")
DEMO_BUFFER = "logs/baseline_lock/demo_push-v2.pkl"


def _records(run_dir: str) -> list:
    out = []
    for path in sorted(glob.glob(os.path.join(run_dir, "experiments", "*.json"))):
        with open(path, encoding="utf-8") as f:
            out.append(json.load(f))
    if not out:
        raise SystemExit(f"no experiment records under {run_dir}")
    return out


# ---------------------------------------------------------------------------
#  What is compared: read from the saved records, never re-derived
# ---------------------------------------------------------------------------


def discover_pairs(run_dir: str) -> list:
    """One entry per evaluated fittable-lane child: its source, its parent's source, its fitting."""
    pairs = []
    for r in _records(run_dir):
        lineage = r.get("lineage") or {}
        if lineage.get("lane") != LANE_FITTABLE or not r.get("fitting") or not r["code"].get("source_path"):
            continue
        reference = lineage["lane_metadata"]["reference"]
        parent_path = os.path.join(_PROJECT_ROOT, reference["source_path"])
        if sha256_file(parent_path) != reference["source_sha256"]:
            raise SystemExit(f"{r['experiment_id']}: parent {reference['source_path']} changed since the run")
        child_path = os.path.join(run_dir, r["code"]["source_path"])
        if sha256_file(child_path) != r["code"]["source_sha256"]:
            raise SystemExit(f"{r['experiment_id']}: child source changed since the run")
        pairs.append({
            "child_id": r["experiment_id"], "candidate_id": r["candidate_id"],
            "child_source_path": child_path, "child_source_sha256": r["code"]["source_sha256"],
            "child_initial_state": os.path.join(run_dir, r["fitting"]["initial_state"]["path"]),
            "parent_id": reference["structure_id"], "parent_source_path": parent_path,
            "parent_source_sha256": reference["source_sha256"],
            "scope": (lineage["lane_metadata"].get("mutation_scope") or {}).get("id"),
            "objective_config": r["fitting"]["objective_config"],
            "probe_manifest_id": r["fitting"]["probe_manifest_id"],
            "run_selected_step": r["fitting"]["selected_step"],
        })
    if not pairs:
        raise SystemExit(f"no evaluated fittable-lane children in {run_dir}")
    return pairs


def common_fitting(pairs: list) -> dict:
    """The fitting settings every child used; refuses a run whose children were fitted differently."""
    settings = {k: pairs[0]["objective_config"][k] for k in FIT_KEYS}
    for p in pairs[1:]:
        other = {k: p["objective_config"][k] for k in FIT_KEYS}
        if other != settings:
            raise SystemExit(f"{p['child_id']} was fitted differently: {other} vs {settings}")
    probes = {p["probe_manifest_id"] for p in pairs}
    if len(probes) != 1:
        raise SystemExit(f"children used different probe manifests: {probes}")
    return {**settings, "probe_manifest_id": probes.pop()}


def performance_incumbent(run_dir: str) -> str:
    with open(os.path.join(run_dir, "incumbents.json"), encoding="utf-8") as f:
        return json.load(f)["performance_incumbent"]["attempt_id"]


def baseline_best(baseline_dir: str) -> dict:
    """The baseline candidate its own run sent to the promotion test: its best screened challenger."""
    tested = [r for r in _records(baseline_dir)
              if ((r.get("evaluation") or {}).get("screening") or {}).get("promotion_tested")]
    if len(tested) != 1:
        raise SystemExit(f"expected one promotion-tested baseline candidate, found {[r['experiment_id'] for r in tested]}")
    r = tested[0]
    return {"id": r["experiment_id"], "source_path": os.path.join(baseline_dir, r["code"]["source_path"]),
            "state_path": os.path.join(baseline_dir, r["fitting"]["selected_state"]["path"]),
            "selected_step": r["fitting"]["selected_step"]}


# ---------------------------------------------------------------------------
#  Statistics: placement-level pairing
# ---------------------------------------------------------------------------


def placement_differences(per_case: dict, pairs: list) -> dict:
    """``{case_id: mean over pairs of child success - parent success}``.

    ``per_case`` maps a policy id to ``{case_id: success}``; every policy must cover the same cases.
    """
    cases = sorted(per_case[pairs[0]["child_id"]])
    for pid in {p["child_id"] for p in pairs} | {p["parent_id"] for p in pairs}:
        if sorted(per_case[pid]) != cases:
            raise ValueError(f"{pid} was not scored on the same cases")
    return {c: float(np.mean([float(per_case[p["child_id"]][c]) - float(per_case[p["parent_id"]][c])
                              for p in pairs])) for c in cases}


def paired(per_case: dict, a: str, b: str) -> dict:
    diffs = [float(per_case[a][c]) - float(per_case[b][c]) for c in sorted(per_case[b])]
    return {"a": a, "b": b, "n": len(diffs), **t_interval(diffs), **win_loss_tie(diffs)}


def summarise(per_case: dict, pairs: list, winner: str, baseline_id: str) -> dict:
    by_case = placement_differences(per_case, pairs)
    primary = t_interval(list(by_case.values()))
    success = {pid: float(np.mean([float(v) for v in cases.values()])) for pid, cases in per_case.items()}
    return {
        "paired_success_difference": primary["mean"],
        "primary": {"statistic": "mean over placements of the mean child-minus-parent success over pairs",
                    "n_placements": len(by_case), "n_pairs": len(pairs), **primary},
        "per_pair": [{"child": p["child_id"], "parent": p["parent_id"], "scope": p["scope"],
                      "child_success": success[p["child_id"]], "parent_success": success[p["parent_id"]],
                      **paired(per_case, p["child_id"], p["parent_id"])} for p in pairs],
        "mean_child_success": float(np.mean([success[p["child_id"]] for p in pairs])),
        "mean_parent_success": float(np.mean([success[p["parent_id"]] for p in pairs])),
        "secondary_r001": paired(per_case, winner, baseline_id) if baseline_id in per_case else None,
        "success_by_policy": success,
        "placement_differences": by_case,
    }


# ---------------------------------------------------------------------------
#  Seeds and manifest
# ---------------------------------------------------------------------------


def ensure_reserved(pool_seed: int, fit_seed: int, out_dir: str, ledger_path: str = SEED_LEDGER) -> str:
    """Reserve both seeds for this out-dir, or confirm they already are. Refuses any other use."""
    return ensure_entries_reserved([(pool_seed, "mt1_pool", "R-005 child vs fitted parent fresh manifest"),
                                    (fit_seed, "fitting", "R-005 child and parent refits")], out_dir, ledger_path)


def ensure_entries_reserved(entries, out_dir: str, ledger_path: str = SEED_LEDGER) -> str:
    """``entries`` is ``[(seed, namespace, use)]``; reserve them all for this out-dir or confirm they already are."""
    artifact = rel(out_dir)
    mine = {(int(e["seed"]), e["namespace"]) for e in read_ledger(ledger_path) if e.get("artifact") == artifact}
    wanted = {(int(seed), ns) for seed, ns, _ in entries}
    if mine == wanted:
        return "already reserved for this out-dir"
    if not mine <= wanted:
        raise SystemExit(f"the ledger holds a different reservation for {artifact}: {sorted(mine - wanted)}")
    # A reservation may be extended with new seeds (never changed); only the new ones are checked and written.
    entries = [(seed, ns, use) for seed, ns, use in entries if (int(seed), ns) not in mine]
    pools = [seed for seed, ns, _ in entries if ns == "mt1_pool"]
    fitting = [seed for seed, ns, _ in entries if ns != "mt1_pool"]
    conflicts = seed_conflicts(pools, fitting, ledger_path=ledger_path)
    if conflicts:
        raise SystemExit("seeds already used:\n  " + "\n  ".join(conflicts))
    stamp = now()
    reserve_seeds([{"seed": int(seed), "namespace": ns, "use": use, "artifact": artifact, "recorded_at": stamp,
                    "backfilled": False} for seed, ns, use in entries], ledger_path=ledger_path)
    return "reserved" if not mine else f"extended with {sorted(int(seed) for seed, _, _ in entries)}"


def fresh_manifest(env_name: str, pool_seed: int, episodes: int, horizon: int, out_dir: str, role: str = "confirmation",
                   also_disjoint_from=()):
    from lares.eval.manifest import SPLIT_DEVELOPMENT, EvaluationManifest, assert_disjoint, build_manifest

    pool_seeds = [int(x) for x in (pool_seed if isinstance(pool_seed, (list, tuple)) else [pool_seed])]
    label = "-".join(str(x) for x in pool_seeds)
    path = os.path.join(out_dir, f"{env_name}_{role}_s{label}.yaml")
    manifest_id = f"{env_name}:development_{role}:s{label}:n{episodes}"
    if os.path.exists(path):
        manifest = EvaluationManifest.load(path)
        if manifest.manifest_id != manifest_id:
            raise SystemExit(f"{rel(path)} holds {manifest.manifest_id}, not {manifest_id}")
        return manifest, path
    manifest, _ = build_manifest(env_name, SPLIT_DEVELOPMENT, pool_seeds, horizon=horizon,
                                 num_episodes=episodes, manifest_id=manifest_id)
    known = {}
    for pattern in ("config/manifests/*.yaml", "config/**/*confirmation*.yaml", "logs/**/*confirmation*.yaml"):
        for p in glob.glob(os.path.join(_PROJECT_ROOT, pattern), recursive=True):
            m = EvaluationManifest.load(p)
            known.setdefault(m.manifest_id, m)
    for m in also_disjoint_from:
        known.setdefault(m.manifest_id, m)
    assert_disjoint(list(known.values()) + [manifest])
    manifest.save(path)
    return manifest, path


# ---------------------------------------------------------------------------
#  One job: fit (or not), then score on the fresh manifest
# ---------------------------------------------------------------------------


def run_job(job) -> dict:
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
    started = time.time()
    schema = get_obs_schema(job["env_name"])
    fit_cfg, horizon = job["fitting"], int(job["horizon"])
    with open(job["source_path"], encoding="utf-8") as f:
        source = f.read()
    policy = instantiate_source(source, 39, 4, schema)
    policy.fixed_action_scale = float(fit_cfg["fixed_action_scale"])
    counters = {"probe_episodes": 0, "evaluation_episodes": 0, "bc_gradient_steps": 0}
    out = {"policy_id": job["policy_id"], "role": job["role"], "source_path": rel(job["source_path"]),
           "source_sha256": sha256_file(job["source_path"])}

    if job.get("check_state"):
        saved = torch.load(job["check_state"], weights_only=True)
        out["initial_state_matches_run"] = state_parameters_sha256(saved) == state_parameters_sha256(policy.state_dict())
        if not out["initial_state_matches_run"]:
            raise RuntimeError(f"{job['policy_id']}: source does not rebuild the run's initial state")
    if job.get("load_state"):
        policy.load_state_dict(torch.load(job["load_state"], weights_only=True))
        out["state_path"] = rel(job["load_state"])

    if job["fit"]:
        development = EvaluationManifest.load(job["development_manifest"])
        probe_manifest = development.tail(10)
        if probe_manifest.manifest_id != fit_cfg["probe_manifest_id"]:
            raise RuntimeError(f"probe manifest {probe_manifest.manifest_id} is not the run's {fit_cfg['probe_manifest_id']}")
        probe_env = make_manifest_env(probe_manifest, development.resolve_pool())

        def probe(pol):
            result = evaluate_manifest(PolicyActor(pol, deterministic=True, name="probe"), probe_manifest, probe_env,
                                       max_steps=horizon)
            counters["probe_episodes"] += len(result.episodes)
            progress = [e.signed_goal_progress for e in result.episodes if e.signed_goal_progress is not None]
            return RolloutScore(result.success_rate, float(np.mean(progress)) if progress else 0.0)

        data = prepare_data(DemoBuffer.load(job["demo_buffer"]), val_fraction=float(fit_cfg["val_fraction"]),
                            split_seed=int(fit_cfg["split_seed"]))
        fit = fit_policy_with_objective(
            policy, data, fit_cfg["objective"], budget=int(fit_cfg["budget_gradient_steps"]),
            batch_size=min(int(fit_cfg["batch_size"]), int(data.train_obs.shape[0])), lr=float(fit_cfg["lr"]),
            clip_grad_norm=float(fit_cfg["clip_grad_norm"]), seed=int(job["fit_seed"]), sampling=fit_cfg["sampling"],
            rollout_probe=probe, checkpoint_steps=[int(s) for s in fit_cfg["checkpoint_steps"]],
            structure_id=job["policy_id"],
        )
        policy.load_state_dict(fit.selected_state)
        counters["bc_gradient_steps"] = int(fit.bc_gradient_steps)
        state_path = os.path.join(job["out_dir"], "states", f"{job['policy_id']}.pt")
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        torch.save(fit.selected_state, state_path)
        out.update(selected_step=int(fit.selected_step), fitting_rejected=int(fit.selected_step) == 0,
                   checkpoint_history=fit.checkpoint_history, objective_config=fit.objective_config,
                   state_path=rel(state_path))

    manifest = EvaluationManifest.load(job["manifest_path"])
    env = make_manifest_env(manifest, manifest.resolve_pool())
    result = evaluate_manifest(PolicyActor(policy, deterministic=True, name=job["policy_id"]), manifest, env,
                               max_steps=horizon)
    counters["evaluation_episodes"] += len(result.episodes)
    labels = {}
    for e in result.episodes:
        labels[e.failure_label] = labels.get(e.failure_label, 0) + 1
    out.update(
        parameters_sha256=state_parameters_sha256(policy.state_dict()), manifest_id=manifest.manifest_id,
        success_rate=result.success_rate, mean_return=result.mean_reward, failure_labels=labels,
        per_case={e.case_id: {"success": bool(e.success), "return": e.episode_return,
                              "signed_goal_progress": e.signed_goal_progress} for e in result.episodes},
        resources=counters, wall_time_seconds=time.time() - started,
    )
    path = os.path.join(job["out_dir"], "policies", f"{job['policy_id']}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    return out


def build_jobs(pairs, fitting, baseline, args, manifest_path) -> list:
    common = {"env_name": args.env_name, "horizon": args.horizon, "fitting": fitting, "fit_seed": args.fit_seed,
              "out_dir": os.path.abspath(args.out_dir), "manifest_path": os.path.abspath(manifest_path),
              "demo_buffer": os.path.join(_PROJECT_ROOT, args.demo_buffer),
              "development_manifest": os.path.join(_PROJECT_ROOT, "config", "manifests", f"{args.env_name}_development.yaml")}
    jobs = [{**common, "policy_id": p["child_id"], "role": "child", "fit": True,
             "source_path": p["child_source_path"], "check_state": p["child_initial_state"]} for p in pairs]
    parents = {}
    for p in pairs:
        parents.setdefault(p["parent_id"], p["parent_source_path"])
    jobs += [{**common, "policy_id": pid, "role": "parent", "fit": True, "source_path": path}
             for pid, path in parents.items()]
    if baseline:
        jobs.append({**common, "policy_id": f"baseline:{baseline['id']}", "role": "baseline", "fit": False,
                     "source_path": baseline["source_path"], "load_state": baseline["state_path"]})
    return jobs


# ---------------------------------------------------------------------------
#  R-007: nested screen replay
# ---------------------------------------------------------------------------

BRANCH_LOWER_SUCCESS = "child_lower_success"
BRANCH_TIED_SUCCESS_LOWER_PROGRESS = "tied_success_lower_or_equal_progress"
BRANCH_ACCEPTED = "accepted"


def screen_decision(child: dict, parent: dict, cases) -> str:
    """Stage F's acceptance rule on ``cases``: the child replaces the parent only if strictly better by the
    checkpoint tie rule (success, then mean signed progress); an exact tie keeps the parent.

    ``child``/``parent`` map case id to ``(success, progress)``.
    """
    from lares.fitting.checkpoints import RolloutScore, is_better_checkpoint

    def score(side):
        return RolloutScore(float(np.mean([side[c][0] for c in cases])), float(np.mean([side[c][1] for c in cases])))

    c, q = score(child), score(parent)
    if is_better_checkpoint(c, q):
        return BRANCH_ACCEPTED
    return BRANCH_LOWER_SUCCESS if c.success < q.success else BRANCH_TIED_SUCCESS_LOWER_PROGRESS


def nested_screen_analysis(child_screen, parent_screen, child_conf, parent_conf, replays: int, seed: int,
                           subset: int = 10) -> dict:
    """Selected-controller confirmation success under a full screen versus seeded ``subset``-case replays.

    Per confirmation placement c: the full screen selects one controller S30; the replays select the child with
    frequency p10, so the expected success under a 10-case screen is p10 * child_c + (1 - p10) * parent_c. The
    paired difference is success(S30, c) minus that, with a t interval over placements.
    """
    cases = sorted(child_screen)
    if sorted(parent_screen) != cases or sorted(child_conf) != sorted(parent_conf):
        raise ValueError("child and parent were not scored on the same cases")
    rng = np.random.default_rng(int(seed))
    branches = {BRANCH_ACCEPTED: 0, BRANCH_LOWER_SUCCESS: 0, BRANCH_TIED_SUCCESS_LOWER_PROGRESS: 0}
    for _ in range(int(replays)):
        draw = [cases[i] for i in sorted(rng.choice(len(cases), size=int(subset), replace=False))]
        branches[screen_decision(child_screen, parent_screen, draw)] += 1
    p10 = branches[BRANCH_ACCEPTED] / float(replays)
    full = screen_decision(child_screen, parent_screen, cases)
    conf_cases = sorted(child_conf)
    selected30 = child_conf if full == BRANCH_ACCEPTED else parent_conf
    diffs = [float(selected30[c]) - (p10 * float(child_conf[c]) + (1 - p10) * float(parent_conf[c])) for c in conf_cases]
    child_minus_parent = [float(child_conf[c]) - float(parent_conf[c]) for c in conf_cases]
    return {
        "paired_success_difference": float(np.mean(diffs)),
        "selected_controller_difference": {"n": len(diffs), **t_interval(diffs)},
        "full_screen_decision": full,
        "ten_case_acceptance_rate": p10,
        "ten_case_branches": branches,
        "replays": int(replays), "replay_seed": int(seed), "subset": int(subset),
        "refitted_child_minus_parent_confirmation": {"n": len(child_minus_parent), **t_interval(child_minus_parent),
                                                      **win_loss_tie(child_minus_parent)},
        "screen_success": {"child": float(np.mean([child_screen[c][0] for c in cases])),
                           "parent": float(np.mean([parent_screen[c][0] for c in cases]))},
        "confirmation_success": {"child": float(np.mean([float(child_conf[c]) for c in conf_cases])),
                                 "parent": float(np.mean([float(parent_conf[c]) for c in conf_cases]))},
    }


def with_ranges(policy, overrides: dict):
    """Install changed ranges on one instance: ``clip_params`` and every reader see them; code is untouched.

    A zero-width range ``(v, v)`` clamps a parameter at ``v`` for the whole fit.
    """
    base = dict(policy.get_param_ranges())
    base.update({k: (float(lo), float(hi)) for k, (lo, hi) in overrides.items()})
    policy.get_param_ranges = lambda _ranges=base: dict(_ranges)
    return policy


def nested_job(job) -> dict:
    """Refit one controller on the shared demonstrations, select on the fresh probe, and score it on every manifest
    in ``job["evaluate"]`` (name -> path). With ``fit: False`` the controller is scored at its declared values."""
    import torch

    torch.set_num_threads(1)
    from lares.core.obs_schema import get_obs_schema
    from lares.core.training_pipeline import ensure_mujoco_headless_gl
    from lares.eval import EvaluationManifest, make_manifest_env
    from lares.eval.runner import PolicyActor, evaluate_manifest
    from lares.fitting.optimizers import state_parameters_sha256
    from lares.fitting.structure_bank import instantiate_source

    ensure_mujoco_headless_gl()
    started = time.time()
    fit_cfg, horizon = job["fitting"], int(job["horizon"])
    with open(job["source_path"], encoding="utf-8") as f:
        policy = instantiate_source(f.read(), 39, 4, get_obs_schema(job["env_name"]))
    policy.fixed_action_scale = float(fit_cfg["fixed_action_scale"])
    if job.get("clamp"):
        values = {n: float(p.detach().reshape(-1)[0]) for n, p in policy.named_parameters()}
        with_ranges(policy, {n: (values[n], values[n]) for n in job["clamp"]})
    if job.get("check_state"):
        if state_parameters_sha256(torch.load(job["check_state"], weights_only=True)) != \
                state_parameters_sha256(policy.state_dict()):
            raise RuntimeError(f"{job['policy_id']}: source does not rebuild the run's initial state")
    counters = {"probe_episodes": 0, **{f"{name}_episodes": 0 for name in job["evaluate"]}}
    out = {"policy_id": job["policy_id"], "role": job["role"], "source_path": rel(job["source_path"]),
           "source_sha256": sha256_file(job["source_path"]), "fit_seed": job.get("fit_seed")}
    if job.get("fit", True):
        out.update(_refit(job, policy, counters))
    else:
        out.update(selected_step=0, checkpoint_history=[])
    out["parameters_sha256"] = state_parameters_sha256(policy.state_dict())
    state_path = os.path.join(job["out_dir"], "states", f"{job.get('label', job['policy_id'])}.pt")
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    torch.save(policy.state_dict(), state_path)
    out["state_path"] = rel(state_path)
    for name, path in job["evaluate"].items():
        manifest = EvaluationManifest.load(path)
        env = make_manifest_env(manifest, manifest.resolve_pool())
        result = evaluate_manifest(PolicyActor(policy, deterministic=True, name=job["policy_id"]), manifest, env,
                                   max_steps=horizon)
        counters[f"{name}_episodes"] += len(result.episodes)
        out[name] = {"manifest_id": manifest.manifest_id, "success_rate": result.success_rate,
                     "per_case": {e.case_id: [float(e.success), float(e.signed_goal_progress)
                                              if e.signed_goal_progress is not None else 0.0]
                                  for e in result.episodes}}
    out.update(resources=counters, wall_time_seconds=time.time() - started)
    path = os.path.join(job["out_dir"], "policies", f"{job.get('label', job['policy_id'])}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    return out


def _refit(job, policy, counters) -> dict:
    """Rollout-selected refit on the job's probe; leaves ``policy`` holding the selected state."""
    from lares.core.training_pipeline import DemoBuffer
    from lares.eval import EvaluationManifest, make_manifest_env
    from lares.eval.runner import PolicyActor, evaluate_manifest
    from lares.fitting.checkpoints import RolloutScore
    from lares.fitting.optimizers import fit_policy_with_objective, prepare_data

    fit_cfg, horizon = job["fitting"], int(job["horizon"])
    probe = EvaluationManifest.load(job["probe_manifest"])
    probe_env = make_manifest_env(probe, probe.resolve_pool())

    def score(pol):
        result = evaluate_manifest(PolicyActor(pol, deterministic=True, name="probe"), probe, probe_env,
                                   max_steps=horizon)
        counters["probe_episodes"] += len(result.episodes)
        progress = [e.signed_goal_progress for e in result.episodes if e.signed_goal_progress is not None]
        return RolloutScore(result.success_rate, float(np.mean(progress)) if progress else 0.0)

    data = prepare_data(DemoBuffer.load(job["demo_buffer"]), val_fraction=float(fit_cfg["val_fraction"]),
                        split_seed=int(fit_cfg["split_seed"]))
    fit = fit_policy_with_objective(
        policy, data, fit_cfg["objective"], budget=int(fit_cfg["budget_gradient_steps"]),
        batch_size=min(int(fit_cfg["batch_size"]), int(data.train_obs.shape[0])), lr=float(fit_cfg["lr"]),
        clip_grad_norm=float(fit_cfg["clip_grad_norm"]), seed=int(job["fit_seed"]), sampling=fit_cfg["sampling"],
        rollout_probe=score, checkpoint_steps=[int(s) for s in job.get("checkpoint_steps") or fit_cfg["checkpoint_steps"]],
        structure_id=job["policy_id"],
    )
    policy.load_state_dict(fit.selected_state)
    return {"selected_step": int(fit.selected_step), "checkpoint_history": fit.checkpoint_history}


def run_nested_screen(args) -> int:
    import multiprocessing as mp

    pairs = {q["child_id"]: q for q in discover_pairs(args.run)}
    if args.child not in pairs:
        raise SystemExit(f"{args.child} is not an evaluated fittable-lane child of {args.run}: {sorted(pairs)}")
    pair, fitting = pairs[args.child], common_fitting(list(pairs.values()))
    os.makedirs(args.out_dir, exist_ok=True)
    status = ensure_entries_reserved([
        (args.screen_seed, "mt1_pool", "R-007 nested screen: 30-case screen manifest"),
        *[(x, "mt1_pool", "R-007 nested screen: 100-case confirmation manifest") for x in args.heldout_seed],
        (args.probe_seed, "mt1_pool", "R-007 nested screen: 10-case checkpoint probe"),
        (args.fit_seed, "fitting", "R-007 nested screen: child and parent refits"),
        (args.replay_seed, "replay", "R-007 nested screen: ten-case subset replays"),
    ], args.out_dir)
    print(f"  pair {pair['child_id']} vs {pair['parent_id']}; seeds: {status}")
    if args.reserve_only:
        return 0
    from lares.core.training_pipeline import ensure_mujoco_headless_gl

    ensure_mujoco_headless_gl()
    probe, probe_path = fresh_manifest(args.env_name, args.probe_seed, 10, args.horizon, args.out_dir, role="probe")
    screen, screen_path = fresh_manifest(args.env_name, args.screen_seed, 30, args.horizon, args.out_dir,
                                         role="screen", also_disjoint_from=[probe])
    conf, conf_path = fresh_manifest(args.env_name, args.heldout_seed, 100, args.horizon, args.out_dir,
                                     role="confirmation", also_disjoint_from=[probe, screen])
    common = {"env_name": args.env_name, "horizon": args.horizon, "fitting": fitting, "fit_seed": args.fit_seed,
              "out_dir": os.path.abspath(args.out_dir), "demo_buffer": os.path.join(_PROJECT_ROOT, args.demo_buffer),
              "probe_manifest": os.path.abspath(probe_path),
              "evaluate": {"screen": os.path.abspath(screen_path), "confirmation": os.path.abspath(conf_path)}}
    jobs = [{**common, "policy_id": pair["child_id"], "role": "child", "source_path": pair["child_source_path"],
             "check_state": pair["child_initial_state"]},
            {**common, "policy_id": pair["parent_id"], "role": "parent", "source_path": pair["parent_source_path"]}]
    with open(os.path.join(args.out_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump({"started_at": now(), "argv": sys.argv, "pair": {k: v for k, v in pair.items()
                                                                   if k != "objective_config"},
                   "fitting": fitting, "seeds": {"screen": args.screen_seed, "confirmation": args.heldout_seed,
                                                 "probe": args.probe_seed, "fit": args.fit_seed,
                                                 "replay": args.replay_seed},
                   "manifests": {"probe": probe.manifest_id, "screen": screen.manifest_id,
                                 "confirmation": conf.manifest_id},
                   "demo_buffer_sha256": sha256_file(common["demo_buffer"])}, f, indent=2, default=str)
    with mp.get_context("spawn").Pool(processes=2) as pool:
        child, parent = pool.map(nested_job, jobs)
    summary = nested_screen_analysis(
        {c: tuple(v) for c, v in child["screen"]["per_case"].items()},
        {c: tuple(v) for c, v in parent["screen"]["per_case"].items()},
        {c: v[0] for c, v in child["confirmation"]["per_case"].items()},
        {c: v[0] for c, v in parent["confirmation"]["per_case"].items()},
        replays=args.replays, seed=args.replay_seed)
    summary.update(child=pair["child_id"], parent=pair["parent_id"],
                   selected_steps={"child": child["selected_step"], "parent": parent["selected_step"]},
                   resources={k: child["resources"][k] + parent["resources"][k] for k in child["resources"]})
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    d, r = summary["selected_controller_difference"], summary["refitted_child_minus_parent_confirmation"]
    print(f"  screen success child {summary['screen_success']['child']:.3f} parent {summary['screen_success']['parent']:.3f}; "
          f"full screen: {summary['full_screen_decision']}; 10-case acceptance {summary['ten_case_acceptance_rate']:.3f} "
          f"{summary['ten_case_branches']}")
    print(f"  selected-controller difference (30 vs 10): {d['mean']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}]")
    print(f"  refitted child - parent on confirmation: {r['mean']:+.3f} [{r['low']:+.3f}, {r['high']:+.3f}] "
          f"w/l/t {r['wins']}/{r['losses']}/{r['ties']}; resources {summary['resources']}")
    return 0


# ---------------------------------------------------------------------------
#  R-010: does the child's advantage survive across fitting seeds?
# ---------------------------------------------------------------------------


def seed_robustness_analysis(child, parent, child_step0, parent_step0, rule) -> dict:
    """``child``/``parent`` map fit seed -> {case: success}; ``*_step0`` map case -> success at declared values.

    Primary: per placement, the mean over seeds of child minus parent, t over placements. Beside it: one
    difference per seed, t over seeds (fitting robustness), and the unchanged qualification rule applied to the
    child's own per-seed fitted-minus-step-0 differences (report only).
    """
    from lares.fitting.replicates import qualification_checks

    seeds = sorted(child)
    if sorted(parent) != seeds:
        raise ValueError("child and parent were not fitted under the same seeds")
    cases = sorted(child_step0)
    for side in (child, parent):
        for k in seeds:
            if sorted(side[k]) != cases:
                raise ValueError("every fit must be scored on the same confirmation cases")
    per_case = [float(np.mean([child[k][c] - parent[k][c] for k in seeds])) for c in cases]
    rate = lambda d: float(np.mean([float(v) for v in d.values()]))  # noqa: E731
    per_seed = {int(k): rate(child[k]) - rate(parent[k]) for k in seeds}
    child_gain = [rate(child[k]) - rate(child_step0) for k in seeds]
    return {
        "paired_success_difference": float(np.mean(per_case)),
        "placement_level": {"n": len(cases), **t_interval(per_case), **win_loss_tie(per_case)},
        "seed_level": {"by_seed": per_seed, **t_interval(list(per_seed.values())),
                       **win_loss_tie(list(per_seed.values()))},
        "success_by_seed": {"child": {int(k): rate(child[k]) for k in seeds},
                            "parent": {int(k): rate(parent[k]) for k in seeds}},
        "step0_success": {"child": rate(child_step0), "parent": rate(parent_step0)},
        "child_qualification_report_only": {"fitted_minus_step0_by_seed": child_gain,
                                            **qualification_checks(child_gain, rule)},
    }


def run_seed_robustness(args) -> int:
    import multiprocessing as mp

    pairs = {q["child_id"]: q for q in discover_pairs(args.run)}
    if args.child not in pairs:
        raise SystemExit(f"{args.child} is not an evaluated fittable-lane child of {args.run}: {sorted(pairs)}")
    pair, fitting = pairs[args.child], common_fitting(list(pairs.values()))
    os.makedirs(args.out_dir, exist_ok=True)
    status = ensure_entries_reserved(
        [(x, "fitting", "R-010 geometry child seed robustness: fit seed") for x in args.fit_seeds]
        + [(args.probe_seed, "mt1_pool", "R-010: 10-case checkpoint probe")]
        + [(x, "mt1_pool", "R-010: confirmation manifest") for x in args.heldout_seed], args.out_dir)
    print(f"  pair {pair['child_id']} vs {pair['parent_id']}; seeds: {status}")
    if args.reserve_only:
        return 0
    from lares.core.training_pipeline import ensure_mujoco_headless_gl

    ensure_mujoco_headless_gl()
    probe, probe_path = fresh_manifest(args.env_name, args.probe_seed, 10, args.horizon, args.out_dir, role="probe")
    conf, conf_path = fresh_manifest(args.env_name, args.heldout_seed, 50 * len(args.heldout_seed), args.horizon,
                                     args.out_dir, role="confirmation", also_disjoint_from=[probe])
    rule = load_yaml(os.path.join(_PROJECT_ROOT, "config", "fittable_bank", "qualification_v2.yaml")
                     )["qualification_rule"]["enters_fittable_seed_bank_if"]
    common = {"env_name": args.env_name, "horizon": args.horizon, "fitting": fitting,
              "out_dir": os.path.abspath(args.out_dir), "demo_buffer": os.path.join(_PROJECT_ROOT, args.demo_buffer),
              "probe_manifest": os.path.abspath(probe_path), "evaluate": {"confirmation": os.path.abspath(conf_path)}}
    sides = {"child": (pair["child_id"], pair["child_source_path"]), "parent": (pair["parent_id"], pair["parent_source_path"])}
    jobs = []
    for role, (pid, src) in sides.items():
        extra = {"check_state": pair["child_initial_state"]} if role == "child" else {}
        jobs.append({**common, **extra, "policy_id": pid, "role": role, "source_path": src, "fit": False,
                     "label": f"{pid}_step0"})
        jobs += [{**common, **extra, "policy_id": pid, "role": role, "source_path": src, "fit_seed": k,
                  "label": f"{pid}_fit{k}"} for k in args.fit_seeds]
    with open(os.path.join(args.out_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump({"started_at": now(), "argv": sys.argv, "pair": {k: v for k, v in pair.items() if k != "objective_config"},
                   "fitting": fitting, "fit_seeds": args.fit_seeds, "rule": rule,
                   "manifests": {"probe": probe.manifest_id, "confirmation": conf.manifest_id},
                   "demo_buffer_sha256": sha256_file(common["demo_buffer"])}, f, indent=2, default=str)
    with mp.get_context("spawn").Pool(processes=min(int(args.workers), len(jobs))) as pool:
        results = pool.map(nested_job, jobs)
    conf_of = lambda r: {c: v[0] for c, v in r["confirmation"]["per_case"].items()}  # noqa: E731
    fitted = {role: {r["fit_seed"]: conf_of(r) for r in results if r["role"] == role and r["fit_seed"] is not None}
              for role in sides}
    step0 = {r["role"]: conf_of(r) for r in results if r["fit_seed"] is None}
    summary = seed_robustness_analysis(fitted["child"], fitted["parent"], step0["child"], step0["parent"], rule)
    summary.update(child=pair["child_id"], parent=pair["parent_id"],
                   selected_steps={f"{r['role']}_{r['fit_seed']}": r["selected_step"] for r in results},
                   resources={k: int(sum(r["resources"].get(k, 0) for r in results))
                              for k in ("probe_episodes", "confirmation_episodes")})
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    pl, sl = summary["placement_level"], summary["seed_level"]
    print(f"  child {summary['success_by_seed']['child']}\n  parent {summary['success_by_seed']['parent']}")
    print(f"  placement-level child - parent: {pl['mean']:+.3f} [{pl['low']:+.3f}, {pl['high']:+.3f}]; "
          f"seed-level: {sl['mean']:+.3f} [{sl['low']:+.3f}, {sl['high']:+.3f}] w/l/t {sl['wins']}/{sl['losses']}/{sl['ties']}")
    q = summary["child_qualification_report_only"]
    print(f"  child qualification (report only): qualifies={q['qualifies']} gains {q['fitted_minus_step0_by_seed']}; "
          f"resources {summary['resources']}")
    return 0


# ---------------------------------------------------------------------------
#  R-013: promotion-set check of a predeclared fitted pair, rebuilt and hash-verified without rollouts
# ---------------------------------------------------------------------------


def rebuild_fitted_state(source_path, fitting, fit_seed, selected_step, demo_buffer, env_name="push-v2"):
    """Refit deterministically and return ``(policy, sha)`` holding the state at ``selected_step``.

    The checkpoint probe is replaced by one that snapshots parameters and runs no episode, so the gradient
    trajectory is the original fit's and nothing is rolled out.
    """
    import copy

    import torch

    from lares.core.obs_schema import get_obs_schema
    from lares.core.training_pipeline import DemoBuffer
    from lares.fitting.checkpoints import RolloutScore
    from lares.fitting.optimizers import fit_policy_with_objective, prepare_data, state_parameters_sha256
    from lares.fitting.structure_bank import instantiate_source

    torch.set_num_threads(1)
    with open(source_path, encoding="utf-8") as f:
        policy = instantiate_source(f.read(), 39, 4, get_obs_schema(env_name))
    policy.fixed_action_scale = float(fitting["fixed_action_scale"])
    steps = [int(s) for s in fitting["checkpoint_steps"]]
    snapshots, order = {}, iter(steps)

    def snapshot(pol):
        snapshots[next(order)] = copy.deepcopy(pol.state_dict())
        return RolloutScore(0.0, 0.0)

    data = prepare_data(DemoBuffer.load(demo_buffer), val_fraction=float(fitting["val_fraction"]),
                        split_seed=int(fitting["split_seed"]))
    fit_policy_with_objective(
        policy, data, fitting["objective"], budget=int(fitting["budget_gradient_steps"]),
        batch_size=min(int(fitting["batch_size"]), int(data.train_obs.shape[0])), lr=float(fitting["lr"]),
        clip_grad_norm=float(fitting["clip_grad_norm"]), seed=int(fit_seed), sampling=fitting["sampling"],
        rollout_probe=snapshot, checkpoint_steps=steps, structure_id="rebuild",
    )
    policy.load_state_dict(snapshots[int(selected_step)])
    return policy, state_parameters_sha256(policy.state_dict())


def run_promotion_check(args) -> int:
    from lares.eval import EvaluationManifest, make_manifest_env
    from lares.eval.runner import PolicyActor, evaluate_manifest

    pairs = {q["child_id"]: q for q in discover_pairs(args.run)}
    pair, fitting = pairs[args.child], common_fitting(list(pairs.values()))
    os.makedirs(args.out_dir, exist_ok=True)
    status = ensure_entries_reserved([(args.eval_seed, "evaluation", "R-013 promotion-set check (unused by a "
                                       "deterministic evaluation; reserved per the advisor)")], args.out_dir)
    print(f"  pair {pair['child_id']} vs {pair['parent_id']}; seeds: {status}")
    if args.reserve_only:
        return 0
    from lares.core.training_pipeline import ensure_mujoco_headless_gl

    ensure_mujoco_headless_gl()
    demo = os.path.join(_PROJECT_ROOT, args.demo_buffer)
    rebuilt = {}
    for role, pid, src in (("child", pair["child_id"], pair["child_source_path"]),
                           ("parent", pair["parent_id"], pair["parent_source_path"])):
        with open(os.path.join(args.from_run, "policies", f"{pid}_fit{args.fit_seed}.json"), encoding="utf-8") as f:
            recorded = json.load(f)
        policy, sha = rebuild_fitted_state(src, fitting, args.fit_seed, recorded["selected_step"], demo, args.env_name)
        rebuilt[role] = {"policy_id": pid, "policy": policy, "selected_step": recorded["selected_step"],
                         "recorded_sha256": recorded["parameters_sha256"], "rebuilt_sha256": sha}
        print(f"  {role} {pid}: step {recorded['selected_step']}, hash {'matches' if sha == recorded['parameters_sha256'] else 'DIFFERS'}")
    mismatched = [r for r in rebuilt.values() if r["rebuilt_sha256"] != r["recorded_sha256"]]
    meta = {r: {k: v for k, v in d.items() if k != "policy"} for r, d in rebuilt.items()}
    if mismatched:
        with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump({"scored": False, "reason": "a rebuilt state does not match its recorded hash", "rebuilt": meta}, f, indent=2)
        raise SystemExit("hash mismatch: nothing scored")
    development = EvaluationManifest.load(os.path.join(_PROJECT_ROOT, "config", "manifests", f"{args.env_name}_development.yaml"))
    promotion = development.slice(10, 40)
    env = make_manifest_env(development, development.resolve_pool())
    per_case = {}
    for role, d in rebuilt.items():
        result = evaluate_manifest(PolicyActor(d["policy"], deterministic=True, name=role), promotion, env,
                                   max_steps=args.horizon)
        per_case[role] = {e.case_id: float(e.success) for e in result.episodes}
        meta[role]["success_rate"] = result.success_rate
    cases = sorted(per_case["child"])
    diffs = [per_case["child"][c] - per_case["parent"][c] for c in cases]
    summary = {"scored": True, "manifest_id": promotion.manifest_id, "paired_success_difference": float(np.mean(diffs)),
               "paired": {"n": len(diffs), **t_interval(diffs), **win_loss_tie(diffs)}, "rebuilt": meta,
               "per_case": per_case, "episodes": 2 * len(cases)}
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    pr = summary["paired"]
    print(f"  {promotion.manifest_id}: child {meta['child']['success_rate']:.3f} parent {meta['parent']['success_rate']:.3f}; "
          f"paired {pr['mean']:+.3f} [{pr['low']}, {pr['high']}] w/l/t {pr['wins']}/{pr['losses']}/{pr['ties']}")
    return 0


# ---------------------------------------------------------------------------
#  R-014: do the geometry child's added parameters need to be fitted?
# ---------------------------------------------------------------------------

ABLATION_CHECKPOINTS = (0, 125, 250, 500, 750, 1000)


def child_only_parameters(child_policy, parent_policy) -> list:
    from lares.search.lanes import scale_parameter_names

    parent = {n for n, _ in parent_policy.named_parameters()}
    scale = set(scale_parameter_names(child_policy))
    return sorted(n for n, _ in child_policy.named_parameters() if n not in parent and n not in scale)


def ablation_analysis(trainable, clamped) -> dict:
    """``{fit seed: {case: success}}`` per arm. Placement level (mean over seeds) and seed level, both t intervals."""
    seeds = sorted(trainable)
    if sorted(clamped) != seeds:
        raise ValueError("both arms must be fitted under the same seeds")
    cases = sorted(trainable[seeds[0]])
    per_case = [float(np.mean([trainable[k][c] - clamped[k][c] for k in seeds])) for c in cases]
    rate = lambda d: float(np.mean(list(d.values())))  # noqa: E731
    per_seed = {int(k): rate(trainable[k]) - rate(clamped[k]) for k in seeds}
    return {"paired_success_difference": float(np.mean(per_case)),
            "placement_level": {"n": len(cases), **t_interval(per_case), **win_loss_tie(per_case)},
            "seed_level": {"by_seed": per_seed, **t_interval(list(per_seed.values()))},
            "success_by_seed": {"trainable": {int(k): rate(trainable[k]) for k in seeds},
                                "clamped": {int(k): rate(clamped[k]) for k in seeds}}}


def run_added_parameter_ablation(args) -> int:
    import multiprocessing as mp

    from lares.core.obs_schema import get_obs_schema
    from lares.fitting.structure_bank import instantiate_source

    pairs = {q["child_id"]: q for q in discover_pairs(args.run)}
    pair, fitting = pairs[args.child], common_fitting(list(pairs.values()))
    schema = get_obs_schema(args.env_name)
    with open(pair["child_source_path"], encoding="utf-8") as f:
        child = instantiate_source(f.read(), 39, 4, schema)
    with open(pair["parent_source_path"], encoding="utf-8") as f:
        parent = instantiate_source(f.read(), 39, 4, schema)
    added = child_only_parameters(child, parent)
    if not added:
        raise SystemExit(f"{args.child} adds no parameter to {pair['parent_id']}; nothing to clamp")
    os.makedirs(args.out_dir, exist_ok=True)
    status = ensure_entries_reserved(
        [(k, "fitting", "R-014 added-parameter ablation: fit seed") for k in args.fit_seeds]
        + [(args.probe_seed, "mt1_pool", "R-014: 10-case checkpoint probe")]
        + [(x, "mt1_pool", "R-014: 50-case confirmation manifest") for x in args.heldout_seed], args.out_dir)
    print(f"  {pair['child_id']}: clamping {added}; seeds: {status}")
    if args.reserve_only:
        return 0
    from lares.core.training_pipeline import ensure_mujoco_headless_gl

    ensure_mujoco_headless_gl()
    probe, probe_path = fresh_manifest(args.env_name, args.probe_seed, 10, args.horizon, args.out_dir, role="probe")
    conf, conf_path = fresh_manifest(args.env_name, args.heldout_seed, 50 * len(args.heldout_seed), args.horizon,
                                     args.out_dir, role="confirmation", also_disjoint_from=[probe])
    common = {"env_name": args.env_name, "horizon": args.horizon, "fitting": fitting, "role": "child",
              "policy_id": pair["child_id"], "source_path": pair["child_source_path"],
              "check_state": pair["child_initial_state"], "checkpoint_steps": list(ABLATION_CHECKPOINTS),
              "out_dir": os.path.abspath(args.out_dir), "demo_buffer": os.path.join(_PROJECT_ROOT, args.demo_buffer),
              "probe_manifest": os.path.abspath(probe_path), "evaluate": {"confirmation": os.path.abspath(conf_path)}}
    jobs = [{**common, "fit_seed": k, "label": f"trainable_fit{k}"} for k in args.fit_seeds]
    jobs += [{**common, "fit_seed": k, "clamp": added, "label": f"clamped_fit{k}"} for k in args.fit_seeds]
    with open(os.path.join(args.out_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump({"started_at": now(), "argv": sys.argv, "child": pair["child_id"], "clamped": added,
                   "checkpoints": ABLATION_CHECKPOINTS, "fitting": fitting,
                   "manifests": {"probe": probe.manifest_id, "confirmation": conf.manifest_id}}, f, indent=2, default=str)
    with mp.get_context("spawn").Pool(processes=len(jobs)) as pool:
        results = pool.map(nested_job, jobs)
    conf_of = lambda r: {c: v[0] for c, v in r["confirmation"]["per_case"].items()}  # noqa: E731
    arms = {"trainable": {}, "clamped": {}}
    for job, r in zip(jobs, results):
        arms["clamped" if job.get("clamp") else "trainable"][job["fit_seed"]] = conf_of(r)
    summary = ablation_analysis(arms["trainable"], arms["clamped"])
    summary.update(child=pair["child_id"], clamped=added,
                   selected_steps={j["label"]: r["selected_step"] for j, r in zip(jobs, results)},
                   resources={k: int(sum(r["resources"].get(k, 0) for r in results))
                              for k in ("probe_episodes", "confirmation_episodes")})
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    pl, sl = summary["placement_level"], summary["seed_level"]
    print(f"  success {summary['success_by_seed']}; steps {summary['selected_steps']}")
    print(f"  trainable - clamped: placement {pl['mean']:+.3f} [{pl['low']}, {pl['high']}] w/l/t "
          f"{pl['wins']}/{pl['losses']}/{pl['ties']}; seed {sl['by_seed']}; resources {summary['resources']}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--run", required=True, help="two-lane run directory")
    p.add_argument("--baseline", default="", help="zero-shot baseline run directory (R-001 secondary)")
    p.add_argument("--pool-seed", type=int, help="fresh evaluation pool (R-005 mode)")
    p.add_argument("--fit-seed", type=int)
    p.add_argument("--research-protocol", choices=["nested-screen", "seed-robustness", "promotion-check",
                                                   "added-parameter-ablation"], default="")
    p.add_argument("--from-run", default="", help="promotion-check: the run whose fit records name the pair")
    p.add_argument("--eval-seed", type=int)
    p.add_argument("--fit-seeds", type=int, nargs="+", help="seed-robustness: fitting seeds")
    p.add_argument("--child", default="", help="nested-screen: the frozen child attempt id")
    p.add_argument("--screen-seed", type=int)
    p.add_argument("--heldout-seed", type=int, nargs="+", help="confirmation pool seed(s); 50 placements each")
    p.add_argument("--probe-seed", type=int)
    p.add_argument("--replay-seed", type=int)
    p.add_argument("--replays", type=int, default=1000)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--env-name", default="push-v2")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--horizon", type=int, default=150)
    p.add_argument("--demo-buffer", default=DEMO_BUFFER)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--reserve-only", action="store_true", help="check and reserve the seeds, then stop")
    args = p.parse_args(argv)
    if args.research_protocol == "nested-screen":
        missing = [n for n in ("child", "fit_seed", "screen_seed", "heldout_seed", "probe_seed", "replay_seed")
                   if getattr(args, n) in (None, "")]
        if missing:
            raise SystemExit(f"nested-screen needs --{', --'.join(m.replace('_', '-') for m in missing)}")
        return run_nested_screen(args)
    if args.research_protocol == "added-parameter-ablation":
        if not (args.child and args.fit_seeds and args.probe_seed is not None and args.heldout_seed):
            raise SystemExit("added-parameter-ablation needs --child, --fit-seeds, --probe-seed and --heldout-seed")
        return run_added_parameter_ablation(args)
    if args.research_protocol == "promotion-check":
        if not (args.child and args.fit_seed is not None and args.from_run and args.eval_seed is not None):
            raise SystemExit("promotion-check needs --child, --fit-seed, --from-run and --eval-seed")
        return run_promotion_check(args)
    if args.research_protocol == "seed-robustness":
        missing = [n for n in ("child", "fit_seeds", "probe_seed", "heldout_seed") if getattr(args, n) in (None, "")]
        if missing:
            raise SystemExit(f"seed-robustness needs --{', --'.join(m.replace('_', '-') for m in missing)}")
        return run_seed_robustness(args)
    if args.pool_seed is None or args.fit_seed is None:
        raise SystemExit("--pool-seed and --fit-seed are required")

    pairs = discover_pairs(args.run)
    fitting = common_fitting(pairs)
    winner = performance_incumbent(args.run)
    if winner not in {q["child_id"] for q in pairs}:
        raise SystemExit(f"the two-lane winner {winner} is not a fittable-lane child")
    baseline = baseline_best(args.baseline) if args.baseline else None
    print(f"  {len(pairs)} children, {len({q['parent_id'] for q in pairs})} unique parents; fitting {fitting}")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"  seeds {args.pool_seed} (pool), {args.fit_seed} (fitting): "
          f"{ensure_reserved(args.pool_seed, args.fit_seed, args.out_dir)}")
    if args.reserve_only:
        return 0

    import multiprocessing as mp

    from lares.core.training_pipeline import ensure_mujoco_headless_gl

    ensure_mujoco_headless_gl()
    manifest, manifest_path = fresh_manifest(args.env_name, args.pool_seed, args.episodes, args.horizon, args.out_dir)
    print(f"  manifest {manifest.manifest_id}: {len(manifest)} cases -> {rel(manifest_path)}")
    demo_sha = sha256_file(os.path.join(_PROJECT_ROOT, args.demo_buffer))
    jobs = build_jobs(pairs, fitting, baseline, args, manifest_path)
    with open(os.path.join(args.out_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump({"started_at": now(), "argv": sys.argv, "pairs": [{k: v for k, v in q.items() if k != "objective_config"}
                   for q in pairs], "fitting": fitting, "fit_seed": args.fit_seed, "demo_buffer": args.demo_buffer,
                   "demo_buffer_sha256": demo_sha, "manifest_id": manifest.manifest_id,
                   "manifest_sha256": sha256_file(manifest_path), "two_lane_winner": winner, "baseline": baseline},
                  f, indent=2, default=str)

    results = {}
    with mp.get_context("spawn").Pool(processes=min(int(args.workers), len(jobs))) as pool:
        for done in pool.imap_unordered(run_job, jobs):
            results[done["policy_id"]] = done
            step = f"step {done['selected_step']}" if "selected_step" in done else "saved state"
            print(f"  {done['policy_id']:<36} {done['role']:<8} {step:<12} success {done['success_rate']:.2f}"
                  f"  ({done['wall_time_seconds']:.0f} s)", flush=True)

    per_case = {pid: {c: v["success"] for c, v in r["per_case"].items()} for pid, r in results.items()}
    baseline_id = f"baseline:{baseline['id']}" if baseline else ""
    summary = summarise(per_case, pairs, winner, baseline_id)
    summary.update(
        manifest_id=manifest.manifest_id, fit_seed=args.fit_seed, pool_seed=args.pool_seed, two_lane_winner=winner,
        baseline=baseline_id or None, selected_steps={pid: r.get("selected_step") for pid, r in results.items()},
        resources={k: sum(int(r["resources"][k]) for r in results.values())
                   for k in ("probe_episodes", "evaluation_episodes", "bc_gradient_steps")},
    )
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    pr = summary["primary"]
    print(f"\n  child - parent, placement-level over {pr['n_placements']} cases x {pr['n_pairs']} pairs: "
          f"{pr['mean']:+.3f} t95 [{pr['low']:+.3f}, {pr['high']:+.3f}]")
    for row in summary["per_pair"]:
        print(f"    {row['child']:<16} {row['child_success']:.2f}  vs {row['parent']:<32} {row['parent_success']:.2f}"
              f"  {row['mean']:+.3f}  w/l/t {row['wins']}/{row['losses']}/{row['ties']}")
    if summary["secondary_r001"]:
        s = summary["secondary_r001"]
        print(f"  R-001 {s['a']} - {s['b']}: {s['mean']:+.3f} t95 [{s['low']:+.3f}, {s['high']:+.3f}]")
    print(f"  resources {summary['resources']}\n  written: {rel(os.path.join(args.out_dir, 'summary.json'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
