"""Tune the R-039 scripted gait on the debug seed only (4098), several settings per batched rollout.

Each setting is run on the same first ``--cases`` placements of the debug pool, all settings in one
Isaac env (settings x cases environments), so one fresh-process rollout scores a whole sweep
reproducibly (D-034: only the first rollout in a process is reproducible). Episodes are counted and
appended to ``--budget-log`` so the protocol's tuning allowance can be audited.

    source ~/IsaacLab/source_env.sh
    python -u scripts/isaac/tune_pen_spin_gait.py --stage 1 --settings sweep.json --cases 10
    python -u scripts/isaac/tune_pen_spin_gait.py --stage 1 --describe       # geometry only, 1 episode
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lares.utils.clock import now_iso  # noqa: E402
from lares.envs.isaac_lab_adapter import close_isaac_app, launch_isaac_app  # noqa: E402

DEBUG_SEED = 4098

p = argparse.ArgumentParser()
p.add_argument("--stage", type=int, default=1)
p.add_argument("--settings", default=None, help="JSON list of gait parameter dicts")
p.add_argument("--cases", type=int, default=10)
p.add_argument("--describe", action="store_true")
p.add_argument("--out", default=None, help="write per-setting results JSON here")
p.add_argument("--budget-log", default="logs/pen_spin/r039_scripted_expert/tuning_budget.jsonl")
args = p.parse_args()

launch_isaac_app(headless=True)
try:
    import numpy as np

    from lares.envs.isaac_pen_spin import cases
    from lares.envs.isaac_pen_spin.runner import PenSpinBatchRunner, reduce_episodes
    from lares.eval.manifest import EpisodeCase, EvaluationManifest

    base = cases.build_manifest(args.stage, DEBUG_SEED, max(args.cases, 1), label="debug")
    sl = cases.obs_slices()

    if args.describe:
        from lares.envs.isaac_pen_spin.controllers import NeutralPose

        runner = PenSpinBatchRunner(args.stage, 1)
        m1 = base.head(1)
        env = runner.env
        snaps = []

        class Probe(NeutralPose):
            def __call__(self, obs, t, ctx):
                if t in (0, 20, 60):
                    o = obs[0].cpu().numpy()
                    snaps.append({"t": t, **{k: np.round(o[sl[k]], 4).tolist() for k in
                                             ("obj_pos", "obj_rot", "in_hand_pos", "fingertip_pos", "hand_dof_pos")}})
                return super().__call__(obs, t, ctx)

        runner.run(m1, Probe())
        idx = env.actuated_dof_indices
        print(json.dumps({
            "joint_names": env.hand.joint_names,
            "actuated": [env.hand.joint_names[i] for i in idx],
            "lower": env.hand_dof_lower_limits[0].cpu().numpy().round(3).tolist(),
            "upper": env.hand_dof_upper_limits[0].cpu().numpy().round(3).tolist(),
            "hand_root_pos": (env.hand.data.root_pos_w[0] - env.scene.env_origins[0]).cpu().numpy().round(4).tolist(),
            "hand_root_quat": env.hand.data.root_quat_w[0].cpu().numpy().round(4).tolist(),
            "snapshots": snaps,
        }, indent=1))
        runner.close()
        sys.exit(0)

    from lares.envs.isaac_pen_spin.scripted_expert import FeedbackGait

    with open(args.settings, encoding="utf-8") as f:
        settings = json.load(f)
    n, S = args.cases, len(settings)
    eps = []
    for j in range(S):
        for c in base.episodes[:n]:
            eps.append(EpisodeCase(f"set{j:02d}-{c.case_id}", c.task_id, c.task_hash, c.reset_seed, c.policy_seed))
    m = EvaluationManifest(f"{base.manifest_id}:sweep{S}x{n}", base.environment, base.pool, base.split, eps)
    runner = PenSpinBatchRunner(args.stage, len(eps))
    ctrl = FeedbackGait.batched([s for s in settings for _ in range(n)])
    t0 = time.time()
    trace = runner.run(m, ctrl)
    rows = reduce_episodes(trace)
    runner.close()
    out = []
    for j, s in enumerate(settings):
        r = rows[j * n:(j + 1) * n]
        spin = trace["spin_rate"][:, j * n:(j + 1) * n]
        L = trace["length"][j * n:(j + 1) * n]
        flips, consist, ydrift = [], [], []
        for i in range(n):
            tail = spin[max(0, L[i] - 60):L[i], i]
            flips.append(int(np.sum(np.diff(np.sign(tail)) != 0)))
            consist.append(float(abs(np.mean(np.sign(tail)))))
            ydrift.append(float(np.mean(tail)))
        # Stages share physics (only thresholds and reward differ), so score every stage's reducer offline.
        cdist = trace["center_dist"][:, j * n:(j + 1) * n]
        comp = trace["completed"][j * n:(j + 1) * n]
        by_stage = {}
        for k, spec in cases.STAGES.items():
            ok = (cdist < spec.success_center_tolerance) & (np.abs(spin) > spec.success_spin_rate)
            by_stage[f"sustained_s{k}"] = float(np.mean([cases.sustained(ok[:L[i], i], L[i], bool(comp[i])) for i in range(n)]))
            by_stage[f"tail_s{k}"] = float(np.mean([ok[max(0, L[i] - 60):L[i], i].mean() if L[i] else 0.0 for i in range(n)]))
        out.append({
            "setting": s,
            **by_stage,
            "tail_sign_flips": float(np.mean(flips)),
            "tail_direction_consistency": float(np.mean(consist)),
            "tail_mean_signed_spin": float(np.mean(ydrift)),
            "sustained": float(np.mean([x["sustained"] for x in r])),
            "completed": float(np.mean([x["completed"] for x in r])),
            "tail_hit_fraction": float(np.mean([x["tail_hit_fraction"] for x in r])),
            "hit_fraction": float(np.mean([x["hit_steps"] / max(1, x["length"]) for x in r])),
            "mean_abs_spin": float(np.mean([x["mean_abs_spin"] for x in r])),
            "mean_length": float(np.mean([x["length"] for x in r])),
        })
    for j, o in enumerate(out):
        print(j, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in o.items() if k != "setting"}),
              json.dumps(o["setting"]), flush=True)
    print(f"{len(eps)} episodes in {time.time() - t0:.1f}s", flush=True)
    os.makedirs(os.path.dirname(args.budget_log), exist_ok=True)
    with open(args.budget_log, "a", encoding="utf-8") as f:
        f.write(json.dumps({"stage": args.stage, "seed": DEBUG_SEED, "episodes": len(eps), "settings": S,
                            "cases": n, "at": now_iso()}) + "\n")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
finally:
    close_isaac_app()
