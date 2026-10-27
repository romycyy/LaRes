"""Exercise the pen-spin batch runner on a throwaway debug manifest (never a frozen one).

Run with Isaac Lab's python on the box:
    source ~/IsaacLab/source_env.sh && python -u scripts/isaac/debug_pen_spin_runner.py --stage 1 --n 4
The debug pool seed (default 4099) is recorded in the seed ledger as debug use.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from lares.envs.isaac_lab_adapter import close_isaac_app, launch_isaac_app  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--stage", type=int, default=1)
p.add_argument("--n", type=int, default=4)
p.add_argument("--seed", type=int, default=4099)
p.add_argument("--determinism", action="store_true", help="PhysX enable_enhanced_determinism")
p.add_argument("--save", default=None, help="save the first run's spin traces to this .npz prefix")
p.add_argument("--repeat", action="store_true", help="run each controller twice to check determinism")
args = p.parse_args()

launch_isaac_app(headless=True)
try:
    import numpy as np

    from lares.envs.isaac_pen_spin import cases
    from lares.envs.isaac_pen_spin.controllers import AUDIT_CONTROLLERS
    from lares.envs.isaac_pen_spin.runner import PenSpinBatchRunner, reduce_episodes

    m = cases.build_manifest(args.stage, args.seed, args.n, label="debug")
    runner = PenSpinBatchRunner(args.stage, args.n, enhanced_determinism=args.determinism)
    for cls in AUDIT_CONTROLLERS:
        runs = []
        for _ in range(2 if args.repeat else 1):
            t0 = time.time()
            tr = runner.run(m, cls(), record_obs=args.repeat, allow_reuse=True)
            runs.append(tr)
            if args.save and len(runs) == 1:
                np.savez(f"{args.save}_{cls.name}.npz", spin=tr["spin_rate"], center=tr["center_dist"], length=tr["length"])
            print(cls.name, f"{time.time() - t0:.1f}s", flush=True)
            for row in reduce_episodes(tr):
                print(json.dumps(row), flush=True)
        if args.repeat:
            d = np.nanmax(np.abs(runs[0]["spin_rate"] - runs[1]["spin_rate"]))
            print("determinism: max |spin diff| between repeats =", float(d), flush=True)
            o0, o1 = runs[0]["obs"], runs[1]["obs"]
            print("obs diff at t=0 per field:", {k: float(np.abs(o0[:, 0, s] - o1[:, 0, s]).max())
                                                for k, s in cases.obs_slices().items()}, flush=True)
            T = min(o0.shape[1], o1.shape[1])
            dev = np.abs(o0[:, :T] - o1[:, :T]).max(axis=(0, 2))
            first = [int(np.argmax(dev > tol)) if (dev > tol).any() else None for tol in (1e-6, 1e-3, 1e-1)]
            print("first step obs differ by >1e-6, >1e-3, >1e-1:", first, flush=True)
    runner.close()
finally:
    close_isaac_app()
