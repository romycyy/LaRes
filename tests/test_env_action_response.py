#!/usr/bin/env python
"""Sweep action dims 0–2 on [-1,1]; report max observation change (see main)."""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402

from lares.core.training_pipeline import ensure_mujoco_headless_gl  # noqa: E402
from scripts.run_full_evolution import (
    DEFAULT_CONFIG_PATH,
    load_config,
    make_env,
)  # noqa: E402


def _obs0(env):
    out = env.reset()
    return np.asarray(out[0] if isinstance(out, tuple) else out, dtype=np.float64)


def _delta(env, a, hold_steps):
    obs, _ = env.reset()
    o0 = np.asarray(obs, dtype=np.float64)
    o = o0
    for _ in range(hold_steps):
        o, _, done, _ = env.step(a)
        if done:
            break
    return np.asarray(o, np.float64), o0


def probe_first_three(env, hold_steps=1000, seed=0):
    """Only action indices 0,1,2. Other actions fixed at 0."""
    dims = range(min(3, env.action_space.shape[0]))
    grid = [1.0, -1.0]
    out = []
    for d in dims:
        for x in grid:
            a = np.zeros(env.action_space.shape, np.float32)
            a[d] = np.float32(x)
            current_delta, initial_delta = _delta(env, a, hold_steps)
            out.append(current_delta)
            print(f"dim: {d}, action: {a}, current_observation: {current_delta}, initial_observation: {initial_delta}")

    # find if there are any entries in out that are all the same
    for i in range(out[0].shape[0]):
        if all(abs(entry[i] - out[0][i]) < 1e-6 for entry in out):
            print(f"All entries are the same for dimension {i}: {out[0][i]}")
        else:
            # print(f"Entries are not the same for dimension {i}: {out[0][i]}")
            pass


def main():
    cfg = load_config(os.path.abspath(DEFAULT_CONFIG_PATH))
    np.random.seed(cfg.seed)
    ensure_mujoco_headless_gl()
    env = make_env(cfg)
    try:
        probe_first_three(env, seed=cfg.seed)
        # print(f"rows: {rows}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
