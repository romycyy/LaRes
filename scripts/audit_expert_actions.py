#!/usr/bin/env python
"""Audit expert action generation and clipping semantics (``spec.md`` FR-7, AC-5).

A likelihood must match the process that produced its targets. This measures
that process rather than assuming it.

MetaWorld's scripted expert is an unbounded proportional controller: it returns
``10 * (desired_position - current_position)`` and only *warns* when the result
leaves ``[-1, 1]``. The environment clips downstream, and Stage 1 clips before
storing. So the mass sitting exactly at an action endpoint is produced by *our*
clip of an unbounded response, not by anything the expert does.

That decides the likelihood question. The stored targets are a censored
observation of an unbounded deterministic signal, so the endpoint mass belongs to
a censored model with a clipped execution contract. A tanh-Gaussian likelihood is
wrong twice over: ``atanh(+-1)`` diverges on the censored fraction, and the
generating process is not a tanh squash.

Usage (from the project root)::

    python scripts/audit_expert_actions.py
    python scripts/audit_expert_actions.py --episodes 50
"""

import argparse
import json
import os
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402

from lares.core.obs_schema import get_obs_schema  # noqa: E402
from lares.core.training_pipeline import (  # noqa: E402
    ensure_mujoco_headless_gl,
    get_expert_policy,
)
from lares.eval import EvaluationManifest, make_manifest_env  # noqa: E402
from lares.eval.manifest import SPLIT_TRAIN  # noqa: E402

AXES = ("dx", "dy", "dz", "gripper")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env-name", default="push-v2")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument(
        "--manifest",
        default=os.path.join(_PROJECT_ROOT, "config", "manifests", "push-v2_train.yaml"),
    )
    p.add_argument(
        "--out-dir", default=os.path.join(_PROJECT_ROOT, "logs", "expert_audit")
    )
    return p.parse_args()


def separator(title):
    print(f"\n{'=' * 76}\n  {title}\n{'=' * 76}")


def main():
    args = parse_args()
    ensure_mujoco_headless_gl()
    os.makedirs(args.out_dir, exist_ok=True)
    schema = get_obs_schema(args.env_name)

    manifest = EvaluationManifest.load(args.manifest)
    cases = manifest.episodes[: args.episodes]
    env = make_manifest_env(manifest, manifest.resolve_pool(), 200)
    expert = get_expert_policy(args.env_name)

    from lares.fitting.phases import expert_phase_labels, PHASE_NAMES

    raw_actions, phases = [], []
    for case in cases:
        obs, _ = env.reset(case)
        for _ in range(args.max_steps):
            raw = np.asarray(expert.get_action(obs), dtype=np.float64)
            raw_actions.append(raw)
            phases.append(expert_phase_labels(obs[None, :], schema)[0])
            obs, _, done, _ = env.step(env.action_space.high * np.clip(raw, -1.0, 1.0))
            if done:
                break
    env.close()

    raw = np.asarray(raw_actions)
    clipped = np.clip(raw, -1.0, 1.0)

    separator("Expert action generation")
    print("  The scripted expert is an unbounded proportional controller:")
    print("    action[0:3] = 10.0 * (desired_position - hand_position)")
    print("    action[3]   = 0.0 or 0.6, a two-valued grab effort")
    print("  It performs no clipping of its own; it warns and returns the raw value.")
    print("  The environment clips to [-1, 1], and Stage 1 clips before storing.")

    separator(f"Raw versus stored actions over {len(raw)} transitions")
    header = f"{'axis':<10}{'raw min':>10}{'raw max':>10}{'|raw|>1':>10}{'stored ==+-1':>14}"
    print(header)
    print("-" * len(header))
    per_axis = {}
    for i, name in enumerate(AXES):
        censored = float((np.abs(raw[:, i]) > 1.0).mean())
        at_bound = float((np.abs(clipped[:, i]) >= 1.0).mean())
        per_axis[name] = {
            "raw_min": float(raw[:, i].min()),
            "raw_max": float(raw[:, i].max()),
            "fraction_censored": censored,
            "fraction_at_endpoint": at_bound,
            "raw_mean": float(raw[:, i].mean()),
            "raw_std": float(raw[:, i].std()),
        }
        print(
            f"{name:<10}{raw[:, i].min():>10.3f}{raw[:, i].max():>10.3f}"
            f"{censored:>10.4f}{at_bound:>14.4f}"
        )
    any_censored = float((np.abs(raw) > 1.0).any(axis=1).mean())
    print(f"\n  transitions censored on at least one axis: {any_censored:.4f}")
    print(f"  largest raw magnitude seen: {np.abs(raw).max():.2f}, "
          f"which is {np.abs(raw).max():.0f}x the action limit")

    separator("Expert phases")
    print("  The expert's own control law has three explicit branches, so the phase")
    print("  label is ground truth rather than inferred from a learned gate.")
    counts = {name: int((np.asarray(phases) == i).sum()) for i, name in enumerate(PHASE_NAMES)}
    total = sum(counts.values()) or 1
    for name, count in counts.items():
        print(f"    {name:<12} {count:>7}  {count / total:.3f}")
    print("\n  Uniform minibatch sampling therefore sees the phases in these")
    print("  proportions, which is why phase-balanced sampling is worth testing.")

    separator("What this means for the objective")
    print("  The stored target is a censored observation of an unbounded deterministic")
    print("  signal. Consequences:")
    print("   1. A tanh-Gaussian likelihood is not the matching model. atanh(+-1) is")
    print("      undefined on the censored fraction, and the generator is not a tanh")
    print("      squash, so the change of variables does not apply.")
    print("   2. The expert has no aleatoric spread at all: p(a|s) is a point mass.")
    print("      Maximum likelihood against a point mass drives the scale to zero.")
    print("   3. Deterministic MSE on tanh(mean) remains the defensible primary")
    print("      objective, with the scale fixed and excluded from optimisation.")
    print("   4. A censored Gaussian is the correct alternative to test, and it")
    print("      requires a clipped execution contract, not the tanh sampler.")

    payload = {
        "env_name": args.env_name,
        "episodes": len(cases),
        "transitions": int(len(raw)),
        "expert": {
            "class": type(expert).__name__,
            "clips_internally": False,
            "proportional_gain": 10.0,
            "grab_effort_values": [0.0, 0.6],
        },
        "per_axis": per_axis,
        "fraction_censored_any_axis": any_censored,
        "max_raw_magnitude": float(np.abs(raw).max()),
        "phase_counts": counts,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = os.path.join(args.out_dir, f"expert_audit_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n  report: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
