"""R-041: does the qualified PPO expert need feedback, or only a replayed gait?

Stage 3, two frozen 40-case manifests. The D-036 expert (checkpoint frozen by path and SHA-256) runs once
on the demonstration manifest with its actions recorded; the first case in manifest order whose episode
is a sustained success (D-034 reducer) supplies the tape, its complete action sequence. On the diagnostic
manifest the expert runs closed loop twice (the R-039 amended reproducibility gate: zero sustained-outcome
flips, at most 4/40 diverged environments; abort otherwise) and the tape is replayed open loop by the
clock alone, with no observation feedback and no retiming. Every rollout is its own process (D-034).

The prediction is scored on the diagnostic manifest: closed-loop run ``a`` minus the replay, paired by case.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

from lares.envs.isaac_pen_spin import cases

FEEDBACK_ABLATION = "feedback_ablation"
STAGE = 3


class ActionTape:
    """Replays one recorded action sequence by time step, ignoring the observation."""

    name = "action_tape"

    def __init__(self, actions: np.ndarray, source: dict):
        self.actions = np.asarray(actions, dtype=np.float32)
        if self.actions.ndim != 2 or self.actions.shape[1] != cases.NUM_ACTIONS:
            raise ValueError(f"tape must be (T, {cases.NUM_ACTIONS}), got {self.actions.shape}")
        self.params = {"source": source, "steps": int(len(self.actions))}

    def reset(self, ctx):
        import torch

        self._tape = torch.as_tensor(self.actions, device=ctx.device)

    def __call__(self, obs, t, ctx):
        # Hold the last action past the tape's end (a sustained-success tape is full length, so unused).
        row = self._tape[min(int(t), len(self._tape) - 1)]
        return row.unsqueeze(0).expand(obs.shape[0], -1).clone()


def select_demonstration(episodes: list[dict], actions: np.ndarray) -> tuple[int, np.ndarray]:
    """First sustained success in manifest order and its executed actions.

    ``actions`` is the runner's recorded trace, shaped (cases, T, 20): ``PenSpinBatchRunner.run`` stacks the
    per-step (cases, 20) arrays on axis 1.
    """
    actions = np.asarray(actions)
    if actions.ndim != 3 or actions.shape[0] != len(episodes):
        raise ValueError(f"actions must be (cases={len(episodes)}, T, {cases.NUM_ACTIONS}), got {actions.shape}")
    for i, e in enumerate(episodes):
        if e["sustained"]:
            return i, actions[i, : int(e["length"])]
    raise ValueError("no sustained-success demonstration on the demonstration manifest")


def manifest_paths(protocol_cfg: dict) -> dict:
    return {"demo": protocol_cfg["demonstration_manifest"], "diagnostic": protocol_cfg["diagnostic_manifest"]}


def build_manifests(protocol_cfg: dict) -> list[str]:
    from lares.eval.manifest import EvaluationManifest

    written = []
    n = int(protocol_cfg["cases"])
    for label, path in manifest_paths(protocol_cfg).items():
        seed = int(protocol_cfg["pool_seeds"][label])
        m = cases.build_manifest(STAGE, seed, n, label=label)
        if os.path.exists(path):
            if EvaluationManifest.load(path).to_dict() != m.to_dict():
                raise ValueError(f"{path} exists and differs from the manifest this protocol builds")
            continue
        m.save(path)
        written.append(path)
    return written


def check_checkpoint(protocol_cfg: dict) -> str:
    from lares.utils.hashing import sha256_file

    path = protocol_cfg["checkpoint"]
    if not os.path.exists(path):
        raise SystemExit(f"frozen checkpoint {path} missing")
    got = sha256_file(path)
    if got != protocol_cfg["checkpoint_sha256"]:
        raise SystemExit(f"checkpoint {path} hash {got} != frozen {protocol_cfg['checkpoint_sha256']}")
    return path


def make_controller(name: str, protocol_cfg: dict, out_dir: str):
    if name == "ppo_expert":
        from lares.envs.isaac_pen_spin.ppo_expert import PPOController

        return PPOController(check_checkpoint(protocol_cfg), int(protocol_cfg["training_seed"]))
    if name == ActionTape.name:
        with open(os.path.join(out_dir, "tape.json"), encoding="utf-8") as f:
            meta = json.load(f)
        return ActionTape(np.load(os.path.join(out_dir, "tape.npy")), meta)
    raise ValueError(f"unknown controller {name!r}")


def _child(config_path: str, manifest_path: str, controller: str, out_path: str, record_obs: bool = False):
    if os.path.exists(out_path):
        print("exists", out_path, flush=True)
        return
    cmd = [sys.executable, "-u", sys.argv[0], "--config", config_path, "--study", FEEDBACK_ABLATION,
           "--cell-manifest", manifest_path, "--controller", controller, "--cell-out", out_path]
    if record_obs:
        cmd.append("--record-obs")
    print("running", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def reproducibility_gate(a_json: str, b_json: str) -> dict:
    from lares.envs.isaac_pen_spin.protocols import MAX_DIVERGED_ENVS, _diverged_envs

    a, b = (np.load(p.replace(".json", ".npz")) for p in (a_json, b_json))
    ra, rb = (json.load(open(p))["episodes"] for p in (a_json, b_json))
    union = (_diverged_envs(a, b, "spin_rate") | _diverged_envs(a, b, "center_dist")
             | {i for i in range(len(ra)) if ra[i]["length"] != rb[i]["length"]})
    flips = sum(x["sustained"] != y["sustained"] for x, y in zip(ra, rb))
    return {"union_diverged_envs": len(union), "sustained_outcome_flips": int(flips),
            "max_diverged_envs": MAX_DIVERGED_ENVS,
            "passes": bool(flips == 0 and len(union) <= MAX_DIVERGED_ENVS)}


def _arm_summary(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "n": n,
        "successes": int(sum(r["sustained"] for r in rows)),
        "success": float(np.mean([r["sustained"] for r in rows])),
        "completed": float(np.mean([r["completed"] for r in rows])),
        "tail_hit_fraction": float(np.mean([r["tail_hit_fraction"] for r in rows])),
        "hit_fraction": float(np.mean([r["hit_steps"] / max(1, r["length"]) for r in rows])),
        "mean_length": float(np.mean([r["length"] for r in rows])),
        "mean_abs_spin": float(np.mean([r["mean_abs_spin"] for r in rows if r["mean_abs_spin"] is not None])),
        "mean_transverse": float(np.mean([r["mean_transverse"] for r in rows if r["mean_transverse"] is not None])),
    }


def aggregate(closed_rows: list[dict], tape_rows: list[dict]) -> dict:
    """Paired closed-loop minus replay sustained success on the diagnostic manifest."""
    from lares.eval.stats import t_summary

    if [r["task_hash"] for r in closed_rows] != [r["task_hash"] for r in tape_rows]:
        raise ValueError("closed-loop and replay rows are not paired case for case")
    d = [float(c["sustained"]) - float(r["sustained"]) for c, r in zip(closed_rows, tape_rows)]
    t = t_summary(d)
    return {
        "paired_success_difference": t["mean"], "paired_t": t,
        "closed_loop": _arm_summary(closed_rows), "action_tape": _arm_summary(tape_rows),
    }


def run_feedback_ablation(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    """Driver: demonstration, gate, replay, aggregate. Children write the cells; nothing is rerun if present."""
    from lares.utils.hashing import sha256_file

    check_checkpoint(protocol_cfg)
    os.makedirs(out_dir, exist_ok=True)
    m = manifest_paths(protocol_cfg)
    demo_out = os.path.join(out_dir, "demonstration_ppo_expert.json")
    _child(config_path, m["demo"], "ppo_expert", demo_out, record_obs=True)
    if not os.path.exists(os.path.join(out_dir, "tape.npy")):
        rows = json.load(open(demo_out))["episodes"]
        idx, tape = select_demonstration(rows, np.load(demo_out.replace(".json", ".npz"))["actions"])
        np.save(os.path.join(out_dir, "tape.npy"), tape)
        meta = {"manifest": m["demo"], "manifest_sha256": sha256_file(m["demo"]),
                "case_index": idx, "case_id": rows[idx]["case_id"], "task_hash": rows[idx]["task_hash"],
                "steps": int(len(tape)), "checkpoint": protocol_cfg["checkpoint"],
                "checkpoint_sha256": protocol_cfg["checkpoint_sha256"], "rule": "first sustained success in manifest order",
                "demonstration_successes": int(sum(r["sustained"] for r in rows)), "demonstration_cases": len(rows)}
        with open(os.path.join(out_dir, "tape.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    closed = [os.path.join(out_dir, f"diagnostic_ppo_expert_{r}.json") for r in "ab"]
    for p in closed:
        _child(config_path, m["diagnostic"], "ppo_expert", p)
    gate = reproducibility_gate(*closed)
    if not gate["passes"]:
        with open(os.path.join(out_dir, "gate_failed.json"), "w") as f:
            json.dump(gate, f, indent=2)
        raise SystemExit(f"reproducibility gate failed, replay not run: {gate}")
    tape_out = os.path.join(out_dir, "diagnostic_action_tape.json")
    _child(config_path, m["diagnostic"], ActionTape.name, tape_out)
    summary = aggregate(json.load(open(closed[0]))["episodes"], json.load(open(tape_out))["episodes"])
    summary["reproducibility_control"] = gate
    summary["tape"] = json.load(open(os.path.join(out_dir, "tape.json")))
    summary["demonstration"] = _arm_summary(json.load(open(demo_out))["episodes"])
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
