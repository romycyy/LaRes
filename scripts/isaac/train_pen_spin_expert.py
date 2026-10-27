#!/usr/bin/env python
"""R-040: train one PPO pen-spin expert on Stage 3 and score it on all three stages.

Run with Isaac Lab's python on the box (``source ~/IsaacLab/source_env.sh``). Phases, each its own
process so every Isaac rollout is the first in its process (D-034):

    --phase gate        try the configured env counts for a few iterations; freeze the largest that fits
    --phase train       PPO on the vendored Stage-3 env, reward unchanged, to the predeclared transition cap
    --phase diagnose    each saved checkpoint on the 40-case debug manifest (seed 4097), one process each
    --phase evaluate    reproducibility control for the PPO cell, then PPO vs the frozen R-039 gait per stage
    (no --phase)        all of the above in order, skipping phases whose outputs exist

Only the final scheduled checkpoint is evaluated on the promotion manifests; checkpoints are
diagnostics. The runner config is the vendored ``ShadowHandSpinPPORunnerCfg`` with
``max_iterations``/``save_interval`` set from the schedule and ``clip_actions = 1.0`` so the
``last_action`` the policy observes in training matches the clamped action the evaluator applies.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import time

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

import yaml  # noqa: E402

PROTOCOL = "ppo_expert"


def load_protocol(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)["research_protocols"][PROTOCOL]


def _self_cmd(args, *extra) -> list:
    return [sys.executable, "-u", os.path.abspath(__file__), "--config", args.config, *extra]


# ---------------------------------------------------------------------------
#  Isaac-side pieces (imported only after the app is launched)
# ---------------------------------------------------------------------------


def make_training_env(num_envs: int, seed: int, device: str):
    import torch

    from lares.envs.isaac_pen_spin.runner import MeasuredShadowHandSpinEnv
    from lares.envs.isaac_pen_spin.shadow_hand_spin_env_cfg import ShadowHandSpinStage3EnvCfg

    class TrainingDiagnosticsEnv(MeasuredShadowHandSpinEnv):
        """Adds end-over-end rate and fall rate to the training log; reward and dones unchanged."""

        def _get_dones(self):
            out_of_reach, time_out = super()._get_dones()
            self.extras.setdefault("log", {})
            self.extras["log"]["transverse_rate"] = self.transverse_rate.mean()
            self.extras["log"]["fall_rate"] = out_of_reach.float().mean()
            return out_of_reach, time_out

        def _get_rewards(self):
            reward = super()._get_rewards()
            cfg = self.cfg
            log = self.extras["log"]
            log["r_spin"] = (cfg.spin_reward_scale * torch.clamp(self.spin_rate.abs() / cfg.target_spin_rate, 0.0, 2.0)).mean()
            log["r_center"] = (cfg.center_reward_scale * torch.exp(-cfg.center_reward_sharpness * self.center_dist)).mean()
            log["r_axis"] = (cfg.axis_reward_scale * torch.exp(-cfg.axis_reward_sharpness * self.axis_error)).mean()
            return reward

    cfg = ShadowHandSpinStage3EnvCfg()
    cfg.scene.num_envs = int(num_envs)
    cfg.sim.device = device
    cfg.seed = int(seed)
    return TrainingDiagnosticsEnv(cfg=cfg)


def make_agent_cfg(seed: int, max_iterations: int, save_interval: int, experiment_name: str):
    from importlib.metadata import version

    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    from lares.envs.isaac_pen_spin.rsl_rl_ppo_cfg import ShadowHandSpinPPORunnerCfg

    agent = ShadowHandSpinPPORunnerCfg()
    agent.seed = int(seed)
    agent.max_iterations = int(max_iterations)
    agent.save_interval = int(save_interval)
    agent.experiment_name = experiment_name
    agent.clip_actions = 1.0
    return handle_deprecated_rsl_rl_cfg(agent, version("rsl-rl-lib"))


def build_runner(env, agent_cfg, log_dir, device):
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    return OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=log_dir, device=device), wrapped


class PPOController:
    """A trained checkpoint behind the evaluator's controller interface."""

    name = "ppo_expert"

    def __init__(self, checkpoint: str, seed: int):
        self.checkpoint = checkpoint
        self.seed = seed
        self.params = {"checkpoint": checkpoint}

    def attach(self, env):
        agent = make_agent_cfg(self.seed, 1, 1, "eval")
        runner, _ = build_runner(env, agent, None, str(env.device))
        runner.load(self.checkpoint)
        self.policy = runner.get_inference_policy(device=str(env.device))

    def reset(self, ctx):
        pass

    def __call__(self, obs, t, ctx):
        import torch
        from tensordict import TensorDict

        with torch.inference_mode():
            return self.policy(TensorDict({"policy": obs}, batch_size=[obs.shape[0]])).clone()


# ---------------------------------------------------------------------------
#  Phases
# ---------------------------------------------------------------------------


def phase_gate(args, proto):
    """Try each env count in its own process; keep the largest that runs, with its throughput."""
    out_dir = proto["log_dir"]
    os.makedirs(out_dir, exist_ok=True)
    trials = {}
    for n in proto["gate_env_counts"]:
        res = os.path.join(out_dir, "gate", f"n{n}.json")
        if not os.path.exists(res):
            os.makedirs(os.path.dirname(res), exist_ok=True)
            subprocess.run(_self_cmd(args, "--phase", "gate-trial", "--num-envs", str(n), "--out", res))
        trials[n] = json.load(open(res)) if os.path.exists(res) else {"ok": False, "error": "process died"}
    ok = [n for n in proto["gate_env_counts"] if trials[n].get("ok")]
    if not ok:
        raise SystemExit(f"no env count fits: {trials}")
    n = max(ok)
    per_it = n * int(proto["num_steps_per_env"])
    iterations = math.ceil(float(proto["max_transitions"]) / per_it)
    frozen = {"num_envs": n, "transitions_per_iteration": per_it, "iterations": iterations,
              "save_interval": max(1, iterations // 10), "trials": trials,
              "estimated_hours": iterations * per_it / trials[n]["steps_per_second"] / 3600.0,
              "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(os.path.join(out_dir, "gate.json"), "w") as f:
        json.dump(frozen, f, indent=2)
    print(json.dumps(frozen, indent=2))


def phase_gate_trial(args, proto):
    import torch

    from lares.envs.isaac_lab_adapter import close_isaac_app, launch_isaac_app

    launch_isaac_app(headless=True)
    out = {"num_envs": args.num_envs, "ok": False}
    try:
        env = make_training_env(args.num_envs, int(proto["training_seed"]), "cuda:0")
        agent = make_agent_cfg(int(proto["training_seed"]), int(proto["gate_iterations"]), 10 ** 9, "gate")
        runner, _ = build_runner(env, agent, None, "cuda:0")
        torch.cuda.synchronize()
        t0 = time.time()
        runner.learn(num_learning_iterations=int(proto["gate_iterations"]), init_at_random_ep_len=True)
        torch.cuda.synchronize()
        dt = time.time() - t0
        steps = int(proto["gate_iterations"]) * args.num_envs * int(proto["num_steps_per_env"])
        out.update(ok=True, seconds=dt, steps=steps, steps_per_second=steps / dt,
                   max_memory_gb=torch.cuda.max_memory_allocated() / 1e9)
        env.close()
    except Exception as e:  # noqa: BLE001 - the gate records any failure, OOM included
        out["error"] = repr(e)[:500]
    finally:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        close_isaac_app()


def phase_train(args, proto):
    from lares.envs.isaac_lab_adapter import close_isaac_app, launch_isaac_app

    gate = json.load(open(os.path.join(proto["log_dir"], "gate.json")))
    launch_isaac_app(headless=True)
    try:
        env = make_training_env(gate["num_envs"], int(proto["training_seed"]), "cuda:0")
        agent = make_agent_cfg(int(proto["training_seed"]), gate["iterations"], gate["save_interval"], "pen_spin_ppo")
        log_dir = os.path.join(proto["log_dir"], "train")
        runner, _ = build_runner(env, agent, log_dir, "cuda:0")
        with open(os.path.join(proto["log_dir"], "train_started.json"), "w") as f:
            json.dump({"gate": gate, "agent_cfg": agent.to_dict(), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, f,
                      indent=2, default=str)
        runner.learn(num_learning_iterations=gate["iterations"], init_at_random_ep_len=True)
        env.close()
    finally:
        close_isaac_app()


def checkpoints(proto) -> list:
    paths = glob.glob(os.path.join(proto["log_dir"], "train", "model_*.pt"))
    return sorted(paths, key=lambda p: int(re.findall(r"model_(\d+)\.pt", p)[0]))


def final_checkpoint(proto) -> str:
    gate = json.load(open(os.path.join(proto["log_dir"], "gate.json")))
    # rsl_rl numbers iterations from 0 and saves the last one as model_{iterations - 1}.
    path = os.path.join(proto["log_dir"], "train", f"model_{gate['iterations'] - 1}.pt")
    if not os.path.exists(path):
        raise SystemExit(f"final scheduled checkpoint {path} missing: training incomplete")
    return path


def run_ppo_cell(args, proto):
    """Child body: the PPO checkpoint on one manifest (shares protocols.run_cell's output format)."""
    from lares.envs.isaac_lab_adapter import close_isaac_app, launch_isaac_app

    launch_isaac_app(headless=True)
    try:
        from lares.envs.isaac_pen_spin import protocols

        ctrl = PPOController(args.checkpoint, int(proto["training_seed"]))
        protocols.run_cell(args.cell_manifest, ctrl, args.out, "cuda:0", record_obs=args.record_obs)
    finally:
        close_isaac_app()


def _ppo_child(args, manifest, checkpoint, out, record_obs=False):
    if os.path.exists(out):
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = _self_cmd(args, "--phase", "cell", "--cell-manifest", manifest, "--checkpoint", checkpoint, "--out", out)
    if record_obs:
        cmd.append("--record-obs")
    print("running", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def phase_diagnose(args, proto):
    from lares.envs.isaac_pen_spin import cases

    rows = []
    for ck in checkpoints(proto):
        it = int(re.findall(r"model_(\d+)\.pt", ck)[0])
        for stage in (3,):  # predicates nest; stage 1/2 rates are scored offline from the same traces
            m = os.path.join(proto["log_dir"], "debug_manifests", f"pen_spin_stage{stage}_debug_s{proto['debug_seed']}.yaml")
            out = os.path.join(proto["log_dir"], "diagnostics", f"model_{it}_stage{stage}.json")
            _ppo_child(args, m, ck, out)
            rows.append({"iteration": it, **summarise_multistage(out)})
    with open(os.path.join(proto["log_dir"], "diagnostics", "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    for r in rows:
        print(json.dumps(r))
    return rows, cases


def summarise_multistage(cell_json: str) -> dict:
    """Sustained rate at every stage from one Stage-3 trajectory set (identical physics, nested predicates)."""
    import numpy as np

    from lares.envs.isaac_pen_spin import cases

    tr = np.load(cell_json.replace(".json", ".npz"))
    L, comp = tr["length"], tr["completed"]
    out = {"completed": float(comp.mean()), "mean_length": float(L.mean())}
    for k, spec in cases.STAGES.items():
        ok = (tr["center_dist"] < spec.success_center_tolerance) & (np.abs(tr["spin_rate"]) > spec.success_spin_rate)
        n = len(L)
        out[f"sustained_s{k}"] = float(np.mean([cases.sustained(ok[:L[i], i], L[i], bool(comp[i])) for i in range(n)]))
        out[f"tail_s{k}"] = float(np.mean([ok[max(0, L[i] - 60):L[i], i].mean() if L[i] else 0.0 for i in range(n)]))
    spin = np.abs(np.nan_to_num(tr["spin_rate"]))
    out["mean_abs_spin"] = float(spin.sum() / max(1, L.sum()))
    out["mean_transverse"] = float(np.nan_to_num(tr["transverse_rate"]).sum() / max(1, L.sum()))
    return out


def phase_evaluate(args, proto):
    import numpy as np

    from lares.envs.isaac_pen_spin import protocols
    from lares.eval.stats import t_summary

    ck = final_checkpoint(proto)
    out_dir = proto["log_dir"]
    # Reproducibility gate for the PPO cell (R-039 amended rule), on the debug seed.
    control = {}
    for stage in (1, 2, 3):
        m = os.path.join(out_dir, "debug_manifests", f"pen_spin_stage{stage}_debug_s{proto['debug_seed']}.yaml")
        outs = [os.path.join(out_dir, "control", f"stage{stage}_ppo_expert_{r}.json") for r in "ab"]
        for o in outs:
            _ppo_child(args, m, ck, o)
        a, b = (np.load(o.replace(".json", ".npz")) for o in outs)
        ra, rb = (json.load(open(o))["episodes"] for o in outs)
        union = protocols._diverged_envs(a, b, "spin_rate") | protocols._diverged_envs(a, b, "center_dist")
        flips = sum(x["sustained"] != y["sustained"] for x, y in zip(ra, rb))
        control[f"stage{stage}/ppo_expert"] = {"union_diverged_envs": len(union), "sustained_outcome_flips": int(flips),
                                               "passes": bool(flips == 0 and len(union) <= protocols.MAX_DIVERGED_ENVS)}
    if not all(c["passes"] for c in control.values()):
        raise SystemExit(f"reproducibility control failed: {control}")
    per_stage = {}
    for stage in (1, 2, 3):
        m = proto["manifests"][stage]
        ppo_out = os.path.join(out_dir, f"stage{stage}_ppo_expert.json")
        gait_out = os.path.join(out_dir, f"stage{stage}_feedback_gait.json")
        _ppo_child(args, m, ck, ppo_out, record_obs=True)
        if not os.path.exists(gait_out):
            subprocess.run([sys.executable, "-u", os.path.join(_ROOT, "scripts", "run_shadowhand_spin_stages.py"),
                            "--config", args.config, "--research-protocol", "scripted_expert_qualification",
                            "--cell-manifest", m, "--controller", "feedback_gait", "--cell-out", gait_out], check=True)
        p = [float(e["sustained"]) for e in json.load(open(ppo_out))["episodes"]]
        g = [float(e["sustained"]) for e in json.load(open(gait_out))["episodes"]]
        d = [pi - gi for pi, gi in zip(p, g)]
        t = t_summary(d)
        per_stage[f"stage{stage}"] = {
            "n": len(d), "ppo_successes": int(sum(p)), "gait_successes": int(sum(g)),
            "ppo_success": float(np.mean(p)), "gait_success": float(np.mean(g)),
            "paired_success_difference": t["mean"], "paired_t": t,
            "qualifies": int(sum(p)) >= protocols.QUALIFY_MIN,
            "promotes": bool(t["mean"] > 0 and t["low"] is not None and t["low"] > 0),
            "ppo_diagnostics": summarise_multistage(ppo_out),
        }
    summary = {"checkpoint": ck, "reproducibility_control": control, "per_stage": per_stage}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def build_manifests(args, proto):
    from lares.envs.isaac_pen_spin import cases
    from lares.eval.manifest import EvaluationManifest

    written = []
    specs = [(k, p, int(proto["promotion_seeds"][k]), int(proto["promotion_cases"]), "promotion")
             for k, p in proto["manifests"].items()]
    specs += [(k, os.path.join(proto["log_dir"], "debug_manifests", f"pen_spin_stage{k}_debug_s{proto['debug_seed']}.yaml"),
               int(proto["debug_seed"]), int(proto["debug_cases"]), "debug") for k in (1, 2, 3)]
    for stage, path, seed, n, label in specs:
        m = cases.build_manifest(int(stage), seed, n, label=label)
        if os.path.exists(path):
            if EvaluationManifest.load(path).to_dict() != m.to_dict():
                raise ValueError(f"{path} exists and differs")
            continue
        m.save(path)
        written.append(path)
    print("wrote", written or "nothing (all present and identical)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(_ROOT, "config", "shadowhand_spin_stages.yaml"))
    ap.add_argument("--research-protocol", default=PROTOCOL)
    ap.add_argument("--phase", default=None,
                    choices=["build-manifests", "gate", "gate-trial", "train", "diagnose", "evaluate", "cell"])
    ap.add_argument("--num-envs", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--cell-manifest", default=None)
    ap.add_argument("--record-obs", action="store_true")
    args = ap.parse_args()
    if args.research_protocol != PROTOCOL:
        raise SystemExit(f"this script runs only {PROTOCOL!r}")
    proto = load_protocol(args.config)
    phases = {"build-manifests": build_manifests, "gate": phase_gate, "gate-trial": phase_gate_trial,
              "train": phase_train, "diagnose": phase_diagnose, "evaluate": phase_evaluate, "cell": run_ppo_cell}
    if args.phase:
        phases[args.phase](args, proto)
        return
    build_manifests(args, proto)
    if not os.path.exists(os.path.join(proto["log_dir"], "gate.json")):
        phase_gate(args, proto)
    gate = json.load(open(os.path.join(proto["log_dir"], "gate.json")))
    if not os.path.exists(os.path.join(proto["log_dir"], "train", f"model_{gate['iterations'] - 1}.pt")):
        subprocess.run(_self_cmd(args, "--phase", "train"), check=True)
    phase_diagnose(args, proto)
    phase_evaluate(args, proto)


if __name__ == "__main__":
    main()
