"""R-051: can the qualified PPO expert recover states the symbolic learner actually visits? Takeover on Stage 3.

The learner is the R-046 block-3 promoted controller (frozen copy in ``config/pen_spin/r046_block3_incumbent/``,
hashed); the expert is the D-036 PPO checkpoint. Branches are reconstructed by closed-loop re-execution, not by
action-tape replay: one frozen 32-case manifest (seed 4310) runs in three fresh Isaac processes with identical
batch composition, and every process attaches and queries both controllers at every step, so the process history
differs only in which action is executed.

* ``donor``: the learner throughout. Its trace fixes each case's branch step.
* ``learner_repeat``: the learner throughout again. Gate 1: every observation and every frozen-rule outcome must be
  bit-identical to the donor's on all 32 cases, else the takeover is not run.
* ``takeover``: the learner until the branch step, then the expert to the end. Gate 2: each switched lane's
  observations must equal the donor's at every step up to and including the one its first expert action reads.

Branch rule (frozen before execution): the first control step ``t`` in [60, 120) after which the pen is still in play
(the episode did not end at ``t``) and the Stage-3 per-step predicate fails by slow spin (``|spin_rate| <= 4.0``) or
off-centre (``center_dist >= 0.08``); the expert's first action is at ``t + 1``. With no such step the switch is at
step 120 if the pen is still in play then. A case that ended before its switch is never switched and counts as a
failure in both arms. Successful takeover qualifies recovery evidence for this learner/checkpoint pair; it is not a
symbolic repair.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin.feedback_ablation import _arm_summary, check_checkpoint
from lares.envs.isaac_pen_spin.feedback_lesions import failure_split

LEARNER_RECOVERY = "learner_recovery"
STAGE = 3
ARMS = ("donor", "learner_repeat", "takeover")
WINDOW = (60, 120)
TRANSITION_STEPS = 30
NEVER = 10**9


def branch_steps(length, pred, center_dist, spin_rate, stage: int = STAGE, window=WINDOW) -> np.ndarray:
    """Per case, the first control step whose action comes from the expert (``NEVER`` if the pen is gone first).

    ``pred``/``center_dist``/``spin_rate`` are the runner's (T, cases) arrays, recorded after each step's action;
    ``length`` counts the steps the case was in play, so the pen is still in play after step ``t`` iff ``length > t + 1``.
    """
    spec = cases.STAGES[stage]
    lo, hi = window
    length = np.asarray(length)
    out = np.full(len(length), NEVER, dtype=np.int64)
    for i in range(len(length)):
        for t in range(lo, hi):
            if length[i] <= t + 1:
                break
            if pred[t, i]:
                continue
            slow = abs(float(spin_rate[t, i])) <= spec.success_spin_rate
            off = float(center_dist[t, i]) >= spec.success_center_tolerance
            if slow or off:
                out[i] = t + 1
                break
        else:
            if length[i] > hi:
                out[i] = hi
    return out


class HandoverController:
    """Learner rows until their branch step, expert rows from it on. Both are evaluated on every step."""

    def __init__(self, learner, expert, steps, name: str):
        self.learner = learner
        self.expert = expert
        self.steps = np.asarray(steps, dtype=np.int64)
        self.name = name
        switched = self.steps < NEVER
        self.params = {"arm": name, "switched_cases": int(switched.sum()),
                       "branch_steps": [int(s) if s < NEVER else None for s in self.steps],
                       "learner": getattr(learner, "params", None), "expert": getattr(expert, "params", None)}

    def attach(self, env):
        for c in (self.learner, self.expert):
            if hasattr(c, "attach"):
                c.attach(env)

    def reset(self, ctx):
        import torch

        self.learner.reset(ctx)
        self.expert.reset(ctx)
        self._steps = torch.as_tensor(self.steps, device=ctx.device)

    def __call__(self, obs, t, ctx):
        a_learner = self.learner(obs, t, ctx).to(obs.device)
        a_expert = self.expert(obs, t, ctx).to(obs.device)
        use = (self._steps <= int(t)).unsqueeze(1)
        return (a_expert * use + a_learner * ~use).clone()


def manifest_path(protocol_cfg: dict) -> str:
    return protocol_cfg["manifest"]


def build_manifests(protocol_cfg: dict) -> list[str]:
    from lares.eval.manifest import EvaluationManifest

    path = manifest_path(protocol_cfg)
    m = cases.build_manifest(STAGE, int(protocol_cfg["pool_seed"]), int(protocol_cfg["cases"]), label="recovery")
    if os.path.exists(path):
        if EvaluationManifest.load(path).to_dict() != m.to_dict():
            raise ValueError(f"{path} exists and differs from the manifest this protocol builds")
        return []
    m.save(path)
    return [path]


def check_learner(protocol_cfg: dict) -> dict:
    from lares.utils.hashing import sha256_file

    lc = protocol_cfg["learner"]
    for key in ("source", "state"):
        got = sha256_file(lc[key])
        if got != lc[f"{key}_sha256"]:
            raise SystemExit(f"learner {key} {lc[key]} hash {got} != frozen {lc[f'{key}_sha256']}")
    return lc


def make_learner(protocol_cfg: dict):
    """The frozen learner (hashes checked), executed as ``clamp(mean, -1, 1)`` like every pipeline evaluation."""
    from lares.envs.isaac_pen_spin.layout import OBS_LAYOUT
    from lares.envs.isaac_pen_spin.pipeline_child import SymbolicController

    lc = check_learner(protocol_cfg)
    with open(lc["source"], encoding="utf-8") as f:
        source = f.read()
    return SymbolicController({"source": source, "state_path": lc["state"],
                               "obs_dim": sum(size for _, size, _ in OBS_LAYOUT), "action_dim": cases.NUM_ACTIONS,
                               "fixed_action_scale": float(lc["fixed_action_scale"]),
                               "execution": "clamp(mean, -1, 1)"}, cases.STAGES[STAGE].task_id)


def make_controller(arm: str, protocol_cfg: dict, out_dir: str):
    from lares.envs.isaac_pen_spin.ppo_expert import PPOController

    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; one of {ARMS}")
    learner = make_learner(protocol_cfg)
    expert = PPOController(check_checkpoint(protocol_cfg), int(protocol_cfg["training_seed"]))
    n = int(protocol_cfg["cases"])
    if arm == "takeover":
        with open(os.path.join(out_dir, "branch_steps.json"), encoding="utf-8") as f:
            steps = json.load(f)["steps"]
        steps = [NEVER if s is None else int(s) for s in steps]
    else:
        steps = [NEVER] * n
    return HandoverController(learner, expert, steps, arm)


def _child(config_path: str, manifest: str, arm: str, out_path: str, study: str = LEARNER_RECOVERY) -> None:
    if os.path.exists(out_path):
        print("exists", out_path, flush=True)
        return
    cmd = [sys.executable, "-u", sys.argv[0], "--config", config_path, "--study", study,
           "--cell-manifest", manifest, "--controller", arm, "--cell-out", out_path, "--record-obs"]
    print("running", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _load(out_path: str):
    return np.load(out_path.replace(".json", ".npz")), json.load(open(out_path))["episodes"]


def identity_gate(a, ra, b, rb) -> dict:
    """Gate 1: bit-identical observations (where either run is in play) and identical outcomes on every case."""
    n = len(ra)
    obs_diff = [i for i in range(n) if not np.array_equal(a["obs"][i, : int(a["length"][i])],
                                                         b["obs"][i, : int(b["length"][i])])]
    out_diff = [i for i in range(n) if (ra[i]["sustained"], ra[i]["completed"], ra[i]["length"])
                != (rb[i]["sustained"], rb[i]["completed"], rb[i]["length"])]
    return {"cases": n, "observation_mismatch_cases": obs_diff, "outcome_mismatch_cases": out_diff,
            "passes": not obs_diff and not out_diff}


def prefix_gate(donor, take, steps) -> dict:
    """Gate 2: each switched lane's observations equal the donor's through the one its first expert action reads."""
    bad = []
    for i, s in enumerate(steps):
        if s >= NEVER:
            continue
        if not np.array_equal(donor["obs"][i, : s + 1], take["obs"][i, : s + 1]):
            first = int(np.argmax((donor["obs"][i, : s + 1] != take["obs"][i, : s + 1]).any(axis=1)))
            bad.append({"case": i, "branch_step": int(s), "first_mismatch_step": first})
    return {"switched_cases": int(sum(s < NEVER for s in steps)), "mismatches": bad, "passes": not bad}


def transitions(donor, take, steps, k: int = TRANSITION_STEPS) -> dict:
    """Matched learner/expert (obs, executed action, next obs) windows from each switched case's branch step."""
    rows = {"case": [], "step": [], "obs": [], "learner_action": [], "learner_next_obs": [],
            "expert_action": [], "expert_next_obs": []}
    T = donor["obs"].shape[1]
    for i, s in enumerate(steps):
        if s >= NEVER:
            continue
        for t in range(s, min(s + k, T - 1)):
            if t + 1 >= min(int(donor["length"][i]), int(take["length"][i])):
                break
            rows["case"].append(i)
            rows["step"].append(t)
            rows["obs"].append(take["obs"][i, t])
            rows["learner_action"].append(donor["actions"][i, t])
            rows["learner_next_obs"].append(donor["obs"][i, t + 1])
            rows["expert_action"].append(take["actions"][i, t])
            rows["expert_next_obs"].append(take["obs"][i, t + 1])
    return {key: np.asarray(v) for key, v in rows.items()}


def aggregate(donor_rows, take_rows, steps) -> dict:
    from lares.eval.stats import t_summary

    if [r["task_hash"] for r in donor_rows] != [r["task_hash"] for r in take_rows]:
        raise ValueError("takeover rows are not paired case for case with the donor")
    d = [float(x["sustained"]) - float(e["sustained"]) for x, e in zip(take_rows, donor_rows)]
    t = t_summary(d)
    switched = [i for i, s in enumerate(steps) if s < NEVER]
    out = {
        "paired_success_difference": t["mean"], "takeover_minus_learner": t,
        "arms": {"learner": {**_arm_summary(donor_rows), **failure_split(donor_rows)},
                 "takeover": {**_arm_summary(take_rows), **failure_split(take_rows)}},
        "switched_cases": len(switched),
        "never_switched_cases": [i for i, s in enumerate(steps) if s >= NEVER],
        "branch_step_quartiles": (np.percentile([steps[i] for i in switched], [25, 50, 75]).tolist()
                                  if switched else None),
        "switched_only": None,
    }
    if switched:
        ds = [d[i] for i in switched]
        out["switched_only"] = {"n": len(switched), "learner_successes": int(sum(donor_rows[i]["sustained"] for i in switched)),
                                "takeover_successes": int(sum(take_rows[i]["sustained"] for i in switched)),
                                "difference": t_summary(ds),
                                "takeover_drops_after_switch": int(sum(not take_rows[i]["completed"] for i in switched)),
                                "learner_drops_after_switch": int(sum(not donor_rows[i]["completed"] for i in switched))}
    return out


def run_learner_recovery(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    """Driver: donor, repeat, Gate 1, branch steps, takeover, Gate 2, aggregate. Nothing is rerun if its output exists."""
    from lares.utils.hashing import sha256_file

    check_checkpoint(protocol_cfg)
    check_learner(protocol_cfg)
    os.makedirs(out_dir, exist_ok=True)
    m = manifest_path(protocol_cfg)
    if not os.path.exists(m):
        raise SystemExit(f"frozen manifest {m} missing; build and commit it before the run")
    out = {arm: os.path.join(out_dir, f"{arm}.json") for arm in ARMS}
    for arm in ("donor", "learner_repeat"):
        _child(config_path, m, arm, out[arm])
    donor, donor_rows = _load(out["donor"])
    rep, rep_rows = _load(out["learner_repeat"])
    gate1 = identity_gate(donor, donor_rows, rep, rep_rows)
    if not gate1["passes"]:
        with open(os.path.join(out_dir, "gate1_failed.json"), "w") as f:
            json.dump(gate1, f, indent=2)
        raise SystemExit(f"Gate 1 failed (branching unqualified), takeover not run: {gate1}")
    steps = branch_steps(donor["length"], donor["pred"], donor["center_dist"], donor["spin_rate"])
    bpath = os.path.join(out_dir, "branch_steps.json")
    payload = {"steps": [int(s) if s < NEVER else None for s in steps], "window": list(WINDOW),
               "rule": "first t in [60,120) still in play with the predicate failing by slow spin or off-centre; "
                       "expert from t+1; else 120 if in play; else never"}
    if os.path.exists(bpath):
        if json.load(open(bpath))["steps"] != payload["steps"]:
            raise SystemExit(f"{bpath} differs from the branch steps the donor trace gives")
    else:
        with open(bpath, "w") as f:
            json.dump(payload, f, indent=2)
    _child(config_path, m, "takeover", out["takeover"])
    take, take_rows = _load(out["takeover"])
    gate2 = prefix_gate(donor, take, steps)
    summary = aggregate(donor_rows, take_rows, steps) if gate2["passes"] else {"paired_success_difference": None}
    summary["gate1_identity"] = gate1
    summary["gate2_prefix"] = gate2
    summary["interpretable"] = bool(gate1["passes"] and gate2["passes"])
    summary["branch_steps"] = payload
    tr = transitions(donor, take, steps)
    np.savez_compressed(os.path.join(out_dir, "transitions.npz"), **tr)
    if len(tr["expert_action"]):
        summary["transitions"] = {"rows": int(len(tr["case"])),
                                  "expert_saturated_fraction": float(np.mean(np.abs(tr["expert_action"]) >= 1.0 - 1e-6)),
                                  "learner_saturated_fraction": float(np.mean(np.abs(tr["learner_action"]) >= 1.0 - 1e-6))}
    summary["manifest"] = {"path": m, "sha256": sha256_file(m)}
    summary["episodes"] = 3 * int(protocol_cfg["cases"])
    summary["limitation"] = ("takeover success qualifies recovery evidence for this learner/checkpoint pair only; it is "
                             "not a symbolic repair and says nothing about which individual labels are useful")
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


__all__ = ["ARMS", "HandoverController", "LEARNER_RECOVERY", "NEVER", "aggregate", "branch_steps", "build_manifests",
           "identity_gate", "make_learner", "prefix_gate", "run_learner_recovery", "transitions"]
