"""Research protocols on the pen-spin task, driven by ``scripts/run_shadowhand_spin_stages.py``.

``episode_rule_audit`` (R-038): ten frozen probe cases per stage, two diagnostic controllers, and both
episode reducers scored offline from the same trajectories. A second rollout in one process is not
reproducible (see ``runner``), so the driver runs each (stage, controller) in its own child process
(``--stage --controller``) and aggregates their JSON afterwards.
"""

from __future__ import annotations

import json
import os
import subprocess

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.eval.manifest import EvaluationManifest, assert_disjoint
from lares.utils.cli import self_argv

EPISODE_RULE_AUDIT = "episode_rule_audit"


# ---------------------------------------------------------------------------------------------------------------------
#  Protocol-config readers and frozen-file idioms shared by every --study module
# ---------------------------------------------------------------------------------------------------------------------


def roles(protocol_cfg: dict) -> dict:
    """``{role: (start, stop)}`` index ranges of a block manifest, from the protocol's ``roles`` block."""
    return {k: (int(v[0]), int(v[1])) for k, v in protocol_cfg["roles"].items()}


def manifest_path(protocol_cfg: dict) -> str:
    """The one frozen manifest a single-manifest study names."""
    return protocol_cfg["manifest"]


def disjoint_from(protocol_cfg: dict) -> list:
    """The committed manifests the protocol's ``disjoint_from`` list names, loaded (for ``assert_disjoint``)."""
    return [EvaluationManifest.load(p) for p in protocol_cfg.get("disjoint_from", [])]


def cached_json(path: str, compute, indent: int = 2):
    """The JSON at ``path`` if it exists, else ``compute()`` written there: the offline-gate idiom (computed once)."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    out = compute()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=indent)
    return out


def freeze_json(path: str, payload: dict, what: str) -> bool:
    """Write ``payload`` at ``path`` unless an identical one is there; refuse a different one. True when written."""
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            if json.load(f) != json.loads(json.dumps(payload)):
                raise SystemExit(f"{path} exists and differs from the {what} this run produces")
        return False
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    return True


def drive_params(state: dict) -> dict:
    """``spin_target`` and ``speed_gain`` of a learner state dict: what ``scalar_tuning.saturation`` recomputes the drive from."""
    return {"spin_target": float(state["spin_target"]), "speed_gain": float(state["speed_gain"])}


def learner_drive_params(protocol_cfg: dict) -> dict:
    """:func:`drive_params` of the protocol's frozen learner (``learner_recovery.make_learner``)."""
    from lares.envs.isaac_pen_spin.learner_recovery import make_learner

    return drive_params(make_learner(protocol_cfg).policy.state_dict())


def load_cell(out_path: str) -> tuple:
    """``(trace arrays, episode rows)`` of one saved cell: the ``.npz`` beside the ``.json`` and its ``episodes``."""
    with open(out_path, encoding="utf-8") as f:
        rows = json.load(f)["episodes"]
    # A plain dict: an ``NpzFile`` decompresses a whole member again on every ``[...]`` access.
    return dict(np.load(out_path.replace(".json", ".npz"))), rows


def in_play(obs, length):
    """``(i, L)`` for every recorded case with in-play steps, ``L = min(length[i], T) > 0`` (``obs`` is ``(cases, T, d)``)."""
    for i in range(obs.shape[0]):
        L = int(min(length[i], obs.shape[1]))
        if L > 0:
            yield i, L


def actions_match(reference, candidate, obs, length, before_case=None) -> dict:
    """The offline identity gate: ``candidate`` must reproduce ``reference``'s actions bit for bit on every archived case.

    Each case's in-play observations go through ``reference`` first, then ``candidate``; ``before_case()`` (if given)
    runs before each case. Returns ``{"rows", "mismatch_cases", "passes"}``.
    """
    import torch

    mism, rows = [], 0
    for i, L in in_play(obs, length):
        if before_case is not None:
            before_case()
        o = torch.as_tensor(obs[i, :L], dtype=torch.float32)
        rows += L
        if not torch.equal(reference(o, 0, None), candidate(o, 0, None)):
            mism.append(i)
    return {"rows": rows, "mismatch_cases": mism, "passes": not mism}


def isaac_shell(isaac_env: str, command: str, *, log: str | None = None) -> None:
    """Run ``command`` in a bash that first sources Isaac Lab's environment script.

    Without ``log`` the child inherits stdout and a failure raises ``CalledProcessError``; with ``log`` its output
    goes to that file and a failure raises ``RuntimeError`` carrying the log's tail.
    """
    cmd = f"source {isaac_env} >/dev/null 2>&1 && {command}"
    if log is None:
        subprocess.run(["bash", "-c", cmd], check=True)
        return
    with open(log, "w", encoding="utf-8") as f:
        proc = subprocess.run(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        with open(log, encoding="utf-8", errors="replace") as f:
            tail = f.read()[-3000:]
        raise RuntimeError(f"Isaac child failed (exit {proc.returncode}); log {log}:\n{tail}")


def audit_manifest_paths(protocol_cfg: dict) -> dict:
    return {int(k): v for k, v in protocol_cfg["manifests"].items()}


def build_audit_manifests(protocol_cfg: dict) -> list[str]:
    """Write the probe manifests the protocol names; refuses to overwrite a different one."""
    written = []
    n = int(protocol_cfg["cases_per_stage"])
    for stage, path in audit_manifest_paths(protocol_cfg).items():
        seed = int(protocol_cfg["pool_seeds"][stage])
        if cases.freeze_manifest(path, cases.build_manifest(stage, seed, n, label="probe")):
            written.append(path)
    return written


def build_single_manifest(protocol_cfg: dict, label: str, stage: int) -> list[str]:
    """The one-manifest protocols' ``build_manifests``: ``pool_seed`` and ``cases`` from the config, frozen at
    ``manifest`` and placement-disjoint from every manifest in ``disjoint_from``."""
    path = manifest_path(protocol_cfg)
    m = cases.build_manifest(stage, int(protocol_cfg["pool_seed"]), int(protocol_cfg["cases"]), label=label)
    assert_disjoint([m] + disjoint_from(protocol_cfg))  # before the write: a refused manifest never reaches disk
    return [path] if cases.freeze_manifest(path, m) else []


def run_audit_cell(stage: int, controller: str, protocol_cfg: dict, out_dir: str, device: str) -> str:
    """Child process body: one controller on one stage's probe manifest (one run per process)."""
    from lares.envs.isaac_pen_spin.runner import PenSpinBatchRunner, reduce_episodes

    ctrl = controller_factory(controller)
    manifest = EvaluationManifest.load(audit_manifest_paths(protocol_cfg)[stage])
    runner = PenSpinBatchRunner(stage, len(manifest), device=device)
    try:
        trace = runner.run(manifest, ctrl)
    finally:
        runner.close()
    np.savez_compressed(
        os.path.join(out_dir, f"trace_stage{stage}_{controller}.npz"),
        **{k: trace[k] for k in ("length", "completed", "pred", "center_dist", "spin_rate",
                                 "transverse_rate", "reward")},
    )
    result = {"stage": stage, "controller": controller, "manifest_id": manifest.manifest_id,
              "max_episode_length": int(trace["max_episode_length"]), "episodes": reduce_episodes(trace)}
    path = _cell_path(out_dir, stage, controller)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return path


def _cell_path(out_dir: str, stage: int, controller: str) -> str:
    return os.path.join(out_dir, f"stage{stage}_{controller}.json")


def aggregate_audit(out_dir: str, stages=(1, 2, 3)) -> dict:
    """Both reducers per stage and controller, and the pooled paired difference sustained - any_hit."""
    from lares.eval.stats import t_summary

    from lares.envs.isaac_pen_spin.controllers import AUDIT_CONTROLLERS

    per_cell, diffs, any_all, sus_all = {}, [], [], []
    for stage in stages:
        for ctrl in (c.name for c in AUDIT_CONTROLLERS):
            with open(_cell_path(out_dir, stage, ctrl), encoding="utf-8") as f:
                eps = json.load(f)["episodes"]
            a = [float(e["any_hit"]) for e in eps]
            s = [float(e["sustained"]) for e in eps]
            per_cell[f"stage{stage}/{ctrl}"] = {
                "n": len(eps),
                "any_hit_rate": float(np.mean(a)),
                "sustained_rate": float(np.mean(s)),
                "completed_rate": float(np.mean([e["completed"] for e in eps])),
                "mean_length": float(np.mean([e["length"] for e in eps])),
                "mean_hit_steps": float(np.mean([e["hit_steps"] for e in eps])),
                "mean_tail_hit_fraction": float(np.mean([e["tail_hit_fraction"] for e in eps])),
                "mean_abs_spin": float(np.mean([e["mean_abs_spin"] for e in eps])),
                "max_abs_spin": float(np.max([e["max_abs_spin"] for e in eps])),
                "mean_transverse": float(np.mean([e["mean_transverse"] for e in eps])),
                "first_hit_steps": [e["first_hit_step"] for e in eps],
            }
            diffs += [si - ai for si, ai in zip(s, a)]
            any_all += a
            sus_all += s
    return {
        "per_stage_controller": per_cell,
        "episodes": len(diffs),
        "success_rate_any_hit": float(np.mean(any_all)),
        "success_rate_sustained": float(np.mean(sus_all)),
        "paired_difference_sustained_minus_any_hit": t_summary(diffs),
    }


def run_episode_rule_audit(protocol_cfg: dict, config_path: str, out_dir: str, device: str) -> dict:
    from lares.envs.isaac_pen_spin.controllers import AUDIT_CONTROLLERS

    os.makedirs(out_dir, exist_ok=True)
    stages = tuple(sorted(audit_manifest_paths(protocol_cfg)))
    for stage in stages:
        for ctrl in (c.name for c in AUDIT_CONTROLLERS):
            if os.path.exists(_cell_path(out_dir, stage, ctrl)):
                continue
            cmd = self_argv(config_path) + ["--research-protocol", EPISODE_RULE_AUDIT, "--stage", str(stage),
                                            "--controller", ctrl, "--log-dir", out_dir]
            print("running", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
    summary = aggregate_audit(out_dir, stages)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------------------
#  R-039: scripted expert qualification
# ---------------------------------------------------------------------------

SCRIPTED_EXPERT_QUALIFICATION = "scripted_expert_qualification"
QUALIFY_MIN = 27  # of 30 promotion cases per stage
SCREEN = (0, 10)
PROMOTION = (10, 40)


def controller_factory(name: str):
    from lares.envs.isaac_pen_spin.controllers import AUDIT_CONTROLLERS
    from lares.envs.isaac_pen_spin.scripted_expert import FeedbackGait, frozen_gait

    table = {c.name: c for c in AUDIT_CONTROLLERS}
    table[FeedbackGait.name] = frozen_gait
    return table[name]()


def build_qualification_manifests(protocol_cfg: dict) -> list[str]:
    written = []
    specs = [(int(k), v, int(protocol_cfg["pool_seeds"][k]), "development") for k, v in protocol_cfg["manifests"].items()]
    specs.append((None, None, int(protocol_cfg["control_seed"]), "debug"))
    for stage, path, seed, label in specs:
        stages = [stage] if stage is not None else [int(k) for k in protocol_cfg["manifests"]]
        for st in stages:
            p = path or os.path.join(protocol_cfg["log_dir"], "control_manifests", f"pen_spin_stage{st}_debug_s{seed}.yaml")
            if cases.freeze_manifest(p, cases.build_manifest(st, seed, int(protocol_cfg["cases_per_stage"]), label=label)):
                written.append(p)
    return written


def run_cell(manifest_path: str, controller: str, out_path: str, device: str, record_obs: bool = False) -> str:
    """Child process body: one controller on one whole manifest, saved as JSON plus an .npz trace."""
    from lares.envs.isaac_pen_spin.runner import PenSpinBatchRunner, reduce_episodes
    from lares.utils.hashing import sha256_file

    manifest = EvaluationManifest.load(manifest_path)
    ctrl = controller_factory(controller) if isinstance(controller, str) else controller
    runner = PenSpinBatchRunner(cases.stage_of(manifest), len(manifest), device=device)
    try:
        if hasattr(ctrl, "attach"):  # a learned policy built against the live env's spaces
            ctrl.attach(runner.env)
        trace = runner.run(manifest, ctrl, record_obs=record_obs)
    finally:
        runner.close()
    keys = ["length", "completed", "pred", "center_dist", "spin_rate", "transverse_rate", "reward"]
    if record_obs:
        keys += ["obs", "actions"]
    np.savez_compressed(out_path.replace(".json", ".npz"), **{k: trace[k] for k in keys})
    result = {
        "stage": cases.stage_of(manifest), "controller": ctrl.name, "manifest_id": manifest.manifest_id,
        "manifest_path": manifest_path, "manifest_sha256": sha256_file(manifest_path),
        "controller_params": getattr(ctrl, "params", None),
        "reducer": "sustained: time-out without a fall and the per-step predicate on all final "
                   f"{cases.SUSTAIN_WINDOW} steps (D-034)",
        "max_episode_length": int(trace["max_episode_length"]), "episodes": reduce_episodes(trace),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return out_path


def child_cell(config_path: str, manifest_path: str, controller: str, out_path: str, *, protocol: str | None = None,
               study: str | None = None, protocol_cfg: dict | None = None, record_obs: bool = False,
               skip_existing: bool = False) -> None:
    """Re-exec this script for one cell (one controller on one manifest in a fresh Isaac process, D-034).

    ``protocol`` names a ``--research-protocol``, ``study`` a ``--study``; exactly one is given. ``study`` defaults to
    ``protocol_cfg["study"]``, which ``run_shadowhand_spin_stages.run_study`` sets to the registry name it was
    invoked under, so a module serving two names re-executes under the right one. With ``skip_existing`` a cell
    whose output exists is not rerun, which is how every driver resumes.
    """
    if study is None and protocol_cfg is not None:
        study = protocol_cfg.get("study")
    if (protocol is None) == (study is None):
        raise ValueError("child_cell takes exactly one of protocol= and study= (or a protocol_cfg naming its study)")
    if skip_existing and os.path.exists(out_path):
        print("exists", out_path, flush=True)
        return
    flag, name = ("--study", study) if study is not None else ("--research-protocol", protocol)
    cmd = self_argv(config_path) + [flag, name, "--cell-manifest", manifest_path, "--controller", controller,
                                    "--cell-out", out_path]
    if record_obs:
        cmd.append("--record-obs")
    print("running", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def study_cell(protocol_cfg: dict, config_path: str, manifest_path: str, controller: str, out_path: str,
               record_obs: bool = True) -> None:
    """A ``--study`` driver's cell: :func:`child_cell` under the study ``protocol_cfg`` names, skipping a saved one."""
    child_cell(config_path, manifest_path, controller, out_path, protocol_cfg=protocol_cfg, record_obs=record_obs,
               skip_existing=True)


# ---------------------------------------------------------------------------------------------------------------------
#  Per-arm row summaries shared by the feedback studies (R-041, R-044, R-047, R-051, R-053)
# ---------------------------------------------------------------------------------------------------------------------


def arm_summary(rows: list[dict]) -> dict:
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


def failure_split(rows: list[dict]) -> dict:
    drops = [r for r in rows if not r["completed"]]
    tail = [r for r in rows if r["completed"] and not r["sustained"]]
    return {"drops": len(drops), "tail_spin_failures": len(tail),
            "median_fall_step": float(np.median([r["length"] for r in drops])) if drops else None}


def aggregate_vs_expert(rows: dict, arms, primary: str) -> dict:
    """Per-arm summaries and each of ``arms`` minus the ``"expert"`` rows, paired by case.

    ``primary`` names the arm whose paired mean is reported as ``paired_success_difference``.
    """
    from lares.eval.stats import t_summary

    base = rows["expert"]
    out = {"arms": {}}
    for arm, rs in rows.items():
        if [r["task_hash"] for r in rs] != [r["task_hash"] for r in base]:
            raise ValueError(f"{arm} rows are not paired case for case with the expert")
        out["arms"][arm] = {**arm_summary(rs), **failure_split(rs)}
    for arm in arms:
        if arm in rows:
            d = [float(x["sustained"]) - float(e["sustained"]) for x, e in zip(rows[arm], base)]
            out[f"{arm}_minus_expert"] = t_summary(d)
    head = out.get(f"{primary}_minus_expert")
    out["paired_success_difference"] = head["mean"] if head else None
    return out


#: Amended gate (advisor consult call_00075, before any development case ran): the first, bit-identity
#: version failed on 5 of 6 cells (GPU PhysX at batch 40 is not bit-deterministic across processes). A cell
#: now passes with zero paired sustained-outcome flips and at most this many of its environments diverging
#: in spin_rate or center_dist (union). Bit-level counts are still reported.
MAX_DIVERGED_ENVS = 4


def diverged_envs(a, b, key: str, tol: float = 0.0) -> set:
    """Environments (columns) whose ``(T, envs)`` array ``key`` differs between two traces by more than ``tol``."""
    x, y = np.nan_to_num(a[key], nan=-1e9), np.nan_to_num(b[key], nan=-1e9)
    return {int(i) for i in np.nonzero((np.abs(x - y) > tol).any(axis=0))[0]}


def reproducibility_control(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    """Each (stage, controller) on the debug manifest in two fresh processes, compared episode by episode."""
    ctrl_dir = os.path.join(out_dir, "control")
    os.makedirs(ctrl_dir, exist_ok=True)
    report = {}
    for stage in sorted(int(k) for k in protocol_cfg["manifests"]):
        mpath = os.path.join(protocol_cfg["log_dir"], "control_manifests",
                             f"pen_spin_stage{stage}_debug_s{int(protocol_cfg['control_seed'])}.yaml")
        for ctrl in protocol_cfg["controllers"]:
            outs, rows = [], []
            for rep in ("a", "b"):
                out = os.path.join(ctrl_dir, f"stage{stage}_{ctrl}_{rep}.json")
                if not os.path.exists(out):
                    child_cell(config_path, mpath, ctrl, out, protocol=SCRIPTED_EXPERT_QUALIFICATION, record_obs=False)
                trace, episodes = load_cell(out)
                outs.append(trace)
                rows.append(episodes)
            spin = diverged_envs(outs[0], outs[1], "spin_rate")
            centre = diverged_envs(outs[0], outs[1], "center_dist")
            union = spin | centre | {i for i in range(len(rows[0]))
                                     if rows[0][i]["length"] != rows[1][i]["length"]}
            flips = sum(r0["sustained"] != r1["sustained"] for r0, r1 in zip(rows[0], rows[1]))
            report[f"stage{stage}/{ctrl}"] = {
                "bit_identical": not union,
                "spin_rate_diverged_envs": len(spin), "center_dist_diverged_envs": len(centre),
                "union_diverged_envs": len(union), "sustained_outcome_flips": int(flips),
                "passes": bool(flips == 0 and len(union) <= MAX_DIVERGED_ENVS),
            }
    with open(os.path.join(ctrl_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def aggregate_qualification(protocol_cfg: dict, out_dir: str) -> dict:
    from lares.eval.stats import t_summary

    base, cand = protocol_cfg["baseline"], protocol_cfg["candidate"]
    per_stage, pooled = {}, []
    for stage in sorted(int(k) for k in protocol_cfg["manifests"]):
        rows = {}
        for ctrl in (base, cand):
            with open(os.path.join(out_dir, f"stage{stage}_{ctrl}.json"), encoding="utf-8") as f:
                rows[ctrl] = json.load(f)["episodes"]
        block = {}
        for split, (lo, hi) in (("screen", SCREEN), ("promotion", PROMOTION)):
            b = [float(e["sustained"]) for e in rows[base][lo:hi]]
            c = [float(e["sustained"]) for e in rows[cand][lo:hi]]
            d = [ci - bi for ci, bi in zip(c, b)]
            block[split] = {
                "n": len(d),
                f"{base}_success": float(np.mean(b)), f"{cand}_success": float(np.mean(c)),
                f"{cand}_successes": int(sum(c)),
                f"{base}_tail_hit_fraction": float(np.mean([e["tail_hit_fraction"] for e in rows[base][lo:hi]])),
                f"{cand}_tail_hit_fraction": float(np.mean([e["tail_hit_fraction"] for e in rows[cand][lo:hi]])),
                f"{cand}_completed": float(np.mean([e["completed"] for e in rows[cand][lo:hi]])),
                f"{cand}_mean_abs_spin": float(np.mean([e["mean_abs_spin"] for e in rows[cand][lo:hi]])),
                f"{cand}_mean_transverse": float(np.mean([e["mean_transverse"] for e in rows[cand][lo:hi]])),
                "paired_difference": t_summary(d),
            }
            if split == "promotion":
                pooled += d
        prom = block["promotion"]
        block["qualifies"] = {c: int(sum(float(e["sustained"]) for e in rows[c][PROMOTION[0]:PROMOTION[1]])) >= QUALIFY_MIN
                              for c in (base, cand)}
        lo = prom["paired_difference"]["low"]
        block["promotes"] = bool(prom["paired_difference"]["mean"] > 0 and lo is not None and lo > 0)
        per_stage[f"stage{stage}"] = block
    return {"per_stage": per_stage, "pooled_promotion_pairs": len(pooled),
            "paired_success_difference": float(np.mean(pooled)) if pooled else None,
            "paired_success_difference_t": t_summary(pooled)}


def run_scripted_expert_qualification(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    control = reproducibility_control(protocol_cfg, config_path, out_dir)
    if not all(c["passes"] for c in control.values()):
        raise SystemExit(f"reproducibility control failed, evaluation not run: {control}")
    for stage in sorted(int(k) for k in protocol_cfg["manifests"]):
        mpath = protocol_cfg["manifests"][stage]
        for ctrl in protocol_cfg["controllers"]:
            out = os.path.join(out_dir, f"stage{stage}_{ctrl}.json")
            if not os.path.exists(out):
                child_cell(config_path, mpath, ctrl, out, protocol=SCRIPTED_EXPERT_QUALIFICATION,
                           record_obs=ctrl == protocol_cfg["candidate"])
    summary = {"reproducibility_control": control, **aggregate_qualification(protocol_cfg, out_dir)}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
