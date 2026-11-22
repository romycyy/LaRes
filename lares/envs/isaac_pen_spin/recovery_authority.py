"""R-114: can any scripted action get a stuck episode of the R-106 finalist rolling again? A diagnostic, no learning.

About 31% of the finalist's episodes are slow at step 120, and under 1% of those recover on their own (R-109's
archive). Four batches of ``4 x 512`` environments run the finalist on shared placements to step 120; from there,
in every *eligible* environment (alive, centred, 30-step mean |axial rate| below 3 rad/s), one slot does nothing and
three slots each apply one frozen 60-step intervention, then hand back to the finalist:

* ``recapture``    -- blend the sent action toward the hand's default (reset) posture over 20 steps and back over 40;
* ``stroke_boost`` -- the incumbent's fixed stroke gain times 1.5 inside the reference;
* ``residual_x3``  -- the finalist's residual with 3 x its coefficients, clipped at 0.30 instead of 0.10.

The finalist's own action and the reference's filter are computed every step for every environment; an intervention
only replaces the action that is sent. Slots rotate by one per batch. A pair is a placement whose control-slot and
probe-slot environments are both eligible. In-process rollouts (not the fresh-process evaluator): authority only.

``python -m lares.envs.isaac_pen_spin.recovery_authority <request.json>`` is the Isaac child.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin import symbolic_reward as sr
from lares.envs.isaac_pen_spin.protocols import freeze_json
from lares.utils.hashing import sha256_file, sha256_text

STUDY = "recovery_authority"
ROLES = ("control", "recapture", "stroke_boost", "residual_x3")
PROBES = ROLES[1:]


# ---------------------------------------------------------------------------------------------------------------------
#  Pure pieces
# ---------------------------------------------------------------------------------------------------------------------


def slot_roles(batch: int) -> list:
    """Slot ``k`` of batch ``b`` holds role ``(k + b) mod 4``."""
    return [ROLES[(k + int(batch)) % len(ROLES)] for k in range(len(ROLES))]


def beta(t: int, cfg: dict) -> float:
    """The recapture blend: 0 outside the window, rising over ``ramp_in`` steps to 1, falling over the rest."""
    t0, n, up = int(cfg["t0"]), int(cfg["window"]), int(cfg["ramp_in"])
    if t < t0 or t >= t0 + n:
        return 0.0
    return (t - t0 + 1) / up if t < t0 + up else (t0 + n - t) / (n - up)


def eligible(alive, centre_at_t0, mean_abs_spin, cfg: dict):
    return alive & (centre_at_t0 < float(cfg["centre_tolerance"])) & (mean_abs_spin < float(cfg["stuck_rate"]))


def placement_provenance(cfg: dict) -> dict:
    from lares.eval.manifest import EvaluationManifest, _hash_rand_vec

    hashes = {int(s): [_hash_rand_vec(v) for v in cases.placements(int(s), int(cfg["cases_per_slot"]))] for s in cfg["placement_seeds"]}
    flat = [h for v in hashes.values() for h in v]
    committed = set()
    for p in sorted(glob.glob("config/manifests/pen_spin_*.yaml")):
        committed |= {c.task_hash for c in EvaluationManifest.load(p).episodes}
    if len(set(flat)) != len(flat) or committed & set(flat):
        raise SystemExit("diagnostic placements collide with each other or with a committed manifest")
    return {"batches": {str(s): {"cases": len(v), "sha256": sha256_text("".join(v)), "slot_roles": slot_roles(b)}
                        for b, (s, v) in enumerate(hashes.items())}, "disjoint_from_committed_pen_spin_cases": len(committed)}


def by_role(batch: dict, b: int, per: int) -> dict:
    """``{role: {field: (per,) array}}`` of one batch's per-environment arrays."""
    return {role: {k: np.asarray(v)[slot * per:(slot + 1) * per] for k, v in batch.items() if k != "starts"}
            for slot, role in enumerate(slot_roles(b))}


def probe_statistics(batches: list, per: int, cfg: dict) -> dict:
    """Per probe: the paired contrast against the control over jointly eligible placements, pooled and per batch."""
    from lares.eval.stats import t_summary, win_loss_tie

    out, pvals = {}, {}
    for probe in PROBES:
        ds, dd, per_batch, tail = [], [], [], {"control": [], "probe": []}
        for b, batch in enumerate(batches):
            r = by_role(batch, b, per)
            joint = r["control"]["eligible"].astype(bool) & r[probe]["eligible"].astype(bool)
            d = r[probe]["sustained"][joint].astype(float) - r["control"]["sustained"][joint].astype(float)
            ds += list(d)
            dd += list((~r[probe]["completed"][joint].astype(bool)).astype(float) - (~r["control"]["completed"][joint].astype(bool)).astype(float))
            per_batch.append({"pairs": int(joint.sum()), "difference": float(d.mean()) if joint.any() else None,
                              "probe_conversions": int(r[probe]["sustained"][joint].sum()), "control_conversions": int(r["control"]["sustained"][joint].sum())})
            tail["control"] += list(r["control"]["tail_abs_spin"][joint])
            tail["probe"] += list(r[probe]["tail_abs_spin"][joint])
        ts, td = t_summary(ds), t_summary(dd, level=0.90)
        clustered = t_summary([p["difference"] for p in per_batch if p["difference"] is not None])
        el = [by_role(batch, b, per)[probe]["eligible"].astype(bool) for b, batch in enumerate(batches)]
        sus = [by_role(batch, b, per)[probe]["sustained"].astype(bool) for b, batch in enumerate(batches)]
        out[probe] = {"pairs": len(ds), "difference": ts["mean"], "ci95": [ts["low"], ts["high"]], "p": ts["p_value"],
                      "win_loss_tie": win_loss_tie(ds), "per_batch": per_batch, "batch_clustered": clustered,
                      "drop_difference": td["mean"], "drop_one_sided_95_upper": td["high"],
                      "drops_ok": bool(td["high"] is not None and td["high"] <= float(cfg["drop_margin"]) + 1e-12),
                      "unpaired_conversion": float(np.concatenate(sus)[np.concatenate(el)].mean()) if np.concatenate(el).any() else None,
                      "eligible": int(np.concatenate(el).sum()),
                      "tail_abs_spin_control_probe": [float(np.mean(tail["control"])), float(np.mean(tail["probe"]))] if ds else None}
        pvals[probe] = 1.0 if ts["p_value"] is None else ts["p_value"]
    family = sr.holm(pvals)
    thr = float(cfg["threshold"])
    for probe, o in out.items():
        o["p_holm"] = family[probe]["p_holm"]
        positive = bool(family[probe]["rejects"] and o["difference"] is not None and o["difference"] > 0)
        cl = o["batch_clustered"]
        agrees = bool(cl["low"] is not None and cl["low"] > 0)
        o["verdict"] = {"magnitude_supported": bool(o["ci95"][0] is not None and o["ci95"][0] >= thr),
                        "magnitude_refuted": bool(o["ci95"][1] is not None and o["ci95"][1] < thr),
                        "positive_case_paired": positive, "confirmed_across_batches": agrees,
                        "useful_scripted_authority": bool(positive and o["drops_ok"] and agrees)}
    return out


def control_summary(batches: list, per: int) -> dict:
    rows = [by_role(batch, b, per)["control"] for b, batch in enumerate(batches)]
    sus, el = np.concatenate([r["sustained"] for r in rows]).astype(bool), np.concatenate([r["eligible"] for r in rows]).astype(bool)
    return {"cases": int(sus.size), "success": float(sus.mean()), "eligible_fraction": float(el.mean()),
            "untreated_conversion": float(sus[el].mean()) if el.any() else None,
            "first_batch_success": float(rows[0]["sustained"].astype(bool).mean())}


def eligibility_agreement(batches: list, per: int) -> float:
    """Mean fraction of placements on which two slots agree about eligibility (all slots are identical before T0)."""
    vals = []
    for b, batch in enumerate(batches):
        r = by_role(batch, b, per)
        for a in ROLES:
            for c in ROLES:
                if a < c:
                    vals.append(float((r[a]["eligible"].astype(bool) == r[c]["eligible"].astype(bool)).mean()))
    return float(np.mean(vals))


# ---------------------------------------------------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------------------------------------------------


def build_manifests(protocol_cfg: dict) -> list[str]:
    return []


def make_controller(name: str, protocol_cfg: dict, out_dir: str | None = None):
    raise SystemExit("the recovery diagnostic runs in its own child process; it has no evaluation cells")


def finalist_theta(cfg: dict) -> list:
    with open(cfg["root_finalist"], encoding="utf-8") as f:
        entry = json.load(f)
    if entry["theta_sha256"] != cfg["root_theta_sha256"] or sr.theta_sha256(entry["theta"]) != entry["theta_sha256"]:
        raise SystemExit("the policy is not the frozen R-106 finalist")
    return entry["theta"]


def normalization(cfg: dict) -> dict:
    if sha256_file(cfg["normalization_path"]) != cfg["normalization_sha256"]:
        raise SystemExit(f"{cfg['normalization_path']} is not the frozen R-103 normalisation")
    with open(cfg["normalization_path"], encoding="utf-8") as f:
        return json.load(f)


def run(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    descriptor = sr.check_reference(protocol_cfg)
    finalist_theta(protocol_cfg)
    normalization(protocol_cfg)
    cfg_sha = sr.config_sha256(protocol_cfg)
    freeze_json(os.path.join(out_dir, "frozen_config.json"),
                {"config_sha256": cfg_sha, "config": sr.frozen_config(protocol_cfg), "reference": descriptor,
                 "placements": placement_provenance(protocol_cfg)}, "frozen configuration")
    resp, marker = os.path.join(out_dir, "rollouts.json"), os.path.join(out_dir, "rollouts.json.started")
    if not os.path.exists(resp):
        if os.path.exists(marker):
            raise SystemExit(f"{marker}: the diagnostic already started and left no result; it is not relaunched")
        req = os.path.join(out_dir, "rollouts.request.json")
        with open(req, "w", encoding="utf-8") as f:
            json.dump({"config_path": os.path.abspath(config_path), "study": STUDY, "config_sha256": cfg_sha, "response_path": resp}, f)
        with open(marker, "w", encoding="utf-8") as f:
            json.dump({"started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}, f)
        with open(os.path.join(out_dir, "rollouts.log"), "w", encoding="utf-8") as log:
            proc = subprocess.run([sys.executable, "-u", "-m", "lares.envs.isaac_pen_spin.recovery_authority", req],
                                  stdout=log, stderr=subprocess.STDOUT, timeout=3600)
        if proc.returncode != 0 or not os.path.exists(resp):
            raise RuntimeError(f"the diagnostic child failed (exit {proc.returncode}); see {out_dir}/rollouts.log")
        os.remove(marker)
    with open(resp, encoding="utf-8") as f:
        raw = json.load(f)
    per = int(protocol_cfg["cases_per_slot"])
    summary = {"study": STUDY, "config_sha256": cfg_sha, "starts": raw["starts"], "ceiling": int(protocol_cfg["start_ceiling"]),
               "batches_run": len(raw["batches"]), "halted": raw.get("halted"), "stroke_gain_constant": raw.get("stroke_gain_constant")}
    if raw.get("halted") is None:
        stats = probe_statistics(raw["batches"], per, protocol_cfg)
        ctrl = control_summary(raw["batches"], per)
        enough = stats["recapture"]["pairs"] >= int(protocol_cfg["min_pairs_total"])
        summary.update({"control": ctrl, "eligibility_agreement_between_slots": eligibility_agreement(raw["batches"], per),
                        "probes": stats, "paired_success_difference": stats["recapture"]["difference"],
                        "paired_success_difference_ci95": stats["recapture"]["ci95"]})
        if not enough:
            summary["halted"] = f"recapture has {stats['recapture']['pairs']} jointly eligible pairs, fewer than {protocol_cfg['min_pairs_total']}: no verdict"
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return summary


# ---------------------------------------------------------------------------------------------------------------------
#  Isaac child
# ---------------------------------------------------------------------------------------------------------------------


class PerEnvGain:
    """The incumbent's fixed stroke gain with a per-environment multiplier (1 everywhere reproduces the constant)."""

    def __init__(self, constant: float, n: int):
        import torch

        self.constant = float(constant)
        self.scale = torch.ones(n, 1)

    def __call__(self, spin):
        return self.constant * self.scale.to(spin.device, spin.dtype)


def sent_action(a_fin, ref, parts: dict, theta, role_of, elig, t: int, neutral, cfg: dict, cap: float):
    """The action sent this step: the finalist's, except in eligible environments of a probe slot inside the window
    (the stroke boost acts inside the reference and is not applied here)."""
    import torch

    t0, n = int(cfg["t0"]), int(cfg["window"])
    if t < t0 or t >= t0 + n:
        return a_fin
    b = beta(t, cfg)
    rec = (elig & (role_of == ROLES.index("recapture"))).unsqueeze(1)
    x3 = (elig & (role_of == ROLES.index("residual_x3"))).unsqueeze(1)
    blended = ((1.0 - b) * a_fin + b * neutral).clamp(-1.0, 1.0)
    scaled = (ref + sr.residual(float(cfg["residual_scale"]) * theta, parts["s"], parts["vhat"], parts["ehat"],
                                float(cfg["residual_scale"]) * cap)).clamp(-1.0, 1.0)
    return torch.where(rec, blended, torch.where(x3, scaled, a_fin))


def _batch(env, ctx, ctrl, gain, vecs, role_of, cfg: dict) -> dict:
    import torch

    from lares.envs.isaac_pen_spin.runner import _write_placements
    from lares.envs.isaac_pen_spin.success_ars import first_episode_scores

    spec = cases.STAGES[3]
    n, dev = env.num_envs, env.device
    t0, win, look = int(cfg["t0"]), int(cfg["window"]), int(cfg["stuck_window"])
    env.reset()
    _write_placements(env, vecs)
    obs = env._get_observations()["policy"]
    ctrl.reset(ctx)
    gain.scale[:] = 1.0
    T = int(env.max_episode_length)
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    completed = torch.zeros(n, dtype=torch.bool, device=dev)
    length = torch.zeros(n, dtype=torch.long, device=dev)
    pred = torch.zeros((T, n), dtype=torch.bool, device=dev)
    spin_sum = torch.zeros(n, device=dev)
    tail_sum, tail_n = torch.zeros(n, device=dev), torch.zeros(n, device=dev)
    elig = torch.zeros(n, dtype=torch.bool, device=dev)
    centre = torch.zeros(n, device=dev)
    starts = n
    boost = role_of == ROLES.index("stroke_boost")
    for t in range(T):
        if t == t0:
            elig = eligible(alive, centre, spin_sum / look, cfg)
        inside = t0 <= t < t0 + win
        gain.scale[:] = (1.0 + (float(cfg["stroke_scale"]) - 1.0) * (elig & boost).float().unsqueeze(1).cpu()) if inside else 1.0
        p = ctrl.parts(obs, t, ctx)
        a_fin = ctrl.mean(p, ctrl.theta).clamp(-1.0, 1.0)
        action = sent_action(a_fin, p["ref"], p, ctrl.theta, role_of, elig, t, ctx.neutral_action.to(a_fin.dtype), cfg, ctrl.cap)
        obs_d, _, terminated, truncated, _ = env.step(action)
        obs = obs_d["policy"]
        done = terminated | truncated
        ctrl.reset_envs(done)
        starts += int(done.sum())
        rate = env.spin_rate.abs()
        if t0 - look <= t < t0:
            spin_sum += torch.where(alive, rate, torch.zeros_like(rate))
        if t == t0 - 1:
            centre = env.center_dist.clone()
        if t >= T - 61:
            tail_sum += torch.where(alive, rate, torch.zeros_like(rate))
            tail_n += alive.float()
        pred[t] = alive & (env.center_dist < spec.success_center_tolerance) & (rate > spec.success_spin_rate)
        length = torch.where(alive, length + 1, length)
        completed = torch.where(alive & done, truncated & ~terminated, completed)
        alive = alive & ~done
        if not bool(alive.any()):
            break
    sc = first_episode_scores(pred.cpu().numpy(), length.cpu().numpy(), completed.cpu().numpy())
    return {"eligible": elig.cpu().numpy().astype(int).tolist(), "sustained": sc["sustained"].astype(int).tolist(),
            "completed": completed.cpu().numpy().astype(int).tolist(), "length": length.cpu().numpy().tolist(),
            "tail_abs_spin": (tail_sum / tail_n.clamp(min=1.0)).cpu().numpy().tolist(), "starts": starts}


def child_main(request_path: str) -> None:
    from lares.envs.isaac_lab_adapter import _ensure_isaaclab_on_path, close_isaac_app, launch_isaac_app
    from lares.utils.cli import load_yaml

    with open(request_path, encoding="utf-8") as f:
        req = json.load(f)
    full = load_yaml(req["config_path"])
    cfg = full["research_protocols"][req["study"]]
    device = str(full["device"])
    _ensure_isaaclab_on_path(str(full["isaaclab_root"]))
    launch_isaac_app(headless=True, device=device)
    code = 0
    try:
        import torch

        from lares.envs.isaac_pen_spin.runner import ControllerContext
        from lares.envs.isaac_pen_spin.success_ars import tile
        from lares.envs.isaac_pen_spin.symbolic_reward_child import make_training_env

        if sr.config_sha256(cfg) != req["config_sha256"]:
            raise SystemExit("the config block changed since the driver froze it")
        per, slots = int(cfg["cases_per_slot"]), len(ROLES)
        env = make_training_env(per * slots, device, int(cfg["placement_seeds"][0]))
        try:
            ctx = ControllerContext(env)
            if sr.executed_joint_order(ctx) != list(sr.JOINTS):
                raise SystemExit("the simulator's joint order is not the actor's")
            ctrl = sr.build_controller(cfg, "finalist", finalist_theta(cfg), normalization(cfg))
            policy = ctrl.ref.inner.policy
            gain = PerEnvGain(policy._stroke_gain.constant, per * slots)
            policy._stroke_gain = gain
            out = {"batches": [], "starts": 0, "halted": None, "stroke_gain_constant": gain.constant}
            for b, seed in enumerate(cfg["placement_seeds"]):
                if out["starts"] + 2 * per * slots + 512 > int(cfg["start_ceiling"]):
                    out["halted"] = f"start ceiling before batch {b}"
                    break
                roles = slot_roles(b)
                role_of = torch.as_tensor(np.repeat([ROLES.index(r) for r in roles], per), device=device)
                res = _batch(env, ctx, ctrl, gain, tile(cases.placements(int(seed), per), slots), role_of, cfg)
                out["starts"] += res["starts"]
                out["batches"].append(res)
                if b == 0:                                    # instrument checks: no verdict, no retuning, no replacement
                    first = by_role(res, 0, per)
                    succ = float(np.mean(first["control"]["sustained"]))
                    pairs = int((first["control"]["eligible"].astype(bool) & first["recapture"]["eligible"].astype(bool)).sum())
                    lo, hi = cfg["first_batch_control_success"]
                    if not (float(lo) <= succ <= float(hi)) or pairs < int(cfg["min_pairs_first_batch"]):
                        out["halted"] = f"instrument check failed after the first batch: control success {succ:.3f}, recapture pairs {pairs}"
                        break
                print("batch", b, "starts", out["starts"], flush=True)
        finally:
            env.close()
        with open(req["response_path"], "w", encoding="utf-8") as f:
            json.dump(out, f)
        print("wrote", req["response_path"], flush=True)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        code = 1
    finally:
        if code:
            os._exit(code)
        close_isaac_app()


if __name__ == "__main__":
    child_main(sys.argv[1])
