"""R-106: the Isaac process of the exact-success ARS study (``python -m lares.envs.isaac_pen_spin.success_ars_child req``).

One fresh process for the smoke stage and one for the whole training run. A rollout writes the given placements into
every environment (as ``runner.PenSpinBatchRunner`` does), runs one per-environment theta, and scores each
environment's first episode by the unchanged sustained rule. Later rollouts of the training process are not
reproducible evaluations (D-034) and are never reported as such.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def rollout(env, ctx, ctrl, vecs, theta_env, *, plain=None, keep_steps: int = 0, reward_cfg=None) -> dict:
    """One batch: placements ``vecs`` (one per environment), residual coefficients ``theta_env`` (one row per
    environment). Returns per-environment first-episode outcomes, residual sums, and the simulator starts made."""
    import torch

    from lares.envs.isaac_pen_spin import cases
    from lares.envs.isaac_pen_spin.runner import _write_placements

    spec = cases.STAGES[3]
    n = env.num_envs
    env.reset()
    _write_placements(env, vecs)
    obs = env._get_observations()["policy"]
    ctrl.reset(ctx)
    if plain is not None:
        plain.reset(ctx)
    T = int(env.max_episode_length)
    dev = env.device
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    completed = torch.zeros(n, dtype=torch.bool, device=dev)
    length = torch.zeros(n, dtype=torch.long, device=dev)
    pred = torch.zeros((T, n), dtype=torch.bool, device=dev)
    prop = torch.zeros(n, device=dev, dtype=torch.float64)
    exe = torch.zeros_like(prop)
    starts, mism, compared = n, 0, 0
    dense = torch.zeros(n, device=dev, dtype=torch.float64)
    seq_obs, seq_act, first_action = [], [], None
    for t in range(T):
        p = ctrl.parts(obs, t, ctx)
        mu = ctrl.mean(p, theta_env)
        action = mu.clamp(-1.0, 1.0)
        if t == 0:
            first_action = action.clone()
        if plain is not None:                               # identity against R-102's class, first episodes only
            a_plain = plain(obs, t, ctx).to(action.device)
            mism += int((action[alive] != a_plain[alive]).any(dim=1).sum())
            compared += int(alive.sum())
        if t < keep_steps:
            seq_obs.append(obs.clone())
            seq_act.append(action.clone())
        prop += torch.where(alive, (mu - p["ref"]).abs().sum(1).double(), torch.zeros_like(prop))
        exe += torch.where(alive, (action - p["ref"]).abs().sum(1).double(), torch.zeros_like(exe))
        obs_d, _, terminated, truncated, _ = env.step(action)
        obs = obs_d["policy"]
        done = terminated | truncated
        ctrl.reset_envs(done)
        starts += int(done.sum())
        if reward_cfg is not None:                           # R-103's step reward, first episodes only
            from lares.envs.isaac_pen_spin.symbolic_reward import step_reward

            r = step_reward(env.center_dist, env.spin_rate, terminated, action, p["ref"], reward_cfg, ctrl.cap)
            dense += torch.where(alive, r.double(), torch.zeros_like(dense))
        step_ok = (env.center_dist < spec.success_center_tolerance) & (env.spin_rate.abs() > spec.success_spin_rate)
        pred[t] = alive & step_ok
        length = torch.where(alive, length + 1, length)
        completed = torch.where(alive & done, truncated & ~terminated, completed)
        alive = alive & ~done
        if not bool(alive.any()):
            break
    from lares.envs.isaac_pen_spin.success_ars import first_episode_scores

    sc = first_episode_scores(pred.cpu().numpy(), length.cpu().numpy(), completed.cpu().numpy())
    return {"sustained": sc["sustained"], "completed": completed.cpu().numpy(), "length": length.cpu().numpy(),
            "tail_hit": sc["tail_hit"], "proposed": prop.cpu().numpy(), "dense": dense.cpu().numpy(),
            "executed": exe.cpu().numpy(), "starts": starts, "transitions": int(length.sum()),
            "env_steps": int((t + 1) * n), "identity": {"compared": compared, "mismatches": mism},
            "seq_obs": seq_obs, "seq_act": seq_act, "first_action": first_action,
            "intact": (length >= keep_steps) if keep_steps else None}


def _setup(protocol_cfg: dict, device: str, seed: int, name: str):
    import torch

    from lares.envs.isaac_pen_spin import success_ars as sa
    from lares.envs.isaac_pen_spin import symbolic_reward as sr
    from lares.envs.isaac_pen_spin.runner import ControllerContext
    from lares.envs.isaac_pen_spin.symbolic_reward_child import make_training_env

    lay = sa.layout(protocol_cfg)
    env = make_training_env(lay["num_envs"], device, seed)
    ctx = ControllerContext(env)
    ctrl = sr.build_controller(protocol_cfg, name, None, sa.normalization(protocol_cfg))
    to_t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=device)   # noqa: E731
    return env, ctx, ctrl, lay, to_t


def smoke(req: dict, protocol_cfg: dict, device: str) -> dict:
    import numpy as np
    import torch

    from lares.envs.isaac_pen_spin import success_ars as sa
    from lares.envs.isaac_pen_spin import symbolic_reward as sr

    sm = protocol_cfg["smoke"]
    env, ctx, ctrl, lay, to_t = _setup(protocol_cfg, device, int(sm["seed"]), "smoke")
    out = {"mode": "smoke", "seed": int(sm["seed"]), "gates": {}}
    try:
        names = sr.executed_joint_order(ctx)
        out["gates"]["joint_order"] = {"passes": names == list(sr.JOINTS)}
        vecs = sa.tile(sa.smoke_placements(protocol_cfg), lay["slots"])
        zero = torch.zeros((lay["num_envs"], sr.N_COEF), device=device)
        t0 = time.time()
        rw = protocol_cfg.get("reward")
        a = rollout(env, ctx, ctrl, vecs, zero, plain=sr.build_reference(protocol_cfg, "smoke_plain", resettable=False),
                    reward_cfg=rw)
        if rw is not None:                                  # R-108: the dense return must be finite; its A/A spread is reported
            dm = a["dense"].reshape(lay["slots"], lay["per"]).mean(axis=1)
            out["gates"]["dense_return"] = {"slot_means": [float(x) for x in dm], "max_abs_slot_difference": float(dm.max() - dm.min()),
                                            "passes": bool(np.isfinite(a["dense"]).all() and dm.min() > 0)}
        out["gates"]["zero_theta_identity"] = {**a["identity"], "passes": a["identity"]["mismatches"] == 0 and a["identity"]["compared"] > 0}
        out["aa"] = sa.aa_statistics(a["sustained"].reshape(lay["slots"], lay["per"]))
        out["aa"]["scorer_live"] = bool(a["sustained"].any())
        out["gates"]["scorer_live"] = {"reference_successes": int(a["sustained"].sum()), "passes": bool(a["sustained"].any())}
        out["aa"]["slot_drops"] = [int(x) for x in (~a["completed"]).reshape(lay["slots"], lay["per"]).sum(axis=1)]

        signs = np.random.default_rng(int(sm["seed"])).choice([-1.0, 1.0], size=sr.N_COEF)
        probe = float(sm["probe_magnitude"]) * signs
        ck = {"theta": sr.theta_record(probe), "theta_sha256": sr.theta_sha256(probe)}
        ck_path = os.path.join(os.path.dirname(req["response_path"]), "smoke_checkpoint.json")
        with open(ck_path, "w", encoding="utf-8") as f:
            json.dump(ck, f)
        cand = np.stack([np.zeros(sr.N_COEF), probe, -probe, probe])
        K = int(sm["reconstruction_steps"])
        b = rollout(env, ctx, ctrl, vecs, to_t(sa.slot_thetas(cand, lay["per"])), keep_steps=K)
        wall = time.time() - t0
        per = lay["per"]
        slot = lambda x, k: x[k * per:(k + 1) * per]   # noqa: E731
        fa = b["first_action"]
        probe_envs = np.r_[per:4 * per]
        ratio = float(b["executed"][probe_envs].sum() / max(b["proposed"][probe_envs].sum(), 1e-12))
        out["gates"]["executed_authority"] = {"pooled_executed_over_proposed": ratio, "zero_slot_proposed": float(slot(b["proposed"], 0).sum()),
                                              "passes": ratio >= float(sm["pooled_ratio_min"]) and float(slot(b["proposed"], 0).sum()) == 0.0}
        dup = float((slot(fa, 1) - slot(fa, 3)).abs().max())
        out["gates"]["per_environment_theta"] = {
            "plus_slots_max_abs_difference_at_step_0": dup,
            "plus_slots_agree_at_step_0": bool(dup <= float(sm["duplicate_slot_tolerance"])),
            "plus_differs_from_minus": bool(not torch.equal(slot(fa, 1), slot(fa, 2))),
            "plus_differs_from_zero": bool(not torch.equal(slot(fa, 1), slot(fa, 0)))}
        g = out["gates"]["per_environment_theta"]
        g["passes"] = all(v for v in g.values() if isinstance(v, bool))
        out["probe_slots"] = {"slot_success": [float(x) for x in b["sustained"].reshape(lay["slots"], per).mean(axis=1)],
                              "note": "slots: zero, +probe, -probe, +probe; a diagnostic, not a selection"}
        # reconstruction: the saved probe through the evaluation factory, replayed on slot 1's stored observations
        with open(ck_path, encoding="utf-8") as f:
            c3 = sr.build_controller(protocol_cfg, "smoke_reconstruction", json.load(f)["theta"], sa.normalization(protocol_cfg))
        c3.reset(ctx)
        intact = b["intact"].to(device)
        mask = torch.zeros(lay["num_envs"], dtype=torch.bool, device=device)
        mask[per:2 * per] = True
        mask &= intact
        bad = 0
        for t in range(K):
            bad += int((c3(b["seq_obs"][t], t, ctx)[mask] != b["seq_act"][t][mask]).any(dim=1).sum())
        out["gates"]["reconstruction"] = {"steps": K, "environments": int(mask.sum()), "mismatches": bad,
                                          "passes": bad == 0 and int(mask.sum()) > 0}
        out["starts"] = a["starts"] + b["starts"]
        out["transitions"] = a["env_steps"] + b["env_steps"]
        out["throughput"] = {"transitions_per_s": out["transitions"] / wall, "wall_time_s": wall}
        out["passes"] = all(g["passes"] for g in out["gates"].values())
    finally:
        env.close()
    return out


def train(req: dict, protocol_cfg: dict, device: str) -> dict:
    import numpy as np

    from lares.envs.isaac_pen_spin import success_ars as sa
    from lares.envs.isaac_pen_spin import symbolic_reward as sr

    tr = protocol_cfg["training"]
    env, ctx, ctrl, lay, to_t = _setup(protocol_cfg, device, int(tr["seed"]), "train")
    ars = sr.ARS(int(tr["seed"]), protocol_cfg["ars"], float(protocol_cfg["actor"]["bound"]))
    cap = int(req["start_cap"])
    objective, rw = req.get("objective", "exact"), protocol_cfg.get("reward")
    if objective not in ("exact", "dense") or (objective == "dense" and rw is None):
        raise SystemExit(f"objective {objective!r} is not available for this protocol")
    starts = transitions = scored = 0
    updates, history, complete, note = [], [sr.theta_record(ars.theta)], True, None
    t0 = time.time()
    try:
        for u in range(lay["updates"]):
            cand = ars.ask(u)
            vecs = sa.tile(sa.training_placements(protocol_cfg, u), lay["slots"])
            slot_scores, drops, prop, success, dense_r = [], [], [], [], []
            for r in range(lay["rollouts"]):
                if starts + 2 * lay["num_envs"] > cap:
                    complete, note = False, f"update {u + 1} rollout {r + 1}: {starts} starts, cap {cap}"
                    break
                rc = sa.rollout_candidates(cand, r, lay)
                res = rollout(env, ctx, ctrl, vecs, to_t(sa.slot_thetas(rc, lay["per"])), reward_cfg=rw)
                starts += res["starts"]
                transitions += res["env_steps"]
                scored += lay["num_envs"]
                success.append(res["sustained"].reshape(lay["slots"], lay["per"]).mean(axis=1))
                dense_r.append(res["dense"].reshape(lay["slots"], lay["per"]).mean(axis=1))
                slot_scores.append(success[-1] if objective == "exact" else dense_r[-1])
                drops.append((~res["completed"]).reshape(lay["slots"], lay["per"]).mean(axis=1))
                prop.append(float(res["proposed"].sum() / max(1, res["transitions"]) / sr.N_JOINTS))
            if not complete:
                break
            scores = sa.ars_scores(slot_scores, lay)
            info = ars.tell(u, scores)
            n = lay["directions"]
            history.append(sr.theta_record(ars.theta))
            updates.append({"update": u + 1, "plus": [float(x) for x in scores[:n]], "minus": [float(x) for x in scores[n:]],
                            "mean_candidate_success": float(np.mean(success)), "sigma_r": info["sigma_r"],
                            "slot_success": [[float(x) for x in v] for v in success],
                            "slot_dense_return": [[float(x) for x in v] for v in dense_r],
                            "slot_drop_rates": [[float(x) for x in d] for d in drops], "mean_abs_proposed": prop,
                            "theta_abs_max": float(np.abs(ars.theta).max()), "theta_l2": float(np.linalg.norm(ars.theta)),
                            "starts_so_far": starts, "wall_time_s": time.time() - t0})
            print("update", json.dumps(updates[-1]), flush=True)
    finally:
        env.close()
    return {"mode": "train", "objective": objective, "seed": int(tr["seed"]), "complete": complete, "note": note, "updates": updates,
            "starts": starts, "transitions": transitions, "scored_episodes": scored, "wall_time_s": time.time() - t0,
            "final_theta": sr.theta_record(ars.theta) if complete else None,
            "final_theta_sha256": sr.theta_sha256(ars.theta) if complete else None, "theta_history": history,
            "exploration": "antithetic parameter perturbations, deterministic actions, exact sustained-success returns"}


def main(request_path: str) -> None:
    from lares.envs.isaac_lab_adapter import _ensure_isaaclab_on_path, close_isaac_app, launch_isaac_app
    from lares.utils.cli import load_yaml

    with open(request_path, encoding="utf-8") as f:
        req = json.load(f)
    cfg = load_yaml(req["config_path"])
    protocol_cfg = cfg["research_protocols"][req["study"]]
    device = str(cfg["device"])
    _ensure_isaaclab_on_path(str(cfg["isaaclab_root"]))
    launch_isaac_app(headless=True, device=device)
    code = 0
    try:
        from lares.envs.isaac_pen_spin import symbolic_reward as sr

        if sr.config_sha256(protocol_cfg) != req["config_sha256"]:
            raise SystemExit("the config block changed since the driver froze it")
        protocol_cfg = {**protocol_cfg, **req.get("cfg_overrides", {})}   # R-128: one lineage's seeds and schedule
        out = smoke(req, protocol_cfg, device) if req["mode"] == "smoke" else train(req, protocol_cfg, device)
        with open(req["response_path"], "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print("wrote", req["response_path"], flush=True)
    except BaseException:  # Isaac's close swallows tracebacks: print before it runs
        traceback.print_exc()
        sys.stdout.flush()
        code = 1
    finally:
        if code:
            os._exit(code)
        close_isaac_app()


if __name__ == "__main__":
    main(sys.argv[1])
