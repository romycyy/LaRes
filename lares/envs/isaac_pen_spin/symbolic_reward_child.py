"""R-103: the Isaac process of the symbolic-reward study (``python -m lares.envs.isaac_pen_spin.symbolic_reward_child req``).

One fresh process per request: the smoke gate, or one ``(arm, block)`` training run. Training uses the environment's
own seeded reset distribution in a persistent process with auto-resets; it is not a reproducible evaluation and is
never scored as one (evaluation cells use ``protocols.run_cell``). Run under Isaac Lab's environment only.
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


class BudgetExhausted(RuntimeError):
    """The run's episode allowance is spent: it stops incomplete, and the protocol stops with it."""


def make_training_env(num_envs: int, device: str, seed: int):
    """The Stage-3 env of ``runner.make_env`` with Isaac Lab's own seeding (reset draws, torch, numpy)."""
    import importlib

    from lares.envs.isaac_pen_spin import cases
    from lares.envs.isaac_pen_spin.runner import MeasuredShadowHandSpinEnv

    cfg = getattr(importlib.import_module("lares.envs.isaac_pen_spin.shadow_hand_spin_env_cfg"), cases.STAGES[3].cfg_class)()
    cfg.scene.num_envs = int(num_envs)
    cfg.sim.device = device
    cfg.seed = int(seed)
    return MeasuredShadowHandSpinEnv(cfg=cfg)


class Loop:
    """Steps the env for a controller, counting transitions, episodes started and falls, under an episode cap."""

    def __init__(self, env, ctrl, episode_cap: int | None):
        from lares.envs.isaac_pen_spin.runner import ControllerContext

        self.env, self.ctrl, self.cap = env, ctrl, episode_cap
        self.ctx = ControllerContext(env)
        self.transitions = self.episodes = self.falls = 0

    def _charge(self, n: int) -> None:
        self.episodes += int(n)
        if self.cap is not None and self.episodes > self.cap:
            raise BudgetExhausted(f"{self.episodes} episodes started exceed the allowance of {self.cap}")

    def reset_all(self, ctrl=None):
        self._charge(self.env.num_envs)
        obs = self.env.reset()[0]["policy"]
        (ctrl or self.ctrl).reset(self.ctx)
        return obs

    def step(self, action, ctrl=None):
        obs_d, _, terminated, truncated, _ = self.env.step(action)
        done = terminated | truncated
        (ctrl or self.ctrl).reset_envs(done)
        self.transitions += self.env.num_envs
        self.falls += int(terminated.sum())
        self._charge(int(done.sum()))
        return obs_d["policy"], terminated, truncated


def _predicate(env):
    from lares.envs.isaac_pen_spin import cases

    spec = cases.STAGES[3]
    return (env.center_dist < spec.success_center_tolerance) & (env.spin_rate.abs() > spec.success_spin_rate)


# ---------------------------------------------------------------------------------------------------------------------
#  Smoke gate
# ---------------------------------------------------------------------------------------------------------------------


def smoke(req: dict, protocol_cfg: dict, device: str) -> dict:
    import numpy as np
    import torch

    from lares.envs.isaac_pen_spin import cases
    from lares.envs.isaac_pen_spin import symbolic_reward as sr
    from lares.envs.isaac_pen_spin.incumbent_residual import ACTUATED

    sm, actor, tr = protocol_cfg["smoke"], protocol_cfg["actor"], protocol_cfg["training"]
    n, H = int(tr["num_envs"]), int(tr["horizon"])
    env = make_training_env(n, device, int(sm["seed"]))
    out = {"mode": "smoke", "seed": int(sm["seed"]), "gates": {}}
    try:
        # ---- phase 1: the reference only (identity, filter reset, normalisation) ------------------------------------
        ctrl = sr.build_controller(protocol_cfg, "smoke_zero", None, sr.unit_normalization())
        plain = sr.build_reference(protocol_cfg, "smoke_plain", resettable=False)
        loop = Loop(env, ctrl, None)
        names = sr.executed_joint_order(loop.ctx)
        out["gates"]["joint_order"] = {"simulator": names, "passes": names == list(sr.JOINTS)}
        obs = loop.reset_all()
        plain.reset(loop.ctx)
        sl_q = cases.obs_slices()["hand_dof_pos"]
        never_reset = torch.ones(n, dtype=torch.bool, device=obs.device)
        mism = compared = filt_bad = reset_events = 0
        sum_v = torch.zeros(sr.N_JOINTS, device=obs.device, dtype=torch.float64)
        sum_e = torch.zeros_like(sum_v)
        count = 0
        alpha = ctrl.ref.alpha

        def filter_check(o):
            nonlocal filt_bad, reset_events
            v_prev = None if ctrl.ref.v_f is None else ctrl.ref.v_f.clone()
            pending = None if ctrl.ref._pending is None else ctrl.ref._pending.clone()
            p = ctrl.parts(o, 0, loop.ctx)
            v = ctrl.ref.velocity(o)
            if v_prev is None:
                expected = v
            else:
                base = v_prev if pending is None else torch.where(pending.unsqueeze(1), v, v_prev)
                expected = base + alpha * (v - base)
            filt_bad += int((ctrl.ref.v_f != expected).any(dim=1).sum())
            if pending is not None:
                reset_events += int(pending.sum())
                filt_bad += int((ctrl.ref.v_f[pending] != v[pending]).any(dim=1).sum())
            return p

        t0 = time.time()
        for t in range(H):
            p = filter_check(obs)
            action = ctrl.mean(p, ctrl.theta).clamp(-1.0, 1.0)
            a_plain = plain(obs, t, loop.ctx).to(action.device)
            mism += int((action[never_reset] != a_plain[never_reset]).any(dim=1).sum())
            compared += int(never_reset.sum())
            sum_v += (ctrl.ref.v_f.double() ** 2).sum(0)
            sum_e += ((p["ref"] - torch.nan_to_num(obs)[:, sl_q][:, list(ACTUATED)].to(p["ref"].dtype)).double() ** 2).sum(0)
            count += n
            obs, terminated, truncated = loop.step(action)
            never_reset &= ~(terminated | truncated)
        filter_check(obs)                                   # the time-out resets of the last step, with no env step
        out["gates"]["zero_theta_identity"] = {"compared_env_steps": compared, "mismatches": mism, "passes": mism == 0 and compared > 0}
        out["gates"]["filter_reset"] = {"reset_events_checked": reset_events, "violations": filt_bad,
                                        "passes": filt_bad == 0 and reset_events > 0}
        rms_v = torch.sqrt(sum_v / count).clamp(min=float(actor["rms_floor_velocity"]))
        rms_e = torch.sqrt(sum_e / count).clamp(min=float(actor["rms_floor_posture"]))
        norm = {"rms_velocity": [float(x) for x in rms_v.float()], "rms_posture": [float(x) for x in rms_e.float()],
                "source": f"{count} in-play reference environment-steps, training seed {int(sm['seed'])}",
                "floors": [float(actor["rms_floor_velocity"]), float(actor["rms_floor_posture"])]}
        out["normalization"] = norm
        out["reference_phase"] = {"falls": loop.falls, "episodes_started": loop.episodes}

        # ---- phase 2: a probe theta through the checkpoint path (authority, likelihood, reconstruction) -------------
        sdir = os.path.dirname(req["response_path"])
        signs = np.random.default_rng(int(sm["seed"])).choice([-1.0, 1.0], size=sr.N_COEF)
        probe = {"theta": sr.theta_record(float(sm["probe_magnitude"]) * signs)}
        probe["theta_sha256"] = sr.theta_sha256(probe["theta"])
        ck_path = os.path.join(sdir, "smoke_checkpoint.json")
        with open(ck_path, "w", encoding="utf-8") as f:
            json.dump(probe, f)

        def from_checkpoint(name):
            with open(ck_path, encoding="utf-8") as f:
                ck = json.load(f)
            return sr.build_controller(protocol_cfg, name, ck["theta"], norm)

        c2 = from_checkpoint("smoke_probe")
        obs = loop.reset_all(c2)
        K = int(sm["reconstruction_steps"])
        seq_obs, seq_act = [], []
        intact = torch.ones(n, dtype=torch.bool, device=obs.device)
        prop = torch.zeros(sr.N_JOINTS, device=obs.device, dtype=torch.float64)
        exe, sat = torch.zeros_like(prop), torch.zeros_like(prop)
        differs = 0
        like = {}
        gen = torch.Generator(device=obs.device)
        gen.manual_seed(int(sm["seed"]))
        sigma = float(protocol_cfg["ppo"]["sigma"])
        for t in range(H):
            p = c2.parts(obs, t, loop.ctx)
            mu = c2.mean(p, c2.theta)
            action = mu.clamp(-1.0, 1.0)
            prop += (mu - p["ref"]).abs().double().sum(0)
            exe += (action - p["ref"]).abs().double().sum(0)
            sat += (action.abs() >= 1.0).double().sum(0)
            differs += int((action != p["ref"]).any(dim=1).sum())
            if t < K:
                seq_obs.append(obs.clone())
                seq_act.append(action.clone())
            if t == int(sm["likelihood_step"]):
                psi = (c2.theta / c2.bound).clone().requires_grad_(True)
                with torch.no_grad():
                    mu_old = p["ref"] + sr.residual(psi * c2.bound, p["s"], p["vhat"], p["ehat"], c2.cap)
                    u = mu_old + sigma * torch.randn(mu_old.shape, generator=gen, device=mu_old.device, dtype=mu_old.dtype)
                    logp = sr.gaussian_logp(u, mu_old, sigma)
                    adv = torch.randn(n, generator=gen, device=mu_old.device, dtype=mu_old.dtype)
                batch = {**p, "u": u, "logp": logp, "mu": mu_old}
                obj, kl, ratio = sr.surrogate(psi, batch, adv, protocol_cfg["ppo"], c2.bound, c2.cap)
                obj.backward()
                g = psi.grad
                like = {"finite_logp": bool(torch.isfinite(logp).all()),
                        "ratio_is_one": bool((ratio == 1.0).all()), "kl_at_unchanged_theta": float(kl),
                        "finite_gradient": bool(torch.isfinite(g).all()),
                        "nonzero_gradient_coefficients": int((g != 0).sum()),
                        "zero_gradient_coefficients": [sr.COEF_NAMES[i] for i in torch.nonzero(g == 0).flatten().tolist()]}
                like["passes"] = bool(like["finite_logp"] and like["ratio_is_one"] and like["finite_gradient"]
                                      and like["nonzero_gradient_coefficients"] == sr.N_COEF and float(kl) == 0.0)
            obs, terminated, truncated = loop.step(action, c2)
            if t < K:
                intact &= ~(terminated | truncated)
        wall = time.time() - t0
        total = n * H
        ratio_j = (exe / prop.clamp(min=1e-12)).cpu().numpy()
        auth = {"probe_magnitude": float(sm["probe_magnitude"]), "differs_fraction": differs / total,
                "pooled_executed_over_proposed": float(exe.sum() / prop.sum().clamp(min=1e-12)),
                "joints_at_or_above_min_ratio": int((ratio_j >= float(sm["joint_ratio_min"])).sum()),
                "per_joint": {j: {"proposed": float(prop[i]) / total, "executed": float(exe[i]) / total,
                                  "ratio": float(ratio_j[i]), "saturated_fraction": float(sat[i]) / total}
                              for i, j in enumerate(sr.JOINTS)}}
        auth["passes"] = bool(auth["differs_fraction"] >= float(sm["differs_min"])
                              and auth["pooled_executed_over_proposed"] >= float(sm["pooled_ratio_min"])
                              and auth["joints_at_or_above_min_ratio"] >= int(sm["joints_min"]))
        out["gates"]["executed_authority"] = auth
        out["gates"]["likelihood"] = like or {"passes": False}

        c3 = from_checkpoint("smoke_reconstruction")
        c3.reset(loop.ctx)
        bad = 0
        for t in range(K):
            bad += int((c3(seq_obs[t], t, loop.ctx)[intact] != seq_act[t][intact]).any(dim=1).sum())
        out["gates"]["reconstruction"] = {"steps": K, "environments": int(intact.sum()), "mismatches": bad,
                                          "theta_sha256": c3.params["theta_sha256"],
                                          "passes": bad == 0 and int(intact.sum()) > 0
                                          and c3.params["theta_sha256"] == probe["theta_sha256"]}
        out["transitions"] = loop.transitions
        out["episodes_started"] = loop.episodes
        out["falls"] = loop.falls
        out["throughput"] = {"transitions_per_s": loop.transitions / wall, "wall_time_s": wall,
                             "note": "both phases, including the second reference instance of the identity check"}
        out["passes"] = all(g["passes"] for g in out["gates"].values())
    finally:
        env.close()
    return out


# ---------------------------------------------------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------------------------------------------------


def _checkpoint(k: int, transitions: int, theta) -> dict:
    from lares.envs.isaac_pen_spin import symbolic_reward as sr

    return {"k": k, "transitions": transitions, "theta": sr.theta_record(theta), "theta_sha256": sr.theta_sha256(theta)}


def train(req: dict, protocol_cfg: dict, device: str) -> dict:
    import numpy as np
    import torch

    from lares.envs.isaac_pen_spin import cases
    from lares.envs.isaac_pen_spin import symbolic_reward as sr

    arm, seed, block = req["arm"], int(req["seed"]), int(req["block"])
    actor, tr, rw = protocol_cfg["actor"], protocol_cfg["training"], protocol_cfg["reward"]
    n, H = int(tr["num_envs"]), int(tr["horizon"])
    bound, cap = float(actor["bound"]), float(actor["cap"])
    sched = sr.schedule(protocol_cfg)
    norm = sr.load_normalization(req["out_dir"])
    env = make_training_env(n, device, seed)
    ctrl = sr.build_controller(protocol_cfg, f"train_b{block}_{arm}", None, norm)
    loop = Loop(env, ctrl, req.get("episode_cap"))
    checkpoints, curve, updates, complete, note = [], [], 0, True, None
    t0 = time.time()

    def row(tag, rewards, preds, spins, extra=None):
        return {"at": tag, "transitions": loop.transitions, "mean_step_reward": float(np.mean(rewards)),
                "predicate_fraction": float(np.mean(preds)), "mean_abs_spin": float(np.mean(spins)),
                "falls_so_far": loop.falls, **(extra or {})}

    try:
        if arm in ("draws", "ars"):
            search = sr.DrawSearch(seed, protocol_cfg["draws"], bound) if arm == "draws" else sr.ARS(seed, protocol_cfg["ars"], bound)
            gens = int(protocol_cfg["draws"]["generations"] if arm == "draws" else protocol_cfg["ars"]["iterations"])
            every = sched["draws_every"] if arm == "draws" else sched["ars_every"]
            for g in range(gens):
                cand = search.ask(g)
                per = n // len(cand)
                theta_env = torch.as_tensor(np.repeat(cand, per, axis=0), dtype=torch.float32, device=device)
                obs = loop.reset_all()
                ret = torch.zeros(n, device=device)
                rs, ps, ss = [], [], []
                for t in range(H):
                    p = ctrl.parts(obs, t, loop.ctx)
                    action = ctrl.mean(p, theta_env.to(p["ref"].dtype)).clamp(-1.0, 1.0)
                    obs, terminated, truncated = loop.step(action)
                    r = sr.step_reward(env.center_dist, env.spin_rate, terminated, action, p["ref"], rw, cap)
                    ret += r
                    rs.append(float(r.mean()))
                    ps.append(float(_predicate(env).float().mean()))
                    ss.append(float(env.spin_rate.abs().mean()))
                if not bool(torch.isfinite(ret).all()):
                    raise FloatingPointError("non-finite training return")
                scores = ret.view(len(cand), per).mean(dim=1).cpu().numpy()
                info = search.tell(g, scores)
                updates += 1 if arm == "ars" else 0
                curve.append(row(f"generation {g + 1}", rs, ps, ss, {
                    "candidate_scores": [float(x) for x in scores], "candidate_abs_max": [float(np.abs(c).max()) for c in cand],
                    "nominee_abs_max": float(np.abs(search.nominee()).max()), **(info or {}),
                    **({"best": dict(search.best)} if arm == "draws" else {})}))
                if (g + 1) % every == 0:
                    checkpoints.append(_checkpoint((g + 1) // every, loop.transitions, search.nominee()))
        elif arm == "ppo":
            pc = protocol_cfg["ppo"]
            sigma, T = float(pc["sigma"]), int(pc["steps"])
            gen = torch.Generator(device=device)
            gen.manual_seed(seed + int(pc["rng_offset"]))
            psi = torch.zeros(sr.N_COEF, device=device, requires_grad=True)
            actor_opt = torch.optim.Adam([psi], lr=float(pc["lr"]))
            critic = sr.Critic(cases.OBS_DIM, pc["critic_hidden"], float(pc["value_scale"])).to(device)
            critic_opt = torch.optim.Adam(critic.parameters(), lr=float(pc["critic_lr"]))
            lr = float(pc["lr"])
            obs = loop.reset_all()
            for upd in range(1, int(pc["updates"]) + 1):
                buf = {k: [] for k in ("obs", "ref", "s", "vhat", "ehat", "u", "logp", "mu", "val", "rew", "term", "trunc")}
                ps, ss = [], []
                with torch.no_grad():
                    theta = psi * bound
                    for t in range(T):
                        p = ctrl.parts(obs, t, loop.ctx)
                        mu = ctrl.mean(p, theta)
                        u = mu + sigma * torch.randn(mu.shape, generator=gen, device=mu.device, dtype=mu.dtype)
                        critic.observe(obs)
                        val = critic(obs)
                        executed = u.clamp(-1.0, 1.0)
                        o_now = obs
                        obs, terminated, truncated = loop.step(executed)
                        r = sr.step_reward(env.center_dist, env.spin_rate, terminated, executed, p["ref"], rw, cap)
                        for k, v in (("obs", torch.nan_to_num(o_now)), ("ref", p["ref"]), ("s", p["s"]), ("vhat", p["vhat"]),
                                     ("ehat", p["ehat"]), ("u", u), ("logp", sr.gaussian_logp(u, mu, sigma)), ("mu", mu),
                                     ("val", val), ("rew", r), ("term", terminated), ("trunc", truncated)):
                            buf[k].append(v.clone())
                        ps.append(float(_predicate(env).float().mean()))
                        ss.append(float(env.spin_rate.abs().mean()))
                    st = {k: torch.stack(v) for k, v in buf.items()}
                    adv, ret = sr.gae(st["rew"], st["val"], critic(obs), st["term"], st["trunc"], float(pc["gamma"]), float(pc["lam"]))
                if not bool(torch.isfinite(st["rew"]).all() and torch.isfinite(adv).all()):
                    raise FloatingPointError("non-finite reward or advantage")
                flat = {k: st[k].reshape(T * n, *st[k].shape[2:]) for k in ("obs", "ref", "s", "vhat", "ehat", "u", "logp", "mu")}
                flat["adv"], flat["ret"] = adv.reshape(-1), ret.reshape(-1)
                train_actor = upd > int(pc["critic_warmup_updates"])
                info = sr.ppo_update(psi, actor_opt, critic, critic_opt, flat, pc, bound, cap, lr, train_actor, gen)
                lr = info["lr"]
                updates += int(train_actor)
                if not bool(torch.isfinite(psi).all()):
                    raise FloatingPointError("non-finite coefficients")
                curve.append(row(f"update {upd}", [float(st["rew"].mean())], ps, ss, {
                    "kl": info["kl"], "value_loss": info["value_loss"], "surrogate": info["surrogate"], "actor_lr": lr,
                    "theta_abs_max": float((psi.detach() * bound).abs().max()),
                    "explained_variance": float(1.0 - (flat["ret"] - st["val"].reshape(-1)).var() / flat["ret"].var().clamp(min=1e-12))}))
                if upd % sched["ppo_every"] == 0:
                    checkpoints.append(_checkpoint(upd // sched["ppo_every"], loop.transitions, (psi.detach() * bound).clamp(-bound, bound)))
        else:
            raise ValueError(f"unknown arm {arm!r}")
    except BudgetExhausted as e:
        complete, note = False, str(e)
    finally:
        env.close()

    final = checkpoints[-1] if checkpoints else None
    with open(os.path.join(req["run_dir"], "checkpoints.json"), "w", encoding="utf-8") as f:
        json.dump({"arm": arm, "block": block, "seed": seed, "checkpoints": checkpoints}, f, indent=1)
    return {"mode": "train", "arm": arm, "block": block, "seed": seed, "complete": complete, "note": note,
            "transitions": loop.transitions, "episodes_started": loop.episodes, "falls": loop.falls,
            "wall_time_s": time.time() - t0, "updates": updates, "num_envs": n,
            "exploration": {"draws": "independent bounded draws, no adaptation",
                            "ars": "antithetic parameter perturbations, deterministic actions",
                            "ppo": f"fixed Gaussian latent-action noise sigma {protocol_cfg['ppo']['sigma']}; the executed "
                                   "action is clamp(u, -1, 1); deployment is deterministic"}[arm],
            "final_theta_sha256": final["theta_sha256"] if final else None,
            "final_theta_abs_max": float(np.abs(final["theta"]).max()) if final else None,
            "curve_first_last": [curve[0], curve[-1]] if curve else [], "curve": curve}


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
