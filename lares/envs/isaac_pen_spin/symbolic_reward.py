"""R-103 (P1): learn a sparse symbolic residual on the D-072-qualified reference from reward. No LLM.

One deterministic actor family for three learners (independent parameter draws, ARS, PPO)::

    residual_j = clip(a_j s + b_g(j) vhat_j + c_g(j) ehat_j, -cap, cap)        (normalised action units)
    action     = clamp(reference_action + residual, -1, 1)

``s`` is the unsigned axial-rate deficit ``clip((w* - |axial rate|) / w*, 0, 1)``, ``vhat`` the reference's own filtered
joint velocity and ``ehat`` the posture error ``reference_action - q``, both scaled by three training-only RMS values
and clipped to ``[-1, 1]``. ``theta = [a (20), b (6), c (6)]`` is 32 bounded coefficients, all zero initially, and
``theta = 0`` reproduces the reference bit for bit. The reference is the D-058 incumbent plus R-102's frozen
``JointMotionResidual``; its per-joint velocity filter is the only controller state, and
:class:`ResettableReference` re-initialises it per environment at an auto-reset.

This module holds everything that needs no simulator: the actor, the reward, the three learners' arithmetic, the
reference descriptor, the selection rule, the inference and the driver. ``symbolic_reward_child`` is the Isaac
process (one per smoke stage or ``(arm, block)`` training run); evaluation cells go through the established
``protocols.run_cell`` path, one whole frozen manifest per fresh process.

The driver runs: reference check -> smoke gate (stops on failure) -> nine training runs -> checkpoint selection on
each block's selection manifest -> finalists frozen and hashed -> fresh evaluation -> six planned contrasts.
"""

from __future__ import annotations

import glob
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin.incumbent_residual import ACTUATED, PARAMS, JointMotionResidual, params_sha256
from lares.envs.isaac_pen_spin.layout import VEL_OBS_SCALE
from lares.envs.isaac_pen_spin.protocols import arm_summary, failure_split, freeze_json, load_cell, study_cell
from lares.utils.hashing import sha256_file, sha256_text

STUDY = "symbolic_reward_residual"
STAGE = 3
ARMS = ("draws", "ars", "ppo")
REFERENCE = "reference"

#: The 20 actuated joints in action order (asserted against the simulator in the smoke stage).
JOINTS = ("WRJ1", "WRJ0", "FFJ3", "MFJ3", "RFJ3", "LFJ4", "THJ4", "FFJ2", "MFJ2", "RFJ2", "LFJ3", "THJ3",
          "FFJ1", "MFJ1", "RFJ1", "LFJ2", "THJ2", "LFJ1", "THJ1", "THJ0")
GROUPS = ("WR", "FF", "MF", "RF", "LF", "TH")
GROUP_OF = tuple(GROUPS.index(j[:2]) for j in JOINTS)
N_JOINTS, N_GROUPS = len(JOINTS), len(GROUPS)
N_COEF = N_JOINTS + 2 * N_GROUPS
COEF_NAMES = tuple([f"a_{j}" for j in JOINTS] + [f"b_{g}" for g in GROUPS] + [f"c_{g}" for g in GROUPS])

#: The six planned contrasts, one Holm family; the first is the registered prediction.
CONTRASTS = (("ppo", "draws"), ("ppo", "ars"), ("ars", "draws"), ("ppo", REFERENCE), ("ars", REFERENCE),
             ("draws", REFERENCE))
PRIMARY = "ppo_minus_draws"


# ---------------------------------------------------------------------------------------------------------------------
#  Frozen configuration and the reference descriptor
# ---------------------------------------------------------------------------------------------------------------------


def frozen_config(protocol_cfg: dict) -> dict:
    """The protocol block without the dispatcher's own ``study`` key, as plain JSON."""
    return json.loads(json.dumps({k: v for k, v in protocol_cfg.items() if k != "study"}, sort_keys=True))


def config_sha256(protocol_cfg: dict) -> str:
    return sha256_text(json.dumps(frozen_config(protocol_cfg), sort_keys=True, separators=(",", ":")))


def reference_params(protocol_cfg: dict) -> dict:
    """R-102's frozen residual parameters, exactly as the config records them, checked against their hash."""
    sel = {k: float(protocol_cfg["reference_residual"][k]) for k in PARAMS}
    got = params_sha256(sel)
    if got != protocol_cfg["reference_residual_sha256"]:
        raise SystemExit(f"reference residual hash {got} != the frozen {protocol_cfg['reference_residual_sha256']}")
    return sel


def reference_descriptor(protocol_cfg: dict) -> dict:
    """Source, state and config hashes of the reference controller (the incumbent asset is read, never written)."""
    from lares.envs.isaac_pen_spin.coordinate_repair import check_asset

    asset = protocol_cfg["incumbent_asset"]
    inc = check_asset(asset)
    here = os.path.dirname(os.path.abspath(__file__))
    return {"name": "d072_qualified_reference", "decisions": ["D-058", "D-070", "D-072"],
            "construction": "JointMotionResidual(reference_residual, incumbent_asset): clamp(incumbent + 2 clip(g "
                            "sign(v_f) max(|v_f| - v0, 0), -b, b) / (hi - lo), -1, 1)",
            "incumbent_asset": asset, "incumbent_json_sha256": sha256_file(os.path.join(asset, "incumbent.json")),
            "incumbent_files": dict(inc["files"]), "stroke_gain_constant": inc["constant"],
            "residual_params": reference_params(protocol_cfg),
            "residual_params_sha256": protocol_cfg["reference_residual_sha256"],
            "residual_source_sha256": sha256_file(os.path.join(here, "incumbent_residual.py"))}


def check_reference(protocol_cfg: dict) -> dict:
    """The committed descriptor must equal the one rebuilt from the files now on disk."""
    path = protocol_cfg["reference_descriptor"]
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing: the reference descriptor is frozen before the study runs")
    with open(path, encoding="utf-8") as f:
        saved = json.load(f)
    now = json.loads(json.dumps(reference_descriptor(protocol_cfg)))
    if saved != now:
        diff = sorted(k for k in set(saved) | set(now) if saved.get(k) != now.get(k))
        raise SystemExit(f"the reference differs from its frozen descriptor {path} in {diff}")
    return saved


# ---------------------------------------------------------------------------------------------------------------------
#  The common symbolic actor
# ---------------------------------------------------------------------------------------------------------------------


def theta_record(theta) -> list[float]:
    """``theta`` as the float32 values the controller uses, in a form JSON stores exactly."""
    arr = np.asarray(torch.as_tensor(theta).detach().cpu().numpy() if torch.is_tensor(theta) else theta,
                     dtype=np.float32).reshape(-1)
    if arr.size != N_COEF:
        raise ValueError(f"theta has {arr.size} coefficients, the actor has {N_COEF}")
    return [float(x) for x in arr]


def theta_sha256(theta) -> str:
    return sha256_text(json.dumps(theta_record(theta), separators=(",", ":")))


def axial_rate(obs):
    """The pen's signed angular rate about its own long axis (rad/s) from the observation: ``(N, 1)``."""
    sl = cases.obs_slices()
    q = obs[:, sl["obj_rot"]]
    q = q / (torch.norm(q, dim=1, keepdim=True) + 1e-8)
    w, x, y, z = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
    axis = torch.cat([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)], dim=1)
    axis = axis / (torch.norm(axis, dim=1, keepdim=True) + 1e-8)
    return torch.sum(obs[:, sl["obj_angvel"]] / VEL_OBS_SCALE * axis, dim=1, keepdim=True)


def residual(theta, s, vhat, ehat, cap: float):
    """The clipped residual for ``theta`` of shape ``(32,)`` (one policy) or ``(N, 32)`` (one per environment)."""
    idx = torch.as_tensor(GROUP_OF, device=vhat.device)
    a = theta[..., :N_JOINTS]
    b = theta[..., N_JOINTS:N_JOINTS + N_GROUPS].index_select(-1, idx)
    c = theta[..., N_JOINTS + N_GROUPS:].index_select(-1, idx)
    return (a * s + b * vhat + c * ehat).clamp(-cap, cap)


def executed_joint_order(ctx) -> list[str]:
    """The joint each action component drives, by short name. The env sorts its actuated indices, so action ``i``
    drives ``joint_names[actuated_dof_indices[i]]``; ``ctx.actuated_joint_names`` is the config's list in another
    (finger) order and is NOT the action order (the first R-103 smoke compared against it and failed its own gate)."""
    return [str(ctx.joint_names[int(i)]).split("_")[-1] for i in ctx.actuated_dof_indices]


def unit_normalization() -> dict:
    """Placeholder statistics (the smoke's reference-only phase runs before any are measured)."""
    return {"rms_velocity": [1.0] * N_JOINTS, "rms_posture": [1.0] * N_JOINTS}


class ResettableReference(JointMotionResidual):
    """R-102's frozen controller with a per-environment re-initialisation of its velocity filter.

    ``reset_envs(mask)`` marks environments whose episode just ended; at the next call their filter is set to the
    current velocity before the unchanged parent update runs, which then leaves it there exactly
    (``v + alpha (v - v) = v``): the same value a fresh controller's first step produces.
    """

    def reset(self, ctx):
        super().reset(ctx)
        self._pending = None

    def reset_envs(self, mask) -> None:
        self._pending = mask.clone() if self._pending is None else (self._pending | mask)

    def velocity(self, obs):
        """The parent's velocity read, unchanged: actuated joints, rad/s, NaN as 0."""
        v = (obs[:, self.sl][:, list(ACTUATED)] / VEL_OBS_SCALE).to(self.span.device, self.span.dtype)
        return torch.nan_to_num(v)

    def __call__(self, obs, t, ctx):
        if self._pending is not None and self.v_f is not None and bool(self._pending.any()):
            self.v_f = torch.where(self._pending.to(self.v_f.device).unsqueeze(1), self.velocity(obs), self.v_f)
        self._pending = None
        return super().__call__(obs, t, ctx)


class SymbolicResidualController:
    """``clamp(reference + residual_theta(features), -1, 1)`` behind the runner's controller interface.

    ``parts`` advances the reference (and its filter) one step and returns the reference action and the three
    features; ``mean`` is the unclamped symbolic mean for any ``theta``, which is what PPO's latent Gaussian is
    centred on. Training calls the two separately; evaluation calls the controller.
    """

    def __init__(self, name: str, reference, normalization: dict, actor_cfg: dict, theta=None):
        self.ref, self.name = reference, name
        self.cap, self.bound = float(actor_cfg["cap"]), float(actor_cfg["bound"])
        self.spin_target = float(actor_cfg["spin_target"])
        self.sigmas = float(actor_cfg["feature_clip_sigmas"])
        self._norm = normalization
        self._theta_cpu = torch.zeros(N_COEF) if theta is None else torch.tensor(theta_record(theta))
        if float(self._theta_cpu.abs().max()) > self.bound + 1e-6:      # float32(0.10) is a hair above 0.10
            raise ValueError("theta leaves the coefficient bound")
        self.params = {"theta": theta_record(self._theta_cpu), "theta_sha256": theta_sha256(self._theta_cpu),
                       "reference": getattr(reference, "params", None), "cap": self.cap, "bound": self.bound,
                       "execution": "clamp(reference + clip(a s + b_g vhat + c_g ehat, -cap, cap), -1, 1)"}
        self._stats = None

    def reset(self, ctx):
        self.ref.reset(ctx)
        dev, dt = self.ref.span.device, self.ref.span.dtype
        self.theta = self._theta_cpu.to(dev, dt)
        self.scale_v = self.sigmas * torch.tensor(self._norm["rms_velocity"], device=dev, dtype=dt)
        self.scale_e = self.sigmas * torch.tensor(self._norm["rms_posture"], device=dev, dtype=dt)
        self.sl_q = cases.obs_slices()["hand_dof_pos"]
        self._stats = {"steps": 0, "proposed": torch.zeros(N_JOINTS, device=dev), "saturated": torch.zeros(N_JOINTS, device=dev),
                       "executed": torch.zeros(N_JOINTS, device=dev), "values": 0}

    def reset_envs(self, mask) -> None:
        self.ref.reset_envs(mask)

    def parts(self, obs, t, ctx) -> dict:
        ref = self.ref(obs, t, ctx)
        dev, dt = self.ref.span.device, self.ref.span.dtype
        ref = ref.to(dev, dt)
        o = torch.nan_to_num(obs.to(dev, dt))
        s = ((self.spin_target - axial_rate(o).abs()) / self.spin_target).clamp(0.0, 1.0)
        vhat = (self.ref.v_f / self.scale_v).clamp(-1.0, 1.0)
        ehat = ((ref - o[:, self.sl_q][:, list(ACTUATED)]) / self.scale_e).clamp(-1.0, 1.0)
        return {"ref": ref, "s": s, "vhat": vhat, "ehat": ehat}

    def mean(self, parts: dict, theta):
        return parts["ref"] + residual(theta, parts["s"], parts["vhat"], parts["ehat"], self.cap)

    def __call__(self, obs, t, ctx):
        p = self.parts(obs, t, ctx)
        mu = self.mean(p, self.theta)
        action = mu.clamp(-1.0, 1.0)
        st = self._stats
        st["steps"] += 1
        st["values"] += int(action.shape[0])
        st["proposed"] += (mu - p["ref"]).abs().sum(0)
        st["executed"] += (action - p["ref"]).abs().sum(0)
        st["saturated"] += (action.abs() >= 1.0).float().sum(0)
        return action

    def activity(self) -> dict:
        st = self._stats
        n = max(1, st["values"])
        prop, exe = st["proposed"].cpu().numpy() / n, st["executed"].cpu().numpy() / n
        return {"values": st["values"], "mean_abs_proposed": float(prop.mean()), "mean_abs_executed": float(exe.mean()),
                "per_joint": {j: {"proposed": float(prop[i]), "executed": float(exe[i]),
                                  "saturated_fraction": float(st["saturated"][i]) / n} for i, j in enumerate(JOINTS)}}

    def extra_trace(self) -> dict:
        # ``run_cell`` calls this after the rollout; the activity goes into the cell's controller_params too.
        self.params["residual_activity"] = self.activity()
        a = self.params["residual_activity"]
        return {"proposed": np.array([a["per_joint"][j]["proposed"] for j in JOINTS]),
                "executed": np.array([a["per_joint"][j]["executed"] for j in JOINTS]),
                "saturated_fraction": np.array([a["per_joint"][j]["saturated_fraction"] for j in JOINTS])}


def build_reference(protocol_cfg: dict, name: str = REFERENCE, inner=None, resettable: bool = True):
    cls = ResettableReference if resettable else JointMotionResidual
    return cls(name, reference_params(protocol_cfg), protocol_cfg["incumbent_asset"], inner=inner,
               control_dt=float(protocol_cfg["control_dt"]))


def build_controller(protocol_cfg: dict, name: str, theta, normalization: dict, inner=None):
    """The one construction of the symbolic actor: training, the smoke's reconstruction check and evaluation cells."""
    return SymbolicResidualController(name, build_reference(protocol_cfg, f"{name}:reference", inner=inner),
                                      normalization, protocol_cfg["actor"], theta)


# ---------------------------------------------------------------------------------------------------------------------
#  Reward
# ---------------------------------------------------------------------------------------------------------------------


def step_reward(center_dist, spin_rate, fell, executed, reference, reward_cfg: dict, cap: float):
    """``exp(-(d/d0)^2) min(|w|/w0, 1) - P [fell] - k mean_j((executed_j - reference_j)/cap)^2`` per environment."""
    roll = torch.exp(-(center_dist / float(reward_cfg["center_scale"])) ** 2) * \
        (spin_rate.abs() / float(reward_cfg["spin_cap"])).clamp(max=1.0)
    cost = float(reward_cfg["residual_cost"]) * (((executed - reference) / cap) ** 2).mean(dim=1)
    return torch.nan_to_num(roll) - float(reward_cfg["fall_penalty"]) * fell.float() - cost


# ---------------------------------------------------------------------------------------------------------------------
#  Parameter-space learners (numpy; the child only supplies returns)
# ---------------------------------------------------------------------------------------------------------------------


def draw_thetas(seed: int, cfg: dict, bound: float) -> np.ndarray:
    """All of a block's independent draws: ``rho z``, ``z ~ U[-bound, bound]^32``, ``rho`` log-uniform in
    ``[rho_min, 1]``; shape ``(generations * per_generation, 32)``."""
    rng = np.random.default_rng(int(seed) + int(cfg["rng_offset"]))
    n = int(cfg["generations"]) * int(cfg["per_generation"])
    z = rng.uniform(-bound, bound, size=(n, N_COEF))
    rho = np.exp(rng.uniform(math.log(float(cfg["rho_min"])), 0.0, size=(n, 1)))
    return rho * z


class DrawSearch:
    """Independent draws: no adaptation; the nominee is the best training score so far (ties: the earlier draw)."""

    def __init__(self, seed: int, cfg: dict, bound: float):
        self.cfg, self.thetas = cfg, draw_thetas(seed, cfg, bound)
        self.per = int(cfg["per_generation"])
        self.scores: list[float] = []
        self.best = None

    def ask(self, generation: int) -> np.ndarray:
        return self.thetas[generation * self.per:(generation + 1) * self.per]

    def tell(self, generation: int, scores) -> None:
        for i, sc in enumerate(scores):
            self.scores.append(float(sc))
            if self.best is None or float(sc) > self.best["score"]:
                self.best = {"draw": generation * self.per + i, "score": float(sc)}

    def nominee(self) -> np.ndarray:
        return np.zeros(N_COEF) if self.best is None else self.thetas[self.best["draw"]]


class ARS:
    """ARS V1-t on the bounded coefficients: antithetic directions, the top ``b`` by ``max(r+, r-)``, the step
    divided by the standard deviation of the returns it uses. The nominee is the current ``theta``."""

    def __init__(self, seed: int, cfg: dict, bound: float):
        self.rng = np.random.default_rng(int(seed) + int(cfg["rng_offset"]))
        self.n, self.top = int(cfg["directions"]), int(cfg["top"])
        self.nu, self.alpha, self.bound = float(cfg["nu"]), float(cfg["alpha"]), float(bound)
        self.theta = np.zeros(N_COEF)
        self.delta = None

    def ask(self, generation: int) -> np.ndarray:
        """``(2 n, 32)`` candidates: rows ``0..n-1`` are ``theta + nu delta``, rows ``n..2n-1`` are ``theta - nu delta``."""
        self.delta = self.rng.standard_normal((self.n, N_COEF))
        return np.clip(np.concatenate([self.theta + self.nu * self.delta, self.theta - self.nu * self.delta]),
                       -self.bound, self.bound)

    def tell(self, generation: int, scores) -> dict:
        r = np.asarray(scores, dtype=float)
        rp, rm = r[:self.n], r[self.n:]
        order = np.argsort(-np.maximum(rp, rm), kind="stable")[:self.top]
        sigma = float(np.concatenate([rp[order], rm[order]]).std())
        if sigma > 1e-8:
            step = self.alpha / (self.top * sigma) * ((rp[order] - rm[order])[:, None] * self.delta[order]).sum(0)
            self.theta = np.clip(self.theta + step, -self.bound, self.bound)
        return {"sigma_r": sigma, "used": [int(i) for i in order]}

    def nominee(self) -> np.ndarray:
        return self.theta.copy()


# ---------------------------------------------------------------------------------------------------------------------
#  PPO arithmetic (the latent-action contract)
# ---------------------------------------------------------------------------------------------------------------------


def gaussian_logp(u, mu, sigma: float):
    """Log-density of the *latent* action ``u ~ N(mu, sigma^2 I)``, summed over joints. The executed action is
    ``clamp(u, -1, 1)``; the clamp belongs to the environment, so no density is evaluated on a clipped action."""
    return (-0.5 * ((u - mu) / sigma) ** 2 - math.log(sigma) - 0.5 * math.log(2.0 * math.pi)).sum(dim=-1)


def gae(rewards, values, last_value, terminated, truncated, gamma: float, lam: float):
    """Advantages and returns for ``(T, N)`` rollouts. A fall ends the return; a time-out is bootstrapped with the
    value of the state it was taken in (the next observation already belongs to the following episode)."""
    T = rewards.shape[0]
    r = rewards + gamma * values * truncated.float()
    done = (terminated | truncated).float()
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(last_value)
    for t in reversed(range(T)):
        nxt = last_value if t == T - 1 else values[t + 1]
        delta = r[t] + gamma * nxt * (1.0 - done[t]) - values[t]
        last = delta + gamma * lam * (1.0 - done[t]) * last
        adv[t] = last
    return adv, adv + values


class Critic(torch.nn.Module):
    """The neural critic (never deployed): a running observation normaliser and an ELU MLP; output times ``scale``."""

    def __init__(self, obs_dim: int, hidden, scale: float):
        super().__init__()
        layers, d = [], obs_dim
        for h in hidden:
            layers += [torch.nn.Linear(d, int(h)), torch.nn.ELU()]
            d = int(h)
        layers.append(torch.nn.Linear(d, 1))
        self.net = torch.nn.Sequential(*layers)
        self.scale = float(scale)
        self.register_buffer("mean", torch.zeros(obs_dim))
        self.register_buffer("var", torch.ones(obs_dim))
        self.register_buffer("count", torch.tensor(1e-4))

    def observe(self, obs) -> None:
        x = torch.nan_to_num(obs.reshape(-1, obs.shape[-1]))
        n = x.shape[0]
        m, v = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + n
        delta = m - self.mean
        self.var.copy_((self.var * self.count + v * n + delta ** 2 * self.count * n / tot) / tot)
        self.mean.copy_(self.mean + delta * n / tot)
        self.count.copy_(tot)

    def forward(self, obs):
        x = ((torch.nan_to_num(obs) - self.mean) / torch.sqrt(self.var + 1e-8)).clamp(-10.0, 10.0)
        return self.net(x).squeeze(-1) * self.scale


def surrogate(psi, batch: dict, adv, cfg: dict, bound: float, cap: float):
    """PPO's clipped surrogate (to maximise) and the exact KL of the old mean's Gaussian from the new one."""
    sigma = float(cfg["sigma"])
    mu = batch["ref"] + residual(psi * bound, batch["s"], batch["vhat"], batch["ehat"], cap)
    ratio = torch.exp(gaussian_logp(batch["u"], mu, sigma) - batch["logp"])
    clip = float(cfg["clip"])
    obj = torch.min(ratio * adv, ratio.clamp(1.0 - clip, 1.0 + clip) * adv).mean()
    kl = (((mu - batch["mu"]) ** 2).sum(-1) / (2.0 * sigma ** 2)).mean()
    return obj, kl, ratio


def ppo_update(psi, actor_opt, critic, critic_opt, roll: dict, cfg: dict, bound: float, cap: float, lr: float,
               train_actor: bool, generator=None) -> dict:
    """One PPO update from one rollout (``roll`` holds flat ``(T N, ...)`` tensors). Returns the new actor rate and
    diagnostics. ``psi = theta / bound`` is projected to ``[-1, 1]`` after every actor step."""
    n = roll["u"].shape[0]
    adv = roll["adv"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    mb = n // int(cfg["minibatches"])
    kls, vlosses, objs = [], [], []
    for _ in range(int(cfg["epochs"])):
        perm = torch.randperm(n, device=adv.device, generator=generator)
        for k in range(int(cfg["minibatches"])):
            idx = perm[k * mb:(k + 1) * mb]
            v = critic(roll["obs"][idx])
            vloss = ((v - roll["ret"][idx]) / critic.scale).pow(2).mean()
            critic_opt.zero_grad()
            vloss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), float(cfg["max_grad_norm"]))
            critic_opt.step()
            vlosses.append(float(vloss))
            if not train_actor:
                continue
            b = {key: roll[key][idx] for key in ("ref", "s", "vhat", "ehat", "u", "logp", "mu")}
            obj, kl, _ = surrogate(psi, b, adv[idx], cfg, bound, cap)
            for g in actor_opt.param_groups:
                g["lr"] = lr
            actor_opt.zero_grad()
            (-obj).backward()
            actor_opt.step()
            with torch.no_grad():
                psi.clamp_(-1.0, 1.0)
            kl = float(kl)
            kls.append(kl)
            objs.append(float(obj))
            target = float(cfg["kl_target"])
            if kl > 2.0 * target:
                lr = max(float(cfg["lr_min"]), lr / 1.5)
            elif kl < 0.5 * target:
                lr = min(float(cfg["lr_max"]), lr * 1.5)
    return {"lr": lr, "kl": float(np.mean(kls)) if kls else None, "value_loss": float(np.mean(vlosses)),
            "surrogate": float(np.mean(objs)) if objs else None}


# ---------------------------------------------------------------------------------------------------------------------
#  Schedule
# ---------------------------------------------------------------------------------------------------------------------


def schedule(protocol_cfg: dict) -> dict:
    """The common budget: every arm spends exactly ``transitions`` and nominates at the same ``checkpoints`` counts."""
    tr, n, h = protocol_cfg["training"], int(protocol_cfg["training"]["num_envs"]), int(protocol_cfg["training"]["horizon"])
    total, k = int(tr["transitions"]), int(tr["checkpoints"])
    d, a, p = protocol_cfg["draws"], protocol_cfg["ars"], protocol_cfg["ppo"]
    checks = {
        "draws generations x envs x horizon": int(d["generations"]) * n * h,
        "draws envs": int(d["per_generation"]) * int(d["envs_per_draw"]) * h * int(d["generations"]),
        "ars iterations x envs x horizon": int(a["iterations"]) * n * h,
        "ars envs": 2 * int(a["directions"]) * int(a["envs_per_candidate"]) * h * int(a["iterations"]),
        "ppo updates x steps x envs": int(p["updates"]) * int(p["steps"]) * n,
    }
    bad = {name: v for name, v in checks.items() if v != total}
    if bad or total > int(tr["cap"]) or int(d["generations"]) % k or int(a["iterations"]) % k or int(p["updates"]) % k:
        raise SystemExit(f"the training schedule is not one common budget of {total} transitions: {bad}")
    return {"transitions": total, "checkpoint_transitions": [total // k * (i + 1) for i in range(k)],
            "draws_every": int(d["generations"]) // k, "ars_every": int(a["iterations"]) // k,
            "ppo_every": int(p["updates"]) // k}


# ---------------------------------------------------------------------------------------------------------------------
#  Manifests and evaluation controllers
# ---------------------------------------------------------------------------------------------------------------------


def blocks(protocol_cfg: dict) -> list[int]:
    return sorted(int(b) for b in protocol_cfg["training"]["blocks"])


def manifest_for(protocol_cfg: dict, role: str, block: int) -> str:
    seed = int(protocol_cfg[f"{role}_seeds"][int(block)])
    return str(protocol_cfg[f"{role}_manifest_template"]).format(seed=seed)


def build_manifests(protocol_cfg: dict) -> list[str]:
    from lares.eval.manifest import EvaluationManifest, assert_disjoint

    mine = {}
    for role in ("selection", "fresh"):
        for b in blocks(protocol_cfg):
            mine[manifest_for(protocol_cfg, role, b)] = cases.build_manifest(
                STAGE, int(protocol_cfg[f"{role}_seeds"][b]), int(protocol_cfg[f"{role}_cases"]), label=f"reward_{role}")
    others = [EvaluationManifest.load(p) for p in sorted(glob.glob("config/manifests/pen_spin_stage3_*.yaml"))
              if p not in mine]
    assert_disjoint(list(mine.values()) + others)
    return [p for p, m in mine.items() if cases.freeze_manifest(p, m)]


def evaluation_allowance(protocol_cfg: dict) -> int:
    """The most evaluation cases the protocol can run: per block, the reference and every checkpoint on the selection
    manifest, then the reference and every finalist on the fresh one."""
    k, a = int(protocol_cfg["training"]["checkpoints"]), len(ARMS)
    return len(blocks(protocol_cfg)) * ((1 + a * k) * int(protocol_cfg["selection_cases"])
                                        + (1 + a) * int(protocol_cfg["fresh_cases"]))


def run_dir(out_dir: str, block: int, arm: str) -> str:
    return os.path.join(out_dir, "train", f"b{int(block)}_{arm}")


def load_normalization(out_dir: str) -> dict:
    with open(os.path.join(out_dir, "normalization.json"), encoding="utf-8") as f:
        return json.load(f)


def load_checkpoints(out_dir: str, block: int, arm: str) -> list[dict]:
    with open(os.path.join(run_dir(out_dir, block, arm), "checkpoints.json"), encoding="utf-8") as f:
        return json.load(f)["checkpoints"]


def make_controller(name: str, protocol_cfg: dict, out_dir: str | None = None):
    """Evaluation cells: ``reference`` (R-102's class, unchanged), ``b<block>_<arm>_c<k>`` (a training checkpoint) or
    ``b<block>_<arm>_final`` (the frozen finalist)."""
    if name == REFERENCE:
        return build_reference(protocol_cfg, REFERENCE, resettable=False)
    out_dir = out_dir or str(protocol_cfg["log_dir"])
    block, arm, which = name.split("_")
    block = int(block[1:])
    if arm not in ARMS:
        raise ValueError(f"unknown controller {name!r}")
    if which == "final":
        with open(os.path.join(out_dir, "finalists.json"), encoding="utf-8") as f:
            entry = json.load(f)["finalists"][f"b{block}_{arm}"]
    else:
        entry = load_checkpoints(out_dir, block, arm)[int(which[1:]) - 1]
    if theta_sha256(entry["theta"]) != entry["theta_sha256"]:
        raise SystemExit(f"{name}: the stored theta does not match its hash")
    return build_controller(protocol_cfg, name, entry["theta"], load_normalization(out_dir))


# ---------------------------------------------------------------------------------------------------------------------
#  Selection and inference
# ---------------------------------------------------------------------------------------------------------------------


def selection_key(rows: list[dict], k: int) -> tuple:
    """More sustained successes, then higher tail-hit fraction, then fewer drops, then the earlier checkpoint."""
    return (-int(sum(r["sustained"] for r in rows)), -float(np.mean([r["tail_hit_fraction"] for r in rows])),
            int(sum(not r["completed"] for r in rows)), int(k))


def select_checkpoint(reference_rows: list[dict], checkpoint_rows: dict) -> int:
    """``0`` (the reference: training rejected) or the 1-based checkpoint with the best :func:`selection_key`."""
    keyed = [(selection_key(reference_rows, 0), 0)] + [(selection_key(rows, k), k) for k, rows in checkpoint_rows.items()]
    return min(keyed)[1]


def holm(p_values: dict, alpha: float = 0.05) -> dict:
    """Holm's step-down adjusted p-values and rejections for a family of named two-sided tests."""
    names = sorted(p_values, key=lambda n: (p_values[n], n))
    m, out, running, open_ = len(names), {}, 0.0, True
    for i, name in enumerate(names):
        running = max(running, min(1.0, (m - i) * float(p_values[name])))
        open_ = open_ and running < alpha
        out[name] = {"p": float(p_values[name]), "p_holm": running, "rejects": bool(open_)}
    return out


def _paired(a_rows: list, b_rows: list, key) -> list[float]:
    if [r["task_hash"] for r in a_rows] != [r["task_hash"] for r in b_rows]:
        raise ValueError("rows are not paired case for case")
    return [key(a) - key(b) for a, b in zip(a_rows, b_rows)]


def contrasts(rows: dict, threshold: float, drop_margin: float) -> dict:
    """``rows[block][arm]`` (``arm`` in ARMS or ``reference``) -> the six planned contrasts.

    Per contrast: the case-paired interval over every fresh pair (conditional on the trained finalists), the interval
    over the per-block means (the only statement across training runs), wins/losses/ties, the paired drop
    difference with its one-sided 95% upper bound, and the Holm-adjusted case-paired test.
    """
    from lares.eval.stats import t_summary, win_loss_tie

    out = {}
    for a, b in CONTRASTS:
        per_block, ds, dd, dh = [], [], [], []
        for blk in sorted(rows):
            d = _paired(rows[blk][a], rows[blk][b], lambda r: float(r["sustained"]))
            per_block.append(float(np.mean(d)))
            ds += d
            dd += _paired(rows[blk][a], rows[blk][b], lambda r: float(not r["completed"]))
            dh += _paired(rows[blk][a], rows[blk][b], lambda r: float(r["tail_hit_fraction"]))
        ts, td = t_summary(ds), t_summary(dd, level=0.90)
        out[f"{a}_minus_{b}"] = {
            "estimate": float(np.mean(per_block)), "case_paired": ts, "win_loss_tie": win_loss_tie(ds),
            "per_block": per_block, "block_level": t_summary(per_block), "tail_hit_fraction": t_summary(dh),
            "drops": {"mean": td["mean"], "one_sided_95_upper": td["high"], "margin": drop_margin,
                      "non_inferior": bool(td["high"] is not None and td["high"] <= drop_margin + 1e-12)},
            "threshold": threshold,
            "magnitude_supported": bool(ts["mean"] >= threshold and ts["low"] is not None and ts["low"] > 0),
            "magnitude_refuted": bool(ts["high"] is not None and ts["high"] < threshold),
            "learner_level_excludes_zero": bool(t_summary(per_block)["low"] is not None and t_summary(per_block)["low"] > 0),
        }
    family = holm({name: c["case_paired"]["p_value"] for name, c in out.items()})
    for name, c in out.items():
        c["holm"] = family[name]
        c["positive_effect_claim"] = bool(family[name]["rejects"] and c["estimate"] > 0)
    return out


# ---------------------------------------------------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------------------------------------------------

CHILD_TIMEOUT_S = 3 * 3600


def _dump(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


AMENDMENT = "config/pen_spin/symbolic_reward_reference/amendment_two_blocks.json"


def amended_blocks(protocol_cfg: dict, out_dir: str, training: dict, sched: dict):
    """The blocks the recorded amendment evaluates, or ``None`` when there is no amendment (the protocol then stops
    incomplete with no evaluation). Every evaluated block must be fully trained in all three arms, with every
    scheduled checkpoint nominee on disk; nothing is trained, extended or replaced."""
    if not os.path.exists(AMENDMENT):
        return None
    wanted = [int(b) for b in _load(AMENDMENT)["evaluated_blocks"]]
    k = int(protocol_cfg["training"]["checkpoints"])
    for b in wanted:
        for arm in ARMS:
            rec = training.get(f"b{b}_{arm}")
            if rec is None or not rec["complete"] or rec["transitions"] != sched["transitions"]:
                raise SystemExit(f"the amendment evaluates block {b}, but b{b}_{arm} is not a complete training run")
            cks = load_checkpoints(out_dir, b, arm)
            if [int(c["k"]) for c in cks] != list(range(1, k + 1)) or \
                    [int(c["transitions"]) for c in cks] != sched["checkpoint_transitions"]:
                raise SystemExit(f"b{b}_{arm} does not hold the {k} scheduled checkpoint nominees")
    return wanted


def run_child(request: dict, request_path: str) -> dict:
    """One fresh Isaac process for a smoke stage or a training run; the response is the child's own record."""
    if os.path.exists(request["response_path"]):
        print("exists", request["response_path"], flush=True)
        return _load(request["response_path"])
    marker = request["response_path"] + ".started"
    if os.path.exists(marker):
        raise SystemExit(f"{marker}: this child already started and left no result. It is never relaunched from its "
                         "seed; the protocol stops here and returns to triage.")
    _dump(request_path, request)
    _dump(marker, {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "request": request_path})
    log = request_path.replace(".request.json", ".log")
    cmd = [sys.executable, "-u", "-m", "lares.envs.isaac_pen_spin.symbolic_reward_child", request_path]
    print("running", " ".join(cmd), flush=True)
    with open(log, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=CHILD_TIMEOUT_S)
    if proc.returncode != 0 or not os.path.exists(request["response_path"]):
        with open(log, encoding="utf-8", errors="replace") as f:
            raise RuntimeError(f"child failed (exit {proc.returncode}); log {log}:\n{f.read()[-3000:]}")
    os.remove(marker)
    return _load(request["response_path"])


def _cell(protocol_cfg: dict, config_path: str, mpath: str, name: str, cell: str, expect_sha: str | None) -> list[dict]:
    study_cell(protocol_cfg, config_path, mpath, name, cell, record_obs=False)
    if expect_sha is not None:
        got = (_load(cell).get("controller_params") or {}).get("theta_sha256")
        if got != expect_sha:
            raise SystemExit(f"{cell}: evaluated theta {got} is not the training record's {expect_sha}")
    return load_cell(cell)[1]


def _cell_summary(rows: list[dict], cell: str | None = None) -> dict:
    out = {**arm_summary(rows), **failure_split(rows)}
    if cell and os.path.exists(cell):
        out["residual_activity"] = (_load(cell).get("controller_params") or {}).get("residual_activity")
    return out


def run(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    if os.path.normpath(out_dir) != os.path.normpath(str(protocol_cfg["log_dir"])):
        raise SystemExit("this study's evaluation cells read checkpoints from the protocol's own log_dir; do not override it")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    descriptor = check_reference(protocol_cfg)
    sched = schedule(protocol_cfg)
    cfg_sha = config_sha256(protocol_cfg)
    freeze_json(os.path.join(out_dir, "frozen_config.json"), {"config_sha256": cfg_sha, "config": frozen_config(protocol_cfg),
                                                              "reference": descriptor, "schedule": sched}, "frozen configuration")
    manifests = {role: {b: cases.require_frozen(manifest_for(protocol_cfg, role, b)) for b in blocks(protocol_cfg)}
                 for role in ("selection", "fresh")}
    base = {"config_path": os.path.abspath(config_path), "study": STUDY, "out_dir": out_dir, "config_sha256": cfg_sha}

    # ---- smoke gate -------------------------------------------------------------------------------------------------
    sdir = os.path.join(out_dir, "smoke")
    os.makedirs(sdir, exist_ok=True)
    smoke = run_child({**base, "mode": "smoke", "response_path": os.path.join(sdir, "smoke.json")},
                      os.path.join(sdir, "smoke.request.json"))
    n_eval = len(blocks(protocol_cfg)) * (1 + len(ARMS) * int(protocol_cfg["training"]["checkpoints"]) + 1 + len(ARMS))
    projected_h = (len(blocks(protocol_cfg)) * len(ARMS) * (sched["transitions"] / smoke["throughput"]["transitions_per_s"] + 60.0)
                   + n_eval * 40.0) / 3600.0
    smoke_ok = bool(smoke["passes"]) and projected_h <= float(protocol_cfg["max_hours"])
    summary = {"study": STUDY, "config_sha256": cfg_sha, "reference": descriptor, "schedule": sched,
               "smoke": {k: v for k, v in smoke.items() if k != "normalization"}, "projected_hours": projected_h,
               "smoke_gate": "pass" if smoke_ok else "fail"}
    if not smoke_ok:
        summary["note"] = "the smoke gate failed: no training or evaluation ran"
        _dump(os.path.join(out_dir, "summary.json"), summary)
        return summary
    freeze_json(os.path.join(out_dir, "normalization.json"), smoke["normalization"], "feature normalisation")

    # ---- training (hard ceiling on episodes started; the evaluation allowance is reserved first) ---------------------
    ceiling, reserve = int(protocol_cfg["episode_ceiling"]), evaluation_allowance(protocol_cfg)
    started = int(smoke["episodes_started"]) + int(protocol_cfg.get("prior_episodes_started", 0))
    training, eval_blocks, stopped_at = {}, None, None
    for b in blocks(protocol_cfg):
        for arm in ARMS:
            d = run_dir(out_dir, b, arm)
            os.makedirs(d, exist_ok=True)
            allowance = ceiling - reserve - started
            rec = None
            if allowance >= int(protocol_cfg["training"]["num_envs"]) or os.path.exists(os.path.join(d, "train.json")):
                rec = run_child({**base, "mode": "train", "arm": arm, "block": b, "run_dir": d, "episode_cap": allowance,
                                 "seed": int(protocol_cfg["training"]["blocks"][b]),
                                 "response_path": os.path.join(d, "train.json")}, os.path.join(d, "train.request.json"))
                training[f"b{b}_{arm}"] = rec
                started += int(rec["episodes_started"])
            if rec is None or not rec["complete"] or rec["transitions"] != sched["transitions"]:
                eval_blocks = amended_blocks(protocol_cfg, out_dir, training, sched)
                if eval_blocks is not None:
                    stopped_at = f"b{b}_{arm}"
                    break
                summary.update({"budget": {"ceiling": ceiling, "evaluation_reserve": reserve, "episodes_started": started},
                                "incomplete": f"b{b}_{arm}: the episode ceiling was reached; the protocol stops with an "
                                              "incomplete result and no arm is extended or shortened on its own",
                                "training": {k: {kk: v[kk] for kk in ("transitions", "episodes_started", "falls", "complete")}
                                             for k, v in training.items()}})
                _dump(os.path.join(out_dir, "summary.json"), summary)
                return summary
        if eval_blocks is not None:
            break
    if eval_blocks is None:
        eval_blocks = blocks(protocol_cfg)

    # ---- checkpoint selection ---------------------------------------------------------------------------------------
    zero_sha = theta_sha256(np.zeros(N_COEF))
    selection, finalists = {}, {}
    for b in eval_blocks:
        mpath = manifests["selection"][b]
        sel_dir = os.path.join(out_dir, "selection", f"b{b}")
        os.makedirs(sel_dir, exist_ok=True)
        ref_rows = _cell(protocol_cfg, config_path, mpath, REFERENCE, os.path.join(sel_dir, "reference_cell.json"), None)
        scored = {zero_sha: ref_rows}                       # a nominee equal to one already scored reuses its rows
        selection[f"b{b}"] = {"manifest": cases.manifest_record(mpath), REFERENCE: _cell_summary(ref_rows)}
        for arm in ARMS:
            ck_rows, ck_info = {}, []
            for ck in load_checkpoints(out_dir, b, arm):
                k, sha = int(ck["k"]), ck["theta_sha256"]
                cell = os.path.join(sel_dir, f"{arm}_c{k}_cell.json")
                reused = sha in scored
                if not reused:
                    scored[sha] = _cell(protocol_cfg, config_path, mpath, f"b{b}_{arm}_c{k}", cell, sha)
                ck_rows[k] = scored[sha]
                ck_info.append({"k": k, "transitions": ck["transitions"], "theta_sha256": sha, "reused_cell": reused,
                                **_cell_summary(scored[sha], None if reused else cell)})
            pick = select_checkpoint(ref_rows, ck_rows)
            selection[f"b{b}"][arm] = {"checkpoints": ck_info, "selected": pick, "training_rejected": pick == 0}
            theta = [0.0] * N_COEF if pick == 0 else load_checkpoints(out_dir, b, arm)[pick - 1]["theta"]
            finalists[f"b{b}_{arm}"] = {"block": b, "arm": arm, "checkpoint": pick, "theta": theta_record(theta),
                                        "theta_sha256": theta_sha256(theta)}
    freeze_json(os.path.join(out_dir, "finalists.json"),
                {"rule": "sustained successes, then tail-hit fraction, then fewer drops, then the earlier checkpoint "
                         "(0 = the reference)", "finalists": finalists}, "finalists")

    # ---- fresh evaluation -------------------------------------------------------------------------------------------
    rows, fresh = {}, {}
    for b in eval_blocks:
        mpath = manifests["fresh"][b]
        fdir = os.path.join(out_dir, "fresh", f"b{b}")
        os.makedirs(fdir, exist_ok=True)
        ref_cell = os.path.join(fdir, "reference_cell.json")
        rows[b] = {REFERENCE: _cell(protocol_cfg, config_path, mpath, REFERENCE, ref_cell, None)}
        fresh[f"b{b}"] = {"manifest": cases.manifest_record(mpath), REFERENCE: _cell_summary(rows[b][REFERENCE])}
        scored = {zero_sha: rows[b][REFERENCE]}
        for arm in ARMS:
            sha = finalists[f"b{b}_{arm}"]["theta_sha256"]
            cell = os.path.join(fdir, f"{arm}_cell.json")
            reused = sha in scored
            if not reused:
                scored[sha] = _cell(protocol_cfg, config_path, mpath, f"b{b}_{arm}_final", cell, sha)
            rows[b][arm] = scored[sha]
            fresh[f"b{b}"][arm] = {"reused_cell": reused, **_cell_summary(scored[sha], None if reused else cell)}

    c = contrasts(rows, float(protocol_cfg["threshold"]), float(protocol_cfg["drop_margin"]))
    head = c[PRIMARY]
    n_cells = len(glob.glob(os.path.join(out_dir, "selection", "*", "*_cell.json"))) + \
        len(glob.glob(os.path.join(out_dir, "fresh", "*", "*_cell.json")))
    eval_eps = sum(len(_load(p)["episodes"]) for p in glob.glob(os.path.join(out_dir, "selection", "*", "*_cell.json"))
                   + glob.glob(os.path.join(out_dir, "fresh", "*", "*_cell.json")))
    summary.update({
        "paired_success_difference": head["estimate"],
        "paired_success_difference_ci95": [head["case_paired"]["low"], head["case_paired"]["high"]],
        "contrasts": c, "selection": selection, "fresh": fresh, "finalists": finalists,
        "training": {k: {kk: v[kk] for kk in ("transitions", "episodes_started", "falls", "wall_time_s", "updates",
                                              "final_theta_sha256", "final_theta_abs_max", "curve_first_last")}
                     for k, v in training.items()},
        "accounting": {"training_transitions": sum(v["transitions"] for v in training.values()),
                       "training_episodes_started": sum(v["episodes_started"] for v in training.values()),
                       "smoke_transitions": smoke["transitions"], "smoke_episodes_started": smoke["episodes_started"],
                       "episodes_started_total": started + eval_eps, "episode_ceiling": ceiling,
                       "evaluation_processes": n_cells,
                       "evaluation_episodes": eval_eps, "llm_calls": 0,
                       "driver_wall_time_s_this_invocation": time.time() - t0},
        "symbolic_complexity": {"coefficients": N_COEF, "features": ["axial_rate_deficit", "filtered_joint_velocity",
                                                                     "posture_error"], "trainable_state": 0},
        "limitation": "a finite pilot: training runs of 460,800 transitions per learner and block; the case-paired "
                      "intervals are conditional on the evaluated finalists; no promotion follows from it",
        "evaluated_blocks": list(eval_blocks),
    })
    if stopped_at is not None:
        summary["incomplete"] = (f"incomplete: {len(eval_blocks)} of {len(blocks(protocol_cfg))} blocks. Training stopped on "
                                 f"the episode ceiling at {stopped_at}; that block is void and excluded from selection "
                                 "and inference (amendment)")
        summary["amendment"] = _load(AMENDMENT)
    _dump(os.path.join(out_dir, "summary.json"), summary)
    return summary


__all__ = ["AMENDMENT", "ARMS", "ARS", "COEF_NAMES", "CONTRASTS", "Critic", "DrawSearch", "GROUP_OF", "JOINTS", "N_COEF", "PRIMARY",
           "REFERENCE", "ResettableReference", "STUDY", "amended_blocks", "SymbolicResidualController", "axial_rate", "build_controller",
           "build_manifests", "build_reference", "check_reference", "config_sha256", "contrasts", "draw_thetas", "executed_joint_order", "gae",
           "gaussian_logp", "holm", "make_controller", "ppo_update", "reference_descriptor", "residual", "run", "schedule",
           "select_checkpoint", "selection_key", "step_reward", "surrogate", "theta_record", "theta_sha256"]
