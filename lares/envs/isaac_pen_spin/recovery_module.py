"""R-116: a learned symbolic re-capture module anchored on the frozen R-106 finalist. No LLM.

D-080 showed that reopening and recatching restarts rolling in a stuck episode, but drops the pen in 9% of treated
episodes and rarely yields a steady, centred roll. This module is that script with 11 bounded parameters, trained
by ARS on the full-episode sustained-success count (so a drop costs what it costs):

    threshold r, decision step tau, ramp-in n_in, ramp-out n_out, blend depth d_g per joint group (6), and a
    post-recapture stroke-gain multiplier m held for 60 steps.

One decision per episode, at step ``T = round(tau)``: eligible if the pen is centred and its mean |axial rate| over
the 30 observations ending at ``T`` is below ``r``. If eligible, with ``u = t - T``::

    beta(u) = min((u + 1)/n_in, 1)            for u < n_in
            = 1 - (u - n_in)/n_out            for n_in <= u < n_in + n_out,        0 afterwards
    sent_j  = clamp((1 - beta d_g(j)) a_root,j + beta d_g(j) a_neutral,j, -1, 1)

At the initial parameters this is the D-080 script exactly. The root's action and filter are computed every step
regardless; the controller reads only the observation, so one class serves training and the fresh-process evaluator.
``python -m lares.envs.isaac_pen_spin.recovery_module <request.json>`` is the Isaac child (smoke, training).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin import recovery_authority as ra
from lares.envs.isaac_pen_spin import symbolic_reward as sr
from lares.envs.isaac_pen_spin.protocols import freeze_json, load_cell, study_cell
from lares.utils.hashing import sha256_text

STUDY = "symbolic_recovery_ars"
#: (name, low, high, initial): the initial point is the D-080 script
PARAMS = (("threshold", 1.0, 4.0, 3.0), ("decision_step", 60.0, 180.0, 120.0), ("ramp_in", 5.0, 40.0, 20.0),
          ("ramp_out", 10.0, 80.0, 40.0), *((f"depth_{g}", 0.0, 1.0, 1.0) for g in sr.GROUPS), ("stroke_multiplier", 1.0, 2.0, 1.0))
#: R-118: an optional second attempt; with ``retry_depth`` 0 the module is the 11-parameter one bit for bit
RETRY_PARAMS = (("retry_gap", 30.0, 150.0, 60.0), ("retry_depth", 0.0, 1.0, 0.0))
ALL_PARAMS = PARAMS + RETRY_PARAMS
NAMES = tuple(p[0] for p in ALL_PARAMS)
LOW, HIGH, INIT = (np.array([p[i] for p in ALL_PARAMS]) for i in (1, 2, 3))
DIM, DIM_RETRY = len(PARAMS), len(ALL_PARAMS)
SCRIPT_X = ((INIT - LOW) / (HIGH - LOW))[:DIM]
RETRY_INIT_TAIL = ((INIT - LOW) / (HIGH - LOW))[DIM:]
RETRY_DEADLINE = 300
WINDOW, BOOST_STEPS = 30, 60
SCRIPT_TOLERANCE = 1e-5
POLICIES = ("reference", "r106", "scripted", "trained")
CONTRASTS = (("trained", "r106"), ("scripted", "r106"), ("trained", "scripted"), ("trained", "reference"),
             ("scripted", "reference"), ("r106", "reference"))


def to_params(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    d = x.shape[-1]
    return LOW[:d] + np.clip(x, 0.0, 1.0) * (HIGH[:d] - LOW[:d])


def x_record(x) -> dict:
    xs = [float(v) for v in np.asarray(x, dtype=np.float64).reshape(-1)]
    if len(xs) not in (DIM, DIM_RETRY):
        raise ValueError(f"the module has {DIM} parameters ({DIM_RETRY} with a second attempt)")
    return {"x": xs, "params": dict(zip(NAMES, [float(v) for v in to_params(xs)])),
            "sha256": sha256_text(json.dumps(xs, separators=(",", ":")))}


def blend(u, n_in, n_out):
    """``beta(u)`` for tensors of equal shape (``u`` may be negative: 0 before the decision)."""
    up = ((u + 1.0) / n_in).clamp(max=1.0)
    down = 1.0 - (u - n_in) / n_out
    zero = torch.zeros_like(up)
    return torch.where(u < 0, zero, torch.where(u < n_in, up, torch.where(u < n_in + n_out, down, zero)))


class RecoveryController:
    """The frozen root controller plus the re-capture module, behind the runner's controller interface.

    ``x`` is the normalised parameter vector, one row or one row per environment. ``enabled=False`` computes
    everything and never triggers (the smoke's identity check)."""

    def __init__(self, name: str, root, x, centre_tolerance: float, enabled: bool = True):
        self.root, self.name, self.enabled = root, name, bool(enabled)
        self.tolerance = float(centre_tolerance)
        self._x = np.asarray(x, dtype=np.float64)
        self.params = {"root_theta_sha256": root.params["theta_sha256"], "enabled": self.enabled,
                       **({"module": x_record(self._x)} if self._x.ndim == 1 else {})}
        self.gain = None

    def set_x(self, x) -> None:
        """Per-environment (or shared) parameters as tensors on the controller's device."""
        dev, dt = self.device, self.dtype
        p = torch.as_tensor(to_params(np.atleast_2d(x)), device=dev, dtype=dt)
        if p.shape[0] == 1:
            p = p.expand(self.n, p.shape[1])
        self.threshold, self.T = p[:, 0], torch.round(p[:, 1]).long()
        self.n_in, self.n_out = p[:, 2], p[:, 3]
        self.depth = p[:, 4:10].index_select(1, torch.as_tensor(sr.GROUP_OF, device=dev))
        self.mult = p[:, 10]
        if p.shape[1] == DIM_RETRY:                           # R-118: the second attempt
            gap, self.kappa = torch.round(p[:, 11]).long(), p[:, 12]
        else:
            gap, self.kappa = torch.zeros_like(self.T), torch.zeros_like(self.mult)
        self.T2 = self.T + torch.round(self.n_in + self.n_out).long() + gap

    def reset(self, ctx):
        self.root.reset(ctx)
        self.device, self.dtype, self.n = self.root.ref.span.device, self.root.ref.span.dtype, int(self.root.ref.span.shape[0])
        self.neutral = ctx.neutral_action.to(self.device, self.dtype)
        if self.gain is None:                                # the incumbent's fixed stroke gain, per environment
            policy = self.root.ref.inner.policy
            self.gain = ra.PerEnvGain(policy._stroke_gain.constant, self.n)
            policy._stroke_gain = self.gain
        self.gain.scale[:] = 1.0
        self.step = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.hist = torch.zeros(self.n, WINDOW, device=self.device, dtype=self.dtype)
        self.elig = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.elig2 = torch.zeros(self.n, dtype=torch.bool, device=self.device)
        self.triggered = self.triggered2 = 0
        self.last_root = None
        self.set_x(self._x)

    def reset_envs(self, mask) -> None:
        self.root.reset_envs(mask)
        m = mask.to(self.device)
        self.step = torch.where(m, torch.zeros_like(self.step), self.step)
        self.elig = self.elig & ~m
        self.elig2 = self.elig2 & ~m
        self.hist = torch.where(m.unsqueeze(1), torch.zeros_like(self.hist), self.hist)

    def signals(self, obs) -> tuple:
        o = torch.nan_to_num(obs.to(self.device, self.dtype))
        sl = cases.obs_slices()
        return sr.axial_rate(o).abs()[:, 0], torch.norm(o[:, sl["obj_pos"]] - o[:, sl["in_hand_pos"]], dim=1)

    def __call__(self, obs, t, ctx):
        rate, centre = self.signals(obs)
        self.hist.scatter_(1, (self.step % WINDOW).unsqueeze(1), rate.unsqueeze(1))
        deciding = self.step == self.T
        if self.enabled:
            new = deciding & (centre < self.tolerance) & (self.hist.mean(dim=1) < self.threshold)
            self.elig = self.elig | new
            self.triggered += int(new.sum())
        u = (self.step - self.T).to(self.dtype)
        beta = torch.where(self.elig, blend(u, self.n_in, self.n_out), torch.zeros_like(u))
        end = self.n_in + self.n_out
        boost = self.elig & (u >= end) & (u < end + BOOST_STEPS)
        if self.enabled:                                     # one retry: after a real first attempt, still slow and centred
            new2 = ((self.step == self.T2) & self.elig & (self.kappa > 0) & (centre < self.tolerance)
                    & (self.hist.mean(dim=1) < self.threshold) & (self.T2.to(self.dtype) + end <= RETRY_DEADLINE))
            self.elig2 = self.elig2 | new2
            self.triggered2 += int(new2.sum())
        u2 = (self.step - self.T2).to(self.dtype)
        beta2 = torch.where(self.elig2, blend(u2, self.n_in, self.n_out), torch.zeros_like(u2))
        beta = beta + beta2 * self.kappa                     # kappa = 0 leaves beta unchanged exactly
        boost = boost | (self.elig2 & (u2 >= end) & (u2 < end + BOOST_STEPS))
        self.gain.scale[:] = torch.where(boost, self.mult, torch.ones_like(self.mult)).unsqueeze(1).cpu()
        a_root = self.root(obs, t, ctx)
        self.last_root = a_root
        w = beta.unsqueeze(1) * self.depth
        sent = torch.where(w > 0, ((1.0 - w) * a_root + w * self.neutral).clamp(-1.0, 1.0), a_root)
        self.step = self.step + 1
        return sent

    def extra_trace(self) -> dict:
        self.params["recovery"] = {"triggered_environment_episodes": int(self.triggered),
                                   "second_attempts": int(self.triggered2)}
        return self.root.extra_trace()


# ---------------------------------------------------------------------------------------------------------------------
#  Construction, manifests, evaluation controllers
# ---------------------------------------------------------------------------------------------------------------------


def build(protocol_cfg: dict, name: str, x, enabled: bool = True, inner=None) -> RecoveryController:
    root = sr.build_controller(protocol_cfg, f"{name}:root", ra.finalist_theta(protocol_cfg), ra.normalization(protocol_cfg), inner=inner)
    return RecoveryController(name, root, x, float(protocol_cfg["centre_tolerance"]), enabled)


def confirmation_manifest(cfg: dict, seed) -> str:
    return str(cfg["confirmation_manifest_template"]).format(seed=int(seed))


def build_manifests(protocol_cfg: dict) -> list[str]:
    import glob

    from lares.eval.manifest import EvaluationManifest, assert_disjoint

    mine = {confirmation_manifest(protocol_cfg, s): cases.build_manifest(3, int(s), int(protocol_cfg["confirmation_cases"]), label="recovery_confirm")
            for s in protocol_cfg["confirmation_seeds"]}
    others = [EvaluationManifest.load(p) for p in sorted(glob.glob("config/manifests/pen_spin_stage3_*.yaml")) if p not in mine]
    assert_disjoint(list(mine.values()) + others)
    return [p for p, m in mine.items() if cases.freeze_manifest(p, m)]


def make_controller(name: str, protocol_cfg: dict, out_dir: str | None = None):
    if name == "reference":
        return sr.build_reference(protocol_cfg, "reference", resettable=False)
    if name == "r106":
        return sr.build_controller(protocol_cfg, "r106", ra.finalist_theta(protocol_cfg), ra.normalization(protocol_cfg))
    if name == "scripted":
        return build(protocol_cfg, "scripted", SCRIPT_X)
    if name == "trained":
        with open(os.path.join(out_dir or str(protocol_cfg["log_dir"]), "finalist.json"), encoding="utf-8") as f:
            entry = json.load(f)
        if x_record(entry["x"])["sha256"] != entry["sha256"]:
            raise SystemExit("the stored module does not match its hash")
        return build(protocol_cfg, "trained", entry["x"])
    raise ValueError(f"unknown controller {name!r}")


# ---------------------------------------------------------------------------------------------------------------------
#  ARS in normalised coordinates
# ---------------------------------------------------------------------------------------------------------------------


def ars_candidates(x, delta, nu: float) -> np.ndarray:
    return np.clip(np.concatenate([x + nu * delta, x - nu * delta]), 0.0, 1.0)


def ars_step(x, delta, scores, alpha: float) -> tuple:
    r = np.asarray(scores, dtype=float)
    n = len(delta)
    rp, rm = r[:n], r[n:]
    sigma = float(r.std())
    if sigma > 1e-8:
        x = np.clip(x + alpha / (n * sigma) * ((rp - rm)[:, None] * delta).sum(0), 0.0, 1.0)
    return x, sigma


# ---------------------------------------------------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------------------------------------------------


def _load(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _dump(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)


def run_child(request: dict, stem: str) -> dict:
    if os.path.exists(request["response_path"]):
        return _load(request["response_path"])
    marker = request["response_path"] + ".started"
    if os.path.exists(marker):
        raise SystemExit(f"{marker}: this child already started and left no result; it is not relaunched")
    _dump(stem + ".request.json", request)
    _dump(marker, {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    with open(stem + ".log", "w", encoding="utf-8") as log:
        proc = subprocess.run([sys.executable, "-u", "-m", "lares.envs.isaac_pen_spin.recovery_module", stem + ".request.json"],
                              stdout=log, stderr=subprocess.STDOUT, timeout=4 * 3600)
    if proc.returncode != 0 or not os.path.exists(request["response_path"]):
        raise RuntimeError(f"child failed (exit {proc.returncode}); see {stem}.log")
    os.remove(marker)
    return _load(request["response_path"])


def contrasts(rows: dict, threshold: float, drop_margin: float) -> dict:
    from lares.envs.isaac_pen_spin.incumbent_residual import confirmation_contrast

    return {f"{a}_minus_{b}": confirmation_contrast({m: (r[a], r[b]) for m, r in rows.items()}, threshold, drop_margin)
            for a, b in CONTRASTS}


def run(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    if os.path.normpath(out_dir) != os.path.normpath(str(protocol_cfg["log_dir"])):
        raise SystemExit("evaluation cells read the finalist from the protocol's own log_dir; do not override it")
    os.makedirs(out_dir, exist_ok=True)
    descriptor = sr.check_reference(protocol_cfg)
    ra.finalist_theta(protocol_cfg)
    ra.normalization(protocol_cfg)
    cfg_sha = sr.config_sha256(protocol_cfg)
    freeze_json(os.path.join(out_dir, "frozen_config.json"),
                {"config_sha256": cfg_sha, "config": sr.frozen_config(protocol_cfg), "reference": descriptor,
                 "parameters": [list(p) for p in PARAMS], "script": x_record(SCRIPT_X)}, "frozen configuration")
    seeds = [int(s) for s in protocol_cfg["confirmation_seeds"]]
    mpaths = {s: cases.require_frozen(confirmation_manifest(protocol_cfg, s)) for s in seeds}
    ceiling = int(protocol_cfg["start_ceiling"])
    reserve = 2 * len(POLICIES) * len(seeds) * int(protocol_cfg["confirmation_cases"])
    base = {"config_path": os.path.abspath(config_path), "study": STUDY, "config_sha256": cfg_sha}

    smoke = run_child({**base, "mode": "smoke", "response_path": os.path.join(out_dir, "smoke.json")}, os.path.join(out_dir, "smoke"))
    summary = {"study": STUDY, "config_sha256": cfg_sha, "smoke": smoke, "smoke_gate": "pass" if smoke["passes"] else "fail"}
    if not smoke["passes"]:
        summary["note"] = "the smoke gate failed: no training or confirmation ran"
        _dump(os.path.join(out_dir, "summary.json"), summary)
        return summary
    train = run_child({**base, "mode": "train", "start_cap": ceiling - reserve - int(smoke["starts"]),
                       "response_path": os.path.join(out_dir, "train.json")}, os.path.join(out_dir, "train"))
    summary["training"] = train
    if not train["complete"]:
        summary["incomplete"] = "the start ceiling was reached during training: no finalist, no confirmation"
        _dump(os.path.join(out_dir, "summary.json"), summary)
        return summary
    final = x_record(train["final_x"])
    freeze_json(os.path.join(out_dir, "finalist.json"), {**final, "rule": "the final ARS mean; no selection"}, "finalist")

    rows = {}
    for s in seeds:
        sub = os.path.join(out_dir, "confirmation", f"s{s}")
        os.makedirs(sub, exist_ok=True)
        rows[f"s{s}"] = {}
        for name in POLICIES:
            cell = os.path.join(sub, f"{name}_cell.json")
            study_cell(protocol_cfg, config_path, mpaths[s], name, cell, record_obs=False)
            if name == "trained" and _load(cell)["controller_params"]["module"]["sha256"] != final["sha256"]:
                raise SystemExit(f"{cell}: the evaluated module is not the frozen finalist")
            rows[f"s{s}"][name] = load_cell(cell)[1]
    c = contrasts(rows, float(protocol_cfg["threshold"]), float(protocol_cfg["drop_margin"]))
    head = c["trained_minus_r106"]["pooled"]["success"]
    triggered = {name: int(sum(_load(os.path.join(out_dir, "confirmation", f"s{s}", f"{name}_cell.json"))["controller_params"]["recovery"]["triggered_environment_episodes"]
                               for s in seeds)) for name in ("scripted", "trained")}
    summary.update({
        "paired_success_difference": head["mean"], "paired_success_difference_ci95": [head["low"], head["high"]],
        "contrasts": c,
        "successes": {n: int(sum(r["sustained"] for m in rows.values() for r in m[n])) for n in POLICIES},
        "drops": {n: int(sum(not r["completed"] for m in rows.values() for r in m[n])) for n in POLICIES},
        "triggered_environment_episodes": triggered, "finalist": final, "script": x_record(SCRIPT_X),
        "accounting": {"smoke_starts": smoke["starts"], "training_starts": train["starts"],
                       "confirmation_charged_starts": reserve, "charged_total": smoke["starts"] + train["starts"] + reserve,
                       "ceiling": ceiling, "llm_calls": 0},
        "limitation": "one ARS trajectory; D-080's authority was measured in-process, this confirmation is the fresh-process evaluator",
    })
    _dump(os.path.join(out_dir, "summary.json"), summary)
    return summary


# ---------------------------------------------------------------------------------------------------------------------
#  Isaac child
# ---------------------------------------------------------------------------------------------------------------------


def _rollout(env, ctx, ctrl, vecs, x_env, *, plain=None, script_check: dict | None = None) -> dict:
    from lares.envs.isaac_pen_spin.runner import _write_placements
    from lares.envs.isaac_pen_spin.success_ars import first_episode_scores

    spec = cases.STAGES[3]
    n, dev = env.num_envs, env.device
    env.reset()
    _write_placements(env, vecs)
    obs = env._get_observations()["policy"]
    ctrl.reset(ctx)
    ctrl.set_x(x_env)
    if plain is not None:
        plain.reset(ctx)
    T = int(env.max_episode_length)
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    completed = torch.zeros(n, dtype=torch.bool, device=dev)
    length = torch.zeros(n, dtype=torch.long, device=dev)
    pred = torch.zeros((T, n), dtype=torch.bool, device=dev)
    starts, mism, compared, script_bad, elig_bad = n, 0, 0, 0, None
    env_rate = torch.zeros(n, device=dev)
    env_centre = torch.zeros(n, device=dev)
    for t in range(T):
        action = ctrl(obs, t, ctx)
        if plain is not None:
            a_plain = plain(obs, t, ctx).to(action.device)
            mism += int((action[alive] != a_plain[alive]).any(dim=1).sum())
            compared += int(alive.sum())
        if script_check is not None:
            # the D-080 script on the same observations: eligibility from the env's own values, the fixed 20/40 blend
            if t == int(script_check["t0"]):
                want = ra.eligible(alive, env_centre, env_rate / int(script_check["stuck_window"]), script_check)
                elig_bad = int((want != ctrl.elig)[alive].sum())
            b = ra.beta(t, script_check)
            expect = torch.where(ctrl.elig.unsqueeze(1), ((1.0 - b) * ctrl.last_root + b * ctrl.neutral).clamp(-1.0, 1.0), ctrl.last_root) \
                if b > 0 else ctrl.last_root
            # the module's schedule is float32 tensor arithmetic, the script's a Python float: equal to rounding
            script_bad += int(((action[alive] - expect[alive]).abs() > SCRIPT_TOLERANCE).any(dim=1).sum())
        obs_d, _, terminated, truncated, _ = env.step(action)
        obs = obs_d["policy"]
        done = terminated | truncated
        ctrl.reset_envs(done)
        starts += int(done.sum())
        if script_check is not None:
            t0, look = int(script_check["t0"]), int(script_check["stuck_window"])
            if t0 - look <= t < t0:
                env_rate += env.spin_rate.abs()
            if t == t0 - 1:
                env_centre = env.center_dist.clone()
        pred[t] = alive & (env.center_dist < spec.success_center_tolerance) & (env.spin_rate.abs() > spec.success_spin_rate)
        length = torch.where(alive, length + 1, length)
        completed = torch.where(alive & done, truncated & ~terminated, completed)
        alive = alive & ~done
        if not bool(alive.any()):
            break
    sc = first_episode_scores(pred.cpu().numpy(), length.cpu().numpy(), completed.cpu().numpy())
    return {"sustained": sc["sustained"], "completed": completed.cpu().numpy(), "starts": starts, "env_steps": int((t + 1) * n),
            "identity": {"compared": compared, "mismatches": mism}, "script_mismatches": script_bad,
            "eligibility_mismatches": elig_bad, "triggered": int(ctrl.triggered), "triggered2": int(ctrl.triggered2)}


def _child(req: dict, cfg: dict, device: str) -> dict:
    from lares.envs.isaac_pen_spin import success_ars as sa
    from lares.envs.isaac_pen_spin.runner import ControllerContext
    from lares.envs.isaac_pen_spin.symbolic_reward_child import make_training_env

    lay = {"slots": 4, "per": int(cfg["cases_per_slot"]), "directions": int(cfg["ars"]["directions"])}
    n = lay["slots"] * lay["per"]
    seed = int(cfg["smoke_seed"] if req["mode"].startswith("smoke") else cfg["training_seed"])
    env = make_training_env(n, device, seed)
    try:
        ctx = ControllerContext(env)
        if sr.executed_joint_order(ctx) != list(sr.JOINTS):
            raise SystemExit("the simulator's joint order is not the actor's")
        if req["mode"] == "smoke":
            out = {"mode": "smoke", "gates": {}}
            vecs = sa.tile(cases.placements(seed, lay["per"]), lay["slots"])
            t0 = time.time()
            off = build(cfg, "smoke_off", SCRIPT_X, enabled=False)
            plain = sr.build_controller(cfg, "smoke_r106", ra.finalist_theta(cfg), ra.normalization(cfg))
            a = _rollout(env, ctx, off, vecs, SCRIPT_X, plain=plain)
            out["gates"]["no_trigger_identity"] = {**a["identity"], "passes": a["identity"]["mismatches"] == 0 and a["identity"]["compared"] > 0 and a["triggered"] == 0}
            sus = a["sustained"].reshape(lay["slots"], lay["per"])
            out["aa"] = sa.aa_statistics(sus)
            out["gates"]["scorer_live"] = {"root_successes": int(sus.sum()), "passes": bool(sus.any())}
            on = build(cfg, "smoke_script", SCRIPT_X)
            check = {"t0": 120, "window": 60, "ramp_in": 20, "stuck_window": WINDOW, "stuck_rate": 3.0, "centre_tolerance": float(cfg["centre_tolerance"])}
            b = _rollout(env, ctx, on, vecs, SCRIPT_X, script_check=check)
            out["gates"]["script_equivalence"] = {"action_mismatch_env_steps": b["script_mismatches"], "eligibility_mismatches": b["eligibility_mismatches"],
                                                  "passes": b["script_mismatches"] == 0 and b["eligibility_mismatches"] is not None
                                                  and b["eligibility_mismatches"] <= int(cfg["smoke_eligibility_mismatch_max"])}
            half = torch.arange(n, device=device) % 2 == 0
            on.step[:] = 7
            on.elig[:] = True
            on.hist[:] = 1.0
            on.reset_envs(half)
            ok = bool((on.step[half] == 0).all() and (~on.elig[half]).all() and (on.hist[half] == 0).all()
                      and (on.step[~half] == 7).all() and on.elig[~half].all() and (on.hist[~half] == 1.0).all())
            out["gates"]["state_reset"] = {"passes": ok}
            out["scripted"] = {"triggered_fraction": b["triggered"] / n, "success": float(b["sustained"].mean()),
                               "drops": int((~b["completed"]).sum()), "root_success_same_placements": float(a["sustained"].mean()),
                               "root_drops": int((~a["completed"]).sum())}
            out["starts"] = a["starts"] + b["starts"]
            out["throughput"] = {"transitions_per_s": (a["env_steps"] + b["env_steps"]) / (time.time() - t0)}
            out["passes"] = all(g["passes"] for g in out["gates"].values())
            return out
        if req["mode"] == "smoke_retry":
            return _smoke_retry(env, ctx, cfg, lay, seed, device)
        x = np.asarray(req.get("init_x", SCRIPT_X), dtype=float).copy()      # R-118 trains from a given point
        dim, draw_dim = len(x), int(req.get("direction_dim", len(x)))
        ctrl = build(cfg, "train", x)
        rng = np.random.default_rng(seed + 2000)
        ars = cfg["ars"]
        cap, starts, transitions, updates, complete, note = int(req["start_cap"]), 0, 0, [], True, None
        t0 = time.time()
        for u in range(int(ars["iterations"])):
            if starts + 4 * n > cap:
                complete, note = False, f"update {u + 1}: {starts} starts, cap {cap}"
                break
            delta = rng.standard_normal((lay["directions"], draw_dim))[:, :dim]   # arms share the leading coordinates' draws
            cand = ars_candidates(x, delta, float(ars["nu"]))
            raw = np.concatenate([x + float(ars["nu"]) * delta, x - float(ars["nu"]) * delta])
            vecs = sa.tile(cases.placements(seed, lay["per"] * int(ars["iterations"]))[lay["per"] * u:lay["per"] * (u + 1)], lay["slots"])
            slot_scores, trig, trig2, drops = [], [], [], []
            for r in range(2):
                rc = sa.rollout_candidates(cand, r, lay)
                res = _rollout(env, ctx, ctrl, vecs, sa.slot_thetas(rc, lay["per"]))
                starts += res["starts"]
                transitions += res["env_steps"]
                slot_scores.append(res["sustained"].reshape(lay["slots"], lay["per"]).mean(axis=1))
                drops.append((~res["completed"]).reshape(lay["slots"], lay["per"]).mean(axis=1))
                trig.append(res["triggered"] / n)
                trig2.append(res["triggered2"] / n)
            scores = sa.ars_scores(slot_scores, lay)
            x, sigma = ars_step(x, delta, scores, float(ars["alpha"]))
            d = lay["directions"]
            updates.append({"update": u + 1, "plus": [float(v) for v in scores[:d]], "minus": [float(v) for v in scores[d:]],
                            "mean_candidate_success": float(scores.mean()), "sigma_r": sigma, "triggered_fraction": trig,
                            "slot_drop_rates": [[float(v) for v in dd] for dd in drops],
                            "clipped_candidate_coordinates": int((raw != cand).sum()),
                            "second_attempt_fraction": trig2,
                            "params": dict(zip(NAMES, [float(v) for v in to_params(x)])), "starts_so_far": starts})
            print("update", json.dumps(updates[-1]), flush=True)
        return {"mode": "train", "seed": seed, "complete": complete, "note": note, "updates": updates, "starts": starts,
                "transitions": transitions, "wall_time_s": time.time() - t0, "final_x": [float(v) for v in x] if complete else None}
    finally:
        env.close()


def _smoke_retry(env, ctx, cfg: dict, lay: dict, seed: int, device: str) -> dict:
    """R-118's gate: RETRY at kappa = 0 is the parent module bit for bit; at kappa = 1 second attempts fire."""
    from lares.envs.isaac_pen_spin import success_ars as sa

    n = lay["slots"] * lay["per"]
    parent = np.asarray(_load(cfg["module_finalist"])["x"], dtype=float)
    x_off = np.concatenate([parent, RETRY_INIT_TAIL])
    x_on = np.concatenate([parent, [RETRY_INIT_TAIL[0], 1.0]])
    vecs = sa.tile(cases.placements(seed, lay["per"]), lay["slots"])
    out = {"mode": "smoke_retry", "gates": {}}
    t0 = time.time()
    a = _rollout(env, ctx, build(cfg, "retry_off", x_off), vecs, x_off, plain=build(cfg, "parent", parent))
    out["gates"]["inheritance_identity"] = {**a["identity"], "second_attempts": a["triggered2"],
                                            "passes": a["identity"]["mismatches"] == 0 and a["identity"]["compared"] > 0 and a["triggered2"] == 0}
    sus = a["sustained"].reshape(lay["slots"], lay["per"])
    out["aa"] = sa.aa_statistics(sus)
    out["gates"]["scorer_live"] = {"parent_successes": int(sus.sum()), "passes": bool(sus.any())}
    on = build(cfg, "retry_on", x_on)
    b = _rollout(env, ctx, on, vecs, x_on)
    earliest = int((on.T2 - on.T - torch.round(on.n_in + on.n_out).long()).min())
    out["gates"]["second_attempt"] = {"first_attempts": b["triggered"], "second_attempts": b["triggered2"], "steps_between_attempts_min": earliest,
                                      "passes": b["triggered2"] > 0 and b["triggered2"] <= b["triggered"] and earliest >= int(RETRY_PARAMS[0][1])}
    half = torch.arange(n, device=device) % 2 == 0
    on.step[:] = 7
    on.elig[:] = True
    on.elig2[:] = True
    on.reset_envs(half)
    out["gates"]["state_reset"] = {"passes": bool((on.step[half] == 0).all() and (~on.elig[half]).all() and (~on.elig2[half]).all()
                                                  and on.elig[~half].all() and on.elig2[~half].all() and (on.step[~half] == 7).all())}
    out["retry_at_full_depth"] = {"success": float(b["sustained"].mean()), "drops": int((~b["completed"]).sum()),
                                  "parent_success_same_placements": float(a["sustained"].mean()), "parent_drops": int((~a["completed"]).sum())}
    out["starts"] = a["starts"] + b["starts"]
    out["throughput"] = {"transitions_per_s": (a["env_steps"] + b["env_steps"]) / (time.time() - t0)}
    out["passes"] = all(g["passes"] for g in out["gates"].values())
    return out


def child_main(request_path: str) -> None:
    from lares.envs.isaac_lab_adapter import _ensure_isaaclab_on_path, close_isaac_app, launch_isaac_app
    from lares.utils.cli import load_yaml

    req = _load(request_path)
    full = load_yaml(req["config_path"])
    cfg = full["research_protocols"][req["study"]]
    device = str(full["device"])
    _ensure_isaaclab_on_path(str(full["isaaclab_root"]))
    launch_isaac_app(headless=True, device=device)
    code = 0
    try:
        if sr.config_sha256(cfg) != req["config_sha256"]:
            raise SystemExit("the config block changed since the driver froze it")
        out = _child(req, cfg, device)
        _dump(req["response_path"], out)
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
