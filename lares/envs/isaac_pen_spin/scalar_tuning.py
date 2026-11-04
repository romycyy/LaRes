"""R-058: budget-matched CMA-ES versus random search on the frozen R-046 learner's spin and centering scalars.

The parent is the R-046 block-3 promoted controller (``config/pen_spin/r046_block3_incumbent``, hashes checked). Only
the twelve whitelisted scalars move (six spin, six centering; ``WHITELIST``), each inside the parent's own declared
``get_param_ranges`` bound, in bound-normalised coordinates on [0, 1]; the gait, grasp pose, observation processing and
execution (``clamp(mean, -1, 1)``) are unchanged.

Per block (one frozen 80-case Stage-3 manifest: screen 0-9, promotion 10-39, fresh 40-79), two arms with the same
budget of 8 generations x 8 candidates on the same 10 screen cases:

* ``cma``: one pycma ``CMAEvolutionStrategy`` (x0 = parent, sigma0 0.2, bounds [0, 1] under pycma's own
  ``BoundTransform``, popsize 8, seeded per block), told the frozen lexicographic screen rank each generation.
* ``random``: a frozen copy of the same initial sampling law: every generation a *fresh* pycma instance with identical
  options under its own seed is asked once and never told, so the arms differ only in adaptation.

Both arms screen each generation in one fresh Isaac process holding 8 candidates x 10 cases = 80 environments
(candidate-major), the same batch shape (D-035, D-037). Each arm's winner is frozen before any promotion or fresh
episode. The two winners and the parent are then each scored in their own process on the 30 promotion cases (strict
paired promotion rule) and on the 40 fresh cases (confirmation only; never selects anything), observations recorded for
the saturation diagnostics (recorded in-play steps only: the runner records nothing after a drop).
"""

from __future__ import annotations

import json
import os

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin.learner_recovery import check_learner, make_learner
from lares.envs.isaac_pen_spin.protocols import arm_summary, child_cell, failure_split
from lares.eval.manifest import EpisodeCase, EvaluationManifest

STUDY = "r058_scalar_cma_vs_random"
STAGE = 3
ARMS = ("cma", "random")
CONTROLLERS = ("cma_winner", "random_winner", "parent")
ROLES = ("screen", "promotion", "fresh")
WHITELIST = {
    "spin": ("stroke_amp", "lift_amp", "spin_target", "speed_gain", "phase_lead", "stance_sharpness"),
    "centering": ("center_gain", "center_length", "safe_radius", "position_sharpness", "tilt_gain", "swing_damping"),
}
RANKING = "sustained successes desc, drops asc, mean tail-hit fraction desc, candidate index asc"
SATURATED_ACTION = 0.99
SATURATED_DRIVE = 0.99
FEEDBACK_ACTIVATION = 0.25  # the unsaturated in-play drive fraction must rise by at least this, interval above zero


# ---------------------------------------------------------------------------------------------------------------------
#  Parameter space
# ---------------------------------------------------------------------------------------------------------------------


def whitelist_names(protocol_cfg: dict | None = None) -> list[str]:
    wl = (protocol_cfg or {}).get("whitelist") or WHITELIST
    return list(wl["spin"]) + list(wl["centering"])


class ParameterSpace:
    """Named scalars with bounds and the parent's values; ``normalise``/``denormalise`` map to and from [0, 1]."""

    def __init__(self, names, lo, hi, parent):
        self.names = list(names)
        self.lo = np.asarray(lo, dtype=np.float64)
        self.hi = np.asarray(hi, dtype=np.float64)
        self.parent = np.asarray(parent, dtype=np.float64)
        if not (len(self.names) == len(self.lo) == len(self.hi) == len(self.parent)):
            raise ValueError("names, bounds and parent values differ in length")
        if np.any(self.hi <= self.lo):
            raise ValueError("every bound must have hi > lo")
        if np.any(self.parent < self.lo) or np.any(self.parent > self.hi):
            raise ValueError("the parent's values must lie inside their declared bounds")

    @property
    def dim(self) -> int:
        return len(self.names)

    def normalise(self, real) -> np.ndarray:
        return (np.asarray(real, dtype=np.float64) - self.lo) / (self.hi - self.lo)

    def denormalise(self, z) -> np.ndarray:
        return self.lo + np.asarray(z, dtype=np.float64) * (self.hi - self.lo)

    def params(self, z) -> dict:
        """``{name: real value}`` for one normalised point."""
        return {n: float(v) for n, v in zip(self.names, self.denormalise(z))}

    def to_dict(self) -> dict:
        return {"names": self.names, "lo": self.lo.tolist(), "hi": self.hi.tolist(), "parent": self.parent.tolist(),
                "parent_normalised": self.normalise(self.parent).tolist()}


def parameter_space(protocol_cfg: dict) -> ParameterSpace:
    """The whitelist's bounds and values read from the frozen parent itself (its ``get_param_ranges`` and state)."""
    learner = make_learner(protocol_cfg)
    ranges = learner.policy.get_param_ranges()
    state = learner.policy.state_dict()
    names = whitelist_names(protocol_cfg)
    for n in names:
        if n not in ranges or n not in state or state[n].numel() != 1:
            raise ValueError(f"whitelisted parameter {n!r} is not a declared scalar of the parent")
    return ParameterSpace(names, [ranges[n][0] for n in names], [ranges[n][1] for n in names],
                          [float(state[n].item()) for n in names])


# ---------------------------------------------------------------------------------------------------------------------
#  Samplers (pycma)
# ---------------------------------------------------------------------------------------------------------------------


def cma_options(protocol_cfg: dict) -> dict:
    o = protocol_cfg["optimizer"]
    return {"sigma0": float(o["sigma0"]), "popsize": int(o["popsize"]), "generations": int(o["generations"]),
            "bounds": [float(o["bounds"][0]), float(o["bounds"][1])], "random_seed_base": int(o["random_seed_base"])}


def _new_es(space: ParameterSpace, opts: dict, seed: int):
    import cma

    x0 = space.normalise(space.parent).tolist()
    # Instance-local randomness: pycma's ``seed`` option seeds the *global* numpy RNG at construction, so two
    # instances would otherwise interleave their draws; ``randn`` makes each instance a function of its own seed.
    return cma.CMAEvolutionStrategy(x0, opts["sigma0"], {"bounds": list(opts["bounds"]), "popsize": opts["popsize"],
                                                          "seed": int(seed), "verbose": -9,
                                                          "randn": np.random.RandomState(int(seed)).randn})


class CMASampler:
    """The adaptive arm: one pycma instance, told the screen rank of every generation."""

    name = "cma"

    def __init__(self, space: ParameterSpace, opts: dict, seed: int):
        self.es = _new_es(space, opts, seed)
        self.seed = int(seed)
        self._pending = None

    def ask(self) -> np.ndarray:
        self._pending = [np.asarray(x, dtype=np.float64) for x in self.es.ask()]
        return np.stack(self._pending)

    def tell(self, ranks) -> None:
        self.es.tell(self._pending, [float(r) for r in ranks])
        self._pending = None

    def state(self) -> dict:
        return {"sigma": float(self.es.sigma), "mean_genotype": [float(v) for v in self.es.mean],
                "countiter": int(self.es.countiter)}


class FreshInstanceSampler:
    """The independent arm: each generation is a fresh pycma instance with the same options, asked once, never told."""

    name = "random"

    def __init__(self, space: ParameterSpace, opts: dict, seed: int):
        self.space, self.opts = space, opts
        self.base = int(opts["random_seed_base"]) + int(seed) * 100
        self.generation = 0

    def ask(self) -> np.ndarray:
        es = _new_es(self.space, self.opts, self.base + self.generation + 1)
        xs = np.stack([np.asarray(x, dtype=np.float64) for x in es.ask()])
        self.generation += 1
        return xs

    def tell(self, ranks) -> None:  # never updates anything: outcome-independent by construction
        return None

    def state(self) -> dict:
        return {"generation": self.generation, "seed_rule": "random_seed_base + 100 * block_seed + generation + 1"}


def make_sampler(arm: str, space: ParameterSpace, opts: dict, seed: int):
    if arm == "cma":
        return CMASampler(space, opts, seed)
    if arm == "random":
        return FreshInstanceSampler(space, opts, seed)
    raise ValueError(f"unknown arm {arm!r}; one of {ARMS}")


# ---------------------------------------------------------------------------------------------------------------------
#  Screen ranking
# ---------------------------------------------------------------------------------------------------------------------


def screen_metrics(rows: list[dict]) -> dict:
    return {"cases": len(rows), "successes": int(sum(r["sustained"] for r in rows)),
            "drops": int(sum(not r["completed"] for r in rows)),
            "tail_hit_fraction": float(np.mean([r["tail_hit_fraction"] for r in rows])) if rows else 0.0,
            "mean_abs_spin": float(np.mean([r["mean_abs_spin"] for r in rows if r["mean_abs_spin"] is not None]))
            if any(r["mean_abs_spin"] is not None for r in rows) else None}


def rank_key(metrics: dict, index: int) -> tuple:
    """Lower is better: ``RANKING``."""
    return (-int(metrics["successes"]), int(metrics["drops"]), -float(metrics["tail_hit_fraction"]), int(index))


def rank_candidates(metrics: list[dict]) -> list[int]:
    """``ranks[i]`` is candidate ``i``'s rank (0 = best) under ``RANKING``; the index breaks every tie."""
    order = sorted(range(len(metrics)), key=lambda i: rank_key(metrics[i], i))
    ranks = [0] * len(metrics)
    for r, i in enumerate(order):
        ranks[i] = r
    return ranks


def split_rows(rows: list[dict], tile: int) -> list[list[dict]]:
    """Candidate-major rows of a tiled screen cell back into one list per candidate."""
    if len(rows) % tile:
        raise ValueError(f"{len(rows)} rows do not tile into {tile} candidates")
    n = len(rows) // tile
    return [rows[k * n:(k + 1) * n] for k in range(tile)]


# ---------------------------------------------------------------------------------------------------------------------
#  Manifests and controllers
# ---------------------------------------------------------------------------------------------------------------------


def manifest_path(protocol_cfg: dict, seed: int) -> str:
    return protocol_cfg["manifest_pattern"].format(seed=int(seed))


def roles(protocol_cfg: dict) -> dict:
    return {k: (int(v[0]), int(v[1])) for k, v in protocol_cfg["roles"].items()}


def build_manifests(protocol_cfg: dict) -> list[str]:
    """One 80-case Stage-3 block manifest per seed, placement-disjoint across blocks; refuses to overwrite a different one."""
    n = int(protocol_cfg["cases"])
    specs = [(manifest_path(protocol_cfg, s), cases.build_manifest(STAGE, int(s), n, label="r058_block"), f"block_s{s}")
             for s in protocol_cfg["seeds"]]
    before = {p for p, _, _ in specs if os.path.exists(p)}
    cases.freeze_manifest_set(specs)
    return [p for p, _, _ in specs if p not in before]


def tiled_manifest(screen: EvaluationManifest, tile: int) -> EvaluationManifest:
    """The screen cases repeated ``tile`` times, candidate-major: rows ``k*n:(k+1)*n`` belong to candidate ``k``."""
    eps = [EpisodeCase(case_id=f"{c.case_id}:c{k}", task_id=c.task_id, task_hash=c.task_hash, reset_seed=c.reset_seed,
                       policy_seed=c.policy_seed) for k in range(int(tile)) for c in screen.episodes]
    return EvaluationManifest(manifest_id=f"{screen.manifest_id}:tiled{int(tile)}", environment=screen.environment,
                              pool=screen.pool, split=screen.split, episodes=eps)


def apply_params(policy, params: dict) -> None:
    """Overwrite the named scalar parameters in place; every other parameter keeps its frozen value."""
    import torch

    for name, value in params.items():
        p = getattr(policy, name)
        if not isinstance(p, torch.nn.Parameter) or p.numel() != 1:
            raise ValueError(f"{name!r} is not a scalar parameter of the policy")
        with torch.no_grad():
            p.copy_(torch.tensor(float(value), dtype=p.dtype))


class TiledCandidates:
    """``tile`` parameterisations of the parent, each driving its own contiguous block of rows."""

    name = "tiled_candidates"

    def __init__(self, learners, candidates: list[dict], label: str):
        if len(learners) != len(candidates):
            raise ValueError("one learner per candidate")
        self.learners = list(learners)
        self.tile = len(learners)
        for learner, params in zip(self.learners, candidates):
            apply_params(learner.policy, params)
        self.params = {"label": label, "tile": self.tile, "candidates": candidates,
                       "execution": "clamp(mean, -1, 1)"}

    def reset(self, ctx):
        for learner in self.learners:
            learner.reset(ctx)

    def __call__(self, obs, t, ctx):
        import torch

        if obs.shape[0] % self.tile:
            raise ValueError(f"{obs.shape[0]} rows do not tile into {self.tile} candidates")
        n = obs.shape[0] // self.tile
        return torch.cat([learner(obs[k * n:(k + 1) * n], t, ctx) for k, learner in enumerate(self.learners)], dim=0)


def write_candidates(path: str, label: str, candidates: list[dict], extra: dict | None = None) -> None:
    """Freeze a candidate file; an existing one must hold exactly the same candidates."""
    payload = {"label": label, "tile": len(candidates), "candidates": candidates, **(extra or {})}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
        same = old["tile"] == payload["tile"] and all(
            set(a) == set(b) and all(np.isclose(a[k], b[k], rtol=0, atol=1e-12) for k in a)
            for a, b in zip(old["candidates"], candidates))
        if not same:
            raise SystemExit(f"{path} exists and holds different candidates than this run generates")
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)


def make_controller(name: str, protocol_cfg: dict, out_dir: str):
    """``candidates:<path relative to out_dir>``: the parent with that file's parameterisations, tiled."""
    kind, _, rel = name.partition(":")
    if kind != "candidates" or not rel:
        raise ValueError(f"unknown controller {name!r}; expected candidates:<file>")
    with open(os.path.join(out_dir, rel), encoding="utf-8") as f:
        spec = json.load(f)
    learners = [make_learner(protocol_cfg) for _ in spec["candidates"]]
    return TiledCandidates(learners, spec["candidates"], spec["label"])


# ---------------------------------------------------------------------------------------------------------------------
#  Saturation diagnostics (offline, from recorded traces)
# ---------------------------------------------------------------------------------------------------------------------


def drive_from_obs(obs: np.ndarray, spin_target: float, speed_gain: float) -> np.ndarray:
    """The parent's spin drive ``0.25 + 0.75 tanh(speed_gain (spin_target - spin))`` recomputed from observations.

    ``spin`` is what the policy sees: ``5 * obj_angvel`` projected on the pen's long axis from ``obj_rot``.
    """
    sl = cases.obs_slices()
    obs = np.asarray(obs, dtype=np.float64)
    omega = 5.0 * obs[..., sl["obj_angvel"]]
    q = obs[..., sl["obj_rot"]]
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    axis = np.stack([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)], axis=-1)
    axis = axis / (np.linalg.norm(axis, axis=-1, keepdims=True) + 1e-8)
    spin = np.sum(omega * axis, axis=-1)
    return 0.25 + 0.75 * np.tanh(float(speed_gain) * (float(spin_target) - spin))


def saturation(trace, params: dict) -> dict:
    """Drive and executed-action saturation over recorded in-play steps, overall and per case."""
    obs, actions, length = np.asarray(trace["obs"]), np.asarray(trace["actions"]), np.asarray(trace["length"])
    n, T = obs.shape[0], obs.shape[1]
    drive_sat, act_sat_axis, per_case_unsat, steps = 0, np.zeros(actions.shape[-1]), [], 0
    for i in range(n):
        L = int(min(length[i], T))
        if L <= 0:
            per_case_unsat.append(None)
            continue
        d = drive_from_obs(obs[i, :L], params["spin_target"], params["speed_gain"])
        sat = d > SATURATED_DRIVE
        drive_sat += int(sat.sum())
        per_case_unsat.append(float(1.0 - sat.mean()))
        act_sat_axis += (np.abs(actions[i, :L]) >= SATURATED_ACTION).sum(axis=0)
        steps += L
    return {"in_play_steps": int(steps),
            "drive_saturated_fraction": float(drive_sat / steps) if steps else None,
            "drive_unsaturated_fraction_per_case": per_case_unsat,
            "action_saturated_fraction": float(act_sat_axis.sum() / (steps * actions.shape[-1])) if steps else None,
            "action_saturated_by_axis": (act_sat_axis / steps).tolist() if steps else None,
            "note": "recorded in-play steps only; the runner records nothing after a drop"}


# ---------------------------------------------------------------------------------------------------------------------
#  Aggregation
# ---------------------------------------------------------------------------------------------------------------------


def _paired(a_rows, b_rows, key):
    if [r["task_hash"] for r in a_rows] != [r["task_hash"] for r in b_rows]:
        raise ValueError("rows are not paired case for case")
    return [float(key(x)) - float(key(y)) for x, y in zip(a_rows, b_rows)]


def _success(r):
    return r["sustained"]


def _drop(r):
    return not r["completed"]


def contrast(a_rows, b_rows) -> dict:
    """Paired success and drop differences (a minus b) with the strict promotion verdict on success."""
    from lares.eval.stats import t_summary, win_loss_tie

    s = _paired(a_rows, b_rows, _success)
    d = _paired(a_rows, b_rows, _drop)
    ts = t_summary(s)
    return {"success": ts, "success_win_loss_tie": win_loss_tie(s), "drops": t_summary(d),
            "drop_counts": [int(sum(_drop(r) for r in a_rows)), int(sum(_drop(r) for r in b_rows))],
            "promotes": bool(ts["mean"] is not None and ts["mean"] > 0 and ts["low"] is not None and ts["low"] > 0)}


def feedback_contrast(a_sat: dict, b_sat: dict) -> dict:
    """Unsaturated in-play drive fraction, a minus b, paired by case; activated at +0.25 with the interval above zero."""
    from lares.eval.stats import t_summary

    pairs = [(x, y) for x, y in zip(a_sat["drive_unsaturated_fraction_per_case"], b_sat["drive_unsaturated_fraction_per_case"])
             if x is not None and y is not None]
    t = t_summary([x - y for x, y in pairs])
    return {"unsaturated_drive_fraction": t, "cases": len(pairs),
            "activated": bool(t["mean"] is not None and t["mean"] >= FEEDBACK_ACTIVATION
                              and t["low"] is not None and t["low"] > 0)}


def block_summary(rows: dict, sats: dict, winners: dict, parent_screen: dict, screens: dict) -> dict:
    """``rows[controller][role]`` and ``sats[controller][role]`` for the three controllers on promotion and fresh."""
    out = {"controllers": {c: {role: {**arm_summary(rows[c][role]), **failure_split(rows[c][role]),
                                     "saturation": {k: v for k, v in sats[c][role].items()
                                                    if k != "drive_unsaturated_fraction_per_case"}}
                               for role in ("promotion", "fresh")} for c in CONTROLLERS},
           "winners": winners, "parent_screen": parent_screen, "screens": screens, "promotion": {}, "fresh": {}}
    for arm in ARMS:
        w = f"{arm}_winner"
        out["promotion"][f"{w}_minus_parent"] = contrast(rows[w]["promotion"], rows["parent"]["promotion"])
        out["fresh"][f"{w}_minus_parent"] = contrast(rows[w]["fresh"], rows["parent"]["fresh"])
        out["fresh"][f"{w}_feedback"] = feedback_contrast(sats[w]["fresh"], sats["parent"]["fresh"])
        fresh = out["fresh"][f"{w}_minus_parent"]
        out["fresh"][f"{w}_qualifies_repair_parent"] = bool(
            out["promotion"][f"{w}_minus_parent"]["promotes"] and fresh["success"]["low"] is not None
            and fresh["success"]["low"] > 0 and fresh["drop_counts"][0] <= fresh["drop_counts"][1])
    out["fresh"]["cma_minus_random"] = contrast(rows["cma_winner"]["fresh"], rows["random_winner"]["fresh"])
    out["promotion"]["cma_minus_random"] = contrast(rows["cma_winner"]["promotion"], rows["random_winner"]["promotion"])
    return out


def summarise(blocks: list[dict]) -> dict:
    """Block-level primary contrast and the pooled secondary ones."""
    from lares.eval.stats import block_contrast

    def fresh_success(b, c):
        return b["controllers"][c]["fresh"]["success"]

    def fresh_drops(b, c):
        return b["controllers"][c]["fresh"]["drops"]

    out = {"blocks": len(blocks), "block_seeds": [b["seed"] for b in blocks],
           "cma_minus_random_fresh": block_contrast(fresh_success(b, "cma_winner") - fresh_success(b, "random_winner")
                                                    for b in blocks),
           "cma_minus_random_fresh_drops": block_contrast(fresh_drops(b, "cma_winner") - fresh_drops(b, "random_winner")
                                                          for b in blocks),
           "per_controller": {}, "promotions": {}, "qualifying_repair_parents": {}, "feedback_activated_blocks": {}}
    out["paired_success_difference"] = out["cma_minus_random_fresh"].get("mean")
    for c in CONTROLLERS:
        out["per_controller"][c] = {
            role: {"success": block_contrast(b["controllers"][c][role]["success"] for b in blocks),
                   "drops_total": int(sum(b["controllers"][c][role]["drops"] for b in blocks)),
                   "successes_total": int(sum(b["controllers"][c][role]["successes"] for b in blocks)),
                   "drive_saturated_fraction": [b["controllers"][c][role]["saturation"]["drive_saturated_fraction"]
                                                for b in blocks],
                   "action_saturated_fraction": [b["controllers"][c][role]["saturation"]["action_saturated_fraction"]
                                                 for b in blocks]}
            for role in ("promotion", "fresh")}
    for arm in ARMS:
        w = f"{arm}_winner"
        out[f"{w}_minus_parent_fresh"] = block_contrast(fresh_success(b, w) - fresh_success(b, "parent") for b in blocks)
        out[f"{w}_minus_parent_fresh_drops"] = block_contrast(fresh_drops(b, w) - fresh_drops(b, "parent") for b in blocks)
        out[f"{w}_minus_parent_promotion"] = block_contrast(
            b["promotion"][f"{w}_minus_parent"]["success"]["mean"] for b in blocks)
        out["promotions"][w] = [b["seed"] for b in blocks if b["promotion"][f"{w}_minus_parent"]["promotes"]]
        out["qualifying_repair_parents"][w] = [b["seed"] for b in blocks if b["fresh"][f"{w}_qualifies_repair_parent"]]
        out["feedback_activated_blocks"][w] = [b["seed"] for b in blocks if b["fresh"][f"{w}_feedback"]["activated"]]
        out[f"{w}_feedback_unsaturated_gain"] = block_contrast(
            b["fresh"][f"{w}_feedback"]["unsaturated_drive_fraction"]["mean"] for b in blocks)
    return out


def decision_artifact(summary: dict, blocks: list[dict]) -> dict:
    """What the deferred follow-ups R-056 and R-057 read; it executes nothing."""
    qualifying = [(b["seed"], w, b["controllers"][w]["fresh"]["success"], b["winners"][w.replace("_winner", "")])
                  for b in blocks for w in ("cma_winner", "random_winner") if b["fresh"][f"{w}_qualifies_repair_parent"]]
    best = max(qualifying, key=lambda q: q[2]) if qualifying else None
    activated = {w: summary["feedback_activated_blocks"][w] for w in ("cma_winner", "random_winner")}
    any_activated = any(activated.values())
    out = {"rule": {"repair_parent": "strict promotion on cases 10-39, fresh success interval lower bound above zero, "
                                     "and no more fresh drops than the parent, in the same block",
                    "feedback_activated": f"unsaturated in-play drive fraction up by at least {FEEDBACK_ACTIVATION} "
                                          "versus the parent on the fresh cases with the paired interval above zero"},
           "qualifying_repair_parents": [{"seed": s, "arm": w, "fresh_success": v, "winner": win}
                                         for s, w, v, win in qualifying],
           "repair_parent": None if best is None else {"seed": best[0], "arm": best[1], "fresh_success": best[2],
                                                       "winner": best[3],
                                                       "caveat": "chosen among qualifying winners by fresh success, "
                                                                 "which is a selection on the confirmation cases"},
           "feedback_activated_blocks": activated,
           "route": ("R-056: repair the tuned winner" if best is not None else
                     "R-057: anatomical-coordinate repair (R-043 re-scoped)" if any_activated else
                     "neither follow-up is enabled by this result")}
    return out


# ---------------------------------------------------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------------------------------------------------


def _cell(config_path: str, manifest: str, controller: str, out_path: str, record_obs: bool) -> None:
    child_cell(config_path, manifest, controller, out_path, study=STUDY, record_obs=record_obs, skip_existing=True)


def _rows(out_path: str) -> list[dict]:
    with open(out_path, encoding="utf-8") as f:
        return json.load(f)["episodes"]


def _trace(out_path: str):
    return np.load(out_path.replace(".json", ".npz"))


def _rel(out_dir: str, path: str) -> str:
    return os.path.relpath(path, out_dir)


def _freeze_json(path: str, payload: dict, what: str) -> None:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            if json.load(f) != json.loads(json.dumps(payload)):
                raise SystemExit(f"{path} exists and differs from the {what} this run produces")
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)


def run_arm(arm: str, space: ParameterSpace, opts: dict, seed: int, bdir: str, tiled_path: str, config_path: str,
            out_dir: str) -> dict:
    """One arm's 8 generations on the block's tiled screen; returns the frozen winner and the per-generation log."""
    adir = os.path.join(bdir, arm)
    os.makedirs(adir, exist_ok=True)
    sampler = make_sampler(arm, space, opts, seed)
    all_metrics, all_params, generations = [], [], []
    for g in range(opts["generations"]):
        xs = sampler.ask()
        if xs.shape != (opts["popsize"], space.dim):
            raise RuntimeError(f"sampler returned {xs.shape}, expected {(opts['popsize'], space.dim)}")
        if np.any(xs < opts["bounds"][0]) or np.any(xs > opts["bounds"][1]):
            raise RuntimeError("pycma returned a point outside its own bounds")
        candidates = [space.params(x) for x in xs]
        cand_path = os.path.join(adir, f"gen{g}_candidates.json")
        write_candidates(cand_path, f"{arm}:s{seed}:gen{g}", candidates,
                         {"normalised": xs.tolist(), "sampler": sampler.state()})
        cell = os.path.join(adir, f"gen{g}.json")
        _cell(config_path, tiled_path, f"candidates:{_rel(out_dir, cand_path)}", cell, False)
        per = [screen_metrics(r) for r in split_rows(_rows(cell), opts["popsize"])]
        ranks = rank_candidates(per)
        sampler.tell(ranks)
        base = len(all_metrics)
        all_metrics += per
        all_params += candidates
        generations.append({"generation": g, "metrics": per, "ranks": ranks,
                            "candidate_indices": list(range(base, base + len(per))), "sampler_after": sampler.state()})
        with open(os.path.join(adir, f"gen{g}_screen.json"), "w", encoding="utf-8") as f:
            json.dump(generations[-1], f, indent=1)
    best = min(range(len(all_metrics)), key=lambda i: rank_key(all_metrics[i], i))
    dup = len(all_params) - len({tuple(sorted(p.items())) for p in all_params})
    winner = {"arm": arm, "index": best, "generation": best // opts["popsize"], "params": all_params[best],
              "screen": all_metrics[best], "candidates": len(all_params), "duplicate_candidates": dup,
              "invalid_candidates": 0, "ranking": RANKING}
    _freeze_json(os.path.join(adir, "winner.json"), winner, "winner")
    return {"winner": winner, "generations": generations}


def run_block(seed: int, protocol_cfg: dict, config_path: str, out_dir: str, space: ParameterSpace, opts: dict) -> dict:
    from lares.utils.hashing import sha256_file

    bdir = os.path.join(out_dir, f"block_s{seed}")
    os.makedirs(bdir, exist_ok=True)
    mpath = manifest_path(protocol_cfg, seed)
    if not os.path.exists(mpath):
        raise SystemExit(f"frozen manifest {mpath} missing; build and commit it before the run")
    m = EvaluationManifest.load(mpath)
    role = cases.role_manifests(m, roles(protocol_cfg), expected_cases=int(protocol_cfg["cases"]), stage=STAGE)
    paths = {}
    for r, rm in role.items():
        paths[r] = os.path.join(bdir, f"{r}.yaml")
        cases.freeze_manifest(paths[r], rm)
    tiled_path = os.path.join(bdir, f"screen_tiled{opts['popsize']}.yaml")
    cases.freeze_manifest(tiled_path, tiled_manifest(role["screen"], opts["popsize"]))

    arms = {arm: run_arm(arm, space, opts, seed, bdir, tiled_path, config_path, out_dir) for arm in ARMS}
    winners = {arm: arms[arm]["winner"] for arm in ARMS}
    _freeze_json(os.path.join(bdir, "winners.json"), winners, "winners")  # frozen before any promotion or fresh episode

    files = {}
    for c in CONTROLLERS:
        params = {} if c == "parent" else winners[c.replace("_winner", "")]["params"]
        files[c] = os.path.join(bdir, f"{c}_params.json")
        write_candidates(files[c], f"{c}:s{seed}", [params])
    rows, sats = {c: {} for c in CONTROLLERS}, {c: {} for c in CONTROLLERS}
    for r in ("promotion", "fresh"):
        for c in CONTROLLERS:
            cell = os.path.join(bdir, f"{c}_{r}.json")
            _cell(config_path, paths[r], f"candidates:{_rel(out_dir, files[c])}", cell, True)
            rows[c][r] = _rows(cell)
            full = {**space.params(space.normalise(space.parent)), **({} if c == "parent" else winners[c.replace("_winner", "")]["params"])}
            sats[c][r] = saturation(_trace(cell), full)
    pcell = os.path.join(bdir, "parent_screen.json")
    _cell(config_path, paths["screen"], f"candidates:{_rel(out_dir, files['parent'])}", pcell, False)
    parent_screen = screen_metrics(_rows(pcell))

    summary = block_summary(rows, sats, winners, parent_screen, {arm: arms[arm]["generations"] for arm in ARMS})
    summary["seed"] = int(seed)
    summary["manifest"] = {"path": mpath, "sha256": sha256_file(mpath), "id": m.manifest_id}
    summary["episodes"] = (2 * opts["generations"] * opts["popsize"] * len(role["screen"]) + len(role["screen"])
                           + 3 * len(role["promotion"]) + 3 * len(role["fresh"]))
    with open(os.path.join(bdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return summary


def run_study(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    """Driver: parameter space, eight blocks, summary and decision artifact. Nothing is rerun if its output exists."""
    import cma

    from lares.utils.hashing import sha256_file

    lc = check_learner(protocol_cfg)
    os.makedirs(out_dir, exist_ok=True)
    space = parameter_space(protocol_cfg)
    opts = cma_options(protocol_cfg)
    _freeze_json(os.path.join(out_dir, "space.json"),
                 {"space": space.to_dict(), "whitelist": protocol_cfg.get("whitelist") or WHITELIST, "optimizer": opts,
                  "pycma_version": cma.__version__, "ranking": RANKING,
                  "random_arm": "fresh pycma instance per generation, identical options, asked once, never told",
                  "parent": lc}, "parameter space")
    blocks = [run_block(int(s), protocol_cfg, config_path, out_dir, space, opts) for s in protocol_cfg["seeds"]]
    summary = summarise(blocks)
    summary["episodes"] = int(sum(b["episodes"] for b in blocks))
    summary["pycma_version"] = cma.__version__
    summary["manifests"] = {b["seed"]: b["manifest"] for b in blocks}
    summary["limitation"] = ("budget-specific: 8 generations x 8 candidates per arm per block on 10 screen cases; "
                             "saturation over recorded in-play steps only; fresh cases confirm and never select")
    decision = decision_artifact(summary, blocks)
    with open(os.path.join(out_dir, "decision.json"), "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=1)
    summary["decision"] = decision
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    summary["space_sha256"] = sha256_file(os.path.join(out_dir, "space.json"))
    return summary


run = run_study

__all__ = ["ARMS", "CONTROLLERS", "CMASampler", "FreshInstanceSampler", "ParameterSpace", "RANKING", "STUDY",
           "TiledCandidates", "WHITELIST", "apply_params", "block_summary", "build_manifests", "contrast",
           "decision_artifact", "drive_from_obs", "feedback_contrast", "make_controller", "make_sampler",
           "parameter_space", "rank_candidates", "rank_key", "run", "run_study", "saturation", "screen_metrics",
           "split_rows", "summarise", "tiled_manifest", "whitelist_names"]
