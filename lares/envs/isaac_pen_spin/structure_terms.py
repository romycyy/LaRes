"""R-109 (P2): the symbolic residual as a term list, and the edit language the grammar and LLM arms share. No Isaac.

``residual_j = clip(sum_terms coef[term, slot(j)] * E_term,j(observation), -cap, cap)``. A term is
``{"expr": E, "target": T}``; ``E`` is a small expression over named, clipped primitives of the observation and the
reference action, ``T`` names which joints the term drives and how its coefficients are shared. The inherited
structure (:data:`BASE_TERMS`, 32 coefficients) is R-103/R-106's actor and reproduces it bit for bit. An edit adds one
term whose coefficients start at zero, so a child structure reproduces its parent exactly until it is trained.

Expressions are JSON: a primitive name, or ``["mul", A, B]``, ``["gate", A, c, "above" | "below"]``, ``["comp", A]``,
``["abs", A]``. Every primitive is ``unit`` (in ``[0, 1]``) or ``signed`` (in ``[-1, 1]``); the operators are typed
accordingly, so an expression's range is known without evaluating it.
"""

from __future__ import annotations

import json
import math

import numpy as np
import torch

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin import symbolic_reward as sr
from lares.envs.isaac_pen_spin.incumbent_residual import ACTUATED
from lares.envs.isaac_pen_spin.layout import VEL_OBS_SCALE
from lares.utils.hashing import sha256_text

FINGERS = ("FF", "MF", "RF", "LF", "TH")
#: finger index of each action joint (the wrist has none)
FINGER_OF = tuple(FINGERS.index(j[:2]) if j[:2] in FINGERS else -1 for j in sr.JOINTS)

#: primitive -> (type, shape); shape is ``scalar`` (N, 1), ``finger`` (per finger, broadcast to its chain) or ``joint``
PRIMITIVES = {
    "spin_deficit": ("unit", "scalar"), "spin_level": ("unit", "scalar"), "centre_dist": ("unit", "scalar"),
    "transverse": ("unit", "scalar"), "tilt_up": ("unit", "scalar"),
    "centre_x": ("signed", "scalar"), "centre_y": ("signed", "scalar"), "centre_z": ("signed", "scalar"),
    "yaw_cos2": ("signed", "scalar"), "yaw_sin2": ("signed", "scalar"),
    "phase_sin": ("signed", "scalar"), "phase_cos": ("signed", "scalar"),
    "tip_dist": ("unit", "finger"), "tip_axial_abs": ("unit", "finger"),
    "vel": ("signed", "joint"), "posture": ("signed", "joint"), "ref": ("signed", "joint"), "pos": ("signed", "joint"),
}
GATE_CENTRES = (0.25, 0.5, 0.75)
GATE_WIDTH = 0.25
OPERATORS = ("mul", "gate", "comp", "abs")

#: target -> the joints it drives, grouped into coefficient slots
def _target_table() -> dict:
    t = {"joints": [[i] for i in range(sr.N_JOINTS)],
         "groups": [[i for i in range(sr.N_JOINTS) if sr.GROUP_OF[i] == g] for g in range(sr.N_GROUPS)],
         "wrist": [[i] for i in range(sr.N_JOINTS) if sr.GROUP_OF[i] == 0]}
    for f, name in enumerate(FINGERS):
        t[f"chain:{name}"] = [[i] for i in range(sr.N_JOINTS) if FINGER_OF[i] == f]
    return t


TARGETS = _target_table()
EDIT_TARGETS = tuple(k for k in TARGETS if k != "joints")          # 'joints' exists only for the inherited term
BASE_TERMS = ({"expr": "spin_deficit", "target": "joints"}, {"expr": "vel", "target": "groups"},
              {"expr": "posture", "target": "groups"})
LIMITS = {"depth": 3, "nodes": 7, "new_coefficients": 8, "coefficients": 48, "total_nodes": 256, "activation": 0.05}


class EditError(ValueError):
    """An edit that is not in the language, or not admissible for this structure."""


# ---------------------------------------------------------------------------------------------------------------------
#  Expressions: static checks
# ---------------------------------------------------------------------------------------------------------------------


def expr_type(e) -> str:
    if isinstance(e, str):
        if e not in PRIMITIVES:
            raise EditError(f"unknown primitive {e!r}")
        return PRIMITIVES[e][0]
    if not isinstance(e, (list, tuple)) or not e or e[0] not in OPERATORS:
        raise EditError(f"not an expression: {e!r}")
    op = e[0]
    if op == "mul":
        if len(e) != 3:
            raise EditError("mul takes two arguments")
        a, b = expr_type(e[1]), expr_type(e[2])
        return "unit" if a == b == "unit" else "signed"
    if op == "gate":
        if len(e) != 4 or e[2] not in GATE_CENTRES or e[3] not in ("above", "below"):
            raise EditError(f"gate takes (A, c in {GATE_CENTRES}, 'above' | 'below')")
        if expr_type(e[1]) != "unit":
            raise EditError("gate needs a [0, 1]-valued argument")
        return "unit"
    if len(e) != 2:
        raise EditError(f"{op} takes one argument")
    a = expr_type(e[1])
    if (op == "comp" and a != "unit") or (op == "abs" and a != "signed"):
        raise EditError("comp needs a [0, 1]-valued argument and abs a [-1, 1]-valued one")
    return "unit"


def _children(e):
    return [] if isinstance(e, str) else ([e[1], e[2]] if e[0] == "mul" else [e[1]])


def depth(e) -> int:
    return 1 + max((depth(c) for c in _children(e)), default=0)


def nodes(e) -> int:
    return 1 + sum(nodes(c) for c in _children(e))


def canonical(e):
    """The expression with ``mul`` arguments in a fixed order, as nested lists (JSON-ready)."""
    if isinstance(e, str):
        return e
    if e[0] == "mul":
        a, b = sorted((canonical(e[1]), canonical(e[2])), key=lambda x: json.dumps(x))
        return ["mul", a, b]
    if e[0] == "gate":
        return ["gate", canonical(e[1]), float(e[2]), e[3]]
    return [e[0], canonical(e[1])]


def term_key(term: dict) -> str:
    return json.dumps({"expr": canonical(term["expr"]), "target": term["target"]}, sort_keys=True)


def n_slots(target: str) -> int:
    if target not in TARGETS:
        raise EditError(f"unknown target {target!r}")
    return len(TARGETS[target])


def n_coefficients(terms) -> int:
    return sum(n_slots(t["target"]) for t in terms)


def coefficient_names(terms) -> list[str]:
    out = []
    for k, t in enumerate(terms):
        for slot in TARGETS[t["target"]]:
            label = sr.JOINTS[slot[0]] if len(slot) == 1 else sr.GROUPS[sr.GROUP_OF[slot[0]]]
            out.append(f"t{k}[{json.dumps(canonical(t['expr']), separators=(',', ':'))}@{t['target']}].{label}")
    return out


# ---------------------------------------------------------------------------------------------------------------------
#  Primitives and evaluation (torch)
# ---------------------------------------------------------------------------------------------------------------------


class Features:
    """The primitives of one batch of observations, computed on demand. ``s``, ``vhat`` and ``ehat`` are passed in
    exactly as the R-103 controller computes them, so the inherited terms reproduce it bit for bit."""

    def __init__(self, obs, ref, s, vhat, ehat):
        self.obs, self.ref = torch.nan_to_num(obs.to(ref.device, ref.dtype)), ref
        self._c = {"spin_deficit": s, "vel": vhat, "posture": ehat, "ref": ref}
        self._g = None

    def _geometry(self) -> dict:
        if self._g is None:
            sl, o = cases.obs_slices(), self.obs
            q = o[:, sl["obj_rot"]]
            q = q / (torch.norm(q, dim=1, keepdim=True) + 1e-8)
            w, x, y, z = q[:, 0:1], q[:, 1:2], q[:, 2:3], q[:, 3:4]
            axis = torch.cat([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)], dim=1)
            axis = axis / (torch.norm(axis, dim=1, keepdim=True) + 1e-8)
            omega = o[:, sl["obj_angvel"]] / VEL_OBS_SCALE
            spin = torch.sum(omega * axis, dim=1, keepdim=True)
            rel = o[:, sl["obj_pos"]] - o[:, sl["in_hand_pos"]]
            tw = torch.cat([w, x], dim=1)
            tw = tw / (torch.norm(tw, dim=1, keepdim=True) + 1e-8)
            tips = o[:, sl["fingertip_pos"]].reshape(o.shape[0], 5, 3)
            off = tips - o[:, sl["obj_pos"]].unsqueeze(1)
            axial = torch.sum(off * axis.unsqueeze(1), dim=2)
            radial = torch.norm(off - axial.unsqueeze(2) * axis.unsqueeze(1), dim=2)
            excess = (axial.abs() - 0.075).clamp(min=0.0)
            self._g = {"axis": axis, "spin": spin, "rel": rel, "tw": tw, "axial": axial,
                       "transverse": torch.norm(omega - spin * axis, dim=1, keepdim=True),
                       "tip": torch.sqrt(radial * radial + excess * excess + 1e-10)}
        return self._g

    def _finger(self, per_finger):
        """(N, 5) -> (N, 20): each chain joint takes its finger's value, the wrist 0."""
        idx = torch.as_tensor([max(f, 0) for f in FINGER_OF], device=per_finger.device)
        mask = torch.as_tensor([1.0 if f >= 0 else 0.0 for f in FINGER_OF], device=per_finger.device, dtype=per_finger.dtype)
        return per_finger.index_select(1, idx) * mask

    def __getitem__(self, name: str):
        if name in self._c:
            return self._c[name]
        g = self._geometry()
        if name == "spin_level":
            v = (g["spin"].abs() / 7.0).clamp(0.0, 1.0)
        elif name == "centre_dist":
            v = (torch.norm(g["rel"], dim=1, keepdim=True) / 0.08).clamp(0.0, 1.0)
        elif name in ("centre_x", "centre_y", "centre_z"):
            k = "xyz".index(name[-1])
            v = (g["rel"][:, k:k + 1] / 0.08).clamp(-1.0, 1.0)
        elif name == "transverse":
            v = (g["transverse"] / 7.0).clamp(0.0, 1.0)
        elif name == "tilt_up":
            v = g["axis"][:, 2:3].abs().clamp(0.0, 1.0)
        elif name == "yaw_cos2":
            v = (g["axis"][:, 0:1] ** 2 - g["axis"][:, 1:2] ** 2).clamp(-1.0, 1.0)
        elif name == "yaw_sin2":
            v = (2.0 * g["axis"][:, 0:1] * g["axis"][:, 1:2]).clamp(-1.0, 1.0)
        elif name == "phase_sin":
            v = (2.0 * g["tw"][:, 0:1] * g["tw"][:, 1:2]).clamp(-1.0, 1.0)
        elif name == "phase_cos":
            v = (g["tw"][:, 0:1] ** 2 - g["tw"][:, 1:2] ** 2).clamp(-1.0, 1.0)
        elif name == "tip_dist":
            v = self._finger((g["tip"] / 0.06).clamp(0.0, 1.0))
        elif name == "tip_axial_abs":
            v = self._finger((g["axial"].abs() / 0.07).clamp(0.0, 1.0))
        elif name == "pos":
            v = self.obs[:, cases.obs_slices()["hand_dof_pos"]][:, list(ACTUATED)].clamp(-1.0, 1.0)
        else:
            raise EditError(f"unknown primitive {name!r}")
        self._c[name] = v
        return v


def evaluate(e, feats: Features):
    """The expression's value, broadcastable to ``(N, 20)``."""
    if isinstance(e, str):
        return feats[e]
    if e[0] == "mul":
        return evaluate(e[1], feats) * evaluate(e[2], feats)
    a = evaluate(e[1], feats)
    if e[0] == "gate":
        d = (a - float(e[2])) if e[3] == "above" else (float(e[2]) - a)
        return (d / GATE_WIDTH).clamp(0.0, 1.0)
    return (1.0 - a) if e[0] == "comp" else a.abs()


def _slot_index(target: str, device):
    idx = [-1] * sr.N_JOINTS
    for k, slot in enumerate(TARGETS[target]):
        for j in slot:
            idx[j] = k
    full = all(i >= 0 for i in idx)
    return (torch.as_tensor([max(i, 0) for i in idx], device=device),
            None if full else torch.as_tensor([1.0 if i >= 0 else 0.0 for i in idx], device=device))


def term_contributions(terms, theta, feats: Features) -> list:
    """Each term's unclipped contribution ``(N, 20)`` for ``theta`` of shape ``(n,)`` or ``(N, n)``."""
    out, at = [], 0
    for t in terms:
        k = n_slots(t["target"])
        idx, mask = _slot_index(t["target"], theta.device)
        coef = theta[..., at:at + k].index_select(-1, idx)
        c = coef * evaluate(t["expr"], feats)
        out.append(c if mask is None else c * mask.to(c.dtype))
        at += k
    if at != theta.shape[-1]:
        raise ValueError(f"theta has {theta.shape[-1]} coefficients, the structure {at}")
    return out


def residual(terms, theta, feats: Features, cap: float):
    parts = term_contributions(terms, theta, feats)
    total = parts[0]
    for c in parts[1:]:
        total = total + c
    return total.clamp(-cap, cap)


# ---------------------------------------------------------------------------------------------------------------------
#  Structures, hashing, the controller
# ---------------------------------------------------------------------------------------------------------------------


def theta_record(theta, n: int | None = None) -> list[float]:
    arr = np.asarray(theta.detach().cpu().numpy() if torch.is_tensor(theta) else theta, dtype=np.float32).reshape(-1)
    if n is not None and arr.size != n:
        raise ValueError(f"theta has {arr.size} coefficients, the structure {n}")
    return [float(x) for x in arr]


def policy_record(terms, theta) -> dict:
    """A structure with its coefficients and the hash that names it."""
    terms = [{"expr": canonical(t["expr"]), "target": t["target"]} for t in terms]
    th = theta_record(theta, n_coefficients(terms))
    return {"terms": terms, "theta": th,
            "sha256": sha256_text(json.dumps({"terms": terms, "theta": th}, sort_keys=True, separators=(",", ":")))}


def check_policy(rec: dict) -> dict:
    if policy_record(rec["terms"], rec["theta"])["sha256"] != rec["sha256"]:
        raise SystemExit("a stored structure does not match its hash")
    return rec


def base_policy(theta32) -> dict:
    return policy_record(BASE_TERMS, theta32)


def extend(policy: dict, edit: dict | None) -> dict:
    """The child of ``policy`` under ``edit``: one more term with zero coefficients (``None``: the unchanged parent)."""
    if edit is None:
        return policy_record(policy["terms"], policy["theta"])
    terms = list(policy["terms"]) + [{"expr": edit["expr"], "target": edit["target"]}]
    return policy_record(terms, list(policy["theta"]) + [0.0] * n_slots(edit["target"]))


class TermController(sr.SymbolicResidualController):
    """The term-list actor behind the runner's controller interface (the R-103 controller with ``mean`` replaced)."""

    def __init__(self, name: str, reference, normalization: dict, actor_cfg: dict, policy: dict):
        self.ref, self.name = reference, name
        self.cap, self.bound = float(actor_cfg["cap"]), float(actor_cfg["bound"])
        self.spin_target = float(actor_cfg["spin_target"])
        self.sigmas = float(actor_cfg["feature_clip_sigmas"])
        self._norm = normalization
        self.terms = [dict(t) for t in policy["terms"]]
        self._theta_cpu = torch.tensor(theta_record(policy["theta"], n_coefficients(self.terms)))
        if float(self._theta_cpu.abs().max()) > self.bound + 1e-6:
            raise ValueError("theta leaves the coefficient bound")
        self.params = {"policy_sha256": policy_record(self.terms, self._theta_cpu)["sha256"], "terms": self.terms,
                       "theta": theta_record(self._theta_cpu), "theta_sha256": policy_record(self.terms, self._theta_cpu)["sha256"],
                       "reference": getattr(reference, "params", None), "cap": self.cap, "bound": self.bound,
                       "execution": "clamp(reference + clip(sum_terms coef * expr, -cap, cap), -1, 1)"}
        self._stats = None

    def parts(self, obs, t, ctx) -> dict:
        p = super().parts(obs, t, ctx)
        p["feats"] = Features(obs, p["ref"], p["s"], p["vhat"], p["ehat"])
        return p

    def mean(self, parts: dict, theta):
        return parts["ref"] + residual(self.terms, theta, parts["feats"], self.cap)


def build_controller(protocol_cfg: dict, name: str, policy: dict, normalization: dict, inner=None):
    return TermController(name, sr.build_reference(protocol_cfg, f"{name}:reference", inner=inner), normalization,
                          protocol_cfg["actor"], check_policy(policy))


# ---------------------------------------------------------------------------------------------------------------------
#  Edits: validation and the grammar sampler
# ---------------------------------------------------------------------------------------------------------------------


def sample_features(sample: dict, normalization: dict, actor_cfg: dict) -> Features:
    """Primitives on a stored observation sample (``obs``, ``ref``, ``v_f`` arrays), as the controller computes them."""
    obs, ref, v_f = (torch.as_tensor(np.asarray(sample[k]), dtype=torch.float32) for k in ("obs", "ref", "v_f"))
    sig = float(actor_cfg["feature_clip_sigmas"])
    s = ((float(actor_cfg["spin_target"]) - sr.axial_rate(torch.nan_to_num(obs)).abs()) / float(actor_cfg["spin_target"])).clamp(0.0, 1.0)
    vhat = (v_f / (sig * torch.tensor(normalization["rms_velocity"]))).clamp(-1.0, 1.0)
    q = torch.nan_to_num(obs)[:, cases.obs_slices()["hand_dof_pos"]][:, list(ACTUATED)]
    ehat = ((ref - q) / (sig * torch.tensor(normalization["rms_posture"]))).clamp(-1.0, 1.0)
    return Features(obs, ref, s, vhat, ehat)


def activation_fraction(edit: dict, feats: Features) -> float:
    """The fraction of sample rows on which the new term can move at least one of its joints."""
    v = evaluate(edit["expr"], feats)
    v = v.expand(v.shape[0], sr.N_JOINTS) if v.shape[1] == 1 else v
    joints = [j for slot in TARGETS[edit["target"]] for j in slot]
    return float((v[:, joints].abs() > 1e-6).any(dim=1).float().mean())


def validate_edit(edit, policy: dict, tried=(), feats: Features | None = None, limits: dict = LIMITS) -> dict:
    """The canonical edit, or :class:`EditError` naming the first rule it breaks."""
    if not isinstance(edit, dict) or "expr" not in edit or "target" not in edit:
        raise EditError('an edit is {"expr": ..., "target": ...}')
    if edit["target"] not in EDIT_TARGETS:
        raise EditError(f"target must be one of {list(EDIT_TARGETS)}")
    expr_type(edit["expr"])
    if depth(edit["expr"]) > limits["depth"] or nodes(edit["expr"]) > limits["nodes"]:
        raise EditError(f"expression exceeds depth {limits['depth']} or {limits['nodes']} nodes")
    out = {"expr": canonical(edit["expr"]), "target": edit["target"]}
    k = n_slots(out["target"])
    if k > limits["new_coefficients"] or n_coefficients(policy["terms"]) + k > limits["coefficients"]:
        raise EditError("the edit exceeds the coefficient limits")
    if sum(nodes(t["expr"]) for t in policy["terms"]) + nodes(out["expr"]) > limits["total_nodes"]:
        raise EditError("the structure would exceed its node limit")
    key = term_key(out)
    if key in {term_key(t) for t in policy["terms"]} or key in {term_key(t) for t in tried}:
        raise EditError("this term already exists in the structure or was already tried in this block")
    if feats is not None:
        frac = activation_fraction(out, feats)
        if frac < limits["activation"]:
            raise EditError(f"the term is nonzero on only {frac:.3f} of the observation sample (minimum {limits['activation']})")
    return out


def _random_expr(rng, d: int, want: str | None):
    if d == 1:
        names = [n for n, (ty, _) in PRIMITIVES.items() if want in (None, ty)]
        return names[int(rng.integers(len(names)))]
    ops = ["mul", "gate", "comp", "abs"] if want != "signed" else ["mul"]
    op = ops[int(rng.integers(len(ops)))]
    if op == "mul":
        other = int(rng.integers(1, d))                      # the second factor is no deeper than the first
        a_want = want if want == "unit" else None
        a = _random_expr(rng, d - 1, a_want)
        b_want = "unit" if want == "unit" else ("signed" if want == "signed" and expr_type(a) == "unit" else None)
        return ["mul", a, _random_expr(rng, other, b_want)]
    if op == "gate":
        return ["gate", _random_expr(rng, d - 1, "unit"), GATE_CENTRES[int(rng.integers(3))], ("above", "below")[int(rng.integers(2))]]
    return [op, _random_expr(rng, d - 1, "unit" if op == "comp" else "signed")]


def sample_edit(rng, policy: dict, tried=(), feats: Features | None = None, limits: dict = LIMITS, attempts: int = 500) -> dict:
    """One admissible edit from the declared distribution: target 'groups' 0.4, a chain 0.5, 'wrist' 0.1; depth 1/2/3
    with probability 0.3/0.4/0.3; primitives and operators uniform over the type-valid ones; resampled until valid."""
    chains = [t for t in EDIT_TARGETS if t.startswith("chain:")]
    for _ in range(attempts):
        u = rng.random()
        target = "groups" if u < 0.4 else (chains[int(rng.integers(len(chains)))] if u < 0.9 else "wrist")
        d = int(rng.choice([1, 2, 3], p=[0.3, 0.4, 0.3]))
        try:
            return validate_edit({"expr": _random_expr(rng, d, None), "target": target}, policy, tried, feats, limits)
        except EditError:
            continue
    raise RuntimeError("the grammar found no admissible edit")


# ---------------------------------------------------------------------------------------------------------------------
#  ARS at any dimension
# ---------------------------------------------------------------------------------------------------------------------


def ars_directions(seed: int, directions: int, dim: int) -> np.ndarray:
    return np.random.default_rng(int(seed)).standard_normal((int(directions), int(dim)))


def ars_candidates(theta, delta, nu: float, bound: float) -> np.ndarray:
    """Rows ``+d0..+d(n-1), -d0..-d(n-1)``."""
    theta = np.asarray(theta, dtype=float)
    return np.clip(np.concatenate([theta + nu * delta, theta - nu * delta]), -bound, bound)


def ars_step(theta, delta, scores, alpha: float, bound: float, top: int | None = None) -> tuple:
    """R-106's update: the step divided by the standard deviation of the returns it uses, then projection."""
    theta, r = np.asarray(theta, dtype=float), np.asarray(scores, dtype=float)
    n = len(delta)
    rp, rm = r[:n], r[n:]
    top = n if top is None else int(top)
    order = np.argsort(-np.maximum(rp, rm), kind="stable")[:top]
    sigma = float(np.concatenate([rp[order], rm[order]]).std())
    if sigma > 1e-8:
        theta = np.clip(theta + alpha / (top * sigma) * ((rp[order] - rm[order])[:, None] * delta[order]).sum(0), -bound, bound)
    return theta, {"sigma_r": sigma}


LANGUAGE_TEXT = """A term is {"expr": E, "target": T}. Its contribution to joint j is coef[slot(j)] * E_j; a new term's
coefficients start at 0 and are then trained (bounded to [-0.10, 0.10]); the summed residual is clipped to
[-0.10, 0.10] action units and added to the reference controller's action.
Targets T (one coefficient per slot): "groups" (6 slots: wrist, FF, MF, RF, LF, TH; drives all 20 joints),
"chain:FF" (FFJ3, FFJ2, FFJ1), "chain:MF" (MFJ3, MFJ2, MFJ1), "chain:RF" (RFJ3, RFJ2, RFJ1),
"chain:LF" (LFJ4, LFJ3, LFJ2, LFJ1), "chain:TH" (THJ4, THJ3, THJ2, THJ1, THJ0), "wrist" (WRJ1, WRJ0).
Primitives with values in [0, 1]: spin_deficit = clip((7 - |axial rate|)/7), spin_level = clip(|axial rate|/7),
centre_dist = clip(|pen centre - in-hand position|/0.08), transverse = clip(|end-over-end angular speed|/7),
tilt_up = |z component of the pen's long axis|; per finger (applied to that finger's chain joints, 0 on the wrist):
tip_dist = clip(fingertip-to-pen distance/0.06), tip_axial_abs = clip(|fingertip coordinate along the pen|/0.07).
Primitives with values in [-1, 1]: centre_x, centre_y, centre_z = clip(offset of the pen from the in-hand position/0.08),
yaw_cos2, yaw_sin2 (the pen axis's heading in the palm plane, invariant to which end is which), phase_sin, phase_cos
(the pen's roll angle about its own axis); per joint: vel (filtered joint velocity, normalised), posture (the
reference's target minus the joint position, normalised), ref (the reference action), pos (the joint position).
Operators: ["mul", A, B]; ["gate", A, c, "above" | "below"] with A in [0, 1] and c in {0.25, 0.5, 0.75}, equal to
clip((A - c)/0.25, 0, 1) or clip((c - A)/0.25, 0, 1); ["comp", A] = 1 - A for A in [0, 1]; ["abs", A] for A in [-1, 1].
Limits: depth at most 3 (a primitive is depth 1), at most 7 nodes, at most 8 new coefficients, 48 in the structure.
The term must not already exist in the structure or among the edits already tried, and must be nonzero on at least
5% of the parent's observations."""

__all__ = ["BASE_TERMS", "EDIT_TARGETS", "EditError", "FINGER_OF", "Features", "LANGUAGE_TEXT", "LIMITS", "PRIMITIVES",
           "TARGETS", "TermController", "activation_fraction", "ars_candidates", "ars_directions", "ars_step",
           "base_policy", "build_controller", "canonical", "check_policy", "coefficient_names", "depth", "evaluate",
           "expr_type", "extend", "n_coefficients", "n_slots", "nodes", "policy_record", "residual", "sample_edit",
           "sample_features", "term_contributions", "term_key", "theta_record", "validate_edit"]
