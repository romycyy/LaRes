"""R-131 (Q4): a small language of recovery modules, shared by the grammar sampler and the LLM (no Isaac).

A module watches three signals computed from the observation (the pen's absolute axial rate, its distance from the
in-hand position, its transverse rate) and returns, per step, a blend weight toward a target posture and a stroke
boost flag. The hand-designed R-116 module is the language's **template**: one decision step; if the pen is centred
and slow, ramp the sent action toward the reset posture and back, then boost the stroke for 60 steps.

A structure is the template plus edits. An **edit changes exactly one component** and adds at most four bounded
parameters, each starting at an identity value, so a child is its parent before training (checked by replay,
:func:`replay_identity`). Every structure is a point of one fixed 27-slot parameter layout (:data:`SLOTS`): a
structure only decides which slots are trained (its *active* coordinates) and the comparison direction of its extra
trigger conditions, so one batch can hold different structures (per-environment parameters and switches).
"""

from __future__ import annotations

import json

import numpy as np
import torch

from lares.utils.hashing import sha256_text

WINDOW, BOOST_STEPS, RETRY_DEADLINE = 30, 60, 300
GROUPS = ("WR", "FF", "MF", "RF", "LF", "TH")
#: (name, low, high, identity / initial value). Slots 0-10 are the template (the scripted D-080 point).
SLOTS = (
    ("threshold", 1.0, 4.0, 3.0), ("decision_step", 60.0, 180.0, 120.0), ("ramp_in", 5.0, 40.0, 20.0), ("ramp_out", 10.0, 80.0, 40.0),
    *((f"depth_{g}", 0.0, 1.0, 1.0) for g in GROUPS), ("stroke_multiplier", 1.0, 2.0, 1.0),
    ("period", 5.0, 60.0, 10.0), ("extra_window", 0.0, 150.0, 0.0),                      # schedule: periodic
    ("second_gap", 20.0, 150.0, 60.0), ("second_scale", 0.0, 1.0, 0.0),                  # schedule: two_step
    ("level_transverse", 0.0, 10.0, None), ("level_trend", -4.0, 4.0, None), ("level_deficit", 0.0, 1.0, None),
    ("deficit_tau", 10.0, 120.0, 60.0),                                                  # conditions
    ("max_hold", 0.0, 60.0, 0.0), ("settle_level", 0.0, 10.0, 3.0),                      # support_end: event
    ("max_extra", 0.0, 115.0, 0.0), ("release_level", 4.0, 10.0, 6.0),                   # boost_end: event
    ("mix_midrange", 0.0, 1.0, 0.0), ("mix_hold", 0.0, 1.0, 0.0),                        # target
    ("retry_gap", 30.0, 150.0, 60.0), ("retry_depth", 0.0, 1.0, 0.0),                    # attempts: 2
)
NAMES = tuple(s[0] for s in SLOTS)
LOW = np.array([s[1] for s in SLOTS])
HIGH = np.array([s[2] for s in SLOTS])
N_SLOTS, N_TEMPLATE = len(SLOTS), 11
I = {n: k for k, n in enumerate(NAMES)}
PRIMITIVES = ("transverse", "trend", "deficit")
OPS = ("lt", "gt")
TEMPLATE = {"schedule": "single", "conditions": [], "support_end": "timed", "boost_end": "timed", "target": "reset", "attempts": 1}
CHOICES = {"schedule": ("single", "periodic", "two_step"), "support_end": ("timed", "event"), "boost_end": ("timed", "event"),
           "target": ("reset", "midrange", "hold"), "attempts": (1, 2)}
#: the slots a component value trains
COMPONENT_SLOTS = {("schedule", "periodic"): ("period", "extra_window"), ("schedule", "two_step"): ("second_gap", "second_scale"),
                   ("support_end", "event"): ("max_hold", "settle_level"), ("boost_end", "event"): ("max_extra", "release_level"),
                   ("target", "midrange"): ("mix_midrange",), ("target", "hold"): ("mix_hold",), ("attempts", 2): ("retry_gap", "retry_depth")}
CONDITION_SLOTS = {"transverse": ("level_transverse",), "trend": ("level_trend",), "deficit": ("level_deficit", "deficit_tau")}
LIMITS = {"edits": 3, "parameters": 23, "conditions": 2, "new_parameters": 4}


class EditError(ValueError):
    """An edit or structure the language does not admit."""


# ---------------------------------------------------------------------------------------------------------------------
#  Structures, edits, parameters
# ---------------------------------------------------------------------------------------------------------------------


def canonical(spec: dict) -> dict:
    """A validated copy of a structure, conditions sorted by primitive."""
    if not isinstance(spec, dict) or set(spec) != set(TEMPLATE):
        raise EditError(f"a structure has exactly the components {sorted(TEMPLATE)}")
    out = {}
    for comp, allowed in CHOICES.items():
        if spec[comp] not in allowed:
            raise EditError(f"{comp} must be one of {list(allowed)}")
        out[comp] = spec[comp]
    conds = spec["conditions"]
    if not isinstance(conds, list) or len(conds) > LIMITS["conditions"]:
        raise EditError(f"at most {LIMITS['conditions']} extra trigger conditions")
    seen = []
    for c in conds:
        if not isinstance(c, dict) or set(c) != {"primitive", "op"} or c["primitive"] not in PRIMITIVES or c["op"] not in OPS:
            raise EditError('a condition is {"primitive": one of ' + str(list(PRIMITIVES)) + ', "op": "lt" or "gt"}')
        if c["primitive"] in seen:
            raise EditError(f"the primitive {c['primitive']} is already a condition")
        seen.append(c["primitive"])
    out["conditions"] = sorted(({"primitive": c["primitive"], "op": c["op"]} for c in conds), key=lambda c: PRIMITIVES.index(c["primitive"]))
    if n_edits(out) > LIMITS["edits"] or len(active(out)) > LIMITS["parameters"]:
        raise EditError(f"at most {LIMITS['edits']} edits and {LIMITS['parameters']} module parameters")
    return out


def n_edits(spec: dict) -> int:
    return sum(spec[c] != TEMPLATE[c] for c in CHOICES) + len(spec["conditions"])


def active(spec: dict) -> list[int]:
    """Slot indices a structure trains, ascending: the template's eleven, then its components' own."""
    names = list(NAMES[:N_TEMPLATE])
    for comp in CHOICES:
        names += COMPONENT_SLOTS.get((comp, spec[comp]), ())
    for c in spec["conditions"]:
        names += CONDITION_SLOTS[c["primitive"]]
    return sorted(I[n] for n in names)


def identity_values(spec: dict) -> np.ndarray:
    """Physical values of all 27 slots at which the structure's non-template components do nothing."""
    v = np.array([LOW[k] if s[3] is None else s[3] for k, s in enumerate(SLOTS)], dtype=float)
    for c in spec["conditions"]:                             # a comparison that is always true: lt at the top, gt at the bottom
        k = I[CONDITION_SLOTS[c["primitive"]][0]]
        v[k] = HIGH[k] if c["op"] == "lt" else LOW[k]
    return v


def ops_vector(spec: dict) -> np.ndarray:
    """Per primitive: 0 no condition, 1 ``lt``, 2 ``gt``."""
    out = np.zeros(len(PRIMITIVES), dtype=np.int64)
    for c in spec["conditions"]:
        out[PRIMITIVES.index(c["primitive"])] = 1 + OPS.index(c["op"])
    return out


def to_unit(values, idx) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return (v - LOW[idx]) / (HIGH[idx] - LOW[idx])


def full_params(spec: dict, x) -> np.ndarray:
    """All 27 slots in physical units for normalised active coordinates ``x`` (``(dim,)`` or ``(rows, dim)``)."""
    idx = active(spec)
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if x.shape[1] != len(idx):
        raise EditError(f"the structure trains {len(idx)} coordinates, got {x.shape[1]}")
    out = np.tile(identity_values(spec), (x.shape[0], 1))
    out[:, idx] = LOW[idx] + np.clip(x, 0.0, 1.0) * (HIGH[idx] - LOW[idx])
    return out


def initial_x(spec: dict) -> np.ndarray:
    idx = active(spec)
    return to_unit(identity_values(spec)[idx], idx)


def record(spec: dict, x) -> dict:
    spec = canonical(spec)
    xs = [float(v) for v in np.asarray(x, dtype=np.float64).reshape(-1)]
    idx = active(spec)
    if len(xs) != len(idx):
        raise EditError(f"the structure trains {len(idx)} coordinates, got {len(xs)}")
    return {"spec": spec, "x": xs, "params": dict(zip([NAMES[k] for k in idx], [float(v) for v in full_params(spec, xs)[0, idx]])),
            "edits": n_edits(spec), "parameters": len(idx),
            "sha256": sha256_text(json.dumps({"spec": spec, "x": xs}, sort_keys=True, separators=(",", ":")))}


def check_record(rec: dict) -> dict:
    if record(rec["spec"], rec["x"])["sha256"] != rec["sha256"]:
        raise SystemExit("a stored module does not match its hash")
    return rec


def template_record() -> dict:
    return record(TEMPLATE, initial_x(TEMPLATE))


def apply_edit(spec: dict, edit) -> dict:
    """The parent with one component changed. Refuses anything that is not exactly one new change."""
    spec = canonical(spec)
    if not isinstance(edit, dict) or "component" not in edit:
        raise EditError('an edit is {"component": <name>, "value": <value>} or {"component": "conditions", "add": {...}}')
    comp = edit["component"]
    child = {**spec, "conditions": list(spec["conditions"])}
    if comp == "conditions":
        if set(edit) - {"component", "add", "rationale"} or "add" not in edit:
            raise EditError('a condition edit is {"component": "conditions", "add": {"primitive": ..., "op": ...}}')
        child["conditions"].append(edit["add"])
    elif comp in CHOICES:
        if set(edit) - {"component", "value", "rationale"} or "value" not in edit:
            raise EditError(f'an edit of {comp} is {{"component": "{comp}", "value": <one of {list(CHOICES[comp])}>}}')
        if spec[comp] != TEMPLATE[comp]:
            raise EditError(f"{comp} was already changed in this structure; each component is changed at most once")
        if edit["value"] == TEMPLATE[comp] or edit["value"] not in CHOICES[comp]:
            raise EditError(f"{comp} must change to one of {[v for v in CHOICES[comp] if v != TEMPLATE[comp]]}")
        child[comp] = edit["value"]
    else:
        raise EditError(f"unknown component {comp!r}; components are {sorted(TEMPLATE)}")
    child = canonical(child)
    if len(active(child)) - len(active(spec)) > LIMITS["new_parameters"]:
        raise EditError("an edit adds at most four parameters")
    return child


def edit_key(edit: dict) -> str:
    e = {k: edit[k] for k in ("component", "value", "add") if k in edit}
    return json.dumps(e, sort_keys=True)


def valid_edits(spec: dict) -> list[dict]:
    """Every admissible single edit of a structure, in a fixed order."""
    cands = [{"component": c, "value": v} for c, vals in CHOICES.items() for v in vals if v != TEMPLATE[c]]
    cands += [{"component": "conditions", "add": {"primitive": p, "op": o}} for p in PRIMITIVES for o in OPS]
    out = []
    for e in cands:
        try:
            apply_edit(spec, e)
            out.append(e)
        except EditError:
            pass
    return out


def sample_edits(rng, spec: dict, k: int, tried=()) -> list[dict]:
    """``k`` distinct edits drawn uniformly without replacement from the valid ones not yet tried."""
    pool = [e for e in valid_edits(spec) if edit_key(e) not in {edit_key(t) for t in tried}]
    order = rng.permutation(len(pool))
    return [pool[int(j)] for j in order[:k]]


def inherit(parent: dict, child_spec: dict) -> np.ndarray:
    """The child's coordinates: the parent's where shared, identity for the new ones."""
    child_spec = canonical(child_spec)
    p_idx = active(parent["spec"])
    px = dict(zip(p_idx, parent["x"]))
    idx = active(child_spec)
    init = dict(zip(idx, initial_x(child_spec)))
    return np.array([px[k] if k in px else init[k] for k in idx], dtype=float)


# ---------------------------------------------------------------------------------------------------------------------
#  The interpreter: per-environment parameters and switches, signals in, blend weight and boost out
# ---------------------------------------------------------------------------------------------------------------------


class ModuleState:
    """All environments' module state. ``params`` is ``(n, 27)`` physical values, ``ops`` ``(n, 3)`` condition switches."""

    def __init__(self, n: int, device, dtype=torch.float32, centre_tolerance: float = 0.08):
        self.n, self.device, self.dtype, self.tol = int(n), device, dtype, float(centre_tolerance)
        self.set(np.tile(identity_values(TEMPLATE), (n, 1)), np.zeros((n, len(PRIMITIVES)), dtype=np.int64))
        self.reset()

    def set(self, params, ops) -> None:
        p = torch.as_tensor(np.asarray(params), device=self.device, dtype=self.dtype)
        o = torch.as_tensor(np.asarray(ops), device=self.device, dtype=torch.long)
        self.p = p.expand(self.n, N_SLOTS).clone() if p.shape[0] == 1 else p
        self.ops = o.expand(self.n, len(PRIMITIVES)).clone() if o.shape[0] == 1 else o
        g = lambda name: self.p[:, I[name]]  # noqa: E731
        self.T = torch.round(g("decision_step")).long()
        self.period = torch.round(g("period")).long().clamp(min=1)
        self.gap2 = torch.round(g("second_gap")).long()
        self.retry_gap = torch.round(g("retry_gap")).long()

    def reset(self, mask=None) -> None:
        z = lambda *shape, dt=None: torch.zeros(*shape, device=self.device, dtype=dt or self.dtype)  # noqa: E731
        if mask is None:
            self.step = z(self.n, dt=torch.long)
            self.hist, self.thist = z(self.n, WINDOW), z(self.n, WINDOW)
            self.deficit = z(self.n)
            self.fired, self.fired2 = z(self.n, dt=torch.bool), z(self.n, dt=torch.bool)
            self.t_fire, self.t_fire2 = z(self.n, dt=torch.long), z(self.n, dt=torch.long)
            self.hold_used, self.extra_used = z(self.n), z(self.n)
            self.released, self.fire_now = z(self.n, dt=torch.bool), z(self.n, dt=torch.bool)
            self.boost_steps = z(self.n)
            return
        m = mask.to(self.device)
        for name in ("step", "t_fire", "t_fire2"):
            setattr(self, name, torch.where(m, torch.zeros_like(getattr(self, name)), getattr(self, name)))
        for name in ("deficit", "hold_used", "extra_used", "boost_steps"):
            setattr(self, name, torch.where(m, torch.zeros_like(getattr(self, name)), getattr(self, name)))
        for name in ("fired", "fired2", "released", "fire_now"):
            setattr(self, name, getattr(self, name) & ~m)
        self.hist = torch.where(m.unsqueeze(1), torch.zeros_like(self.hist), self.hist)
        self.thist = torch.where(m.unsqueeze(1), torch.zeros_like(self.thist), self.thist)

    def _timeline(self, u, hold):
        """Blend weight and ramp-out progress ``v`` for steps-since-fire ``u`` with ``hold`` held steps already used."""
        n_in, n_out = self.p[:, I["ramp_in"]], self.p[:, I["ramp_out"]]
        v = u - n_in - hold
        up = ((u + 1.0) / n_in).clamp(max=1.0)
        down = 1.0 - v / n_out
        zero = torch.zeros_like(up)
        beta = torch.where(u < 0, zero, torch.where(u < n_in, up, torch.where(v < 0, torch.ones_like(up), torch.where(v < n_out, down, zero))))
        return beta, v - n_out

    def __call__(self, rate, centre, transverse, enabled: bool = True) -> tuple:
        """One control step. Returns ``(beta, boost, fire_now)``: the blend weight in [0, 1], the boost flag and
        whether the first trigger fired on this step (the controller stores the root action it then holds)."""
        g = lambda name: self.p[:, I[name]]  # noqa: E731
        col = (self.step % WINDOW).unsqueeze(1)
        self.hist.scatter_(1, col, rate.unsqueeze(1))
        self.thist.scatter_(1, col, transverse.unsqueeze(1))
        mean_rate, mean_tr = self.hist.mean(dim=1), self.thist.mean(dim=1)
        # trend: the mean of the 15 newest rates minus the mean of the 15 before them (ring order from the step counter)
        age = (self.step.unsqueeze(1) - torch.arange(WINDOW, device=self.device).unsqueeze(0)) % WINDOW
        new = (age < WINDOW // 2).to(self.dtype)
        trend = ((self.hist * new).sum(1) - (self.hist * (1.0 - new)).sum(1)) / (WINDOW // 2)
        self.deficit = self.deficit + ((4.0 - rate).clamp(0.0, 4.0) / 4.0 - self.deficit) / g("deficit_tau")
        rng = lambda name: (float(LOW[I[name]]), float(HIGH[I[name]]))  # noqa: E731
        prim = (mean_tr.clamp(*rng("level_transverse")), trend.clamp(*rng("level_trend")), self.deficit.clamp(*rng("level_deficit")))
        extra = torch.ones(self.n, dtype=torch.bool, device=self.device)
        for k, name in enumerate(("level_transverse", "level_trend", "level_deficit")):
            op = self.ops[:, k]
            extra = extra & torch.where(op == 1, prim[k] <= g(name), torch.where(op == 2, prim[k] >= g(name), torch.ones_like(extra)))
        centred = centre < self.tol
        since = self.step - self.T
        at_base = since == 0
        at_periodic = (since > 0) & (since.to(self.dtype) <= g("extra_window")) & (since % self.period == 0)
        at_second = (since == self.gap2) & (g("second_scale") > 0)
        slow = mean_rate < g("threshold")
        slow2 = mean_rate < g("threshold") * g("second_scale")
        want = ((at_base | at_periodic) & slow) | (at_second & slow2)
        self.fire_now = want & centred & extra & ~self.fired if enabled else torch.zeros_like(self.fired)
        self.t_fire = torch.where(self.fire_now, self.step, self.t_fire)
        self.fired = self.fired | self.fire_now

        u = (self.step - self.t_fire).to(self.dtype)
        n_in = g("ramp_in")
        holding = self.fired & (u >= n_in) & ((u - n_in - self.hold_used) < 1.0) & (self.hold_used < g("max_hold")) & (mean_tr > g("settle_level")) \
            & ((u - n_in - self.hold_used) >= 0)
        self.hold_used = self.hold_used + holding.to(self.dtype)
        self.holding = holding
        beta, w = self._timeline(u, self.hold_used)
        beta = torch.where(self.fired, beta, torch.zeros_like(beta))
        in_boost = self.fired & (w >= 0)
        timed = in_boost & (w < BOOST_STEPS)
        want_extra = in_boost & (w >= BOOST_STEPS) & ~self.released & (self.extra_used < g("max_extra")) & (mean_rate < g("release_level"))
        self.released = self.released | (in_boost & (w >= BOOST_STEPS) & ~want_extra)
        self.extra_used = self.extra_used + want_extra.to(self.dtype)
        self.extending = want_extra
        boost = timed | want_extra

        # one retry (R-118's semantics, timed): after a real first attempt, still slow and centred, before the deadline
        end = g("ramp_in") + g("ramp_out")
        t2 = self.t_fire + torch.round(end).long() + self.retry_gap
        new2 = (self.fired & ~self.fired2 & (self.step == t2) & (g("retry_depth") > 0) & centred & slow
                & (t2.to(self.dtype) + end <= RETRY_DEADLINE)) if enabled else torch.zeros_like(self.fired)
        self.t_fire2 = torch.where(new2, self.step, self.t_fire2)
        self.fired2 = self.fired2 | new2
        u2 = (self.step - self.t_fire2).to(self.dtype)
        b2, w2 = self._timeline(u2, torch.zeros_like(u2))
        beta = beta + torch.where(self.fired2, b2, torch.zeros_like(b2)) * g("retry_depth")
        boost = boost | (self.fired2 & (w2 >= 0) & (w2 < BOOST_STEPS))
        self.boost_steps = self.boost_steps + boost.to(self.dtype)
        self.step = self.step + 1
        return beta.clamp(0.0, 1.0), boost, self.fire_now


def replay(spec: dict, x, signals: dict, centre_tolerance: float = 0.08) -> dict:
    """A structure's outputs on recorded signal histories ``{"rate", "centre", "transverse"}``, each ``(steps, cases)``;
    an optional ``"done"`` array replays the live controller's resets at episode ends."""
    rate = torch.as_tensor(np.asarray(signals["rate"]), dtype=torch.float32)
    n = rate.shape[1]
    st = ModuleState(n, rate.device, centre_tolerance=centre_tolerance)
    st.set(full_params(spec, x), ops_vector(spec)[None])
    centre = torch.as_tensor(np.asarray(signals["centre"]), dtype=torch.float32)
    tr = torch.as_tensor(np.asarray(signals["transverse"]), dtype=torch.float32)
    done = None if signals.get("done") is None else torch.as_tensor(np.asarray(signals["done"]), dtype=torch.bool)
    betas, boosts, fired = [], [], torch.zeros(n, dtype=torch.bool)
    for t in range(rate.shape[0]):
        b, k, f = st(rate[t], centre[t], tr[t])
        betas.append(b.clone())
        boosts.append(k.clone())
        fired |= f
        if done is not None:                                 # the live controller's per-environment reset at an episode's end
            st.reset(done[t])
    return {"beta": torch.stack(betas).numpy(), "boost": torch.stack(boosts).numpy(), "fired": fired.numpy()}


def replay_identity(parent: dict, child_spec: dict, signals: dict, tol: float = 1e-6) -> dict:
    """The child at its inherited coordinates gives the parent's outputs on every recorded step (a checked
    equivalence on these histories, not a proof for every history)."""
    a = replay(parent["spec"], parent["x"], signals)
    b = replay(child_spec, inherit(parent, child_spec), signals)
    diff = float(np.abs(a["beta"] - b["beta"]).max())
    flips = int((a["boost"] != b["boost"]).sum())
    return {"max_abs_beta_difference": diff, "boost_mismatches": flips, "cases": int(a["beta"].shape[1]),
            "parent_fired": int(a["fired"].sum()), "passes": bool(diff <= tol and flips == 0)}


LANGUAGE_TEXT = """A recovery module watches the pen while a fixed root controller rolls it. It can fire ONCE per episode. When it fires
it blends the sent action toward a target posture (weight ramps up over ramp_in steps, then down over ramp_out steps,
with a per-finger-group depth), then multiplies the root's stroke amplitude for 60 steps ("boost"). The signals it
may use, all computed from the observation every step (60 steps per second, 360-step episode, success is judged on the
last 60 steps): mean |axial rate| of the last 30 steps (rad/s; the task needs > 4), the pen's distance from the
in-hand position (it must be < 0.08 m to fire), the mean transverse (end-over-end) rate of the last 30 steps,
"trend" = mean rate of the newest 15 steps minus the 15 before, and "deficit" = a leaky average of max(0, 4 - rate)/4.

The TEMPLATE structure is
  {"schedule": "single", "conditions": [], "support_end": "timed", "boost_end": "timed", "target": "reset", "attempts": 1}
with 11 trained parameters: threshold (fire if mean rate < threshold), decision_step (the one step at which it
checks), ramp_in, ramp_out, six group depths (wrist, four fingers, thumb) and stroke_multiplier.

An EDIT changes exactly ONE component (each component at most once per structure; at most 2 extra conditions):
  {"component": "schedule", "value": "periodic"}    re-check the same condition every `period` steps after decision_step,
                                                     up to `extra_window` steps later, until it fires          (+2 parameters)
  {"component": "schedule", "value": "two_step"}    one extra check `second_gap` steps after decision_step with its own,
                                                     lower threshold (threshold x second_scale)                (+2)
  {"component": "conditions", "add": {"primitive": P, "op": "lt"|"gt"}}   P in "transverse" | "trend" | "deficit": the
                                                     trigger also requires P below / above a trained level     (+1, deficit +2)
  {"component": "support_end", "value": "event"}    after the ramp-in, hold the full blend while the transverse rate is
                                                     above `settle_level`, for at most `max_hold` steps        (+2)
  {"component": "boost_end", "value": "event"}      after the 60 boost steps, keep boosting while the mean rate is below
                                                     `release_level`, for at most `max_extra` more steps      (+2)
  {"component": "target", "value": "midrange"}      blend toward a mix of the reset posture and the mid-range posture (+1)
  {"component": "target", "value": "hold"}          blend toward a mix of the reset posture and the root's own action at
                                                     the moment of firing                                      (+1)
  {"component": "attempts", "value": 2}             one more attempt `retry_gap` steps after the first ends, with depth
                                                     scaled by `retry_depth`, if still slow and centred        (+2)
Every new parameter starts at a value where the new component does nothing, so an edited module behaves exactly as
its parent until its parameters are trained (by a derivative-free search on the success count, a few updates)."""

__all__ = ["CHOICES", "EditError", "I", "LANGUAGE_TEXT", "LIMITS", "ModuleState", "NAMES", "N_SLOTS", "N_TEMPLATE", "PRIMITIVES", "SLOTS",
           "TEMPLATE", "active", "apply_edit", "canonical", "check_record", "edit_key", "full_params", "identity_values", "inherit",
           "initial_x", "n_edits", "ops_vector", "record", "replay", "replay_identity", "sample_edits", "template_record", "to_unit",
           "valid_edits"]
