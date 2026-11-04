"""R-053: does the symbolic learner use signed axial-spin feedback? Sign reversal of the observed axial velocity.

The learner is the frozen R-051 learner (the R-046 block-3 promoted controller). A source audit, done before any
episode and recorded in the summary, shows it reads ``obj_angvel`` and drives the fingers by
``speed_gain * (spin_target - spin)`` with ``spin = omega . pen_axis``, so the sign of the axial component is a live
input. The intervention reflects the observed angular velocity across the plane normal to the pen's long axis,
``omega' = omega - 2 (omega . a) a`` with ``a`` the axis computed from the observed ``obj_rot``: the axial component
changes sign, the transverse component is kept, and no other observation column, simulator state, reward or timing
changes. ``obj_angvel`` is the only observation field that carries the pen's angular velocity, so nothing derived
needs updating.

One frozen 32-case Stage-3 manifest (seed 4400), three fresh processes with identical batches (D-034, D-035):
``learner``, ``learner_identity`` (the same learner behind the wrapper with the identity transform: Gate 1 requires
it bit-identical to ``learner`` on every case, else the reversal is not run), and ``axial_reversed``. The result is
dependence under an observation shift, not proof of a stabilising feedback law.
"""

from __future__ import annotations

import json
import os
import re


from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin.feedback_ablation import _arm_summary
from lares.envs.isaac_pen_spin.feedback_lesions import failure_split
from lares.envs.isaac_pen_spin.learner_recovery import _load, check_learner, identity_gate, make_learner
from lares.envs.isaac_pen_spin.protocols import build_single_manifest

LEARNER_AXIAL_SIGN = "learner_axial_sign"
STAGE = 3
ARMS = ("learner", "learner_identity", "axial_reversed")


def reflect_axial(obs, transform: str = "reverse"):
    """``obs`` with ``obj_angvel``'s component along the pen axis negated (``"reverse"``) or untouched (``"identity"``)."""
    import torch

    sl = cases.obs_slices()
    out = obs.clone()
    if transform == "identity":
        return out
    if transform != "reverse":
        raise ValueError(f"unknown transform {transform!r}")
    q = obs[:, sl["obj_rot"]]
    w, xyz = q[:, :1], q[:, 1:]
    ex = torch.zeros_like(xyz)
    ex[:, 0] = 1.0
    t = 2.0 * torch.cross(xyz, ex, dim=1)
    axis = ex + w * t + torch.cross(xyz, t, dim=1)
    axis = axis / axis.norm(dim=1, keepdim=True).clamp_min(1e-12)
    om = obs[:, sl["obj_angvel"]]
    out[:, sl["obj_angvel"]] = om - 2.0 * (om * axis).sum(dim=1, keepdim=True) * axis
    return out


class AxialSignController:
    """Feeds the learner the transformed observation; checks no column outside ``obj_angvel`` changed."""

    def __init__(self, inner, transform: str, name: str):
        self.inner = inner
        self.transform = transform
        self.name = name
        self.params = {"arm": name, "transform": transform, "field": "obj_angvel",
                       "rule": "omega - 2 (omega . a) a, a from the observed obj_rot", **(getattr(inner, "params", {}) or {})}

    def attach(self, env):
        if hasattr(self.inner, "attach"):
            self.inner.attach(env)

    def reset(self, ctx):
        self.inner.reset(ctx)

    def __call__(self, obs, t, ctx):
        import torch

        out = reflect_axial(obs, self.transform)
        sl = cases.obs_slices()["obj_angvel"]
        keep = torch.ones(obs.shape[1], dtype=torch.bool, device=obs.device)
        keep[sl] = False
        if not torch.equal(out[:, keep], obs[:, keep]):
            raise AssertionError("the transform changed an observation column outside obj_angvel")
        return self.inner(out, t, ctx)


def source_audit(protocol_cfg: dict) -> dict:
    """Static check, before any episode, that the learner reads ``obj_angvel`` and uses its axial projection."""
    with open(check_learner(protocol_cfg)["source"], encoding="utf-8") as f:
        src = f.read()
    reads = sorted(set(re.findall(r'obs_field\(obs, "(\w+)"\)', src)))
    lines = [ln.strip() for ln in src.splitlines() if re.search(r"\bomega\b|\bspin\b", ln)]
    return {"fields_read": reads, "reads_obj_angvel": "obj_angvel" in reads,
            "axial_projection": any(re.search(r"omega \* axis", ln) for ln in lines),
            "axial_drive": any("spin_target - spin" in ln for ln in lines),
            "lines": lines[:12]}


def manifest_path(protocol_cfg: dict) -> str:
    return protocol_cfg["manifest"]


def build_manifests(protocol_cfg: dict) -> list[str]:
    return build_single_manifest(protocol_cfg, "axial_sign", STAGE)


def make_controller(arm: str, protocol_cfg: dict, out_dir: str | None = None):
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; one of {ARMS}")
    learner = make_learner(protocol_cfg)
    return AxialSignController(learner, "reverse" if arm == "axial_reversed" else "identity", arm)


def _child(config_path: str, manifest: str, arm: str, out_path: str) -> None:
    from lares.envs.isaac_pen_spin import learner_recovery

    learner_recovery._child(config_path, manifest, arm, out_path, study=LEARNER_AXIAL_SIGN)


def aggregate(base_rows, rev_rows) -> dict:
    from lares.eval.stats import t_summary, win_loss_tie

    if [r["task_hash"] for r in base_rows] != [r["task_hash"] for r in rev_rows]:
        raise ValueError("reversed rows are not paired case for case with the learner")
    d = [float(x["sustained"]) - float(e["sustained"]) for x, e in zip(rev_rows, base_rows)]
    t = t_summary(d)
    return {"paired_success_difference": t["mean"], "reversed_minus_learner": t,
            "win_loss_tie": win_loss_tie(d),
            "arms": {"learner": {**_arm_summary(base_rows), **failure_split(base_rows)},
                     "axial_reversed": {**_arm_summary(rev_rows), **failure_split(rev_rows)}}}


def run_learner_axial_sign(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    """Driver: audit, learner, identity repeat, Gate 1, reversal, aggregate. Nothing is rerun if its output exists."""
    from lares.utils.hashing import sha256_file

    audit = source_audit(protocol_cfg)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "source_audit.json"), "w") as f:
        json.dump(audit, f, indent=2)
    if not (audit["reads_obj_angvel"] and audit["axial_projection"]):
        raise SystemExit(f"source audit: the learner does not use the axial angular velocity, no episode run: {audit}")
    m = manifest_path(protocol_cfg)
    if not os.path.exists(m):
        raise SystemExit(f"frozen manifest {m} missing; build and commit it before the run")
    out = {arm: os.path.join(out_dir, f"{arm}.json") for arm in ARMS}
    for arm in ("learner", "learner_identity"):
        _child(config_path, m, arm, out[arm])
    base, base_rows = _load(out["learner"])
    ident, ident_rows = _load(out["learner_identity"])
    gate = identity_gate(base, base_rows, ident, ident_rows)
    if not gate["passes"]:
        with open(os.path.join(out_dir, "gate_failed.json"), "w") as f:
            json.dump(gate, f, indent=2)
        raise SystemExit(f"identity gate failed, reversal not run: {gate}")
    _child(config_path, m, "axial_reversed", out["axial_reversed"])
    rev, rev_rows = _load(out["axial_reversed"])
    summary = aggregate(base_rows, rev_rows)
    summary["identity_gate"] = gate
    summary["source_audit"] = audit
    summary["manifest"] = {"path": m, "sha256": sha256_file(m)}
    summary["episodes"] = len(ARMS) * int(protocol_cfg["cases"])
    summary["limitation"] = ("sign reversal measures dependence under an observation shift for this learner; it does not "
                             "establish that the feedback stabilises spin, nor anything about other controllers")
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


run = run_learner_axial_sign

__all__ = ["ARMS", "AxialSignController", "LEARNER_AXIAL_SIGN", "aggregate", "build_manifests", "reflect_axial", "run",
           "run_learner_axial_sign", "source_audit"]
