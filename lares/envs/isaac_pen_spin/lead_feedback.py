"""R-064 (revised on the D-058 incumbent): spin feedback through the stroke's phase lead, not its amplitude.

The incumbent's periodic stroke has no tempo: its phase is the pen's own measured roll angle (the twist of
``obj_rot``) shifted by the scalar ``phase_lead``. The only timing lever is that lead. Here one audited substitution
makes the lead ``phase_lead + offset`` with ``offset`` either a constant (the open-loop controls) or
``clip(gain (target - y), -0.6, 0.6)`` with ``y`` a causal low-pass of the pen's signed axial rate (the feedback
settings). The identity form returns exactly ``0.0`` and must reproduce the incumbent bit for bit offline.

Twelve candidates and the parent on one frozen 40-case screen manifest, each in its own process; the winner per family
(sustained successes, fewer drops, index) frozen before confirmation; both winners and the parent on the confirmation
manifest's promotion (10-39) and fresh (40-119) cases, traces recorded.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin import channel_lesions as cl
from lares.envs.isaac_pen_spin import coordinate_repair as cr
from lares.envs.isaac_pen_spin import scalar_tuning as st
from lares.envs.isaac_pen_spin.learner_recovery import _load
from lares.envs.isaac_pen_spin.protocols import arm_summary, child_cell, failure_split
from lares.eval.manifest import EvaluationManifest

STUDY = "r064_lead_feedback"
STAGE = 3
FAMILIES = ("feedback", "constant")
PANELS = ("promotion", "fresh")
GAIN_THRESHOLD = 0.10
CONTROL_DT = 1.0 / 60.0
_LEAD = re.compile(r"        sin_lead = torch\.sin\(self\.phase_lead\)\n        cos_lead = torch\.cos\(self\.phase_lead\)\n")


def substituted_source(evaluated_source: str) -> str:
    """``lead = self.phase_lead + self._lead_offset(spin)`` replaces ``self.phase_lead`` inside sin_lead and cos_lead."""
    if len(_LEAD.findall(evaluated_source)) != 1:
        raise SystemExit("the incumbent source does not contain exactly one sin_lead / cos_lead pair")
    return _LEAD.sub("        lead = self.phase_lead + self._lead_offset(spin)\n"
                     "        sin_lead = torch.sin(lead)\n        cos_lead = torch.cos(lead)\n", evaluated_source)


def alpha_for(tau_s: float, dt: float = CONTROL_DT) -> float:
    return float(dt / (float(tau_s) + dt))


class LeadOffset:
    """``identity`` (exactly 0.0), ``constant`` (a fixed offset) or ``feedback`` (clipped gain times filtered rate error)."""

    def __init__(self, mode: str, constant: float = 0.0, target: float = 0.0, gain: float = 0.0,
                 alpha: float = 1.0, clip: float = 0.6):
        self.mode, self.constant, self.target, self.gain = mode, float(constant), float(target), float(gain)
        self.alpha, self.clip = float(alpha), float(clip)
        self.state = None
        self.sum, self.at_clip, self.count, self.rate_sum = 0.0, 0, 0, 0.0

    def reset(self):
        self.state = None

    def __call__(self, spin):
        import torch

        if self.mode == "identity":
            return 0.0
        if self.mode == "constant":
            off = torch.full_like(spin, self.constant)
        elif self.mode == "feedback":
            s = spin.detach()
            if self.state is None or self.state.shape != s.shape:
                self.state = s.clone()
            self.state = self.state + self.alpha * (s - self.state)
            off = torch.clamp(self.gain * (self.target - self.state), -self.clip, self.clip)
            with torch.no_grad():
                self.rate_sum += float(self.state.sum())
        else:
            raise ValueError(self.mode)
        with torch.no_grad():
            self.sum += float(off.sum())
            self.at_clip += int((off.abs() >= self.clip - 1e-6).sum())
            self.count += int(off.numel())
        return off

    def activity(self) -> dict:
        n = max(1, self.count)
        return {"mean_offset": self.sum / n if self.count else None, "at_clip_fraction": self.at_clip / n if self.count else None,
                "mean_filtered_rate": self.rate_sum / n if (self.count and self.mode == "feedback") else None, "values": self.count}


def candidate_names(protocol_cfg: dict) -> dict:
    """``{family: [arm names]}`` in index order."""
    fb = [f"fb_t{float(t):g}_g{float(g):+g}" for t in protocol_cfg["targets"] for g in protocol_cfg["gains"]]
    const = [f"const_{float(c):+g}" for c in protocol_cfg["offsets"]]
    return {"feedback": fb, "constant": const}


def parse_arm(arm: str) -> dict:
    if arm == "parent":
        return {"mode": "identity"}
    m = re.fullmatch(r"fb_t([0-9.]+)_g([+-][0-9.]+)", arm)
    if m:
        return {"mode": "feedback", "target": float(m.group(1)), "gain": float(m.group(2))}
    m = re.fullmatch(r"const_([+-][0-9.]+)", arm)
    if m:
        return {"mode": "constant", "constant": float(m.group(1))}
    raise ValueError(f"unknown arm {arm!r}")


class LeadController:
    """The incumbent with its phase lead offset; resets the rate filter at every episode start."""

    def __init__(self, asset_dir: str, arm: str, spec: dict, alpha: float, clip: float):
        from lares.envs.isaac_pen_spin.pipeline_child import SymbolicController
        from lares.envs.isaac_pen_spin.stroke_feedback import StrokeGain

        base = cr.incumbent_spec(asset_dir)
        src = substituted_source(base["source"])
        self.inner = SymbolicController({**base, "source": src}, cases.STAGES[STAGE].task_id)
        self.inner.policy._stroke_gain = StrokeGain("constant", base["stroke_gain_constant"])
        self.offset = LeadOffset(spec["mode"], spec.get("constant", 0.0), spec.get("target", 0.0), spec.get("gain", 0.0), alpha, clip)
        self.inner.policy._lead_offset = self.offset
        self.name = arm
        self._params = {**self.inner.params, "arm": arm, **spec, "alpha": alpha, "clip": clip,
                        "stroke_gain_constant": base["stroke_gain_constant"], "substitution": "lead = phase_lead + offset(spin)"}

    @property
    def params(self) -> dict:
        return {**self._params, "offset_activity": self.offset.activity()}

    def reset(self, ctx):
        self.offset.reset()
        self.inner.reset(ctx)

    def __call__(self, obs, t, ctx):
        return self.inner(obs, t, ctx)


def make_controller(arm: str, protocol_cfg: dict, out_dir: str | None = None):
    if arm == "parent":
        return cr.incumbent_controller(protocol_cfg["incumbent_asset"])
    return LeadController(protocol_cfg["incumbent_asset"], arm, parse_arm(arm), alpha_for(float(protocol_cfg["filter_tau_s"])),
                          float(protocol_cfg["clip"]))


def offline_identity_gate(protocol_cfg: dict, archive_npz: str, max_steps: int | None = None) -> dict:
    import torch

    data = np.load(archive_npz)
    obs, length = data["obs"], data["length"]
    parent = cr.incumbent_controller(protocol_cfg["incumbent_asset"])
    ident = LeadController(protocol_cfg["incumbent_asset"], "identity", {"mode": "identity"}, 1.0, 0.6)
    mism, rows = [], 0
    for i in range(obs.shape[0]):
        L = int(min(length[i], obs.shape[1]))
        if max_steps:
            L = min(L, int(max_steps))
        if L <= 0:
            continue
        o = torch.as_tensor(obs[i, :L], dtype=torch.float32)
        rows += L
        if not torch.equal(parent(o, 0, None), ident(o, 0, None)):
            mism.append(i)
    return {"archive": archive_npz, "rows": rows, "mismatch_cases": mism, "passes": not mism}


def manifest_files(protocol_cfg: dict) -> dict:
    return {"screen": protocol_cfg["screen_manifest"], "confirmation": protocol_cfg["confirmation_manifest"]}


def roles(protocol_cfg: dict) -> dict:
    return {k: (int(v[0]), int(v[1])) for k, v in protocol_cfg["roles"].items()}


def build_manifests(protocol_cfg: dict) -> list[str]:
    from lares.eval.manifest import assert_disjoint

    files = manifest_files(protocol_cfg)
    specs = [(files["screen"], cases.build_manifest(STAGE, int(protocol_cfg["screen_seed"]), int(protocol_cfg["screen_cases"]), label="r064_screen"), "screen"),
             (files["confirmation"], cases.build_manifest(STAGE, int(protocol_cfg["confirmation_seed"]), int(protocol_cfg["confirmation_cases"]), label="r064_confirmation"), "confirmation")]
    before = {p for p, _, _ in specs if os.path.exists(p)}
    cases.freeze_manifest_set(specs)
    assert_disjoint([m for _, m, _ in specs] + [EvaluationManifest.load(p) for p in protocol_cfg.get("disjoint_from", [])])
    return [p for p, _, _ in specs if p not in before]


def screen_key(m: dict, index: int) -> tuple:
    return (-int(m["successes"]), int(m["drops"]), int(index))


def _panel(rows_a, rows_b) -> dict:
    c = st.contrast(rows_a, rows_b)
    return {**c, "drops_not_increased": bool(c["drop_counts"][0] <= c["drop_counts"][1]),
            "drop_increase_established": bool(c["drops"]["low"] is not None and c["drops"]["low"] > 0.0)}


def aggregate(rows: dict, sats: dict, screens: dict, winners: dict, activity: dict) -> dict:
    ctrls = ("parent", "feedback", "constant")
    out = {"controllers": {c: {p: {**arm_summary(rows[c][p]), **failure_split(rows[c][p]), **cl.tail_failure_split(rows[c][p]),
                                  "saturation": {k: v for k, v in sats[c][p].items() if k != "drive_unsaturated_fraction_per_case"},
                                  "offset_activity": activity.get(c, {}).get(p)}
                              for p in PANELS} for c in ctrls},
           "screens": screens, "winners": winners, "contrasts": {}}
    for p in PANELS:
        out["contrasts"][p] = {"feedback_minus_constant": _panel(rows["feedback"][p], rows["constant"][p]),
                               "feedback_minus_parent": _panel(rows["feedback"][p], rows["parent"][p]),
                               "constant_minus_parent": _panel(rows["constant"][p], rows["parent"][p])}
    t = out["contrasts"]["fresh"]["feedback_minus_constant"]["success"]
    out["paired_success_difference"] = t["mean"]
    out["prediction"] = {"threshold": GAIN_THRESHOLD,
                         "supported": bool(t["mean"] is not None and t["mean"] >= GAIN_THRESHOLD and t["low"] is not None and t["low"] > 0.0),
                         "refuted": bool(t["high"] is not None and t["high"] < GAIN_THRESHOLD)}
    out["adoption"] = {}
    for fam in FAMILIES:
        pro, fr = out["contrasts"]["promotion"][f"{fam}_minus_parent"], out["contrasts"]["fresh"][f"{fam}_minus_parent"]
        out["adoption"][fam] = bool(pro["promotes"] and fr["success"]["low"] is not None and fr["success"]["low"] > 0.0
                                    and pro["drops_not_increased"] and fr["drops_not_increased"])
    return out


def _child(config_path: str, manifest: str, arm: str, out_path: str, record: bool) -> None:
    child_cell(config_path, manifest, arm, out_path, study=STUDY, record_obs=record, skip_existing=True)


def _activity(cell: str):
    with open(cell, encoding="utf-8") as f:
        return (json.load(f).get("controller_params") or {}).get("offset_activity")


def run_lead_feedback(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    from lares.utils.hashing import sha256_file

    asset = protocol_cfg["incumbent_asset"]
    inc = cr.check_asset(asset)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "lead_feedback_source.py"), "w", encoding="utf-8") as f:
        f.write(substituted_source(cr.incumbent_spec(asset)["source"]))
    gate_path = os.path.join(out_dir, "identity_gate.json")
    if os.path.exists(gate_path):
        with open(gate_path, encoding="utf-8") as f:
            gate = json.load(f)
    else:
        gate = offline_identity_gate(protocol_cfg, protocol_cfg["identity_archive"])
        with open(gate_path, "w", encoding="utf-8") as f:
            json.dump(gate, f, indent=2)
    if not gate["passes"]:
        raise SystemExit(f"offline identity gate failed, no episode run: {gate}")
    files = manifest_files(protocol_cfg)
    for p in files.values():
        if not os.path.exists(p):
            raise SystemExit(f"frozen manifest {p} missing; build and commit it before the run")
    conf = EvaluationManifest.load(files["confirmation"])
    role = cases.role_manifests(conf, roles(protocol_cfg), expected_cases=int(protocol_cfg["confirmation_cases"]), stage=STAGE)
    paths = {"screen": files["screen"]}
    for p, rm in role.items():
        paths[p] = os.path.join(out_dir, f"{p}.yaml")
        cases.freeze_manifest(paths[p], rm)
    names = candidate_names(protocol_cfg)

    # Screening: twelve candidates and the parent, one process each on the 40 screen cases.
    screens = {}
    for fam in FAMILIES:
        for i, arm in enumerate(names[fam]):
            cell = os.path.join(out_dir, "screen", f"{arm}.json")
            os.makedirs(os.path.dirname(cell), exist_ok=True)
            _child(config_path, paths["screen"], arm, cell, False)
            rows = _load(cell)[1]
            screens[arm] = {"family": fam, "index": i, **st.screen_metrics(rows), "offset_activity": _activity(cell)}
    pcell = os.path.join(out_dir, "screen", "parent.json")
    _child(config_path, paths["screen"], "parent", pcell, False)
    screens["parent"] = {"family": "parent", "index": -1, **st.screen_metrics(_load(pcell)[1])}
    winners = {fam: min(names[fam], key=lambda a: screen_key(screens[a], screens[a]["index"])) for fam in FAMILIES}
    wpath = os.path.join(out_dir, "winners.json")
    st._freeze_json(wpath, {fam: {"arm": winners[fam], "screen": {k: v for k, v in screens[winners[fam]].items() if k != "offset_activity"},
                                  "rule": "sustained successes, then fewer drops, then index"} for fam in FAMILIES}, "winners")

    # Confirmation: both winners and the parent on promotion and fresh, traces recorded.
    sd_state = __import__("torch").load(os.path.join(asset, "state.pt"), weights_only=True)
    params = {"spin_target": float(sd_state["spin_target"]), "speed_gain": float(sd_state["speed_gain"])}
    rows, sats, activity = {}, {}, {}
    for c, arm in (("parent", "parent"), ("feedback", winners["feedback"]), ("constant", winners["constant"])):
        rows[c], sats[c], activity[c] = {}, {}, {}
        for p in PANELS:
            cell = os.path.join(out_dir, "confirmation", f"{c}_{p}.json")
            os.makedirs(os.path.dirname(cell), exist_ok=True)
            _child(config_path, paths[p], arm, cell, True)
            trace, rows[c][p] = _load(cell)
            sats[c][p] = st.saturation(trace, params)
            activity[c][p] = _activity(cell)
    summary = aggregate(rows, sats, screens, {fam: winners[fam] for fam in FAMILIES}, activity)
    summary["identity_gate"] = gate
    summary["incumbent"] = {"asset": asset, "constant": inc["constant"], "files": inc["files"]}
    summary["filter"] = {"tau_s": float(protocol_cfg["filter_tau_s"]), "alpha": alpha_for(float(protocol_cfg["filter_tau_s"])),
                         "clip": float(protocol_cfg["clip"]), "targets": list(protocol_cfg["targets"]), "gains": list(protocol_cfg["gains"]),
                         "offsets": list(protocol_cfg["offsets"])}
    summary["manifests"] = {k: {"path": v, "sha256": sha256_file(v)} for k, v in files.items()}
    summary["episodes"] = int((len(names["feedback"]) + len(names["constant"]) + 1) * int(protocol_cfg["screen_cases"])
                              + 3 * sum(len(role[p]) for p in PANELS))
    summary["decision"] = {"adopt": [f for f in FAMILIES if summary["adoption"][f]],
                           "route": ("adopt " + ", ".join(f"{f} ({winners[f]})" for f in FAMILIES if summary["adoption"][f])
                                     if any(summary["adoption"].values()) else "keep the D-058 incumbent"),
                           "rule": "strict promotion against the parent on cases 10-39, fresh lower bound above zero, no more observed drops"}
    summary["limitation"] = "one block, one learner, one filter time constant; six settings per family"
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


run = run_lead_feedback

__all__ = ["FAMILIES", "GAIN_THRESHOLD", "LeadController", "LeadOffset", "PANELS", "STUDY", "aggregate", "alpha_for",
           "build_manifests", "candidate_names", "make_controller", "offline_identity_gate", "parse_arm", "run",
           "run_lead_feedback", "screen_key", "substituted_source"]
