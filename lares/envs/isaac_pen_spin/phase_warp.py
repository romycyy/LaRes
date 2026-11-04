"""R-086 (re-nominated R-076): reshape the D-058 incumbent's stroke within each roll cycle.

The incumbent's stroke phase is the pen's roll angle ``phi = atan2(sin_roll, cos_roll)`` from the quaternion twist,
shifted by ``phase_lead`` and the five finger offsets. One audited substitution applies the warp
``h(phi) = phi + epsilon sin(phi)`` at the roll stage, and a second re-centres and rescales each finger's warped
``sin_phase`` / ``cos_phase`` channel to the unwarped channel's cycle mean and RMS, with constants computed offline on a
3600-point phase grid. Only the within-cycle shape of the stroke changes; amplitude, phase lead, compression (zero),
stroke gain, centring and grasp are unchanged. At ``epsilon = 0`` both substitutions return their inputs untouched
(``atan2`` followed by ``sin`` is not bit-exact), which is the offline identity gate.

Both warps and the parent on the 40 screen cases (one process each); the winner frozen; winner and parent on the
promotion (10-39) and fresh (80-159) cases of the same frozen 160-case manifest, traces recorded.
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

STUDY = "r086_phase_warp"
STAGE = 3
PANELS = ("promotion", "fresh")
GAIN_THRESHOLD = 0.10
GRID = 3600
FINGER_COS = np.array([1.0, 0.0, -1.0, 0.0, -1.0])
FINGER_SIN = np.array([0.0, 1.0, 0.0, -1.0, 0.0])
_ROLL = re.compile(r"        cos_roll = tw \* tw - tx \* tx\n        sin_roll = 2\.0 \* tw \* tx\n")
_PHASE = re.compile(r"        sin_phase = \(\n            sin_base \* self\.phase_cos \+ cos_base \* self\.phase_sin\n        \)\n"
                    r"        cos_phase = \(\n            cos_base \* self\.phase_cos - sin_base \* self\.phase_sin\n        \)\n")


def substituted_source(evaluated_source: str) -> str:
    if len(_ROLL.findall(evaluated_source)) != 1 or len(_PHASE.findall(evaluated_source)) != 1:
        raise SystemExit("the incumbent source does not contain exactly one roll stage and one phase stage")
    src = _ROLL.sub("        cos_roll, sin_roll = self._warp(tw * tw - tx * tx, 2.0 * tw * tx)\n", evaluated_source)
    return _PHASE.sub("        sin_phase, cos_phase = self._normalise(\n"
                      "            sin_base * self.phase_cos + cos_base * self.phase_sin,\n"
                      "            cos_base * self.phase_cos - sin_base * self.phase_sin\n        )\n", src)


def warp(phi, eps: float):
    return phi + float(eps) * np.sin(phi)


def normalisation_constants(eps: float, lead: float, grid: int = GRID) -> dict:
    """Per finger: the warped channels' cycle mean and RMS, and the unwarped RMS, on a uniform roll-angle grid."""
    phi = np.linspace(-np.pi, np.pi, int(grid), endpoint=False)
    base = warp(phi, eps) + float(lead)
    theta = np.arctan2(FINGER_SIN, FINGER_COS)  # 0, pi/2, pi, -pi/2, pi
    sp = np.sin(base[:, None] + theta[None, :])
    cp = np.cos(base[:, None] + theta[None, :])
    base0 = phi + float(lead)
    sp0 = np.sin(base0[:, None] + theta[None, :])
    cp0 = np.cos(base0[:, None] + theta[None, :])
    rms = lambda a: np.sqrt(np.mean(a * a, axis=0))
    return {"eps": float(eps), "lead": float(lead), "grid": int(grid),
            "sin_mean": sp.mean(axis=0).tolist(), "sin_rms": rms(sp).tolist(), "sin_rms0": rms(sp0).tolist(),
            "cos_mean": cp.mean(axis=0).tolist(), "cos_rms": rms(cp).tolist(), "cos_rms0": rms(cp0).tolist(),
            "sin_mean0": sp0.mean(axis=0).tolist(), "cos_mean0": cp0.mean(axis=0).tolist()}


class PhaseWarp:
    """``_warp`` and ``_normalise`` for one epsilon; at 0 both return their inputs untouched."""

    def __init__(self, eps: float, constants: dict | None):
        import torch

        self.eps = float(eps)
        self.constants = constants
        if self.eps != 0.0:
            c = constants
            self.sm = torch.tensor(c["sin_mean"], dtype=torch.float32).view(1, -1)
            self.ss = torch.tensor(np.array(c["sin_rms0"]) / np.array(c["sin_rms"]), dtype=torch.float32).view(1, -1)
            self.cm = torch.tensor(c["cos_mean"], dtype=torch.float32).view(1, -1)
            self.cs = torch.tensor(np.array(c["cos_rms0"]) / np.array(c["cos_rms"]), dtype=torch.float32).view(1, -1)
            self.sm0 = torch.tensor(c["sin_mean0"], dtype=torch.float32).view(1, -1)
            self.cm0 = torch.tensor(c["cos_mean0"], dtype=torch.float32).view(1, -1)

    def _warp(self, cos_roll, sin_roll):
        import torch

        if self.eps == 0.0:
            return cos_roll, sin_roll
        phi = torch.atan2(sin_roll, cos_roll)
        h = phi + self.eps * torch.sin(phi)
        return torch.cos(h), torch.sin(h)

    def _normalise(self, sin_phase, cos_phase):
        if self.eps == 0.0:
            return sin_phase, cos_phase
        dev = sin_phase.device
        s = (sin_phase - self.sm.to(dev)) * self.ss.to(dev) + self.sm0.to(dev)
        c = (cos_phase - self.cm.to(dev)) * self.cs.to(dev) + self.cm0.to(dev)
        return s, c


class WarpController:
    def __init__(self, asset_dir: str, arm: str, eps: float, constants: dict | None):
        from lares.envs.isaac_pen_spin.pipeline_child import SymbolicController
        from lares.envs.isaac_pen_spin.stroke_feedback import StrokeGain

        base = cr.incumbent_spec(asset_dir)
        self.inner = SymbolicController({**base, "source": substituted_source(base["source"])}, cases.STAGES[STAGE].task_id)
        self.inner.policy._stroke_gain = StrokeGain("constant", base["stroke_gain_constant"])
        w = PhaseWarp(eps, constants)
        self.inner.policy._warp = w._warp
        self.inner.policy._normalise = w._normalise
        self.name = arm
        self.params = {**self.inner.params, "arm": arm, "eps": float(eps), "stroke_gain_constant": base["stroke_gain_constant"],
                       "substitution": "h(phi) = phi + eps sin(phi) at the roll stage; per-finger channels re-centred and rescaled",
                       "constants_sha256": constants.get("sha256") if constants else None}

    def reset(self, ctx):
        self.inner.reset(ctx)

    def __call__(self, obs, t, ctx):
        return self.inner(obs, t, ctx)


def candidate_names(protocol_cfg: dict) -> list[str]:
    return [f"warp_{float(e):+g}" for e in protocol_cfg["epsilons"]]


def parse_arm(arm: str) -> float:
    m = re.fullmatch(r"warp_([+-][0-9.]+)", arm)
    if not m:
        raise ValueError(f"unknown arm {arm!r}")
    return float(m.group(1))


def frozen_lead(asset_dir: str) -> float:
    import torch

    return float(torch.load(os.path.join(asset_dir, "state.pt"), weights_only=True)["phase_lead"])


def freeze_constants(protocol_cfg: dict, out_dir: str) -> dict:
    """``constants.json``: the normalisation constants of every epsilon, with a hash, before any episode."""
    from lares.utils.hashing import sha256_text

    path = os.path.join(out_dir, "constants.json")
    lead = frozen_lead(protocol_cfg["incumbent_asset"])
    payload = {}
    for e in protocol_cfg["epsilons"]:
        c = normalisation_constants(float(e), lead)
        c["sha256"] = sha256_text(json.dumps({k: v for k, v in c.items()}, sort_keys=True))
        payload[f"{float(e):+g}"] = c
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
        if {k: v["sha256"] for k, v in old.items()} != {k: v["sha256"] for k, v in payload.items()}:
            raise SystemExit(f"{path} exists with different constants")
        return old
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    return payload


def make_controller(arm: str, protocol_cfg: dict, out_dir: str | None = None):
    if arm == "parent":
        return cr.incumbent_controller(protocol_cfg["incumbent_asset"])
    eps = parse_arm(arm)
    with open(os.path.join(out_dir, "constants.json"), encoding="utf-8") as f:
        constants = json.load(f)[f"{eps:+g}"]
    return WarpController(protocol_cfg["incumbent_asset"], arm, eps, constants)


def offline_identity_gate(protocol_cfg: dict, archive_npz: str, max_steps: int | None = None) -> dict:
    import torch

    data = np.load(archive_npz)
    obs, length = data["obs"], data["length"]
    parent = cr.incumbent_controller(protocol_cfg["incumbent_asset"])
    ident = WarpController(protocol_cfg["incumbent_asset"], "identity", 0.0, None)
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


def realised_channels(trace: dict, eps: float, lead: float, constants: dict | None) -> dict:
    """The per-finger phase channels' mean and RMS on recorded in-play steps, from the observed quaternion twist."""
    sl = cases.obs_slices()["obj_rot"]
    obs, length = np.asarray(trace["obs"]), np.asarray(trace["length"])
    rows = []
    for i in range(obs.shape[0]):
        L = int(min(length[i], obs.shape[1]))
        if L > 0:
            rows.append(obs[i, :L, sl].astype(np.float64))
    q = np.concatenate(rows) if rows else np.zeros((0, 4))
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    tw, tx = q[:, 0], q[:, 1]
    n = np.sqrt(tw * tw + tx * tx) + 1e-8
    tw, tx = tw / n, tx / n
    phi = np.arctan2(2.0 * tw * tx, tw * tw - tx * tx)
    base = warp(phi, eps) + lead
    theta = np.arctan2(FINGER_SIN, FINGER_COS)
    sp = np.sin(base[:, None] + theta[None, :])
    cp = np.cos(base[:, None] + theta[None, :])
    if eps != 0.0 and constants:
        sp = (sp - np.array(constants["sin_mean"])) * (np.array(constants["sin_rms0"]) / np.array(constants["sin_rms"])) + np.array(constants["sin_mean0"])
        cp = (cp - np.array(constants["cos_mean"])) * (np.array(constants["cos_rms0"]) / np.array(constants["cos_rms"])) + np.array(constants["cos_mean0"])
    return {"steps": int(len(phi)), "sin_mean": sp.mean(axis=0).tolist(), "sin_rms": np.sqrt((sp * sp).mean(axis=0)).tolist(),
            "cos_mean": cp.mean(axis=0).tolist(), "cos_rms": np.sqrt((cp * cp).mean(axis=0)).tolist()}


def manifest_path(protocol_cfg: dict) -> str:
    return protocol_cfg["manifest"]


def roles(protocol_cfg: dict) -> dict:
    return {k: (int(v[0]), int(v[1])) for k, v in protocol_cfg["roles"].items()}


def build_manifests(protocol_cfg: dict) -> list[str]:
    from lares.eval.manifest import assert_disjoint

    path = manifest_path(protocol_cfg)
    m = cases.build_manifest(STAGE, int(protocol_cfg["pool_seed"]), int(protocol_cfg["cases"]), label="r086_block")
    written = cases.freeze_manifest(path, m)
    assert_disjoint([m] + [EvaluationManifest.load(p) for p in protocol_cfg.get("disjoint_from", [])])
    return [path] if written else []


def screen_key(m: dict, index: int) -> tuple:
    return (-int(m["successes"]), int(m["drops"]), int(index))


def _panel(rows_a, rows_b) -> dict:
    c = st.contrast(rows_a, rows_b)
    return {**c, "drops_not_increased": bool(c["drop_counts"][0] <= c["drop_counts"][1]),
            "drop_increase_established": bool(c["drops"]["low"] is not None and c["drops"]["low"] > 0.0)}


def aggregate(rows: dict, sats: dict, screens: dict, winner: str, channels: dict) -> dict:
    out = {"controllers": {c: {p: {**arm_summary(rows[c][p]), **failure_split(rows[c][p]), **cl.tail_failure_split(rows[c][p]),
                                  "saturation": {k: v for k, v in sats[c][p].items() if k != "drive_unsaturated_fraction_per_case"},
                                  "channels": channels.get(c, {}).get(p)}
                              for p in PANELS} for c in ("parent", "winner")},
           "screens": screens, "winner": winner,
           "contrasts": {p: {"winner_minus_parent": _panel(rows["winner"][p], rows["parent"][p])} for p in PANELS}}
    t = out["contrasts"]["fresh"]["winner_minus_parent"]["success"]
    out["paired_success_difference"] = t["mean"]
    out["prediction"] = {"threshold": GAIN_THRESHOLD,
                         "supported": bool(t["mean"] is not None and t["mean"] >= GAIN_THRESHOLD and t["low"] is not None and t["low"] > 0.0),
                         "refuted": bool(t["high"] is not None and t["high"] < GAIN_THRESHOLD)}
    pro, fr = out["contrasts"]["promotion"]["winner_minus_parent"], out["contrasts"]["fresh"]["winner_minus_parent"]
    out["adopts"] = bool(pro["promotes"] and fr["success"]["low"] is not None and fr["success"]["low"] > 0.0
                         and pro["drops_not_increased"] and fr["drops_not_increased"])
    return out


def _child(config_path: str, manifest: str, arm: str, out_path: str, record: bool) -> None:
    child_cell(config_path, manifest, arm, out_path, study=STUDY, record_obs=record, skip_existing=True)


def run_phase_warp(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    from lares.utils.hashing import sha256_file

    asset = protocol_cfg["incumbent_asset"]
    inc = cr.check_asset(asset)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "phase_warp_source.py"), "w", encoding="utf-8") as f:
        f.write(substituted_source(cr.incumbent_spec(asset)["source"]))
    constants = freeze_constants(protocol_cfg, out_dir)
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
    mpath = manifest_path(protocol_cfg)
    if not os.path.exists(mpath):
        raise SystemExit(f"frozen manifest {mpath} missing; build and commit it before the run")
    m = EvaluationManifest.load(mpath)
    role = cases.role_manifests(m, roles(protocol_cfg), expected_cases=int(protocol_cfg["cases"]), stage=STAGE)
    paths = {}
    for p, rm in role.items():
        paths[p] = os.path.join(out_dir, f"{p}.yaml")
        cases.freeze_manifest(paths[p], rm)
    names = candidate_names(protocol_cfg)
    screens = {}
    for i, arm in enumerate(names + ["parent"]):
        cell = os.path.join(out_dir, "screen", f"{arm}.json")
        os.makedirs(os.path.dirname(cell), exist_ok=True)
        _child(config_path, paths["screen"], arm, cell, False)
        screens[arm] = {"index": i, **st.screen_metrics(_load(cell)[1])}
    winner = min(names, key=lambda a: screen_key(screens[a], screens[a]["index"]))
    st._freeze_json(os.path.join(out_dir, "winner.json"), {"arm": winner, "screen": screens[winner],
                                                           "rule": "sustained successes, then fewer drops, then candidate order"}, "winner")
    import torch

    sd = torch.load(os.path.join(asset, "state.pt"), weights_only=True)
    params = {"spin_target": float(sd["spin_target"]), "speed_gain": float(sd["speed_gain"])}
    lead = float(sd["phase_lead"])
    rows, sats, channels = {}, {}, {}
    for c, arm in (("parent", "parent"), ("winner", winner)):
        rows[c], sats[c], channels[c] = {}, {}, {}
        eps = 0.0 if c == "parent" else parse_arm(arm)
        for p in PANELS:
            cell = os.path.join(out_dir, "confirmation", f"{c}_{p}.json")
            os.makedirs(os.path.dirname(cell), exist_ok=True)
            _child(config_path, paths[p], arm, cell, True)
            trace, rows[c][p] = _load(cell)
            sats[c][p] = st.saturation(trace, params)
            channels[c][p] = realised_channels(trace, eps, lead, constants.get(f"{eps:+g}") if eps else None)
    summary = aggregate(rows, sats, screens, winner, channels)
    summary["identity_gate"] = gate
    summary["constants"] = {k: {"sha256": v["sha256"], "sin_mean": v["sin_mean"], "sin_rms": v["sin_rms"]} for k, v in constants.items()}
    summary["incumbent"] = {"asset": asset, "constant": inc["constant"], "files": inc["files"], "phase_lead": lead}
    summary["manifest"] = {"path": mpath, "sha256": sha256_file(mpath), "id": m.manifest_id}
    summary["episodes"] = int((len(names) + 1) * len(role["screen"]) + 2 * sum(len(role[p]) for p in PANELS))
    summary["decision"] = {"adopt": summary["adopts"], "route": f"adopt {winner}" if summary["adopts"] else "keep the D-058 incumbent",
                           "rule": "strict promotion against the parent on cases 10-39, fresh lower bound above zero, no more observed drops"}
    summary["limitation"] = "one block, one learner, two warp magnitudes; the warp is applied to the pen's own roll angle"
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


run = run_phase_warp

__all__ = ["GAIN_THRESHOLD", "GRID", "PANELS", "STUDY", "PhaseWarp", "WarpController", "aggregate", "build_manifests",
           "candidate_names", "freeze_constants", "make_controller", "normalisation_constants", "offline_identity_gate",
           "parse_arm", "realised_channels", "run", "run_phase_warp", "screen_key", "substituted_source", "warp"]
