"""R-075: low-pass filter the reactive capture curl instead of removing it.

D-054 split the parent's capture curl into a useful static part (a constant preload restores spin rate) and a harmful
reactive part (it doubles drops). A first-order low-pass on ``extra_curl`` alone keeps the slow component and removes
the impulses: ``y_t = y_{t-1} + alpha (x_t - y_{t-1})``, ``alpha = dt / (tau + dt)`` with ``tau`` = 100 ms at 60 Hz,
per environment, initialised to the first raw value at reset. One audited source substitution wraps the ``extra_curl``
expression in ``self._curl_filter(...)``; the wrapper with ``alpha = 1`` returns the raw value exactly (the offline
identity gate). Three arms (parent, exact zero-curl clamp, filtered curl) on one frozen 200-case manifest.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin import channel_lesions as cl
from lares.envs.isaac_pen_spin import scalar_tuning as st
from lares.envs.isaac_pen_spin.learner_recovery import _load, check_learner, make_learner
from lares.envs.isaac_pen_spin.protocols import arm_summary, child_cell, failure_split
from lares.eval.manifest import EvaluationManifest

STUDY = "r075_curl_bandwidth"
STAGE = 3
ARMS = ("parent", "zero_curl", "filtered_curl")
PANELS = ("promotion", "fresh")
GAIN_THRESHOLD = 0.10
CONTROL_DT = 1.0 / 60.0
_CURL = re.compile(r"extra_curl = self\.capture_curl \* \(\n(\s*)\(1\.0 - contact\) \+ 0\.75 \* \(1\.0 - retained\)\n(\s*)\)\n")


def alpha_for(tau_s: float, dt: float = CONTROL_DT) -> float:
    return float(dt / (float(tau_s) + dt))


def substituted_source(source: str) -> str:
    """Wrap the extra_curl expression in ``self._curl_filter(...)``; everything else is untouched."""
    if len(_CURL.findall(source)) != 1:
        raise SystemExit("the parent source does not contain exactly one capture-curl expression")
    return _CURL.sub(lambda m: f"extra_curl = self._curl_filter(self.capture_curl * (\n{m.group(1)}(1.0 - contact) + 0.75 * (1.0 - retained)\n{m.group(2)}))\n", source)


class CurlFilter:
    """Per-row first-order low-pass with state; ``alpha == 1`` passes the raw value through exactly."""

    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.state = None
        self.raw_sum, self.filtered_sum, self.count = 0.0, 0.0, 0

    def reset(self):
        self.state = None

    def __call__(self, x):
        import torch

        if self.alpha >= 1.0:
            y = x
        else:
            if self.state is None or self.state.shape != x.shape:
                self.state = x.detach().clone()
            self.state = self.state + self.alpha * (x.detach() - self.state)
            y = self.state
        with torch.no_grad():
            self.raw_sum += float(x.sum())
            self.filtered_sum += float(y.sum())
            self.count += int(x.numel())
        return y


class FilteredCurlController:
    """The parent with ``extra_curl`` low-passed; resets the filter state at every episode start."""

    name = "filtered_curl"

    def __init__(self, protocol_cfg: dict, alpha: float, name: str):
        from lares.envs.isaac_pen_spin.layout import OBS_LAYOUT
        from lares.envs.isaac_pen_spin.pipeline_child import SymbolicController

        lc = check_learner(protocol_cfg)
        with open(lc["source"], encoding="utf-8") as f:
            source = f.read()
        self.inner = SymbolicController({"source": substituted_source(source), "state_path": lc["state"],
                                         "obs_dim": sum(size for _, size, _ in OBS_LAYOUT), "action_dim": cases.NUM_ACTIONS,
                                         "fixed_action_scale": float(lc["fixed_action_scale"]),
                                         "execution": "clamp(mean, -1, 1)"}, cases.STAGES[STAGE].task_id)
        self.filter = CurlFilter(alpha)
        self.inner.policy._curl_filter = self.filter
        self.name = name
        self._params = {**self.inner.params, "arm": name, "alpha": float(alpha), "filter": "first-order low-pass on extra_curl",
                        "rule": "y = y + alpha (x - y), per environment, initialised to the first raw value at reset"}

    @property
    def params(self) -> dict:
        """Read after the rollout by ``run_cell``, so the curl means cover the episodes just run."""
        return {**self._params, "curl_means": self.curl_means()}

    def reset(self, ctx):
        self.filter.reset()
        self.inner.reset(ctx)

    def __call__(self, obs, t, ctx):
        return self.inner(obs, t, ctx)

    def curl_means(self) -> dict:
        n = max(1, self.filter.count)
        return {"raw_mean_curl": self.filter.raw_sum / n, "filtered_mean_curl": self.filter.filtered_sum / n, "values": n}


def manifest_path(protocol_cfg: dict) -> str:
    return protocol_cfg["manifest"]


def roles(protocol_cfg: dict) -> dict:
    return {k: (int(v[0]), int(v[1])) for k, v in protocol_cfg["roles"].items()}


def build_manifests(protocol_cfg: dict) -> list[str]:
    from lares.eval.manifest import assert_disjoint

    path = manifest_path(protocol_cfg)
    m = cases.build_manifest(STAGE, int(protocol_cfg["pool_seed"]), int(protocol_cfg["cases"]), label="r075_block")
    written = cases.freeze_manifest(path, m)
    assert_disjoint([m] + [EvaluationManifest.load(p) for p in protocol_cfg.get("disjoint_from", [])])
    return [path] if written else []


def make_controller(arm: str, protocol_cfg: dict, out_dir: str | None = None):
    if arm == "parent":
        return make_learner(protocol_cfg)
    if arm == "zero_curl":
        return cl.lesioned_learner(protocol_cfg, "curl_clamp", 0.0, arm)
    if arm == "filtered_curl":
        return FilteredCurlController(protocol_cfg, alpha_for(float(protocol_cfg["filter_tau_s"])), arm)
    raise ValueError(f"unknown arm {arm!r}; one of {ARMS}")


def offline_identity_gate(protocol_cfg: dict, archive_npz: str, max_steps: int | None = None) -> dict:
    import torch

    data = np.load(archive_npz)
    obs, length = data["obs"], data["length"]
    parent = make_learner(protocol_cfg)
    ident = FilteredCurlController(protocol_cfg, 1.0, "identity")
    mism, rows = [], 0
    for i in range(obs.shape[0]):
        L = int(min(length[i], obs.shape[1]))
        if max_steps:
            L = min(L, int(max_steps))
        if L <= 0:
            continue
        ident.reset(None)
        o = torch.as_tensor(obs[i, :L], dtype=torch.float32)
        rows += L
        if not torch.equal(parent(o, 0, None), ident(o, 0, None)):
            mism.append(i)
    return {"archive": archive_npz, "rows": rows, "mismatch_cases": mism, "passes": not mism}


def _panel(rows_a, rows_b) -> dict:
    c = st.contrast(rows_a, rows_b)
    return {**c, "drops_not_increased": bool(c["drop_counts"][0] <= c["drop_counts"][1]),
            "drop_increase_established": bool(c["drops"]["low"] is not None and c["drops"]["low"] > 0.0)}


def aggregate(rows: dict, sats: dict) -> dict:
    out = {"arms": {a: {p: {**arm_summary(rows[a][p]), **failure_split(rows[a][p]), **cl.tail_failure_split(rows[a][p]),
                            "saturation": {k: v for k, v in sats[a][p].items() if k != "drive_unsaturated_fraction_per_case"}}
                        for p in PANELS} for a in ARMS},
           "contrasts": {p: {"filtered_curl_minus_parent": _panel(rows["filtered_curl"][p], rows["parent"][p]),
                             "filtered_curl_minus_zero_curl": _panel(rows["filtered_curl"][p], rows["zero_curl"][p]),
                             "zero_curl_minus_parent": _panel(rows["zero_curl"][p], rows["parent"][p])} for p in PANELS}}
    f = out["contrasts"]["fresh"]["filtered_curl_minus_parent"]
    t = f["success"]
    out["paired_success_difference"] = t["mean"]
    out["prediction"] = {"threshold": GAIN_THRESHOLD,
                         "supported": bool(t["mean"] is not None and t["mean"] >= GAIN_THRESHOLD and t["low"] is not None
                                           and t["low"] > 0.0 and not f["drop_increase_established"]),
                         "refuted": bool((t["high"] is not None and t["high"] < GAIN_THRESHOLD) or f["drop_increase_established"])}
    q = True
    for ctl in ("parent", "zero_curl"):
        pro = out["contrasts"]["promotion"][f"filtered_curl_minus_{ctl}"]
        fr = out["contrasts"]["fresh"][f"filtered_curl_minus_{ctl}"]
        q = q and pro["promotes"] and fr["success"]["low"] is not None and fr["success"]["low"] > 0.0 \
            and pro["drops_not_increased"] and fr["drops_not_increased"]
    out["qualifies_repair_parent"] = bool(q)
    return out


def _child(config_path: str, manifest: str, arm: str, out_path: str) -> None:
    child_cell(config_path, manifest, arm, out_path, study=STUDY, record_obs=True, skip_existing=True)


def run_curl_bandwidth(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    from lares.utils.hashing import sha256_file

    lc = check_learner(protocol_cfg)
    os.makedirs(out_dir, exist_ok=True)
    with open(lc["source"], encoding="utf-8") as f:
        source = f.read()
    audit = cl.source_audit(source)
    if not audit["passes"]:
        raise SystemExit(f"source audit failed, no episode run: {audit}")
    with open(os.path.join(out_dir, "filtered_curl_source.py"), "w", encoding="utf-8") as f:
        f.write(substituted_source(source))
    alpha = alpha_for(float(protocol_cfg["filter_tau_s"]))
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
    sd = make_learner(protocol_cfg).policy.state_dict()
    params = {"spin_target": float(sd["spin_target"]), "speed_gain": float(sd["speed_gain"])}
    rows, sats = {a: {} for a in ARMS}, {a: {} for a in ARMS}
    for p in PANELS:
        for a in ARMS:
            cell = os.path.join(out_dir, f"{a}_{p}.json")
            _child(config_path, paths[p], a, cell)
            trace, rows[a][p] = _load(cell)
            sats[a][p] = st.saturation(trace, params)
    summary = aggregate(rows, sats)
    summary["filter"] = {"tau_s": float(protocol_cfg["filter_tau_s"]), "alpha": alpha, "dt": CONTROL_DT}
    curl = {}
    for p in PANELS:
        with open(os.path.join(out_dir, f"filtered_curl_{p}.json"), encoding="utf-8") as f:
            curl[p] = json.load(f).get("controller_params", {}).get("curl_means")
    summary["curl_means"] = curl
    summary["identity_gate"] = gate
    summary["source_audit"] = audit
    summary["manifest"] = {"path": mpath, "sha256": sha256_file(mpath), "id": m.manifest_id}
    summary["episodes"] = len(ARMS) * sum(len(role[p]) for p in PANELS)
    summary["decision"] = {"repair_parent": ({"controller": "filtered_curl", "alpha": alpha,
                                              "source": os.path.join(out_dir, "filtered_curl_source.py")}
                                             if summary["qualifies_repair_parent"] else None),
                           "rule": "strict promotion against both controls, fresh lower bound above zero against both, "
                                   "no observed drop increase against either on either panel"}
    summary["limitation"] = "one block, one learner, one frozen time constant (100 ms); no other bandwidth was tried"
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


run = run_curl_bandwidth

__all__ = ["ARMS", "CONTROL_DT", "CurlFilter", "FilteredCurlController", "GAIN_THRESHOLD", "PANELS", "STUDY",
           "aggregate", "alpha_for", "build_manifests", "make_controller", "offline_identity_gate", "run",
           "run_curl_bandwidth", "substituted_source"]
