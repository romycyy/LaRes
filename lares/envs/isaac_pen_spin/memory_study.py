"""R-112 (P3): does a slow memory of the spin deficit help where an instantaneous or lightly filtered one does not?

Three arms from the frozen R-106 finalist. Each keeps its 32 coefficients and adds the same 26 zero-initialised
ones, a posture offset per joint and a gain on the reference action per joint group, both multiplied by one gate::

    instant        gate = spin_deficit                      (the deficit this step)
    single_filter  gate = deficit_fast                      (through the reference's own ~11-step filter)
    two_timescale  gate = deficit_slow                      (a further 60-step average, 0 at reset)

Only the gate differs: the arms share placements and ARS direction streams. A memoryless gate cannot tell a normal
start (slow at step 20) from a stuck episode (slow at step 150); the slow state can. Training and evaluation reuse
R-109's Isaac child (``structure_search_child``: smoke, train, cell); this module is the driver and the contrasts.
"""

from __future__ import annotations

import json
import os

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin import structure_search as ss
from lares.envs.isaac_pen_spin import structure_terms as st
from lares.envs.isaac_pen_spin import symbolic_reward as sr
from lares.envs.isaac_pen_spin.protocols import freeze_json, load_cell

STUDY = "compact_memory_matched"
GATES = {"instant": "spin_deficit", "single_filter": "deficit_fast", "two_timescale": "deficit_slow"}
ARMS = tuple(GATES)
ROOT, REFERENCE = "r106", "reference"
CONTRASTS = (("two_timescale", "single_filter"), ("single_filter", "instant"), ("instant", ROOT), ("single_filter", ROOT),
             ("two_timescale", ROOT), ("instant", REFERENCE), ("single_filter", REFERENCE), ("two_timescale", REFERENCE),
             (ROOT, REFERENCE))
HOLM_FAMILY = tuple(f"{a}_minus_{ROOT}" for a in ARMS)


def arm_terms(arm: str) -> list:
    g = GATES[arm]
    return [{"expr": g, "target": "joints"}, {"expr": st.canonical(["mul", g, "ref"]), "target": "groups"}]


def arm_policy(cfg: dict, arm: str) -> dict:
    """The root's three terms and coefficients, bit for bit, plus the arm's two gated terms at zero."""
    root = ss.root_policy(cfg)
    terms = list(root["terms"]) + arm_terms(arm)
    return st.policy_record(terms, list(root["theta"]) + [0.0] * (st.n_coefficients(terms) - len(root["theta"])))


def confirmation_manifest(cfg: dict, seed) -> str:
    return str(cfg["confirmation_manifest_template"]).format(seed=int(seed))


def build_manifests(protocol_cfg: dict) -> list[str]:
    import glob

    from lares.eval.manifest import EvaluationManifest, assert_disjoint

    mine = {confirmation_manifest(protocol_cfg, s): cases.build_manifest(3, int(s), int(protocol_cfg["confirmation_cases"]), label="memory_confirm")
            for s in protocol_cfg["confirmation_seeds"][0]}
    others = [EvaluationManifest.load(p) for p in sorted(glob.glob("config/manifests/pen_spin_stage3_*.yaml")) if p not in mine]
    assert_disjoint(list(mine.values()) + others)
    return [p for p, m in mine.items() if cases.freeze_manifest(p, m)]


def make_controller(name: str, protocol_cfg: dict, out_dir: str | None = None):
    raise SystemExit("this study's cells run through structure_search_child (mode 'cell'), not the dispatcher's cell path")


def contrasts(rows: dict, threshold: float, drop_margin: float) -> dict:
    """``rows[manifest][policy]`` -> pooled and manifest-clustered summaries per contrast; Holm over the root comparisons."""
    from lares.envs.isaac_pen_spin.incumbent_residual import confirmation_contrast

    out = {f"{a}_minus_{b}": confirmation_contrast({m: (r[a], r[b]) for m, r in rows.items()}, threshold, drop_margin)
           for a, b in CONTRASTS}
    h = sr.holm({k: out[k]["pooled"]["success"]["p_value"] for k in HOLM_FAMILY})
    for k in HOLM_FAMILY:
        out[k]["holm"] = h[k]
        out[k]["useful_against_root"] = bool(h[k]["rejects"] and out[k]["pooled"]["adopts"])
    return out


def _dump(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)


def run(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    if os.path.normpath(out_dir) != os.path.normpath(str(protocol_cfg["study_dir"])):
        raise SystemExit("the study writes to its configured study_dir; do not override it")
    cfg = {**protocol_cfg, "config_path": config_path}
    os.makedirs(out_dir, exist_ok=True)
    descriptor = sr.check_reference(cfg)
    ss.normalization(cfg)
    root = ss.root_policy(cfg)
    policies = {arm: arm_policy(cfg, arm) for arm in ARMS}
    freeze_json(os.path.join(out_dir, "frozen_config.json"),
                {"config_sha256": ss.config_sha256(cfg), "config": ss.frozen_config(cfg), "reference": descriptor,
                 "root_policy_sha256": root["sha256"], "placements": ss.placement_provenance(cfg),
                 "initial_policies": {a: {"sha256": p["sha256"], "terms": p["terms"], "coefficients": len(p["theta"])} for a, p in policies.items()}},
                "frozen configuration")
    seeds = [int(s) for s in cfg["confirmation_seeds"][0]]
    manifests = [cases.require_frozen(confirmation_manifest(cfg, s)) for s in seeds]
    ceiling = int(cfg["start_ceiling"])
    names = (REFERENCE, ROOT, *ARMS)
    reserve = 2 * len(names) * len(manifests) * int(cfg["confirmation_cases"])
    updates = int(cfg["candidates"]) * int(cfg["updates_per_candidate"])

    smoke = ss.run_child(cfg, {"mode": "smoke", "response_path": os.path.join(out_dir, "smoke.json")}, os.path.join(out_dir, "smoke"))
    summary = {"study": STUDY, "config_sha256": ss.config_sha256(cfg), "smoke": smoke, "smoke_gate": "pass" if smoke["passes"] else "fail"}
    if not smoke["passes"]:
        summary["note"] = "the smoke gate failed: no training or confirmation ran"
        _dump(os.path.join(out_dir, "summary.json"), summary)
        return summary
    started, training, finalists = int(smoke["starts"]), {}, {}
    for arm in ARMS:
        stem = os.path.join(out_dir, f"train_{arm}")
        tr = ss.run_child(cfg, {"mode": "train", "block": 0, "arm": arm, "round": 0, "start_cap": ceiling - reserve - started,
                                "jobs": [{"policy": policies[arm], "updates": list(range(updates)), "snapshots": [updates]}],
                                "response_path": stem + ".json"}, stem)
        started += int(tr["starts"])
        if not tr["complete"]:
            summary["incomplete"] = f"the start ceiling was reached in the {arm} arm: no confirmation"
            _dump(os.path.join(out_dir, "summary.json"), summary)
            return summary
        finalists[arm] = tr["jobs"][0]["snapshots"][0]
        training[arm] = {"starts": tr["starts"], "transitions": tr["transitions"], "wall_time_s": tr["wall_time_s"],
                         "updates": tr["jobs"][0]["updates"]}
    freeze_json(os.path.join(out_dir, "finalists.json"),
                {"rule": "each arm's final ARS parameter mean; no checkpoint selection", "root": root, "finalists": finalists}, "finalists")

    rows = {}
    for mpath in manifests:
        mkey = os.path.basename(mpath).replace(".yaml", "")
        cdir = os.path.join(out_dir, "confirmation", mkey)
        os.makedirs(cdir, exist_ok=True)
        rows[mkey] = {}
        for name, pol in ((REFERENCE, REFERENCE), (ROOT, root), *((arm, finalists[arm]) for arm in ARMS)):
            cell = os.path.join(cdir, f"{name}_cell.json")
            if not os.path.exists(cell):
                ss.run_child(cfg, {"mode": "cell", "name": name, "policy": pol, "manifest": mpath, "cell_out": cell,
                                   "response_path": cell.replace("_cell.json", "_done.json")}, cell.replace("_cell.json", ""), seeded=False)
            if pol != REFERENCE and ss._load(cell)["controller_params"]["policy_sha256"] != pol["sha256"]:
                raise SystemExit(f"{cell}: the evaluated policy is not the frozen one")
            rows[mkey][name] = load_cell(cell)[1]
    c = contrasts(rows, float(cfg["threshold"]), float(cfg["drop_margin"]))
    head = c["two_timescale_minus_single_filter"]["pooled"]["success"]
    n_cells = len(names) * len(manifests)

    def new_part(arm):
        th = np.array(finalists[arm]["theta"])
        return {"sha256": finalists[arm]["sha256"], "theta_l2": float(np.linalg.norm(th)), "inherited_l2_change": float(np.linalg.norm(th[:32] - np.array(root["theta"]))),
                "posture_offsets_abs_max": float(np.abs(th[32:52]).max()), "posture_offsets_l2": float(np.linalg.norm(th[32:52])),
                "reference_gains": dict(zip(sr.GROUPS, [float(x) for x in th[52:58]]))}

    summary.update({
        "paired_success_difference": head["mean"], "paired_success_difference_ci95": [head["low"], head["high"]],
        "contrasts": c,
        "successes": {n: int(sum(r["sustained"] for m in rows.values() for r in m[n])) for n in names},
        "drops": {n: int(sum(not r["completed"] for m in rows.values() for r in m[n])) for n in names},
        "finalists": {arm: new_part(arm) for arm in ARMS}, "training": training,
        "accounting": {"simulator_starts_before_confirmation": started, "confirmation_cells": n_cells,
                       "confirmation_charged_starts": 2 * n_cells * int(cfg["confirmation_cases"]),
                       "charged_total": started + 2 * n_cells * int(cfg["confirmation_cases"]), "ceiling": ceiling, "llm_calls": 0},
        "limitation": "one matched set of ARS trajectories: a pilot, conditional on these runs",
    })
    _dump(os.path.join(out_dir, "summary.json"), summary)
    return summary


__all__ = ["ARMS", "CONTRASTS", "GATES", "STUDY", "arm_policy", "arm_terms", "build_manifests", "contrasts", "make_controller", "run"]
