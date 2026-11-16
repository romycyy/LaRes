"""R-089: qualify the frozen D-062 warp (epsilon -0.35) against the D-058 incumbent on one 90-pair instrument.

The same instrument that qualified the clamp (D-057): three independent frozen 40-case manifests, cases 10-39 of each,
parent and candidate each panel in its own process, one paired t interval over the 90 declared differences under the
standing rule. The candidate is resolved from the archived R-086 run (its substituted source and normalisation
constants, hash-checked); the study aborts if either is missing or differs, and never reconstructs it.
"""

from __future__ import annotations

import json
import os

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin import channel_lesions as cl
from lares.envs.isaac_pen_spin import clamp_qualification as cq
from lares.envs.isaac_pen_spin import coordinate_repair as cr
from lares.envs.isaac_pen_spin import phase_warp as pw
from lares.envs.isaac_pen_spin import scalar_tuning as st
from lares.envs.isaac_pen_spin.protocols import (arm_summary, cached_json, disjoint_from, drive_params, failure_split,
                                                 load_cell, roles, study_cell)

STUDY = "r089_warp_qualification_90"
STAGE = 3
ARMS = ("parent", "warp")
PANEL = "promotion"
GAIN_THRESHOLD = 0.05


def check_candidate(protocol_cfg: dict) -> dict:
    """The archived D-062 warp: its source must equal the substitution of the incumbent's evaluated source, its
    constants file must match the frozen hash, and the epsilon must be present in it."""
    from lares.utils.hashing import sha256_file, sha256_text

    cand = protocol_cfg["candidate"]
    for key in ("source", "constants"):
        if not os.path.exists(cand[key]):
            raise SystemExit(f"archived candidate {key} {cand[key]} missing; the candidate is never reconstructed")
        got = sha256_file(cand[key])
        if got != cand[f"{key}_sha256"]:
            raise SystemExit(f"archived candidate {key} hash {got} != frozen {cand[f'{key}_sha256']}")
    with open(cand["source"], encoding="utf-8") as f:
        archived = f.read()
    rebuilt = pw.substituted_source(cr.incumbent_spec(protocol_cfg["incumbent_asset"])["source"])
    if sha256_text(rebuilt) != sha256_text(archived):
        raise SystemExit("the archived warp source is not the substitution of the incumbent's evaluated source")
    with open(cand["constants"], encoding="utf-8") as f:
        constants = json.load(f)
    key = f"{float(cand['epsilon']):+g}"
    if key not in constants:
        raise SystemExit(f"epsilon {key} absent from the archived constants")
    return {"epsilon": float(cand["epsilon"]), "constants": constants[key], "source_sha256": cand["source_sha256"],
            "constants_sha256": cand["constants_sha256"]}


def make_controller(arm: str, protocol_cfg: dict, out_dir: str | None = None):
    if arm == "parent":
        return cr.incumbent_controller(protocol_cfg["incumbent_asset"])
    if arm == "warp":
        c = check_candidate(protocol_cfg)
        return pw.WarpController(protocol_cfg["incumbent_asset"], f"warp_{c['epsilon']:+g}", c["epsilon"], c["constants"])
    raise ValueError(f"unknown arm {arm!r}; one of {ARMS}")


def manifest_paths(protocol_cfg: dict) -> list[str]:
    return list(protocol_cfg["manifests"])


def build_manifests(protocol_cfg: dict) -> list[str]:
    from lares.eval.manifest import assert_disjoint

    specs = [(path, cases.build_manifest(STAGE, int(seed), int(protocol_cfg["cases"]), label="r089_block"))
             for path, seed in zip(manifest_paths(protocol_cfg), protocol_cfg["pool_seeds"])]
    written = [path for path, m in specs if cases.freeze_manifest(path, m)]
    assert_disjoint([m for _, m in specs] + disjoint_from(protocol_cfg))
    return written


def _panel(parent_rows, warp_rows, parent_sat, warp_sat) -> dict:
    c = st.contrast(warp_rows, parent_rows)
    return {**c, "arms": {"parent": {**arm_summary(parent_rows), **failure_split(parent_rows), **cl.tail_failure_split(parent_rows),
                                     "saturation": {k: v for k, v in parent_sat.items() if k != "drive_unsaturated_fraction_per_case"}},
                          "warp": {**arm_summary(warp_rows), **failure_split(warp_rows), **cl.tail_failure_split(warp_rows),
                                   "saturation": {k: v for k, v in warp_sat.items() if k != "drive_unsaturated_fraction_per_case"}}},
            "drops_not_increased": bool(c["drop_counts"][0] <= c["drop_counts"][1])}


def aggregate(parent_rows, warp_rows, parent_sat, warp_sat, per_panel: dict | None = None) -> dict:
    panel = _panel(parent_rows, warp_rows, parent_sat, warp_sat)
    t = panel["success"]
    return {"panel": panel, "per_panel": per_panel or {}, "pairs": len(parent_rows), "paired_success_difference": t["mean"],
            "prediction": {"threshold": GAIN_THRESHOLD,
                           "supported": bool(t["mean"] is not None and t["mean"] >= GAIN_THRESHOLD and t["low"] is not None and t["low"] > 0.0),
                           "refuted": bool(t["high"] is not None and t["high"] < GAIN_THRESHOLD)},
            "qualifies": bool(panel["promotes"]), "adopts": bool(panel["promotes"] and panel["drops_not_increased"])}


def run_warp_qualification(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    asset = protocol_cfg["incumbent_asset"]
    inc = cr.check_asset(asset)
    cand = check_candidate(protocol_cfg)
    os.makedirs(out_dir, exist_ok=True)
    gate = cached_json(os.path.join(out_dir, "identity_gate.json"),
                       lambda: pw.offline_identity_gate(protocol_cfg, protocol_cfg["identity_archive"]))
    if not gate["passes"]:
        raise SystemExit(f"offline identity gate failed, no episode run: {gate}")
    import torch

    params = drive_params(torch.load(os.path.join(asset, "state.pt"), weights_only=True))
    rows, sats_parts, manifests, per_panel, episodes = {a: [] for a in ARMS}, {a: [] for a in ARMS}, [], {}, 0
    for mpath in manifest_paths(protocol_cfg):
        m = cases.load_frozen(mpath)
        role = cases.role_manifests(m, roles(protocol_cfg), expected_cases=int(protocol_cfg["cases"]), stage=STAGE)
        tag = f"s{m.pool.seeds[0]}"
        ppath = os.path.join(out_dir, f"{PANEL}_{tag}.yaml")
        cases.freeze_manifest(ppath, role[PANEL])
        prow = {}
        for a in ARMS:
            cell = os.path.join(out_dir, f"{a}_{PANEL}_{tag}.json")
            study_cell(protocol_cfg, config_path, ppath, a, cell)
            trace, prow[a] = load_cell(cell)
            rows[a] += prow[a]
            sats_parts[a].append(st.saturation(trace, params))
        per_panel[tag] = _panel(prow["parent"], prow["warp"], sats_parts["parent"][-1], sats_parts["warp"][-1])
        manifests.append({**cases.manifest_record(mpath, m), "panel": list(roles(protocol_cfg)[PANEL])})
        episodes += len(ARMS) * len(role[PANEL])
    sats = {a: cq._merge_saturation(sats_parts[a]) for a in ARMS}
    summary = aggregate(rows["parent"], rows["warp"], sats["parent"], sats["warp"], per_panel)
    summary["identity_gate"] = gate
    summary["candidate"] = cand
    summary["incumbent"] = {"asset": asset, "constant": inc["constant"], "files": inc["files"]}
    summary["manifests"] = manifests
    summary["instrument"] = ("one paired t interval over every prospectively declared pair of the listed panels; "
                             "a one-off instrument of the existing strict rule, not a standing gate change")
    summary["episodes"] = episodes
    summary["decision"] = {"adopt": summary["adopts"], "route": "adopt the warp as the incumbent" if summary["adopts"] else "keep the D-058 incumbent",
                           "rule": "positive mean and lower bound above zero over the 90 pairs, and no more candidate drops"}
    summary["limitation"] = (f"{summary['pairs']} pairs; the paired standard error at this base rate is about "
                             f"{0.31 / summary['pairs'] ** 0.5:.2f}; no retry, no extension, no historical pooling")
    with open(os.path.join(out_dir, "qualification.json"), "w", encoding="utf-8") as f:
        json.dump({"study": STUDY, "pairs": summary["pairs"], "qualifies": summary["qualifies"], "adopts": summary["adopts"],
                   "paired_success_difference": summary["paired_success_difference"],
                   "interval": [summary["panel"]["success"]["low"], summary["panel"]["success"]["high"]],
                   "drop_counts_warp_parent": summary["panel"]["drop_counts"], "candidate": cand, "manifests": manifests}, f, indent=2)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


run = run_warp_qualification

__all__ = ["ARMS", "GAIN_THRESHOLD", "PANEL", "STUDY", "aggregate", "build_manifests", "check_candidate", "make_controller",
           "manifest_paths", "run", "run_warp_qualification"]
