"""R-077: qualify the exact D-052 zero-curl clamp on a fresh frozen 30-case promotion panel under the standing rule.

Three fresh panels (D-052, D-054, D-055) all favoured the clamp over the parent, but each 30-case promotion panel missed
the strict rule (positive paired difference with the t-interval lower bound above zero) by 0.006 to 0.02. Pooling does
not replace the rule, so this is the rule's own measurement: the parent and the exact clamp, each in its own process, on
cases 10-39 of a frozen 40-case manifest (cases 0-9 unused). No selection, no extension, no retry.
"""

from __future__ import annotations

import json
import os

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin import channel_lesions as cl
from lares.envs.isaac_pen_spin import curl_confirmation as cc
from lares.envs.isaac_pen_spin import scalar_tuning as st
from lares.envs.isaac_pen_spin.learner_recovery import learner_source
from lares.envs.isaac_pen_spin.protocols import (cached_json, disjoint_from, learner_drive_params, load_cell,
                                                 manifest_path, roles, study_cell)

STUDY = "r077_clamp_qualification"
STUDY_90 = "r080_zero_curl_qualification_90"
STAGE = 3
ARMS = cc.ARMS  # ("parent", "zero_curl")
PANEL = "promotion"
GAIN_THRESHOLD = 0.05


def manifest_paths(protocol_cfg: dict) -> list[str]:
    """R-077 names one manifest; R-080 names several (one per seed), each contributing its cases 10-39."""
    if "manifests" in protocol_cfg:
        return list(protocol_cfg["manifests"])
    return [manifest_path(protocol_cfg)]


def pool_seeds(protocol_cfg: dict) -> list[int]:
    return [int(s) for s in protocol_cfg["pool_seeds"]] if "pool_seeds" in protocol_cfg else [int(protocol_cfg["pool_seed"])]


def build_manifests(protocol_cfg: dict) -> list[str]:
    from lares.eval.manifest import assert_disjoint

    label = protocol_cfg.get("label", "r077_block")
    specs = [(path, cases.build_manifest(STAGE, seed, int(protocol_cfg["cases"]), label=label))
             for path, seed in zip(manifest_paths(protocol_cfg), pool_seeds(protocol_cfg))]
    written = [path for path, m in specs if cases.freeze_manifest(path, m)]
    assert_disjoint([m for _, m in specs] + disjoint_from(protocol_cfg))
    return written


make_controller = cc.make_controller


def aggregate(parent_rows, clamp_rows, parent_sat, clamp_sat, per_panel: dict | None = None) -> dict:
    """The instrument: one paired interval over every pair given (30 for R-077, 90 for R-080)."""
    panel = cc.panel_contrast(parent_rows, clamp_rows, parent_sat, clamp_sat)
    t = panel["success"]
    return {"panel": panel, "per_panel": per_panel or {}, "pairs": len(parent_rows), "paired_success_difference": t["mean"],
            "prediction": {"threshold": GAIN_THRESHOLD,
                           "supported": bool(t["mean"] is not None and t["mean"] >= GAIN_THRESHOLD and t["low"] is not None and t["low"] > 0.0),
                           "refuted": bool(t["high"] is not None and t["high"] < GAIN_THRESHOLD)},
            "qualifies": bool(panel["promotes"]),
            "repair_parent_eligible": bool(panel["promotes"] and panel["drops_not_increased"])}


def _merge_saturation(parts: list[dict]) -> dict:
    """Step-weighted merge of per-panel saturation summaries into one."""
    steps = sum(p["in_play_steps"] for p in parts)
    if not steps:
        return parts[0]
    w = [p["in_play_steps"] / steps for p in parts]
    axes = [sum(wi * p["action_saturated_by_axis"][k] for wi, p in zip(w, parts)) for k in range(len(parts[0]["action_saturated_by_axis"]))]
    return {"in_play_steps": steps,
            "drive_saturated_fraction": sum(wi * p["drive_saturated_fraction"] for wi, p in zip(w, parts)),
            "drive_unsaturated_fraction_per_case": sum((p["drive_unsaturated_fraction_per_case"] for p in parts), []),
            "action_saturated_fraction": sum(wi * p["action_saturated_fraction"] for wi, p in zip(w, parts)),
            "action_saturated_by_axis": axes, "note": parts[0]["note"]}


def run_qualification(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    source = learner_source(protocol_cfg)
    os.makedirs(out_dir, exist_ok=True)
    audit = cl.source_audit(source)
    if not audit["passes"]:
        raise SystemExit(f"source audit failed, no episode run: {audit}")
    with open(os.path.join(out_dir, "zero_curl_source.py"), "w", encoding="utf-8") as f:
        f.write(cl.substituted_source(source, cc.LESION, 0.0))
    gate = cached_json(os.path.join(out_dir, "identity_gate.json"),
                       lambda: cl.offline_identity_gate(protocol_cfg, protocol_cfg["identity_archive"]))
    if not gate["lesions"][cc.LESION]["passes"]:
        raise SystemExit(f"offline identity gate failed for {cc.LESION}, no episode run: {gate}")
    params = learner_drive_params(protocol_cfg)
    rows, sats_parts, manifests, per_panel = {a: [] for a in ARMS}, {a: [] for a in ARMS}, [], {}
    episodes = 0
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
        per_panel[tag] = cc.panel_contrast(prow["parent"], prow["zero_curl"], sats_parts["parent"][-1], sats_parts["zero_curl"][-1])
        manifests.append({**cases.manifest_record(mpath, m), "panel": list(roles(protocol_cfg)[PANEL])})
        episodes += len(ARMS) * len(role[PANEL])
    sats = {a: _merge_saturation(sats_parts[a]) for a in ARMS}
    summary = aggregate(rows["parent"], rows["zero_curl"], sats["parent"], sats["zero_curl"], per_panel)
    summary["identity_gate"] = gate["lesions"][cc.LESION]
    summary["source_audit"] = audit
    summary["manifest"] = manifests[0] if len(manifests) == 1 else None
    summary["manifests"] = manifests
    summary["instrument"] = ("one paired t interval over every prospectively declared pair of the listed panels; "
                             "a one-off instrument of the existing strict rule, not a standing gate change")
    summary["episodes"] = episodes
    summary["decision"] = {"repair_parent": ({"controller": "zero_curl", "source": os.path.join(out_dir, "zero_curl_source.py")}
                                             if summary["repair_parent_eligible"] else None),
                           "rule": "strict promotion on cases 10-39 (positive difference, lower bound above zero) and no observed drop increase"}
    summary["limitation"] = (f"{summary['pairs']} pairs; the paired standard error at this base rate is about "
                             f"{0.31 / summary['pairs'] ** 0.5:.2f}; no retry, no extension, no historical pooling")
    qpath = os.path.join(out_dir, "qualification.json")
    with open(qpath, "w", encoding="utf-8") as f:
        json.dump({"study": STUDY_90 if len(manifests) > 1 else STUDY, "pairs": summary["pairs"], "qualifies": summary["qualifies"],
                   "repair_parent_eligible": summary["repair_parent_eligible"], "paired_success_difference": summary["paired_success_difference"],
                   "interval": [summary["panel"]["success"]["low"], summary["panel"]["success"]["high"]],
                   "drop_counts_clamp_parent": summary["panel"]["drop_counts"], "manifests": manifests}, f, indent=2)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


run = run_qualification

__all__ = ["ARMS", "GAIN_THRESHOLD", "PANEL", "STUDY", "STUDY_90", "aggregate", "build_manifests", "make_controller",
           "manifest_paths", "pool_seeds", "run", "run_qualification"]
