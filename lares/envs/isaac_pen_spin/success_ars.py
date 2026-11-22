"""R-106: ARS on the exact sustained-success count, in R-103's 32-coefficient residual family. No LLM.

R-103 trained on a dense per-step reward and selected checkpoints on 64 cases; R-104 showed that mean reward can rise
through episodes that still fail and that 64 cases cannot rank policies a few points apart. Here the objective is the
episode rule itself and nothing is selected::

    return(candidate) = sustained successes on 512 shared placements / 512

Each ARS update is two rollouts of ``4 x 512`` environments in one persistent training process: a rollout holds two
antithetic pairs (``+d0, -d0, +d1, -d1``), every signed candidate on the **same** 512 placements (slot ``k`` is
environments ``512 k .. 512 k + 511``), placements written into the simulator as the evaluator writes them, each
environment scored on its first episode only. The final parameter mean is frozen and hashed, then confirmed once
against the reference on twelve fresh manifests through the established one-manifest-per-process evaluator.

The actor, the reference, ARS and the normalisation are R-103's, imported from ``symbolic_reward`` unchanged.
"""

from __future__ import annotations

import glob
import itertools
import json
import os
import subprocess
import sys
import time

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin import symbolic_reward as sr
from lares.envs.isaac_pen_spin.protocols import freeze_json, load_cell, study_cell
from lares.utils.hashing import sha256_file

STUDY = "symbolic_exact_success_ars"
STAGE = 3
REFERENCE, FINALIST = "reference", "finalist"
CHILD = "lares.envs.isaac_pen_spin.success_ars_child"
CHILD_TIMEOUT_S = 4 * 3600


# ---------------------------------------------------------------------------------------------------------------------
#  Placements and the slot layout
# ---------------------------------------------------------------------------------------------------------------------


def layout(protocol_cfg: dict) -> dict:
    tr = protocol_cfg["training"]
    slots, per = int(tr["slots"]), int(tr["cases_per_slot"])
    a = protocol_cfg["ars"]
    if 2 * int(a["directions"]) != slots * int(tr["rollouts_per_update"]) or int(a["top"]) > int(a["directions"]):
        raise SystemExit("the slot layout does not hold every signed candidate of an update exactly once")
    return {"slots": slots, "per": per, "num_envs": slots * per, "rollouts": int(tr["rollouts_per_update"]),
            "updates": int(a["iterations"]), "directions": int(a["directions"])}


def training_placements(protocol_cfg: dict, update: int) -> np.ndarray:
    lay = layout(protocol_cfg)
    allv = cases.placements(int(protocol_cfg["training"]["seed"]), lay["per"] * lay["updates"])
    return allv[lay["per"] * update:lay["per"] * (update + 1)]


def smoke_placements(protocol_cfg: dict) -> np.ndarray:
    return cases.placements(int(protocol_cfg["smoke"]["seed"]), layout(protocol_cfg)["per"])


def placement_hashes(vecs) -> list[str]:
    from lares.eval.manifest import _hash_rand_vec

    return [_hash_rand_vec(v) for v in vecs]


def placement_provenance(protocol_cfg: dict) -> dict:
    """Hashes of every training and smoke placement, checked disjoint from each other and from every committed
    pen-spin manifest (the MetaWorld splits, including the locked one, are never read)."""
    from lares.eval.manifest import EvaluationManifest

    lay = layout(protocol_cfg)
    train = [h for u in range(lay["updates"]) for h in placement_hashes(training_placements(protocol_cfg, u))]
    smoke = placement_hashes(smoke_placements(protocol_cfg))
    committed = set()
    for p in sorted(glob.glob("config/manifests/pen_spin_*.yaml")):
        committed |= {c.task_hash for c in EvaluationManifest.load(p).episodes}
    mine = train + smoke
    if len(set(mine)) != len(mine) or committed & set(mine):
        raise SystemExit("training or smoke placements collide with each other or with a committed manifest")
    from lares.utils.hashing import sha256_text

    return {"training_seed": int(protocol_cfg["training"]["seed"]), "training_cases": len(train),
            "training_sha256": sha256_text("".join(train)), "smoke_seed": int(protocol_cfg["smoke"]["seed"]),
            "smoke_cases": len(smoke), "smoke_sha256": sha256_text("".join(smoke)),
            "disjoint_from_committed_pen_spin_cases": len(committed)}


def tile(vecs: np.ndarray, slots: int) -> np.ndarray:
    """The same placements in every slot: ``(slots * n, d)``, slot-major."""
    return np.tile(np.asarray(vecs), (int(slots), 1))


def slot_thetas(candidates: np.ndarray, per: int) -> np.ndarray:
    """One candidate per slot: ``(slots * per, 32)``, slot-major."""
    return np.repeat(np.asarray(candidates, dtype=np.float64), int(per), axis=0)


def rollout_candidates(cand: np.ndarray, rollout: int, lay: dict) -> np.ndarray:
    """``ARS.ask`` returns rows ``+d0..+d(n-1), -d0..-d(n-1)``; rollout ``r`` holds the antithetic pairs of its
    directions side by side: ``+d, -d, +d', -d'``."""
    n, pairs = lay["directions"], lay["slots"] // 2
    dirs = range(rollout * pairs, (rollout + 1) * pairs)
    return np.stack([cand[i + s * n] for i in dirs for s in (0, 1)])


def ars_scores(slot_scores: list, lay: dict) -> np.ndarray:
    """Per-rollout slot scores back into ``ARS.tell`` order (all plus rows, then all minus rows)."""
    n, pairs = lay["directions"], lay["slots"] // 2
    out = np.zeros(2 * n)
    for r, sc in enumerate(slot_scores):
        for j in range(pairs):
            out[r * pairs + j], out[n + r * pairs + j] = sc[2 * j], sc[2 * j + 1]
    return out


def first_episode_scores(pred, length, completed) -> dict:
    """Each environment's first episode by the evaluator's own rule (``cases.sustained``): ended by time-out, and the
    predicate held on the final ``SUSTAIN_WINDOW`` steps of the episode's OWN recorded length. A time-out episode has
    ``HORIZON - 1`` recorded steps (the first R-106 smoke required 360 and scored every episode a failure)."""
    pred, length, completed = np.asarray(pred, dtype=bool), np.asarray(length, dtype=int), np.asarray(completed, dtype=bool)
    w = cases.SUSTAIN_WINDOW
    sus = np.array([cases.sustained(pred[:, i], int(length[i]), bool(completed[i])) for i in range(pred.shape[1])])
    tail = np.array([float(pred[:int(L), i][-w:].mean()) if L else 0.0 for i, L in enumerate(length)])
    return {"sustained": sus, "tail_hit": tail}


def aa_statistics(sustained) -> dict:
    """The noise floor of a training contrast: identical policies on identical placements in different slots."""
    s = np.asarray(sustained, dtype=bool)
    rates = s.mean(axis=1)
    pairs = list(itertools.combinations(range(s.shape[0]), 2))
    disc = [float((s[a] != s[b]).mean()) for a, b in pairs]
    diff = [float(rates[a] - rates[b]) for a, b in pairs]
    return {"slot_success": [float(x) for x in rates], "pairwise_discordance": disc,
            "mean_discordance": float(np.mean(disc)), "slot_differences": diff,
            "sd_of_slot_differences": float(np.std(diff, ddof=1)), "max_abs_slot_difference": float(np.max(np.abs(diff))),
            "cases": int(s.shape[1])}


# ---------------------------------------------------------------------------------------------------------------------
#  Manifests and evaluation controllers
# ---------------------------------------------------------------------------------------------------------------------


def confirmation_manifest(protocol_cfg: dict, seed) -> str:
    return str(protocol_cfg["confirmation_manifest_template"]).format(seed=int(seed))


def build_manifests(protocol_cfg: dict) -> list[str]:
    from lares.eval.manifest import EvaluationManifest, assert_disjoint

    mine = {confirmation_manifest(protocol_cfg, s): cases.build_manifest(
        STAGE, int(s), int(protocol_cfg["confirmation_cases"]), label="exact_success_confirm")
        for s in protocol_cfg["confirmation_seeds"]}
    others = [EvaluationManifest.load(p) for p in sorted(glob.glob("config/manifests/pen_spin_stage3_*.yaml")) if p not in mine]
    assert_disjoint(list(mine.values()) + others)
    return [p for p, m in mine.items() if cases.freeze_manifest(p, m)]


def normalization(protocol_cfg: dict) -> dict:
    """R-103's frozen feature normalisation (the family is identical), verified by hash."""
    path = protocol_cfg["normalization_path"]
    if sha256_file(path) != protocol_cfg["normalization_sha256"]:
        raise SystemExit(f"{path} is not the frozen R-103 normalisation")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_controller(name: str, protocol_cfg: dict, out_dir: str | None = None):
    if name == REFERENCE:
        return sr.build_reference(protocol_cfg, REFERENCE, resettable=False)
    if name != FINALIST:
        raise ValueError(f"unknown controller {name!r}")
    with open(os.path.join(out_dir or str(protocol_cfg["log_dir"]), "finalist.json"), encoding="utf-8") as f:
        entry = json.load(f)
    if sr.theta_sha256(entry["theta"]) != entry["theta_sha256"]:
        raise SystemExit("the stored finalist does not match its hash")
    return sr.build_controller(protocol_cfg, FINALIST, entry["theta"], normalization(protocol_cfg))


# ---------------------------------------------------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------------------------------------------------


def _dump(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_child(request: dict, request_path: str) -> dict:
    """One fresh Isaac process (smoke or the whole training run). A child that started and left no result is never
    relaunched from its seed."""
    if os.path.exists(request["response_path"]):
        print("exists", request["response_path"], flush=True)
        return _load(request["response_path"])
    marker = request["response_path"] + ".started"
    if os.path.exists(marker):
        raise SystemExit(f"{marker}: this child already started and left no result; it is not relaunched")
    _dump(request_path, request)
    _dump(marker, {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    log = request_path.replace(".request.json", ".log")
    cmd = [sys.executable, "-u", "-m", CHILD, request_path]
    print("running", " ".join(cmd), flush=True)
    with open(log, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=CHILD_TIMEOUT_S)
    if proc.returncode != 0 or not os.path.exists(request["response_path"]):
        with open(log, encoding="utf-8", errors="replace") as f:
            raise RuntimeError(f"child failed (exit {proc.returncode}); log {log}:\n{f.read()[-3000:]}")
    os.remove(marker)
    return _load(request["response_path"])


def run(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    from lares.envs.isaac_pen_spin.incumbent_residual import cell_row, confirmation_contrast

    if os.path.normpath(out_dir) != os.path.normpath(str(protocol_cfg["log_dir"])):
        raise SystemExit("evaluation cells read the finalist from the protocol's own log_dir; do not override it")
    os.makedirs(out_dir, exist_ok=True)
    descriptor = sr.check_reference(protocol_cfg)
    normalization(protocol_cfg)
    lay = layout(protocol_cfg)
    cfg_sha = sr.config_sha256(protocol_cfg)
    freeze_json(os.path.join(out_dir, "frozen_config.json"),
                {"config_sha256": cfg_sha, "config": sr.frozen_config(protocol_cfg), "reference": descriptor,
                 "placements": placement_provenance(protocol_cfg)}, "frozen configuration")
    seeds = [int(s) for s in protocol_cfg["confirmation_seeds"]]
    mpaths = {s: cases.require_frozen(confirmation_manifest(protocol_cfg, s)) for s in seeds}
    ceiling = int(protocol_cfg["start_ceiling"])
    reserve = 2 * 2 * len(seeds) * int(protocol_cfg["confirmation_cases"])       # two policies, charged at two starts
    base = {"config_path": os.path.abspath(config_path), "study": STUDY, "out_dir": out_dir, "config_sha256": cfg_sha}

    smoke = run_child({**base, "mode": "smoke", "response_path": os.path.join(out_dir, "smoke.json")},
                      os.path.join(out_dir, "smoke.request.json"))
    prior = int(protocol_cfg.get("prior_starts", 0))        # a failed earlier smoke attempt stays charged
    n_train = lay["updates"] * lay["rollouts"] * lay["num_envs"] * int(protocol_cfg["training"]["horizon"])
    projected_h = (n_train / smoke["throughput"]["transitions_per_s"] + 2 * len(seeds) * 40.0) / 3600.0
    ok = bool(smoke["passes"]) and projected_h <= float(protocol_cfg["max_hours"])
    summary = {"study": STUDY, "config_sha256": cfg_sha, "smoke": smoke, "projected_hours": projected_h,
               "smoke_gate": "pass" if ok else "fail", "layout": lay}
    if not ok:
        summary["note"] = "the smoke gate failed: no training or confirmation ran"
        _dump(os.path.join(out_dir, "summary.json"), summary)
        return summary

    train = run_child({**base, "mode": "train", "start_cap": ceiling - reserve - int(smoke["starts"]) - prior,
                       "response_path": os.path.join(out_dir, "train.json")}, os.path.join(out_dir, "train.request.json"))
    summary["training"] = {k: v for k, v in train.items() if k != "theta_history"}
    if not train["complete"]:
        summary["incomplete"] = "the start ceiling was reached during training: no finalist, no confirmation"
        _dump(os.path.join(out_dir, "summary.json"), summary)
        return summary
    freeze_json(os.path.join(out_dir, "finalist.json"),
                {"theta": train["final_theta"], "theta_sha256": train["final_theta_sha256"],
                 "rule": "the final ARS parameter mean; no checkpoint selection, no fallback to the reference"}, "finalist")

    dt = float(protocol_cfg["control_dt"])
    by_manifest, cells = {}, {}
    for s in seeds:
        sub = os.path.join(out_dir, "confirmation", f"s{s}")
        os.makedirs(sub, exist_ok=True)
        rows = {}
        for name in (REFERENCE, FINALIST):
            cell = os.path.join(sub, f"{name}_cell.json")
            study_cell(protocol_cfg, config_path, mpaths[s], name, cell, record_obs=False)
            if name == FINALIST and _load(cell)["controller_params"]["theta_sha256"] != train["final_theta_sha256"]:
                raise SystemExit(f"{cell}: the evaluated theta is not the frozen finalist")
            _, rows[name] = load_cell(cell)
            cells[f"s{s}/{name}"] = {k: v for k, v in cell_row(cell, dt).items() if k != "residual_activity"}
        by_manifest[f"s{s}"] = (rows[FINALIST], rows[REFERENCE])
    c = confirmation_contrast(by_manifest, float(protocol_cfg["threshold"]), float(protocol_cfg["drop_margin"]))
    p = c["pooled"]
    activity = _load(os.path.join(out_dir, "confirmation", f"s{seeds[0]}", f"{FINALIST}_cell.json"))["controller_params"].get("residual_activity")
    summary.update({
        "paired_success_difference": p["success"]["mean"],
        "paired_success_difference_ci95": [p["success"]["low"], p["success"]["high"]],
        "finalist_minus_reference": c, "cells": cells, "adopts": p["adopts"], "promotes": p["promotes"],
        "finalist": {"theta_sha256": train["final_theta_sha256"], "theta_abs_max": float(np.abs(train["final_theta"]).max()),
                     "theta_l2": float(np.linalg.norm(train["final_theta"])),
                     "theta": dict(zip(sr.COEF_NAMES, train["final_theta"]))},
        "finalist_residual_activity_first_manifest": activity,
        "accounting": {"prior_attempt_starts": prior, "smoke_starts": smoke["starts"], "training_starts": train["starts"],
                       "confirmation_cases": 2 * c["pairs"], "confirmation_charged_starts": 4 * c["pairs"],
                       "charged_total": prior + smoke["starts"] + train["starts"] + 4 * c["pairs"], "ceiling": ceiling,
                       "training_transitions": train["transitions"], "training_scored_episodes": train["scored_episodes"],
                       "llm_calls": 0},
        "limitation": "one ARS trajectory in one residual family: this learner configuration against the reference, not "
                      "ARS against other learners and not the exhaustion of the family",
    })
    _dump(os.path.join(out_dir, "summary.json"), summary)
    return summary


__all__ = ["FINALIST", "REFERENCE", "STUDY", "aa_statistics", "ars_scores", "build_manifests", "confirmation_manifest",
           "first_episode_scores", "layout", "make_controller", "normalization", "placement_hashes", "placement_provenance", "rollout_candidates",
           "run", "slot_thetas", "smoke_placements", "tile", "training_placements"]
