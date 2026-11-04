"""R-060: re-rank the R-058 archive's top candidates on larger, independent screens.

D-048 found that both optimisers' ten-case screen winners (0.44 / 0.40 on their screen) fall to the parent's level on
promotion (0.233 / 0.237 vs 0.221): the selection, not the search, looked like the bottleneck. This study keeps the
archive (``logs/pen_spin/r058_scalar_cma_vs_random``: 64 screened candidates per arm per block, parameters and
ten-case metrics on disk) and changes only the selection: each arm's original top 8 by the frozen R-058 ranking (always
containing the original winner) is rescreened on 40 new cases, re-ranked by the same lexicographic rule, and the
re-ranked winner is compared with the original winner and the parent on 30 promotion and 40 fresh cases, one process
per controller per manifest (D-034, D-035). New block seed ``s'`` re-ranks R-058 block ``4510 + (s' - 4550)``.

The R-059 question (how much of a drive-override action difference final clipping removes) is answered offline from
the R-058 recorded traces by :func:`clipping_diagnostic`; it runs no episode.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin import scalar_tuning as st
from lares.envs.isaac_pen_spin.learner_recovery import check_learner
from lares.envs.isaac_pen_spin.protocols import arm_summary, child_cell, failure_split
from lares.eval.manifest import EvaluationManifest

STUDY = "r060_archive_rescreen"
STAGE = 3
ARMS = st.ARMS
CONTROLLERS = ("cma_reranked", "random_reranked", "cma_original", "random_original", "parent")
SHORTLIST = 8


# ---------------------------------------------------------------------------------------------------------------------
#  Archive
# ---------------------------------------------------------------------------------------------------------------------


def archive_block_seed(protocol_cfg: dict, seed: int) -> int:
    a = protocol_cfg["archive"]
    return int(a["block_seed_base"]) + (int(seed) - int(protocol_cfg["seeds"][0]))


def load_archive_arm(archive_dir: str, block_seed: int, arm: str, generations: int, popsize: int) -> dict:
    """Every candidate of one R-058 arm with its ten-case metrics, in candidate-index order, plus the file hashes."""
    from lares.utils.hashing import sha256_file

    adir = os.path.join(archive_dir, f"block_s{block_seed}", arm)
    params, metrics, files = [], [], {}
    for g in range(generations):
        cpath, spath = os.path.join(adir, f"gen{g}_candidates.json"), os.path.join(adir, f"gen{g}_screen.json")
        for p in (cpath, spath):
            if not os.path.exists(p):
                raise SystemExit(f"archive incomplete: {p} missing")
            files[os.path.relpath(p, archive_dir)] = sha256_file(p)
        with open(cpath, encoding="utf-8") as f:
            c = json.load(f)
        with open(spath, encoding="utf-8") as f:
            s = json.load(f)
        if len(c["candidates"]) != popsize or len(s["metrics"]) != popsize:
            raise SystemExit(f"archive generation {g} of {adir} does not hold {popsize} candidates")
        if s["candidate_indices"] != list(range(len(params), len(params) + popsize)):
            raise SystemExit(f"archive generation {g} of {adir}: candidate indices out of order")
        params += c["candidates"]
        metrics += s["metrics"]
    wpath = os.path.join(adir, "winner.json")
    if not os.path.exists(wpath):
        raise SystemExit(f"archive incomplete: {wpath} missing")
    files[os.path.relpath(wpath, archive_dir)] = sha256_file(wpath)
    with open(wpath, encoding="utf-8") as f:
        winner = json.load(f)
    return {"params": params, "metrics": metrics, "winner_index": int(winner["index"]), "files": files}


def shortlist(archive: dict, n: int = SHORTLIST) -> dict:
    """The archive's top ``n`` by the frozen R-058 ranking; refuses an archive whose winner the ranking does not reproduce."""
    ranks = st.rank_candidates(archive["metrics"])
    order = sorted(range(len(ranks)), key=lambda i: ranks[i])
    if order[0] != archive["winner_index"]:
        raise SystemExit(f"the frozen ranking puts candidate {order[0]} first but the archive's winner is "
                         f"{archive['winner_index']}; archive not reproduced")
    top = order[:n]
    return {"indices": top, "params": [archive["params"][i] for i in top],
            "screen_metrics": [archive["metrics"][i] for i in top], "original_winner_index": archive["winner_index"],
            "archive_files": archive["files"], "ranking": st.RANKING, "shortlist": n}


# ---------------------------------------------------------------------------------------------------------------------
#  Manifests and controllers
# ---------------------------------------------------------------------------------------------------------------------


def manifest_path(protocol_cfg: dict, seed: int) -> str:
    return protocol_cfg["manifest_pattern"].format(seed=int(seed))


def roles(protocol_cfg: dict) -> dict:
    return {k: (int(v[0]), int(v[1])) for k, v in protocol_cfg["roles"].items()}


def build_manifests(protocol_cfg: dict) -> list[str]:
    """One 110-case block manifest per seed, disjoint across blocks and from every manifest in ``disjoint_from``."""
    from lares.eval.manifest import assert_disjoint

    n = int(protocol_cfg["cases"])
    specs = [(manifest_path(protocol_cfg, s), cases.build_manifest(STAGE, int(s), n, label="r060_block"), f"block_s{s}")
             for s in protocol_cfg["seeds"]]
    before = {p for p, _, _ in specs if os.path.exists(p)}
    cases.freeze_manifest_set(specs)
    others = [EvaluationManifest.load(p) for p in protocol_cfg.get("disjoint_from", [])]
    assert_disjoint([m for _, m, _ in specs] + others)
    return [p for p, _, _ in specs if p not in before]


make_controller = st.make_controller


# ---------------------------------------------------------------------------------------------------------------------
#  Aggregation
# ---------------------------------------------------------------------------------------------------------------------


def block_summary(rows: dict, sats: dict, rescreen: dict, winners: dict) -> dict:
    """``rows[controller][role]``, ``sats[controller][role]`` for the five controllers on promotion and fresh."""
    out = {"controllers": {c: {role: {**arm_summary(rows[c][role]), **failure_split(rows[c][role]),
                                     "saturation": {k: v for k, v in sats[c][role].items()
                                                    if k != "drive_unsaturated_fraction_per_case"}}
                               for role in ("promotion", "fresh")} for c in CONTROLLERS},
           "rescreen": rescreen, "winners": winners, "promotion": {}, "fresh": {}}
    for arm in ARMS:
        r, o = f"{arm}_reranked", f"{arm}_original"
        same = winners[arm]["reranked_index"] == winners[arm]["original_index"]
        out["fresh"][f"{r}_minus_{o}"] = {**st.contrast(rows[r]["fresh"], rows[o]["fresh"]), "same_candidate": same}
        out["promotion"][f"{r}_minus_{o}"] = {**st.contrast(rows[r]["promotion"], rows[o]["promotion"]), "same_candidate": same}
        for c in (r, o):
            out["promotion"][f"{c}_minus_parent"] = st.contrast(rows[c]["promotion"], rows["parent"]["promotion"])
            out["fresh"][f"{c}_minus_parent"] = st.contrast(rows[c]["fresh"], rows["parent"]["fresh"])
            out["fresh"][f"{c}_feedback"] = st.feedback_contrast(sats[c]["fresh"], sats["parent"]["fresh"])
            fresh = out["fresh"][f"{c}_minus_parent"]
            out["fresh"][f"{c}_qualifies_repair_parent"] = bool(
                out["promotion"][f"{c}_minus_parent"]["promotes"] and fresh["success"]["low"] is not None
                and fresh["success"]["low"] > 0 and fresh["drop_counts"][0] <= fresh["drop_counts"][1])
    out["fresh"]["reranked_minus_original_mean_over_arms"] = float(np.mean(
        [out["fresh"][f"{arm}_reranked_minus_{arm}_original"]["success"]["mean"] for arm in ARMS]))
    return out


def summarise(blocks: list[dict]) -> dict:
    from lares.eval.stats import block_contrast

    def fs(b, c):
        return b["controllers"][c]["fresh"]["success"]

    def fd(b, c):
        return b["controllers"][c]["fresh"]["drops"]

    out = {"blocks": len(blocks), "block_seeds": [b["seed"] for b in blocks],
           "reranked_minus_original_fresh": block_contrast(b["fresh"]["reranked_minus_original_mean_over_arms"]
                                                           for b in blocks),
           "per_controller": {}, "promotions": {}, "qualifying_repair_parents": {}, "feedback_activated_blocks": {},
           "same_candidate_blocks": {}}
    out["paired_success_difference"] = out["reranked_minus_original_fresh"].get("mean")
    for arm in ARMS:
        r, o = f"{arm}_reranked", f"{arm}_original"
        out[f"{r}_minus_{o}_fresh"] = block_contrast(fs(b, r) - fs(b, o) for b in blocks)
        out[f"{r}_minus_{o}_fresh_drops"] = block_contrast(fd(b, r) - fd(b, o) for b in blocks)
        out["same_candidate_blocks"][arm] = [b["seed"] for b in blocks if b["fresh"][f"{r}_minus_{o}"]["same_candidate"]]
        for c in (r, o):
            out[f"{c}_minus_parent_fresh"] = block_contrast(fs(b, c) - fs(b, "parent") for b in blocks)
            out[f"{c}_minus_parent_fresh_drops"] = block_contrast(fd(b, c) - fd(b, "parent") for b in blocks)
            out[f"{c}_minus_parent_promotion"] = block_contrast(
                b["promotion"][f"{c}_minus_parent"]["success"]["mean"] for b in blocks)
            out["promotions"][c] = [b["seed"] for b in blocks if b["promotion"][f"{c}_minus_parent"]["promotes"]]
            out["qualifying_repair_parents"][c] = [b["seed"] for b in blocks if b["fresh"][f"{c}_qualifies_repair_parent"]]
            out["feedback_activated_blocks"][c] = [b["seed"] for b in blocks if b["fresh"][f"{c}_feedback"]["activated"]]
            out[f"{c}_feedback_unsaturated_gain"] = block_contrast(
                b["fresh"][f"{c}_feedback"]["unsaturated_drive_fraction"]["mean"] for b in blocks)
    for c in CONTROLLERS:
        out["per_controller"][c] = {
            role: {"success": block_contrast(b["controllers"][c][role]["success"] for b in blocks),
                   "successes_total": int(sum(b["controllers"][c][role]["successes"] for b in blocks)),
                   "drops_total": int(sum(b["controllers"][c][role]["drops"] for b in blocks)),
                   "drive_saturated_fraction": [b["controllers"][c][role]["saturation"]["drive_saturated_fraction"]
                                                for b in blocks],
                   "action_saturated_fraction": [b["controllers"][c][role]["saturation"]["action_saturated_fraction"]
                                                 for b in blocks]}
            for role in ("promotion", "fresh")}
    out["rescreen"] = {"parent_success_per_block": [b["rescreen"]["parent"]["successes"] / b["rescreen"]["cases"]
                                                    for b in blocks],
                       "shortlist_best_success_per_block": {arm: [max(m["successes"] for m in b["rescreen"][arm]["metrics"])
                                                                  / b["rescreen"]["cases"] for b in blocks]
                                                            for arm in ARMS},
                       "shortlist_mean_success_per_block": {arm: [float(np.mean([m["successes"] for m in b["rescreen"][arm]["metrics"]]))
                                                                  / b["rescreen"]["cases"] for b in blocks]
                                                            for arm in ARMS}}
    return out


def decision_artifact(summary: dict, blocks: list[dict]) -> dict:
    qualifying = [(b["seed"], c, b["controllers"][c]["fresh"]["success"], b["winners"][c.split("_")[0]])
                  for b in blocks for c in CONTROLLERS if c != "parent" and b["fresh"][f"{c}_qualifies_repair_parent"]]
    best = max(qualifying, key=lambda q: q[2]) if qualifying else None
    activated = {c: summary["feedback_activated_blocks"][c] for c in CONTROLLERS if c != "parent"}
    return {"rule": {"repair_parent": "strict promotion on the 30 promotion cases, fresh success interval lower bound "
                                      "above zero, and no more fresh drops than the parent, in the same block",
                     "feedback_activated": f"unsaturated in-play drive fraction up by at least {st.FEEDBACK_ACTIVATION} "
                                           "versus the parent on the fresh cases with the paired interval above zero"},
            "qualifying_repair_parents": [{"seed": s, "controller": c, "fresh_success": v, "winner": w}
                                          for s, c, v, w in qualifying],
            "repair_parent": None if best is None else {"seed": best[0], "controller": best[1], "fresh_success": best[2],
                                                        "winner": best[3],
                                                        "caveat": "chosen among qualifying winners by fresh success, "
                                                                  "which is a selection on the confirmation cases"},
            "feedback_activated_blocks": activated,
            "route": ("R-056 / R-061: repair the qualifying winner" if best is not None else
                      "R-057: anatomical-coordinate repair (R-043 re-scoped)" if any(activated.values()) else
                      "neither follow-up is enabled by this result")}


# ---------------------------------------------------------------------------------------------------------------------
#  R-059 offline diagnostic: clipping-induced loss of drive-override authority
# ---------------------------------------------------------------------------------------------------------------------

_DRIVE_RE = re.compile(r"drive = 0\.25 \+ 0\.75 \* torch\.tanh\(\s*self\.speed_gain \* \(self\.spin_target - spin\)\s*\)")
_CLIP_RE = re.compile(r"mean = torch\.clamp\(mean, -1\.0, 1\.0\)")


def instrumented_policy(protocol_cfg: dict, params: dict):
    """The frozen parent with ``_drive_delta`` added to its drive and the final clamp removed (pre-clip mean)."""
    from lares.envs.isaac_pen_spin.learner_recovery import make_learner

    learner = make_learner(protocol_cfg)
    with open(check_learner(protocol_cfg)["source"], encoding="utf-8") as f:
        src = f.read()
    if not _DRIVE_RE.search(src) or not _CLIP_RE.search(src):
        raise SystemExit("the parent source no longer matches the drive law / final clamp this diagnostic instruments")
    src = _DRIVE_RE.sub("drive = 0.25 + 0.75 * torch.tanh(self.speed_gain * (self.spin_target - spin)) + self._drive_delta",
                        src)
    src = _CLIP_RE.sub("mean = mean", src)
    import torch

    from lares.core.obs_schema import active_schema, get_obs_schema
    from lares.core.policy_generation import GENERATED_IMPORTS

    ns: dict = {}
    exec(GENERATED_IMPORTS + src, ns)
    with active_schema(get_obs_schema(cases.STAGES[STAGE].task_id)):
        policy = ns["GeneratedPolicy"](cases.OBS_DIM, cases.NUM_ACTIONS)
    policy.load_state_dict(learner.policy.state_dict())
    st.apply_params(policy, params)
    policy._drive_delta = 0.0
    policy.eval()
    return policy


def clipping_diagnostic(protocol_cfg: dict, traces: list, params_per_trace: list, delta: float = 0.10,
                        max_steps: int | None = None) -> dict:
    """Per axis, mean |a(d+delta) - a(d-delta)| before and after the final clamp, paired per episode.

    ``traces`` are recorded ``(obs, length)`` pairs (an ``.npz`` of a promotion or fresh cell); the reduction
    ``pre - post`` is averaged within each episode and the paired t interval is taken over episodes.
    """
    import torch

    from lares.eval.stats import t_summary

    pre_all, post_all, red = [], [], []
    for trace, params in zip(traces, params_per_trace):
        policy = instrumented_policy(protocol_cfg, params)
        obs, length = np.asarray(trace["obs"]), np.asarray(trace["length"])
        for i in range(obs.shape[0]):
            L = int(min(length[i], obs.shape[1]))
            if max_steps:
                L = min(L, int(max_steps))
            if L <= 0:
                continue
            o = torch.as_tensor(obs[i, :L], dtype=torch.float32)
            with torch.no_grad():
                policy._drive_delta = -float(delta)
                lo, _ = policy(o)
                policy._drive_delta = float(delta)
                hi, _ = policy(o)
            pre = (hi - lo).abs().mean(dim=0).numpy()
            post = (hi.clamp(-1, 1) - lo.clamp(-1, 1)).abs().mean(dim=0).numpy()
            pre_all.append(pre)
            post_all.append(post)
            red.append(pre - post)
    pre_all, post_all, red = np.stack(pre_all), np.stack(post_all), np.stack(red)
    by_axis = [t_summary(red[:, k]) for k in range(red.shape[1])]
    return {"delta": float(delta), "episodes": int(red.shape[0]),
            "action_variation_by_axis_pre_clip": pre_all.mean(axis=0).tolist(),
            "action_variation_by_axis_post_clip": post_all.mean(axis=0).tolist(),
            "reduction_by_axis": [{"mean": t["mean"], "low": t["low"], "high": t["high"]} for t in by_axis],
            "reduction_mean_over_axes": t_summary(red.mean(axis=1)),
            "reduction_max_axis": {"axis": int(np.argmax(red.mean(axis=0))), **by_axis[int(np.argmax(red.mean(axis=0)))]},
            "note": "R-059 offline: the parent's drive term shifted by -delta / +delta on the recorded in-play "
                    "observations; nothing executed, no episode run"}


# ---------------------------------------------------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------------------------------------------------


def _cell(config_path: str, manifest: str, controller: str, out_path: str, record_obs: bool) -> None:
    child_cell(config_path, manifest, controller, out_path, study=STUDY, record_obs=record_obs, skip_existing=True)


def run_block(seed: int, protocol_cfg: dict, config_path: str, out_dir: str, space: st.ParameterSpace) -> dict:
    from lares.utils.hashing import sha256_file

    a = protocol_cfg["archive"]
    bdir = os.path.join(out_dir, f"block_s{seed}")
    os.makedirs(bdir, exist_ok=True)
    mpath = manifest_path(protocol_cfg, seed)
    if not os.path.exists(mpath):
        raise SystemExit(f"frozen manifest {mpath} missing; build and commit it before the run")
    m = EvaluationManifest.load(mpath)
    role = cases.role_manifests(m, roles(protocol_cfg), expected_cases=int(protocol_cfg["cases"]), stage=STAGE)
    paths = {}
    for r, rm in role.items():
        paths[r] = os.path.join(bdir, f"{r}.yaml")
        cases.freeze_manifest(paths[r], rm)
    per_proc = int(protocol_cfg["rescreen_candidates_per_process"])
    n_short = int(protocol_cfg["shortlist"])
    if n_short % per_proc:
        raise SystemExit("shortlist must split evenly into rescreen processes")
    tiled_path = os.path.join(bdir, f"rescreen_tiled{per_proc}.yaml")
    cases.freeze_manifest(tiled_path, st.tiled_manifest(role["rescreen"], per_proc))

    archive_seed = archive_block_seed(protocol_cfg, seed)
    rescreen, winners = {"cases": len(role["rescreen"]), "archive_block_seed": archive_seed}, {}
    for arm in ARMS:
        adir = os.path.join(bdir, arm)
        os.makedirs(adir, exist_ok=True)
        sl = shortlist(load_archive_arm(a["log_dir"], archive_seed, arm, int(a["generations"]), int(a["popsize"])), n_short)
        st._freeze_json(os.path.join(adir, "shortlist.json"), sl, "shortlist")  # before any episode
        metrics = []
        for p in range(n_short // per_proc):
            chunk = sl["params"][p * per_proc:(p + 1) * per_proc]
            cpath = os.path.join(adir, f"rescreen{p}_candidates.json")
            st.write_candidates(cpath, f"{arm}:s{seed}:rescreen{p}", chunk,
                                {"shortlist_indices": sl["indices"][p * per_proc:(p + 1) * per_proc]})
            cell = os.path.join(adir, f"rescreen{p}.json")
            _cell(config_path, tiled_path, f"candidates:{st._rel(out_dir, cpath)}", cell, False)
            metrics += [st.screen_metrics(r) for r in st.split_rows(st._rows(cell), per_proc)]
        ranks = st.rank_candidates(metrics)
        best = ranks.index(0)
        rescreen[arm] = {"metrics": metrics, "ranks": ranks, "archive_indices": sl["indices"],
                         "original_screen_metrics": sl["screen_metrics"]}
        winners[arm] = {"reranked_shortlist_position": best, "reranked_index": sl["indices"][best],
                        "reranked_params": sl["params"][best], "reranked_rescreen": metrics[best],
                        "original_index": sl["original_winner_index"], "original_params": sl["params"][0],
                        "original_shortlist_position": 0}
        if sl["indices"][0] != sl["original_winner_index"]:
            raise SystemExit("shortlist position 0 is not the original winner")
    st._freeze_json(os.path.join(bdir, "winners.json"), winners, "winners")  # frozen before promotion or fresh

    files = {}
    for c in CONTROLLERS:
        if c == "parent":
            params = {}
        else:
            arm, kind = c.split("_")
            params = winners[arm][f"{kind}_params"]
        files[c] = os.path.join(bdir, f"{c}_params.json")
        st.write_candidates(files[c], f"{c}:s{seed}", [params])
    pcell = os.path.join(bdir, "parent_rescreen.json")
    _cell(config_path, paths["rescreen"], f"candidates:{st._rel(out_dir, files['parent'])}", pcell, False)
    rescreen["parent"] = st.screen_metrics(st._rows(pcell))

    rows, sats = {c: {} for c in CONTROLLERS}, {c: {} for c in CONTROLLERS}
    parent_full = space.params(space.normalise(space.parent))
    for r in ("promotion", "fresh"):
        for c in CONTROLLERS:
            cell = os.path.join(bdir, f"{c}_{r}.json")
            _cell(config_path, paths[r], f"candidates:{st._rel(out_dir, files[c])}", cell, True)
            rows[c][r] = st._rows(cell)
            with open(files[c], encoding="utf-8") as f:
                over = json.load(f)["candidates"][0]
            sats[c][r] = st.saturation(st._trace(cell), {**parent_full, **over})

    summary = block_summary(rows, sats, rescreen, winners)
    summary["seed"] = int(seed)
    summary["manifest"] = {"path": mpath, "sha256": sha256_file(mpath), "id": m.manifest_id}
    summary["episodes"] = (2 * n_short * len(role["rescreen"]) + len(role["rescreen"])
                           + len(CONTROLLERS) * (len(role["promotion"]) + len(role["fresh"])))
    with open(os.path.join(bdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    return summary


def run_study(protocol_cfg: dict, config_path: str, out_dir: str) -> dict:
    """Driver: archive preflight for every block, then the blocks, summary and decision artifact; resumable."""
    from lares.utils.hashing import sha256_file

    lc = check_learner(protocol_cfg)
    os.makedirs(out_dir, exist_ok=True)
    a = protocol_cfg["archive"]
    space = st.parameter_space(protocol_cfg)
    preflight = {}
    for s in protocol_cfg["seeds"]:
        bs = archive_block_seed(protocol_cfg, int(s))
        for arm in ARMS:
            sl = shortlist(load_archive_arm(a["log_dir"], bs, arm, int(a["generations"]), int(a["popsize"])),
                           int(protocol_cfg["shortlist"]))
            preflight[f"s{s}:{arm}"] = {"archive_block_seed": bs, "indices": sl["indices"],
                                        "original_winner_index": sl["original_winner_index"]}
    st._freeze_json(os.path.join(out_dir, "preflight.json"),
                    {"archive": a, "shortlist": int(protocol_cfg["shortlist"]), "parent": lc, "space": space.to_dict(),
                     "blocks": preflight, "ranking": st.RANKING}, "preflight")
    blocks = [run_block(int(s), protocol_cfg, config_path, out_dir, space) for s in protocol_cfg["seeds"]]
    summary = summarise(blocks)
    summary["episodes"] = int(sum(b["episodes"] for b in blocks))
    summary["manifests"] = {b["seed"]: b["manifest"] for b in blocks}
    summary["limitation"] = ("only each arm's original top 8 of 64 are re-ranked; 40 rescreen cases resolve differences "
                             "of about 0.15 per candidate, not 0.05; fresh cases confirm and never select")
    decision = decision_artifact(summary, blocks)
    with open(os.path.join(out_dir, "decision.json"), "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=1)
    summary["decision"] = decision
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    summary["preflight_sha256"] = sha256_file(os.path.join(out_dir, "preflight.json"))
    return summary


run = run_study

__all__ = ["ARMS", "CONTROLLERS", "SHORTLIST", "STUDY", "archive_block_seed", "block_summary", "build_manifests",
           "clipping_diagnostic", "decision_artifact", "instrumented_policy", "load_archive_arm", "make_controller",
           "run", "run_study", "shortlist", "summarise"]
