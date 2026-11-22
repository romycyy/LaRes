"""R-109 (P2): does structural evolution add value over continued fitting of a fixed structure? The driver.

Three independent blocks start from the frozen R-106 finalist (D-075). Per block and round every arm spends twelve
exact-success ARS updates (R-106's learner, unchanged) and submits two candidates to one selection gate:

* ``fixed``   -- one twelve-update continuation of the parent's structure; its snapshots after 6 and 12 updates.
* ``grammar`` -- two children of the parent, each the parent plus one term sampled from the edit language, six updates each.
* ``llm``     -- the same, the term proposed by the model in the same language (strict JSON, validated before any
  simulation, one linked correction; an edit still invalid becomes an inherited no-op candidate).

Edit generation reads only information one step behind selection: the smoke's parent sample in round 1, the retained
parent's sample and statistics from the previous round's selection in round 2. Selection is in-process on shared
placements (parent, candidate 0, candidate 1, parent again; three batches, rotated slots); a candidate replaces the
parent only through the strict paired rule with Holm over the two candidates and the drop bound. Finalists are
hashed before any confirmation case; confirmation is the established one-manifest-per-fresh-process evaluator.

The driver runs in ``.venv-metaworld`` (it makes the LLM calls) and launches every Isaac step as a child through
``protocols.isaac_shell``. ``structure_terms`` holds the actor and the language; ``structure_search_child`` the
simulator side.
"""

from __future__ import annotations

import glob
import json
import os
import re
import threading
import time

import numpy as np

from lares.envs.isaac_pen_spin import cases
from lares.envs.isaac_pen_spin import structure_terms as st
from lares.envs.isaac_pen_spin import symbolic_reward as sr
from lares.envs.isaac_pen_spin.protocols import freeze_json, isaac_shell, load_cell
from lares.utils.hashing import sha256_file, sha256_text

STUDY = "structure_search"
CONFIG_KEY = "pen_spin_structure"
ARMS = ("fixed", "grammar", "llm")
ROOT, REFERENCE = "r106", "reference"
CHILD = "lares.envs.isaac_pen_spin.structure_search_child"
#: keys the entry point folds into the block; they are not part of the frozen configuration
RUNTIME_KEYS = ("model", "reasoning_effort", "isaac_env", "config_path", "study")
#: the planned contrasts; the first is the registered prediction
CONTRASTS = (("llm", "grammar"), ("grammar", "fixed"), ("llm", "fixed"), ("fixed", ROOT), ("grammar", ROOT), ("llm", ROOT),
             ("fixed", REFERENCE), ("grammar", REFERENCE), ("llm", REFERENCE), (ROOT, REFERENCE))


# ---------------------------------------------------------------------------------------------------------------------
#  Frozen configuration, seeds and placements
# ---------------------------------------------------------------------------------------------------------------------


def frozen_config(cfg: dict) -> dict:
    return json.loads(json.dumps({k: v for k, v in cfg.items() if k not in RUNTIME_KEYS}, sort_keys=True))


def config_sha256(cfg: dict) -> str:
    return sha256_text(json.dumps(frozen_config(cfg), sort_keys=True, separators=(",", ":")))


def normalization(cfg: dict) -> dict:
    if sha256_file(cfg["normalization_path"]) != cfg["normalization_sha256"]:
        raise SystemExit(f"{cfg['normalization_path']} is not the frozen R-103 normalisation")
    with open(cfg["normalization_path"], encoding="utf-8") as f:
        return json.load(f)


def root_policy(cfg: dict) -> dict:
    """The R-106 finalist as the inherited three-term structure."""
    with open(cfg["root_finalist"], encoding="utf-8") as f:
        entry = json.load(f)
    if entry["theta_sha256"] != cfg["root_theta_sha256"] or sr.theta_sha256(entry["theta"]) != entry["theta_sha256"]:
        raise SystemExit("the root is not the frozen R-106 finalist")
    return st.base_policy(entry["theta"])


def schedule(cfg: dict) -> dict:
    k, u, r = int(cfg["candidates"]), int(cfg["updates_per_candidate"]), int(cfg["rounds"])
    if k != 2:
        raise SystemExit("the selection layout holds exactly two candidates")
    return {"candidates": k, "updates_per_candidate": u, "rounds": r, "updates_per_round": k * u,
            "slices": k * u * r, "per": int(cfg["layout"]["cases_per_slot"])}


def global_update(cfg: dict, candidate: int, local: int) -> int:
    """Structural candidate ``k``, local update ``i`` -> ``u = U k + i``; the fixed continuation uses ``0 .. K U - 1``."""
    return int(cfg["updates_per_candidate"]) * int(candidate) + int(local)


def training_placements(cfg: dict, block: int, rnd: int, u: int) -> np.ndarray:
    s = schedule(cfg)
    i = s["updates_per_round"] * int(rnd) + int(u)
    return cases.placements(int(cfg["block_seeds"][int(block)]), s["per"] * s["slices"])[s["per"] * i:s["per"] * (i + 1)]


def direction_seed(cfg: dict, block: int, rnd: int, u: int) -> int:
    return int(cfg["block_seeds"][int(block)]) + 2000 + 100 * (schedule(cfg)["updates_per_round"] * int(rnd) + int(u))


def selection_seeds(cfg: dict, block: int, rnd: int) -> list:
    return [int(s) for s in cfg["selection_seeds"][int(block)][int(rnd)]]


def grammar_seed(cfg: dict, block: int, rnd: int, k: int) -> int:
    return int(cfg["block_seeds"][int(block)]) + 9000 + 10 * int(rnd) + int(k)


def placement_provenance(cfg: dict) -> dict:
    """Every training, selection and smoke placement, checked disjoint from each other and from the committed manifests."""
    from lares.eval.manifest import EvaluationManifest, _hash_rand_vec

    s, per = schedule(cfg), int(cfg["layout"]["cases_per_slot"])
    groups = {f"train_b{b}": cases.placements(int(seed), per * s["slices"]) for b, seed in enumerate(cfg["block_seeds"])}
    for b in range(len(cfg["block_seeds"])):
        for r in range(s["rounds"]):
            for seed in selection_seeds(cfg, b, r):
                groups[f"select_s{seed}"] = cases.placements(seed, per)
    groups["smoke"] = cases.placements(int(cfg["smoke"]["seed"]), per)
    hashes = {k: [_hash_rand_vec(v) for v in vecs] for k, vecs in groups.items()}
    flat = [h for v in hashes.values() for h in v]
    committed = set()
    for p in sorted(glob.glob("config/manifests/pen_spin_*.yaml")):
        committed |= {c.task_hash for c in EvaluationManifest.load(p).episodes}
    if len(set(flat)) != len(flat) or committed & set(flat):
        raise SystemExit("training, selection or smoke placements collide with each other or with a committed manifest")
    return {"groups": {k: {"cases": len(v), "sha256": sha256_text("".join(v))} for k, v in hashes.items()},
            "placements": len(flat), "disjoint_from_committed_pen_spin_cases": len(committed)}


def confirmation_manifest(cfg: dict, seed) -> str:
    return str(cfg["confirmation_manifest_template"]).format(seed=int(seed))


def freeze_manifests(cfg: dict) -> dict:
    """The confirmation manifests (twelve per block), disjoint from every committed Stage-3 manifest."""
    from lares.eval.manifest import EvaluationManifest, assert_disjoint

    mine = {confirmation_manifest(cfg, s): cases.build_manifest(3, int(s), int(cfg["confirmation_cases"]), label="structure_confirm")
            for block in cfg["confirmation_seeds"] for s in block}
    others = [EvaluationManifest.load(p) for p in sorted(glob.glob("config/manifests/pen_spin_stage3_*.yaml")) if p not in mine]
    assert_disjoint(list(mine.values()) + others)
    return {"written": [p for p, m in mine.items() if cases.freeze_manifest(p, m)], "manifests": len(mine)}


# ---------------------------------------------------------------------------------------------------------------------
#  Structures in one batch
# ---------------------------------------------------------------------------------------------------------------------


def union_layout(policies: list) -> tuple:
    """The union of the policies' terms (first appearance order) and each policy's coefficients embedded in it; a term
    a policy does not have gets zeros, which contribute exactly nothing."""
    terms, keys = [], []
    for p in policies:
        for t in p["terms"]:
            if st.term_key(t) not in keys:
                keys.append(st.term_key(t))
                terms.append({"expr": st.canonical(t["expr"]), "target": t["target"]})
    offsets, at = {}, 0
    for key, t in zip(keys, terms):
        offsets[key] = at
        at += st.n_slots(t["target"])
    thetas = []
    for p in policies:
        th, src = np.zeros(at), 0
        for t in p["terms"]:
            k = st.n_slots(t["target"])
            th[offsets[st.term_key(t)]:offsets[st.term_key(t)] + k] = p["theta"][src:src + k]
            src += k
        thetas.append(th)
    return terms, thetas


# ---------------------------------------------------------------------------------------------------------------------
#  The selection gate
# ---------------------------------------------------------------------------------------------------------------------


#: selection rules: the registered strict gate, and the amended one (advisor consult call_00244, outcome-informed)
RULE_STRICT, RULE_BEST = "strict_holm", "best_success_drop_bound"


def selection_decision(rows: dict, drop_margin: float, alpha: float = 0.05, rule: str = RULE_STRICT) -> dict:
    """``rows[role] = {"sustained": [...], "completed": [...]}`` over the shared selection placements.

    A candidate passes with a positive paired success difference against the parent, a Holm-adjusted (two
    candidates) two-sided paired t-test below ``alpha`` and a one-sided 95% upper bound on its paired drop-rate
    difference of at most ``drop_margin``. Of the passing candidates the larger mean wins; otherwise the parent stays.
    """
    from lares.eval.stats import t_summary

    par_s, par_c = np.asarray(rows["parent"]["sustained"], float), np.asarray(rows["parent"]["completed"], float)
    out, pvals = {}, {}
    for role in ("c0", "c1"):
        ds = np.asarray(rows[role]["sustained"], float) - par_s
        dd = (1.0 - np.asarray(rows[role]["completed"], float)) - (1.0 - par_c)
        ts, td = t_summary(ds), t_summary(dd, level=0.90)
        out[role] = {"difference": ts["mean"], "ci95": [ts["low"], ts["high"]], "p": ts["p_value"],
                     "successes": int(np.asarray(rows[role]["sustained"]).sum()), "drop_difference": td["mean"],
                     "drop_one_sided_95_upper": td["high"], "drops_ok": bool(td["high"] is not None and td["high"] <= drop_margin + 1e-12)}
        pvals[role] = 1.0 if ts["p_value"] is None else ts["p_value"]
    family = sr.holm(pvals, alpha)
    for role in out:
        out[role]["p_holm"] = family[role]["p_holm"]
        out[role]["passes"] = bool(family[role]["rejects"] and out[role]["difference"] > 0 and out[role]["drops_ok"])
    passing = [r for r in ("c0", "c1") if out[r]["passes"]]
    strict = max(passing, key=lambda r: (out[r]["difference"], -int(r[1]))) if passing else "parent"
    # Amended rule: the most sustained successes among the parent and the candidates within the drop bound; a tie
    # keeps the parent, then the earlier candidate. The Holm test above is still reported but does not gate.
    best = "parent"
    for r in ("c0", "c1"):
        top = int(par_s.sum()) if best == "parent" else out[best]["successes"]
        if out[r]["drops_ok"] and out[r]["successes"] > top:
            best = r
    if rule not in (RULE_STRICT, RULE_BEST):
        raise SystemExit(f"unknown selection rule {rule!r}")
    chosen = strict if rule == RULE_STRICT else best
    dup_s = np.asarray(rows["parent_dup"]["sustained"], float)
    return {"selected": chosen, "rule": rule, "selected_under_registered_strict_rule": strict, "candidates": out, "parent_successes": int(par_s.sum()), "cases": int(par_s.size),
            "duplicate_parent": {"difference": float((dup_s - par_s).mean()), "discordance": float((dup_s != par_s).mean())}}


def merge_stats(batch_stats: list) -> dict:
    """The mean of a role's per-batch summaries (counts are summed)."""
    out = {}
    for key, v in batch_stats[0].items():
        if isinstance(v, dict):
            out[key] = {k: float(np.mean([b[key][k] for b in batch_stats])) for k in v}
        elif key in ("cases", "successes", "drops", "slow_spin", "off_centre"):
            out[key] = int(sum(b[key] for b in batch_stats))
        else:
            out[key] = float(np.mean([b[key] for b in batch_stats]))
    out["success"] = out["successes"] / max(1, out["cases"])
    return out


# ---------------------------------------------------------------------------------------------------------------------
#  Edit generation
# ---------------------------------------------------------------------------------------------------------------------

SYSTEM_PROMPT = ("You improve a symbolic controller for a Shadow Hand that must keep a pen in the hand and roll it about "
                 "its own long axis faster than 4 rad/s, centred, for the whole final second of a 6 s episode. You may "
                 "only ADD ONE TERM to the controller's residual, written in the edit language you are given. Reply with "
                 "one JSON object and nothing else.")


def edit_prompt(parent: dict, stats: dict, tried: list) -> str:
    names = st.coefficient_names(parent["terms"])
    coefs = "\n".join(f"  {n} = {v:+.4f}" for n, v in zip(names, parent["theta"]))
    tried_txt = "\n".join(f"  {json.dumps(t['edit'])} -> {t['outcome']}" for t in tried) or "  (none yet)"
    return (f"EDIT LANGUAGE\n{st.LANGUAGE_TEXT}\n\n"
            f"CURRENT STRUCTURE ({len(parent['terms'])} terms, {len(parent['theta'])} coefficients; trained values):\n"
            + "\n".join(f"  term {k}: {json.dumps(t)}" for k, t in enumerate(parent["terms"])) + f"\n{coefs}\n\n"
            "THE CURRENT CONTROLLER'S BEHAVIOUR (its most recent measured rollouts; an episode succeeds only if it is never "
            "dropped and rolls fast and centred on every step of its final second):\n"
            f"{json.dumps(stats, indent=1)}\n\n"
            f"EDITS ALREADY TRIED IN THIS SEARCH, WITH THEIR OUTCOME:\n{tried_txt}\n\n"
            "Propose the one new term most likely to raise the success rate after its coefficients are trained from zero "
            "by a derivative-free search on the success count. The roll direction differs between episodes, so prefer "
            "quantities that do not assume a roll sign. Reply exactly as "
            '{"expr": <expression>, "target": "<target>", "rationale": "<one sentence>"}.')


def parse_edit(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        raise st.EditError("the reply contains no JSON object")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise st.EditError(f"the reply is not valid JSON: {e}") from None
    if not isinstance(obj, dict):
        raise st.EditError("the reply is not a JSON object")
    return obj


class EditGenerator:
    """One generation call per slot and at most one linked correction; every call is a ledger entry."""

    def __init__(self, client, cfg: dict, run_root: str):
        from lares.search.generation import LLMCallLedger

        self.client, self.cfg = client, cfg
        self.ledger = LLMCallLedger(run_root)
        self.lock = threading.Lock()

    def _call(self, messages: list, label: str, purpose: str) -> tuple:
        from lares.core.policy_generation import _call_llm

        t0 = time.time()
        responses, pt, ct, tt = _call_llm(self.client, 1, self.cfg["model"], messages, 1.0,
                                          reasoning_effort=self.cfg["reasoning_effort"])
        text = getattr(getattr(responses[0], "message", None), "content", None) or ""
        with self.lock:
            rec = self.ledger.record(label=label, model=self.cfg["model"], temperature=1.0, messages=messages, responses=[text],
                                     usage={"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt},
                                     wall_time_seconds=time.time() - t0,
                                     decoding={"n": 1, "temperature": 1.0, "reasoning_effort": self.cfg["reasoning_effort"]},
                                     purpose=purpose)
        return text, rec.call_id, int(tt or 0)

    def propose(self, parent: dict, stats: dict, tried: list, feats, purpose: str) -> dict:
        from lares.search.generation import LABEL_CORRECTION, LABEL_GENERATION

        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": edit_prompt(parent, stats, tried)}]
        out = {"edit": None, "rationale": None, "calls": [], "tokens": 0, "errors": []}
        for attempt in range(int(self.cfg["max_calls_per_slot"])):
            text, call_id, tokens = self._call(messages, LABEL_GENERATION if attempt == 0 else LABEL_CORRECTION, purpose)
            out["calls"].append(call_id)
            out["tokens"] += tokens
            try:
                obj = parse_edit(text)
                out["edit"] = st.validate_edit(obj, parent, [t["edit"] for t in tried], feats)
                out["rationale"] = str(obj.get("rationale", ""))[:500]
                return out
            except st.EditError as e:
                out["errors"].append(str(e))
                messages = messages + [{"role": "assistant", "content": text},
                                       {"role": "user", "content": f"That edit is not admissible: {e}. Reply with one corrected JSON object."}]
        return out                                          # still invalid: an inherited no-op candidate


def load_sample_features(cfg: dict, path: str):
    with np.load(path) as z:
        sample = {k: z[k] for k in ("obs", "ref", "v_f")}
    return st.sample_features(sample, normalization(cfg), cfg["actor"])


# ---------------------------------------------------------------------------------------------------------------------
#  Children
# ---------------------------------------------------------------------------------------------------------------------


def _dump(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)


def _load(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run_child(cfg: dict, request: dict, stem: str, seeded: bool = True) -> dict:
    """One fresh Isaac process. A seeded child that started and left no result is never relaunched."""
    resp = request["response_path"]
    if os.path.exists(resp):
        return _load(resp)
    marker = resp + ".started"
    if seeded and os.path.exists(marker):
        raise SystemExit(f"{marker}: this child already started and left no result; it is not relaunched")
    req_path = stem + ".request.json"
    _dump(req_path, {**request, "config_path": cfg["config_path"], "config_key": CONFIG_KEY, "config_sha256": config_sha256(cfg)})
    if seeded:
        _dump(marker, {"started_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    print("running", request["mode"], stem, flush=True)
    isaac_shell(cfg["isaac_env"], f"python -u -m {CHILD} {req_path}", log=stem + ".log")
    if not os.path.exists(resp):
        raise RuntimeError(f"child left no result; log {stem}.log")
    if seeded:
        os.remove(marker)
    return _load(resp)


# ---------------------------------------------------------------------------------------------------------------------
#  Inference
# ---------------------------------------------------------------------------------------------------------------------


def contrasts(rows: dict, threshold: float, drop_margin: float) -> dict:
    """``rows[block][manifest][policy]`` -> per contrast the per-block case-paired summaries, the equal-weight
    three-block t interval (the algorithm-level statistic) and the pooled case-paired and manifest-clustered ones."""
    from lares.envs.isaac_pen_spin.incumbent_residual import confirmation_contrast
    from lares.eval.stats import t_summary

    out = {}
    for a, b in CONTRASTS:
        per_block = {blk: confirmation_contrast({m: (r[a], r[b]) for m, r in ms.items()}, threshold, drop_margin)
                     for blk, ms in rows.items()}
        pooled = confirmation_contrast({f"{blk}/{m}": (r[a], r[b]) for blk, ms in rows.items() for m, r in ms.items()},
                                       threshold, drop_margin)
        means = [per_block[blk]["pooled"]["success"]["mean"] for blk in sorted(per_block)]
        across = t_summary(means)
        out[f"{a}_minus_{b}"] = {
            "estimate": float(np.mean(means)), "per_block": means, "across_blocks": across,
            "across_blocks_supported": bool(across["low"] is not None and across["low"] >= threshold),
            "across_blocks_refuted": bool(across["high"] is not None and across["high"] < threshold),
            "across_blocks_excludes_zero": bool(across["low"] is not None and across["low"] > 0),
            "pooled_case_paired": pooled["pooled"]["success"], "pooled_win_loss_tie": pooled["pooled"]["success_win_loss_tie"],
            "pooled_manifest_clustered": pooled["manifest_clustered"], "pooled_drops": pooled["pooled"]["drops"],
            "pooled_promotes": pooled["pooled"]["promotes"], "pooled_adopts": pooled["pooled"]["adopts"],
            "per_block_case_paired": {str(blk): {"mean": c["pooled"]["success"]["mean"], "low": c["pooled"]["success"]["low"],
                                                 "high": c["pooled"]["success"]["high"], "successes": c["successes"],
                                                 "adopts": c["pooled"]["adopts"]} for blk, c in per_block.items()}}
    return out


# ---------------------------------------------------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------------------------------------------------


def _jobs(cfg: dict, arm: str, parent: dict, edits: list) -> list:
    k, u = int(cfg["candidates"]), int(cfg["updates_per_candidate"])
    if arm == "fixed":
        return [{"policy": parent, "updates": list(range(k * u)), "snapshots": [u * (i + 1) for i in range(k)]}]
    return [{"policy": st.extend(parent, edits[i]), "updates": [global_update(cfg, i, j) for j in range(u)], "snapshots": [u]}
            for i in range(k)]


def run_study(cfg: dict, client) -> dict:
    sdir = cfg["study_dir"]
    os.makedirs(sdir, exist_ok=True)
    descriptor = sr.check_reference(cfg)
    normalization(cfg)
    root = root_policy(cfg)
    sched = schedule(cfg)
    blocks = list(range(len(cfg["block_seeds"])))
    freeze_json(os.path.join(sdir, "frozen_config.json"),
                {"config_sha256": config_sha256(cfg), "config": frozen_config(cfg), "reference": descriptor,
                 "root_policy_sha256": root["sha256"], "placements": placement_provenance(cfg), "language": st.LANGUAGE_TEXT,
                 "limits": st.LIMITS}, "frozen configuration")
    manifests = {b: [cases.require_frozen(confirmation_manifest(cfg, s)) for s in cfg["confirmation_seeds"][b]] for b in blocks}
    ceiling = int(cfg["start_ceiling"])
    policies_per_block = 2 + len(ARMS)
    reserve = 2 * policies_per_block * sum(len(m) for m in manifests.values()) * int(cfg["confirmation_cases"])

    smoke = run_child(cfg, {"mode": "smoke", "response_path": os.path.join(sdir, "smoke.json")}, os.path.join(sdir, "smoke"))
    summary = {"study": STUDY, "config_sha256": config_sha256(cfg), "smoke": smoke, "schedule": sched,
               "smoke_gate": "pass" if smoke["passes"] else "fail"}
    if not smoke["passes"]:
        summary["note"] = "the smoke gate failed: no generation, training or confirmation ran"
        _dump(os.path.join(sdir, "summary.json"), summary)
        return summary
    started = int(smoke["starts"])
    generator = EditGenerator(client, cfg, sdir)

    parent = {(b, arm): root for b in blocks for arm in ARMS}
    tried = {(b, arm): [] for b in blocks for arm in ARMS}
    info = {(b, arm): {"sample": os.path.join(sdir, "parent_sample.npz"), "stats": smoke["parent_stats"],
                       "source": "the seed-%d smoke (shared initialisation information)" % int(cfg["smoke"]["seed"])}
            for b in blocks for arm in ARMS}
    history = {}
    for r in range(sched["rounds"]):
        # ---- generation: no simulator; information is one step behind selection --------------------------------------
        edits = {}
        for b in blocks:
            path = os.path.join(sdir, f"candidates_b{b}_grammar_r{r}.json")
            if not os.path.exists(path):
                feats = load_sample_features(cfg, info[(b, "grammar")]["sample"])
                got = []
                for k in range(sched["candidates"]):
                    rng = np.random.default_rng(grammar_seed(cfg, b, r, k))
                    got.append(st.sample_edit(rng, parent[(b, "grammar")], [t["edit"] for t in tried[(b, "grammar")]] + got, feats))
                _dump(path, {"block": b, "arm": "grammar", "round": r, "information": info[(b, "grammar")]["source"],
                             "parent_sha256": parent[(b, "grammar")]["sha256"], "edits": got})
            edits[(b, "grammar")] = _load(path)["edits"]
        from lares.envs.isaac_pen_spin.llm_pool import GenerationPool

        pending = [b for b in blocks if not os.path.exists(os.path.join(sdir, f"candidates_b{b}_llm_r{r}.json"))]
        slots = {b: [] for b in pending}
        for k in range(sched["candidates"]):                 # slot 1 after slot 0, so it sees slot 0's edit as tried
            pool = GenerationPool(int(cfg.get("llm_workers", 16)))
            for b in pending:
                seen = tried[(b, "llm")] + [{"edit": s["edit"], "outcome": "proposed in this round, not yet trained"}
                                            for s in slots[b] if s["edit"] is not None]
                pool.submit(b, generator.propose, parent[(b, "llm")], info[(b, "llm")]["stats"], seen,
                            load_sample_features(cfg, info[(b, "llm")]["sample"]), f"structure edit b{b} r{r} k{k}")
            for b in pending:
                slots[b].append(pool.result(b))
            pool.close()
        for b in pending:
            _dump(os.path.join(sdir, f"candidates_b{b}_llm_r{r}.json"),
                  {"block": b, "arm": "llm", "round": r, "information": info[(b, "llm")]["source"],
                   "parent_sha256": parent[(b, "llm")]["sha256"], "edits": [s["edit"] for s in slots[b]], "slots": slots[b]})
        for b in blocks:
            edits[(b, "llm")] = _load(os.path.join(sdir, f"candidates_b{b}_llm_r{r}.json"))["edits"]
            edits[(b, "fixed")] = [None, None]

        # ---- training, then selection (its placements are revealed only now) -----------------------------------------
        for b in blocks:
            for arm in ARMS:
                stem = os.path.join(sdir, f"train_b{b}_{arm}_r{r}")
                tr = run_child(cfg, {"mode": "train", "block": b, "arm": arm, "round": r, "start_cap": ceiling - reserve - started,
                                     "jobs": _jobs(cfg, arm, parent[(b, arm)], edits[(b, arm)]), "response_path": stem + ".json"}, stem)
                started += int(tr["starts"])
                if not tr["complete"]:
                    summary["incomplete"] = f"the start ceiling was reached in {stem}: no finalists, no confirmation"
                    _dump(os.path.join(sdir, "summary.json"), summary)
                    return summary
                cands = [s for job in tr["jobs"] for s in job["snapshots"]]
                freeze_json(os.path.join(sdir, f"trained_b{b}_{arm}_r{r}.json"),
                            {"candidates": cands, "edits": edits[(b, arm)]}, "trained candidates")
                stem = os.path.join(sdir, f"select_b{b}_{arm}_r{r}")
                sel = run_child(cfg, {"mode": "select", "block": b, "arm": arm, "round": r, "parent": parent[(b, arm)],
                                      "candidates": cands, "response_path": stem + ".json"}, stem)
                started += int(sel["starts"])
                dec = selection_decision(sel["rows"], float(cfg["drop_margin"]), rule=cfg.get("selection_rule", RULE_STRICT))
                role = dec["selected"]
                new_parent = parent[(b, arm)] if role == "parent" else cands[int(role[1])]
                for k, e in enumerate(edits[(b, arm)]):
                    if e is not None:
                        c = dec["candidates"][f"c{k}"]
                        tried[(b, arm)].append({"edit": e, "outcome": ("selected" if role == f"c{k}" else "not selected")
                                                + f": success difference {c['difference']:+.4f} against its parent on "
                                                f"{dec['cases']} shared cases (Holm p {c['p_holm']:.3f})"})
                info[(b, arm)] = {"sample": sel["sample_paths"][role], "stats": merge_stats(sel["batch_stats"][role]),
                                  "source": f"round {r} selection rollouts of the retained parent ({role})"}
                history[f"b{b}_{arm}_r{r}"] = {"edits": edits[(b, arm)], "decision": dec, "parent_before": parent[(b, arm)]["sha256"],
                                               "parent_after": new_parent["sha256"], "terms_after": new_parent["terms"],
                                               "coefficients_after": len(new_parent["theta"]),
                                               "training": [{"final": j["updates"][-1], "first": j["updates"][0]} for j in tr["jobs"]]}
                parent[(b, arm)] = new_parent
                _dump(os.path.join(sdir, "history.partial.json"), history)

    finalists = {f"b{b}_{arm}": parent[(b, arm)] for b in blocks for arm in ARMS}
    freeze_json(os.path.join(sdir, "finalists.json"),
                {"rule": "each arm's parent after the last round's selection gate", "root": root, "finalists": finalists}, "finalists")

    # ---- confirmation: one whole frozen manifest per fresh process ---------------------------------------------------
    rows, cells = {}, {}
    for b in blocks:
        rows[b] = {}
        for mpath in manifests[b]:
            mkey = os.path.basename(mpath).replace(".yaml", "")
            cdir = os.path.join(sdir, "confirmation", f"b{b}", mkey)
            os.makedirs(cdir, exist_ok=True)
            rows[b][mkey], scored = {}, {}
            for name, pol in ((REFERENCE, REFERENCE), (ROOT, root), *((arm, finalists[f"b{b}_{arm}"]) for arm in ARMS)):
                key = pol if pol == REFERENCE else pol["sha256"]
                if key not in scored:                         # a finalist identical to an evaluated policy reuses its cell
                    cell = os.path.join(cdir, f"{name}_cell.json")
                    if not os.path.exists(cell):
                        run_child(cfg, {"mode": "cell", "name": name, "policy": pol, "manifest": mpath, "cell_out": cell,
                                        "response_path": cell.replace("_cell.json", "_done.json")}, cell.replace("_cell.json", ""), seeded=False)
                    if pol != REFERENCE and _load(cell)["controller_params"]["policy_sha256"] != pol["sha256"]:
                        raise SystemExit(f"{cell}: the evaluated policy is not the frozen one")
                    scored[key] = load_cell(cell)[1]
                rows[b][mkey][name] = scored[key]
        cells[f"b{b}"] = {name: {"successes": int(sum(r["sustained"] for m in rows[b].values() for r in m[name])),
                                 "drops": int(sum(not r["completed"] for m in rows[b].values() for r in m[name])),
                                 "cases": int(sum(len(m[name]) for m in rows[b].values()))}
                          for name in (REFERENCE, ROOT, *ARMS)}
    c = contrasts(rows, float(cfg["threshold"]), float(cfg["drop_margin"]))
    head = c["llm_minus_grammar"]
    n_cells = len(glob.glob(os.path.join(sdir, "confirmation", "*", "*", "*_cell.json")))
    summary.update({
        "paired_success_difference": head["estimate"],
        "paired_success_difference_across_blocks_ci95": [head["across_blocks"]["low"], head["across_blocks"]["high"]],
        "contrasts": c, "cells": cells, "history": history,
        "finalists": {k: {"sha256": v["sha256"], "terms": v["terms"], "coefficients": len(v["theta"]),
                          "is_root": v["sha256"] == root["sha256"], "theta_l2": float(np.linalg.norm(v["theta"]))}
                      for k, v in finalists.items()},
        "llm": generator.ledger.totals(),
        "accounting": {"simulator_starts_before_confirmation": started, "confirmation_cells": n_cells,
                       "confirmation_charged_starts": 2 * n_cells * int(cfg["confirmation_cases"]),
                       "charged_total": started + 2 * n_cells * int(cfg["confirmation_cases"]), "ceiling": ceiling},
        "limitation": "three search blocks: an algorithm pilot; the across-block interval governs. The selection rule was "
                      "amended after the first two selection decisions (outcome-informed; see the amendment file), and the "
                      "grammar arm's RNG streams overlap between neighbouring blocks, so the across-block t intervals are "
                      "nominal and exploratory, not calibrated independent-search uncertainty",
        "selection_rule": cfg.get("selection_rule", RULE_STRICT),
    })
    _dump(os.path.join(sdir, "summary.json"), summary)
    return summary


__all__ = ["RULE_BEST", "RULE_STRICT", "ARMS", "CONFIG_KEY", "CONTRASTS", "EditGenerator", "STUDY", "config_sha256", "confirmation_manifest", "contrasts",
           "direction_seed", "edit_prompt", "freeze_manifests", "frozen_config", "global_update", "grammar_seed", "merge_stats",
           "normalization", "parse_edit", "placement_provenance", "root_policy", "run_study", "schedule", "selection_decision",
           "selection_seeds", "training_placements", "union_layout"]
