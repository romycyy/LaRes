"""R-131 (Q4 of ``docs/NEXT_RESEARCH_DIRECTION_V2.md``): can an LLM construct a better recovery module than a grammar?

Three arms on the frozen R-106 residual, five independent blocks, one frozen inner learner (ARS: four antithetic
direction pairs, every signed candidate on the same 512 placements of one 2,048-environment batch). Every arm spends
36 updates the same way: two rounds of three branches x 4 updates, a frozen ranking of the three trained branches on
512 fresh placements, then 12 updates of the round-2 winner.

* ``fixed``: the three branches are stochastic continuations of the unchanged template;
* ``grammar``: branch 0 continues the parent structure, branches 1 and 2 are edits drawn uniformly from the language;
* ``llm``: the same, with the two edits proposed by the model (one call and at most one linked correction per slot).

The language, validator, sampler and interpreter are ``module_language``; an edit must reproduce its parent on the
smoke's recorded signal histories before it is trained. The driver runs in the LLM venv and launches this module as
its own Isaac child (``python -m lares.envs.isaac_pen_spin.module_search <request.json>``).

    python scripts/run_full_evolution.py --config config/pen_spin_plateau_repair.yaml --study module_search
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lares.envs.isaac_pen_spin import cases  # noqa: E402
from lares.envs.isaac_pen_spin import module_language as ml  # noqa: E402
from lares.envs.isaac_pen_spin import recovery_authority as ra  # noqa: E402
from lares.envs.isaac_pen_spin import recovery_module as rm  # noqa: E402
from lares.envs.isaac_pen_spin import recovery_tail_audit as ta  # noqa: E402
from lares.envs.isaac_pen_spin import symbolic_reward as sr  # noqa: E402
from lares.envs.isaac_pen_spin.protocols import freeze_json, isaac_shell, load_cell  # noqa: E402
from lares.utils.hashing import sha256_file, sha256_text  # noqa: E402

STUDY = "module_search"
CONFIG_KEY = "pen_spin_module_search"
ARMS = ("fixed", "grammar", "llm")
CONTROLS = ("r106", "d072", "r118", "r121")
CHILD = "lares.envs.isaac_pen_spin.module_search"
RUNTIME_KEYS = ("model", "reasoning_effort", "isaac_env", "config_path", "study", "log_dir")
ROLES = {"train": 1, "direction": 2, "select": 3, "continue": 4, "continue_direction": 5, "grammar": 6}
ARM_CODE = {None: 0, "fixed": 1, "grammar": 2, "llm": 3}
LABEL_CFG = {"early_drop_step": 60, "slow_hit_max": 0.2, "near_hit_min": 0.8}
SLOTS = 4


def frozen_config(cfg: dict) -> dict:
    return json.loads(json.dumps({k: v for k, v in cfg.items() if k not in RUNTIME_KEYS}, sort_keys=True))


def config_sha256(cfg: dict) -> str:
    return sha256_text(json.dumps(frozen_config(cfg), sort_keys=True, separators=(",", ":")))


def _dump(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)


def _load(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------------------------------
#  Streams: every random draw is keyed by its full tuple
# ---------------------------------------------------------------------------------------------------------------------


def stream_key(cfg: dict, block: int, role: str, rnd: int = 0, branch: int = 0, update: int = 0, arm: str | None = None) -> tuple:
    """``(block seed, arm, round, branch, role, update)``. Streams shared between arms carry arm ``0``."""
    return (int(cfg["block_seeds"][block]), ARM_CODE[arm], int(rnd), int(branch), ROLES[role], int(update))


def rng_for(key: tuple):
    return np.random.default_rng(np.random.SeedSequence(list(key)))


def placements(cfg: dict, block: int, role: str, rnd: int = 0, update: int = 0) -> np.ndarray:
    """One batch of 512 placements. Shared by every arm and every branch of a round (the key has neither)."""
    return rng_for(stream_key(cfg, block, role, rnd, 0, update)).uniform(-1.0, 1.0, size=(int(cfg["layout"]["cases_per_slot"]), cases.PLACEMENT_DIM))


def directions(cfg: dict, block: int, role: str, rnd: int, branch: int, update: int, idx: list) -> np.ndarray:
    """Four directions over the structure's active coordinates: draws over all 27 slots, so two structures share
    the draws of the coordinates they share (and two arms the whole stream of the same branch)."""
    full = rng_for(stream_key(cfg, block, role, rnd, branch, update)).standard_normal((int(cfg["ars"]["directions"]), ml.N_SLOTS))
    return full[:, idx]


def all_stream_keys(cfg: dict) -> dict:
    """Every stream the study draws, by a readable name; intentional sharing is one key under one name."""
    out = {}
    for b in range(len(cfg["block_seeds"])):
        for r in range(int(cfg["rounds"])):
            out[f"b{b}/r{r}/select"] = stream_key(cfg, b, "select", r)
            out[f"b{b}/r{r}/grammar"] = stream_key(cfg, b, "grammar", r, arm="grammar")
            for u in range(int(cfg["updates_per_branch"])):
                out[f"b{b}/r{r}/train/u{u}"] = stream_key(cfg, b, "train", r, 0, u)
                for k in range(int(cfg["branches"])):
                    out[f"b{b}/r{r}/direction/k{k}/u{u}"] = stream_key(cfg, b, "direction", r, k, u)
        for u in range(int(cfg["continuation_updates"])):
            out[f"b{b}/continue/u{u}"] = stream_key(cfg, b, "continue", 0, 0, u)
            out[f"b{b}/continue_direction/u{u}"] = stream_key(cfg, b, "continue_direction", 0, 0, u)
    return out


def stream_provenance(cfg: dict) -> dict:
    """Keys are unique, placements are disjoint from each other and from every committed pen-spin manifest."""
    from lares.envs.isaac_pen_spin.success_ars import placement_hashes
    from lares.eval.manifest import EvaluationManifest

    keys = all_stream_keys(cfg)
    if len(set(keys.values())) != len(keys):
        raise SystemExit("two streams share a key")
    if len(set(int(s) for s in cfg["block_seeds"])) != len(cfg["block_seeds"]):
        raise SystemExit("block seeds repeat")
    hashes = []
    for b in range(len(cfg["block_seeds"])):
        for r in range(int(cfg["rounds"])):
            hashes += placement_hashes(placements(cfg, b, "select", r))
            for u in range(int(cfg["updates_per_branch"])):
                hashes += placement_hashes(placements(cfg, b, "train", r, u))
        for u in range(int(cfg["continuation_updates"])):
            hashes += placement_hashes(placements(cfg, b, "continue", 0, u))
    hashes += placement_hashes(cases.placements(int(cfg["smoke_seed"]), int(cfg["layout"]["cases_per_slot"])))
    committed = set()
    for p in sorted(glob.glob("config/manifests/pen_spin_*.yaml") + glob.glob("config/manifests/shadowhand_spin_*.yaml")):
        committed |= {c.task_hash for c in EvaluationManifest.load(p).episodes}
    if len(set(hashes)) != len(hashes) or committed & set(hashes):
        raise SystemExit("search placements collide with each other or with a committed manifest")
    return {"streams": len(keys), "placements": len(hashes), "placements_sha256": sha256_text("".join(hashes)),
            "disjoint_from_committed_pen_spin_cases": len(committed),
            "aliases": "training, selection and continuation placements are shared by the three arms (and by a round's branches); "
                       "direction streams are shared by the three arms per branch, over the coordinates their structures share"}


# ---------------------------------------------------------------------------------------------------------------------
#  Manifests and controllers
# ---------------------------------------------------------------------------------------------------------------------


def confirmation_manifest(cfg: dict, seed) -> str:
    return str(cfg["confirmation_manifest_template"]).format(seed=int(seed))


def freeze_manifests(cfg: dict) -> list:
    from lares.eval.manifest import EvaluationManifest, assert_disjoint

    mine = {confirmation_manifest(cfg, s): cases.build_manifest(3, int(s), int(cfg["confirmation_cases"]), label="modsearch")
            for s in cfg["confirmation_seeds"]}
    paths = sorted(glob.glob("config/manifests/pen_spin_stage3_*.yaml") + glob.glob("config/manifests/shadowhand_spin_*.yaml"))
    assert_disjoint(list(mine.values()) + [EvaluationManifest.load(p) for p in paths if p not in mine])
    return [p for p, m in mine.items() if cases.freeze_manifest(p, m)]


class ModuleController:
    """The frozen root controller plus one interpreted module per environment, behind the runner's interface."""

    def __init__(self, name: str, root, centre_tolerance: float, rec: dict | None = None):
        self.root, self.name, self.tol, self.rec = root, name, float(centre_tolerance), rec
        self.params = {"root_theta_sha256": root.params["theta_sha256"], **({"module": rec} if rec else {})}
        self.gain, self.enabled = None, True

    def reset(self, ctx):
        self.root.reset(ctx)
        self.root.startup_offsets = None
        span = self.root.ref.span
        self.device, self.dtype, self.n = span.device, span.dtype, int(span.shape[0])
        self.neutral = ctx.neutral_action.to(self.device, self.dtype)
        if self.gain is None:                                # the incumbent's fixed stroke gain, per environment
            policy = self.root.ref.inner.policy
            self.gain = ra.PerEnvGain(policy._stroke_gain.constant, self.n)
            policy._stroke_gain = self.gain
        self.gain.scale[:] = 1.0
        self.state = ml.ModuleState(self.n, self.device, self.dtype, self.tol)
        self.held = torch.zeros(self.n, sr.N_JOINTS, device=self.device, dtype=self.dtype)
        self.totals = {"fired": 0, "hold_steps": 0, "extra_boost_steps": 0, "boost_steps": 0, "abs_action_change": 0.0, "steps": 0}
        if self.rec is not None:
            self.set_modules(ml.full_params(self.rec["spec"], self.rec["x"]), ml.ops_vector(self.rec["spec"])[None])

    def set_modules(self, params, ops) -> None:
        self.state.set(params, ops)
        self.state.reset()
        self.depth = self.state.p[:, 4:10].index_select(1, torch.as_tensor(sr.GROUP_OF, device=self.device))

    def reset_envs(self, mask) -> None:
        self.root.reset_envs(mask)
        self.state.reset(mask)

    def signals(self, obs) -> tuple:
        o = torch.nan_to_num(obs.to(self.device, self.dtype))
        sl = cases.obs_slices()
        axial = sr.axial_rate(o)[:, 0]
        total = torch.norm(o[:, sl["obj_angvel"]] / sr.VEL_OBS_SCALE, dim=1)
        transverse = torch.sqrt((total * total - axial * axial).clamp(min=0.0))
        return axial.abs(), torch.norm(o[:, sl["obj_pos"]] - o[:, sl["in_hand_pos"]], dim=1), transverse

    def __call__(self, obs, t, ctx):
        rate, centre, transverse = self.signals(obs)
        self.last_signals = (rate, centre, transverse)
        st = self.state
        beta, boost, fire = st(rate, centre, transverse, self.enabled)
        mult = st.p[:, ml.I["stroke_multiplier"]]
        self.gain.scale[:] = torch.where(boost, mult, torch.ones_like(mult)).unsqueeze(1).cpu()
        a_root = self.root(obs, t, ctx)
        self.held = torch.where(fire.unsqueeze(1), a_root, self.held)
        target = self.neutral * (1.0 - st.p[:, ml.I["mix_midrange"]] - st.p[:, ml.I["mix_hold"]]).unsqueeze(1) \
            + self.held * st.p[:, ml.I["mix_hold"]].unsqueeze(1)
        w = beta.unsqueeze(1) * self.depth
        sent = torch.where(w > 0, ((1.0 - w) * a_root + w * target).clamp(-1.0, 1.0), a_root)
        self.last_beta, self.last_boost = beta, boost
        tot = self.totals
        tot["fired"] += int(fire.sum())
        tot["hold_steps"] += int(st.holding.sum())
        tot["extra_boost_steps"] += int(st.extending.sum())
        tot["boost_steps"] += int(boost.sum())
        tot["abs_action_change"] += float((sent - a_root).abs().sum())
        tot["steps"] += self.n
        return sent

    def extra_trace(self) -> dict:
        self.params["recovery"] = {"triggered_environment_episodes": int(self.totals["fired"]), **{k: v for k, v in self.totals.items() if k != "fired"}}
        return self.root.extra_trace()


def build_root(cfg: dict, name: str, inner=None):
    return rm.StartupRoot(f"{name}:root", sr.build_reference(cfg, f"{name}:root:reference", inner=inner), ra.normalization(cfg),
                          cfg["actor"], ra.finalist_theta(cfg))


def build_controller(cfg: dict, name: str, rec: dict | None, inner=None) -> ModuleController:
    return ModuleController(name, build_root(cfg, name, inner), float(cfg["centre_tolerance"]), None if rec is None else ml.check_record(rec))


def control_x(cfg: dict, name: str) -> np.ndarray:
    entry = rm._load(cfg["controls"][name]["path"])
    if entry["sha256"] != cfg["controls"][name]["sha256"]:
        raise SystemExit(f"{name}: the control asset is not the frozen one")
    return np.asarray(entry["x"], dtype=float)


def make_policy(cfg: dict, name: str, rec: dict | None = None):
    """A confirmation policy: a search finalist (its record) or one of the frozen controls."""
    if rec is not None:
        return build_controller(cfg, name, rec)
    if name == "d072":
        return sr.build_reference(cfg, "d072", resettable=False)
    if name == "r106":
        return sr.build_controller(cfg, "r106", ra.finalist_theta(cfg), ra.normalization(cfg))
    if name == "r118":
        return rm.build(cfg, "r118", control_x(cfg, "r118"), original_bounds=True)
    if name == "r121":
        return rm.build(cfg, "r121", control_x(cfg, "r121"))
    raise ValueError(f"unknown policy {name!r}")


# ---------------------------------------------------------------------------------------------------------------------
#  Selection and inference
# ---------------------------------------------------------------------------------------------------------------------


def rank_branches(slots: list, drop_margin: float) -> dict:
    """``slots`` = the entering parent then the trained branches, each ``{"successes", "drops", "cases"}``. The winner
    is the eligible branch (drop-rate increase over the parent within the margin) with the most sustained successes;
    ties go to branch 0 (the parent structure), then the earlier branch; with none eligible the parent is retained."""
    parent, branches = slots[0], slots[1:]
    base = parent["drops"] / parent["cases"]
    eligible = [k for k, s in enumerate(branches) if s["drops"] / s["cases"] - base <= drop_margin]
    if not eligible:
        return {"winner": None, "eligible": [], "retained": "parent", "rule": "no eligible branch"}
    best = max(eligible, key=lambda k: (branches[k]["successes"], -k))
    return {"winner": int(best), "eligible": eligible, "retained": f"branch_{best}",
            "successes": [s["successes"] for s in slots], "drops": [s["drops"] for s in slots]}


def block_contrast(rows: dict, a: str, b: str) -> float:
    """Mean paired success difference of two policies over every case of every manifest."""
    return float(np.mean([float(x["sustained"]) - float(y["sustained"]) for m in rows.values() for x, y in zip(m[a], m[b])]))


def analyse(rows: dict, cfg: dict, finalists: dict) -> dict:
    """``rows[manifest][policy]``; ``finalists[f"b{b}_{arm}"]`` records. Blocks are the units of every arm contrast."""
    from lares.envs.isaac_pen_spin.incumbent_residual import confirmation_contrast
    from lares.eval.stats import t_summary

    blocks, thr = range(len(cfg["block_seeds"])), float(cfg["threshold"])

    def over_blocks(a, b):
        d = [block_contrast(rows, f"b{k}_{a}" if a in ARMS else a, f"b{k}_{b}" if b in ARMS else b) for k in blocks]
        ts = t_summary(d)
        return {"per_block": d, "mean": ts["mean"], "ci95": [ts["low"], ts["high"]], "p_value": ts["p_value"], "blocks": len(d)}

    head = over_blocks("llm", "grammar")
    head["verdict"] = "supported" if head["ci95"][0] >= thr else ("refuted" if head["ci95"][1] < thr else "inconclusive")
    secondary = {f"{a}_minus_{b}": over_blocks(a, b) for a, b in (("grammar", "fixed"), ("llm", "fixed"), ("fixed", "r106"),
                                                                 ("grammar", "r106"), ("llm", "r106"))}
    qual = {}
    for name in finalists:
        c = confirmation_contrast({m: (r[name], r["r121"]) for m, r in rows.items()}, 0.0, float(cfg["drop_margin"]))
        p = c["pooled"]
        qual[f"{name}_minus_r121"] = {"mean": p["success"]["mean"], "ci95": [p["success"]["low"], p["success"]["high"]],
                                      "p_value": p["success"]["p_value"], "drop_upper_95": p["drops"]["one_sided_95_upper"],
                                      "qualifies": bool((p["success"]["low"] or 0) > 0 and p["drops"]["non_inferior"])}
    family = {**secondary, **qual}
    h = sr.holm({k: (1.0 if v["p_value"] is None else v["p_value"]) for k, v in family.items()})
    for k, v in family.items():
        v["holm"] = h[k]
    n = sum(len(m["r106"]) for m in rows.values())
    table = {}
    for p in [*finalists, *CONTROLS]:
        table[p] = {"success": float(np.mean([r["sustained"] for m in rows.values() for r in m[p]])),
                    "successes": int(sum(r["sustained"] for m in rows.values() for r in m[p])),
                    "drops": int(sum(not r["completed"] for m in rows.values() for r in m[p])), "cases": n,
                    "labels": {name: int(sum(ta.label(r, LABEL_CFG) == name for m in rows.values() for r in m[p])) for name in ta.LABELS}}
        if p in finalists:
            table[p].update(spec=finalists[p]["spec"], edits=finalists[p]["edits"], parameters=finalists[p]["parameters"])
    arms = {a: {"success": [table[f"b{k}_{a}"]["success"] for k in blocks], "edits": [table[f"b{k}_{a}"]["edits"] for k in blocks],
                "parameters": [table[f"b{k}_{a}"]["parameters"] for k in blocks]} for a in ARMS}
    for a in ARMS:
        arms[a]["mean_success"] = float(np.mean(arms[a]["success"]))
    return {"paired_success_difference": head["mean"], "paired_success_difference_ci95": head["ci95"], "primary": head,
            "secondary": secondary, "qualification_against_joint": qual, "policies": table, "arms": arms,
            "precision_note": "five blocks: every arm contrast is a t interval on 4 degrees of freedom"}


# ---------------------------------------------------------------------------------------------------------------------
#  Generation
# ---------------------------------------------------------------------------------------------------------------------

SYSTEM_PROMPT = ("You improve one component of a symbolic controller for a Shadow Hand that must keep a pen in the hand and roll it "
                 "about its own long axis faster than 4 rad/s, centred, on every step of the final second of a 6 s episode. A fixed "
                 "root controller does the rolling; you may only change the structure of its RECOVERY MODULE, by ONE edit in the "
                 "language you are given. Reply with one JSON object and nothing else.")


def edit_prompt(cfg: dict, parent: dict, stats: dict, tried: list) -> str:
    params = "\n".join(f"  {k} = {v:.4g}" for k, v in parent["params"].items())
    tried_txt = "\n".join(f"  {json.dumps(t['edit'])} -> {t['outcome']}" for t in tried) or "  (none yet)"
    options = "\n".join(f"  {json.dumps(e)}" for e in ml.valid_edits(parent["spec"]))
    return (f"MODULE LANGUAGE\n{ml.LANGUAGE_TEXT}\n\n"
            f"PRIOR FINDINGS ABOUT THIS TASK (fixed text, the same in every search)\n{cfg['prior_findings']}\n\n"
            f"CURRENT STRUCTURE\n  {json.dumps(parent['spec'])}\nits trained parameter values:\n{params}\n\n"
            "THE CURRENT MODULE'S MEASURED BEHAVIOUR (its most recent rollouts on the root controller):\n"
            f"{json.dumps(stats, indent=1)}\n\n"
            f"EDITS ALREADY TRIED IN THIS SEARCH, WITH THEIR OUTCOME:\n{tried_txt}\n\n"
            f"THE ADMISSIBLE EDITS OF THE CURRENT STRUCTURE:\n{options}\n\n"
            "Choose the ONE edit most likely to raise the success rate after 4 to 16 updates of a derivative-free search on the "
            "success count train its new parameters from their do-nothing values. Reply exactly as one of the admissible edit "
            'objects with an added "rationale" field of one sentence.')


def parse_edit(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        raise ml.EditError("the reply contains no JSON object")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        raise ml.EditError(f"the reply is not valid JSON: {e}") from None
    if not isinstance(obj, dict):
        raise ml.EditError("the reply is not a JSON object")
    return obj


def validate_edit(edit, parent: dict, tried: list, signals: dict | None) -> dict:
    """The shared validator of both search arms: admissible, new in this search, and an identity on the smoke histories."""
    child = ml.apply_edit(parent["spec"], edit)
    clean = {k: edit[k] for k in ("component", "value", "add") if k in edit}
    if ml.edit_key(clean) in {ml.edit_key(t) for t in tried}:
        raise ml.EditError("this edit was already tried in this search")
    if signals is not None:
        ident = ml.replay_identity(parent, child, signals)
        if not ident["passes"]:
            raise ml.EditError(f"the edit does not reproduce its parent before training: {ident}")
    return clean


class EditGenerator:
    """One generation call per slot and at most one linked correction; every call is a ledger entry."""

    def __init__(self, client, cfg: dict, run_root: str):
        from lares.search.generation import LLMCallLedger

        self.client, self.cfg = client, cfg
        self.ledger = LLMCallLedger(run_root)
        self.lock = threading.Lock()

    def _call(self, messages: list, label: str, purpose: str) -> tuple:
        from lares.core.policy_generation import _call_llm

        if sum(len(m["content"]) for m in messages) > int(self.cfg["max_input_chars"]):
            raise RuntimeError("a generation prompt exceeds the agreed input cap")
        t0 = time.time()
        responses, pt, ct, tt = _call_llm(self.client, 1, self.cfg["model"], messages, 1.0, reasoning_effort=self.cfg["reasoning_effort"],
                                          max_completion_tokens=int(self.cfg["max_output_tokens"]))
        text = getattr(getattr(responses[0], "message", None), "content", None) or ""
        with self.lock:
            rec = self.ledger.record(label=label, model=self.cfg["model"], temperature=1.0, messages=messages, responses=[text],
                                     usage={"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt},
                                     wall_time_seconds=time.time() - t0,
                                     decoding={"n": 1, "temperature": 1.0, "reasoning_effort": self.cfg["reasoning_effort"],
                                               "max_completion_tokens": int(self.cfg["max_output_tokens"])}, purpose=purpose)
        return text, rec.call_id, {"prompt": int(pt or 0), "completion": int(ct or 0), "total": int(tt or 0), "wall_s": time.time() - t0}

    def propose(self, parent: dict, stats: dict, tried: list, taken: list, signals, purpose: str) -> dict:
        from lares.search.generation import LABEL_CORRECTION, LABEL_GENERATION

        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": edit_prompt(self.cfg, parent, stats, tried)}]
        if taken:
            messages[1]["content"] += ("\n\nAnother slot of this round already takes " + "; ".join(json.dumps(t) for t in taken)
                                       + ". Choose a DIFFERENT edit.")
        out = {"edit": None, "rationale": None, "calls": [], "usage": [], "errors": []}
        for attempt in range(int(self.cfg["max_calls_per_slot"])):
            text, call_id, usage = self._call(messages, LABEL_GENERATION if attempt == 0 else LABEL_CORRECTION, purpose)
            out["calls"].append(call_id)
            out["usage"].append(usage)
            try:
                obj = parse_edit(text)
                out["edit"] = validate_edit(obj, parent, [t["edit"] for t in tried] + list(taken), signals)
                out["rationale"] = str(obj.get("rationale", ""))[:500]
                return out
            except ml.EditError as e:
                out["errors"].append(str(e))
                messages = messages + [{"role": "assistant", "content": text},
                                       {"role": "user", "content": f"That edit is not admissible: {e}. Reply with one corrected JSON object."}]
        return out                                          # still invalid: a logged parent-copy branch


def grammar_edits(cfg: dict, block: int, rnd: int, parent: dict, tried: list, signals) -> list:
    """Two edits drawn uniformly without replacement from the valid, untried, replay-identical edits."""
    rng = rng_for(stream_key(cfg, block, "grammar", rnd, arm="grammar"))
    out = []
    for e in ml.sample_edits(rng, parent["spec"], len(ml.valid_edits(parent["spec"])), [t["edit"] for t in tried]):
        try:
            out.append({"edit": validate_edit(e, parent, [t["edit"] for t in tried], signals), "rationale": "uniform draw", "calls": [], "usage": [], "errors": []})
        except ml.EditError as err:
            out.append({"edit": None, "rationale": None, "calls": [], "usage": [], "errors": [str(err)]})
        if len(out) == int(cfg["branches"]) - 1:
            break
    return out


def branch_records(parent: dict, proposals: list, arm: str, branches: int) -> list:
    """Branch 0 is the parent structure; the others are the proposed edits, or a parent copy where a slot has none."""
    out = [dict(parent, edit=None)]
    for k in range(branches - 1):
        prop = proposals[k] if arm != "fixed" and k < len(proposals) else None
        if prop is None or prop["edit"] is None:
            out.append(dict(parent, edit=None, parent_copy=arm != "fixed"))
        else:
            spec = ml.apply_edit(parent["spec"], prop["edit"])
            out.append(dict(ml.record(spec, ml.inherit(parent, spec)), edit=prop["edit"]))
    return out


# ---------------------------------------------------------------------------------------------------------------------
#  Driver
# ---------------------------------------------------------------------------------------------------------------------


class Budget:
    def __init__(self, path: str, cfg: dict):
        self.path, self.cfg = path, cfg
        self.data = _load(path) if os.path.exists(path) else {"starts": 0, "transitions": 0, "wall_s": 0.0, "children": 0, "relaunches": []}

    def room(self, starts: int) -> bool:
        return (self.data["starts"] + starts <= int(self.cfg["start_ceiling"])
                and self.data["wall_s"] < 60.0 * float(self.cfg["max_minutes"]))

    def charge(self, out: dict, wall: float) -> None:
        self.data["starts"] += int(out.get("starts", 0))
        self.data["transitions"] += int(out.get("transitions", 0))
        self.data["wall_s"] += float(wall)
        self.data["children"] += 1
        _dump(self.path, self.data)


def run_child(cfg: dict, request: dict, stem: str, budget: Budget, need: int) -> dict:
    """One Isaac process. A finished child is reused; a rerun after a failure is the identical request and is logged."""
    resp = request["response_path"]
    if os.path.exists(resp):
        return _load(resp)
    if not budget.room(need):
        raise SystemExit(f"a ceiling leaves no room for {stem} ({budget.data})")
    if os.path.exists(stem + ".request.json"):              # an earlier attempt died: charge it at its full allowance
        budget.data["relaunches"].append({"child": stem, "charged_starts": int(need)})
        budget.data["starts"] += int(need)
        _dump(budget.path, budget.data)
    _dump(stem + ".request.json", {**request, "config_path": cfg["config_path"], "config_key": CONFIG_KEY, "config_sha256": config_sha256(cfg)})
    print("running", request["mode"], stem, flush=True)
    t0 = time.time()
    isaac_shell(cfg["isaac_env"], f"python -u -m {CHILD} {stem}.request.json", log=stem + ".log")
    if not os.path.exists(resp):
        raise RuntimeError(f"child left no result; log {stem}.log")
    out = _load(resp)
    budget.charge(out, time.time() - t0)
    return out


def smoke_signals(sdir: str) -> dict:
    with np.load(os.path.join(sdir, "smoke_signals.npz")) as z:
        return {k: z[k] for k in ("rate", "centre", "transverse", "done", "beta", "boost")}


def run_round(cfg: dict, sdir: str, budget: Budget, block: int, rnd: int, arm: str, parent: dict, proposals: list, tried: list) -> dict:
    """Train the round's three branches, score them with the entering parent, apply the frozen ranking."""
    adir = os.path.join(sdir, f"block_{block}", arm, f"round_{rnd}")
    os.makedirs(adir, exist_ok=True)
    path = os.path.join(adir, "round.json")
    if os.path.exists(path):
        return _load(path)
    nb, per_update = int(cfg["branches"]), 2 * SLOTS * int(cfg["layout"]["cases_per_slot"])
    branches = branch_records(parent, proposals, arm, nb)
    tr = run_child(cfg, {"mode": "train_round", "block": block, "round": rnd, "branches": [{"spec": b["spec"], "x": b["x"]} for b in branches],
                         "response_path": os.path.join(adir, "train.json")}, os.path.join(adir, "train"), budget,
                   2 * nb * int(cfg["updates_per_branch"]) * per_update)
    trained = [ml.record(b["spec"], t["final_x"]) for b, t in zip(branches, tr["branches"])]
    sel = run_child(cfg, {"mode": "select", "block": block, "round": rnd,
                          "slots": [{"spec": p["spec"], "x": p["x"]} for p in (parent, *trained)],
                          "response_path": os.path.join(adir, "select.json")}, os.path.join(adir, "select"), budget, 2 * per_update)
    decision = rank_branches(sel["slots"], float(cfg["drop_margin"]))
    winner = parent if decision["winner"] is None else trained[decision["winner"]]
    for k in range(1, nb):
        if branches[k].get("edit") is not None:
            s = sel["slots"][k + 1]
            tried.append({"edit": branches[k]["edit"], "round": rnd,
                          "outcome": f"{s['successes']} of {s['cases']} validation successes after {cfg['updates_per_branch']} updates "
                                     f"(parent structure branch: {sel['slots'][1]['successes']}; entering parent: {sel['slots'][0]['successes']})"
                                     + ("; SELECTED" if decision["winner"] == k else "")})
    out = {"block": block, "round": rnd, "arm": arm, "entering_parent": parent, "branches": branches, "trained": trained,
           "training": [{"mean_candidate_success": [u["mean_candidate_success"] for u in t["updates"]]} for t in tr["branches"]],
           "selection": sel["slots"], "decision": decision, "winner": winner, "tried": list(tried),
           "proposals": proposals if arm != "fixed" else None}
    _dump(path, out)
    return out


def prompt_stats(rnd_out: dict | None, smoke: dict) -> dict:
    """What a proposer is told about its parent: the smoke's template statistics, or the last round's selection."""
    if rnd_out is None:
        return {"module": "the untrained template", **smoke["template_stats"]}
    k = rnd_out["decision"]["winner"]
    slot = rnd_out["selection"][0 if k is None else k + 1]
    return {"module": "the selected branch of the previous round", **{x: slot[x] for x in ("cases", "successes", "drops", "labels", "authority")},
            "training_mean_candidate_success_per_update": rnd_out["training"][0 if k is None else k]["mean_candidate_success"]}


def run_study(cfg: dict, client) -> dict:
    sdir = cfg["study_dir"]
    os.makedirs(sdir, exist_ok=True)
    descriptor = sr.check_reference(cfg)
    ra.finalist_theta(cfg)
    ra.normalization(cfg)
    for name in ("r118", "r121"):
        control_x(cfg, name)
    freeze_manifests(cfg)
    cfg_sha = config_sha256(cfg)
    nblocks, rounds, nb = len(cfg["block_seeds"]), int(cfg["rounds"]), int(cfg["branches"])
    freeze_json(os.path.join(sdir, "frozen_config.json"),
                {"config_sha256": cfg_sha, "config": frozen_config(cfg), "reference": descriptor, "template": ml.template_record(),
                 "language": {"slots": [list(s) for s in ml.SLOTS], "limits": ml.LIMITS, "text_sha256": sha256_text(ml.LANGUAGE_TEXT),
                              "valid_template_edits": ml.valid_edits(ml.TEMPLATE)},
                 "streams": stream_provenance(cfg), "selection_rule": rank_branches.__doc__,
                 "code": {f: sha256_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), f)) for f in ("module_language.py", "module_search.py")}},
                "frozen configuration")
    budget = Budget(os.path.join(sdir, "accounting.json"), cfg)
    per_update = 2 * SLOTS * int(cfg["layout"]["cases_per_slot"])
    summary = {"study": STUDY, "config_sha256": cfg_sha}

    smoke = run_child(cfg, {"mode": "smoke", "out_dir": sdir, "response_path": os.path.join(sdir, "smoke.json")}, os.path.join(sdir, "smoke"),
                      budget, 20000)
    signals = smoke_signals(sdir)
    rep = ml.replay(ml.TEMPLATE, ml.initial_x(ml.TEMPLATE), signals)
    beta_gap = float(np.abs(rep["beta"] - signals["beta"]).max())
    smoke["gates"]["replay_matches_live"] = {"max_abs_beta_difference": beta_gap, "boost_mismatches": int((rep["boost"] != signals["boost"]).sum()),
                                             "environment_steps": int(rep["beta"].size), "replayed_fired": int(rep["fired"].sum()),
                                             "passes": bool(beta_gap <= 1e-6 and (rep["boost"] == signals["boost"]).all())}
    ident = {ml.edit_key(e): ml.replay_identity(ml.template_record(), ml.apply_edit(ml.TEMPLATE, e), signals) for e in ml.valid_edits(ml.TEMPLATE)}
    smoke["gates"]["every_template_edit_is_an_identity"] = {"edits": len(ident), "failing": [k for k, v in ident.items() if not v["passes"]],
                                                            "passes": all(v["passes"] for v in ident.values())}
    ok = bool(smoke["passes"]) and all(g["passes"] for g in smoke["gates"].values())
    summary.update(smoke=smoke, smoke_gate="pass" if ok else "fail")
    if not ok:
        summary["note"] = "the smoke gate failed: no generation, training or confirmation ran"
        _dump(os.path.join(sdir, "summary.json"), summary)
        return summary

    gen = EditGenerator(client, cfg, sdir)
    parents = {(b, a): ml.template_record() for b in range(nblocks) for a in ARMS}
    tried = {(b, a): [] for b in range(nblocks) for a in ARMS}
    last = {(b, a): None for b in range(nblocks) for a in ARMS}
    for rnd in range(rounds):
        def llm_slots(b, rnd=rnd):
            path = os.path.join(sdir, f"block_{b}", "llm", f"round_{rnd}", "generated.json")
            if os.path.exists(path):
                return _load(path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            out = []
            for k in range(nb - 1):                          # the second slot sees the first's edit, so the two differ
                taken = [p["edit"] for p in out if p["edit"] is not None]
                out.append(gen.propose(parents[(b, "llm")], prompt_stats(last[(b, "llm")], smoke), tried[(b, "llm")], taken, signals,
                                       f"R-131 block {b} round {rnd} slot {k}"))
            _dump(path, out)
            return out

        with ThreadPoolExecutor(max_workers=int(cfg["llm_workers"])) as pool:
            llm_props = dict(zip(range(nblocks), pool.map(llm_slots, range(nblocks))))
        for b in range(nblocks):
            for arm in ARMS:
                props = [] if arm == "fixed" else (llm_props[b] if arm == "llm" else
                                                   grammar_edits(cfg, b, rnd, parents[(b, arm)], tried[(b, arm)], signals))
                out = run_round(cfg, sdir, budget, b, rnd, arm, parents[(b, arm)], props, tried[(b, arm)])
                parents[(b, arm)], last[(b, arm)], tried[(b, arm)] = out["winner"], out, out["tried"]

    finalists = {}
    for b in range(nblocks):
        for arm in ARMS:
            adir = os.path.join(sdir, f"block_{b}", arm)
            p = parents[(b, arm)]
            tr = run_child(cfg, {"mode": "continue", "block": b, "policy": {"spec": p["spec"], "x": p["x"]},
                                 "response_path": os.path.join(adir, "continue.json")}, os.path.join(adir, "continue"), budget,
                           2 * int(cfg["continuation_updates"]) * per_update)
            finalists[f"b{b}_{arm}"] = ml.record(p["spec"], tr["final_x"])
    freeze_json(os.path.join(sdir, "finalists.json"), {"rule": "each arm's terminal ARS mean after its 36 updates; no selection on confirmation data",
                                                       "finalists": finalists}, "finalists")

    rows, names = {}, [*finalists, *CONTROLS]
    for s in (int(v) for v in cfg["confirmation_seeds"]):
        mpath = cases.require_frozen(confirmation_manifest(cfg, s))
        sub = os.path.join(sdir, "confirmation", f"s{s}")
        os.makedirs(sub, exist_ok=True)
        rows[f"s{s}"] = {}
        for name in names:
            cell = os.path.join(sub, f"{name}_cell.json")
            run_child(cfg, {"mode": "cell", "name": name, "policy": finalists.get(name), "manifest": mpath, "cell_out": cell,
                            "response_path": os.path.join(sub, f"{name}.done.json")}, os.path.join(sub, name), budget,
                      2 * int(cfg["confirmation_cases"]))
            if name in finalists and _load(cell)["controller_params"]["module"]["sha256"] != finalists[name]["sha256"]:
                raise SystemExit(f"{cell}: the evaluated module is not the frozen finalist")
            rows[f"s{s}"][name] = load_cell(cell)[1]

    summary.update(analyse(rows, cfg, finalists))
    authority = {}
    for name in finalists:
        tot = [_load(os.path.join(sdir, "confirmation", f"s{s}", f"{name}_cell.json"))["controller_params"]["recovery"] for s in cfg["confirmation_seeds"]]
        authority[name] = {k: float(sum(t[k] for t in tot)) for k in tot[0]}
        authority[name]["mean_abs_action_change_per_env_step"] = authority[name]["abs_action_change"] / max(1.0, authority[name]["steps"])
    usage = [u for b in range(nblocks) for r in range(rounds)
             for p in _load(os.path.join(sdir, f"block_{b}", "llm", f"round_{r}", "generated.json")) for u in p["usage"]]
    slots = [p for b in range(nblocks) for r in range(rounds) for p in _load(os.path.join(sdir, f"block_{b}", "llm", f"round_{r}", "generated.json"))]
    summary.update({
        "executed_authority": authority,
        "search": {f"b{b}_{a}": [{"round": r, "edits": [br.get("edit") for br in _load(os.path.join(sdir, f"block_{b}", a, f"round_{r}", "round.json"))["branches"]],
                                  "decision": _load(os.path.join(sdir, f"block_{b}", a, f"round_{r}", "round.json"))["decision"]}
                                 for r in range(rounds)] for b in range(nblocks) for a in ARMS},
        "llm": {"calls": len(usage), "max_calls": nblocks * rounds * (nb - 1) * int(cfg["max_calls_per_slot"]),
                "prompt_tokens": int(sum(u["prompt"] for u in usage)), "completion_tokens": int(sum(u["completion"] for u in usage)),
                "total_tokens": int(sum(u["total"] for u in usage)), "wall_s": float(sum(u["wall_s"] for u in usage)),
                "slots": len(slots), "invalid_slots": int(sum(p["edit"] is None for p in slots)), "model": cfg["model"],
                "reasoning_effort": cfg["reasoning_effort"]},
        "accounting": {**budget.data, "wall_minutes": budget.data["wall_s"] / 60.0, "start_ceiling": int(cfg["start_ceiling"]),
                       "max_minutes": float(cfg["max_minutes"]),
                       "scored_training_episodes": nblocks * len(ARMS) * (rounds * nb * int(cfg["updates_per_branch"]) + int(cfg["continuation_updates"])) * per_update,
                       "selection_episodes": nblocks * len(ARMS) * rounds * SLOTS * int(cfg["layout"]["cases_per_slot"]),
                       "confirmation_episodes": len(names) * len(cfg["confirmation_seeds"]) * int(cfg["confirmation_cases"])},
        "limitation": "five search blocks on one frozen residual root; a 13-edit language; each structural arm reaches at most two edits"})
    _dump(os.path.join(sdir, "summary.json"), summary)
    return summary


# ---------------------------------------------------------------------------------------------------------------------
#  Isaac child
# ---------------------------------------------------------------------------------------------------------------------


def _rollout(env, ctx, ctrl, vecs, params, ops, record: bool = False) -> dict:
    """One batch: placements ``vecs`` and one module (``params`` row, ``ops`` row) per environment; first episodes only."""
    from lares.envs.isaac_pen_spin.runner import _write_placements
    from lares.envs.isaac_pen_spin.success_ars import first_episode_scores

    spec = cases.STAGES[3]
    n, dev = env.num_envs, env.device
    env.reset()
    _write_placements(env, vecs)
    obs = env._get_observations()["policy"]
    ctrl.reset(ctx)
    ctrl.set_modules(params, ops)
    T = int(env.max_episode_length)
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    completed = torch.zeros(n, dtype=torch.bool, device=dev)
    length = torch.zeros(n, dtype=torch.long, device=dev)
    pred = torch.zeros((T, n), dtype=torch.bool, device=dev)
    final_centre = torch.zeros(n, device=dev)
    fired = torch.zeros(n, dtype=torch.bool, device=dev)
    hold, extra, change = torch.zeros(n, device=dev), torch.zeros(n, device=dev), torch.zeros(n, device=dev)
    sig = {k: [] for k in ("rate", "centre", "transverse", "beta", "boost", "done")} if record else None
    starts = n
    for t in range(T):
        action = ctrl(obs, t, ctx)
        st = ctrl.state
        if record:
            for k, v in zip(("rate", "centre", "transverse"), ctrl.last_signals):
                sig[k].append(v.detach().cpu().numpy().astype(np.float32))
            sig["beta"].append(ctrl.last_beta.detach().cpu().numpy().astype(np.float32))
            sig["boost"].append(ctrl.last_boost.detach().cpu().numpy())
        fired |= alive & st.fire_now.to(dev)
        hold += torch.where(alive, st.holding.to(dev).float(), torch.zeros_like(hold))
        extra += torch.where(alive, st.extending.to(dev).float(), torch.zeros_like(extra))
        obs_d, _, terminated, truncated, _ = env.step(action)
        obs = obs_d["policy"]
        done = terminated | truncated
        ctrl.reset_envs(done)
        if record:
            sig["done"].append(done.detach().cpu().numpy())
        starts += int(done.sum())
        pred[t] = alive & (env.center_dist < spec.success_center_tolerance) & (env.spin_rate.abs() > spec.success_spin_rate)
        final_centre = torch.where(alive, env.center_dist, final_centre)
        length = torch.where(alive, length + 1, length)
        completed = torch.where(alive & done, truncated & ~terminated, completed)
        alive = alive & ~done
        if not bool(alive.any()):
            break
    sc = first_episode_scores(pred.cpu().numpy(), length.cpu().numpy(), completed.cpu().numpy())
    out = {"sustained": sc["sustained"], "tail_hit": sc["tail_hit"], "completed": completed.cpu().numpy(), "length": length.cpu().numpy(),
           "final_centre": final_centre.cpu().numpy(), "fired": fired.cpu().numpy(), "hold": hold.cpu().numpy(), "extra": extra.cpu().numpy(),
           "starts": starts, "env_steps": int((t + 1) * n)}
    if record:
        out["signals"] = {k: np.stack(v) for k, v in sig.items()}
    return out


def smoke_probe() -> dict:
    """The smoke's second structure: three edits (the language's depth cap), each trained away from its identity value."""
    tpl = ml.template_record()
    spec = ml.canonical({"schedule": "periodic", "conditions": [], "support_end": "event", "boost_end": "event", "target": "reset", "attempts": 1})
    x, idx = ml.inherit(tpl, spec), ml.active(spec)
    for name, value in (("extra_window", 60.0), ("max_hold", 20.0), ("settle_level", 0.5), ("max_extra", 60.0), ("release_level", 8.0)):
        k = ml.I[name]
        x[idx.index(k)] = (value - ml.LOW[k]) / (ml.HIGH[k] - ml.LOW[k])
    return ml.record(spec, x)


def slot_stats(res: dict, sl: slice) -> dict:
    rows = [{"sustained": bool(res["sustained"][i]), "completed": bool(res["completed"][i]), "length": int(res["length"][i]),
             "final_center_dist": float(res["final_centre"][i]), "tail_hit_fraction": float(res["tail_hit"][i])}
            for i in range(sl.start, sl.stop)]
    fired = res["fired"][sl]
    return {"cases": len(rows), "successes": int(sum(r["sustained"] for r in rows)), "drops": int(sum(not r["completed"] for r in rows)),
            "labels": {name: int(sum(ta.label(r, LABEL_CFG) == name for r in rows)) for name in ta.LABELS},
            "authority": {"fired_fraction": float(fired.mean()), "success_among_fired": float(res["sustained"][sl][fired].mean()) if fired.any() else None,
                          "mean_hold_steps_among_fired": float(res["hold"][sl][fired].mean()) if fired.any() else 0.0,
                          "mean_extra_boost_steps_among_fired": float(res["extra"][sl][fired].mean()) if fired.any() else 0.0}}


def _tile_modules(policies: list, per: int) -> tuple:
    """Per-environment parameter and switch rows for one policy per slot (slot-major)."""
    params = np.concatenate([np.repeat(ml.full_params(p["spec"], p["x"]), per, axis=0) for p in policies])
    ops = np.concatenate([np.repeat(ml.ops_vector(p["spec"])[None], per, axis=0) for p in policies])
    return params, ops


def _train(env, ctx, ctrl, cfg: dict, spec: dict, x0, n_updates: int, place, direct) -> dict:
    """``n_updates`` ARS updates of one structure from ``x0``; ``place(u)`` and ``direct(u, idx)`` give its streams."""
    from lares.envs.isaac_pen_spin import success_ars as sa

    per = int(cfg["layout"]["cases_per_slot"])
    lay = {"slots": SLOTS, "per": per, "directions": int(cfg["ars"]["directions"])}
    idx = ml.active(spec)
    ops = np.repeat(ml.ops_vector(spec)[None], SLOTS * per, axis=0)
    x = np.asarray(x0, dtype=float).copy()
    updates, starts, steps = [], 0, 0
    for u in range(n_updates):
        delta = direct(u, idx)
        cand = rm.ars_candidates(x, delta, float(cfg["ars"]["nu"]))
        vecs = sa.tile(place(u), SLOTS)
        slot_scores, drops, fired = [], [], []
        for r in range(2):
            rc = sa.rollout_candidates(cand, r, lay)
            res = _rollout(env, ctx, ctrl, vecs, np.repeat(ml.full_params(spec, rc), per, axis=0), ops)
            starts += res["starts"]
            steps += res["env_steps"]
            slot_scores.append(res["sustained"].reshape(SLOTS, per).mean(axis=1))
            drops.append(float((~res["completed"]).mean()))
            fired.append(float(res["fired"].mean()))
        scores = sa.ars_scores(slot_scores, lay)
        x, sigma = rm.ars_step(x, delta, scores, float(cfg["ars"]["alpha"]))
        d = lay["directions"]
        updates.append({"update": u + 1, "plus": [float(v) for v in scores[:d]], "minus": [float(v) for v in scores[d:]],
                        "mean_candidate_success": float(scores.mean()), "sigma_r": sigma, "drop_rate": drops, "fired_fraction": fired})
        print("update", json.dumps(updates[-1]), flush=True)
    return {"final_x": [float(v) for v in x], "updates": updates, "starts": starts, "transitions": steps}


def _child(req: dict, cfg: dict, device: str) -> dict:
    from lares.envs.isaac_pen_spin import protocols
    from lares.envs.isaac_pen_spin import success_ars as sa
    from lares.envs.isaac_pen_spin.runner import ControllerContext
    from lares.envs.isaac_pen_spin.symbolic_reward_child import make_training_env

    mode = req["mode"]
    if mode == "cell":
        ctrl = make_policy(cfg, req["name"], req.get("policy"))
        protocols.run_cell(req["manifest"], ctrl, req["cell_out"], device, record_obs=False)
        return {"mode": "cell", "cell": req["cell_out"], "starts": 2 * int(cfg["confirmation_cases"]), "transitions": 0}
    per = int(cfg["layout"]["cases_per_slot"])
    n = SLOTS * per
    seed = int(cfg["smoke_seed"]) if mode == "smoke" else int(cfg["block_seeds"][int(req["block"])])
    env = make_training_env(n, device, seed)
    try:
        ctx = ControllerContext(env)
        if sr.executed_joint_order(ctx) != list(sr.JOINTS):
            raise SystemExit("the simulator's joint order is not the actor's")
        ctrl = build_controller(cfg, mode, None)
        b = int(req.get("block", 0))
        if mode == "train_round":
            rnd, out = int(req["round"]), []
            for k, br in enumerate(req["branches"]):
                out.append(_train(env, ctx, ctrl, cfg, br["spec"], br["x"], int(cfg["updates_per_branch"]),
                                  lambda u: placements(cfg, b, "train", rnd, u),
                                  lambda u, idx, k=k: directions(cfg, b, "direction", rnd, k, u, idx)))
            return {"mode": mode, "branches": out, "starts": sum(o["starts"] for o in out), "transitions": sum(o["transitions"] for o in out)}
        if mode == "continue":
            p = req["policy"]
            out = _train(env, ctx, ctrl, cfg, p["spec"], p["x"], int(cfg["continuation_updates"]),
                         lambda u: placements(cfg, b, "continue", 0, u), lambda u, idx: directions(cfg, b, "continue_direction", 0, 0, u, idx))
            return {"mode": mode, **out}
        if mode == "select":
            params, ops = _tile_modules(req["slots"], per)
            res = _rollout(env, ctx, ctrl, sa.tile(placements(cfg, b, "select", int(req["round"])), SLOTS), params, ops)
            return {"mode": mode, "slots": [slot_stats(res, slice(k * per, (k + 1) * per)) for k in range(SLOTS)],
                    "starts": res["starts"], "transitions": res["env_steps"]}
        # smoke: the template on every slot (A/A, live signals for the replay check), then a probe with every component active
        t0 = time.time()
        vecs = sa.tile(cases.placements(seed, per), SLOTS)
        tpl = ml.template_record()
        a = _rollout(env, ctx, ctrl, vecs, *_tile_modules([tpl] * SLOTS, per), record=True)
        np.savez_compressed(os.path.join(req["out_dir"], "smoke_signals.npz"), **a["signals"])
        sus = a["sustained"].reshape(SLOTS, per)
        out = {"mode": "smoke", "gates": {}, "template_fired": int(a["fired"].sum()), "aa": sa.aa_statistics(sus),
               "template_stats": slot_stats(a, slice(0, n))}
        out["gates"]["scorer_live"] = {"successes": int(sus.sum()), "passes": bool(sus.any())}
        out["gates"]["template_triggers"] = {"fired_fraction": float(a["fired"].mean()), "passes": bool(0.1 < a["fired"].mean() < 0.6)}
        out["gates"]["equal_slots_agree"] = {"max_abs_slot_difference": out["aa"]["max_abs_slot_difference"],
                                             "passes": bool(out["aa"]["max_abs_slot_difference"] <= float(cfg["smoke_aa_tolerance"]))}
        probe = smoke_probe()
        c = _rollout(env, ctx, ctrl, vecs, *_tile_modules([tpl, probe, tpl, probe], per))
        s2 = c["sustained"].reshape(SLOTS, per).mean(axis=1)
        fired = c["fired"].reshape(SLOTS, per).mean(axis=1)
        hold, extra = c["hold"].reshape(SLOTS, per).sum(axis=1), c["extra"].reshape(SLOTS, per).sum(axis=1)
        out["gates"]["per_slot_structures"] = {
            "slot_success": [float(v) for v in s2], "fired_fraction": [float(v) for v in fired],
            "hold_steps": [float(v) for v in hold], "extra_boost_steps": [float(v) for v in extra],
            "passes": bool(abs(s2[0] - s2[2]) <= float(cfg["smoke_aa_tolerance"]) and abs(s2[1] - s2[3]) <= float(cfg["smoke_aa_tolerance"])
                           and hold[0] == 0 and extra[0] == 0 and hold[2] == 0 and extra[2] == 0 and hold[1] > 0 and extra[1] > 0
                           and hold[3] > 0 and extra[3] > 0)}
        out["starts"] = a["starts"] + c["starts"]
        out["transitions"] = a["env_steps"] + c["env_steps"]
        out["throughput"] = {"transitions_per_s": out["transitions"] / (time.time() - t0)}
        out["passes"] = all(g["passes"] for g in out["gates"].values())
        return out
    finally:
        env.close()


def child_main(request_path: str) -> None:
    from lares.envs.isaac_lab_adapter import DEFAULT_ISAACLAB_ROOT, _ensure_isaaclab_on_path, close_isaac_app, launch_isaac_app
    from lares.utils.cli import load_yaml

    req = _load(request_path)
    cfg = load_yaml(req["config_path"])[req["config_key"]]
    device = str(req.get("device", "cuda:0"))
    _ensure_isaaclab_on_path(DEFAULT_ISAACLAB_ROOT)
    launch_isaac_app(headless=True, device=device)
    code = 0
    try:
        if config_sha256(cfg) != req["config_sha256"]:
            raise SystemExit("the config block changed since the driver froze it")
        _dump(req["response_path"], _child(req, cfg, device))
        print("wrote", req["response_path"], flush=True)
    except BaseException:  # Isaac's close swallows tracebacks: print before it runs
        traceback.print_exc()
        sys.stdout.flush()
        code = 1
    finally:
        if code:
            os._exit(code)
        close_isaac_app()


__all__ = ["ARMS", "CONFIG_KEY", "CONTROLS", "STUDY", "EditGenerator", "ModuleController", "all_stream_keys", "analyse", "branch_records",
           "build_controller", "config_sha256", "directions", "edit_prompt", "freeze_manifests", "grammar_edits", "make_policy", "parse_edit",
           "placements", "rank_branches", "run_study", "stream_key", "stream_provenance", "validate_edit"]

if __name__ == "__main__":
    child_main(sys.argv[1])
