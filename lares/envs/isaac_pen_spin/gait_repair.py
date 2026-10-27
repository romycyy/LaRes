"""R-046: plateau-triggered scaffold continuation on Stage-3 pen spinning (the frozen R-039 gait kept or replaceable).

Driven by ``scripts/run_full_evolution.py --config config/pen_spin_plateau_repair.yaml --study gait_preserving_repair``.
Approved with advisor call_00091; execution wrappers reconciled with call_00094. Order (each step skipped if its output
exists, so an interrupted study resumes without rerunning a case):

1. ``freeze``: one 94-case manifest per block seed, split by index into role manifests (``ROLES``), and one 40-case
   parity manifest. No Isaac.
2. ``parity``: the frozen gait through R-039's evaluator (``protocols.run_cell``) and through the pipeline backend's
   unchanged scaffold, separate processes, 40 cases. Version 1 (zero sustained flips, at most 4 diverged) failed on
   one late flip (37/40 bit-identical; divergence from steps 185-340); version 2 (call_00095, before any development
   case) requires every environment identical through step 150 and at most 4 diverged, flips reported.
3. Per block: a cold prefix (the R-042 prefix: a searched single-phase exploration on the plain 157-d schema, 24
   settings on the 4-case prefix probe, screen, promotion; the unchanged plateau trigger). A block whose trigger does
   not fire is reported missing. Otherwise:
   * the expert once on the 12-case expert slice, traces recorded with the gait run as a shadow along each trajectory;
     qualification on cases 0-9 (at least 9 sustained), the traced episodes are the first two sustained successes in
     the order 10, 11, 0, 1, ...;
   * ``unchanged_gait``: the gait alone; ``bounded_residual`` and ``whole_edit``: one generated correction each, from the
     same request (gait source and parameters, 32 uniform keyframes with the gait's shadow action) except the execution
     rule, plus at most one linked correction. Each LLM arm scores its policy and five ``rollout_search.perturb`` draws
     on the 8-case arm probe and keeps the best by the checkpoint tie rule.
   * every arm: screen (10), promotion (30) against the prefix incumbent (null when the trigger fired) under the
     strict paired rule, then the 30 fresh cases. Final incumbent: the arm's candidate if it promoted, else the prefix
     incumbent (null, 0 on fresh cases).
4. ``summary.json``: primary bounded_residual minus whole_edit (final incumbents, fresh), paired over eligible blocks;
   secondary bounded_residual minus unchanged_gait, and the selected settings' fresh scores.
"""

from __future__ import annotations

import copy
import inspect
import json
import os
from types import SimpleNamespace

import numpy as np

from lares.envs.isaac_pen_spin import cases, gait_scaffold
from lares.envs.isaac_pen_spin.layout import GAIT_SCAFFOLD_TASK_ID

STAGE = 3
#: Index ranges of a block manifest, role-disjoint by construction.
ROLES = {"prefix_probe": (0, 4), "development": (4, 44), "arm_probe": (44, 52), "expert": (52, 64), "fresh": (64, 94)}
BLOCK_CASES = 94
LLM_ARMS = ("bounded_residual", "whole_edit")
ARMS = ("unchanged_gait",) + LLM_ARMS
MODE = {"unchanged_gait": "unchanged", "bounded_residual": "residual", "whole_edit": "whole"}
PRIMARY = ("bounded_residual", "whole_edit")
SECONDARY = (("bounded_residual", "unchanged_gait"),)
SEARCH_ORDER = ("original", "local", "global", "local", "global", "local")
QUALIFY_CASES, QUALIFY_MIN = 10, 9
TRACE_ORDER = (10, 11) + tuple(range(10))
SCREEN, PROMOTION = 10, 30
MIN_ELIGIBLE_BLOCKS = 3


# ---------------------------------------------------------------------------------------------------------------------
#  Manifests
# ---------------------------------------------------------------------------------------------------------------------


def manifest_file(label: str, seed: int) -> str:
    return cases.manifest_path(STAGE, label, seed)


def freeze_manifests(gcfg: dict) -> dict:
    from lares.eval.manifest import EvaluationManifest

    specs = [("gait", int(s), BLOCK_CASES, f"block{b}") for b, s in enumerate(gcfg["block_seeds"])]
    specs.append(("parity", int(gcfg["parity_seed"]), int(gcfg["parity_cases"]), "parity"))
    built, hashes = {}, {}
    for label, seed, n, role in specs:
        m = cases.build_manifest(STAGE, seed, n, label=label)
        path = manifest_file(label, seed)
        if os.path.exists(path):
            if EvaluationManifest.load(path).to_dict() != m.to_dict():
                raise ValueError(f"{path} exists and differs from the manifest this study builds")
        else:
            m.save(path)
        for c in m.episodes:
            if c.task_hash in hashes:
                raise ValueError(f"{role} shares placement {c.task_hash} with {hashes[c.task_hash]}")
            hashes[c.task_hash] = role
        built[role] = path
    return built


def role_manifest(m, role: str):
    """Cases ``ROLES[role]`` of a block manifest, under an id naming the role (so its own slices stay unambiguous)."""
    from lares.eval.manifest import EvaluationManifest

    a, b = ROLES[role]
    return EvaluationManifest(manifest_id=f"{m.manifest_id}:{role}", environment=m.environment, pool=m.pool,
                              split=m.split, episodes=list(m.episodes[a:b]))


def block_roles(seed: int) -> dict:
    from lares.eval.manifest import EvaluationManifest

    m = EvaluationManifest.load(manifest_file("gait", seed))
    if len(m) != BLOCK_CASES:
        raise ValueError(f"block manifest for seed {seed} has {len(m)} cases, not {BLOCK_CASES}")
    roles = {r: role_manifest(m, r) for r in ROLES}
    dev = roles["development"]
    roles["screen"], roles["promotion"] = dev.head(SCREEN), dev.slice(SCREEN, SCREEN + PROMOTION)
    return roles


# ---------------------------------------------------------------------------------------------------------------------
#  Parity: the gait through R-039's evaluator and through the backend's unchanged scaffold
# ---------------------------------------------------------------------------------------------------------------------


def run_cell_gait(gcfg: dict, manifest_path: str, out: str, config_path: str) -> dict:
    """The frozen gait through ``protocols.run_cell`` (R-039's evaluator), its own Isaac process."""
    import subprocess

    if not os.path.exists(out):
        cmd = (f"source {gcfg['isaac_env']} >/dev/null 2>&1 && python -u scripts/run_shadowhand_spin_stages.py "
               f"--config {config_path} --research-protocol scripted_expert_qualification "
               f"--cell-manifest {manifest_path} --controller feedback_gait --cell-out {out}")
        subprocess.run(["bash", "-c", cmd], check=True)
    data = np.load(out.replace(".json", ".npz"))
    return {"rows": json.load(open(out))["episodes"], "spin_rate": data["spin_rate"], "center_dist": data["center_dist"]}


def first_divergence(a: dict, b: dict) -> dict:
    """``{env: first step whose spin_rate or center_dist differ}`` for every diverged environment."""
    out = {}
    for key in ("spin_rate", "center_dist"):
        x, y = np.nan_to_num(a[key], nan=-1e9), np.nan_to_num(b[key], nan=-1e9)
        d = np.abs(x - y) > 0
        for e in np.nonzero(d.any(axis=0))[0]:
            out[int(e)] = min(out.get(int(e), 10 ** 9), int(np.argmax(d[:, e])))
    for i, (x, y) in enumerate(zip(a["rows"], b["rows"])):
        if x["length"] != y["length"]:
            out[i] = min(out.get(i, 10 ** 9), min(int(x["length"]), int(y["length"])))
    return out


def wrapper_gate_v2(parity_dir: str, match_through: int, max_diverged: int) -> dict:
    """R-046 parity gate, version 2 (amended with advisor consult call_00095 after v1 failed, before any development
    case): every environment matches through ``match_through`` steps and at most ``max_diverged`` diverge overall;
    sustained flips are reported, not gating. Recomputed from the saved parity traces; no rollout."""
    ref_json = os.path.join(parity_dir, "run_cell_gait.json")
    ref = np.load(ref_json.replace(".json", ".npz"))
    resp = json.load(open(os.path.join(parity_dir, "parity_0001.response.json")))
    new = np.load(resp["trace_path"]) if os.path.exists(resp.get("trace_path", "")) else np.load(
        os.path.join(parity_dir, "parity_0001.trace.npz"))
    a = {"rows": resp["episodes"], "spin_rate": new["spin_rate"], "center_dist": new["center_dist"]}
    b = {"rows": json.load(open(ref_json))["episodes"], "spin_rate": ref["spin_rate"], "center_dist": ref["center_dist"]}
    first = first_divergence(a, b)
    flips = [i for i, (x, y) in enumerate(zip(a["rows"], b["rows"])) if bool(x["sustained"]) != bool(y["sustained"])]
    early = {e: s for e, s in first.items() if s < match_through}
    return {"version": 2, "rule": f"every environment identical through step {match_through}; at most {max_diverged} "
                                  f"of {len(a['rows'])} diverge overall; sustained flips reported, not gating",
            "first_divergence_step": {str(k): v for k, v in sorted(first.items())}, "diverged_envs": len(first),
            "diverged_before_match_step": len(early), "sustained_outcome_flips": len(flips), "flipped_envs": flips,
            "passes": bool(not early and len(first) <= max_diverged), "amended_by": "advisor consult call_00095"}


def parity_gate(gcfg: dict, study_dir: str, stream_factory) -> dict:
    from lares.envs.isaac_pen_spin.pipeline_backend import GaitActor
    from lares.envs.isaac_pen_spin.study import compare_cells
    from lares.eval.manifest import EvaluationManifest

    path = os.path.join(study_dir, "parity.json")
    if os.path.exists(path):
        return json.load(open(path))
    mpath = manifest_file("parity", int(gcfg["parity_seed"]))
    m = EvaluationManifest.load(mpath)
    pdir = os.path.join(study_dir, "parity")
    os.makedirs(pdir, exist_ok=True)
    ref = run_cell_gait(gcfg, mpath, os.path.join(pdir, "run_cell_gait.json"), gcfg["gait_protocol_config"])
    got = stream_factory("parity", pdir).collect_traces(GaitActor(gcfg["gait_sha256"]), m)
    data = np.load(got["trace_path"])
    new = {"rows": got["episodes"], "spin_rate": data["spin_rate"], "center_dist": data["center_dist"]}
    report = compare_cells(new, ref, len(m))
    report["episodes"] = 2 * len(m)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return report


# ---------------------------------------------------------------------------------------------------------------------
#  Expert traces with the gait's shadow action, and the request
# ---------------------------------------------------------------------------------------------------------------------


def expert_traces(bdir: str, roles: dict, stream_factory, expert) -> dict:
    """The expert once on the expert slice (gait shadowed); never rerun once started."""
    from lares.repair import pen_spin_traces as pst

    path = os.path.join(bdir, "expert.json")
    if os.path.exists(path):
        return json.load(open(path))
    started = os.path.join(bdir, "expert.started")
    if os.path.exists(started):
        raise SystemExit(f"{bdir}: the expert collection was interrupted earlier; its cases are never rerun")
    os.makedirs(bdir, exist_ok=True)
    open(started, "w").close()
    got = stream_factory("expert", os.path.join(bdir, "expert")).collect_traces(expert, roles["expert"])
    if "gait_actions" not in got:
        raise ValueError("the expert traces carry no shadow gait actions")
    eps = []
    for i, row in enumerate(got["episodes"]):
        L = int(row["length"])
        rows = pst.episode_rows(got["obs"][i], got["actions"][i], L)
        for r, g in zip(rows, np.asarray(got["gait_actions"][i, :L])):
            r["gait_action"] = g.tolist()
        eps.append({"case": row["case_id"], "success": bool(row["sustained"]), "length": L, "rows": rows})
    qualify = sum(e["success"] for e in eps[:QUALIFY_CASES])
    traced = [eps[i] for i in TRACE_ORDER if i < len(eps) and eps[i]["success"]][:2]
    out = {"manifest_id": roles["expert"].manifest_id, "successes": int(sum(e["success"] for e in eps)),
           "episodes": len(eps), "qualify_successes": int(qualify), "qualify_cases": QUALIFY_CASES,
           "qualified": bool(qualify >= QUALIFY_MIN and len(traced) == 2), "traced": traced,
           "traced_case_ids": [e["case"] for e in traced], "trace_path": got["trace_path"]}
    with open(path, "w") as f:
        json.dump(out, f)
    return out


HEADER = ("Successful expert episodes (the expert is a closed-loop policy). Per kept step: t; STATE as in the trace "
          "format below; GAIT = the frozen gait's action at this step along the expert's own trajectory (what "
          "gait_action would read); ACTION = the expert's executed action (action order).")

EXECUTION_RULE = {
    "residual": ("Your controller's output is a correction delta. The executed action is\n"
                 "  clamp(gait_action + clamp(delta, -{bound}, {bound}), -1, 1)\n"
                 "so the gait always runs and each component of your correction is limited to +/-{bound}."),
    "whole": ("Your controller's output is a correction delta. The executed action is\n"
              "  clamp(gait_action + delta, -1, 1)\n"
              "so delta = 0 keeps the gait unchanged and delta = target - gait_action replaces it with any action."),
}

REQUEST_TEMPLATE = """Below is a frozen scripted finger gait for this task. It does not reliably {goal_phrase}: it
succeeded on 7 of 30 and 9 of 30 fresh episodes in two earlier evaluations, failing mostly by dropping the pen or
letting the spin rate dip near the end. An expert policy solves this task; it succeeded on {successes} of {episodes}
episodes here. It is closed-loop: replaying one of its recorded action sequences without feedback succeeded on 0 of
40 episodes, and holding either its pen-state or its hand-state observations fixed at their first values made it fail
on 40 and 39 of 40.

The gait runs every step with its own clock and integral state; you cannot change it. Its action for the current step
is given to your controller as the observation field gait_action.

{execution_rule}

Write a controller (class GeneratedPolicy) whose output is that correction. Read observations only through
self.obs_field (gait_action included), and declare every parameter with get_param_ranges.

The frozen gait (its parameters are FROZEN_R039; t is the control step, control_dt = 1/60 s):

```python
{gait_source}
FROZEN_R039 = {gait_params}
```
"""


def gait_source() -> str:
    from lares.envs.isaac_pen_spin import scripted_expert as se

    return "\n\n".join([inspect.getsource(se.quat_rotate_x), inspect.getsource(se.FeedbackGait)]).strip()


def render_row(r: dict) -> str:
    from lares.repair import pen_spin_traces as pst

    base = pst.render_row(r)
    head, action = base.split(" ACTION=")
    return f"{head} GAIT={pst._fmt(r['gait_action'], 3)} ACTION={action}"


def repair_request(arm: str, expert: dict, bound: float) -> tuple[str, dict]:
    """The request for one LLM arm and its packet metadata; the two arms differ only in the execution rule."""
    from lares.repair import pen_spin_traces as pst
    from lares.utils.hashing import sha256_text

    from lares.envs.isaac_pen_spin.scripted_expert import FROZEN_R039

    lines, kept = ["", f"{HEADER} {pst.HEADER} {pst.KEYFRAMES} steps per episode are shown."], []
    for k, e in enumerate(expert["traced"]):
        idx = pst.uniform_keyframes(len(e["rows"]))
        kept.append(idx)
        lines.append(f"Episode {k + 1} ({len(e['rows'])} steps):")
        lines += [render_row(e["rows"][t]) for t in idx]
    block = "\n".join(lines)
    head = REQUEST_TEMPLATE.format(goal_phrase=pst.GOAL_PHRASE, successes=expert["successes"],
                                   episodes=expert["episodes"],
                                   execution_rule=EXECUTION_RULE[MODE[arm]].format(bound=bound),
                                   gait_source=gait_source(), gait_params=json.dumps(FROZEN_R039, sort_keys=True))
    text = head + block
    return text, {"arm": arm, "keyframe_rule": "uniform", "keyframes": pst.KEYFRAMES, "kept_steps": kept,
                  "packet_chars": len(block), "request_chars": len(text), "packet_sha256": sha256_text(block),
                  "request_sha256": sha256_text(text), "traced_case_ids": expert["traced_case_ids"]}


# ---------------------------------------------------------------------------------------------------------------------
#  Generation (one call plus at most one linked correction, through the R-037 slot generator)
# ---------------------------------------------------------------------------------------------------------------------


def generate(arm: str, request: str, packet: dict, adir: str, client, model: str, reasoning_effort: str,
             max_calls: int = 2) -> dict:
    """The arm's generated correction (or why there is none). Calls are ledgered in ``<adir>/llm_calls.jsonl``."""
    import pickle

    from lares.core.policy_generation import PlateauSlotGenerator
    from lares.core.training_pipeline import load_policy_prompt_assets
    from lares.search.generation import LLMCallLedger

    os.makedirs(adir, exist_ok=True)
    data_pkl = os.path.join(adir, "data.pkl")
    with open(data_pkl, "wb") as f:
        pickle.dump([{"obs": np.zeros(177)}], f)
    ledger = LLMCallLedger(adir)
    gen = PlateauSlotGenerator(client, SimpleNamespace(model=model, reasoning_effort=reasoning_effort),
                               os.path.join(adir, "gen_0"), 177, cases.NUM_ACTIONS, GAIT_SCAFFOLD_TASK_ID,
                               load_policy_prompt_assets(GAIT_SCAFFOLD_TASK_ID), SimpleNamespace(n=2), data_pkl, ledger,
                               llm_transcript_path=os.path.join(adir, "llm_transcript.log"), max_total_calls=max_calls,
                               max_repair_per_candidate=1)
    meta = {"lane": "gait_scaffold", "arm": arm, "parent": "frozen_r039_gait", **packet}
    attempts = gen.trace_repair(1, {"candidate_id": "frozen_r039_gait"}, request, meta)
    final = attempts[-1]
    out = {"attempts": [{"attempt_id": c.attempt_id, "valid": c.valid, "validation": c.validation,
                         "llm_call_ids": c.llm_call_ids} for c in attempts],
           "valid": bool(final.valid), "llm_calls": len(ledger), "llm_tokens": ledger.totals()["llm_total_tokens"],
           "policy": final.policy if final.valid else None, "code": final.code if final.valid else None}
    return out


# ---------------------------------------------------------------------------------------------------------------------
#  Arms
# ---------------------------------------------------------------------------------------------------------------------


def score(actor, manifest, stream) -> dict:
    from lares.eval.runner import evaluate_manifest

    res = evaluate_manifest(actor, manifest, stream)
    eps = res.episodes
    return {"manifest_id": manifest.manifest_id, "success": float(res.success_rate),
            "tail_hit_fraction": float(np.mean([e.signed_goal_progress for e in eps])),
            "per_case": [float(e.success) for e in eps], "failure_labels": [e.failure_label for e in eps],
            "_result": res}


def public(s: dict) -> dict:
    return {k: v for k, v in s.items() if not k.startswith("_")}


def scaffolded(policy, arm: str, gcfg: dict):
    policy._scaffold = {"mode": MODE[arm], "bound": float(gcfg["residual_bound"]), "gait_sha256": gcfg["gait_sha256"],
                        "schema_env_id": GAIT_SCAFFOLD_TASK_ID}
    return policy


def search_settings(policy, seed: int) -> list:
    from lares.search import rollout_search

    rng = np.random.default_rng(np.random.SeedSequence(int(seed)))
    out = []
    for i, kind in enumerate(SEARCH_ORDER):
        p = copy.deepcopy(policy)
        if kind != "original":
            rollout_search.perturb(p, kind, rng)
        out.append((f"search{i}_{kind}", p))
    return out


def evaluate_arm(arm: str, actor_for, settings: list, roles: dict, stream, incumbent_id) -> dict:
    """Probe (LLM arms), screen, promotion against the prefix incumbent, fresh; returns the arm's record."""
    from lares.eval.promotion import compare_to_incumbent
    from lares.fitting.checkpoints import RolloutScore, is_better_checkpoint

    if incumbent_id is not None:
        raise ValueError("an eligible block's prefix incumbent is null by the trigger rule")
    probes, best, best_i = [], None, 0
    if len(settings) > 1:
        for i, (label, p) in enumerate(settings):
            s = score(actor_for(label, p), roles["arm_probe"], stream)
            probes.append({"label": label, **public(s)})
            r = RolloutScore(s["success"], s["tail_hit_fraction"])
            if is_better_checkpoint(r, best):
                best, best_i = r, i
    label, chosen = settings[best_i]
    actor = actor_for(label, chosen)
    screen = score(actor, roles["screen"], stream)
    promo = score(actor, roles["promotion"], stream)
    excluded = [roles["screen"].manifest_id, roles["arm_probe"].manifest_id, roles["prefix_probe"].manifest_id]
    decision = compare_to_incumbent(promo["_result"], None, f"{arm}:{label}", None, excluded_manifest_ids=excluded)
    fresh = score(actor, roles["fresh"], stream)
    promoted = bool(decision.promoted)
    return {"arm": arm, "probe": probes, "selected_label": label, "screen": public(screen), "promotion": public(promo),
            "promotion_decision": decision.to_dict(), "promoted": promoted, "fresh_selected": public(fresh),
            "fresh_incumbent": {"success": fresh["success"] if promoted else 0.0,
                                "per_case": fresh["per_case"] if promoted else [0.0] * len(fresh["per_case"]),
                                "candidate": label if promoted else None,
                                "rule": "the arm's candidate if it promoted, else the prefix incumbent (null, 0)"}}


def run_arm(arm: str, b: int, gcfg: dict, roles: dict, expert: dict, bdir: str, stream_factory, client) -> dict:
    import torch

    from lares.envs.isaac_pen_spin.pipeline_backend import GaitActor
    from lares.eval.runner import PolicyActor

    adir = os.path.join(bdir, arm)
    path = os.path.join(adir, "arm.json")
    if os.path.exists(path):
        return json.load(open(path))
    if os.path.exists(os.path.join(adir, "isaac")):
        raise SystemExit(f"{adir} was interrupted after spending episodes; an interrupted arm is not rerun")
    os.makedirs(adir, exist_ok=True)
    seed = int(gcfg["block_seeds"][b])
    extra = {}
    if arm == "unchanged_gait":
        gait = GaitActor(gcfg["gait_sha256"])
        settings, actor_for = [("frozen_gait", None)], (lambda label, p: gait)
    else:
        request, packet = repair_request(arm, expert, float(gcfg["residual_bound"]))
        with open(os.path.join(adir, "request.txt"), "w", encoding="utf-8") as f:
            f.write(request)
        g = generate(arm, request, packet, adir, client, gcfg["model"], gcfg["reasoning_effort"])
        extra = {"packet": packet, "generation": {k: v for k, v in g.items() if k not in ("policy",)}}
        if not g["valid"]:
            out = {"arm": arm, "valid": False, "promoted": False, "selected_label": None,
                   "fresh_incumbent": {"success": 0.0, "candidate": None,
                                       "rule": "no valid correction: the prefix incumbent (null, 0)"}, **extra}
            with open(path, "w") as f:
                json.dump(out, f, indent=2, default=str)
            return out
        policy = scaffolded(g["policy"], arm, gcfg)
        settings = search_settings(policy, seed)
        actor_for = lambda label, p: PolicyActor(p, deterministic=True, name=f"{arm}:{label}")  # noqa: E731
    stream = stream_factory(arm, os.path.join(adir, "isaac"))
    out = evaluate_arm(arm, actor_for, settings, roles, stream, None)
    out.update(valid=True, isaac_processes=stream.calls, isaac_episodes=stream.episodes, **extra)
    if arm != "unchanged_gait":
        chosen = dict(settings)[out["selected_label"]]
        torch.save(chosen.state_dict(), os.path.join(adir, "selected_state.pt"))
        with open(os.path.join(adir, "selected_source.py"), "w", encoding="utf-8") as f:
            f.write(chosen._generated_source)
        out["parameters"] = {k: v.detach().cpu().numpy().tolist() for k, v in chosen.named_parameters()}
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    return out


# ---------------------------------------------------------------------------------------------------------------------
#  Blocks and summary
# ---------------------------------------------------------------------------------------------------------------------


def summarise(blocks: list[dict]) -> dict:
    from lares.eval.stats import t_summary

    eligible = [b for b in blocks if b.get("eligible")]
    out = {"blocks": blocks, "eligible_blocks": len(eligible), "min_eligible_blocks": MIN_ELIGIBLE_BLOCKS}
    for a, b in (PRIMARY,) + SECONDARY:
        for key in ("fresh_incumbent", "fresh_selected"):
            d = [blk["arms"][a][key]["success"] - blk["arms"][b][key]["success"] for blk in eligible
                 if key in blk["arms"][a] and key in blk["arms"][b]]
            out[f"{a}_minus_{b}_{key}"] = {"per_block": d, **(t_summary(d) if len(d) >= 2 else {"n": len(d)})}
    out["paired_success_difference"] = out[f"{PRIMARY[0]}_minus_{PRIMARY[1]}_fresh_incumbent"].get("mean")
    out["conclusive"] = len(eligible) >= MIN_ELIGIBLE_BLOCKS
    return out


def run_study(gcfg: dict, stream_factory, expert, run_prefix, client) -> dict:
    """``run_prefix(block, roles, log_dir)`` runs the cold prefix in its own process and returns its prefix state."""
    sdir = gcfg["study_dir"]
    os.makedirs(sdir, exist_ok=True)
    gait_scaffold.check_gait(gcfg["gait_sha256"])
    freeze_manifests(gcfg)
    parity = parity_gate(gcfg, sdir, stream_factory)
    print("parity gate (v1):", json.dumps(parity))
    if int(gcfg.get("parity_gate_version", 1)) >= 2:
        v1 = parity
        parity = wrapper_gate_v2(os.path.join(sdir, "parity"), int(gcfg["parity_match_through_step"]),
                                 int(gcfg["parity_max_diverged"]))
        parity["v1"] = v1
        print("parity gate (v2):", json.dumps({k: v for k, v in parity.items() if k != "v1"}))
    if not parity["passes"]:
        raise SystemExit(f"R-046 parity gate failed; nothing else runs: {parity}")
    blocks = []
    for b, seed in enumerate(gcfg["block_seeds"]):
        roles = block_roles(int(seed))
        bdir = os.path.join(sdir, f"block{b}")
        state = run_prefix(b, roles, os.path.join(bdir, "prefix"))
        row = {"block": b, "seed": int(seed), "trigger": state["trigger"], "eligible": bool(state["trigger"]["fired"]),
               "prefix_incumbent": state["incumbent_id"]}
        if not row["eligible"]:
            print(f"block {b}: no plateau ({state['trigger']['reason']}); reported as missing")
            blocks.append(row)
            continue
        ex = expert_traces(bdir, roles, stream_factory, expert)
        row["expert"] = {k: v for k, v in ex.items() if k != "traced"}
        if not ex["qualified"]:
            row.update(eligible=False, missing="the expert did not qualify on this block's expert slice")
            blocks.append(row)
            continue
        row["arms"] = {arm: run_arm(arm, b, gcfg, roles, ex, bdir, stream_factory, client) for arm in ARMS}
        blocks.append(row)
        json.dump(summarise(blocks), open(os.path.join(sdir, "summary.partial.json"), "w"), indent=2, default=str)
    summary = summarise(blocks)
    summary["parity"] = parity
    with open(os.path.join(sdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


__all__ = ["ARMS", "EXECUTION_RULE", "LLM_ARMS", "ROLES", "block_roles", "evaluate_arm", "expert_traces",
           "freeze_manifests", "parity_gate", "repair_request", "run_study", "summarise"]
