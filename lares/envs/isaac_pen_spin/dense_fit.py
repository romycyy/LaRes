"""R-045: dense expert-label fitting versus rollout search on the frozen R-042 prefix controllers (Stage 3).

Driven by ``scripts/run_full_evolution.py --config config/pen_spin_plateau_repair.yaml --study dense_feedback_fit``.
No LLM call. Order (each step skipped if its output exists, so an interrupted study resumes without rerunning a case):

1. ``freeze``: one 92-case manifest per block seed (screen 0-9, promotion 10-39, probe 40-49, expert 50-61,
   fresh 62-91) and one 40-case parity manifest. No Isaac.
2. ``parity``: the D-036 expert on the parity manifest in two separate processes of the pipeline backend; zero
   sustained flips and at most ``batch // 10`` diverged environments, or the study stops before any arm.
3. Per block: the student is R-042 block ``b``'s prefix candidate (source and selected state pinned by sha256).
   The expert runs once on the expert slice with traces recorded; its (observation, clamped action) pairs over every
   step are the dense labels.
   * ``rollout_search``: the student's selected parameters, then three ``rollout_search.perturb`` draws in the order
     local, global, local from ``SeedSequence(block seed)``.
   * ``dense_fit``: full-batch Adam (lr 0.01) on the MSE between the policy mean and the expert's action, ``clip_params``
     after every step, checkpoints at steps 0/250/500/1000. The loss is a diagnostic only.
   Each arm scores its four settings on the same 10 probe cases (one process each) and keeps the best by
   ``is_better_checkpoint`` (success, then tail-hit fraction, ties to the earlier setting). The selected setting is
   screened (10), tested for promotion on 30 against the null incumbent under the strict paired rule, and scored on
   the 30 fresh cases whether or not it promoted. The primary per-arm score is the final incumbent's fresh success:
   the selected setting's if it promoted, else 0 (the null incumbent).
4. ``summary.json``: per-block outcomes, the block-paired primary (dense_fit minus rollout_search) and the secondary
   contrast of selected settings on fresh cases.
"""

from __future__ import annotations

import copy
import json
import os

import numpy as np

from lares.envs.isaac_pen_spin import cases

STAGE = 3
ARMS = ("rollout_search", "dense_fit")
PRIMARY = ("dense_fit", "rollout_search")
SEARCH_ORDER = ("original", "local", "global", "local")
FIT_CHECKPOINTS = (0, 250, 500, 1000)
FIT_LR = 0.01
#: Index ranges of a block manifest, role-disjoint by construction.
SLICES = {"screen": (0, 10), "promotion": (10, 40), "probe": (40, 50), "expert": (50, 62), "fresh": (62, 92)}
BLOCK_CASES = 92
MIN_ELIGIBLE_BLOCKS = 3


# ---------------------------------------------------------------------------------------------------------------------
#  Manifests
# ---------------------------------------------------------------------------------------------------------------------


def manifest_file(label: str, seed: int) -> str:
    return cases.manifest_path(STAGE, label, seed)


def freeze_manifests(dcfg: dict) -> dict:
    """Build (or verify) the block and parity manifests; refuse a placement shared between any two."""
    specs = [("dense", int(s), BLOCK_CASES, f"block{b}") for b, s in enumerate(dcfg["block_seeds"])]
    specs.append(("parity", int(dcfg["parity_seed"]), int(dcfg["parity_cases"]), "parity"))
    return cases.freeze_manifest_set((manifest_file(label, seed), cases.build_manifest(STAGE, seed, n, label=label), role)
                                     for label, seed, n, role in specs)


def block_slices(seed: int) -> dict:
    from lares.eval.manifest import EvaluationManifest

    m = EvaluationManifest.load(manifest_file("dense", seed))
    if len(m) != BLOCK_CASES:
        raise ValueError(f"block manifest for seed {seed} has {len(m)} cases, not {BLOCK_CASES}")
    return {role: m.slice(a, b) for role, (a, b) in SLICES.items()}


# ---------------------------------------------------------------------------------------------------------------------
#  Students
# ---------------------------------------------------------------------------------------------------------------------


def check_student(student: dict) -> None:
    from lares.utils.hashing import sha256_file

    for key in ("source", "state"):
        got = sha256_file(student[key])
        if got != student[f"{key}_sha256"]:
            raise ValueError(f"{student[key]}: sha256 {got} does not match the pinned {student[f'{key}_sha256']}")


def build_policy(source: str, state: dict, env_id: str, scale: float):
    from lares.core.obs_schema import active_schema, get_obs_schema
    from lares.core.policy_generation import GENERATED_IMPORTS

    ns: dict = {}
    exec(GENERATED_IMPORTS + source, ns)
    with active_schema(get_obs_schema(env_id)):
        policy = ns["GeneratedPolicy"](cases.OBS_DIM, cases.NUM_ACTIONS)
    policy.fixed_action_scale = float(scale)
    policy.load_state_dict(state)
    policy._generated_source = source
    policy.eval()
    return policy


def load_student(student: dict, env_id: str, scale: float):
    import torch

    check_student(student)
    source = open(student["source"], encoding="utf-8").read()
    return build_policy(source, torch.load(student["state"], weights_only=True), env_id, scale)


def parameter_snapshot(policy) -> dict:
    return {k: v.detach().cpu().numpy().tolist() for k, v in policy.named_parameters()}


# ---------------------------------------------------------------------------------------------------------------------
#  The two arms' settings
# ---------------------------------------------------------------------------------------------------------------------


def search_settings(student, seed: int) -> list[tuple[str, object]]:
    """``[(label, policy)]``: the student unchanged, then local, global, local draws."""
    from lares.search import rollout_search

    rng = np.random.default_rng(np.random.SeedSequence(int(seed)))
    out = []
    for i, kind in enumerate(SEARCH_ORDER):
        p = copy.deepcopy(student)
        if kind != "original":
            rollout_search.perturb(p, kind, rng)
        out.append((f"search{i}_{kind}", p))
    return out


def expert_pairs(collected: dict) -> tuple[np.ndarray, np.ndarray]:
    """Every step's (observation, executed action) from the expert traces, actions clamped to [-1, 1]."""
    obs, act = [], []
    for i, row in enumerate(collected["episodes"]):
        L = int(row["length"])
        obs.append(np.asarray(collected["obs"][i, :L], dtype=np.float32))
        act.append(np.clip(np.asarray(collected["actions"][i, :L], dtype=np.float32), -1.0, 1.0))
    return np.concatenate(obs), np.concatenate(act)


def fit_settings(student, obs: np.ndarray, actions: np.ndarray, seed: int,
                 checkpoints=FIT_CHECKPOINTS, lr: float = FIT_LR) -> tuple[list, list]:
    """``([(label, policy)], loss log)``: full-batch Adam on MSE(mean, expert action), clipped every step."""
    import torch

    torch.manual_seed(int(seed))
    policy = copy.deepcopy(student)
    policy.train()
    x, y = torch.as_tensor(obs), torch.as_tensor(actions)
    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)

    def loss_now():
        mean, _ = policy(x)
        return torch.mean((mean - y) ** 2)

    out, log, step = [], [], 0
    for target in sorted(checkpoints):
        while step < target:
            opt.zero_grad()
            loss = loss_now()
            loss.backward()
            opt.step()
            policy.clip_params()
            step += 1
        with torch.no_grad():
            value = float(loss_now())
        snap = copy.deepcopy(policy)
        snap.eval()
        out.append((f"fit_step{target}", snap))
        log.append({"step": target, "mse": value})
    return out, log


# ---------------------------------------------------------------------------------------------------------------------
#  Scoring
# ---------------------------------------------------------------------------------------------------------------------


def score(policy, name: str, manifest, stream) -> dict:
    """One manifest in one process: per-case success, tail-hit fraction and failure label."""
    from lares.eval.runner import PolicyActor, evaluate_manifest

    res = evaluate_manifest(PolicyActor(policy, deterministic=True, name=name), manifest, stream)
    eps = res.episodes
    return {"manifest_id": manifest.manifest_id, "success": float(res.success_rate),
            "tail_hit_fraction": float(np.mean([e.signed_goal_progress for e in eps])),
            "per_case": [float(e.success) for e in eps], "failure_labels": [e.failure_label for e in eps],
            "_result": res}


def public(s: dict) -> dict:
    return {k: v for k, v in s.items() if not k.startswith("_")}


def select(scored: list[dict]) -> int:
    """Index of the best setting by the frozen tie rule (success, then tail-hit, ties to the earlier)."""
    from lares.fitting.checkpoints import RolloutScore, is_better_checkpoint

    best, best_i = None, None
    for i, s in enumerate(scored):
        r = RolloutScore(s["success"], s["tail_hit_fraction"])
        if is_better_checkpoint(r, best):
            best, best_i = r, i
    return best_i


def run_arm(arm: str, settings: list, slices: dict, stream_factory, adir: str, extra: dict | None = None) -> dict:
    """Probe every setting, select one, screen, promotion-test against the null incumbent, score on fresh cases."""
    import torch

    from lares.eval.promotion import compare_to_incumbent

    path = os.path.join(adir, "arm.json")
    if os.path.exists(path):
        return json.load(open(path))
    os.makedirs(adir, exist_ok=True)
    stream = stream_factory(arm, os.path.join(adir, "isaac"))
    probes = []
    for label, policy in settings:
        s = score(policy, label, slices["probe"], stream)
        probes.append({"label": label, "parameters": parameter_snapshot(policy), **public(s)})
    i = select(probes)
    label, chosen = settings[i]
    torch.save(chosen.state_dict(), os.path.join(adir, "selected_state.pt"))
    with open(os.path.join(adir, "selected_source.py"), "w", encoding="utf-8") as f:
        f.write(chosen._generated_source)
    screen = score(chosen, label, slices["screen"], stream)
    promo = score(chosen, label, slices["promotion"], stream)
    decision = compare_to_incumbent(promo["_result"], None, f"{arm}:{label}", None,
                                    excluded_manifest_ids=[slices["screen"].manifest_id, slices["probe"].manifest_id])
    fresh = score(chosen, label, slices["fresh"], stream)
    promoted = bool(decision.promoted)
    out = {"arm": arm, "probe": probes, "selected_index": i, "selected_label": label,
           "screen": public(screen), "promotion": public(promo), "promotion_decision": decision.to_dict(),
           "promoted": promoted, "fresh_selected": public(fresh),
           "fresh_incumbent": {"success": fresh["success"] if promoted else 0.0,
                               "per_case": fresh["per_case"] if promoted else [0.0] * len(fresh["per_case"]),
                               "candidate": label if promoted else None,
                               "rule": "the selected setting's fresh score if it promoted, else the null incumbent's 0"},
           "isaac_processes": stream.calls, "isaac_episodes": stream.episodes, **(extra or {})}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    return out


# ---------------------------------------------------------------------------------------------------------------------
#  Parity, expert labels, blocks
# ---------------------------------------------------------------------------------------------------------------------


def parity_gate(dcfg: dict, study_dir: str, stream_factory, expert) -> dict:
    from lares.envs.isaac_pen_spin.study import compare_cells
    from lares.eval.manifest import EvaluationManifest

    path = os.path.join(study_dir, "parity.json")
    if os.path.exists(path):
        return json.load(open(path))
    m = EvaluationManifest.load(manifest_file("parity", int(dcfg["parity_seed"])))
    runs = []
    for k in ("a", "b"):
        got = stream_factory(f"parity_{k}", os.path.join(study_dir, "parity")).collect_traces(expert, m)
        data = np.load(got["trace_path"])
        runs.append({"rows": got["episodes"], "spin_rate": data["spin_rate"], "center_dist": data["center_dist"]})
    report = compare_cells(runs[0], runs[1], len(m))
    report["episodes"] = 2 * len(m)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return report


def expert_labels(bdir: str, slices: dict, stream_factory, expert) -> tuple[dict, np.ndarray, np.ndarray]:
    """The expert once on the block's expert slice, traces recorded; never rerun once started."""
    path = os.path.join(bdir, "expert.json")
    if os.path.exists(path):
        meta = json.load(open(path))
        data = np.load(meta["trace_path"])
        collected = {"episodes": meta["episodes"], "obs": data["obs"], "actions": data["actions"]}
        return meta, *expert_pairs(collected)
    started = os.path.join(bdir, "expert.started")
    if os.path.exists(started):
        raise SystemExit(f"{bdir}: the expert collection was interrupted earlier; its cases are never rerun")
    os.makedirs(bdir, exist_ok=True)
    open(started, "w").close()
    collected = stream_factory("expert", os.path.join(bdir, "expert")).collect_traces(expert, slices["expert"])
    obs, act = expert_pairs(collected)
    meta = {"manifest_id": slices["expert"].manifest_id, "episodes": collected["episodes"],
            "successes": int(sum(r["sustained"] for r in collected["episodes"])),
            "pairs": int(len(obs)), "trace_path": collected["trace_path"]}
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta, obs, act


def run_block(b: int, dcfg: dict, study_dir: str, stream_factory, expert, env_id: str, scale: float) -> dict:
    seed = int(dcfg["block_seeds"][b])
    bdir = os.path.join(study_dir, f"block{b}")
    slices = block_slices(seed)
    student = load_student(dcfg["students"][b], env_id, scale)
    meta, obs, act = expert_labels(bdir, slices, stream_factory, expert)
    row = {"block": b, "seed": seed, "student": dcfg["students"][b], "expert_successes": meta["successes"],
           "expert_episodes": len(meta["episodes"]), "expert_pairs": meta["pairs"], "arms": {}}
    row["arms"]["rollout_search"] = run_arm("rollout_search", search_settings(student, seed), slices, stream_factory,
                                            os.path.join(bdir, "rollout_search"))
    fit_path = os.path.join(bdir, "dense_fit")
    fitted, loss_log = (None, None)
    if not os.path.exists(os.path.join(fit_path, "arm.json")):
        fitted, loss_log = fit_settings(student, obs, act, seed)
    row["arms"]["dense_fit"] = run_arm("dense_fit", fitted or [], slices, stream_factory, fit_path,
                                       extra={"fit": {"loss": loss_log, "lr": FIT_LR, "checkpoints": FIT_CHECKPOINTS,
                                                      "objective": "mean squared error, mean vs clamped expert action",
                                                      "selection": "rollouts on the probe slice; loss is diagnostic"}})
    for arm in ARMS:
        a = row["arms"][arm]
        row[arm] = {"fresh_incumbent": a["fresh_incumbent"]["success"], "fresh_selected": a["fresh_selected"]["success"],
                    "promoted": a["promoted"], "selected": a["selected_label"]}
    return row


def summarise(blocks: list[dict]) -> dict:
    from lares.eval.stats import block_contrast

    out = {"blocks": blocks, "eligible_blocks": len(blocks), "min_eligible_blocks": MIN_ELIGIBLE_BLOCKS}
    a, b = PRIMARY
    for key in ("fresh_incumbent", "fresh_selected"):
        out[f"{a}_minus_{b}_{key}"] = block_contrast(blk[a][key] - blk[b][key] for blk in blocks)
    out["paired_success_difference"] = out[f"{a}_minus_{b}_fresh_incumbent"].get("mean")
    out["conclusive"] = len(blocks) >= MIN_ELIGIBLE_BLOCKS
    return out


def run_study(dcfg: dict, stream_factory, expert, env_id: str, scale: float) -> dict:
    sdir = dcfg["study_dir"]
    os.makedirs(sdir, exist_ok=True)
    freeze_manifests(dcfg)
    for s in dcfg["students"]:
        check_student(s)
    parity = parity_gate(dcfg, sdir, stream_factory, expert)
    print("parity gate:", json.dumps(parity))
    if not parity["passes"]:
        raise SystemExit(f"R-045 parity gate failed; no arm runs: {parity}")
    blocks = []
    for b in range(len(dcfg["block_seeds"])):
        blocks.append(run_block(b, dcfg, sdir, stream_factory, expert, env_id, scale))
        json.dump(summarise(blocks), open(os.path.join(sdir, "summary.partial.json"), "w"), indent=2)
    summary = summarise(blocks)
    summary["parity"] = parity
    with open(os.path.join(sdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


__all__ = ["ARMS", "FIT_CHECKPOINTS", "SLICES", "expert_pairs", "fit_settings", "freeze_manifests", "run_arm",
           "run_study", "search_settings", "select", "summarise"]
