"""R-048: does the expert's local feedback map help a gait-preserving residual more than raw expert labels?

Driven by ``scripts/run_full_evolution.py --config config/pen_spin_plateau_repair.yaml --study local_feedback_evidence``.
Approved with advisor call_00098. Ungated: every block runs (D-041 had too few plateau-eligible blocks); the
production plateau trigger is unchanged and nothing here bears on it. Order per block (each step skipped if its output
exists; an interrupted step that spent episodes is never rerun):

1. ``parent``: the frozen R-039 gait (unchanged scaffold) on the block's screen (10, traces recorded), probe (10),
   promotion (30) and fresh (30) cases: 80 episodes.
2. ``states``: two of the parent's screen episodes, failures first (screen order), and from each ``STATES_PER_EPISODE``
   uniformly spaced states up to its last step, plus ``EXTRA_PER_EPISODE`` further states between them for the
   raw-label arm's equal budget.
3. ``queries``: the qualified PPO expert (D-036) queried at every state and, for the evidence states, at symmetric
   perturbations of eight coordinates (pen-centre x/y/z, axial rate, and the flexion of each finger), in one Isaac
   process with no episode. Step sizes are ``STEP_FRACTION`` of each coordinate's 5-95 percentile range over a frozen
   expert demonstration (``demo_trace``, pinned by SHA-256). Sensitivities are half the difference of the executed
   (clamped) actions; saturation and each state's distance from the demonstration are logged. They are the network's
   responses at states the expert never visited, not verified recovery actions.
4. ``raw_labels`` and ``sensitivity``: two candidate slots each (one call plus at most one linked correction per slot)
   from requests that differ only in the evidence block, the raw block padded with extra labelled states to at least
   the sensitivity block's length. Both run as bounded residuals (``|delta| <= residual_bound``) on the gait. Each
   arm screens its valid candidates (10 cases each), keeps the better by the checkpoint tie rule, scores it on the
   probe (10, recorded only), then promotion (30) against the parent under the strict paired rule, then fresh (30):
   at most 90 episodes per arm. Final incumbent: the candidate if it promoted, else the parent.
5. ``summary.json``: primary sensitivity minus raw_labels on fresh final incumbents, paired over blocks; secondary the
   selected candidates' fresh scores and each arm minus the parent.
"""

from __future__ import annotations

import json
import os

import numpy as np

from lares.envs.isaac_pen_spin import cases, gait_scaffold
from lares.envs.isaac_pen_spin.gait_repair import (EXECUTION_RULE, gait_source, generate, public, scaffolded,
                                                   score)

STAGE = 3
LABEL = "local_feedback"
#: Index ranges of a block manifest, role-disjoint by construction (development = screen + promotion).
ROLES = {"screen": (0, 10), "promotion": (10, 40), "probe": (40, 50), "fresh": (50, 80)}
BLOCK_CASES = 80
ARMS = ("raw_labels", "sensitivity")
PRIMARY = ("sensitivity", "raw_labels")
SLOTS = 2
STATES_PER_EPISODE = 16
EXTRA_PER_EPISODE = 48
STEP_FRACTION = 0.1
TOP_K = 3
#: Action order (sorted actuated joints), as in ``scripted_expert.A``.
ACTION_JOINTS = ("WRJ1", "WRJ0", "FFJ3", "MFJ3", "RFJ3", "LFJ4", "THJ4", "FFJ2", "MFJ2", "RFJ2", "LFJ3", "THJ3", "FFJ1",
                 "MFJ1", "RFJ1", "LFJ2", "THJ2", "LFJ1", "THJ1", "THJ0")

# hand_dof_pos joint order: WRJ1, WRJ0, FFJ3, MFJ3, RFJ3, LFJ4, THJ4, FFJ2, MFJ2, RFJ2, LFJ3, THJ3, FFJ1, MFJ1, RFJ1,
# LFJ2, THJ2, FFJ0, MFJ0, RFJ0, LFJ1, THJ1, LFJ0, THJ0.
FINGER_FLEX = {"ff_flex": (7, 12, 17), "mf_flex": (8, 13, 18), "rf_flex": (9, 14, 19), "lf_flex": (15, 20, 22)}
COORDS = ("offset_x", "offset_y", "offset_z", "axial_rate") + tuple(FINGER_FLEX)


# ---------------------------------------------------------------------------------------------------------------------
#  Manifests
# ---------------------------------------------------------------------------------------------------------------------


def manifest_file(seed: int) -> str:
    return cases.manifest_path(STAGE, LABEL, seed)


def freeze_manifests(lcfg: dict) -> dict:
    from lares.eval.manifest import EvaluationManifest

    built, hashes = {}, {}
    for b, seed in enumerate(lcfg["block_seeds"]):
        m = cases.build_manifest(STAGE, int(seed), BLOCK_CASES, label=LABEL)
        path = manifest_file(int(seed))
        if os.path.exists(path):
            if EvaluationManifest.load(path).to_dict() != m.to_dict():
                raise ValueError(f"{path} exists and differs from the manifest this study builds")
        else:
            m.save(path)
        for c in m.episodes:
            if c.task_hash in hashes:
                raise ValueError(f"block{b} shares placement {c.task_hash} with {hashes[c.task_hash]}")
            hashes[c.task_hash] = f"block{b}"
        built[f"block{b}"] = path
    return built


def block_roles(seed: int) -> dict:
    from lares.eval.manifest import EvaluationManifest

    m = EvaluationManifest.load(manifest_file(seed))
    if len(m) != BLOCK_CASES:
        raise ValueError(f"block manifest for seed {seed} has {len(m)} cases, not {BLOCK_CASES}")
    out = {}
    for role, (a, b) in ROLES.items():
        out[role] = EvaluationManifest(manifest_id=f"{m.manifest_id}:{role}", environment=m.environment, pool=m.pool,
                                       split=m.split, episodes=list(m.episodes[a:b]))
    return out


# ---------------------------------------------------------------------------------------------------------------------
#  Coordinates, perturbations and sensitivities (numpy only)
# ---------------------------------------------------------------------------------------------------------------------


def coordinates(obs: np.ndarray) -> np.ndarray:
    """``(N, 8)``: pen offset from in_hand_pos (m), roll about the pen axis (rad/s), mean flexion of each finger."""
    sl = cases.obs_slices()
    obs = np.asarray(obs, dtype=np.float64)
    offset = obs[:, sl["obj_pos"]] - obs[:, sl["in_hand_pos"]]
    axis = cases.pen_axis(obs[:, sl["obj_rot"]])
    roll = np.sum(obs[:, sl["obj_angvel"]] / cases.VEL_OBS_SCALE * axis, axis=-1)
    hand = obs[:, sl["hand_dof_pos"]]
    flex = [hand[:, list(j)].mean(axis=1) for j in FINGER_FLEX.values()]
    return np.column_stack([offset, roll] + flex)


def step_sizes(demo_obs: np.ndarray) -> np.ndarray:
    """``STEP_FRACTION`` of each coordinate's 5-95 percentile range over the demonstration states."""
    c = coordinates(demo_obs)
    lo, hi = np.percentile(c, 5, axis=0), np.percentile(c, 95, axis=0)
    h = STEP_FRACTION * (hi - lo)
    if np.any(h <= 0):
        raise ValueError(f"degenerate demonstration range for {[COORDS[i] for i in np.where(h <= 0)[0]]}")
    return h


def perturb(obs: np.ndarray, k: int, delta: float) -> np.ndarray:
    """``obs`` with coordinate ``k`` moved by ``delta`` (its own units); nothing else changes."""
    sl = cases.obs_slices()
    out = np.array(obs, dtype=np.float32, copy=True)
    if k < 3:
        out[:, sl["obj_pos"].start + k] += delta
    elif k == 3:
        axis = cases.pen_axis(out[:, sl["obj_rot"]])
        out[:, sl["obj_angvel"]] += (delta * cases.VEL_OBS_SCALE * axis).astype(np.float32)
    else:
        joints = list(FINGER_FLEX.values())[k - 4]
        cols = [sl["hand_dof_pos"].start + j for j in joints]
        out[:, cols] += delta
    return out


def query_batch(states: np.ndarray, h: np.ndarray) -> np.ndarray:
    """The rows to query: the states, then for each coordinate the + and - perturbations."""
    rows = [states]
    for k in range(len(COORDS)):
        rows += [perturb(states, k, +h[k]), perturb(states, k, -h[k])]
    return np.concatenate(rows, axis=0)


def sensitivities(raw: np.ndarray, n: int) -> dict:
    """From the queried raw actions: base actions, half-differences of clamped actions ``(n, 8, 20)``, saturation."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape[0] != n * (1 + 2 * len(COORDS)):
        raise ValueError(f"{raw.shape[0]} queried actions do not match {n} states")
    clamped = np.clip(raw, -1.0, 1.0)
    base = clamped[:n]
    sens = np.zeros((n, len(COORDS), raw.shape[1]))
    for k in range(len(COORDS)):
        plus = clamped[n * (1 + 2 * k): n * (2 + 2 * k)]
        minus = clamped[n * (2 + 2 * k): n * (3 + 2 * k)]
        sens[:, k] = 0.5 * (plus - minus)
    return {"base": base, "sens": sens, "saturated_fraction": float(np.mean(np.abs(raw) > 1.0)),
            "base_saturated_fraction": float(np.mean(np.abs(raw[:n]) > 1.0))}


def demo_distance(states: np.ndarray, demo_obs: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Each state's distance to the nearest demonstration state, coordinates in units of their step sizes."""
    a, d = coordinates(states) / h, coordinates(demo_obs) / h
    return np.sqrt(((a[:, None, :] - d[None, :, :]) ** 2).sum(-1)).min(axis=1)


# ---------------------------------------------------------------------------------------------------------------------
#  States from the parent's screen traces
# ---------------------------------------------------------------------------------------------------------------------


def choose_episodes(rows: list[dict]) -> list[int]:
    """Two screen episodes, failures first, each group in screen order."""
    order = [i for i, r in enumerate(rows) if not r["sustained"]] + [i for i, r in enumerate(rows) if r["sustained"]]
    return order[:2]


def spaced(length: int, n: int, exclude=()) -> list[int]:
    """``n`` uniformly spaced steps of ``range(length)``, avoiding ``exclude`` where the episode allows."""
    pool = [t for t in range(int(length)) if t not in set(exclude)]
    if len(pool) <= n:
        return pool
    return [pool[int(round(k * (len(pool) - 1) / (n - 1)))] for k in range(n)]


def select_states(traces: dict) -> list[dict]:
    """``[{episode, case, success, length, evidence_steps, extra_steps}]`` for the two chosen screen episodes."""
    out = []
    for i in choose_episodes(traces["episodes"]):
        row = traces["episodes"][i]
        L = int(row["length"])
        ev = spaced(L, STATES_PER_EPISODE)
        out.append({"episode": i, "case": row["case_id"], "success": bool(row["sustained"]), "length": L,
                    "evidence_steps": ev, "extra_steps": spaced(L, EXTRA_PER_EPISODE, exclude=ev)})
    return out


# ---------------------------------------------------------------------------------------------------------------------
#  Requests
# ---------------------------------------------------------------------------------------------------------------------

REQUEST_TEMPLATE = """Below is a frozen scripted finger gait for this task. It does not reliably {goal_phrase}: it
succeeded on about 13 to 15 of 30 fresh episodes in recent evaluations, failing mostly by dropping the pen or letting
the spin rate dip near the end. An expert policy solves this task (40 of 40 episodes in a recent evaluation). It is
closed-loop: replaying its recorded actions without feedback succeeded on 0 of 40 episodes, holding either its
pen-state or its hand-state observations fixed made it fail on 40 and 39 of 40, and feeding either group to it four
control steps late made it fail on 40 of 40.

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

RAW_HEADER = ("Evidence: states the gait itself visited in two of its own episodes on this block ({episodes}). The "
              "expert never ran from these states; EXPERT is the action the expert policy outputs when given each "
              "state (queried, clamped to [-1, 1], action order). Per row: t; STATE as defined below; GAIT = the "
              "gait's executed action at that step; EXPERT = the queried expert action.")
SENS_HEADER = (" SENS lists, for each of eight state coordinates, how the queried expert action changes when only that "
               "coordinate is moved by +h versus -h (half the difference, clamped actions; the {top} largest action "
               "components by magnitude, named by action joint). Coordinates and step sizes h: {steps}. "
               "offset_* = pen centre minus in_hand_pos (m); axial_rate = roll about the pen's axis (rad/s); "
               "*_flex = all flexion joints of one finger moved together (normalised joint units). These are the "
               "network's local responses at states it never visited, not verified recovery actions. AVERAGE SENS "
               "is the mean over the shown states with the fraction whose sign agrees.")


def _row_text(obs_row: np.ndarray, gait: np.ndarray, expert: np.ndarray, t: int) -> str:
    from lares.repair import pen_spin_traces as pst

    r = pst.episode_rows(obs_row[None, :], expert[None, :], 1)[0]
    head, action = pst.render_row(dict(r, t=t)).split(" ACTION=")
    return f"{head} GAIT={pst._fmt(gait, 3)} EXPERT={action}"


def _sens_text(sens: np.ndarray) -> str:
    parts = []
    for k, name in enumerate(COORDS):
        top = np.argsort(-np.abs(sens[k]))[:TOP_K]
        parts.append(f"{name}: " + ", ".join(f"{ACTION_JOINTS[j]} {sens[k, j]:+.3f}" for j in top))
    return "    SENS " + "; ".join(parts)


def _average_text(sens: np.ndarray) -> str:
    mean = sens.mean(axis=0)
    agree = (np.sign(sens) == np.sign(mean)[None]).mean(axis=0)
    lines = ["AVERAGE SENS (per coordinate: action joint mean (sign agreement), the 6 largest by |mean|):"]
    for k, name in enumerate(COORDS):
        top = np.argsort(-np.abs(mean[k]))[:6]
        lines.append(f"  {name}: " + ", ".join(f"{ACTION_JOINTS[j]} {mean[k, j]:+.3f} ({agree[k, j]:.2f})"
                                               for j in top))
    return "\n".join(lines)


def evidence_blocks(ev: dict, h: np.ndarray) -> dict:
    """The two arms' evidence text; the raw block is padded with extra labelled states to at least the other's size."""
    from lares.repair import pen_spin_traces as pst

    names = ", ".join(f"{e['case']} ({'succeeded' if e['success'] else 'failed'}, {e['length']} steps)"
                      for e in ev["episodes"])
    header = RAW_HEADER.format(episodes=names) + " " + pst.HEADER.split("Per kept step: t; ")[1].split(" ACTION")[0]
    steps = ", ".join(f"{c} {v:.4g}" for c, v in zip(COORDS, h))
    sens_lines = [header + SENS_HEADER.format(top=TOP_K, steps=steps)]
    raw_lines = [header]
    for k, e in enumerate(ev["episodes"]):
        sens_lines.append(f"Episode {k + 1}:")
        raw_lines.append(f"Episode {k + 1}:")
        for j, t in enumerate(e["evidence_steps"]):
            i = e["evidence_index"][j]
            row = _row_text(ev["obs"][i], ev["gait"][i], ev["expert"][i], t)
            sens_lines += [row, _sens_text(ev["sens"][i])]
            raw_lines.append(row)
    sens_lines.append(_average_text(ev["sens_evidence"]))
    sens_text = "\n".join(sens_lines)
    # Pad: interleave the two episodes' extra states in time order until the raw block is at least as long.
    extra = []
    for k, e in enumerate(ev["episodes"]):
        extra += [(k, t, i) for t, i in zip(e["extra_steps"], e["extra_index"])]
    added = {0: [], 1: []}
    for k, t, i in sorted(extra, key=lambda x: (x[1], x[0])):
        if len("\n".join(_assemble_raw(raw_lines, ev, added))) >= len(sens_text):
            break
        added[k].append((t, i))
    raw_text = "\n".join(_assemble_raw(raw_lines, ev, added))
    return {"raw_labels": raw_text, "sensitivity": sens_text,
            "raw_extra_states": sum(len(v) for v in added.values()),
            "raw_chars": len(raw_text), "sensitivity_chars": len(sens_text)}


def _assemble_raw(raw_lines: list[str], ev: dict, added: dict) -> list[str]:
    """The raw block with each episode's evidence and added states merged in time order."""
    out = [raw_lines[0]]
    for k, e in enumerate(ev["episodes"]):
        out.append(f"Episode {k + 1}:")
        rows = [(t, e["evidence_index"][j]) for j, t in enumerate(e["evidence_steps"])] + added.get(k, [])
        out += [_row_text(ev["obs"][i], ev["gait"][i], ev["expert"][i], t) for t, i in sorted(rows)]
    return out


def request_for(arm: str, block_text: str, bound: float) -> tuple[str, dict]:
    from lares.envs.isaac_pen_spin.scripted_expert import FROZEN_R039
    from lares.repair import pen_spin_traces as pst
    from lares.utils.hashing import sha256_text

    head = REQUEST_TEMPLATE.format(goal_phrase=pst.GOAL_PHRASE,
                                   execution_rule=EXECUTION_RULE["residual"].format(bound=bound),
                                   gait_source=gait_source(), gait_params=json.dumps(FROZEN_R039, sort_keys=True))
    text = head + "\n" + block_text
    return text, {"arm": arm, "packet_chars": len(block_text), "request_chars": len(text),
                  "packet_sha256": sha256_text(block_text), "request_sha256": sha256_text(text)}


# ---------------------------------------------------------------------------------------------------------------------
#  Block steps
# ---------------------------------------------------------------------------------------------------------------------


def _once(path: str, started: str):
    if os.path.exists(path):
        return json.load(open(path))
    if os.path.exists(started):
        raise SystemExit(f"{started}: this step was interrupted after spending episodes; it is never rerun")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(started, "w").close()
    return None


def run_parent(lcfg: dict, roles: dict, bdir: str, stream_factory) -> dict:
    """The gait on screen (traced), probe, promotion (against null, recorded) and fresh."""
    from lares.envs.isaac_pen_spin.pipeline_backend import GaitActor
    from lares.eval.promotion import compare_to_incumbent

    path = os.path.join(bdir, "parent.json")
    done = _once(path, os.path.join(bdir, "parent.started"))
    if done is not None:
        return done
    gait = GaitActor(lcfg["gait_sha256"])
    stream = stream_factory("parent", os.path.join(bdir, "parent"))
    tr = stream.collect_traces(gait, roles["screen"])
    screen_rows = tr["episodes"]
    out = {"screen": {"manifest_id": roles["screen"].manifest_id,
                      "success": float(np.mean([r["sustained"] for r in screen_rows])),
                      "tail_hit_fraction": float(np.mean([r["tail_hit_fraction"] for r in screen_rows])),
                      "per_case": [float(r["sustained"]) for r in screen_rows]},
           "trace_path": tr["trace_path"], "screen_rows": screen_rows}
    for role in ("probe", "promotion", "fresh"):
        s = score(gait, roles[role], stream)
        if role == "promotion":
            out["promotion_returns"] = [float(e.episode_return) for e in s["_result"].episodes]
            out["promotion_vs_null"] = compare_to_incumbent(
                s["_result"], None, "frozen_gait", None,
                excluded_manifest_ids=[roles["screen"].manifest_id, roles["probe"].manifest_id]).to_dict()
        out[role] = public(s)
    out.update(isaac_processes=stream.calls, isaac_episodes=stream.episodes)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    return out


def run_queries(lcfg: dict, roles: dict, parent: dict, bdir: str, stream_factory, expert) -> dict:
    """States from the parent's screen traces and the expert's queried actions; saved as ``evidence.npz``."""
    from lares.utils.hashing import sha256_file

    npz = os.path.join(bdir, "evidence.npz")
    meta_path = os.path.join(bdir, "evidence.json")
    if os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        data = np.load(npz)
        return _evidence(meta, data)
    if sha256_file(lcfg["demo_trace"]) != lcfg["demo_trace_sha256"]:
        raise SystemExit(f"{lcfg['demo_trace']} does not match its pinned SHA-256")
    demo = np.load(lcfg["demo_trace"])
    demo_obs = np.concatenate([demo["obs"][i, :int(L)] for i, L in enumerate(demo["length"])], axis=0)
    h = step_sizes(demo_obs)
    trace = np.load(parent["trace_path"])
    episodes = select_states({"episodes": parent["screen_rows"]})
    obs, gait, ev_idx, ex_idx, n = [], [], [], [], 0
    for e in episodes:
        steps = e["evidence_steps"] + e["extra_steps"]
        obs.append(trace["obs"][e["episode"], steps])
        gait.append(trace["actions"][e["episode"], steps])
        e["evidence_index"] = list(range(n, n + len(e["evidence_steps"])))
        e["extra_index"] = list(range(n + len(e["evidence_steps"]), n + len(steps)))
        n += len(steps)
    obs, gait = np.concatenate(obs).astype(np.float32), np.concatenate(gait)
    ev_rows = [i for e in episodes for i in e["evidence_index"]]
    batch = np.concatenate([obs, query_batch(obs[ev_rows], h)], axis=0)
    stream = stream_factory("queries", os.path.join(bdir, "queries"))
    raw = stream.query_actions(expert, batch, roles["screen"])
    expert_all = np.clip(raw[:n], -1.0, 1.0)
    s = sensitivities(raw[n:], len(ev_rows))
    sens = np.zeros((n, len(COORDS), raw.shape[1]))
    sens[ev_rows] = s["sens"]
    np.savez_compressed(npz, obs=obs, gait=gait, expert=expert_all, sens=sens, raw=raw)
    dist = demo_distance(obs[ev_rows], demo_obs, h)
    meta = {"episodes": episodes, "evidence_rows": ev_rows, "step_sizes": dict(zip(COORDS, h.tolist())),
            "queries": int(len(batch)), "saturated_fraction": s["saturated_fraction"],
            "base_saturated_fraction": s["base_saturated_fraction"],
            "demo_distance": {"median": float(np.median(dist)), "max": float(dist.max()), "per_state": dist.tolist()},
            "demo_trace": lcfg["demo_trace"], "demo_states": int(len(demo_obs)),
            "evidence_sha256": sha256_file(npz)}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return _evidence(meta, np.load(npz))


def _evidence(meta: dict, data) -> dict:
    ev = dict(meta)
    ev.update(obs=data["obs"], gait=data["gait"], expert=data["expert"], sens=data["sens"])
    ev["sens_evidence"] = data["sens"][meta["evidence_rows"]]
    return ev


def run_arm(arm: str, lcfg: dict, roles: dict, parent: dict, text: str, bdir: str, stream_factory, client) -> dict:
    from lares.eval.promotion import compare_to_incumbent
    from lares.eval.runner import PolicyActor
    from lares.fitting.checkpoints import RolloutScore, is_better_checkpoint

    adir = os.path.join(bdir, arm)
    path = os.path.join(adir, "arm.json")
    if os.path.exists(path):
        return json.load(open(path))
    if os.path.exists(os.path.join(adir, "isaac")):
        raise SystemExit(f"{adir} was interrupted after spending episodes; an interrupted arm is not rerun")
    os.makedirs(adir, exist_ok=True)
    bound = float(lcfg["residual_bound"])
    request, packet = request_for(arm, text, bound)
    with open(os.path.join(adir, "request.txt"), "w", encoding="utf-8") as f:
        f.write(request)
    slots, policies = [], []
    for k in range(SLOTS):
        g = generate(arm, request, dict(packet, slot=k), os.path.join(adir, f"slot{k}"), client, lcfg["model"],
                     lcfg["reasoning_effort"], max_calls=int(lcfg.get("max_calls_per_slot", 2)))
        slots.append({k2: v for k2, v in g.items() if k2 != "policy"})
        policies.append(scaffolded(g["policy"], "bounded_residual", lcfg) if g["valid"] else None)
    out = {"arm": arm, "packet": packet, "slots": slots,
           "llm_calls": sum(s["llm_calls"] for s in slots), "llm_tokens": sum(s["llm_tokens"] for s in slots)}
    valid = [(k, p) for k, p in enumerate(policies) if p is not None]
    parent_fresh = parent["fresh"]
    if not valid:
        out.update(valid=False, promoted=False, selected_slot=None, isaac_episodes=0,
                   fresh_incumbent={"success": parent_fresh["success"], "per_case": parent_fresh["per_case"],
                                    "candidate": "parent", "rule": "no valid candidate: the parent"})
        with open(path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        return out
    stream = stream_factory(arm, os.path.join(adir, "isaac"))
    actors = {k: PolicyActor(p, deterministic=True, name=f"{arm}:slot{k}") for k, p in valid}
    screens, best, best_k = {}, None, None
    for k, _ in valid:
        s = score(actors[k], roles["screen"], stream)
        screens[k] = public(s)
        r = RolloutScore(s["success"], s["tail_hit_fraction"])
        if is_better_checkpoint(r, best):
            best, best_k = r, k
    actor = actors[best_k]
    probe = score(actor, roles["probe"], stream)
    promo = score(actor, roles["promotion"], stream)
    parent_promo = _parent_result(parent, roles["promotion"])
    decision = compare_to_incumbent(promo["_result"], parent_promo, f"{arm}:slot{best_k}", "frozen_gait",
                                    excluded_manifest_ids=[roles["screen"].manifest_id, roles["probe"].manifest_id])
    fresh = score(actor, roles["fresh"], stream)
    promoted = bool(decision.promoted)
    out.update(valid=True, screens={str(k): v for k, v in screens.items()}, selected_slot=best_k,
               probe=public(probe), promotion=public(promo), promotion_decision=decision.to_dict(),
               promoted=promoted, fresh_selected=public(fresh),
               fresh_incumbent={"success": fresh["success"] if promoted else parent_fresh["success"],
                                "per_case": fresh["per_case"] if promoted else parent_fresh["per_case"],
                                "candidate": f"slot{best_k}" if promoted else "parent",
                                "rule": "the arm's candidate if it promoted against the parent, else the parent"},
               isaac_processes=stream.calls, isaac_episodes=stream.episodes,
               selected_code=dict(valid)[best_k]._generated_source)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    return out


def _parent_result(parent: dict, manifest):
    """The parent's promotion ManifestResult rebuilt from its saved per-case scores (paired by case)."""
    from types import SimpleNamespace

    from lares.eval.runner import ManifestResult

    p = parent["promotion"]
    if p["manifest_id"] != manifest.manifest_id:
        raise ValueError("the parent's promotion record is for another manifest")
    res = ManifestResult(actor_name="frozen_gait", manifest_id=manifest.manifest_id, action_mode="deterministic",
                         schema_id=cases.STAGES[STAGE].task_id)
    for case, ok, ret in zip(manifest.episodes, p["per_case"], parent["promotion_returns"]):
        res.episodes.append(SimpleNamespace(case_id=case.case_id, task_hash=case.task_hash, success=bool(ok),
                                            episode_return=float(ret), signed_goal_progress=None))
    return res


# ---------------------------------------------------------------------------------------------------------------------
#  Study
# ---------------------------------------------------------------------------------------------------------------------


def summarise(blocks: list[dict]) -> dict:
    from lares.eval.stats import t_summary

    out = {"blocks": blocks, "n_blocks": len(blocks)}
    pairs = [PRIMARY, ("sensitivity", "parent"), ("raw_labels", "parent")]
    for a, b in pairs:
        for key in ("fresh_incumbent", "fresh_selected"):
            d = []
            for blk in blocks:
                x = blk["arms"][a].get(key)
                y = blk["parent"]["fresh"] if b == "parent" else blk["arms"][b].get(key)
                if x is not None and y is not None:
                    d.append(x["success"] - y["success"])
            out[f"{a}_minus_{b}_{key}"] = {"per_block": d, **(t_summary(d) if len(d) >= 2 else {"n": len(d)})}
    out["paired_success_difference"] = out["sensitivity_minus_raw_labels_fresh_incumbent"].get("mean")
    out["promoted"] = {a: sum(bool(blk["arms"][a].get("promoted")) for blk in blocks) for a in ARMS}
    return out


def run_study(lcfg: dict, stream_factory, expert, client) -> dict:
    sdir = lcfg["study_dir"]
    os.makedirs(sdir, exist_ok=True)
    gait_scaffold.check_gait(lcfg["gait_sha256"])
    freeze_manifests(lcfg)
    blocks = []
    for b, seed in enumerate(lcfg["block_seeds"]):
        roles = block_roles(int(seed))
        bdir = os.path.join(sdir, f"block{b}")
        parent = run_parent(lcfg, roles, bdir, stream_factory)
        ev = run_queries(lcfg, roles, parent, bdir, stream_factory, expert)
        texts = evidence_blocks(ev, np.array([ev["step_sizes"][c] for c in COORDS]))
        row = {"block": b, "seed": int(seed),
               "parent": {k: parent[k] for k in ("screen", "probe", "promotion", "fresh", "promotion_vs_null")},
               "evidence": {k: ev[k] for k in ("episodes", "step_sizes", "queries", "saturated_fraction",
                                               "base_saturated_fraction", "demo_distance", "evidence_sha256")},
               "packets": {k: texts[k] for k in ("raw_extra_states", "raw_chars", "sensitivity_chars")}}
        row["arms"] = {arm: run_arm(arm, lcfg, roles, parent, texts[arm], bdir, stream_factory, client)
                       for arm in ARMS}
        blocks.append(row)
        json.dump(summarise(blocks), open(os.path.join(sdir, "summary.partial.json"), "w"), indent=2, default=str)
    summary = summarise(blocks)
    with open(os.path.join(sdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    return summary


__all__ = ["ARMS", "COORDS", "ROLES", "block_roles", "coordinates", "evidence_blocks", "freeze_manifests", "perturb",
           "query_batch", "run_study", "select_states", "sensitivities", "step_sizes", "summarise"]
