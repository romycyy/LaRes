"""R-042: expert demonstrations and keyframe trace packets for plateau repair on Shadow Hand pen spinning.

The PPO expert succeeds only through state feedback (D-037), so a trace shows state-action pairs: at each kept step,
the pen's state as the policy observes it, the hand's joint positions, and the executed action. Two keyframe rules
pick the 16 steps per trace, frozen before any outcome (R-042 protocol):

* ``uniform``: ``t_k = round(k (L - 1) / 15)``, ``k = 0..15``.
* ``event``: endpoints ``0`` and ``L - 1`` always kept; interior steps scored
  ``s_t = ||a_t - a_{t-1}|| / m_a + ||w_t - w_{t-1}|| / m_w`` (``a`` the executed action, ``w`` the pen's world angular
  velocity in rad/s, ``m_*`` the medians over ``t = 1..L-1`` of the same trace's step-to-step norms, a zero median
  replaced by 1), taken greedily by descending score, ties to the earlier step, skipping any step within
  ``MIN_SPACING`` of one already kept, until 16 are kept; listed in time order.

Both arms render the same fields with the same rounding, so the packets differ only in which steps they show.
"""

from __future__ import annotations

import json

import numpy as np

from lares.utils.hashing import sha256_text

KEYFRAMES = 16
MIN_SPACING = 6
UNIFORM = "uniform"
EVENT = "event"
KEYFRAME_RULES = (UNIFORM, EVENT)
TRACE_EPISODES = 2

GOAL_PHRASE = "keep the pen in the hand and roll it about its long axis fast enough for the whole episode"
HEADER = ("Successful expert episodes (the expert is a closed-loop policy: each action is its response to the state "
          "on the same line). Per kept step: t; STATE pen_offset = pen centre minus in_hand_pos (m), pen_axis = the "
          "pen's long axis in the world frame, pen_angvel = pen angular velocity (rad/s, world frame, NOT scaled by "
          "0.2), roll = signed spin rate about the pen's own axis (rad/s), hand = hand_dof_pos exactly as the policy "
          "observes it (24 joint positions normalised to [-1, 1], not radians; hand joint order); ACTION = the 20 "
          "executed actions (action order).")

REQUEST_TEMPLATE = """Below is a controller for this task. It does not reliably {goal_phrase}. An expert
policy solves this task; it succeeded on {successes} of {episodes} demonstration episodes. It is closed-loop: in one
test, replaying one of its recorded action sequences without feedback succeeded on 0 of 40 episodes, against 40 of
40 for the expert itself.

Rewrite the controller so that it succeeds on its own; no expert is available. You may change any part of it. Keep
the class name GeneratedPolicy, read observations only through self.obs_field, and declare every parameter with
get_param_ranges.

```python
{parent}
```
{parameters}"""


def episode_rows(obs: np.ndarray, actions: np.ndarray, length: int) -> list[dict]:
    """Per-step state-action rows of one episode (unrounded), read through the named observation layout."""
    from lares.envs.isaac_pen_spin import cases

    sl = cases.obs_slices()
    obs = np.asarray(obs[: int(length)], dtype=np.float64)
    act = np.asarray(actions[: int(length)], dtype=np.float64)
    q = obs[:, sl["obj_rot"]]
    axis = cases.pen_axis(q)
    angvel = obs[:, sl["obj_angvel"]] / cases.VEL_OBS_SCALE
    roll = np.sum(angvel * axis, axis=-1)
    offset = obs[:, sl["obj_pos"]] - obs[:, sl["in_hand_pos"]]
    return [{"t": t, "pen_offset": offset[t].tolist(), "pen_axis": axis[t].tolist(), "pen_angvel": angvel[t].tolist(),
             "roll": float(roll[t]), "hand": obs[t, sl["hand_dof_pos"]].tolist(), "action": act[t].tolist()}
            for t in range(int(length))]


def demonstrations_from_traces(manifest, collected: dict) -> dict:
    """The demonstration record in the shape ``traces.collect_expert_traces`` returns."""
    episodes = []
    for i, (case, row) in enumerate(zip(manifest.episodes, collected["episodes"])):
        if row["case_id"] != case.case_id:
            raise ValueError("demonstration rows do not follow the manifest's cases")
        episodes.append({"case": case.case_id, "success": float(row["sustained"]),
                         "tail_hit_fraction": float(row["tail_hit_fraction"]), "length": int(row["length"]),
                         "rows": episode_rows(collected["obs"][i], collected["actions"][i], row["length"])})
    return {"manifest_id": manifest.manifest_id, "episodes": episodes,
            "successes": int(sum(e["success"] for e in episodes)), "trace_path": collected.get("trace_path")}


# ---------------------------------------------------------------------------------------------------------------------
#  Keyframe rules
# ---------------------------------------------------------------------------------------------------------------------


def uniform_keyframes(length: int, n: int = KEYFRAMES) -> list[int]:
    L = int(length)
    if L < n:
        raise ValueError(f"a trace of {L} steps cannot show {n} keyframes")
    return [int(round(k * (L - 1) / (n - 1))) for k in range(n)]


def event_scores(rows: list[dict]) -> np.ndarray:
    """``s_t`` for every step (``s_0`` and ``s_{L-1}`` are not used; the endpoints are always kept)."""
    a = np.asarray([r["action"] for r in rows], dtype=np.float64)
    w = np.asarray([r["pen_angvel"] for r in rows], dtype=np.float64)
    da = np.linalg.norm(np.diff(a, axis=0), axis=-1)
    dw = np.linalg.norm(np.diff(w, axis=0), axis=-1)
    ma, mw = float(np.median(da)), float(np.median(dw))
    ma, mw = (ma if ma > 0 else 1.0), (mw if mw > 0 else 1.0)
    return np.concatenate([[0.0], da / ma + dw / mw])


def event_keyframes(rows: list[dict], n: int = KEYFRAMES, spacing: int = MIN_SPACING) -> list[int]:
    L = len(rows)
    if L < n:
        raise ValueError(f"a trace of {L} steps cannot show {n} keyframes")
    s = event_scores(rows)
    kept = [0, L - 1]
    # Descending score, ties to the earlier step: sort by (-score, t).
    for t in sorted(range(1, L - 1), key=lambda t: (-s[t], t)):
        if len(kept) == n:
            break
        if all(abs(t - k) >= spacing for k in kept):
            kept.append(t)
    if len(kept) < n:
        raise ValueError(f"only {len(kept)} keyframes fit {spacing}-step spacing in {L} steps")
    return sorted(kept)


def keyframes(rows: list[dict], rule: str) -> list[int]:
    if rule == UNIFORM:
        return uniform_keyframes(len(rows))
    if rule == EVENT:
        return event_keyframes(rows)
    raise ValueError(f"unknown keyframe rule {rule!r}; one of {KEYFRAME_RULES}")


# ---------------------------------------------------------------------------------------------------------------------
#  Rendering
# ---------------------------------------------------------------------------------------------------------------------


def _fmt(values, dp: int) -> str:
    return "[" + ", ".join(f"{float(v):.{dp}f}" for v in values) + "]"


def render_row(r: dict) -> str:
    return (f"  t={r['t']} STATE pen_offset={_fmt(r['pen_offset'], 4)} pen_axis={_fmt(r['pen_axis'], 3)} "
            f"pen_angvel={_fmt(r['pen_angvel'], 2)} roll={r['roll']:.2f} hand={_fmt(r['hand'], 3)} "
            f"ACTION={_fmt(r['action'], 3)}")


def select_traces(demos: dict, n: int = TRACE_EPISODES) -> list:
    """The first ``n`` sustained-success episodes, in manifest order."""
    return [e for e in demos["episodes"] if e["success"] and e["rows"]][:n]


def trace_block(selected: list, rule: str) -> tuple[str, list]:
    """The rendered packet and the kept steps per episode."""
    lines, kept = ["", f"{HEADER} {KEYFRAMES} steps per episode are shown."], []
    for k, e in enumerate(selected):
        idx = keyframes(e["rows"], rule)
        kept.append(idx)
        lines.append(f"Episode {k + 1} ({len(e['rows'])} steps):")
        lines += [render_row(e["rows"][t]) for t in idx]
    return "\n".join(lines), kept


def parameters_block(parameters: dict | None) -> str:
    if not parameters:
        return ""
    def fmt(v):
        return f"{v:.4g}" if isinstance(v, (int, float)) else "[" + ", ".join(f"{x:.4g}" for x in v) + "]"

    vals = ", ".join(f"{k}={fmt(v)}" for k, v in sorted(parameters.items()))
    return f"\nThe rollout search selected these parameter values for this controller: {vals}\n"


def repair_request(parent_source: str, successes: int, episodes: int, selected: list, rule: str,
                   parameters: dict | None = None) -> tuple[str, dict]:
    """The request text and its packet metadata (kept steps, sizes, hashes)."""
    head = (REQUEST_TEMPLATE.replace("{goal_phrase}", GOAL_PHRASE).replace("{successes}", str(successes))
            .replace("{episodes}", str(episodes)).replace("{parent}", parent_source.strip())
            .replace("{parameters}", parameters_block(parameters)))
    block, kept = trace_block(selected, rule)
    text = head + block
    return text, {"keyframe_rule": rule, "keyframes": KEYFRAMES, "min_spacing": MIN_SPACING, "kept_steps": kept,
                  "packet_chars": len(block), "request_chars": len(text), "packet_sha256": sha256_text(block),
                  "traces_case_ids": [e["case"] for e in selected]}


def traces_sha256(selected: list) -> str:
    return sha256_text(json.dumps(selected, sort_keys=True))


__all__ = ["EVENT", "KEYFRAMES", "KEYFRAME_RULES", "MIN_SPACING", "UNIFORM", "demonstrations_from_traces",
           "episode_rows", "event_keyframes", "event_scores", "keyframes", "render_row", "repair_request",
           "select_traces", "trace_block", "traces_sha256", "uniform_keyframes"]
