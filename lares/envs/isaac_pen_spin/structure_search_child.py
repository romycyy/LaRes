"""R-109: the Isaac process of the structure-search study (``python -m lares.envs.isaac_pen_spin.structure_search_child req``).

One fresh process per request, launched by the driver through ``protocols.isaac_shell``:

* ``smoke``  -- inheritance, activation, scorer and A/A checks; saves the shared round-1 observation sample.
* ``train``  -- exact-success ARS for one ``(block, arm, round)``: each job trains one structure from inherited
  coefficients on its own placement slices and returns its snapshots.
* ``select`` -- parent, two candidates and the parent again on shared placements, three batches with rotated slots.
* ``cell``   -- one policy on one whole frozen manifest through the established evaluator (``protocols.run_cell``).

Slots of one batch may hold different structures: the batch runs the union of their terms, and a slot's
coefficients for a term it does not have are zero, which contributes exactly nothing.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

SAMPLE_STEPS = (40, 100, 160, 220, 280, 340)


def rollout(env, ctx, ctrl, vecs, theta_env, *, plain=None, sample: bool = False) -> dict:
    """One batch on the given placements with one coefficient row per environment; first episodes only.

    Returns per-environment outcomes and residual sums, the simulator starts made, and (``sample``) the alive
    environments' observations, reference actions and filter values at :data:`SAMPLE_STEPS`."""
    import numpy as np
    import torch

    from lares.envs.isaac_pen_spin import cases
    from lares.envs.isaac_pen_spin import structure_terms as st
    from lares.envs.isaac_pen_spin.runner import _write_placements
    from lares.envs.isaac_pen_spin.success_ars import first_episode_scores

    spec = cases.STAGES[3]
    n, dev = env.num_envs, env.device
    env.reset()
    _write_placements(env, vecs)
    obs = env._get_observations()["policy"]
    ctrl.reset(ctx)
    if plain is not None:
        plain.reset(ctx)
    T = int(env.max_episode_length)
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    completed = torch.zeros(n, dtype=torch.bool, device=dev)
    length = torch.zeros(n, dtype=torch.long, device=dev)
    pred = torch.zeros((T, n), dtype=torch.bool, device=dev)
    zero = lambda *shape: torch.zeros(*shape, device=dev, dtype=torch.float64)   # noqa: E731
    prop, exe, sat = zero(n, 20), zero(n, 20), zero(n, 20)
    term_abs = zero(n, len(ctrl.terms))
    spin_abs, final_dist = zero(n), zero(n)
    starts, mism, compared, first_action = n, 0, 0, None
    rows = {"obs": [], "ref": [], "v_f": [], "env": []}
    for t in range(T):
        p = ctrl.parts(obs, t, ctx)
        mu = ctrl.mean(p, theta_env)
        action = mu.clamp(-1.0, 1.0)
        if t == 0:
            first_action = action.clone()
        if plain is not None:
            a_plain = plain(obs, t, ctx).to(action.device)
            mism += int((action[alive] != a_plain[alive]).any(dim=1).sum())
            compared += int(alive.sum())
        if sample and t in SAMPLE_STEPS:
            keep = alive.nonzero().flatten()
            rows["obs"].append(obs[keep].cpu().numpy())
            rows["ref"].append(p["ref"][keep].cpu().numpy())
            rows["v_f"].append(ctrl.ref.v_f[keep].cpu().numpy())
            rows["env"].append(keep.cpu().numpy())
        live = alive.unsqueeze(1).double()
        prop += live * (mu - p["ref"]).abs().double()
        exe += live * (action - p["ref"]).abs().double()
        sat += live * (action.abs() >= 1.0).double()
        for k, c in enumerate(st.term_contributions(ctrl.terms, theta_env, p["feats"])):
            term_abs[:, k] += alive.double() * c.abs().mean(dim=1).double()
        obs_d, _, terminated, truncated, _ = env.step(action)
        obs = obs_d["policy"]
        done = terminated | truncated
        ctrl.reset_envs(done)
        starts += int(done.sum())
        step_ok = (env.center_dist < spec.success_center_tolerance) & (env.spin_rate.abs() > spec.success_spin_rate)
        pred[t] = alive & step_ok
        spin_abs += alive.double() * env.spin_rate.abs().double()
        final_dist = torch.where(alive, env.center_dist.double(), final_dist)
        length = torch.where(alive, length + 1, length)
        completed = torch.where(alive & done, truncated & ~terminated, completed)
        alive = alive & ~done
        if not bool(alive.any()):
            break
    L = length.cpu().numpy()
    sc = first_episode_scores(pred.cpu().numpy(), L, completed.cpu().numpy())
    steps = np.maximum(L, 1)[:, None]
    out = {"sustained": sc["sustained"], "completed": completed.cpu().numpy(), "length": L,
           "final_center_dist": final_dist.cpu().numpy(), "mean_abs_spin": spin_abs.cpu().numpy() / steps[:, 0],
           "proposed": prop.cpu().numpy() / steps, "executed": exe.cpu().numpy() / steps,
           "saturated": sat.cpu().numpy() / steps, "term_abs": term_abs.cpu().numpy() / steps,
           "starts": starts, "env_steps": int((t + 1) * n), "identity": {"compared": compared, "mismatches": mism},
           "first_action": first_action}
    if sample:
        out["sample"] = {k: np.concatenate(v) for k, v in rows.items()}
    return out


def slot_stats(res: dict, sl: slice, term_names: list) -> dict:
    """One slot's summary: what the LLM arm may read about a parent, and what the driver records per candidate."""
    import numpy as np

    from lares.envs.isaac_pen_spin import cases
    from lares.envs.isaac_pen_spin import symbolic_reward as sr

    tol = cases.STAGES[3].success_center_tolerance
    sus, comp = res["sustained"][sl], res["completed"][sl]
    tail = comp & ~sus
    off = tail & (res["final_center_dist"][sl] >= tol)
    return {"cases": int(sus.size), "successes": int(sus.sum()), "success": float(sus.mean()), "drops": int((~comp).sum()),
            "slow_spin": int((tail & ~off).sum()), "off_centre": int(off.sum()),
            "mean_abs_axial_rate": float(res["mean_abs_spin"][sl].mean()),
            "per_joint_executed_residual": {j: float(res["executed"][sl][:, i].mean()) for i, j in enumerate(sr.JOINTS)},
            "per_joint_saturation": {j: float(res["saturated"][sl][:, i].mean()) for i, j in enumerate(sr.JOINTS)},
            "per_term_mean_abs_contribution": {name: float(res["term_abs"][sl][:, k].mean()) for k, name in enumerate(term_names)}}


def _env_and_ctx(protocol_cfg: dict, device: str, seed: int):
    from lares.envs.isaac_pen_spin.runner import ControllerContext
    from lares.envs.isaac_pen_spin.symbolic_reward_child import make_training_env

    lay = protocol_cfg["layout"]
    env = make_training_env(int(lay["slots"]) * int(lay["cases_per_slot"]), device, seed)
    return env, ControllerContext(env)


def _union_controller(protocol_cfg: dict, policies: list, name: str):
    from lares.envs.isaac_pen_spin import structure_search as ss
    from lares.envs.isaac_pen_spin import structure_terms as st

    terms, thetas = ss.union_layout(policies)
    ctrl = st.build_controller(protocol_cfg, name, st.policy_record(terms, [0.0] * st.n_coefficients(terms)), ss.normalization(protocol_cfg))
    return ctrl, terms, thetas


def _lay(protocol_cfg: dict) -> dict:
    lay = protocol_cfg["layout"]
    return {"slots": int(lay["slots"]), "per": int(lay["cases_per_slot"]), "directions": int(protocol_cfg["ars"]["directions"]),
            "num_envs": int(lay["slots"]) * int(lay["cases_per_slot"]), "rollouts": 2}


def _save_sample(path: str, sample: dict, envs, rows: int) -> None:
    import numpy as np

    keep = np.isin(sample["env"], envs).nonzero()[0][:rows]
    np.savez_compressed(path, obs=sample["obs"][keep], ref=sample["ref"][keep], v_f=sample["v_f"][keep])


def smoke(req: dict, protocol_cfg: dict, device: str) -> dict:
    import numpy as np
    import torch

    from lares.envs.isaac_pen_spin import cases
    from lares.envs.isaac_pen_spin import structure_search as ss
    from lares.envs.isaac_pen_spin import structure_terms as st
    from lares.envs.isaac_pen_spin import success_ars as sa
    from lares.envs.isaac_pen_spin import symbolic_reward as sr

    sm, lay = protocol_cfg["smoke"], _lay(protocol_cfg)
    env, ctx = _env_and_ctx(protocol_cfg, device, int(sm["seed"]))
    out = {"mode": "smoke", "seed": int(sm["seed"]), "gates": {}}
    try:
        parent = ss.root_policy(protocol_cfg)
        probe = st.validate_edit(sm["probe_edit"], parent)
        child = st.extend(parent, probe)
        ctrl, terms, thetas = _union_controller(protocol_cfg, [parent, child], "smoke")
        names = st.coefficient_names(terms)
        to_t = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)   # noqa: E731
        out["gates"]["joint_order"] = {"passes": sr.executed_joint_order(ctx) == list(sr.JOINTS)}
        vecs = sa.tile(cases.placements(int(sm["seed"]), lay["per"]), lay["slots"])
        r106 = sr.build_controller(protocol_cfg, "smoke_r106", parent["theta"][:sr.N_COEF], ss.normalization(protocol_cfg))
        t0 = time.time()
        # A: every slot is the parent written as the union structure (the child's new term at zero), against R-106's class
        a = rollout(env, ctx, ctrl, vecs, to_t(sa.slot_thetas(np.stack([thetas[1]] * lay["slots"]), lay["per"])), plain=r106, sample=True)
        out["gates"]["inheritance_identity"] = {**a["identity"], "passes": a["identity"]["mismatches"] == 0 and a["identity"]["compared"] > 0,
                                                "note": "base terms plus one zero-coefficient new term equal the R-106 controller bit for bit"}
        sus = a["sustained"].reshape(lay["slots"], lay["per"])
        out["aa"] = sa.aa_statistics(sus)
        out["gates"]["scorer_live"] = {"parent_successes": int(sus.sum()), "passes": bool(sus.any())}
        sdir = os.path.dirname(req["response_path"])
        _save_sample(os.path.join(sdir, "parent_sample.npz"), a["sample"], np.arange(lay["num_envs"]), int(protocol_cfg["sample_rows"]))
        term_names = [json.dumps(t["expr"], separators=(",", ":")) + "@" + t["target"] for t in terms]
        out["parent_stats"] = slot_stats(a, slice(0, lay["num_envs"]), term_names)
        # B: zero, +probe, -probe, +probe on the new term's coefficients
        k = st.n_slots(probe["target"])
        plus = np.array(thetas[1], dtype=float)
        plus[-k:] = float(sm["probe_magnitude"])
        minus = np.array(thetas[1], dtype=float)
        minus[-k:] = -float(sm["probe_magnitude"])
        b = rollout(env, ctx, ctrl, vecs, to_t(sa.slot_thetas(np.stack([thetas[1], plus, minus, plus]), lay["per"])))
        wall = time.time() - t0
        per = lay["per"]
        fa = b["first_action"]
        slot = lambda x, i: x[i * per:(i + 1) * per]   # noqa: E731
        dup = float((slot(fa, 1) - slot(fa, 3)).abs().max())
        base_exec, probe_exec = float(slot(b["executed"], 0).mean()), float(slot(b["executed"], 1).mean())
        out["gates"]["activation"] = {"plus_slots_max_abs_difference_at_step_0": dup, "new_term_contribution": float(slot(b["term_abs"], 1)[:, -1].mean()),
                                      "zero_slot_new_term_contribution": float(slot(b["term_abs"], 0)[:, -1].mean()),
                                      "executed_residual_zero_vs_probe": [base_exec, probe_exec],
                                      "passes": bool(dup <= float(sm["duplicate_slot_tolerance"]) and float(slot(b["term_abs"], 1)[:, -1].mean()) > 0
                                                     and float(slot(b["term_abs"], 0)[:, -1].mean()) == 0.0
                                                     and not torch.equal(slot(fa, 1), slot(fa, 2)))}
        out["probe_slots"] = {"slot_success": [float(x) for x in b["sustained"].reshape(lay["slots"], per).mean(axis=1)],
                              "coefficients": names[-k:], "note": "slots: zero, +probe, -probe, +probe; a diagnostic only"}
        out["starts"] = a["starts"] + b["starts"]
        out["transitions"] = a["env_steps"] + b["env_steps"]
        out["throughput"] = {"transitions_per_s": out["transitions"] / wall, "wall_time_s": wall}
        out["passes"] = all(g["passes"] for g in out["gates"].values())
    finally:
        env.close()
    return out


def train(req: dict, protocol_cfg: dict, device: str) -> dict:
    """Jobs ``{"policy", "updates": [global u ...], "snapshots": [local counts ...]}`` for one (block, arm, round)."""
    import numpy as np
    import torch

    from lares.envs.isaac_pen_spin import structure_search as ss
    from lares.envs.isaac_pen_spin import structure_terms as st
    from lares.envs.isaac_pen_spin import success_ars as sa

    lay, a = _lay(protocol_cfg), protocol_cfg["ars"]
    block, rnd = int(req["block"]), int(req["round"])
    seed = int(protocol_cfg["block_seeds"][block])
    env, ctx = _env_and_ctx(protocol_cfg, device, seed + 50 + rnd)
    cap, starts, transitions, complete, note = int(req["start_cap"]), 0, 0, True, None
    bound = float(protocol_cfg["actor"]["bound"])
    results, t0 = [], time.time()
    try:
        for job in req["jobs"]:
            policy = st.check_policy(job["policy"])
            ctrl = st.build_controller(protocol_cfg, "train", policy, ss.normalization(protocol_cfg))
            theta = np.array(policy["theta"], dtype=float)
            snaps, log = [], []
            for i, u in enumerate(job["updates"]):
                if starts + 4 * lay["num_envs"] > cap:
                    complete, note = False, f"job {len(results)} update {i}: {starts} starts, cap {cap}"
                    break
                delta = st.ars_directions(ss.direction_seed(protocol_cfg, block, rnd, u), lay["directions"], theta.size)
                cand = st.ars_candidates(theta, delta, float(a["nu"]), bound)
                vecs = sa.tile(ss.training_placements(protocol_cfg, block, rnd, u), lay["slots"])
                slot_scores = []
                for r in range(lay["rollouts"]):
                    rc = sa.rollout_candidates(cand, r, lay)
                    res = rollout(env, ctx, ctrl, vecs, torch.as_tensor(sa.slot_thetas(rc, lay["per"]), dtype=torch.float32, device=device))
                    starts += res["starts"]
                    transitions += res["env_steps"]
                    slot_scores.append(res["sustained"].reshape(lay["slots"], lay["per"]).mean(axis=1))
                scores = sa.ars_scores(slot_scores, lay)
                theta, info = st.ars_step(theta, delta, scores, float(a["alpha"]), bound, int(a["top"]))
                n = lay["directions"]
                log.append({"u": int(u), "plus": [float(x) for x in scores[:n]], "minus": [float(x) for x in scores[n:]],
                            "mean_candidate_success": float(scores.mean()), "sigma_r": info["sigma_r"],
                            "theta_l2": float(np.linalg.norm(theta)), "theta_abs_max": float(np.abs(theta).max())})
                print("update", json.dumps({"job": len(results), **log[-1], "starts": starts}), flush=True)
                if i + 1 in job["snapshots"]:
                    snaps.append(st.policy_record(policy["terms"], theta))
            if not complete:
                break
            results.append({"snapshots": snaps, "updates": log})
    finally:
        env.close()
    return {"mode": "train", "block": block, "arm": req["arm"], "round": rnd, "complete": complete, "note": note,
            "jobs": results, "starts": starts, "transitions": transitions, "wall_time_s": time.time() - t0}


def select(req: dict, protocol_cfg: dict, device: str) -> dict:
    """Parent, candidate 0, candidate 1, parent again, on shared placements; slots rotate by one per batch."""
    import numpy as np
    import torch

    from lares.envs.isaac_pen_spin import cases
    from lares.envs.isaac_pen_spin import structure_search as ss
    from lares.envs.isaac_pen_spin import structure_terms as st
    from lares.envs.isaac_pen_spin import success_ars as sa

    lay = _lay(protocol_cfg)
    block, rnd = int(req["block"]), int(req["round"])
    roles = ("parent", "c0", "c1", "parent_dup")
    policies = [st.check_policy(req["parent"]), st.check_policy(req["candidates"][0]), st.check_policy(req["candidates"][1])]
    ctrl, terms, thetas = _union_controller(protocol_cfg, policies, "select")
    by_role = {"parent": thetas[0], "c0": thetas[1], "c1": thetas[2], "parent_dup": thetas[0]}
    term_names = [json.dumps(t["expr"], separators=(",", ":")) + "@" + t["target"] for t in terms]
    seeds = ss.selection_seeds(protocol_cfg, block, rnd)
    env, ctx = _env_and_ctx(protocol_cfg, device, int(seeds[0]))
    per = lay["per"]
    rows = {r: {"sustained": [], "completed": []} for r in roles}
    stats = {r: [] for r in roles}
    samples = {r: {"obs": [], "ref": [], "v_f": []} for r in roles}
    starts = transitions = 0
    try:
        for bi, seed in enumerate(seeds):
            order = [roles[(k + bi) % len(roles)] for k in range(lay["slots"])]          # rotated slot assignment
            vecs = sa.tile(cases.placements(int(seed), per), lay["slots"])
            theta_env = torch.as_tensor(sa.slot_thetas(np.stack([by_role[r] for r in order]), per), dtype=torch.float32, device=device)
            res = rollout(env, ctx, ctrl, vecs, theta_env, sample=True)
            starts += res["starts"]
            transitions += res["env_steps"]
            for k, role in enumerate(order):
                sl = slice(k * per, (k + 1) * per)
                rows[role]["sustained"].append(res["sustained"][sl])
                rows[role]["completed"].append(res["completed"][sl])
                stats[role].append(slot_stats(res, sl, term_names))
                keep = np.isin(res["sample"]["env"], np.arange(k * per, (k + 1) * per))
                for key in samples[role]:
                    samples[role][key].append(res["sample"][key][keep])
    finally:
        env.close()
    sdir = os.path.dirname(req["response_path"])
    stem = os.path.basename(req["response_path"]).replace(".json", "")
    sample_paths = {}
    for role in ("parent", "c0", "c1"):                      # a sample for every selection slot: the retained one is reused
        path = os.path.join(sdir, f"{stem}_{role}_sample.npz")
        n = int(protocol_cfg["sample_rows"])
        np.savez_compressed(path, **{k: np.concatenate(v)[:n] for k, v in samples[role].items()})
        sample_paths[role] = path
    return {"mode": "select", "block": block, "arm": req["arm"], "round": rnd, "seeds": [int(s) for s in seeds],
            "rows": {r: {k: np.concatenate(v).astype(int).tolist() for k, v in d.items()} for r, d in rows.items()},
            "batch_stats": stats, "sample_paths": sample_paths, "starts": starts, "transitions": transitions}


def cell(req: dict, protocol_cfg: dict, device: str) -> dict:
    """One policy on one whole frozen manifest through the established evaluator."""
    from lares.envs.isaac_pen_spin import protocols
    from lares.envs.isaac_pen_spin import structure_search as ss
    from lares.envs.isaac_pen_spin import structure_terms as st
    from lares.envs.isaac_pen_spin import symbolic_reward as sr

    if req["policy"] == "reference":
        ctrl = sr.build_reference(protocol_cfg, "reference", resettable=False)
    else:
        ctrl = st.build_controller(protocol_cfg, req["name"], st.check_policy(req["policy"]), ss.normalization(protocol_cfg))
    protocols.run_cell(req["manifest"], ctrl, req["cell_out"], device, record_obs=False)
    return {"mode": "cell", "cell": req["cell_out"]}


def main(request_path: str) -> None:
    from lares.envs.isaac_lab_adapter import DEFAULT_ISAACLAB_ROOT, _ensure_isaaclab_on_path, close_isaac_app, launch_isaac_app
    from lares.utils.cli import load_yaml

    with open(request_path, encoding="utf-8") as f:
        req = json.load(f)
    protocol_cfg = load_yaml(req["config_path"])[req["config_key"]]
    device = str(req.get("device", "cuda:0"))
    _ensure_isaaclab_on_path(DEFAULT_ISAACLAB_ROOT)
    launch_isaac_app(headless=True, device=device)
    code = 0
    try:
        from lares.envs.isaac_pen_spin import structure_search as ss

        if ss.config_sha256(protocol_cfg) != req["config_sha256"]:
            raise SystemExit("the config block changed since the driver froze it")
        out = {"smoke": smoke, "train": train, "select": select, "cell": cell}[req["mode"]](req, protocol_cfg, device)
        with open(req["response_path"], "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
        print("wrote", req["response_path"], flush=True)
    except BaseException:  # Isaac's close swallows tracebacks: print before it runs
        traceback.print_exc()
        sys.stdout.flush()
        code = 1
    finally:
        if code:
            os._exit(code)
        close_isaac_app()


if __name__ == "__main__":
    main(sys.argv[1])
