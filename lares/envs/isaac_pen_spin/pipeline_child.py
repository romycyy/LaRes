"""R-042: one pen-spin evaluation in a fresh Isaac process (``python -m lares.envs.isaac_pen_spin.pipeline_child req``).

Reads a request written by :class:`~lares.envs.isaac_pen_spin.pipeline_backend.IsaacPenSpinStream`, builds the actor
(a generated symbolic policy from its source and state, executed as ``clamp(mean, -1, 1)``; or the frozen PPO expert,
its checkpoint hash checked), runs every case of the manifest in one batch with ``PenSpinBatchRunner`` (which zeroes
the previous action at reset), and writes the response: per-episode rows and, when asked, the observation and action
traces. Run under Isaac Lab's environment only.
"""

from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class SymbolicController:
    """A generated SymbolicPolicy behind the runner's controller interface."""

    name = "symbolic_policy"

    def __init__(self, spec: dict, env_id: str):
        import torch

        from lares.core.obs_schema import active_schema, get_obs_schema
        from lares.core.policy_generation import GENERATED_IMPORTS

        namespace: dict = {}
        exec(GENERATED_IMPORTS + spec["source"], namespace)
        with active_schema(get_obs_schema(env_id)):
            policy = namespace["GeneratedPolicy"](int(spec["obs_dim"]), int(spec["action_dim"]))
        policy.fixed_action_scale = float(spec["fixed_action_scale"])
        policy.load_state_dict(torch.load(spec["state_path"], weights_only=True))
        policy.eval()
        self.policy = policy
        self.params = {"execution": spec["execution"]}

    def reset(self, ctx):
        reset = getattr(self.policy, "reset_diagnostics", None)
        if callable(reset):
            reset()

    def mean(self, obs):
        """The policy mean, unclamped (NaN read as 0)."""
        # The policy runs on the CPU, as every MetaWorld evaluation does: generated code builds constants with
        # torch.tensor(...) inside forward, which would otherwise sit on another device than the observation.
        import torch

        with torch.no_grad():
            mean, _ = self.policy(obs.detach().float().cpu())
        return torch.nan_to_num(mean, nan=0.0)

    def __call__(self, obs, t, ctx):
        return self.mean(obs).clamp(-1.0, 1.0).to(obs.device)


def build_controller(spec: dict, env_id: str):
    """The request's actor. ``schema_env_id`` (R-046) names the schema a policy reads when it is not the stage's."""
    from lares.envs.isaac_pen_spin import gait_scaffold

    if spec.get("shadow_gait"):
        return gait_scaffold.ShadowGait(build_controller({k: v for k, v in spec.items() if k != "shadow_gait"}, env_id))
    if spec["kind"] == "gait":
        return gait_scaffold.ScaffoldController("unchanged", gait_sha256=spec["gait_sha256"])
    if spec["kind"] == "symbolic" and spec.get("scaffold"):
        sc = spec["scaffold"]
        policy = SymbolicController(spec, spec.get("schema_env_id", env_id))
        return gait_scaffold.ScaffoldController(sc["mode"], policy.mean, float(sc["bound"]), sc["gait_sha256"])
    if spec["kind"] == "symbolic":
        return SymbolicController(spec, spec.get("schema_env_id", env_id))
    if spec["kind"] == "ppo":
        from lares.envs.isaac_pen_spin.ppo_expert import PPOController
        from lares.utils.hashing import sha256_file

        got = sha256_file(spec["checkpoint"])
        if got != spec["checkpoint_sha256"]:
            raise SystemExit(f"checkpoint {spec['checkpoint']} hash {got} != frozen {spec['checkpoint_sha256']}")
        return PPOController(spec["checkpoint"], int(spec["training_seed"]))
    raise ValueError(f"unknown actor kind {spec['kind']!r}")


def write_reset_observations(req: dict, manifest, stage: int) -> None:
    """Each case's observation after its placement is written, before any step (the reconstruction batch)."""
    from lares.envs.isaac_pen_spin import cases
    from lares.envs.isaac_pen_spin.runner import _write_placements, make_env

    env = make_env(stage, len(manifest), req.get("device", "cuda:0"))
    try:
        env.reset()
        _write_placements(env, cases.resolve_placements(manifest))
        obs = env._get_observations()["policy"].detach().cpu().numpy()
    finally:
        env.close()
    with open(req["response_path"], "w", encoding="utf-8") as f:
        json.dump({"request_path": req["manifest_path"], "manifest_id": manifest.manifest_id,
                   "observations": obs.tolist(), "episodes": []}, f)


def write_queried_actions(req: dict, stage: int) -> None:
    """The actor's raw output at given observations (R-048): no placement, no step, no episode."""
    import numpy as np
    import torch

    from lares.envs.isaac_pen_spin import cases
    from lares.envs.isaac_pen_spin.runner import make_env
    from lares.utils.hashing import sha256_file

    obs = np.load(req["query_path"])["obs"].astype(np.float32)
    device = req.get("device", "cuda:0")
    ctrl = build_controller(req["actor"], cases.STAGES[stage].task_id)
    env = make_env(stage, 1, device)
    try:
        env.reset()
        if hasattr(ctrl, "attach"):
            ctrl.attach(env)
        out = []
        for a in range(0, len(obs), 256):
            o = torch.as_tensor(obs[a:a + 256], device=device)
            out.append(ctrl(o, 0, None).detach().cpu().numpy())
    finally:
        env.close()
    path = req["response_path"].replace(".response.json", ".actions.npz")
    np.savez_compressed(path, actions=np.concatenate(out, axis=0))
    with open(req["response_path"], "w", encoding="utf-8") as f:
        json.dump({"request_path": req["manifest_path"], "actions_path": path, "actions_sha256": sha256_file(path),
                   "queries": int(len(obs)), "episodes": []}, f)


def main(request_path: str) -> None:
    from lares.envs.isaac_lab_adapter import close_isaac_app, launch_isaac_app

    with open(request_path, encoding="utf-8") as f:
        req = json.load(f)
    launch_isaac_app(headless=True, device=req.get("device", "cuda:0"))
    try:
        import numpy as np

        from lares.envs.isaac_pen_spin import cases
        from lares.envs.isaac_pen_spin.pipeline_backend import rows_from_trace
        from lares.envs.isaac_pen_spin.runner import PenSpinBatchRunner
        from lares.eval.manifest import EvaluationManifest
        from lares.utils.hashing import sha256_file

        manifest = EvaluationManifest.load(req["manifest_path"])
        stage = cases.stage_of(manifest)
        if stage != int(req["stage"]):
            raise SystemExit(f"request stage {req['stage']} != manifest stage {stage}")
        if req.get("mode") == "reset_obs":
            write_reset_observations(req, manifest, stage)
            return
        if req.get("mode") == "query":
            write_queried_actions(req, stage)
            return
        ctrl = build_controller(req["actor"], cases.STAGES[stage].task_id)
        runner = PenSpinBatchRunner(stage, len(manifest), device=req.get("device", "cuda:0"))
        if hasattr(ctrl, "attach"):
            ctrl.attach(runner.env)
        try:
            trace = runner.run(manifest, ctrl, record_obs=True)
        finally:
            runner.close()
        rows = rows_from_trace(trace)
        response = {"request_path": request_path, "manifest_id": manifest.manifest_id, "stage": stage,
                    "actor_kind": req["actor"]["kind"], "episodes": rows,
                    "reducer": f"sustained: time-out without a fall and the per-step predicate on all final "
                               f"{cases.SUSTAIN_WINDOW} steps (D-034)"}
        if req.get("record"):
            path = req["response_path"].replace(".response.json", ".trace.npz")
            extra = {"gait_actions": ctrl.gait_actions()} if hasattr(ctrl, "gait_actions") else {}
            np.savez_compressed(path, obs=trace["obs"], actions=trace["actions"], length=trace["length"],
                                completed=trace["completed"], pred=trace["pred"], spin_rate=trace["spin_rate"],
                                center_dist=trace["center_dist"], **extra)
            response["trace_path"] = path
            response["trace_sha256"] = sha256_file(path)
        with open(req["response_path"], "w", encoding="utf-8") as f:
            json.dump(response, f, indent=1, default=float)
    finally:
        close_isaac_app()


if __name__ == "__main__":
    main(sys.argv[1])
