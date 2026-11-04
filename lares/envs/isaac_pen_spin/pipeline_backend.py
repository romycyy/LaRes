"""R-042: the pen-spin task as a pipeline stream, one fresh Isaac process per evaluation.

``evaluate_manifest`` hands a manifest to :meth:`IsaacPenSpinStream.evaluate_manifest_batch` when the env it is given
is one of these streams. The stream runs in the pipeline's own Python (no Isaac import): it writes a request (the
policy's source and state, or the frozen PPO expert's checkpoint; the manifest path; whether to record traces), runs
``pipeline_child`` under Isaac Lab's environment, and turns the returned rows into an ordinary
:class:`~lares.eval.runner.ManifestResult`. Every evaluation is its own process because a second rollout in one Isaac
process is not reproducible (D-034), and the batch is the whole manifest, so a case's batch is the same every time the
same manifest is scored.

Episode semantics are the frozen D-034 sustained reducer at the stream's stage: ``success`` is sustained success,
``signed_goal_progress`` carries the tail-hit fraction (the share of the final 60 steps that satisfied the per-step
predicate), the only continuous progress term the pipeline ranks by. The executed action is the policy mean clamped
to [-1, 1] (the env's normalised joint-target interface), not ``tanh(mean)`` as MetaWorld's ``PolicyActor`` executes.
Fields with no pen-spin meaning (object proximity, grasp) are ``None``.
"""

from __future__ import annotations

import io
import json
import os
import shlex
import subprocess
import tempfile
from types import SimpleNamespace

import numpy as np

from lares.envs.isaac_pen_spin import cases

#: How a stream starts its child: Isaac Lab's environment script, then the child module with the request path.
DEFAULT_ISAAC_ENV = "~/IsaacLab/source_env.sh"
CHILD_MODULE = "lares.envs.isaac_pen_spin.pipeline_child"
EXECUTION = "clamp(mean, -1, 1)"


class IsaacExpertActor:
    """The frozen D-036 PPO expert as a pipeline actor (it only runs inside a child process)."""

    action_mode = "deterministic"

    def __init__(self, checkpoint: str, checkpoint_sha256: str, training_seed: int, shadow_gait: bool = False):
        self.checkpoint = checkpoint
        self.checkpoint_sha256 = checkpoint_sha256
        self.training_seed = int(training_seed)
        self.shadow_gait = bool(shadow_gait)
        self.name = "ppo_expert"

    def spec(self) -> dict:
        spec = {"kind": "ppo", "checkpoint": self.checkpoint, "checkpoint_sha256": self.checkpoint_sha256,
                "training_seed": self.training_seed}
        if self.shadow_gait:
            spec["shadow_gait"] = True
        return spec


class NeutralActor:
    """R-049: every joint held at its default pose, the scored stand-in for the null incumbent."""

    action_mode = "deterministic"
    name = "neutral_pose"

    def spec(self) -> dict:
        return {"kind": "neutral"}

    def reset(self, case):  # pragma: no cover - the child drives it
        pass

    def act(self, obs):  # pragma: no cover - never stepped in this process
        raise RuntimeError("the neutral pose runs only inside a pen-spin child process")


class GaitActor:
    """R-046: the frozen R-039 gait alone, as a pipeline actor (it only runs inside a child process)."""

    action_mode = "deterministic"
    name = "unchanged_gait"

    def __init__(self, gait_sha256: str):
        self.gait_sha256 = gait_sha256

    def spec(self) -> dict:
        return {"kind": "gait", "gait_sha256": self.gait_sha256}

    def reset(self, case):  # pragma: no cover - the child drives the gait
        pass

    def act(self, obs):  # pragma: no cover - never stepped in this process
        raise RuntimeError("the gait scaffold runs only inside a pen-spin child process")


def actor_spec(actor) -> tuple[dict, bytes | None]:
    """``(spec, state bytes)`` for a request: a symbolic policy by source and state, or the PPO expert."""
    if callable(getattr(actor, "spec", None)):  # the PPO expert, the unchanged gait (R-046): no state to ship
        return actor.spec(), None
    policy = getattr(actor, "policy", None)
    if policy is None:
        raise TypeError(f"{type(actor).__name__}: a pen-spin stream runs symbolic policies or the PPO expert only")
    if not getattr(actor, "deterministic", True):
        raise ValueError("a pen-spin stream executes the deterministic action only")
    source = getattr(policy, "_generated_source", None)
    if not source:
        raise ValueError("the policy carries no _generated_source; it cannot be rebuilt in an Isaac process")
    import torch

    buf = io.BytesIO()
    torch.save({k: v.detach().cpu() for k, v in policy.state_dict().items()}, buf)
    spec = {"kind": "symbolic", "source": source, "obs_dim": int(policy.obs_dim), "action_dim": int(policy.action_dim),
            "fixed_action_scale": float(getattr(policy, "fixed_action_scale", 0.1)), "execution": EXECUTION}
    scaffold = getattr(policy, "_scaffold", None)  # R-046: the policy's mean is a correction to the frozen gait
    if scaffold:
        spec.update(scaffold=dict(scaffold), schema_env_id=scaffold["schema_env_id"],
                    execution=f"gait scaffold, mode {scaffold['mode']}")
    residual = getattr(policy, "_parent_residual", None)  # R-084: a bounded residual on the D-058 incumbent
    if residual:
        spec.update(parent_residual=dict(residual), schema_env_id=residual["schema_env_id"],
                    execution=f"parent residual, bound {residual['bound']}, decoder {residual['decoder_name']}")
    return spec, buf.getvalue()


def _row_to_record(row: dict, task_id: str):
    from lares.eval.runner import EpisodeRecord

    return EpisodeRecord(
        case_id=row["case_id"], task_id=task_id, success=float(row["sustained"]), episode_return=float(row["return"]),
        length=int(row["length"]), terminal_obj_to_target=row["final_center_dist"],
        initial_obj_to_target=row.get("initial_center_dist"), near_object_rate=None, first_near_object_step=None,
        grasp_success_rate=None, action_saturation_rate=row.get("action_saturation_rate"),
        timed_out=bool(row["completed"]), signed_goal_progress=float(row["tail_hit_fraction"]),
        failure_label=row.get("failure_label"),
    )


def failure_label(row: dict) -> str | None:
    """Coarse pen-spin failure cause: ``dropped``, ``slow_spin`` (centred but not spinning), ``off_centre``."""
    if row["sustained"]:
        return None
    if not row["completed"]:
        return "dropped"
    if row["tail_hit_fraction"] >= 1.0:
        return None
    return "slow_spin" if row["final_center_dist"] is not None and row["final_center_dist"] < 0.08 else "off_centre"


class IsaacPenSpinStream:
    """One pipeline stream (development, probe, demonstration, ...) backed by fresh Isaac processes."""

    def __init__(self, stage: int, work_dir: str, name: str, isaac_env: str = DEFAULT_ISAAC_ENV,
                 device: str = "cuda:0", runner=None):
        self.stage = int(stage)
        self.spec = cases.STAGES[self.stage]
        self.work_dir = os.path.abspath(work_dir)
        self.name = name
        self.isaac_env = isaac_env
        self.device = device
        #: ``runner(request_path) -> None`` writes the response; the default starts the Isaac child. Tests inject one.
        self._runner = runner or self._run_child
        self.observation_space = SimpleNamespace(shape=(cases.OBS_DIM,), low=-np.inf, high=np.inf)
        self.action_space = SimpleNamespace(shape=(cases.NUM_ACTIONS,), low=-np.ones(cases.NUM_ACTIONS),
                                            high=np.ones(cases.NUM_ACTIONS))
        self.calls = 0
        self.episodes = 0

    # -- process -------------------------------------------------------------------------------------------------
    def _run_child(self, request_path: str) -> None:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        cmd = (f"source {self.isaac_env} >/dev/null 2>&1 && cd {shlex.quote(root)} && "
               f"python -u -m {CHILD_MODULE} {shlex.quote(request_path)}")
        log = request_path.replace(".request.json", ".log")
        with open(log, "w", encoding="utf-8") as f:
            proc = subprocess.run(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            tail = open(log, encoding="utf-8", errors="replace").read()[-3000:]
            raise RuntimeError(f"pen-spin child failed (exit {proc.returncode}); log {log}:\n{tail}")

    def _call(self, actor, manifest, record: bool, mode: str = "evaluate", extra: dict | None = None) -> dict:
        if cases.stage_of(manifest) != self.stage:
            raise ValueError(f"{manifest.manifest_id} is not a stage-{self.stage} manifest")
        os.makedirs(self.work_dir, exist_ok=True)
        self.calls += 1
        stem = os.path.join(self.work_dir, f"{self.name}_{self.calls:04d}")
        spec, state = actor_spec(actor) if actor is not None else ({"kind": "none"}, None)
        if state is not None:
            with open(stem + ".state.pt", "wb") as f:
                f.write(state)
            spec["state_path"] = stem + ".state.pt"
        manifest_path = stem + ".manifest.yaml"
        manifest.save(manifest_path)
        request = {"stage": self.stage, "mode": mode, "manifest_path": manifest_path, "manifest_id": manifest.manifest_id,
                   "actor": spec, "record": bool(record), "device": self.device,
                   "response_path": stem + ".response.json", **(extra or {})}
        request_path = stem + ".request.json"
        with open(request_path, "w", encoding="utf-8") as f:
            json.dump(request, f, indent=1)
        self._runner(request_path)
        with open(request["response_path"], encoding="utf-8") as f:
            response = json.load(f)
        if mode in ("reset_obs", "query"):
            return response
        if [r["case_id"] for r in response["episodes"]] != [c.case_id for c in manifest.episodes]:
            raise ValueError(f"{request['response_path']}: rows do not follow the manifest's cases")
        self.episodes += len(response["episodes"])
        return response

    # -- pipeline interface --------------------------------------------------------------------------------------
    def evaluate_manifest_batch(self, actor, manifest, max_steps=None, schema=None):
        """The whole manifest in one fresh Isaac process; returns a ManifestResult (sustained success)."""
        from lares.eval.runner import ManifestResult

        response = self._call(actor, manifest, record=False)
        by_case = {c.case_id: c for c in manifest.episodes}
        result = ManifestResult(actor_name=getattr(actor, "name", type(actor).__name__),
                                manifest_id=manifest.manifest_id, action_mode="deterministic",
                                schema_id=self.spec.task_id)
        for row in response["episodes"]:
            result.episodes.append(_row_to_record(row, by_case[row["case_id"]].task_id))
        result.process = {"request": response.get("request_path"), "response_sha256": response.get("sha256")}
        return result

    def collect_traces(self, actor, manifest) -> dict:
        """Rows plus per-step observation and action traces (cases, T, dim), for expert demonstrations.

        An actor with ``shadow_gait`` set also returns the frozen gait's action along the same trajectory (R-046).
        """
        response = self._call(actor, manifest, record=True)
        data = np.load(response["trace_path"])
        out = {"episodes": response["episodes"], "obs": data["obs"], "actions": data["actions"],
               "trace_path": response["trace_path"]}
        if "gait_actions" in data:
            out["gait_actions"] = data["gait_actions"]
        return out

    def reset_observations(self, manifest) -> np.ndarray:
        """Every case's observation right after its placement is written (no step, no episode), ``(cases, 157)``."""
        response = self._call(None, manifest, record=False, mode="reset_obs")
        obs = np.asarray(response["observations"], dtype=np.float32)
        if obs.shape != (len(manifest), self.observation_space.shape[0]):
            raise ValueError(f"reset observations have shape {obs.shape}")
        return obs

    def query_actions(self, actor, obs: np.ndarray, manifest) -> np.ndarray:
        """The actor's raw (unclamped) output at each row of ``obs`` ``(N, 157)``, in a fresh Isaac process with no
        episode (R-048). ``manifest`` only names the stage."""
        os.makedirs(self.work_dir, exist_ok=True)
        qpath = os.path.join(self.work_dir, f"{self.name}_{self.calls + 1:04d}.query.npz")
        np.savez_compressed(qpath, obs=np.asarray(obs, dtype=np.float32))
        response = self._call(actor, manifest, record=False, mode="query", extra={"query_path": qpath})
        acts = np.load(response["actions_path"])["actions"]
        if acts.shape != (len(obs), cases.NUM_ACTIONS):
            raise ValueError(f"queried actions have shape {acts.shape}")
        return acts

    def reset(self, case=None):
        raise RuntimeError("a pen-spin stream scores whole manifests; it has no single-episode reset")

    def close(self):
        pass


def make_temp_stream(stage: int, name: str, runner=None) -> IsaacPenSpinStream:
    """A stream writing under a fresh temporary directory (tests)."""
    return IsaacPenSpinStream(stage, tempfile.mkdtemp(prefix=f"penspin_{name}_"), name, runner=runner)


def rows_from_trace(trace: dict) -> list[dict]:
    """Pipeline rows from a runner trace: the reducer rows plus the fields the stream maps onto an EpisodeRecord."""
    rows = cases.reduce_episodes(trace)
    for i, r in enumerate(rows):
        L = r["length"]
        r["initial_center_dist"] = float(trace["center_dist"][0, i]) if L else None
        if "actions" in trace:
            a = np.asarray(trace["actions"][i, :L])
            r["action_saturation_rate"] = float(np.mean(np.any(np.abs(a) > 0.99, axis=-1))) if L else None
        r["failure_label"] = failure_label(r)
    return rows


__all__ = ["EXECUTION", "GaitActor", "IsaacExpertActor", "IsaacPenSpinStream", "actor_spec", "failure_label", "make_temp_stream",
           "rows_from_trace"]
