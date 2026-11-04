#!/usr/bin/env python
"""Run a LaRes-style three-stage smoke pipeline on Isaac Lab ShadowHandSpin."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from lares.core.symbolic_policy import SymbolicPolicy  # noqa: E402
from lares.core.training_logger import (  # noqa: E402
    DATASET_EPISODE_RETURN,
    DATASET_SPIN_RATE,
    DATASET_SUCCESS,
    TrainingLogger,
)
from lares.core.training_pipeline import DemoBuffer, behavioral_cloning, rl_finetune  # noqa: E402
from lares.envs.isaac_lab_adapter import (  # noqa: E402
    IsaacLabSingleEnvAdapter,
    _ensure_isaaclab_on_path,
    close_isaac_app,
    launch_isaac_app,
)
from lares.eval.manifest import synthetic_cases  # noqa: E402
from lares.utils.cli import load_yaml  # noqa: E402
from lares.utils.clock import run_stamp  # noqa: E402


class ShadowHandSpinSymbolicPolicy(SymbolicPolicy):
    """Tiny trainable symbolic policy for smoke-testing the LaRes stages."""

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__(obs_dim, action_dim)
        self.obs_gain = torch.nn.Parameter(torch.zeros(action_dim))
        self.obs_bias = torch.nn.Parameter(torch.zeros(action_dim))
        self.log_std = torch.nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, obs):
        obs_slice = obs[:, : self.action_dim]
        mean = self.obs_gain.unsqueeze(0) * obs_slice + self.obs_bias.unsqueeze(0)
        std = torch.nn.functional.softplus(self.log_std).unsqueeze(0).expand_as(mean) + 0.05
        return mean, std

    def get_param_ranges(self):
        return {
            "obs_gain": (-5.0, 5.0),
            "obs_bias": (-3.0, 3.0),
            "log_std": (-5.0, 2.0),
        }


# Optional keys and their fallbacks, applied once so each default lives in exactly one place.
_CONFIG_DEFAULTS = {
    "enable_cameras": False,
    "gif_deterministic": True,
    "gif_dir": "gifs",
    "gif_fps": 20,
    "record_stage_gifs": False,
}


def load_config(path: str) -> SimpleNamespace:
    cfg = {**_CONFIG_DEFAULTS, **(load_yaml(path) or {})}
    # Derived from other keys, so these cannot live in the static defaults above.
    cfg.setdefault("gif_max_steps", cfg["max_steps"])
    if "pipeline_task" not in cfg:
        raise ValueError(f"{path}: 'pipeline_task' names the one Isaac env every stage runs against")
    return SimpleNamespace(**cfg)


def write_summary(run_dir: str, summary: dict) -> str:
    summary_path = os.path.join(run_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary_path


def make_env(task_name: str, cfg: SimpleNamespace) -> IsaacLabSingleEnvAdapter:
    return IsaacLabSingleEnvAdapter(
        task_name=task_name,
        num_envs=int(cfg.num_envs),
        device=str(cfg.device),
        isaaclab_root=str(cfg.isaaclab_root),
        max_episode_steps=int(cfg.max_steps),
        enable_cameras=bool(cfg.record_stage_gifs or cfg.enable_cameras),
        render_mode="rgb_array" if cfg.record_stage_gifs else None,
    )


def scripted_spin_action(step: int, action_dim: int, noise: float) -> np.ndarray:
    action = np.zeros(action_dim, dtype=np.float32)
    phase = 0.35 * step
    # Wrist oscillation plus light finger closure: deliberately simple, enough
    # to produce nontrivial demonstrations for the smoke BC stage.
    if action_dim >= 2:
        action[0] = 0.55 * np.sin(phase)
        action[1] = 0.35 * np.cos(phase)
    if action_dim > 2:
        action[2:] = 0.15
    if noise > 0:
        action += np.random.normal(0.0, noise, size=action_dim).astype(np.float32)
    return np.clip(action, -1.0, 1.0)


def _policy_action(policy: ShadowHandSpinSymbolicPolicy, obs: np.ndarray, deterministic: bool) -> np.ndarray:
    device = next(policy.parameters()).device
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        mean, std = policy(obs_t)
        if deterministic:
            action_t = mean
        else:
            action_t = torch.distributions.Normal(mean, std).sample()
    # ``mean``/``std`` parameterise the *pre-tanh* Gaussian (see SymbolicPolicy and
    # behavioral_cloning), so squash before acting — otherwise the GIF rollout follows a
    # different policy than rl_finetune/evaluate_policy, which both use tanh(...).
    action_t = torch.tanh(action_t)
    action = action_t.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    return np.clip(action, -1.0, 1.0)


def _draw_bar(frame: np.ndarray, x: int, y: int, width: int, value: float, color: tuple[int, int, int]) -> None:
    value = float(np.clip(value, 0.0, 1.0))
    frame[y : y + 10, x : x + width] = 55
    frame[y : y + 10, x : x + int(width * value)] = color


def _state_visualization_frame(step: int, max_steps: int, action: np.ndarray, info: dict) -> np.ndarray:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:, :, 0] = 18
    frame[:, :, 1] = 22
    frame[:, :, 2] = 30

    progress = (step + 1) / max(max_steps, 1)
    spin = abs(float(info.get("spin_rate", 0.0)))
    center = float(info.get("center_dist", 0.0))
    axis = float(info.get("axis_error", 0.0))
    action_mag = float(np.linalg.norm(action) / max(np.sqrt(action.size), 1.0))

    _draw_bar(frame, 24, 32, 272, progress, (80, 160, 255))
    _draw_bar(frame, 24, 76, 272, min(spin / 20.0, 1.0), (255, 180, 80))
    _draw_bar(frame, 24, 120, 272, min(center / 0.3, 1.0), (255, 90, 90))
    _draw_bar(frame, 24, 164, 272, min(axis, 1.0), (140, 220, 120))
    _draw_bar(frame, 24, 208, 272, min(action_mag, 1.0), (190, 120, 255))

    # A compact indicator that moves with spin/action so fallback GIFs still show behavior over time.
    dot_x = int(24 + 272 * progress)
    dot_y = int(120 + 42 * np.sin(0.2 * step + spin))
    frame[max(0, dot_y - 4) : min(frame.shape[0], dot_y + 5), max(0, dot_x - 4) : min(frame.shape[1], dot_x + 5)] = (
        255,
        255,
        255,
    )
    return frame


def record_shadowhand_gif(
    env,
    path: str,
    action_fn,
    max_steps: int,
    fps: int,
) -> dict:
    import imageio

    frames = []
    use_render = False
    episode_return = 0.0
    success = False
    tracked: dict[str, list[float]] = {"spin_rate": [], "center_dist": [], "axis_error": []}

    obs, _ = env.reset()
    # The RTX renderer emits blank frames for the first few calls; prime it before
    # step 0 decides whether this recording uses real frames or the fallback.
    warmup = getattr(env, "warmup_renderer", None)
    if callable(warmup):
        attempts = warmup()
        print(f"  [stage_gif] renderer warmup: {'ready after %s attempts' % attempts if attempts else 'never became ready'}")

    for step in range(max_steps):
        action = action_fn(obs, step)
        obs, reward, done, info = env.step(action)

        try:
            # The adapter's render() already returns a coerced (H, W, 3) uint8 frame or None.
            frame = env.render()
        except Exception as exc:
            if step == 0:
                print(f"  [stage_gif] Dropping initial render frame: {exc!r}")
            frame = None
        if step == 0:
            # Every frame of a GIF must share one shape, and the fallback visualisation is
            # not the render resolution, so pick one source on the first step instead of
            # interleaving the two and having imageio reject the whole animation.
            use_render = frame is not None
        if use_render:
            # Hold the last rendered frame if a later render comes back empty.
            frames.append(frame if frame is not None else frames[-1])
        else:
            frames.append(_state_visualization_frame(step, max_steps, action, info))

        episode_return += float(reward)
        success = success or float(info.get("success", 0.0)) > 0.0
        for key, values in tracked.items():
            if key in info:
                values.append(float(info[key]))
        if done:
            break

    saved = False
    if frames:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        imageio.mimsave(path, frames, duration=1000.0 / float(fps))
        saved = True

    return {
        "path": path,
        "saved": saved,
        # Whether the GIF shows the simulator or the synthetic status bars. Without this
        # a failed camera still reports saved=true and looks like a successful recording.
        "frame_source": ("render" if use_render else "fallback") if frames else "none",
        "num_frames": len(frames),
        "episode_return": episode_return,
        "success": success,
        **{f"mean_{key}": float(np.mean(v)) if v else 0.0 for key, v in tracked.items()},
    }


def record_stage_gif(
    stage_name: str,
    env,
    cfg: SimpleNamespace,
    run_dir: str,
    policy: ShadowHandSpinSymbolicPolicy | None = None,
) -> dict:
    path = os.path.join(run_dir, str(cfg.gif_dir), f"{stage_name}.gif")
    max_steps = int(cfg.gif_max_steps)
    fps = int(cfg.gif_fps)
    deterministic = bool(cfg.gif_deterministic)

    if policy is None:
        action_fn = lambda _obs, step: scripted_spin_action(  # noqa: E731
            step,
            env.action_space.shape[0],
            0.0,
        )
    else:
        policy.eval()
        action_fn = lambda obs, _step: _policy_action(policy, obs, deterministic)  # noqa: E731

    try:
        result = record_shadowhand_gif(env, path, action_fn, max_steps=max_steps, fps=fps)
    except Exception as exc:
        result = {
            "path": path,
            "saved": False,
            "num_frames": 0,
            "error": repr(exc),
        }
        print(f"  [stage_gif] Failed to record {stage_name}: {exc!r}")
    return result


def evaluate_stage_policy(policy, env, cases, max_steps: int) -> dict:
    """Deterministic evaluation for the Isaac stack.

    Local rather than shared with ``training_pipeline.evaluate_policy``: that
    one replays a MetaWorld ``EvaluationManifest``, and Isaac Lab exposes no
    per-episode placements to build one from.
    """
    total_reward, total_success = 0.0, 0.0
    policy.eval()
    for case in cases:
        obs, _ = env.reset(case)
        episode_success = 0.0
        for _ in range(int(max_steps)):
            action = _policy_action(policy, obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward)
            episode_success = max(episode_success, float(info.get("success", 0.0)))
            if done:
                break
        total_success += episode_success
    n = max(1, len(cases))
    return {"mean_reward": total_reward / n, "success_rate": total_success / n}


def collect_scripted_dataset(env, cfg: SimpleNamespace, logger: TrainingLogger) -> tuple[DemoBuffer, dict]:
    buffer = DemoBuffer()
    episode_returns, successes, spin_rates = [], [], []
    cases = synthetic_cases(int(cfg.dataset_episodes), prefix="shadowhand-train")

    for ep, case in enumerate(cases):
        obs, _ = env.reset(case)
        episode_return = 0.0
        episode_success = 0.0
        episode_spin = []
        for step in range(int(cfg.max_steps)):
            action = scripted_spin_action(step, env.action_space.shape[0], float(cfg.scripted_noise))
            next_obs, reward, done, info = env.step(action)
            buffer.add(obs, action, reward, next_obs, done, episode_id=case.case_id)
            episode_return += reward
            episode_success = max(episode_success, float(info.get("success", 0.0)))
            if "spin_rate" in info:
                episode_spin.append(float(info["spin_rate"]))
            obs = next_obs
            if done:
                break
        episode_returns.append(episode_return)
        successes.append(episode_success)
        spin_rates.append(float(np.mean(episode_spin)) if episode_spin else 0.0)
        logger.log_metrics(
            stage="dataset",
            update=ep,
            metrics={
                DATASET_EPISODE_RETURN: episode_return,
                DATASET_SUCCESS: episode_success,
                DATASET_SPIN_RATE: spin_rates[-1],
            },
            task_name=cfg.task_name,
        )

    stats = {
        "num_transitions": len(buffer),
        "num_episodes": int(cfg.dataset_episodes),
        "mean_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "mean_success": float(np.mean(successes)) if successes else 0.0,
        "mean_spin_rate": float(np.mean(spin_rates)) if spin_rates else 0.0,
    }
    return buffer, stats


def pen_spin_studies() -> dict:
    """``{study name: module}`` for ``--study``. Every module exposes ``build_manifests(protocol_cfg)``,
    ``make_controller(name, protocol_cfg, out_dir)`` and ``run(protocol_cfg, config_path, out_dir)``; a new study is
    one module with those three names and one entry here."""
    from lares.envs.isaac_pen_spin import (archive_rescreen, axial_sign, feedback_ablation, feedback_delay,
                                           feedback_lesions, learner_recovery, scalar_tuning, stroke_lesion)

    return {feedback_ablation.FEEDBACK_ABLATION: feedback_ablation,
            feedback_lesions.FEEDBACK_LESIONS: feedback_lesions,
            feedback_delay.FEEDBACK_DELAY: feedback_delay,
            learner_recovery.LEARNER_RECOVERY: learner_recovery,
            axial_sign.LEARNER_AXIAL_SIGN: axial_sign,
            scalar_tuning.STUDY: scalar_tuning,
            archive_rescreen.STUDY: archive_rescreen,
            stroke_lesion.STUDY: stroke_lesion}


def run_study(args, cfg: SimpleNamespace) -> None:
    """``--study feedback_ablation`` (R-041), ``feedback_lesions`` (R-044), ``feedback_delay`` (R-047),
    ``learner_recovery`` (R-051), ``learner_axial_sign`` (R-053), ``r058_scalar_cma_vs_random`` (R-058),
    ``r060_archive_rescreen`` (R-060) or ``r062_stroke_lesion`` (R-062): the driver, or one child cell."""
    studies = pen_spin_studies()
    if args.study not in studies:
        raise SystemExit(f"unknown study {args.study!r}; one of {sorted(studies)}")
    fa = studies[args.study]
    protocol_cfg = (getattr(cfg, "research_protocols", None) or {}).get(args.study)
    if protocol_cfg is None:
        raise SystemExit(f"{args.config} declares no research protocol {args.study!r}")
    out_dir = args.log_dir or str(protocol_cfg["log_dir"])
    if args.manifest_seed is not None and int(args.manifest_seed) != int(protocol_cfg["pool_seed"]):
        raise SystemExit(f"--manifest-seed {args.manifest_seed} != the protocol's pool_seed {protocol_cfg['pool_seed']}")
    if args.build_manifests:
        print("wrote", fa.build_manifests(protocol_cfg) or "nothing (all present and identical)")
        return
    if args.cell_manifest:
        from lares.envs.isaac_pen_spin import protocols

        _ensure_isaaclab_on_path(str(cfg.isaaclab_root))
        launch_isaac_app(headless=True, device=str(cfg.device))
        try:
            ctrl = fa.make_controller(args.controller, protocol_cfg, out_dir)
            print("wrote", protocols.run_cell(args.cell_manifest, ctrl, args.cell_out, str(cfg.device),
                                              record_obs=args.record_obs))
        finally:
            close_isaac_app()
        return
    summary = fa.run(protocol_cfg, args.config, out_dir)
    print(json.dumps({k: v for k, v in summary.items() if k != "tape"}, indent=2))


def run_research_protocol(args, cfg: SimpleNamespace) -> None:
    from lares.envs.isaac_pen_spin import protocols

    protocol_cfg = (getattr(cfg, "research_protocols", None) or {}).get(args.research_protocol)
    if protocol_cfg is None:
        raise SystemExit(f"{args.config} declares no research protocol {args.research_protocol!r}")
    if args.research_protocol == protocols.SCRIPTED_EXPERT_QUALIFICATION:
        if args.build_manifests:
            print("wrote", protocols.build_qualification_manifests(protocol_cfg) or "nothing (all present and identical)")
            return
        if args.cell_manifest:
            _ensure_isaaclab_on_path(str(cfg.isaaclab_root))
            launch_isaac_app(headless=True, device=str(cfg.device))
            try:
                print("wrote", protocols.run_cell(args.cell_manifest, args.controller, args.cell_out,
                                                  str(cfg.device), record_obs=args.record_obs))
            finally:
                close_isaac_app()
            return
        out_dir = args.log_dir or str(protocol_cfg["log_dir"])
        summary = protocols.run_scripted_expert_qualification(protocol_cfg, args.config, out_dir)
        print(json.dumps({k: v for k, v in summary.items() if k != "per_stage"}, indent=2))
        return
    if args.research_protocol != protocols.EPISODE_RULE_AUDIT:
        raise SystemExit(f"unknown research protocol {args.research_protocol!r}")
    if args.build_manifests:
        print("wrote", protocols.build_audit_manifests(protocol_cfg) or "nothing (all present and identical)")
        return
    out_dir = args.log_dir or str(protocol_cfg["log_dir"])
    if args.stage is not None:
        _ensure_isaaclab_on_path(str(cfg.isaaclab_root))
        launch_isaac_app(headless=True, device=str(cfg.device))
        try:
            os.makedirs(out_dir, exist_ok=True)
            print("wrote", protocols.run_audit_cell(args.stage, args.controller, protocol_cfg, out_dir, str(cfg.device)))
        finally:
            close_isaac_app()
        return
    summary = protocols.run_episode_rule_audit(protocol_cfg, args.config, out_dir, str(cfg.device))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_stage_controller"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="ShadowHandSpin three-stage smoke pipeline")
    parser.add_argument("--config", default=os.path.join(_PROJECT_ROOT, "config", "shadowhand_spin_stages.yaml"))
    parser.add_argument("--research-protocol", default=None,
                        help="run a named protocol from the config's research_protocols block instead of the smoke stages")
    parser.add_argument("--study", default=None,
                        help="run a named study (feedback_ablation, feedback_lesions, feedback_delay, learner_recovery, learner_axial_sign, r058_scalar_cma_vs_random, r060_archive_rescreen, r062_stroke_lesion) from the config's research_protocols block")
    parser.add_argument("--build-manifests", action="store_true",
                        help="with --research-protocol: write the protocol's frozen manifests and exit (no Isaac)")
    parser.add_argument("--stage", type=int, default=None, help=argparse.SUPPRESS)  # protocol child process
    parser.add_argument("--controller", default=None, help=argparse.SUPPRESS)  # protocol child process
    parser.add_argument("--cell-manifest", default=None, help=argparse.SUPPRESS)  # protocol child process
    parser.add_argument("--cell-out", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--record-obs", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--log-dir", "--output-dir", dest="log_dir", default=None,
                        help="protocol output directory (default: the protocol's log_dir)")
    parser.add_argument("--manifest-seed", type=int, default=None,
                        help="with --study: the pool seed the command expects; refused if the config declares another")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.study:
        run_study(args, cfg)
        return
    if args.research_protocol:
        run_research_protocol(args, cfg)
        return
    run_id = run_stamp()
    run_dir = os.path.abspath(os.path.join(str(cfg.log_dir), run_id))
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy2(args.config, os.path.join(run_dir, "config.yaml"))

    logger = TrainingLogger(log_dir=run_dir, run_id="training_dynamics", task_name=cfg.task_name)
    summary = {"run_dir": run_dir, "config": vars(cfg)}
    record_gifs = bool(cfg.record_stage_gifs)
    if record_gifs:
        summary["gifs"] = {}

    def maybe_record_gif(stage_key: str, gif_name: str, policy=None) -> None:
        """Record a stage GIF into the summary, or do nothing when GIFs are off."""
        if not record_gifs:
            return
        print(f"\nRecording {stage_key} GIF")
        result = record_stage_gif(gif_name, env, cfg, run_dir, policy=policy)
        summary["gifs"][stage_key] = result
        if result.get("frame_source") == "fallback":
            print(
                f"  [stage_gif] WARNING: {gif_name}.gif contains synthetic status bars, "
                "not simulator frames — the renderer produced no image."
            )

    env = None
    try:
        env = make_env(str(cfg.pipeline_task), cfg)

        print("\nStage 1: scripted dataset collection")
        demo_buffer, dataset_stats = collect_scripted_dataset(env, cfg, logger)
        demo_path = os.path.join(run_dir, "demo_shadowhand_spin.pkl")
        demo_buffer.save(demo_path)
        summary["stage1"] = dataset_stats | {"demo_path": demo_path}
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        maybe_record_gif("stage1", "stage1_scripted")

        print("\nStage 2: behavioral cloning")
        policy = ShadowHandSpinSymbolicPolicy(obs_dim, action_dim)
        policy.validate()
        bc_stats = behavioral_cloning(
            policy,
            demo_buffer,
            num_steps=int(cfg.bc_steps),
            batch_size=int(cfg.bc_batch_size),
            lr=float(cfg.bc_lr),
            log_interval=max(1, int(cfg.bc_steps) // 2),
            logger=logger,
            task_name=cfg.task_name,
            log_every_n_steps=1,
        )
        summary["stage2"] = bc_stats
        maybe_record_gif("stage2", "stage2_bc", policy=policy)

        print("\nStage 3: RL fine-tuning")
        rl_cases = synthetic_cases(
            int(cfg.rl_episodes_per_iter) * max(1, int(cfg.rl_iterations)),
            prefix="shadowhand-rl",
        )
        rl_stats = rl_finetune(
            policy,
            env,
            rl_cases,
            num_iterations=int(cfg.rl_iterations),
            episodes_per_iter=int(cfg.rl_episodes_per_iter),
            lr=float(cfg.rl_lr),
            max_steps=int(cfg.max_steps),
            log_interval=1,
            logger=logger,
            task_name=cfg.task_name,
        )
        eval_cases = synthetic_cases(int(cfg.eval_episodes), prefix="shadowhand-eval")
        eval_stats = evaluate_stage_policy(policy, env, eval_cases, int(cfg.max_steps))
        summary["stage3"] = rl_stats | {"eval": eval_stats}
        maybe_record_gif("stage3", "stage3_rl", policy=policy)

        model_path = os.path.join(run_dir, "shadowhand_spin_policy.pt")
        torch.save(policy.state_dict(), model_path)
        summary["policy_path"] = model_path

        figures_dir = os.path.join(run_dir, "figures")
        plot_cmd = [
            sys.executable,
            os.path.join(_PROJECT_ROOT, "scripts", "plot_training_dynamics.py"),
            "--log-path",
            logger.log_path,
            "--output-dir",
            figures_dir,
        ]
        print("\nPlotting training dynamics")
        plot_result = subprocess.run(plot_cmd, cwd=_PROJECT_ROOT, text=True, capture_output=True)
        summary["plot"] = {
            "command": " ".join(plot_cmd),
            "returncode": plot_result.returncode,
            "stdout": plot_result.stdout[-4000:],
            "stderr": plot_result.stderr[-4000:],
            "figures_dir": figures_dir,
        }
        if plot_result.returncode != 0:
            print(plot_result.stdout)
            print(plot_result.stderr)
            raise RuntimeError("plot_training_dynamics.py failed")

        summary_path = write_summary(run_dir, summary)
        print("\nShadowHandSpin smoke pipeline complete")
        print(f"  run_dir     : {run_dir}")
        print(f"  log_jsonl   : {logger.log_path}")
        print(f"  summary     : {summary_path}")
        print(f"  figures_dir : {summary['plot']['figures_dir']}")
        if record_gifs:
            print(f"  gifs_dir    : {os.path.join(run_dir, str(cfg.gif_dir))}")

    finally:
        if env is not None:
            env.close()
        close_isaac_app()


if __name__ == "__main__":
    main()
