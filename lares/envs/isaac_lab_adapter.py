"""Small adapter that exposes an Isaac Lab vector env as a single LaRes env."""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

_APP_LAUNCHER = None
_SIMULATION_APP = None
_LAUNCH_OPTS: dict | None = None

# Overridable so the package does not hardcode one machine's checkout location.
DEFAULT_ISAACLAB_ROOT = os.environ.get("ISAACLAB_ROOT", "/home/ubuntu/IsaacLab")


def _ensure_isaaclab_on_path(isaaclab_root: str) -> None:
    source_dir = os.path.join(isaaclab_root, "source")
    paths = [
        os.path.join(source_dir, "isaaclab"),
        os.path.join(source_dir, "isaaclab_tasks"),
        os.path.join(source_dir, "isaaclab_assets"),
        os.path.join(source_dir, "isaaclab_rl"),
    ]
    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)


def launch_isaac_app(headless: bool = True, device: str = "cuda:0", enable_cameras: bool = False) -> Any:
    """Launch Isaac Sim once for Isaac Lab tasks."""
    global _APP_LAUNCHER, _SIMULATION_APP, _LAUNCH_OPTS
    opts = {"headless": bool(headless), "enable_cameras": bool(enable_cameras), "device": str(device)}
    if _SIMULATION_APP is not None:
        # Isaac Sim is one-per-process, so a second launch cannot change these.
        def _conflicts(key, want):
            live = _LAUNCH_OPTS[key]
            if key == "enable_cameras":
                # A capability: an app launched with cameras still serves requests
                # that do not need them, but not the reverse.
                return want and not live
            return live != want

        conflicts = {k: (_LAUNCH_OPTS[k], v) for k, v in opts.items() if _conflicts(k, v)}
        if conflicts:
            detail = ", ".join(f"{k}: launched with {was!r}, requested {now!r}" for k, (was, now) in conflicts.items())
            raise RuntimeError(f"Isaac Sim is already running with different options ({detail}); restart the process.")
        return _SIMULATION_APP

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "yes")
    from isaaclab.app import AppLauncher

    # AppLauncher may add its own defaults to the dict it is handed, so snapshot the
    # requested options first: the conflict check above compares against them.
    requested = dict(opts)
    _APP_LAUNCHER = AppLauncher(opts)
    _SIMULATION_APP = _APP_LAUNCHER.app
    _LAUNCH_OPTS = requested
    return _SIMULATION_APP


def close_isaac_app() -> None:
    global _APP_LAUNCHER, _SIMULATION_APP, _LAUNCH_OPTS
    if _SIMULATION_APP is not None:
        try:
            _SIMULATION_APP.close()
        finally:
            # Drop the handles even if close() raised, otherwise a later launch_isaac_app()
            # hands back a dead app instead of starting a new one.
            _SIMULATION_APP = None
            _APP_LAUNCHER = None
            _LAUNCH_OPTS = None


class IsaacLabSingleEnvAdapter:
    """Expose the first Isaac Lab vector environment through a Gym-like API."""

    def __init__(
        self,
        task_name: str,
        num_envs: int = 1,
        device: str = "cuda:0",
        isaaclab_root: str = DEFAULT_ISAACLAB_ROOT,
        max_episode_steps: int | None = None,
        enable_cameras: bool = False,
        render_mode: str | None = None,
    ):
        _ensure_isaaclab_on_path(isaaclab_root)
        launch_isaac_app(headless=True, device=device, enable_cameras=enable_cameras)

        import gymnasium as gym
        import torch
        import isaaclab_tasks  # noqa: F401
        from gymnasium import spaces
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        self.torch = torch
        self.task_name = task_name
        self.num_envs = int(num_envs)
        self.max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
        self._render_product = None
        self._rgb_annotator = None

        env_cfg = parse_env_cfg(task_name, device=device, num_envs=self.num_envs)
        if render_mode == "rgb_array":
            env_cfg.viewer.cam_prim_path = "/World/ShadowHandGifCamera"
            env_cfg.viewer.eye = (1.0, -1.35, 1.0)
            env_cfg.viewer.lookat = (0.0, -0.39, 0.55)
            env_cfg.viewer.resolution = (640, 480)
        make_kwargs = {"cfg": env_cfg}
        if render_mode is not None:
            make_kwargs["render_mode"] = render_mode
        self.env = gym.make(task_name, **make_kwargs)
        # ``unwrapped`` walks the wrapper chain, so resolve the device once here
        # rather than on every step().
        self._device = self.env.unwrapped.device
        if render_mode == "rgb_array":
            self._ensure_render_camera()

        action_shape = getattr(self.env.action_space, "shape", None)
        if action_shape is None:
            action_dim = int(getattr(self.env.unwrapped, "num_actions"))
        else:
            action_dim = int(action_shape[-1])
        self.action_space = spaces.Box(-np.ones(action_dim, dtype=np.float32), np.ones(action_dim, dtype=np.float32))

        obs, _ = self.env.reset()
        obs_np = self._first_obs(obs)
        self.observation_space = spaces.Box(
            -np.inf * np.ones_like(obs_np, dtype=np.float32),
            np.inf * np.ones_like(obs_np, dtype=np.float32),
        )

    def reset(self):
        self._elapsed_steps = 0
        obs, info = self.env.reset()
        return self._first_obs(obs), self._info_for_first(info)

    def step(self, action):
        self._elapsed_steps += 1
        action_np = np.asarray(action, dtype=np.float32).reshape(1, -1)
        action_np = np.clip(action_np, -1.0, 1.0)
        action_batch = np.repeat(action_np, self.num_envs, axis=0)
        action_t = self.torch.tensor(action_batch, dtype=self.torch.float32, device=self._device)

        obs, reward, terminated, truncated, info = self.env.step(action_t)
        reward_f = self._first_scalar(reward)
        # Combine the two done flags on-device so they cost one sync, not two.
        if hasattr(terminated, "detach") and hasattr(truncated, "detach"):
            done = bool(
                (terminated.detach().reshape(-1)[0] | truncated.detach().reshape(-1)[0]).item()
            )
        else:
            done = bool(self._first_scalar(terminated) or self._first_scalar(truncated))
        if self.max_episode_steps is not None and self._elapsed_steps >= self.max_episode_steps:
            done = True
        info_first = self._info_for_first(info)
        return self._first_obs(obs), reward_f, done, info_first

    def close(self):
        self.env.close()

    def render(self):
        """Return an RGB frame when the wrapped Isaac Lab env was created with ``rgb_array`` rendering."""
        render_fn = getattr(self.env, "render", None)
        if callable(render_fn):
            try:
                frame = self._coerce_rgb_frame(render_fn())
                if frame is not None:
                    return frame
            except Exception:
                pass
        return self._render_with_replicator()

    def _first_obs(self, obs):
        if isinstance(obs, dict):
            obs = obs.get("policy", next(iter(obs.values())))
        if hasattr(obs, "detach"):
            # Slice env 0 on-device first: transferring the whole (num_envs, obs_dim)
            # batch and then discarding all but row 0 wastes num_envs-1 rows of traffic.
            if obs.ndim > 1:
                obs = obs[0]
            return obs.detach().cpu().numpy().astype(np.float32, copy=False)
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            return obs
        return obs[0].copy()

    def _first_scalar(self, value) -> float:
        if hasattr(value, "detach"):
            # One scalar sync instead of copying the full (num_envs,) vector back.
            return float(value.detach().reshape(-1)[0].item())
        return float(np.asarray(value).reshape(-1)[0])

    def _info_for_first(self, info):
        out = {}
        if not isinstance(info, dict):
            return out
        log = info.get("log", {})
        if isinstance(log, dict):
            # Isaac Lab puts every reward term in info["log"]. Stack the tensor entries
            # and move them across in a single sync rather than one per key.
            keys, scalars = [], []
            for key, value in log.items():
                try:
                    if hasattr(value, "detach"):
                        keys.append(key)
                        scalars.append(value.detach().reshape(-1)[0])
                    else:
                        out[key] = float(np.asarray(value).reshape(-1)[0])
                except Exception:
                    pass
            if scalars:
                try:
                    packed = self.torch.stack(scalars).cpu().numpy()
                    out.update(zip(keys, (float(v) for v in packed)))
                except Exception:
                    for key, value in zip(keys, scalars):
                        try:
                            out[key] = float(value.item())
                        except Exception:
                            pass
        if "success_rate" in out:
            out["success"] = float(out["success_rate"] > 0.0)
        return out

    def _render_with_replicator(self):
        try:
            import omni.replicator.core as rep

            if self._rgb_annotator is None:
                viewer = self.env.unwrapped.cfg.viewer
                self._render_product = rep.create.render_product(viewer.cam_prim_path, viewer.resolution)
                self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
                try:
                    self._rgb_annotator.attach(self._render_product)
                except Exception:
                    self._rgb_annotator.attach([self._render_product])

            sim = getattr(self.env.unwrapped, "sim", None)
            if sim is not None:
                sim.render()
            return self._coerce_rgb_frame(self._rgb_annotator.get_data())
        except Exception:
            return None

    def _ensure_render_camera(self):
        try:
            import isaacsim.core.utils.prims as prim_utils
            from pxr import UsdGeom

            viewer = self.env.unwrapped.cfg.viewer
            if not prim_utils.is_prim_path_valid(viewer.cam_prim_path):
                cam_prim = prim_utils.create_prim(viewer.cam_prim_path, prim_type="Camera")
                UsdGeom.Camera(cam_prim)
            self.env.unwrapped.sim.set_camera_view(
                eye=viewer.eye,
                target=viewer.lookat,
                camera_prim_path=viewer.cam_prim_path,
            )
        except Exception:
            pass

    @staticmethod
    def _coerce_rgb_frame(frame):
        if frame is None:
            return None
        arr = np.asarray(frame)
        if arr.size == 0:
            return None
        if arr.ndim != 3 or arr.shape[-1] < 3:
            return None
        arr = arr[..., :3]
        if arr.dtype == np.uint8:
            return arr
        if np.issubdtype(arr.dtype, np.floating):
            if arr.size and float(np.nanmax(arr)) <= 1.0:
                arr = arr * 255.0
            arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        return np.clip(arr, 0, 255).astype(np.uint8)
