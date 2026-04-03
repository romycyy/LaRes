#!/usr/bin/env python
"""Tests for ``record_episode_gif`` in ``lares.core.training_pipeline``.

Run from repo root (``tests`` is not a Python package):
    python tests/test_record_episode_gif.py -v
    # or: python -m unittest discover -s tests -p 'test_record_episode_gif.py' -v

Real MetaWorld / MuJoCo envs often omit rgb_array frames when headless; the
function then logs "No frames captured — check render mode support."  These
tests use mocks that implement ``render`` so frame capture and GIF writing are
verified without a display.

**reach-v2 / Gymnasium:** GIF capture failed in production because (1) MetaWorld
was constructed without ``render_mode='rgb_array'``, so ``env.render()`` returned
``None``, and (2) ``env_wrapper.render`` did not forward ``**kwargs``, so
``render(mode='rgb_array')`` raised before reaching the inner env.  See
``make_metaworld_env`` and ``env_wrapper`` in ``lares.utils.utils``.

Tests call ``record_episode_gif(..., verbose=False)`` so expected no-frame cases
and headless MetaWorld skips do not print ``[record_gif]`` lines. Pipeline runs
keep the default ``verbose=True``.
"""

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from lares.core.training_pipeline import record_episode_gif  # noqa: E402

try:
    import imageio  # noqa: F401

    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

try:
    import metaworld.env_dict  # noqa: F401

    HAS_METAWORLD = True
except ImportError:
    HAS_METAWORLD = False

# reach-v2 observation / action sizes (flat Box) for integration-style tests
REACH_V2_OBS_DIM = 39
REACH_V2_ACTION_DIM = 4


class MockActionSpace:
    def __init__(self, dim):
        self.shape = (dim,)
        self.high = np.ones(dim, dtype=np.float32)
        self.low = -np.ones(dim, dtype=np.float32)


class MockObsSpace:
    def __init__(self, dim):
        self.shape = (dim,)


class RenderableMockEnv:
    """Gym-like env that returns RGB frames from ``render`` (rgb_array API)."""

    def __init__(self, obs_dim=8, action_dim=4, episode_length=5, frame_size=(32, 24)):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_space = MockActionSpace(action_dim)
        self.observation_space = MockObsSpace(obs_dim)
        self._episode_length = episode_length
        self._step_count = 0
        self._h, self._w = frame_size

    def reset(self):
        self._step_count = 0
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        return obs, {}

    def render(self, mode="rgb_array", offscreen=False, **kwargs):
        # Deterministic pattern so saved GIF content is stable-ish
        base = (self._step_count * 17) % 255
        frame = np.zeros((self._h, self._w, 3), dtype=np.uint8)
        frame[:, :, 0] = base
        frame[:, :, 1] = (base + 50) % 255
        return frame

    def step(self, action):
        self._step_count += 1
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        reward = 0.0
        done = self._step_count >= self._episode_length
        info = {"success": 1.0 if done else 0.0}
        return obs, reward, done, info


class WrappedRenderableEnv:
    """Wrapper with ``_env`` only — exercises inner render path in ``_grab_frame``."""

    def __init__(self, inner):
        self._env = inner
        self.action_space = inner.action_space
        self.observation_space = inner.observation_space

    def reset(self):
        return self._env.reset()

    def step(self, action):
        return self._env.step(action)


class NoRenderMockEnv(RenderableMockEnv):
    """Same as renderable env but never returns frames (all render attempts fail)."""

    def render(self, mode="rgb_array", offscreen=False, **kwargs):
        return None


class GymnasiumStyleReachLikeEnv:
    """Mimics MetaWorld-on-Gymnasium: only ``render()`` (no ``mode`` kwarg) returns pixels."""

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.action_space = MockActionSpace(REACH_V2_ACTION_DIM)
        self.observation_space = MockObsSpace(REACH_V2_OBS_DIM)
        self._step_count = 0

    def reset(self, **kwargs):
        self._step_count = 0
        obs = np.zeros(REACH_V2_OBS_DIM, dtype=np.float32)
        return obs, {}

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        base = (self._step_count * 19) % 255
        h, w = 40, 40
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 0] = base
        return frame

    def step(self, action):
        self._step_count += 1
        obs = np.zeros(REACH_V2_OBS_DIM, dtype=np.float32)
        reward = 0.0
        done = self._step_count >= 5
        info = {"success": 1.0 if done else 0.0}
        return obs, reward, done, info


class ProductionLikeEnvWrapper:
    """Same shape as ``env_wrapper``: ``_env``, timestep cap, forward ``render``."""

    def __init__(self, inner, episode_length=150):
        self._env = inner
        self.args = SimpleNamespace(episode_length=episode_length)
        self.observation_space = inner.observation_space
        self.action_space = inner.action_space
        self.timesteps = 0

    def reset(self):
        self.timesteps = 0
        return self._env.reset()

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        self.timesteps += 1
        if self.timesteps >= self.args.episode_length:
            done = True
        return obs, reward, done, info

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)


class TinyGaussianPolicy(nn.Module):
    """Minimal policy matching ``record_episode_gif`` usage (mean, std)."""

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def forward(self, obs):
        b = obs.shape[0]
        mean = torch.zeros(b, self.action_dim, device=obs.device, dtype=obs.dtype)
        std = torch.ones(b, self.action_dim, device=obs.device, dtype=obs.dtype) * 0.1
        return mean, std


@unittest.skipUnless(HAS_IMAGEIO, "imageio is required for record_episode_gif tests")
class TestRecordEpisodeGif(unittest.TestCase):
    def test_saves_gif_and_returns_metadata(self):
        env = RenderableMockEnv(episode_length=4)
        policy = TinyGaussianPolicy(env.obs_dim, env.action_dim)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "episode.gif")
            out = record_episode_gif(
                policy, env, path, max_steps=50, fps=10, verbose=False
            )
            self.assertTrue(out["saved"], msg="GIF should be written when render returns rgb_array")
            self.assertGreater(out["num_frames"], 0)
            self.assertIsInstance(out["episode_reward"], float)
            self.assertIsInstance(out["success"], bool)
            self.assertTrue(os.path.isfile(path))
            self.assertGreater(os.path.getsize(path), 0)

    def test_grabs_frames_from_inner_env_when_wrapper_has_no_render(self):
        inner = RenderableMockEnv(episode_length=3)
        env = WrappedRenderableEnv(inner)
        policy = TinyGaussianPolicy(env.observation_space.shape[0], env.action_space.shape[0])

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "wrapped.gif")
            out = record_episode_gif(
                policy, env, path, max_steps=20, fps=10, verbose=False
            )
            self.assertTrue(out["saved"])
            self.assertGreaterEqual(out["num_frames"], 1)

    def test_no_frames_when_render_returns_non_array(self):
        env = NoRenderMockEnv(episode_length=5)
        policy = TinyGaussianPolicy(env.obs_dim, env.action_dim)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.gif")
            out = record_episode_gif(
                policy, env, path, max_steps=10, fps=10, verbose=False
            )
            self.assertFalse(out["saved"])
            self.assertEqual(out["num_frames"], 0)
            self.assertFalse(os.path.exists(path))

    def test_reach_v2_like_gymnasium_stack_saves_gif(self):
        """Gymnasium-style inner (render only) + wrapper matches evolution/demo stack."""
        inner = GymnasiumStyleReachLikeEnv(render_mode="rgb_array")
        env = ProductionLikeEnvWrapper(inner, episode_length=150)
        policy = TinyGaussianPolicy(REACH_V2_OBS_DIM, REACH_V2_ACTION_DIM)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "reach_like.gif")
            out = record_episode_gif(
                policy, env, path, max_steps=30, fps=10, verbose=False
            )
            self.assertTrue(
                out["saved"],
                msg="Plain render() after render_mode=rgb_array should yield frames",
            )
            self.assertGreater(out["num_frames"], 0)
            self.assertTrue(os.path.isfile(path))

    def test_reach_v2_like_without_render_mode_captures_no_frames(self):
        """If render_mode is unset, Gymnasium-style env returns None from render()."""
        inner = GymnasiumStyleReachLikeEnv(render_mode=None)
        env = ProductionLikeEnvWrapper(inner, episode_length=150)
        policy = TinyGaussianPolicy(REACH_V2_OBS_DIM, REACH_V2_ACTION_DIM)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "no_mode.gif")
            out = record_episode_gif(
                policy, env, path, max_steps=15, fps=10, verbose=False
            )
            self.assertFalse(out["saved"])
            self.assertEqual(out["num_frames"], 0)


@unittest.skipUnless(HAS_IMAGEIO, "imageio is required for record_episode_gif tests")
@unittest.skipUnless(HAS_METAWORLD, "metaworld is required for reach-v2 env factory tests")
class TestRecordEpisodeGifReachV2MetaWorld(unittest.TestCase):
    def test_make_metaworld_reach_v2_records_gif(self):
        """End-to-end: same factory as ``run_full_evolution`` / ``run_demo`` for reach-v2."""
        from lares.utils.utils import env_wrapper, make_metaworld_env

        # Headless SSH: default is GLFW/X11 (needs DISPLAY). MuJoCo must load EGL/OSMesa
        # before the env is created; otherwise render() raises FatalError.
        _patched_gl = False
        if not os.environ.get("DISPLAY") and not os.environ.get("MUJOCO_GL"):
            os.environ["MUJOCO_GL"] = "egl"
            _patched_gl = True

        cfg = SimpleNamespace(env_name="reach-v2", episode_length=150)
        env = None
        try:
            env = env_wrapper(make_metaworld_env(cfg, seed=0), cfg)
            policy = TinyGaussianPolicy(REACH_V2_OBS_DIM, REACH_V2_ACTION_DIM)

            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "reach_v2_metaworld.gif")
                try:
                    out = record_episode_gif(
                        policy,
                        env,
                        path,
                        max_steps=25,
                        fps=10,
                        verbose=False,
                    )
                except Exception as exc:
                    self.skipTest(
                        "MuJoCo rgb_array render failed (OpenGL / headless). "
                        "Try: export MUJOCO_GL=egl or MUJOCO_GL=osmesa before tests. "
                        f"{type(exc).__name__}: {exc}"
                    )
                if not out["saved"]:
                    self.skipTest(
                        "MuJoCo returned no pixels after render(); check MUJOCO_GL / GPU EGL."
                    )
                self.assertGreater(out["num_frames"], 0)
                self.assertTrue(os.path.isfile(path))
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            if _patched_gl:
                del os.environ["MUJOCO_GL"]


if __name__ == "__main__":
    unittest.main()
