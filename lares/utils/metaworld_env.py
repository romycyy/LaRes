"""MetaWorld environment factory and episode wrapper for the evolution pipeline."""

import numpy as np
import metaworld.env_dict as _env_dict
from gym.wrappers.time_limit import TimeLimit

from lares.envs.rlkit.envs.wrappers import NormalizedBoxEnv


class _GetDict:
    """Picklable callable that implements the V2 get_dict() API for a V3 metaworld env.

    Defined at module level (not as a closure) so multiprocessing spawn can
    pickle it when serialising worker objects.
    """

    def __init__(self, env_instance):
        self.env = env_instance

    def __call__(self):
        env = self.env
        tcp = env.get_endeff_pos().copy()
        target = getattr(env, "_target_pos", np.zeros(3)).copy()
        init_tcp = np.array(getattr(env, "init_tcp", tcp))
        obj_init_pos = np.array(getattr(env, "obj_init_pos", np.zeros(3)))
        hand_init_pos = np.array(getattr(env, "hand_init_pos", tcp))

        try:
            obj = env.data.body("obj").xpos.copy()
        except Exception:
            try:
                obs = env._get_obs()
                obj = obs[4:7].copy()
            except Exception:
                obj = np.zeros(3)

        return {
            "tcp": tcp,
            "obj": obj,
            "target": target,
            "init_tcp": init_tcp,
            "hand_pos": tcp,
            "gripper": tcp,
            "handle": obj,
            "_target_pos": target,
            "target_pos": target,
            "current_pos": obj,
            "obj_init_pos": obj_init_pos,
            "hand_init_pos": hand_init_pos,
            "actions": np.zeros(env.action_space.shape[0]),
        }


def _patch_get_dict(env_instance):
    """Attach a picklable get_dict() to a V3 metaworld env instance."""
    env_instance.get_dict = _GetDict(env_instance)


def _unwrap_through_wrappers(env):
    """Walk ``TimeLimit`` / ``NormalizedBoxEnv`` (``ProxyEnv``) to the inner env."""
    cur = env
    seen = set()
    for _ in range(32):
        if id(cur) in seen:
            break
        seen.add(id(cur))
        nxt = None
        if hasattr(cur, "env"):
            nxt = cur.env
        elif hasattr(cur, "_wrapped_env"):
            nxt = cur._wrapped_env
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return cur


def make_metaworld_env(cfg, seed):
    env_name = cfg.env_name
    env_name_v3 = env_name.replace("-v2", "-v3").replace("-v1", "-v3")
    if env_name_v3 not in _env_dict.ALL_V3_ENVIRONMENTS:
        raise ValueError(
            f"Environment '{env_name}' (looked up as '{env_name_v3}') not found in ALL_V3_ENVIRONMENTS"
        )

    use_mt1 = getattr(cfg, "use_mt1", False)
    if use_mt1:
        import metaworld as mw

        try:
            mt1 = mw.MT1(env_name_v3)
        except Exception as e:
            raise ValueError(
                f"use_mt1=True but MT1({env_name_v3!r}) failed. "
                f"Use a benchmark name MT1 supports (e.g. push-v3). Original error: {e}"
            ) from e
        if env_name_v3 not in mt1.train_classes:
            raise ValueError(
                f"use_mt1=True but {env_name_v3!r} not in MT1.train_classes "
                f"(keys sample: {list(mt1.train_classes.keys())[:5]} ...)"
            )
        env_cls = mt1.train_classes[env_name_v3]
        try:
            env = env_cls(render_mode="rgb_array", camera_id=2)
        except TypeError:
            env = env_cls()
        _patch_get_dict(env)
        train_tasks = tuple(mt1.train_tasks)
        if not train_tasks:
            raise ValueError(f"MT1({env_name_v3!r}) returned no train_tasks")
        env.mt1_train_tasks = train_tasks
        env.seed(seed)
        env.set_task(train_tasks[int(seed) % len(train_tasks)])
        return TimeLimit(NormalizedBoxEnv(env), env.max_path_length)

    env_cls = _env_dict.ALL_V3_ENVIRONMENTS[env_name_v3]

    try:
        env = env_cls(render_mode="rgb_array", camera_id=2)
    except TypeError:
        env = env_cls()
    _patch_get_dict(env)

    env._freeze_rand_vec = False
    env._set_task_called = True
    env.seed(seed)

    return TimeLimit(NormalizedBoxEnv(env), env.max_path_length)


class env_wrapper:
    def __init__(self, env, args):
        self._env = env
        self.args = args
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self):
        self.timesteps = 0
        inner = _unwrap_through_wrappers(self._env)
        tasks = getattr(inner, "mt1_train_tasks", None)
        if tasks is not None:
            if not hasattr(self, "_mt1_rng"):
                self._mt1_rng = np.random.default_rng(int(getattr(self.args, "seed", 0)))
            idx = int(self._mt1_rng.integers(0, len(tasks)))
            inner.set_task(tasks[idx])
        obs, info = self._env.reset()
        return obs, info

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        self.timesteps += 1
        ep_len = self.args.episode_length
        if self.timesteps >= ep_len:
            done = True
        return obs, reward, done, info

    def render(self, *args, **kwargs):
        """Forward to the wrapped env so GIF recording can use rgb_array / offscreen APIs."""
        return self._env.render(*args, **kwargs)

    def seed(self, seed):
        self._env.seed(seed)

    def close(self):
        """Forward to the wrapped env so MuJoCo / EGL contexts are freed deterministically."""
        close_fn = getattr(self._env, "close", None)
        if callable(close_fn):
            close_fn()
