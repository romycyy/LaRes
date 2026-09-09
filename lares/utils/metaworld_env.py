"""MetaWorld environment factory and episode wrapper for the evolution pipeline.

Every reset names the placement it wants.  There is no hidden task stream: the
caller passes an :class:`~lares.eval.manifest.EpisodeCase`, the wrapper installs
that placement and reseeds the simulator, and the episode that follows is
bit-identical no matter what ran before it (``spec.md`` FR-1 / AC-0).
"""

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


def make_metaworld_env(cfg, seed, tasks=None):
    """Build the wrapped MetaWorld env.

    Args:
        cfg: namespace with ``env_name`` and optional ``use_mt1``.
        seed: initial simulator seed.  Every episode reseeds from its own case,
            so this only fixes the state before the first reset.
        tasks: mapping ``{task_id: metaworld.Task}`` (or an iterable of tasks)
            when ``cfg.use_mt1`` is set.  Required on that path: the pool seed
            is part of the evaluation contract, so the env never draws its own.

    Returns:
        ``TimeLimit(NormalizedBoxEnv(env))``.  Wrap in :class:`env_wrapper` to
        get the per-case reset API.
    """
    env_name = cfg.env_name
    env_name_v3 = env_name.replace("-v2", "-v3").replace("-v1", "-v3")
    if env_name_v3 not in _env_dict.ALL_V3_ENVIRONMENTS:
        raise ValueError(
            f"Environment '{env_name}' (looked up as '{env_name_v3}') not found in ALL_V3_ENVIRONMENTS"
        )

    use_mt1 = getattr(cfg, "use_mt1", False)
    env_cls = _env_dict.ALL_V3_ENVIRONMENTS[env_name_v3]
    try:
        env = env_cls(render_mode="rgb_array", camera_id=2)
    except TypeError:
        env = env_cls()
    _patch_get_dict(env)
    env.seed(seed)

    if use_mt1:
        if tasks is None:
            raise ValueError(
                "use_mt1=True requires an explicit task pool. Build one with "
                "lares.eval.manifest.build_task_pool(...) and pass "
                "tasks=pool.task_index(); MT1 constructed without a seed draws "
                "different placements on every run."
            )
        task_list = list(tasks.values()) if isinstance(tasks, dict) else list(tasks)
        if not task_list:
            raise ValueError("task pool is empty")
        # Install one placement so the env is usable before the first cased reset.
        env.set_task(task_list[0])
    else:
        # Non-MT1 tasks resample the placement at reset. Route that draw through
        # the env's own generator so a per-case seed makes it reproducible;
        # the default path uses the process-global numpy RNG.
        env._freeze_rand_vec = False
        env._set_task_called = True
        env.seeded_rand_vec = True

    return TimeLimit(NormalizedBoxEnv(env), env.max_path_length)


class env_wrapper:
    """Per-case episode wrapper.

    ``reset`` takes the :class:`~lares.eval.manifest.EpisodeCase` it should run.
    Nothing about the episode depends on how many episodes preceded it, so
    dataset collection, GIF recording and debug rollouts cannot shift a later
    evaluation case.
    """

    def __init__(self, env, args, tasks=None):
        self._env = env
        self.args = args
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.use_mt1 = bool(getattr(args, "use_mt1", False))
        if isinstance(tasks, dict):
            self._task_index = dict(tasks)
        elif tasks is not None:
            self._task_index = {str(i): t for i, t in enumerate(tasks)}
        else:
            self._task_index = None
        if self.use_mt1 and not self._task_index:
            raise ValueError("use_mt1=True requires a {task_id: Task} mapping")
        self.current_case = None

    def reset(self, case=None):
        """Reset onto ``case``.

        Args:
            case: an ``EpisodeCase``.  Required.  Installs ``case.task_id`` and
                reseeds the simulator with ``case.reset_seed``.
        """
        if case is None:
            raise ValueError(
                "env_wrapper.reset() requires an EpisodeCase. Evaluation must name "
                "its placement and reset seed; see lares.eval.manifest."
            )
        self.timesteps = 0
        self.current_case = case
        inner = _unwrap_through_wrappers(self._env)
        if self._task_index is not None:
            task_id = getattr(case, "task_id", None)
            if task_id not in self._task_index:
                raise KeyError(
                    f"case {getattr(case, 'case_id', case)!r} names task_id "
                    f"{task_id!r}, which is not in this env's pool"
                )
            inner.set_task(self._task_index[task_id])
        inner.seed(int(case.reset_seed))
        obs, info = self._env.reset()
        return obs, info

    def step(self, action):
        out = self._env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = terminated or truncated
        else:
            obs, reward, done, info = out
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
