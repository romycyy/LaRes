"""Named observation accessors, validated against the installed environment.

Generated policies read the observation through names, never through raw indices
(``spec.md`` FR-2 / AC-1).  A missing table entry once sent the LLM a blank
observation layout for ``push-v2``; it invented indices, read the puck's
quaternion as its position, and every candidate that generation scored about 16.
Nothing in the pipeline could tell that apart from a bad control law.

A schema is only trusted once :func:`ObsSchema.check_against_env` has compared
each field against the value the simulator reports for it.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ObsField:
    """One named slice of the flat observation vector."""

    name: str
    start: int
    stop: int
    description: str
    required: bool = False

    @property
    def size(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class ObsSchema:
    """The observation layout for one task, plus the fields a policy must use."""

    env_id: str
    obs_dim: int
    fields: tuple[ObsField, ...]

    def __post_init__(self):
        seen = set()
        for f in self.fields:
            if f.name in seen:
                raise ValueError(f"duplicate observation field {f.name!r}")
            seen.add(f.name)
            if not (0 <= f.start < f.stop <= self.obs_dim):
                raise ValueError(
                    f"field {f.name!r} slice [{f.start}:{f.stop}] is outside "
                    f"an observation of {self.obs_dim} dimensions"
                )

    # -- lookup -----------------------------------------------------------

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields)

    @property
    def required_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fields if f.required)

    def field(self, name: str) -> ObsField:
        for f in self.fields:
            if f.name == name:
                return f
        raise KeyError(
            f"{name!r} is not a field of the {self.env_id} observation schema. "
            f"Available: {', '.join(self.names)}"
        )

    def slice(self, obs, name: str):
        """``(batch, size)`` view of one named field.

        Accepts a ``(batch, obs_dim)`` tensor, or a ``(obs_dim,)`` one which is
        treated as a single-row batch.
        """
        f = self.field(name)
        t = obs if torch.is_tensor(obs) else torch.as_tensor(np.asarray(obs))
        if t.dim() == 1:
            t = t.unsqueeze(0)
        if t.shape[-1] != self.obs_dim:
            raise ValueError(
                f"observation has {t.shape[-1]} dimensions, schema {self.env_id} "
                f"expects {self.obs_dim}"
            )
        return t[:, f.start : f.stop]

    def describe(self) -> str:
        """Human-readable layout, for prompts and error messages."""
        lines = [f"Named observation fields for {self.env_id} (obs_dim={self.obs_dim}):"]
        for f in self.fields:
            tag = " [required]" if f.required else ""
            lines.append(f"  {f.name:<16} size {f.size}  {f.description}{tag}")
        return "\n".join(lines)

    # -- validation against the simulator ---------------------------------

    def check_against_env(self, env, case=None, atol: float = 1e-6) -> list[str]:
        """Compare each checkable field against what the simulator reports.

        Returns a list of discrepancy strings; empty means the layout holds.
        Only fields the environment exposes a direct accessor for are checked;
        the rest are reported as unverifiable rather than silently passed.
        """
        from lares.utils.metaworld_env import _unwrap_through_wrappers

        obs, _ = env.reset(case)
        inner = _unwrap_through_wrappers(getattr(env, "_env", env))
        obs = np.asarray(obs, dtype=np.float64)

        problems: list[str] = []
        if obs.shape[-1] != self.obs_dim:
            problems.append(
                f"observation is {obs.shape[-1]}-dimensional, schema says {self.obs_dim}"
            )
            return problems

        # Multi-object tasks return every object's position (quaternion) back to back; the first object is obj.
        checks = {
            "tcp": lambda: inner.get_endeff_pos(),
            "obj": lambda: np.asarray(inner._get_pos_objects()).ravel()[:3],
            "obj_quat": lambda: np.asarray(inner._get_quat_objects()).ravel()[:4],
            "obj2": lambda: np.asarray(inner._get_pos_objects()).ravel()[3:6],
            "goal": lambda: inner._target_pos,
        }
        for name, getter in checks.items():
            if name not in self.names:
                continue
            f = self.field(name)
            try:
                expected = np.asarray(getter(), dtype=np.float64).ravel()
            except Exception as exc:
                problems.append(f"{name}: environment accessor failed ({exc!r})")
                continue
            actual = obs[f.start : f.stop]
            if expected.shape != actual.shape:
                problems.append(
                    f"{name}: slice [{f.start}:{f.stop}] has size {actual.size}, "
                    f"environment reports size {expected.size}"
                )
                continue
            if not np.allclose(actual, expected, atol=atol):
                problems.append(
                    f"{name}: slice [{f.start}:{f.stop}] = {actual} but the environment "
                    f"reports {expected}"
                )
        return problems

    def unverifiable_names(self) -> tuple[str, ...]:
        """Fields with no direct simulator accessor, so never machine-checked."""
        checkable = {"tcp", "obj", "obj_quat", "obj2", "goal"}
        return tuple(n for n in self.names if n not in checkable)


# ---------------------------------------------------------------------------
#  Task schemas
# ---------------------------------------------------------------------------

# MetaWorld V3 assembles the flat observation as
#   [endeff_pos(3), gripper_apart(1), padded object block(14), previous 18, goal(3)]
# See metaworld.sawyer_xyz_env._get_curr_obs_combined_no_goal.
_METAWORLD_V3_COMMON = (
    ObsField("tcp", 0, 3, "end-effector position (x, y, z)", required=True),
    ObsField("gripper", 3, 4, "normalised gripper opening, 1.0 = fully open"),
    ObsField("obj", 4, 7, "manipulated object position (x, y, z)", required=True),
    ObsField("obj_quat", 7, 11, "manipulated object orientation quaternion"),
    ObsField("prev_tcp", 18, 21, "end-effector position one step earlier"),
    ObsField("prev_gripper", 21, 22, "gripper opening one step earlier"),
    ObsField("prev_obj", 22, 25, "object position one step earlier"),
    ObsField("goal", 36, 39, "goal position (x, y, z)", required=True),
)

#: R-024 (approved 2026-10-22): the second object block, verified against the simulator before use. MetaWorld lays out
#: each object's position then quaternion, so a second object's position sits at obs[11:14]. Only hammer-v2 among the
#: registered tasks has one (the nail); its quaternion is not exposed.
_HAMMER_V3 = _METAWORLD_V3_COMMON + (
    ObsField("obj2", 11, 14, "second object position (x, y, z): the nail in hammer-v2"),
)

OBS_SCHEMAS: dict[str, ObsSchema] = {
    env_id: ObsSchema(env_id, 39, _METAWORLD_V3_COMMON)
    for env_id in (
        "push-v2",
        "reach-v2",
        "pick-place-v2",
        "window-close-v2",
        "window-open-v2",
        "button-press-v2",
        "door-close-v2",
        "door-open-v2",
        "drawer-open-v2",
        "drawer-close-v2",
        # R-024: single-object harder tasks on the same layout (obj = the manipulated object, read from source).
        "assembly-v2",
        "peg-insert-side-v2",
        "shelf-place-v2",
        "basketball-v2",
        "soccer-v2",
    )
}
OBS_SCHEMAS["hammer-v2"] = ObsSchema("hammer-v2", 39, _HAMMER_V3)


def _pen_spin_schema(stage: int) -> ObsSchema:
    """R-042: the Shadow Hand pen-spin task's 157-d "full" observation, from ``cases.OBS_LAYOUT``.

    Not checked by :meth:`ObsSchema.check_against_env` (that reads MetaWorld accessors); the layout is the env's
    own ``compute_full_observations`` order, and ``tests/test_pen_spin_pipeline`` checks these slices against it.
    The required fields are the ones a spinning controller cannot do without: the pen's position, orientation and
    angular velocity, and the hand's joint positions.
    """
    from lares.envs.isaac_pen_spin.layout import OBS_LAYOUT, STAGE_TASK_IDS

    required = {"obj_pos", "obj_rot", "obj_angvel", "hand_dof_pos"}
    fields, start = [], 0
    for name, size, description in OBS_LAYOUT:
        fields.append(ObsField(name, start, start + size, description, required=name in required))
        start += size
    return ObsSchema(STAGE_TASK_IDS[stage], start, tuple(fields))


def _gait_scaffold_schema() -> ObsSchema:
    """R-046: the Stage-3 layout plus ``gait_action`` (177-d); ``gait_action`` is readable but not required."""
    from lares.envs.isaac_pen_spin.layout import GAIT_FIELD, GAIT_SCAFFOLD_TASK_ID

    base = _pen_spin_schema(3)
    name, size, description = GAIT_FIELD
    fields = tuple(base.fields) + (ObsField(name, base.obs_dim, base.obs_dim + size, description, required=False),)
    return ObsSchema(GAIT_SCAFFOLD_TASK_ID, base.obs_dim + size, fields)


for _stage in (1, 2, 3):
    _s = _pen_spin_schema(_stage)
    OBS_SCHEMAS[_s.env_id] = _s
del _stage, _s
OBS_SCHEMAS[_gait_scaffold_schema().env_id] = _gait_scaffold_schema()


def get_obs_schema(env_id: str) -> ObsSchema:
    """Schema for ``env_id``, or a clear failure naming what is missing.

    Refusing an unknown task is the point: a blank layout is what produced the
    ``push-v2`` failure this module exists to prevent.
    """
    try:
        return OBS_SCHEMAS[env_id]
    except KeyError:
        raise KeyError(
            f"No observation schema for {env_id!r}. Add one to "
            f"lares/core/obs_schema.py and verify it with "
            f"ObsSchema.check_against_env before generating policies for this task. "
            f"Known: {', '.join(sorted(OBS_SCHEMAS))}"
        ) from None


# ---------------------------------------------------------------------------
#  Active schema
# ---------------------------------------------------------------------------

_ACTIVE: ObsSchema | None = None
#: Per-thread stack of schemas bound by :class:`active_schema`. Concurrent generation (several studies' LLM slots on
#: worker threads) constructs policies at the same time; a single global binding would let one thread's ``__exit__``
#: unbind the schema while another thread is mid-construction.
_LOCAL = threading.local()


def set_active_schema(schema: ObsSchema | None) -> None:
    """Bind the process-wide schema new :class:`SymbolicPolicy` instances will read through.

    An :class:`active_schema` block on the current thread takes precedence over this binding while it is open.
    """
    global _ACTIVE
    _ACTIVE = schema


def get_active_schema() -> ObsSchema | None:
    stack = getattr(_LOCAL, "stack", None)
    if stack:
        return stack[-1]
    return _ACTIVE


class active_schema:
    """Context manager binding a schema for the duration of a block, on the current thread only."""

    def __init__(self, schema: ObsSchema | None):
        self.schema = schema

    def __enter__(self) -> ObsSchema | None:
        if not hasattr(_LOCAL, "stack"):
            _LOCAL.stack = []
        _LOCAL.stack.append(self.schema)
        return self.schema

    def __exit__(self, *exc) -> None:
        _LOCAL.stack.pop()
        return None
