"""Per-episode geometry and action diagnostics (``spec.md`` 7.3, FR-3, AC-2).

A success rate says a candidate failed. These say *how*: whether the arm ever
reached the object, whether the object moved at all, whether it moved toward the
goal or away from it, and whether the controller was railed against its action
limits the whole time. Those are different repairs.

Everything here is measured through :class:`~lares.core.obs_schema.ObsSchema`, so
a field the schema does not carry is reported as unavailable rather than
silently computed from the wrong slice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lares.core.obs_schema import ObsSchema

#: The arm counts as having reached the object below this separation, in metres.
#: Wider than MetaWorld's own 0.03 m ``near_object`` flag so a near miss is not
#: labelled "never reached".
REACH_RADIUS = 0.05

#: Object displacement below this is indistinguishable from simulator jitter.
MOVED_THRESHOLD = 0.02

#: An action component above this magnitude is treated as railed.
SATURATION_THRESHOLD = 0.99

FAILURE_LABELS = (
    "success",
    "never_reached_object",
    "object_not_moved",
    "pushed_away_from_goal",
    "stopped_short_of_goal",
)


@dataclass
class RolloutGeometry:
    """Accumulates geometry and action statistics over one episode.

    Call :meth:`observe` with the observation each action was chosen from and the
    action taken, then :meth:`finish` with the final observation.
    """

    schema: ObsSchema | None
    action_dim: int
    success_radius: float = 0.05

    _tcp: list = field(default_factory=list, init=False)
    _obj: list = field(default_factory=list, init=False)
    _goal: list = field(default_factory=list, init=False)
    _actions: list = field(default_factory=list, init=False)
    _usable: bool = field(default=False, init=False)

    def __post_init__(self):
        self._usable = self.schema is not None and all(
            name in self.schema.names for name in ("tcp", "obj", "goal")
        )

    @property
    def available(self) -> bool:
        return self._usable

    def _row(self, obs, name):
        return np.asarray(self.schema.slice(obs, name), dtype=np.float64).reshape(-1)

    def observe(self, obs, action=None) -> None:
        if action is not None:
            self._actions.append(np.asarray(action, dtype=np.float64).reshape(-1))
        if not self._usable:
            return
        self._tcp.append(self._row(obs, "tcp"))
        self._obj.append(self._row(obs, "obj"))
        self._goal.append(self._row(obs, "goal"))

    def finish(self, final_obs) -> None:
        if not self._usable or final_obs is None:
            return
        self._tcp.append(self._row(final_obs, "tcp"))
        self._obj.append(self._row(final_obs, "obj"))
        self._goal.append(self._row(final_obs, "goal"))

    # -- results ----------------------------------------------------------

    def action_statistics(self) -> dict:
        """Per-axis saturation and step-to-step variation."""
        if not self._actions:
            return {
                "action_saturation_by_axis": None,
                "action_variation_by_axis": None,
                "action_mean_by_axis": None,
            }
        arr = np.asarray(self._actions, dtype=np.float64)
        saturation = (np.abs(arr) > SATURATION_THRESHOLD).mean(axis=0)
        variation = (
            np.abs(np.diff(arr, axis=0)).mean(axis=0)
            if arr.shape[0] > 1
            else np.zeros(arr.shape[1])
        )
        return {
            "action_saturation_by_axis": [float(x) for x in saturation],
            "action_variation_by_axis": [float(x) for x in variation],
            "action_mean_by_axis": [float(x) for x in arr.mean(axis=0)],
        }

    def geometry_statistics(self) -> dict:
        """Approach, displacement and goal-progress measures.

        Every field is ``None`` when the schema cannot supply it, so an
        unavailable measurement is never read as a zero.
        """
        empty = {
            "initial_goal_distance": None,
            "final_goal_distance": None,
            "signed_goal_progress": None,
            "lateral_drift": None,
            "object_displacement": None,
            "min_tcp_object_distance": None,
            "final_tcp_object_distance": None,
        }
        if not self._usable or len(self._obj) < 2:
            return empty

        tcp = np.asarray(self._tcp)
        obj = np.asarray(self._obj)
        goal = np.asarray(self._goal)

        d_initial = float(np.linalg.norm(obj[0] - goal[0]))
        d_final = float(np.linalg.norm(obj[-1] - goal[-1]))
        displacement_vec = obj[-1] - obj[0]

        # Drift is the part of the object's motion perpendicular to the straight
        # line it should have travelled. Separates "pushed the wrong way" from
        # "pushed the right way but not far enough".
        to_goal = goal[0] - obj[0]
        norm = np.linalg.norm(to_goal)
        if norm > 1e-9:
            unit = to_goal / norm
            along = float(np.dot(displacement_vec, unit))
            lateral = float(np.linalg.norm(displacement_vec - along * unit))
        else:
            lateral = None

        separation = np.linalg.norm(tcp - obj, axis=1)
        return {
            "initial_goal_distance": d_initial,
            "final_goal_distance": d_final,
            "signed_goal_progress": d_initial - d_final,
            "lateral_drift": lateral,
            "object_displacement": float(np.linalg.norm(displacement_vec)),
            "min_tcp_object_distance": float(separation.min()),
            "final_tcp_object_distance": float(separation[-1]),
        }

    def failure_label(self, success: float) -> str | None:
        """Coarse cause of failure, or ``None`` when geometry is unavailable.

        Heuristic, and deliberately coarse: the thresholds it uses are reported
        alongside it so a reader can see what the label means rather than
        trusting the word.
        """
        if float(success) > 0:
            return "success"
        stats = self.geometry_statistics()
        if stats["min_tcp_object_distance"] is None:
            return None
        if stats["min_tcp_object_distance"] > REACH_RADIUS:
            return "never_reached_object"
        if stats["object_displacement"] < MOVED_THRESHOLD:
            return "object_not_moved"
        if stats["signed_goal_progress"] <= 0.0:
            return "pushed_away_from_goal"
        return "stopped_short_of_goal"

    def thresholds(self) -> dict:
        return {
            "reach_radius_m": REACH_RADIUS,
            "moved_threshold_m": MOVED_THRESHOLD,
            "saturation_threshold": SATURATION_THRESHOLD,
            "success_radius_m": self.success_radius,
        }


def aggregate_by_axis(values, action_dim: int) -> list | None:
    """Mean of per-axis vectors across episodes, or ``None`` if none were measured."""
    rows = [v for v in values if v is not None]
    if not rows:
        return None
    arr = np.asarray(rows, dtype=float)
    return [float(x) for x in arr.mean(axis=0)]


def aggregate_gate_statistics(per_episode) -> dict | None:
    """Average each gate's statistics across the episodes that exposed it.

    ``first_crossing_step`` averages only over episodes where the gate actually
    crossed, and the count of those episodes is reported, so "crossed late" and
    "usually never crossed" do not look the same.
    """
    rows = [g for g in per_episode if g]
    if not rows:
        return None
    names = sorted({name for row in rows for name in row})
    out: dict[str, dict] = {}
    for name in names:
        entries = [row[name] for row in rows if name in row]
        crossings = [
            e["first_crossing_step"]
            for e in entries
            if e.get("first_crossing_step") is not None
        ]
        out[name] = {
            "episodes_measured": len(entries),
            "mean": float(np.mean([e["mean"] for e in entries])),
            "min": float(np.min([e["min"] for e in entries])),
            "max": float(np.max([e["max"] for e in entries])),
            "occupancy_above_half": float(
                np.mean([e["occupancy_above_half"] for e in entries])
            ),
            "episodes_with_crossing": len(crossings),
            "mean_first_crossing_step": float(np.mean(crossings)) if crossings else None,
            "mean_transitions": float(np.mean([e["transitions"] for e in entries])),
        }
    return out
