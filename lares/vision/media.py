"""Choosing what the vision model sees (``spec.md`` 7.4, FR-11).

The selection rule is fixed and declared here, before any analysis runs, for the
same reason the promotion rule is: a rule chosen after looking at the frames
selects the frames that support the story. Five moments, in timestep order:

``start``            the first state of the episode
``closest_approach`` the step where the hand came nearest the object
``gate_transition``  the first crossing of 0.5 by any exposed gate
``max_object_move``  the step where the object moved furthest in one step
``termination``      the last state

A moment that cannot be measured is omitted and named in ``unavailable``, rather
than substituted with a nearby frame that would look like a measurement.

Final-test media never leaves this module. FR-11 says visual analysis must never
be returned to the search loop from the final split, so capture refuses that
manifest outright rather than relying on a caller to remember.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from lares.vision.schema import MediaItem

#: The rule, named so it can be recorded on every analysis and compared across
#: runs. Change the moments and change this string.
MEDIA_SELECTION_RULE = "v1: start, closest_approach, gate_transition, max_object_move, termination"

MOMENTS = ("start", "closest_approach", "gate_transition", "max_object_move", "termination")

GATE_THRESHOLD = 0.5


class FinalTestMediaRefused(RuntimeError):
    """Raised when media is requested from the final-test split."""


@dataclass
class EpisodeMedia:
    """Frames from one episode, with the per-step series that selected them."""

    case_id: str
    frames: list = field(default_factory=list)
    tcp: list = field(default_factory=list)
    obj: list = field(default_factory=list)
    goal: list = field(default_factory=list)
    gates: list = field(default_factory=list)
    unavailable: list = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.tcp)


class EpisodeRecorder:
    """A `run_episode` observer that renders and keeps the per-step geometry.

    Hangs off the single rollout every actor already goes through, so this is
    not another copy of the rollout loop.
    """

    def __init__(self, env, schema, render=True):
        self.env = env
        self.schema = schema
        self.render = render
        self.media = EpisodeMedia(case_id="")

    def _row(self, obs, name):
        return np.asarray(self.schema.slice(np.asarray(obs)[None, :], name),
                          dtype=np.float64).reshape(-1)

    def _frame(self):
        if not self.render:
            return None
        render = getattr(self.env, "render", None)
        if not callable(render):
            return None
        try:
            return render()
        except Exception:
            # A missing renderer must not take the episode down with it; the
            # geometry is still worth having and the caller is told.
            self.render = False
            return None

    def __call__(self, step: dict) -> None:
        frame = self._frame()
        self.media.frames.append(frame)
        if self.schema is not None:
            obs = step["obs"]
            self.media.tcp.append(self._row(obs, "tcp"))
            self.media.obj.append(self._row(obs, "obj"))
            self.media.goal.append(self._row(obs, "goal"))
        gates = step.get("gates")
        self.media.gates.append(
            {k: float(np.asarray(v).mean()) for k, v in gates.items()} if gates else {}
        )


def record_episode_media(actor, env, case, manifest_split, schema, max_steps=150,
                         render=True):
    """Roll out one case, keeping frames and geometry.

    Refuses the final-test split. FR-11's requirement that final-test media never
    reach the search loop is enforced here rather than trusted to a caller.
    """
    from lares.eval.manifest import SPLIT_FINAL_TEST
    from lares.eval.runner import run_episode

    if manifest_split == SPLIT_FINAL_TEST:
        raise FinalTestMediaRefused(
            f"media was requested for {case.case_id} on the final-test split. "
            f"Final-test media and the analysis of it must never return to the "
            f"search loop (FR-11)."
        )
    recorder = EpisodeRecorder(env, schema, render=render)
    record = run_episode(actor, env, case, max_steps=max_steps, schema=schema,
                         observer=recorder)
    recorder.media.case_id = case.case_id
    if not any(f is not None for f in recorder.media.frames):
        recorder.media.unavailable.append("frames: the environment returned none")
    return record, recorder.media


# ---------------------------------------------------------------------------
#  The selection rule
# ---------------------------------------------------------------------------


def _closest_approach(media: EpisodeMedia):
    if media.length < 1 or not media.obj:
        return None
    distances = np.linalg.norm(np.asarray(media.tcp) - np.asarray(media.obj), axis=1)
    return int(np.argmin(distances))


def _max_object_move(media: EpisodeMedia):
    if media.length < 2:
        return None
    steps = np.linalg.norm(np.diff(np.asarray(media.obj), axis=0), axis=1)
    if not steps.size or float(steps.max()) <= 0.0:
        return None
    return int(np.argmax(steps)) + 1


def _gate_transition(media: EpisodeMedia):
    names = sorted({k for g in media.gates for k in g})
    if not names:
        return None
    for name in names:
        series = [g.get(name) for g in media.gates]
        previous = None
        for t, value in enumerate(series):
            if value is None:
                continue
            if previous is not None and (previous < GATE_THRESHOLD) != (value < GATE_THRESHOLD):
                return t
            previous = value
    return None


def select_media(media: EpisodeMedia, prefix: str = "frame") -> tuple:
    """Apply the fixed rule. Returns ``(items, unavailable)``.

    Items are unique and in timestep order. A moment that cannot be measured is
    listed in ``unavailable`` rather than replaced.
    """
    if media.length == 0:
        return [], ["every moment: the episode recorded no steps"]

    chosen = {}
    unavailable = list(media.unavailable)
    picks = {
        "start": 0,
        "closest_approach": _closest_approach(media),
        "gate_transition": _gate_transition(media),
        "max_object_move": _max_object_move(media),
        "termination": media.length - 1,
    }
    for moment in MOMENTS:
        timestep = picks.get(moment)
        if timestep is None:
            unavailable.append(f"{moment}: not measurable from this episode")
            continue
        timestep = int(np.clip(timestep, 0, media.length - 1))
        # A moment that lands on a frame already chosen extends that frame's
        # reason rather than showing the model the same picture twice.
        if timestep in chosen:
            chosen[timestep].reason += f", {moment}"
            continue
        chosen[timestep] = MediaItem(
            frame_or_clip_id=f"{prefix}_t{timestep:04d}",
            start_timestep=timestep,
            end_timestep=timestep,
            reason=moment,
        )
    return [chosen[t] for t in sorted(chosen)], unavailable


def save_media(media: EpisodeMedia, items, directory: str) -> list:
    """Write the selected frames, and record the path on each item."""
    try:
        import imageio.v2 as imageio
    except ImportError:  # pragma: no cover - depends on the environment
        return items
    os.makedirs(directory, exist_ok=True)
    for item in items:
        frame = (media.frames[item.start_timestep]
                 if item.start_timestep < len(media.frames) else None)
        if frame is None:
            continue
        path = os.path.join(directory, f"{item.frame_or_clip_id}.png")
        imageio.imwrite(path, np.asarray(frame))
        item.path = path
    return items
