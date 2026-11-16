"""Launch and close the one Isaac Sim app a process may hold, with Isaac Lab on the path."""

from __future__ import annotations

import os
import sys
from typing import Any

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
