"""R-048 pre-launch check: the PPO expert queried at given observations through the stream's query mode.

Runs in the orchestrator's venv; the query itself runs in a fresh Isaac process with no episode. Uses observations from
the pinned expert demonstration, so the expert's queried actions there should match its executed (clamped) actions.

    .venv-metaworld/bin/python scripts/isaac/check_expert_query.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np  # noqa: E402

from lares.utils.cli import load_yaml  # noqa: E402
from lares.envs.isaac_pen_spin import cases  # noqa: E402
from lares.envs.isaac_pen_spin.pipeline_backend import IsaacExpertActor, IsaacPenSpinStream  # noqa: E402


def main():
    cfg = load_yaml("config/pen_spin_plateau_repair.yaml")
    e, lf = cfg["pen_spin"]["expert"], cfg["local_feedback_evidence"]
    demo = np.load(lf["demo_trace"])
    obs, act = demo["obs"][0, :40:4], demo["actions"][0, :40:4]
    stream = IsaacPenSpinStream(3, tempfile.mkdtemp(prefix="query_check_"), "check",
                                isaac_env=cfg["pen_spin"]["isaac_env"])
    expert = IsaacExpertActor(e["checkpoint"], e["checkpoint_sha256"], int(e["training_seed"]))
    raw = stream.query_actions(expert, obs, cases.build_manifest(3, 1, 1, label="check"))
    err = np.abs(np.clip(raw, -1, 1) - act).max()
    print(f"queries={len(obs)} episodes={stream.episodes} max|queried-executed|={err:.2e}")
    assert stream.episodes == 0 and err < 1e-4, err
    print("OK")


if __name__ == "__main__":
    main()
