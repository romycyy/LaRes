"""R-048 executed-switch audit (advisor call_00100): zero episodes, zero LLM calls, expert forward queries only.

Reads the saved block-0 evidence states (``evidence.npz`` rows ``evidence_rows``), queries the PPO expert on the frozen
grid ``local_feedback.switch_grid()`` (+-1 demonstration range per coordinate, 20 points), and writes the coverage of
executed-action sign switches. Every response is the expert model's extrapolation at learner states, not a verified
recovery action.

    .venv-metaworld/bin/python scripts/isaac/audit_expert_switches.py logs/pen_spin_local_feedback_evidence/block0
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np  # noqa: E402

from lares.utils.cli import load_yaml  # noqa: E402
from lares.envs.isaac_pen_spin import cases, local_feedback as lf  # noqa: E402
from lares.envs.isaac_pen_spin.pipeline_backend import IsaacExpertActor, IsaacPenSpinStream  # noqa: E402
from lares.utils.hashing import sha256_file  # noqa: E402


def main(bdir: str):
    cfg = load_yaml("config/pen_spin_plateau_repair.yaml")
    e, lcfg = cfg["pen_spin"]["expert"], cfg["local_feedback_evidence"]
    meta = json.load(open(os.path.join(bdir, "evidence.json")))
    states = np.load(os.path.join(bdir, "evidence.npz"))["obs"][meta["evidence_rows"]]
    if sha256_file(lcfg["demo_trace"]) != lcfg["demo_trace_sha256"]:
        raise SystemExit("demonstration trace does not match its pinned SHA-256")
    demo = np.load(lcfg["demo_trace"])
    demo_obs = np.concatenate([demo["obs"][i, :int(L)] for i, L in enumerate(demo["length"])], axis=0)
    ranges = lf.step_sizes(demo_obs) / lf.STEP_FRACTION
    batch = lf.switch_batch(states, ranges)
    out_dir = os.path.join(bdir, "switch_audit")
    stream = IsaacPenSpinStream(3, out_dir, "switch", isaac_env=cfg["pen_spin"]["isaac_env"])
    expert = IsaacExpertActor(e["checkpoint"], e["checkpoint_sha256"], int(e["training_seed"]))
    raw = stream.query_actions(expert, batch, cases.build_manifest(3, 1, 1, label="audit"))
    a = lf.switch_audit(raw, len(states))
    np.savez_compressed(os.path.join(out_dir, "switch_audit.npz"), nearest=a.pop("nearest"), changes=a.pop("changes"),
                        raw=raw)
    a.update(states=int(len(states)), queries=int(len(batch)), episodes=stream.episodes, llm_calls=0,
             demo_ranges=dict(zip(lf.COORDS, ranges.tolist())), source=os.path.join(bdir, "evidence.npz"))
    with open(os.path.join(out_dir, "switch_audit.json"), "w") as f:
        json.dump(a, f, indent=2)
    print(json.dumps(a, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
