"""R-071: paired axial-direction reversal rates of the R-068 zero-curl clamp versus the parent, from archived traces.

    python scripts/isaac/axial_reversals.py --run-dir logs/pen_spin/r068_capture_curl_confirmation \
        --out logs/pen_spin/r071_axial_reversals/summary.json --hysteresis 1.0 --persistence-steps 3

Offline and numpy only: no simulator, no LLM; a missing record ends the analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from lares.envs.isaac_pen_spin.axial_reversals import analyse  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--hysteresis", type=float, default=1.0)
    p.add_argument("--persistence-steps", type=int, default=3)
    a = p.parse_args()
    summary = analyse(a.run_dir, a.hysteresis, a.persistence_steps)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "panels"}, indent=1))
    for panel, v in summary["panels"].items():
        t = v["phase_transitions"]
        print(panel, "zero_curl - parent reversals/s", round(t["mean"], 4), [round(t["low"], 4), round(t["high"], 4)],
              v["win_loss_tie"], {arm: round(x["reversals_per_second"], 4) for arm, x in v["arms"].items()})


if __name__ == "__main__":
    main()
