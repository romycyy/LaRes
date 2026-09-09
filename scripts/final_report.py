#!/usr/bin/env python
"""Regenerate the aggregate report from saved artifacts (``spec.md`` AC-7).

Reads files and nothing else. No simulator, no model call, no fitting. Run it on
a fresh checkout of the logs and it produces the same report, which is what
"reproducible from saved artifacts" has to mean.

Quantities AC-7 asks for that no artifact carries are listed as absent, together
with where each would come from. Filling them with zeros would make the report
look complete while resting on numbers nobody measured.

Usage (from the project root)::

    python scripts/final_report.py
    python scripts/final_report.py --json logs/final/aggregate_report.json
"""

import argparse
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from lares.eval.final_report import build_final_report, render  # noqa: E402

LOGS = os.path.join(_PROJECT_ROOT, "logs")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--experiments", nargs="+", default=[
        os.path.join(LOGS, "evolution", "experiments"),
        os.path.join(LOGS, "repair_studies", "experiments"),
    ])
    p.add_argument("--repair-studies", default=os.path.join(LOGS, "repair_studies"))
    p.add_argument("--ablations", default=os.path.join(LOGS, "repair_ablation"))
    p.add_argument("--final", default=os.path.join(LOGS, "final"))
    p.add_argument("--ablation-budget", type=int, default=None,
                   help="which matched budget to report; the largest by default")
    p.add_argument("--json", default=None, help="also write the report as JSON")
    return p.parse_args()


def main():
    args = parse_args()
    report = build_final_report(
        args.experiments, args.repair_studies, args.ablations, args.final,
        ablation_budget=args.ablation_budget,
    )
    print(render(report))
    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, sort_keys=True, default=str)
        print(f"\n  written: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
