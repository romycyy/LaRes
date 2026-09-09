#!/usr/bin/env python
"""Regenerate the search summary from saved experiment records (FR-10, AC-4, AC-7).

Reads only what was written to disk. If a number cannot be produced from the
records alone, the records are incomplete and this says so rather than
reconstructing it from terminal output.

Usage (from the project root)::

    python scripts/summarise_experiments.py
    python scripts/summarise_experiments.py --log-dir ./logs/evolution
"""

import argparse
import json
import os
import sys
from collections import Counter

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from lares.search import STATUS_INVALID, STATUS_PROMOTED, load_records  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--log-dir", default=os.path.join(_PROJECT_ROOT, "logs", "evolution"))
    p.add_argument("--out", default="", help="Write the summary as JSON to this path.")
    return p.parse_args()


def separator(title):
    print(f"\n{'=' * 76}\n  {title}\n{'=' * 76}")


def main():
    args = parse_args()
    records_dir = os.path.join(args.log_dir, "experiments")
    records = load_records(records_dir)
    if not records:
        print(f"No experiment records under {records_dir}.")
        print("A search run writes one per attempted candidate, including invalid ones.")
        return 1

    separator(f"{len(records)} attempted candidates in {records_dir}")
    statuses = Counter(r.status for r in records)
    for status, count in sorted(statuses.items()):
        print(f"  {status:<12} {count}")

    attempted = len(records)
    invalid = statuses.get(STATUS_INVALID, 0)
    reached = attempted - invalid
    repairs = [
        r.validation.get("repair_attempts", 0)
        for r in records
        if r.status == STATUS_INVALID
    ]
    separator("Generation reliability (AC-4)")
    print(f"  reached rollout validation : {reached}/{attempted} "
          f"({reached / attempted:.1%})")
    print(f"  target                     : at least 95% after no more than one repair")
    if repairs:
        print(f"  repair attempts on failures: max {max(repairs)}, mean {sum(repairs)/len(repairs):.2f}")
    over_budget = [r for r in records if r.validation.get("repair_attempts", 0) > 1]
    if over_budget:
        print(f"  WARNING: {len(over_budget)} candidate(s) used more than one repair")

    separator("Per-generation")
    by_generation: dict[int, list] = {}
    for r in records:
        by_generation.setdefault(r.generation, []).append(r)
    header = f"{'gen':>4}{'attempted':>11}{'invalid':>9}{'promoted':>10}{'best success':>14}"
    print(header)
    print("-" * len(header))
    for gen in sorted(by_generation):
        group = by_generation[gen]
        successes = [
            r.evaluation.get("success_rate")
            for r in group
            if r.evaluation.get("success_rate") is not None
        ]
        print(
            f"{gen:>4}{len(group):>11}"
            f"{sum(1 for r in group if r.status == STATUS_INVALID):>9}"
            f"{sum(1 for r in group if r.status == STATUS_PROMOTED):>10}"
            f"{(max(successes) if successes else float('nan')):>14.3f}"
        )

    separator("Traceability (AC-4)")
    missing_hash = [r.candidate_id for r in records if not r.code.get("source_hash")]
    missing_manifest = [
        r.candidate_id
        for r in records
        if r.status not in (STATUS_INVALID,) and not r.evaluation.get("manifest_ids")
    ]
    missing_mode = [
        r.candidate_id
        for r in records
        if r.status not in (STATUS_INVALID,) and not r.evaluation.get("action_mode")
    ]
    for label, missing in (
        ("code hash", missing_hash),
        ("manifest ids", missing_manifest),
        ("action mode", missing_mode),
    ):
        status = "complete" if not missing else f"MISSING on {missing}"
        print(f"  {label:<14} {status}")

    separator("Interventions")
    interventions = [r for r in records if r.intervention]
    if not interventions:
        print("  none recorded; nothing is excluded from fitness ranking")
    else:
        for r in interventions:
            print(f"  {r.candidate_id}: {r.intervention_detail}")

    separator("Failures preserved as evidence")
    for r in records:
        if r.status != STATUS_INVALID:
            continue
        error = (r.validation.get("errors") or [""])[0].splitlines()
        print(f"  {r.candidate_id:<24} {r.validation.get('stage','')}: "
              f"{error[0][:80] if error else ''}")

    summary = {
        "records_dir": records_dir,
        "attempted": attempted,
        "statuses": dict(statuses),
        "reached_validation_fraction": reached / attempted,
        "per_generation": {
            str(gen): {
                "attempted": len(group),
                "invalid": sum(1 for r in group if r.status == STATUS_INVALID),
                "promoted": sum(1 for r in group if r.status == STATUS_PROMOTED),
            }
            for gen, group in sorted(by_generation.items())
        },
        "traceability_gaps": {
            "code_hash": missing_hash,
            "manifest_ids": missing_manifest,
            "action_mode": missing_mode,
        },
        "interventions": [r.candidate_id for r in interventions],
    }
    out = args.out or os.path.join(args.log_dir, "experiment_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  summary: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
