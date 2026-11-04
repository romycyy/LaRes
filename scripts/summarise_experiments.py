#!/usr/bin/env python
"""Regenerate the search summary from saved experiment records (FR-10, AC-4, AC-7).

Reads only what was written to disk: the records, the LLM-call ledger, the
feedback bundles, the state and report artifacts, and an optional provenance
sidecar for runs recorded before lineage existed. If a number cannot be produced
from those files, the files are incomplete and this says so rather than
reconstructing it from terminal output.

A traceability field is reported ``complete`` only when *every* record carries
that evidence and it verifies: lineage, feedback, prompt hashes, LLM usage, state
artifacts, manifests, resource accounting and an exact reconstruction
(``debug_spec.md`` D11). A legacy record is never complete.

Usage (from the project root)::

    python scripts/summarise_experiments.py
    python scripts/summarise_experiments.py --log-dir ./logs/evolution
    python scripts/summarise_experiments.py --log-dir <run> --no-reconstruct
"""

import argparse
import json
import os
import sys
from collections import Counter

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from lares.utils.cli import separator  # noqa: E402
from lares.search import (  # noqa: E402
    LINEAGE_STATUSES,
    STATUS_INVALID,
    STATUS_PROMOTED,
    LLMCallLedger,
    load_records,
)
from lares.search.provenance import (  # noqa: E402
    lineage_status,
    load_sidecar,
    reconcile_resources,
    traceability_summary,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--log-dir", default=os.path.join(_PROJECT_ROOT, "logs", "evolution"))
    p.add_argument("--out", default="", help="Write the summary as JSON to this path.")
    p.add_argument(
        "--no-reconstruct",
        action="store_true",
        help="Skip rebuilding each policy. Reconstruction is then reported incomplete.",
    )
    return p.parse_args()


def _attempt(record) -> int:
    value = (record.lineage or {}).get("attempt")
    return int(value) if isinstance(value, int) else 0


def correction_summary(records, ledger) -> dict:
    """Corrections attempted, how many succeeded, and what the correction calls cost."""
    corrections = [r for r in records if (r.lineage or {}).get("previous_attempt_id")]
    call_ids = []
    for r in corrections:
        for call in r.llm.get("calls") or []:
            if call.get("label") == "correction":
                call_ids.append(call["call_id"])
    totals = ledger.totals(call_ids) if ledger is not None and len(ledger) else None
    return {
        "attempted": len(corrections),
        "succeeded": sum(1 for r in corrections if r.validation.get("passed") is True),
        "succeeded_attempt_ids": [r.experiment_id for r in corrections if r.validation.get("passed") is True],
        "correction_calls": len(set(call_ids)),
        "correction_total_tokens": totals["llm_total_tokens"] if totals else None,
    }


def main():
    args = parse_args()
    records_dir = os.path.join(args.log_dir, "experiments")
    records = load_records(records_dir)
    if not records:
        print(f"No experiment records under {records_dir}.")
        print("A search run writes one per attempted candidate, including invalid ones.")
        return 1
    sidecar = load_sidecar(args.log_dir)
    ledger_path = os.path.join(args.log_dir, LLMCallLedger.FILENAME)
    ledger = LLMCallLedger(args.log_dir) if os.path.isfile(ledger_path) else None

    generated = [r for r in records if not r.intervention]
    interventions = [r for r in records if r.intervention]

    separator(f"{len(records)} records in {records_dir}")
    statuses = Counter(r.status for r in records)
    for status, count in sorted(statuses.items()):
        print(f"  {status:<12} {count}")

    separator("Lineage status (debug_spec.md section 6)")
    lineage = Counter(lineage_status(r, sidecar) for r in generated)
    for name in LINEAGE_STATUSES:
        print(f"  {name:<18} {lineage.get(name, 0)}")
    if sidecar:
        print(f"  provenance sidecar version {sidecar.get('sidecar_version')} applied; raw records unchanged")

    # -- reliability: attempts are not slots ------------------------------------
    slots = {}
    for r in generated:
        key = (r.generation, r.candidate_id)
        if key not in slots or _attempt(r) > _attempt(slots[key]):
            slots[key] = r
    validated_attempts = sum(1 for r in generated if r.validation.get("passed") is True)
    validated_slots = sum(1 for r in slots.values() if r.validation.get("passed") is True)
    within_one = sum(
        1 for r in slots.values() if r.validation.get("passed") is True and _attempt(r) <= 1
    )
    corrections = correction_summary(generated, ledger)
    separator("Generation reliability (AC-4)")
    print(f"  attempts                        : {len(generated)}")
    print(f"  slots                           : {len(slots)}")
    print(f"  attempts that passed validation : {validated_attempts}")
    print(f"  slots whose final attempt passed: {validated_slots}/{len(slots)} ({validated_slots / len(slots):.1%})")
    print(f"  slots valid within one correction: {within_one}/{len(slots)}  (target: at least 95%)")
    print(f"  corrections attempted / succeeded: {corrections['attempted']} / {corrections['succeeded']}"
          + (f"  ({corrections['correction_total_tokens']} tokens)" if corrections["correction_total_tokens"] is not None else ""))
    over_budget = [r.experiment_id for r in generated if _attempt(r) > 1]
    if over_budget:
        print(f"  WARNING: attempts beyond one correction: {over_budget}")

    separator("Per-generation")
    by_generation: dict[int, list] = {}
    for r in generated:
        by_generation.setdefault(r.generation, []).append(r)
    header = f"{'gen':>4}{'attempts':>10}{'slots':>7}{'invalid':>9}{'promoted':>10}{'best screen':>13}{'best expanded':>15}"
    print(header)
    print("-" * len(header))
    per_generation = {}
    for gen in sorted(by_generation):
        group = by_generation[gen]
        screen = [r.evaluation.get("success_rate") for r in group if r.evaluation.get("success_rate") is not None]
        expanded = [r.evaluation.get("expanded_success_rate") for r in group
                    if r.evaluation.get("expanded_success_rate") is not None]
        row = {
            "attempts": len(group),
            "slots": len({r.candidate_id for r in group}),
            "invalid": sum(1 for r in group if r.status == STATUS_INVALID),
            "promoted": sum(1 for r in group if r.status == STATUS_PROMOTED),
            "best_screen_success": max(screen) if screen else None,
            "best_expanded_success": max(expanded) if expanded else None,
        }
        per_generation[str(gen)] = row
        fmt = lambda v, w: f"{v:>{w}.3f}" if v is not None else f"{'n/a':>{w}}"  # noqa: E731
        print(f"{gen:>4}{row['attempts']:>10}{row['slots']:>7}{row['invalid']:>9}{row['promoted']:>10}"
              f"{fmt(row['best_screen_success'], 13)}{fmt(row['best_expanded_success'], 15)}")

    separator("Traceability (AC-4, debug_spec.md D11)")
    traceability = traceability_summary(generated, args.log_dir, verify_reconstruction=not args.no_reconstruct)
    for name, field in traceability.items():
        if field["complete"]:
            print(f"  {name:<16} complete")
        else:
            missing = sorted(field["gaps"])
            shown = ", ".join(missing[:4]) + (f" and {len(missing) - 4} more" if len(missing) > 4 else "")
            print(f"  {name:<16} INCOMPLETE on {len(missing)} record(s): {shown}")
    if args.no_reconstruct:
        print("  (reconstruction not re-run; --no-reconstruct)")

    reconciliation = reconcile_resources(generated, ledger, args.log_dir)
    separator("Resources")
    if ledger is None:
        print("  no LLM-call ledger in this run; token totals cannot be reconciled")
    else:
        llm = reconciliation["llm"]
        print(f"  LLM calls {llm['ledger_calls']}, tokens {llm['ledger_total_tokens']} "
              f"({'reconciles' if llm['reconciles'] else 'DOES NOT reconcile'} on unique call ids)")
    streams = reconciliation["episodes"]["from_records"]
    for stream, counts in streams.items():
        print(f"  {stream:<16} episodes {counts['episodes']:>6}   simulator steps {counts['environment_steps']:>8}")
    print(f"  gradient steps   {reconciliation['bc_gradient_steps_total']}")
    if "reconciles" in reconciliation["episodes"]:
        print(f"  screen/expanded episodes {'reconcile' if reconciliation['episodes']['reconciles'] else 'DO NOT reconcile'} "
              f"with the saved reports")

    separator("Interventions")
    if not interventions:
        print("  none recorded; nothing is excluded from fitness ranking")
    else:
        for r in interventions:
            print(f"  {r.candidate_id}: {r.intervention_detail}")

    separator("Failures preserved as evidence")
    for r in generated:
        if r.status != STATUS_INVALID:
            continue
        error = str((r.validation.get("errors") or [""])[0]).splitlines()
        print(f"  {r.experiment_id:<24} {r.validation.get('stage', '')}: {error[0][:80] if error else ''}")

    lanes_summary = None
    if any((r.lineage or {}).get("lane") for r in generated):
        from lares.search.lanes import lane_resource_totals

        lanes_summary = lane_resource_totals(generated, ledger)
        separator("Resources by lane (two-lane search)")
        for lane in ("fittable", "exploration"):
            t = lanes_summary[lane]
            print(f"  {lane:<12} slots {t['slots']}  gradient steps {t['bc_gradient_steps']}  probe episodes "
                  f"{t['probe_episodes']}  screen episodes {t['screen_episodes']}  promotion episodes "
                  f"{t['expanded_episodes']}  simulator steps {t['simulator_steps']}  LLM calls {t['llm_calls']} "
                  f"({t['llm_total_tokens']} tokens)")
        if lanes_summary["calls_crossing_lanes"]:
            print(f"  WARNING: calls referenced by both lanes: {lanes_summary['calls_crossing_lanes']}")

    summary = {
        "records_dir": records_dir,
        "records": len(records),
        "attempts": len(generated),
        "slots": len(slots),
        "statuses": dict(statuses),
        "lineage_status": {name: lineage.get(name, 0) for name in LINEAGE_STATUSES},
        "validated_attempts": validated_attempts,
        "validated_slots": validated_slots,
        "slots_valid_within_one_correction": within_one,
        "corrections": corrections,
        "per_generation": per_generation,
        "traceability": traceability,
        "reconstruction_rerun": not args.no_reconstruct,
        "resources": reconciliation,
        "interventions": [r.candidate_id for r in interventions],
        "sidecar_version": sidecar.get("sidecar_version") if sidecar else None,
        "lanes": lanes_summary,
    }
    out = args.out or os.path.join(args.log_dir, "experiment_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  summary: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
