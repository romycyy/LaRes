#!/usr/bin/env python
"""Compare evolution runs on their shared expanded development cases, from saved files only.

Each run's best candidate is the one with the highest expanded success, ties broken by
mean signed goal progress. Runs are compared *paired*, case by case, on the case ids both
expanded reports contain, with a small-sample t interval on the per-case difference
(``debug_spec.md`` Stage E asks not to rely on a normal 1.96 x SE interval).

Read the result with the selection effect in mind: a best-of-N on the same cases it was
selected on is optimistic, more so for the run with the larger N, and the number of
expanded candidates per run is printed for that reason. Development cases only; this
script never opens the final-test split.

Usage (from the project root)::

    python scripts/compare_evolution_runs.py --baseline logs/evolution_review_20260913 \\
        --run logs/evolution_debug_full
"""

import argparse
import json
import math
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402

from lares.search import STATUS_INVALID, load_records  # noqa: E402
from lares.search.provenance import lineage_status, load_sidecar  # noqa: E402


def expanded_report_path(run_dir, record):
    ref = (record.evaluation.get("reports") or {}).get("fitted_expanded")
    if ref:
        return os.path.join(run_dir, ref["path"])
    legacy = os.path.join(run_dir, f"gen_{record.generation}", "reports", f"{record.candidate_id}_expanded.json")
    return legacy if os.path.isfile(legacy) else None


def run_candidates(run_dir):
    records = [r for r in load_records(os.path.join(run_dir, "experiments")) if not r.intervention]
    sidecar = load_sidecar(run_dir)
    out = []
    for r in records:
        if r.status == STATUS_INVALID:
            continue
        path = expanded_report_path(run_dir, r)
        report = None
        if path and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                report = json.load(f)
        episodes = (report or {}).get("episodes") or []
        progress = [e["signed_goal_progress"] for e in episodes if e.get("signed_goal_progress") is not None]
        out.append({
            "experiment_id": r.experiment_id,
            "candidate_id": r.candidate_id,
            "generation": r.generation,
            "num_parameters": r.code.get("num_parameters"),
            "lineage_status": lineage_status(r, sidecar),
            "mode": (r.lineage or {}).get("mode") or "",
            "parent_ids": r.parent_ids,
            "screen_success": r.evaluation.get("success_rate"),
            "zero_shot_screen_success": (r.evaluation.get("checkpoints") or {}).get("zero_shot"),
            "selected_step": r.fitting.get("selected_step"),
            "expanded_manifest_id": (report or {}).get("manifest_id"),
            "expanded_success": float(np.mean([e["success"] for e in episodes])) if episodes else None,
            "expanded_progress": float(np.mean(progress)) if progress else None,
            "per_case": {e["case_id"]: (float(e["success"]), e.get("signed_goal_progress")) for e in episodes},
        })
    return records, out


def best(candidates):
    scored = [c for c in candidates if c["expanded_success"] is not None]
    if not scored:
        return None
    return max(scored, key=lambda c: (c["expanded_success"], c["expanded_progress"] or -math.inf))


def t_interval(diffs, level=0.95):
    from scipy import stats

    arr = np.asarray(diffs, dtype=float)
    n = arr.size
    if n < 2:
        return None
    se = arr.std(ddof=1) / math.sqrt(n)
    half = stats.t.ppf(0.5 + level / 2, n - 1) * se
    return [float(arr.mean() - half), float(arr.mean() + half)]


def paired(a, b):
    shared = sorted(set(a["per_case"]) & set(b["per_case"]))
    success = [a["per_case"][c][0] - b["per_case"][c][0] for c in shared]
    progress = [
        a["per_case"][c][1] - b["per_case"][c][1]
        for c in shared
        if a["per_case"][c][1] is not None and b["per_case"][c][1] is not None
    ]
    return {
        "n_shared_cases": len(shared),
        "success_mean_difference": float(np.mean(success)) if success else None,
        "success_t95": t_interval(success),
        "wins": sum(1 for d in success if d > 0),
        "losses": sum(1 for d in success if d < 0),
        "ties": sum(1 for d in success if d == 0),
        "progress_mean_difference": float(np.mean(progress)) if progress else None,
        "progress_t95": t_interval(progress),
    }


def summarise(run_dir):
    records, candidates = run_candidates(run_dir)
    slots = {(r.generation, r.candidate_id) for r in records}
    by_gen = {}
    for c in candidates:
        by_gen.setdefault(c["generation"], []).append(c)
    return {
        "run_dir": run_dir,
        "attempts": len(records),
        "slots": len(slots),
        "evaluated": len(candidates),
        "expanded": sum(1 for c in candidates if c["expanded_success"] is not None),
        "screen_success_mean": float(np.mean([c["screen_success"] for c in candidates])) if candidates else None,
        "candidates_with_screen_success": sum(1 for c in candidates if (c["screen_success"] or 0) > 0),
        "mean_parameters": float(np.mean([c["num_parameters"] for c in candidates if c["num_parameters"]])) if candidates else None,
        "best": best(candidates),
        "best_by_generation": {str(g): best(cs) for g, cs in sorted(by_gen.items())},
        "candidates": candidates,
    }


def strip(c):
    return None if c is None else {k: v for k, v in c.items() if k != "per_case"}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--baseline", required=True)
    p.add_argument("--run", required=True, action="append", help="Repeatable.")
    p.add_argument("--out", default="")
    args = p.parse_args()

    base = summarise(args.baseline)
    result = {"baseline": {**base, "best": strip(base["best"]), "candidates": [strip(c) for c in base["candidates"]],
                           "best_by_generation": {g: strip(c) for g, c in base["best_by_generation"].items()}},
              "runs": []}

    def line(label, s):
        b = s["best"]
        print(f"  {label:<44} attempts {s['attempts']:>3}  slots {s['slots']:>3}  evaluated {s['evaluated']:>3}  "
              f"expanded {s['expanded']:>3}  mean screen {s['screen_success_mean'] or 0:.3f}  "
              f"mean params {s['mean_parameters'] or 0:.1f}")
        if b:
            print(f"      best {b['experiment_id']} (gen {b['generation']}, {b['num_parameters']} params): expanded success "
                  f"{b['expanded_success']:.3f}, progress {b['expanded_progress'] or 0:+.4f} on {b['expanded_manifest_id']}")
        for g, c in s["best_by_generation"].items():
            if c:
                print(f"      gen {g} best {c['experiment_id']}: expanded {c['expanded_success']:.3f}")

    print("\nBaseline")
    line(args.baseline, base)
    for run_dir in args.run:
        s = summarise(run_dir)
        print("\nRun")
        line(run_dir, s)
        comparison = paired(s["best"], base["best"]) if s["best"] and base["best"] else None
        if comparison:
            ci = comparison["success_t95"]
            print(f"      paired vs baseline best on {comparison['n_shared_cases']} shared cases: success "
                  f"{comparison['success_mean_difference']:+.3f}"
                  + (f" t95 [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "")
                  + f"  wins/losses/ties {comparison['wins']}/{comparison['losses']}/{comparison['ties']}")
        result["runs"].append({
            **s, "best": strip(s["best"]), "candidates": [strip(c) for c in s["candidates"]],
            "best_by_generation": {g: strip(c) for g, c in s["best_by_generation"].items()},
            "paired_best_vs_baseline_best": comparison,
        })
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n  written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
