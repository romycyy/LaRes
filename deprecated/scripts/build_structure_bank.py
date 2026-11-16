#!/usr/bin/env python
"""Freeze a bank of symbolic-policy structures for the fitting comparison (Phase 3).

Reads the candidates an earlier search produced from ``logs/evolution/gen_*/results.pkl``,
rewrites their raw observation slices into named accessors, proves the rewrite
changed nothing, and writes each one to ``lares/fitting/structures/``.

Structures written here are the fixed thing in the optimizer comparison: they
come from what the pipeline actually generated, not from a set chosen to make one
optimizer look good.

Usage (from the project root)::

    python scripts/build_structure_bank.py
    python scripts/build_structure_bank.py --dry-run
"""

import argparse
import glob
import os
import pickle
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from lares.core.obs_schema import get_obs_schema  # noqa: E402
from lares.core.policy_validator import validate_policy  # noqa: E402
from lares.fitting.structure_bank import (  # noqa: E402
    STRUCTURES_DIR,
    check_port_equivalence,
    instantiate_source,
    port_raw_indices,
)

HEADER = '''# Frozen structure for the Phase 3 fitting comparison. Do not edit by hand:
# regenerate with scripts/build_structure_bank.py.
# provenance: {provenance}
# ported: raw observation slices rewritten to named accessors; outputs verified
# identical to the original on random observations at two scales.
'''


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--env-name", default="push-v2")
    p.add_argument("--results-glob", default=os.path.join(_PROJECT_ROOT, "logs", "evolution", "gen_*", "results.pkl"))
    p.add_argument("--out-dir", default=STRUCTURES_DIR)
    p.add_argument("--obs-dim", type=int, default=39)
    p.add_argument("--action-dim", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    schema = get_obs_schema(args.env_name)
    os.makedirs(args.out_dir, exist_ok=True)

    written, skipped = [], []
    seen_sources = set()
    for path in sorted(glob.glob(args.results_glob)):
        generation = os.path.basename(os.path.dirname(path))
        with open(path, "rb") as f:
            payload = pickle.load(f)
        for index, candidate in enumerate(payload.get("candidates", [])):
            code = candidate["code"]
            structure_id = f"{generation}_cand{index}"
            if code.strip() in seen_sources:
                skipped.append((structure_id, "duplicate source"))
                continue
            seen_sources.add(code.strip())

            ported, unported = port_raw_indices(code, schema)
            if unported:
                skipped.append((structure_id, f"unportable slices {sorted(set(unported))}"))
                continue
            try:
                if not check_port_equivalence(
                    code, ported, args.obs_dim, args.action_dim, schema
                ):
                    skipped.append((structure_id, "ported output differs from original"))
                    continue
                policy = instantiate_source(ported, args.obs_dim, args.action_dim, schema)
                report = validate_policy(
                    policy, args.obs_dim, args.action_dim, source=ported, schema=schema
                )
            except Exception as exc:
                skipped.append((structure_id, f"{type(exc).__name__}: {exc}"))
                continue
            if not report.ok:
                skipped.append((structure_id, f"validation: {sorted(report.categories)}"))
                continue

            provenance = (
                f"{generation} candidate {index} of the 2026-08-28 search; "
                f"reported score {candidate.get('score', float('nan')):.2f}"
            )
            body = HEADER.format(provenance=provenance) + "\n" + ported.strip() + "\n"
            out_path = os.path.join(args.out_dir, f"{structure_id}.py")
            if not args.dry_run:
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(body)
            written.append((structure_id, policy.count_parameters(), out_path))

    print(f"\nPorted {len(written)} structures into {args.out_dir}")
    for sid, nparams, _ in written:
        print(f"  {sid:<20} {nparams:>3} parameters")
    if skipped:
        print(f"\nSkipped {len(skipped)}:")
        for sid, why in skipped:
            print(f"  {sid:<20} {why}")
    if args.dry_run:
        print("\n(dry run: nothing written)")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
