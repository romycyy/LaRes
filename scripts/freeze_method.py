#!/usr/bin/env python
"""Freeze the method before final testing (``spec.md`` Phase 7 step 1, AC-7).

Digests everything a final run must not change afterwards: model identifiers,
every prompt template, the structure sources, the committed manifests, the
optimizer settings, the selection rule and the budgets. The record is written
once and refuses to overwrite itself, because a freeze rewritten after a
final-test number has been seen records nothing.

Writing this record is also what unlocks the final-test manifest.
``evaluate_manifest`` refuses that split outside a `final_test_session`, so a
search cannot look at held-out data and carry on as though it had not.

Usage (from the project root)::

    python scripts/freeze_method.py --freeze-id v1
    python scripts/freeze_method.py --verify            # report drift since the freeze
"""

import argparse
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import yaml  # noqa: E402

from lares.eval.freeze import (  # noqa: E402
    build_freeze,
    hash_directory,
    load_freeze,
)

PROMPT_DIR = os.path.join(_PROJECT_ROOT, "lares", "utils", "policy_prompts")
MANIFEST_DIR = os.path.join(_PROJECT_ROOT, "config", "manifests")
STRUCTURE_DIR = os.path.join(_PROJECT_ROOT, "lares", "fitting", "structures")
CONFIG = os.path.join(_PROJECT_ROOT, "config", "run_full_evolution.yaml")

#: The modules a final run must not change under it. Listed rather than globbed
#: so adding a module is a deliberate act recorded in this file.
FROZEN_MODULES = (
    "lares/core/symbolic_policy.py",
    "lares/core/policy_validator.py",
    "lares/core/obs_schema.py",
    "lares/core/policy_generation.py",
    "lares/core/training_pipeline.py",
    "lares/eval/runner.py",
    "lares/eval/promotion.py",
    "lares/eval/diagnostics.py",
    "lares/fitting/optimizers.py",
    "lares/fitting/objectives.py",
    "lares/repair/repair.py",
    "lares/repair/hypotheses.py",
    "lares/search/archive.py",
    "lares/search/feedback.py",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--freeze-id", default="v1")
    p.add_argument("--out-dir", default=os.path.join(_PROJECT_ROOT, "logs", "final"))
    p.add_argument("--config", default=CONFIG)
    p.add_argument("--verify", action="store_true",
                   help="compare what is on disk against an existing freeze")
    p.add_argument("--notes", default="")
    return p.parse_args()


def current_code_hashes():
    from lares.eval.freeze import hash_file

    code = {m: hash_file(os.path.join(_PROJECT_ROOT, m)) for m in FROZEN_MODULES}
    code.update({f"structures/{k}": v for k, v in hash_directory(STRUCTURE_DIR, ".py").items()})
    return code


def main():
    args = parse_args()

    if args.verify:
        record = load_freeze(args.out_dir)
        report = record.verify(
            prompts=hash_directory(PROMPT_DIR, ".txt"),
            code=current_code_hashes(),
            manifests=hash_directory(MANIFEST_DIR, ".yaml"),
        )
        print(f"freeze {record.freeze_id}, written {record.created_at}")
        if report["matches"]:
            print("  nothing has drifted since the freeze")
            return 0
        for section, detail in report["drift"].items():
            print(f"  {section}:")
            for kind in ("changed", "missing", "added"):
                if detail[kind]:
                    print(f"    {kind}: {detail[kind]}")
        print("\n  A final-test number measured after this drift is not a number about the "
              "frozen method.")
        return 1

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    from lares.eval.promotion import PromotionPolicy
    from lares.fitting.benchmark import SELECTION_RULE

    policy = PromotionPolicy()
    record = build_freeze(
        freeze_id=args.freeze_id,
        models={
            "generation_model": cfg.get("model", "unset"),
            "generation_temperature": cfg.get("temperature", "unset"),
            "two_phase": cfg.get("policy_gen_two_phase", False),
            "impl_mode": cfg.get("policy_impl_mode", "unset"),
        },
        prompt_dir=PROMPT_DIR,
        manifest_dir=MANIFEST_DIR,
        code_files={m: os.path.join(_PROJECT_ROOT, m) for m in FROZEN_MODULES},
        optimizer={
            "objective": "deterministic_mse",
            "method": "adam_baseline",
            "bc_steps": cfg.get("bc_steps", "unset"),
            "batch_size": cfg.get("bc_batch_size", "unset"),
            "learning_rate": cfg.get("bc_lr", "unset"),
            "checkpoint_rule": "best rollout probe, then final",
            "rollout_checkpoint_probes": cfg.get("rollout_checkpoint_probes", 4),
            "benchmark_selection_rule": SELECTION_RULE,
        },
        selection={
            "screen_episodes": policy.screen_episodes,
            "expanded_episodes": policy.expanded_episodes,
            "fallback_top_k": policy.fallback_top_k,
            "promotion": "paired success difference against the incumbent",
            "tie_break": "mean signed goal progress",
        },
        budgets={
            "generations": cfg.get("num_generations", "unset"),
            "population": cfg.get("num_policies", "unset"),
            "max_repair_per_candidate": cfg.get("max_repair_per_candidate", "unset"),
            "final_test_episodes": 100,
            "independent_searches": 5,
        },
        notes=args.notes,
    )
    code = current_code_hashes()
    record.code = code
    path = record.save(args.out_dir)
    print(f"froze {len(record.prompts)} prompts, {len(record.code)} source files, "
          f"{len(record.manifests)} manifests")
    print(f"  written: {path}")
    print("  The final-test manifest is now unlockable, one session at a time, "
          "and every look at it is appended to the ledger beside this record.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
