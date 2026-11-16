#!/usr/bin/env python
"""Write a versioned provenance sidecar for a run recorded before lineage existed.

``debug_spec.md`` section 6. Old raw records are never rewritten: nothing here
may make a record look as if it captured evidence it did not. What *can* be
recovered from other files is written to ``provenance_sidecar.v<N>.json`` next
to the records, with the file hashes it was recovered from and the fields that
remain unknown.

Recovered, where the files support it:

* ``context_parent_ids`` for generation ``g >= 1``: the parents shown in full in
  ``gen_<g-1>/feedback.md`` (``#### Parent N: <id>`` headings). Candidates whose
  diagnostics alone were shown are listed separately.
* The positional idea mapping ``gen<g>_cand<k> -> idea<k>``, only when the
  transcript shows exactly one ``policy_implement_<k>`` call per idea, in order,
  and the generation has one record per idea. Otherwise the mapping is not
  claimed.

Never recovered, because the model never declared them: semantic ``parent_ids``,
``proposal_id``, a structured ``PolicyIdea``, LLM call ids and token usage per
candidate, selected-state artifacts and simulator-step counts.

Usage (from the project root)::

    python scripts/recover_provenance.py --log-dir logs/evolution_review_20260913
"""

import argparse
import glob
import json
import os
import re
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from lares.utils.clock import now_iso  # noqa: E402
from lares.search import (  # noqa: E402
    LINEAGE_LEGACY_MISSING,
    LINEAGE_PARTIAL_RECOVERY,
    load_records,
)
from lares.search.provenance import SIDECAR_PREFIX  # noqa: E402
from lares.utils.hashing import sha256_file, sha256_text  # noqa: E402

UNKNOWN_ALWAYS = [
    "parent_ids (semantic parents were never declared)",
    "proposal_id (no MutationProposal existed)",
    "idea_id as a structured PolicyIdea (ideas were free text)",
    "slot and attempt ids allocated before generation",
    "llm_call_ids, prompt and response hashes per candidate",
    "llm token usage per candidate",
    "initial, final and selected state artifacts",
    "objective configuration and selected checkpoint step",
    "simulator steps (resources.environment_steps held gradient steps)",
    "reconstruction",
]

_CALL = re.compile(r"^call (\d+) — (\S+) — ", re.MULTILINE)
_FULL_PARENT = re.compile(r"^#### Parent \d+: (\S+)\s*$", re.MULTILINE)
_DIAGNOSTIC = re.compile(r"^#### (gen\d+_cand\d+)\s*$", re.MULTILINE)




def transcript_blocks(path):
    """Call labels per generation, split where a new ideation call begins."""
    if not os.path.isfile(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        labels = [m.group(2) for m in _CALL.finditer(f.read())]
    blocks = []
    for label in labels:
        if label.startswith("policy_ideas") and not label.endswith("retry"):
            blocks.append([])
        if blocks:
            blocks[-1].append(label)
    return blocks


def feedback_context(log_dir, generation):
    path = os.path.join(log_dir, f"gen_{generation - 1}", "feedback.md")
    if generation < 1 or not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    full = list(dict.fromkeys(_FULL_PARENT.findall(text)))
    diagnostics = [c for c in dict.fromkeys(_DIAGNOSTIC.findall(text)) if c not in full]
    return {"path": path, "context_parent_ids": full, "diagnostics_only_ids": diagnostics}


def next_version(log_dir):
    versions = []
    for path in glob.glob(os.path.join(log_dir, f"{SIDECAR_PREFIX}*.json")):
        stem = os.path.basename(path)[len(SIDECAR_PREFIX):-len(".json")]
        if stem.isdigit():
            versions.append(int(stem))
    return max(versions, default=0) + 1


def recover(log_dir):
    records = load_records(os.path.join(log_dir, "experiments"))
    if not records:
        raise SystemExit(f"no records under {log_dir}/experiments")
    transcript = os.path.join(log_dir, "llm_evolution_transcript.log")
    blocks = transcript_blocks(transcript)
    sources = {}
    if os.path.isfile(transcript):
        sources[os.path.relpath(transcript, log_dir)] = sha256_file(transcript)

    by_generation = {}
    for r in records:
        by_generation.setdefault(r.generation, []).append(r)

    out = {}
    for generation, group in sorted(by_generation.items()):
        ideas_path = os.path.join(log_dir, f"gen_{generation}", f"Iter_{generation}_ideas.json")
        ideas = None
        if os.path.isfile(ideas_path):
            with open(ideas_path, "r", encoding="utf-8") as f:
                ideas = json.load(f).get("ideas")
            sources[os.path.relpath(ideas_path, log_dir)] = sha256_file(ideas_path)
        labels = blocks[generation] if generation < len(blocks) else []
        implement = [l for l in labels if l.startswith("policy_implement_")]
        expected = [f"policy_implement_{k}" for k in range(len(ideas or []))]
        mapping_ok = bool(ideas) and implement == expected and len(group) == len(ideas)
        mapping_evidence = (
            f"transcript generation block {generation} has implementation calls {implement}; "
            f"{len(ideas or [])} ideas in {os.path.relpath(ideas_path, log_dir)}; {len(group)} records"
        )
        context = feedback_context(log_dir, generation)
        if context:
            sources[os.path.relpath(context["path"], log_dir)] = sha256_file(context["path"])

        for r in sorted(group, key=lambda r: r.candidate_id):
            match = re.fullmatch(rf"gen{generation}_cand(\d+)", r.candidate_id)
            recovered, unknown = {}, list(UNKNOWN_ALWAYS)
            if mapping_ok and match and int(match.group(1)) < len(ideas):
                k = int(match.group(1))
                text = str(ideas[k])
                recovered["idea_index"] = k
                recovered["idea_text_sha256"] = sha256_text(text)
                recovered["idea_text_excerpt"] = text[:240]
                recovered["idea_mapping"] = "positional"
                recovered["idea_mapping_evidence"] = mapping_evidence
            else:
                unknown.append(f"idea mapping (not positional on the evidence: {mapping_evidence})")
            if generation >= 1:
                if context:
                    recovered["context_parent_ids"] = context["context_parent_ids"]
                    recovered["diagnostics_only_ids"] = context["diagnostics_only_ids"]
                    recovered["context_source"] = os.path.relpath(context["path"], log_dir)
                else:
                    unknown.append("context_parent_ids (no feedback.md for the previous generation)")
            status = LINEAGE_PARTIAL_RECOVERY if recovered else LINEAGE_LEGACY_MISSING
            out[r.experiment_id] = {
                "candidate_id": r.candidate_id,
                "generation": generation,
                "lineage_status": status,
                "recovered": recovered,
                "unknown_fields": unknown,
                "raw_record_parent_ids": r.parent_ids,
                "raw_record_idea_id": r.idea_id,
                "raw_record_proposal_id": r.proposal_id,
            }
    return out, sources


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--log-dir", default=os.path.join(_PROJECT_ROOT, "logs", "evolution_review_20260913"))
    args = p.parse_args()
    entries, sources = recover(args.log_dir)
    version = next_version(args.log_dir)
    payload = {
        "sidecar_version": version,
        "created_at": now_iso(),
        "policy": "raw records are not modified; this sidecar holds only what other files support",
        "source_files_sha256": sources,
        "records": entries,
    }
    path = os.path.join(args.log_dir, f"{SIDECAR_PREFIX}{version}.json")
    with open(path, "x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    for experiment_id, entry in entries.items():
        recovered = entry["recovered"]
        print(f"  {experiment_id:<14} {entry['lineage_status']:<17} "
              f"idea={recovered.get('idea_index', '?')}  context={recovered.get('context_parent_ids', '-')}")
    print(f"\n  sidecar: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
