"""What a saved record can actually support (``debug_spec.md`` Stage C and section 6).

A summary used to call traceability "complete" when three fields were present,
while lineage, prompts, states and resources could all be missing. This module
answers the stronger questions from the files alone:

* ``lineage_status``: complete, partial_recovery, legacy_missing or
  failed_validation, honouring a versioned provenance sidecar for old runs.
* ``traceability_gaps``: per evidence field, why it is not complete.
* ``reconstruct_actions`` / ``reconstruct_record``: rebuild the policy from the
  record's hashed source and selected state and compare its actions exactly.
* ``reconcile_resources``: run totals against unique LLM call ids and against the
  episodes the saved reports contain.
"""

from __future__ import annotations

import glob
import io
import json
import os

from lares.search.generation import LLMCallLedger, verify_artifact
from lares.utils.hashing import sha256_bytes
from lares.search.resources import EPISODE_STREAMS, resource_problems
from lares.search.schemas import (
    EVALUATED_STATUSES,
    LINEAGE_COMPLETE,
    LINEAGE_FAILED_VALIDATION,
    LINEAGE_LEGACY_MISSING,
    LINEAGE_STATUSES,
    ORIGIN_GENERATED,
    RECONSTRUCTION_VERIFIED,
    STATUS_INVALID,
)

TRACEABILITY_FIELDS = (
    "lineage",
    "feedback",
    "prompt_hashes",
    "llm_usage",
    "state_artifacts",
    "manifests",
    "resources",
    "reconstruction",
)

SIDECAR_PREFIX = "provenance_sidecar.v"

#: Which resource stream each saved report's episodes belong to.
REPORT_STREAMS = {
    "zero_shot": "screen",
    "intermediate": "screen",
    "fitted_screen": "screen",
    "fitted_expanded": "expanded",
    # Two-lane search: the fittable challenger's step-0 state, scored on the promotion set.
    "zero_shot_expanded": "expanded",
}


# ---------------------------------------------------------------------------
#  Lineage status
# ---------------------------------------------------------------------------


def load_sidecar(log_dir: str) -> dict:
    """The newest ``provenance_sidecar.v<N>.json`` in ``log_dir``, or ``{}``."""
    paths = glob.glob(os.path.join(log_dir, f"{SIDECAR_PREFIX}*.json"))
    if not paths:
        return {}

    def version(path):
        stem = os.path.basename(path)[len(SIDECAR_PREFIX):-len(".json")]
        return int(stem) if stem.isdigit() else -1

    with open(max(paths, key=version), "r", encoding="utf-8") as f:
        return json.load(f)


def lineage_status(record, sidecar: dict | None = None) -> str:
    """The lineage completeness of one record. A sidecar entry takes precedence."""
    entry = ((sidecar or {}).get("records") or {}).get(record.experiment_id)
    if entry and entry.get("lineage_status") in LINEAGE_STATUSES:
        return entry["lineage_status"]
    if record.origin != ORIGIN_GENERATED or not record.lineage:
        return LINEAGE_LEGACY_MISSING
    if record.status == STATUS_INVALID or not record.validation.get("passed"):
        return LINEAGE_FAILED_VALIDATION
    status = record.lineage.get("status")
    return status if status in LINEAGE_STATUSES else LINEAGE_LEGACY_MISSING


# ---------------------------------------------------------------------------
#  Reconstruction
# ---------------------------------------------------------------------------


def _read_artifact_bytes(run_root: str, ref: dict) -> bytes:
    """The artifact's bytes, verified against its recorded hash in one read."""
    problem = verify_artifact(run_root, ref, read=False)
    if problem:
        raise ValueError(problem)
    with open(os.path.join(run_root, ref["path"]), "rb") as f:
        data = f.read()
    actual = sha256_bytes(data)
    if actual != ref["sha256"]:
        raise ValueError(f"{ref['path']!r} hash {actual[:12]} does not match recorded {ref['sha256'][:12]}")
    return data


def load_tensor_artifact(run_root: str, ref: dict):
    import torch

    return torch.load(io.BytesIO(_read_artifact_bytes(run_root, ref)), weights_only=True)


def rebuild_policy(run_root: str, reconstruction: dict):
    """Build the policy from the record's source and load its selected state."""
    from lares.core.obs_schema import active_schema, get_obs_schema
    from lares.core.policy_generation import GENERATED_IMPORTS

    source = _read_artifact_bytes(run_root, reconstruction["source"]).decode("utf-8")
    namespace: dict = {}
    exec(GENERATED_IMPORTS + source, namespace)
    if "GeneratedPolicy" not in namespace:
        raise ValueError("the recorded source defines no GeneratedPolicy")
    schema = get_obs_schema(reconstruction["env_id"])
    with active_schema(schema):
        policy = namespace["GeneratedPolicy"](int(reconstruction["obs_dim"]), int(reconstruction["action_dim"]))
    policy.fixed_action_scale = float(reconstruction["fixed_action_scale"])
    policy._generated_source = source
    policy.load_state_dict(load_tensor_artifact(run_root, reconstruction["selected_state"]))
    policy.eval()
    return policy


def reconstruct_actions(run_root: str, reconstruction: dict):
    """``tanh(mean)`` of the rebuilt policy on the recorded probe batch."""
    import torch

    policy = rebuild_policy(run_root, reconstruction)
    probe = load_tensor_artifact(run_root, reconstruction["probe_batch"])
    with torch.no_grad():
        return torch.tanh(policy(probe)[0]).detach().clone()


def reconstruct_record(record, run_root: str) -> dict:
    """Rebuild from a record alone and compare against its stored expected actions."""
    import torch

    recon = record.reconstruction or {}
    missing = [k for k in ("source", "selected_state", "probe_batch", "expected_actions") if not recon.get(k)]
    if missing:
        return {"identical": False, "problems": [f"reconstruction lacks {missing}"]}
    try:
        actual = reconstruct_actions(run_root, recon)
        expected = load_tensor_artifact(run_root, recon["expected_actions"])
    except Exception as exc:
        return {"identical": False, "problems": [f"{type(exc).__name__}: {exc}"]}
    identical = bool(actual.shape == expected.shape and torch.equal(actual, expected))
    diff = float((actual - expected).abs().max()) if actual.shape == expected.shape and actual.numel() else None
    return {"identical": identical, "max_abs_diff": diff, "problems": [] if identical else ["actions differ"]}


# ---------------------------------------------------------------------------
#  Traceability
# ---------------------------------------------------------------------------


def _feedback_problems(record, run_root: str) -> list:
    from lares.search.generation import FeedbackBundle

    if record.generation == 0:
        return []
    feedback_id = (record.lineage or {}).get("feedback_id")
    if not feedback_id:
        return ["no feedback id"]
    path = os.path.join(run_root, f"gen_{record.generation - 1}", "feedback_bundle.json")
    if not os.path.isfile(path):
        return [f"feedback bundle {os.path.relpath(path, run_root)} is missing"]
    with open(path, "r", encoding="utf-8") as f:
        bundle = FeedbackBundle.from_dict(json.load(f))
    problems = []
    if bundle.feedback_id != feedback_id:
        problems.append(f"record names {feedback_id}, the bundle on disk is {bundle.feedback_id}")
    try:
        bundle.validate(run_root)
    except Exception as exc:
        problems.append(str(exc))
    if list(bundle.context_parent_ids) != list(record.lineage.get("context_parent_ids") or []):
        problems.append("context parents differ from the bundle on disk")
    return problems


def traceability_gaps(record, run_root: str, ledger: LLMCallLedger | None = None,
                      verify_reconstruction: bool = True) -> dict:
    """``{field: [reasons]}``; an empty list means that field is complete for this record."""
    gaps = {name: [] for name in TRACEABILITY_FIELDS}
    if record.origin != ORIGIN_GENERATED or not record.lineage:
        for name in TRACEABILITY_FIELDS:
            gaps[name].append(f"{record.origin} record carries no generated-candidate provenance")
        return gaps

    evaluated = record.status in EVALUATED_STATUSES
    lineage = record.lineage
    if evaluated:
        if lineage.get("status") != LINEAGE_COMPLETE:
            gaps["lineage"].append(f"lineage status is {lineage.get('status')}")
        gaps["lineage"].extend(lineage.get("problems") or [])
        if not record.idea_id:
            gaps["lineage"].append("no idea id")
        if record.generation > 0 and not record.proposal_id:
            gaps["lineage"].append("no proposal id")
    for key in ("slot", "attempt", "attempt_id"):
        if lineage.get(key) is None:
            gaps["lineage"].append(f"no {key}")
    if evaluated or record.validation.get("stage") not in ("ideation", "budget_exhausted"):
        gaps["feedback"].extend(_feedback_problems(record, run_root))

    ledger = ledger if ledger is not None else LLMCallLedger(run_root)
    ledger_problems = ledger.verify()
    if not record.llm_call_ids:
        gaps["prompt_hashes"].append("no LLM call ids")
    embedded = {c.get("call_id"): c for c in (record.llm.get("calls") or [])}
    for call_id in record.llm_call_ids:
        if call_id not in ledger:
            gaps["prompt_hashes"].append(f"{call_id} is not in the run ledger")
            gaps["llm_usage"].append(f"{call_id} usage unavailable")
            continue
        call = ledger.get(call_id)
        gaps["prompt_hashes"].extend(f"{call_id} {p}" for p in ledger_problems.get(call_id, []))
        ref = embedded.get(call_id) or {}
        if ref.get("prompt_sha256") != call.prompt_sha256 or ref.get("response_sha256") != call.response_sha256:
            gaps["prompt_hashes"].append(f"{call_id} embedded hashes disagree with the ledger")
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if not isinstance(getattr(call, key, None), int):
                gaps["llm_usage"].append(f"{call_id} has no {key}")
    for key in ("llm_calls", "llm_prompt_tokens", "llm_completion_tokens", "llm_total_tokens"):
        if key not in record.resources:
            gaps["llm_usage"].append(f"resources lack {key}")

    gaps["resources"].extend(resource_problems(record.resources))

    if evaluated:
        for key in ("initial_state", "final_state", "selected_state"):
            problem = verify_artifact(run_root, record.fitting.get(key))
            if problem:
                gaps["state_artifacts"].append(f"{key}: {problem}")
        if record.fitting.get("selected_step") is None:
            gaps["state_artifacts"].append("no selected step")
        if not record.evaluation.get("manifest_ids"):
            gaps["manifests"].append("no manifest ids")
        environment = record.evaluation.get("environment") or {}
        if not (environment.get("canonical_env_id") and environment.get("resolved_registry_env_id")):
            gaps["manifests"].append("no canonical/resolved environment ids")
        reports = record.evaluation.get("reports") or {}
        if not reports:
            gaps["manifests"].append("no report references")
        for name, ref in reports.items():
            problem = verify_artifact(run_root, ref)
            if problem:
                gaps["manifests"].append(f"report {name}: {problem}")
        recon = record.reconstruction or {}
        if recon.get("status") != RECONSTRUCTION_VERIFIED:
            gaps["reconstruction"].append(f"reconstruction status is {recon.get('status')!r}")
        if verify_reconstruction:
            outcome = reconstruct_record(record, run_root)
            if not outcome["identical"]:
                gaps["reconstruction"].extend(outcome["problems"])
    return gaps


def traceability_summary(records, run_root: str, verify_reconstruction: bool = True) -> dict:
    """Per field: complete only if *no* record has a gap in it."""
    ledger = LLMCallLedger(run_root)
    per_field = {name: {} for name in TRACEABILITY_FIELDS}
    for record in records:
        for name, reasons in traceability_gaps(record, run_root, ledger, verify_reconstruction).items():
            if reasons:
                per_field[name][record.experiment_id] = reasons
    return {
        name: {"complete": not missing, "gaps": missing}
        for name, missing in per_field.items()
    }


# ---------------------------------------------------------------------------
#  Resource reconciliation
# ---------------------------------------------------------------------------


def reconcile_resources(records, ledger: LLMCallLedger | None, run_root: str | None = None) -> dict:
    """Check run totals against unique call ids and against saved report episodes.

    LLM: the tokens of every *unique* call id any record references must equal the
    ledger's total, and the sum of each record's exclusive usage plus each shared
    call counted once must equal the same number. A shared ideation call is
    therefore counted once, never once per candidate.

    Episodes: screen and expanded episodes and steps summed from records must
    equal what the referenced report files contain. Probe episodes have no saved
    report, so they are reported from records only and say so.
    """
    out: dict = {}
    if ledger is not None:
        referenced = []
        for record in records:
            referenced.extend(record.llm_call_ids or [])
        unique = list(dict.fromkeys(referenced))
        from_records = ledger.totals(unique)
        total = ledger.totals()
        shared = [c for c in unique if c in ledger and ledger.get(c).shared]
        shared_tokens = ledger.totals(shared)["llm_total_tokens"]
        exclusive_tokens = sum(int(r.resources.get("llm_total_tokens", 0) or 0) for r in records)
        out["llm"] = {
            "ledger_calls": total["llm_calls"],
            "ledger_total_tokens": total["llm_total_tokens"],
            "referenced_unique_calls": from_records["llm_calls"],
            "referenced_unique_total_tokens": from_records["llm_total_tokens"],
            "exclusive_plus_shared_total_tokens": exclusive_tokens + shared_tokens,
            "unreferenced_call_ids": [c.call_id for c in ledger.records() if c.call_id not in set(unique)],
            "unknown_call_ids": from_records["unknown_call_ids"],
            "reconciles": (
                from_records["llm_total_tokens"] == total["llm_total_tokens"]
                and exclusive_tokens + shared_tokens == total["llm_total_tokens"]
                and not from_records["unknown_call_ids"]
            ),
        }

    streams = {s: {"episodes": 0, "environment_steps": 0} for s in EPISODE_STREAMS}
    for record in records:
        for s in EPISODE_STREAMS:
            streams[s]["episodes"] += int(record.resources.get(f"{s}_episodes", 0) or 0)
            streams[s]["environment_steps"] += int(record.resources.get(f"{s}_environment_steps", 0) or 0)
    episodes = {"from_records": streams}
    if run_root is not None:
        from_reports = {s: {"episodes": 0, "environment_steps": 0} for s in ("screen", "expanded")}
        unreadable = []
        for record in records:
            for name, ref in ((record.evaluation or {}).get("reports") or {}).items():
                stream = REPORT_STREAMS.get(name)
                if stream is None:
                    continue
                if verify_artifact(run_root, ref):
                    unreadable.append(f"{record.experiment_id}:{name}")
                    continue
                with open(os.path.join(run_root, ref["path"]), "r", encoding="utf-8") as f:
                    report = json.load(f)
                rows = report.get("episodes") or []
                from_reports[stream]["episodes"] += len(rows)
                from_reports[stream]["environment_steps"] += int(sum(int(e.get("length") or 0) for e in rows))
        episodes["from_reports"] = from_reports
        episodes["unreadable_reports"] = unreadable
        episodes["reconciles"] = not unreadable and all(
            streams[s] == from_reports[s] for s in ("screen", "expanded")
        )
        episodes["probe_note"] = "probe episodes have no saved report; counted from records only"
    out["episodes"] = episodes
    out["simulator_steps_total"] = sum(v["environment_steps"] for v in streams.values())
    out["bc_gradient_steps_total"] = sum(int(r.resources.get("bc_gradient_steps", 0) or 0) for r in records)
    return out
