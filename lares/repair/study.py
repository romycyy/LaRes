"""Run a diagnostic intervention study on one policy (``spec.md`` FR-9, AC-6).

Takes a fitted policy, finds its recurrent failures, forms competing
explanations for each, tests every one with a controlled substitution on the same
cases, and records the outcome including the explanations the evidence rejected.

Nothing here changes a policy. The study produces evidence for a repair; the
repair itself is a separate, bounded edit whose acceptance is decided by
:func:`~lares.repair.hypotheses.evaluate_repair`.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from lares.eval.report import CHECKPOINT_FITTED, build_report
from lares.eval.runner import PolicyActor, evaluate_manifest, paired_difference
from lares.repair.hypotheses import (
    InterventionOutcome,
    competing_explanations,
    episode_metric_mean,
    find_clusters,
    rank_explanations,
)
from lares.repair.interventions import InterventionActor, available_gates


@dataclass
class ClusterStudy:
    """Everything learned about one recurrent failure."""

    cluster: dict
    explanations: list = field(default_factory=list)
    outcomes: list = field(default_factory=list)
    supported: str | None = None
    rejected: list = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "cluster": self.cluster,
            "explanations": self.explanations,
            "outcomes": [o.to_dict() if hasattr(o, "to_dict") else o for o in self.outcomes],
            "supported": self.supported,
            "rejected": self.rejected,
            "note": self.note,
        }


def run_study(
    policy,
    structure,
    manifest,
    env,
    env_name: str,
    schema,
    max_steps: int = 150,
    max_clusters: int = 2,
    progress=None,
) -> dict:
    """Diagnose a policy's recurrent failures by controlled substitution."""
    started = time.time()
    control_result = evaluate_manifest(
        PolicyActor(policy, deterministic=True, name="control"),
        manifest,
        env,
        max_steps=max_steps,
    )
    control_report = build_report(
        control_result, f"{structure.structure_id}:control", CHECKPOINT_FITTED,
        action_dim=structure.action_dim,
    )
    clusters = find_clusters(control_report)

    probe = torch.as_tensor(
        np.stack([np.zeros(structure.obs_dim, dtype=np.float32)] * 4)
    )
    # Probe on a real observation so a distance gate is not trivially shut.
    obs, _ = env.reset(manifest.episodes[0])
    probe = torch.as_tensor(np.stack([obs] * 4), dtype=torch.float32)
    gates = available_gates(policy, probe)

    studies = []
    rollouts = 1
    for cluster in clusters[:max_clusters]:
        explanations = competing_explanations(cluster, gates=gates)
        study = ClusterStudy(cluster=cluster.to_dict(),
                             explanations=[e.to_dict() for e in explanations])
        if len(explanations) < 2:
            study.note = (
                "fewer than two applicable explanations; a single account is a guess "
                "with a substitution attached, not a diagnosis"
            )
            studies.append(study)
            continue

        for explanation in explanations:
            actor = InterventionActor(
                policy, explanation.substitution, env_name, schema,
                name=f"{structure.structure_id}:{explanation.name}",
            )
            result = evaluate_manifest(actor, manifest, env, max_steps=max_steps)
            rollouts += 1
            if progress is not None:
                progress(cluster.label, explanation.name, result.success_rate)

            before = episode_metric_mean(
                control_result, explanation.focused_metric, cluster.case_ids
            )
            after = episode_metric_mean(
                result, explanation.focused_metric, cluster.case_ids
            )
            study.outcomes.append(
                InterventionOutcome(
                    explanation=explanation.name,
                    substitution=explanation.substitution.to_dict(),
                    focused_metric=explanation.focused_metric,
                    control_value=before,
                    intervened_value=after,
                    check=explanation.check(before, after),
                    control_success=control_result.success_rate,
                    intervened_success=result.success_rate,
                    paired_success=paired_difference(result, control_result, "success"),
                    n_cases=len(cluster.case_ids),
                )
            )

        ranked = rank_explanations(study.outcomes)
        best = ranked[0] if ranked else None
        if best is not None and best.check.get("supported"):
            study.supported = best.explanation
            study.rejected = [
                o.explanation for o in ranked[1:] if o.explanation != best.explanation
            ]
        else:
            study.rejected = [o.explanation for o in ranked]
            study.note = (
                "no explanation was supported: every substitution failed to move its "
                "focused metric, so the fault is elsewhere or is not localised to one "
                "of the components tested"
            )
        studies.append(study)

    policy.clear_gate_overrides()
    return {
        "structure_id": structure.structure_id,
        "manifest_id": manifest.manifest_id,
        "control": {
            "success_rate": control_result.success_rate,
            "mean_return": control_result.mean_reward,
            "failure_labels": control_report.rollout.failure_labels,
        },
        "gates": gates,
        "clusters_found": [c.to_dict() for c in clusters],
        "studies": [s.to_dict() for s in studies],
        "rollout_episodes": rollouts * len(manifest),
        #: Of those, the ones that were expert-assisted or gate-forced. The
        #: control run is not one, and neither is anything a repair spends.
        "intervention_episodes": (rollouts - 1) * len(manifest),
        "wall_time_seconds": time.time() - started,
    }


def run_repair(
    policy,
    structure,
    payload,
    data,
    schema,
    manifest,
    env,
    control_result=None,
    held_out=None,
    operator: str = "search",
    budget: int = 400,
    search_episodes: int = 200,
    population: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
    max_steps: int = 150,
):
    """Generate one bounded repair from a study and validate it (Phase 6, 4-5).

    Two operators. ``refit`` moves the in-scope parameters toward the expert on
    the scope's phases. ``search`` moves them against the focused metric the
    explanation predicted, which is the only one of the two that can correct a
    geometry term the expert data itself produced; it costs rollout episodes and
    those are counted.

    Focused validation first, on the cluster the diagnosis was about, then the
    protected-case regression check. ``held_out`` is scored afterwards and takes
    no part in the decision, so it stays an honest estimate of what the repair
    is worth away from the cases that produced it.
    """
    from lares.repair.hypotheses import Explanation
    from lares.repair.repair import accept_repair, apply_repair, propose_repair

    chosen = None
    for study in payload["studies"]:
        if not study.get("supported"):
            continue
        for explanation in study["explanations"]:
            if explanation["name"] == study["supported"]:
                chosen = (study, Explanation.from_dict(explanation))
                break
        if chosen:
            break
    if chosen is None:
        return {
            "attempted": False,
            "reason": "no explanation was supported, so there is nothing to repair",
        }, None

    study, explanation = chosen
    cluster_case_ids = study["cluster"]["case_ids"]
    repair = propose_repair(policy, explanation, data, schema)
    if repair.is_empty:
        return {
            "attempted": False,
            "reason": f"no parameter is in scope for {explanation.suspected_component}",
            "repair": repair.to_dict(),
        }, None

    episodes = 0
    if operator == "refit":
        repaired = apply_repair(
            policy, repair, data, schema, budget=budget, batch_size=batch_size, lr=lr, seed=seed
        )
    elif operator == "search":
        from lares.repair.repair import search_repair

        repaired, _ = search_repair(
            policy, repair, manifest, env, focused_case_ids=cluster_case_ids,
            budget_episodes=search_episodes, population=population, seed=seed,
            max_steps=max_steps,
        )
        episodes += repair.cost.get("rollout_episodes", 0)
    else:
        raise ValueError(f"operator must be 'refit' or 'search', got {operator!r}")

    if control_result is None:
        control_result = evaluate_manifest(
            PolicyActor(policy, deterministic=True, name="before"), manifest, env,
            max_steps=max_steps,
        )
        episodes += len(manifest)
    after_result = evaluate_manifest(
        PolicyActor(repaired, deterministic=True, name="after"), manifest, env,
        max_steps=max_steps,
    )
    episodes += len(manifest)
    decision = accept_repair(control_result, after_result, repair, cluster_case_ids)

    record = {
        "attempted": True,
        "cluster": study["cluster"]["label"],
        "explanation": explanation.name,
        "statement": explanation.statement,
        "operator": operator,
        "repair": repair.to_dict(),
        "decision": decision.to_dict(),
        "focused_cases": len(cluster_case_ids),
        "success_before": control_result.success_rate,
        "success_after": after_result.success_rate,
        "return_before": control_result.mean_reward,
        "return_after": after_result.mean_reward,
        "paired_success": paired_difference(after_result, control_result, "success"),
        "rollout_episodes": episodes,
    }
    if held_out is not None:
        before_held = evaluate_manifest(
            PolicyActor(policy, deterministic=True, name="before"), held_out, env,
            max_steps=max_steps,
        )
        after_held = evaluate_manifest(
            PolicyActor(repaired, deterministic=True, name="after"), held_out, env,
            max_steps=max_steps,
        )
        record["held_out"] = {
            "manifest_id": held_out.manifest_id,
            "n_cases": len(held_out),
            "success_before": before_held.success_rate,
            "success_after": after_held.success_rate,
            "paired_success": paired_difference(after_held, before_held, "success"),
            "note": "scored after the decision; took no part in it",
        }
        record["rollout_episodes"] += 2 * len(held_out)
    return record, repaired


def records_from_study(payload, structure, generation: int = 0, manifest_ids=None) -> list:
    """Turn a study into experiment records (``spec.md`` FR-10).

    One record per substitution, each marked ``intervention=True`` with the
    substitution spelled out, plus one for the repair if a repair ran. Without
    these the intervention rollouts exist only in a study file, and a search
    summary regenerated from records alone would not know they happened.

    The repair record is *not* an intervention: it is a policy that ran with no
    expert assistance, so it may be ranked. Every intervention record may not.
    """
    from lares.search.schemas import (
        STATUS_EVALUATED,
        ExperimentRecord,
        source_hash,
    )

    code_hash = source_hash(structure.source)
    manifests = list(manifest_ids or [payload.get("manifest_id", "")])
    records = []
    for index, study in enumerate(payload.get("studies", [])):
        for outcome in study.get("outcomes", []):
            substitution = outcome.get("substitution", {})
            records.append(
                ExperimentRecord(
                    experiment_id=f"{payload['structure_id']}:c{index}:{outcome['explanation']}",
                    candidate_id=payload["structure_id"],
                    generation=int(generation),
                    status=STATUS_EVALUATED,
                    code={"source_hash": code_hash, "structure_id": payload["structure_id"]},
                    evaluation={
                        "manifest_ids": manifests,
                        "action_mode": "deterministic",
                        "success_rate": outcome.get("intervened_success"),
                        "focused_metric": outcome.get("focused_metric"),
                        "check": outcome.get("check", {}),
                        "cluster": study.get("cluster", {}).get("label"),
                    },
                    intervention=True,
                    intervention_detail=substitution.get("label", "substitution"),
                    final_disposition=(
                        "supported" if outcome["explanation"] == study.get("supported")
                        else "rejected"
                    ),
                )
            )

    repair = payload.get("repair")
    if repair and repair.get("attempted"):
        decision = repair.get("decision", {})
        records.append(
            ExperimentRecord(
                experiment_id=f"{payload['structure_id']}:repair",
                candidate_id=f"{payload['structure_id']}:repaired",
                generation=int(generation),
                status=STATUS_EVALUATED,
                parent_ids=[payload["structure_id"]],
                code={
                    "source_hash": code_hash,
                    "structure_id": payload["structure_id"],
                    "num_parameters": getattr(structure, "num_parameters", None),
                },
                evaluation={
                    "manifest_ids": manifests,
                    "action_mode": "deterministic",
                    "success_rate": repair.get("success_after"),
                    "mean_return": repair.get("return_after"),
                    "paired_success": repair.get("paired_success"),
                    "held_out": repair.get("held_out"),
                    "repair": repair.get("repair"),
                    "decision": decision,
                },
                final_disposition="accepted" if decision.get("accepted") else "rejected",
            )
        )
    for record in records:
        record.validate()
    return records


def summarise(payload) -> str:
    """The study rendered for a terminal and for a repair prompt."""
    lines = [
        f"Control: success {payload['control']['success_rate']:.3f}, "
        f"labels {payload['control']['failure_labels']}",
    ]
    forceable = [g["gate"] for g in payload["gates"] if g["forceable"]]
    unforceable = [g["gate"] for g in payload["gates"] if not g["forceable"]]
    if payload["gates"]:
        lines.append(f"Gates: forceable {forceable}, telemetry-only {unforceable}")
    for study in payload["studies"]:
        cluster = study["cluster"]
        lines.append("")
        lines.append(f"Cluster {cluster['label']} ({cluster['size']} cases)")
        for key, value in cluster["summary"].items():
            if value is not None:
                lines.append(f"    {key:<26} {value:+.4f}")
        for outcome in study["outcomes"]:
            check = outcome["check"]
            if not check.get("measurable"):
                verdict = "unmeasurable"
                detail = ""
            else:
                verdict = "SUPPORTED" if check["supported"] else "rejected"
                detail = (
                    f"{check['metric']} {check['before']:+.4f} -> {check['after']:+.4f} "
                    f"({check['delta']:+.4f}, wanted {check['predicted_direction']})"
                )
            lines.append(
                f"  {outcome['explanation']:<26} {verdict:<10} "
                f"success {outcome['control_success']:.3f} -> "
                f"{outcome['intervened_success']:.3f}   {detail}"
            )
        if study["supported"]:
            lines.append(f"  => supported: {study['supported']}; rejected {study['rejected']}")
        else:
            lines.append(f"  => {study['note']}")
    lines.append("")
    lines.append(
        "Every row above is expert-assisted or gate-forced. These are diagnoses, "
        "not policy scores, and cannot enter fitness ranking."
    )
    return "\n".join(lines)


def save(payload, directory: str, name: str = "") -> str:
    os.makedirs(directory, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(directory, f"{name or payload['structure_id']}_{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return path
