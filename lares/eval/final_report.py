"""The aggregate report, assembled from saved artifacts only (AC-7).

AC-7's last requirement is that one documented command regenerates the aggregate
report from saved experiment artifacts. This module is the assembly; nothing
here runs a policy, opens a simulator or calls a model. If a quantity is not in
a file on disk it is named in ``unavailable`` and left out, following the same
rule ``EvaluationReport`` follows: an unmeasured number reported as zero is
worse than an absent one, because it looks measured.

The report is deliberately assembled from four independent artifact streams, so
a claim can be traced back to the run that produced it:

``experiments/``      one ``ExperimentRecord`` per attempted candidate
``repair_studies/``   diagnostic studies and the repairs they produced
``repair_ablation/``  the matched-cost arm comparison
``final/``            the freeze record and the final-test look-ledger
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import asdict, dataclass, field

import numpy as np

#: Quantities AC-7 requires in the final report. Each maps to the artifact that
#: would carry it, so a missing one names where to add instrumentation rather
#: than merely reading "unavailable".
REQUIRED_QUANTITIES = {
    "success": "experiment records: evaluation.success_rate",
    "return": "experiment records: evaluation.mean_return",
    "final_distance": "evaluation reports: rollout.final_goal_distance",
    "invalid_program_rate": "experiment records: status",
    "repair_acceptance_rate": "repair studies: repair.decision.accepted",
    "complexity": "experiment records: code.num_parameters",
    "deterministic_performance": "experiment records: evaluation.action_mode",
    "sampled_performance": "experiment records: evaluation.action_mode",
    "simulator_steps": "experiment records: resources.environment_steps",
    "expert_labels": "aggregation stats: expert_queries",
    "numerical_evaluations": "experiment records: fitting.steps",
    "llm_calls": "experiment records: llm.calls",
    "llm_tokens": "experiment records: llm.tokens",
    "wall_time": "experiment records: resources.wall_time_seconds",
    "inference_latency": "experiment records: resources.inference_latency_seconds",
}


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def load_records(directory: str) -> list:
    """Every experiment record under ``directory``, newest last."""
    from lares.search.schemas import ExperimentRecord

    records = []
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        with open(path, encoding="utf-8") as f:
            records.append(ExperimentRecord.from_dict(json.load(f)))
    return records


def load_json_dir(directory: str) -> list:
    out = []
    for path in sorted(glob.glob(os.path.join(directory, "*.json"))):
        with open(path, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


@dataclass
class FinalReport:
    """What the saved artifacts actually support saying."""

    generated_at: str = ""
    sources: dict = field(default_factory=dict)
    search: dict = field(default_factory=dict)
    validity: dict = field(default_factory=dict)
    repair: dict = field(default_factory=dict)
    ablation: dict = field(default_factory=dict)
    final_test: dict = field(default_factory=dict)
    resources: dict = field(default_factory=dict)
    #: Quantities AC-7 asks for that no artifact carries, with where they belong.
    unavailable: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def search_section(records) -> dict:
    """Candidate outcomes, over every attempted candidate including failures."""
    from lares.search.schemas import STATUS_EVALUATED, STATUS_PROMOTED

    rankable = [r for r in records if r.eligible_for_ranking]
    scored = [r for r in rankable if r.status in (STATUS_EVALUATED, STATUS_PROMOTED)]
    successes = [r.evaluation.get("success_rate") for r in scored]
    return {
        "records": len(records),
        "rankable": len(rankable),
        "scored": len(scored),
        "interventions": sum(1 for r in records if r.intervention),
        "mean_success": _mean(successes),
        "best_success": max([s for s in successes if s is not None], default=None),
        "mean_return": _mean([r.evaluation.get("mean_return") for r in scored]),
        "mean_parameters": _mean([r.code.get("num_parameters") for r in scored]),
        "action_modes": sorted({r.evaluation.get("action_mode") for r in scored if r.evaluation.get("action_mode")}),
        "generations": sorted({r.generation for r in records}),
    }


def validity_section(records) -> dict:
    """Invalid-program rate, over every candidate the generator produced."""
    from lares.search.schemas import STATUS_INVALID

    attempted = [r for r in records if not r.intervention]
    invalid = [r for r in attempted if r.status == STATUS_INVALID]
    return {
        "attempted": len(attempted),
        "invalid": len(invalid),
        "invalid_program_rate": (len(invalid) / len(attempted)) if attempted else None,
        "error_categories": sorted({
            e.get("category", "unknown") if isinstance(e, dict) else str(e)
            for r in invalid for e in r.validation.get("errors", [])
        }),
    }


def repair_section(studies) -> dict:
    """Repair acceptance, and how many explanations the evidence rejected."""
    attempted = [s for s in studies if s.get("repair", {}).get("attempted")]
    accepted = [s for s in attempted if s["repair"]["decision"].get("accepted")]
    outcomes = [o for s in studies for st in s.get("studies", []) for o in st.get("outcomes", [])]
    supported = [o for o in outcomes if o.get("check", {}).get("supported")]
    return {
        "studies": len(studies),
        "clusters_targeted": sum(len(s.get("studies", [])) for s in studies),
        "explanations_tested": len(outcomes),
        "explanations_supported": len(supported),
        "explanations_rejected": len(outcomes) - len(supported),
        "repairs_attempted": len(attempted),
        "repairs_accepted": len(accepted),
        "repair_acceptance_rate": (len(accepted) / len(attempted)) if attempted else None,
        "intervention_episodes": sum(s.get("intervention_episodes", 0) for s in studies),
    }


def ablation_section(runs, budget=None) -> dict:
    """Per-arm held-out means, for runs that share a budget.

    Arms are only comparable at a matched cost, so runs at different budgets are
    never pooled. When several budgets are present the largest is used and the
    others are named, rather than averaged into a number that describes no
    experiment.
    """
    budgets = sorted({int(r.get("budget_screen_episodes", 0)) for r in runs})
    if not runs:
        return {"runs": 0}
    chosen = int(budget) if budget is not None else (budgets[-1] if budgets else 0)
    kept = [r for r in runs if int(r.get("budget_screen_episodes", 0)) == chosen]
    rows = [row for run in kept for row in run.get("rows", [])]
    if not rows:
        return {"runs": 0, "budgets_present": budgets}
    arms = {}
    for row in rows:
        arms.setdefault(row["arm"], []).append(row)
    return {
        "runs": len(kept),
        "budget_screen_episodes": chosen,
        "budgets_present": budgets,
        "budgets_excluded": [b for b in budgets if b != chosen],
        "replicates": len(rows),
        "arms": {
            arm: {
                "replicates": len(arm_rows),
                "held_out_success": _mean([r["held_out_success"] for r in arm_rows]),
                "start_success": _mean([r["baseline_success"] for r in arm_rows]),
                "screen_episodes": _mean([r["cost"]["screen_episodes"] for r in arm_rows]),
                "adam_steps": _mean([r["cost"]["adam_steps"] for r in arm_rows]),
                "expert_assisted_episodes": _mean(
                    [r["cost"]["expert_assisted_episodes"] for r in arm_rows]
                ),
            }
            for arm, arm_rows in sorted(arms.items())
        },
    }


def final_test_section(directory: str) -> dict:
    """The freeze, and every look at the held-out data."""
    from lares.eval.freeze import FREEZE_FILENAME, LEDGER_FILENAME, ledger_entries

    freeze_path = os.path.join(directory, FREEZE_FILENAME)
    entries = ledger_entries(os.path.join(directory, LEDGER_FILENAME))
    section = {
        "frozen": os.path.isfile(freeze_path),
        "looks": len(entries),
        "distinct_candidates": len({e["key"] for e in entries}),
        "rescores": sum(1 for e in entries if e.get("rescore")),
        "entries": entries,
    }
    if os.path.isfile(freeze_path):
        with open(freeze_path, encoding="utf-8") as f:
            freeze = json.load(f)
        section["freeze_id"] = freeze.get("freeze_id")
        section["frozen_at"] = freeze.get("created_at")
    if not entries:
        section["note"] = (
            "the final-test split has never been evaluated, which is the correct "
            "state before the method is frozen and Phase 7 begins"
        )
    return section


def resource_section(records, studies, ablations) -> dict:
    return {
        "simulator_steps_recorded": sum(
            int(r.resources.get("environment_steps") or 0) for r in records
        ),
        "numerical_evaluations": sum(int(r.fitting.get("steps") or 0) for r in records),
        "intervention_episodes": sum(s.get("intervention_episodes", 0) for s in studies),
        "ablation_screen_episodes": sum(
            row["cost"]["screen_episodes"] for run in ablations for row in run.get("rows", [])
        ),
    }


def missing_quantities(report: FinalReport) -> dict:
    """AC-7 quantities no artifact carries, and where each would come from."""
    present = {
        "success": report.search.get("mean_success") is not None,
        "return": report.search.get("mean_return") is not None,
        "final_distance": False,
        "invalid_program_rate": report.validity.get("invalid_program_rate") is not None,
        "repair_acceptance_rate": report.repair.get("repair_acceptance_rate") is not None,
        "complexity": report.search.get("mean_parameters") is not None,
        "deterministic_performance": "deterministic" in report.search.get("action_modes", []),
        "sampled_performance": "sampled" in report.search.get("action_modes", []),
        "simulator_steps": bool(report.resources.get("simulator_steps_recorded")),
        "expert_labels": False,
        "numerical_evaluations": bool(report.resources.get("numerical_evaluations")),
        "llm_calls": False,
        "llm_tokens": False,
        "wall_time": False,
        "inference_latency": False,
    }
    return {q: REQUIRED_QUANTITIES[q] for q, ok in present.items() if not ok}


def build_final_report(
    experiments_dirs,
    repair_studies_dir: str,
    ablation_dir: str,
    final_dir: str,
    ablation_budget=None,
) -> FinalReport:
    """Assemble the aggregate report. Reads files; runs nothing."""
    import time

    if isinstance(experiments_dirs, str):
        experiments_dirs = [experiments_dirs]
    records = []
    for directory in experiments_dirs:
        if os.path.isdir(directory):
            records.extend(load_records(directory))
    studies = load_json_dir(repair_studies_dir) if os.path.isdir(repair_studies_dir) else []
    ablations = load_json_dir(ablation_dir) if os.path.isdir(ablation_dir) else []

    report = FinalReport(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        sources={
            "experiments": list(experiments_dirs), "experiment_records": len(records),
            "repair_studies": repair_studies_dir, "studies": len(studies),
            "ablations": ablation_dir, "ablation_runs": len(ablations),
            "final": final_dir,
        },
        search=search_section(records),
        validity=validity_section(records),
        repair=repair_section(studies),
        ablation=ablation_section(ablations, budget=ablation_budget),
        final_test=final_test_section(final_dir),
    )
    report.resources = resource_section(records, studies, ablations)
    report.unavailable = missing_quantities(report)
    return report


def render(report: FinalReport) -> str:
    """The report as text, with the gaps stated rather than hidden."""
    lines = [f"Aggregate report, generated {report.generated_at}", ""]
    lines.append(
        f"Sources: {report.sources['experiment_records']} experiment records, "
        f"{report.sources['studies']} repair studies, "
        f"{report.sources['ablation_runs']} ablation runs"
    )

    search = report.search
    lines += ["", "Search"]
    lines.append(f"  candidates attempted        {search['records']}")
    lines.append(f"  scored                      {search['scored']}")
    lines.append(f"  intervention runs excluded  {search['interventions']}")
    for label, key in (("mean success", "mean_success"), ("best success", "best_success"),
                       ("mean return", "mean_return"), ("complexity, mean parameters", "mean_parameters")):
        value = search.get(key)
        lines.append(f"  {label:<28}" + ("not measured" if value is None else f"{value:.4f}"))

    validity = report.validity
    rate = validity.get("invalid_program_rate")
    lines += ["", "Validity"]
    lines.append(f"  attempted                   {validity['attempted']}")
    lines.append("  invalid-program rate        " + ("not measured" if rate is None else f"{rate:.3f}"))

    repair = report.repair
    lines += ["", "Repair"]
    lines.append(f"  explanations tested         {repair['explanations_tested']}")
    lines.append(f"  supported / rejected        {repair['explanations_supported']} / {repair['explanations_rejected']}")
    accept = repair.get("repair_acceptance_rate")
    lines.append(f"  repairs accepted            {repair['repairs_accepted']} of {repair['repairs_attempted']}"
                 + ("" if accept is None else f"  ({accept:.3f})"))

    if report.ablation.get("runs"):
        lines += ["", f"Ablation at {report.ablation['budget_screen_episodes']} screen "
                      f"episodes per arm, held-out success"]
        for arm, stats in report.ablation["arms"].items():
            lines.append(
                f"  {arm:<20}{stats['held_out_success']:.3f}   "
                f"start {stats['start_success']:.3f}   "
                f"{stats['screen_episodes']:.0f} screen episodes   n={stats['replicates']}"
            )
        excluded = report.ablation.get("budgets_excluded")
        if excluded:
            lines.append(f"  runs at {excluded} screen episodes are not pooled in: arms are "
                         f"comparable only at a matched cost")

    resources = report.resources
    lines += ["", "Resources recorded"]
    lines.append(f"  simulator steps             {resources.get('simulator_steps_recorded', 0)}")
    lines.append(f"  numerical evaluations       {resources.get('numerical_evaluations', 0)}")
    lines.append(f"  intervention episodes       {resources.get('intervention_episodes', 0)}")
    lines.append(f"  ablation screen episodes    {resources.get('ablation_screen_episodes', 0)}")
    modes = search.get("action_modes") or []
    lines.append(f"  action modes evaluated      {', '.join(modes) if modes else 'none recorded'}")

    final = report.final_test
    lines += ["", "Final test"]
    lines.append(f"  frozen                      {final['frozen']}")
    lines.append(f"  looks at held-out data      {final['looks']}")
    if final.get("note"):
        lines.append(f"  {final['note']}")

    if report.unavailable:
        lines += ["", f"Not measured ({len(report.unavailable)} of {len(REQUIRED_QUANTITIES)} "
                      f"AC-7 quantities). Each names where it would come from:"]
        for quantity, source in sorted(report.unavailable.items()):
            lines.append(f"  {quantity:<28}{source}")
        lines.append("")
        lines.append("  These are absent, not zero. A report that filled them in would read as "
                     "complete while resting on numbers nobody measured.")
    return "\n".join(lines)
