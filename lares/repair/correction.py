"""Correction-LLM repair and the clean repair ablation (``debug_spec.md`` Stage F, D10).

The pilot ablation (``scripts/ablate_repair.py``) mixed unequal fitting and
search budgets and had no LLM in it. This module is the replacement. Every arm
starts from one *common preprocessing* of the structure: fitting with its
checkpoint chosen by rollout probes, then one screen run of the result. Only
after that do the arms differ:

``no_repair``                 keep the preprocessed parent.
``llm_evidence_free``         a correction LLM sees the parent's code and nothing measured.
``llm_numeric``               ... plus the numeric evidence packet from the screen run.
``llm_numeric_intervention``  ... plus the outcomes of controlled substitutions.
``llm_numeric_vision``        the numeric packet plus a validated visual analysis of the parent's worst
                              failed screen episodes from a local vision model (R-006; not in Stage F's six).
``local_search``              perturb every declared parameter around the parent.
``global_search``             draw every declared parameter uniformly from its range.

An LLM arm's repair chain is: evidence packet -> one ``MutationProposal`` in
``repair`` mode, tied to the parent id and the packet's feedback id -> one
implementation with at most one linked validation correction -> the same
preprocessing as the parent (refit plus rollout-selected checkpoint) -> a screen
run. Every arm then replaces the parent under one declared rule
(:data:`ACCEPTANCE_RULE`), and every arm's result is scored on held-out cases
that took no part in any decision.

Budgets are matched on repair-evaluation episodes, not only reported: the
search arms get :meth:`StageFBudget.repair_episode_cap` episodes, the most an
LLM arm can spend by construction. LLM calls, tokens, gradient steps,
expert-assisted episodes and wall time are different currencies and are
recorded per arm in a :class:`~lares.search.resources.ResourceLedger` instead
of being converted into one another.
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from lares.eval.report import CHECKPOINT_FITTED, build_report
from lares.eval.runner import PolicyActor, evaluate_manifest
from lares.eval.stats import t_summary, win_loss_tie
from lares.fitting.checkpoints import CHECKPOINT_TIE_RULE, RolloutScore, is_better_checkpoint
from lares.fitting.objectives import DETERMINISTIC_MSE
from lares.fitting.optimizers import fit_policy_with_objective, state_parameters_sha256
from lares.repair.hypotheses import episode_metric_mean, find_clusters
from lares.search.resources import ResourceLedger
from lares.utils.hashing import sha256_text

ARM_NO_REPAIR = "no_repair"
ARM_LLM_EVIDENCE_FREE = "llm_evidence_free"
ARM_LLM_NUMERIC = "llm_numeric"
ARM_LLM_INTERVENTION = "llm_numeric_intervention"
ARM_LOCAL_SEARCH = "local_search"
ARM_GLOBAL_SEARCH = "global_search"
ARM_LLM_VISION = "llm_numeric_vision"
#: Stage F's frozen six; the default arm set.
ARMS = (
    ARM_NO_REPAIR,
    ARM_LLM_EVIDENCE_FREE,
    ARM_LLM_NUMERIC,
    ARM_LLM_INTERVENTION,
    ARM_LOCAL_SEARCH,
    ARM_GLOBAL_SEARCH,
)
#: Every arm a run may name.
ALL_ARMS = ARMS + (ARM_LLM_VISION,)

EVIDENCE_NONE = "none"
EVIDENCE_NUMERIC = "numeric"
EVIDENCE_INTERVENTION = "numeric_intervention"
EVIDENCE_VISION = "numeric_vision"
#: What each LLM arm is shown. The arms differ in this and nothing else.
LLM_ARM_EVIDENCE = {
    ARM_LLM_EVIDENCE_FREE: EVIDENCE_NONE,
    ARM_LLM_NUMERIC: EVIDENCE_NUMERIC,
    ARM_LLM_INTERVENTION: EVIDENCE_INTERVENTION,
    ARM_LLM_VISION: EVIDENCE_VISION,
}
SEARCH_ARMS = (ARM_LOCAL_SEARCH, ARM_GLOBAL_SEARCH)

#: Step size of the local search, as a fraction of each parameter's declared range.
LOCAL_SEARCH_SIGMA = 0.1

ACCEPTANCE_RULE = (
    "an arm replaces the preprocessed parent only when its result is strictly better on the "
    f"screen manifest by the checkpoint tie rule ({CHECKPOINT_TIE_RULE}); an exact tie keeps the parent"
)

#: Chain outcomes, one per arm per replicate.
OUTCOME_KEPT_PARENT = "kept_parent"
OUTCOME_REPLACED = "replaced"
OUTCOME_GENERATION_FAILED = "generation_failed"
OUTCOME_NOT_A_REPAIR = "not_a_repair"


# ---------------------------------------------------------------------------
#  Budget
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StageFBudget:
    """Everything an arm may spend, declared before any arm runs."""

    screen_episodes: int
    probe_episodes: int
    fit_budget: int
    checkpoint_steps: tuple
    #: Controlled substitutions the intervention arm may run (explanations tested).
    max_substitutions: int = 2
    #: Implementation plus linked corrections per LLM arm.
    max_llm_calls: int = 4
    max_repair_per_candidate: int = 1

    @property
    def refit_probe_episodes(self) -> int:
        return len(self.checkpoint_steps) * self.probe_episodes if self.fit_budget else 0

    @property
    def repair_episode_cap(self) -> int:
        """Most episodes an LLM arm can spend: substitutions, refit probes, the child's screen run.

        The substitution study reuses the common screen run of the parent as its
        control, so the control is not charged to any arm.
        """
        substitutions = self.max_substitutions * self.screen_episodes
        return substitutions + self.refit_probe_episodes + self.screen_episodes

    @property
    def search_candidates(self) -> int:
        """Screen runs a search arm buys with the same cap."""
        return max(1, self.repair_episode_cap // self.screen_episodes)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["checkpoint_steps"] = list(self.checkpoint_steps)
        d.update(repair_episode_cap=self.repair_episode_cap, search_candidates=self.search_candidates,
                 refit_probe_episodes=self.refit_probe_episodes)
        return d


def check_isolation(screen, probe, held_out) -> None:
    """Refuse overlapping streams: the screen decides, the probe selects, the held-out set only scores.

    Cases are compared by placement hash, as the committed splits are; a case id
    is only an index within its manifest. Synthetic cases have no placement, so
    they fall back to the case id.
    """
    streams = {"screen": screen, "probe": probe, "held_out": held_out}
    ids = {name: {c.task_hash or c.case_id for c in m.episodes} for name, m in streams.items()}
    names = list(ids)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = ids[a] & ids[b]
            if overlap:
                raise ValueError(f"{a} and {b} manifests overlap on {sorted(overlap)[:5]}")


# ---------------------------------------------------------------------------
#  Scoring and preprocessing
# ---------------------------------------------------------------------------


def score_of(result) -> RolloutScore:
    progress = episode_metric_mean(result, "signed_goal_progress")
    return RolloutScore(float(result.success_rate), float(progress if progress is not None else -1e9))


def evaluate(policy, manifest, env, ledger, stream, max_steps, name):
    started = time.time()
    result = evaluate_manifest(PolicyActor(policy, deterministic=True, name=name), manifest, env,
                               max_steps=max_steps)
    ledger.add_episodes(stream, result)
    ledger.add_wall_time("evaluation", time.time() - started)
    return result


def preprocess(policy, data, probe, env, budget: StageFBudget, seed: int, ledger: ResourceLedger,
               max_steps: int = 150, structure_id: str = "policy"):
    """The common preprocessing: fit, probe every scheduled checkpoint, keep the best.

    Applied identically to each replicate's parent and to every LLM child, so an
    LLM arm's child is compared after the same treatment its parent received.
    Returns ``(policy holding the selected state, summary dict)``.
    """
    if not budget.fit_budget:
        state = copy.deepcopy(policy.state_dict())
        return policy, {"fit_budget": 0, "selected_step": 0,
                        "parameters_sha256": state_parameters_sha256(state), "checkpoint_history": []}

    def rollout_probe(candidate):
        result = evaluate(candidate, probe, env, ledger, "probe", max_steps, f"{structure_id}:probe")
        return score_of(result)

    started = time.time()
    fitted = fit_policy_with_objective(
        policy, data, DETERMINISTIC_MSE, budget=budget.fit_budget, seed=seed,
        rollout_probe=rollout_probe, checkpoint_steps=budget.checkpoint_steps, structure_id=structure_id,
    )
    ledger.add("bc_gradient_steps", budget.fit_budget)
    ledger.add_wall_time("fitting", time.time() - started)
    return policy, {
        "fit_budget": budget.fit_budget,
        "selected_step": fitted.selected_step,
        "parameters_sha256": state_parameters_sha256(fitted.selected_state),
        "checkpoint_history": list(getattr(fitted, "checkpoint_history", []) or []),
    }


# ---------------------------------------------------------------------------
#  Evidence packet
# ---------------------------------------------------------------------------


def episode_table(result) -> str:
    header = "  case          success  label                    closest(m)  progress(m)  drift(m)"
    lines = [header]

    def num(value):
        return f"{value:+.4f}" if value is not None else "   n/a "

    for e in result.episodes:
        lines.append(
            f"  {str(e.case_id)[-12:]:<13} {e.success:>6.0f}  {str(e.failure_label or 'success'):<24} "
            f"{num(e.min_tcp_object_distance):>10}  {num(e.signed_goal_progress):>11}  {num(e.lateral_drift):>8}"
        )
    return "\n".join(lines)


def build_evidence_text(level: str, parent_id: str, parent_source: str, control_result=None,
                        control_report=None, study=None, vision_block=None) -> str:
    """The text a correction LLM is shown. Levels nest: each adds to the one before.

    The intervention and vision levels each add one section to the numeric packet; they are
    siblings, not nested in each other.
    """
    from lares.repair.study import summarise as summarise_study
    from lares.search.feedback import CandidateEvidence, diagnostics_block

    if level not in (EVIDENCE_NONE, EVIDENCE_NUMERIC, EVIDENCE_INTERVENTION, EVIDENCE_VISION):
        raise ValueError(f"unknown evidence level {level!r}")
    if level == EVIDENCE_VISION and vision_block is None:
        raise ValueError("the vision level needs a visual-analysis block (an empty analysis says so itself)")
    lines = [
        "# Repair request",
        "",
        f"Propose exactly one repair of parent `{parent_id}`: mode \"repair\", parent_ids [\"{parent_id}\"].",
        "Change only the component you believe is at fault and predict which measured quantity moves.",
        "",
        f"## Parent `{parent_id}` source",
        "",
        "```python",
        parent_source.strip(),
        "```",
        "",
    ]
    if level == EVIDENCE_NONE:
        lines.append("No evaluation evidence is provided for this request. Reason from the code and the task alone.")
        return "\n".join(lines) + "\n"
    if control_result is None or control_report is None:
        raise ValueError(f"evidence level {level!r} needs the parent's screen result and report")
    lines += [
        f"## Numeric evidence: parent `{parent_id}` on {control_result.manifest_id} "
        f"({len(control_result.episodes)} cases)",
        "",
        diagnostics_block(CandidateEvidence(candidate_id=parent_id, code=parent_source, report=control_report)),
        "",
        "Per case:",
        episode_table(control_result),
        "",
    ]
    clusters = find_clusters(control_report)
    if clusters:
        lines.append("Recurrent failures: " + ", ".join(f"{c.label} ({c.size} cases)" for c in clusters))
        lines.append("")
    if level == EVIDENCE_INTERVENTION:
        if study is None:
            raise ValueError("the intervention level needs a substitution study")
        lines += ["## Controlled substitutions (expert-assisted; diagnoses, not scores)", "",
                  summarise_study(study), ""]
    if level == EVIDENCE_VISION:
        lines += ["## Visual failure analysis (local vision model; supporting evidence, checked against the numbers)",
                  "", vision_block.strip(), ""]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
#  Arms
# ---------------------------------------------------------------------------


@dataclass
class ArmContext:
    """What every arm of one replicate shares."""

    env: object
    env_name: str
    schema: object
    data: object
    screen: object
    probe: object
    budget: StageFBudget
    seed: int
    max_steps: int = 150
    #: LLM plumbing; unused by the non-LLM arms.
    client: object = None
    llm_args: object = None
    prompts: dict = None
    param_budget: tuple = None
    #: ``(policy, structure, manifest, env, env_name, schema, max_steps) -> study payload``.
    study_fn: object = None
    #: ``(policy, parent_id, control_result, ctx, run_dir) -> {"block", "run", "episodes_rerun"}``; see
    #: :func:`vision_evidence`. Needed only by the vision arm.
    vision_fn: object = None
    #: Env for the probe manifest when it comes from another task pool than the screen; else ``env``.
    probe_env: object = None

    @property
    def probe_stream_env(self):
        return self.probe_env if self.probe_env is not None else self.env


def _parameter_names(policy) -> list:
    ranges = policy.get_param_ranges()
    return sorted(name for name, _ in policy.named_parameters() if name in ranges)


def _propose(policy, arm, rng):
    ranges = policy.get_param_ranges()
    with torch.no_grad():
        for name, param in policy.named_parameters():
            if name not in ranges:
                continue
            lo, hi = float(ranges[name][0]), float(ranges[name][1])
            if arm == ARM_LOCAL_SEARCH:
                noise = rng.normal(0.0, LOCAL_SEARCH_SIGMA * (hi - lo), size=tuple(param.shape))
                param.add_(torch.as_tensor(noise, dtype=param.dtype))
            else:
                param.copy_(torch.as_tensor(rng.uniform(lo, hi, size=tuple(param.shape)), dtype=param.dtype))
    policy.clip_params()


def run_search_arm(arm, parent, parent_score, ctx: ArmContext, ledger: ResourceLedger):
    """Spend the matched episode cap on parameter candidates; keep the best by the tie rule."""
    if arm not in SEARCH_ARMS:
        raise ValueError(f"{arm!r} is not a search arm")
    rng = np.random.default_rng(ctx.seed)
    started = time.time()
    best, best_score, trace = None, None, []
    for index in range(ctx.budget.search_candidates):
        candidate = copy.deepcopy(parent)
        _propose(candidate, arm, rng)
        result = evaluate(candidate, ctx.screen, ctx.env, ledger, "repair", ctx.max_steps, f"{arm}:{index}")
        score = score_of(result)
        trace.append({"index": index, **score.to_dict()})
        if is_better_checkpoint(score, best_score):
            best, best_score = candidate, score
    ledger.add_wall_time("repair", time.time() - started)
    replaced = best is not None and is_better_checkpoint(best_score, parent_score)
    chain = {
        "parameters": _parameter_names(parent),
        "sigma_fraction": LOCAL_SEARCH_SIGMA if arm == ARM_LOCAL_SEARCH else None,
        "candidates": trace,
        "best_screen": best_score.to_dict() if best_score else None,
        "outcome": OUTCOME_REPLACED if replaced else OUTCOME_KEPT_PARENT,
    }
    return (best if replaced else parent), chain


def run_llm_arm(arm, parent_id, parent, parent_source, control_result, ctx: ArmContext, ledger: ResourceLedger,
                run_dir: str):
    """One correction-LLM repair chain: packet, proposal, implementation, refit, screen."""
    from lares.core.policy_generation import generate_candidates
    from lares.search.generation import LLMCallLedger, write_feedback_bundle
    from lares.search.schemas import MODE_REPAIR

    level = LLM_ARM_EVIDENCE[arm]
    os.makedirs(run_dir, exist_ok=True)
    chain = {"evidence_level": level}

    control_report = build_report(control_result, f"{parent_id}:control", CHECKPOINT_FITTED,
                                  action_dim=int(ctx.env.action_space.shape[0]))
    study = None
    if level == EVIDENCE_INTERVENTION:
        started = time.time()
        study_fn = ctx.study_fn
        if study_fn is None:
            from lares.repair.study import run_study as study_fn
        study = study_fn(parent, _NamedStructure(parent_id, parent_source, ctx), ctx.screen, ctx.env,
                         ctx.env_name, ctx.schema, max_steps=ctx.max_steps, max_clusters=1,
                         control_result=control_result, max_substitutions=ctx.budget.max_substitutions)
        ledger.add("expert_assisted_episodes", study["intervention_episodes"])
        ledger.add("expert_assisted_environment_steps", study["intervention_environment_steps"])
        ledger.add_wall_time("repair", time.time() - started)
        with open(os.path.join(run_dir, "study.json"), "w", encoding="utf-8") as f:
            json.dump(study, f, indent=2, default=str)
        chain["study"] = {"path": "study.json", "supported": [s.get("supported") for s in study["studies"]],
                          "rollout_episodes": study["rollout_episodes"],
                          "intervention_episodes": study["intervention_episodes"]}
    vision_block = None
    if level == EVIDENCE_VISION:
        if ctx.vision_fn is None:
            raise ValueError(f"{arm} needs ArmContext.vision_fn")
        started = time.time()
        vision = ctx.vision_fn(parent, parent_id, control_result, ctx, run_dir)
        # The media reruns replay screen cases with rendering on: repair-evaluation episodes of this arm.
        ledger.add("repair_episodes", int(vision["episodes_rerun"]))
        ledger.add_wall_time("repair", time.time() - started)
        vision_block = vision["block"]
        with open(os.path.join(run_dir, "vision.json"), "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in vision.items() if k != "block"}, f, indent=2, default=str)
        chain["vision"] = {"path": "vision.json", **{k: vision["run"].get(k) for k in (
            "model_id", "episodes_analysed", "episodes_rerun", "frames_shown", "analyses_returned",
            "analyses_rejected", "conflicts")}}

    text = build_evidence_text(level, parent_id, parent_source, control_result, control_report, study,
                               vision_block=vision_block)
    bundle = write_feedback_bundle(run_dir, 0, ctx.screen.manifest_id, [parent_id], text)
    chain["evidence_packet"] = bundle.to_dict()

    llm_ledger = LLMCallLedger(run_dir)
    data_pkl = os.path.join(run_dir, "data.pkl")
    import pickle

    with open(data_pkl, "wb") as f:
        pickle.dump([{"obs": np.zeros(int(ctx.env.observation_space.shape[0]))}], f)
    started = time.time()
    candidates = generate_candidates(
        ctx.client, ctx.llm_args, os.path.join(run_dir, "gen_1"), 1,
        int(ctx.env.observation_space.shape[0]), int(ctx.env.action_space.shape[0]), ctx.env_name,
        ctx.prompts, 1, data_pkl, llm_ledger, feedback_bundle=bundle, feedback_text=text,
        parent_sources={parent_id: parent_source},
        llm_transcript_path=os.path.join(run_dir, "llm_transcript.jsonl"),
        max_total_calls=ctx.budget.max_llm_calls, max_repair_per_candidate=ctx.budget.max_repair_per_candidate,
        param_budget=ctx.param_budget,
    )
    ledger.add_wall_time("generation", time.time() - started)
    totals = llm_ledger.totals()
    ledger.add_llm_totals(totals)
    chain["llm"] = {"call_ids": [r.call_id for r in llm_ledger.records()], "totals": totals,
                    "ledger": "llm_calls.jsonl"}
    chain["attempts"] = [
        {"attempt_id": c.attempt_id, "previous_attempt_id": c.previous_attempt_id,
         "validation": {k: c.validation.get(k) for k in ("passed", "stage", "errors")},
         "lineage": c.lineage_dict(bundle), "code_sha256": _sha(c.code)}
        for c in candidates
    ]
    final = candidates[-1] if candidates else None
    if final is None or not final.valid:
        chain["outcome"] = OUTCOME_GENERATION_FAILED
        return parent, chain
    proposal = final.proposal
    chain["proposal"] = proposal.to_dict() if proposal else None
    if proposal is None or proposal.mode != MODE_REPAIR or list(proposal.parent_ids) != [parent_id]:
        chain["outcome"] = OUTCOME_NOT_A_REPAIR
        return parent, chain

    child, child_prep = preprocess(final.policy, ctx.data, ctx.probe, ctx.probe_stream_env, ctx.budget, ctx.seed, ledger,
                                   ctx.max_steps, structure_id=final.attempt_id)
    child_result = evaluate(child, ctx.screen, ctx.env, ledger, "repair", ctx.max_steps, f"{arm}:child")
    child_score, parent_score = score_of(child_result), score_of(control_result)
    replaced = is_better_checkpoint(child_score, parent_score)
    metric = proposal.predicted_metric
    chain.update(
        child={"attempt_id": final.attempt_id, "code_sha256": _sha(final.code), "preprocessing": child_prep,
               "screen": child_score.to_dict()},
        prediction=proposal.check_outcome(episode_metric_mean(control_result, metric),
                                          episode_metric_mean(child_result, metric)),
        outcome=OUTCOME_REPLACED if replaced else OUTCOME_KEPT_PARENT,
    )
    with open(os.path.join(run_dir, "child_policy_source.py"), "w", encoding="utf-8") as f:
        f.write(final.code)
    return (child if replaced else parent), chain


def vision_evidence(analyst, schema, limit: int = 3, save_media: bool = True):
    """A ``vision_fn`` for :class:`ArmContext`: analyse the parent's worst failed screen episodes.

    Re-runs up to ``limit`` failed screen cases with rendering on, shows the analyst the frames the fixed
    media rule selects, validates and merges each analysis with the episode's numbers, and returns the
    repair-prompt block. Rejected analyses are recorded and dropped; when nothing survives, the block says
    so, and the arm still runs, so an empty packet is part of the assigned-arm result.
    """
    from lares.vision import analyse_failures, evidence_block

    def vision_fn(policy, parent_id, control_result, ctx, run_dir):
        actor = PolicyActor(policy, deterministic=True, name=parent_id)
        analyses, merged, run = analyse_failures(
            actor, ctx.env, ctx.screen, control_result, analyst, schema, candidate_id=parent_id,
            max_steps=ctx.max_steps, limit=limit, media_dir=os.path.join(run_dir, "media") if save_media else "",
        )
        for analysis in analyses:
            analysis.save(os.path.join(run_dir, "analyses"))
        return {"block": evidence_block(merged, analyses), "run": run.to_dict(),
                "episodes_rerun": int(run.episodes_rerun), "merged": [m.to_dict() for m in merged]}

    return vision_fn


class _NamedStructure:
    """The two attributes :func:`lares.repair.study.run_study` reads from a structure."""

    def __init__(self, structure_id, source, ctx):
        self.structure_id = structure_id
        self.source = source
        self.obs_dim = int(ctx.env.observation_space.shape[0])
        self.action_dim = int(ctx.env.action_space.shape[0])


def _sha(text) -> str | None:
    if text is None:
        return None
    return sha256_text(text)


# ---------------------------------------------------------------------------
#  One replicate
# ---------------------------------------------------------------------------


def run_replicate(structure_id: str, source: str, build, ctx: ArmContext, held_out, out_dir: str,
                  arms=ARMS, held_out_env=None) -> dict:
    """Common preprocessing once, then every arm, then held-out scoring of each arm's result.

    ``build()`` returns a fresh policy at the structure's declared values.
    ``held_out_env`` is needed when the held-out cases come from another task pool.
    Writes ``<out_dir>/<arm>/chain.json`` per arm and returns the replicate dict.
    """
    held_out_env = held_out_env if held_out_env is not None else ctx.env
    unknown = [a for a in arms if a not in ALL_ARMS]
    if unknown:
        raise ValueError(f"unknown arms {unknown}; expected a subset of {ALL_ARMS}")
    check_isolation(ctx.screen, ctx.probe, held_out)
    os.makedirs(out_dir, exist_ok=True)

    common = ResourceLedger()
    parent, prep = preprocess(build(), ctx.data, ctx.probe, ctx.probe_stream_env, ctx.budget, ctx.seed, common,
                              ctx.max_steps, structure_id=structure_id)
    control = evaluate(parent, ctx.screen, ctx.env, common, "screen", ctx.max_steps, f"{structure_id}:control")
    parent_score = score_of(control)
    parent_state = copy.deepcopy(parent.state_dict())

    rows = []
    for arm in arms:
        ledger = ResourceLedger()
        working = copy.deepcopy(parent)
        working.load_state_dict(parent_state)
        run_dir = os.path.join(out_dir, arm)
        os.makedirs(run_dir, exist_ok=True)
        if arm == ARM_NO_REPAIR:
            final, chain = working, {"outcome": OUTCOME_KEPT_PARENT}
        elif arm in SEARCH_ARMS:
            final, chain = run_search_arm(arm, working, parent_score, ctx, ledger)
        else:
            final, chain = run_llm_arm(arm, structure_id, working, source, control, ctx, ledger, run_dir)
        scored = evaluate(final, held_out, held_out_env, ResourceLedger(), "expanded", ctx.max_steps, f"{arm}:held_out")
        spent = ledger.counters["repair_episodes"] + ledger.counters["probe_episodes"] \
            + ledger.counters["expert_assisted_episodes"]
        if spent > ctx.budget.repair_episode_cap:
            raise RuntimeError(f"{arm} spent {spent} repair-evaluation episodes, over the cap "
                               f"{ctx.budget.repair_episode_cap}")
        chain.update(
            arm=arm,
            structure_id=structure_id,
            seed=ctx.seed,
            parent={"structure_id": structure_id, "source_sha256": _sha(source), "preprocessing": prep,
                    "screen": parent_score.to_dict()},
            final_parameters_sha256=state_parameters_sha256(final.state_dict()),
            held_out={"manifest_id": held_out.manifest_id, "success": scored.success_rate,
                      "progress": episode_metric_mean(scored, "signed_goal_progress"),
                      "per_case": {e.case_id: float(e.success) for e in scored.episodes}},
            resources=ledger.to_dict(),
            repair_episodes_spent=spent,
            repair_episode_cap=ctx.budget.repair_episode_cap,
            acceptance_rule=ACCEPTANCE_RULE,
        )
        with open(os.path.join(run_dir, "chain.json"), "w", encoding="utf-8") as f:
            json.dump(chain, f, indent=2, default=str)
        rows.append(chain)

    replicate = {
        "structure_id": structure_id,
        "seed": ctx.seed,
        "budget": ctx.budget.to_dict(),
        "manifests": {"screen": ctx.screen.manifest_id, "probe": ctx.probe.manifest_id,
                      "held_out": held_out.manifest_id},
        "common_preprocessing": {**prep, "screen": parent_score.to_dict(), "resources": common.to_dict()},
        "rows": rows,
    }
    with open(os.path.join(out_dir, "replicate.json"), "w", encoding="utf-8") as f:
        json.dump(replicate, f, indent=2, default=str)
    return replicate


# ---------------------------------------------------------------------------
#  Verification and summary
# ---------------------------------------------------------------------------


def verify_chain(chain: dict, run_dir: str) -> list:
    """Why an LLM arm's saved chain does not trace, if it does not. Empty means complete."""
    from lares.search.generation import FeedbackBundle, LLMCallLedger

    problems = []
    arm = chain.get("arm")
    if arm not in LLM_ARM_EVIDENCE:
        return problems
    if chain.get("evidence_level") != LLM_ARM_EVIDENCE[arm]:
        problems.append(f"{arm} was shown evidence level {chain.get('evidence_level')!r}")
    try:
        bundle = FeedbackBundle.from_dict(chain["evidence_packet"])
        bundle.validate(run_dir)
    except Exception as exc:  # noqa: BLE001 - every defect is reported, not raised
        problems.append(f"evidence packet does not verify: {exc}")
        bundle = None
    if bundle is not None and bundle.context_parent_ids != [chain["parent"]["structure_id"]]:
        problems.append(f"packet context {bundle.context_parent_ids} is not the parent alone")
    ledger_path = os.path.join(run_dir, "llm_calls.jsonl")
    if not os.path.isfile(ledger_path):
        problems.append("no LLM-call ledger")
    else:
        ledger = LLMCallLedger(run_dir)
        for call_id, reasons in ledger.verify().items():
            problems.append(f"{call_id}: {'; '.join(reasons)}")
        if sorted(chain["llm"]["call_ids"]) != sorted(r.call_id for r in ledger.records()):
            problems.append("the chain's call ids differ from the ledger's")
    for attempt in chain.get("attempts", []):
        lineage = attempt["lineage"]
        if bundle is not None and lineage.get("feedback_id") != bundle.feedback_id:
            problems.append(f"{attempt['attempt_id']} is not tied to the packet's feedback id")
    if chain.get("outcome") in (OUTCOME_REPLACED, OUTCOME_KEPT_PARENT):
        proposal = chain.get("proposal") or {}
        if proposal.get("parent_ids") != [chain["parent"]["structure_id"]]:
            problems.append("the proposal is not tied to the parent id")
        if "child" not in chain:
            problems.append("an evaluated repair has no child record")
    return problems


def replicate_interval(diffs, level: float = 0.95) -> dict:
    """Mean and small-sample t interval over replicate-level paired differences."""
    s = t_summary(diffs, level)
    return {"n_replicates": s["n"], "mean_difference": s["mean"], **win_loss_tie(diffs),
            "ci95_low": s["low"], "ci95_high": s["high"]}


COST_KEYS = ("repair_episodes", "probe_episodes", "expert_assisted_episodes", "bc_gradient_steps",
             "llm_calls", "llm_total_tokens")


def summarise_replicates(replicates, reference: str = ARM_NO_REPAIR) -> dict:
    """Per-arm held-out success and cost, and each arm minus ``reference`` paired over replicates."""
    by_arm = {}
    for rep in replicates:
        for row in rep["rows"]:
            by_arm.setdefault(row["arm"], {})[(rep["structure_id"], rep["seed"])] = row
    arms = {}
    for arm, rows in by_arm.items():
        values = list(rows.values())
        arms[arm] = {
            "n_replicates": len(values),
            "held_out_success": float(np.mean([r["held_out"]["success"] for r in values])),
            "outcomes": {o: sum(1 for r in values if r.get("outcome") == o) for o in
                         sorted({r.get("outcome") for r in values})},
            "mean_cost": {k: float(np.mean([r["resources"][k] for r in values])) for k in COST_KEYS},
            "mean_wall_time_seconds": float(np.mean([sum(r["resources"]["wall_time_seconds"].values())
                                                     for r in values])),
        }
    paired = {}
    base = by_arm.get(reference, {})
    for arm, rows in by_arm.items():
        if arm == reference:
            continue
        shared = sorted(set(base) & set(rows))
        paired[arm] = replicate_interval(
            [rows[k]["held_out"]["success"] - base[k]["held_out"]["success"] for k in shared]
        )
    return {"reference": reference, "arms": arms, "paired": paired}
