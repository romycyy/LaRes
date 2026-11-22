# Next research direction: reward-trained symbolic policies

Prepared 2026-11-22 for `/research-cycle` in `romycyy/LaRes`, branch `staging`.
Reviewed GitHub `RESEARCH_HANDOFF.md`, the current handoff, the research-cycle skill and CLI at commit `2d0907908ed8b57f91d517fcf8b4888e8be395ca`.

This is a direction and proposal brief, not an approved protocol or a replacement for the skill. It takes effect when the user invokes it. Refresh the ledger before acting; later decisions supersede this snapshot. Use local proposal labels P1-P4 below; let the tooling allocate actual R-ids.

## Direction

Continue Stage-3 Shadow Hand axial rolling by testing reward-trained symbolic residuals and incremental policy construction. Use the frozen D-072-qualified controller as the experimental reference, without changing the saved default incumbent. First compare independent parameter draws, ARS/CEM and PPO on one identical sparse, state-conditioned symbolic residual family. If an inner learner produces a confirmed useful gain, compare fixed structure, grammar-based structural evolution and LLM-guided structural evolution with that learner and inherited parameters. Investigate compact memory or anchored component replacement only when the preceding results motivate them. Preserve the task, frozen evaluation manifests and strict paired promotion/drop criteria. Run one protocol at a time. No controller-generation LLM calls during P1; advisor calls for the research loop remain allowed. Later controller-generation calls need an agreed protocol and explicit call budget. Do not repeat R-102, retune its confirmation cases, reopen cold whole-controller generation, or resume expert cloning. Continue evidence-driven follow-ups within this direction through the existing loop, including its escalation and exhaustion rules.

## Evidence that proposals must address

| Decision | Established result | Consequence |
|---|---|---|
| D-072 / R-102 | Frozen joint-motion residual: 267/672 successes versus 230/672; paired +0.055, 95% CI [0.024, 0.086]; drops 34 versus 34 | A qualified working reference exists. Its mechanism is unproven and it has not replaced the default incumbent asset. |
| D-048 / R-058 | CMA-ES versus random search on the old learner's 12 constants: +0.003 [-0.076, 0.082]; neither improves the parent | Repeating that scalar-tuning study is insufficient. P1 changes the residual family and tests policy gradients. |
| D-065, D-068 | Cold generation is near its floor under the tested modular and contact-output contracts | Build on competent behavior; a new representation name alone is not a mechanism. |
| D-039, D-045, D-047 | Expert fitting, takeover and expert-trace repair do not improve this learner reliably | Reward and closed-loop evaluation must decide success; expert-action matching is not the objective. |
| D-069, D-071 | Free-space fingertips track requests; geometrically close fingers track poorly and some joints saturate | Measure executed action authority. Do not introduce the failed contact-frame decoder into P1. Proximity is a geometric proxy, not a contact-force measurement. |

Read the corresponding ledger entries and artifacts before triage. Some standing constraints and early sections of generated summaries are stale; the latest decisions and current code win.

## How to feed this file into the skill

1. Read `.claude/skills/research-cycle/SKILL.md`, `research/STATE.md`, `docs/CHATGPT_HANDOFF_2026-11-22.md` and the relevant decision blocks in `RESEARCH_HANDOFF.md`.
2. The user's invocation of this file supplies a new direction. Set `research.py direct` to the actual text of **Direction**, not merely this filename. Resume open work before proposing more.
3. With nothing open, use `propose --n 1 --focus <actual current-proposal text>`. Start with P1. The CLI does not open a Markdown path passed to `--focus`; pass the section contents. Use structured subprocess arguments rather than interpolated shell commands.
4. Let the advisor and implementer revise and triage the proposal normally. Include its controls, exact hypothesis, metric, bounds, costs and dependencies in the idea record so subsequent advisor calls can see them. The advisor does not automatically see files the implementer read.
5. Use the existing propose/triage/position/start/record/verdict/analyse/decide workflow. No hand-created approvals or decisions; do not edit generated `STATE.md`, bypass checks with `--force`, or weaken validation.

Example for setting the direction without shell interpolation, from the repository root:

```python
from pathlib import Path
import subprocess
import sys

brief = Path("docs/NEXT_RESEARCH_DIRECTION.md").read_text()
direction = brief.split("## Direction\n", 1)[1].split("\n## ", 1)[0].strip()
subprocess.run([sys.executable, "scripts/research.py", "direct", direction], check=True)
```

## P1 — Learn a symbolic residual from reward

**Question:** can reward learning improve a state-conditioned symbolic actor beyond both the qualified reference and independent parameter draws?

**Reference:** reconstruct the exact D-058 controller plus `JointMotionResidual` using R-102's frozen `selected` parameters and hash `2a509ef5ffc148cab2a513bc2c7339cca9a620b50dec2f7a988fd2b041c58cd1`. Read exact values from the recorded config; do not reconstruct them from rounded prose. Save a separate experiment-local reference descriptor with source/state/config hashes. Leave `config/pen_spin/d058_fixed_stroke_incumbent/` unchanged. This avoids requiring a default-incumbent adoption decision to begin research.

**One actor family for every learner:**

`action = clamp(reference_action + residual_theta(features), -1, 1)`.

Freeze a sparse feature-to-joint mapping with about 20-40 coefficients. Candidate features: normalized joint velocity, fixed filtered joint velocity, posture error, centering error and unsigned axial-rate error. Freeze normalization using training data only. Initialize the additional residual identically to zero. Freeze the reference and filter constants in P1; no new trainable hidden-state dynamics yet. Check that zero coefficients reproduce reference actions bit for bit, nonzero coefficients affect executed actions, and per-environment state resets correctly.

**Arms:** independent parameter draws; one parameter-space learner (ARS or CEM, chosen before running); PPO with this symbolic actor and a neural critic. Same coefficients, initial deterministic policy, parameter bounds, features, output constraints, training cases and reward. Freeze and record each arm's exploration scheme; report deterministic deployment separately from stochastic training. A neural critic is permitted; the deployed actor remains symbolic.

**Primary proposed prediction:** `paired_success_difference`, direction `increase`, threshold `0.05`, PPO minus independent draws on fresh deterministic evaluation. Also preregister the two necessary contrasts: PPO versus ARS/CEM, and each trained family versus the qualified reference. Apply multiplicity correction to promotion claims; do not choose the primary contrast after seeing results. The implementer and advisor may revise the prediction before spending.

**Initial scale:** three independent training seeds per learner, initially 500,000 aggregate simulator transitions per seed: 4.5 million training transitions total. All parameter-ranking rollouts count toward this budget. Give all arms the same validation schedule and account for validation separately. Estimate episode count and wall time from an agreed smoke run; record early terminations and actual transitions. This is a finite feasibility pilot, not proof of learnability or nonlearnability. Any extension is a new preregistered follow-up, not an automatic rerun until significance.

**Before the main comparison:** include a small runner/identity/likelihood smoke stage inside the approved protocol. Validate training-to-evaluation reconstruction. Define the dense training reward once for all arms: retained, centered axial rolling with bounded speed reward, a fall penalty and modest action cost. Freeze its formula and coefficients before the comparison; do not select reward variants using fresh outcomes. Preserve the original sustained-success definition.

**PPO details that must be correct:** save the action variable used in the likelihood calculation; do not apply an ordinary Gaussian density to an already clipped action. Use a documented latent-action or bounded-distribution contract. Store/reset filter state correctly. Do not assume archived BC/GRPO smoke functions implement this experiment. If training memory parameters later, recompute sequences consistently instead of shuffling stale parameter-dependent hidden states.

**Interpretation:** PPO winning supports action-level credit assignment; ARS/CEM winning supports parameter-space learning; both beating independent draws supports learned inner optimization in this family. Reward-only gains do not count as task improvement. If proposed changes vanish after clipping, investigate authority before increasing training. If meaningful changes execute but all learners fail, examine representation and objective mismatch.

**Advance to P2 only after** at least one inner learner has a frozen finalist with prospectively confirmed sustained-success improvement over the qualified reference and passes the drop bound. A pilot winner alone does not open P2.

## P2 — Does structural evolution add value?

**Prerequisite:** P1's useful inner learner has passed confirmation.

**Hypothesis:** small structural edits with inherited parameters find better controllers than fitting one fixed structure; LLM-proposed edits may be better than grammar-based edits.

Compare fixed structure, grammar/GP edits, and LLM edits with the same inner learner. Equalize total training/evaluation transitions, optimizer-update allocation, candidate evaluations, complexity cap and validation access. For fixed structure, spend the matching allocation on independent restarts or continued optimization under a rule frozen in advance.

Allowed edits: add/replace a feature, product term, bounded gate, short filter or feature-to-joint-group connection. Preserve unchanged parameters. Initialize new contributions to zero and verify behavioral inheritance; avoid initializing a new branch so every learning gradient is dead. Keep expression depth, node count, coefficient count and memory dimension bounded. No arbitrary neural actor hidden inside generated code.

**Primary proposed prediction:** `paired_success_difference increase >= 0.05`, LLM edits minus grammar edits. Also measure grammar edits versus fixed structure. Compare algorithms across independent search blocks; within each block, pair finalist evaluations on the same cases. Use at least three blocks for a pilot and size stronger inference prospectively rather than treating many episodes from one search as independent search runs. Budget all generation/correction calls and use the existing `llm_pool` with 16 workers when multi-block generation is approved.

If grammar edits help but LLM edits do not, continue with grammar search. If neither helps, do not claim a structural-evolution contribution. Broad LLM-structure/numerical-parameter separation is existing prior art.

## P3 — Conditional memory study

Use only if P1 traces suggest temporal information may help. Compare instantaneous velocity, the existing fixed filter and a matched small two-timescale representation, using the winning inner learner. Keep features/output authority and coefficient budgets comparable. Do not simultaneously change the optimizer, reward and structure search.

**Proposed prediction:** `paired_success_difference increase >= 0.05`, two-timescale versus single-filter residual. Use closed-loop success as primary; tracking error is diagnostic. Only evolve memory-update equations after this controlled test supports their value. P3 can precede P2 only if its dependency is recorded and triage agrees.

## P4 — Conditional anchored component construction

Use if whole-residual learning or whole-policy construction remains ineffective despite verified action authority. Keep a competent anchor for unchanged outputs and replace one predefined joint/finger-chain group at a time, optimizing the hybrid's rollout return. Compare incremental replacement with all-selected-components-at-once under the same total budget and symbolic complexity cap.

**Proposed prediction:** `paired_success_difference increase >= 0.05`, incremental versus simultaneous construction. Begin with the qualified symbolic controller. A separate teacher-compression study may use the PPO expert from episode reset; do not assume it can recover arbitrary learner states or reuse failed takeover labels. Report any neural outputs that remain. Greedy group order can fail; freeze the ordering or budget its selection explicitly.

## Shared evaluation and result contract

- Stage 3 remains 60 Hz, 360 steps. Sustained success means no fall and the centered/absolute-spin predicate on all final 60 steps. Report failures including slow spin, off-centering and drops.
- Pick seeds from the live ledger, accounting for reserved manifests; no hardcoded new seeds or R-ids here. Do not reuse R-102's confirmation cases for tuning.
- Freeze evaluation batches and use one whole manifest per fresh Isaac process. Keep batch shape and case layout fixed across paired arms. Repeated training resets do not replace the established evaluator.
- Keep training, checkpoint selection and confirmation disjoint. Choose confirmation size prospectively from pilot discordance for a 5-percentage-point effect; report case-paired and manifest-level uncertainty. No repeated confirmation until the interval crosses zero.
- Preserve the repository's strict success-promotion rule. For drop qualification, require the one-sided 95% upper bound on challenger-minus-reference drop rate to be <= 0.05. Distinguish evidence of a positive effect from evidence that its magnitude exceeds 0.05.
- Record `paired_success_difference` for the declared primary contrast, intervals, per-seed/per-block scores, reference comparisons, wins/losses/ties, drop differences/bounds, tail-hit fraction, train/validation curves, proposed/executed residual norms by joint, saturation, symbolic complexity, simulator transitions/episodes, wall time, optimizer updates and LLM calls/tokens by role. Diagnostic fields need not become prediction names.
- Retain per-case records, hashes, full config and reconstructable checkpoints. Write numbers to `research_result.json` under `metrics` and `controls`, with locally available artifact paths and deviations in `notes`.
- Freeze scientific assumptions and costs through triage. Include new module paths in `protocol.code_changes`; use the existing dispatcher `scripts/run_shadowhand_spin_stages.py` and `config/shadowhand_spin_stages.yaml` where appropriate. New study registry keys are implementation work, not existing commands. A protocol is runnable only after the script/config/manifests and study handler exist. Do not weaken checks to make planned files pass.
- Add only necessary runner/optimizer/interface tests. Use the normal GPU environment and operational workflow. Never touch locked evaluation splits or modify historical decisions.
- After each agreed decision: regenerate state, append handoff through tooling, commit and push code plus ledger and small result summaries together. Report what was decided, what is running and the next dependency. Stop at the existing escalation/exhaustion conditions or when the user interrupts; do not ask for approval of every ordinary idea.

## Literature pointers

- [Bosio et al., LLMs + gradient-free controller optimization](https://arxiv.org/abs/2510.00373): outer structural synthesis, inner numerical learning.
- [Code to Control](https://arxiv.org/abs/2609.38733): synthesized features with ARS-fitted action mapping; no matched MuJoCo interaction-efficiency claim.
- [Deep Symbolic Policy, ICML 2021](https://proceedings.mlr.press/v139/landajuela21a.html): rollout-trained expression generator and progressive action anchoring.
- [CEM-RL, ICLR 2019](https://arxiv.org/abs/1810.01222): population search plus actor-critic improvement; neural-policy evidence, not a direct symbolic-hand result.
- [PROPEL, NeurIPS 2019](https://arxiv.org/abs/1907.05431): improve a mixed policy and project to programs; projection is a separate potential bottleneck.
- [Memory-based symbolic GP, 2026](https://link.springer.com/article/10.1007/s10710-025-09530-9): evolve compact memory and symbolic readouts on dynamical-system benchmarks.

These motivate hypotheses. None establishes that the proposed method solves this exact task.
