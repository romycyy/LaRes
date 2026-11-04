# Session summary: tuning, lesioning and repairing the Shadow Hand pen-spin learner (2026-11-04)

This document summarises one research-loop session (R-056 to R-091, decisions D-048 to D-063). The authoritative
record is the ledger: `research/ideas/R-0xx.json`, `research/decisions.jsonl`, the decision blocks in
`RESEARCH_HANDOFF.md`, and each run's `logs/pen_spin/<run>/research_result.json`. Numbers here are copied from those
files; if they disagree, the ledger wins.

## 1. Direction and what was asked

Resume from `docs/SESSION_2026-10-27_PEN_SPIN_SUMMARY.md`. First compare CMA-ES and random search at equal rollout
budgets on the frozen R-046 controller's spin and centring parameters, both winners against the parent on fresh paired
cases, reporting sustained success, drops and saturation. If tuning improves success without more drops, use the
winner for repair; if feedback activates without improving success, prioritise R-043. Defer cold search and R-055;
require verified learner-state recovery before imitation. llm_pool with 16 workers, high reasoning effort. One
protocol at a time; commit and push each decision to staging. Final test locked throughout.

The loop ran under the advisor-triage workflow with gpt-6.1-sol as advisor. The final-test split was never opened.

## 2. Bottom line

1. **Scalar tuning of the frozen R-046 learner is closed.** Budget-matched CMA-ES and random search (8 blocks, 64
   candidates per arm, 12,000 episodes) gave +0.003 [-0.076, +0.082] between the two on fresh cases, and neither winner
   differed from the parent (D-048). Re-ranking the archived top candidates on 40-case screens gave -0.002 [-0.043,
   +0.040] (D-049): the ten-case screen winners' scores were selection noise (0.352 on their screen, 0.221 on new
   cases, parent level on fresh).
2. **The learner's capture curl was its defect.** A lesion panel (D-051) found that zeroing the reactive capture curl
   (the term that tightens the grasp when contact or centring deteriorates) cut drops 13 to 5 and off-centre failures
   18 to 3 on 80 cases. The exact clamp then beat the parent on three fresh panels (+0.150, +0.069, +0.075), replaced
   compression by a constant preload failed (D-054), retiming the curl was worse than both (D-055), and after four
   30-case promotion panels each landed within 0.0004 to 0.02 of the strict rule (D-052, D-054 to D-056), a
   predeclared 90-pair instrument qualified it: +0.167 [+0.040, +0.293], drops 4 vs 10 (D-057).
3. **A stronger open-loop stroke is the one repair that transferred.** Instantaneous spin feedback on the stroke's
   amplitude lost to its own constant-gain control (D-058), but that control (the clamp with its periodic stroke scaled
   by a fixed 1.408) beat the qualified clamp on both panels (+0.133 promotion, +0.150 fresh, equal drops) and was
   adopted as the incumbent (D-059), frozen with hashes under `config/pen_spin/d058_fixed_stroke_incumbent/`.
4. **Nothing further qualified on the incumbent.** Anonymous-coordinate LLM residuals were too small to matter in either
   basis (D-060, R-043 branch); phase-lead feedback and constant lead offsets were flat (D-061); a within-cycle warp of
   the stroke phase gained +0.0875 [+0.015, +0.160] on one fresh panel (D-062) but -0.022 [-0.105, +0.061] on the
   90-pair instrument (D-063). The incumbent stands at roughly 0.35 to 0.44 fresh success with few drops; its failure
   mode is slow spin.
5. **The advisor abstained twice in a row** (calls 00178 and 00180): the direction's sequence is complete. The loop
   guard now counts zero-idea propose calls, so `propose` refuses until the user sets a new direction.

## 3. Ideas and decisions, in order

| Idea | Decision | Outcome | Question | Key number |
|---|---|---|---|---|
| R-058 | D-048 | refuted | Budget-matched CMA-ES vs random search on 12 scalars | +0.003 [-0.076, +0.082] fresh over 8 blocks; neither beats the parent |
| R-060 | D-049 | refuted | Re-rank the archive's top 8 per arm on 40-case screens | -0.002 [-0.043, +0.040]; selection inflation 0.352 to 0.221 to parent level |
| R-062 | D-050 | inconclusive | Does the periodic stroke generate the spin? | lesion -0.088 [-0.168, -0.007]; the pen still spins at the same mean rate |
| R-065 | D-051 | refuted | Which reactive channel carries residual spin? | rotation correction inert; capture-curl clamp +0.125 [-0.014, +0.264], drops 13 to 5 |
| R-068 | D-052 | supported | Confirm the zero-curl clamp on fresh cases | +0.150 [+0.052, +0.248] (52 vs 28 of 160), drops 21 to 7; promotion missed by 0.006 |
| R-071 | D-053 | refuted | Does curl removal cause spin reversals? | +0.16 reversals/s [+0.05, +0.27], below the 0.5 threshold |
| R-073 | D-054 | refuted | Constant preload instead of reactive curl | 0.000 [-0.086, +0.086] vs the clamp, twice the drops |
| R-075 | D-055 | refuted | 100 ms low-pass on the curl | -0.094 [-0.170, -0.018] vs the parent; more drops |
| R-077 | D-056 | inconclusive | Qualify the clamp on a fresh 30-case panel | +0.233 [-0.0004, +0.467] |
| R-080 | D-057 | supported | Qualify the clamp on a 90-pair instrument | +0.167 [+0.040, +0.293], drops 4 vs 10; repair parent |
| R-081 | D-058 | refuted | Spin feedback on the stroke vs constant gain | feedback -0.163 [-0.271, -0.054]; the constant gain beats the parent +0.150 fresh |
| R-083 | D-059 | supported | Adopt the fixed-stroke controller (zero spend) | incumbent frozen with hashes |
| R-084 | D-060 | inconclusive | R-043: anatomical vs joint residual coordinates | -0.025 [-0.104, +0.054]; residual norms 0.02 of a 0.15 bound |
| R-064 | D-061 | refuted | Phase-lead feedback vs constant lead offsets | -0.025 [-0.112, +0.062]; twelve settings flat |
| R-086 | D-062 | inconclusive | Within-cycle warp of the stroke phase | +0.0875 [+0.015, +0.160] fresh; promotion missed by 0.014 |
| R-089 | D-063 | inconclusive | Qualify the warp on a 90-pair instrument | -0.022 [-0.105, +0.061]; not adopted |

Rejected at triage: R-059 (an associational audit; answered offline), R-074 (same), R-091 (a stop rule, not an
experiment), R-090 (cancelled by its own condition). Deferred: R-056, R-057, R-061, R-063, R-066, R-067, R-069,
R-070, R-072, R-076, R-078, R-079, R-082, R-085, R-087, R-088, and the earlier R-043, R-052, R-055.

## 4. Findings worth carrying forward

- **The learner's reactive grasp tightening loses the pen.** Removing it is the single largest transferable gain found
  on this learner. Static compression keeps spin but drops more; delayed compression is worse than both.
- **Open-loop stroke amplitude is the lever; feedback on it is not.** Amplitude feedback (D-058), phase-lead feedback
  (D-061) and anonymous residuals (D-060) all failed; a fixed 1.41 gain won. The learner's roll direction differs
  across cases, so any signed-rate target is ill-posed (D-061).
- **Thirty-pair promotion panels cannot resolve effects of 0.1 to 0.2 at this base rate.** Four candidates landed
  within 0.0004 to 0.02 of the strict lower bound; the 90-pair instrument (three standard panels, the same rule on the
  pooled differences) resolved two of them in opposite directions (D-057 qualified, D-063 did not).
- **Fresh panels can mislead on their own.** D-062's +0.0875 [+0.015, +0.160] did not replicate (D-063).
- **Seed bookkeeping.** A frozen manifest reserves its seed for the idea that names it; `check_idea`, the briefing and
  the state page now distinguish reserved from consumed seeds (R-077 round 3).

## 5. Infrastructure added (all under `lares/envs/isaac_pen_spin/` unless stated)

| Module | What it does | Study |
|---|---|---|
| `scalar_tuning.py` | pycma CMA-ES vs never-told fresh-instance random search, tiled screening, saturation diagnostics | `r058_scalar_cma_vs_random` |
| `archive_rescreen.py` | re-rank archived candidates; offline clipping diagnostic | `r060_archive_rescreen` |
| `stroke_lesion.py`, `channel_lesions.py` | audited source-substitution lesions with offline bit-identical gates | `r062_stroke_lesion`, `r065_channel_lesions` |
| `curl_confirmation.py`, `clamp_qualification.py` | exact-clamp confirmation; the 30-pair and 90-pair instruments | `r068_…`, `r077_…`, `r080_…` |
| `axial_reversals.py`, `scripts/isaac/axial_reversals.py` | offline reversal audit | `R-071` |
| `constant_preload.py`, `curl_bandwidth.py` | preload and low-pass curl variants | `r073_…`, `r075_…` |
| `stroke_feedback.py` | gain on the stroke (feedback, constant, identity) | `r081_stroke_feedback` |
| `coordinate_repair.py` (+ `ParentScaffold` schema, `parent_residual` actor kind) | LLM residuals in orthonormal coordinates on a frozen parent | `r084_coordinate_repair` (run_full_evolution) |
| `lead_feedback.py`, `phase_warp.py`, `warp_qualification.py` | phase-lead feedback; within-cycle warp; its 90-pair instrument | `r064_…`, `r086_…`, `r089_…` |
| `scripts/isaac/adopt_fixed_stroke.py` | verifies the frozen D-058 incumbent asset | `R-083` |
| `lares/research/protocol.py`, `state.py`, `advisor.py` | reserved-vs-consumed seeds; zero-idea abstentions counted by the loop guard | — |

**Frozen assets.** `config/pen_spin/d058_fixed_stroke_incumbent/` (parent source and state, evaluated source,
constant, `incumbent.json` with hashes). Manifests `config/manifests/pen_spin_stage3_r0xx_*.yaml`. pycma 4.5.0 is
installed in both box venvs and in `lares-local` (not checked in).

## 6. How to resume

```bash
conda run -n lares-local python scripts/research.py state     # the guard blocks propose until a new direction
python scripts/research.py direct "..."                        # the user's new direction
gpu start && gpu test tests.test_pen_spin tests.test_pen_spin_pipeline tests.test_research
```

The box was stopped at the end of this session. Every study resumes from its own log directory (cells are never rerun).
