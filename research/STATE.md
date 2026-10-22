# Research state

_Generated 2026-10-22T05:02:59 by `python scripts/research.py state` from `research/ideas/` and `research/decisions.jsonl`. Do not edit by hand._

| status | count |
|---|---|
| escalated | 0 |
| analysed | 0 |
| running | 0 |
| approved | 1 |
| proposed | 0 |
| decided | 9 |
| rejected | 3 |

## Needs attention

- **R-009** [approved] Test LLM bounds-only edits against matched random bounds
  - `python scripts/research.py start R-009` then run its command

## Last 5 decisions

- D-009 · R-013 · **inconclusive** · 2026-10-22
  R-013 rebuilt the predeclared seed-2900 fitted states of the geometry child gen0_cand5_a0 and its parent simple_damped_push by deterministic refit (no rollouts; both parameter hashes match R-010's records) and scored both on the promotion set (development 10-39). Child 1.000, parent 0.967: +0.033 [-0.035, +0.102], 1/0/29. The interval spans zero, so the bank-admission requirement (strictly positive lower bound) is not met, and the refutation condition is not met either. The parent fails only one promotion case, a ceiling that 30 cases cannot resolve; the child's advantage stands on the fresh-case evidence of D-004 and D-006.
  Next: Bank admission of gen0_cand5_a0 needs a ruling on whether fresh-case evidence may substitute for the promotion-set rule; R-009 (LLM bounds-only edits) is approved and next.
- D-008 · R-012 · **refuted** · 2026-10-22
  R-012 asked the LLM (gpt-6.1-sol, 9 calls) for code-only mutations of two reach-push seed-bank parents, two geometry-only and two gain/damping-only per parent, all valid, and fitted each child and parent identically under two seeds (probe s2922, confirmation s2923). Geometry minus gain/damping: -0.237 [-0.315, -0.160]; geometry minus parent: -0.193 [-0.234, -0.151]; gain/damping minus parent: +0.045 [-0.019, +0.109]. Three of four geometry children damaged a working controller (e.g. sharp_gate 0.98 -> 0.54). The damped-push geometry win (D-004, D-006) does not transfer as a scope rule; caveat: these edits were code-only, while the winning child added parameters.
  Next: Treat gen0_cand5_a0 as a single validated structure; run its cheap bank-admission check; do not spend further on scope rules.
- D-007 · R-011 · **refuted** · 2026-10-22
  R-011 pooled 15 candidates per structure (original, 3 local and 4 global search draws, 7 imitation checkpoints) under one rollout probe and compared the pick against equal-budget search-only (15 draws) and imitation-only (15 checkpoints) selectors on four frozen weak evolved structures (probe s2912, confirmation s2913). Pooled minus search-only -0.055, 97.5% [-0.110, +0.000]; pooled minus imitation-only +0.000 [-0.033, +0.033]: the predicted +0.05 against both is refuted. Search's lead rests on one lucky global draw on cand2_a1 (0.56) outside the pooled subset; imitation wins cand1_a0; the two additional structures stay near zero under every selector. Parameter optimisation does not rescue weak zero-shot evolved structures at this budget.
  Next: Prefer structural changes known to work (geometry-scope mutation, D-006) over optimiser mixing; test their transfer to other parents (R-012).
- D-006 · R-010 · **supported** · 2026-10-22
  R-010 refitted the LLM geometry-scope child gen0_cand5_a0 and its parent simple_damped_push under five fitting seeds (2900-2904; probe s2905) and scored both on 50 fresh cases (s2906). The child reached 1.00 under every seed, the parent 0.88-0.92: child minus parent +0.104, placement-level 95% [+0.023, +0.185], seed-level [+0.083, +0.125], 5/0/0. The advantage does not depend on one fit. Report only: the child passes the unchanged qualification rule (0.00 untrained, 1.00 fitted), but the promotion-set (development 10-39) comparison was not run, so bank admission is not yet justified.
  Next: Run the development 10-39 comparison (or a qualification run) before admitting gen0_cand5_a0 as a fittable_seed; test geometry-scope transfer to other parents (R-012).
- D-005 · R-008 · **inconclusive** · 2026-10-22
  R-008 compared direct rollout search (original + 3 local + 4 global draws) against rollout-selected imitation (8 checkpoints to 1000 steps), each with 80 probe episodes, from the frozen original parameters of the two weak evolved Stage F structures (seed 2812, split 2813, probe s2810, confirmation s2811). Search minus imitation: +0.080 [+0.002, +0.158]; the predicted +0.10 is neither supported nor refuted. The average hides a complete split: on baseline61_gen0_cand2_a1 one global draw reached 0.52 while imitation never left step 0 (0.00); on baseline61_gen0_cand1_a0 imitation reached 0.36 at step 125 while no search draw succeeded. The two optimisers fail on different structures; the probe agreed with confirmation in both.
  Next: Test pooling search candidates and imitation checkpoints under one rollout probe at the combined budget, over more weak structures.

## Standing constraints

- LLM model for every pipeline call is gpt-6.1-sol with reasoning_effort high and temperature 1 (the only temperature the model accepts). _(CLAUDE.md, decided 2026-10-13)_
- Validation (imitation) loss correlates with development success at +0.19, the wrong sign; selecting on loss cannot work. Behavioural cloning is non-monotonic in gradient steps: success peaks near 1000 steps and is zero by 3000. _(RESEARCH_HANDOFF.md, Phase 5)_
- Imitation fitting is disabled by default (bc_steps 0). If fitting is turned on it must be rollout-selected on a probe manifest disjoint from the screen; fixed-budget fitting is harmful. _(Stage E, logs/stage_e/20260914_032500; config/run_full_evolution.yaml)_
- Promotion requires a strictly positive paired success difference whose paired t interval lower bound is above zero on the promotion set (development cases 10-39). Tied success never promotes. _(lares/eval/promotion.py, decided 2026-09-14)_
- The saved incumbent scores 0.000 success on the development manifest, not 0.30. _(RESEARCH_HANDOFF.md, 'The incumbent does not reproduce')_
- Only the five fittable_seed structures (simple_reach_push_sharp_gate, simple_damped_push, simple_reach_push, simple_reach_push_damped, simple_reach_push_shared_gain) may be fittable-lane parents; the standoff family is a non-fittable control or retired. _(config/fittable_bank/structure_roles.yaml)_
- Stage F: no correction-LLM repair arm beat no_repair; search arms led, LLM arms were flat. _(RESEARCH_HANDOFF.md 'Stage F: correction-LLM repair does not beat no repair (2026-10-13)')_
- The gpt-6.1-sol baseline rerun (nine unseeded zero-shot candidates) reached zero success; the gpt-5.6-sol run of the same config reached 0.333 on the promotion set. _(logs/baseline_gpt61_sol_high, logs/baseline_gpt56_sol_high; RESEARCH_HANDOFF.md 2026-10-13)_
- The AC-6 repair ablation did not support the repair loop: optimizer-only 0.650 vs full loop 0.517 held-out success over six replicates. _(RESEARCH_HANDOFF.md, 'The repair ablation (AC-6, E6, E9)')_
- The final-test split (config/manifests/push-v2_final_test.yaml) is locked until Phase 7, which needs the user's approval and a FreezeRecord. Qwen/E10 is also blocked. _(lares/eval/freeze.py; CLAUDE.md)_
- Seeds 2400-2424 are held for config/fittable_bank/qualification_v2.yaml (predeclared, not run). New protocols pick seeds from 2600 upward. _(config/experiment_seed_ledger.jsonl; CLAUDE.md)_
- Every protocol runs on the remote GPU box via `gpu run`, which bills by the hour; the implementer stops it afterwards. Live LLM or GPU spend needs the user's per-idea approval. _(CLAUDE.md, 'Remote GPU server')_
- Spent seeds: [0, 1, 2, 1000, 1001, 1002, 2000, 2100, 2200, 2201, 2202, 2203, 2204, 2300, 2301, 2302, 2303, 2304, 2400, 2401, 2402, 2403, 2404, 2410, 2411, 2412, 2413, 2414, 2420, 2421, 2422, 2423, 2424, 2500, 2501, 2502, 2610, 2611, 2620, 2621, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 2700, 2701, 2702, 2703, 2704, 2705, 2710, 2800, 2801, 2802, 2803, 2804, 2805, 2810, 2811, 2812, 2813, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2910, 2911, 2912, 2913, 2920, 2921, 2922, 2923, 2940, 3000, 3001]

## Pointers

- Narrative findings: `RESEARCH_HANDOFF.md` (appended once per decision)
- Frozen history: `IMPLEMENTATION_PROGRESS.md`, `DEBUG_PROGRESS.md`
- Advisor calls: `research/llm_calls.jsonl`; briefings: `research/briefings/`
