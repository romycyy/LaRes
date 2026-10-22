# Research state

_Generated 2026-10-22T04:44:38 by `python scripts/research.py state` from `research/ideas/` and `research/decisions.jsonl`. Do not edit by hand._

| status | count |
|---|---|
| escalated | 0 |
| analysed | 0 |
| running | 0 |
| approved | 0 |
| proposed | 3 |
| decided | 7 |
| rejected | 2 |

## Needs attention

- **R-003** [proposed] Rotating the development screen removes selection on fixed cases
  - USER decides: `python scripts/research.py approve R-003` or `reject R-003 --reason ...`
- **R-009** [proposed] Test LLM bounds-only edits against matched random bounds
  - USER decides: `python scripts/research.py approve R-009` or `reject R-009 --reason ...`
- **R-012** [proposed] Transfer geometry-only mutation beyond damped push
  - USER decides: `python scripts/research.py approve R-012` or `reject R-012 --reason ...`

## Last 5 decisions

- D-007 · R-011 · **refuted** · 2026-10-22
  R-011 pooled 15 candidates per structure (original, 3 local and 4 global search draws, 7 imitation checkpoints) under one rollout probe and compared the pick against equal-budget search-only (15 draws) and imitation-only (15 checkpoints) selectors on four frozen weak evolved structures (probe s2912, confirmation s2913). Pooled minus search-only -0.055, 97.5% [-0.110, +0.000]; pooled minus imitation-only +0.000 [-0.033, +0.033]: the predicted +0.05 against both is refuted. Search's lead rests on one lucky global draw on cand2_a1 (0.56) outside the pooled subset; imitation wins cand1_a0; the two additional structures stay near zero under every selector. Parameter optimisation does not rescue weak zero-shot evolved structures at this budget.
  Next: Prefer structural changes known to work (geometry-scope mutation, D-006) over optimiser mixing; test their transfer to other parents (R-012).
- D-006 · R-010 · **supported** · 2026-10-22
  R-010 refitted the LLM geometry-scope child gen0_cand5_a0 and its parent simple_damped_push under five fitting seeds (2900-2904; probe s2905) and scored both on 50 fresh cases (s2906). The child reached 1.00 under every seed, the parent 0.88-0.92: child minus parent +0.104, placement-level 95% [+0.023, +0.185], seed-level [+0.083, +0.125], 5/0/0. The advantage does not depend on one fit. Report only: the child passes the unchanged qualification rule (0.00 untrained, 1.00 fitted), but the promotion-set (development 10-39) comparison was not run, so bank admission is not yet justified.
  Next: Run the development 10-39 comparison (or a qualification run) before admitting gen0_cand5_a0 as a fittable_seed; test geometry-scope transfer to other parents (R-012).
- D-005 · R-008 · **inconclusive** · 2026-10-22
  R-008 compared direct rollout search (original + 3 local + 4 global draws) against rollout-selected imitation (8 checkpoints to 1000 steps), each with 80 probe episodes, from the frozen original parameters of the two weak evolved Stage F structures (seed 2812, split 2813, probe s2810, confirmation s2811). Search minus imitation: +0.080 [+0.002, +0.158]; the predicted +0.10 is neither supported nor refuted. The average hides a complete split: on baseline61_gen0_cand2_a1 one global draw reached 0.52 while imitation never left step 0 (0.00); on baseline61_gen0_cand1_a0 imitation reached 0.36 at step 125 while no search draw succeeded. The two optimisers fail on different structures; the probe agreed with confirmation in both.
  Next: Test pooling search candidates and imitation checkpoints under one rollout probe at the combined budget, over more weak structures.
- D-004 · R-007 · **refuted** · 2026-10-22
  R-007 refitted the frozen geometry-scope child gen0_cand5_a0 and its parent simple_damped_push identically (fit seed 2802, probe s2804), scored both on a fresh 30-case screen (s2800) and a 100-case confirmation set (s2801+s2805), and replayed Stage F's acceptance rule on 1000 ten-case subsets. A 30-case screen selects a better controller than a 10-case screen by only +0.0044 [+0.0019, +0.0069]: the ten-case screen keeps the child 96% of the time, rejecting it only on tied success. The predicted +0.02 is refuted; screen size is not what blocks good edits. The refitted child beats its parent by +0.110 [+0.048, +0.172], 11/0/89, replicating R-005's +0.14: this one LLM geometry mutation is a real improvement.
  Next: Consider admitting gen0_cand5_a0 to the fittable seed bank via the qualification rule; investigate why geometry-scope mutation works where correction-LLM repair does not.
- D-003 · R-006 · **refuted** · 2026-10-22
  R-006 added a local VLM evidence arm to correction-LLM repair: Qwen2.5-VL-7B-Instruct (pinned cc59489, NF4 language weights, fp16 compute, 7.55 GiB on the T4) analysed the parent's worst failed screen episodes (five rule-selected, upright frames each) and its validated, geometry-merged analysis was appended to the Stage F numeric packet. Nine independent cells (three frozen Stage F structures x bundles 2620-2628), held-out pool 2629, arms no_repair / llm_numeric / llm_numeric_vision at matched repair-episode caps. Vision minus numeric held-out success: +0.000 [+0.000, +0.000], 0/0/9; all arms 0.440. Every packet carried valid analyses (23 returned, 0 rejected, 3 stage conflicts), but all 18 LLM children scored at or below their refitted parent on the screen and were rejected, so neither evidence level changed any outcome. Predicted +0.05 refuted; localisation benefit untested because no edit survived refit. With Stage F and R-005, the correction LLM's structural edits, not its evidence, are the bottleneck.
  Next: Do not add more evidence to correction-LLM repair; test whether any LLM edit can survive refit, or redirect effort to parameter search.

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
- Spent seeds: [0, 1, 2, 1000, 1001, 1002, 2000, 2100, 2200, 2201, 2202, 2203, 2204, 2300, 2301, 2302, 2303, 2304, 2400, 2401, 2402, 2403, 2404, 2410, 2411, 2412, 2413, 2414, 2420, 2421, 2422, 2423, 2424, 2500, 2501, 2502, 2610, 2611, 2620, 2621, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 2700, 2701, 2702, 2703, 2704, 2705, 2710, 2800, 2801, 2802, 2803, 2804, 2805, 2810, 2811, 2812, 2813, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2910, 2911, 2912, 2913, 3000, 3001]

## Pointers

- Narrative findings: `RESEARCH_HANDOFF.md` (appended once per decision)
- Frozen history: `IMPLEMENTATION_PROGRESS.md`, `DEBUG_PROGRESS.md`
- Advisor calls: `research/llm_calls.jsonl`; briefings: `research/briefings/`
