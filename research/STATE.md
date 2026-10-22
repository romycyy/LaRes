# Research state

_Generated 2026-10-22T07:14:53 by `python scripts/research.py state` from `research/ideas/` and `research/decisions.jsonl`. Do not edit by hand._

| status | count |
|---|---|
| escalated | 0 |
| analysed | 0 |
| running | 3 |
| approved | 0 |
| proposed | 0 |
| decided | 15 |
| rejected | 3 |

## Needs attention

- **R-019** [running] 1. Test seedless generator transfer to grasp-and-transport
  - run the protocol, then `python scripts/research.py record R-019 --results <json>`
- **R-020** [running] 2. Test whether search needs grasp-and-lift-aware selection
  - run the protocol, then `python scripts/research.py record R-020 --results <json>`
- **R-021** [running] 3. Test explicit observation and gripper-action grounding
  - run the protocol, then `python scripts/research.py record R-021 --results <json>`

## Last 5 decisions

- D-015 · R-018 · **refuted** · 2026-10-22
  R-018 added an exploration-lane option (config/two_lane_search_v2.yaml): single-phase generation with code-derived idea records, treated by 24-candidate independent rollout search on the probe. Six paired runs per configuration (seeds 3100-3111), incumbents scored on per-block fresh manifests (pool seeds 3130-3135). v2 minus control incumbent success: -0.007, run-level [-0.035, +0.022]; the predicted +0.05 is refuted. Every incumbent in all twelve runs came from the unchanged fittable lane at 1.00 promotion success, so the task is saturated and exploration candidates could only tie. The new lane itself works: exploration screen success rose from 0.03 to 0.78 on average (best per run 0.10 to 0.97) with fewer LLM calls (10 vs 11-12).
  Next: Keep v2 as an option, not the default, on push-v2; evaluate the v2 exploration lane where no seed bank saturates the task (a harder MetaWorld task, or the fittable lane disabled), and confirm its screen winners on fresh cases.
- D-014 · R-017 · **supported** · 2026-10-22
  R-017 compared seedless generation by the single-phase prompt with the two-phase ideas-then-implementation generator the exploration lane and the gpt-6.1-sol baseline use, four blocks, one structure per generator per block (13 gpt-6.1-sol calls), each given 24-candidate independent rollout search under two seeds (probe s3046, confirmation s3047). Single-phase minus two-phase: +0.698, block-level [+0.395, +1.000]; single-phase won every block (0.44-1.00 vs 0.00-0.32). Untrained, two-phase scored 0.00 everywhere (as in the baseline run) and single-phase 0.00-0.90. Supported: the generator bundle (prompt wording plus design/implementation split), not phasing alone, is what the result isolates; part of the gain exists before search (+0.25).
  Next: Switch the exploration lane to single-phase generation plus 24-candidate rollout search and rerun the end-to-end search against the current configuration on a fresh confirmation set.
- D-013 · R-015 · **refuted** · 2026-10-22
  R-015 appended a parameterization contract (smooth sigmoid gates, unit-direction drives scaled by bounded gains, 6-10 bounded parameters) to the single-phase seedless generation prompt and compared it with the same prompt without the contract: three blocks, one gpt-6.1-sol structure per prompt (all valid first time), each given 24-candidate independent rollout search under two seeds (probe s3025, confirmation s3026). Contract minus baseline: -0.293, block-level [-0.655, +0.069], placement-level [-0.365, -0.221]; worse in every block, untrained and searched. The predicted +0.10 is refuted. Side observation: the plain single-phase prompt's structures scored 0.16-0.88 untrained and 0.80-1.00 after search, against 0.00 for the two-phase gpt-6.1-sol baseline run.
  Next: Test single-phase versus two-phase seedless generation under rollout search with more blocks; if it holds, switch the exploration lane's prompt and add rollout search.
- D-012 · R-016 · **inconclusive** · 2026-10-22
  R-016 compared progress-guided adaptive parameter search (two batches of 8 from a Gaussian over the top-3 candidates) with budget-matched independent local/global draws, 24 candidates per arm with the first 8 shared, on two weak evolved structures x two search seeds (5-case probe s3032, confirmation s3033). Adaptive minus independent: +0.070 placement-level [+0.038, +0.102], 14/0/36, but cell-level [-0.069, +0.209]; the predicted +0.08 is neither supported nor refuted. All gains are on baseline61_gen0_cand2_a1 (0.94 vs 0.76, 0.10 vs 0.00); gen0_cand1_a0 tied on a shared-batch pick. Notably, 24-candidate rollout search lifted these 0.00-zero-shot structures to 0.42-0.94 in three of four cells.
  Next: Confirm on more weak structures with a larger probe before changing the exploration lane; 24-candidate rollout search itself looks like a strong rescue for zero-shot structures.
- D-011 · R-014 · **refuted** · 2026-10-22
  R-014 clamped the geometry child's only added parameter (approach_standoff, fixed at its declared 0.02) and compared it with the fully trainable child under identical fitting (seeds 3010/3011, probe s3012, confirmation s3013). Both reach 1.00 under both seeds: +0.000 [0, 0], 0/0/50. The predicted +0.05 is refuted: the child's advantage over simple_damped_push (D-006) comes from its changed forward geometry with a fixed 0.02 standoff, not from fitting the added dimension.
  Next: A hand-written simple_damped_push variant with a fixed 0.02 standoff would reproduce the gain without an LLM; R-016 (adaptive search) is next.

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
- Spent seeds: [0, 1, 2, 1000, 1001, 1002, 2000, 2100, 2200, 2201, 2202, 2203, 2204, 2300, 2301, 2302, 2303, 2304, 2400, 2401, 2402, 2403, 2404, 2410, 2411, 2412, 2413, 2414, 2420, 2421, 2422, 2423, 2424, 2500, 2501, 2502, 2610, 2611, 2620, 2621, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 2700, 2701, 2702, 2703, 2704, 2705, 2710, 2800, 2801, 2802, 2803, 2804, 2805, 2810, 2811, 2812, 2813, 2820, 2822, 2823, 2824, 2825, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2910, 2911, 2912, 2913, 2920, 2921, 2922, 2923, 2940, 3000, 3001, 3010, 3011, 3012, 3013, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3030, 3031, 3032, 3033, 3040, 3041, 3042, 3043, 3044, 3045, 3046, 3047, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3130, 3131, 3132, 3133, 3134, 3135]

## Pointers

- Narrative findings: `RESEARCH_HANDOFF.md` (appended once per decision)
- Frozen history: `IMPLEMENTATION_PROGRESS.md`, `DEBUG_PROGRESS.md`
- Advisor calls: `research/llm_calls.jsonl`; briefings: `research/briefings/`
