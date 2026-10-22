# Research state

_Generated 2026-10-22T07:57:41 by `python scripts/research.py state` from `research/ideas/` and `research/decisions.jsonl`. Do not edit by hand._

| status | count |
|---|---|
| escalated | 0 |
| analysed | 0 |
| running | 1 |
| approved | 0 |
| proposed | 1 |
| decided | 18 |
| rejected | 3 |

## Needs attention

- **R-022** [running] 1. Scan eight tasks for failures of seedless generation plus search
  - run the protocol, then `python scripts/research.py record R-022 --results <json>`
- **R-023** [proposed] 2. Test structural breadth instead of deeper parameter search
  - USER decides: `python scripts/research.py approve R-023` or `reject R-023 --reason ...`

## Last 5 decisions

- D-018 · R-019 · **inconclusive** · 2026-10-22
  R-019 compared single-phase and two-phase seedless generation on pick-place-v2 under identical 24-candidate rollout search (four blocks, shared controls with R-020/R-021). Single minus two-phase: +0.283 block-level [-0.480, +1.046], inconclusive: one two-phase structure found no successful candidate (1.00 vs 0.00) while the other blocks differ by at most 0.10. The suite's main finding: with no seed bank, single-phase gpt-6.1-sol structures plus 24-candidate search solve pick-place-v2 (8/8 at 1.00 on 30 fresh cases; 7/8 succeed untrained).
  Next: Find a task where single-phase zero-shot fails before testing generator or selector variants further; consider the D-014 recipe as the seedless default.
- D-017 · R-021 · **refuted** · 2026-10-22
  R-021 appended a source-verified interface table (action scaling, gripper polarity, observation meanings, read from MetaWorld 3 source) to the single-phase pick-place-v2 prompt and compared it with the unchanged prompt under identical 24-candidate search, four paired blocks. Grounded minus control: +0.000 [0, 0]; the predicted +0.10 is refuted at a ceiling: both arms reach 1.00 after search, and untrained the plain prompt was if anything better (4/4 vs 3/4 structures succeeding at step 0).
  Next: Keep the schema-only prompt; do not add the interface table.
- D-016 · R-020 · **refuted** · 2026-10-22
  R-020 (pick-place-v2 shared-control suite) re-ranked the same 24-candidate search pools of four single-phase structures by success, then grasp-and-lift rate, then progress, against the original success-then-progress selector. Lift-aware minus original: +0.000 [0, 0]; the predicted +0.10 is refuted. Ceiling: every pool already held fully successful candidates and both selectors picked the same one, so the lift key was never consulted.
  Next: Selection is not the bottleneck when pools contain successes; revisit only where successes are sparse.
- D-015 · R-018 · **refuted** · 2026-10-22
  R-018 added an exploration-lane option (config/two_lane_search_v2.yaml): single-phase generation with code-derived idea records, treated by 24-candidate independent rollout search on the probe. Six paired runs per configuration (seeds 3100-3111), incumbents scored on per-block fresh manifests (pool seeds 3130-3135). v2 minus control incumbent success: -0.007, run-level [-0.035, +0.022]; the predicted +0.05 is refuted. Every incumbent in all twelve runs came from the unchanged fittable lane at 1.00 promotion success, so the task is saturated and exploration candidates could only tie. The new lane itself works: exploration screen success rose from 0.03 to 0.78 on average (best per run 0.10 to 0.97) with fewer LLM calls (10 vs 11-12).
  Next: Keep v2 as an option, not the default, on push-v2; evaluate the v2 exploration lane where no seed bank saturates the task (a harder MetaWorld task, or the fittable lane disabled), and confirm its screen winners on fresh cases.
- D-014 · R-017 · **supported** · 2026-10-22
  R-017 compared seedless generation by the single-phase prompt with the two-phase ideas-then-implementation generator the exploration lane and the gpt-6.1-sol baseline use, four blocks, one structure per generator per block (13 gpt-6.1-sol calls), each given 24-candidate independent rollout search under two seeds (probe s3046, confirmation s3047). Single-phase minus two-phase: +0.698, block-level [+0.395, +1.000]; single-phase won every block (0.44-1.00 vs 0.00-0.32). Untrained, two-phase scored 0.00 everywhere (as in the baseline run) and single-phase 0.00-0.90. Supported: the generator bundle (prompt wording plus design/implementation split), not phasing alone, is what the result isolates; part of the gain exists before search (+0.25).
  Next: Switch the exploration lane to single-phase generation plus 24-candidate rollout search and rerun the end-to-end search against the current configuration on a fresh confirmation set.

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
- Spent seeds: [0, 1, 2, 1000, 1001, 1002, 2000, 2100, 2200, 2201, 2202, 2203, 2204, 2300, 2301, 2302, 2303, 2304, 2400, 2401, 2402, 2403, 2404, 2410, 2411, 2412, 2413, 2414, 2420, 2421, 2422, 2423, 2424, 2500, 2501, 2502, 2610, 2611, 2620, 2621, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 2700, 2701, 2702, 2703, 2704, 2705, 2710, 2800, 2801, 2802, 2803, 2804, 2805, 2810, 2811, 2812, 2813, 2820, 2822, 2823, 2824, 2825, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2910, 2911, 2912, 2913, 2920, 2921, 2922, 2923, 2940, 3000, 3001, 3010, 3011, 3012, 3013, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3030, 3031, 3032, 3033, 3040, 3041, 3042, 3043, 3044, 3045, 3046, 3047, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3130, 3131, 3132, 3133, 3134, 3135, 3200, 3201, 3202, 3205, 3210, 3211, 3212, 3213, 3214, 3215, 3220, 3221, 3222, 3223]

## Pointers

- Narrative findings: `RESEARCH_HANDOFF.md` (appended once per decision)
- Frozen history: `IMPLEMENTATION_PROGRESS.md`, `DEBUG_PROGRESS.md`
- Advisor calls: `research/llm_calls.jsonl`; briefings: `research/briefings/`
