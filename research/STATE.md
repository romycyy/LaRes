# Research state

_Generated 2026-10-22T10:21:37 by `python scripts/research.py state` from `research/ideas/` and `research/decisions.jsonl`. Do not edit by hand._

| status | count |
|---|---|
| escalated | 0 |
| analysed | 0 |
| running | 1 |
| approved | 1 |
| proposed | 1 |
| decided | 29 |
| rejected | 4 |

## Needs attention

- **R-034** [running] Transport expert-trace edits across assembly parent structures
  - run the protocol, then `python scripts/research.py record R-034 --results <json>`
- **R-035** [approved] Find another recoverable terminal-phase plateau
  - `python scripts/research.py start R-035` then run its command
- **R-036** [proposed] Plateau-triggered trace repair versus exploration restart
  - USER decides: `python scripts/research.py approve R-036` or `reject R-036 --reason ...`

## Last 5 decisions

- D-029 · R-033 · **inconclusive** · 2026-10-22
  R-033 tried to test trace-guided editing on a fresh hammer-v2 plateau parent, but the plateau gate failed: 24-candidate search of the fresh structure already produced probe successes, so no handoff or edit stage ran (inconclusive, 120 episodes, 1 call). Hammer is not a plateau for single-phase generation plus search; assembly-v2 remains the only plateau found.
  Next: Scan shelf-place, basketball and soccer for another plateau, or test from a fresh assembly parent.
- D-028 · R-032 · **supported** · 2026-10-22
  R-032 replaced D-027's learner handoff traces with expert-only demonstrations (10/10 successful; rows from when the nut first rose 3 cm). Edits of the recoverable assembly-v2 parent that saw two such traces scored 1.00 on 30 fresh cases at their declared values (no search) in both blocks; summary-only edits scored 0.00: +1.000 [1.0, 1.0], supported. Learner-state alignment is not needed: plain expert demonstrations carry the placement geometry the edit lacks.
  Next: Test transfer to hammer (R-033) and from a fresh generated parent; integrate trace-conditioned editing as a repair step.
- D-027 · R-031 · **supported** · 2026-10-22
  R-031 repeated R-029 with the pipeline's standard single validity correction and ten learner-to-expert handoff demonstrations (6/10 successful continuations). In both blocks the edit of the recoverable D-020 assembly-v2 parent that saw two successful expert placement traces scored 1.00 on 30 fresh promotion cases, while the summary-only edit scored 0.00: +1.000 [1.0, 1.0], 30/0/0, supported. The trace edits already succeed untrained (probe 1.0 at step 0). Three of four edits needed their one correction, which is why R-029 never tested the traces. The code reads observations through the schema; constants are centimetre-scale offsets. Narrow: one parent, one demonstration set, two blocks.
  Next: Generalise trace-conditioned editing: another parent or fresh structure, another failing task, and as a repair step in the two-lane pipeline where an expert exists.
- D-026 · R-030 · **inconclusive** · 2026-10-22
  R-030 planned an expert-calibrated contrast of learner success on impact tasks (hammer, soccer) against insertion (peg-insert-side). MetaWorld's scripted expert scored 1.00 on hammer, 0.90 on soccer and only 0.63 on peg-insert-side on the frozen screen cases, failing the predeclared >= 0.90 gate, so no structure was generated: inconclusive, 90 episodes, 0 LLM calls.
  Next: Replace the insertion control with a task whose expert is reliable, or scan hammer and soccer as further testbeds.
- D-025 · R-029 · **refuted** · 2026-10-22
  R-029 compared assembly-v2 placement edits of the recoverable D-020 parent given the handoff summary alone or the summary plus a successful expert continuation trace (only 1 of 5 handoffs succeeded), in two blocks under identical 24-candidate search. Both summary-only edits were valid and scored 0.00; both trace edits failed validation on gate_not_a_lever (unused post-lift gates) and, with no correction allowed, score 0.00. Trace minus summary-only: +0.000 [0, 0]; refuted for the procedure as specified, while the trace mechanism itself was never evaluated.
  Next: Rerun with one validity correction and more successful continuations if trace evidence is still worth testing; otherwise move to the remaining hard tasks.

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
- Spent seeds: [0, 1, 2, 1000, 1001, 1002, 2000, 2100, 2200, 2201, 2202, 2203, 2204, 2300, 2301, 2302, 2303, 2304, 2400, 2401, 2402, 2403, 2404, 2410, 2411, 2412, 2413, 2414, 2420, 2421, 2422, 2423, 2424, 2500, 2501, 2502, 2600, 2601, 2602, 2603, 2604, 2605, 2606, 2607, 2610, 2611, 2612, 2613, 2614, 2615, 2616, 2617, 2618, 2619, 2620, 2621, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 2700, 2701, 2702, 2703, 2704, 2705, 2710, 2800, 2801, 2802, 2803, 2804, 2805, 2810, 2811, 2812, 2813, 2820, 2822, 2823, 2824, 2825, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2910, 2911, 2912, 2913, 2920, 2921, 2922, 2923, 2940, 3000, 3001, 3010, 3011, 3012, 3013, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3030, 3031, 3032, 3033, 3040, 3041, 3042, 3043, 3044, 3045, 3046, 3047, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3130, 3131, 3132, 3133, 3134, 3135, 3200, 3201, 3202, 3205, 3210, 3211, 3212, 3213, 3214, 3215, 3220, 3221, 3222, 3223, 3300, 3301, 3302, 3303, 3304, 3305, 3310, 3311, 3312, 3313, 3314, 3315, 3320, 3321, 3330, 3331, 3332, 3333, 3334, 3335, 3400, 3410, 3420, 3421, 3422, 3500, 3501, 3502, 3503, 3505, 3506, 3507, 3520, 3521, 3522, 3523, 3524, 3525, 3526, 3527, 3528, 3529, 3540, 3541, 3544, 3600, 3601, 3602, 3603, 3604, 3605, 3606, 3607, 3608, 3609, 3710, 3711, 3712, 3713, 3720, 3721, 3722, 3723, 3724, 3725, 3726]

## Pointers

- Narrative findings: `RESEARCH_HANDOFF.md` (appended once per decision)
- Frozen history: `IMPLEMENTATION_PROGRESS.md`, `DEBUG_PROGRESS.md`
- Advisor calls: `research/llm_calls.jsonl`; briefings: `research/briefings/`
