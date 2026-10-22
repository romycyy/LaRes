# Research state

_Generated 2026-10-22T03:07:58 by `python scripts/research.py state` from `research/ideas/` and `research/decisions.jsonl`. Do not edit by hand._

| status | count |
|---|---|
| escalated | 0 |
| analysed | 0 |
| running | 0 |
| approved | 0 |
| proposed | 2 |
| decided | 2 |
| rejected | 2 |

## Needs attention

- **R-003** [proposed] Rotating the development screen removes selection on fixed cases
  - USER decides: `python scripts/research.py approve R-003` or `reject R-003 --reason ...`
- **R-006** [proposed] Vision evidence from a local Qwen VLM makes correction-LLM repair beat numeric-only repair
  - USER decides: `python scripts/research.py approve R-006` or `reject R-006 --reason ...`

## Last 5 decisions

- D-002 · R-004 · **refuted** · 2026-10-22
  R-004 fitted the five seed-bank structures under five data-varying bundles (own demo sample, trajectory split, probe manifest and fitting seed; seeds 2400-2404, 2410-2414) and under a matched minibatch-only control (fixed sample/split 2700, probe 2710, fitting seeds 2701-2705), replicate r of both arms scored on the same qualification manifest (pool seed 2420+r). Between-replicate SE of bank-mean success: 0.0075 vs 0.0056, difference +0.0019, paired bootstrap 95% [-0.0034, +0.0054]; the predicted +0.01 is refuted and the direction is inconclusive. Bank-mean success 0.896 vs 0.911. Minibatch-only qualification did not materially understate bank-level uncertainty. One exception: simple_reach_push_shared_gain rejected fitting (step 0 selected) under 2 of 5 data draws, so its fitted success is data-sensitive (sd 0.088 vs 0.037).
  Next: Keep the seed bank as is; treat simple_reach_push_shared_gain as data-sensitive when reading single-run results.
- D-001 · R-005 · **refuted** · 2026-10-22
  R-005 refitted each of the six fittable-lane children of logs/two_lane_search and its seed-bank parent identically (fit seed 2611, same demos, deterministic_mse, budget 1000, checkpoints {0,500,1000} probe-selected on development 40-49) and scored all on 50 fresh cases (pool seed 2610). Child minus parent, averaged over pairs per placement: -0.020, t95 [-0.046, +0.006]; the predicted +0.05 is refuted, and the direction is inconclusive. Per pair: the geometry-scope mutation of simple_damped_push gained +0.14 (7/0/43), cand0 lost -0.16, two children tied their parents on every case. All 11 fits selected step 1000. The two-lane winner beats the zero-success gpt-6.1-sol baseline by +0.98 [+0.94, +1.02] (R-001), but it ties its own fitted parent, so that gain is the seed bank plus rollout-selected fitting, not the LLM mutation.
  Next: Treat seed bank + rollout-selected fitting as the operative method; test the LLM mutation step only where it showed a signal (geometry scope), or redirect LLM budget to repair with richer evidence.

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
- Spent seeds: [0, 1, 2, 1000, 1001, 1002, 2000, 2100, 2200, 2201, 2202, 2203, 2204, 2300, 2301, 2302, 2303, 2304, 2400, 2401, 2402, 2403, 2404, 2410, 2411, 2412, 2413, 2414, 2420, 2421, 2422, 2423, 2424, 2500, 2501, 2502, 2610, 2611, 2700, 2701, 2702, 2703, 2704, 2705, 2710, 3000, 3001]

## Pointers

- Narrative findings: `RESEARCH_HANDOFF.md` (appended once per decision)
- Frozen history: `IMPLEMENTATION_PROGRESS.md`, `DEBUG_PROGRESS.md`
- Advisor calls: `research/llm_calls.jsonl`; briefings: `research/briefings/`
