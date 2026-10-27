# Research state

_Generated 2026-10-27T17:10:41 by `python scripts/research.py state` from `research/ideas/`, `research/decisions.jsonl`, `research/direction.json`. Do not edit by hand._

## Direction

Continue researching the single-step exploration pipeline with plateau-triggered trace repair, extended to pen spinning on the Shadow Hand (Isaac Lab Isaac-ShadowHandSpin-Stage{1,2,3}-Direct-v0, vendored in lares/envs/isaac_pen_spin; 157-d full obs, 20-d action, 60 Hz, 360-step episodes). Expert demonstrations: first try a hand-written scripted spinning controller; also search online for an existing trained pen-spinning policy; if neither qualifies, train an RL expert (rsl_rl PPO on the box). No budget ceiling on GPU hours or LLM calls. Known gaps to turn into questions: the env's spin_rate is roll about the pen's own long axis (not end-over-end twirl), success is per step with no episode rule, and the pipeline (manifests, obs schema, runner, records) is MetaWorld-only today. Debug issues, improve the pipeline, and turn unexpected findings into concrete research questions; document results and limitations.  _(set 2026-10-27)_

| status | count |
|---|---|
| escalated | 0 |
| analysed | 0 |
| running | 0 |
| approved | 2 |
| proposed | 0 |
| deferred | 1 |
| decided | 42 |
| rejected | 4 |

## Needs attention

- **R-048** [approved] Expose the expert's local feedback map instead of isolated labels
  - `python scripts/research.py start R-048` then run its command
- **R-049** [approved] Require separate retention and spin feedback in cold exploration
  - `python scripts/research.py start R-049` then run its command

## Deferred (right idea, not now; `triage R-xxx` re-opens one)

- R-043 Do anatomical action coordinates improve single-step exploration?: This is a useful action-representation experiment, but its 1536 episodes and at most 16 LLM calls should follow R-042's validated Isaac integration and direct-a

## Last 5 decisions

- D-042 · R-047 · **supported** · 2026-10-27
  Predicted a paired decrease of at least 0.25; observed -1.000 (40/40 -> 0/40) for a four-step pen-state delay, with the reproducibility control bit-identical (0 diverged, 0 flips). The hand-state delay is equally fatal (0/40, 39 drops). The claim is limited to this checkpoint under a temporal joint-distribution shift; with one delay value there is no dose-response, so the failure threshold lies at or below 67 ms but is otherwise unmeasured.
  Next: If latency matters for the pipeline, a dose-response (1, 2 steps) would locate the threshold; for generated controllers, the implication is that retention needs current-step pen and hand feedback, which R-049's two-channel requirement tests directly.
- D-041 · R-046 · **inconclusive** · 2026-10-27
  The predeclared rule needs >=3 eligible blocks; the plateau trigger fired in only 1 of 4 (blocks 0 and 2 had a probe success, block 3 promoted a cold prefix candidate at +0.333 [0.154, 0.512]), so the primary contrast is undefined and the trigger was not weakened. Block 1 is suggestive only: whole_edit replaced the gait (delta = desired - gait) and dropped the pen in 24/30 fresh cases (0/30), while bounded_residual matched the unchanged gait (0.467 vs 0.433; 9 wins 8 losses 13 ties), i.e. preservation, not improvement. The unexpected finding is the cold prefix's non-zero base rate on Stage 3 (3/4 blocks with probe successes, 1 promotion) after R-042 found 0/4 under the same search; whether Stage 3 is a floor depends on seeds, which a base-rate study should measure before any further plateau-gated comparison.
  Next: Measure the cold single-step search's Stage 3 base rate (probe-success and promotion rates) over many prefix blocks, and run the gait-residual comparison ungated (every block) instead of behind the plateau trigger.
- D-040 · R-044 · **supported** · 2026-10-27
  Pen-state clamped minus the unmodified expert is -0.975 [-1.026,-0.924] over 40 paired cases, below the predicted -0.30 with the interval far from it; the reproducibility control is exact (0 flips, 0 diverged). Within the narrowed hypothesis (this checkpoint, under clamping, with hand proprioception and action history live) the prediction is supported. Two limits: the hand-state clamp is nearly as destructive (-0.950 [-1.021,-0.879]), so pen state is not uniquely required; and clamping shifts the input distribution, so this is dependence, not necessity. The failure split is informative: pen clamps produce early drops (39/40, median step 49, matching D-037's replay falls), hand clamps split into drops (22) and completed-but-slow spins (17).
  Next: A symbolic controller needs both pen-state retention feedback and proprioceptive spin feedback; next, measure whether generated controllers on Stage 1/2 (non-zero base rate) improve when the prompt states this two-channel requirement.
- D-039 · R-045 · **refuted** · 2026-10-27
  Primary dense_fit minus rollout_search final-incumbent fresh success is 0.000 [0,0] over four blocks (neither arm promoted anywhere); the interval lies below the predicted +0.15, so the prediction is refuted under this protocol. As predeclared, an all-zero result cannot separate insufficient structure from ineffective fitting, so this does not show dense labels are useless in general. Mechanistic evidence: fitting succeeded as optimisation (MSE 2-4x lower, about 0.33 floor per action dimension) and rollouts chose fitted checkpoints in 3/4 blocks, yet no fitted setting scored a single probe success, and in block 2 every fitted checkpoint collapsed tail-hit (0.336 to 0.03-0.06). Lower imitation loss did not track rollout quality, matching the MetaWorld Stage E finding. The only success came from rollout search (block 3 local draw: 3/30 promotion, 3/30 fresh, not promoted). Expert 48/48 and parity 0 flips rule out integration faults. Secondary selected-setting contrast -0.025 [-0.105,+0.055].
  Next: Stage 3 symbolic controllers sit at the floor under prompting, sparse traces and dense fitting alike; measure base rates on Stage 1/2 (R-044's lesion study can tell which feedback a structure must contain) before another Stage-3 repair comparison.
- D-038 · R-042 · **refuted** · 2026-10-27
  The predeclared primary, event minus uniform fresh sustained success paired over four blocks, is exactly 0.000 with a degenerate [0,0] interval; the protocol's conclusiveness rule (>=3 eligible blocks) is met with 4/4 triggers fired, and the interval excludes the predicted +0.20, so the prediction as stated is refuted. The caveat is a floor: all twelve continuation arms ended with the null incumbent, so the study cannot rank keyframe rules at any effect size it could detect and does not show trace repair is useless in general. Development evidence agrees there is no trace benefit: promotion-set successes no_trace 2/120, uniform 0/120, event 1/120; exploratory tail-hit contrasts event-uniform -0.047 [-0.195,+0.102], uniform-no_trace +0.079 [-0.088,+0.245]. Trace-arm candidates in blocks 0-1 spin while held (highest tail-hit 0.40, 0.335) but drop the pen in 28-30 of 30 promotion cases, consistent with D-037: the expert's competence is its feedback map, which 32 sparse state-action keyframes from 2 episodes do not convey to a 6-10 parameter controller. Demonstrations 10/10 and parity 0 flips, so the null is not an integration fault.
  Next: Stage 3 is at the floor for single-step symbolic exploration; next test a dense, fitted use of the expert (e.g. parameter fitting of the edited controller to expert state-action pairs, rollout-selected) or move the repair study to Stage 1/2 where generated controllers have non-zero base rates, so a trace effect is measurable.

## Last 5 triage rounds (full log: `research/triage.jsonl`)

- R-049 round 1 · advisor approve / implementer approve → agreed (approved)
- R-048 round 1 · advisor approve / implementer approve → agreed (approved)
- R-047 round 1 · advisor approve / implementer approve → agreed (approved)
- R-046 round 2 · advisor approve / implementer approve → agreed (approved)
- R-044 round 2 · advisor approve / implementer approve → agreed (approved)

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
- Every protocol runs on the remote GPU box via `gpu run`, which bills by the hour; the box stays up across a session's steps and is stopped at the end. Since 2026-10-26 an idea is approved when the advisor and the implementer state the same triage position (approve); the user sets the research direction and rules only on ideas that exhaust their negotiation rounds. _(CLAUDE.md 'Development Workflow'; research/triage.jsonl)_
- Pen-spin task (vendored lares/envs/isaac_pen_spin): Isaac Lab Shadow Hand, 157-d 'full' observation, 20-d joint-target action, 60 Hz control, 360-step (6 s) episodes, num_envs default 256. Success is evaluated per step (pen centre within 0.10/0.10/0.08 m of the in-hand position and |spin_rate| above 1/3/4 rad/s for Stage 1/2/3); there is no episode-level success rule. spin_rate is the pen's angular velocity about its own long axis (roll), not end-over-end twirl. Isaac Lab runs only under ~/IsaacLab/source_env.sh (its own python 3.11 venv), not .venv-metaworld. _(lares/envs/isaac_pen_spin (2026-10-27, scripts/isaac/probe_pen_spin_cfg.py))_
- No pretrained pen-spinning policy for this task exists publicly as of 2026-10-27: the closest work (Wang et al., 'Lessons from Learning to Spin Pens', CoRL 2024) uses an Allegro hand in Isaac Gym and end-over-end spinning. The user's order for the expert: scripted controller first; if it cannot qualify, train an RL expert with rsl_rl PPO (ShadowHandSpinPPORunnerCfg) on the box. No budget ceiling on GPU hours or LLM calls. _(user instruction 2026-10-27; web search 2026-10-27)_
- The policy-search pipeline (manifests, obs schema, run_episode, EXPERT_POLICY_MAP, run_full_evolution.py) is MetaWorld-only. The Isaac path today is the pre-manifest smoke script scripts/run_shadowhand_spin_stages.py with its own rollout copy (no action_space.high scaling, num_envs 1). _(CLAUDE.md 'Things that will bite you'; scripts/run_shadowhand_spin_stages.py)_
- Spent seeds: [0, 1, 2, 1000, 1001, 1002, 2000, 2100, 2200, 2201, 2202, 2203, 2204, 2300, 2301, 2302, 2303, 2304, 2400, 2401, 2402, 2403, 2404, 2410, 2411, 2412, 2413, 2414, 2420, 2421, 2422, 2423, 2424, 2500, 2501, 2502, 2600, 2601, 2602, 2603, 2604, 2605, 2606, 2607, 2610, 2611, 2612, 2613, 2614, 2615, 2616, 2617, 2618, 2619, 2620, 2621, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 2700, 2701, 2702, 2703, 2704, 2705, 2710, 2800, 2801, 2802, 2803, 2804, 2805, 2810, 2811, 2812, 2813, 2820, 2822, 2823, 2824, 2825, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2910, 2911, 2912, 2913, 2920, 2921, 2922, 2923, 2940, 3000, 3001, 3010, 3011, 3012, 3013, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3030, 3031, 3032, 3033, 3040, 3041, 3042, 3043, 3044, 3045, 3046, 3047, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3130, 3131, 3132, 3133, 3134, 3135, 3200, 3201, 3202, 3205, 3210, 3211, 3212, 3213, 3214, 3215, 3220, 3221, 3222, 3223, 3300, 3301, 3302, 3303, 3304, 3305, 3310, 3311, 3312, 3313, 3314, 3315, 3320, 3321, 3330, 3331, 3332, 3333, 3334, 3335, 3400, 3410, 3420, 3421, 3422, 3500, 3501, 3502, 3503, 3505, 3506, 3507, 3520, 3521, 3522, 3523, 3524, 3525, 3526, 3527, 3528, 3529, 3540, 3541, 3544, 3600, 3601, 3602, 3603, 3604, 3605, 3606, 3607, 3608, 3609, 3710, 3711, 3712, 3713, 3720, 3721, 3722, 3723, 3724, 3725, 3726, 3800, 3801, 3802, 3803, 3804, 3805, 3806, 3807, 3810, 3811, 3812, 3813, 3814, 3815, 3816, 3817, 3818, 3820, 3821, 3822, 3823, 3824, 3825, 3900, 3901, 3902, 3903, 3904, 3910, 3911, 3912, 3913, 3914, 4000, 4001, 4002, 4010, 4011, 4012, 4020, 4030, 4031, 4032, 4097, 4098, 4099, 4100, 4101, 4120, 4121, 4122, 4123, 4124, 4125, 4126, 4127, 4128, 4129, 4130, 4131, 4132, 4133, 4134, 4135, 4136, 4137, 4140, 4170, 4171, 4172, 4173, 4174, 4180, 4181, 4182, 4183, 4184, 4190, 4191, 4192, 4193, 4200]

## Pointers

- Narrative findings: `RESEARCH_HANDOFF.md` (appended once per decision)
- Frozen history: `IMPLEMENTATION_PROGRESS.md`, `DEBUG_PROGRESS.md`
- Every proposal: `research/proposals.jsonl`; every triage round: `research/triage.jsonl`
- Advisor calls: `research/llm_calls.jsonl`; briefings: `research/briefings/`
