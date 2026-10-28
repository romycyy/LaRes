# Research state

_Generated 2026-10-27T22:30:13 by `python scripts/research.py state` from `research/ideas/`, `research/decisions.jsonl`, `research/direction.json`. Do not edit by hand._

## Direction

Continue researching the single-step exploration pipeline with plateau-triggered trace repair, extended to pen spinning on the Shadow Hand (Isaac Lab Isaac-ShadowHandSpin-Stage{1,2,3}-Direct-v0, vendored in lares/envs/isaac_pen_spin; 157-d full obs, 20-d action, 60 Hz, 360-step episodes). Expert demonstrations: first try a hand-written scripted spinning controller; also search online for an existing trained pen-spinning policy; if neither qualifies, train an RL expert (rsl_rl PPO on the box). No budget ceiling on GPU hours or LLM calls. Known gaps to turn into questions: the env's spin_rate is roll about the pen's own long axis (not end-over-end twirl), success is per step with no episode rule, and the pipeline (manifests, obs schema, runner, records) is MetaWorld-only today. Debug issues, improve the pipeline, and turn unexpected findings into concrete research questions; document results and limitations.  _(set 2026-10-27)_

| status | count |
|---|---|
| escalated | 0 |
| analysed | 0 |
| running | 0 |
| approved | 1 |
| proposed | 0 |
| deferred | 3 |
| decided | 46 |
| rejected | 5 |

## Needs attention

- **R-054** [approved] Repair with successful traces from the learner's own grasp basin
  - `python scripts/research.py start R-054` then run its command

## Deferred (right idea, not now; `triage R-xxx` re-opens one)

- R-043 Do anatomical action coordinates improve single-step exploration?: This is a useful action-representation experiment, but its 1536 episodes and at most 16 LLM calls should follow R-042's validated Isaac integration and direct-a
- R-052 3. Repair with verified learner-state corrections, not expert-only traces: Verified learner-state corrections are meaningfully different from the expert-only keyframes and off-distribution switch maps already tested. However, neither f
- R-055 Train the expert's recovery basin on learner-prefix states: Changing the expert's training-state distribution is a legitimate recovery-basin experiment, distinct from unchanged-expert takeover and filtering existing corr

## Last 5 decisions

- D-046 · R-053 · **inconclusive** · 2026-10-27
  Identity gate passed (bit-identical on 32/32). Predicted a decrease of at least 0.15; observed -0.031 [-0.176, +0.113] (4/32 vs 5/32, 27 ties). The predeclared refutation needs the interval entirely above -0.15 and its lower bound is -0.176, so neither the prediction nor its refutation is established; the interval includes zero. Post hoc, the reason the lever is weak is visible: the frozen parameters (spin_target 12.16, speed_gain 0.81) saturate the drive 0.25 + 0.75 tanh(gain (target - spin)) above 0.99 on 94.6% of the learner's in-play steps, and reversing the axial sign changes it by more than 0.05 on only 10.4% of steps. The source audit found a code path, but at its operating point the learner is effectively an open-loop fixed-amplitude stroke in axial spin; reversal also did not slow the pen (|spin| 7.39 vs 6.89).
  Next: Treat the learner's spin as open loop when designing repair evidence (R-054). A sharper dependence test would set the operating point inside the unsaturated range (e.g. spin_target at its declared 8.0) or replace the measured axial rate with the target (zero error) rather than reversing it.
- D-045 · R-051 · **refuted** · 2026-10-27
  Both gates passed (repeat bit-identical on 32/32; 28/28 takeover prefixes match), so the contrast is interpretable. Predicted takeover minus learner >= +0.25; observed +0.031 [-0.183, +0.246] (8/32 vs 7/32). The upper bound is below +0.25, which the predeclared rule names as refutation; confidence 0.75 rather than higher because the bound clears the threshold by only 0.004 with 32 cases. The aggregate hides opposing effects: on 21 already-failing learner states takeover rescued 6 but dropped 6 the learner kept; on the 7 states where the learner was succeeding it broke 5 (all drops 38-126 steps after the switch). Takeover doubles spin (10.3 vs 4.8 rad/s) but drops the pen in 11 switched cases vs 0 for the learner, so the expert's recovery basin covers some slow-spin learner states but not the learner's own grasp; expert labels at these states are 99.85% saturated.
  Next: Do not collect expert takeover transitions as repair evidence for this learner as-is (R-052 should stay deferred or be re-scoped to the 6 rescued cases only). Closed-loop prefix re-execution is qualified for branching. A useful next question is whether the learner's own successful episodes (not the expert's) can serve as within-learner demonstrations for its failing cases.
- D-044 · R-049 · **refuted** · 2026-10-27
  Predicted two_channel minus baseline fresh-incumbent success of at least +0.15 with the interval establishing it; observed 0.000 [0.000, 0.000] over 8 blocks. Both arms kept the neutral-pose null incumbent everywhere: 0 of 32 candidates per arm had a screen success (0/320 screen episodes each), 0 promotions; the only non-zero cell is two_channel block 1 (2/30 promotion, not promoted). Confidence is 0.7 because both arms sit at the floor and 4 candidates per block gives little resolution. Diagnostics: generated candidates already retain the pen (13% of screen episodes dropped vs 97% for the neutral pose) and fail on slow_spin/off_centre, so the bottleneck on Stages 1/2 is sustained axial spin near the centre, not retention; and these per-step-looser stages scored 0/8 blocks while R-046's 24-candidate Stage 3 prefix had probe successes in 3/4, so candidates per block, not stage difficulty, is the untested candidate driver of the base rate.
  Next: Measure the cold-search per-candidate success rate with a larger candidate budget on Stage 1 and Stage 3 under one protocol, and target spin generation (axial angular-velocity feedback) rather than retention.
- D-043 · R-048 · **inconclusive** · 2026-10-27
  No paired comparison ran, so the +0.15 prediction was not evaluated. The PPO expert is bang-bang (99.98% of executed components at +/-1), which made clamped finite differences zero almost everywhere; the executed-switch audit found switches in only 2.8% of state-coordinate-action triples within one demonstration range (median distance 0.7 ranges), so the evidence representation would be impoverished and the comparison would test it rather than the feedback-map hypothesis. Closed on the advisor's recommendation (call_00101) as insufficient demonstrated local signal on the sampled grid, not falsification.
  Next: Run R-049 (two-channel cold exploration on Stages 1/2). A future feedback-map idea needs a smooth expert (e.g. pre-saturation logits from a policy trained with an action penalty) or switch maps at on-distribution states.
- D-042 · R-047 · **supported** · 2026-10-27
  Predicted a paired decrease of at least 0.25; observed -1.000 (40/40 -> 0/40) for a four-step pen-state delay, with the reproducibility control bit-identical (0 diverged, 0 flips). The hand-state delay is equally fatal (0/40, 39 drops). The claim is limited to this checkpoint under a temporal joint-distribution shift; with one delay value there is no dose-response, so the failure threshold lies at or below 67 ms but is otherwise unmeasured.
  Next: If latency matters for the pipeline, a dose-response (1, 2 steps) would locate the threshold; for generated controllers, the implication is that retention needs current-step pen and hand feedback, which R-049's two-channel requirement tests directly.

## Last 5 triage rounds (full log: `research/triage.jsonl`)

- R-055 round 1 · advisor defer / implementer defer → agreed (deferred)
- R-054 round 1 · advisor approve / implementer approve → agreed (approved)
- R-053 round 1 · advisor approve / implementer approve → agreed (approved)
- R-051 round 3 · advisor approve / implementer approve → agreed (approved)
- R-051 round 2 · advisor revise / implementer revise → agreed (proposed)

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
- Spent seeds: [0, 1, 2, 1000, 1001, 1002, 2000, 2100, 2200, 2201, 2202, 2203, 2204, 2300, 2301, 2302, 2303, 2304, 2400, 2401, 2402, 2403, 2404, 2410, 2411, 2412, 2413, 2414, 2420, 2421, 2422, 2423, 2424, 2500, 2501, 2502, 2600, 2601, 2602, 2603, 2604, 2605, 2606, 2607, 2610, 2611, 2612, 2613, 2614, 2615, 2616, 2617, 2618, 2619, 2620, 2621, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 2700, 2701, 2702, 2703, 2704, 2705, 2710, 2800, 2801, 2802, 2803, 2804, 2805, 2810, 2811, 2812, 2813, 2820, 2822, 2823, 2824, 2825, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2910, 2911, 2912, 2913, 2920, 2921, 2922, 2923, 2940, 3000, 3001, 3010, 3011, 3012, 3013, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3030, 3031, 3032, 3033, 3040, 3041, 3042, 3043, 3044, 3045, 3046, 3047, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3130, 3131, 3132, 3133, 3134, 3135, 3200, 3201, 3202, 3205, 3210, 3211, 3212, 3213, 3214, 3215, 3220, 3221, 3222, 3223, 3300, 3301, 3302, 3303, 3304, 3305, 3310, 3311, 3312, 3313, 3314, 3315, 3320, 3321, 3330, 3331, 3332, 3333, 3334, 3335, 3400, 3410, 3420, 3421, 3422, 3500, 3501, 3502, 3503, 3505, 3506, 3507, 3520, 3521, 3522, 3523, 3524, 3525, 3526, 3527, 3528, 3529, 3540, 3541, 3544, 3600, 3601, 3602, 3603, 3604, 3605, 3606, 3607, 3608, 3609, 3710, 3711, 3712, 3713, 3720, 3721, 3722, 3723, 3724, 3725, 3726, 3800, 3801, 3802, 3803, 3804, 3805, 3806, 3807, 3810, 3811, 3812, 3813, 3814, 3815, 3816, 3817, 3818, 3820, 3821, 3822, 3823, 3824, 3825, 3900, 3901, 3902, 3903, 3904, 3910, 3911, 3912, 3913, 3914, 4000, 4001, 4002, 4010, 4011, 4012, 4020, 4030, 4031, 4032, 4097, 4098, 4099, 4100, 4101, 4120, 4121, 4122, 4123, 4124, 4125, 4126, 4127, 4128, 4129, 4130, 4131, 4132, 4133, 4134, 4135, 4136, 4137, 4140, 4170, 4171, 4172, 4173, 4174, 4180, 4181, 4182, 4183, 4184, 4190, 4191, 4192, 4193, 4200, 4210, 4211, 4212, 4213, 4214, 4215, 4220, 4221, 4222, 4223, 4224, 4225, 4226, 4227, 4310, 4400]

## Pointers

- Narrative findings: `RESEARCH_HANDOFF.md` (appended once per decision)
- Frozen history: `IMPLEMENTATION_PROGRESS.md`, `DEBUG_PROGRESS.md`
- Every proposal: `research/proposals.jsonl`; every triage round: `research/triage.jsonl`
- Advisor calls: `research/llm_calls.jsonl`; briefings: `research/briefings/`
