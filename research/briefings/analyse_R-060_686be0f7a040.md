# Research briefing (analyse_R-060) — generated 2026-11-04T05:14:28

## User's research direction
Resume Shadow Hand pen spinning from docs/SESSION_2026-10-27_PEN_SPIN_SUMMARY.md. First compare CMA-ES and random search at equal rollout budgets on the frozen R-046 controller's spin and centering parameters (config/pen_spin/r046_block3_incumbent). Compare both winners with the parent on fresh paired cases. Report sustained success, drops, and saturation. Keep final test locked. If tuning improves success without more drops, use the winner for repair. If feedback activates without improving success, prioritize R-043. Defer cold search and R-055. Require verified learner-state recovery before imitation. Use llm_pool with 16 workers and high reasoning effort. Run one protocol at a time.  _(set 2026-11-04)_
- earlier direction (2026-10-27 to 2026-11-04): Continue researching the single-step exploration pipeline with plateau-triggered trace repair, extended to pen spinning on the Shadow Hand (Isaac Lab Isaac-ShadowHandSpin-Stage{1,2,3}-Direct-v0, vendored in lares/envs/isaac_pen_spin; 157-d full obs, 20-d action, 60 Hz, 360-step episodes). Expert demonstrations: first try a hand-written scripted spinning controller; also search online for an existing trained pen-spinning policy; if neither qualifies, train an RL expert (rsl_rl PPO on the box). No budget ceiling on GPU hours or LLM calls. Known gaps to turn into questions: the env's spin_rate is roll about the pen's own long axis (not end-over-end twirl), success is per step with no episode rule, and the pipeline (manifests, obs schema, runner, records) is MetaWorld-only today. Debug issues, improve the pipeline, and turn unexpected findings into concrete research questions; document results and limitations.

## Standing constraints (each with its source; do not propose anything that violates one)
- LLM model for every pipeline call is gpt-6.1-sol with reasoning_effort high and temperature 1 (the only temperature the model accepts).  _(source: CLAUDE.md, decided 2026-10-13)_
- Validation (imitation) loss correlates with development success at +0.19, the wrong sign; selecting on loss cannot work. Behavioural cloning is non-monotonic in gradient steps: success peaks near 1000 steps and is zero by 3000.  _(source: RESEARCH_HANDOFF.md, Phase 5)_
- Imitation fitting is disabled by default (bc_steps 0). If fitting is turned on it must be rollout-selected on a probe manifest disjoint from the screen; fixed-budget fitting is harmful.  _(source: Stage E, logs/stage_e/20260914_032500; config/run_full_evolution.yaml)_
- Promotion requires a strictly positive paired success difference whose paired t interval lower bound is above zero on the promotion set (development cases 10-39). Tied success never promotes.  _(source: lares/eval/promotion.py, decided 2026-09-14)_
- The saved incumbent scores 0.000 success on the development manifest, not 0.30.  _(source: RESEARCH_HANDOFF.md, 'The incumbent does not reproduce')_
- Only the five fittable_seed structures (simple_reach_push_sharp_gate, simple_damped_push, simple_reach_push, simple_reach_push_damped, simple_reach_push_shared_gain) may be fittable-lane parents; the standoff family is a non-fittable control or retired.  _(source: config/fittable_bank/structure_roles.yaml)_
- Stage F: no correction-LLM repair arm beat no_repair; search arms led, LLM arms were flat.  _(source: RESEARCH_HANDOFF.md 'Stage F: correction-LLM repair does not beat no repair (2026-10-13)')_
- The gpt-6.1-sol baseline rerun (nine unseeded zero-shot candidates) reached zero success; the gpt-5.6-sol run of the same config reached 0.333 on the promotion set.  _(source: logs/baseline_gpt61_sol_high, logs/baseline_gpt56_sol_high; RESEARCH_HANDOFF.md 2026-10-13)_
- The AC-6 repair ablation did not support the repair loop: optimizer-only 0.650 vs full loop 0.517 held-out success over six replicates.  _(source: RESEARCH_HANDOFF.md, 'The repair ablation (AC-6, E6, E9)')_
- The final-test split (config/manifests/push-v2_final_test.yaml) is locked until Phase 7, which needs the user's approval and a FreezeRecord. Qwen/E10 is also blocked.  _(source: lares/eval/freeze.py; CLAUDE.md)_
- Seeds 2400-2424 are held for config/fittable_bank/qualification_v2.yaml (predeclared, not run). New protocols pick seeds from 2600 upward.  _(source: config/experiment_seed_ledger.jsonl; CLAUDE.md)_
- Every protocol runs on the remote GPU box via `gpu run`, which bills by the hour; the box stays up across a session's steps and is stopped at the end. Since 2026-10-26 an idea is approved when the advisor and the implementer state the same triage position (approve); the user sets the research direction and rules only on ideas that exhaust their negotiation rounds.  _(source: CLAUDE.md 'Development Workflow'; research/triage.jsonl)_
- Pen-spin task (vendored lares/envs/isaac_pen_spin): Isaac Lab Shadow Hand, 157-d 'full' observation, 20-d joint-target action, 60 Hz control, 360-step (6 s) episodes, num_envs default 256. Success is evaluated per step (pen centre within 0.10/0.10/0.08 m of the in-hand position and |spin_rate| above 1/3/4 rad/s for Stage 1/2/3); there is no episode-level success rule. spin_rate is the pen's angular velocity about its own long axis (roll), not end-over-end twirl. Isaac Lab runs only under ~/IsaacLab/source_env.sh (its own python 3.11 venv), not .venv-metaworld.  _(source: lares/envs/isaac_pen_spin (2026-10-27, scripts/isaac/probe_pen_spin_cfg.py))_
- No pretrained pen-spinning policy for this task exists publicly as of 2026-10-27: the closest work (Wang et al., 'Lessons from Learning to Spin Pens', CoRL 2024) uses an Allegro hand in Isaac Gym and end-over-end spinning. The user's order for the expert: scripted controller first; if it cannot qualify, train an RL expert with rsl_rl PPO (ShadowHandSpinPPORunnerCfg) on the box. No budget ceiling on GPU hours or LLM calls.  _(source: user instruction 2026-10-27; web search 2026-10-27)_
- The policy-search pipeline (manifests, obs schema, run_episode, EXPERT_POLICY_MAP, run_full_evolution.py) is MetaWorld-only. The Isaac path today is the pre-manifest smoke script scripts/run_shadowhand_spin_stages.py with its own rollout copy (no action_space.high scaling, num_envs 1).  _(source: CLAUDE.md 'Things that will bite you'; scripts/run_shadowhand_spin_stages.py)_
- Spent seeds (never reuse): [0, 1, 2, 1000, 1001, 1002, 2000, 2100, 2200, 2201, 2202, 2203, 2204, 2300, 2301, 2302, 2303, 2304, 2400, 2401, 2402, 2403, 2404, 2410, 2411, 2412, 2413, 2414, 2420, 2421, 2422, 2423, 2424, 2500, 2501, 2502, 2600, 2601, 2602, 2603, 2604, 2605, 2606, 2607, 2610, 2611, 2612, 2613, 2614, 2615, 2616, 2617, 2618, 2619, 2620, 2621, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 2700, 2701, 2702, 2703, 2704, 2705, 2710, 2800, 2801, 2802, 2803, 2804, 2805, 2810, 2811, 2812, 2813, 2820, 2822, 2823, 2824, 2825, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2910, 2911, 2912, 2913, 2920, 2921, 2922, 2923, 2940, 3000, 3001, 3010, 3011, 3012, 3013, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3030, 3031, 3032, 3033, 3040, 3041, 3042, 3043, 3044, 3045, 3046, 3047, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3130, 3131, 3132, 3133, 3134, 3135, 3200, 3201, 3202, 3205, 3210, 3211, 3212, 3213, 3214, 3215, 3220, 3221, 3222, 3223, 3300, 3301, 3302, 3303, 3304, 3305, 3310, 3311, 3312, 3313, 3314, 3315, 3320, 3321, 3330, 3331, 3332, 3333, 3334, 3335, 3400, 3410, 3420, 3421, 3422, 3500, 3501, 3502, 3503, 3505, 3506, 3507, 3520, 3521, 3522, 3523, 3524, 3525, 3526, 3527, 3528, 3529, 3540, 3541, 3544, 3600, 3601, 3602, 3603, 3604, 3605, 3606, 3607, 3608, 3609, 3710, 3711, 3712, 3713, 3720, 3721, 3722, 3723, 3724, 3725, 3726, 3800, 3801, 3802, 3803, 3804, 3805, 3806, 3807, 3810, 3811, 3812, 3813, 3814, 3815, 3816, 3817, 3818, 3820, 3821, 3822, 3823, 3824, 3825, 3900, 3901, 3902, 3903, 3904, 3910, 3911, 3912, 3913, 3914, 4000, 4001, 4002, 4010, 4011, 4012, 4020, 4030, 4031, 4032, 4097, 4098, 4099, 4100, 4101, 4120, 4121, 4122, 4123, 4124, 4125, 4126, 4127, 4128, 4129, 4130, 4131, 4132, 4133, 4134, 4135, 4136, 4137, 4140, 4170, 4171, 4172, 4173, 4174, 4180, 4181, 4182, 4183, 4184, 4190, 4191, 4192, 4193, 4200, 4210, 4211, 4212, 4213, 4214, 4215, 4220, 4221, 4222, 4223, 4224, 4225, 4226, 4227, 4310, 4400, 4500, 4501, 4502, 4503, 4504, 4505, 4506, 4507, 4510, 4511, 4512, 4513, 4514, 4515, 4516, 4517, 4550, 4551, 4552, 4553, 4554, 4555, 4556, 4557]
- The final-test split is locked. No protocol may read or score it.

## Open ideas (already proposed; do not duplicate)
### R-060 [analysed] 2. Re-rank archived CMA-ES and random candidates on larger screens
Hypothesis: Independent larger-screen re-ranking can recover transferable scalar candidates that ten-case extreme-score selection discarded.
Prediction: paired_success_difference increase by >= 0.05 vs R-058 original ten-case-selected winners, re-evaluated on the same fresh cases
Falsified by: The paired interval upper bound for re-ranked minus original-winner fresh sustained success is below 0.05; an interval spanning the threshold is inconclusive.
Protocol: gpu run 'bash -lc "source ~/IsaacLab/source_env.sh && python scripts/run_shadowhand_spin_stages.py --config config/shadowhand_spin_stages.yaml --study r060_archive_rescreen --log-dir logs/pen_spin/r060_archive_rescreen"' (episodes=8240, llm_calls=0, seeds=[])
Triage so far: 2 round(s); last advisor approve, implementer approve; Approve the implementer's amended selection-mechanism test using scripts/run_shadowhand_spin_stages.py and config/shadowhand_spin_stages.yaml, seeds 4550-4557, and a budget cap of 8240 simulator episo

## Deferred ideas (judged right but not now; re-triaged when the direction fits)
- R-043 Do anatomical action coordinates improve single-step exploration?: This is a useful action-representation experiment, but its 1536 episodes and at most 16 LLM calls should follow R-042's validated Isaac integration and direct-action exploration results. Its weakest p
- R-052 3. Repair with verified learner-state corrections, not expert-only traces: Verified learner-state corrections are meaningfully different from the expert-only keyframes and off-distribution switch maps already tested. However, neither faithful branching nor useful recovery ev
- R-055 Train the expert's recovery basin on learner-prefix states: Changing the expert's training-state distribution is a legitimate recovery-basin experiment, distinct from unchanged-expert takeover and filtering existing corrections. However, spending 99032 episode
- R-056 R-057: Repair a qualifying tuned winner with slip-gated spin: The 1,800-episode, at-most-32-call repair budget should wait until R-058 establishes a qualifying tuned parent; neither simulator nor LLM spending is justified before that result. Multiplicative slip 
- R-057 R-058: Conditional R-043 anatomical-coordinate repair: This is the appropriate priority branch if R-058 establishes activated feedback without a demonstrated success gain, but its 1,680 episodes and at most 32 LLM calls should not be spent before that pre
- R-061 3. Repair a qualifying winner with clipping-aware command allocation: Joint command allocation differs from R-056's slip-gated repair, but spending 2400 episodes and up to 128 LLM calls requires demonstrated downstream authority loss and a strictly promoted, fresh-confi

## Recently rejected at triage (do not re-propose without saying what changed)
- R-001 Two-lane search beats the unseeded baseline on a fresh pool seed: merged into R-005 (same seed-2610 cases)
- R-002 The gpt-6.1-sol zero-success baseline is model, not draw: Rejected by the advisor (gpt-6.1-sol, consult call_00003) under the user's 2026-10-22 session delegation: a rerun tests recurrence, not a model effect, and its prediction contradicts its hypothesis.
- R-003 Rotating the development screen removes selection on fixed cases: Rejected by the advisor (gpt-6.1-sol, consult call_00015) under the user's 2026-10-22 session delegation: the historical control confounds rotation with candidate draws; reconsideration needs a contem
- R-023 2. Test structural breadth instead of deeper parameter search: Dropped by the advisor (gpt-6.1-sol, consult call) under the user's 2026-10-22 session delegation: its predeclared skip rule applies, since R-022 (D-019) found no task below 0.80; the mechanism is unt
- R-050 1. Resolve same-placement tape divergence before branching traces: I accept the implementer's argument: closed-loop prefix re-execution with its own reproducibility gate removes tape fidelity as a prerequisite for R-051. This audit's predicted paired_success_differen
- R-059 1. Measure actuator headroom downstream of the saturated spin drive: I accept the implementer's decisive argument: the requested non-executed interventions can be computed from saved R-058 observations, so 32 new simulator episodes and zero LLM calls are unnecessary. T

## Decided ideas (most recent first)
- D-048 on R-058: **refuted** — R-046 scalar tuning: budget-matched CMA-ES versus random search
  Predicted that budget-matched CMA-ES would beat random search by at least +0.10 fresh sustained success when tuning the frozen R-046 learner's twelve spin and centering scalars inside their declared bounds (8 blocks, 64 candidates per arm per block on 10 screen cases, winners and parent scored on 30 promotion and 40 fresh cases each, 12,000 episodes, no LLM). Observed +0.003 [-0.076, +0.082] over the eight block contrasts: the upper bound is below +0.10, the predeclared refutation. Neither winner differs from the parent on fresh cases (CMA -0.006 [-0.041, +0.029]; random -0.009 [-0.080, +0.061]) or on promotion pooled over blocks; one strict promotion in sixteen (CMA, block 4515, 10/30 vs 6/30) did not confirm on fresh cases. Drops 46/45/47 of 320 with every interval through zero. The spin drive stays saturated above 0.99 on most recorded in-play steps for every controller (parent 94-99%, CMA winners 82-99%; unsaturated gain +0.046 [-0.004, +0.095]), far from the +0.25 activation margin, so tuning did not activate spin feedback. Not predeclared: CMA-ES did adapt on the screen (mean candidate screen successes 1.8 to 2.5 of 10 across generations, screen drops per generation 14.1 to 12.7; random flat at 2.0 and 15.0) and the screen winners scored 0.44/0.40 against the parent's 0.225 on the same ten cases, yet 0.233/0.237 against 0.221 on promotion: at this success rate selection on ten screen cases is selection noise, and the twelve scalars do not contain a better controller that ten cases can identify. No (block, arm) qualifies as a repair parent and no block activates feedback, so R-056 and R-057 stay deferred.
  Next step then proposed: Neither follow-up is enabled: no tuned winner qualifies as a repair parent (R-056 stays deferred) and no block activates feedback (R-057 stays deferred). Scalar tuning of this learner at this screen size is exhausted; a transferable gain would need a screen large enough to resolve 0.05 differences (about 40+ cases per candidate) or a structural change rather than a scalar one.
- D-047 on R-054: **refuted** — Repair with successful traces from the learner's own grasp basin
  Predicted self minus expert final-incumbent fresh success of at least +0.15, paired over eight blocks. Observed 0.000 with a degenerate [0, 0] interval (8 ties): no candidate in either arm passed the strict promotion rule against the frozen parent, so both arms kept the parent (fresh mean 0.225) in every block. The interval's upper bound is below +0.15, which refutes the prediction under this protocol (4 candidates per arm per block, one linked correction). Secondary and not predeclared: self-trace candidates kept some parent behaviour (27/320 candidate screen successes, 3 candidates beat the parent screen, block-level mean candidate screen self minus expert +0.084 [-0.008, +0.177], 7 wins 0 losses 1 tie) while expert-trace candidates scored 0/320 and dropped the pen in 200/320 screen episodes versus 97/320 for self and 14/80 for the parent. So the evidence source changes how much an edit damages the grasp, but neither source produced an improvement over the parent; the claim that self traces give more transferable repairs is not established at the final-incumbent level.
  Next step then proposed: none
- D-046 on R-053: **inconclusive** — Does the symbolic learner actually use signed axial-spin feedback?
  Identity gate passed (bit-identical on 32/32). Predicted a decrease of at least 0.15; observed -0.031 [-0.176, +0.113] (4/32 vs 5/32, 27 ties). The predeclared refutation needs the interval entirely above -0.15 and its lower bound is -0.176, so neither the prediction nor its refutation is established; the interval includes zero. Post hoc, the reason the lever is weak is visible: the frozen parameters (spin_target 12.16, speed_gain 0.81) saturate the drive 0.25 + 0.75 tanh(gain (target - spin)) above 0.99 on 94.6% of the learner's in-play steps, and reversing the axial sign changes it by more than 0.05 on only 10.4% of steps. The source audit found a code path, but at its operating point the learner is effectively an open-loop fixed-amplitude stroke in axial spin; reversal also did not slow the pen (|spin| 7.39 vs 6.89).
  Next step then proposed: Treat the learner's spin as open loop when designing repair evidence (R-054). A sharper dependence test would set the operating point inside the unsaturated range (e.g. spin_target at its declared 8.0) or replace the measured axial rate with the target (zero error) rather than reversing it.
- D-045 on R-051: **refuted** — 2. Can the expert recover from actual symbolic-controller states?
  Both gates passed (repeat bit-identical on 32/32; 28/28 takeover prefixes match), so the contrast is interpretable. Predicted takeover minus learner >= +0.25; observed +0.031 [-0.183, +0.246] (8/32 vs 7/32). The upper bound is below +0.25, which the predeclared rule names as refutation; confidence 0.75 rather than higher because the bound clears the threshold by only 0.004 with 32 cases. The aggregate hides opposing effects: on 21 already-failing learner states takeover rescued 6 but dropped 6 the learner kept; on the 7 states where the learner was succeeding it broke 5 (all drops 38-126 steps after the switch). Takeover doubles spin (10.3 vs 4.8 rad/s) but drops the pen in 11 switched cases vs 0 for the learner, so the expert's recovery basin covers some slow-spin learner states but not the learner's own grasp; expert labels at these states are 99.85% saturated.
  Next step then proposed: Do not collect expert takeover transitions as repair evidence for this learner as-is (R-052 should stay deferred or be re-scoped to the 6 rescued cases only). Closed-loop prefix re-execution is qualified for branching. A useful next question is whether the learner's own successful episodes (not the expert's) can serve as within-learner demonstrations for its failing cases.
- D-044 on R-049: **refuted** — Require separate retention and spin feedback in cold exploration
  Predicted two_channel minus baseline fresh-incumbent success of at least +0.15 with the interval establishing it; observed 0.000 [0.000, 0.000] over 8 blocks. Both arms kept the neutral-pose null incumbent everywhere: 0 of 32 candidates per arm had a screen success (0/320 screen episodes each), 0 promotions; the only non-zero cell is two_channel block 1 (2/30 promotion, not promoted). Confidence is 0.7 because both arms sit at the floor and 4 candidates per block gives little resolution. Diagnostics: generated candidates already retain the pen (13% of screen episodes dropped vs 97% for the neutral pose) and fail on slow_spin/off_centre, so the bottleneck on Stages 1/2 is sustained axial spin near the centre, not retention; and these per-step-looser stages scored 0/8 blocks while R-046's 24-candidate Stage 3 prefix had probe successes in 3/4, so candidates per block, not stage difficulty, is the untested candidate driver of the base rate.
  Next step then proposed: Measure the cold-search per-candidate success rate with a larger candidate budget on Stage 1 and Stage 3 under one protocol, and target spin generation (axial angular-velocity feedback) rather than retention.
- D-043 on R-048: **inconclusive** — Expose the expert's local feedback map instead of isolated labels
  No paired comparison ran, so the +0.15 prediction was not evaluated. The PPO expert is bang-bang (99.98% of executed components at +/-1), which made clamped finite differences zero almost everywhere; the executed-switch audit found switches in only 2.8% of state-coordinate-action triples within one demonstration range (median distance 0.7 ranges), so the evidence representation would be impoverished and the comparison would test it rather than the feedback-map hypothesis. Closed on the advisor's recommendation (call_00101) as insufficient demonstrated local signal on the sampled grid, not falsification.
  Next step then proposed: Run R-049 (two-channel cold exploration on Stages 1/2). A future feedback-map idea needs a smooth expert (e.g. pre-saturation logits from a policy trained with an action penalty) or switch maps at on-distribution states.
- D-042 on R-047: **supported** — Measure the expert's retention-feedback latency budget
  Predicted a paired decrease of at least 0.25; observed -1.000 (40/40 -> 0/40) for a four-step pen-state delay, with the reproducibility control bit-identical (0 diverged, 0 flips). The hand-state delay is equally fatal (0/40, 39 drops). The claim is limited to this checkpoint under a temporal joint-distribution shift; with one delay value there is no dose-response, so the failure threshold lies at or below 67 ms but is otherwise unmeasured.
  Next step then proposed: If latency matters for the pipeline, a dose-response (1, 2 steps) would locate the threshold; for generated controllers, the implication is that retention needs current-step pen and hand feedback, which R-049's two-channel requirement tests directly.
- D-041 on R-046: **inconclusive** — 3. Preserve the scripted gait while repairing retention feedback
  The predeclared rule needs >=3 eligible blocks; the plateau trigger fired in only 1 of 4 (blocks 0 and 2 had a probe success, block 3 promoted a cold prefix candidate at +0.333 [0.154, 0.512]), so the primary contrast is undefined and the trigger was not weakened. Block 1 is suggestive only: whole_edit replaced the gait (delta = desired - gait) and dropped the pen in 24/30 fresh cases (0/30), while bounded_residual matched the unchanged gait (0.467 vs 0.433; 9 wins 8 losses 13 ties), i.e. preservation, not improvement. The unexpected finding is the cold prefix's non-zero base rate on Stage 3 (3/4 blocks with probe successes, 1 promotion) after R-042 found 0/4 under the same search; whether Stage 3 is a floor depends on seeds, which a base-rate study should measure before any further plateau-gated comparison.
  Next step then proposed: Measure the cold single-step search's Stage 3 base rate (probe-success and promotion rates) over many prefix blocks, and run the gait-residual comparison ungated (every block) instead of behind the plateau trigger.

## Available protocol scripts (reuse before writing new ones)
scripts/ablate_repair.py, scripts/analyse_failures.py, scripts/audit_expert_actions.py, scripts/benchmark_fitting.py, scripts/build_structure_bank.py, scripts/compare_child_vs_parent.py, scripts/compare_evolution_runs.py, scripts/compare_generation_modes.py, scripts/compare_objectives.py, scripts/diagnose_repair.py, scripts/evaluate_on_fresh_manifest.py, scripts/final_report.py, scripts/freeze_method.py, scripts/lock_baseline.py, scripts/plot_training_dynamics.py, scripts/qualify_fittable_bank.py, scripts/recover_provenance.py, scripts/research.py, scripts/run_full_evolution.py, scripts/run_shadowhand_spin_stages.py, scripts/stage_e.py, scripts/stage_f.py, scripts/summarise_experiments.py, scripts/visualize_expert_policy.py
## Available configs
config/assembly_two_lane_repair.yaml, config/debug_smoke_two_generation.yaml, config/environment.yaml, config/fittable_bank/manifest_provenance.json, config/fittable_bank/push-v2_development_confirmation_s2300.yaml, config/fittable_bank/qualification.yaml, config/fittable_bank/qualification_v2.yaml, config/fittable_bank/seed_bank.json, config/fittable_bank/structure_roles.yaml, config/fittable_bank/variants_frozen.json, config/pen_spin/r054_evidence/expert.json, config/pen_spin/r054_evidence/self.json, config/pen_spin_plateau_repair.yaml, config/run_full_evolution.yaml, config/shadowhand_spin_stages.yaml, config/stage_e/manifest_provenance.json, config/stage_e/push-v2_development_confirmation_s2200.yaml, config/stage_e/selection.json, config/stage_e/stage_e.yaml, config/stage_f/manifest_provenance.json, config/stage_f/push-v2_development_heldout_s2500.yaml, config/stage_f/selection.json, config/stage_f/stage_f.yaml, config/stage_f_vision/manifest_provenance.json, config/stage_f_vision/probes/push-v2_probe_s2620.yaml, config/stage_f_vision/probes/push-v2_probe_s2621.yaml, config/stage_f_vision/probes/push-v2_probe_s2622.yaml, config/stage_f_vision/probes/push-v2_probe_s2623.yaml, config/stage_f_vision/probes/push-v2_probe_s2624.yaml, config/stage_f_vision/probes/push-v2_probe_s2625.yaml, config/stage_f_vision/probes/push-v2_probe_s2626.yaml, config/stage_f_vision/probes/push-v2_probe_s2627.yaml, config/stage_f_vision/probes/push-v2_probe_s2628.yaml, config/stage_f_vision/push-v2_development_heldout_s2629.yaml, config/stage_f_vision/stage_f_vision.yaml, config/two_lane_search.yaml, config/two_lane_search_v2.yaml
## Metrics a prediction may name
action_saturation_by_axis, action_variation_by_axis, failure_label, failure_labels, final_goal_distance, fitting.bound_activity, fitting.checkpoint_step, fitting.gradient_norms, fitting.loss_by_phase, fitting.num_steps, fitting.objective, fitting.parameters_sha256, fitting.split, fitting.train_loss_by_axis, fitting.train_loss_end, fitting.train_loss_start, fitting.validation_loss_by_axis, fitting.validation_loss_end, gate_statistics, initial_goal_distance, lateral_drift, llm_calls, llm_total_tokens, mean_return, min_tcp_object_distance, num_episodes, object_displacement, paired_interval_lower, paired_success_difference, phase_occupancy, phase_transitions, promotion_success_rate, proximity_rate, return_se, screen_success_rate, signed_goal_progress, simulator_steps, slots_valid_within_one_correction, success_rate, success_se, timeout_rate, wall_time_seconds

## Where this stands

Phases 0 through 6 are implemented and measured. FR-4 and FR-11 are implemented.
Phase 7's machinery is built but its experiment has not run.

**Status corrected by `debug_spec.md` (2026-09-13).** A review found that the live evolutionary path
did not carry what the phase statuses claimed: generation-1 records had empty lineage, the live
fitter still trained the scale under the legacy objective, a selected checkpoint could be reported
with another checkpoint's diagnostics, and gradient steps were stored as `environment_steps`. The
corrected statuses are below. Stages A–C of the debug spec are implemented and pass a mocked
two-generation run; see `DEBUG_PROGRESS.md`.

| Phase | State |
|---|---|
| 0 measurement lock | complete, 7/7 AC-0 checks |
| 1 trusted interface | complete, 6/6 AC-1 items |
| 2 evaluation and reporting | 8/9 AC-2 items; the ninth is E10, which is blocked |
| 3 fitting comparison | complete, 6/6 AC-3 items |
| 4 search evidence and archive | **reopened**; lineage and bounded-edit semantics integrated, live smoke test passed; edit generations evaluated in Stage F |
| 5 objectives and data coverage | machinery complete; live integration done in the debug cycle; Stage E confirmation not run |
| 6 intervention-guided repair | **pilot**; the ablation does not test correction-LLM repair and its budgets were unequal |
| 7 freeze and final validation | **blocked** through Stage F of `debug_spec.md` |

Three results decide what is worth doing next, and all three run against the
plan's assumptions:

1. **Fitting the expert better does not build a better controller.** Validation
   loss correlates with development success at +0.19, the wrong sign, and
   development success collapses to zero well before the loss minimum.
2. **In a pilot, the repair loop did not beat spending the same budget on the
   optimizer.** Across the six structure/seed pairs `optimizer_only - full` was
   +0.35, +0.20, +0.05, +0.25, -0.05 and 0.00, mean +0.133. The interval first
   reported, [+0.008, +0.259], used 1.96 x SE; the small-sample t interval (5
   degrees of freedom) is about [-0.031, +0.298] and includes zero. This supports
   a directional optimizer advantage, not a significance claim, and the arms did
   not share preprocessing or a correction-LLM path (`debug_spec.md` D10).
   The controlled substitution changed the chosen component in two of six
   replicates and the outcome in one.
3. **The biggest lever on this task is the fitting schedule**, not the structure
   search or the repair. Recovering the pre-collapse checkpoint takes one
   structure from 0.000 to 1.000 held-out success.

Two things are blocked on a decision rather than on implementation:

- **Phase 7's experiment.** Five independent searches plus at least 100
  final-test episodes per finalist. The largest single spend in the plan.
- **E10.** It needs a Qwen checkpoint chosen *and* a correction-LLM repair path,
  because the current repair operators read no prompt and so cannot be affected
  by an evidence block. That same path is what the AC-6 result argues for:
  explanations generated rather than drawn from a fixed table.

The full list of open decisions is at the end of this file.


## Questions requiring a research decision

-2. **Raised by the bank study.** (a) Four of the five qualifying references are reach-push variants, so the
   seeded lane starts from essentially one design family plus `simple_damped_push`. Should the six seeded
   mutations be spread across references by a fixed allocation (for example, damped push guaranteed at least
   one slot) rather than ranked? (b) Should future qualification vary the data split as well as the minibatch
   seed, since the current seeds do not produce independent replicates? (c) The standoff family never succeeds;
   is there a reason to keep it in any bank?


-1. **Raised by Stage E.** (a) The frozen rule pooled both families, and the default it produced removes the
   route to the strongest policies yet found. Should fitting stay disabled for generated structures while
   remaining available, rollout-selected only, for structures that are fitting-dependent, and if so, how is
   that decided before seeing a structure's fitted score? A predeclared per-structure probe (fit only when a
   step-500 or step-1000 checkpoint beats step 0 on the probe cases) is one option; it is what
   rollout selection already does, at the cost of the probe episodes. (b) The generation prompts now reward
   untrained behaviour. That produced fitting-resistant structures capped near 0.40, while two hand-written
   six-parameter structures reach 0.9 after fitting. Should the next search ask for fittable structures, or
   seed the search with the bank references? (c) Evidence about fitting for fitting-dependent structures
   rests on three bank structures; more are needed before any decision specific to them.


0. **Answered on 2026-09-14** (details in `DEBUG_PROGRESS.md`, "Decisions of 2026-09-14"): fitting is an
   experimental treatment decided by Stage E's four conditions; promotion requires a paired interval
   above zero on an independent promotion manifest and never promotes on a tie; the next search runs one
   budget-matched initial generation with no edit generations; seed 2200 and fitting seeds 2200-2204 are
   reserved for Stage E. The original questions follow for the record.
   **New, from the debug-cycle runs.**
   (a) *Should behavioural cloning stay in the loop?* For the structures the current prompts produce, every
   promoted candidate selected step 0 and 2000 steps drive the incumbent to 0.000. Options: keep fitting
   behind the step-0 probe (current: correct, but it spends 30,000 gradient steps and 112,500 probe simulator
   steps per 15-candidate run for nothing), cut the budget and schedule, or replace imitation with direct
   search on rollouts. Stage E's predeclared arms (fixed 2000, rollout-selected, phase-balanced) will
   quantify this; its prediction for the fixed-2000 arm is now close to zero.
   (b) *Should promotion require the paired interval to exclude zero, or at least forbid promotion on tied
   success?* The current rule promoted twice on differences held-out data does not support.
   (c) *Is the edit loop worth its cost?* Generations 1–2 cost roughly two thirds of the run's LLM spend and
   did not improve held-out success. The feedback may need to show per-case differences between parent and
   child, since several repairs changed code without changing any outcome.
   (d) MT1 pool seed 2100 is spent on this comparison. Stage E needs another unused seed.

1. **Answered by the debug cycle.** Section 2.3's reliability target asks that 95% of candidates
   compile and pass validation after at most one repair. Generation now allows exactly one linked
   correction per slot (`max_repair_per_candidate=1`) and no outer retry loop, and the summary
   reports "slots valid within one correction" against the 95% target. What remains open is whether
   the target is reachable: in two Tier 3 runs gpt-4o-mini produced schema-valid structured ideas,
   but its one-slot implementation failed validation after its correction both times (an unknown
   observation field, then a policy that ignored `goal`). The previous loop hid this by retrying up
   to 50 calls. Measure the rate on the live smoke test before deciding whether prompts, the model,
   or the target should change.
2. FR-11 needs a small Qwen vision model. `transformers` is not installed, and the T4 is Turing, so
   fp16 rather than bf16. Which model and quantisation should be frozen? Everything around the model
   is now built and tested, so this decision plus an install is all that stands between here and a
   first real analysis. `QwenAnalyst` raises rather than defaulting, so nothing is frozen by accident.
3. The development split holds 50 placements and the search loop will see them repeatedly. Section 14
   warns about overfitting development through repeated feedback. Should development rotate on a
   predeclared schedule, and if so how?
4. **Imitation loss does not predict task success here, and following it destroys the controller.**
   Phase 5 answered the first half of this: the checkpoint is now chosen on development rollouts
   rather than on loss, which is the only rule that can find the peak. What remains open is whether
   behavioural cloning should keep its current role at all. A structure that scores 1.000 at 1000
   steps and 0.000 at 3000 is being *found* by the rollout probe, not by the fitting, so the fitting
   is closer to a random search over an unhelpfully parameterised path. Options: keep it as an
   initialiser and select on rollouts, which is what is implemented now; replace it with direct
   search on rollout success, which is cheap for six-parameter structures and would not need a
   reward; or keep both and treat the loss purely as a diagnostic.
5. The Phase 3 evidence rests on ten development cases per configuration and success rates near
   0.05. Should the comparison be re-run at 30 cases before the optimizer default is frozen? That
   costs roughly three times the 1120 seconds the current run took. The Phase 5 finding makes this
   more urgent, because every Phase 3 number was measured at the fixed 2000-step budget that sits
   past the peak, so the optimizer comparison may have been comparing points on the wrong side of it.
6. **The scripted expert is not a reliable recovery controller** (open decision 6, answered). About
   a third of its labels at learner-visited states fail to reduce even its own tracking error, worst
   in the descend phase. Should learner-state aggregation filter on the counterfactual check rather
   than only on the cheap validity test? That would raise label quality at the cost of one extra
   simulator step per accepted label.
7. `simple_reach_push` reaches 1.000 on all 50 development cases. It is hand-written, not
   LLM-generated, so it says nothing about the search. Should it become a reference target the
   search is measured against, and should the final-test evaluation of it wait until Phase 7 as the
   spec requires? The final-test manifest has not been touched.

8. **The protected-case regression tolerance (open decision 3) is answered at 0.05 paired success,
   but that number cannot be validated on a policy that scores 0.000.** Nothing can fail it, so the
   fallback progress check is what is actually doing the work today. Once a candidate scores above
   zero the success tolerance starts to bind, and 0.05 on ten screening cases means a single lost
   case is inside tolerance while two are outside. Should the tolerance be expressed in cases rather
   than in rate, so it does not change meaning with the screen size?
9. **The competing explanations are a fixed table keyed on the failure label.** They cover all four
   labels and always offer at least two accounts blaming different components, which is what makes a
   substitution informative. But the table is hand-written, so the search cannot propose an
   explanation the table does not contain. Should the correction LLM propose the competing
   explanations, with the table as a fallback? That is the natural place for FR-11's visual evidence
   to enter as well.
10. **The ablation's `full` arm tests only the two label-based explanations, not the gate one.**
   Holding the diagnostic cost fixed across structures made the arms easier to match, but it means a
   structure whose real fault is a gate that never opens cannot have that found. Worth a follow-up
   run that passes the available gates to `competing_explanations`.
