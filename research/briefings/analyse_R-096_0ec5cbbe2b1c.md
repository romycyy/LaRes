# Research briefing (analyse_R-096) — generated 2026-11-20T06:31:52

## User's research direction
Test two representation changes for cold pen-spin generation on Stage 3, independently, native sustained success primary, final test locked: (M1) LLM-generated modular policies (MODULE_SPEC + named methods + generated composition inside GeneratedPolicy, no role hints) vs monolithic, same prompt otherwise, 24-draw search for both arms; (C1) a no-LLM contact-frame decoder gate (capsule frames, fingertip Jacobians, requested vs observed local motion); (C2, only if C1 passes) LLM contact-frame velocity rules vs direct joint generation at matched budget. No BC, RL, curriculum, or gradient fitting. One protocol at a time on the box.  _(set 2026-11-20)_
- earlier direction (2026-10-27 to 2026-11-04): Continue researching the single-step exploration pipeline with plateau-triggered trace repair, extended to pen spinning on the Shadow Hand (Isaac Lab Isaac-ShadowHandSpin-Stage{1,2,3}-Direct-v0, vendored in lares/envs/isaac_pen_spin; 157-d full obs, 20-d action, 60 Hz, 360-step episodes). Expert demonstrations: first try a hand-written scripted spinning controller; also search online for an existing trained pen-spinning policy; if neither qualifies, train an RL expert (rsl_rl PPO on the box). No budget ceiling on GPU hours or LLM calls. Known gaps to turn into questions: the env's spin_rate is roll about the pen's own long axis (not end-over-end twirl), success is per step with no episode rule, and the pipeline (manifests, obs schema, runner, records) is MetaWorld-only today. Debug issues, improve the pipeline, and turn unexpected findings into concrete research questions; document results and limitations.
- earlier direction (2026-11-04 to 2026-11-20): Resume Shadow Hand pen spinning from docs/SESSION_2026-10-27_PEN_SPIN_SUMMARY.md. First compare CMA-ES and random search at equal rollout budgets on the frozen R-046 controller's spin and centering parameters (config/pen_spin/r046_block3_incumbent). Compare both winners with the parent on fresh paired cases. Report sustained success, drops, and saturation. Keep final test locked. If tuning improves success without more drops, use the winner for repair. If feedback activates without improving success, prioritize R-043. Defer cold search and R-055. Require verified learner-state recovery before imitation. Use llm_pool with 16 workers and high reasoning effort. Run one protocol at a time.

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
- Spent seeds (consumed by a run or a committed manifest nobody open owns; never reuse): [0, 1, 2, 1000, 1001, 1002, 2000, 2100, 2200, 2201, 2202, 2203, 2204, 2300, 2301, 2302, 2303, 2304, 2400, 2401, 2402, 2403, 2404, 2410, 2411, 2412, 2413, 2414, 2420, 2421, 2422, 2423, 2424, 2500, 2501, 2502, 2600, 2601, 2602, 2603, 2604, 2605, 2606, 2607, 2610, 2611, 2612, 2613, 2614, 2615, 2616, 2617, 2618, 2619, 2620, 2621, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 2700, 2701, 2702, 2703, 2704, 2705, 2710, 2800, 2801, 2802, 2803, 2804, 2805, 2810, 2811, 2812, 2813, 2820, 2822, 2823, 2824, 2825, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2910, 2911, 2912, 2913, 2920, 2921, 2922, 2923, 2940, 3000, 3001, 3010, 3011, 3012, 3013, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3030, 3031, 3032, 3033, 3040, 3041, 3042, 3043, 3044, 3045, 3046, 3047, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3130, 3131, 3132, 3133, 3134, 3135, 3200, 3201, 3202, 3205, 3210, 3211, 3212, 3213, 3214, 3215, 3220, 3221, 3222, 3223, 3300, 3301, 3302, 3303, 3304, 3305, 3310, 3311, 3312, 3313, 3314, 3315, 3320, 3321, 3330, 3331, 3332, 3333, 3334, 3335, 3400, 3410, 3420, 3421, 3422, 3500, 3501, 3502, 3503, 3505, 3506, 3507, 3520, 3521, 3522, 3523, 3524, 3525, 3526, 3527, 3528, 3529, 3540, 3541, 3544, 3600, 3601, 3602, 3603, 3604, 3605, 3606, 3607, 3608, 3609, 3710, 3711, 3712, 3713, 3720, 3721, 3722, 3723, 3724, 3725, 3726, 3800, 3801, 3802, 3803, 3804, 3805, 3806, 3807, 3810, 3811, 3812, 3813, 3814, 3815, 3816, 3817, 3818, 3820, 3821, 3822, 3823, 3824, 3825, 3900, 3901, 3902, 3903, 3904, 3910, 3911, 3912, 3913, 3914, 4000, 4001, 4002, 4010, 4011, 4012, 4020, 4030, 4031, 4032, 4097, 4098, 4099, 4100, 4101, 4120, 4121, 4122, 4123, 4124, 4125, 4126, 4127, 4128, 4129, 4130, 4131, 4132, 4133, 4134, 4135, 4136, 4137, 4140, 4170, 4171, 4172, 4173, 4174, 4180, 4181, 4182, 4183, 4184, 4190, 4191, 4192, 4193, 4200, 4210, 4211, 4212, 4213, 4214, 4215, 4220, 4221, 4222, 4223, 4224, 4225, 4226, 4227, 4310, 4400, 4500, 4501, 4502, 4503, 4504, 4505, 4506, 4507, 4510, 4511, 4512, 4513, 4514, 4515, 4516, 4517, 4550, 4551, 4552, 4553, 4554, 4555, 4556, 4557, 4560, 4580, 4581, 4610, 4620, 4631, 4642, 4653, 4660, 4661, 4662, 4663, 4672, 4673, 4690, 4700, 4701, 4702, 4710, 4711, 4712, 4713, 4714, 4715, 4716, 4717, 4720, 4721, 4722, 4723]
- Reserved seeds (held only by an open idea's own frozen manifest; that idea may run on them, no other may use them): {4724: 'R-096'}
- The final-test split is locked. No protocol may read or score it.

## Open ideas (already proposed; do not duplicate)
### R-096 [analysed] C1c: 80-case replication of the settled-window paired-contrast decoder gate
Hypothesis: The frozen capsule-frame/Jacobian decoder produces sign-correct local fingertip motion during settled, pen-following pulses without reducing retention beyond the fixed margin.
Prediction: timeout_rate increase by >= -0.1 vs neutral-hold cell on the same 80-case seed-4724 manifest
Falsified by: Hold-minus-neutral timeout rate < -0.10 refutes the retention prediction; Jacobian audit median >= 0.15, pooled paired sign agreement < 0.80, or pooled median paired gain outside [0.3, 1.5] refutes decoder qualification. Fewer than 20 eligible pairs in any contrast makes qualification inconclusive, not supported. D-066 left coverage unresolved: lf_normal had 16 pairs and th_normal 19, with 29/40 cases alive at step 180. This explicitly repeats the same mechanism with twice the cases, not relaxed criteria or a retry of seed 4722.
Protocol: gpu run bash -lc 'source ~/IsaacLab/source_env.sh && python scripts/run_shadowhand_spin_stages.py --config config/shadowhand_spin_stages.yaml --study contact_settled_gate_80' (episodes=2560, llm_calls=0, seeds=[4724])
Triage so far: 1 round(s); last advisor approve, implementer approve; This unchanged-decoder replication addresses D-066's unresolved coverage rather than reopening a settled mechanism, making it the right next C1 spend. Reusing scripts/run_shadowhand_spin_stages.py and

## Deferred ideas (judged right but not now; re-triaged when the direction fits)
- R-043 Do anatomical action coordinates improve single-step exploration?: This is a useful action-representation experiment, but its 1536 episodes and at most 16 LLM calls should follow R-042's validated Isaac integration and direct-action exploration results. Its weakest p
- R-052 3. Repair with verified learner-state corrections, not expert-only traces: Verified learner-state corrections are meaningfully different from the expert-only keyframes and off-distribution switch maps already tested. However, neither faithful branching nor useful recovery ev
- R-055 Train the expert's recovery basin on learner-prefix states: Changing the expert's training-state distribution is a legitimate recovery-basin experiment, distinct from unchanged-expert takeover and filtering existing corrections. However, spending 99032 episode
- R-056 R-057: Repair a qualifying tuned winner with slip-gated spin: The 1,800-episode, at-most-32-call repair budget should wait until R-058 establishes a qualifying tuned parent; neither simulator nor LLM spending is justified before that result. Multiplicative slip 
- R-057 R-058: Conditional R-043 anatomical-coordinate repair: This is the appropriate priority branch if R-058 establishes activated feedback without a demonstrated success gain, but its 1,680 episodes and at most 32 LLM calls should not be spent before that pre
- R-061 3. Repair a qualifying winner with clipping-aware command allocation: Joint command allocation differs from R-056's slip-gated repair, but spending 2400 episodes and up to 128 LLM calls requires demonstrated downstream authority loss and a strictly promoted, fresh-confi
- R-063 Anticipatory centering without changing the spin stroke: Predictive centering introduces a genuinely new observation pathway, but its 380 simulator episodes and zero LLM calls should wait for R-062's structural diagnosis. Its weakest point is that extrapola
- R-066 2. Separate online centering from the frozen spin machinery: This tests centering-feedback necessity, unlike R-063's anticipatory replacement, but its 160 episodes and zero LLM calls should wait for R-065's localization result. Its weakest point is decision val
- R-067 3. Test whether inter-finger stroke coordination matters: Relative-phase disruption is distinct from D-050's stroke removal and R-064's tempo intervention, but its 240 episodes and zero LLM calls are lower priority than locating the residual spin channel. It
- R-069 2. Test whether curl removal prevents damage or recovers the grasp: This 272-episode, zero-LLM diagnostic becomes useful only after R-068 qualifies the exact reset clamp; it differs from D-045 by testing the newly identified within-learner intervention rather than exp
- R-070 3. Repair slow spin with stroke-synchronous contact unloading: Stroke-synchronous contact unloading is distinct from the deferred timing, relative-phase, and clipping repairs, but its 1,210 episodes and at most 32 LLM calls require a qualified zero-curl parent fi
- R-072 2. Test redundant stabilization after capture-curl removal: The zero-curl operating point leaves a genuine redundancy question that D-051's parent-controller lesions did not settle. However, 480 episodes and zero LLM calls buy a localization result rather than
- R-076 Test nonreciprocal stroke shape on the zero-curl operating point: I accept the implementer's sequencing argument: R-075 may change the appropriate operating point, so committing 760 episodes and zero LLM calls to waveform shaping now is premature. The reciprocity cl
- R-078 Give spin feedback direct authority over the raw stroke: Direct stroke scaling tests an authority pathway that scalar tuning did not reach, while the constant-amplitude control distinguishes feedback from stronger open-loop actuation. Defer its 330 episodes
- R-079 Transport the stroke into the pen's instantaneous tangential plane: Geometric stroke transport is distinct from the rotation-correction lesion, and the initial-axis control would separate online transport from static remapping. Defer its 450 episodes and zero LLM call
- R-082 Conditional R-079: transport the stroke with the pen axis: Instantaneous-axis transport versus reset-axis transport isolates online geometric tracking, which the rotation-correction lesion did not test. Defer its 450 simulator episodes and zero LLM calls unti
- R-085 Conditional R-043: isolate anatomical language grounding: This cleanly separates anatomical language grounding from numerical decoder effects, but its prerequisite is an established R-084 anatomical advantage and parent qualification. Its paired_success_diff
- R-087 2. Re-nominate R-082: isolate online pen-axis transport: Online-versus-reset transport isolates changing-axis geometry, but R-086 should finish first because adoption could change the appropriate parent. Its weakest point is whether a separable geometric st
- R-088 3. Re-nominate R-067: lesion inter-finger stroke coordination: Relative-phase disruption tests coordination rather than the common phase lead tested in D-061, but it localizes a mechanism without supplying a better controller. Its 160-episode, zero-LLM budget sho
- R-094 C2: Contact-frame velocity rules versus direct joint generation: The matched-budget contact-output comparison fits the direction, but its 3,400 episodes and at most 64 LLM calls must wait for a recorded C1 pass with the exact decoder hash. Reuse scripts/run_full_ev

## Recently rejected at triage (do not re-propose without saying what changed)
- R-023 2. Test structural breadth instead of deeper parameter search: Dropped by the advisor (gpt-6.1-sol, consult call) under the user's 2026-10-22 session delegation: its predeclared skip rule applies, since R-022 (D-019) found no task below 0.80; the mechanism is unt
- R-050 1. Resolve same-placement tape divergence before branching traces: I accept the implementer's argument: closed-loop prefix re-execution with its own reproducibility gate removes tape fidelity as a prerequisite for R-051. This audit's predicted paired_success_differen
- R-059 1. Measure actuator headroom downstream of the saturated spin drive: I accept the implementer's decisive argument: the requested non-executed interventions can be computed from saved R-058 observations, so 32 new simulator episodes and zero LLM calls are unnecessary. T
- R-074 Audit whether capture-curl impulses predict drops: I accept the implementer's decisive argument: this association cannot establish impulse causality, and neither a positive nor negative result would change the next decision to test R-075. Extending sc
- R-090 2. Conditional open-loop contact-dwell calibration: I accepted the implementer's deferral while R-089 was pending, but D-063 now records failed qualification and more drops, triggering this protocol's explicit cancellation condition. Spending 420 episo
- R-091 3. Abstain: no third research protocol within this focus: This is a useful administrative stop constraint, but not a separate experiment: R-089 already prohibits retries, and R-090 already cancels after failed qualification. Its zero-episode, zero-LLM comman

## Decided ideas (most recent first)
- D-066 on R-095: **supported** — C1b: settled-window paired-contrast contact decoder gate
  C1b, the settled-window paired-contrast contact decoder gate (no LLM; 400 calibration episodes on seed 4723, then 32 whole-manifest cells = 1,280 episodes on seed 4722 with the frozen block hashed beforehand). R-093's controller pulsed at steps 150-180, after the pen has come to rest in the closed reference posture, with pen-following on the commanded fingertip, and scored per case by the paired contrast of the positive and negative pulse. The registered retention prediction is supported (0.725 vs 0.025, +0.70 against a -0.10 margin). The gate is INCONCLUSIVE on coverage alone: two of 15 fingertip-axis contrasts have fewer than 20 eligible pairs (lf_normal 16, th_normal 19) because only 29 of 40 cases still hold the pen at step 180. Every measured check passes: Jacobian audit 0.0047, pooled paired sign agreement 1.000 over 394 pairs, pooled median paired gain 0.62, all 15 contrasts at sign agreement 1.00 with gains 0.38-0.94 (normal lowest), thumb 0.61; unpaired sign agreement 0.999. By the declared rule the decoder is not qualified and R-094 stays deferred.
  Next step then proposed: Replicate once with the identical frozen block (hash 7219ec6d...), no calibration, on a fresh 80-case manifest with every threshold unchanged, including the absolute minimum of 20 eligible pairs per contrast. A pass qualifies that exact decoder for R-094; anything else ends the contact branch under this direction.
- D-065 on R-092: **refuted** — M1: Modular vs monolithic cold generation, searched
  M1: contract-validated modular vs monolithic cold generation on Stage 3, both arms searched (8 blocks, seeds 4710-4717; 4 slots per arm; the best screened slot gets the 24-candidate probe search; promotion against the scored neutral-pose null; fresh; 6,080 episodes, 82 LLM calls, 1.22 M tokens, 2 h 44 min). The modular arm's prompt adds only a MODULE_SPEC requirement with no role hints, enforced by an AST contract (observation reads inside declared modules, forward composes module outputs, an observation-to-action bypass is a validation failure). Predicted at least +0.05 fresh final-incumbent success; observed +0.0083 [-0.011, +0.028]: the upper bound is below +0.05, so the predicted magnitude is refuted, and no improvement is established. One modular promotion (block 1: 4/30 vs the null's 0/30, fresh 2/30), none monolithic. Secondaries: searched winner on fresh +0.025 [-0.011, +0.061] (inconclusive), final-window hit fraction -0.029 [-0.123, +0.065], probe-success candidates +0.125 per block [-1.6, +1.9], winner's drops -3.9 per block [-10.0, +2.3], 0 of 64 candidates with a screen success, 32/32 slots valid within one correction in both arms (the contract cost no validity). Cold Stage-3 generation is at its floor for both representations. Post hoc: left free to decompose, the model wrote state-reader modules plus a retention and a rolling module in most selected controllers, the split R-049 had hinted.
  Next step then proposed: Do not run modular conditions C and D on cold generation; test the modular format where it could matter, diagnosis and repair of a working controller. The contact branch continues with a settled-window paired-contrast decoder gate.
- D-064 on R-093: **supported** — C1: Contact-frame decoder gate
  C1, the contact-frame decoder gate (no LLM; 800 calibration episodes on seed 4721, then 32 whole-manifest cells = 1,280 episodes on seed 4720 with constants hashed beforehand). The registered prediction, zero-request hold timeout rate no more than 0.10 below the neutral hold, is supported: 0.775 vs 0.000 (the closed reference posture retains the pen where the neutral pose never does; neither spins). The gate itself FAILED: Jacobian audit 0.004 (< 0.15), coverage 40/40 eligible in all 30 directional cells and median gain 0.526 (in [0.3, 1.5]) pass, but pooled requested-sign agreement is 0.6825 against 0.80. The failure is a request-independent relative motion of about +0.17 m/s along the normal (the pen settling and the fingers closing at steps 10-40), five times the 0.03 m/s pulse; the frozen setting does not follow the pen because every pen-following setting lost 60-90% of pairs to decoder bounds in calibration. Post hoc and not pre-declared, the paired positive-minus-negative contrast per case gives sign agreement 0.938 and gain 0.54 (thumb 0.10-0.43). The decoder is NOT qualified and R-094 (C2) stays deferred. Durable engineering findings: PhysX Jacobians and the observed fingertip linear velocity refer to each distal link's centre of mass, 15-16 mm from the link origin the observed position reports (uncorrected audit error 0.20, corrected 0.004); targets are unsmoothed; PD stiffness 1.0, damping 0.1; fingertip origins sit 4-6 cm from the pen at step 10 except the thumb.
  Next step then proposed: A second, pre-declared gate on fresh seeds: pulses after the pen has settled in the closed reference posture, pen-following on, paired positive-minus-negative contrast per case as the tracking statistic, thumb reported separately; re-triage R-094 only on a pass.
- D-063 on R-089: **inconclusive** — 1. Qualify the frozen negative warp with one 90-pair instrument
  The 90-pair instrument (as D-057) for the frozen D-062 warp (epsilon -0.35 on the D-058 incumbent's stroke phase, archived source and constants hash-checked, identity gate bit for bit): three independent frozen 40-case manifests (seeds 4700-4702), cases 10-39 of each, parent and warp each panel in its own process, one paired interval over the 90 declared differences; 180 episodes, no LLM. Predicted at least +0.05; observed -0.022 [-0.105, +0.061] (29 vs 31 of 90; 6 wins, 8 losses, 76 ties), drops 6 vs 4: the interval includes zero and its upper bound exceeds +0.05, so the magnitude is neither supported nor refuted, but the warp does not qualify (lower bound below zero) and would not be adoptable (more drops); no panel favoured it (8 vs 9, 13 vs 13, 8 vs 9). D-062's fresh gain (+0.0875 [+0.015, +0.160]) did not replicate. The D-058 incumbent stands; R-090 cancels by its own condition. With D-061 (phase lead) and this, the stroke-side mechanisms re-nominated after the direction's named branches have all failed to qualify on the adopted incumbent.
  Next step then proposed: R-090 cancels by its own condition. With this, every stroke-side mechanism re-nominated after the direction's named branches (lead feedback D-061, waveform shape D-062/this) has failed to qualify on the adopted incumbent; the incumbent (zero-curl clamp, stroke x 1.408) stands at about 0.35-0.44 fresh success with few drops and slow spin as its failure mode. Unless the advisor can name a mechanism these results do not already predict, the direction's sequence is complete and the loop should stop and report.
- D-062 on R-086: **inconclusive** — 1. Re-nominate R-076: reshape the stroke within each roll cycle
  Re-nominated R-076 on the D-058 incumbent: a within-cycle warp of the stroke phase, h(phi) = phi + epsilon sin(phi) on the pen's own roll angle (epsilon -0.35 or +0.35) with per-finger channels re-centred and rescaled to the unwarped cycle mean and RMS (constants frozen on a 3600-point grid), everything else unchanged; identity bit for bit at epsilon 0. Both warps and the parent screened on 40 frozen cases (warp -0.35 17/40, parent 16/40, warp +0.35 13/40), the winner frozen, winner and parent on promotion (10-39) and fresh (80-159) of the same seed-4690 manifest; 340 episodes, no LLM. Predicted at least +0.10 fresh success over the parent; observed +0.0875 [+0.015, +0.160] (32 vs 25 of 80; 8 wins, 1 loss, 71 ties; drops 3 vs 3): an improvement is established on the fresh panel but the predicted magnitude is neither supported nor refuted. Promotion +0.100 [-0.014, +0.214] (10 vs 7; 3 wins, 0 losses; drops 4 vs 4) misses the strict lower bound by 0.014, so the warp is not adopted and the incumbent stands. The negative warp slows the stroke's phase around the roll angle's zero and speeds it near +-pi; the positive warp did the opposite and screened below the parent. In-play |spin| 4.38 vs 4.04 and slow-spin tail failures 39 vs 47. This is the first stroke-shape change with an established fresh gain on the adopted incumbent and the fourth candidate on this learner whose 30-pair promotion panel landed just short of the strict rule while a fresh panel favoured it; the 90-pair instrument that qualified the clamp (D-057) is the predeclared way to settle it.
  Next step then proposed: The negative warp is the first stroke-shape change with an established fresh gain on the adopted incumbent (and the only candidate besides the fixed gain to win against it). Qualify it the way the clamp was qualified: one predeclared 90-pair instrument (three cases-10-39 panels) under the standing rule, 180 episodes, then adopt or not. If it qualifies, it becomes the incumbent and the warp magnitude could be swept open-loop (one more screen) before any further mechanism work; if not, the incumbent stands and the sequence is complete.
- D-061 on R-064: **refuted** — Close spin feedback through the stroke's phase lead, not its amplitude
  Revised onto the D-058 incumbent, whose stroke phase is locked to the pen's measured roll angle (no tempo), so the timing lever is the phase lead. Six feedback settings (lead offset = clip(gain (target - y), +-0.6 rad), y a 0.25 s low-pass of the signed axial rate; targets 4, 6, 8 rad/s x gains +-0.10) and six constant lead offsets (+-0.1, +-0.2, +-0.4 rad) were screened with the parent on a frozen 40-case manifest (one process each), the winner per family frozen (feedback: target 6, gain +0.10; constant: +0.2 rad), and both winners and the parent scored on the confirmation manifest's promotion (10-39) and fresh (40-119) cases; identity gate bit for bit; 850 episodes, no LLM. Predicted at least +0.10 fresh success for feedback over the constant winner; observed -0.025 [-0.112, +0.062] (26 vs 28 of 80): refuted. Neither family beats the parent (fresh feedback -0.088 [-0.189, +0.014], constant -0.063 [-0.144, +0.019]; promotion 12 and 11 vs 10, neither strict); no adoption, the incumbent stands. The twelve lead settings screened flat (11-16 of 40 vs the parent's 13): the phase lead is not a lever for this controller in either form. The feedback was active (mean offset +0.39 rad, a third of steps at the clip) but its signed-rate target is ill-posed: the filtered signed rate averaged about 1 rad/s against an in-play |spin| of 4.5 because the parent's roll direction differs across cases; any future rate feedback on this learner must use an unsigned or per-case-direction rate. With D-058 (amplitude feedback), D-060 (R-043) and this, every mechanism the direction named has been run on the adopted incumbent; the stroke amplitude (D-058's fixed gain) is the one lever that moved it.
  Next step then proposed: The stroke's amplitude (D-058's fixed gain) was the one lever that moved this learner; its phase lead is not one, and instantaneous or filtered signed-rate feedback has now failed on both. Any further feedback on this learner must use an unsigned or per-case-direction rate. With R-043 (D-060) and this, every mechanism the direction named has been run on the adopted incumbent; the direction's sequence is complete unless the user extends it.
- D-060 on R-084: **inconclusive** — R-043 on the current parent: anatomical versus joint residual coordinates
  The R-043 branch on the D-058 incumbent: an LLM wrote bounded residual corrections (executed action = clamp(parent + clamp(D c, -0.15, 0.15), -1, 1)) in anonymous orthonormal coordinates, D the identity (joint arm) or a frozen synergy basis (anatomical arm), from one shared request; 8 slots per arm with one linked correction (31 calls, 459k tokens; 8/8 and 6/8 valid), each candidate screened on 20 frozen cases in its own process, finalists frozen (slot 7 in both arms, 9/20 and 8/20), then the finalists and the parent on the confirmation manifest's promotion (10-39) and fresh (40-119) cases (610 episodes; identity gate bit for bit on 23,484 archived observations). Predicted at least +0.05 fresh success for the anatomical finalist over the joint finalist; observed -0.025 [-0.104, +0.054] (33 vs 35 of 80; 4 wins, 6 losses, 70 ties): the interval spans zero and its upper bound is not below +0.05, so neither supported nor refuted. Neither finalist differs from the parent (fresh 35 and 33 vs 34; promotion 12 and 14 vs 14); no strict promotion, no adoption; the D-058 incumbent stands. The traces explain it: the residuals used about a tenth of their authority (mean executed-minus-parent norm 0.024 joint, 0.013 anatomical against the 0.15 bound), so both arms are the parent plus a small perturbation and the coordinate basis was never exercised; with anonymous coordinates the model cannot aim a correction and writes a cautious one. Deviation stated: candidates were screened in their own 20-environment processes rather than one tiled process per arm, the same batch shape for both arms. With this, every branch the direction named has been run: scalar tuning (D-048, D-049), the lesion panel and the curl removal that qualified and was adopted (D-050 to D-059), stroke feedback (D-058) and R-043 (this); the open problem on the incumbent is its slow-spin tail.
  Next step then proposed: Treat R-043 as answered for this procedure: with anonymous coordinates and one cohort, LLM residuals on this incumbent are too small to move anything, in either basis. If representation is tested again it needs either disclosed coordinate semantics (R-085's grounding arm, which the deferral makes conditional on an advantage that did not appear) or a residual that is forced to use its authority (a minimum-norm or screened-amplitude contract). Otherwise the D-058 incumbent stands and the slow-spin tail is the open problem; the direction's branches (tuning, repair of the winner, R-043) have all been run.
- D-059 on R-083: **supported** — Adopt the fixed-stroke control, with one-shot confirmation only if required
  Administrative adoption, on the agreed zero-spend branch, of R-081's constant-gain control as the pen-spin incumbent: the D-057-qualified zero-curl clamp of the frozen R-046 learner with its periodic stroke (the rolling * gait term) scaled by the fixed 1.4081491479357957 and no feedback. Its comparison against the qualified parent in R-081 was prospective, on frozen seed-4663 panels, with no selection among candidates: promotion +0.133 [+0.004, +0.262] (16 vs 12; the strict rule passed), fresh +0.150 [+0.049, +0.251] (35 vs 23), drops equal on both panels (3 vs 3, 4 vs 4); every adoption criterion met, a control designation being no disqualification. No episode, no LLM call, seed 4670 unused and its fallback manifest never built. The exact evaluated artefacts are frozen with sha256 hashes under config/pen_spin/d058_fixed_stroke_incumbent/ (parent source and state, the evaluated substituted source, constant_gain.json, incumbent.json) and verified by scripts/isaac/adopt_fixed_stroke.py. The incumbent's remaining failure mode is slow spin (38 of 80 fresh tail failures) with few drops (4 of 80); R-084 (the R-043 branch) runs on it as parent.
  Next step then proposed: R-084 (the R-043 branch) runs on this incumbent as its parent.

## Available protocol scripts (reuse before writing new ones)
scripts/evaluate_on_fresh_manifest.py, scripts/final_report.py, scripts/freeze_method.py, scripts/lock_baseline.py, scripts/plot_training_dynamics.py, scripts/research.py, scripts/run_full_evolution.py, scripts/run_shadowhand_spin_stages.py, scripts/summarise_experiments.py
## Available configs
config/assembly_two_lane_repair.yaml, config/debug_smoke_two_generation.yaml, config/environment.yaml, config/fittable_bank/manifest_provenance.json, config/fittable_bank/push-v2_development_confirmation_s2300.yaml, config/fittable_bank/qualification.yaml, config/fittable_bank/qualification_v2.yaml, config/fittable_bank/seed_bank.json, config/fittable_bank/structure_roles.yaml, config/fittable_bank/variants_frozen.json, config/pen_spin/d058_fixed_stroke_incumbent/constant_gain.json, config/pen_spin/d058_fixed_stroke_incumbent/incumbent.json, config/pen_spin/d058_fixed_stroke_incumbent/research_result.json, config/pen_spin/r054_evidence/expert.json, config/pen_spin/r054_evidence/self.json, config/pen_spin_plateau_repair.yaml, config/run_full_evolution.yaml, config/shadowhand_spin_stages.yaml, config/stage_e/manifest_provenance.json, config/stage_e/push-v2_development_confirmation_s2200.yaml, config/stage_e/selection.json, config/stage_e/stage_e.yaml, config/stage_f/manifest_provenance.json, config/stage_f/push-v2_development_heldout_s2500.yaml, config/stage_f/selection.json, config/stage_f/stage_f.yaml, config/stage_f_vision/manifest_provenance.json, config/stage_f_vision/probes/push-v2_probe_s2620.yaml, config/stage_f_vision/probes/push-v2_probe_s2621.yaml, config/stage_f_vision/probes/push-v2_probe_s2622.yaml, config/stage_f_vision/probes/push-v2_probe_s2623.yaml, config/stage_f_vision/probes/push-v2_probe_s2624.yaml, config/stage_f_vision/probes/push-v2_probe_s2625.yaml, config/stage_f_vision/probes/push-v2_probe_s2626.yaml, config/stage_f_vision/probes/push-v2_probe_s2627.yaml, config/stage_f_vision/probes/push-v2_probe_s2628.yaml, config/stage_f_vision/push-v2_development_heldout_s2629.yaml, config/stage_f_vision/stage_f_vision.yaml, config/two_lane_search.yaml, config/two_lane_search_v2.yaml
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
