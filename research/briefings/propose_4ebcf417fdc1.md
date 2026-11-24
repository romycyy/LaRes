# Research briefing (propose) — generated 2026-11-23T21:51:50

## User's research direction
Continue Stage-3 Shadow Hand axial-rolling research from the frozen R-121 JOINT controller. Prioritize independent training reproducibility, then robustness of frozen controllers. Reproduce the successful residual-plus-recovery construction across independent training seeds before claiming a reliable learning algorithm. Evaluate longer horizons and predeclared physical/reset perturbations without fitting. Use archived failures to decide whether recovery hand-back deserves a state-dependent transition study. Then investigate whether LLM-guided construction can discover useful recovery-module structure against grammar search and the hand-designed template under matched budgets. Preserve the original six-second task and evaluation contract. Use JOINT as the experimental reference, preserve all existing assets, and leave default-incumbent installation separate. Run one protocol at a time through the existing advisor/implementer loop. Continue justified follow-ups within this direction; honor its escalation and exhaustion rules. No new global budget ceiling is imposed; each protocol must declare finite simulator, wall-time and model-call costs before agreement. No controller-generation calls for Q1–Q3. Q4 requires a separately agreed generation budget. Advisor calls for the research loop remain permitted throughout.  _(set 2026-11-22)_
- earlier direction (2026-11-21 to 2026-11-22): No LLM calls in any protocol. Test bounded residual repairs of the working fixed-stroke incumbent (config/pen_spin/d058_fixed_stroke_incumbent) on Stage 3, in four separately gated steps: (S1) reproduce the incumbent on fixed cases and quantify its slow-spin failures, with angular progress measured along each case's own roll direction; (S2) qualify the contact-frame decoder DURING ROLLING on top of the incumbent stroke, reporting requested, feasible (after bounds and least squares) and achieved motion separately, with tracking error, directional agreement and bound activity as pass/fail criteria (controllability gate); (S3) only if S2 passes, compare incumbent alone, a hand-authored contact-frame residual and a hand-authored anonymous-coordinate joint residual, with matched tuning budgets (same parameter count, same number of probe evaluations, same tuning method, tuned on probe cases and frozen before any fresh case), candidates ranked by tail-hit fraction with sustained unsigned angular progress and retention inspected; (S4) evaluate the selected controllers on fresh paired cases, 90 pairs per comparison, Holm correction over the two comparisons, promotion requires the strict paired sustained-success rule AND a pre-declared numeric non-inferiority bound on drops (usefulness gate). Residuals must not assume a roll sign. If S2 fails, the anonymous residual may still be compared against the incumbent alone. A failed hand-authored controller shelves that decoder/controller family only. No BC, RL, curriculum or gradient fitting. One protocol at a time on the box.
- earlier direction (2026-11-22 to 2026-11-22): No LLM calls in any protocol. Two finite follow-ups to D-069 and D-070 on Stage 3, on the fixed-stroke incumbent (config/pen_spin/d058_fixed_stroke_incumbent). (A) A transfer audit: on about 40 fresh cases, run the incumbent alone and with forefinger/ring circumferential pulses along and against each case's own roll direction, logging clamp activity, joint motion, fingertip and pen-surface velocity, and contact impulses; the prediction is on paired roll progress, pro minus anti. (B) One confirmatory replication of D-070's frozen residual, unchanged, on about 670 fresh pairs against the incumbent, with the strict paired success rule and the drop bound. No retuning, no new controller search, no further replication after (B). One protocol at a time on the box.
- earlier direction (2026-11-22 to 2026-11-22): Continue Stage-3 Shadow Hand axial rolling by testing reward-trained symbolic residuals and incremental policy construction. Use the frozen D-072-qualified controller as the experimental reference, without changing the saved default incumbent. First compare independent parameter draws, ARS/CEM and PPO on one identical sparse, state-conditioned symbolic residual family. If an inner learner produces a confirmed useful gain, compare fixed structure, grammar-based structural evolution and LLM-guided structural evolution with that learner and inherited parameters. Investigate compact memory or anchored component replacement only when the preceding results motivate them. Preserve the task, frozen evaluation manifests and strict paired promotion/drop criteria. Run one protocol at a time. No controller-generation LLM calls during P1; advisor calls for the research loop remain allowed. Later controller-generation calls need an agreed protocol and explicit call budget. Do not repeat R-102, retune its confirmation cases, reopen cold whole-controller generation, or resume expert cloning. Continue evidence-driven follow-ups within this direction through the existing loop, including its escalation and exhaustion rules.

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
- Spent seeds (consumed by a run or a committed manifest nobody open owns; never reuse): [0, 1, 2, 1000, 1001, 1002, 2000, 2100, 2200, 2201, 2202, 2203, 2204, 2300, 2301, 2302, 2303, 2304, 2400, 2401, 2402, 2403, 2404, 2410, 2411, 2412, 2413, 2414, 2420, 2421, 2422, 2423, 2424, 2500, 2501, 2502, 2600, 2601, 2602, 2603, 2604, 2605, 2606, 2607, 2610, 2611, 2612, 2613, 2614, 2615, 2616, 2617, 2618, 2619, 2620, 2621, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 2700, 2701, 2702, 2703, 2704, 2705, 2710, 2800, 2801, 2802, 2803, 2804, 2805, 2810, 2811, 2812, 2813, 2820, 2822, 2823, 2824, 2825, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2910, 2911, 2912, 2913, 2920, 2921, 2922, 2923, 2940, 3000, 3001, 3010, 3011, 3012, 3013, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3030, 3031, 3032, 3033, 3040, 3041, 3042, 3043, 3044, 3045, 3046, 3047, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3130, 3131, 3132, 3133, 3134, 3135, 3200, 3201, 3202, 3205, 3210, 3211, 3212, 3213, 3214, 3215, 3220, 3221, 3222, 3223, 3300, 3301, 3302, 3303, 3304, 3305, 3310, 3311, 3312, 3313, 3314, 3315, 3320, 3321, 3330, 3331, 3332, 3333, 3334, 3335, 3400, 3410, 3420, 3421, 3422, 3500, 3501, 3502, 3503, 3505, 3506, 3507, 3520, 3521, 3522, 3523, 3524, 3525, 3526, 3527, 3528, 3529, 3540, 3541, 3544, 3600, 3601, 3602, 3603, 3604, 3605, 3606, 3607, 3608, 3609, 3710, 3711, 3712, 3713, 3720, 3721, 3722, 3723, 3724, 3725, 3726, 3800, 3801, 3802, 3803, 3804, 3805, 3806, 3807, 3810, 3811, 3812, 3813, 3814, 3815, 3816, 3817, 3818, 3820, 3821, 3822, 3823, 3824, 3825, 3900, 3901, 3902, 3903, 3904, 3910, 3911, 3912, 3913, 3914, 4000, 4001, 4002, 4010, 4011, 4012, 4020, 4030, 4031, 4032, 4097, 4098, 4099, 4100, 4101, 4120, 4121, 4122, 4123, 4124, 4125, 4126, 4127, 4128, 4129, 4130, 4131, 4132, 4133, 4134, 4135, 4136, 4137, 4140, 4170, 4171, 4172, 4173, 4174, 4180, 4181, 4182, 4183, 4184, 4190, 4191, 4192, 4193, 4200, 4210, 4211, 4212, 4213, 4214, 4215, 4220, 4221, 4222, 4223, 4224, 4225, 4226, 4227, 4310, 4400, 4500, 4501, 4502, 4503, 4504, 4505, 4506, 4507, 4510, 4511, 4512, 4513, 4514, 4515, 4516, 4517, 4550, 4551, 4552, 4553, 4554, 4555, 4556, 4557, 4560, 4580, 4581, 4610, 4620, 4631, 4642, 4653, 4660, 4661, 4662, 4663, 4672, 4673, 4690, 4700, 4701, 4702, 4710, 4711, 4712, 4713, 4714, 4715, 4716, 4717, 4720, 4721, 4722, 4723, 4724, 4730, 4731, 4732, 4733, 4734, 4735, 4736, 4737, 4740, 4741, 4742, 4744, 4750, 4751, 4752, 4753, 4754, 4755, 4756, 4757, 4758, 4764, 4765, 4766, 4767, 4768, 4769, 4780, 4781, 4782, 4783, 4784, 4785, 4786, 4787, 4788, 4789, 4790, 4791, 4822, 4823, 4824, 4825, 4826, 4827, 4828, 4829, 4830, 4831, 4832, 4833, 4856, 4857, 4858, 4859, 4860, 4861, 4862, 4863, 4864, 4865, 4866, 4867, 4868, 4869, 4870, 4871, 4872, 4873, 4874, 4875, 4876, 4877, 4878, 4879, 4880, 4881, 4882, 4883, 4884, 4885, 4886, 4887, 4888, 4889, 4890, 4891, 4895, 4896, 4897, 4898, 4899, 4900, 4901, 4902, 4903, 4904, 4905, 4906, 4942, 4943, 4944, 4945, 4946, 4947, 4948, 4949, 4950, 4951, 4952, 4953, 4971, 4972, 4973, 4974, 4975, 4976, 4977, 4978, 4979, 4980, 4981, 4982, 4999, 5000, 5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009, 5010, 5013, 5014, 5015, 5016, 5017, 5018, 5019, 5020, 5021, 5022, 5023, 5024, 5025, 5041, 5042, 5043, 5044, 5045, 5046, 5047, 5048, 5049, 5050, 5051, 5052, 5053, 5054, 5055, 5056, 5057, 5058, 5059, 5060, 5061, 5062, 5063, 5076, 5077, 5078, 5079, 5080, 5081, 5082, 5083, 5084, 5085, 5086, 5087, 5088, 5089, 5090, 5091, 5092, 5093, 5114, 5115, 5116, 5117, 5118, 5119, 5120, 5121, 5122, 5123, 5124, 5125, 5126, 5127, 5128, 5129, 5130, 5131, 5132, 5133, 5134, 5135, 5136, 5137, 5240, 5241, 5242, 5243, 5244, 5245, 5246, 5247, 5248, 5249, 5250, 5251, 5252, 5253, 5254, 5255, 5256, 5257, 5258, 5259, 5260, 5261, 5262, 5263, 5264, 5265, 5266, 5267, 5268, 5269, 5270, 5271, 5272, 5273, 5310, 5311, 5312, 5313, 5314, 5315, 5316, 5317, 5318, 5319, 5320, 5321, 5354, 5355, 5356, 5357, 5358, 5359, 5360, 5361, 5362, 5363, 5364, 5365, 5366, 5367, 5368, 5369, 5370, 5371, 5372, 5373, 5374, 5375, 5376, 5377, 5378, 5379, 5380, 5381]
- The final-test split is locked. No protocol may read or score it.

## Open ideas (already proposed; do not duplicate)
(none)

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
- R-098 I3: Diagnose fingertip-to-pen transfer after a failed useful repair: This 120-episode, zero-LLM diagnostic could distinguish fingertip tracking from object-level transfer, but only after R-097 passes and R-099's contact controller fails usefulness qualification. Its we
- R-105 Conditional objective-only ARS ablation with a sustained-tail return: The objective-only contrast addresses a mechanism D-073 left unresolved, but its 18,444-episode ceiling and 0 controller-generation LLM calls should wait for R-104's alignment and power findings. Its 
- R-107 2. Conditional anchored finger-chain replacement versus simultaneous fitting: Anchored replacement is a distinct intervention, but its 91,200 estimated simulator starts, 94,000-start ceiling and zero controller-generation LLM calls should wait for R-106 and the existing escalat
- R-111 2. Conditional P4: construct native posture and stroke components: Native posture-then-stroke fitting tests a different intervention from deferred finger-chain replacement, motivated by the residual-family plateau, but its capture/posture dependency remains unresolve
- R-113 2. Conditional fallback: learn an odd roll-direction residual: Direction-odd feedback differs from D-077's unsigned terms, but the negative-direction failure association may reflect poor capture rather than a repairable cause. Its estimated 225,000 simulator epis
- R-115 2. Conditional R-111 revision: native posture then stroke construction: This explicitly revises R-111 by replacing weak reset-geometry evidence with causal intervention evidence, so the earlier deferral does not settle the revised mechanism. Its paired_success_difference 
- R-117 2. Conditional: event-terminated versus timed recovery support: Event-terminated support tests a distinct hand-back mechanism rather than repeating D-079's filtered residuals, but it is justified only after R-116 establishes useful recovery and traces implicate pr
- R-119 2. Conditional joint adaptation of the residual and recovery module: Joint adaptation tests coupling left unresolved by D-077 and D-081, but its 406,912 episodes and zero LLM calls should wait until R-118 determines the appropriate structure and qualified parent. The p
- R-123 2. Conditional: trigger the single recovery from an early slow latch: Wait for R-122: this protocol is justified only if neither finalist qualifies over JOINT and the JOINT archive still shows substantial slow-at-step-60 failures awaiting recovery. Advancing the sole re
- R-125 2. Conditional R-123 revision: frozen early-latch authority gate: Using persistent early slowness to advance the sole re-capture tests discrete recovery authority, unlike D-079's additive memory residual or D-085's capture offsets. Its weakest point is whether enoug
- R-127 Matched-budget staged versus simultaneous residual–recovery fitting: The matched initialization and parameter-release mask isolate construction schedule, a contrast not settled by D-084's coordinate comparison or D-085's continuation from JOINT. However, its 404,864 si

## Recently rejected at triage (do not re-propose without saying what changed)
- R-074 Audit whether capture-curl impulses predict drops: I accept the implementer's decisive argument: this association cannot establish impulse causality, and neither a positive nor negative result would change the next decision to test R-075. Extending sc
- R-090 2. Conditional open-loop contact-dwell calibration: I accepted the implementer's deferral while R-089 was pending, but D-063 now records failed qualification and more drops, triggering this protocol's explicit cancellation condition. Spending 420 episo
- R-091 3. Abstain: no third research protocol within this focus: This is a useful administrative stop constraint, but not a separate experiment: R-089 already prohibits retries, and R-090 already cancels after failed qualification. Its zero-episode, zero-LLM comman
- R-100 Resolve D-070 with 360 fresh pairs and the unchanged frozen residual: This frozen-controller replication addresses uncertainty left open by D-070; its predicted magnitude is falsified if the paired interval upper bound is below +0.05. However, spending 720 simulator epi
- R-132 1. Diagnose recovery-training dependence on reset-table reuse: I accept the implementer's audit: update-specific placement slices and 184,320 distinct placement hashes remove the historical reset-reuse premise. Comparing REFRESH with an artificial FIXED arm canno
- R-136 3. Audit Q4's selection instrument using archived finalists: I accept the implementer's decisive objection: none of the 15 frozen finalists has a parameter-matched search score because scoring preceded the 12-update continuation. The registered paired_success_d

## Decided ideas (most recent first)
- D-095 on R-137: **inconclusive** — 1. Prospectively isolate JOINT's initial-tilt sensitivity
  Prospective follow-up to D-094: frozen JOINT, PPO and D-072 on four fresh manifests under reset conditions that scale the rotation draws and the position / joint draws separately. Confirmatory: JOINT tilt-only 1.5x minus JOINT position-only 1.5x on 512 identical cases -0.139 [-0.189, -0.089]; the direction is established, the registered -0.15 is not (inconclusive). Against nominal (0.777) JOINT loses 0.225 under tilt-only, 0.086 under position-only and 0.316 under both; PPO loses 0.105 and D-072 0.104 under tilt-only. The wide-reset weakness is mostly an initial-orientation effect, larger for JOINT than for the expert.
  Next step then proposed: The wide-reset weakness is mostly an initial-orientation effect, shared in part by the expert. Improving it is a capture-authority question (early drops before the module acts), outside what ARS continuation, mixtures or module edits changed in this direction: a new direction for the user.
- D-094 on R-135: **inconclusive** — 2. Localize frozen JOINT's reset weakness without another rollout
  Archive-only audit of the frozen JOINT controller at 1.5x resets by reconstructed initial geometry. Registered: standardized highest-minus-lowest transverse-displacement quartile success -0.056 [-0.177, +0.064] on R-129's 512 cases (inconclusive against -0.15); separate replication on R-133's 1,536 cases -0.172 [-0.239, -0.105]. Descriptive and identical in both archives: success falls with the pen's initial tilt, 0.81 / 0.59 / 0.38 / 0.16 and 0.82 / 0.53 / 0.33 / 0.18 across tilt quartiles, while axial displacement is flat; the wide-reset weakness of D-089 is mostly an initial-orientation effect (unregistered; needs a prospective test). No episode was run.
  Next step then proposed: The wide-reset weakness looks like an initial-orientation effect; that is a new mechanism to test prospectively (for example frozen JOINT on resets that scale tilt and position separately) and, if it holds, a capture-authority question beyond this direction's scope. Hand the choice of direction back to the user.
- D-093 on R-134: **refuted** — 1. Archive-first: does long recovery ramp-out suppress Stage-B rolling?
  Archive-only audit of R-128's stage-B cells: among re-captured episodes the rolling-step fraction rises across the end of recovery support by 0.166 / 0.321 / 0.186 / 0.092 / 0.313 in lineages 0-4; the weak lineage 2 minus the mean of the others is -0.037 [-0.064, -0.011] against a predicted +0.10: refuted. All five stage-B modules have long ramp-outs and stroke multipliers near 1, so lineage 2 is not distinctive at that stage. No episode was run.
  Next step then proposed: Do not pursue support-release explanations of the Q1 variance. Run the already approved R-135 archive audit, then let the loop's exhaustion rule return the decision to the user.
- D-092 on R-133: **refuted** — Mixed-reset training of the JOINT controller against matched nominal continuation
  Follow-up to D-089: five paired blocks of JOINT continuation (36 ARS updates of 43 coordinates) that differ only in writing half of each training batch at reset scale 1.5 (reset_mixture.py). MIXED minus NOMINAL on fresh expanded resets +0.004 [-0.012, +0.021] against a registered +0.08: refuted, no directional improvement. Expanded-reset success: frozen JOINT 0.462, NOMINAL arm 0.465, MIXED arm 0.469; nominal resets 0.771 / 0.766 / 0.767; no endpoint improves on JOINT in either condition and 0 of 5 MIXED endpoints are useful. Training-distribution coverage is not the lever for the wide-reset weakness in this controller family. 3.03 M starts, 336 minutes, no model calls.
  Next step then proposed: Training-distribution coverage is not the lever for the wide-reset weakness inside this controller family. With Q1-Q4 and both named follow-ups decided, do not propose further ARS continuation or mixture variants; the remaining failure mass (early drops before the module acts, off-centre and persistent-slow captures from wide resets) points at capture authority, which is a new direction for the user to set.
- D-091 on R-131: **inconclusive** — Q4: Matched-budget recovery-module construction
  Q4 of direction V2: on the frozen R-106 residual, five independent blocks of three matched arms (the fixed template, grammar edits and LLM edits of the recovery module in a 13-edit language; module_language.py, module_search.py), every arm 36 ARS updates with one frozen ranking rule, 15 finalists and four frozen controls on twelve fresh manifests (1,536 cases each). LLM minus grammar +0.030, five-block 95% interval [-0.017, +0.076]: inconclusive, no LLM advantage established. Arm means: fixed 0.742, grammar 0.713, LLM 0.743; neither search beats the fixed template (LLM minus fixed +0.001 [-0.095, +0.097]); all beat the root (0.572) by 0.14-0.17; none reaches the frozen JOINT (0.801). 20 generation calls of 40 allowed, all valid first time, 49,014 tokens; 4.64 M starts, 483 minutes. The LLM chose blend-target edits first and boost/trigger edits second; the ranking kept target midrange (three blocks) and event-held support (two). A smoke-gate tolerance was amended before any generation (advisor consult call_00310) and a first launch died on an implementer bug in the smoke.
  Next step then proposed: Do not repeat this comparison at the same size. If construction is pursued, the bottleneck to address first is the selection instrument (4 updates and 512 placements cannot rank structures a few points apart) and the between-block variance, not the proposer. The direction's four questions are now each decided once; propose only a follow-up that the results justify (for example the reset-distribution weakness from D-089), otherwise let the loop stop for a new direction.
- D-090 on R-130: **refuted** — Q3 first spend: archive-only recovery hand-back diagnosis
  Q3 of direction V2, first spend (archive only, no episode): the archived JOINT and transplant cells of R-124 and R-126 aligned to the frozen module's decision (step 149, reconstructed per case from the trace), the end of its support (183) and the end of its stroke boost (243). Failures that follow a healthy hand-back within 30 steps are 48 of 3,072 JOINT cases (0.0156 [0.011, 0.020]) and 67 of 4,096 (0.0164 [0.012, 0.020]), the transplant 40 of 4,096, against a 0.06 gate; 0.018-0.021 even with a 60-step window. Q3 closes without a hand-back simulation. JOINT's failures are elsewhere: 7.8% of cases fail untriggered (mostly early drops), 7.2% trigger and roll again but are not rolling steadily when support and boost end, 3.2% trigger and never roll again. Trigger flags are reconstructed (stable under one-step re-indexing; no cell exceeds its recorded count).
  Next step then proposed: Close Q3. The archive points elsewhere: 7.2% of cases trigger and re-capture without reaching steady rolling by the end of support and boost, 3.2% trigger and never roll again, 7.8% fail untriggered (mostly early drops). Move to Q4 (LLM vs grammar construction of the recovery module) with a separately agreed generation budget; R-129's reset-distribution weakness is a separate training-distribution question.
- D-089 on R-129: **supported** — Q2: Frozen-controller retention and one-factor robustness
  Q2 of direction V2: four frozen controllers (R-121 JOINT, the R-126 transplant, D-072, the PPO expert) on a predeclared one-factor panel, 204 cells on ten fresh manifests, no fitting (frozen_robustness.py). Primary: JOINT at 0.8x pen friction minus nominal JOINT -0.015 [-0.035, +0.005] over 1,280 pairs, inside the -0.05 margin (non-inferiority supported; not an improvement). Horizons: no retention loss, JOINT's terminal success 0.775 at 12 s and 0.781 at 30 s against 0.766 on the first six seconds, no drop after second 4. Mass, friction, radius and length at +/-10-20% leave JOINT within -0.023 to +0.041 of nominal (512 pairs each, nothing Holm-significant). The expanded reset distribution (1.5x) is the weakness: JOINT 0.770 -> 0.486 (-0.281 [-0.331, -0.232]), transplant -0.322, D-072 -0.141, PPO -0.125. The transplant is within +/-0.035 of JOINT in every condition. PPO 0.994 nominal, 0.887 at 0.9x length. Re-capture trigger counts are unavailable in evaluation cells.
  Next step then proposed: The robustness gap is capture from wider resets, not dynamics or retention. That is a training-distribution question (propose separately, never retune on this panel). Proceed to Q3's archive-only diagnosis of recovery hand-back on JOINT/transplant traces, noting that evaluation cells do not record trigger events.
- D-088 on R-128: **inconclusive** — Q1: Five independent replications of the frozen construction recipe
  Q1 of direction V2: the R-106 -> R-116 -> R-118 -> R-121 construction recipe repeated in five independent training lineages from the D-072 reference (training_replication.py), every endpoint evaluated with D-072 and the frozen historical JOINT asset on 24 fresh manifests (3,072 cases per policy). Equal-weight mean JOINT-stage gain over D-072 +0.308, two-level 95% interval [0.247, 0.370] on 4 df: inconclusive against the registered +0.30, far from zero. Per-lineage JOINT-stage success 0.751, 0.750, 0.667, 0.767, 0.781 (mean 0.743, sd 0.044) against D-072 0.435 and the historical JOINT 0.792, which beats all five (the historical lineage was a favourable draw). Stage means: A 0.546 (sd 0.010), B 0.651 (0.052), C 0.730 (0.036), D 0.743. All five stage-D endpoints qualify against D-072; stages A, B, C qualify over their parent in 5 of 5 lineages; the JOINT stage qualifies over stage C in 1 of 5. No child death, no rerun; 3,105,435 starts, 422 minutes, no model calls. An implementer bug in the divergence check (float32 bound) was fixed before any outcome; finished stages were reused unchanged.
  Next step then proposed: Treat the recipe as reproducible in sign and large in size but with a between-lineage sd of about 0.04 and one clearly weaker lineage; keep the historical JOINT asset as the frozen reference for Q2 (robustness of frozen controllers), reporting policy robustness separately from learning reliability. A training-variance diagnosis of stage B (lineage 2's long ramp-out) is a candidate follow-up, archive-first.

## Available protocol scripts (reuse before writing new ones)
scripts/evaluate_on_fresh_manifest.py, scripts/final_report.py, scripts/freeze_method.py, scripts/lock_baseline.py, scripts/plot_training_dynamics.py, scripts/research.py, scripts/run_full_evolution.py, scripts/run_shadowhand_spin_stages.py, scripts/summarise_experiments.py
## Available configs
config/assembly_two_lane_repair.yaml, config/debug_smoke_two_generation.yaml, config/environment.yaml, config/fittable_bank/manifest_provenance.json, config/fittable_bank/push-v2_development_confirmation_s2300.yaml, config/fittable_bank/qualification.yaml, config/fittable_bank/qualification_v2.yaml, config/fittable_bank/seed_bank.json, config/fittable_bank/structure_roles.yaml, config/fittable_bank/variants_frozen.json, config/pen_spin/d058_fixed_stroke_incumbent/constant_gain.json, config/pen_spin/d058_fixed_stroke_incumbent/incumbent.json, config/pen_spin/d058_fixed_stroke_incumbent/research_result.json, config/pen_spin/r054_evidence/expert.json, config/pen_spin/r054_evidence/self.json, config/pen_spin/r106_success_ars_finalist/descriptor.json, config/pen_spin/r106_success_ars_finalist/finalist.json, config/pen_spin/r116_recovery_finalist/descriptor.json, config/pen_spin/r116_recovery_finalist/finalist.json, config/pen_spin/r118_continue_finalist/descriptor.json, config/pen_spin/r118_continue_finalist/finalist.json, config/pen_spin/r121_joint_finalist/descriptor.json, config/pen_spin/r121_joint_finalist/finalist.json, config/pen_spin/r126_transplant_finalist/finalist.json, config/pen_spin/symbolic_reward_reference/amendment_r106_smoke.json, config/pen_spin/symbolic_reward_reference/amendment_r109_selection.json, config/pen_spin/symbolic_reward_reference/amendment_two_blocks.json, config/pen_spin/symbolic_reward_reference/reference.json, config/pen_spin_plateau_repair.yaml, config/run_full_evolution.yaml, config/shadowhand_spin_stages.yaml, config/stage_e/manifest_provenance.json, config/stage_e/push-v2_development_confirmation_s2200.yaml, config/stage_e/selection.json, config/stage_e/stage_e.yaml, config/stage_f/manifest_provenance.json, config/stage_f/push-v2_development_heldout_s2500.yaml, config/stage_f/selection.json, config/stage_f/stage_f.yaml, config/stage_f_vision/manifest_provenance.json, config/stage_f_vision/probes/push-v2_probe_s2620.yaml, config/stage_f_vision/probes/push-v2_probe_s2621.yaml, config/stage_f_vision/probes/push-v2_probe_s2622.yaml, config/stage_f_vision/probes/push-v2_probe_s2623.yaml, config/stage_f_vision/probes/push-v2_probe_s2624.yaml, config/stage_f_vision/probes/push-v2_probe_s2625.yaml, config/stage_f_vision/probes/push-v2_probe_s2626.yaml, config/stage_f_vision/probes/push-v2_probe_s2627.yaml, config/stage_f_vision/probes/push-v2_probe_s2628.yaml, config/stage_f_vision/push-v2_development_heldout_s2629.yaml, config/stage_f_vision/stage_f_vision.yaml, config/two_lane_search.yaml, config/two_lane_search_v2.yaml
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
