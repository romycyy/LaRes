# Research briefing (triage) — generated 2026-10-27T14:13:56

## User's research direction
Continue researching the single-step exploration pipeline with plateau-triggered trace repair, extended to pen spinning on the Shadow Hand (Isaac Lab Isaac-ShadowHandSpin-Stage{1,2,3}-Direct-v0, vendored in lares/envs/isaac_pen_spin; 157-d full obs, 20-d action, 60 Hz, 360-step episodes). Expert demonstrations: first try a hand-written scripted spinning controller; also search online for an existing trained pen-spinning policy; if neither qualifies, train an RL expert (rsl_rl PPO on the box). No budget ceiling on GPU hours or LLM calls. Known gaps to turn into questions: the env's spin_rate is roll about the pen's own long axis (not end-over-end twirl), success is per step with no episode rule, and the pipeline (manifests, obs schema, runner, records) is MetaWorld-only today. Debug issues, improve the pipeline, and turn unexpected findings into concrete research questions; document results and limitations.  _(set 2026-10-27)_

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
- Spent seeds (never reuse): [0, 1, 2, 1000, 1001, 1002, 2000, 2100, 2200, 2201, 2202, 2203, 2204, 2300, 2301, 2302, 2303, 2304, 2400, 2401, 2402, 2403, 2404, 2410, 2411, 2412, 2413, 2414, 2420, 2421, 2422, 2423, 2424, 2500, 2501, 2502, 2600, 2601, 2602, 2603, 2604, 2605, 2606, 2607, 2610, 2611, 2612, 2613, 2614, 2615, 2616, 2617, 2618, 2619, 2620, 2621, 2622, 2623, 2624, 2625, 2626, 2627, 2628, 2629, 2700, 2701, 2702, 2703, 2704, 2705, 2710, 2800, 2801, 2802, 2803, 2804, 2805, 2810, 2811, 2812, 2813, 2820, 2822, 2823, 2824, 2825, 2900, 2901, 2902, 2903, 2904, 2905, 2906, 2910, 2911, 2912, 2913, 2920, 2921, 2922, 2923, 2940, 3000, 3001, 3010, 3011, 3012, 3013, 3020, 3021, 3022, 3023, 3024, 3025, 3026, 3030, 3031, 3032, 3033, 3040, 3041, 3042, 3043, 3044, 3045, 3046, 3047, 3100, 3101, 3102, 3103, 3104, 3105, 3106, 3107, 3108, 3109, 3110, 3111, 3130, 3131, 3132, 3133, 3134, 3135, 3200, 3201, 3202, 3205, 3210, 3211, 3212, 3213, 3214, 3215, 3220, 3221, 3222, 3223, 3300, 3301, 3302, 3303, 3304, 3305, 3310, 3311, 3312, 3313, 3314, 3315, 3320, 3321, 3330, 3331, 3332, 3333, 3334, 3335, 3400, 3410, 3420, 3421, 3422, 3500, 3501, 3502, 3503, 3505, 3506, 3507, 3520, 3521, 3522, 3523, 3524, 3525, 3526, 3527, 3528, 3529, 3540, 3541, 3544, 3600, 3601, 3602, 3603, 3604, 3605, 3606, 3607, 3608, 3609, 3710, 3711, 3712, 3713, 3720, 3721, 3722, 3723, 3724, 3725, 3726, 3800, 3801, 3802, 3803, 3804, 3805, 3806, 3807, 3810, 3811, 3812, 3813, 3814, 3815, 3816, 3817, 3818, 3820, 3821, 3822, 3823, 3824, 3825, 3900, 3901, 3902, 3903, 3904, 3910, 3911, 3912, 3913, 3914, 4000, 4001, 4002, 4010, 4011, 4012, 4020, 4030, 4031, 4032, 4097, 4098, 4099, 4100, 4101, 4120, 4121, 4122, 4123, 4124, 4125, 4126, 4127, 4128, 4129, 4130, 4131, 4132, 4133, 4134, 4135, 4136, 4137]
- The final-test split is locked. No protocol may read or score it.

## Open ideas (already proposed; do not duplicate)
### R-044 [proposed] 1. Identify which state feedback the qualified expert needs
Hypothesis: Instantaneous pen-state feedback is necessary for the PPO expert to retain the pen, even when hand proprioception and action history remain available.
Prediction: paired_success_difference decrease by >= 0.3 vs baseline
Falsified by: The paired interval for pen-clamped minus baseline success lies entirely above -0.30. An interval containing zero is not evidence that pen feedback matters.
Protocol: gpu run bash -lc 'source ~/IsaacLab/source_env.sh && python scripts/run_shadowhand_spin_stages.py --config config/shadowhand_spin_stages.yaml --study feedback_lesions --stage 3 --seed 4140 --cases 40 --arms expert,expert_repeat,pen_state_clamped,hand_state_clamped --manifest config/manifests/pen_spin_feedback_lesions_s4140.yaml' (episodes=160, llm_calls=0, seeds=[4140])
Triage so far: 1 round(s); last advisor revise, implementer revise; This cheaply localizes the qualified checkpoint's dependence on feedback channels, unlike D-037's removal of all feedback. However, clamping creates distribution shift, and sustained-success loss does
### R-045 [proposed] Fit dense expert labels into the R-042 prefix controllers, selecting only by rollouts
Hypothesis: Dense expert state-action labels move a frozen symbolic pen-spin controller's bounded parameters toward sustained success more effectively than an equally budgeted rollout-only parameter search over the same structure, bounds and probe cases.
Prediction: paired_success_difference increase by >= 0.15 vs rollout_search arm: the same frozen R-042 prefix controller per block, its selected parameters plus three predeclared bounded draws (rollout_search.perturb: local, global, local) scored on the same 10 probe cases
Falsified by: The paired interval of block-level fresh incumbent differences (dense_fit minus rollout_search) lies entirely below +0.15. An interval containing zero is no evidence of improvement; an all-zero result cannot distinguish insufficient controller capacity from ineffective fitting.
Protocol: gpu run 'source ~/.profile >/dev/null 2>&1; nohup .venv-metaworld/bin/python -u scripts/run_full_evolution.py --config config/pen_spin_plateau_repair.yaml --study dense_feedback_fit > logs_r045.log 2>&1 &' (episodes=1008, llm_calls=0, seeds=[4170, 4171, 4172, 4173, 4174])
Triage so far: 2 round(s); last advisor revise, implementer revise; This isolates whether dense expert labels improve feedback-parameter search, which D-038's sparse prompting did not test. I accept the implementer's seed-collision objection and matched-arm specificat
### R-046 [proposed] 3. Preserve the scripted gait while repairing retention feedback
Hypothesis: An immutable scripted gait preserves contact coordination while a bounded symbolic residual repairs retention failures that whole-controller trace edits otherwise introduce.
Prediction: paired_success_difference increase by >= 0.2 vs baseline
Falsified by: With at least three eligible plateau blocks, the paired interval for residual minus whole-edit fresh incumbent success lies entirely below +0.20. An interval containing zero is not evidence of improvement; fewer than three triggered blocks make the repair comparison inconclusive.
Protocol: gpu run bash -lc 'source ~/IsaacLab/source_env.sh && python scripts/run_full_evolution.py --config config/pen_spin_plateau_repair.yaml --study gait_preserving_repair --stage 3 --block-seeds 4160 4161 4162 4163 --parity-seed 4164 --arms whole_edit,bounded_residual --repair-parent frozen_r039_gait --max-prefix-screen-episodes 120 --repair-variants 6 --screen-cases 10 --promotion-cases 30 --fresh-cases 30 --expert-cases 12 --model gpt-6.1-sol --reasoning-effort high --temperature 1' (episodes=1688, llm_calls=24, seeds=[4160, 4161, 4162, 4163, 4164])
Triage so far: 1 round(s); last advisor revise, implementer revise; Protecting an existing contact scaffold tests a different mechanism from D-038's keyframe selection and directly addresses destructive whole-controller edits. However, without an unchanged-gait contin

## Deferred ideas (judged right but not now; re-triaged when the direction fits)
- R-043 Do anatomical action coordinates improve single-step exploration?: This is a useful action-representation experiment, but its 1536 episodes and at most 16 LLM calls should follow R-042's validated Isaac integration and direct-action exploration results. Its weakest p

## Recently rejected at triage (do not re-propose without saying what changed)
- R-001 Two-lane search beats the unseeded baseline on a fresh pool seed: merged into R-005 (same seed-2610 cases)
- R-002 The gpt-6.1-sol zero-success baseline is model, not draw: Rejected by the advisor (gpt-6.1-sol, consult call_00003) under the user's 2026-10-22 session delegation: a rerun tests recurrence, not a model effect, and its prediction contradicts its hypothesis.
- R-003 Rotating the development screen removes selection on fixed cases: Rejected by the advisor (gpt-6.1-sol, consult call_00015) under the user's 2026-10-22 session delegation: the historical control confounds rotation with candidate draws; reconsideration needs a contem
- R-023 2. Test structural breadth instead of deeper parameter search: Dropped by the advisor (gpt-6.1-sol, consult call) under the user's 2026-10-22 session delegation: its predeclared skip rule applies, since R-022 (D-019) found no task below 0.80; the mechanism is unt

## Decided ideas (most recent first)
- D-038 on R-042: **refuted** — Plateau-triggered trace repair on pen spinning: event-selected vs uniform keyframes, with a no-trace arm
  The predeclared primary, event minus uniform fresh sustained success paired over four blocks, is exactly 0.000 with a degenerate [0,0] interval; the protocol's conclusiveness rule (>=3 eligible blocks) is met with 4/4 triggers fired, and the interval excludes the predicted +0.20, so the prediction as stated is refuted. The caveat is a floor: all twelve continuation arms ended with the null incumbent, so the study cannot rank keyframe rules at any effect size it could detect and does not show trace repair is useless in general. Development evidence agrees there is no trace benefit: promotion-set successes no_trace 2/120, uniform 0/120, event 1/120; exploratory tail-hit contrasts event-uniform -0.047 [-0.195,+0.102], uniform-no_trace +0.079 [-0.088,+0.245]. Trace-arm candidates in blocks 0-1 spin while held (highest tail-hit 0.40, 0.335) but drop the pen in 28-30 of 30 promotion cases, consistent with D-037: the expert's competence is its feedback map, which 32 sparse state-action keyframes from 2 episodes do not convey to a 6-10 parameter controller. Demonstrations 10/10 and parity 0 flips, so the null is not an integration fault.
  Next step then proposed: Stage 3 is at the floor for single-step symbolic exploration; next test a dense, fitted use of the expert (e.g. parameter fitting of the edited controller to expert state-action pairs, rollout-selected) or move the repair study to Stage 1/2 where generated controllers have non-zero base rates, so a trace effect is measurable.
- D-037 on R-041: **supported** — Does the qualified expert need feedback, or only a replayed gait?
  R-041 asked whether the qualified PPO pen-spin expert (D-036) needs state feedback or only a replayable gait. On 40 fresh Stage-3 cases the closed-loop expert succeeded 40/40 while a clock-only replay of its first successful demonstration (359 actions) succeeded 0/40: paired +1.000, supported. The replay dropped the pen in 30/40 episodes (steps 17-355) and the survivors passed 35% of tail steps; end-over-end rate rose from 2.8 to 4.3 rad/s. A pre-launch smoke test showed that even on its own placement the tape tracks the recording only through step 16, then drifts to failure once the batch's other environments differ. The expert's success depends on corrections; a single tape is not a sufficient demonstration. Limitation: one tape; the corrective mechanism is not identified. A smoke test also caught and fixed a tape-slicing bug (the runner stores actions case-major).
  Next step then proposed: Revise R-042: Isaac backend for run_full_evolution and plateau-triggered trace repair on pen spinning, with a no-trace arm beside uniform and event-selected keyframe arms; traces must expose state-action corrections, not only action timelines.
- D-036 on R-040: **inconclusive** — 3. Conditional PPO fallback: does reward optimization yield a usable expert?
  PPO expert (one Stage-3 policy, seed 4020, 1.08e9 transitions, final checkpoint only) qualifies on every stage: 29/30, 30/30, 30/30 sustained vs the frozen gait's 21/13/9 on fresh 30-case manifests (seeds 4030-4032). Paired gains +0.267 [0.099, 0.435], +0.567 [0.378, 0.755], +0.700 [0.526, 0.874]; the predeclared >= +0.10 per-stage prediction is established on stages 2-3 but not stage 1 (interval contains +0.10), so the prediction is inconclusive while the expert qualifies and promotes everywhere. Reproducibility control: zero diverged envs. The only failure is a completed, centred episode missing 1 of 60 tail steps. Spin is axial roll ~17 rad/s, transverse ~2.8 rad/s.
  Next step then proposed: Use model_8239 as the pen-spin trace-repair expert: export traces on a disjoint demonstration manifest and build the pen-spin plateau-repair pipeline.
- D-035 on R-039: **supported** — 2. Can feedback-controlled finger gaiting qualify a scripted expert?
  R-039 tested whether a scripted feedback finger gait could qualify as the pen-spin expert. Tuned on a debug seed only and frozen before the development cases, the gait regulates roll rate through the long fingers' flexion (the fingers act as a variable slope), which turns the open-loop oscillator's rocking into fast one-directional spin (21-26 rad/s mean). On 90 pooled promotion pairs it beat the frozen R-038 oscillator by +0.411 [0.307, 0.515] sustained success, promoting on every stage: supported. It does not qualify as an expert on any stage: 14/30, 16/30 and 7/30 sustained successes against the 27/30 rule, with failures from drops and tail dips, not speed (stage 3's 0.08 m tolerance costs most); it also tumbles the pen end over end (3.5-4.8 rad/s). The frozen bit-identity reproducibility control failed (GPU PhysX at batch 40 is not bit-deterministic across processes: 1-2/40 environments diverge after step 78) and was amended before any development case ran to zero outcome flips and at most 4/40 diverged environments, which passed (0 flips in 240 pairs). The user's fallback, a PPO expert, now applies to all three stages.
  Next step then proposed: Re-triage R-040 (PPO expert) for all three stages, recalibrated as argued in its round-1 triage: budget in env steps with many parallel envs, success and roll logged separately from return, a check that the reward (centring 3.0 vs spin 0.15 at stage 1) does not teach holding still, and the D-034 reducer for qualification. Separate idea: should the frozen gait's 47-53% sustained success seed the policy-search pipeline as a non-expert prior, or the tail-fraction near-misses inform a less brittle reducer?
- D-034 on R-038: **supported** — 1. Audit transient hits versus sustained axial-roll success
  R-038 built the first manifest-driven evaluation path for the Shadow Hand pen-spin task (frozen reset placements written into the simulator, a named 157-d observation layout, per-step traces) and audited the episode success rule the env leaves undefined. On 60 identical trajectories (3 stages x 10 probe cases x neutral hand and open-loop finger oscillator), any-hit success was 0.983 and sustained success (time-out without a fall, predicate true on all of the final 60 steps) 0.000: paired -0.983 [-1.017, -0.950], supported. Any-hit is degenerate: a neutral hand scores 1.0 because the dropped pen rolls on landing at steps 5-8, while it drops the pen in 29/30 episodes. The oscillator keeps the pen in hand to time-out in 29/30 episodes and passes 84/56/39% of tail steps (stage 1/2/3) at about 4 rad/s of roll, but never all 60, so whether 'all steps' is too strict for a periodic gait is open. Two infrastructure findings: the previous run's action leaked into the next run's observation, and a second rollout in one Isaac process is not reproducible (the first is bit-identical across processes), so protocols run one process per cell. The task rewards roll about the pen's own axis; end-over-end rotation stays below 0.8 rad/s.
  Next step then proposed: Re-triage R-039 (scripted feedback gait) with the sustained rule frozen as the qualification reducer, and report tail-hit fraction beside it so a periodic gait's near-misses are visible. Open question to propose: is 'all 60 steps' too strict for any periodic gait, i.e. should the rule be a tail-fraction or mean-rate criterion? Decide it before any expert is qualified, not after.
- D-033 on R-037: **supported** — Trace repair inside the real orchestrator: an end-to-end assembly-v2 run of run_full_evolution.py
  R-037 integrated plateau-triggered expert-trace repair into EvolutionOrchestrator and made run_full_evolution.py task-generic (manifests from pool seeds, expert demonstrations via EXPERT_POLICY_MAP), passing a 21-test mocked gate before any spend. Live on assembly-v2, two blocks of paired repair-on/repair-off production runs: all four runs hit the plateau trigger; with repair on, demonstrations qualified 10/10 and the trace repair became the final incumbent through the unchanged strict promotion rule (30/30 on promotion cases), scoring 30/30 on frozen fresh cases in both blocks, while every repair-off candidate scored 0/30 there (the repair-off incumbent stayed null). Paired fresh difference +1.000 in each block (n=2 blocks, degenerate interval), 1560 episodes and 8 calls, every record traceable. The real pipeline reproduces D-032 without the study-mode emulation.
  Next step then proposed: Make plateau-triggered trace repair a standard pipeline option; check the trigger stays silent and costless on non-plateau tasks; look for a second plateau task.
- D-032 on R-036: **supported** — Plateau-triggered trace repair versus exploration restart
  R-036 piloted plateau-triggered trace repair inside the pipeline's selection and promotion on assembly-v2 (one block). The exploration lane plateaued (0/24 probe successes) and no prefix candidate passed promotion, so the incumbent stayed at 0.00. Forking at that checkpoint, the trace-repair branch (whole-controller edit with two expert-only traces) produced a controller at 1.00 that passed the strict paired promotion rule; the fresh no-trace generation stayed at 0.00. Final incumbents: +1.000 [1.0, 1.0], supported as a pilot. The pipeline steps were reproduced in a study mode, not run_full_evolution.py.
  Next step then proposed: Build trace repair into the orchestrator with a plateau trigger and expert-demonstration source, generalise run_full_evolution.py beyond push-v2, and confirm with an independent block.
- D-031 on R-035: **refuted** — Find another recoverable terminal-phase plateau
  R-035 scanned shelf-place, basketball and soccer for a second recoverable plateau. All three experts qualified (1.00, 1.00, 0.90), and one single-phase structure plus 24-candidate search already succeeded on every task's probe (22, 20 and 23 of 24 candidates), so no task is a plateau: refuted. Across all scans, assembly-v2 is the only MetaWorld task where single-phase generation plus search plateaus, and expert-trace editing solves it (D-027, D-028, D-030).
  Next step then proposed: Integrate trace-guided repair into the two-lane pipeline (R-036).

## Available protocol scripts (reuse before writing new ones)
scripts/ablate_repair.py, scripts/analyse_failures.py, scripts/audit_expert_actions.py, scripts/benchmark_fitting.py, scripts/build_structure_bank.py, scripts/compare_child_vs_parent.py, scripts/compare_evolution_runs.py, scripts/compare_generation_modes.py, scripts/compare_objectives.py, scripts/diagnose_repair.py, scripts/evaluate_on_fresh_manifest.py, scripts/final_report.py, scripts/freeze_method.py, scripts/lock_baseline.py, scripts/plot_training_dynamics.py, scripts/qualify_fittable_bank.py, scripts/recover_provenance.py, scripts/research.py, scripts/run_full_evolution.py, scripts/run_shadowhand_spin_stages.py, scripts/stage_e.py, scripts/stage_f.py, scripts/summarise_experiments.py, scripts/visualize_expert_policy.py
## Available configs
config/assembly_two_lane_repair.yaml, config/debug_smoke_two_generation.yaml, config/environment.yaml, config/fittable_bank/manifest_provenance.json, config/fittable_bank/push-v2_development_confirmation_s2300.yaml, config/fittable_bank/qualification.yaml, config/fittable_bank/qualification_v2.yaml, config/fittable_bank/seed_bank.json, config/fittable_bank/structure_roles.yaml, config/fittable_bank/variants_frozen.json, config/pen_spin_plateau_repair.yaml, config/run_full_evolution.yaml, config/shadowhand_spin_stages.yaml, config/stage_e/manifest_provenance.json, config/stage_e/push-v2_development_confirmation_s2200.yaml, config/stage_e/selection.json, config/stage_e/stage_e.yaml, config/stage_f/manifest_provenance.json, config/stage_f/push-v2_development_heldout_s2500.yaml, config/stage_f/selection.json, config/stage_f/stage_f.yaml, config/stage_f_vision/manifest_provenance.json, config/stage_f_vision/probes/push-v2_probe_s2620.yaml, config/stage_f_vision/probes/push-v2_probe_s2621.yaml, config/stage_f_vision/probes/push-v2_probe_s2622.yaml, config/stage_f_vision/probes/push-v2_probe_s2623.yaml, config/stage_f_vision/probes/push-v2_probe_s2624.yaml, config/stage_f_vision/probes/push-v2_probe_s2625.yaml, config/stage_f_vision/probes/push-v2_probe_s2626.yaml, config/stage_f_vision/probes/push-v2_probe_s2627.yaml, config/stage_f_vision/probes/push-v2_probe_s2628.yaml, config/stage_f_vision/push-v2_development_heldout_s2629.yaml, config/stage_f_vision/stage_f_vision.yaml, config/two_lane_search.yaml, config/two_lane_search_v2.yaml
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
