# Contact-Frame Pilot — Implementation Handoff

Updated: 2026-11-19. Status: proposed; no new code or experiments implemented.

## 1. Goal and verified baseline

Test whether LLM-generated finger–pen relative-motion rules outperform direct joint-action generation at matched cost. Run independently of the modular-policy pilot; curriculum learning is deferred.

Repository: [romycyy/LaRes](https://github.com/romycyy/LaRes), branch `staging`. Reviewed head: `cf3201affdae026882e036f0ed166d6591de3669`. Read `docs/PIPELINE_STATE.md` and `CLAUDE.md` first; inspect newer commits before implementation. The state document describes some results from earlier commits, not a new measurement at this head.

| Existing contract | Value / consequence |
| --- | --- |
| Task | Axial rolling around the pen's local X axis, not end-over-end spinning |
| Stage 3 success | Complete 360 steps without falling; centre distance < 0.08 m and absolute axial rate > 4 rad/s on every final 60 steps |
| Direction | Either spin direction qualifies. Do not replace the metric with a signed angle target. |
| Timing / action | 60 Hz; 20 normalized joint-position targets; execute `clamp(mean, -1, 1)`, not tanh |
| Observation | 157-component full vector; use named schema accessors; velocity observations are scaled by 0.2 |
| Parameters / fitting | Bounded parameters already exist; gradient fitting is off by default. Standard rollout search uses 24 parameter draws per generated structure. |
| Documented incumbent | `config/pen_spin/d058_fixed_stroke_incumbent/`; 35/80 sustained successes on one fresh panel; typical failure is slow spin while retaining the pen |
| Reproducibility | One whole frozen manifest per fresh Isaac process. Outcomes depend on batch composition; do not split or regroup a manifest. |

These are repository-documented facts, with task/execution details checked against source. Reproduce the incumbent before comparing new controllers. The old “small signed rotation” task may be used only as a labelled adapter diagnostic; the main study retains native sustained success.

## 2. Change being tested

A local frame describes normal, circumferential, and axial motion near a finger–pen interaction. The LLM writes bounded relative-velocity rules in these coordinates. Trusted geometry and a numerical decoder translate them into the existing 20 joint targets.

Use a capsule model matching `shadow_hand_spin_env_cfg.py`, not an infinite cylinder. Handle end caps, ambiguous normals, and frame continuity. Proximity does not establish physical contact or support. The paper-inspired pilot is not a full reproduction of generalized contact-frame mechanics or force control.

Fix the available fingertip/frame set for the first study, preferably all five fingertips. Let the LLM specify the motion rules and roles. Any manually chosen active subset is an experimental prior that must be recorded.

## 3. Code map

Paths below exist unless marked **new**. Final names for new files are suggestions.

| Work | Integration point |
| --- | --- |
| Generation and validation | `lares/core/policy_generation.py`, `lares/core/policy_validator.py`, `lares/core/obs_schema.py`, `lares/utils/policy_prompts/` |
| Contact geometry / decoder | **New** `lares/envs/isaac_pen_spin/contact_control.py` |
| Actor serialization / child construction | `lares/envs/isaac_pen_spin/pipeline_backend.py`, `pipeline_child.py` |
| Policy rebuilding / runner | `dense_fit.build_policy`, `runner.ControllerContext`, `runner.PenSpinBatchRunner` |
| Actual dynamics and observation semantics | `shadow_hand_spin_env.py`, `shadow_hand_spin_env_cfg.py`, `layout.py`; inherited Isaac Lab action/observation code must also be inspected |
| Study and records | `block_study.py`, `protocols.py`, `cases.py`; register an LLM study through `scripts/run_full_evolution.py` |

The existing `SymbolicController` runs generated policies on CPU and expects 20 joint-action outputs. Jacobians are not supplied in its 157-component observation. Therefore this pilot needs an explicit actor/decoder path, not just a prompt edit.

## 4. Implement in order

1. **Freeze the baseline.** Verify incumbent hashes with `scripts/isaac/adopt_fixed_stroke.py`; reproduce a recorded manifest with its exact ordering and batch size. Record source, state, execution contract, and environment versions.
2. **Build trusted geometry.** Recover positions and velocities with correct scales, quaternion convention, frames, and environment origins. Get fingertip Jacobians through the installed Isaac API. Use the true actuated-joint indices, not assumed contiguous finger blocks. Account for coupled passive joints; check effective Jacobians numerically.
3. **Validate a decoder without an LLM.** Use small calibration commands; measure requested versus observed local motion, residuals, joint-limit activity, and retention. These are engineering checks, not learned-policy results. Do not increase search until basic tracking is reliable.
4. **Add a representation-aware actor contract.** Keep `GeneratedPolicy` as the generated class, but distinguish policy-output dimension `3*K` from environment-action dimension 20. Select the observation schema and validator dimensions explicitly. Serialize representation, geometry/scales, bounds, state, and source; dispatch to a new contact controller in `pipeline_child.build_controller`. Trusted controller code may attach to the environment; generated code receives only declared observations.
5. **Generate and search.** Each policy returns `(mean, self.fixed_std(mean))`, now shaped `(batch, 3*K)` for this representation. Define mean as normalized local-velocity requests; the decoder applies recorded m/s scales. Use named accessors and declared bounded parameters. Reuse single-phase generation and the existing 24-draw rollout search, with gradient fitting off.
6. **Evaluate and record.** Use existing study helpers, manifests, seed ledger, reports, and strict promotion. Record contact-specific diagnostics separately and return measured failures to the generator.

The new actor must define behavior for offline observation-only queries: either serialize all decoder state required to reproduce actions, or explicitly reject that query mode. Never reconstruct an arbitrary simulator state and present its Jacobian-dependent action as an exact replay.

### Minimal decoder sketch

This is a proposed kinematic approximation, not the paper's full formulation. For fingertip Jacobian `J_i`, local-to-world frame `R_i`, local velocity request `u_i`, and object-surface point `p_i`:

\[
v_{o,i}=v_o+\omega_o\times(p_i-p_o),\qquad
\dot q^*=\arg\min_{\dot q}\sum_i\|W_i(J_i\dot q-v_{o,i}-R_i u_i)\|^2+\lambda\|\dot q\|^2.
\]

Apply velocity and predicted joint-position bounds. Convert `q_target = q + dt * qdot` into normalized position targets using actual joint limits; inspect inherited target smoothing before claiming tracking. A nearest surface point can change identity, so validate moving-frame semantics rather than treating it as a fixed material point.

A kinematic solve cannot guarantee friction or grasp stability. Separate tracking error, slipping, and loss of contact where measurable. Supporting fingers may move with the pen; do not pin them in world coordinates while demanding rotation.

## 5. Comparison and completion checks

| Arm | Purpose |
| --- | --- |
| Direct joint generation | Existing representation baseline |
| Hand-authored contact controller | Decoder diagnostic only; not an LLM-discovery result |
| LLM contact rules, declared initial parameters | Zero-shot generation result |
| Same contact structures, 24 parameter draws | Representation comparison against direct generation with the same search treatment |

Compare zero-shot with zero-shot and searched with searched. Match generated structures, parameter budgets, underlying state information, and total simulator steps; report LLM tokens, failed candidates, diagnostic steps, solver cost, and wall time. If contact features improve results, add a direct-joint arm with the same derived features to separate input improvements from the decoder.

Keep native Stage 3 sustained success primary; report spin-speed distribution, final-window hit fraction, retention/drops, centring, decoder residuals, and saturation. Use independent generation blocks and paired manifests. A 30-pair panel is a screen, not sufficient evidence for small improvements; use the existing 90-pair instrument when qualification requires its precision. Confirm on fresh development cases. Leave final-test access locked under the existing freeze protocol.

Completion means: correct frame/Jacobian/action tests; reproducible source-and-state reconstruction in a fresh Isaac process; bounded decoder behavior under infeasible targets; baseline replay unchanged; a matched comparison with per-case evidence and uncertainty. Differentiable arithmetic does not establish gradients through contact. Do not add BC, RL fine-tuning, curriculum, or modular generation to this study.

Implement and validate locally first. Experimental execution follows the repository's research protocol; this document records no new run approval or result.

## 6. References

| Reference | Link / code status |
| --- | --- |
| Relative Contact Velocity-Controlled Hand-Object Mechanism for Dexterous Tool Manipulation | [Paper](https://arxiv.org/abs/2609.25619), [author-linked project](https://sunyuw.github.io/hom-sim/). Public implementation repository **not verified**. |
| RAX: Scaling Online Exploration for Robot Agents | [Paper](https://arxiv.org/html/2610.22496), [verified repository](https://github.com/robot-agent-exploration/rax), [DexJoCo native branch](https://github.com/robot-agent-exploration/rax/tree/dexjoco/native), [container branch](https://github.com/robot-agent-exploration/rax/tree/dexjoco/container). Branches are author-linked; not audited or executed here. |

RAX is a reference for exploration-to-program workflows; its frozen LLM revises programs without policy-parameter training. That overlaps our existing loop, so a separate RAX phase-program experiment is deferred. Neither reference establishes autonomous symbolic pen-spinning performance.
