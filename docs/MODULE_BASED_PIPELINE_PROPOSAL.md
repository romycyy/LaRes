# Modular Policy Generation — Implementation Handoff

Updated: 2026-11-19. Status: proposed; repository inspected, no implementation or experiment run.

## 1. Goal and design decision

Replace monolithic policy generation with **LLM-generated sparse functions and LLM-generated composition**. The pipeline supplies generic interfaces, validation, execution, and evidence; humans do not implement task-specific modules or combine candidates manually.

For the first pilot, keep one outer `GeneratedPolicy(SymbolicPolicy)` class. Put modules in named methods inside it. This avoids replacing the current loader and Isaac serialization path. Separate module files and cross-candidate libraries come later.

Rotation, damping, and support are optional prompt-level examples, not built-in module types or mandatory pipeline logic. A hinted pilot tests that prior; a no-hint condition is required to test autonomous decomposition. The LLM writes composition in both conditions.

Run this independently of contact-frame control, curriculum learning, and new gradient-fitting methods.

## 2. Verified starting point

Repository: [romycyy/LaRes](https://github.com/romycyy/LaRes), branch `staging`, reviewed head `cf3201affdae026882e036f0ed166d6591de3669`. Read `docs/PIPELINE_STATE.md` and `CLAUDE.md`; inspect subsequent changes before implementing.

- Generated policies already use sparse PyTorch expressions, named observations, and bounded `nn.Parameter`s; neural layers are forbidden.
- `forward(obs)` returns `(mean, self.fixed_std(mean))`. For ordinary pen-spin policies, both have shape `(batch, 20)`; deterministic execution clamps mean to `[-1, 1]`.
- `_extract_code_string` retains source starting at `GeneratedPolicy`; helpers placed before it are lost. Keep initial modules and metadata inside the class.
- Single-phase generation is the established baseline. Do not introduce a separate ideation call as part of the modularity comparison.
- Gradient fitting is off by default. Standard search uses 24 parameter draws per structure; retain that treatment for both arms.
- Stage 3 is axial roll in either direction: centre < 0.08 m and absolute spin > 4 rad/s on all final 60 steps, with no fall during the 360-step episode. Typical incumbent failures are slow spin, not loss of retention.
- Every evaluation uses the whole frozen manifest in a fresh Isaac process; batch composition affects outcomes. Preserve manifest ordering and batch size.

## 3. Code map

| Change | Existing integration point |
| --- | --- |
| Add opt-in generation format / prompt | `lares/core/policy_generation.py`, `lares/utils/policy_prompts/` |
| Add module interface checks | `lares/core/policy_validator.py`; optionally **new** `lares/core/module_contract.py` |
| Retain external policy behavior | `lares/core/symbolic_policy.py`, `lares/core/obs_schema.py` |
| Save provenance and feedback | `lares/search/generation.py`, `lares/search/feedback.py`, existing experiment records |
| Verify subprocess reconstruction | `lares/envs/isaac_pen_spin/dense_fit.py`, `pipeline_backend.py`, `pipeline_child.py` |
| Collect module traces / compare arms | Existing `runner.py`, `protocols.py`, `cases.py`, `block_study.py`; register the LLM study in `scripts/run_full_evolution.py` |

The current Isaac symbolic controller resets diagnostics and evaluates on CPU. Keep version 1 modules stateless: do not silently introduce recurrence or per-episode memory that lacks a reset contract.

## 4. Generated artifact and implementation order

1. **Reproduce baseline.** Verify the frozen incumbent with `scripts/isaac/adopt_fixed_stroke.py`; replay a recorded manifest before changing behavior.
2. **Define a generic module manifest.** Store class-level serializable metadata describing each method's role, inputs, outputs, shapes, units, coordinate frames, bounds, dependencies, and validity assumptions. Keep constants inside the class. Count all module and coordinator computation in one complexity budget.
3. **Update the single-phase prompt.** Request one complete class containing the manifest, module methods, declared parameters/ranges, and composed `forward`. Add a config switch so the monolithic arm uses its original prompt. The LLM chooses numerical laws and connections.
4. **Validate modules and composition.** Check signatures, declared graph edges, shapes, units/frames, finite outputs, bounds, declared parameters, and graph consistency. Retain existing observation, batch-independence, gate, and serialization checks. Use an explicitly restricted expression subset where needed; arbitrary Python metadata alone cannot prove dimensional correctness or dependency compliance.
5. **Add diagnostic execution.** Capture detached module outputs, input/output sensitivity, activation conditions, and final-action influence. Provide generic bounded overrides or compatible neutral replacements. Clear diagnostics between evaluations. Normal execution must produce identical actions with diagnostic collection on or off.
6. **Integrate feedback and artifacts.** Save the full source, manifest, parameter state, interface version, and their hashes. Add measured module failures to the evidence block. A local module edit still produces a new complete candidate with lineage and a complete-policy evaluation.
7. **Run the staged comparisons below.** Use existing orchestration and research protocol rather than creating a second evolution engine.

Proposed structure, not a new installed API:

```python
class GeneratedPolicy(SymbolicPolicy):
    MODULE_SPEC = {...}  # interfaces and graph generated by the LLM

    def module_a(self, named_inputs):
        ...  # sparse tensor expressions

    def module_b(self, named_inputs):
        ...

    def forward(self, obs):
        ...  # named observation reads, module calls, generated composition
        return mean, self.fixed_std(mean)

    def get_param_ranges(self):
        ...
```

Modules may emit intermediate physical quantities, not necessarily actions. Desired angular velocity and angular acceleration form different interfaces; they cannot simply be added. A momentum module needs a mass/inertia model or estimate. Velocity damping should be labelled damping.

Prevent decorative modules: expose coordinator observation-to-action bypasses, enforce declared dependencies, bound interface dimensions, and report removable/inactive modules. Conditional inactivity is valid. Do not require an ablated rotation module to reduce success in every state; compensation and interactions are possible.

## 5. Experiments and acceptance

| Condition | What changes | What it tests |
| --- | --- | --- |
| A: Monolithic | Existing generation/search | Reference |
| B: Generic modular | Named functions, generated composition, ordinary interface checks | Function decomposition |
| C: Physical modular | B plus physical interface contracts and checks | Physical structure beyond function splitting |
| D: Targeted repair | C plus module-intervention evidence for revisions | Value of targeted diagnostics |

First implement A versus B. Add C and D only after the basic format works. For C versus B, derived observation information must be equal. For D versus C, give C the corresponding budget for ordinary revisions/evaluations. Keep task, action representation, available state, total expression size, parameter count, fitting treatment, and generation budget comparable. Count all diagnostic episodes, failed candidates, LLM usage, and simulator steps; report wall time too.

Test the hinted versus LLM-chosen decomposition separately. Afterwards test reuse by freezing a module across changed spin targets or reset conditions, allowing remaining code to adapt. Compare with regeneration and whole-controller reuse at equal cost. Matching signatures alone does not establish compatible input distributions or transfer.

Primary metric: native sustained success. Also report retention, spin-speed distribution, final-window hit fraction, valid-generation rate, repair cost, and success versus interaction budget. Use repeated independent generation blocks and paired manifests. Preserve the strict paired promotion rule, multiplicity handling, seed ledger, and final-test lock. A 30-pair screen can miss modest effects; use the existing 90-pair qualification instrument where appropriate.

Completion checks:

- Monolithic behavior and source/state replay remain unchanged.
- A modular class survives extraction, subprocess validation, reconstruction, and execution in a fresh Isaac process.
- Invalid interfaces and undeclared dependencies are rejected; all parameters remain discoverable and serializable.
- Diagnostics are action-neutral; explicit interventions affect the intended path and are recorded as interventions, never ranked as ordinary candidates.
- The full-controller comparison reports all attempts and uncertainty. Module labels alone are not evidence of physical meaning.

Meaningful tests should cover these contracts and integration boundaries. Follow repository commands and research bookkeeping for actual runs; this handoff does not start experiments.

## 6. Related work and candidate insight

**Modular generation is established.** The candidate insight is whether LLM-generated physical interfaces make sparse continuous controllers easier to diagnose, repair, and reuse at matched cost. This review does not establish novelty. Differentiable programs, verification, and structure/parameter optimization also have prior work.

| Work | Relevant lesson | Primary reference / code |
| --- | --- | --- |
| Code as Policies | Hierarchical helper generation and reactive robot programs already exist | [Project](https://code-as-policies.github.io/), [code](https://github.com/google-research/google-research/tree/master/code_as_policies) |
| ModuLoop | Modular low-level code synthesis with execution probes and debugging | [Paper](https://arxiv.org/abs/2606.03047), [publisher](https://ieeexplore.ieee.org/document/11207505/); code unverified |
| MeMo | Useful division of labor requires more than splitting a network; expert-provided assembly partitions differ from our functional decomposition | [Paper](https://arxiv.org/abs/2407.01567), [code](https://github.com/MeganTj/MeMo) |
| RMPflow | Combine task-space policies through geometrically consistent transformations | [Paper](https://arxiv.org/abs/1811.07049); code not verified here |
| RMPfusion | Learnable physical-policy fusion is established | [Proceedings](https://proceedings.mlr.press/v100/mukadam20a.html); code not verified here |
| MAESTRO | Curated tool orchestration and feedback matter; tool count alone is insufficient | [Project](https://maestro-robot.github.io/), [code](https://github.com/penn-pal-lab/maestro) |
| ARCHITECT | Modular programs, trace-localized corrections, and persistent skills; includes human and VLM correction conditions | [Project](https://robo-architect.github.io/), [code](https://github.com/robo-architect/architect-franka) |
| CoLMO | LLM structure search plus gradient-free parameter tuning is established | [Paper](https://arxiv.org/abs/2510.00373), [project/code link](https://sites.google.com/berkeley.edu/colmo) |
| DiPRL | Differentiable program training can lose performance when hardened into discrete programs | [Paper](https://arxiv.org/abs/2605.18508); abstract reviewed, code unverified |
| NESYRO | Symbolic verification and interactive validation of generated robot programs already exist | [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/6d13ce54347c65845614d01ced1dbe23-Paper-Conference.pdf); code unverified |

Sources were checked on 2026-11-19; no external implementation was installed or reproduced. ModuLoop and DiPRL coverage is limited to abstracts/publisher metadata. Physical contracts do not inherit RMPflow stability guarantees, and differentiable module arithmetic does not make simulator contacts differentiable.
