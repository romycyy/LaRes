# Pipeline Refinement Spec — making `push-v2` work reliably

**Status:** draft for discussion. Nothing here is implemented.
**Scope:** the symbolic-policy evolution loop (`lares/core/training_pipeline.py`,
`lares/core/policy_generation.py`, `lares/utils/policy_prompts/*`).
**Baseline to beat:** best evolved policy on `push-v2` = 402.76 reward / 0.30 success as
reported by the run; ~233 / 0.13 on honest re-evaluation. Expert = 1068.34 / 1.00.

---

## 0. Where we actually are

Two bugs were fixed to get from 16.60/0.00 to the current numbers:

| Fix | Result |
|---|---|
| (baseline, HEAD) | 16.60 / 0.00 |
| BC loss: NLL-on-atanh → `MSE(tanh(mean), a)` | 37.42 / 0.00 |
| + `push-v2` added to `obs_description_dict` | 402.76 / 0.30 |

The second fix was the dominant one: the observation layout sent to the LLM was the empty
string, so it invented indices. That class of failure — a silently missing table entry —
is now warned about, but the deeper lesson is that **the pipeline had no signal that would
have caught it**. A policy reading the puck's quaternion as its position scores ~16, and a
policy with a genuinely bad control law also scores ~16. Nothing in the feedback loop
distinguishes them. That is the theme of this document.

Three problems, in the order the user raised them:

1. **Feedback to the LLM is nearly information-free** (§1)
2. **The idea → code gap in two-phase generation** (§2)
3. **MSE is the wrong BC objective for a policy that must emit a distribution** (§3)

Plus one problem found while writing this that is arguably larger than any of them:

4. **Candidate ranking is not a valid comparison** (§4)

---

## 1. Feedback

### 1.1 What the LLM currently receives

`llm_evolution()` builds `code_feedback` from `code_feedback.txt`:

```
Based on the current symbolic policy, after training for 2000 environment steps:
- Success rate: 0.3
- Mean episode reward (environment reward): 402.76

Evaluation details across multiple episodes:
{'mean_reward': 402.76, 'success_rate': 0.3}
```

`current_output` is `str(best_prev["eval"])` — literally the same two scalars again.
The header "Evaluation details across multiple episodes" promises a breakdown and delivers
a repeat. **Two floats is the entire performance signal for a structural redesign.**

Three further losses:

- **`elite_num` is a lie.** `EvolutionOrchestrator` passes `elite_num` results forward, but
  `llm_evolution()` does `max(previous_results, key=score)` and discards the rest. With
  `elite_num: 2` the second elite is loaded, pickled, and ignored. The LLM never sees two
  candidates side by side, so it can never learn *which structural difference mattered*.
- **`train_steps` is wrong.** It is `bc_steps + rl_iterations * episodes * 150`; with RL off
  this reports "2000 environment steps" for what were 2000 *gradient* steps on an offline
  buffer. The LLM is being told something false about the training regime.
- **Failed candidates vanish.** A candidate that fails codegen validation is repaired or
  dropped; the next generation never learns that e.g. `torch.clamp(x, float, tensor)` is
  not a valid call.

### 1.2 What is available and thrown away

MetaWorld's `info` dict, confirmed live on `push-v3`:

```
success, near_object, grasp_success, grasp_reward, in_place_reward,
obj_to_target, unscaled_reward
```

`evaluate_policy()` reads only `info["success"]` and returns two floats. Everything that
would localise a failure is discarded at the point of measurement.

### 1.3 Proposal — a structured evaluation report

Replace the two-scalar return with a diagnostic record, aggregated over episodes:

| Field | Source | Diagnoses |
|---|---|---|
| `success_rate`, `mean_reward` | as today | overall |
| `near_object_rate` | `info["near_object"]` ever true | did it even reach the puck? |
| `obj_to_target_final` (mean, p25, p75) | `info["obj_to_target"]` at t=T | how close did the puck get? |
| `obj_to_target_initial` | at t=0 | needed to tell "pushed it wrong" from "never moved it" |
| `obj_displacement` | ‖puck_T − puck_0‖ | **never touched vs. touched and missed** |
| `tcp_min_dist_to_obj` | from obs | approach phase working? |
| `action_saturation_rate` | \|tanh(mean)\| > 0.99 per dim | is the controller railed? |
| `phase_occupancy` | *see below* | are the LLM's own gates ever firing? |
| `time_to_first_contact` | first step `near_object` | too slow to matter? |
| per-episode table | 10 rows | variance is visible, not hidden in a mean |

`phase_occupancy` is the interesting one and needs a contract change: ask the generated
policy to optionally expose `self.last_gates: dict[str, Tensor]` in `forward()`. Then the
report can say *"`w_push` averaged 0.02 across the episode — your push phase never
activated"*, which is precisely the feedback a control engineer would want and which no
scalar reward can convey. Cost: one optional attribute, ignored if absent.

**Also feed back, per generation:**
- All `pop_size` candidates with `(code, score, diagnostics)` — not just the best. Ranked,
  so the LLM can do the credit assignment itself.
- BC loss trajectory (start, end, whether it plateaued). A structure that cannot even fit
  the expert data is a *capacity* problem, and BC loss says so at a fraction of the noise
  of a 10-episode success rate. **This is the cheapest high-value signal we are not using.**
- Codegen failures from the previous generation, with the error text.

**Open questions for you:**
- (a) How much of this goes in the prompt? All five candidates with full code is maybe 6-8k
  tokens/generation. Probably fine for gpt-5, but do we want the top-2 with full code and
  the rest as score+diagnostics only?
- (b) Do we want the diagnostics *computed* by us, or do we hand over per-episode traces and
  let the model do the analysis? The former is cheaper and more reliable; the latter might
  spot things we did not think to measure.
- (c) Should feedback be **comparative** ("candidate 3 differed from candidate 1 only in the
  descend gate and scored 4× lower") or just a ranked list? Comparative is stronger but
  requires us to diff the structures, which is hard in general.

---

## 2. The two-phase idea → code gap

### 2.1 What is actually happening

Ideation is working better than expected. Gen-2 ideas were ~200 words each, genuinely
distinct (three-phase gating / impedance control / potential field / four-stage with
anticipatory braking / vector field), and used the correct obs indices once the layout was
fixed. The prompt asks for "3–8 sentences: concrete enough that an engineer could implement
it without guessing the high-level structure", and it delivers roughly that.

The gap is downstream. Those ideas turn into **19–31 learnable parameters** each:

```
cand 0: 168 lines, 22 params   cand 3: 218 lines, 31 params
cand 1: 151 lines, 19 params   cand 4: 144 lines, 19 params
cand 2: 153 lines, 22 params
```

with entries like `alpha_r: (5.0, 200.0)`, `alpha_v: (5.0, 200.0)`, `alpha_b: (5.0, 200.0)`
— three sigmoid sharpness parameters spanning 1.5 orders of magnitude, fit by 2000 Adam
steps with `clip_params()` projection after every step. The idea may be sound and the
implementation faithful, and the thing still scores 200 because the parameters never
reached a good basin.

So the failure is not "the LLM cannot implement the idea". It is **the idea is not
specified tightly enough to be *initialisable***. Nothing in the current prompt asks:
*what should this policy do before any training at all?*

### 2.2 Proposal — make ideas carry an executable contract

Change the ideation output schema from a free-text string to a structured object:

```json
{"ideas": [{
  "name": "staged-push-with-standoff",
  "summary": "...",
  "phases": [{"name": "approach", "activates_when": "...", "action": "..."}],
  "params": [{"name": "r_back", "role": "standoff distance behind puck",
              "init": 0.05, "range": [0.02, 0.15], "units": "m"}],
  "obs_indices_used": {"tcp": "0:3", "obj": "4:7", "goal": "36:39"},
  "zero_shot_behaviour": "at init values, the arm should descend behind the puck and
                          push toward the goal at moderate speed even with no training",
  "expected_failure_mode": "may overshoot past the goal; the brake gate is untuned"
}]}
```

Three things this buys:

1. **`init` values become a design decision, argued for in phase 1**, rather than a
   throwaway in phase 2. We can then *test* them: instantiate and evaluate the policy
   before BC. If the untrained policy scores ~0 and the trained one scores ~0, the problem
   is the structure; if untrained is decent and trained is worse, BC is destroying it.
   (Strongly suspect we will find cases of the latter. Worth measuring first.)
2. **`params` is a checklist for `get_param_ranges()`.** Right now `clip_params()` silently
   skips any parameter missing from the ranges dict — an undeclared parameter is simply
   unconstrained, with no warning. A declared param list lets us assert coverage.
3. **`obs_indices_used` is checkable against the layout table** before we ever run the code.
   The `push-v2` disaster would have been caught by asserting that the declared `obj` slice
   matches the documented one.

**A parameter budget.** Ask for ≤10 learnable parameters, and require the idea to state
which are *structural* (thresholds, standoffs — physically interpretable, tight ranges) vs.
*gains* (which BC can move freely). 31 parameters on 22.5k transitions is not
over-parameterised in the statistical sense, but it is badly conditioned, and every extra
sigmoid sharpness is a plateau in the loss surface.

**Complexity ladder.** Consider forcing generation 0 to be simple (≤6 params, ≤2 phases)
and only allowing complexity to grow when a simpler structure has been shown to plateau.
Currently generation 0 already proposes four-stage controllers with anticipatory braking.

### 2.3 Implementation-phase fixes (cheap, independent)

- **Batched implementation writes 5 policies in one response.** Later slots are plausibly
  worse than earlier ones (attention/effort decay across a long generation). We have
  `policy_impl_mode: per_idea` already — worth an A/B before redesigning anything.
- **Add the known codegen traps to `new_code_output_tip.txt`.** `torch.clamp(x, float,
  tensor)` is invalid and cost 3 of 5 candidates in one run; the fix is
  `torch.minimum`/`torch.maximum`. This is a two-line prompt edit with a measured payoff.
- **A `SymbolicPolicy` self-check helper** the generated code could call, that validates
  shapes, range coverage, and finiteness — turning a subprocess failure into a precise
  message.

---

## 3. The BC objective

### 3.1 The user's objection is correct

`MSE(tanh(mean), a_expert) + 0.01 * std.mean()` is not a distributional fit. The std term
is a pure regulariser that drives std to its lower bound irrespective of the data; the mean
term is deterministic regression. We are training a Gaussian policy with a loss that
ignores its own likelihood. It was the right emergency fix (it beat the alternatives in an
A/B: 126.20 vs 81.33 for raw-MSE and 11.63 for the NLL-on-atanh that shipped in `1248ef7`)
but it is not the right objective.

### 3.2 Why the obvious fix — tanh-Gaussian NLL — failed

Measured on the collected expert buffer (22,500 transitions, `push-v2`):

| dim | mean | std | \|a\| > 0.99 |
|---|---|---|---|
| 0 (x) | +0.005 | 0.209 | 1.0% |
| 1 (y) | +0.188 | 0.353 | **10.4%** |
| 2 (z) | −0.446 | 0.262 | **7.1%** |
| 3 (grip) | +0.495 | 0.228 | 0.0% |

**17.6% of transitions saturate at least one dimension.** Under tanh squashing, emitting
`a = ±1` requires pre-tanh `mean → ±∞`. `atanh(±0.999) = ±3.8`; `atanh(±1) = ±∞`. Meanwhile
`get_param_ranges()` typically bounds gains at 10-20 and `clip_params()` projects onto that
box after **every** gradient step. So the target is not merely hard to reach — it is
**outside the representable set by construction**, and the loss spends its gradient budget
driving parameters into their upper clips.

This is the actual lesson: *any* loss that regresses in pre-tanh space fights the parameter
box. MSE-on-`tanh` degrades gracefully (its gradient vanishes as the output saturates, so it
pushes as far as the box allows and stops). NLL-on-`atanh` diverges.

### 3.3 The second problem: the expert is deterministic

MetaWorld's expert is a scripted controller. There is **no aleatoric spread to fit** —
p(a|s) is a point mass. Maximum likelihood against a point mass drives std → 0, which is
exactly what the `0.01 * std.mean()` hack does by other means, more honestly. So "fit the
action distribution" cannot mean "fit the expert's distribution", because the expert does
not have one. It has to mean one of:

- **(A) std = epistemic residual.** Fit `mean` to the expert, then fit `std` to the
  *policy's own residual* `|tanh(mean) − a|`. This is well-posed, is a real maximum-likelihood
  problem, and yields a genuinely useful quantity: the policy is loud where its structure
  cannot express the expert and quiet where it can. Heteroscedastic regression, standard.
- **(B) std = exploration schedule.** Do not learn it from BC at all; set it by an annealing
  schedule for RL and let RL tune it. Honest, trivial, and admits that BC has nothing to say
  about it.
- **(C) std = learned, with an entropy floor.** Keep the likelihood objective but add
  `−β·H[π]` so it cannot collapse. Standard in SAC-style pipelines.

**Recommendation: (A), with (B) as the fallback if RL is ever re-enabled.** (A) makes the
policy output a real distribution — the user's requirement — without inventing variance that
is not in the data, and it hands §1 a free diagnostic: *"your policy's residual std is 0.4
on the z axis during the descend phase; the structure cannot express what the expert does
there."* That is exactly the structural feedback the LLM needs.

### 3.4 Concrete candidates to evaluate

Let `u = tanh(mean)` (executed action), `a` = expert action.

| Option | Loss | Notes |
|---|---|---|
| **B0** current | `MSE(u,a) + 0.01·mean(std)` | baseline, 126.20 in A/B |
| **B1** tanh-Gaussian NLL, clipped targets | `−log N(atanh(clip(a,±0.999)); mean, std) − Σlog(1−u²)` | needs §3.5 |
| **B2** two-stage heteroscedastic | stage 1: `MSE(u,a)`; stage 2: freeze mean-params, fit std to residuals | option (A) |
| **B3** joint heteroscedastic Gaussian in **action space** | `½·((u−a)/std)² + log std` | option (A), single stage, no atanh anywhere |
| **B4** B3 + entropy floor | `B3 + β·relu(log σ_min − log σ)` | prevents σ→0 blowing up the first term |

**B3 is the one I would try first.** It is a genuine likelihood — a Gaussian over the
*executed* action rather than the pre-tanh variable — so it never touches `atanh`, never
leaves the parameter box, and reduces to B0's mean term when std is held fixed. It is not
the exact tanh-Gaussian density (it drops the Jacobian and ignores that the support is
bounded), which is a defensible approximation to state explicitly, or to fix by using a
**truncated** Gaussian or a Beta on `(a+1)/2` if we want it to be exactly right.

**Open question:** do we care about the density being *correct* on `[−1,1]`, or only about
having a well-calibrated, trainable scale? If RL is genuinely off for good, B3 is enough. If
GRPO comes back, the ratio `π_new/π_old` needs a correct density and we should do the full
tanh-Gaussian with the Jacobian term, and then §3.5 becomes mandatory.

### 3.5 Prerequisite: fix the parameterisation, not just the loss

Whatever loss we pick, the saturation problem stands. Options:

- **Clip expert targets to ±0.999** — cheap, hides the problem, loses the 17.6%.
- **Let the policy emit a scale it can actually reach**: require an explicit output gain
  `mean = g · (symbolic expression)` with `g ∈ [1, 20]`, so saturation is reachable within
  the box. Small prompt change, likely large effect.
- **Drop tanh; clip instead.** `a = clip(mean, −1, 1)` matches the expert's own saturating
  behaviour exactly and makes ±1 trivially representable. Costs differentiability at the
  boundary (dead gradient) and breaks the reparameterisation trick for RL.
- **Weight the loss by** `1 − a²` so saturated transitions contribute less — principled
  under the tanh-Gaussian Jacobian, and it stops 17.6% of the data from dominating.

These interact with the loss choice; worth deciding together.

---

## 4. Ranking validity (found while writing this — possibly the biggest item)

`env_wrapper.reset()` (`lares/utils/utils.py:2540`) resamples an MT1 task on **every**
reset, from a `_mt1_rng` created once and never reset:

```python
idx = int(self._mt1_rng.integers(0, len(tasks)))
inner.set_task(tasks[idx])
```

So candidate 1 is evaluated on tasks 0-9 of the stream, candidate 2 on tasks 10-19, and so
on. **No two candidates are scored on the same task instances.** With `eval_episodes: 10`
and goal/puck placements varying substantially in difficulty, the ranking is largely noise.

This is not speculation. Re-evaluating the *saved* best policy gave 175.79 / 293.04 /
197.24 / 179.54 over four 20-episode batches, and 232.96 over 30 — against the 402.76 the
run reported from its 10 episodes. **The reported winner of each generation is
substantially a lottery**, which means the elite fed back to the LLM is often not the best
structure, and the "402.76 after 3 generations" improvement curve is partly an artefact of
taking a max over noisy draws.

**Proposal (cheap, do this first):**
- Fix a held-out eval task set: seed `_mt1_rng` identically before every candidate's
  evaluation, so all candidates see **the same** N tasks (common random numbers). This alone
  removes most of the between-candidate variance at zero extra compute.
- Raise `eval_episodes` to ~25-30, or make it adaptive (evaluate the top-3 further).
- Report a confidence interval alongside the mean, and refuse to call a generation an
  improvement when the intervals overlap.
- Keep a separate *test* task set, never used for ranking, for the final reported number.

**Open question:** with common random numbers, do we still want MT1 resampling at all
during BC/eval, or should `push-v2` be a fixed single task until the pipeline demonstrably
works? Simpler target first is arguably the whole point of "make it work on a simple task".

---

## 5. Suggested ordering

Ordered by (measured evidence of impact) ÷ (effort):

1. **§4 common random numbers + more eval episodes.** Small diff. Without it we cannot
   measure whether anything else in this document helps.
2. **§1.3 structured diagnostics** from the `info` dict + all-candidate feedback + BC-loss
   reporting. Medium diff, no LLM changes.
3. **§2.3 codegen traps in the prompt** and a `per_idea` A/B. Tiny diff.
4. **§3 loss redesign (B3 or B2)** together with §3.5 output-gain parameterisation.
5. **§2.2 structured idea schema** with `init` values and the pre-BC zero-shot evaluation.
6. **§2.2 parameter budget / complexity ladder.**

Items 1-3 are strictly instrumentation and prompt text; they change no algorithm and can be
validated on one run each. Items 4-6 are the substantive redesign and should be judged
against the fixed eval set from item 1.

---

## 6. Things deliberately not proposed

- **Re-enabling RL.** Out of scope per the user; noted only because §3.4's choice between B3
  and a full tanh-Gaussian depends on whether GRPO ever comes back.
- **Changing the symbolic-policy restriction.** No `nn.Linear` is the point of the project.
- **Multi-seed BC restarts per candidate.** Would probably help the §2.1 conditioning
  problem a lot (`pop_size × n_restarts` BC runs is cheap — 2000 steps is seconds), but it
  treats the symptom rather than the badly-conditioned parameterisation. Worth discussing
  as an alternative to §2.2's parameter budget.

---

## 7. Questions I want your answer on before writing code

1. §1 — how much per-candidate detail goes in the prompt, and computed-diagnostics vs.
   raw-traces-for-the-model-to-analyse?
2. §3 — is a correct density on `[−1,1]` a requirement (RL may return), or is a
   well-calibrated trainable scale enough (B3)?
3. §3.5 — output gain, target clipping, or drop tanh for clip? These are mutually exclusive.
4. §4 — fixed single task for `push-v2` until the pipeline works, or keep MT1 with common
   random numbers?
5. §2.2 — is a parameter budget acceptable, or does capping expressiveness defeat the
   purpose of LLM structure search? (Alternative: keep it uncapped, add multi-seed restarts.)
