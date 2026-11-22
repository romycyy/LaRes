# Research handoff for ChatGPT (2026-11-22)

Prepared by Claude Code on 2026-11-22. A snapshot for an outside reader. The live, generated view is
`research/STATE.md` (regenerated for this handoff with `python scripts/research.py state`); the pipeline
description is `docs/PIPELINE_STATE.md`; the narrative decision log is `RESEARCH_HANDOFF.md`.

## Where the code is

- Repository: `github.com/romycyy/LaRes`, branch **`staging`**.
- Last research commit before this handoff: `67a74ba` (R-102 / D-072). This file, the regenerated
  `research/STATE.md` and the small result summaries listed below were committed on top of it; the commit
  that adds this file is the handoff commit.
- `main` is at `9f398d7` and is far behind `staging`; all research since 2026-09 is on `staging`.

## What is running

**Nothing.** The GPU box (EC2 `LLM-symbolic-backup`, g4dn.2xlarge) is **stopped**. No idea is `running`,
`approved`, `proposed`, `analysed` or `escalated` in the ledger (72 decided, 20 deferred, 10 rejected). No
experiment was started for this handoff.

## Current direction (set by the user 2026-11-22)

> No LLM calls in any protocol. Two finite follow-ups to D-069 and D-070 on Stage 3, on the fixed-stroke
> incumbent (`config/pen_spin/d058_fixed_stroke_incumbent`). (A) A transfer audit ... (B) One confirmatory
> replication of D-070's frozen residual, unchanged, on about 670 fresh pairs ... No retuning, no new
> controller search, no further replication after (B).

Both (A) = R-101 and (B) = R-102 are done, so **this direction is exhausted**. The next step needs a new
direction from the user, plus one pending call: whether to install D-070's residual as the incumbent asset.

## Task in one paragraph

Shadow Hand pen spinning in Isaac Lab (`Isaac-ShadowHandSpin-Stage3`, 157-d observation, 20-d joint-target
action, 60 Hz, 360-step episodes). Success is ours, per episode: `cases.sustained` = time-out without a fall
and the Stage-3 predicate (pen centred, |axial roll rate| > 4 rad/s) on all of the final 60 steps. Every
comparison is **paired on frozen manifests**, one manifest per fresh Isaac process (a second rollout in one
process is not reproducible, D-034). The incumbent controller is the D-058 fixed-stroke clamp (fresh sustained
success about 0.34-0.44, few drops; slow spin is the dominant failure).

## Latest completed experiments (current and previous direction, 2026-11-21 to 2026-11-22)

All four used 0 LLM calls. Intervals are 95%. "Paired" = per-case difference on shared manifests.

### R-102 / D-072, **supported**: confirmation of D-070's frozen residual

- Protocol: `python scripts/run_shadowhand_spin_stages.py --study incumbent_residual_confirmation`; 8 fresh
  manifests (seeds 4751-4758) x 84 cases x 2 arms = 1,344 episodes, 16 fresh processes, about 12 min. Residual
  verified by hash `2a509ef5ffc148cab2a513bc2c7339cca9a620b50dec2f7a988fd2b041c58cd1`, unchanged
  (draw7: gain 0.0047 s, deadband 0.23 rad/s, tau 0.185 s, cap 0.103 rad; config block in
  `config/shadowhand_spin_stages.yaml` around line 1358).
- **Paired sustained-success difference +0.055 [+0.024, +0.086]**, 672 pairs; 267/672 (0.397) vs 230/672
  (0.342); wins/losses/ties 77/40/555.
- Manifest-clustered t interval (7 df): **[+0.026, +0.084]**; positive in 7 of 8 manifests (s4752: -0.024).
- Drops 34 vs 34; one-sided 95% upper bound on the drop difference **+0.006** (gate: <= +0.05).
- Secondary: tail-hit fraction +0.071 [+0.040, +0.102]; slow-spin failures 347 vs 377; unsigned roll 25.4 vs
  20.8 rad.
- Both adoption gates pass. Point estimate matches D-070's +0.056, so that was not a selection artefact.
  Caveats: interval includes effects below +0.05; the mechanism is not established; the incumbent asset
  `config/pen_spin/d058_fixed_stroke_incumbent` has **not** been changed (user's call).
- Artifacts: `logs/pen_spin/r102_residual_confirmation/summary.json`, `.../research_result.json`.

### R-101 / D-071, **refuted**: forefinger + ring transfer audit

- `--study transfer_audit`; 3 cells x 40 cases (seed 4750) = 120 episodes, about 2 min of Isaac.
- Authority adequate: fingertip displacement 0.69 (ff) and 0.82 (rf) of the full 20 mm; every eligible case
  positive; no speed bound, cap or clamp active.
- **Roll progress pro minus anti, steps 150-209: -0.054 rad [-0.147, +0.038]**, 32 eligible cases, against a
  predicted +0.10.
- Post hoc (not pre-declared): forefinger, ring and little fingertips sit about 58 mm from the pen; only the
  middle fingertip and thumb are at the surface (within 15 mm on 0.92 and 0.89 of steps). D-069's "tracking"
  fingers were moving in free space.
- Artifacts: `logs/pen_spin/r101_transfer_audit/summary.json`, `.../posthoc_contact_distance.json`,
  `.../research_result.json`.

### R-099 / D-070, **inconclusive**: anonymous joint-motion residual, search + 90-pair fresh check

- `--study incumbent_residual` then `--study incumbent_residual_fresh`; 2,000 search episodes (24 draws x 80
  probe cases, seed 4742) + 180 fresh (seed 4744), about 20 min.
- Probe: winner 39/80 vs incumbent 31/80 (best of 24, so inflated).
- **Fresh: +0.056 [-0.041, +0.152]**, 41/90 vs 36/90, W/L/T 12/7/71; drops 1 vs 1. A 90-pair instrument
  resolves about +/-0.10 (paired SE 0.048), so it could not decide a +0.05 effect; R-102 then did.
- Artifacts: `logs/pen_spin/r099_incumbent_residual/search_summary.json`, `.../fresh_summary.json`,
  `.../research_result.json`.

### R-097 / D-069, **refuted**: reproduce the incumbent; contact-frame decoder gate during rolling

- `--study rolling_baseline`, `rolling_decoder_calibration`, `rolling_decoder_gate`; 80 + 800 + 2,480 = 3,360
  episodes (seeds 4741 / 4740), about 35 min.
- S1 reproduces: **28/80 = 0.350, Wilson [0.255, 0.459]**; 42 slow spin, 8 early drops, 2 off-centre.
- S2 gate **fails**: pooled paired sign agreement **0.724, case-clustered [0.710, 0.739]** vs required 0.80;
  feasible/requested 0.445 (min 0.70); bound active on 0.44 of pulse steps (max 0.20); tracking error 0.84
  (max 0.70). Instrument checks pass (Jacobian audit 0.0025, in-play prefixes bit-identical).
- Post hoc: forefinger/ring track (0.994 / 1.000), middle 0.411, thumb 0.315, little 0.899 with bounds active
  0.71. Incumbent's own action is saturated at +/-1 on MFJ2 (79% of steps) and LFJ1 (64%).
- Artifacts: `logs/pen_spin/r097_rolling_decoder_gate/{baseline,calibration,gate}_summary.json`,
  `.../research_result.json`.

### Earlier in the same week (one line each; details in `docs/PIPELINE_STATE.md` and `RESEARCH_HANDOFF.md`)

| Idea | Decision | Result |
|---|---|---|
| R-094 LLM contact-frame rules vs joint targets | D-068 refuted | 0.000 [0, 0] fresh final-incumbent success over 8 blocks; searched winner -0.0125 [-0.054, +0.029]; decoder outside its qualified regime (bound active 97%) |
| R-096 settled-window decoder gate, 80 cases | D-067 supported | paired sign agreement 1.000 over 901 pairs, gain 0.615 |
| R-095 settled-window decoder gate, 40 cases | D-066 inconclusive | tracking checks pass; coverage short (16 and 19 of 20 pairs) |
| R-092 modular vs monolithic cold generation | D-065 refuted | +0.0083 [-0.011, +0.028] over 8 blocks (predicted +0.05); 0 of 64 candidates had a screen success |
| R-093 contact-frame decoder gate (settling window) | D-064 gate failed | sign agreement 0.6825 < 0.80; retention prediction held (0.775 vs 0.000) |
| R-100 360-pair replication | rejected at triage | exceeded the direction's 90-pair gate, about 55% power; replaced by R-102 |

## Failures and open problems

Scientific:
- **Cold LLM generation on Stage 3 is at its floor** for both monolithic and modular representations
  (D-065, D-068; one promotion in 16 arms). Trace repair (D-038, D-047), dense imitation fitting (D-039) and
  expert takeover (D-045) did not lift it.
- **Contact-frame control does not transfer to rolling on this incumbent** (D-068, D-069, D-071). The two
  fingers that track are not touching the pen; the two that hold it (middle, thumb) do not track and are
  partly saturated. Further contact work would need to act on the middle finger and thumb: a new direction.
- **Slow spin remains the dominant failure** even with the qualified residual (347 of 672 episodes).
- The only positive changes to the incumbent so far: the zero-curl clamp + fixed 1.408 stroke scaling
  (D-057 to D-059) and now the joint-motion residual (D-072, +0.055).

Operational (all resolved before results were taken; none changed a result):
- R-101's first launch ran the driver under the wrong Python (`gpu run`'s venv instead of
  `source ~/IsaacLab/source_env.sh`) and failed before any episode.
- R-099's first fresh invocation crashed: the driver's record file shared its name with a cell
  (`selected.json`), so that cell was skipped; renamed `frozen_selected.json` and resumed, nothing retuned.
- R-097: the prefix-identity check first included post-termination auto-reset steps; restricted to in-play
  steps before the gate ran.
- Two `gpu sync` guard gaps are open (server-side untracked additions are not detected; a synced untracked
  file deleted locally stays on the server). See `docs/PIPELINE_STATE.md`, "GPT review on 2026-11-19".
- The MetaWorld final-test split is still locked (Phase 7 not started); no final-test episode has run.

## Artifact paths

- Ledger: `research/ideas/R-0xx.json` (hypothesis, protocol, triage rounds, result metrics, verdicts),
  `research/decisions.jsonl`, `research/triage.jsonl`, `research/proposals.jsonl`,
  `research/llm_calls.jsonl`, `research/briefings/`, `research/direction.json`, `research/STATE.md`.
- Result summaries (small JSON; `logs/` is gitignored, these were force-added for this handoff):
  `logs/pen_spin/r097_rolling_decoder_gate/`, `logs/pen_spin/r099_incumbent_residual/`,
  `logs/pen_spin/r101_transfer_audit/`, `logs/pen_spin/r102_residual_confirmation/`. Only the summaries were
  copied back from the box; per-case cell files, where kept, are on the box under `~/LaRes/logs/pen_spin/`.
- Code: `lares/envs/isaac_pen_spin/incumbent_residual.py` (R-099, R-102), `transfer_audit.py` (R-101),
  `rolling_decoder_gate.py` (R-097), `contact_control.py`, `contact_gate.py`; driver
  `scripts/run_shadowhand_spin_stages.py --study <name>`; protocol constants in
  `config/shadowhand_spin_stages.yaml` (`research_protocols`).
- Incumbent asset: `config/pen_spin/d058_fixed_stroke_incumbent/` (verified by
  `scripts/isaac/adopt_fixed_stroke.py`). Manifests: `config/manifests/pen_spin_stage3_*.yaml`.
- Session summaries: `docs/SESSION_2026-10-27_PEN_SPIN_SUMMARY.md`, `docs/SESSION_2026-11-04_PEN_SPIN_SUMMARY.md`.

## Decisions waiting on the user

1. Install the D-070 / D-072 residual (hash `2a509ef5...`) as the incumbent asset, or keep D-058.
2. A new research direction: the current one declared R-102 the last replication.
