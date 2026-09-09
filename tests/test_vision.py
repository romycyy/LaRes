#!/usr/bin/env python
"""Tests for VLM-assisted failure analysis (``spec.md`` FR-11, AC-2).

The vision model is the one component that will confidently describe things it
did not see. These tests are the contract that makes that detectable: a claim
must cite a frame that was shown and a timestep inside it, a recommendation must
name something the pipeline measures, and a disagreement with the simulator is
recorded rather than resolved.

Nothing here loads a model.

Run from the project root::

    python -m unittest tests.test_vision
"""

import json
import os
import sys
import tempfile
import unittest

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402

from lares.core.obs_schema import get_obs_schema  # noqa: E402
from lares.eval.manifest import (  # noqa: E402
    SPLIT_DEVELOPMENT,
    SPLIT_FINAL_TEST,
    synthetic_manifest,
)
from lares.eval.runner import ConstantActor, EpisodeRecord, run_episode  # noqa: E402
from lares.vision import (  # noqa: E402
    AGREE,
    analyse_failures,
    evidence_block,
    failed_cases,
    CONFLICT,
    MEDIA_SELECTION_RULE,
    MOMENTS,
    UNCHECKABLE,
    AnalysisRequest,
    EpisodeMedia,
    Evidence,
    FinalTestMediaRefused,
    MediaItem,
    QwenAnalyst,
    ScriptedAnalyst,
    VisualFailureAnalysis,
    merge_all,
    merge_analysis,
    parse_analysis,
    record_episode_media,
    repair_prompt_block,
    select_media,
)

OBS_DIM = 39
ACT_DIM = 4
SCHEMA = get_obs_schema("push-v2")


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------


def a_media_item(id="frame_t0000", start=0, end=0, reason="start"):
    return MediaItem(frame_or_clip_id=id, start_timestep=start, end_timestep=end,
                     reason=reason)


def an_analysis(**overrides):
    base = dict(
        analysis_id="a1",
        candidate_id="c0",
        case_id="case_0",
        model_id="scripted",
        prompt_version="v1",
        media=[a_media_item(), a_media_item("frame_t0040", 40, 40, "closest_approach")],
        observed_stage="approach",
        failure_summary="the hand stops short of the puck",
        evidence=[Evidence("the gripper is still above the puck", "frame_t0040", 40)],
        candidate_hypotheses=["the descent never triggers"],
        uncertainties=["cannot tell whether the fingers closed"],
        recommended_numeric_checks=["min_tcp_object_distance"],
    )
    base.update(overrides)
    return VisualFailureAnalysis(**base)


def an_episode(case_id="case_0", label="never_reached_object", **overrides):
    base = dict(
        case_id=case_id, task_id="t", success=0.0, episode_return=100.0, length=150,
        terminal_obj_to_target=0.2, initial_obj_to_target=0.3, near_object_rate=0.0,
        first_near_object_step=None, grasp_success_rate=0.0, action_saturation_rate=0.0,
        timed_out=True, failure_label=label, min_tcp_object_distance=0.09,
        object_displacement=0.01, signed_goal_progress=0.01,
    )
    base.update(overrides)
    return EpisodeRecord(**base)


def a_reply(**overrides):
    payload = {
        "observed_stage": "approach",
        "failure_summary": "the hand stops short",
        "evidence": [{"statement": "gripper above puck", "frame_or_clip_id": "frame_t0000",
                      "timestep": 0}],
        "candidate_hypotheses": ["descent never triggers"],
        "uncertainties": ["fingers not visible"],
        "recommended_numeric_checks": ["min_tcp_object_distance"],
    }
    payload.update(overrides)
    return json.dumps(payload)


class _Space:
    def __init__(self, dim):
        self.shape = (dim,)
        self.high = np.ones(dim, dtype=np.float32)
        self.low = -np.ones(dim, dtype=np.float32)


class MockEnv:
    """Moves the hand toward the puck, then stops, and renders a frame index."""

    def __init__(self, episode_length=6, renders=True):
        self.observation_space = _Space(OBS_DIM)
        self.action_space = _Space(ACT_DIM)
        self._episode_length = episode_length
        self._step = 0
        self.renders = renders

    def _obs(self):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        # tcp closes on the object until the midpoint, then retreats.
        t = self._step
        half = self._episode_length / 2.0
        approach = abs(half - t) / half
        obs[0:3] = [approach * 0.1, 0.0, 0.0]
        # obj jumps once, late, so max_object_move is a distinct timestep.
        obs[4:7] = [0.0, 0.0, 0.0] if t < self._episode_length - 1 else [0.05, 0.0, 0.0]
        obs[36:39] = [0.3, 0.0, 0.0]
        return obs

    def reset(self, case=None):
        if case is None:
            raise ValueError("case required")
        self._step = 0
        return self._obs(), {}

    def step(self, action):
        self._step += 1
        done = self._step >= self._episode_length
        return self._obs(), 1.0, done, {"success": 0.0, "obj_to_target": 0.3}

    def render(self):
        if not self.renders:
            raise RuntimeError("no renderer")
        return np.full((4, 4, 3), self._step, dtype=np.uint8)


def a_recorded_media(length=6, gates=None, obj_jump_at=5):
    media = EpisodeMedia(case_id="case_0")
    for t in range(length):
        media.frames.append(np.full((2, 2, 3), t, dtype=np.uint8))
        half = length / 2.0
        media.tcp.append(np.array([abs(half - t) / half * 0.1, 0.0, 0.0]))
        media.obj.append(np.array([0.05 if t >= obj_jump_at else 0.0, 0.0, 0.0]))
        media.goal.append(np.array([0.3, 0.0, 0.0]))
        media.gates.append(dict(gates[t]) if gates else {})
    return media


# ---------------------------------------------------------------------------
#  The contract
# ---------------------------------------------------------------------------


class TestAnalysisContract(unittest.TestCase):
    def test_a_complete_analysis_validates(self):
        self.assertTrue(an_analysis().validate())

    def test_evidence_citing_a_frame_never_shown_is_refused(self):
        analysis = an_analysis(
            evidence=[Evidence("the puck moved", "frame_t9999", 9999)]
        )
        with self.assertRaises(ValueError) as caught:
            analysis.validate()
        self.assertIn("never shown to the model", str(caught.exception))

    def test_evidence_citing_a_timestep_outside_its_frame_is_refused(self):
        analysis = an_analysis(
            evidence=[Evidence("the puck moved", "frame_t0000", 77)]
        )
        with self.assertRaises(ValueError) as caught:
            analysis.validate()
        self.assertIn("spans", str(caught.exception))

    def test_a_timestep_inside_a_clip_span_is_accepted(self):
        analysis = an_analysis(
            media=[a_media_item("clip_a", 10, 40, "closest_approach")],
            evidence=[Evidence("the hand slows", "clip_a", 25)],
        )
        self.assertTrue(analysis.validate())

    def test_an_analysis_with_no_evidence_is_an_assertion_and_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            an_analysis(evidence=[]).validate()
        self.assertIn("assertion", str(caught.exception))

    def test_an_analysis_must_name_the_media_it_saw(self):
        with self.assertRaises(ValueError):
            an_analysis(media=[]).validate()

    def test_duplicate_media_ids_are_refused(self):
        with self.assertRaises(ValueError):
            an_analysis(media=[a_media_item(), a_media_item()]).validate()

    def test_a_clip_that_ends_before_it_starts_is_refused(self):
        with self.assertRaises(ValueError):
            an_analysis(media=[a_media_item("clip", 40, 10)]).validate()

    def test_every_identity_field_is_required(self):
        for field in ("analysis_id", "candidate_id", "case_id", "model_id", "prompt_version"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    an_analysis(**{field: ""}).validate()

    def test_it_must_recommend_a_numeric_check(self):
        # Visual evidence cannot promote, reject or repair on its own, so it has
        # to say what would confirm or refute it.
        with self.assertRaises(ValueError) as caught:
            an_analysis(recommended_numeric_checks=[]).validate()
        self.assertIn("cannot promote, reject or repair", str(caught.exception))

    def test_a_check_the_pipeline_cannot_run_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            an_analysis(recommended_numeric_checks=["vibes"]).validate()
        self.assertIn("does not measure", str(caught.exception))

    def test_it_round_trips_through_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = an_analysis().save(tmp)
            with open(path, encoding="utf-8") as f:
                loaded = VisualFailureAnalysis.from_dict(json.load(f))
        self.assertEqual(loaded.to_dict(), an_analysis().to_dict())
        self.assertTrue(loaded.validate())


class TestParsing(unittest.TestCase):
    def test_the_model_does_not_supply_its_own_identity(self):
        # Candidate, case, model and prompt version are facts about the request.
        analysis = parse_analysis(
            a_reply(candidate_id="something_else"), "c0", "case_0", "qwen-x",
            [a_media_item()],
        )
        self.assertEqual(analysis.candidate_id, "c0")
        self.assertEqual(analysis.case_id, "case_0")
        self.assertEqual(analysis.model_id, "qwen-x")

    def test_the_analysis_id_follows_the_reply(self):
        first = parse_analysis(a_reply(), "c0", "case_0", "m", [a_media_item()])
        same = parse_analysis(a_reply(), "c0", "case_0", "m", [a_media_item()])
        other = parse_analysis(
            a_reply(failure_summary="different"), "c0", "case_0", "m", [a_media_item()]
        )
        self.assertEqual(first.analysis_id, same.analysis_id)
        self.assertNotEqual(first.analysis_id, other.analysis_id)

    def test_a_parsed_reply_still_has_to_validate(self):
        analysis = parse_analysis(
            a_reply(evidence=[{"statement": "x", "frame_or_clip_id": "nope", "timestep": 0}]),
            "c0", "case_0", "m", [a_media_item()],
        )
        with self.assertRaises(ValueError):
            analysis.validate()

    def test_runtime_metadata_is_carried_for_audit(self):
        analysis = parse_analysis(
            a_reply(), "c0", "case_0", "m", [a_media_item()],
            runtime={"quantization": "fp16", "latency_seconds": 1.5},
        )
        self.assertEqual(analysis.runtime["quantization"], "fp16")


# ---------------------------------------------------------------------------
#  Media selection
# ---------------------------------------------------------------------------


class TestMediaSelection(unittest.TestCase):
    def test_the_rule_is_declared_and_versioned(self):
        self.assertIn("v1", MEDIA_SELECTION_RULE)
        for moment in MOMENTS:
            self.assertIn(moment, MEDIA_SELECTION_RULE)

    def test_it_selects_the_declared_moments(self):
        items, unavailable = select_media(a_recorded_media(length=6))
        reasons = " ".join(i.reason for i in items)
        self.assertIn("start", reasons)
        self.assertIn("closest_approach", reasons)
        self.assertIn("termination", reasons)
        self.assertIn("max_object_move", reasons)

    def test_frames_come_back_in_timestep_order(self):
        items, _ = select_media(a_recorded_media(length=8))
        self.assertEqual([i.start_timestep for i in items],
                         sorted(i.start_timestep for i in items))

    def test_the_same_episode_always_selects_the_same_frames(self):
        first, _ = select_media(a_recorded_media(length=8))
        second, _ = select_media(a_recorded_media(length=8))
        self.assertEqual([i.frame_or_clip_id for i in first],
                         [i.frame_or_clip_id for i in second])

    def test_a_moment_landing_on_a_chosen_frame_is_not_shown_twice(self):
        items, _ = select_media(a_recorded_media(length=3))
        ids = [i.frame_or_clip_id for i in items]
        self.assertEqual(len(ids), len(set(ids)))

    def test_an_unmeasurable_moment_is_named_not_substituted(self):
        # No gates recorded, so there is no gate transition to show.
        _, unavailable = select_media(a_recorded_media(length=6))
        self.assertTrue(any("gate_transition" in u for u in unavailable))

    def test_a_gate_crossing_is_selected_when_gates_exist(self):
        gates = [{"contact": v} for v in (0.1, 0.2, 0.9, 0.9, 0.9, 0.9)]
        items, unavailable = select_media(a_recorded_media(length=6, gates=gates))
        reasons = " ".join(i.reason for i in items)
        self.assertIn("gate_transition", reasons)
        self.assertFalse(any("gate_transition" in u for u in unavailable))

    def test_a_gate_that_never_crosses_is_not_a_transition(self):
        gates = [{"contact": 0.1} for _ in range(6)]
        _, unavailable = select_media(a_recorded_media(length=6, gates=gates))
        self.assertTrue(any("gate_transition" in u for u in unavailable))

    def test_an_object_that_never_moves_has_no_movement_frame(self):
        media = a_recorded_media(length=6, obj_jump_at=99)
        _, unavailable = select_media(media)
        self.assertTrue(any("max_object_move" in u for u in unavailable))

    def test_an_empty_episode_selects_nothing(self):
        items, unavailable = select_media(EpisodeMedia(case_id="c"))
        self.assertEqual(items, [])
        self.assertTrue(unavailable)

    def test_at_most_one_frame_per_declared_moment(self):
        items, _ = select_media(a_recorded_media(length=20))
        self.assertLessEqual(len(items), len(MOMENTS))


class TestRecording(unittest.TestCase):
    def _case(self, split=SPLIT_DEVELOPMENT):
        return synthetic_manifest(1, label="push-v2", split=split).episodes[0]

    def test_recording_hangs_off_the_shared_rollout(self):
        record, media = record_episode_media(
            ConstantActor(np.zeros(ACT_DIM)), MockEnv(), self._case(),
            SPLIT_DEVELOPMENT, SCHEMA, max_steps=6,
        )
        self.assertEqual(record.case_id, media.case_id)
        # One observation per step, plus the terminal state.
        self.assertEqual(len(media.tcp), record.length + 1)

    def test_the_observer_cannot_change_the_episode(self):
        plain = run_episode(ConstantActor(np.zeros(ACT_DIM)), MockEnv(), self._case(),
                            max_steps=6, schema=SCHEMA)
        recorded, _ = record_episode_media(
            ConstantActor(np.zeros(ACT_DIM)), MockEnv(), self._case(),
            SPLIT_DEVELOPMENT, SCHEMA, max_steps=6,
        )
        self.assertEqual(plain.episode_return, recorded.episode_return)
        self.assertEqual(plain.length, recorded.length)
        self.assertEqual(plain.failure_label, recorded.failure_label)

    def test_final_test_media_is_refused_outright(self):
        # FR-11: final-test media must never return to the search loop.
        with self.assertRaises(FinalTestMediaRefused):
            record_episode_media(
                ConstantActor(np.zeros(ACT_DIM)), MockEnv(),
                self._case(SPLIT_FINAL_TEST), SPLIT_FINAL_TEST, SCHEMA, max_steps=6,
            )

    def test_a_missing_renderer_loses_frames_not_the_episode(self):
        record, media = record_episode_media(
            ConstantActor(np.zeros(ACT_DIM)), MockEnv(renders=False), self._case(),
            SPLIT_DEVELOPMENT, SCHEMA, max_steps=6,
        )
        self.assertEqual(record.length, 6)
        self.assertTrue(any("frames" in u for u in media.unavailable))

    def test_frames_are_captured_when_the_env_renders(self):
        _, media = record_episode_media(
            ConstantActor(np.zeros(ACT_DIM)), MockEnv(), self._case(),
            SPLIT_DEVELOPMENT, SCHEMA, max_steps=6,
        )
        self.assertTrue(all(f is not None for f in media.frames))
        self.assertEqual(media.unavailable, [])


# ---------------------------------------------------------------------------
#  Merging with the numbers
# ---------------------------------------------------------------------------


class TestMerge(unittest.TestCase):
    def test_an_agreeing_stage_is_marked_agreeing(self):
        merged = merge_analysis(an_analysis(observed_stage="approach"), an_episode())
        self.assertEqual(merged.stage_verdict, AGREE)
        self.assertFalse(merged.has_conflict)

    def test_a_disagreement_is_marked_not_resolved(self):
        merged = merge_analysis(an_analysis(observed_stage="push"), an_episode())
        self.assertEqual(merged.stage_verdict, CONFLICT)
        conflict = merged.conflicts[0]
        self.assertEqual(conflict["visual"], "push")
        self.assertEqual(conflict["numerical_label"], "never_reached_object")
        self.assertIn("neither is preferred", conflict["note"])

    def test_a_stage_it_cannot_check_is_not_a_conflict(self):
        merged = merge_analysis(an_analysis(observed_stage=""), an_episode())
        self.assertEqual(merged.stage_verdict, UNCHECKABLE)
        self.assertFalse(merged.has_conflict)

    def test_an_episode_with_no_label_cannot_be_contradicted(self):
        merged = merge_analysis(an_analysis(), an_episode(label=None))
        self.assertEqual(merged.stage_verdict, UNCHECKABLE)

    def test_recommended_checks_are_run_against_the_episode(self):
        merged = merge_analysis(an_analysis(), an_episode(min_tcp_object_distance=0.07))
        check = merged.checks[0]
        self.assertEqual(check["metric"], "min_tcp_object_distance")
        self.assertAlmostEqual(check["value"], 0.07)
        self.assertEqual(check["status"], "measured")

    def test_a_check_the_episode_does_not_carry_is_uncheckable_not_zero(self):
        merged = merge_analysis(
            an_analysis(recommended_numeric_checks=["lateral_drift"]),
            an_episode(lateral_drift=None),
        )
        self.assertIsNone(merged.checks[0]["value"])
        self.assertEqual(merged.checks[0]["status"], UNCHECKABLE)

    def test_visual_evidence_never_decides_anything(self):
        merged = merge_analysis(an_analysis(), an_episode())
        self.assertFalse(merged.visual_evidence_decided_anything)

    def test_merge_all_pairs_each_analysis_with_its_own_case(self):
        analyses = [an_analysis(case_id="case_0"), an_analysis(case_id="case_1")]
        episodes = [an_episode("case_1"), an_episode("case_0")]
        merged = merge_all(analyses, episodes)
        self.assertEqual([m.case_id for m in merged], ["case_0", "case_1"])

    def test_an_analysis_with_no_matching_episode_is_dropped(self):
        merged = merge_all([an_analysis(case_id="ghost")], [an_episode("case_0")])
        self.assertEqual(merged, [])


class TestRepairPromptBlock(unittest.TestCase):
    def test_the_block_carries_the_claim_and_the_measurement_together(self):
        analyses = [an_analysis()]
        block = repair_prompt_block(merge_all(analyses, [an_episode()]), analyses)
        self.assertIn("gripper is still above the puck", block)
        self.assertIn("min_tcp_object_distance", block)

    def test_the_block_says_the_evidence_decides_nothing(self):
        analyses = [an_analysis()]
        block = repair_prompt_block(merge_all(analyses, [an_episode()]), analyses)
        self.assertIn("can promote, reject or repair a candidate on its own", block)
        self.assertIn("supporting evidence", block)

    def test_a_conflict_is_visible_in_the_block(self):
        analyses = [an_analysis(observed_stage="push")]
        block = repair_prompt_block(merge_all(analyses, [an_episode()]), analyses)
        self.assertIn("CONFLICT", block)
        self.assertIn("neither is resolved", block)

    def test_every_claim_carries_its_frame_reference(self):
        analyses = [an_analysis()]
        block = repair_prompt_block(merge_all(analyses, [an_episode()]), analyses)
        self.assertIn("frame_t0040", block)
        self.assertIn("t=40", block)

    def test_no_analysis_produces_a_block_that_says_so(self):
        self.assertIn("None available", repair_prompt_block([], []))


# ---------------------------------------------------------------------------
#  The analyst interface
# ---------------------------------------------------------------------------


class TestAnalysts(unittest.TestCase):
    def _request(self):
        return AnalysisRequest(
            candidate_id="c0", case_id="case_0", media=[a_media_item()],
            frames=[np.zeros((2, 2, 3), dtype=np.uint8)],
        )

    def test_a_scripted_reply_becomes_a_valid_analysis(self):
        analysis = ScriptedAnalyst([a_reply()]).analyse(self._request())
        self.assertTrue(analysis.validate())

    def test_the_runtime_metadata_the_spec_asks_for_is_recorded(self):
        analysis = ScriptedAnalyst([a_reply()]).analyse(self._request())
        for key in ("quantization", "prompt_version", "media_selection_rule",
                    "latency_seconds", "frames_shown"):
            self.assertIn(key, analysis.runtime)

    def test_an_analyst_with_no_replies_produces_nothing(self):
        # This is the E10 control arm: numeric diagnostics only, same code path.
        self.assertIsNone(ScriptedAnalyst([]).analyse(self._request()))

    def test_the_prompt_only_offers_checks_the_pipeline_measures(self):
        from lares.search.schemas import measurable_metrics

        prompt = ScriptedAnalyst([]).prompt(self._request())
        self.assertNotIn("{measurable}", prompt)
        self.assertIn(sorted(measurable_metrics())[0], prompt)

    def test_the_prompt_tells_the_model_its_output_is_not_decisive(self):
        prompt = ScriptedAnalyst([]).prompt(self._request())
        self.assertIn("supporting evidence", prompt)
        self.assertIn("cannot tell", prompt)

    def test_qwen_refuses_to_pick_a_checkpoint_on_its_own(self):
        with self.assertRaises(ValueError) as caught:
            QwenAnalyst()
        self.assertIn("open research decision", str(caught.exception))

    def test_qwen_names_its_model_and_quantization_for_audit(self):
        analyst = QwenAnalyst(model_id="Qwen/Qwen2-VL-2B-Instruct", quantization="fp16")
        self.assertEqual(analyst.quantization, "fp16")
        self.assertIn("Qwen", analyst.model_id)


class EchoAnalyst(ScriptedAnalyst):
    """Cites whichever frames it was actually shown.

    A real model sees the frame ids in its prompt. A test double that invents
    them would be rejected by the contract, which is the contract working.
    """

    def __init__(self, overrides=None, model_id="scripted"):
        super().__init__([], model_id=model_id)
        self.overrides = dict(overrides or {})

    def analyse(self, request):
        first = request.media[0]
        payload = json.loads(a_reply(**self.overrides))
        for claim in payload["evidence"]:
            claim["frame_or_clip_id"] = first.frame_or_clip_id
            claim["timestep"] = first.start_timestep
        self.replies = [json.dumps(payload)]
        return super().analyse(request)


class TestAnalysisPipeline(unittest.TestCase):
    """The integration point: failed episodes in, merged evidence out."""

    def setUp(self):
        from lares.eval.runner import ManifestResult, evaluate_manifest

        self.manifest = synthetic_manifest(3, label="push-v2", split=SPLIT_DEVELOPMENT)
        self.env = MockEnv()
        self.actor = ConstantActor(np.zeros(ACT_DIM))
        self.result = evaluate_manifest(self.actor, self.manifest, self.env, max_steps=6,
                                        schema=SCHEMA)

    def _run(self, analyst, limit=2, **kw):
        return analyse_failures(
            self.actor, self.env, self.manifest, self.result,
            analyst, SCHEMA, candidate_id="c0", max_steps=6, limit=limit, **kw
        )

    def test_only_failed_episodes_are_analysed(self):
        self.assertTrue(all(e.success <= 0.0 for e in failed_cases(self.result)))

    def test_the_worst_failures_come_first(self):
        progress = [e.signed_goal_progress for e in failed_cases(self.result)]
        self.assertEqual(progress, sorted(progress, key=lambda p: p if p is not None else 0.0))

    def test_the_budget_limits_how_many_episodes_are_analysed(self):
        _, _, run = self._run(EchoAnalyst(), limit=2)
        self.assertLessEqual(run.episodes_analysed, 2)

    def test_a_returned_analysis_is_merged_with_its_episode(self):
        analyses, merged, run = self._run(EchoAnalyst(), limit=1)
        self.assertEqual(run.analyses_returned, 1)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].case_id, analyses[0].case_id)

    def test_an_invalid_analysis_is_dropped_and_the_reason_kept(self):
        bad = a_reply(evidence=[{"statement": "x", "frame_or_clip_id": "ghost",
                                 "timestep": 0}])
        analyses, _, run = self._run(ScriptedAnalyst([bad]), limit=1)
        self.assertEqual(analyses, [])
        self.assertEqual(run.analyses_rejected, 1)
        self.assertIn("never shown", run.rejection_reasons[0])

    def test_no_analyst_output_is_the_numeric_only_arm(self):
        analyses, merged, run = self._run(ScriptedAnalyst([]), limit=2)
        self.assertEqual(analyses, [])
        self.assertEqual(merged, [])
        self.assertGreater(run.episodes_rerun, 0)

    def test_the_run_records_what_the_spec_asks_to_be_audited(self):
        _, _, run = self._run(EchoAnalyst(), limit=1)
        self.assertEqual(run.media_selection_rule, MEDIA_SELECTION_RULE)
        self.assertEqual(run.model_id, "scripted")
        self.assertGreater(run.frames_shown, 0)
        self.assertGreaterEqual(run.latency_seconds, 0.0)

    def test_unavailable_moments_are_reported_per_case(self):
        _, _, run = self._run(EchoAnalyst(), limit=1)
        self.assertTrue(any("gate_transition" in u for u in run.unavailable))

    def test_conflicts_are_counted(self):
        # The mock episodes are labelled stopped_short_of_goal, so a model that
        # reports the approach stage is contradicting the geometry.
        _, _, run = self._run(EchoAnalyst({"observed_stage": "approach"}), limit=1)
        self.assertEqual(run.conflicts, 1)
        self.assertEqual(run.analyses_returned, 1)

    def test_an_agreeing_analysis_is_not_counted_as_a_conflict(self):
        _, _, run = self._run(EchoAnalyst({"observed_stage": "push"}), limit=1)
        self.assertEqual(run.conflicts, 0)
        self.assertEqual(run.analyses_returned, 1)

    def test_the_evidence_block_is_produced_from_the_merge(self):
        analyses, merged, _ = self._run(EchoAnalyst(), limit=1)
        self.assertIn("Visual analysis", evidence_block(merged, analyses))

    def test_a_final_test_manifest_is_refused_by_the_pipeline_too(self):
        from lares.eval.runner import ManifestResult

        final = synthetic_manifest(2, label="push-v2", split=SPLIT_FINAL_TEST)
        result = ManifestResult(
            actor_name="a", manifest_id=final.manifest_id, action_mode="deterministic",
            episodes=[an_episode(c.case_id) for c in final.episodes],
        )
        with self.assertRaises(FinalTestMediaRefused):
            analyse_failures(
                self.actor, self.env, final, result, EchoAnalyst(),
                SCHEMA, candidate_id="c0", max_steps=6, limit=1,
            )


if __name__ == "__main__":
    unittest.main()
