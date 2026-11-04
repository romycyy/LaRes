"""The research-idea ledger against a scripted advisor (no API key, no simulator).

    python -m unittest tests.test_research
"""

import json
import os
import sys
import tempfile
import unittest

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))

from lares.research import advisor as adv  # noqa: E402
from lares.research import protocol as proto  # noqa: E402
from lares.research.briefing import build_briefing  # noqa: E402
from lares.research.schemas import (  # noqa: E402
    PROPOSALS_LOG,
    RESEARCH_METRICS,
    SOURCE_ADVISOR,
    SOURCE_IMPLEMENTER,
    STAGE_TRIAGE,
    STAGE_VERDICT,
    STATUS_ANALYSED,
    STATUS_APPROVED,
    STATUS_DECIDED,
    STATUS_DEFERRED,
    STATUS_ESCALATED,
    STATUS_PROPOSED,
    STATUS_REJECTED,
    STATUS_RUNNING,
    TRIAGE_LOG,
    Prediction,
    Protocol,
    ResearchIdea,
    ResearchSchemaError,
    Verdict,
    apply_revision,
    decide,
    load_decisions,
    load_direction,
    load_ideas,
    load_jsonl,
    next_idea_id,
    record_triage_advice,
    record_triage_position,
    rule_triage,
    set_direction,
)
from lares.research.state import loop_guard, render_state  # noqa: E402
import research as cli  # noqa: E402

METRICS = set(RESEARCH_METRICS)


def make_idea(idea_id="R-001", metric="promotion_success_rate", seeds=(2600,), **kw):
    fields = dict(
        id=idea_id,
        title="Fixed 4:2 allocation beats ranked parents",
        hypothesis="Guaranteeing damped-push slots raises promotion success because ranked selection collapses to one family.",
        prediction=Prediction(metric=metric, direction="increase", threshold=0.1, against="baseline"),
        protocol=Protocol(script="scripts/run_full_evolution.py", config="config/two_lane_search.yaml",
                          command="python scripts/run_full_evolution.py --config config/two_lane_search.yaml",
                          seeds=list(seeds), episodes=400, llm_calls=12, needs_llm=True),
        falsified_by="promotion success no higher than the ranked run on the same manifest",
    )
    fields.update(kw)
    return ResearchIdea(**fields)


def raw_idea(seed=2601, metric="promotion_success_rate", script="scripts/run_full_evolution.py"):
    return {
        "title": "Rotate the development screen",
        "hypothesis": "Rotating screen cases removes the +0.19 wrong-sign selection effect.",
        "prediction": {"metric": metric, "direction": "increase", "threshold": 0.05, "against": "baseline"},
        "falsified_by": "no change in promotion success under rotation",
        "protocol": {"script": script, "config": "", "command": "python " + script, "seeds": [seed],
                     "manifests": ["config/manifests/push-v2_development.yaml"], "episodes": 300, "llm_calls": 10,
                     "needs_gpu": True, "needs_llm": True},
        "cost_estimate": {"episodes": 300, "llm_calls": 10},
        "open_question_refs": [3],
    }


class ScriptedCall:
    def __init__(self, replies):
        self.replies = list(replies)
        self.messages = []

    def __call__(self, messages):
        self.messages.append(messages)
        text = self.replies.pop(0)
        return text, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, "scripted", {"temperature": 1.0}


class TestSchemas(unittest.TestCase):
    def test_prediction_must_name_a_measured_metric(self):
        with self.assertRaises(ResearchSchemaError):
            make_idea(metric="vibes").validate(METRICS)
        make_idea().validate(METRICS)

    def test_protocol_refuses_final_test(self):
        idea = make_idea(protocol=Protocol(script="scripts/x.py", manifests=["config/manifests/push-v2_final_test.yaml"]))
        with self.assertRaises(ResearchSchemaError):
            idea.validate(METRICS)

    def test_transitions_are_ordered(self):
        idea = make_idea()
        with self.assertRaises(ResearchSchemaError):
            idea.transition(STATUS_RUNNING)        # nothing runs before approval
        idea.transition(STATUS_DEFERRED)
        idea.transition(STATUS_PROPOSED)
        idea.transition(STATUS_APPROVED)
        idea.transition(STATUS_RUNNING)
        with self.assertRaises(ResearchSchemaError):
            idea.transition(STATUS_PROPOSED)
        self.assertEqual([h["to"] for h in idea.history], [STATUS_DEFERRED, STATUS_PROPOSED, STATUS_APPROVED, STATUS_RUNNING])

    def test_prediction_check_direction(self):
        p = Prediction(metric="promotion_success_rate", direction="increase", threshold=0.1)
        self.assertTrue(p.check(0.2, 0.35)["matched"])
        self.assertFalse(p.check(0.2, 0.25)["matched"])
        self.assertIsNone(p.check(None, 0.3)["matched"])

    def test_advisor_verdict_must_cite(self):
        with self.assertRaises(ResearchSchemaError):
            Verdict(source=SOURCE_ADVISOR, verdict="supported", confidence=0.9).validate()
        Verdict(source=SOURCE_IMPLEMENTER, verdict="supported", confidence=0.9).validate()

    def test_save_load_and_ids(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(next_idea_id(root), "R-001")
            idea = make_idea()
            idea.validate(METRICS)
            idea.save(root)
            self.assertEqual(next_idea_id(root), "R-002")
            loaded = load_ideas(root)[0]
            self.assertEqual(loaded.to_dict(), idea.to_dict())


class TestProtocolChecks(unittest.TestCase):
    def test_spent_and_reserved_seeds_are_refused(self):
        with tempfile.TemporaryDirectory() as root:
            ledger = os.path.join(root, "seeds.jsonl")
            with open(ledger, "w") as f:
                f.write(json.dumps({"seed": 2600, "namespace": "mt1_pool", "use": "spent"}) + "\n")
            other = make_idea("R-001", seeds=(2601,))
            other.save(root)
            idea = make_idea("R-002", seeds=(2600, 2601, 2602))
            problems = proto.check_idea(idea, root, ledger_path=ledger, metrics=METRICS)
            self.assertTrue(any("2600 is spent" in p for p in problems))
            self.assertTrue(any("2601 is already declared by R-001" in p for p in problems))
            self.assertFalse(any("2602" in p for p in problems))

    def test_missing_script_is_reported_and_start_needs_approval(self):
        with tempfile.TemporaryDirectory() as root:
            ledger = os.path.join(root, "seeds.jsonl")
            idea = make_idea(protocol=Protocol(script="scripts/does_not_exist.py", episodes=10))
            self.assertTrue(any("does not exist" in p for p in proto.check_idea(idea, root, ledger_path=ledger, metrics=METRICS)))
            good = make_idea()
            good.save(root)
            with self.assertRaises(ResearchSchemaError):
                proto.start_run(good, root, ledger_path=ledger)   # still proposed
            good.approval = {"by": "user", "at": "now"}
            good.transition(STATUS_APPROVED)
            proto.start_run(good, root, ledger_path=ledger)
            self.assertEqual(good.status, STATUS_RUNNING)
            self.assertTrue(good.run["git_sha"])


class TestAdvisorAndDecisions(unittest.TestCase):
    def _root_with_results(self, root):
        idea = make_idea()
        idea.approval = {"by": "user", "at": "now"}
        idea.transition(STATUS_APPROVED)
        idea.transition(STATUS_RUNNING)
        idea.result = {"metrics": {"promotion_success_rate": 0.4}, "controls": {"promotion_success_rate": 0.2},
                       "artifacts": ["logs/x/screening.json"], "notes": "paired interval [0.05, 0.35]"}
        idea.prediction_outcome = idea.prediction.check(0.2, 0.4)
        idea.transition(STATUS_ANALYSED)
        idea.save(root)
        return idea

    def test_propose_parses_strictly_and_records_the_call(self):
        with tempfile.TemporaryDirectory() as root:
            good = raw_idea(seed=2601)
            bad = raw_idea(seed=2602, metric="not_measured")
            call = ScriptedCall([json.dumps([good, bad])])
            advisor = adv.Advisor(call, root)
            briefing = build_briefing(root, "propose")
            out = advisor.propose(briefing, n=2)
            self.assertEqual(len(out["ideas"]), 1)
            self.assertEqual(out["ideas"][0].id, "R-001")
            self.assertEqual(out["ideas"][0].briefing_sha256, briefing["sha256"])
            self.assertEqual(len(out["rejected"]), 1)
            self.assertIn("nothing measures", out["rejected"][0]["reason"])
            self.assertEqual(len(advisor.ledger), 1)
            self.assertEqual(advisor.ledger.records()[0].label, "research_propose")
            self.assertIn("Standing constraints", call.messages[0][1]["content"])

    def test_propose_wrong_count_raises(self):
        with tempfile.TemporaryDirectory() as root:
            advisor = adv.Advisor(ScriptedCall([json.dumps([raw_idea()])]), root)
            with self.assertRaises(ResearchSchemaError):
                advisor.propose(build_briefing(root, "propose"), n=3)

    def test_analyse_requires_known_artifacts(self):
        with tempfile.TemporaryDirectory() as root:
            idea = self._root_with_results(root)
            reply = {"verdict": "supported", "confidence": 0.8, "reasoning": "x", "cited_artifacts": ["logs/other.json"]}
            advisor = adv.Advisor(ScriptedCall([json.dumps(reply)]), root)
            with self.assertRaises(ResearchSchemaError):
                advisor.analyse(idea, build_briefing(root, "a"))
            reply["cited_artifacts"] = ["logs/x/screening.json"]
            advisor = adv.Advisor(ScriptedCall([json.dumps(reply)]), root)
            verdict = advisor.analyse(idea, build_briefing(root, "a"))
            self.assertEqual(verdict.verdict, "supported")
            self.assertTrue(verdict.call_id)

    def test_consult_records_the_call_and_checks_every_id(self):
        with tempfile.TemporaryDirectory() as root:
            idea = make_idea()
            idea.save(root)
            entry = {"id": idea.id, "recommendation": "revise", "reasoning": "x", "suggested_changes": ["y"]}
            reply = {"ideas": [entry], "order": [idea.id], "next_step": "z"}
            advisor = adv.Advisor(ScriptedCall([json.dumps(reply)]), root)
            out = advisor.consult([idea], build_briefing(root, "c"), question="what first?")
            self.assertEqual(out["ideas"][0]["recommendation"], "revise")
            self.assertTrue(out["call_id"])
            self.assertEqual(out["question"], "what first?")
            for bad in ({"ideas": [], "order": []},
                        {"ideas": [dict(entry, recommendation="approve")], "order": []},
                        {"ideas": [entry], "order": ["R-999"]}):
                advisor = adv.Advisor(ScriptedCall([json.dumps(bad)]), root)
                with self.assertRaises(ResearchSchemaError):
                    advisor.consult([idea], build_briefing(root, "c"))
            self.assertEqual(load_ideas(root)[0].status, idea.status)   # advice changes no idea

    def test_agreement_decides_and_disagreement_escalates(self):
        with tempfile.TemporaryDirectory() as root:
            idea = self._root_with_results(root)
            idea.add_verdict(Verdict(SOURCE_ADVISOR, "supported", 0.8, cited_artifacts=["logs/x/screening.json"]))
            idea.add_verdict(Verdict(SOURCE_IMPLEMENTER, "inconclusive", 0.6, reasoning="interval lower bound 0.05 on 30 cases"))
            # One advisor verdict so far: below the round cap the idea stays analysed for another exchange.
            self.assertIsNone(decide(idea, root, max_rounds=2))
            self.assertEqual(idea.status, STATUS_ANALYSED)
            idea.add_verdict(Verdict(SOURCE_ADVISOR, "supported", 0.8, reasoning="[revised after review] holds",
                                     cited_artifacts=["logs/x/screening.json"]))
            self.assertIsNone(decide(idea, root, max_rounds=2))
            self.assertEqual(idea.status, STATUS_ESCALATED)
            self.assertEqual(idea.escalation["stage"], STAGE_VERDICT)
            self.assertEqual(load_decisions(root), [])
            # The user rules; the disagreement is kept in the entry.
            d = decide(idea, root, user_ruling={"outcome": "inconclusive", "note": "rerun at 50 cases"}, summary="User ruled.")
            self.assertEqual(idea.status, STATUS_DECIDED)
            self.assertFalse(d.agreed)
            self.assertEqual(load_decisions(root)[0]["user_ruling"]["note"], "rerun at 50 cases")
            self.assertIn("inconclusive", render_state(root))

    def test_agreement_writes_decision(self):
        with tempfile.TemporaryDirectory() as root:
            idea = self._root_with_results(root)
            idea.add_verdict(Verdict(SOURCE_ADVISOR, "supported", 0.8, reasoning="a", cited_artifacts=["logs/x/screening.json"]))
            idea.add_verdict(Verdict(SOURCE_IMPLEMENTER, "supported", 0.7, reasoning="b", next_experiment="c"))
            d = decide(idea, root)
            self.assertTrue(d.agreed)
            self.assertEqual(d.id, "D-001")
            self.assertEqual(d.next_step, "c")
            self.assertEqual(idea.decision_id, "D-001")

    def test_review_can_revise_the_advisor(self):
        with tempfile.TemporaryDirectory() as root:
            idea = self._root_with_results(root)
            idea.add_verdict(Verdict(SOURCE_ADVISOR, "supported", 0.8, cited_artifacts=["logs/x/screening.json"]))
            idea.add_verdict(Verdict(SOURCE_IMPLEMENTER, "inconclusive", 0.6, reasoning="r"))
            reply = {"agree_with_implementer": True, "revised_verdict": "inconclusive", "confidence": 0.6,
                     "reasoning": "the control was 10 cases", "disagreement_is_about": "data"}
            advisor = adv.Advisor(ScriptedCall([json.dumps(reply)]), root)
            advisor.review(idea)
            self.assertEqual(idea.verdict_from(SOURCE_ADVISOR).verdict, "inconclusive")
            self.assertIsNotNone(decide(idea, root))


class TestTriage(unittest.TestCase):
    def _proposed(self, root, idea_id="R-001", seeds=(97001,)):
        idea = make_idea(idea_id, seeds=seeds)
        idea.validate(METRICS)
        idea.save(root)
        return idea

    def test_agreement_approves_with_both_parties_recorded(self):
        with tempfile.TemporaryDirectory() as root:
            idea = self._proposed(root)
            record_triage_advice(idea, {"decision": "approve", "reasoning": "r", "priority": 1}, "call_00001", "abc")
            res = record_triage_position(idea, "approve", "agree", root=root)
            self.assertEqual(res["outcome"], "agreed")
            self.assertEqual(idea.status, STATUS_APPROVED)
            self.assertEqual(idea.approval["by"], "advisor+implementer")
            self.assertEqual(idea.approval["advisor_call_id"], "call_00001")
            self.assertEqual(idea.triage["agreed_round"], 1)
            log = load_jsonl(root, TRIAGE_LOG)
            self.assertEqual((log[0]["advisor"]["decision"], log[0]["implementer"]["position"]), ("approve", "approve"))

    def test_disagreement_runs_another_round_then_escalates(self):
        with tempfile.TemporaryDirectory() as root:
            idea = self._proposed(root)
            record_triage_advice(idea, {"decision": "reject", "reasoning": "dup"}, "call_1")
            with self.assertRaises(ResearchSchemaError):
                record_triage_advice(idea, {"decision": "reject", "reasoning": "dup"}, "call_2")   # round still open
            res = record_triage_position(idea, "approve", "not a duplicate: D-003 used 10 cases", root=root, max_rounds=2)
            self.assertEqual(res["outcome"], "disagreed")
            self.assertEqual(idea.status, STATUS_PROPOSED)
            record_triage_advice(idea, {"decision": "reject", "reasoning": "still a dup"}, "call_2")
            res = record_triage_position(idea, "approve", "still disagree", root=root, max_rounds=2)
            self.assertEqual(res["outcome"], "escalated")
            self.assertEqual(idea.status, STATUS_ESCALATED)
            self.assertEqual(idea.escalation["stage"], STAGE_TRIAGE)
            self.assertEqual(len(load_jsonl(root, TRIAGE_LOG)), 2)
            # Only the user moves it on.
            rule_triage(idea, "approve", "run it, the control differs", root=root)
            self.assertEqual(idea.status, STATUS_APPROVED)
            self.assertEqual(idea.approval["by"], "user")
            self.assertIn("user ruling", render_state(root))

    def test_advisor_changing_its_mind_in_round_two_agrees(self):
        with tempfile.TemporaryDirectory() as root:
            idea = self._proposed(root)
            record_triage_advice(idea, {"decision": "defer", "reasoning": "wait"}, "call_1")
            record_triage_position(idea, "approve", "it blocks everything else", root=root)
            record_triage_advice(idea, {"decision": "approve", "reasoning": "accepted: it blocks R-002"}, "call_2")
            res = record_triage_position(idea, "approve", "agreed", root=root)
            self.assertEqual((res["outcome"], idea.status, idea.triage["agreed_round"]), ("agreed", STATUS_APPROVED, 2))

    def test_both_revise_then_revision_then_approve(self):
        with tempfile.TemporaryDirectory() as root:
            idea = self._proposed(root)
            record_triage_advice(idea, {"decision": "revise", "reasoning": "threshold too low",
                                        "suggested_changes": ["threshold 0.2"]}, "call_1")
            res = record_triage_position(idea, "revise", "fair", root=root)
            self.assertEqual((res["outcome"], idea.status), ("agreed", STATUS_PROPOSED))
            apply_revision(idea, {"prediction": {"threshold": 0.2}}, note="advisor round 1", root=root, metrics=METRICS)
            self.assertEqual(load_ideas(root)[0].prediction.threshold, 0.2)
            self.assertEqual(idea.revisions[0]["round"], 1)
            with self.assertRaises(ResearchSchemaError):
                apply_revision(idea, {"status": "approved"}, root=root, metrics=METRICS)
            record_triage_advice(idea, {"decision": "approve", "reasoning": "ok now"}, "call_2")
            record_triage_position(idea, "approve", "ok", root=root, changes_applied=["threshold 0.2"])
            self.assertEqual(idea.status, STATUS_APPROVED)

    def test_defer_and_reject_outcomes(self):
        with tempfile.TemporaryDirectory() as root:
            a = self._proposed(root, "R-001", seeds=(97001,))
            b = self._proposed(root, "R-002", seeds=(97002,))
            record_triage_advice(a, {"decision": "defer", "reasoning": "after R-002"}, "c1")
            record_triage_position(a, "defer", "yes", root=root)
            record_triage_advice(b, {"decision": "reject", "reasoning": "duplicate of D-001"}, "c1")
            record_triage_position(b, "reject", "yes", root=root)
            self.assertEqual([i.status for i in load_ideas(root)], [STATUS_DEFERRED, STATUS_REJECTED])
            # A deferred idea can be re-triaged; it returns to proposed first.
            record_triage_advice(a, {"decision": "approve", "reasoning": "now"}, "c2")
            self.assertEqual(a.status, STATUS_PROPOSED)
            with self.assertRaises(ResearchSchemaError):
                record_triage_advice(b, {"decision": "approve", "reasoning": "x"}, "c3")   # rejected is final
            self.assertIn("R-002", build_briefing(root, "propose")["text"])   # recently rejected is shown to the advisor

    def test_advisor_triage_call_is_strict_and_sees_history(self):
        with tempfile.TemporaryDirectory() as root:
            idea = self._proposed(root)
            record_triage_advice(idea, {"decision": "reject", "reasoning": "dup"}, "c1")
            record_triage_position(idea, "approve", "the control differs", root=root)
            good = {"ideas": [{"id": "R-001", "decision": "approve", "reasoning": "accepted the control argument", "priority": 1}],
                    "order": ["R-001"], "next_step": "start"}
            call = ScriptedCall([json.dumps(good)])
            advisor = adv.Advisor(call, root)
            out = advisor.triage([idea], build_briefing(root, "triage"), direction="finish the fittable lane")
            self.assertEqual(out["ideas"]["R-001"]["decision"], "approve")
            self.assertEqual(advisor.ledger.records()[0].label, "research_triage")
            prompt = call.messages[0][1]["content"]
            self.assertIn("the control differs", prompt)            # the implementer's objection reaches the advisor
            self.assertIn("finish the fittable lane", prompt)
            for bad in ({"ideas": [], "order": []},
                        {"ideas": [{"id": "R-001", "decision": "maybe", "reasoning": "x"}]},
                        {"ideas": [{"id": "R-001", "decision": "revise", "reasoning": "x", "suggested_changes": []}]},
                        {"ideas": [{"id": "R-001", "decision": "approve", "reasoning": ""}]},
                        {"ideas": [{"id": "R-001", "decision": "approve", "reasoning": "x"}], "order": ["R-009"]}):
                with self.assertRaises(ResearchSchemaError):
                    adv.Advisor(ScriptedCall([json.dumps(bad)]), root).triage([idea], build_briefing(root, "t"))

    def test_direction_history_and_loop_guard(self):
        with tempfile.TemporaryDirectory() as root:
            set_direction("first", root)
            set_direction("second", root)
            d = load_direction(root)
            self.assertEqual((d["text"], d["history"][0]["text"]), ("second", "first"))
            self.assertIn("second", build_briefing(root, "propose")["text"])
            # Two propose calls whose every idea was rejected trip the guard.
            from lares.research.schemas import log_proposal
            for call_id, idea_id, seed in (("call_1", "R-001", 97001), ("call_2", "R-002", 97002)):
                idea = self._proposed(root, idea_id, seeds=(seed,))
                log_proposal(root, call_id, "sha", "second", -1, idea.to_dict(), idea_id=idea_id)
                record_triage_advice(idea, {"decision": "reject", "reasoning": "off direction"}, call_id)
                record_triage_position(idea, "reject", "agree", root=root)
            self.assertEqual(loop_guard(root)["consecutive_all_rejected"], 2)
            with self.assertRaises(SystemExit):
                cli.main(["--root", root, "propose", "--dry-run"])
            self.assertEqual(len(load_jsonl(root, PROPOSALS_LOG)), 2)


class TestCli(unittest.TestCase):
    def test_full_cycle_without_llm(self):
        with tempfile.TemporaryDirectory() as root:
            # The protocol check reads the real seed ledger, which grows as research runs (R-022 spent 2600), so
            # this uses a seed far outside any range the project reserves.
            idea = make_idea(seeds=(97001,))
            idea.validate(METRICS)
            idea.save(root)
            cli.main(["--root", root, "direct", "test the fixed allocation"])
            self.assertEqual(load_direction(root)["text"], "test the fixed allocation")
            reply = {"ideas": [{"id": "R-001", "decision": "approve", "reasoning": "cheap and decisive", "priority": 1}],
                     "order": ["R-001"], "next_step": "start it"}
            cli.ADVISOR_FACTORY = lambda args: adv.Advisor(ScriptedCall([json.dumps(reply)]), root)
            try:
                cli.main(["--root", root, "triage"])
            finally:
                cli.ADVISOR_FACTORY = None
            self.assertEqual(load_ideas(root)[0].status, STATUS_PROPOSED)   # awaits the implementer
            cli.main(["--root", root, "position", "R-001", "--position", "approve", "--reasoning", "agree: 150 episodes, no LLM"])
            approved = load_ideas(root)[0]
            self.assertEqual(approved.status, STATUS_APPROVED)
            self.assertEqual(approved.approval["by"], "advisor+implementer")
            self.assertEqual(load_jsonl(root, TRIAGE_LOG)[0]["outcome"], "agreed")
            cli.main(["--root", root, "start", "R-001"])
            results = os.path.join(root, "r.json")
            with open(results, "w") as f:
                json.dump({"metrics": {"promotion_success_rate": 0.4}, "controls": {"promotion_success_rate": 0.2},
                           "artifacts": ["logs/missing.json"]}, f)
            with self.assertRaises(SystemExit):
                cli.main(["--root", root, "record", "R-001", "--results", results])   # artifact missing locally
            cli.main(["--root", root, "record", "R-001", "--results", results, "--allow-missing-artifacts"])
            loaded = load_ideas(root)[0]
            self.assertEqual(loaded.status, STATUS_ANALYSED)
            self.assertTrue(loaded.prediction_outcome["matched"])
            cli.main(["--root", root, "verdict", "R-001", "--verdict", "supported", "--confidence", "0.7", "--reasoning", "x"])
            loaded = load_ideas(root)[0]
            loaded.add_verdict(Verdict(SOURCE_ADVISOR, "supported", 0.8, cited_artifacts=["logs/missing.json"]))
            loaded.save(root)
            cli.main(["--root", root, "decide", "R-001", "--summary", "done"])
            self.assertEqual(load_ideas(root)[0].status, STATUS_DECIDED)
            self.assertTrue(os.path.isfile(os.path.join(root, "STATE.md")))

    def test_propose_dry_run_builds_briefing(self):
        with tempfile.TemporaryDirectory() as root:
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()) as out:
                cli.main(["--root", root, "propose", "--dry-run"])
            self.assertIn("Standing constraints", out.getvalue())
            self.assertTrue(os.listdir(os.path.join(root, "briefings")))


if __name__ == "__main__":
    unittest.main()


class TestSeedReservations(unittest.TestCase):
    def test_own_frozen_manifest_reserves_rather_than_spends(self):
        with tempfile.TemporaryDirectory() as root:
            open_idea = make_idea("R-010", seeds=(4653,))
            open_idea.protocol = Protocol(**{**open_idea.protocol.__dict__, "manifests": ("config/manifests/x_s4653.yaml",)})
            open_idea.save(root)
            closed = make_idea("R-011", seeds=())
            closed.protocol = Protocol(**{**closed.protocol.__dict__, "manifests": ("config/manifests/y_s4654.yaml",)})
            closed.status = "decided"                                    # not saved: a decided idea's manifest is spent
            used = {4653: "manifest config/manifests/x_s4653.yaml", 4654: "manifest config/manifests/y_s4654.yaml",
                    2600: "mt1_pool: spent"}
            consumed, reserved = proto.split_seed_uses(used, load_ideas(root) + [closed])
            self.assertEqual(reserved, {4653: "R-010"})
            self.assertEqual(sorted(consumed), [2600, 4654])


class TestLoopGuardAbstention(unittest.TestCase):
    def test_zero_idea_propose_calls_count_as_rejected_in_full(self):
        from lares.research.schemas import log_proposal
        from lares.research.state import loop_guard

        with tempfile.TemporaryDirectory() as root:
            idea = make_idea("R-001")
            idea.save(root)
            log_proposal(root, "call_1", "sha", "dir", index=0, raw={}, idea_id="R-001")
            log_proposal(root, "call_2", "sha", "dir", index=-1, raw={}, idea_id="", parse_error="advisor abstained: 0 ideas")
            log_proposal(root, "call_3", "sha", "dir", index=0, raw={}, idea_id="", parse_error="unparseable")
            self.assertEqual(loop_guard(root)["consecutive_all_rejected"], 2)
            log_proposal(root, "call_4", "sha", "dir", index=0, raw={}, idea_id="R-001")
            self.assertEqual(loop_guard(root)["consecutive_all_rejected"], 0)
