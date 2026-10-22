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
    RESEARCH_METRICS,
    SOURCE_ADVISOR,
    SOURCE_IMPLEMENTER,
    STATUS_ANALYSED,
    STATUS_APPROVED,
    STATUS_DECIDED,
    STATUS_ESCALATED,
    STATUS_PROPOSED,
    STATUS_RUNNING,
    Prediction,
    Protocol,
    ResearchIdea,
    ResearchSchemaError,
    Verdict,
    decide,
    load_decisions,
    load_ideas,
    next_idea_id,
)
from lares.research.state import render_state  # noqa: E402
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
        idea.transition(STATUS_APPROVED)
        idea.transition(STATUS_RUNNING)
        with self.assertRaises(ResearchSchemaError):
            idea.transition(STATUS_PROPOSED)
        self.assertEqual([h["to"] for h in idea.history], [STATUS_APPROVED, STATUS_RUNNING])

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
            self.assertIsNone(decide(idea, root))
            self.assertEqual(idea.status, STATUS_ESCALATED)
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


class TestCli(unittest.TestCase):
    def test_full_cycle_without_llm(self):
        with tempfile.TemporaryDirectory() as root:
            # The protocol check reads the real seed ledger, which grows as research runs (R-022 spent 2600), so
            # this uses a seed far outside any range the project reserves.
            idea = make_idea(seeds=(97001,))
            idea.validate(METRICS)
            idea.save(root)
            cli.main(["--root", root, "approve", "R-001", "--note", "ok"])
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
