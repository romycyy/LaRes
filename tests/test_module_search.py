"""R-131: seed streams, matched accounting, the frozen ranking, the shared validator, generation and inference (no Isaac,
no model call)."""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lares.envs.isaac_pen_spin import module_language as ml  # noqa: E402
from lares.envs.isaac_pen_spin import module_search as ms  # noqa: E402
from lares.utils.cli import load_yaml  # noqa: E402
from tests.test_module_language import signals  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
CFG = {**load_yaml(os.path.join(ROOT, "config", "pen_spin_plateau_repair.yaml"))[ms.CONFIG_KEY], "model": "m", "reasoning_effort": "high"}
PARENT = ml.record(ml.TEMPLATE, np.r_[0.9, 0.5, 0.4, 0.2, np.full(6, 0.7), 0.5])


class TestStreams(unittest.TestCase):
    def test_keys_are_unique_full_tuples(self):
        keys = ms.all_stream_keys(CFG)
        self.assertEqual(len(keys), 5 * (2 * (1 + 1 + 4 + 12) + 24))
        self.assertEqual(len(set(keys.values())), len(keys))
        self.assertTrue(all(len(k) == 6 and k[0] in CFG["block_seeds"] for k in keys.values()))
        self.assertEqual({k[0] for k in keys.values()}, set(range(5300, 5305)))

    def test_placements_and_directions_are_shared_as_declared(self):
        a = ms.placements(CFG, 0, "train", 1, 2)
        self.assertEqual(a.shape[0], 512)
        np.testing.assert_array_equal(a, ms.placements(CFG, 0, "train", 1, 2))
        self.assertFalse(np.array_equal(a, ms.placements(CFG, 1, "train", 1, 2)))
        self.assertFalse(np.array_equal(a, ms.placements(CFG, 0, "train", 0, 2)))
        self.assertFalse(np.array_equal(a, ms.placements(CFG, 0, "select", 1)))
        child = ml.apply_edit(ml.TEMPLATE, {"component": "boost_end", "value": "event"})
        d0 = ms.directions(CFG, 0, "direction", 0, 1, 3, ml.active(ml.TEMPLATE))
        d1 = ms.directions(CFG, 0, "direction", 0, 1, 3, ml.active(child))
        self.assertEqual((d0.shape, d1.shape), ((4, 11), (4, 13)))
        np.testing.assert_array_equal(d0, d1[:, :11])              # shared coordinates share their draws
        self.assertFalse(np.array_equal(d0, ms.directions(CFG, 0, "direction", 0, 2, 3, ml.active(ml.TEMPLATE))))

    def test_provenance_and_registry(self):
        p = ms.stream_provenance(CFG)
        self.assertEqual(p["placements"], 5 * (2 * (1 + 4) + 12) * 512 + 512)
        self.assertEqual(ms.freeze_manifests(CFG), [])
        with self.assertRaises(SystemExit):
            ms.stream_provenance({**CFG, "block_seeds": [5300, 5300, 5302, 5303, 5304]})
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import run_full_evolution as rfe

        self.assertEqual(rfe.PEN_SPIN_STUDIES[ms.STUDY].config_key, ms.CONFIG_KEY)

    def test_matched_accounting_is_the_registered_one(self):
        per_update = 2 * 4 * 512
        updates = CFG["rounds"] * CFG["branches"] * CFG["updates_per_branch"] + CFG["continuation_updates"]
        self.assertEqual(updates, 36)
        self.assertEqual(5 * 3 * updates * per_update, 2_211_840)
        self.assertEqual(5 * 3 * CFG["rounds"] * 4 * 512, 61_440)
        self.assertEqual((15 + 4) * 12 * 128, 29_184)
        self.assertEqual(5 * CFG["rounds"] * (CFG["branches"] - 1) * CFG["max_calls_per_slot"], 40)
        for arm in ms.ARMS:                                       # every arm trains three branches per round
            props = [] if arm == "fixed" else [{"edit": {"component": "target", "value": "hold"}}, {"edit": None}]
            self.assertEqual(len(ms.branch_records(PARENT, props, arm, 3)), 3)


class TestSmoke(unittest.TestCase):
    def test_probe_is_admissible_and_every_edit_is_active(self):
        p = ms.smoke_probe()                                      # the first launch built a four-edit probe, past the cap
        self.assertEqual((p["edits"], p["parameters"]), (3, 17))
        self.assertEqual(ml.check_record(p)["params"]["max_extra"], 60.0)
        sig = signals()
        tpl = ml.template_record()
        a, b = ml.replay(tpl["spec"], tpl["x"], sig), ml.replay(p["spec"], p["x"], sig)
        self.assertGreaterEqual(b["fired"].sum(), a["fired"].sum())
        self.assertGreater(b["boost"].sum(), a["boost"].sum())
        self.assertGreater(b["beta"].sum(), a["beta"].sum())

    def test_a_dead_child_is_charged_on_relaunch(self):
        with tempfile.TemporaryDirectory() as d:
            budget = ms.Budget(os.path.join(d, "a.json"), CFG)
            stem = os.path.join(d, "smoke")
            open(stem + ".request.json", "w").write("{}")
            req = {"mode": "smoke", "response_path": stem + ".json"}

            def shell(env, cmd, log=None):
                ms._dump(req["response_path"], {"starts": 16000, "transitions": 5})

            with mock.patch.object(ms, "isaac_shell", shell):
                ms.run_child({**CFG, "isaac_env": "x"}, req, stem, budget, 20000)
            self.assertEqual((budget.data["starts"], budget.data["relaunches"][0]["charged_starts"]), (36000, 20000))


class TestSelection(unittest.TestCase):
    def test_frozen_ranking(self):
        s = lambda ok, drops=10: {"successes": ok, "drops": drops, "cases": 512}  # noqa: E731
        self.assertEqual(ms.rank_branches([s(300), s(310), s(330), s(320)], 0.05)["winner"], 1)
        self.assertEqual(ms.rank_branches([s(300), s(330), s(330), s(330)], 0.05)["winner"], 0)        # ties: the parent structure
        self.assertEqual(ms.rank_branches([s(300), s(310), s(330), s(330)], 0.05)["winner"], 1)        # then the earlier branch
        self.assertEqual(ms.rank_branches([s(300), s(310), s(400, 40), s(305)], 0.05)["winner"], 0)    # drop bound: (40-10)/512 > 0.05
        self.assertEqual(ms.rank_branches([s(300), s(310), s(400, 35), s(305)], 0.05)["winner"], 1)
        none = ms.rank_branches([s(300), s(310, 60), s(400, 60), s(305, 60)], 0.05)
        self.assertEqual((none["winner"], none["retained"]), (None, "parent"))
        self.assertEqual(ms.rank_branches([s(300), s(200), s(190), s(180)], 0.05)["winner"], 0)        # a branch is kept even below the parent

    def test_branches(self):
        props = [{"edit": {"component": "boost_end", "value": "event"}}, {"edit": None}]
        b = ms.branch_records(PARENT, props, "llm", 3)
        self.assertEqual([x["spec"]["boost_end"] for x in b], ["timed", "event", "timed"])
        self.assertEqual((b[0]["x"], b[2]["x"], b[2]["parent_copy"]), (PARENT["x"], PARENT["x"], True))
        self.assertEqual(b[1]["x"][:11], PARENT["x"])
        self.assertEqual(b[1]["parameters"], 13)
        f = ms.branch_records(PARENT, props, "fixed", 3)
        self.assertTrue(all(x["spec"] == ml.TEMPLATE and x["edit"] is None for x in f))


class FakeLedger:
    def __init__(self, root):
        self.n = 0

    def record(self, **kw):
        self.n += 1
        self.last = kw
        return mock.Mock(call_id=f"call_{self.n}")


class TestGeneration(unittest.TestCase):
    def _gen(self, replies):
        seq, seen = iter(replies), []

        def fake(client, n, model, messages, temperature, reasoning_effort=None, max_completion_tokens=None):
            seen.append({"messages": messages, "cap": max_completion_tokens, "temperature": temperature, "effort": reasoning_effort})
            return [mock.Mock(message=mock.Mock(content=next(seq)))], 900, 2100, 3000

        with mock.patch("lares.search.generation.LLMCallLedger", FakeLedger):
            g = ms.EditGenerator(None, CFG, tempfile.mkdtemp())
        return g, seen, mock.patch("lares.core.policy_generation._call_llm", fake)

    def test_validator_is_shared_and_strict(self):
        sig = signals()
        e = {"component": "boost_end", "value": "event", "rationale": "x"}
        self.assertEqual(ms.validate_edit(e, PARENT, [], sig), {"component": "boost_end", "value": "event"})
        with self.assertRaises(ml.EditError):
            ms.validate_edit(e, PARENT, [{"component": "boost_end", "value": "event"}], sig)
        with self.assertRaises(ml.EditError):
            ms.validate_edit({"component": "boost_end", "value": "timed"}, PARENT, [], sig)
        with mock.patch.object(ml, "replay_identity", return_value={"passes": False}):
            with self.assertRaises(ml.EditError):
                ms.validate_edit(e, PARENT, [], sig)

    def test_one_call_then_one_linked_correction(self):
        sig = signals()
        g, seen, patch = self._gen(['{"component": "boost_end", "value": "event", "rationale": "keep boosting"}'])
        with patch:
            out = g.propose(PARENT, {"successes": 300}, [], [], sig, "t")
        self.assertEqual((out["edit"], len(out["calls"]), out["usage"][0]["total"]), ({"component": "boost_end", "value": "event"}, 1, 3000))
        self.assertEqual((seen[0]["cap"], seen[0]["temperature"], seen[0]["effort"]), (32000, 1.0, "high"))
        g, seen, patch = self._gen(["no json here", '{"component": "target", "value": "hold"}'])
        with patch:
            out = g.propose(PARENT, {}, [], [], sig, "t")
        self.assertEqual((out["edit"]["value"], len(out["calls"]), len(out["errors"])), ("hold", 2, 1))
        self.assertIn("not admissible", seen[1]["messages"][-1]["content"])
        g, seen, patch = self._gen(['{"component": "target", "value": "hold"}', '{"component": "target", "value": "hold"}'])
        with patch:                                               # a duplicate of the other slot, twice: a parent copy, no third call
            out = g.propose(PARENT, {}, [], [{"component": "target", "value": "hold"}], sig, "t")
        self.assertEqual((out["edit"], len(out["calls"])), (None, 2))
        self.assertIn("DIFFERENT", seen[0]["messages"][1]["content"])

    def test_prompt_carries_what_the_protocol_declares(self):
        tried = [{"edit": {"component": "attempts", "value": 2}, "outcome": "300 of 512"}]
        p = ms.edit_prompt(CFG, PARENT, {"successes": 300, "labels": {"persistent_slow": 90}}, tried)
        for needle in ("MODULE LANGUAGE", "PRIOR FINDINGS", "7.2%", "persistent_slow", '"attempts"', "ADMISSIBLE EDITS", "threshold ="):
            self.assertIn(needle, p)
        self.assertEqual(p.count('{"component"'), 1 + len(ml.valid_edits(PARENT["spec"])) + ml.LANGUAGE_TEXT.count('{"component"'))
        self.assertLess(len(ms.SYSTEM_PROMPT) + len(p), CFG["max_input_chars"])
        self.assertNotIn("def ", p)                               # no code

    def test_grammar_edits_are_valid_distinct_and_reproducible(self):
        sig = signals()
        a = ms.grammar_edits(CFG, 2, 0, PARENT, [], sig)
        self.assertEqual(len(a), 2)
        self.assertNotEqual(ml.edit_key(a[0]["edit"]), ml.edit_key(a[1]["edit"]))
        self.assertEqual([x["edit"] for x in a], [x["edit"] for x in ms.grammar_edits(CFG, 2, 0, PARENT, [], sig)])
        self.assertNotEqual([x["edit"] for x in a], [x["edit"] for x in ms.grammar_edits(CFG, 2, 1, PARENT, [], sig)])
        again = ms.grammar_edits(CFG, 2, 0, PARENT, [{"edit": a[0]["edit"]}, {"edit": a[1]["edit"]}], sig)
        self.assertFalse({ml.edit_key(x["edit"]) for x in again} & {ml.edit_key(x["edit"]) for x in a})


class TestInference(unittest.TestCase):
    def _rows(self, gap):
        rng = np.random.default_rng(0)
        prob = {"r106": 0.55, "d072": 0.43, "r118": 0.76, "r121": 0.90}
        fin = {}
        for b in range(5):
            for arm, q in (("fixed", 0.66), ("grammar", 0.66 + 0.01 * b), ("llm", 0.66 + 0.01 * b + gap)):
                prob[f"b{b}_{arm}"] = q
                fin[f"b{b}_{arm}"] = ml.template_record()
        rows = {}
        for m in range(12):
            u = rng.random(128)
            rows[f"s{m}"] = {p: [{"task_hash": f"h{m}_{i}", "sustained": bool(u[i] < q), "completed": True, "tail_hit_fraction": float(u[i] < q),
                                  "length": 359, "final_center_dist": 0.01} for i in range(128)] for p, q in prob.items()}
        return rows, fin

    def test_blocks_are_the_units(self):
        rows, fin = self._rows(0.10)
        out = ms.analyse(rows, CFG, fin)
        self.assertEqual((out["primary"]["blocks"], out["primary"]["verdict"]), (5, "supported"))
        self.assertAlmostEqual(out["paired_success_difference"], 0.10, delta=0.02)
        self.assertEqual(len(out["secondary"]) + len(out["qualification_against_joint"]), 20)
        self.assertFalse(any(v["qualifies"] for v in out["qualification_against_joint"].values()))
        self.assertEqual(out["policies"]["b0_llm"]["cases"], 1536)
        self.assertEqual(len(out["arms"]["llm"]["success"]), 5)
        json.dumps(out)
        self.assertEqual(ms.analyse(*self._rows(0.0)[:1], CFG, fin)["primary"]["verdict"], "refuted")


if __name__ == "__main__":
    unittest.main()
