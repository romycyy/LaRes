"""R-128: lineage seeds, the archived schedules and inheritance, endpoint reconstruction, the rerun rule and the
two-level inference (no Isaac)."""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lares.envs.isaac_pen_spin import recovery_joint as rj  # noqa: E402
from lares.envs.isaac_pen_spin import recovery_module as rm  # noqa: E402
from lares.envs.isaac_pen_spin import recovery_retry as rr  # noqa: E402
from lares.envs.isaac_pen_spin import symbolic_reward as sr  # noqa: E402
from lares.envs.isaac_pen_spin import training_replication as tr  # noqa: E402
from lares.utils.cli import load_yaml  # noqa: E402
from tests.test_recovery_module import Incumbent, ctx  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
ALL = load_yaml(os.path.join(ROOT, "config", "shadowhand_spin_stages.yaml"))["research_protocols"]
CFG = ALL[tr.STUDY]


def fake_rows(rng, success: dict, manifests=24, n=128):
    """``rows[manifest][policy]`` with the given success probabilities (shared case difficulty within a manifest)."""
    rows = {}
    for m in range(manifests):
        u = rng.random(n)
        rows[f"s{m}"] = {p: [{"task_hash": f"h{m}_{i}", "sustained": bool(u[i] < q), "completed": True, "tail_hit_fraction": float(u[i] < q),
                              "length": 359, "final_center_dist": 0.01} for i in range(n)] for p, q in success.items()}
    return rows


class TestPlan(unittest.TestCase):
    def test_lineages_policies_and_registry(self):
        lins = tr.lineages(CFG)
        self.assertEqual(len(lins), 5)
        self.assertEqual(sorted(s for lin in lins for s in lin.values()), list(range(5200, 5220)))
        self.assertEqual(len(tr.policies(CFG)), 22)
        self.assertEqual(len(CFG["confirmation_seeds"]), 24)
        with self.assertRaises(SystemExit):
            tr.lineages({**CFG, "lineages": [[1, 2, 3, 4], [4, 5, 6, 7]]})
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import run_shadowhand_spin_stages as rs

        self.assertEqual(rs.PEN_SPIN_STUDIES[tr.STUDY], "training_replication")
        self.assertEqual(tr.build_manifests(CFG), [])

    def test_streams_are_disjoint(self):
        p = tr.placement_provenance(CFG)
        self.assertEqual(p["training_cases"], 5 * 72 * 512)
        self.assertEqual(len({v["sha256"] for v in p["streams"].values()}), 21)

    def test_schedules_are_the_archived_ones(self):
        a = tr.stage_overrides(CFG, "A", 5200)
        old = ALL["symbolic_exact_success_ars"]
        self.assertEqual({**a["training"], "seed": 0}, {**old["training"], "seed": 0})
        self.assertEqual(a["ars"], old["ars"])
        for stage, key in (("B", "symbolic_recovery_ars"), ("C", "recovery_retry_vs_continue"), ("D", "recovery_joint_matched")):
            o = tr.stage_overrides(CFG, stage, 7, "p", "h")
            self.assertEqual(o["ars"], ALL[key]["ars"])
            self.assertEqual((o["training_seed"], o["root_finalist"], o["root_theta_sha256"]), (7, "p", "h"))
            self.assertEqual(CFG["cases_per_slot"], ALL[key]["cases_per_slot"])
        self.assertEqual(CFG["module_bounds"], ALL["recovery_joint_matched"]["module_bounds"])
        self.assertEqual(tr.scored_per_update(CFG), 4096)
        self.assertEqual(sum(tr.updates(CFG, s) for s in tr.STAGES), 72)

    def test_inheritance_matches_the_historical_lineage(self):
        hist = ALL["recovery_joint_matched"]
        theta = rm._load(hist["root_finalist"])["theta"]
        x118 = rj.parent_x(hist)
        np.testing.assert_array_equal(tr.joint_initial(CFG, x118, theta), rj.initial_x(hist, "joint"))
        np.testing.assert_array_equal(tr.joint_scale(CFG), rj.coord_scale(hist, "joint"))
        root = {"path": "r.json", "theta": theta, "theta_sha256": "h"}
        b = tr.stage_request(CFG, "B", 1, None, root)
        self.assertTrue(b["original_bounds"] and "init_x" not in b)          # the child starts from the scripted point
        x116 = rr.parent_x(ALL["recovery_retry_vs_continue"])
        c = tr.stage_request(CFG, "C", 1, {"x": list(x116)}, root)
        self.assertEqual((c["direction_dim"], c["original_bounds"], c["init_x"]), (rm.DIM_RETRY, True, list(x116)))
        d = tr.stage_request(CFG, "D", 1, {"x": list(x118)}, root)
        self.assertEqual((d["direction_dim"], d["original_bounds"], len(d["init_x"])), (rm.DIM_JOINT, False, 43))
        self.assertNotIn("root_finalist", tr.stage_request(CFG, "A", 1, None, None)["cfg_overrides"])

    def test_divergence(self):
        self.assertIsNone(tr.divergent("A", {"final_theta": [0.1] * 32}, 0.10))
        self.assertIsNotNone(tr.divergent("A", {"final_theta": [0.2] * 32}, 0.10))
        self.assertIsNotNone(tr.divergent("B", {"final_x": [float("nan")] * 11}, 0.10))
        self.assertIsNone(tr.divergent("D", {"final_x": [0.0, 1.0] + [0.5] * 41}, 0.10))


class TestEndpoints(unittest.TestCase):
    def test_a_lineage_is_rebuilt_on_its_own_root(self):
        theta = sr.theta_record(0.03 * np.ones(32))
        with tempfile.TemporaryDirectory() as out:
            os.makedirs(tr.lineage_dir(out, 2))
            a = {"theta": theta, "theta_sha256": sr.theta_sha256(theta)}
            xd = tr.joint_initial(CFG, rm.SCRIPT_X, theta)
            ends = {"A": a, "B": rm.x_record(rm.SCRIPT_X), "D": rm.x_record(xd, *rm.bounds(CFG))}
            rm._dump(os.path.join(tr.lineage_dir(out, 2), "endpoints.json"), {"endpoints": ends})
            rm._dump(tr.root_path(out, 2), a)
            real = rm.build
            calls = []

            def build(cfg, name, x, **kw):
                calls.append((cfg["root_finalist"], cfg["root_theta_sha256"], kw.get("original_bounds", False)))
                return real(cfg, name, x, inner=Incumbent(), **kw)

            with mock.patch.object(rm, "build", build):
                b = tr.make_controller("L2_B", CFG, out)
                d = tr.make_controller("L2_D", CFG, out)
            self.assertEqual(calls, [(tr.root_path(out, 2), a["theta_sha256"], True), (tr.root_path(out, 2), a["theta_sha256"], False)])
            self.assertEqual(b.params["root_theta_sha256"], a["theta_sha256"])
            self.assertNotEqual(a["theta_sha256"], CFG["root_theta_sha256"])
            self.assertEqual(b.params["module"]["sha256"], ends["B"]["sha256"])
            self.assertEqual(d.params["module"]["sha256"], ends["D"]["sha256"])
            b.reset(ctx(2))
            self.assertTrue(torch.allclose(b.root.theta.reshape(-1)[:32].double(), torch.tensor(theta, dtype=torch.float64)))
            with self.assertRaises(SystemExit):
                tr.make_controller("L2_C", CFG, out)                          # the lineage froze no stage C
            for bad in ("L9", "x", "L2_E"):
                with self.assertRaises(ValueError):
                    tr.make_controller(bad, CFG, out)
            ends["B"]["sha256"] = "0" * 64
            rm._dump(os.path.join(tr.lineage_dir(out, 2), "endpoints.json"), {"endpoints": ends})
            with self.assertRaises(SystemExit):
                tr.make_controller("L2_B", CFG, out)

    def test_endpoint_records(self):
        self.assertEqual(tr.endpoint_record(CFG, "A", {"final_theta": [0.0] * 32, "final_theta_sha256": "h"}), {"theta": [0.0] * 32, "theta_sha256": "h"})
        x = [0.5] * 43
        self.assertEqual(tr.endpoint_record(CFG, "D", {"final_x": x})["sha256"], rm.x_record(x, *rm.bounds(CFG))["sha256"])
        self.assertAlmostEqual(tr.endpoint_record(CFG, "D", {"final_x": x})["params"]["threshold"], 3.5)
        self.assertAlmostEqual(tr.endpoint_record(CFG, "C", {"final_x": [0.5] * 11})["params"]["threshold"], 2.5)


class TestRerunRule(unittest.TestCase):
    def _run(self, exits):
        out = tempfile.mkdtemp()
        ledger = tr.Ledger(os.path.join(out, "accounting.json"), 10 ** 6, 60)
        req = {"response_path": os.path.join(out, "train_B.json")}
        seq = iter(exits)

        def fake(cmd, stdout=None, **kw):
            code = next(seq)
            stdout.write('update {"starts_so_far": 8200}\n')
            stdout.flush()
            if code == 0:
                rm._dump(req["response_path"], {"starts": 98000, "complete": True})
            return mock.Mock(returncode=code)

        with mock.patch.object(tr.subprocess, "run", fake):
            return tr.run_child("B", req, os.path.join(out, "train_B"), ledger, "L0_B"), ledger

    def test_infrastructure_death_is_rerun_and_charged(self):
        res, ledger = self._run([134, 0])
        self.assertEqual(res["starts"], 98000)
        self.assertEqual(ledger.data["starts"], {"L0_B:died": 8200, "L0_B": 98000})
        self.assertEqual([a["ok"] for a in ledger.data["attempts"]], [False, True])

    def test_three_deaths_end_the_stage(self):
        res, ledger = self._run([134, 134, 134])
        self.assertIsNone(res)
        self.assertEqual(len(ledger.data["attempts"]), 3)
        self.assertEqual(ledger.charged, 3 * 8200)

    def test_ledger_persists_and_refuses_past_its_ceilings(self):
        with tempfile.TemporaryDirectory() as out:
            p = os.path.join(out, "a.json")
            a = tr.Ledger(p, 1000, 1)
            a.charge("x", 600, 10.0)
            b = tr.Ledger(p, 1000, 1)
            self.assertEqual(b.charged, 600)
            self.assertTrue(b.room(400))
            self.assertFalse(b.room(401))
            b.charge("y", 0, 60.0)
            self.assertFalse(b.room(1))


class TestInference(unittest.TestCase):
    def test_two_level_components(self):
        rng = np.random.default_rng(1)
        rows = np.array([0.26, 0.34, 0.38, 0.36, 0.32])
        pure = tr.two_level(np.tile(rows[:, None], (1, 24)), 0.30)
        self.assertAlmostEqual(pure["se"], rows.std(ddof=1) / np.sqrt(5))
        self.assertAlmostEqual(pure["variance_components"]["manifest"], 0.0)
        self.assertEqual(pure["verdict"], "inconclusive")
        self.assertAlmostEqual(pure["low"], pure["lineage_t_interval"][0])
        y = 0.36 + 0.002 * rng.standard_normal((5, 1)) + 0.03 * rng.standard_normal((1, 24)) + 0.01 * rng.standard_normal((5, 24))
        t = tr.two_level(y, 0.30)
        self.assertEqual(t["verdict"], "supported")
        self.assertAlmostEqual(np.sqrt(t["variance_components"]["manifest"]), 0.03, delta=0.012)
        self.assertEqual(tr.two_level(y - 0.2, 0.30)["verdict"], "refuted")
        self.assertEqual(tr.two_level(y[:1], 0.30)["verdict"], "inconclusive")

    def test_analysis_with_every_lineage(self):
        rng = np.random.default_rng(0)
        prob = {tr.REFERENCE: 0.43, tr.HISTORICAL: 0.80}
        for k in range(5):
            prob.update({f"L{k}_A": 0.55, f"L{k}_B": 0.64, f"L{k}_C": 0.76, f"L{k}_D": 0.79 + 0.01 * k})
        ends = {k: {s: {} for s in tr.STAGES} for k in range(5)}
        out = tr.analyse(fake_rows(rng, prob), CFG, ends)
        self.assertEqual(out["primary"]["verdict"], "supported")
        self.assertAlmostEqual(out["paired_success_difference"], 0.38, delta=0.02)
        self.assertEqual(len(out["qualification"]), 25)
        self.assertTrue(out["qualified"]["L0_A_minus_d072"] and out["qualified"]["L4_D_minus_d072"])
        self.assertEqual(out["milestones"]["D"]["success"]["n"], 5)
        self.assertEqual(sorted(out["replicated_joint_minus_historical_joint"]), [f"L{k}_D" for k in range(5)])
        self.assertEqual(out["policies"]["d072"]["cases"], 24 * 128)
        self.assertEqual(sum(out["policies"]["L0_D"]["labels"].values()), 24 * 128)
        json.dumps(out)

    def test_a_missing_endpoint_enters_as_bounds(self):
        rng = np.random.default_rng(0)
        prob = {tr.REFERENCE: 0.43, tr.HISTORICAL: 0.80}
        ends = {}
        for k in range(5):
            stages = tr.STAGES if k else ("A", "B")              # lineage 0 ended in stage C
            prob.update({f"L{k}_{s}": q for s, q in zip(tr.STAGES, (0.55, 0.64, 0.76, 0.80)) if s in stages})
            ends[k] = {s: {} for s in stages}
        out = tr.analyse(fake_rows(rng, prob), CFG, ends)
        p = out["primary"]
        self.assertEqual((out["missing_lineages"], p["missing_lineages"]), ([0], [0]))
        self.assertLess(p["missing_as_all_failures"]["mean"], p["missing_as_all_successes"]["mean"])
        self.assertEqual(p["verdict"], "inconclusive")
        self.assertEqual(len(out["qualification"]), 4 * 4 + 2 + 4)
        self.assertEqual(out["milestones"]["D"]["lineages"], [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
