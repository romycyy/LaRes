"""R-104: the archive-only alignment audit's scores, concordance, bootstrap and decision rule (no Isaac)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lares.envs.isaac_pen_spin import symbolic_reward_audit as au  # noqa: E402
from lares.utils.cli import load_yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = load_yaml(os.path.join(ROOT, "config", "shadowhand_spin_stages.yaml"))["research_protocols"][au.STUDY]


def cell(spin, lengths, completed, sustained):
    """A trace of constant-per-case centre distance 0 and per-step axial rate ``spin`` (T, n)."""
    spin = np.asarray(spin, dtype=float)
    rows = [{"completed": bool(c), "sustained": bool(s), "task_hash": f"h{i}"} for i, (c, s) in enumerate(zip(completed, sustained))]
    return {"center_dist": np.zeros_like(spin), "spin_rate": spin, "length": np.asarray(lengths)}, rows


class TestScores(unittest.TestCase):
    def test_dense_and_tail(self):
        spin = np.full((360, 3), 6.0)
        spin[:, 1] = 3.0                                     # half the cap all episode
        spin[330:, 2] = 0.0                                  # fast, then stalls in the tail
        s = au.episode_scores(*cell(spin, [360, 360, 360], [1, 1, 1], [1, 0, 0]), CFG)
        np.testing.assert_allclose(s["dense"], [360.0, 180.0, 330.0])
        np.testing.assert_allclose(s["tail"], [360.0, 180.0, 0.0])

    def test_fall_and_unreached_steps(self):
        spin = np.full((360, 1), -6.0)                       # unsigned
        trace, rows = cell(spin, [100], [0], [0])
        trace["spin_rate"][100:] = np.nan
        s = au.episode_scores(trace, rows, CFG)
        np.testing.assert_allclose(s["dense"], [100.0 - 5.0])
        np.testing.assert_allclose(s["tail"], [-5.0])

    def test_missing_field_is_reported(self):
        trace, rows = cell(np.ones((360, 1)), [360], [1], [1])
        del trace["spin_rate"]
        with self.assertRaises(KeyError):
            au.episode_scores(trace, rows, CFG)


def policy(dense, tail, sustained, length=360):
    n = len(dense)
    return {"dense": np.asarray(dense, float), "tail": np.asarray(tail, float), "sustained": np.asarray(sustained, bool),
            "length": np.full(n, length), "task_hash": [f"h{i}" for i in range(n)]}


class TestConcordance(unittest.TestCase):
    def test_counts_only_discordant_cases_and_half_for_ties(self):
        a = policy([10, 10, 10, 10], [5, 5, 5, 5], [1, 1, 0, 1])
        b = policy([20, 5, 10, 10], [1, 5, 9, 5], [0, 0, 1, 1])        # case 3 is concordant on success: ignored
        t = au.case_tables({"a": a, "b": b}, 0.0)
        np.testing.assert_array_equal(t["den"], [1, 1, 1, 0])
        np.testing.assert_array_equal(t["dense"], [0, 1, 0.5, 0])
        np.testing.assert_array_equal(t["tail"], [1, 0.5, 1, 0])
        c = au.concordance({0: t})
        self.assertAlmostEqual(c["dense"], 0.5)
        self.assertAlmostEqual(c["tail"], 2.5 / 3)
        self.assertAlmostEqual(c["difference"], 2.5 / 3 - 0.5)

    def test_blocks_average_equally_and_pairs_must_match(self):
        a, b = policy([1, 1], [1, 1], [1, 0]), policy([0, 2], [0, 0], [0, 1])
        t0 = au.case_tables({"a": a, "b": b}, 0.0)                      # dense 1.0, tail 0.5 + ... 
        big = {k: np.tile(v, 50) for k, v in au.case_tables({"a": b, "b": b}, 0.0).items()}
        self.assertEqual(float(big["den"].sum()), 0.0)
        c = au.concordance({0: t0, 1: t0})
        self.assertAlmostEqual(c["dense"], 1.0)
        bad = dict(b, task_hash=["x", "y"])
        with self.assertRaises(ValueError):
            au.case_tables({"a": a, "b": bad}, 0.0)

    def test_vulnerable_margins(self):
        a, b = policy([10.0, 10.0], [0, 0], [1, 1]), policy([9.0, 0.0], [0, 0], [0, 0])
        t = au.case_tables({"a": a, "b": b}, 0.01)                      # bound 0.01 * 720 = 7.2
        np.testing.assert_array_equal(t["dense_vulnerable"], [1, 0])

    def test_bootstrap_is_seeded_and_brackets_the_point(self):
        rng = np.random.default_rng(0)
        sa, sb = rng.random(128) < 0.5, rng.random(128) < 0.4
        mk = lambda s: policy(rng.normal(size=128) + 2 * s, rng.normal(size=128) + 4 * s, s)   # noqa: E731
        t = {0: au.case_tables({"a": mk(sa), "b": mk(sb)}, 0.01), 1: au.case_tables({"a": mk(sb), "b": mk(sa)}, 0.01)}
        c1, c2 = au.bootstrap(t, 200, 4771), au.bootstrap(t, 200, 4771)
        self.assertEqual(c1, c2)
        p = au.concordance(t)
        self.assertTrue(c1["difference"][0] <= p["difference"] <= c1["difference"][1])
        self.assertGreater(p["tail"], p["dense"])

    def test_verdict_rule(self):
        self.assertEqual(au.verdict([0.11, 0.2], 0.10), "supported")
        self.assertEqual(au.verdict([-0.05, 0.09], 0.10), "refuted")
        self.assertEqual(au.verdict([0.0, 0.2], 0.10), "inconclusive")

    def test_sizing(self):
        a, b = policy(np.zeros(100), np.zeros(100), [1] * 30 + [0] * 70), policy(np.zeros(100), np.zeros(100), [0] * 20 + [1] * 10 + [0] * 70)
        s = au.sizing({0: {"a": a, "b": b}}, 0.05, 50, 1)["b0:a_vs_b"]
        self.assertAlmostEqual(s["discordance"], 0.2)
        self.assertAlmostEqual(s["pairs_for_80pct_power"], 7.848879734 * (0.2 - 0.0025) / 0.0025, places=3)

    def test_no_controller_and_registered(self):
        with self.assertRaises(SystemExit):
            au.make_controller("x", CFG)
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import run_shadowhand_spin_stages as rs

        self.assertEqual(rs.PEN_SPIN_STUDIES[au.STUDY], "symbolic_reward_audit")


if __name__ == "__main__":
    unittest.main()
