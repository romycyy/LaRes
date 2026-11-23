"""R-133: the paired plan, the mixed-row allocation, endpoint reconstruction under a condition and the inference (no Isaac)."""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lares.envs.isaac_pen_spin import recovery_joint as rj  # noqa: E402
from lares.envs.isaac_pen_spin import recovery_module as rm  # noqa: E402
from lares.envs.isaac_pen_spin import reset_mixture as mx  # noqa: E402
from lares.utils.cli import load_yaml  # noqa: E402
from tests.test_recovery_module import Incumbent  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
ALL = load_yaml(os.path.join(ROOT, "config", "shadowhand_spin_stages.yaml"))["research_protocols"]
CFG = ALL[mx.STUDY]


class TestPlan(unittest.TestCase):
    def test_blocks_seeds_and_registry(self):
        b = mx.blocks(CFG)
        self.assertEqual(len(b), 5)
        used = [v for x in b for v in x.values()] + mx.seeds_for(CFG, "nominal") + mx.seeds_for(CFG, "expanded")
        self.assertEqual(sorted(used), list(range(5344, 5378)))
        self.assertIn(CFG["smoke_seed"], range(5344, 5378))
        self.assertEqual(len(mx.policies(CFG)), 11)
        self.assertEqual(mx.build_manifests(CFG), [])
        self.assertEqual(mx.placement_provenance(CFG)["latent_placements"], 5 * 36 * 512 + 512)
        with self.assertRaises(SystemExit):
            mx.blocks({**CFG, "placement_seeds": CFG["optimizer_seeds"]})
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import run_shadowhand_spin_stages as rs

        self.assertEqual(rs.PEN_SPIN_STUDIES[mx.STUDY], "reset_mixture")

    def test_arms_differ_only_in_the_reset_allocation(self):
        block = mx.blocks(CFG)[2]
        a, b = mx.train_request(CFG, block, "nominal"), mx.train_request(CFG, block, "mixed")
        self.assertEqual((a["reset_mix"], b["reset_mix"]), (None, {"rows": 256, "scale": 1.5}))
        self.assertEqual({k: v for k, v in a.items() if k != "reset_mix"}, {k: v for k, v in b.items() if k != "reset_mix"})
        self.assertEqual((a["placement_seed"], a["cfg_overrides"]["training_seed"]), (5351, 5346))
        self.assertEqual(a["cfg_overrides"]["ars"], {"iterations": 36, "directions": 4, "nu": 0.1, "alpha": 0.05})
        hist = ALL["recovery_joint_matched"]                       # the JOINT stage's own semantics
        self.assertEqual(a["coord_scale"], [float(v) for v in rj.coord_scale(hist, "joint")])
        self.assertEqual((a["direction_dim"], a["original_bounds"], len(a["init_x"])), (43, False, 43))
        np.testing.assert_array_equal(a["init_x"], mx.joint_x(CFG))
        with self.assertRaises(SystemExit):
            mx.joint_x({**CFG, "joint_sha256": "0" * 64})

    def test_mixed_rows(self):
        v = np.arange(512 * 3, dtype=float).reshape(512, 3) + 1.0
        out = rm.mixed_rows(v, {"rows": 256, "scale": 1.5})
        np.testing.assert_array_equal(out[:256], v[:256])
        np.testing.assert_array_equal(out[256:], 1.5 * v[256:])
        self.assertIs(rm.mixed_rows(v, None), v)
        self.assertEqual(v[300, 0], 901.0)                         # the input is not modified


class TestCells(unittest.TestCase):
    def test_endpoints_run_under_their_condition(self):
        with tempfile.TemporaryDirectory() as out:
            x = np.clip(mx.joint_x(CFG) + 0.01, 0, 1)
            rm._dump(os.path.join(out, "finalists.json"), {"finalists": {"b1_mixed": rm.x_record(x, *rm.bounds(CFG))}})
            real = rm.build
            with mock.patch.object(rm, "build", lambda cfg, name, xx, **kw: real(cfg, name, xx, inner=Incumbent(), **kw)):
                c = mx.make_controller("b1_mixed@expanded", CFG, out)
                j = mx.make_controller("joint@nominal", CFG, out)
            self.assertEqual((c.env_condition, j.env_condition), ({"reset_scale": 1.5}, {}))
            self.assertEqual(c.params["module"]["sha256"], rm.x_record(x, *rm.bounds(CFG))["sha256"])
            self.assertEqual(j.params["module"]["sha256"], CFG["joint_sha256"])
            self.assertTrue(callable(c.extra_trace))
            for bad in ("b1_mixed", "b9_mixed@nominal", "joint@wide"):
                with self.assertRaises(ValueError):
                    mx.make_controller(bad, CFG, out)


class TestInference(unittest.TestCase):
    def _rows(self, gain):
        rng = np.random.default_rng(0)
        prob = {"nominal": {"joint": 0.79}, "expanded": {"joint": 0.50}}
        for k in range(5):
            prob["nominal"].update({f"b{k}_nominal": 0.79, f"b{k}_mixed": 0.78})
            prob["expanded"].update({f"b{k}_nominal": 0.48 + 0.005 * k, f"b{k}_mixed": 0.48 + 0.005 * k + gain})
        rows = {}
        for cond, pr in prob.items():
            rows[cond] = {}
            for m in range(12):
                u = rng.random(128)
                rows[cond][f"s{m}"] = {p: [{"task_hash": f"{cond}{m}_{i}", "sustained": bool(u[i] < q), "completed": True,
                                            "tail_hit_fraction": float(u[i] < q), "length": 359, "final_center_dist": 0.01} for i in range(128)]
                                       for p, q in pr.items()}
        return rows

    def test_primary_and_usefulness(self):
        out = mx.analyse(self._rows(0.15), CFG)
        self.assertEqual((out["primary"]["verdict"], out["primary"]["df"], out["primary"]["contains_zero"]), ("supported", 4, False))
        self.assertAlmostEqual(out["paired_success_difference"], 0.15, delta=0.02)
        self.assertEqual(out["useful_mixed_endpoints"], 5)
        self.assertFalse(out["usefulness"]["b0_nominal"]["useful"])
        self.assertEqual(out["policies"]["expanded"]["joint"]["cases"], 1536)
        self.assertEqual(out["arms"]["expanded"]["mixed"]["n"], 5)
        json.dumps(out)
        none = mx.analyse(self._rows(0.0), CFG)
        self.assertEqual((none["primary"]["verdict"], none["useful_mixed_endpoints"]), ("refuted", 0))


if __name__ == "__main__":
    unittest.main()
