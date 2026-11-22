"""R-124: the consolidated ladder's assets, controllers and contrasts (no Isaac)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lares.envs.isaac_pen_spin import ladder_confirmation as lc  # noqa: E402
from lares.envs.isaac_pen_spin import recovery_module as rm  # noqa: E402
from lares.utils.cli import load_yaml  # noqa: E402
from tests.test_recovery_module import Incumbent, ctx, obs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
CFG = load_yaml(os.path.join(ROOT, "config", "shadowhand_spin_stages.yaml"))["research_protocols"][lc.STUDY]


class TestLadder(unittest.TestCase):
    def test_assets_are_the_frozen_ones(self):
        rec = lc.asset_record(CFG)
        self.assertEqual(set(rec), set(lc.POLICIES))
        self.assertEqual((len(lc.module_x(CFG, "r116")), len(lc.module_x(CFG, "r118")), len(lc.module_x(CFG, "r121"))), (11, 11, 43))
        with self.assertRaises(SystemExit):
            lc.module_x({**CFG, "assets": {**CFG["assets"], "r118": {**CFG["assets"]["r118"], "sha256": "0" * 64}}}, "r118")

    def test_each_module_asset_is_read_in_its_own_ranges(self):
        p116 = rm.to_params(lc.module_x(CFG, "r116"))                       # original ranges
        p118 = rm.to_params(lc.module_x(CFG, "r118"))
        p121 = rm.to_params(lc.module_x(CFG, "r121"), *rm.bounds(CFG))      # widened ranges
        self.assertAlmostEqual(p116[0], 2.64, places=2)
        self.assertAlmostEqual(p118[0], 4.0, places=2)
        self.assertAlmostEqual(p121[0], 4.18, places=2)
        self.assertAlmostEqual(p121[3], 6.34, places=2)                     # below the original 10-step bound
        c = rm.build(CFG, "r118", lc.module_x(CFG, "r118"), inner=Incumbent(), original_bounds=True)
        c.reset(ctx(2))
        c(obs(2, [1.0, 1.0]), 0, None)
        self.assertAlmostEqual(float(c.threshold[0]), 4.0, places=4)

    def test_manifests_registry_and_family(self):
        self.assertEqual(lc.build_manifests(CFG), [])
        self.assertEqual(len(CFG["confirmation_seeds"]), 24)
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import run_shadowhand_spin_stages as rs

        self.assertEqual(rs.PEN_SPIN_STUDIES[lc.STUDY], "ladder_confirmation")
        self.assertEqual(len(lc.FAMILY), 7)
        with self.assertRaises(ValueError):
            lc.make_controller("r999", CFG)

    def test_contrasts_and_labels(self):
        rng = np.random.default_rng(0)
        rows = {}
        for m in range(24):
            level, prev = {}, rng.random(128) < 0.35
            for p, add in zip(lc.POLICIES, (0.0, 0.08, 0.15, 0.2, 0.25, 0.2)):
                prev = prev | (rng.random(128) < add)
                level[p] = [{"task_hash": f"h{m}_{i}", "sustained": bool(v), "completed": True, "tail_hit_fraction": float(v), "length": 359,
                             "final_center_dist": 0.02} for i, v in enumerate(prev)]
            rows[f"s{m}"] = level
        c = lc.contrasts(rows, CFG)
        self.assertEqual(set(c), {f"{a}_minus_{b}" for a, b in lc.FAMILY})
        self.assertTrue(all(v["qualifies"] for v in c.values()))
        self.assertTrue(c[lc.PRIMARY]["pooled"]["prediction"]["supported"])                 # the +0.30 threshold applies to the primary only
        self.assertEqual(c["r106_minus_d072"]["pooled"]["prediction"]["threshold"], 0.05)
        self.assertEqual(sum(lc.label_table(rows)["r121"].values()), 24 * 128)


if __name__ == "__main__":
    unittest.main()
