"""R-120: the tail audit's label rule and paired contrast (no Isaac)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lares.envs.isaac_pen_spin import recovery_tail_audit as ta  # noqa: E402
from lares.utils.cli import load_yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = load_yaml(os.path.join(ROOT, "config", "shadowhand_spin_stages.yaml"))["research_protocols"][ta.STUDY]


def row(sustained=False, completed=True, length=359, centre=0.02, hit=0.0):
    return {"sustained": sustained, "completed": completed, "length": length, "final_center_dist": centre, "tail_hit_fraction": hit}


class TestAudit(unittest.TestCase):
    def test_labels_are_exhaustive_and_exclusive(self):
        got = [ta.label(r, CFG) for r in (row(sustained=True, hit=1.0), row(completed=False, length=40), row(completed=False, length=160),
                                           row(centre=0.09), row(hit=0.2), row(hit=0.5), row(hit=0.8))]
        self.assertEqual(got, list(ta.LABELS))

    def test_paired_difference(self):
        a = ["persistent_slow"] * 30 + ["success"] * 70
        b = ["success"] * 20 + ["persistent_slow"] * 10 + ["success"] * 70
        p = ta.paired(a, b, [i % 4 for i in range(100)], "persistent_slow")
        self.assertAlmostEqual(p["difference"], 0.20)
        self.assertEqual((p["a_count"], p["b_count"]), (30, 10))
        self.assertGreater(p["ci95"][0], 0.05)

    def test_registered(self):
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import run_shadowhand_spin_stages as rs

        self.assertEqual(rs.PEN_SPIN_STUDIES[ta.STUDY], "recovery_tail_audit")
        with self.assertRaises(SystemExit):
            ta.make_controller("x", CFG)


if __name__ == "__main__":
    unittest.main()
