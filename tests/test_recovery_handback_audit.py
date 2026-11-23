"""R-130: the reconstructed trigger, the hand-back boundaries and the reachable-failure screen (no Isaac)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lares.envs.isaac_pen_spin import recovery_handback_audit as ha  # noqa: E402
from lares.utils.cli import load_yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
CFG = load_yaml(os.path.join(ROOT, "config", "shadowhand_spin_stages.yaml"))["research_protocols"][ha.STUDY]
TL = ha.timeline(CFG)


def case(slow_until=None, lose_at=None, drop_at=None, off_centre_from=None, recover_at=None):
    """One 359-step case: rolling at 8 rad/s, optionally slow up to a step, losing rolling from a step, dropping."""
    spin = np.full(360, 8.0)
    centre = np.full(360, 0.02)
    if slow_until is not None:
        spin[:slow_until] = 1.0
    if lose_at is not None:
        spin[lose_at:recover_at] = 1.0
    if off_centre_from is not None:
        centre[off_centre_from:] = 0.2
    L = 359 if drop_at is None else drop_at
    pred = (spin > 4.0) & (centre < 0.08)
    pred[L:] = False
    ok = bool(L == 359 and pred[299:359].all())
    row = {"task_hash": "h", "sustained": ok, "completed": L == 359, "length": L, "final_center_dist": float(centre[L - 1]),
           "tail_hit_fraction": float(pred[max(0, L - 60):L].mean())}
    return spin, centre, pred, L, row


def cell(*cs):
    return ({"spin_rate": np.stack([c[0] for c in cs], 1), "center_dist": np.stack([c[1] for c in cs], 1),
             "pred": np.stack([c[2] for c in cs], 1), "length": np.array([c[3] for c in cs])}, [c[4] for c in cs])


class TestAudit(unittest.TestCase):
    def test_timeline_is_the_frozen_modules(self):
        self.assertEqual((TL["decision_step"], TL["support_steps"], TL["boost_steps"], TL["rate_window"]), (149, 35, 60, 30))
        self.assertEqual((TL["support_end"], TL["boost_end"]), (183, 243))
        self.assertAlmostEqual(TL["threshold"], 4.18, places=2)
        with self.assertRaises(SystemExit):
            ha.timeline({**CFG, "module_sha256": "0" * 64})

    def test_trigger_uses_the_modules_window(self):
        b1, b2 = TL["support_end"], TL["boost_end"]
        trace, rows = cell(case(),                                   # rolling at the decision: no trigger
                           case(slow_until=b1 - 20),                 # slow through the decision: trigger
                           case(slow_until=140),                     # 21 slow + 9 fast observations: mean 3.1, trigger
                           case(slow_until=125),                     # 6 slow + 24 fast: mean 6.6, no trigger
                           case(slow_until=200, drop_at=100),        # dead before the decision
                           case(slow_until=200, off_centre_from=100))
        trig = ha.reconstruct_trigger(trace["spin_rate"], trace["center_dist"], trace["length"], TL)
        self.assertEqual([bool(v) for v in trig], [False, True, True, False, False, False])
        edge = cell(case(slow_until=131))[0]                         # 12 slow of 30: mean 5.2 now, 11 slow one step later
        edge["spin_rate"][119:131, 0] = 0.0
        edge["spin_rate"][131:149, 0] = 6.9                          # mean 4.14 at the decision, 4.37 one step later
        self.assertTrue(ha.reconstruct_trigger(edge["spin_rate"], edge["center_dist"], edge["length"], TL)[0])
        self.assertFalse(ha.reconstruct_trigger(edge["spin_rate"], edge["center_dist"], edge["length"], TL, shift=1)[0])

    def test_categories(self):
        b1, b2 = TL["support_end"], TL["boost_end"]
        slow = b1 - 15                                               # slow at the decision, rolling again 15 steps before support ends
        trace, rows = cell(
            case(slow_until=slow),                                   # triggered success
            case(lose_at=250),                                       # never triggered, fails later
            case(slow_until=slow, lose_at=b1 + 10),                  # healthy at support end, lost 10 steps later
            case(slow_until=slow, lose_at=b2 + 25, recover_at=b2 + 40),   # healthy at boost end, lost 25 later, recovers, fails the window
            case(slow_until=slow, lose_at=b2 + 80),                  # healthy, lost well outside the window
            case(slow_until=359),                                    # never re-captured
            case(slow_until=359, drop_at=170),                       # dropped during the blend
            case(slow_until=b2 + 20, lose_at=b2 + 60),               # rolling only after both boundaries, then lost
        )
        rows[3]["sustained"] = False
        got = ha.classify(trace, rows, TL, CFG)
        self.assertEqual([g["category"] for g in got],
                         ["success", "untriggered_failure", "reachable", "reachable", "later_loss", "never_recaptured",
                          "dropped_during_blend", "pre_boundary_loss"])
        self.assertEqual((got[2]["boundary"], got[2]["loss_delay"], got[3]["boundary"], got[3]["loss_delay"]), ("support_end", 10, "boost_end", 25))
        self.assertEqual((got[2]["rolling_recovers"], got[3]["rolling_recovers"]), (False, True))
        self.assertEqual(ha.classify(trace, rows, TL, CFG, window=15)[3]["category"], "later_loss")
        self.assertEqual(ha.classify(trace, rows, TL, CFG, window=60)[4]["category"], "later_loss")

    def test_audit_gate_and_registry(self):
        b1 = TL["support_end"]
        good, lost = case(slow_until=b1 - 15), case(slow_until=b1 - 15, lose_at=b1 + 10)

        def cells(k):
            return [(f"s{m}", *cell(*([lost] * k + [good] * (20 - k))), 12, "sha") for m in range(6)]

        a = ha.audit(cells(4), TL, CFG)
        self.assertEqual((a["cases"], a["reachable"]["count"], a["categories"]["success"]["count"]), (120, 24, 96))
        self.assertEqual(a["triggered"], {"reconstructed": 120, "successes": 96, "failures": 24, "recorded_all_episodes": 72,
                                          "cells_exceeding_recorded_upper_bound": 6})
        self.assertEqual(a["reachable_by_boundary"], {"support_end": 24, "boost_end": 0})
        self.assertEqual(sum(a["reachable_by_final_label"].values()), 24)
        g = ha.gate({"r124_joint": a, "r126_joint": a}, CFG)
        self.assertEqual((g["verdict"], g["simulation_eligible"]), ("supported", True))
        low = ha.audit(cells(0), TL, CFG)
        g = ha.gate({"r124_joint": a, "r126_joint": low}, CFG)
        self.assertEqual((g["verdict"], g["simulation_eligible"]), ("refuted", False))
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import run_shadowhand_spin_stages as rs

        self.assertEqual(rs.PEN_SPIN_STUDIES[ha.STUDY], "recovery_handback_audit")
        with self.assertRaises(SystemExit):
            ha.make_controller("x", CFG)


if __name__ == "__main__":
    unittest.main()
