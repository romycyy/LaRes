"""R-118: the optional second re-capture attempt and the two-arm study (no Isaac)."""

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lares.envs.isaac_pen_spin import recovery_module as rm  # noqa: E402
from lares.envs.isaac_pen_spin import recovery_retry as rr  # noqa: E402
from lares.utils.cli import load_yaml  # noqa: E402
from tests.test_recovery_module import Incumbent, ctx, obs  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
CFG = load_yaml(os.path.join(ROOT, "config", "shadowhand_spin_stages.yaml"))["research_protocols"][rr.STUDY]


def make(x, n=4):
    c = rm.build(CFG, "t", x, inner=Incumbent())
    c.reset(ctx(n))
    return c


class TestRetry(unittest.TestCase):
    def test_initial_points(self):
        a, b = rr.initial_x(CFG, "continue"), rr.initial_x(CFG, "retry")
        self.assertEqual((len(a), len(b)), (11, 13))
        np.testing.assert_array_equal(b[:11], a)
        p = rm.x_record(b)["params"]
        self.assertEqual((p["retry_gap"], p["retry_depth"]), (60.0, 0.0))
        self.assertAlmostEqual(p["depth_TH"], 0.42, places=1)                      # the learned R-116 thumb depth

    def test_kappa_zero_is_the_parent_bit_for_bit(self):
        parent, retry = make(rr.initial_x(CFG, "continue")), make(rr.initial_x(CFG, "retry"))
        for t in range(330):
            o = obs(4, [1.0, 6.0, 2.0, 1.5])
            self.assertTrue(torch.equal(parent(o, t, None), retry(o, t, None)), t)
        self.assertEqual((retry.triggered, retry.triggered2), (parent.triggered, 0))

    def test_second_attempt_fires_only_after_a_first_and_while_still_slow(self):
        x = rr.initial_x(CFG, "retry")
        x[12] = 1.0
        c = make(x)
        T, T2 = int(c.T[0]), int(c.T2[0])
        self.assertGreaterEqual(T2 - T - round(float(c.n_in[0] + c.n_out[0])), 30)
        sent, root = [], []
        for t in range(330):
            fast_after = 6.0 if t > T + 5 else 1.0                                  # environment 1 recovers after its first attempt
            sent.append(c(obs(4, [1.0, fast_after, 6.0, 1.0]), t, None).clone())
            root.append(c.last_root.clone())
        self.assertEqual(c.elig.tolist(), [True, True, False, True])
        self.assertEqual(c.elig2.tolist(), [True, False, False, True])            # no first attempt or recovered: no retry
        self.assertFalse(torch.equal(sent[T2 + 5][0], root[T2 + 5][0]))
        self.assertTrue(torch.equal(sent[T2 + 5][1], root[T2 + 5][1]))
        self.assertTrue(torch.equal(sent[T2 - 1][0], root[T2 - 1][0]))             # nothing between the two attempts

    def test_no_retry_that_cannot_finish_by_the_deadline(self):
        x = rr.initial_x(CFG, "retry")
        x[12], x[11], x[1] = 1.0, 1.0, 1.0                                         # decision at 180, gap 150
        c = make(x)
        for t in range(359):
            c(obs(4, [1.0] * 4), t, None)
        self.assertEqual((c.triggered, c.triggered2), (4, 0))

    def test_shared_direction_columns(self):
        full = np.random.default_rng(4970 + 2000).standard_normal((4, rm.DIM_RETRY))
        again = np.random.default_rng(4970 + 2000).standard_normal((4, rm.DIM_RETRY))[:, :rm.DIM]
        np.testing.assert_array_equal(full[:, :11], again)

    def test_frozen_files_registry_and_budget(self):
        self.assertEqual(rr.build_manifests(CFG), [])
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import run_shadowhand_spin_stages as rs

        self.assertEqual(rs.PEN_SPIN_STUDIES[rr.STUDY], "recovery_retry")
        self.assertLessEqual(2 * 24 * 2 * 2 * 2048 + 2 * 2 * 2048 + 2 * 5 * 12 * 128, CFG["start_ceiling"])
        with self.assertRaises(SystemExit):
            rr.parent_x({**CFG, "module_sha256": "0" * 64})

    def test_contrasts(self):
        rng = np.random.default_rng(0)
        rows = {}
        for m in range(12):
            base = rng.random(128) < 0.45
            mk = lambda s: [{"task_hash": f"h{m}_{i}", "sustained": bool(v), "completed": True, "tail_hit_fraction": float(v)}   # noqa: E731
                            for i, v in enumerate(s)]
            mod = base | (rng.random(128) < 0.3)
            rows[f"s{m}"] = {"reference": mk(base), "r106": mk(base), "r116": mk(mod), "continue": mk(mod), "retry": mk(mod | (rng.random(128) < 0.2))}
        c = rr.contrasts(rows, 0.05, 0.05)
        self.assertEqual(len(c), 9)
        self.assertTrue(c["retry_minus_r116"]["useful"])
        self.assertFalse(c["continue_minus_r116"]["useful"])


if __name__ == "__main__":
    unittest.main()
