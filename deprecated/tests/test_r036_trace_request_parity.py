#!/usr/bin/env python
"""DEPRECATED: the plateau-repair request is byte-identical to R-036's, cut from tests/test_plateau_repair_mock.py.

Archived 2026-11-09: it compares the live lares/repair/traces.py against scripts/compare_generation_modes.py, which moved to deprecated/scripts/. The live request was checked equal on that date. Kept for reference, not run: its imports name the module's original location, so restore the
module first (see deprecated/README.md).
"""

import os
import sys
import unittest

import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

sys.path.insert(0, os.path.join(_PROJECT_ROOT, "deprecated", "scripts"))

from lares.repair import traces as trace_source  # noqa: E402


class TestTraceSource(unittest.TestCase):
    def test_the_assembly_request_is_byte_identical_to_r036(self):
        import compare_generation_modes as cgm

        rows = [{"t": t, "tcp": [0.1, 0.2, 0.3], "gripper": 0.5, "obj": [0.0, 0.6, 0.05], "goal": [0.0, 0.8, 0.1],
                 "action": [0.1, -0.2, 0.3, 1.0]} for t in range(12)]
        episodes = [{"case": f"c{i}", "success": 1.0, "rows": rows} for i in range(3)]
        selected = trace_source.select_traces({"episodes": episodes})
        ours = trace_source.repair_request("assembly-v2", "class GeneratedPolicy: pass\n", 10, 10, selected)
        block = cgm.trace_block({"episodes": episodes}).replace(
            "Successful expert continuations after your lift", "Successful expert episodes after the expert lifted the nut")
        theirs = cgm.WHOLE_CONTROLLER_REQUEST.replace("{successes}", "10").replace("{episodes}", "10").replace(
            "{parent}", "class GeneratedPolicy: pass") + "\n" + block
        self.assertEqual(ours, theirs)


if __name__ == "__main__":
    unittest.main()
