"""R-083 (D-058): the administrative adoption of the fixed-stroke controller as the pen-spin incumbent.

Nothing runs here. The exact evaluated artefacts of R-081's constant-gain control were frozen under
``config/pen_spin/d058_fixed_stroke_incumbent/`` (parent source and state, the evaluated substituted source, the
constant with its exact float, and ``incumbent.json`` with every file's sha256). This script only verifies them.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSET = os.path.join(ROOT, "config", "pen_spin", "d058_fixed_stroke_incumbent")


def verify() -> dict:
    with open(os.path.join(ASSET, "incumbent.json"), encoding="utf-8") as f:
        inc = json.load(f)
    bad = {}
    for name, expected in inc["files"].items():
        with open(os.path.join(ASSET, name), "rb") as f:
            got = hashlib.sha256(f.read()).hexdigest()
        if got != expected:
            bad[name] = (expected, got)
    with open(os.path.join(ASSET, "constant_gain.json"), encoding="utf-8") as f:
        if json.load(f)["constant"] != inc["constant"]:
            bad["constant"] = "incumbent.json and constant_gain.json disagree"
    return {"asset": ASSET, "constant": inc["constant"], "files": list(inc["files"]), "drift": bad, "passes": not bad}


if __name__ == "__main__":
    out = verify()
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["passes"] else 1)
