"""Run scripts/run_triage_candidates.py on the arm-2 materials pool instead of AmalgaMatch (same runner, swapped loader).

  python scripts/arm2_run_candidates.py --pools gt,core,transform,control --out results/arm2/candidates.csv
(--root is ignored; the pool root defaults to C:/Users/frank/Documents/materials-bench.) NOT run during assembly:
the study is pre-registered and outcomes must stay unseen until then.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_triage_candidates as rtc  # noqa: E402

from cma.materials_pool import MaterialsPoolLoader  # noqa: E402

rtc.AmalgaMatchLoader = MaterialsPoolLoader  # main() resolves the loader name at call time

if __name__ == "__main__":
    if "--out" not in sys.argv:
        sys.argv += ["--out", "results/arm2/candidates.csv"]
    rtc.main()
