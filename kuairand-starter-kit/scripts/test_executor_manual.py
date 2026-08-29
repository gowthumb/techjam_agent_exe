#!/usr/bin/env python3
"""Run one hand-written learning-rate patch through the deterministic executor."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.executor import run_candidate
from agent.state import RunState


def main() -> int:
    state = RunState.from_baseline(ROOT / "baseline.py")
    state.best_metrics = {"GAUC": 0.6674, "nDCG@5": 0.5357, "primary": 0.6016}
    diff = "\n".join(
        (
            "<<<<<<< SEARCH",
            "def run_fm(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True):",
            "=======",
            "def run_fm(splits, k=16, lr=0.0005, epochs=40, bs=8192, patience=4, seed=0, verbose=True):",
            ">>>>>>> REPLACE",
        )
    )
    result = run_candidate(
        state,
        diff,
        data_dir=ROOT / "KuaiRand-Pure" / "data",
        cache_dir=ROOT / ".cache",
        runs_dir=ROOT / "runs",
        hypothesis="halve FM learning rate",
        rationale="Test a smaller optimization step against the published validation baseline.",
    )
    print("%s: %s" % (result.status.upper(), result.metrics or result.error_trace))
    return 0 if result.status in {"accepted", "rejected"} else 1


if __name__ == "__main__":
    raise SystemExit(main())