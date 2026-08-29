#!/usr/bin/env python3
"""Force a syntax failure, then let the real Debugger repair it."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.attempt import attempt_hypothesis
from agent.state import RunState


def main() -> int:
    state = RunState.from_baseline(ROOT / "baseline.py")
    state.best_metrics = {"GAUC": 0.6674, "nDCG@5": 0.5357, "primary": 0.6016}
    hypothesis = {
        "description": "Increase FM L2 regularization from 1e-6 to 1e-5.",
        "rationale": "Use a conservative regularization change after repairing a deliberately invalid first attempt.",
        "target_module": "model",
    }
    broken_diff = "\n".join((
        "<<<<<<< SEARCH",
        "    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):",
        "=======",
        "    def __init__(self, dim, k=16, lr=0.001, l2=1e-5, seed=0)",
        ">>>>>>> REPLACE",
    ))
    result = attempt_hypothesis(
        state,
        hypothesis,
        max_retries=3,
        data_dir=ROOT / "KuaiRand-Pure" / "data",
        cache_dir=ROOT / ".cache",
        runs_dir=ROOT / "runs",
        initial_diff=broken_diff,
    )
    print("retries used: %d" % result.retries_used)
    print("iteration number: %d" % state.iteration_num)
    print("final status: %s" % result.status)
    if result.iteration_result is not None:
        print("metrics: %s" % result.iteration_result.metrics)
    else:
        print("errors: %s" % result.errors)
    return 0 if result.status in {"accepted", "rejected"} else 1


if __name__ == "__main__":
    raise SystemExit(main())