#!/usr/bin/env python3
"""Exercise the real Coder and Debugger retry flow on a modest training-loop change."""
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
        "description": "Adjust the training loop to emphasize negative examples modestly without changing run_fm's signature.",
        "rationale": "A small negative-sampling-style adjustment may better focus ranking training on informative negatives while remaining much simpler than BPR.",
        "target_module": "loss_function",
    }
    result = attempt_hypothesis(
        state,
        hypothesis,
        max_retries=3,
        data_dir=ROOT / "KuaiRand-Pure" / "data",
        cache_dir=ROOT / ".cache",
        runs_dir=ROOT / "runs",
    )
    print("retries used: %d" % result.retries_used)
    print("final status: %s" % result.status)
    if result.iteration_result is not None:
        print("metrics: %s" % result.iteration_result.metrics)
    else:
        print("errors: %s" % result.errors)
    return 0 if result.status in {"accepted", "rejected"} else 1


if __name__ == "__main__":
    raise SystemExit(main())