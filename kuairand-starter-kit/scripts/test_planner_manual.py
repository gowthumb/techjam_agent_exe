#!/usr/bin/env python3
"""Exercise Planner selection for a fresh and duplicate-constrained run state."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.planner import propose_hypothesis
from agent.state import RunState


BASELINE_METRICS = {"GAUC": 0.6674, "nDCG@5": 0.5357, "primary": 0.6016}


def print_result(label: str, result) -> None:
    print(label + " hypothesis: " + str(result.hypothesis))
    print("token usage: input=%d output=%d total=%d" % (
        result.input_tokens,
        result.output_tokens,
        result.input_tokens + result.output_tokens,
    ))


def main() -> int:
    fresh_state = RunState(current_code="", best_metrics=BASELINE_METRICS)
    fresh_result = propose_hypothesis(fresh_state)
    print_result("fresh", fresh_result)

    prior_rejection = {
        "iteration_num": 1,
        "hypothesis": "increase embedding dimension to k=32",
        "rationale": "Test extra model capacity.",
        "code_diff": "",
        "metrics": {"valid": {"GAUC": 0.65, "nDCG@5": 0.53, "primary": 0.59}},
        "status": "rejected",
        "error_trace": None,
        "wall_time_s": 1.0,
        "tokens_used": 0,
    }
    constrained_state = RunState(
        current_code="", best_metrics=BASELINE_METRICS, experiment_history=[prior_rejection]
    )
    constrained_result = propose_hypothesis(constrained_state)
    print_result("rejected-k32 constrained", constrained_result)
    forbidden = ("embedding", "k=32", "k = 32")
    assert not any(term in constrained_result.hypothesis["description"].lower() for term in forbidden)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())