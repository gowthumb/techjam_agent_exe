#!/usr/bin/env python3
"""Run one small Fugu-generated patch through the deterministic Executor."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.coder import propose_patch
from agent.executor import run_candidate
from agent.patcher import apply_patch, validate_syntax
from agent.state import RunState


def main() -> int:
    state = RunState.from_baseline(ROOT / "baseline.py")
    state.best_metrics = {"GAUC": 0.6674, "nDCG@5": 0.5357, "primary": 0.6016}
    hypothesis = {
        "description": "Increase FM L2 regularization from 1e-6 to 1e-5.",
        "rationale": "A small regularization adjustment is a low-risk implementation check before attempting harder loss changes.",
        "target_module": "model",
    }
    coder_result = propose_patch(state.current_code, hypothesis)
    print("Raw Fugu diff:\n" + coder_result.raw_response)
    try:
        validate_syntax(apply_patch(state.current_code, coder_result.diff))
    except Exception as error:
        print("Patch applied cleanly: no (%s: %s)" % (type(error).__name__, error))
        print("token usage: input=%d output=%d total=%d" % (
            coder_result.input_tokens,
            coder_result.output_tokens,
            coder_result.input_tokens + coder_result.output_tokens,
        ))
        return 1
    print("Patch applied cleanly: yes")
    result = run_candidate(
        state,
        coder_result.diff,
        data_dir=ROOT / "KuaiRand-Pure" / "data",
        cache_dir=ROOT / ".cache",
        runs_dir=ROOT / "runs",
        hypothesis=hypothesis["description"],
        rationale=hypothesis["rationale"],
        tokens_used=coder_result.input_tokens + coder_result.output_tokens,
    )
    print("%s: %s" % (result.status.upper(), result.metrics or result.error_trace))
    print("token usage: input=%d output=%d total=%d" % (
        coder_result.input_tokens,
        coder_result.output_tokens,
        coder_result.input_tokens + coder_result.output_tokens,
    ))
    return 0 if result.status in {"accepted", "rejected"} else 1


if __name__ == "__main__":
    raise SystemExit(main())