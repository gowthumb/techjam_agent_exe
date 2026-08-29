#!/usr/bin/env python3
"""Prove a permanently broken repair path is abandoned without crashing a run."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.attempt import attempt_hypothesis
from agent.coder import CoderResult
from agent.state import RunState


MAX_RETRIES = 3
HYPOTHESIS = {
    "description": "Increase FM L2 regularization from 1e-6 to 1e-5.",
    "rationale": "Prove that a permanently broken patch is logged and abandoned safely.",
    "target_module": "model",
}
BROKEN_DIFF = "\n".join((
    "<<<<<<< SEARCH",
    "    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):",
    "=======",
    "    def __init__(self, dim, k=16, lr=0.001, l2=1e-5, seed=0)",
    ">>>>>>> REPLACE",
))


def main() -> int:
    state = RunState.from_baseline(ROOT / "baseline.py")
    always_broken = CoderResult(BROKEN_DIFF, BROKEN_DIFF, 0, 0)
    with tempfile.TemporaryDirectory() as temporary_directory, patch(
        "agent.attempt.fix_patch", return_value=always_broken
    ) as debugger:
        result = attempt_hypothesis(
            state,
            HYPOTHESIS,
            max_retries=MAX_RETRIES,
            runs_dir=Path(temporary_directory),
            initial_diff=BROKEN_DIFF,
        )
        log_path = Path(temporary_directory) / state.run_id / "iterations.jsonl"
        entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

    abandoned_entries = [entry for entry in entries if entry["status"] == "abandoned"]
    abandoned = abandoned_entries[0] if len(abandoned_entries) == 1 else None
    expected_attempts = MAX_RETRIES + 1
    assert result.status == "abandoned"
    assert result.iteration_result is None
    assert state.iteration_num == 0
    assert debugger.call_count == MAX_RETRIES
    assert len(result.attempted_diffs) == expected_attempts
    assert len(result.errors) == expected_attempts
    assert abandoned is not None
    assert abandoned["hypothesis"] == HYPOTHESIS["description"]
    assert abandoned["code_diff"] == result.attempted_diffs
    assert abandoned["error_trace"].count("SyntaxError:") == expected_attempts
    print("status: %s" % result.status)
    print("debugger repair attempts: %d" % debugger.call_count)
    print("candidate attempts: %d" % len(result.attempted_diffs))
    print("iteration_num unchanged: %s (%d)" % (state.iteration_num == 0, state.iteration_num))
    print("abandoned entries: %d" % len(abandoned_entries))
    print("abandoned log fields: %s" % ", ".join(sorted(abandoned)))
    print("logged diffs: %d; logged errors: %d" % (len(abandoned["code_diff"]), len(result.errors)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())