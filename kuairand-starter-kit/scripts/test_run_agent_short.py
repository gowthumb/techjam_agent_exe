#!/usr/bin/env python3
"""Run two real autonomous iterations and verify finalization and persistence."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.state import RunState
from scripts.run_agent import _new_state, run_loop


def main() -> int:
    os.environ.setdefault("LLM_RATE_LIMIT_BACKOFF_S", "1")
    os.environ.setdefault("LLM_REQUEST_TIMEOUT_S", "300")
    run_id = "short-autonomous-loop"
    runs_dir = ROOT / "runs"
    state = _new_state()
    state.run_id = run_id
    summary = run_loop(
        state,
        max_iterations=2,
        max_wallclock_hours=6,
        data_dir=ROOT / "KuaiRand-Pure" / "data",
        runs_dir=runs_dir,
        cache_dir=ROOT / ".cache",
    )
    state_path = runs_dir / run_id / "state.json"
    summary_path = runs_dir / run_id / "summary.json"
    persisted_state = RunState.load(state_path)
    persisted_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["stopping_reason"] == "iteration cap"
    assert summary["total_iterations"] == 2
    assert summary["final_test_metrics"]
    assert persisted_state.iteration_num == 2
    assert persisted_state.total_tokens == summary["total_tokens"]
    assert persisted_state.total_wall_clock_s == summary["total_wall_clock_s"]
    assert persisted_summary == summary
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("persistence verified: state.json and summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())