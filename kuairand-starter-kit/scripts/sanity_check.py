#!/usr/bin/env python3
"""Reproduce the published FM baseline and create an initial audited run."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from agent.logging_utils import log_iteration
from agent.runner import run, score_final_on_test
from agent.state import RunState


def iteration_entry(status, metrics, wall_time_s, error_trace=None):
    return {
        "iteration_num": 0,
        "hypothesis": "reproduce official baseline",
        "rationale": "Verify the locked evaluation contract and deterministic starting point.",
        "code_diff": "",
        "metrics": metrics,
        "status": status,
        "error_trace": error_trace,
        "wall_time_s": wall_time_s,
        "tokens_used": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPOSITORY_ROOT / "KuaiRand-Pure" / "data")
    parser.add_argument("--cache-dir", type=Path, default=REPOSITORY_ROOT / ".cache")
    parser.add_argument("--runs-dir", type=Path, default=REPOSITORY_ROOT / "runs")
    parser.add_argument("--run-id", help="Optional stable identifier for the created run directory.")
    args = parser.parse_args()

    state = RunState.from_baseline(REPOSITORY_ROOT / "baseline.py")
    if args.run_id:
        state.run_id = args.run_id
    started_at = time.monotonic()

    try:
        validation_result = run(state.current_code, args.data_dir, args.cache_dir)
        if validation_result["status"] != "ok":
            raise RuntimeError(validation_result.get("error_trace", "Validation execution timed out."))
        final_result = score_final_on_test(state.current_code, args.data_dir, args.cache_dir)
        if final_result["status"] != "ok":
            raise RuntimeError(final_result.get("error_trace", "Final test execution timed out."))
        results = {
            "valid": validation_result["metrics"]["valid"],
            "test": final_result["metrics"]["test"],
        }

        expected_primary = json.loads((REPOSITORY_ROOT / "baseline_scores.json").read_text(encoding="utf-8"))["scores"]["fm_official"]["test"]["primary"]
        test_primary = results["test"]["primary"]
        metrics = {"valid": results["valid"], "test": results["test"]}
        wall_time_s = time.monotonic() - started_at
        if abs(test_primary - expected_primary) > 0.002:
            message = "test primary %.4f differs from published %.4f by more than 0.002" % (test_primary, expected_primary)
            log_iteration(state, iteration_entry("rejected", metrics, wall_time_s, message), args.runs_dir)
            print("FAIL: " + message)
            return 1

        state.best_metrics = {
            metric: results["valid"][metric]
            for metric in ("GAUC", "nDCG@5", "primary")
        }
        log_iteration(state, iteration_entry("accepted", metrics, wall_time_s), args.runs_dir)
        print("PASS: test primary %.4f matches published %.4f within 0.002" % (test_primary, expected_primary))
        return 0
    except Exception:
        wall_time_s = time.monotonic() - started_at
        error_trace = traceback.format_exc()
        log_iteration(state, iteration_entry("error", None, wall_time_s, error_trace), args.runs_dir)
        print("FAIL: baseline sanity check crashed", file=sys.stderr)
        print(error_trace, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())