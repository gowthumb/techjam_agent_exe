#!/usr/bin/env python3
"""Run the autonomous Planner, Coder, Debugger, and Executor loop."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import monotonic
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.attempt import attempt_hypothesis
from agent.llm_client import LLMError, reset_quota_pause_budget
from agent.planner import propose_hypothesis
from agent.runner import score_final_on_test
from agent.state import RunState


def _run_directory(runs_dir: Path, state: RunState) -> Path:
    directory = runs_dir / state.run_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _new_state() -> RunState:
    state = RunState.from_baseline(ROOT / "baseline.py")
    scores = json.loads((ROOT / "baseline_scores.json").read_text(encoding="utf-8"))
    state.best_metrics = scores["scores"]["fm_official"]["valid"]
    return state


def _stopping_reason(state: RunState, max_iterations: int, max_wallclock_s: float, consecutive_abandoned: int) -> Optional[str]:
    if state.iteration_num >= max_iterations:
        return "iteration cap"
    if state.total_wall_clock_s >= max_wallclock_s:
        return "wallclock cap"
    if consecutive_abandoned >= 5:
        return "abandonment safety valve"
    from agent.executor import check_convergence
    if check_convergence(state):
        return "converged"
    return None


def run_loop(
    state: RunState,
    *,
    max_iterations: int,
    max_wallclock_hours: float,
    data_dir: Path,
    runs_dir: Path,
    cache_dir: Path,
    finalize_on_test: bool = True,
) -> dict:
    """Run until convergence, a budget cap, or repeated pre-scoring abandonment."""
    reset_quota_pause_budget()
    consecutive_abandoned = 0
    reason = None
    run_directory = _run_directory(runs_dir, state)
    try:
        while reason is None:
            reason = _stopping_reason(state, max_iterations, max_wallclock_hours * 3600, consecutive_abandoned)
            if reason is not None:
                break
            pass_started_at = monotonic()
            try:
                planner_result = propose_hypothesis(state)
            except Exception:
                state.total_wall_clock_s += monotonic() - pass_started_at
                state.save(run_directory / "state.json")
                raise
            state.total_tokens += planner_result.input_tokens + planner_result.output_tokens
            accounted_before_attempt = state.total_wall_clock_s
            try:
                attempt_result = attempt_hypothesis(
                    state,
                    planner_result.hypothesis,
                    max_retries=3,
                    data_dir=data_dir,
                    cache_dir=cache_dir,
                    runs_dir=runs_dir,
                )
            except Exception:
                state.total_wall_clock_s += max(0.0, monotonic() - pass_started_at - (state.total_wall_clock_s - accounted_before_attempt))
                state.save(run_directory / "state.json")
                raise
            attempt_elapsed_s = monotonic() - pass_started_at
            runner_accounted_s = state.total_wall_clock_s - accounted_before_attempt
            state.total_wall_clock_s += max(0.0, attempt_elapsed_s - runner_accounted_s)
            consecutive_abandoned = consecutive_abandoned + 1 if attempt_result.status == "abandoned" else 0
            state.save(run_directory / "state.json")
        final_test_metrics = None
        if finalize_on_test:
            final_started_at = monotonic()
            final_result = score_final_on_test(state.current_code, data_dir, cache_dir)
            state.total_wall_clock_s += monotonic() - final_started_at
            if final_result["status"] != "ok":
                raise RuntimeError(final_result.get("error_trace", "Final test scoring timed out."))
            final_test_metrics = final_result["metrics"]["test"]
        summary = {
            "stopping_reason": reason,
            "best_validation_metrics": state.best_metrics,
            "final_test_metrics": final_test_metrics,
            "total_iterations": state.iteration_num,
            "total_tokens": state.total_tokens,
            "total_wall_clock_s": state.total_wall_clock_s,
            "manual_interventions": state.manual_interventions,
            "consecutive_abandoned": consecutive_abandoned,
        }
        (run_directory / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        state.save(run_directory / "state.json")
        return summary
    except Exception:
        state.save(run_directory / "state.json")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--max-wallclock-hours", type=float, default=6)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "KuaiRand-Pure" / "data")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache")
    parser.add_argument("--skip-final-test", action="store_true", help="Stop and persist without reading test metrics.")
    args = parser.parse_args()
    state_path = args.runs_dir / args.run_id / "state.json" if args.run_id else None
    if state_path is not None and state_path.exists():
        state = RunState.load(state_path)
    else:
        state = _new_state()
        if args.run_id:
            state.run_id = args.run_id
    try:
        summary = run_loop(
            state,
            max_iterations=args.max_iterations,
            max_wallclock_hours=args.max_wallclock_hours,
            data_dir=args.data_dir,
            runs_dir=args.runs_dir,
            cache_dir=args.cache_dir,
            finalize_on_test=not args.skip_final_test,
        )
    except LLMError as error:
        print("STOPPED: %s" % error, file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())