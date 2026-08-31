#!/usr/bin/env python3
"""Run the autonomous Planner, Coder, Debugger, and Executor loop."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
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


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _progress(message: str) -> None:
    """Emit one high-level orchestrator progress line, matching the LLM client's prefix."""
    print("[%s] %s" % (_now(), message), flush=True)


def _fmt_metrics(metrics: Optional[dict]) -> str:
    """Render GAUC / nDCG@5 / primary from a metrics dict, tolerating missing keys."""
    if not metrics:
        return "n/a"
    parts = ["%s %.5f" % (key, metrics[key]) for key in ("primary", "GAUC", "nDCG@5") if metrics.get(key) is not None]
    return " / ".join(parts) if parts else "n/a"


def _short(text: object, limit: int = 240) -> str:
    collapsed = " ".join(str(text).split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"


def _run_directory(runs_dir: Path, state: RunState) -> Path:
    directory = runs_dir / state.run_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# bench -> baseline module path. Pure's has a precomputed baseline_scores.json;
# 1K/27K don't, so their starting best_metrics comes from actually running the
# baseline once (cheap on 1K, expensive but self-verifying on 27K -- see
# scripts/run_agent_scaled.py for the recommended 1K-first path instead of
# running this script directly with --bench 27k).
_BASELINE_PATH = {
    "pure": ROOT / "baseline.py",
    "1k": ROOT / "baseline_1k.py",
    "27k": ROOT / "baseline_27k.py",
}


def _new_state(bench: str = "pure", data_dir: Optional[Path] = None, cache_dir: Optional[Path] = None, timeout_s: Optional[float] = None) -> RunState:
    state = RunState.from_baseline(_BASELINE_PATH[bench])
    if bench == "pure":
        scores = json.loads((ROOT / "baseline_scores.json").read_text(encoding="utf-8"))
        state.best_metrics = scores["scores"]["fm_official"]["valid"]
        return state
    from agent.executor import _BENCH_DATA_DIR, _BENCH_TIMEOUT_S
    from agent.runner import run as run_candidate_code
    _progress("seeding %s baseline metrics (one run of %s) ..." % (bench.upper(), _BASELINE_PATH[bench].name))
    result = run_candidate_code(
        state.current_code,
        data_dir if data_dir is not None else _BENCH_DATA_DIR[bench],
        cache_dir,
        timeout_s if timeout_s is not None else _BENCH_TIMEOUT_S[bench],
        bench=bench,
    )
    if result["status"] != "ok":
        raise RuntimeError("Could not establish a %s baseline: %s" % (bench.upper(), result.get("error_trace", result["status"])))
    state.best_metrics = result["metrics"]["valid"]
    _progress("%s baseline seeded | valid %s" % (bench.upper(), _fmt_metrics(state.best_metrics)))
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
    bench: str = "pure",
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
            _progress(
                "iteration %d starting | scored so far %d/%d | tokens used %d | wall-clock %.0fs | best valid %s"
                % (
                    state.iteration_num + 1,
                    state.iteration_num,
                    max_iterations,
                    state.total_tokens,
                    state.total_wall_clock_s,
                    _fmt_metrics(state.best_metrics),
                )
            )
            try:
                planner_result = propose_hypothesis(state, bench=bench)
            except LLMError:
                state.total_wall_clock_s += monotonic() - pass_started_at
                state.save(run_directory / "state.json")
                raise
            except Exception as error:
                state.total_wall_clock_s += monotonic() - pass_started_at
                consecutive_abandoned += 1
                _progress(
                    "iteration %d planner produced no usable hypothesis (%s: %s) — skipping (%d in a row)"
                    % (state.iteration_num + 1, type(error).__name__, error, consecutive_abandoned)
                )
                state.save(run_directory / "state.json")
                continue
            state.total_tokens += planner_result.input_tokens + planner_result.output_tokens
            _progress(
                "iteration %d hypothesis (%s): %s"
                % (
                    state.iteration_num + 1,
                    planner_result.hypothesis.get("target_module", "?"),
                    _short(planner_result.hypothesis["description"]),
                )
            )
            accounted_before_attempt = state.total_wall_clock_s
            try:
                attempt_result = attempt_hypothesis(
                    state,
                    planner_result.hypothesis,
                    max_retries=3,
                    data_dir=data_dir,
                    cache_dir=cache_dir,
                    runs_dir=runs_dir,
                    bench=bench,
                )
            except Exception:
                state.total_wall_clock_s += max(0.0, monotonic() - pass_started_at - (state.total_wall_clock_s - accounted_before_attempt))
                state.save(run_directory / "state.json")
                raise
            attempt_elapsed_s = monotonic() - pass_started_at
            runner_accounted_s = state.total_wall_clock_s - accounted_before_attempt
            state.total_wall_clock_s += max(0.0, attempt_elapsed_s - runner_accounted_s)
            consecutive_abandoned = consecutive_abandoned + 1 if attempt_result.status == "abandoned" else 0
            latest_valid = None
            if attempt_result.iteration_result is not None and attempt_result.iteration_result.metrics:
                latest_valid = attempt_result.iteration_result.metrics.get("valid")
            if attempt_result.status == "abandoned":
                _progress(
                    "iteration %d ABANDONED after %d retries (no validation score) | tokens used %d | pass %.0fs"
                    % (state.iteration_num + 1, attempt_result.retries_used, state.total_tokens, attempt_elapsed_s)
                )
            else:
                _progress(
                    "iteration %d %s | latest valid %s | best valid %s | tokens used %d | pass %.0fs"
                    % (
                        state.iteration_num,
                        attempt_result.status.upper(),
                        _fmt_metrics(latest_valid),
                        _fmt_metrics(state.best_metrics),
                        state.total_tokens,
                        attempt_elapsed_s,
                    )
                )
            state.save(run_directory / "state.json")
        final_test_metrics = None
        if finalize_on_test:
            final_started_at = monotonic()
            final_result = score_final_on_test(state.current_code, data_dir, cache_dir, bench=bench)
            state.total_wall_clock_s += monotonic() - final_started_at
            if final_result["status"] != "ok":
                raise RuntimeError(final_result.get("error_trace", "Final test scoring timed out."))
            final_test_metrics = final_result["metrics"]["test"]
        _progress(
            "run complete (%s) | scored iterations %d | best valid %s | final test %s | tokens used %d | wall-clock %.0fs"
            % (
                reason,
                state.iteration_num,
                _fmt_metrics(state.best_metrics),
                _fmt_metrics(final_test_metrics),
                state.total_tokens,
                state.total_wall_clock_s,
            )
        )
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
    parser.add_argument("--bench", choices=["pure", "1k", "27k"], default="pure",
                        help="Which benchmark to target. 1k/27k use sparse-Adam baselines and the "
                             "HARDWARE_AWARENESS.md/ONEK_RESULTS.md-aware Planner context; see "
                             "scripts/run_agent_scaled.py for the recommended 1K-first, "
                             "27K-confirmation workflow instead of running --bench 27k directly.")
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--max-wallclock-hours", type=float, default=6)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache")
    parser.add_argument("--skip-final-test", action="store_true", help="Stop and persist without reading test metrics.")
    args = parser.parse_args()
    from agent.executor import _BENCH_DATA_DIR
    data_dir = args.data_dir if args.data_dir is not None else _BENCH_DATA_DIR[args.bench]
    state_path = args.runs_dir / args.run_id / "state.json" if args.run_id else None
    if state_path is not None and state_path.exists():
        state = RunState.load(state_path)
    else:
        state = _new_state(args.bench, data_dir, args.cache_dir)
        if args.run_id:
            state.run_id = args.run_id
    try:
        summary = run_loop(
            state,
            max_iterations=args.max_iterations,
            max_wallclock_hours=args.max_wallclock_hours,
            data_dir=data_dir,
            runs_dir=args.runs_dir,
            cache_dir=args.cache_dir,
            finalize_on_test=not args.skip_final_test,
            bench=args.bench,
        )
    except LLMError as error:
        print("STOPPED: %s" % error, file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())