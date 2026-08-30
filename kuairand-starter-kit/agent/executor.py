"""Deterministic patch, validation, acceptance, and stopping logic."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Dict, Optional

from agent import runner
from agent.logging_utils import log_iteration
from agent.llm_client import resolve_model
from agent.patcher import apply_patch, validate_syntax
from agent.state import RunState


_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class IterationResult:
    status: str
    metrics: Optional[Dict[str, Dict[str, float]]] = None
    error_trace: Optional[str] = None


def _entry(state: RunState, diff: str, status: str, metrics: Optional[Dict[str, Any]], wall_time_s: float, error_trace: Optional[str], hypothesis: str, rationale: str) -> Dict[str, Any]:
    return {
        "iteration_num": state.iteration_num,
        "hypothesis": hypothesis,
        "rationale": rationale,
        "code_diff": diff,
        "metrics": metrics,
        "status": status,
        "error_trace": error_trace,
        "wall_time_s": wall_time_s,
        "tokens_used": 0,
        "role_models": {
            "planner": resolve_model("PLANNER"),
            "coder": resolve_model("CODER"),
            "debugger": resolve_model("DEBUGGER"),
        },
    }


def run_candidate(
    state: RunState,
    diff: str,
    data_dir: Path | str = _ROOT / "KuaiRand-Pure" / "data",
    cache_dir: Optional[Path | str] = None,
    runs_dir: Path | str = _ROOT / "runs",
    timeout_s: float = 300,
    hypothesis: str = "manual candidate patch",
    rationale: str = "Deterministic executor evaluation.",
    tokens_used: int = 0,
) -> IterationResult:
    """Apply and evaluate a patch, accepting only strict validation improvements."""
    started_at = monotonic()
    try:
        candidate_code = apply_patch(state.current_code, diff)
        validate_syntax(candidate_code)
    except Exception as error:
        state.retry_count += 1
        error_trace = "%s: %s" % (type(error).__name__, error)
        entry = _entry(state, diff, "error", None, monotonic() - started_at, error_trace, hypothesis, rationale)
        entry["tokens_used"] = tokens_used
        log_iteration(state, entry, runs_dir)
        return IterationResult("error", error_trace=error_trace)

    result = runner.run(candidate_code, data_dir, cache_dir, timeout_s)
    wall_time_s = monotonic() - started_at
    if result["status"] != "ok":
        state.retry_count += 1
        error_trace = result.get("error_trace", "Candidate execution timed out.")
        entry = _entry(state, diff, "error", None, wall_time_s, error_trace, hypothesis, rationale)
        entry["tokens_used"] = tokens_used
        log_iteration(state, entry, runs_dir)
        return IterationResult(result["status"], error_trace=error_trace)

    metrics = result["metrics"]
    valid_metrics = metrics["valid"]
    previous_primary = None if state.best_metrics is None else state.best_metrics["primary"]
    accepted = previous_primary is None or valid_metrics["primary"] > previous_primary
    status = "accepted" if accepted else "rejected"
    if accepted:
        state.current_code = candidate_code
        state.best_metrics = {metric: valid_metrics[metric] for metric in ("GAUC", "nDCG@5", "primary")}
    state.retry_count = 0
    state.iteration_num += 1
    entry = _entry(state, diff, status, metrics, wall_time_s, None, hypothesis, rationale)
    entry["tokens_used"] = tokens_used
    log_iteration(state, entry, runs_dir)
    return IterationResult(status, metrics=metrics)


def check_convergence(state: RunState) -> bool:
    """Apply epsilon=0.002 over three scored intervals after an accepted improvement."""
    best_so_far = []
    best = float("-inf")
    accepted_improvement = False
    for entry in state.experiment_history:
        valid_metrics = (entry.get("metrics") or {}).get("valid")
        if valid_metrics is not None:
            best = max(best, valid_metrics["primary"])
            best_so_far.append(best)
            accepted_improvement = accepted_improvement or entry.get("status") == "accepted"
    return accepted_improvement and len(best_so_far) >= 4 and best_so_far[-1] - best_so_far[-4] <= 0.002


def check_caps(state: RunState) -> bool:
    """Return whether the iteration or wall-clock budget has been exhausted."""
    return state.iteration_num >= 50 or state.total_wall_clock_s >= 6 * 3600