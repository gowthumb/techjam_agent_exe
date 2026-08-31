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

# Mirrors knowledge_base.yaml decision_protocol.single_run_band: a single-run
# validation delta below this sits inside the measured seed-noise band (official
# seed sd ~0.0008) and is not treated as a real improvement. 1K's own seed sd is
# 4x Pure's (0.0022) and 27K's is unmeasured, so this band is a Pure-calibrated
# floor, not a validated gate at scale -- scripts/run_agent_scaled.py layers a
# 3-seed replication check on top before trusting any 1K accept.
_ACCEPTANCE_BAND = 0.0016
# README convergence parameters: epsilon ~= 2.5 sigma, N = 3 non-improving iterations.
_CONVERGENCE_EPSILON = 0.002
_CONVERGENCE_WINDOW = 3

# Per-benchmark defaults. Timeouts are set well above the measured worst case
# (knowledge_base/HARDWARE_AWARENESS.md's per-benchmark table) rather than at
# it, since a Coder-authored patch can legitimately run slower than the
# baseline it was derived from.
_BENCH_DATA_DIR = {
    "pure": _ROOT / "KuaiRand-Pure" / "data",
    "1k": _ROOT / "KuaiRand-1K" / "data",
    "27k": _ROOT / "KuaiRand-27K" / "data",
}
# Pure: ~40s/run. 1K: load ~60s + ~30-45s/epoch, up to 40 epochs -> budget 30min.
# 27K: load ~2063s (~34min) + ~750-774s/epoch, patience=4 -> budget 4h with margin.
_BENCH_TIMEOUT_S = {"pure": 300.0, "1k": 1800.0, "27k": 14400.0}


@dataclass
class IterationResult:
    status: str
    metrics: Optional[Dict[str, Dict[str, float]]] = None
    error_trace: Optional[str] = None


def _entry(state: RunState, diff: str, status: str, metrics: Optional[Dict[str, Any]], wall_time_s: float, error_trace: Optional[str], hypothesis: str, rationale: str, bench: str) -> Dict[str, Any]:
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
        "bench": bench,
        "role_models": {
            "planner": resolve_model("PLANNER"),
            "coder": resolve_model("CODER"),
            "debugger": resolve_model("DEBUGGER"),
        },
    }


def run_candidate(
    state: RunState,
    diff: str,
    data_dir: Optional[Path | str] = None,
    cache_dir: Optional[Path | str] = None,
    runs_dir: Path | str = _ROOT / "runs",
    timeout_s: Optional[float] = None,
    hypothesis: str = "manual candidate patch",
    rationale: str = "Deterministic executor evaluation.",
    tokens_used: int = 0,
    bench: str = "pure",
    seed: Optional[int] = None,
) -> IterationResult:
    """Apply and evaluate a patch, accepting only strict validation improvements.

    ``data_dir``/``timeout_s`` default per ``bench`` (see _BENCH_DATA_DIR /
    _BENCH_TIMEOUT_S above) when not given explicitly.
    """
    if bench not in _BENCH_DATA_DIR:
        raise ValueError("Unknown bench %r; expected one of: %s" % (bench, ", ".join(_BENCH_DATA_DIR)))
    if data_dir is None:
        data_dir = _BENCH_DATA_DIR[bench]
    if timeout_s is None:
        timeout_s = _BENCH_TIMEOUT_S[bench]
    started_at = monotonic()
    try:
        candidate_code = apply_patch(state.current_code, diff)
        validate_syntax(candidate_code)
    except Exception as error:
        state.retry_count += 1
        error_trace = "%s: %s" % (type(error).__name__, error)
        entry = _entry(state, diff, "error", None, monotonic() - started_at, error_trace, hypothesis, rationale, bench)
        entry["tokens_used"] = tokens_used
        log_iteration(state, entry, runs_dir)
        return IterationResult("error", error_trace=error_trace)

    result = runner.run(candidate_code, data_dir, cache_dir, timeout_s, bench=bench, seed=seed)
    wall_time_s = monotonic() - started_at
    if result["status"] != "ok":
        state.retry_count += 1
        error_trace = result.get("error_trace", "Candidate execution timed out.")
        entry = _entry(state, diff, "error", None, wall_time_s, error_trace, hypothesis, rationale, bench)
        entry["tokens_used"] = tokens_used
        log_iteration(state, entry, runs_dir)
        return IterationResult(result["status"], error_trace=error_trace)

    metrics = result["metrics"]
    valid_metrics = metrics["valid"]
    previous_primary = None if state.best_metrics is None else state.best_metrics["primary"]
    accepted = previous_primary is None or valid_metrics["primary"] > previous_primary + _ACCEPTANCE_BAND
    status = "accepted" if accepted else "rejected"
    if accepted:
        state.current_code = candidate_code
        state.best_metrics = {metric: valid_metrics[metric] for metric in ("GAUC", "nDCG@5", "primary")}
    state.retry_count = 0
    state.iteration_num += 1
    entry = _entry(state, diff, status, metrics, wall_time_s, None, hypothesis, rationale, bench)
    entry["tokens_used"] = tokens_used
    log_iteration(state, entry, runs_dir)
    return IterationResult(status, metrics=metrics)


def check_convergence(
    state: RunState,
    epsilon: float = _CONVERGENCE_EPSILON,
    window: int = _CONVERGENCE_WINDOW,
) -> bool:
    """Converged when the best validation primary has not improved past ``epsilon``
    across ``window`` scored iterations since the most recent accepted improvement.

    Scored iterations *before* the first acceptance do not count: a run that has
    not yet beaten its starting point is still searching, not plateaued. Error and
    abandoned iterations carry no validation score and are skipped entirely.
    """
    primaries_since_acceptance: list[float] = []
    seen_acceptance = False
    for entry in state.experiment_history:
        valid_metrics = (entry.get("metrics") or {}).get("valid")
        if valid_metrics is None:
            continue
        if entry.get("status") == "accepted":
            seen_acceptance = True
            primaries_since_acceptance = [valid_metrics["primary"]]
        elif seen_acceptance:
            primaries_since_acceptance.append(valid_metrics["primary"])
    if not seen_acceptance or len(primaries_since_acceptance) <= window:
        return False
    anchor = primaries_since_acceptance[0]
    return max(primaries_since_acceptance) - anchor <= epsilon


def check_caps(state: RunState) -> bool:
    """Return whether the iteration or wall-clock budget has been exhausted."""
    return state.iteration_num >= 50 or state.total_wall_clock_s >= 6 * 3600