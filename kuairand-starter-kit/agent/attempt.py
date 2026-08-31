"""Retry orchestration that recovers from pre-scoring candidate failures."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from agent.coder import propose_patch
from agent.debugger import fix_patch
from agent.executor import IterationResult, run_candidate
from agent.llm_client import resolve_model
from agent.logging_utils import log_iteration
from agent.state import RunState


_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class AttemptResult:
    status: str
    iteration_result: Optional[IterationResult]
    attempted_diffs: list[str]
    errors: list[str]
    retries_used: int


def attempt_hypothesis(
    state: RunState,
    hypothesis: Dict[str, str],
    max_retries: int = 3,
    *,
    data_dir: Path | str = _ROOT / "KuaiRand-Pure" / "data",
    cache_dir: Optional[Path | str] = None,
    runs_dir: Path | str = _ROOT / "runs",
    timeout_s: float = 300,
    initial_diff: Optional[str] = None,
    initial_tokens: int = 0,
    dataset: str = "pure",
) -> AttemptResult:
    """Try Coder then Debugger repairs until a candidate receives a validation score."""
    state.retry_count = 0
    attempted_diffs: list[str] = []
    errors: list[str] = []
    failed_diff = initial_diff or "<no valid Search/Replace block produced>"
    tokens_used = initial_tokens
    if initial_diff is None:
        try:
            coder_result = propose_patch(state.current_code, hypothesis)
            failed_diff = coder_result.diff
            tokens_used = coder_result.input_tokens + coder_result.output_tokens
        except Exception as error:
            errors.append("Coder patch generation failed: %s: %s" % (type(error).__name__, error))

    if not errors:
        attempted_diffs.append(failed_diff)
        result = run_candidate(
            state, failed_diff, data_dir, cache_dir, runs_dir, timeout_s,
            hypothesis["description"], hypothesis["rationale"], tokens_used, dataset,
        )
        if result.status in {"accepted", "rejected"}:
            return AttemptResult(result.status, result, attempted_diffs, errors, 0)
        errors.append(result.error_trace or "Candidate failed before validation scoring.")
        failed_diff = attempted_diffs[-1]
    else:
        attempted_diffs.append(failed_diff)

    for retry_index in range(1, max_retries + 1):
        try:
            repair = fix_patch(state.current_code, failed_diff, errors[-1])
            failed_diff = repair.diff
            tokens_used = repair.input_tokens + repair.output_tokens
        except Exception as error:
            errors.append("Debugger patch generation failed: %s: %s" % (type(error).__name__, error))
            attempted_diffs.append(failed_diff)
            continue
        attempted_diffs.append(failed_diff)
        result = run_candidate(
            state, failed_diff, data_dir, cache_dir, runs_dir, timeout_s,
            hypothesis["description"], hypothesis["rationale"], tokens_used, dataset,
        )
        if result.status in {"accepted", "rejected"}:
            return AttemptResult(result.status, result, attempted_diffs, errors, retry_index)
        errors.append(result.error_trace or "Candidate failed before validation scoring.")

    log_iteration(state, {
        "iteration_num": state.iteration_num,
        "hypothesis": hypothesis["description"],
        "rationale": hypothesis["rationale"],
        "code_diff": attempted_diffs,
        "metrics": None,
        "status": "abandoned",
        "error_trace": "\n\n".join(errors),
        "wall_time_s": 0.0,
        "tokens_used": 0,
        "role_models": {
            "planner": resolve_model("PLANNER"),
            "coder": resolve_model("CODER"),
            "debugger": resolve_model("DEBUGGER"),
        },
    }, runs_dir)
    return AttemptResult("abandoned", None, attempted_diffs, errors, max_retries)
