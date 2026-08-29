"""Retry orchestration that recovers from pre-scoring candidate failures."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Dict, Optional

from agent.coder import CoderResult, propose_patch
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
) -> AttemptResult:
    """Try Coder then Debugger repairs until a candidate receives a validation score."""
    state.retry_count = 0
    attempted_diffs: list[str] = []
    errors: list[str] = []
    if initial_diff is None:
        coder_result = propose_patch(state.current_code, hypothesis)
        diff = coder_result.diff
        tokens_used = coder_result.input_tokens + coder_result.output_tokens
    else:
        diff = initial_diff
        tokens_used = initial_tokens

    for retry_index in range(max_retries + 1):
        attempted_diffs.append(diff)
        result = run_candidate(
            state, diff, data_dir, cache_dir, runs_dir, timeout_s,
            hypothesis["description"], hypothesis["rationale"], tokens_used,
        )
        if result.status in {"accepted", "rejected"}:
            return AttemptResult(result.status, result, attempted_diffs, errors, retry_index)
        errors.append(result.error_trace or "Candidate failed before validation scoring.")
        if retry_index == max_retries:
            break
        repair = fix_patch(state.current_code, diff, errors[-1])
        diff = repair.diff
        tokens_used = repair.input_tokens + repair.output_tokens

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