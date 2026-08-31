"""Deterministic patch, validation, acceptance, and stopping logic."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Dict, Optional

from agent import dependencies, runner
from agent.logging_utils import log_auto_install, log_iteration
from agent.llm_client import resolve_model
from agent.patcher import apply_patch, validate_syntax
from agent.state import RunState


_ROOT = Path(__file__).resolve().parents[1]

# Per-benchmark minimum validation-primary delta to accept a candidate as an
# improvement over the current best, keyed by bench.
#
# Pure's 0.0016 mirrors knowledge_base.yaml decision_protocol.single_run_band:
# a single-run delta below this sits inside Pure's measured 5-seed noise
# (sd ~0.0008) and is not treated as a real improvement.
#
# 1K's went through two revisions before landing back on Pure's number, and
# the reasoning from both is worth keeping:
#   - Raised to 0.032 at one point, deliberately far past anything 1K's own
#     noise (sd 0.0022) would require, after a Pure-track false accept made a
#     wide margin look prudent. In practice this made acceptance nearly
#     unreachable: the one confirmed, 3-seed-replicated 1K win in this
#     codebase's history (a sparse-Adagrad swap) was itself a +0.0018
#     single-run delta -- smaller than even a "moderate" 0.005-0.007 band a
#     later revision proposed. A whole subsequent run (10 iterations, real
#     mechanisms, best delta +0.0034) confirmed 0.032 was calibrated to a
#     magnitude this benchmark has never produced, on either side.
#   - Reset to 0.0016, matching Pure, because that is the exact value that
#     already found 1K's one real result empirically (not a theoretical
#     argument) -- 0.0018 > 0.0016. The hackathon's own convergence rule
#     (score_agent - score_baseline, no minimum-delta floor; converged when
#     validation hasn't improved by more than epsilon=0.002 over 3
#     iterations) has no acceptance threshold at all, so an internal band
#     stricter than what the benchmark can produce only costs the ability to
#     ever keep a real, gradeable improvement. The actual defense against a
#     false positive is unchanged and does the real work: mandatory 3-seed
#     replication (scripts/maximize_1k.py) before any accept is trusted, not
#     this screen.
#
# 27K currently out of scope (no evidence to justify a different number);
# defaults to Pure's.
_ACCEPTANCE_BAND = {"pure": 0.0016, "1k": 0.0016, "27k": 0.0016}
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
# Pure: ~40s/run. 1K: load ~60s + ~30-45s/epoch, up to 40 epochs -> worst case
# ~1800s; budgeted with margin since 1K is now the sole target (no budget held
# back for a 27K run). 27K: load ~2063s (~34min) + ~750-774s/epoch, patience=4
# -> budget 4h with margin. (27K is currently out of scope -- this machine's
# archive is incomplete, see HARDWARE_AWARENESS.md rule 6 -- but the timeout
# stays defined so resuming it later is a data problem, not a code one.)
_BENCH_TIMEOUT_S = {"pure": 300.0, "1k": 2700.0, "27k": 14400.0}

_METRIC_KEYS = ("GAUC", "nDCG@5", "primary")

# What a no-op patch actually looks like in practice (observed directly, not
# hypothetical): a Coder diff that adds a parameter/capability but leaves its
# default at the value that reproduces the original computation exactly (a
# weight of 1.0, an epsilon of 0.0), or a diff that never wires into the
# active code path at all. Either way the candidate trains and scores
# successfully -- it isn't an error -- but its valid metrics come out
# bit-for-bit identical to the current best's, because nothing about the
# actual computation changed. That is reliably detectable (independent
# stochastic training reproducing 3 floats to the bit by chance is not a
# realistic coincidence) and is a different failure mode from "the mechanism
# was tried and lost": one wastes an iteration testing nothing, the other is
# real information. Routed back through the same Debugger-repair path as a
# runtime error (see agent/attempt.py) rather than silently logged as
# "rejected," so it gets a real second chance within the same hypothesis's
# retry budget instead of quietly costing a full Planner round-trip for
# nothing.
_NO_OP_MESSAGE = (
    "This patch's validation metrics (GAUC / nDCG@5 / primary) are bit-for-bit identical to the "
    "current best's. That is not a negative result for the hypothesis -- it means the patch did not "
    "change the model's computation at all (most likely: a new parameter left at its identity/no-op "
    "default -- weight 1.0, probability or epsilon 0.0 -- or a diff that never wires into the active "
    "code path). Implement the hypothesis's actual mechanism: pick ONE concrete, non-identity value "
    "for any parameter the hypothesis names and bake it in as what actually executes. The Executor "
    "calls run_fm(splits) with no extra keyword arguments beyond an optional seed, so whatever "
    "default you set is the only configuration that will be tested."
)


def _metrics_match(a: Dict[str, float], b: Dict[str, float]) -> bool:
    """True if two metrics dicts are identical on every key that matters for acceptance."""
    return all(a.get(key) == b.get(key) for key in _METRIC_KEYS)


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
    if result["status"] != "ok":
        # A missing third-party import isn't a code problem the Coder/Debugger
        # can patch their way out of -- try installing it once (never more
        # than once per module per run; see agent/dependencies.py) and retry
        # the SAME code before falling through to the normal error path. This
        # never touches the Coder/Debugger retry budget in agent/attempt.py.
        missing = dependencies.missing_module(result.get("error_trace"))
        if missing and missing not in state.attempted_installs:
            state.attempted_installs.append(missing)
            install_result = dependencies.install(missing)
            log_auto_install(state, missing, install_result.ok, install_result.message, runs_dir)
            if install_result.ok:
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
    previous_metrics = state.best_metrics
    previous_primary = None if previous_metrics is None else previous_metrics["primary"]
    accepted = previous_primary is None or valid_metrics["primary"] > previous_primary + _ACCEPTANCE_BAND[bench]
    no_op = not accepted and previous_metrics is not None and _metrics_match(valid_metrics, previous_metrics)
    if no_op:
        # Trained and scored successfully -- not an error -- but tested nothing.
        # Mirror the error path: don't advance iteration_num/reset retry_count,
        # so attempt_hypothesis's existing "anything but accepted/rejected goes
        # back to the Debugger" logic gives this hypothesis a real repair
        # attempt within its own retry budget instead of silently wasting the
        # iteration on a candidate that never ran.
        state.retry_count += 1
        entry = _entry(state, diff, "no_op", metrics, wall_time_s, _NO_OP_MESSAGE, hypothesis, rationale, bench)
        entry["tokens_used"] = tokens_used
        log_iteration(state, entry, runs_dir)
        return IterationResult("no_op", metrics=metrics, error_trace=_NO_OP_MESSAGE)

    status = "accepted" if accepted else "rejected"
    if accepted:
        state.current_code = candidate_code
        state.best_metrics = {metric: valid_metrics[metric] for metric in _METRIC_KEYS}
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