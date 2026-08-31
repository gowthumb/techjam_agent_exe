#!/usr/bin/env python3
"""Maximize KuaiRand-1K's score via the standard agent loop, replicated for trust.

27K is out of scope for this script -- this machine's KuaiRand-27K archive is
incomplete (knowledge_base/HARDWARE_AWARENESS.md rule 6) and, independent of
that, ONEK_RESULTS.md's own record is that nothing has ever beaten 1K's
untuned baseline across ~12 axes, so 1K itself is where any further search
should happen anyway. Every iteration here searches and scores against
KuaiRand-1K only, with the full 50-iteration / 6h budget -- no budget is held
back for a 27K confirmation the way an earlier version of this workflow did.

The Planner's knowledge base for this run is knowledge_base.yaml PLUS
knowledge_base/SCALE_DIRECTIVES.md, a condensed distillation of
ONEK_RESULTS.md and HARDWARE_AWARENESS.md's operational directives
(agent/planner.py injects it whenever bench != "pure" -- see
_scaled_bench_context there; ONEK_RESULTS.md's "Token-usage pass" has the
full before/after on why it's condensed rather than the ~63KB of raw source).
It's the record of what has and hasn't beaten 1K's baseline across BPR, SSM,
three affinity fields, k/lr sweeps, embedding noise, and two pairwise-GBDT
objectives, so it's the primary thing steering the Planner away from
re-proposing an already-negative axis and toward the few genuinely untested
ones (CatBoost YetiRank's cat_features fix -- now actually runnable, catboost
is installed; content/side-information features that don't require a
trained-from-scratch ID embedding).

After the loop stops (converged, or the iteration/wallclock cap), the winning
candidate -- if any candidate ever beat the baseline -- is replicated over
--replication-seeds seeds before being reported as the final result. This is
the same discipline that has already caught two false single-seed leads in
this exact codebase (ONEK_RESULTS.md Phase 10's lr_0.0005, Phase 19's
xgb_ndcg): a lead that doesn't survive 3-seed replication is noise, not an
improvement, no matter how good it looked on the one seed the search happened
to run.

Run:  python scripts/maximize_1k.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from agent.executor import _ACCEPTANCE_BAND, _BENCH_DATA_DIR, _BENCH_TIMEOUT_S  # noqa: E402
from agent.llm_client import LLMError, reset_quota_pause_budget  # noqa: E402
from agent.runner import run as run_candidate_code  # noqa: E402
from agent.runner import score_final_on_test  # noqa: E402
from agent.state import RunState  # noqa: E402
from run_agent import _fmt_metrics, _new_state, _progress, run_loop  # noqa: E402


def _has_accepted(state: RunState) -> bool:
    return any(entry.get("status") == "accepted" for entry in state.experiment_history)


def _search_breakdown(state: RunState, baseline_primary: float, band: float) -> Dict[str, Any]:
    """Categorize every logged attempt and find the best GENUINE one -- i.e. one that
    actually tested a different computation, not a no-op patch or a pre-scoring error.

    This exists so a run's "nothing beat the baseline" verdict is honest about how
    much of the run was real signal vs. wasted iterations (agent/executor.py's
    no-op detection): a run with several no_op entries didn't test as many real
    hypotheses as its iteration count suggests, and reporting only "N rejected" would
    overstate how much was actually learned.
    """
    counts = {"accepted": 0, "rejected": 0, "no_op": 0, "error": 0, "abandoned": 0}
    best_genuine = None
    for entry in state.experiment_history:
        status = entry.get("status")
        if status in counts:
            counts[status] += 1
        if status in ("accepted", "rejected"):
            primary = ((entry.get("metrics") or {}).get("valid") or {}).get("primary")
            if primary is not None and (best_genuine is None or primary > best_genuine):
                best_genuine = primary
    return {
        "counts": counts,
        "best_genuine_valid_primary": best_genuine,
        "best_genuine_delta_vs_baseline": None if best_genuine is None else best_genuine - baseline_primary,
        "band_required": band,
    }


def _replicate(code: str, data_dir: Path, cache_dir, timeout_s: float, seeds: list[int]) -> list[dict]:
    runs = []
    for seed in seeds:
        _progress("replication seed %d starting ..." % seed)
        result = run_candidate_code(code, data_dir, cache_dir, timeout_s, bench="1k", seed=seed)
        if result["status"] != "ok":
            _progress("replication seed %d FAILED: %s" % (seed, result.get("error_trace", result["status"])))
            continue
        primary = result["metrics"]["valid"]["primary"]
        _progress("replication seed %d | valid primary %.5f" % (seed, primary))
        runs.append({"seed": seed, "valid": result["metrics"]["valid"]})
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id")
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--max-wallclock-hours", type=float, default=6.0)
    parser.add_argument("--replication-seeds", default="0,1,2",
                        help="Comma-separated seeds to replicate the winning candidate over before "
                             "reporting it as the final result.")
    parser.add_argument("--data-dir", type=Path, default=_BENCH_DATA_DIR["1k"])
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache")
    parser.add_argument("--skip-final-test", action="store_true", help="Skip the final test-set read.")
    args = parser.parse_args()
    seeds = [int(s) for s in args.replication_seeds.split(",") if s.strip()]

    reset_quota_pause_budget()
    state_path = args.runs_dir / args.run_id / "state.json" if args.run_id else None
    if state_path is not None and state_path.exists():
        state = RunState.load(state_path)
    else:
        state = _new_state("1k", args.data_dir, args.cache_dir)
        if args.run_id:
            state.run_id = args.run_id
    baseline_primary = state.best_metrics["primary"]
    # run_loop writes its own logs to runs_dir/state.run_id -- put this report there too.
    run_directory = args.runs_dir / state.run_id
    run_directory.mkdir(parents=True, exist_ok=True)

    _progress("=== searching KuaiRand-1K (bench=1k) | baseline valid primary %.5f ===" % baseline_primary)
    try:
        summary = run_loop(
            state,
            max_iterations=args.max_iterations,
            max_wallclock_hours=args.max_wallclock_hours,
            data_dir=args.data_dir,
            runs_dir=args.runs_dir,
            cache_dir=args.cache_dir,
            finalize_on_test=False,
            bench="1k",
        )
    except LLMError as error:
        print("STOPPED: %s" % error, file=sys.stderr)
        return 1
    _progress("search complete: %s" % json.dumps(summary))

    band = _ACCEPTANCE_BAND["1k"]
    breakdown = _search_breakdown(state, baseline_primary, band)
    _progress(
        "search breakdown: %d accepted, %d rejected, %d no-op (didn't actually test their hypothesis), "
        "%d error, %d abandoned | best genuine attempt valid primary %s"
        % (
            breakdown["counts"]["accepted"], breakdown["counts"]["rejected"], breakdown["counts"]["no_op"],
            breakdown["counts"]["error"], breakdown["counts"]["abandoned"],
            "n/a" if breakdown["best_genuine_valid_primary"] is None else "%.5f" % breakdown["best_genuine_valid_primary"],
        )
    )

    report = {
        "baseline_1k_valid_primary": baseline_primary,
        "search_summary": summary,
        "search_breakdown": breakdown,
        "accepted_any_improvement": _has_accepted(state),
        "replication": None,
        "final_test": None,
        "checkpoint_path": None,
        "verdict": None,
    }

    if not report["accepted_any_improvement"]:
        best_genuine = breakdown["best_genuine_valid_primary"]
        best_delta = breakdown["best_genuine_delta_vs_baseline"]
        if best_genuine is None:
            genuine_line = (
                "Every scored attempt this run was a no-op or an error -- none actually tested a "
                "different computation, so this run produced no real evidence either way."
            )
        else:
            genuine_line = (
                "Best genuinely-tested attempt this run: valid primary %.5f (delta %+.5f vs. baseline). "
                "The acceptance band requires %+.4f -- the same value Pure uses (2x its measured 5-seed "
                "noise), reused here because it's the exact band that already found 1K's one confirmed, "
                "3-seed-replicated win (+0.0018 single-run). A delta below this is inside measurement "
                "noise, not a real effect this run failed to reach." % (best_genuine, best_delta, band)
            )
        report["verdict"] = (
            "No candidate beat the 1K baseline (valid primary %.5f) inside this run's budget. %s This "
            "is consistent with ONEK_RESULTS.md's own record of ~12 prior negative axes on this exact "
            "benchmark -- the baseline itself is the maximized result found here, not a failure of "
            "search effort. See ONEK_RESULTS.md's 'What this means for the 1K recommendation' for the "
            "genuinely untested axes worth trying next (CatBoost YetiRank's cat_features fix; "
            "content/side-information features)." % (baseline_primary, genuine_line)
        )
    else:
        _progress("=== replicating the winning candidate over seeds %s ===" % seeds)
        runs = _replicate(state.current_code, args.data_dir, args.cache_dir, _BENCH_TIMEOUT_S["1k"], seeds)
        primaries = [r["valid"]["primary"] for r in runs]
        mean_primary = sum(primaries) / len(primaries) if primaries else None
        report["replication"] = {"runs": runs, "mean_valid_primary": mean_primary}
        if mean_primary is None:
            report["verdict"] = "All replication seeds failed to score; the search-time accept is unconfirmed."
        elif mean_primary > baseline_primary + band:
            report["verdict"] = (
                "CONFIRMED: %d-seed mean valid primary %.5f beats the 1K baseline %.5f by more than the "
                "acceptance band %.4f. This maximizes 1K's score within this run's budget."
                % (len(primaries), mean_primary, baseline_primary, band)
            )
        else:
            report["verdict"] = (
                "NOT CONFIRMED: %d-seed mean valid primary %.5f does not clear the 1K baseline %.5f + band "
                "%.4f. This is exactly the single-run trap the replication gate exists to catch (see "
                "ONEK_RESULTS.md Phase 10's lr_0.0005 and Phase 19's xgb_ndcg, both of which looked like real "
                "leads on one seed and evaporated on replication). The 1K baseline stands as the maximized "
                "result found this run." % (len(primaries), mean_primary, baseline_primary, band)
            )
    _progress(report["verdict"])

    if not args.skip_final_test:
        _progress("scoring final test metrics (seed 0) ...")
        # Save the trained weights only for a genuine convergence, not a
        # cap-hit -- see scripts/run_agent.py's run_loop for the same rule and
        # rationale. checkpoint_path is best-effort: state.current_code not
        # accepting it just means checkpoint_saved comes back False, never an
        # error (agent/runner.score_final_on_test's docstring).
        checkpoint_path = run_directory / "model_checkpoint.npz"
        converged = summary.get("stopping_reason") == "converged"
        final_result = score_final_on_test(
            state.current_code, args.data_dir, args.cache_dir, _BENCH_TIMEOUT_S["1k"], bench="1k",
            checkpoint_path=checkpoint_path if converged else None,
        )
        if final_result["status"] == "ok":
            report["final_test"] = final_result["metrics"]["test"]
            checkpoint_saved = bool(final_result.get("checkpoint_saved"))
            report["checkpoint_path"] = str(checkpoint_path) if checkpoint_saved else None
            _progress(
                "final test | %s%s"
                % (
                    _fmt_metrics(final_result["metrics"]["test"]),
                    (" | checkpoint saved to %s" % checkpoint_path) if checkpoint_saved else "",
                )
            )
        else:
            report["final_test"] = {"status": final_result["status"], "error_trace": final_result.get("error_trace")}
            _progress("final test scoring FAILED: %s" % final_result.get("error_trace", final_result["status"]))

    (run_directory / "maximize_1k_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
