#!/usr/bin/env python3
"""1K-first, 27K-confirmation pipeline for improving KuaiRand-27K's score.

Per the discussion this script implements: don't run the LLM-driven
Planner/Coder/Debugger/Executor loop against KuaiRand-27K directly. 27K costs
~2.5h of compute per attempt even with data cached (load alone is ~34min,
knowledge_base/ONEK_RESULTS.md Phase 20) -- there is no budget inside a 6h
ceiling for the search itself, let alone the 3-seed replication this codebase's
own decision protocol requires before trusting a lead (two single-seed leads
have already evaporated on replication: Phase 10's lr_0.0005, Phase 19's
xgb_ndcg). 1K costs ~1min/epoch and sits in the same item-cold-start regime, so
it is where a hypothesis should be tried, accepted or rejected, and replicated
-- 27K is reserved for a single, explicitly-labeled confirmation run of
whatever survives that gate.

Stages:
  1. Iterate the standard agent loop against KuaiRand-1K (bench="1k"), gated on
     1K's own actually-measured baseline (run once here, not a hardcoded
     number).
  2. If any candidate was accepted, replicate the final winning code over
     --replication-seeds seeds (default 0,1,2) using the SAME code, no new
     patch -- runner.run(..., seed=N). Promote only if the seed-mean still
     clears the acceptance band against the 1K baseline. This is the exact
     discipline that caught the two false leads above; skipping it here would
     repeat that mistake at 27K's cost instead of 1K's.
  3. If promoted, mechanically retarget the winning code from data_1k to
     data_27k (only the import line changes -- both modules share the same
     load/encode/FIELDS contract) and run it ONCE on KuaiRand-27K via
     agent.runner.score_confirm, reporting valid AND test explicitly labeled as
     a single seed, not replicated -- exactly how ONEK_RESULTS.md Phase 20
     reported its own only 27K number, and for the same reason: 27K's own seed
     noise has never been measured, so a single run here is one data point, not
     a confidence-intervaled result.

Run:  python scripts/run_agent_scaled.py --max-iterations 12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import monotonic
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))

from agent.executor import _ACCEPTANCE_BAND, _BENCH_DATA_DIR, _BENCH_TIMEOUT_S  # noqa: E402
from agent.llm_client import LLMError, reset_quota_pause_budget  # noqa: E402
from agent.runner import run as run_candidate_code  # noqa: E402
from agent.runner import score_confirm  # noqa: E402
from agent.state import RunState  # noqa: E402
from run_agent import _fmt_metrics, _new_state, _progress, _short, run_loop  # noqa: E402


def _has_accepted(state: RunState) -> bool:
    return any(entry.get("status") == "accepted" for entry in state.experiment_history)


def _replicate(
    code: str, data_dir: Path, cache_dir: Optional[Path], timeout_s: float, bench: str, seeds: list[int]
) -> list[dict]:
    """Run the SAME code over several seeds via runner.run (validation only)."""
    runs = []
    for seed in seeds:
        _progress("replication seed %d starting (bench=%s) ..." % (seed, bench.upper()))
        result = run_candidate_code(code, data_dir, cache_dir, timeout_s, bench=bench, seed=seed)
        if result["status"] != "ok":
            _progress("replication seed %d FAILED: %s" % (seed, result.get("error_trace", result["status"])))
            continue
        primary = result["metrics"]["valid"]["primary"]
        _progress("replication seed %d | valid primary %.5f" % (seed, primary))
        runs.append({"seed": seed, "valid": result["metrics"]["valid"]})
    return runs


def _promote_code_to_27k(code: str) -> Optional[str]:
    """Mechanically retarget a winning 1K candidate at 27K's data module.

    Only the import line differs between data_1k.py and data_27k.py (identical
    load/encode/FIELDS contract -- see data_27k.py's docstring), so this is a
    literal string substitution, not a re-patch. Returns None if the expected
    import line isn't found verbatim, rather than guessing at a rewrite --
    silently promoting the wrong code would misattribute whatever the 27K run
    produces.
    """
    marker = "from data_1k import"
    if marker not in code:
        return None
    return code.replace(marker, "from data_27k import", 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id")
    parser.add_argument("--max-iterations", type=int, default=15, help="1K search budget (iterations).")
    parser.add_argument("--max-wallclock-hours", type=float, default=4.0,
                        help="1K search budget (hours). Left below 6h so a 27K confirmation, if promoted, "
                             "still fits inside a 6h total ceiling alongside it.")
    parser.add_argument("--replication-seeds", default="0,1,2",
                        help="Comma-separated seeds to replicate a 1K accept over before promoting it.")
    parser.add_argument("--data-dir-1k", type=Path, default=_BENCH_DATA_DIR["1k"])
    parser.add_argument("--data-dir-27k", type=Path, default=_BENCH_DATA_DIR["27k"])
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache")
    parser.add_argument("--skip-27k", action="store_true",
                        help="Stop after the 1K search + replication stage; never touch 27K's data/budget.")
    parser.add_argument("--force-27k", action="store_true",
                        help="Run the 27K confirmation on the 1K baseline even if no candidate was promoted "
                             "(re-establishes ONEK_RESULTS.md Phase 20's own number against current code).")
    args = parser.parse_args()
    seeds = [int(s) for s in args.replication_seeds.split(",") if s.strip()]

    reset_quota_pause_budget()

    # ---- Stage 1: iterate on 1K ----
    _progress("=== stage 1: searching KuaiRand-1K (bench=1k) ===")
    state_path = args.runs_dir / args.run_id / "state.json" if args.run_id else None
    if state_path is not None and state_path.exists():
        state = RunState.load(state_path)
    else:
        state = _new_state("1k", args.data_dir_1k, args.cache_dir)
        if args.run_id:
            state.run_id = args.run_id
    baseline_1k_primary = state.best_metrics["primary"]
    # run_loop (stage 1) writes its own logs to runs_dir/state.run_id -- put the
    # promotion record there too, whether or not --run-id was given, so
    # everything from one invocation of this script lives in one directory.
    run_directory = args.runs_dir / state.run_id
    run_directory.mkdir(parents=True, exist_ok=True)

    try:
        summary_1k = run_loop(
            state,
            max_iterations=args.max_iterations,
            max_wallclock_hours=args.max_wallclock_hours,
            data_dir=args.data_dir_1k,
            runs_dir=args.runs_dir,
            cache_dir=args.cache_dir,
            finalize_on_test=False,
            bench="1k",
        )
    except LLMError as error:
        print("STOPPED during 1K search: %s" % error, file=sys.stderr)
        return 1
    _progress("stage 1 complete: %s" % json.dumps(summary_1k))

    promotion = {
        "baseline_1k_valid_primary": baseline_1k_primary,
        "accepted_on_1k": _has_accepted(state),
        "replication": None,
        "promoted": False,
        "promotion_reason": None,
        "kuairand_27k_confirmation": None,
    }

    should_replicate = promotion["accepted_on_1k"]
    if not should_replicate:
        promotion["promotion_reason"] = "no candidate beat the 1K baseline; nothing to replicate or promote"
        _progress(promotion["promotion_reason"])

    if should_replicate:
        # ---- Stage 2: replicate the winning 1K candidate ----
        _progress("=== stage 2: replicating the winning 1K candidate over seeds %s ===" % seeds)
        runs = _replicate(state.current_code, args.data_dir_1k, args.cache_dir, _BENCH_TIMEOUT_S["1k"], "1k", seeds)
        primaries = [r["valid"]["primary"] for r in runs]
        mean_primary = sum(primaries) / len(primaries) if primaries else None
        promotion["replication"] = {"runs": runs, "mean_valid_primary": mean_primary}
        if mean_primary is None:
            promotion["promotion_reason"] = "all replication seeds failed to score; not promoting"
        elif mean_primary > baseline_1k_primary + _ACCEPTANCE_BAND:
            promotion["promoted"] = True
            promotion["promotion_reason"] = (
                "%d-seed mean valid primary %.5f clears the 1K baseline %.5f by more than the "
                "acceptance band %.4f" % (len(primaries), mean_primary, baseline_1k_primary, _ACCEPTANCE_BAND)
            )
        else:
            promotion["promotion_reason"] = (
                "%d-seed mean valid primary %.5f does NOT clear the 1K baseline %.5f + band %.4f -- "
                "this is exactly the single-run trap the replication gate exists to catch "
                "(see knowledge_base/ONEK_RESULTS.md Phase 10 and Phase 19)"
                % (len(primaries), mean_primary, baseline_1k_primary, _ACCEPTANCE_BAND)
            )
        _progress(promotion["promotion_reason"])

    # ---- Stage 3: confirm on 27K, only if promoted (or explicitly forced) ----
    run_confirmation = (promotion["promoted"] or args.force_27k) and not args.skip_27k
    if args.skip_27k:
        _progress("--skip-27k set: stopping before touching KuaiRand-27K.")
    elif run_confirmation:
        code_for_27k = _promote_code_to_27k(state.current_code) if promotion["promoted"] else (
            (ROOT / "baseline_27k.py").read_text(encoding="utf-8")
        )
        if promotion["promoted"] and code_for_27k is None:
            _progress(
                "promoted candidate's code no longer contains the literal 'from data_1k import' import line "
                "-- refusing to guess at a rewrite. Not running the 27K confirmation; retarget "
                "baseline_27k.py by hand with the same hypothesis instead."
            )
        else:
            _progress("=== stage 3: single confirmation run on KuaiRand-27K (bench=27k) ===")
            _progress("this can take multiple hours the first time (uncached load ~34min + training) -- see "
                      "knowledge_base/HARDWARE_AWARENESS.md's per-benchmark table.")
            started_at = monotonic()
            result = score_confirm(
                code_for_27k, args.data_dir_27k, args.cache_dir, _BENCH_TIMEOUT_S["27k"], bench="27k", seed=0,
            )
            elapsed_s = monotonic() - started_at
            if result["status"] != "ok":
                promotion["kuairand_27k_confirmation"] = {
                    "status": result["status"], "error_trace": result.get("error_trace"), "wall_time_s": elapsed_s,
                }
                _progress("27K confirmation FAILED (%s) after %.0fs: %s"
                          % (result["status"], elapsed_s, result.get("error_trace", "")))
            else:
                promotion["kuairand_27k_confirmation"] = {
                    "status": "ok",
                    "valid": result["metrics"]["valid"],
                    "test": result["metrics"]["test"],
                    "wall_time_s": elapsed_s,
                    "seed": 0,
                    "caution": (
                        "SINGLE SEED, NOT REPLICATED. 27K's own seed noise has never been measured "
                        "(1K's was already 4x Pure's). Treat this as one data point confirming the "
                        "config runs cleanly, not a tuned or confidence-intervaled result -- exactly "
                        "how ONEK_RESULTS.md Phase 20 reported its own number."
                    ),
                }
                _progress("27K confirmation done in %.0fs | valid %s | test %s"
                          % (elapsed_s, _fmt_metrics(result["metrics"]["valid"]), _fmt_metrics(result["metrics"]["test"])))
    else:
        _progress("no candidate promoted and --force-27k not set: skipping the 27K confirmation run.")

    (run_directory / "promotion.json").write_text(json.dumps(promotion, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"stage_1_summary": summary_1k, "promotion": promotion}, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
