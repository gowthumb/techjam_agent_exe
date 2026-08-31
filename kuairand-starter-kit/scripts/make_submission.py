#!/usr/bin/env python3
"""Generate submission.csv (+ a checkpoint) directly from any completed run's winning code.

Generalizes make_pure_submission.py / make_1k_submission.py, which only know
about the two hand-extracted "best" candidate files -- this reads
runs/<run_id>/state.json's current_code straight from a run and scores it,
so a fresh scripts/run_agent.py or scripts/maximize_1k.py run doesn't need its
winning code manually copied into a new best_*_candidate.py file first.

Scores state.current_code ONCE on the real test split via
agent/runner.score_final_on_test -- the same subprocess-isolated path the
orchestrator itself uses, at seed 0 (the loop's search never varies seed
during iteration, so this reproduces exactly the code that was accepted) --
and saves a weights checkpoint alongside it. Writes the submission through
submit.py's own format functions, then independently re-reads and re-scores
it as a check.

Run:  python scripts/make_submission.py --run-id <id> --bench pure
      python scripts/make_submission.py --run-id <id> --bench 1k
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.executor import _BENCH_DATA_DIR, _BENCH_TIMEOUT_S  # noqa: E402
from agent.runner import score_final_on_test  # noqa: E402
from agent.state import RunState  # noqa: E402
from evaluate import evaluate  # noqa: E402
from submit import read_submission, write_submission  # noqa: E402


def _pure_rows(data_dir: Path):
    from data import load
    return load(str(data_dir))["test"]


def _1k_rows():
    from data_1k import raw_rows
    user_ids, video_ids, labels = raw_rows("test")
    n = len(user_ids)
    return list(zip(
        [None] * n, [str(u) for u in user_ids.tolist()], [str(v) for v in video_ids.tolist()],
        [None] * n, [None] * n, [None] * n, labels.tolist(),
    ))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", required=True, help="Directory name under runs/ to score.")
    ap.add_argument("--bench", required=True, choices=["pure", "1k"])
    ap.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    ap.add_argument("--cache-dir", type=Path, default=ROOT / ".cache")
    ap.add_argument("--out", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    state_path = a.runs_dir / a.run_id / "state.json"
    if not state_path.exists():
        print(f"no such run: {state_path}", file=sys.stderr)
        return 1
    state = RunState.load(state_path)
    out = a.out or str(ROOT / f"submission_{a.bench}_test.csv")
    checkpoint_path = a.checkpoint or str(ROOT / "checkpoints" / f"{a.bench}_{a.run_id}_checkpoint.npz")

    print(f"run {a.run_id}: best valid {state.best_metrics}")
    print(f"scoring on bench={a.bench} test split (seed={a.seed}) ...")
    result = score_final_on_test(
        state.current_code, _BENCH_DATA_DIR[a.bench], a.cache_dir, _BENCH_TIMEOUT_S[a.bench],
        bench=a.bench, seed=a.seed, checkpoint_path=checkpoint_path,
    )
    if result["status"] != "ok":
        print("FAILED:", result.get("error_trace", result["status"]), file=sys.stderr)
        return 1
    t = result["metrics"]["test"]
    print(f"  test  GAUC {t['GAUC']:.4f} | nDCG@5 {t['nDCG@5']:.4f} | primary {t['primary']:.4f}")
    if result.get("checkpoint_saved"):
        print(f"  saved checkpoint to {checkpoint_path}")
    else:
        print("  (no checkpoint saved: this candidate's run_fm doesn't accept checkpoint_path)")

    rows = _pure_rows(_BENCH_DATA_DIR["pure"]) if a.bench == "pure" else _1k_rows()
    if len(result["test_scores"]) != len(rows):
        print(
            f"test_scores length {len(result['test_scores'])} != row count {len(rows)} -- "
            "row-order assumption broke; not writing a submission.", file=sys.stderr,
        )
        return 1

    write_submission(out, rows, result["test_scores"])
    print(f"wrote {out} ({len(rows):,} rows)")

    scores = read_submission(out, rows)
    r = evaluate([x[1] for x in rows], [x[6] for x in rows], scores)
    print(f"check: format + alignment OK | re-scored primary {r['primary']:.4f}")
    if abs(r["primary"] - t["primary"]) > 1e-6:
        print(f"WARNING: re-read score ({r['primary']:.6f}) != training run's own score ({t['primary']:.6f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
