#!/usr/bin/env python3
"""Generate the KuaiRand-Pure test submission from best_pure_candidate.py.

best_pure_candidate.py is the winning candidate from agent run
5cb9e936da024858815cd932df6ecfb7 (see that file's own docstring for the full
provenance and an important caveat about what was and wasn't actually
validated before it was accepted -- read it before treating this as final).

Trains run_fm(splits, seed=0, return_predictions=True) -- seed 0 because the
agent loop's search never varies seed during iteration (agent/executor.py),
so this reproduces exactly the run that was scored -- and writes the
submission via submit.py's own write_submission(), so the file is
byte-for-byte in the format submit.py --check/--score expects. Then runs that
same check-and-score pass so a fresh run and its own validation live in one
command.

Run:  python scripts/make_pure_submission.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import load  # noqa: E402
from submit import read_submission, write_submission  # noqa: E402
from evaluate import evaluate  # noqa: E402
import best_pure_candidate as candidate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "KuaiRand-Pure" / "data"))
    ap.add_argument("--out", default=str(ROOT / "submission_pure_test.csv"))
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "pure_checkpoint.npz"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    rows = splits["test"]

    print(f"training best_pure_candidate.run_fm (seed={a.seed}) ...")
    res = candidate.run_fm(splits, seed=a.seed, return_predictions=True, checkpoint_path=a.checkpoint)
    print(f"saved trained weights to {a.checkpoint}")
    v, t = res["valid"], res["test"]
    print(f"  valid  GAUC {v['GAUC']:.4f} | nDCG@5 {v['nDCG@5']:.4f} | primary {v['primary']:.4f}")
    print(f"  test   GAUC {t['GAUC']:.4f} | nDCG@5 {t['nDCG@5']:.4f} | primary {t['primary']:.4f}")

    write_submission(a.out, rows, res["test_scores"])
    print(f"\nwrote {a.out} ({len(rows):,} rows)")

    scores = read_submission(a.out, rows)
    print(f"check: format + alignment OK, {len(scores):,} rows")
    r = evaluate([x[1] for x in rows], [x[6] for x in rows], scores)
    print(f"score: GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
