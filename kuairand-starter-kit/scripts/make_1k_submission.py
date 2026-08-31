#!/usr/bin/env python3
"""Generate the KuaiRand-1K test submission (bonus benchmark) from best_1k_candidate.py.

best_1k_candidate.py is the winning candidate from agent run
0f6a4083fba54b798c7bb6f87c0a73a9 (see that file's own docstring for the full
provenance) -- sparse Adagrad instead of sparse Adam, the one 1K result in
this codebase's entire research history that beat the untuned baseline AND
survived 3-seed replication (mean valid +0.0024 vs. baseline).

Trains run_fm(splits, seed=0, return_predictions=True) -- seed 0, matching the
replication run's own seed 0 exactly, so this reproduces a specific already-
scored result rather than a fresh unreplicated one -- and writes the
submission in submit.py's exact schema (row_id,user_id,video_id,score), using
submit.py's own write_submission()/read_submission() so the format is
verifiably identical to the Pure submission's, not a reimplementation. Also
saves the trained model's weights (checkpoint.py) so the run's actual model
state is persisted, not just the code that reproduces it.

data_1k.py's opaque splits handle doesn't carry raw (user_id, video_id) rows
the way data.py's does for Pure (see that module's own docstring for why) --
data_1k.raw_rows("test") reconstructs them in encode()'s exact row order.

Run:  python scripts/make_1k_submission.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_1k import load, raw_rows  # noqa: E402
from submit import read_submission, write_submission  # noqa: E402
from evaluate import evaluate  # noqa: E402
import best_1k_candidate as candidate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "submission_1k_test.csv"))
    ap.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "1k_checkpoint.npz"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    print("loading KuaiRand-1K ...")
    splits = load(None)
    user_ids, video_ids, labels = raw_rows("test")
    # submit.py's write_submission/read_submission/evaluate() were built for
    # data.py's 7-tuple row shape (date, user_id, video_id, author_id, tab,
    # duration_ms, label) and only ever index positions 1, 2, and 6 -- reuse
    # them unmodified by building rows in that same shape rather than
    # reimplementing CSV writing or alignment checking for 1K. data.py's own
    # user_id/video_id come straight from csv.DictReader, i.e. strings, and
    # read_submission() compares against whatever the CSV round-trip produces
    # (also strings) -- str() these explicitly so int64 vs. "same value as a
    # string" doesn't look like a misalignment.
    user_id_strs = [str(u) for u in user_ids.tolist()]
    video_id_strs = [str(v) for v in video_ids.tolist()]
    rows = list(zip(
        [None] * len(user_ids), user_id_strs, video_id_strs,
        [None] * len(user_ids), [None] * len(user_ids), [None] * len(user_ids), labels.tolist(),
    ))
    print(f"  {len(rows):,} test rows")

    print(f"training best_1k_candidate.run_fm (sparse Adagrad, seed={a.seed}) ...")
    res = candidate.run_fm(splits, seed=a.seed, return_predictions=True, checkpoint_path=a.checkpoint)
    print(f"saved trained weights to {a.checkpoint}")
    v, t = res["valid"], res["test"]
    print(f"  valid  GAUC {v['GAUC']:.4f} | nDCG@5 {v['nDCG@5']:.4f} | primary {v['primary']:.4f}")
    print(f"  test   GAUC {t['GAUC']:.4f} | nDCG@5 {t['nDCG@5']:.4f} | primary {t['primary']:.4f}")

    test_scores = res["test_scores"]
    if len(test_scores) != len(rows):
        raise ValueError(
            "test_scores length %d != raw test row count %d -- row-order assumption "
            "between data_1k.encode() and data_1k.raw_rows() broke; do not submit "
            "this file." % (len(test_scores), len(rows))
        )

    write_submission(a.out, rows, test_scores)
    print(f"\nwrote {a.out} ({len(rows):,} rows)")

    scores = read_submission(a.out, rows)
    print(f"check: format + alignment OK, {len(scores):,} rows")
    r = evaluate([x[1] for x in rows], [x[6] for x in rows], scores)
    print(f"score: GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
    if abs(r["primary"] - t["primary"]) > 1e-6:
        print(
            "WARNING: re-read score (%.6f) does not match the training run's own test score "
            "(%.6f) -- investigate before submitting." % (r["primary"], t["primary"])
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
