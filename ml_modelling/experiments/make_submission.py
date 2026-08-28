"""Train the KB's recommended config and emit a submission the official tools accept.

This closes the loop: a recommendation that cannot produce a legal submission is
not a recommendation. Writes the file in the starter kit's exact row order and then
hands it to `submit.py --check` (and `--score` on valid) for adjudication.

  python experiments/make_submission.py --split valid    # check + score locally
  python experiments/make_submission.py --split test     # the real submission
"""
import os, sys, csv, argparse, subprocess
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from explib import dataset as D, features as F, fm, harness as H

KIT = os.path.join(D.REPO, 'kuairand-starter-kit')

# The KB's recommended operating point (knowledge_base.yaml -> candidate_models.fm_bpr)
RECOMMENDED = dict(loss='bpr', lr=0.0002, k=6, l2=1e-6, epochs=60, patience=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='valid', choices=['valid', 'test'])
    ap.add_argument('--out', default=None)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    out = a.out or os.path.join(D.REPO, f'submission_{a.split}.csv')
    logs = D.load_logs()
    masks = D.split_slices(logs)
    enc, dim, _ = F.encode_splits(logs, masks, F.BASELINE_FIELDS)

    print(f'training recommended config: {RECOMMENDED}')
    m, info = fm.train(enc, dim, loss=RECOMMENDED['loss'], k=RECOMMENDED['k'],
                       lr=RECOMMENDED['lr'], l2=RECOMMENDED['l2'],
                       epochs=RECOMMENDED['epochs'], patience=RECOMMENDED['patience'],
                       seed=a.seed, evaluator=H.score, verbose=False)
    print(f"  best epoch {info['best_epoch']}, valid primary {info['best_valid_primary']}")

    X, y, u = enc[a.split]
    scores = m.predict(X)
    assert np.isfinite(scores).all(), 'non-finite score would be rejected by submit.py'

    mask = masks[a.split]
    uid = logs['user_id'][mask]
    vid = logs['video_id'][mask]
    with open(out, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['row_id', 'user_id', 'video_id', 'score'])
        for i in range(len(scores)):
            w.writerow([i, int(uid[i]), int(vid[i]), f'{float(scores[i]):.6g}'])
    print(f'wrote {out} ({len(scores)} rows)')

    # Adjudicate with the official tool, not with our own opinion of correctness.
    for flag in (['--check'], ['--score'] if a.split == 'valid' else None):
        if flag is None:
            continue
        cmd = [sys.executable, 'submit.py', *flag, '--split', a.split, out]
        print(f"\n$ {' '.join(flag)} --split {a.split}")
        # submit.py prints a checkmark and Chinese text; on a cp1252 console that
        # raises on the print, long after validation itself has succeeded.
        env = dict(os.environ, PYTHONIOENCODING='utf-8')
        r = subprocess.run(cmd, cwd=KIT, capture_output=True, text=True,
                           encoding='utf-8', env=env)
        print((r.stdout or '').strip() or (r.stderr or '').strip())
        if r.returncode != 0:
            print(f'  submit.py exited {r.returncode}')
            return r.returncode
    return 0


if __name__ == '__main__':
    sys.exit(main())
