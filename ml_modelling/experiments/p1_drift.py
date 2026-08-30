"""Phase 1 / axis G: temporal drift.

Two measured facts motivate this axis:
  * the long_view base rate falls from 0.3366 on train to 0.3134 on valid/test;
  * training volume is wildly front-loaded -- 59% of the 1.14M train rows land in
    just three days (2022-04-10..12), while the eval period runs at ~15K rows/day.
So the FM is fit mostly on a fortnight-old, higher-engagement traffic regime and
then asked to rank a later, quieter one.

Meanwhile ID coverage is essentially total (99.8% of test videos and 96.7% of test
users appear in train), so this is a drift problem, not a cold-start problem.

Two interventions, both cheap:
  recency weighting  -- weight row i by exp(-(t_end - t_i)/tau) so recent traffic
                        dominates the gradient without discarding old rows;
  recent window      -- train only on the last N days.

Run:  python experiments/p1_drift.py
"""
import os, sys, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from explib import dataset as D, features as F, fm, harness as H

HYP_W = ('Training is dominated by early high-volume days while evaluation is a later, '
         'lower-engagement regime; down-weighting stale rows should align the fitted '
         'distribution with the evaluated one')
HYP_D = ('If drift is what limits the baseline, dropping stale days outright should help '
         'despite the large loss of training rows; if it hurts monotonically, volume '
         'matters more than recency and the KB should say so')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--taus', default='2,4,7,14',
                    help='recency half-life in days; comma list')
    ap.add_argument('--windows', default='3,5,7,10',
                    help='train only on the last N days; comma list')
    ap.add_argument('--loss', default='pointwise')
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--skip', default='', help='comma list: taus,windows')
    a = ap.parse_args()

    logs = D.load_logs()
    masks = D.split_slices(logs)
    enc, dim, _ = F.encode_splits(logs, masks, F.BASELINE_FIELDS)
    Xtr, ytr, utr = enc['train']

    # day index within train, 0 = oldest
    tr_dates = logs['date'][masks['train']]
    days = np.unique(tr_dates)
    last = days.max()
    age_days = np.array([np.searchsorted(days, last) - np.searchsorted(days, d)
                         for d in tr_dates], dtype=np.float32)
    print(f'train spans {len(days)} days; rows per day: '
          f'{[int((tr_dates == d).sum()) for d in days]}\n')

    skip = set(a.skip.split(',')) if a.skip else set()

    if 'taus' not in skip:
        for tau in [float(x) for x in a.taus.split(',')]:
            w = np.exp(-age_days / tau).astype(np.float32)
            w = w / w.mean()                      # keep the average step size fixed
            cfg = dict(model='fm', loss=a.loss, k=16, lr=a.lr, l2=1e-6, bs=8192,
                       epochs=a.epochs, patience=4, seed=a.seed,
                       fields=F.BASELINE_FIELDS, recency_tau_days=tau,
                       effective_rows=float(w.sum() ** 2 / (w ** 2).sum()))
            with H.Experiment(f'1G-recency-tau{tau:g}', phase='1G', axis='temporal_drift',
                              hypothesis=HYP_W, config=cfg, tags=['drift', 'recency']) as ex:
                m, info = fm.train(enc, dim, loss=a.loss, k=16, lr=a.lr,
                                   epochs=a.epochs, seed=a.seed, evaluator=H.score,
                                   verbose=False, row_weight=w)
                ex.record_train(**{k: v for k, v in info.items() if k != 'history'})
                ex.record_train(history=info['history'])
                for sp in ('valid', 'test'):
                    X, y, u = enc[sp]
                    ex.record_metrics(sp, H.score(u, y, m.predict(X)))

    if 'windows' not in skip:
        for win in [int(x) for x in a.windows.split(',')]:
            keep = age_days < win
            sub = dict(enc)
            sub['train'] = (Xtr[keep], ytr[keep], utr[keep])
            cfg = dict(model='fm', loss=a.loss, k=16, lr=a.lr, l2=1e-6, bs=8192,
                       epochs=a.epochs, patience=4, seed=a.seed,
                       fields=F.BASELINE_FIELDS, train_window_days=win,
                       train_rows=int(keep.sum()),
                       train_rows_pct=round(100 * float(keep.mean()), 1))
            with H.Experiment(f'1G-window-{win}d', phase='1G', axis='temporal_drift',
                              hypothesis=HYP_D, config=cfg, tags=['drift', 'window']) as ex:
                m, info = fm.train(sub, dim, loss=a.loss, k=16, lr=a.lr,
                                   epochs=a.epochs, seed=a.seed, evaluator=H.score,
                                   verbose=False)
                ex.record_train(**{k: v for k, v in info.items() if k != 'history'})
                ex.record_train(history=info['history'])
                for sp in ('valid', 'test'):
                    X, y, u = enc[sp]
                    ex.record_metrics(sp, H.score(u, y, m.predict(X)))

    print(H.summarize([r for r in H.read_log() if r['phase'] == '1G']))


if __name__ == '__main__':
    main()
