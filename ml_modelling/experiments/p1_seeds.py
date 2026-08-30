"""Phase 1 / axis H: how much of what we are reading is noise.

Every verdict in this workstream leans on a noise band. The official figure is a
test std of 0.0008 over 5 seeds for the baseline FM -- but nothing published says
what the *valid* std is, and valid is the split every decision is actually made on.
Valid is 125K rows over 7 days against test's 171K over 10, so its std is very
plausibly larger, which would mean valid-selected verdicts are noisier than the
band we have been applying.

This measures both, for the pointwise control and for the BPR variant, over 5
seeds. The result sets the KB's noise band and tells the agent how many replicates
a claimed improvement needs before it is believable.

Run:  python experiments/p1_seeds.py
"""
import os, sys, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from explib import dataset as D, features as F, fm, harness as H

HYP = ('Valid is 27% smaller than test and spans 3 fewer days, so its seed-to-seed std '
       'should be at least as large as the official 0.0008 test std; if it is materially '
       'larger, the decision threshold used on valid must be widened accordingly')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='0,1,2,3,4')
    ap.add_argument('--configs', default='pointwise,bpr')
    a = ap.parse_args()

    logs = D.load_logs()
    masks = D.split_slices(logs)
    enc, dim, _ = F.encode_splits(logs, masks, F.BASELINE_FIELDS)
    seeds = [int(s) for s in a.seeds.split(',')]

    SETUPS = {
        'pointwise': dict(loss='pointwise', lr=0.001, epochs=40, patience=4),
        'bpr':       dict(loss='bpr', lr=0.0002, epochs=60, patience=6),
    }

    collected = {}
    for name in a.configs.split(','):
        s = SETUPS[name]
        vals = {'valid': [], 'test': []}
        for seed in seeds:
            cfg = dict(model='fm', k=16, l2=1e-6, bs=8192, seed=seed,
                       fields=F.BASELINE_FIELDS, **s)
            with H.Experiment(f'1H-seedstudy-{name}-seed{seed}', phase='1H',
                              axis='seed_variance', hypothesis=HYP, config=cfg,
                              tags=['seeds', name]) as ex:
                m, info = fm.train(enc, dim, loss=s['loss'], k=16, lr=s['lr'],
                                   epochs=s['epochs'], patience=s['patience'],
                                   seed=seed, evaluator=H.score, verbose=False)
                ex.record_train(**{k: v for k, v in info.items() if k != 'history'})
                for sp in ('valid', 'test'):
                    X, y, u = enc[sp]
                    mm = H.score(u, y, m.predict(X))
                    ex.record_metrics(sp, mm)
                    vals[sp].append(mm['primary'])
        collected[name] = vals

    print('\n=== seed variance ===')
    print(f"{'config':12s} {'split':6s} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}  n")
    for name, vals in collected.items():
        for sp in ('valid', 'test'):
            v = np.array(vals[sp])
            print(f'{name:12s} {sp:6s} {v.mean():8.4f} {v.std(ddof=1):8.4f} '
                  f'{v.min():8.4f} {v.max():8.4f}  {len(v)}')

    if len(collected) == 2:
        for sp in ('valid', 'test'):
            p = np.array(collected['pointwise'][sp])
            b = np.array(collected['bpr'][sp])
            d = b.mean() - p.mean()
            se = np.sqrt(p.var(ddof=1) / len(p) + b.var(ddof=1) / len(b))
            print(f'\nbpr - pointwise on {sp}: {d:+.4f}  (se {se:.4f}, '
                  f'{"separated" if abs(d) > 2 * se else "NOT separated"} at 2 se)')


if __name__ == '__main__':
    main()
