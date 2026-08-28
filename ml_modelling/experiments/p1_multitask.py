"""Phase 1 / axis C: auxiliary feedback signals as multi-task supervision.

Only long_view is scored, but the log carries 11 more signals plus play_time_ms.
Each is free supervision on the *same* shared embedding table. The question the KB
needs answered is not "does multi-task help" in the abstract but *which signals*
help, which are inert, and which actively fight the primary task.

Control: T=1 with A frozen at ones is algebraically the starter kit's FM, so the
single-task row here must land on the baseline. Any auxiliary's effect is then
measured against that same code path.

Run:  python experiments/p1_multitask.py [--only name,...]
"""
import os, sys, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from explib import dataset as D, features as F, mtfm, harness as H

# name -> (column builder, description)
AUX = {
    'is_click':         lambda lg: (lg['is_click'] != 0).astype(np.float32),
    'is_like':          lambda lg: (lg['is_like'] != 0).astype(np.float32),
    'is_follow':        lambda lg: (lg['is_follow'] != 0).astype(np.float32),
    'is_comment':       lambda lg: (lg['is_comment'] != 0).astype(np.float32),
    'is_forward':       lambda lg: (lg['is_forward'] != 0).astype(np.float32),
    'is_hate':          lambda lg: (lg['is_hate'] != 0).astype(np.float32),
    'is_profile_enter': lambda lg: (lg['is_profile_enter'] != 0).astype(np.float32),
    # watch-time derived: did the view reach the end of the clip
    'play_complete':    lambda lg: (lg['play_time_ms'] >= lg['duration_ms']).astype(np.float32),
    # watch-time derived: watched more than half the clip
    'play_half':        lambda lg: (lg['play_time_ms'] >= 0.5 * np.maximum(lg['duration_ms'], 1)
                                    ).astype(np.float32),
}

SINGLES = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward',
           'is_hate', 'is_profile_enter', 'play_complete', 'play_half']

HYP_SINGLE = ('Auxiliary signal shares structure with long_view, so supervising it on the '
              'shared embedding table should regularize V and improve long_view ranking')
HYP_CTRL = ('CONTROL: T=1 with A frozen at ones is algebraically the kit FM, so this must '
            'land on the baseline; if it does not, the multi-task code path is not comparable')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None)
    ap.add_argument('--weight', type=float, default=0.3)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--combo', default=None,
                    help='comma list of aux names to train together')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()

    logs = D.load_logs()
    masks = D.split_slices(logs)
    enc, dim, _ = F.encode_splits(logs, masks, F.BASELINE_FIELDS)
    Xtr, ytr, _ = enc['train']
    Xva, yva, uva = enc['valid']
    Xte, yte, ute = enc['test']
    print(f'dim={dim}\n')

    # positive rate of each auxiliary on train -- a signal with almost no
    # positives cannot supply gradient, and that is worth recording, not guessing
    tr = masks['train']
    rates = {n: float(AUX[n](logs)[tr].mean()) for n in SINGLES}
    print('auxiliary positive rates (train): ' +
          ', '.join(f'{n}={r:.4f}' for n, r in rates.items()) + '\n')

    plans = []
    if a.combo:
        names = a.combo.split(',')
        plans.append((f'1C-combo-{"+".join(names)}-w{a.weight:g}', names, HYP_SINGLE))
    else:
        plans.append(('1C-control-singletask', [], HYP_CTRL))
        for n in SINGLES:
            plans.append((f'1C-aux-{n}-w{a.weight:g}', [n], HYP_SINGLE))

    for exp_id, names, hyp in plans:
        if a.only and not any(s in exp_id for s in a.only.split(',')):
            continue
        cols = [(logs['long_view'] != 0).astype(np.float32)] + [AUX[n](logs) for n in names]
        Y = np.stack(cols, axis=1)
        Ytr = Y[tr]
        weights = [1.0] + [a.weight] * len(names)
        cfg = dict(model='mtfm', k=16, lr=a.lr, l2=1e-6, bs=8192, epochs=a.epochs,
                   patience=4, seed=a.seed, fields=F.BASELINE_FIELDS,
                   aux_tasks=names, task_weights=weights,
                   aux_positive_rate={n: round(rates[n], 5) for n in names},
                   learn_A=bool(names))
        with H.Experiment(exp_id, phase='1C', axis='multi_task',
                          hypothesis=hyp, config=cfg,
                          tags=['multitask'] + names) as ex:
            m, info = mtfm.train(Xtr, Ytr, Xva, yva, uva, dim, weights,
                                 k=16, lr=a.lr, epochs=a.epochs, seed=a.seed,
                                 evaluator=H.score, verbose=not a.quiet,
                                 learn_A=bool(names))
            ex.record_train(**{k: v for k, v in info.items() if k != 'history'})
            ex.record_train(history=info['history'])
            ex.record_metrics('valid', H.score(uva, yva, m.predict(Xva)))
            ex.record_metrics('test', H.score(ute, yte, m.predict(Xte)))
        print()

    print(H.summarize())


if __name__ == '__main__':
    main()
