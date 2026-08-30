"""Phase 1 / axis A: loss function alignment.

Hypothesis (organizer-flagged as the most likely headroom): the baseline FM
optimizes pointwise logloss, but the metric is a within-user *ranking* metric.
Global calibration is irrelevant to a within-user ranking, so a loss that only
constrains within-user order should extract more signal from the same model,
same features, same capacity.

Control: the same FM/encoder/optimizer trained with pointwise logloss must
reproduce the official baseline (valid primary 0.6016). If the control drifts,
nothing downstream is trustworthy.

The ranking losses are swept over lr because their gradient scale differs from
logloss by construction -- comparing them at the baseline's lr only would
confound "wrong loss" with "wrong step size".

Run:  python experiments/p1_losses.py [--only substr,...] [--seed N]
"""
import os, sys, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from explib import dataset as D, features as F, fm, harness as H

H_POINT = 'CONTROL: my FM+encoder with pointwise logloss reproduces the official baseline'
H_LIST = ('Within-user softmax CE optimizes the same within-user ordering the metric '
          'scores, so it should beat pointwise logloss at equal capacity')
H_BPR = ('Within-user pairwise BPR optimizes exactly the quantity GAUC measures '
         '(P(pos ranked above neg) within a user), so it should beat pointwise')
H_HYB = ('Pointwise supplies a dense, well-conditioned signal for rare ids while '
         'listwise supplies the rank signal; the sum should beat either alone')

RUNS = [
    ('1A-control-pointwise',  'pointwise', dict(lr=0.001),              H_POINT),
    ('1A-listwise-lr0.001',   'listwise',  dict(lr=0.001),              H_LIST),
    ('1A-listwise-lr0.005',   'listwise',  dict(lr=0.005),              H_LIST),
    ('1A-listwise-lr0.02',    'listwise',  dict(lr=0.02),               H_LIST),
    ('1A-bpr-lr0.001',        'bpr',       dict(lr=0.001),              H_BPR),
    ('1A-bpr-lr0.005',        'bpr',       dict(lr=0.005),              H_BPR),
    ('1A-bpr-lr0.02',         'bpr',       dict(lr=0.02),               H_BPR),
    ('1A-hybrid-lam1',        'hybrid',    dict(lr=0.001, lam=1.0),     H_HYB),
    ('1A-hybrid-lam0.3',      'hybrid',    dict(lr=0.001, lam=0.3),     H_HYB),
]

TRAIN_ONLY = {'lam', 'pairs_per_pos', 'users_per_batch'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None, help='comma-separated exp_id substrings')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=40)
    a = ap.parse_args()

    logs = D.load_logs()
    masks = D.split_slices(logs)
    enc, dim, _ = F.encode_splits(logs, masks, F.BASELINE_FIELDS)
    rows = {k: len(v[1]) for k, v in enc.items()}
    print(f"fields={F.BASELINE_FIELDS} dim={dim} rows={rows}\n")

    for exp_id, loss, kw, hyp in RUNS:
        if a.only and not any(s in exp_id for s in a.only.split(',')):
            continue
        eid = exp_id + (f'-seed{a.seed}' if a.seed else '')
        cfg = dict(model='fm', loss=loss, k=16, l2=1e-6, bs=8192,
                   epochs=a.epochs, patience=4, seed=a.seed,
                   fields=F.BASELINE_FIELDS, **kw)
        with H.Experiment(eid, phase='1A', axis='loss_function',
                          hypothesis=hyp, config=cfg, tags=['loss', loss]) as ex:
            m, info = fm.train(enc, dim, loss=loss, k=cfg['k'], l2=cfg['l2'],
                               epochs=a.epochs, bs=cfg['bs'], patience=cfg['patience'],
                               seed=a.seed, evaluator=H.score, verbose=True, **kw)
            ex.record_train(**{k: v for k, v in info.items() if k != 'history'})
            ex.record_train(history=info['history'])
            for sp in ('valid', 'test'):
                X, y, u = enc[sp]
                ex.record_metrics(sp, H.score(u, y, m.predict(X)))
        print()

    print(H.summarize())


if __name__ == '__main__':
    main()
