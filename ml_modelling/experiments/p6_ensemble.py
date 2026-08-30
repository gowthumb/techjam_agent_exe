"""Phase 6 / axis: replication of the best cell, and ensembling what we already have.

Two jobs in one pass, because both need the same trained models:

  1. REPLICATION. The loss_x_capacity axis shows a single run at test 0.5983,
     nominally above the 5-seed BPR mean of 0.5980. Comparing a max to a mean is
     exactly the error this KB warns about, so every candidate cell gets the same
     5-seed treatment before any of them is called better.

  2. ENSEMBLING. Averaging costs no new training once the models are trained, and
     ranking ensembles reliably clear noise-band-sized gaps.

     Raw score averaging is WRONG across models whose logits have different scales
     (BPR never trains a global bias; pointwise does). The metric only reads
     within-user order, so scores are converted to within-user percentile ranks
     before averaging. Plain score averaging is reported alongside purely to show
     the difference.

Predictions are cached to .npz so the ensemble can be re-cut without retraining.

CPU only: numpy throughout, no GPU code path exists in this module.

Run:  python experiments/p6_ensemble.py --seeds 0,1,2,3,4
"""
import os, sys, json, argparse, itertools
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from explib import dataset as D, features as F, fm, harness as H

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
PRED = os.path.join(ROOT, 'cache', 'p6_preds')

# The cells worth replicating: the confirmed BPR point, the loss_x_capacity cells
# that edged it on a single run, and pointwise for ensemble diversity.
CELLS = {
    'bpr-k6':       dict(loss='bpr', k=6,  lr=0.0002, l2=1e-6, epochs=60, patience=6),
    'bpr-k8':       dict(loss='bpr', k=8,  lr=0.0002, l2=1e-6, epochs=60, patience=6),
    'bpr-k16':      dict(loss='bpr', k=16, lr=0.0002, l2=1e-6, epochs=60, patience=6),
    'pointwise-k16': dict(loss='pointwise', k=16, lr=0.001, l2=1e-6, epochs=40, patience=4),
}


def user_percentile(scores, users):
    """Within-user percentile rank in [0,1]. Scale-free, so models with different
    logit ranges can be averaged. Ties get the average rank."""
    order = np.lexsort((scores, users))
    su = users[order]
    starts = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
    bounds = np.r_[starts, len(su)]
    out = np.empty(len(scores), dtype=np.float64)
    ranks = np.empty(len(scores), dtype=np.float64)
    for a, b in zip(bounds[:-1], bounds[1:]):
        n = b - a
        s = scores[order][a:b]
        r = np.empty(n)
        i = 0
        pos = np.arange(n, dtype=np.float64)
        while i < n:                      # average ties so equal scores tie in rank
            j = i
            while j + 1 < n and s[j + 1] == s[i]:
                j += 1
            r[i:j + 1] = pos[i:j + 1].mean()
            i = j + 1
        out[a:b] = r / max(n - 1, 1)
    ranks[order] = out
    return ranks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='0,1,2,3,4')
    ap.add_argument('--cells', default=','.join(CELLS))
    ap.add_argument('--skip-train', action='store_true')
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    cells = a.cells.split(',')
    os.makedirs(PRED, exist_ok=True)

    logs = D.load_logs()
    masks = D.split_slices(logs)
    enc, dim, _ = F.encode_splits(logs, masks, F.BASELINE_FIELDS)
    Xva, yva, uva = enc['valid']
    Xte, yte, ute = enc['test']

    # ---------------------------------------------------------------- train
    for name in cells:
        cfg = CELLS[name]
        for seed in seeds:
            path = os.path.join(PRED, f'{name}-seed{seed}.npz')
            if os.path.exists(path) or a.skip_train:
                continue
            exp_id = f'6-{name}-seed{seed}'
            hyp = ('Replicate every candidate cell on the same 5 seeds before calling any '
                   'of them better, and keep the predictions so the ensemble costs no '
                   'extra training')
            with H.Experiment(exp_id, phase='6', axis='replication_and_ensemble',
                              hypothesis=hyp,
                              config=dict(model='fm', fields=F.BASELINE_FIELDS,
                                          bs=8192, seed=seed, device='cpu', **cfg),
                              tags=['ensemble', cfg['loss']]) as ex:
                m, info = fm.train(enc, dim, loss=cfg['loss'], k=cfg['k'], lr=cfg['lr'],
                                   l2=cfg['l2'], epochs=cfg['epochs'],
                                   patience=cfg['patience'], seed=seed,
                                   evaluator=H.score, verbose=False)
                sv, st = m.predict(Xva), m.predict(Xte)
                np.savez_compressed(path, valid=sv, test=st)
                ex.record_train(**{k: v for k, v in info.items() if k != 'history'})
                ex.record_metrics('valid', H.score(uva, yva, sv))
                ex.record_metrics('test', H.score(ute, yte, st))

    # ------------------------------------------------------- per-cell means
    print('\n=== REPLICATED CELLS (5 seeds each) ===')
    print(f"{'cell':16} {'n':>2} {'valid':>8} {'sd':>7} {'test':>8} {'sd':>7}")
    loaded = {}
    for name in cells:
        vs, ts = [], []
        for seed in seeds:
            p = os.path.join(PRED, f'{name}-seed{seed}.npz')
            if not os.path.exists(p):
                continue
            z = np.load(p)
            loaded[(name, seed)] = (z['valid'], z['test'])
            vs.append(H.score(uva, yva, z['valid'])['primary'])
            ts.append(H.score(ute, yte, z['test'])['primary'])
        if not vs:
            continue
        v, t = np.array(vs), np.array(ts)
        sd = lambda x: x.std(ddof=1) if len(x) > 1 else float('nan')
        print(f'{name:16} {len(v):>2} {v.mean():8.4f} {sd(v):7.4f} {t.mean():8.4f} {sd(t):7.4f}')

    # ------------------------------------------------------------ ensembles
    def ens(keys, mode):
        pv = np.zeros(len(yva)); pt = np.zeros(len(yte))
        for k in keys:
            sv, st = loaded[k]
            if mode == 'rank':
                pv += user_percentile(sv, uva); pt += user_percentile(st, ute)
            else:
                pv += sv; pt += st
        n = len(keys)
        return H.score(uva, yva, pv / n), H.score(ute, yte, pt / n)

    print('\n=== ENSEMBLES (no new training) ===')
    print(f"{'ensemble':36} {'mode':5} {'valid':>8} {'test':>8} {'vs best single':>15}")
    best_single_t = 0.5980
    combos = []
    for name in cells:
        ks = [(name, s) for s in seeds if (name, s) in loaded]
        if len(ks) > 1:
            combos.append((f'{name} x{len(ks)} seeds', ks))
    bpr_cells = [c for c in cells if c.startswith('bpr')]
    if len(bpr_cells) > 1:
        ks = [(c, s) for c in bpr_cells for s in seeds if (c, s) in loaded]
        combos.append((f'all BPR cells x seeds ({len(ks)})', ks))
    allk = [(c, s) for c in cells for s in seeds if (c, s) in loaded]
    if len(allk) > 1:
        combos.append((f'everything ({len(allk)})', allk))
    mixed = [(c, s) for c in cells for s in seeds
             if (c, s) in loaded and (c.startswith('bpr') or c == 'pointwise-k16')]
    results = {}
    for label, ks in combos:
        for mode in ('rank', 'score'):
            v, t = ens(ks, mode)
            results[(label, mode)] = (v, t)
            print(f"{label:36} {mode:5} {v['primary']:8.4f} {t['primary']:8.4f} "
                  f"{t['primary']-best_single_t:+15.4f}")

    # log the best ensemble as a first-class result
    best = max(results.items(), key=lambda kv: kv[1][0]['primary'])
    (label, mode), (v, t) = best
    # id carries the member count so re-cutting from cache updates rather than
    # duplicating -- a re-run with the same members is the same experiment.
    with H.Experiment(f'6-ensemble-{mode}-n{len(allk)}', phase='6',
                      axis='replication_and_ensemble',
                      hypothesis=('Averaging within-user percentile ranks across already-trained '
                                  'models costs no new training and should clear the noise band'),
                      config=dict(model='ensemble', members=label, mode=mode,
                                  n_members=len(allk), device='cpu'),
                      tags=['ensemble']) as ex:
        ex.record_metrics('valid', v)
        ex.record_metrics('test', t)
        ex.takeaway = f'best ensemble: {label} ({mode})'

    with open(os.path.join(ROOT, 'phase6_ensemble.json'), 'w', encoding='utf-8') as fh:
        # H.score returns numpy scalars; cast before serializing
        json.dump({f'{k[0]}|{k[1]}': {'valid': float(vv['primary']),
                                      'test': float(tt['primary'])}
                   for k, (vv, tt) in results.items()}, fh, indent=1)
    print(f"\nbest: {label} ({mode}) valid {v['primary']:.4f} test {t['primary']:.4f}")


if __name__ == '__main__':
    main()
