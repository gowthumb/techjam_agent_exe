"""Phase 8: get the multi-task effect directly, without the multi-task machinery.

Phase 1C established that auxiliary heads act as a REGULARIZER, not as transfer:
the gain scales with how sparse the auxiliary label is, and a pure random label at
0.1% positive rate scores +0.0010 — the same as is_follow. If regularisation is
what is on offer, buy it directly and skip the extra head, the extra task weights
and the extra forward pass.

Gaussian noise on the embeddings during training is the cheapest form of it. The
noise is additive, so dE'/dV = I and the FM gradient keeps its exact shape — this
costs one RNG draw per batch and nothing else.

Tested under BOTH losses, because the aux-head result was measured under pointwise
and the recommended config uses BPR; a regulariser that only helps the loss we no
longer use would be a non-result.

Reference points (5 seeds, from Phase 1C/1H):
  single-task control, learnable A     valid 0.6014
  best auxiliary head (is_follow)      valid 0.6025   (+0.0011)
  random-noise auxiliary head          valid 0.6024   (+0.0010)
  BPR k=16 lr 0.0002                   valid 0.6038

CPU only.

Run:  python experiments/p8_noise_reg.py
"""
import os, sys, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from explib import dataset as D, features as F, fm, harness as H

SETUPS = {
    'pointwise': dict(loss='pointwise', k=16, lr=0.001, l2=1e-6, epochs=40, patience=4),
    'bpr':       dict(loss='bpr', k=16, lr=0.0002, l2=1e-6, epochs=60, patience=6),
}

HYP = ('Auxiliary heads were shown to act as regularisers rather than as transfer '
       '(a random label matched the best real signal). Gaussian embedding noise is the '
       'same mechanism bought directly: if it reproduces the +0.001 band, the multi-task '
       'machinery is redundant; if it does not, the aux-head effect is something else')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sigmas', default='0,0.002,0.005,0.01,0.02,0.05')
    ap.add_argument('--losses', default='pointwise,bpr')
    ap.add_argument('--seeds', default='0,1,2')
    a = ap.parse_args()

    logs = D.load_logs()
    masks = D.split_slices(logs)
    enc, dim, _ = F.encode_splits(logs, masks, F.BASELINE_FIELDS)

    out = {}
    for lname in a.losses.split(','):
        s = SETUPS[lname]
        for sig in [float(x) for x in a.sigmas.split(',')]:
            vs, ts = [], []
            for seed in [int(x) for x in a.seeds.split(',')]:
                eid = f'8-{lname}-noise{sig:g}-seed{seed}'
                with H.Experiment(eid, phase='8', axis='noise_regularizer',
                                  hypothesis=HYP,
                                  config=dict(model='fm', emb_noise=sig, bs=8192,
                                              seed=seed, device='cpu',
                                              fields=F.BASELINE_FIELDS, **s),
                                  tags=['regularizer', lname]) as ex:
                    m, info = fm.train(enc, dim, loss=s['loss'], k=s['k'], lr=s['lr'],
                                       l2=s['l2'], epochs=s['epochs'],
                                       patience=s['patience'], seed=seed,
                                       evaluator=H.score, verbose=False,
                                       emb_noise=sig)
                    ex.record_train(**{k: v for k, v in info.items() if k != 'history'})
                    for sp in ('valid', 'test'):
                        X, y, u = enc[sp]
                        mm = H.score(u, y, m.predict(X))
                        ex.record_metrics(sp, mm)
                        (vs if sp == 'valid' else ts).append(mm['primary'])
            out[(lname, sig)] = (np.array(vs), np.array(ts))

    print('\n=== EMBEDDING NOISE AS A REGULARIZER (3 seeds each) ===')
    print(f"{'loss':10} {'sigma':>7} {'valid':>8} {'sd':>7} {'test':>8} {'sd':>7} {'d vs sigma=0':>13}")
    for lname in a.losses.split(','):
        base = out.get((lname, 0.0))
        for (ln, sig), (v, t) in sorted(out.items()):
            if ln != lname:
                continue
            sd = lambda x: x.std(ddof=1) if len(x) > 1 else float('nan')
            d = v.mean() - base[0].mean() if base is not None else float('nan')
            print(f'{ln:10} {sig:>7g} {v.mean():8.4f} {sd(v):7.4f} '
                  f'{t.mean():8.4f} {sd(t):7.4f} {d:+13.4f}')


if __name__ == '__main__':
    main()
