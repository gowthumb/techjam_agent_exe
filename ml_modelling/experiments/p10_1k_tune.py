"""Phase 10: actually optimising KuaiRand-1K (not just transfer-testing it).

Phase 5 established that 1K is an item cold-start regime and that the Pure recipe
loses there. This phase treats 1K as its own problem, using what its structure
implies:

  * best_epoch=2 at lr=0.001 means the model overfits almost immediately ->
    the levers are regularisation (embedding noise, validated on Pure) and a
    smaller step with a longer budget.
  * k=16 beat k=4 by +0.005 and k=64 fell back -> the optimum is interior but
    k=32 was never tested. Fill the gap.
  * 73.9% of test rows have a train-warm AUTHOR even though only 15.1% have a
    warm video -> the author embedding is the item-side signal that survives
    cold-start. It is already one of the five fields; what is untested is
    leaning on it harder via (user, author) affinity, which is 26.5% warm here
    vs 2.6% on Pure (where it was a dead end for exactly that reason).

All runs: sparse Adam (vocab ~2.9M), CPU only, baseline_ref='none' so no
misleading delta against Pure's rungs is recorded.

Run:  python experiments/p10_1k_tune.py --stage reg      # noise + lr + k32
      python experiments/p10_1k_tune.py --stage affinity # (user,author) causal
"""
import os, sys, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from explib import dataset as D, benchmarks as B, features as F, fm, harness as H, history as HI
sys.path.insert(0, os.path.dirname(__file__))
from p5_scale_transfer import prepare

BASE = dict(loss='pointwise', k=16, lr=0.001, l2=1e-6, epochs=40, patience=4)


def run(exp_id, cfg, enc, dim, seed, hyp, axis, emb_noise=0.0):
    full = dict(model='fm', benchmark='1k', bs=8192, seed=seed, device='cpu',
                sparse_updates=True, emb_noise=emb_noise,
                fields=list(F.BASELINE_FIELDS), **cfg)
    with H.Experiment(exp_id, phase='10', axis=axis, hypothesis=hyp, config=full,
                      tags=['1k', 'tune'], baseline_ref='none') as ex:
        m, info = fm.train(enc, dim, loss=cfg['loss'], k=cfg['k'], lr=cfg['lr'],
                           l2=cfg['l2'], epochs=cfg['epochs'],
                           patience=cfg['patience'], seed=seed, evaluator=H.score,
                           verbose=False, sparse=True, emb_noise=emb_noise)
        ex.record_train(**{k: v for k, v in info.items() if k != 'history'})
        ex.record_train(history=info['history'])
        for sp in ('valid', 'test'):
            X, y, u = enc[sp]
            ex.record_metrics(sp, H.score(u, y, m.predict(X)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['reg', 'affinity'])
    ap.add_argument('--seeds', default='0')
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]

    print('loading 1k ...')
    logs, masks, enc, dim = prepare('1k')
    print(f'  dim={dim}')

    if a.stage == 'reg':
        hyp_n = ('1K overfits by epoch 2, so regularisation is the natural lever; '
                 'embedding noise is the cheap regulariser validated on Pure, tried '
                 'here at the strengths that bracketed the Pure optimum')
        for sig in (0.05, 0.1, 0.2):
            for seed in seeds:
                run(f'10-1k-pointwise-noise{sig:g}-seed{seed}', BASE, enc, dim,
                    seed, hyp_n, 'onek_regularisation', emb_noise=sig)
        hyp_lr = ('best_epoch=2 at lr=0.001 is the too-fast-fit signature; a smaller '
                  'step with a longer budget may find a better minimum')
        for lr in (0.0005, 0.0002):
            cfg = dict(BASE, lr=lr, epochs=60, patience=6)
            for seed in seeds:
                run(f'10-1k-pointwise-lr{lr:g}-ep60-seed{seed}', cfg, enc, dim,
                    seed, hyp_lr, 'onek_regularisation')
        hyp_k = ('k=16 (0.6438) beat k=4 and k=64 on 1K but k=32 was never tested; '
                 'the bracket-the-optimum prior requires filling the interior gap')
        cfg = dict(BASE, k=32)
        for seed in seeds:
            run(f'10-1k-pointwise-k32-seed{seed}', cfg, enc, dim, seed,
                hyp_k, 'onek_capacity')

    elif a.stage == 'affinity':
        hyp = ('(user, author) affinity was a Pure dead end at 2.6% warm coverage; on '
               '1K it is 26.5% warm and the author is the item-side signal that '
               'survives cold-start (73.9% of test rows author-warm), so the feature '
               'has the coverage it lacked')
        vids, auths = B.load_video_authors('1k')
        order = np.argsort(vids)
        pos = np.clip(np.searchsorted(vids[order], logs['video_id']), 0, len(vids) - 1)
        hit = vids[order][pos] == logs['video_id']
        author = np.where(hit, auths[order][pos], -1)
        specs = [('aff_user_author', [logs['user_id'], author])]
        print('building causal (user,author) affinity (date-resolution ordering; '
              'minimal load has no time_ms — ties resolve in file order) ...')
        extra, fitted, _ = HI.build_affinity_fields(logs, masks, specs, mode='causal')
        col = extra['aff_user_author']
        for sp, m in masks.items():
            print(f'  {sp}: warm rows {float((col[m] != 0).mean()):.1%}')
        # append the affinity bucket as a 6th field via the int fast path
        cols6 = {'aff_user_author': col.astype(np.int64)}
        Xa, dima, _ = F.encode_int_fields(cols6, masks['train'])
        X5 = {sp: enc[sp][0] for sp in enc}
        enc6 = {sp: (np.concatenate([X5[sp], Xa[masks[sp]] + dim], axis=1),
                     enc[sp][1], enc[sp][2]) for sp in enc}
        cfg = dict(BASE)
        for seed in seeds:
            run(f'10-1k-pointwise-aff-user_author-seed{seed}', cfg, enc6,
                dim + dima, seed, hyp, 'onek_affinity')

    rows = [r for r in H.read_log() if r['phase'] == '10']
    print()
    print(H.summarize(rows))


if __name__ == '__main__':
    main()
