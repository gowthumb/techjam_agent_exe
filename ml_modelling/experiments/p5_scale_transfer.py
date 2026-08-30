"""Phase 5: do the Pure-derived KB priors transfer to a larger benchmark?

KuaiRand-1K is ~8x KuaiRand-Pure. The KB marks three things as
`needs_revalidation` at scale -- k, learning rate, and iteration pacing -- and
flags the over-parameterisation finding as the claim most likely to invert once
there is more data to support capacity. This tests exactly those.

Stages (run them in order; each is a separate invocation so a long one can be
skipped without losing the others):

  --stage facts     dataset shape, split dates, label rates, ID overlap. Cheap.
                    Confirms the official split dates apply to this release at all.
  --stage headto2   the two configs that matter, N seeds each:
                      baseline  pointwise k=16 lr=0.001   (what the kit ships)
                      KB pick   bpr       k=6  lr=0.0002  (what the KB recommends)
                    Answers "does the one confirmed Pure finding survive 8x data".
  --stage capacity  k sweep under both losses. Answers "is capacity still flat".
  --stage lr        lr sweep for BPR. Answers "did the optimum move".

Cost control: --subsample N trains on the first N train rows (evaluation is always
on the full split). Use it for the sweeps, never for headto2.

Run:  python experiments/p5_scale_transfer.py --bench 1k --stage facts
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from explib import dataset as D, benchmarks as B, features as F, fm, harness as H

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

BASELINE_CFG = dict(loss='pointwise', k=16, lr=0.001, l2=1e-6, epochs=40, patience=4)
KB_CFG = dict(loss='bpr', k=6, lr=0.0002, l2=1e-6, epochs=60, patience=6)


def prepare(bench, subsample=None):
    """Encode the five baseline fields via the integer fast path.

    Same five fields and same conventions as Pure, so the comparison is like for
    like; only the encoder implementation differs (see features.encode_int_fields).
    """
    logs = B.load_logs(bench, minimal=True)
    masks = D.split_slices(logs)
    tr = masks['train']

    vids, auths = B.load_video_authors(bench)
    order = np.argsort(vids)
    pos = np.searchsorted(vids[order], logs['video_id'])
    pos = np.clip(pos, 0, len(vids) - 1)
    hit = vids[order][pos] == logs['video_id']
    author = np.where(hit, auths[order][pos], -1)

    edges = np.quantile(logs['duration_ms'][tr], np.linspace(0, 1, 11)[1:-1])
    dur_bucket = np.searchsorted(edges, logs['duration_ms']).astype(np.int64)

    cols = {'user_id': logs['user_id'], 'video_id': logs['video_id'],
            'author_id': author, 'tab': logs['tab'].astype(np.int64),
            'dur_bucket': dur_bucket}
    X, dim, unseen = F.encode_int_fields(cols, tr, order=list(F.BASELINE_FIELDS))
    print('  per-field vocab / share of rows unseen in train:')
    for k, v in unseen.items():
        print(f"    {k:12s} vocab={v['vocab']:>9,}  unseen={v['unseen_rate_all']:.1%}")

    y = (logs[D.LABEL] != 0).astype(np.float32)
    enc = {sp: (X[m], y[m], logs['user_id'][m]) for sp, m in masks.items()}

    if subsample:
        Xt, yt, ut = enc['train']
        n = min(subsample, len(yt))
        idx = np.sort(np.random.default_rng(0).choice(len(yt), n, replace=False))
        enc['train'] = (Xt[idx], yt[idx], ut[idx])
        print(f'  subsampled train to {n:,} of {len(yt):,} rows')
    return logs, masks, enc, dim


def run_one(exp_id, bench, cfg, enc, dim, seed, hypothesis, axis, subsample=None):
    full = dict(model='fm', benchmark=bench, fields=F.BASELINE_FIELDS,
                bs=8192, seed=seed, subsample=subsample,
                sparse_updates=(bench != 'pure'), **cfg)
    # The Pure baseline is not a valid reference on another benchmark, so record
    # no delta rather than a misleading one; comparisons are made within-benchmark.
    ref = None if bench == 'pure' else 'none'
    with H.Experiment(exp_id, phase='5', axis=axis, hypothesis=hypothesis,
                      config=full, tags=['scale', bench, cfg['loss']],
                      baseline_ref=ref) as ex:
        # dim is ~4.4M on 1K, so the dense Adam update is infeasible; sparse mode
        # was verified to land inside the noise band on Pure before being used here.
        m, info = fm.train(enc, dim, loss=cfg['loss'], k=cfg['k'], lr=cfg['lr'],
                           l2=cfg['l2'], epochs=cfg['epochs'], patience=cfg['patience'],
                           seed=seed, evaluator=H.score, verbose=True,
                           sparse=(bench != 'pure'))
        ex.record_train(**{k: v for k, v in info.items() if k != 'history'})
        ex.record_train(history=info['history'])
        for sp in ('valid', 'test'):
            X, y, u = enc[sp]
            ex.record_metrics(sp, H.score(u, y, m.predict(X)))
        return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bench', default='1k', choices=list(B.BENCHMARKS))
    ap.add_argument('--stage', required=True,
                    choices=['facts', 'headto2', 'capacity', 'lr'])
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--subsample', type=int, default=None)
    ap.add_argument('--ks', default='4,16,64')
    ap.add_argument('--lrs', default='0.0001,0.0002,0.001')
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]

    if a.stage == 'facts':
        t0 = time.time()
        facts = B.describe(a.bench)
        print(json.dumps(facts, indent=2))
        out = os.path.join(ROOT, f'phase5_facts_{a.bench}.json')
        with open(out, 'w', encoding='utf-8') as fh:
            json.dump(facts, fh, indent=1)
        print(f'\nwrote {out}  ({time.time()-t0:.0f}s)')

        pure = json.load(open(os.path.join(ROOT, 'phase5_facts_pure.json'), encoding='utf-8')) \
            if os.path.exists(os.path.join(ROOT, 'phase5_facts_pure.json')) else None
        if pure:
            print(f"\nscale vs pure: {facts['rows_total']/pure['rows_total']:.1f}x rows, "
                  f"{facts['users_total']/pure['users_total']:.1f}x users")
        return

    print(f'loading {a.bench} ...')
    t0 = time.time()
    logs, masks, enc, dim = prepare(a.bench, a.subsample)
    split_rows = {k: int(v.sum()) for k, v in masks.items()}
    print(f"  {len(logs['user_id']):,} rows, dim={dim}, loaded in {time.time()-t0:.0f}s")
    print(f"  split rows: {split_rows}")

    tag = f'-sub{a.subsample}' if a.subsample else ''

    if a.stage == 'headto2':
        hyp = ('The one confirmed Pure finding is that BPR beats pointwise by +0.0021 '
               'valid. Ranking losses avoid fitting the global rate, which is a property '
               'of the metric rather than of the data volume, so it should transfer; '
               'if it does not, the KB cannot claim loss choice is scale-invariant')
        for name, cfg in (('baseline', BASELINE_CFG), ('kbpick', KB_CFG)):
            for seed in seeds:
                run_one(f'5-{a.bench}-{name}{tag}-seed{seed}', a.bench, cfg, enc, dim,
                        seed, hyp, 'scale_transfer_loss', a.subsample)

    elif a.stage == 'capacity':
        hyp = ('On Pure, k is flat from 1 to 16 and wasteful above 32. More data should '
               'support more capacity, so this is the KB claim most likely to invert at '
               'scale; if the optimum moves up, scale_transfer must say so explicitly')
        for k in [int(x) for x in a.ks.split(',')]:
            for cfg_name, base in (('pointwise', BASELINE_CFG), ('bpr', KB_CFG)):
                cfg = dict(base, k=k)
                run_one(f'5-{a.bench}-{cfg_name}-k{k}{tag}-seed{seeds[0]}', a.bench,
                        cfg, enc, dim, seeds[0], hyp, 'scale_transfer_capacity',
                        a.subsample)

    elif a.stage == 'lr':
        hyp = ('BPR peaked at lr=0.0002 on Pure. Gradient scale per step depends on batch '
               'composition, which changes with more users per batch, so the optimum may '
               'move at scale')
        for lr in [float(x) for x in a.lrs.split(',')]:
            cfg = dict(KB_CFG, lr=lr)
            run_one(f'5-{a.bench}-bpr-lr{lr:g}{tag}-seed{seeds[0]}', a.bench, cfg,
                    enc, dim, seeds[0], hyp, 'scale_transfer_lr', a.subsample)

    rows = [r for r in H.read_log() if r['phase'] == '5' and r['config'].get('benchmark') == a.bench]
    print()
    print(H.summarize(rows))


if __name__ == '__main__':
    main()
