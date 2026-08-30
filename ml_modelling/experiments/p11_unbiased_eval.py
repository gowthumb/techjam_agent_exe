"""Phase 11 / axis: unbiased evaluation via the randomly-exposed-video log.

GROUNDING. `KNOWLEDGE_BASE_PLAN.md` Phase 0 ("make sure your numbers are
trustworthy") + Phase 2 KB section "Calibration reference"; starter kit README
unexplored direction #7 ("log_random_*.csv 是随机曝光日志，可作为额外的无偏验证集").

QUESTION. Every verdict in this workstream is selected on `valid`, which is a
sample the production recommender chose to show -- it carries exposure bias. The
KB already documents a "valid blind spot": BPR is +0.0021 on valid but +0.0032 on
test; embedding noise is flat on valid but consistently better on test. Would an
unbiased selection split have caught those? And is it a reliable enough signal to
use as a tie-breaker inside the noise band?

WHAT THIS RUNS. A handful of configs whose valid-vs-test behaviour is already
known from the log, each scored on all four splits (valid / test / rand_valid /
rand_test). Then: rank-correlation of each selection split against `test`, and a
direct check of the KB's blind-spot cases.

rand_test is report-only (spans the hidden-test period). Tie-breaking, if it is
adopted, uses rand_valid only.

Run:  python experiments/p11_unbiased_eval.py
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from explib import dataset as D, features as F, fm, harness as H, unbiased as U

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

HYP = ('valid is exposure-biased; the KB documents interventions that move test but not '
       'valid. Scoring known configs on the randomly-exposed log tests whether an unbiased '
       'split tracks test better on exactly those cases, and whether it is stable enough to '
       'break ties inside the 0.0016 noise band')

# (exp_id, loss, kwargs, seeds) -- configs whose valid/test story is already in the log
CONFIGS = [
    ('11-pointwise-k16',          'pointwise', dict(k=16, lr=0.001,  epochs=40, patience=4),              [0, 1]),
    ('11-pointwise-k2',           'pointwise', dict(k=2,  lr=0.001,  epochs=40, patience=4),              [0]),
    ('11-pointwise-k16-noise0.1', 'pointwise', dict(k=16, lr=0.001,  epochs=40, patience=4, emb_noise=0.1), [0, 1]),
    ('11-bpr-k6',                 'bpr',       dict(k=6,  lr=0.0002, epochs=60, patience=6),              [0, 1]),
    ('11-bpr-k16',                'bpr',       dict(k=16, lr=0.0002, epochs=60, patience=6),              [0, 1]),
    ('11-bpr-k16-noise0.1',       'bpr',       dict(k=16, lr=0.0002, epochs=60, patience=6, emb_noise=0.1), [0, 1]),
    ('11-listwise-lr0.001',       'listwise',  dict(k=16, lr=0.001,  epochs=40, patience=4),              [0]),
]


def spearman(a, b):
    ar = np.argsort(np.argsort(a)); br = np.argsort(np.argsort(b))
    return float(np.corrcoef(ar, br)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None)
    a = ap.parse_args()

    done = {r['exp_id'] for r in H.read_log() if r['phase'] == '11'}

    logs = D.load_logs()
    masks = D.split_slices(logs)
    enc, dim, encoder = F.encode_splits(logs, masks, F.BASELINE_FIELDS)
    print(f'fields={F.BASELINE_FIELDS} dim={dim}')

    print('encoding the randomly-exposed log with the frozen train encoder ...')
    rand_enc = U.load_random_encoded(encoder)
    for s, (X, y, u) in rand_enc.items():
        print(f'  {s}: {len(y)} rows, {len(np.unique(u))} users, label rate {y.mean():.4f}')

    rungs = U.calibration_rungs(rand_enc)
    print('\nunbiased calibration rungs:')
    for s, r in rungs.items():
        print(f"  {s}: random {r['random']:.4f}  popularity {r['popularity']:.4f}  "
              f"oracle {r['oracle']:.4f}")

    results = []
    for exp_id, loss, kw, seeds in CONFIGS:
        if a.only and a.only not in exp_id:
            continue
        for seed in seeds:
            eid = exp_id + (f'-seed{seed}' if seed else '')
            if eid in done:
                print(f'  skip {eid} (already in log)')
                continue
            cfg = dict(model='fm', loss=loss, l2=1e-6, bs=8192, seed=seed,
                       fields=F.BASELINE_FIELDS, emb_noise=kw.get('emb_noise', 0.0),
                       **{k: v for k, v in kw.items() if k != 'emb_noise'})
            with H.Experiment(eid, phase='11', axis='unbiased_evaluation',
                              hypothesis=HYP, config=cfg,
                              tags=['unbiased', loss]) as ex:
                m, info = fm.train(enc, dim, loss=loss, k=kw['k'], lr=kw['lr'],
                                   l2=1e-6, epochs=kw['epochs'], patience=kw['patience'],
                                   seed=seed, evaluator=H.score, verbose=False,
                                   emb_noise=kw.get('emb_noise', 0.0))
                ex.record_train(**{k: v for k, v in info.items() if k != 'history'})
                scores = U.evaluate_all(m.predict, enc, rand_enc)
                for sp, sc in scores.items():
                    ex.record_metrics(sp, sc)
                row = {'exp_id': eid, 'seed': seed,
                       **{sp: round(float(sc['primary']), 5) for sp, sc in scores.items()}}
                results.append(row)
                print(f"  {eid:30s} valid {row['valid']:.4f}  test {row['test']:.4f}  "
                      f"rand_valid {row['rand_valid']:.4f}  rand_test {row['rand_test']:.4f}")

    # ---------------------------------------------------------------- analysis
    # read every phase-11 record from the log (not just this run's fresh ones)
    results = []
    for r in H.read_log():
        if r['phase'] != '11' or not r['metrics'].get('valid'):
            continue
        results.append({'exp_id': r['exp_id'],
                        **{sp: round(float(r['metrics'][sp]['primary']), 5)
                           for sp in ('valid', 'test', 'rand_valid', 'rand_test')
                           if sp in r['metrics']}})

    # collapse to per-config means so seed noise does not dominate the correlation
    by_cfg = {}
    for r in results:
        key = r['exp_id'].rsplit('-seed', 1)[0]
        by_cfg.setdefault(key, []).append(r)
    agg = []
    for key, rs in by_cfg.items():
        agg.append({'config': key,
                    **{sp: float(np.mean([r[sp] for r in rs]))
                       for sp in ('valid', 'test', 'rand_valid', 'rand_test')}})

    v = np.array([r['valid'] for r in agg])
    t = np.array([r['test'] for r in agg])
    rv = np.array([r['rand_valid'] for r in agg])
    corr = {'valid_vs_test': spearman(v, t),
            'rand_valid_vs_test': spearman(rv, t)}

    # blind-spot check: does rand_valid move with test on the noise cases?
    def get(cfg):
        return next((r for r in agg if r['config'] == cfg), None)
    blind = {}
    base_p, noise_p = get('11-pointwise-k16'), get('11-pointwise-k16-noise0.1')
    base_b, noise_b = get('11-bpr-k16'), get('11-bpr-k16-noise0.1')
    if base_p and noise_p:
        blind['pointwise_noise0.1'] = {
            'd_valid': round(noise_p['valid'] - base_p['valid'], 5),
            'd_test': round(noise_p['test'] - base_p['test'], 5),
            'd_rand_valid': round(noise_p['rand_valid'] - base_p['rand_valid'], 5)}
    if base_b and noise_b:
        blind['bpr_noise0.1'] = {
            'd_valid': round(noise_b['valid'] - base_b['valid'], 5),
            'd_test': round(noise_b['test'] - base_b['test'], 5),
            'd_rand_valid': round(noise_b['rand_valid'] - base_b['rand_valid'], 5)}
    pw, bpr = get('11-pointwise-k16'), get('11-bpr-k16')
    if pw and bpr:
        blind['bpr_vs_pointwise'] = {
            'd_valid': round(bpr['valid'] - pw['valid'], 5),
            'd_test': round(bpr['test'] - pw['test'], 5),
            'd_rand_valid': round(bpr['rand_valid'] - pw['rand_valid'], 5)}

    print('\n=== SELECTION-SPLIT RELIABILITY ===')
    print(f"  spearman(valid, test)      = {corr['valid_vs_test']:+.3f}")
    print(f"  spearman(rand_valid, test) = {corr['rand_valid_vs_test']:+.3f}")
    print('\n=== KB BLIND-SPOT CASES (delta vs the matched baseline) ===')
    for name, d in blind.items():
        agree = 'agrees with test' if (d['d_rand_valid'] > 0) == (d['d_test'] > 0) else 'DISAGREES with test'
        print(f"  {name:22s} valid {d['d_valid']:+.4f}  test {d['d_test']:+.4f}  "
              f"rand_valid {d['d_rand_valid']:+.4f}  ({agree})")

    out = {'configs': agg, 'seed_runs': results, 'correlation': corr,
           'blind_spot': blind, 'unbiased_rungs': rungs}
    with open(os.path.join(ROOT, 'phase11_unbiased.json'), 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {os.path.join(ROOT, 'phase11_unbiased.json')}")


if __name__ == '__main__':
    main()
