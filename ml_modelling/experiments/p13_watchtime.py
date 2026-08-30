"""Phase 13 / axis: watch-time regression head (CWM-style censored loss).

GROUNDING. `KNOWLEDGE_BASE_PLAN.md` Phase 1c (multi-task / auxiliary signals);
starter kit README direction #4 (watch-time modelling / CWM, flagged as the
direction with "研究深度").

HYPOTHESIS. `long_view` is a deterministic threshold on watch time -- `1` iff
`play_time_ms >= min(duration_ms, 18000)` (matches 97.9% of rows). The binary
label is a coarsened view of a continuous signal the log carries in full. A
regression head on the watch ratio, sharing the embedding table, is trained on
strictly more information than the binary head and gets gradient from every row.

This is NOT the Phase 1C multi-task result. Phase 1C tried the 11 other BINARY
feedback signals and found a random label matched the best real one -- the heads
were regularisers, not transfer. The watch ratio is (a) continuous and (b)
mechanically the thing the label thresholds, so it is a different test.

CENSORING (CWM). A completed play (`play_time_ms >= duration_ms`) right-censors
the true watch desire. For those rows the regression loss is ONE-SIDED: it
penalises predicting less than the observed ratio, not more.

CONTROLS (KB control_rule):
  * two-sided Huber -- if it matches the one-sided version, censoring does nothing.
  * random-continuous target -- a regression head on uniform noise at the same
    scale. If it matches the real watch-ratio head, the head is a regulariser
    (the Phase 1C finding), not watch-time transfer.
  * w_wt = 0 -- single-task; must land on the FM baseline.

Ranking score: head 0 (long_view), head 1 (watch ratio), or a rank-blend,
selected on valid, reported on all four splits (Phase 11).

Stages:  --stage main | control | replicate
Run:  python experiments/p13_watchtime.py --stage main
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from explib import dataset as D, features as F, wtfm, harness as H, unbiased as U

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STATE = os.path.join(ROOT, 'cache', 'p13_best.json')

HYP = ('long_view is a deterministic threshold on watch time; a regression head on the '
       'continuous watch ratio shares the embedding table and trains on strictly more '
       'information than the binary head. One-sided (censored) loss for completed plays. '
       'KNOWLEDGE_BASE_PLAN.md Phase 1c; README direction #4')


def setup():
    logs = D.load_logs()
    masks = D.split_slices(logs)
    enc, dim, encoder = F.encode_splits(logs, masks, F.BASELINE_FIELDS)
    rand_enc = U.load_random_encoded(encoder)
    tr = masks['train']
    pt = logs['play_time_ms'][tr].astype(np.float64)
    du = np.maximum(logs['duration_ms'][tr].astype(np.float64), 1.0)
    y_wt = np.clip(pt / du, 0.0, 2.0).astype(np.float32)
    censored = (pt >= du)
    y_lv = (logs['long_view'][tr] != 0).astype(np.float32)
    return enc, dim, rand_enc, y_lv, y_wt, censored


def run(eid, enc, dim, rand_enc, y_lv, y_wt, censored, *, w_wt, one_sided,
        select_head, seed=0, epochs=40, patience=4, lr=0.001, extra=None):
    if eid in {r['exp_id'] for r in H.read_log() if r['phase'] == '13'}:
        print(f'  skip {eid} (already logged)')
        return None
    cfg = dict(model='wtfm', k=16, lr=lr, l2=1e-6, bs=8192, seed=seed,
               fields=F.BASELINE_FIELDS, w_wt=w_wt, one_sided=one_sided,
               huber_delta=0.5, select_head=select_head, epochs=epochs,
               patience=patience, **(extra or {}))
    with H.Experiment(eid, phase='13', axis='multi_task', hypothesis=HYP,
                      config=cfg, tags=['multitask', 'watchtime']) as ex:
        m, info = wtfm.train(enc['train'][0], y_lv, y_wt, censored, *enc['valid'], dim,
                             w_wt=w_wt, one_sided=one_sided, k=16, lr=lr, l2=1e-6,
                             epochs=epochs, patience=patience, seed=seed,
                             evaluator=H.score, verbose=False, select_head=select_head)
        ex.record_train(**{k: v for k, v in info.items() if k != 'history'})

        def score_split(X, y, u):
            p0, p1 = m.predict_both(X)
            s = (p0 if select_head == 'lv' else p1 if select_head == 'wt'
                 else wtfm.rank_blend([p0, p1], u, [0.5, 0.5]))
            return H.score(u, y, s)

        sc = {name: score_split(*src) for name, src in
              (('valid', enc['valid']), ('test', enc['test']),
               ('rand_valid', rand_enc['rand_valid']), ('rand_test', rand_enc['rand_test']))}
        for sp, s in sc.items():
            ex.record_metrics(sp, s)
    row = {sp: round(float(sc[sp]['primary']), 5) for sp in sc}
    print(f"  {eid:46s} v {row['valid']:.4f} t {row['test']:.4f} "
          f"rv {row['rand_valid']:.4f} rt {row['rand_test']:.4f}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['main', 'control', 'replicate'])
    a = ap.parse_args()
    enc, dim, rand_enc, y_lv, y_wt, censored = setup()
    args = (enc, dim, rand_enc, y_lv, y_wt, censored)

    if a.stage == 'main':
        best = None
        for w in (0.3, 0.5, 1.0):
            r = run(f'13-wt-w{w:g}-onesided-ranklv', *args, w_wt=w,
                    one_sided=True, select_head='lv')
            if r and (best is None or r['valid'] > best['valid']):
                best = dict(r, w_wt=w)
        for head in ('wt', 'blend'):
            run(f'13-wt-w0.5-onesided-rank{head}', *args, w_wt=0.5,
                one_sided=True, select_head=head)
        if best:
            json.dump(best, open(STATE, 'w'))
            print(f"\nmain best: w_wt={best['w_wt']} valid {best['valid']:.4f}")

    elif a.stage == 'control':
        w = json.load(open(STATE)).get('w_wt', 1.0)
        # w_wt = 0 must reproduce the FM baseline -- the sanity check that the
        # negative at w > 0 is real and not a wtfm defect.
        run('13-wt-w0-singletask', *args, w_wt=0.0, one_sided=True, select_head='lv',
            extra={'control': 'single_task_must_match_baseline'})
        run(f'13-wt-w{w:g}-twosided-ranklv', *args, w_wt=w, one_sided=False,
            select_head='lv', extra={'control': 'two_sided_huber'})
        rng = np.random.default_rng(13)
        y_rand = (rng.random(len(y_wt)) * 2.0).astype(np.float32)
        enc2 = (enc, dim, rand_enc, y_lv, y_rand, censored)
        run(f'13-wt-w{w:g}-randomtarget-ranklv', *enc2, w_wt=w, one_sided=False,
            select_head='lv', extra={'control': 'random_continuous_target'})

    elif a.stage == 'replicate':
        w = json.load(open(STATE))['w_wt']
        for seed in range(3):
            sfx = f'-seed{seed}' if seed else '-rep-seed0'
            run(f'13-wt-w{w:g}-onesided-ranklv{sfx}', *args, w_wt=w,
                one_sided=True, select_head='lv', seed=seed)
            run(f'13-wt-w0-singletask{sfx}', *args, w_wt=0.0, one_sided=True,
                select_head='lv', seed=seed)

    print('\n' + H.summarize([r for r in H.read_log() if r['phase'] == '13']))


if __name__ == '__main__':
    main()
