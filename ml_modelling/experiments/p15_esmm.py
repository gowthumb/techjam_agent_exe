"""Phase 15 / axis: ESMM-style multiplicative decomposition.

GROUNDING. `KNOWLEDGE_BASE_PLAN.md` Phase 1c names an "ESMM-style shared-embedding
multi-task setup" explicitly; primer A.2/A.3 (the impression -> click -> deeper
engagement funnel).

HYPOTHESIS. A row with `is_click = 0` has play_time ~ 0 and therefore
`long_view ~ 0` deterministically, so the funnel is real and
`P(long_view) = P(click) * P(long_view | click)` is a well-posed factorisation.
The click head is supervised directly; the conversion head only through the
long_view label (ESMM's trick -- no direct conversion label, no sample-selection
bias). Scored by `p_ctr * p_cvr`.

This is NOT the Phase 1C `is_click` result. That added `is_click` as a co-equal
0.3-weighted auxiliary head and found it HARMFUL (the seesaw: a dense signal
competing for shared capacity). ESMM composes the heads MULTIPLICATIVELY and
never supervises the conversion head directly -- a different mechanism.

CONTROL (KB control_rule): `no_gate` -- the same two-head net scored by
sigma(z_cvr) alone. Isolates the multiplicative structure from "a second head".

Also run on top of the BPR head is out of scope for this phase (ESMM is
inherently pointwise on the composed probability); the honest comparison is
ESMM vs the pointwise FM baseline, and separately vs BPR as the standing best.

Stages:  --stage main | replicate
Run:  python experiments/p15_esmm.py --stage main
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from explib import dataset as D, features as F, esmm, harness as H, unbiased as U

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STATE = os.path.join(ROOT, 'cache', 'p15_best.json')

HYP = ('P(long_view) = P(click) * P(long_view|click); is_click=0 rows are long_view=0 by '
       'construction so the funnel is real. Conversion head supervised only through the '
       'long_view label (ESMM). KNOWLEDGE_BASE_PLAN.md Phase 1c; primer A.2/A.3')


def setup():
    logs = D.load_logs()
    masks = D.split_slices(logs)
    enc, dim, encoder = F.encode_splits(logs, masks, F.BASELINE_FIELDS)
    rand_enc = U.load_random_encoded(encoder)
    tr = masks['train']
    y_lv = (logs['long_view'][tr] != 0).astype(np.float32)
    y_click = (logs['is_click'][tr] != 0).astype(np.float32)
    return enc, dim, rand_enc, y_lv, y_click


def run(eid, enc, dim, rand_enc, y_lv, y_click, *, w_click, mode, seed=0,
        lr=0.001, epochs=40, patience=4, extra=None):
    if eid in {r['exp_id'] for r in H.read_log() if r['phase'] == '15'}:
        print(f'  skip {eid}')
        return None
    cfg = dict(model='esmm', k=16, lr=lr, l2=1e-6, bs=8192, seed=seed,
               fields=F.BASELINE_FIELDS, w_click=w_click, score_mode=mode,
               epochs=epochs, patience=patience, **(extra or {}))
    with H.Experiment(eid, phase='15', axis='multi_task', hypothesis=HYP,
                      config=cfg, tags=['multitask', 'esmm']) as ex:
        m, info = esmm.train(enc['train'][0], y_lv, y_click, *enc['valid'], dim,
                             w_click=w_click, mode=mode, k=16, lr=lr, l2=1e-6,
                             epochs=epochs, patience=patience, seed=seed,
                             evaluator=H.score, verbose=False)
        ex.record_train(**{k: v for k, v in info.items() if k != 'history'})
        sc = U.evaluate_all(lambda X: m.predict(X, mode=mode), enc, rand_enc)
        for sp, s in sc.items():
            ex.record_metrics(sp, s)
    row = {sp: round(float(sc[sp]['primary']), 5) for sp in sc}
    print(f"  {eid:40s} v {row['valid']:.4f} t {row['test']:.4f} "
          f"rv {row['rand_valid']:.4f} rt {row['rand_test']:.4f}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['main', 'replicate'])
    a = ap.parse_args()
    enc, dim, rand_enc, y_lv, y_click = setup()
    args = (enc, dim, rand_enc, y_lv, y_click)
    print(f'click rate {y_click.mean():.3f}, long_view rate {y_lv.mean():.3f}, '
          f'P(lv|click) {y_lv[y_click > 0].mean():.3f}, P(lv|noclick) {y_lv[y_click == 0].mean():.4f}')

    if a.stage == 'main':
        best = None
        for w in (0.3, 1.0):
            r = run(f'15-esmm-w{w:g}-gate', *args, w_click=w, mode='esmm')
            if r and (best is None or r['valid'] > best['valid']):
                best = dict(r, w_click=w)
        run('15-esmm-w1-nogate', *args, w_click=1.0, mode='cvr_only',
            extra={'control': 'no_multiplicative_gate'})
        if best:
            json.dump(best, open(STATE, 'w'))
            print(f"\nmain best: w_click={best['w_click']} valid {best['valid']:.4f}")

    elif a.stage == 'replicate':
        w = json.load(open(STATE))['w_click']
        for seed in range(3):
            sfx = f'-seed{seed}' if seed else '-rep-seed0'
            run(f'15-esmm-w{w:g}-gate{sfx}', *args, w_click=w, mode='esmm', seed=seed)

    print('\n' + H.summarize([r for r in H.read_log() if r['phase'] == '15']))


if __name__ == '__main__':
    main()
