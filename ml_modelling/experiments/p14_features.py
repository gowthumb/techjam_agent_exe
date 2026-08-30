"""Phase 14 / axis: duration-regime and video-freshness features.

GROUNDING. `KNOWLEDGE_BASE_PLAN.md` Phase 1b (bucketing / hashing, explicit
crosses, per-field ablation); starter kit README directions #4 and #6 (duration,
time features). The KB ranks feature work below the loss ("spend the iteration on
the loss"), so this runs on top of the confirmed BPR config, not pointwise.

HYPOTHESES.
  duration_regime -- `long_view` bifurcates at duration 18000 ms: for videos
    <=18s it is complete-play, for >18s it is watch>=18s. These are different
    prediction problems with different base rates. The baseline's flat 10-way
    `dur_bucket` cannot represent the discontinuity; a 0/1 regime field can, and
    the FM will cross it with user_id / author_id implicitly. This is NOT the
    static-side-feature dead end -- it is an interaction the model currently
    lacks, not a redundant user-side column.
  video_age_bucket -- days between upload and impression. A fresh video behaves
    differently (novelty, trending), and age varies across a user's own
    impression list, so unlike a pure user-side feature it CAN change the
    within-user order. Never tried (only author/tag/music were pulled video-side).
  dur_buckets=20 -- a finer duration grid, cheap to check while here.

CONTROL (KB control_rule): duration_regime with its values shuffled across rows.
If the shuffled field helps as much, the gain is capacity/regularisation, not the
regime information.

Stages:  --stage main | control | replicate
Run:  python experiments/p14_features.py --stage main
"""
import os, sys, csv, json, argparse, datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from explib import dataset as D, features as F, fm, harness as H, unbiased as U

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STATE = os.path.join(ROOT, 'cache', 'p14_best.json')

HYP = ('long_view bifurcates at duration 18s and video freshness varies within a user list; '
       'a regime field and an age bucket are interactions the baseline five fields cannot '
       'represent. KNOWLEDGE_BASE_PLAN.md Phase 1b; README directions #4/#6')

BPR = dict(loss='bpr', k=16, lr=0.0002, epochs=60, patience=6)


def _date_ord(date_ints):
    d = np.asarray(date_ints, dtype=np.int64)
    uniq = np.unique(d)
    lut = {int(x): datetime.date(int(x) // 10000, (int(x) // 100) % 100, int(x) % 100).toordinal()
           for x in uniq}
    keys = np.array(sorted(lut)); vals = np.array([lut[k] for k in keys])
    return vals[np.searchsorted(keys, d)]


def build_feature_cols(logs):
    """Integer columns over ALL rows, for injection via Encoder(extra_cols=...)."""
    dur = logs['duration_ms'].astype(np.float64)
    regime = (dur <= 18000).astype(np.int32)                       # 0 long, 1 short

    # video upload date -> age in days at impression time
    upload = {}
    path = os.path.join(D.DATA_DIR, 'video_features_basic_pure.csv')
    with open(path, newline='') as fh:
        for r in csv.DictReader(fh):
            s = r.get('upload_dt', '')
            if s and s not in ('NA', ''):
                try:
                    y, m, d = s.split('-')
                    upload[int(r['video_id'])] = datetime.date(int(y), int(m), int(d)).toordinal()
                except ValueError:
                    pass
    imp_ord = _date_ord(logs['date'])
    up_ord = np.array([upload.get(int(v), -1) for v in logs['video_id']], dtype=np.int64)
    age = np.where(up_ord >= 0, imp_ord - up_ord, -1)
    edges = np.array([1, 3, 7, 14, 30, 90])
    age_bucket = np.where(age < 0, 0, 1 + np.searchsorted(edges, age)).astype(np.int32)
    return {'duration_regime': regime, 'video_age_bucket': age_bucket}, age


def encode_with(logs, masks, extra_fields, extra_cols, dur_buckets=10):
    fields = list(F.BASELINE_FIELDS) + list(extra_fields)
    return F.encode_splits(logs, masks, fields, dur_buckets=dur_buckets,
                           extra_cols=extra_cols)


def run(eid, enc, dim, rand_enc, *, seed=0, extra=None):
    if eid in {r['exp_id'] for r in H.read_log() if r['phase'] == '14'}:
        print(f'  skip {eid}')
        return None
    cfg = dict(model='fm', bs=8192, seed=seed, fields='see extra', **BPR, **(extra or {}))
    with H.Experiment(eid, phase='14', axis='feature_engineering', hypothesis=HYP,
                      config=cfg, tags=['feature', 'bpr']) as ex:
        m, info = fm.train(enc, dim, loss='bpr', k=16, lr=0.0002, l2=1e-6,
                           epochs=60, patience=6, seed=seed, evaluator=H.score,
                           verbose=False)
        ex.record_train(**{k: v for k, v in info.items() if k != 'history'})
        sc = U.evaluate_all(m.predict, enc, rand_enc)
        for sp, s in sc.items():
            ex.record_metrics(sp, s)
    row = {sp: round(float(sc[sp]['primary']), 5) for sp in sc}
    print(f"  {eid:44s} v {row['valid']:.4f} t {row['test']:.4f} "
          f"rv {row['rand_valid']:.4f} rt {row['rand_test']:.4f}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['main', 'control', 'replicate'])
    a = ap.parse_args()

    logs = D.load_logs()
    masks = D.split_slices(logs)
    cols, age = build_feature_cols(logs)
    warm = float((age >= 0)[masks['test']].mean())
    print(f'video_age warm coverage on test: {warm:.1%}  '
          f'(regime split: {cols["duration_regime"].mean():.2f} short)')

    def rand_for(encoder):
        return U.load_random_encoded(
            encoder, extra_col_builder=lambda rl: build_feature_cols(rl)[0])

    if a.stage == 'main':
        best = None
        variants = [
            ('14-bpr-regime',        ['duration_regime'], {'duration_regime': cols['duration_regime']}, 10),
            ('14-bpr-videoage',      ['video_age_bucket'], {'video_age_bucket': cols['video_age_bucket']}, 10),
            ('14-bpr-regime+age',    ['duration_regime', 'video_age_bucket'], cols, 10),
            ('14-bpr-durbuckets20',  [], {}, 20),
        ]
        for eid, xf, xc, db in variants:
            enc, dim, encoder = encode_with(logs, masks, xf, xc, dur_buckets=db)
            r = run(eid, enc, dim, rand_for(encoder))
            if r and (best is None or r['valid'] > best['valid']):
                best = dict(r, eid=eid, extra_fields=xf, dur_buckets=db)
        if best:
            json.dump({k: best[k] for k in ('eid', 'extra_fields', 'dur_buckets', 'valid')},
                      open(STATE, 'w'))
            print(f"\nmain best: {best['eid']} valid {best['valid']:.4f}")

    elif a.stage == 'control':
        rng = np.random.default_rng(14)
        shuffled = cols['duration_regime'][rng.permutation(len(cols['duration_regime']))]
        enc, dim, encoder = encode_with(logs, masks, ['duration_regime'],
                                        {'duration_regime': shuffled}, dur_buckets=10)
        run('14-bpr-regime-shuffled', enc, dim, rand_for(encoder),
            extra={'control': 'shuffled_regime'})

    elif a.stage == 'replicate':
        b = json.load(open(STATE))
        xf = b['extra_fields']
        xc = {k: cols[k] for k in xf}
        for seed in range(3):
            enc, dim, encoder = encode_with(logs, masks, xf, xc, dur_buckets=b['dur_buckets'])
            sfx = f'-seed{seed}' if seed else '-rep-seed0'
            run(b['eid'] + sfx, enc, dim, rand_for(encoder), seed=seed)

    print('\n' + H.summarize([r for r in H.read_log() if r['phase'] == '14']))


if __name__ == '__main__':
    main()
