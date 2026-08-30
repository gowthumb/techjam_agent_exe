"""Unbiased evaluation via the randomly-exposed-video log.

`KNOWLEDGE_BASE_PLAN.md` Phase 0 ("before exploring anything, make sure your
numbers are trustworthy") and the starter kit README's unexplored direction #7:
`log_random_4_22_to_5_08_pure.csv` is 1.19M *randomly exposed* impressions over
the valid+test period. Under random exposure the `long_view` rate is 0.085 vs
0.31 in the biased standard logs -- it is a genuinely different distribution.

WHAT THIS IS FOR. Not a score to optimize. A second, bias-free model-selection
signal: a config chosen on `valid` (which is whatever the production recommender
chose to show) can be fitting the exposure policy rather than user preference.
The KB already documents a "valid blind spot" -- interventions flat on valid but
consistently better on test. An unbiased split is the check for exactly that.

CONVENTIONS (match the standard splits so scoring is like-for-like):
  * split by the official date windows: rand_valid 20220422-28, rand_test
    20220429-0508;
  * encode with a FROZEN train-fitted `features.Encoder` -- unseen ids fall to
    that field's UNK slot, no vocabulary is refit;
  * tie-breaking among near-equal configs uses `rand_valid` ONLY. `rand_test`
    spans the hidden-test period and is report-only.

Scoring is delegated to the starter kit's `evaluate.py` via `harness.score`,
never reimplemented.
"""
import numpy as np
from . import dataset as D, harness as H

RAND_SPLITS = {'rand_valid': (20220422, 20220428),
               'rand_test':  (20220429, 20220508)}


def load_random_encoded(encoder, extra_col_builder=None):
    """Encode every random-exposure row with a frozen train-fitted `encoder`.

    Returns {split: (X, y, users)} for rand_valid / rand_test -- the same shape
    `features.encode_splits` produces, so an experiment scores it with the exact
    predict -> H.score path it already uses for valid/test.

    extra_col_builder: optional f(random_logs) -> {field_name: int array over all
    random-log rows}. Needed when the encoder was fit with injected columns
    (Phase 14 features): the same columns must be recomputed for the random log
    and handed to the encoder before transform.
    """
    logs = D.load_logs(random_log=True)
    if extra_col_builder is not None:
        rcols = extra_col_builder(logs)
        encoder.extra_cols = {k: v for k, v in rcols.items() if k in encoder.fields}
    d = logs['date']
    out = {}
    for name, (lo, hi) in RAND_SPLITS.items():
        idx = np.flatnonzero((d >= lo) & (d <= hi))
        X = encoder.transform(logs, idx)
        y = (logs[D.LABEL][idx] != 0).astype(np.float32)
        u = logs['user_id'][idx].copy()
        out[name] = (X, y, u)
    return out


def evaluate_all(predict_fn, enc, rand_enc):
    """Score one model on all four splits in a single call.

    predict_fn : X -> real-valued scores (e.g. `model.predict`)
    enc        : {split: (X, y, u)} for valid/test from features.encode_splits
    rand_enc   : {split: (X, y, u)} from load_random_encoded
    -> {'valid', 'test', 'rand_valid', 'rand_test'}: full H.score dicts
    """
    out = {}
    for name, src in (('valid', enc['valid']), ('test', enc['test']),
                      ('rand_valid', rand_enc['rand_valid']),
                      ('rand_test', rand_enc['rand_test'])):
        X, y, u = src
        out[name] = H.score(u, y, predict_fn(X))
    return out


def calibration_rungs(rand_enc, seed=0):
    """random-scoring and item-popularity rungs on the unbiased splits.

    The KB's `calibration` section judges every score against fixed rungs. The
    biased splits have them (baseline_scores.json); the unbiased splits need
    their own, because their label rate is a third of the biased one and the
    absolute numbers are not comparable across the two.
    """
    logs = D.load_logs(random_log=True)
    d = logs['date']
    y_all = (logs[D.LABEL] != 0).astype(np.float64)
    # popularity: smoothed long_view rate per video, fit on STANDARD train only
    std = D.load_logs()
    sm = D.split_slices(std)['train']
    ys = (std[D.LABEL][sm] != 0).astype(np.float64)
    vid = std['video_id'][sm]
    gmean = float(ys.mean())
    pos = np.bincount(vid, weights=ys, minlength=int(std['video_id'].max()) + 1)
    imp = np.bincount(vid, minlength=int(std['video_id'].max()) + 1).astype(np.float64)
    prior = 20.0
    rng = np.random.default_rng(seed)
    out = {}
    for name, (lo, hi) in RAND_SPLITS.items():
        idx = np.flatnonzero((d >= lo) & (d <= hi))
        u = logs['user_id'][idx]
        y = y_all[idx]
        v = logs['video_id'][idx]
        v = np.clip(v, 0, len(pos) - 1)
        pop = (pos[v] + prior * gmean) / (imp[v] + prior)
        pop = np.where(imp[v] > 0, pop, gmean)
        out[name] = {
            'random':     H.score(u, y, rng.random(len(y)))['primary'],
            'popularity': H.score(u, y, pop)['primary'],
            'oracle':     H.score(u, y, y + rng.random(len(y)) * 1e-6)['primary'],
            'label_rate': round(float(y.mean()), 4),
            'rows': int(len(y)), 'users': int(len(np.unique(u))),
        }
    return out
