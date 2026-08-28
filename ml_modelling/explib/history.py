"""Causal history / affinity features.

The organizers' note that "pure user-side features contribute exactly 0" is a
consequence of within-user ranking: anything constant inside a user cannot change
the within-user order. History features dodge that because they are keyed on
(user, something-about-the-item) -- they vary across a user's own impressions.

LEAKAGE CONTRACT (the part that must not be got wrong):
  * Counters are built from TRAIN LABELS ONLY. valid/test labels never enter a
    feature. In the real evaluation the test labels do not exist; a feature that
    reads them would score brilliantly here and collapse on the hidden set.
  * Inside train, a row must not see its own label. Two supported modes:
      'causal' -- a row sees only that key's strictly-earlier train rows
                  (chronological by time_ms). Matches deployment, but early rows
                  have thin history, so train and eval see different-quality
                  features.
      'loo'    -- a row sees the full train counters minus its own contribution.
                  Uniform feature quality, standard leave-one-out target encoding.
  * valid/test rows always read the frozen full-train counters.

Rates are smoothed toward the global train rate with a Beta prior, then bucketed
into categorical ids so they drop straight into the existing FM embedding table.
"""
import numpy as np


def _dense_key(*cols):
    """Map a tuple of int columns to dense ids, plus the inverse lookup dict."""
    stacked = np.stack(cols, axis=1)
    uniq, inv = np.unique(stacked, axis=0, return_inverse=True)
    lookup = {tuple(int(v) for v in row): i for i, row in enumerate(uniq)}
    return inv.astype(np.int64), lookup, len(uniq)


def _causal_prior_counts(kid, order_key, y):
    """Per-row (positives, impressions) over strictly-earlier rows of the same key.

    Fully vectorized: sort by (key, time), take a group-local exclusive cumsum.
    """
    order = np.lexsort((order_key, kid))
    k = kid[order]
    yy = y[order].astype(np.float64)
    n = len(k)
    new = np.r_[True, k[1:] != k[:-1]]
    gid = np.cumsum(new) - 1
    start = np.flatnonzero(new)
    csum = np.cumsum(yy)
    excl = csum - yy                       # global exclusive cumsum
    base = np.r_[0.0, csum][start]         # global cumsum just before each group
    pos = excl - base[gid]
    cnt = (np.arange(n) - start[gid]).astype(np.float64)
    out_pos = np.empty(n); out_cnt = np.empty(n)
    out_pos[order] = pos
    out_cnt[order] = cnt
    return out_pos, out_cnt


class AffinityFeature:
    """One (key -> long_view rate) history feature, fit on train, applied anywhere.

    key_cols: names of log columns forming the key, e.g. ('user_id','author_id').
              'author_id' is resolved from the video-side table by the caller and
              passed in as an already-materialized column.
    """

    def __init__(self, name, prior=20.0, mode='causal', n_buckets=16):
        self.name = name
        self.prior = prior
        self.mode = mode
        self.n_buckets = n_buckets
        self.gmean = None
        self.tot_pos = None
        self.tot_cnt = None
        self.lookup = None
        self.edges = None

    def fit_transform_train(self, key_cols, time_col, y):
        kid, lookup, nk = _dense_key(*key_cols)
        self.lookup = lookup
        self.gmean = float(y.mean())
        self.tot_pos = np.bincount(kid, weights=y.astype(np.float64), minlength=nk)
        self.tot_cnt = np.bincount(kid, minlength=nk).astype(np.float64)
        if self.mode == 'causal':
            pos, cnt = _causal_prior_counts(kid, time_col, y)
        elif self.mode == 'loo':
            pos = self.tot_pos[kid] - y
            cnt = self.tot_cnt[kid] - 1.0
        else:
            raise ValueError(f'unknown mode {self.mode}')
        rate = (pos + self.prior * self.gmean) / (cnt + self.prior)
        self.edges = np.quantile(rate, np.linspace(0, 1, self.n_buckets + 1)[1:-1])
        return rate, cnt

    def transform(self, key_cols):
        """Eval rows read the frozen full-train counters. Unseen key -> prior only."""
        n = len(key_cols[0])
        pos = np.zeros(n); cnt = np.zeros(n)
        cols = [np.asarray(c) for c in key_cols]
        get = self.lookup.get
        for i in range(n):
            j = get(tuple(int(c[i]) for c in cols), -1)
            if j >= 0:
                pos[i] = self.tot_pos[j]
                cnt[i] = self.tot_cnt[j]
        rate = (pos + self.prior * self.gmean) / (cnt + self.prior)
        return rate, cnt

    def bucketize(self, rate, cnt, cold_bucket=True):
        """Rate -> categorical id. Rows with no history get their own id, because
        'no evidence' is a different statement from 'evidence of an average rate'."""
        b = np.searchsorted(self.edges, rate).astype(np.int32)
        if cold_bucket:
            b = np.where(cnt > 0, b + 1, 0).astype(np.int32)
        return b


def author_column(logs, vid2author):
    """Materialize author_id per log row (UNK -> -1)."""
    return np.array([vid2author.get(int(v), -1) for v in logs['video_id']], dtype=np.int64)


def build_affinity_fields(logs, masks, specs, mode='causal', prior=20.0, n_buckets=16):
    """Build several affinity features at once.

    specs: list of (field_name, [key column arrays over ALL rows]).
    Returns {field_name: int32 array over all rows} plus the fitted objects.
    """
    y_all = (logs['long_view'] != 0).astype(np.float64)
    tr = masks['train']
    t = logs['time_ms'] if 'time_ms' in logs else logs['date'].astype(np.int64)
    out, fitted = {}, {}
    for name, key_cols in specs:
        af = AffinityFeature(name, prior=prior, mode=mode, n_buckets=n_buckets)
        rate_tr, cnt_tr = af.fit_transform_train([c[tr] for c in key_cols], t[tr], y_all[tr])
        col = np.zeros(len(y_all), dtype=np.int32)
        col[tr] = af.bucketize(rate_tr, cnt_tr)
        for split, m in masks.items():
            if split == 'train':
                continue
            rate, cnt = af.transform([c[m] for c in key_cols])
            col[m] = af.bucketize(rate, cnt)
        out[name] = col
        fitted[name] = af
    return out, fitted
