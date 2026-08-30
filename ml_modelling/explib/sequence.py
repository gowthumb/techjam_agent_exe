"""Causal user behaviour sequences + a DIN-style attention model.

The starter kit's features are entirely flat: no feature encodes what the user did
before the impression being scored. The organizers list this as the single largest
untouched direction, so it gets a real model rather than a proxy.

SEQUENCE CONSTRUCTION -- two kinds, with different leakage properties:

  exposure : the last L videos shown to this user, strictly earlier in time.
             Uses no labels at all, so it is safe on every split and may cross
             into the eval period exactly as it would at serving time.

  positive : the last L videos this user long-viewed. This reads labels, so it is
             built from TRAIN-PERIOD LABELS ONLY; an eval row sees the user's
             train-period positives and nothing newer. Without that restriction it
             would read the very labels being scored.

Both are built vectorized (lexsort + per-group lag gather), not with a Python loop
over 1.4M rows.
"""
import numpy as np

PAD = 0          # id 0 is reserved for padding in every sequence vocabulary


def build_sequences(user_ids, item_ids, time_ms, L=20, valid_mask=None):
    """Last-L causal history per row.

    valid_mask: if given, only rows where it is True may ENTER a history (used to
    restrict the positive sequence to train-period rows). Every row still RECEIVES
    a history.

    Returns (H, hist_len): H is (N, L) int32 of item ids, most recent first, padded
    with PAD; hist_len is (N,) the number of real entries.
    """
    n = len(user_ids)
    order = np.lexsort((time_ms, user_ids))
    u = user_ids[order]
    it = item_ids[order].astype(np.int64)
    keep = np.ones(n, dtype=bool) if valid_mask is None else valid_mask[order]

    new = np.r_[True, u[1:] != u[:-1]]
    gid = np.cumsum(new) - 1
    start = np.flatnonzero(new)
    pos_in_group = np.arange(n) - start[gid]

    # Rank of each row among the *keepable* rows before it, within its user.
    kc = np.cumsum(keep)
    base = np.r_[0, kc][start]
    n_kept_before = (kc - keep) - base[gid]          # exclusive count of kept rows

    # Index (in sorted space) of the j-th most recent kept row for each row.
    kept_idx = np.flatnonzero(keep)
    kept_group_base = np.searchsorted(kept_idx, start)   # first kept slot per group

    H = np.zeros((n, L), dtype=np.int32)
    for lag in range(1, L + 1):
        has = n_kept_before >= lag
        if not has.any():
            break
        slot = kept_group_base[gid[has]] + (n_kept_before[has] - lag)
        H[has, lag - 1] = it[kept_idx[slot]]
    hist_len = np.minimum(n_kept_before, L).astype(np.int32)

    out_H = np.zeros_like(H)
    out_len = np.zeros_like(hist_len)
    out_H[order] = H
    out_len[order] = hist_len
    return out_H, out_len


def sequence_stats(H, hist_len, masks):
    return {sp: {'mean_hist_len': float(hist_len[m].mean()),
                 'pct_empty': float((hist_len[m] == 0).mean()),
                 'pct_full': float((hist_len[m] == H.shape[1]).mean())}
            for sp, m in masks.items()}
