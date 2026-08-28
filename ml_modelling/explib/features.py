"""Configurable categorical feature encoding.

Reproduces the starter kit's 5-field encoding exactly when asked for
BASELINE_FIELDS, but lets any experiment add/remove fields by name so feature
ablations are a one-line config change rather than a forked data.py.

Convention (inherited from the kit, do not change): vocabularies are fit on
*train only*; every field gets a trailing UNK slot; ids are offset so all fields
share one flat embedding table.
"""
import numpy as np
from . import dataset as D

BASELINE_FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']

# Fields sourced from the video-side CSV, keyed by video_id.
_VIDEO_FIELDS = set(D.VIDEO_CAT_COLS)
# Fields sourced from the user-side CSV, keyed by user_id.
_USER_FIELDS = set(D.USER_CAT_COLS)


class Encoder:
    """Fits vocabularies on train rows, transforms any split to an int32 matrix."""

    def __init__(self, fields=None, dur_buckets=10, extra_cols=None):
        self.fields = list(fields or BASELINE_FIELDS)
        self.dur_buckets = dur_buckets
        # extra_cols: {field_name: int array over ALL rows} -- precomputed
        # columns (e.g. causal affinity buckets) injected by an experiment.
        self.extra_cols = dict(extra_cols or {})
        self.vocabs = None
        self.field_dims = None
        self.offsets = None
        self.total_dim = None
        self._edges = None
        self._vmap = None
        self._umap = None

    # ---- raw string values per field, one array per field ----
    def _raw(self, logs, idx):
        cols = []
        for f in self.fields:
            if f in self.extra_cols:
                cols.append(self.extra_cols[f][idx].astype(str))
            elif f == 'user_id':
                cols.append(logs['user_id'][idx].astype(str))
            elif f == 'video_id':
                cols.append(logs['video_id'][idx].astype(str))
            elif f == 'tab':
                cols.append(logs['tab'][idx].astype(str))
            elif f == 'dur_bucket':
                b = np.searchsorted(self._edges, logs['duration_ms'][idx])
                cols.append(b.astype(str))
            elif f == 'hour':
                cols.append((logs['hourmin'][idx] // 100).astype(str))
            elif f == 'weekday':
                # date is YYYYMMDD; only used for drift experiments (phase 1f)
                cols.append((logs['date'][idx] % 7).astype(str))
            elif f in _VIDEO_FIELDS:
                j = self._vmap['pos'].get
                ids = logs['video_id'][idx]
                col = self._vmap[f]
                pos = np.array([self._vmap['pos'].get(int(v), -1) for v in ids])
                out = np.where(pos >= 0, col[np.clip(pos, 0, None)], 'UNK')
                cols.append(out.astype(str))
            elif f in _USER_FIELDS:
                ids = logs['user_id'][idx]
                col = self._umap[f]
                pos = np.array([self._umap['pos'].get(int(u), -1) for u in ids])
                out = np.where(pos >= 0, col[np.clip(pos, 0, None)], 'UNK')
                cols.append(out.astype(str))
            else:
                raise KeyError(f'unknown field: {f}')
        return cols

    def _ensure_side(self):
        if any(f in _VIDEO_FIELDS for f in self.fields) and self._vmap is None:
            ids, cats, _ = D.load_video_features()
            self._vmap = {'pos': {int(v): i for i, v in enumerate(ids)}}
            for i, c in enumerate(D.VIDEO_CAT_COLS):
                self._vmap[c] = cats[:, i].astype(str)
        if any(f in _USER_FIELDS for f in self.fields) and self._umap is None:
            ids, cats, _ = D.load_user_features()
            self._umap = {'pos': {int(u): i for i, u in enumerate(ids)}}
            for i, c in enumerate(D.USER_CAT_COLS):
                self._umap[c] = cats[:, i].astype(str)

    def fit(self, logs, train_idx):
        self._ensure_side()
        if 'dur_bucket' in self.fields:
            self._edges = np.quantile(logs['duration_ms'][train_idx],
                                      np.linspace(0, 1, self.dur_buckets + 1)[1:-1])
        cols = self._raw(logs, train_idx)
        self.vocabs = []
        for col in cols:
            uniq = np.unique(col)
            self.vocabs.append({v: i for i, v in enumerate(uniq)})
        self.field_dims = [len(v) + 1 for v in self.vocabs]      # +1 UNK slot
        self.offsets = np.cumsum([0] + self.field_dims[:-1]).astype(np.int32)
        self.total_dim = int(sum(self.field_dims))
        return self

    def transform(self, logs, idx):
        cols = self._raw(logs, idx)
        n = len(cols[0])
        X = np.empty((n, len(self.fields)), dtype=np.int32)
        for i, col in enumerate(cols):
            vocab, unk, off = self.vocabs[i], len(self.vocabs[i]), self.offsets[i]
            X[:, i] = np.fromiter((vocab.get(v, unk) for v in col),
                                  dtype=np.int32, count=n) + off
        return X


def encode_splits(logs, masks, fields=None, dur_buckets=10, extra_cols=None):
    """-> (enc dict {split: (X, y, users)}, total_dim, Encoder)."""
    e = Encoder(fields, dur_buckets, extra_cols).fit(logs, masks['train'])
    out = {}
    for name, m in masks.items():
        out[name] = (e.transform(logs, m),
                     (logs[D.LABEL][m] != 0).astype(np.float32),
                     logs['user_id'][m])
    return out, e.total_dim, e
