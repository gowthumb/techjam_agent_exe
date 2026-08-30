"""KuaiRand-Pure data layer for the ML-modelling workstream.

Extends the starter kit's `data.py` in two ways the exploration needs:
  1. keeps *all* 19 log columns (the 11 unscored feedback signals, play_time_ms,
     hourmin, is_rand ...), not just the 5 baseline fields;
  2. caches the parsed logs to .npz so a run costs ~1s instead of ~60s of CSV parsing.

Row order is byte-identical to the starter kit's `data.load()[split]` -- verified by
`verify_row_order_matches_starter_kit()`. Submission alignment depends on this.
"""
import os, csv, hashlib, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
KIT = os.path.join(REPO, 'kuairand-starter-kit')
DATA_DIR = os.path.join(KIT, 'KuaiRand-Pure', 'data')
CACHE_DIR = os.path.join(REPO, 'ml_modelling', 'cache')

LABEL = 'long_view'
SPLITS = {'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}

# Order matters: the starter kit reads these two files in this order and keeps
# each file's original row order after filtering by date.
LOG_FILES = ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv')
RANDOM_LOG = 'log_random_4_22_to_5_08_pure.csv'

# All 19 log columns except user_id/video_id (kept separately as int64).
INT_COLS = ['date', 'hourmin', 'is_click', 'is_like', 'is_follow', 'is_comment',
            'is_forward', 'is_hate', 'long_view', 'is_profile_enter', 'is_rand', 'tab']
# Unix ms timestamps overflow int32, so they get their own int64 column. This is
# the ordering key for every causal feature -- date alone only resolves to a day.
BIG_INT_COLS = ['time_ms']
FLOAT_COLS = ['play_time_ms', 'duration_ms', 'profile_stay_time', 'comment_stay_time']

# The 11 feedback signals other than long_view that are logged but not scored.
AUX_SIGNALS = ['is_click', 'is_like', 'is_follow', 'is_comment', 'is_forward',
               'is_hate', 'is_profile_enter']

VIDEO_CAT_COLS = ['author_id', 'video_type', 'upload_type', 'music_id', 'music_type', 'tag']
VIDEO_NUM_COLS = ['video_duration', 'server_width', 'server_height']
USER_CAT_COLS = ['user_active_degree', 'is_lowactive_period', 'is_live_streamer',
                 'is_video_author', 'follow_user_num_range', 'fans_user_num_range',
                 'friend_user_num_range', 'register_days_range'] + \
                [f'onehot_feat{i}' for i in range(18)]


def _cache_key(*parts):
    return hashlib.md5('|'.join(str(p) for p in parts).encode()).hexdigest()[:12]


def _parse_logs(files):
    """Parse log CSVs into a dict of numpy arrays, preserving file+row order."""
    uid, vid = [], []
    ints = {c: [] for c in INT_COLS}
    bigs = {c: [] for c in BIG_INT_COLS}
    flts = {c: [] for c in FLOAT_COLS}
    for f in files:
        with open(os.path.join(DATA_DIR, f), newline='') as fh:
            for r in csv.DictReader(fh):
                uid.append(int(r['user_id']))
                vid.append(int(r['video_id']))
                for c in INT_COLS:
                    ints[c].append(int(float(r[c])))
                for c in BIG_INT_COLS:
                    bigs[c].append(int(float(r[c])))
                for c in FLOAT_COLS:
                    flts[c].append(float(r[c]))
    out = {'user_id': np.asarray(uid, dtype=np.int64),
           'video_id': np.asarray(vid, dtype=np.int64)}
    for c in INT_COLS:
        out[c] = np.asarray(ints[c], dtype=np.int32)
    for c in BIG_INT_COLS:
        out[c] = np.asarray(bigs[c], dtype=np.int64)
    for c in FLOAT_COLS:
        out[c] = np.asarray(flts[c], dtype=np.float32)
    return out


def load_logs(random_log=False, refresh=False):
    """Standard (train/valid/test) logs, or the unbiased random-exposure log.

    Returns a dict of equal-length numpy arrays, one per column.
    """
    files = (RANDOM_LOG,) if random_log else LOG_FILES
    path = os.path.join(CACHE_DIR, f'logs_{_cache_key(*files, "v2-time_ms")}.npz')
    if os.path.exists(path) and not refresh:
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    os.makedirs(CACHE_DIR, exist_ok=True)
    out = _parse_logs(files)
    np.savez_compressed(path, **out)
    return out


def _parse_side(fname, key, cat_cols, num_cols):
    """Parse a side-feature CSV into id -> encoded row. Categoricals stay as strings."""
    cats, nums, ids = [], [], []
    with open(os.path.join(DATA_DIR, fname), newline='') as fh:
        for r in csv.DictReader(fh):
            ids.append(int(r[key]))
            cats.append([r[c] for c in cat_cols])
            nums.append([float(r[c]) if r[c] not in ('', 'NA') else np.nan for c in num_cols])
    return (np.asarray(ids, dtype=np.int64),
            np.asarray(cats, dtype=object),
            np.asarray(nums, dtype=np.float32))


def load_video_features():
    return _parse_side('video_features_basic_pure.csv', 'video_id',
                       VIDEO_CAT_COLS, VIDEO_NUM_COLS)


def load_user_features():
    return _parse_side('user_features_pure.csv', 'user_id', USER_CAT_COLS, [])


def split_slices(logs):
    """Boolean masks per split, matching the official date ranges."""
    d = logs['date']
    return {name: (d >= lo) & (d <= hi) for name, (lo, hi) in SPLITS.items()}


def group_index(user_ids):
    """Contiguous-group index for within-user losses/metrics.

    Returns (order, starts) where order sorts rows by user (stable, so original
    row order is preserved inside each user) and starts marks group boundaries,
    i.e. group g occupies order[starts[g]:starts[g+1]].
    """
    order = np.argsort(user_ids, kind='stable')
    su = user_ids[order]
    bounds = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
    starts = np.r_[bounds, len(su)]
    return order, starts


def verify_row_order_matches_starter_kit():
    """Assert this loader reproduces `data.load()` row-for-row. Cheap insurance:
    submission row_id alignment is defined by the starter kit's ordering."""
    import sys
    sys.path.insert(0, KIT)
    import data as kit_data
    kit = kit_data.load(DATA_DIR)
    logs = load_logs()
    masks = split_slices(logs)
    report = {}
    for name in SPLITS:
        m = masks[name]
        ku = [int(x[1]) for x in kit[name]]
        kv = [int(x[2]) for x in kit[name]]
        ky = [x[6] for x in kit[name]]
        ok = (len(ku) == int(m.sum())
              and np.array_equal(np.asarray(ku), logs['user_id'][m])
              and np.array_equal(np.asarray(kv), logs['video_id'][m])
              and np.array_equal(np.asarray(ky, dtype=np.int32),
                                 (logs[LABEL][m] != 0).astype(np.int32)))
        report[name] = {'rows': int(m.sum()), 'matches_starter_kit': bool(ok)}
    return report


if __name__ == '__main__':
    rep = verify_row_order_matches_starter_kit()
    print(json.dumps(rep, indent=2))
    assert all(v['matches_starter_kit'] for v in rep.values()), 'ROW ORDER MISMATCH'
    print('OK: row order identical to starter kit data.load()')
