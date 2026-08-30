"""Multi-benchmark data layer (KuaiRand-Pure / 1K / 27K).

Phase 5 needs the same pipeline pointed at a bigger benchmark. Rather than
parameterize `dataset.py` -- whose Pure path is verified row-for-row against the
starter kit and determines 100% of the primary score -- this module wraps it and
resolves file names per benchmark. `dataset.py` stays untouched, so nothing here
can regress Pure.

File names are discovered by glob, not hardcoded, because the suffix differs per
release (`_pure` / `_1k` / `_27k`) and the date ranges in the file names may not.
The official split dates are shared across releases, so `dataset.SPLITS` applies
unchanged -- but `describe()` prints the observed date range so that assumption is
checked rather than trusted.
"""
import os, csv, glob, hashlib
import numpy as np
from . import dataset as D

BENCHMARKS = {
    'pure': 'KuaiRand-Pure',
    '1k':   'KuaiRand-1K',
    '27k':  'KuaiRand-27K',
}


def data_dir(bench):
    d = os.path.join(D.KIT, BENCHMARKS[bench], 'data')
    if not os.path.isdir(d):
        raise FileNotFoundError(f'{bench}: {d} not found — download and extract it first')
    return d


def resolve_files(bench):
    """-> (ordered standard logs, random log, user features, video basic features)."""
    d = data_dir(bench)
    std = sorted(glob.glob(os.path.join(d, 'log_standard_*.csv')))
    rnd = sorted(glob.glob(os.path.join(d, 'log_random_*.csv')))
    usr = sorted(glob.glob(os.path.join(d, 'user_features_*.csv')))
    vid = sorted(glob.glob(os.path.join(d, 'video_features_basic_*.csv')))
    if not std:
        raise FileNotFoundError(f'{bench}: no log_standard_*.csv in {d}')
    return std, (rnd[0] if rnd else None), (usr[0] if usr else None), (vid[0] if vid else None)


# The columns the scale experiments actually need: the five baseline fields, the
# label, and the split key. The other 13 exist for the multi-task and watch-time
# axes, which the KB already rules out, so carrying them at 1K/27K scale is pure
# memory cost.
MINIMAL_COLS = {'int': ['date', 'tab', 'long_view'], 'big': [], 'float': ['duration_ms']}


def _parse_log_files(files, int_cols, big_cols, flt_cols, max_rows=None):
    """Chunked parse of one or more log CSVs into a column dict.

    Shared by load_logs and load_random_logs -- the standard and random logs are
    the same 19-column schema, so there is nothing benchmark- or split-specific
    here, only which files get handed in.

    Parses in chunks, converting each chunk to numpy immediately. Peak memory is
    then one chunk of Python objects rather than the whole file's worth -- at 1K
    scale the full-file lists are ~200M int objects, which is what actually
    exhausts RAM, not the final arrays.
    """
    CHUNK = 2_000_000
    parts, n, stop = [], 0, False
    for f in files:
        with open(f, newline='') as fh:
            rdr = csv.reader(fh)
            ix = {name: i for i, name in enumerate(next(rdr))}
            buf_u, buf_v = [], []
            buf_i = {c: [] for c in int_cols}
            buf_b = {c: [] for c in big_cols}
            buf_f = {c: [] for c in flt_cols}

            def flush():
                if not buf_u:
                    return
                d = {'user_id': np.asarray(buf_u, dtype=np.int64),
                     'video_id': np.asarray(buf_v, dtype=np.int64)}
                for c in int_cols:
                    d[c] = np.asarray(buf_i[c], dtype=np.int32)
                for c in big_cols:
                    d[c] = np.asarray(buf_b[c], dtype=np.int64)
                for c in flt_cols:
                    d[c] = np.asarray(buf_f[c], dtype=np.float32)
                parts.append(d)
                buf_u.clear(); buf_v.clear()
                for c in int_cols:
                    buf_i[c].clear()
                for c in big_cols:
                    buf_b[c].clear()
                for c in flt_cols:
                    buf_f[c].clear()

            for row in rdr:
                buf_u.append(int(row[ix['user_id']]))
                buf_v.append(int(row[ix['video_id']]))
                for c in int_cols:
                    buf_i[c].append(int(float(row[ix[c]])))
                for c in big_cols:
                    buf_b[c].append(int(float(row[ix[c]])))
                for c in flt_cols:
                    buf_f[c].append(float(row[ix[c]]))
                n += 1
                if len(buf_u) >= CHUNK:
                    flush()
                if max_rows and n >= max_rows:
                    stop = True
                    break
            flush()
        if stop:
            break

    keys = ['user_id', 'video_id'] + int_cols + big_cols + flt_cols
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    del parts
    return out


def load_logs(bench='pure', refresh=False, max_rows=None, minimal=False):
    """Parsed logs for a benchmark, cached to npz. Same dict shape as dataset.load_logs.

    minimal=True keeps only the columns the scale experiments need.
    """
    if bench == 'pure' and max_rows is None and not minimal:
        return D.load_logs(refresh=refresh)      # the verified path, unchanged

    int_cols = MINIMAL_COLS['int'] if minimal else D.INT_COLS
    big_cols = MINIMAL_COLS['big'] if minimal else D.BIG_INT_COLS
    flt_cols = MINIMAL_COLS['float'] if minimal else D.FLOAT_COLS

    std, _, _, _ = resolve_files(bench)
    tag = hashlib.md5(('|'.join(os.path.basename(f) for f in std)
                       + f'|{max_rows}|{minimal}').encode()).hexdigest()[:12]
    path = os.path.join(D.CACHE_DIR, f'logs_{bench}_{tag}.npz')
    if os.path.exists(path) and not refresh:
        with np.load(path) as z:
            return {k: z[k] for k in z.files}

    out = _parse_log_files(std, int_cols, big_cols, flt_cols, max_rows)
    os.makedirs(D.CACHE_DIR, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def load_random_logs(bench='pure', refresh=False, minimal=False):
    """Parsed RANDOM-EXPOSURE log for a benchmark -- the unbiased-eval source.

    Same shape and caching convention as load_logs, pointed at log_random_*.csv
    instead of log_standard_*.csv. Delegates to dataset.load_logs(random_log=True)
    for Pure (the verified path) exactly like load_logs does for the standard case.
    """
    if bench == 'pure' and not minimal:
        return D.load_logs(random_log=True, refresh=refresh)

    int_cols = MINIMAL_COLS['int'] if minimal else D.INT_COLS
    big_cols = MINIMAL_COLS['big'] if minimal else D.BIG_INT_COLS
    flt_cols = MINIMAL_COLS['float'] if minimal else D.FLOAT_COLS

    _, rnd, _, _ = resolve_files(bench)
    if rnd is None:
        raise FileNotFoundError(f'{bench}: no log_random_*.csv in {data_dir(bench)}')
    tag = hashlib.md5((os.path.basename(rnd) + f'|{minimal}').encode()).hexdigest()[:12]
    path = os.path.join(D.CACHE_DIR, f'randlogs_{bench}_{tag}.npz')
    if os.path.exists(path) and not refresh:
        with np.load(path) as z:
            return {k: z[k] for k in z.files}

    out = _parse_log_files([rnd], int_cols, big_cols, flt_cols)
    os.makedirs(D.CACHE_DIR, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def load_video_features(bench='pure'):
    if bench == 'pure':
        return D.load_video_features()
    _, _, _, vid = resolve_files(bench)
    ids, cats, nums = [], [], []
    with open(vid, newline='') as fh:
        for r in csv.DictReader(fh):
            ids.append(int(r['video_id']))
            cats.append([r[c] for c in D.VIDEO_CAT_COLS])
            nums.append([float(r[c]) if r[c] not in ('', 'NA') else np.nan
                         for c in D.VIDEO_NUM_COLS])
    return (np.asarray(ids, dtype=np.int64), np.asarray(cats, dtype=object),
            np.asarray(nums, dtype=np.float32))


def describe(bench, logs=None, minimal=True):
    """Facts a KB entry would need before trusting any transferred prior."""
    logs = load_logs(bench, minimal=minimal) if logs is None else logs
    masks = D.split_slices(logs)
    y = (logs[D.LABEL] != 0).astype(float)
    out = {
        'benchmark': bench,
        'rows_total': int(len(y)),
        'date_range': [int(logs['date'].min()), int(logs['date'].max())],
        'users_total': int(len(np.unique(logs['user_id']))),
        'videos_total': int(len(np.unique(logs['video_id']))),
        'splits': {}, 'label_rate': {},
    }
    for sp, m in masks.items():
        out['splits'][sp] = {'rows': int(m.sum()),
                             'users': int(len(np.unique(logs['user_id'][m])))}
        out['label_rate'][sp] = round(float(y[m].mean()), 4) if m.any() else None
    tr_u = set(logs['user_id'][masks['train']].tolist())
    tr_v = set(logs['video_id'][masks['train']].tolist())
    for sp in ('valid', 'test'):
        m = masks[sp]
        if not m.any():
            continue
        u = set(logs['user_id'][m].tolist())
        v = set(logs['video_id'][m].tolist())
        out['splits'][sp]['users_seen_in_train'] = round(len(u & tr_u) / max(len(u), 1), 4)
        out['splits'][sp]['videos_seen_in_train'] = round(len(v & tr_v) / max(len(v), 1), 4)
    return out


def load_video_authors(bench):
    """video_id -> author_id as plain int arrays.

    load_video_features parses all six categorical columns into an object array,
    which is ~26M Python strings on 1K. Phase 5 only needs the author, and needs it
    as an int, so read just those two columns.
    """
    _, _, _, vid = resolve_files(bench)
    ids, auth = [], []
    with open(vid, newline='') as fh:
        rdr = csv.reader(fh)
        ix = {n: i for i, n in enumerate(next(rdr))}
        vi, ai = ix['video_id'], ix['author_id']
        for row in rdr:
            ids.append(int(float(row[vi])))
            auth.append(int(float(row[ai])) if row[ai] not in ('', 'NA') else -1)
    return np.asarray(ids, dtype=np.int64), np.asarray(auth, dtype=np.int64)
