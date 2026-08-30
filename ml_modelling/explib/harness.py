"""Experiment logging + evaluation harness.

Every experiment in this workstream appends one JSON object to
`ml_modelling/experiments.jsonl`. The schema is deliberately the same shape the
autonomous agent's run log uses (hypothesis / config / metrics / takeaway), so
offline findings and the agent's own trials are directly comparable and greppable.

Scoring is delegated to the starter kit's `evaluate.py` -- never reimplemented.
"""
import os, sys, json, time, socket, platform, datetime, contextlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
KIT = os.path.join(REPO, 'kuairand-starter-kit')
LOG_PATH = os.path.join(REPO, 'ml_modelling', 'experiments.jsonl')

# The Windows console defaults to cp1252, which raises on the checkmark and the
# Chinese text the starter kit prints. Fail soft on output encoding rather than
# losing a finished run to a print statement.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

if KIT not in sys.path:
    sys.path.insert(0, KIT)
from evaluate import evaluate as _kit_evaluate   # noqa: E402  (the pinned scorer)

# Calibration rungs from baseline_scores.json -- every result is judged against
# these, never against 1.0. oracle is the real denominator for headroom.
CALIBRATION = {
    'random':           {'valid': 0.4834, 'test': 0.4753},
    'item_popularity':  {'valid': 0.5807, 'test': 0.5715},
    'official_baseline': {'valid': 0.6016, 'test': 0.5946},
    'oracle_ceiling':   {'valid': 0.8484, 'test': 0.8645},
}
SEED_STD = 0.0008          # FM std over 5 seeds (official)
NOISE_BAND = 2 * SEED_STD  # 0.0016: below this, a delta is not a result


def score(user_ids, labels, scores):
    """Official metric. user_ids may be ints; evaluate() only groups by identity."""
    return _kit_evaluate(list(user_ids), list(labels), list(scores))


def headroom_pct(primary, split='test'):
    """Fraction of the *reachable* interval (random -> oracle) consumed."""
    lo = CALIBRATION['random'][split]
    hi = CALIBRATION['oracle_ceiling'][split]
    return 100.0 * (primary - lo) / (hi - lo)


def classify(delta_valid_primary):
    """Turn a delta into a verdict using the official noise band, not vibes."""
    if delta_valid_primary is None:
        return 'unknown'
    if delta_valid_primary > NOISE_BAND:
        return 'positive'
    if delta_valid_primary < -NOISE_BAND:
        return 'negative'
    return 'neutral'


class Experiment:
    """Context manager that times a run and appends one record to the log.

        with Experiment('1a-k64', phase='1a', axis='latent_dim',
                        hypothesis='...', config={...}) as ex:
            ...
            ex.record_metrics('valid', m_valid)
            ex.takeaway = '...'
    """

    def __init__(self, exp_id, phase, axis, hypothesis, config, tags=None,
                 baseline_ref=None):
        # baseline_ref: {'valid':x,'test':y} to compare against, or the string
        # 'none' when no comparable baseline exists (e.g. a different benchmark).
        # Defaults to the official KuaiRand-Pure baseline.
        self.baseline_ref = (CALIBRATION['official_baseline'] if baseline_ref is None
                             else baseline_ref)
        self.exp_id = exp_id
        self.phase = phase
        self.axis = axis
        self.hypothesis = hypothesis
        self.config = config
        self.tags = tags or []
        self.metrics = {}
        self.train_info = {}
        self.takeaway = None
        self.error = None
        self._t0 = None

    def record_metrics(self, split, m):
        self.metrics[split] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                               for k, v in m.items()}

    def record_train(self, **kw):
        self.train_info.update(kw)

    def __enter__(self):
        self._t0 = time.time()
        print(f"[{self.exp_id}] {self.hypothesis}")
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            self.error = f'{exc_type.__name__}: {exc}'
        rec = self.to_record(time.time() - self._t0)
        append_record(rec)
        v = self.metrics.get('valid', {})
        if v:
            d = rec['delta_vs_baseline']['valid_primary']
            dtxt = f'{d:+.4f} vs baseline' if d is not None else 'no comparable baseline'
            print(f"[{self.exp_id}] valid primary {v.get('primary', float('nan')):.4f} "
                  f"({dtxt}) -> {rec['outcome']}  [{rec['seconds']:.0f}s]")
        return False   # never swallow exceptions

    def to_record(self, seconds):
        delta = {}
        ref = self.baseline_ref
        for sp in ('valid', 'test'):
            p = self.metrics.get(sp, {}).get('primary')
            delta[f'{sp}_primary'] = (round(p - ref[sp], 5)
                                      if p is not None and isinstance(ref, dict) else None)
        outcome = ('failed' if self.error
                   else classify(delta['valid_primary'])
                   if delta['valid_primary'] is not None else 'unknown')
        return {
            'exp_id': self.exp_id,
            'ts': datetime.datetime.now().isoformat(timespec='seconds'),
            'phase': self.phase,
            'axis': self.axis,
            'hypothesis': self.hypothesis,
            'config': self.config,
            'metrics': self.metrics,
            'train': self.train_info,
            'delta_vs_baseline': delta,
            'headroom_pct_valid': (round(headroom_pct(self.metrics['valid']['primary'], 'valid'), 2)
                                   if 'valid' in self.metrics
                                   and ref is CALIBRATION['official_baseline'] else None),
            'baseline_ref': (ref if isinstance(ref, dict) else 'none'),
            'outcome': outcome,
            'takeaway': self.takeaway,
            'error': self.error,
            'seconds': round(seconds, 1),
            'tags': self.tags,
            'env': {'python': platform.python_version(), 'host': socket.gethostname()},
        }


def append_record(rec, path=LOG_PATH, timeout=30.0):
    """Append one JSON line under an exclusive lock.

    Sweeps run in parallel to fit the compute budget, so several processes append
    to this file at once. A multi-KB line is not atomically appended on Windows,
    so take a lock rather than hope.
    """
    line = json.dumps(rec, ensure_ascii=False, default=_json_default) + '\n'
    lock = path + '.lock'
    deadline = time.time() + timeout
    fd = None
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.time() > deadline:
                # Never lose a result to a stale lock: give up on it and append.
                break
            time.sleep(0.05)
    try:
        with open(path, 'a', encoding='utf-8') as fh:
            fh.write(line)
    finally:
        if fd is not None:
            try:
                os.close(fd)
                os.unlink(lock)
            except OSError:
                pass


def _json_default(o):
    """numpy scalars/arrays are pervasive in these records; make them serializable."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f'Object of type {type(o).__name__} is not JSON serializable')


def read_log():
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, encoding='utf-8') as fh:
        return [json.loads(l) for l in fh if l.strip()]


def summarize(rows=None, sort_by='valid'):
    rows = read_log() if rows is None else rows
    rows = sorted(rows, key=lambda r: -(r['metrics'].get(sort_by, {}).get('primary') or -9))
    w = max([len(r['exp_id']) for r in rows] + [10])
    head = f"{'exp_id':{w}}  {'phase':6} {'valid':>7} {'test':>7} {'d_valid':>8} {'outcome':>8}  s"
    out = [head, '-' * len(head)]
    for r in rows:
        v = r['metrics'].get('valid', {}).get('primary')
        t = r['metrics'].get('test', {}).get('primary')
        d = r['delta_vs_baseline']['valid_primary']
        out.append(f"{r['exp_id']:{w}}  {r['phase']:6} "
                   f"{(f'{v:.4f}' if v is not None else '-'):>7} "
                   f"{(f'{t:.4f}' if t is not None else '-'):>7} "
                   f"{(f'{d:+.4f}' if d is not None else '-'):>8} "
                   f"{r['outcome']:>8}  {r['seconds']:.0f}")
    return '\n'.join(out)


if __name__ == '__main__':
    print(summarize())
