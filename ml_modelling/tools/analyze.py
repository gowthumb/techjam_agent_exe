"""Turn experiments.jsonl into the per-axis view the KB is written from.

Also flags log hygiene problems (duplicate exp_ids, failed runs) so a KB claim is
never derived from a record that quietly overwrote another.

  python tools/analyze.py                # per-axis summary
  python tools/analyze.py --axis loss_function --detail
  python tools/analyze.py --check        # hygiene only
"""
import os, sys, argparse, collections
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from explib import harness as H

BASE = H.CALIBRATION['official_baseline']


def hygiene(rows):
    problems = []
    seen = collections.Counter(r['exp_id'] for r in rows)
    for eid, n in seen.items():
        if n > 1:
            problems.append(f'duplicate exp_id x{n}: {eid}')
    for r in rows:
        if r['outcome'] == 'failed':
            problems.append(f"failed run: {r['exp_id']} -- {r['error']}")
        if not r['metrics'].get('valid'):
            problems.append(f"no valid metrics: {r['exp_id']}")
    return problems


def axis_view(rows, axis):
    sel = [r for r in rows if r['axis'] == axis and r['metrics'].get('valid')]
    sel.sort(key=lambda r: -r['metrics']['valid']['primary'])
    return sel


def fmt(r):
    v = r['metrics']['valid']
    t = r['metrics'].get('test', {})
    dv = r['delta_vs_baseline']['valid_primary']
    dt = r['delta_vs_baseline']['test_primary']
    return (f"  {r['exp_id']:44s} valid {v['primary']:.4f} ({dv:+.4f})  "
            f"test {t.get('primary', float('nan')):.4f} ({dt:+.4f})  "
            f"{r['outcome']:8s} {r['seconds']:5.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--axis', default=None)
    ap.add_argument('--detail', action='store_true')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--ids', action='store_true', help='full id-readability audit')
    a = ap.parse_args()

    rows = H.read_log()
    probs = hygiene(rows)
    print(f'{len(rows)} records, {len(set(r["axis"] for r in rows))} axes')
    if probs:
        print('\nLOG HYGIENE PROBLEMS:')
        for p in probs:
            print('  ! ' + p)
    else:
        print('log hygiene: clean')

    unreadable = audit_ids(rows)
    both = [x for x in unreadable if len(x[3]) == len(x[2])]
    if unreadable:
        print(f'\nID READABILITY: {len(unreadable)} axis/parameter pairs are not readable '
              f'from the exp_id ({len(both)} where NO value is readable).')
        print('  Run with --ids for the full list. These do not collide, but a reader')
        print('  cannot tell from the id which value a run used -- the phase-7 failure mode.')
        for axis, key, vals, missing in both[:6]:
            print(f'   ! {axis:24} {key:14} {", ".join(vals[:4])}')
    if a.check:
        return

    if a.ids:
        print()
        for axis, key, vals, missing in unreadable:
            print(f'  {axis:24} {key:18} values: {", ".join(vals[:5])}')
            print(f'  {"":24} {"":18} unreadable: {", ".join(missing)}')
        return

    axes = [a.axis] if a.axis else sorted({r['axis'] for r in rows})
    for ax in axes:
        sel = axis_view(rows, ax)
        if not sel:
            continue
        best, worst = sel[0], sel[-1]
        pos = sum(1 for r in sel if r['outcome'] == 'positive')
        neu = sum(1 for r in sel if r['outcome'] == 'neutral')
        neg = sum(1 for r in sel if r['outcome'] == 'negative')
        # consistency of the test-side delta, which is the out-of-period check
        tds = [r['delta_vs_baseline']['test_primary'] for r in sel
               if r['delta_vs_baseline']['test_primary'] is not None]
        better_on_test = sum(1 for d in tds if d > 0)
        print(f'\n=== {ax}  ({len(sel)} runs: {pos} positive / {neu} neutral / {neg} negative)')
        print(f'    best  {best["exp_id"]}  valid {best["metrics"]["valid"]["primary"]:.4f} '
              f'({best["delta_vs_baseline"]["valid_primary"]:+.4f})')
        print(f'    worst {worst["exp_id"]} valid {worst["metrics"]["valid"]["primary"]:.4f} '
              f'({worst["delta_vs_baseline"]["valid_primary"]:+.4f})')
        print(f'    beat baseline on test: {better_on_test}/{len(tds)} runs; '
              f'median test delta {sorted(tds)[len(tds)//2]:+.4f}' if tds else '')
        med_s = sorted(r['seconds'] for r in sel)[len(sel) // 2]
        print(f'    median runtime {med_s:.0f}s')
        if a.detail or a.axis:
            for r in sel:
                print(fmt(r))




# ---------------------------------------------------------------- id schema
def audit_ids(rows):
    """Which parameters cannot be read off the exp_id?

    hygiene() already guarantees no two runs share an id, so a true "mis-tuned run
    masquerading as a tuned one" cannot exist *within* the log. The phase-7 failure
    was subtler: the run was unique, but nothing in its id said lr=0.002, so it read
    as comparable to a run at lr=0.0002 when it was not.

    So the well-posed question is readability, not collision: for every parameter
    that varies inside an axis, is each distinct value recoverable from the ids of
    the runs that used it? Numbers are compared with %g so 1.0/1 and 0.001/.001
    match the way they are actually written into ids.
    """
    import collections as _c
    by_axis = _c.defaultdict(list)
    for r in rows:
        by_axis[r['axis']].append(r)

    IGNORE = {'fields', 'aux_positive_rate', 'task_weights', 'aux_tasks', 'blocks',
              'n_features', 'n_categorical', 'n_params', 'members', 'device',
              'effective_rows', 'train_rows', 'train_rows_pct', 'metric',
              'ndcg_eval_at', 'objective', 'verbosity', 'benchmark', 'subsample',
              'sparse_updates', 'mode', 'model', 'bs'}

    def tokens(v):
        """How this value could plausibly be written into an id."""
        out = {str(v)}
        try:
            f = float(v)
            out |= {f'{f:g}', f'{f:g}'.replace('0.', '.'), str(int(f)) if f == int(f) else ''}
        except (TypeError, ValueError):
            pass
        return {t for t in out if t}

    unreadable = []
    for axis, sel in sorted(by_axis.items()):
        keys = set()
        for r in sel:
            keys |= set((r.get('config') or {}).keys())
        for key in sorted(keys - IGNORE):
            vals = _c.defaultdict(list)
            for r in sel:
                v = (r.get('config') or {}).get(key)
                if isinstance(v, (list, dict)) or v is None:
                    continue
                vals[str(v)].append(r)
            if len(vals) < 2:
                continue
            missing = [sv for sv, rs in vals.items()
                       if not any(any(t in r['exp_id'] for t in tokens(sv)) for r in rs)]
            if missing:
                unreadable.append((axis, key, sorted(vals), missing))
    return unreadable


if __name__ == '__main__':
    main()
