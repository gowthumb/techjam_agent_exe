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
    if a.check:
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


if __name__ == '__main__':
    main()
