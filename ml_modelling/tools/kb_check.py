"""Validate knowledge_base.yaml against experiments.jsonl.

The KB is a directive the agent acts on, so a stale or invented number in it is
worse than no number. Every entry that makes an empirical claim carries an
`evidence` block naming the exp_ids it came from; this tool checks that:

  * the KB parses and has every required top-level section;
  * every exp_id cited under `evidence` exists in the log, exactly once;
  * `calibration` matches the organizers' baseline_scores.json;
  * every cited run's actual valid/test delta is printed next to the claim, so a
    reviewer can see whether the prose still matches the data.

  python tools/kb_check.py            # validate + print evidence table
  python tools/kb_check.py --quiet    # exit code only (0 = clean)
"""
import os, sys, json, argparse, collections
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml
from explib import harness as H

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# The KB was packaged into a top-level deliverable folder; this lab still owns the
# experiment log it is validated against.
KB_PATH = os.path.join(ROOT, '..', 'knowledge_base', 'knowledge_base.yaml')
SCORES = os.path.join(ROOT, '..', 'kuairand-starter-kit', 'baseline_scores.json')

REQUIRED = ['meta', 'calibration', 'decision_protocol', 'validated_search_space',
            'priors', 'feature_engineering_menu', 'multi_task_signals',
            'candidate_models', 'architecture_ladder', 'diagnostics',
            'dead_ends', 'scale_transfer']


def walk_evidence(node, path=''):
    """Yield (path, evidence dict) for every `evidence` block in the tree."""
    if isinstance(node, dict):
        if 'evidence' in node and isinstance(node['evidence'], dict):
            yield path, node['evidence']
        for k, v in node.items():
            yield from walk_evidence(v, f'{path}.{k}' if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk_evidence(v, f'{path}[{i}]')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()
    problems = []

    with open(KB_PATH, encoding='utf-8') as fh:
        kb = yaml.safe_load(fh)

    for sec in REQUIRED:
        if sec not in kb:
            problems.append(f'missing required section: {sec}')

    rows = H.read_log()
    by_id = collections.defaultdict(list)
    for r in rows:
        by_id[r['exp_id']].append(r)

    # calibration must match the organizers' published numbers
    with open(SCORES, encoding='utf-8') as fh:
        official = json.load(fh)['scores']
    want = {'random': 'random', 'popularity': 'item_popularity',
            'official_baseline': 'fm_official', 'oracle_ceiling': 'oracle_ceiling'}
    cal = kb.get('calibration', {})
    for kb_key, off_key in want.items():
        for split in ('valid', 'test'):
            got = (cal.get(kb_key) or {}).get(split)
            exp = official[off_key][split]['primary']
            if got is None:
                problems.append(f'calibration.{kb_key}.{split} missing')
            elif abs(got - exp) > 1e-9:
                problems.append(f'calibration.{kb_key}.{split} = {got}, '
                                f'baseline_scores.json says {exp}')

    table = []
    for path, ev in walk_evidence(kb):
        for eid in ev.get('exp_ids', []):
            hits = by_id.get(eid, [])
            if not hits:
                problems.append(f'{path}: cites unknown exp_id "{eid}"')
                continue
            if len(hits) > 1:
                problems.append(f'{path}: exp_id "{eid}" appears {len(hits)}x in the log')
            r = hits[0]
            table.append((path, eid,
                          r['metrics'].get('valid', {}).get('primary'),
                          r['delta_vs_baseline']['valid_primary'],
                          r['delta_vs_baseline']['test_primary'],
                          r['outcome']))

    if not a.quiet:
        print(f'KB sections: {len(kb)}  |  log records: {len(rows)}  |  '
              f'evidence citations: {len(table)}')
        if table:
            print(f"\n{'kb path':46s} {'exp_id':42s} {'valid':>7} {'dval':>8} {'dtest':>8}  outcome")
            print('-' * 122)
            for path, eid, v, dv, dt, out in table:
                print(f'{path:46s} {eid:42s} '
                      f'{(f"{v:.4f}" if v is not None else "-"):>7} '
                      f'{(f"{dv:+.4f}" if dv is not None else "-"):>8} '
                      f'{(f"{dt:+.4f}" if dt is not None else "-"):>8}  {out}')

    if problems:
        print('\nPROBLEMS:')
        for p in problems:
            print('  ! ' + p)
        return 1
    print('\nkb_check: OK -- every cited run exists and calibration matches the official file')
    return 0


if __name__ == '__main__':
    sys.exit(main())
