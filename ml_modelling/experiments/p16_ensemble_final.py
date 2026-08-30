"""Phase 16 / axis: diverse ensemble + alternative submission.

GROUNDING. `KNOWLEDGE_BASE_PLAN.md` Phase 1d (cost vs gain; when to escalate);
extends the existing Phase 6 ensemble. MLE-bench medal solutions lean heavily on
ensembling for variance reduction -- Phase 6 found seed-averaging one FM family
pays ~0 (members too correlated), so this ensembles across DIFFERENT mechanisms
(pointwise / BPR / SSM / watch-time / ESMM), which is where decorrelation lives.

WHAT IT DOES. Retrains a curated diverse set at seed 0-2, caches predictions on
all four splits, then rank-averages (within-user percentile, scale-free) over
combinations. Selects on `valid`; applies the Phase 11 `rand_valid` veto. Writes
the winner to `submission_alt_{split}.csv` at repo root and validates it with the
official `submit.py` -- it is written ALONGSIDE the BPR submission, not as a
replacement (make_submission.py is untouched).

Run:  python experiments/p16_ensemble_final.py --members bpr,pointwise,ssm,wt,esmm
"""
import os, sys, csv, json, subprocess, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from explib import dataset as D, features as F, fm, wtfm, esmm as E, harness as H, unbiased as U

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
KIT = os.path.join(D.REPO, 'kuairand-starter-kit')
PRED = os.path.join(ROOT, 'cache', 'p16_preds')

HYP = ('ensemble across DIFFERENT training objectives (pointwise/BPR/SSM/watch-time/ESMM), '
       'not seeds of one -- Phase 6 showed same-family seed-averaging pays ~0. Rank-average, '
       'select on valid, veto on rand_valid. KNOWLEDGE_BASE_PLAN.md Phase 1d')


def within_user_pct(scores, users):
    """Within-user percentile rank in [0,1], ties averaged. Copied from
    p6_ensemble.user_percentile so the two ensembles are directly comparable."""
    scores = np.asarray(scores); users = np.asarray(users)
    order = np.lexsort((scores, users))
    su = users[order]
    starts = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
    bounds = np.r_[starts, len(su)]
    out = np.empty(len(scores), dtype=np.float64)
    for a, b in zip(bounds[:-1], bounds[1:]):
        n = b - a
        s = scores[order][a:b]
        r = np.empty(n)
        i = 0
        pos = np.arange(n, dtype=np.float64)
        while i < n:
            j = i
            while j + 1 < n and s[j + 1] == s[i]:
                j += 1
            r[i:j + 1] = pos[i:j + 1].mean()
            i = j + 1
        out[a:b] = r / max(n - 1, 1)
    z = np.empty(len(scores), dtype=np.float64)
    z[order] = out
    return z


def build_members(names, enc, dim, rand_enc, logs, masks):
    """-> {member_name: {split: scores}}  (seed 0, one model each)"""
    os.makedirs(PRED, exist_ok=True)
    tr = masks['train']
    out = {}
    for name in names:
        path = os.path.join(PRED, f'{name}.npz')
        if os.path.exists(path):
            z = np.load(path)
            out[name] = {k: z[k] for k in z.files}
            print(f'  {name}: cached')
            continue
        print(f'  training {name} ...')
        if name == 'pointwise':
            m, _ = fm.train(enc, dim, loss='pointwise', k=16, lr=0.001, epochs=40,
                            patience=4, seed=0, evaluator=H.score, verbose=False)
            pf = m.predict
        elif name == 'bpr':
            m, _ = fm.train(enc, dim, loss='bpr', k=16, lr=0.0002, epochs=60,
                            patience=6, seed=0, evaluator=H.score, verbose=False)
            pf = m.predict
        elif name == 'ssm':
            # p12_final.json is written by hand after the 5-seed replication picks
            # the winning (lr, temp, n, k); falls back to the grid best.
            fp = os.path.join(ROOT, 'cache', 'p12_final.json')
            conf = json.load(open(fp)) if os.path.exists(fp) else \
                json.load(open(os.path.join(ROOT, 'cache', 'p12_best.json')))
            m, _ = fm.train(enc, dim, loss='ssm', k=conf.get('k', 16), lr=conf['lr'],
                            temp=conf['temp'], neg_per_pos=conf.get('neg_per_pos', 8),
                            epochs=45, patience=6, seed=0, evaluator=H.score, verbose=False)
            pf = m.predict
        elif name == 'wt':
            b = json.load(open(os.path.join(ROOT, 'cache', 'p13_best.json')))
            pt = logs['play_time_ms'][tr].astype(np.float64)
            du = np.maximum(logs['duration_ms'][tr].astype(np.float64), 1.0)
            y_wt = np.clip(pt / du, 0, 2).astype(np.float32); cen = (pt >= du)
            y_lv = (logs['long_view'][tr] != 0).astype(np.float32)
            m, _ = wtfm.train(enc['train'][0], y_lv, y_wt, cen, *enc['valid'], dim,
                              w_wt=b['w_wt'], one_sided=True, k=16, lr=0.001, epochs=40,
                              patience=4, seed=0, evaluator=H.score, verbose=False,
                              select_head='lv')
            pf = lambda X: m.predict_both(X)[0]
        elif name == 'esmm':
            b = json.load(open(os.path.join(ROOT, 'cache', 'p15_best.json')))
            y_lv = (logs['long_view'][tr] != 0).astype(np.float32)
            y_cl = (logs['is_click'][tr] != 0).astype(np.float32)
            m, _ = E.train(enc['train'][0], y_lv, y_cl, *enc['valid'], dim,
                           w_click=b['w_click'], mode='esmm', k=16, lr=0.001, epochs=40,
                           patience=4, seed=0, evaluator=H.score, verbose=False)
            pf = lambda X: m.predict(X, mode='esmm')
        else:
            raise ValueError(name)
        sc = {'valid': pf(enc['valid'][0]), 'test': pf(enc['test'][0]),
              'rand_valid': pf(rand_enc['rand_valid'][0]),
              'rand_test': pf(rand_enc['rand_test'][0])}
        np.savez_compressed(path, **sc)
        out[name] = sc
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--members', default='bpr,pointwise,ssm,wt,esmm')
    ap.add_argument('--write-submission', action='store_true')
    a = ap.parse_args()
    names = sorted(x for x in a.members.split(',') if x)   # canonical order -> stable exp_id

    logs = D.load_logs()
    masks = D.split_slices(logs)
    enc, dim, encoder = F.encode_splits(logs, masks, F.BASELINE_FIELDS)
    rand_enc = U.load_random_encoded(encoder)
    U_ = {'valid': enc['valid'][2], 'test': enc['test'][2],
          'rand_valid': rand_enc['rand_valid'][2], 'rand_test': rand_enc['rand_test'][2]}
    Y = {'valid': enc['valid'][1], 'test': enc['test'][1],
         'rand_valid': rand_enc['rand_valid'][1], 'rand_test': rand_enc['rand_test'][1]}

    members = build_members(names, enc, dim, rand_enc, logs, masks)
    pct = {n: {sp: within_user_pct(members[n][sp], U_[sp]) for sp in U_} for n in members}

    def score_combo(combo):
        r = {}
        for sp in U_:
            avg = np.mean([pct[n][sp] for n in combo], axis=0)
            r[sp] = H.score(U_[sp], Y[sp], avg)['primary']
        return r

    combos = [tuple(sorted(members))]
    for n in members:
        combos.append((n,))
    if 'bpr' in members and 'ssm' in members:
        combos.append(tuple(sorted(['bpr', 'ssm'])))
        combos.append(tuple(sorted(x for x in ('bpr', 'ssm', 'pointwise') if x in members)))
    combos = list(dict.fromkeys(combos))

    print(f"\n{'combo':38s} {'valid':>8} {'test':>8} {'r_val':>8} {'r_test':>8}")
    results = {}
    for c in combos:
        r = score_combo(c)
        results['+'.join(c)] = r
        print(f"{'+'.join(c):38s} {r['valid']:8.4f} {r['test']:8.4f} "
              f"{r['rand_valid']:8.4f} {r['rand_test']:8.4f}")

    base_rv = results.get('bpr', {}).get('rand_valid', 0)
    ranked = sorted(results.items(),
                    key=lambda kv: (kv[1]['valid']
                                    - (999 if kv[1]['rand_valid'] < base_rv - 0.003 else 0)),
                    reverse=True)
    best_name, best = ranked[0]
    print(f"\nbest by valid (rand_valid veto applied): {best_name}  "
          f"valid {best['valid']:.4f}  test {best['test']:.4f}")

    eid = f'16-ensemble-{best_name}'
    if eid not in {r['exp_id'] for r in H.read_log() if r['phase'] == '16'}:
        with H.Experiment(eid, phase='16', axis='ensemble', hypothesis=HYP,
                          config=dict(model='ensemble', members=best_name, mode='rank',
                                      device='cpu'), tags=['ensemble']) as ex:
            for sp in U_:
                ex.record_metrics(sp, H.score(U_[sp], Y[sp],
                                  np.mean([pct[n][sp] for n in best_name.split('+')], axis=0)))
            ex.takeaway = f'diverse rank-ensemble of {best_name}'
    else:
        print(f'  {eid} already logged')

    json.dump({k: {sp: float(v) for sp, v in r.items()} for k, r in results.items()},
              open(os.path.join(ROOT, 'phase16_ensemble.json'), 'w'), indent=1)

    if a.write_submission:
        best_combo = best_name.split('+')
        for split in ('valid', 'test'):
            avg = np.mean([pct[n][split] for n in best_combo], axis=0)
            mask = masks[split]
            uid = logs['user_id'][mask]; vid = logs['video_id'][mask]
            out = os.path.join(D.REPO, f'submission_alt_{split}.csv')
            with open(out, 'w', newline='') as fh:
                w = csv.writer(fh)
                w.writerow(['row_id', 'user_id', 'video_id', 'score'])
                for i in range(len(avg)):
                    w.writerow([i, int(uid[i]), int(vid[i]), f'{float(avg[i]):.6g}'])
            print(f'wrote {out}')
            env = dict(os.environ, PYTHONIOENCODING='utf-8')
            for flag in (['--check'], ['--score'] if split == 'valid' else None):
                if flag is None:
                    continue
                r = subprocess.run([sys.executable, 'submit.py', *flag, '--split', split, out],
                                   cwd=KIT, capture_output=True, text=True, encoding='utf-8', env=env)
                print((r.stdout or r.stderr or '').strip())


if __name__ == '__main__':
    main()
