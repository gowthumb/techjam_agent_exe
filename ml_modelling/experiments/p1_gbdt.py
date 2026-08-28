"""Phase 1 / axis F: gradient-boosted trees as a parallel branch to the FM ladder.

The organizers measured that FM capacity is not the bottleneck, which makes a
tree model interesting for a different reason than "bigger model": trees consume
numeric features (causal affinity rates, durations, side-table statistics) that
the categorical-only FM cannot represent at all, and LightGBM's `lambdarank`
optimizes a within-group ranking objective directly.

Feature blocks are switchable so the KB can record what each is worth:
  base   - the 5 baseline fields as categoricals + duration/hour numerics
  aff    - causal affinity rates + evidence counts (train-labels-only, see history.py)
  user   - user-side profile columns (useless alone under within-user ranking,
           but a tree can cross them with item-side columns)
  vstat  - video_features_statistic_pure.csv aggregates.
           *** CAVEAT: these aggregates are computed over the whole dataset
           period, so they partially encode test-period outcomes. Reported
           separately and never folded into a headline number. ***

Run:  python experiments/p1_gbdt.py --blocks base,aff --objective lambdarank
"""
import os, sys, argparse, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import lightgbm as lgb
from explib import dataset as D, harness as H, history as HI

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from sweep import build_columns, AFFINITY_SPECS

AFF_KEYS = ['user_tab', 'user_dur', 'user_tag', 'video', 'author', 'user_author']


def build_matrix(logs, masks, blocks):
    feats, names, cat_idx = [], [], []

    def add(col, name, categorical=False):
        if categorical:
            cat_idx.append(len(names))
        feats.append(np.asarray(col, dtype=np.float32))
        names.append(name)

    cols = build_columns(logs, masks, {'author_id', 'music_id', 'tag', 'dur_bucket'})

    if 'base' in blocks:
        add(logs['user_id'], 'user_id', True)
        add(logs['video_id'], 'video_id', True)
        add(cols['author_id'], 'author_id', True)
        add(logs['tab'], 'tab', True)
        add(cols['dur_bucket'], 'dur_bucket', True)
        add(cols['tag'], 'tag', True)
        add(cols['music_id'], 'music_id', True)
        add(np.log1p(np.maximum(logs['duration_ms'], 0)), 'log_duration_ms')
        add(logs['hourmin'] // 100, 'hour')

    if 'aff' in blocks:
        specs = [(f'aff_{n}', [cols[c] if c in cols else logs[c]
                               for c in AFFINITY_SPECS[n]]) for n in AFF_KEYS]
        _, _, raw = HI.build_affinity_fields(logs, masks, specs, mode='causal')
        for n in AFF_KEYS:
            r, c = raw[f'aff_{n}']
            add(r, f'aff_{n}_rate')
            add(np.log1p(c), f'aff_{n}_logcnt')

    if 'user' in blocks:
        uids, ucats, _ = D.load_user_features()
        upos = {int(u): i for i, u in enumerate(uids)}
        uidx = np.array([upos.get(int(u), -1) for u in logs['user_id']])
        for j, name in enumerate(D.USER_CAT_COLS):
            vals = ucats[:, j].astype(str)
            uniq = {v: i for i, v in enumerate(np.unique(vals))}
            codes = np.array([uniq[v] for v in vals], dtype=np.int32)
            add(np.where(uidx >= 0, codes[np.clip(uidx, 0, None)], -1), f'u_{name}', True)

    if 'vstat' in blocks:
        import csv
        stats, ids = [], []
        path = os.path.join(D.DATA_DIR, 'video_features_statistic_pure.csv')
        with open(path, newline='') as fh:
            rdr = csv.DictReader(fh)
            cols_v = [c for c in rdr.fieldnames if c != 'video_id']
            for r in rdr:
                ids.append(int(r['video_id']))
                stats.append([float(r[c]) if r[c] not in ('', 'NA') else np.nan
                              for c in cols_v])
        S = np.asarray(stats, dtype=np.float32)
        vpos = {v: i for i, v in enumerate(ids)}
        vidx = np.array([vpos.get(int(v), -1) for v in logs['video_id']])
        for j, c in enumerate(cols_v):
            col = np.where(vidx >= 0, S[np.clip(vidx, 0, None), j], np.nan)
            add(np.log1p(np.clip(col, 0, None)) if 'cnt' in c or 'num' in c else col,
                f'vs_{c}')

    X = np.stack(feats, axis=1)
    return X, names, cat_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--blocks', default='base,aff')
    ap.add_argument('--objective', default='lambdarank', choices=['lambdarank', 'binary'])
    ap.add_argument('--leaves', type=int, default=63)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--rounds', type=int, default=1500)
    ap.add_argument('--early', type=int, default=60)
    ap.add_argument('--min-data', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--exp-id', default=None)
    a = ap.parse_args()

    blocks = a.blocks.split(',')
    logs = D.load_logs()
    masks = D.split_slices(logs)
    print(f'building features: blocks={blocks} ...')
    t0 = time.time()
    X, names, cat_idx = build_matrix(logs, masks, blocks)
    y = (logs['long_view'] != 0).astype(np.int32)
    print(f'  {X.shape[1]} features ({len(cat_idx)} categorical) in {time.time()-t0:.0f}s')

    def part(split):
        m = masks[split]
        u = logs['user_id'][m]
        o = np.argsort(u, kind='stable')          # lambdarank needs contiguous groups
        Xs, ys, us = X[m][o], y[m][o], u[o]
        _, cnts = np.unique(us, return_counts=True)
        return Xs, ys, us, cnts

    Xtr, ytr, utr, gtr = part('train')
    Xva, yva, uva, gva = part('valid')
    Xte, yte, ute, gte = part('test')

    params = dict(objective=a.objective, metric='ndcg' if a.objective == 'lambdarank' else 'auc',
                  ndcg_eval_at=[5], learning_rate=a.lr, num_leaves=a.leaves,
                  min_data_in_leaf=a.min_data, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  max_cat_threshold=64, cat_smooth=20.0, cat_l2=10.0,
                  num_threads=os.cpu_count(), seed=a.seed, verbosity=-1)

    dtr = lgb.Dataset(Xtr, ytr, feature_name=names, categorical_feature=cat_idx,
                      free_raw_data=False)
    dva = lgb.Dataset(Xva, yva, reference=dtr, feature_name=names,
                      categorical_feature=cat_idx, free_raw_data=False)
    if a.objective == 'lambdarank':
        dtr.set_group(gtr); dva.set_group(gva)

    exp_id = a.exp_id or f"1F-lgbm-{a.objective}-{'+'.join(blocks)}-lv{a.leaves}"
    cfg = dict(model='lightgbm', blocks=blocks, n_features=X.shape[1],
               n_categorical=len(cat_idx), **{k: v for k, v in params.items()
                                              if k not in ('num_threads',)})
    hyp = (f"LightGBM {a.objective} on feature blocks {blocks}: trees can use numeric "
           f"affinity rates and side-table statistics that the categorical-only FM "
           f"cannot represent, so this tests a different axis than FM capacity")
    if 'vstat' in blocks:
        hyp += ' | CAVEAT: vstat aggregates span the test period'

    with H.Experiment(exp_id, phase='1F', axis='gbdt', hypothesis=hyp, config=cfg,
                      tags=['gbdt', 'lightgbm', a.objective]) as ex:
        evals = {}
        booster = lgb.train(params, dtr, num_boost_round=a.rounds,
                            valid_sets=[dva], valid_names=['valid'],
                            callbacks=[lgb.early_stopping(a.early, verbose=False),
                                       lgb.log_evaluation(100),
                                       lgb.record_evaluation(evals)])
        ex.record_train(best_iteration=booster.best_iteration,
                        epochs_run=booster.current_iteration(),
                        best_epoch=booster.best_iteration)
        imp = sorted(zip(names, booster.feature_importance('gain')),
                     key=lambda t: -t[1])[:15]
        ex.record_train(top_features=[(n, round(float(g), 1)) for n, g in imp])
        for sp, (Xs, ys, us) in (('valid', (Xva, yva, uva)), ('test', (Xte, yte, ute))):
            s = booster.predict(Xs, num_iteration=booster.best_iteration)
            ex.record_metrics(sp, H.score(us, ys, s))
        print('\ntop features by gain:')
        for n, g in imp:
            print(f'  {n:28s} {g:12.1f}')

    print()
    print(H.summarize())


if __name__ == '__main__':
    main()
