"""Phase 19: gradient-boosted trees under a pairwise objective, on 1K.

Phase 9 (rank:pairwise / YetiRank) was written to close the ledger's most-flagged
gap -- GBDT was only ever judged on pointwise `binary` and nDCG-weighted
`lambdarank`, never a genuinely pairwise objective -- but was never executed on
EITHER benchmark. This runs it on 1K, where the case for trees is different from
Pure: 1K's regime is item cold-start (85% of test videos unseen in train), and a
tree can split on affinity RATES and evidence COUNTS, which are numeric and
therefore defined even for a cold item's neighbourhood, unlike an FM embedding
row that a cold item simply does not have.

FEATURE SCOPE -- deliberately narrower than Pure's p1_gbdt.build_matrix:
  base  user_id, video_id, author_id, tab, dur_bucket (categorical) +
        log_duration_ms, hour (numeric) -- reuses benchmarks.load_video_authors,
        NOT dataset.load_video_features, because the latter parses all 6 video
        categorical columns (music_id, tag, ...) into a Python string array,
        which is the ~26M-object problem load_video_authors was built to avoid.
        tag/music_id are consequently OUT of scope for 1K's GBDT pass.
  aff   causal affinity rate + evidence count for (user,tab), (user,dur),
        (video), (author), (user,author) -- explib/history.py is benchmark-
        agnostic (operates on whatever logs/cols it is given), so this reuses it
        directly. (user,tag) is dropped along with tag itself.
  user, vstat -- EXCLUDED for 1K. `user` because Pure's own KB already
        establishes pure user-side features contribute exactly 0 under
        within-user ranking (structural, not benchmark-specific, so 1K does not
        get a fresh chance to disprove it here). `vstat` because its 1K source
        (video_features_statistic_1k.csv) is 3.4GB and the block was never a
        winner even on Pure, where it carries a leakage caveat besides.

All CPU: xgboost tree_method=hist, catboost task_type=CPU.

Run:  python experiments/p19_1k_gbdt.py --models xgb_pairwise,xgb_ndcg
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from explib import dataset as D, benchmarks as B, harness as H, history as HI

HYP = ('GBDT was only ever judged on pointwise/lambdarank objectives -- rank:pairwise '
       'is the fair test, and it was never run on 1K at all despite 1K being the regime '
       '(item cold-start) where numeric affinity features have the clearest structural '
       'reason to beat an embedding table that cold items simply do not have a row in')

AFF_KEYS_1K = [('user_tab', ('user_id', 'tab')),
               ('user_dur', ('user_id', 'dur_bucket')),
               ('video', ('video_id',)),
               ('author', ('author_id',)),
               ('user_author', ('user_id', 'author_id'))]


def build_matrix_1k(logs, masks, author):
    feats, names, cat_idx = [], [], []

    def add(col, name, categorical=False):
        if categorical:
            cat_idx.append(len(names))
        feats.append(np.asarray(col, dtype=np.float32))
        names.append(name)

    tr = masks['train']
    edges = np.quantile(logs['duration_ms'][tr], np.linspace(0, 1, 11)[1:-1])
    dur_bucket = np.searchsorted(edges, logs['duration_ms']).astype(np.int64)

    add(logs['user_id'], 'user_id', True)
    add(logs['video_id'], 'video_id', True)
    add(author, 'author_id', True)
    add(logs['tab'], 'tab', True)
    add(dur_bucket, 'dur_bucket', True)
    add(np.log1p(np.maximum(logs['duration_ms'], 0)), 'log_duration_ms')
    add(logs['hourmin'] // 100 if 'hourmin' in logs else np.zeros(len(logs['user_id'])), 'hour')

    cols_by_name = {'user_id': logs['user_id'], 'tab': logs['tab'],
                    'dur_bucket': dur_bucket, 'video_id': logs['video_id'],
                    'author_id': author}
    specs = [(f'aff_{n}', [cols_by_name[c] for c in keys]) for n, keys in AFF_KEYS_1K]
    print(f'  building {len(specs)} affinity fields on {len(logs["user_id"]):,} rows ...')
    _, _, raw = HI.build_affinity_fields(logs, masks, specs, mode='causal')
    for n, _ in AFF_KEYS_1K:
        r, c = raw[f'aff_{n}']
        add(r, f'aff_{n}_rate')
        add(np.log1p(c), f'aff_{n}_logcnt')

    X = np.stack(feats, axis=1)
    return X, names, cat_idx


def parts(X, y, users, masks, split):
    m = masks[split]
    u = users[m]
    o = np.argsort(u, kind='stable')
    Xs, ys, us = X[m][o], y[m][o], u[o]
    _, cnts = np.unique(us, return_counts=True)
    return Xs, ys, us, cnts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='xgb_pairwise,xgb_ndcg,catboost_yetirank')
    ap.add_argument('--rounds', type=int, default=800)
    ap.add_argument('--early', type=int, default=60)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--depth', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    print('loading 1k (minimal columns) ...')
    logs = B.load_logs('1k', minimal=True)
    masks = D.split_slices(logs)
    vids, auths = B.load_video_authors('1k')
    order = np.argsort(vids)
    pos = np.clip(np.searchsorted(vids[order], logs['video_id']), 0, len(vids) - 1)
    hit = vids[order][pos] == logs['video_id']
    author = np.where(hit, auths[order][pos], -1)

    t0 = time.time()
    X, names, cat_idx = build_matrix_1k(logs, masks, author)
    y = (logs[D.LABEL] != 0).astype(np.int32)
    users = logs['user_id']
    print(f'  {X.shape[1]} features ({len(cat_idx)} categorical) in {time.time()-t0:.0f}s')

    Xtr, ytr, utr, gtr = parts(X, y, users, masks, 'train')
    Xva, yva, uva, gva = parts(X, y, users, masks, 'valid')
    Xte, yte, ute, gte = parts(X, y, users, masks, 'test')
    nthread = os.cpu_count()

    for model in a.models.split(','):
        eid = f"19-1k-{model}-base+aff-seed{a.seed}"
        cfg = dict(model=model, benchmark='1k', blocks=['base', 'aff'],
                   n_features=X.shape[1], lr=a.lr, depth=a.depth, rounds=a.rounds,
                   seed=a.seed, device='cpu')
        with H.Experiment(eid, phase='19', axis='onek_gbdt', hypothesis=HYP,
                          config=cfg, tags=['1k', 'gbdt', model],
                          baseline_ref='none') as ex:
            t0 = time.time()
            if model.startswith('xgb'):
                import xgboost as xgb
                obj = 'rank:pairwise' if model.endswith('pairwise') else 'rank:ndcg'
                dtr = xgb.DMatrix(Xtr, label=ytr, group=gtr, nthread=nthread)
                dva = xgb.DMatrix(Xva, label=yva, group=gva, nthread=nthread)
                dte = xgb.DMatrix(Xte, nthread=nthread)
                params = {'objective': obj, 'eta': a.lr, 'max_depth': a.depth,
                          'subsample': .8, 'colsample_bytree': .8, 'lambda': 1.0,
                          'eval_metric': 'ndcg@5', 'tree_method': 'hist',
                          'nthread': nthread, 'seed': a.seed, 'device': 'cpu'}
                bst = xgb.train(params, dtr, num_boost_round=a.rounds,
                                evals=[(dva, 'valid')],
                                early_stopping_rounds=a.early, verbose_eval=100)
                sv = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
                st = bst.predict(dte, iteration_range=(0, bst.best_iteration + 1))
                ex.record_train(best_iteration=int(bst.best_iteration))
            else:
                from catboost import CatBoostRanker, Pool
                ptr = Pool(Xtr, ytr, group_id=utr)
                pva = Pool(Xva, yva, group_id=uva)
                pte = Pool(Xte, group_id=ute)
                cb = CatBoostRanker(loss_function='YetiRank', iterations=a.rounds,
                                    learning_rate=a.lr, depth=min(a.depth, 8),
                                    random_seed=a.seed, task_type='CPU',
                                    thread_count=nthread, verbose=100,
                                    early_stopping_rounds=a.early)
                cb.fit(ptr, eval_set=pva, use_best_model=True)
                sv, st = cb.predict(pva), cb.predict(pte)
                ex.record_train(best_iteration=int(cb.get_best_iteration() or 0))
            ex.record_train(seconds_train=round(time.time() - t0, 1))
            ex.record_metrics('valid', H.score(uva, yva, sv))
            ex.record_metrics('test', H.score(ute, yte, st))

    print()
    print(H.summarize([r for r in H.read_log() if r['phase'] == '19']))


if __name__ == '__main__':
    main()
