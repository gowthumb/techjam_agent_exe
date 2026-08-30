"""Phase 9: gradient-boosted trees under a genuinely PAIRWISE objective.

The Phase 1F GBDT pass used LightGBM `binary` and `lambdarank`. Neither is the
pairwise objective that wins on this metric: `binary` is pointwise, and lambdarank
optimises an nDCG-weighted lambda rather than plain pair order. So the GBDT branch
was never given the loss that made the FM work, and calling it a dead end on that
evidence was premature.

This runs:
  xgboost  rank:pairwise   -- literally -log sigmoid(z_pos - z_neg) over groups
  xgboost  rank:ndcg       -- for contrast within the same library
  catboost YetiRank        -- a different pairwise formulation entirely
  lightgbm lambdarank      -- carried over as the Phase 1F reference point

All grouped by user (the metric's grouping), all on the same feature blocks, all
CPU: xgboost tree_method=hist, catboost task_type=CPU, lightgbm default.

Run:  python experiments/p9_gbdt_pairwise.py --blocks base,aff
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from explib import dataset as D, harness as H
sys.path.insert(0, os.path.dirname(__file__))
from p1_gbdt import build_matrix

HYP = ('The GBDT branch was judged on pointwise binary and lambdarank objectives, never on '
       'the pairwise objective that produced the only confirmed FM gain. A pairwise-ranking '
       'GBDT is the fair test of whether trees are actually the wrong model family here')


def parts(X, y, users, masks, split):
    m = masks[split]
    u = users[m]
    o = np.argsort(u, kind='stable')          # ranking objectives need contiguous groups
    Xs, ys, us = X[m][o], y[m][o], u[o]
    _, cnts = np.unique(us, return_counts=True)
    return Xs, ys, us, cnts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--blocks', default='base,aff')
    ap.add_argument('--models', default='xgb_pairwise,xgb_ndcg,catboost_yetirank')
    ap.add_argument('--rounds', type=int, default=800)
    ap.add_argument('--early', type=int, default=60)
    ap.add_argument('--lr', type=float, default=0.05)
    ap.add_argument('--depth', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    blocks = a.blocks.split(',')
    logs = D.load_logs()
    masks = D.split_slices(logs)
    print(f'building features: blocks={blocks} ...')
    t0 = time.time()
    X, names, cat_idx = build_matrix(logs, masks, blocks)
    y = (logs['long_view'] != 0).astype(np.int32)
    users = logs['user_id']
    print(f'  {X.shape[1]} features in {time.time()-t0:.0f}s')

    Xtr, ytr, utr, gtr = parts(X, y, users, masks, 'train')
    Xva, yva, uva, gva = parts(X, y, users, masks, 'valid')
    Xte, yte, ute, gte = parts(X, y, users, masks, 'test')
    nthread = os.cpu_count()

    for model in a.models.split(','):
        eid = f"9-{model}-{'+'.join(blocks)}"
        cfg = dict(model=model, blocks=blocks, n_features=X.shape[1],
                   lr=a.lr, depth=a.depth, rounds=a.rounds, seed=a.seed, device='cpu')
        with H.Experiment(eid, phase='9', axis='gbdt_pairwise', hypothesis=HYP,
                          config=cfg, tags=['gbdt', model]) as ex:
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
    print(H.summarize([r for r in H.read_log() if r['phase'] in ('9', '1F')]))


if __name__ == '__main__':
    main()
