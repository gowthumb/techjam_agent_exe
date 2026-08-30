"""Phase 12 / axis: sampled-softmax (InfoNCE) loss + negative sampling.

GROUNDING. `KNOWLEDGE_BASE_PLAN.md` Phase 1a (loss / optimizer); the KB's #1
prior ("change the loss, not the capacity" -- the objective is the only axis that
has ever moved this score). Starter kit README unexplored direction #1.

HYPOTHESIS. The KB retired `listwise` as dead, but `listwise` is a softmax over
the whole impression list with a uniform-over-positives target -- a poor fit when
~1/3 of the list is positive. Sampled softmax contrasts ONE positive against a
few sampled negatives (standard implicit-feedback ranking loss; Wu et al., TOIS
2024 report it beats BPR broadly and reduces overconfidence). It optimizes a
smooth surrogate for top-k order, which is what nDCG@5 rewards.

Ranking losses on this data need a SMALLER lr than logloss (KB: BPR peaks at
0.0002). SSM normalises per example, so lr is swept -- comparing SSM at the
pointwise lr would confound "wrong loss" with "wrong step size", the mistake that
nearly killed the BPR finding. The grid runs at k=8 (KB: k is a cost knob, flat
1..16) for speed; the winner is re-confirmed at k=16.

CONTROLS (KB control_rule):
  * ssm_global -- negatives from all rows, not the same user. If it matches
    within-user SSM, the within-user structure of the negatives does no work.
  * bpr pairs_per_pos in {2,4} -- the existing BPR path with more negatives was
    never swept; checks that any SSM edge is the softmax, not just more negatives.

Every run is scored on valid / test / rand_valid / rand_test (Phase 11).

Stages:
  --stage grid       lr x temp at n=4, k=8
  --stage neg        neg_per_pos in {8,16} and lr/2 at the grid best; k=16 check
  --stage control    ssm_global at best; bpr ppp in {2,4}
  --stage replicate  best SSM + BPR reference, 5 seeds each, k=16

Run:  python experiments/p12_ssm_loss.py --stage grid
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from explib import dataset as D, features as F, fm, harness as H, unbiased as U

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
STATE = os.path.join(ROOT, 'cache', 'p12_best.json')

HYP = ('sampled softmax / InfoNCE contrasts one positive against a few sampled negatives -- '
       'a smooth surrogate for top-k order, unlike the retired uniform-target listwise. '
       'Ranking losses need a smaller lr here, so lr is swept before the loss is judged '
       '(KNOWLEDGE_BASE_PLAN.md Phase 1a; KB prior "change the loss, not the capacity")')


def setup():
    logs = D.load_logs()
    masks = D.split_slices(logs)
    enc, dim, encoder = F.encode_splits(logs, masks, F.BASELINE_FIELDS)
    rand_enc = U.load_random_encoded(encoder)
    return enc, dim, rand_enc


def run(eid, cfg_extra, enc, dim, rand_enc, seed=0, **train_kw):
    if eid in {r['exp_id'] for r in H.read_log() if r['phase'] == '12'}:
        print(f'  skip {eid} (already logged)')
        return None
    cfg = dict(model='fm', bs=8192, seed=seed, fields=F.BASELINE_FIELDS, **cfg_extra)
    with H.Experiment(eid, phase='12', axis='loss_function', hypothesis=HYP,
                      config=cfg, tags=['loss', cfg_extra.get('loss', 'ssm')]) as ex:
        m, info = fm.train(enc, dim, evaluator=H.score, verbose=False, seed=seed, **train_kw)
        ex.record_train(**{k: v for k, v in info.items() if k != 'history'})
        sc = U.evaluate_all(m.predict, enc, rand_enc)
        for sp, s in sc.items():
            ex.record_metrics(sp, s)
    row = {sp: round(float(sc[sp]['primary']), 5) for sp in sc}
    print(f"  {eid:36s} v {row['valid']:.4f}  t {row['test']:.4f}  "
          f"rv {row['rand_valid']:.4f}  rt {row['rand_test']:.4f}  "
          f"[ep{info['best_epoch']}/{info['epochs_run']}]")
    return dict(row, best_epoch=info['best_epoch'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True,
                    choices=['grid', 'neg', 'control', 'replicate'])
    a = ap.parse_args()
    enc, dim, rand_enc = setup()

    if a.stage == 'grid':
        best = None
        for lr in (0.0005, 0.001):
            for temp in (0.5, 1.0, 2.0):
                eid = f'12-ssm-lr{lr:g}-t{temp:g}-n4-k8'
                r = run(eid, dict(loss='ssm', k=8, lr=lr, temp=temp, neg_per_pos=4,
                                  epochs=40, patience=5),
                        enc, dim, rand_enc,
                        loss='ssm', k=8, lr=lr, l2=1e-6, temp=temp, neg_per_pos=4,
                        epochs=40, patience=5)
                if r and (best is None or r['valid'] > best['valid']):
                    best = dict(r, lr=lr, temp=temp)
        if best:
            json.dump(best, open(STATE, 'w'))
            print(f"\ngrid best: lr={best['lr']} temp={best['temp']} "
                  f"valid {best['valid']:.4f} (peaked ep {best['best_epoch']})")

    elif a.stage == 'neg':
        # move to k=16 (the validated capacity for the other losses) and pin
        # (lr, temp, n). The grid put the peak at lr=0.0005 with an early best
        # epoch, so lr=0.0003 is also tried; temp 1 vs 2 were within noise on
        # valid but split on test/rand_valid, so both go forward.
        for lr in (0.0003, 0.0005):
            for temp in (1.0, 2.0):
                eid = f'12-ssm-lr{lr:g}-t{temp:g}-n8-k16'
                run(eid, dict(loss='ssm', k=16, lr=lr, temp=temp, neg_per_pos=8,
                              epochs=45, patience=6),
                    enc, dim, rand_enc,
                    loss='ssm', k=16, lr=lr, l2=1e-6, temp=temp, neg_per_pos=8,
                    epochs=45, patience=6)
        run('12-ssm-lr0.0005-t1-n16-k16',
            dict(loss='ssm', k=16, lr=0.0005, temp=1.0, neg_per_pos=16,
                 epochs=45, patience=6),
            enc, dim, rand_enc,
            loss='ssm', k=16, lr=0.0005, l2=1e-6, temp=1.0, neg_per_pos=16,
            epochs=45, patience=6)

    elif a.stage == 'control':
        b = json.load(open(STATE))
        run(f"12-ssm-global-lr{b['lr']:g}-t{b['temp']:g}-n4-k8",
            dict(loss='ssm', k=8, lr=b['lr'], temp=b['temp'], neg_per_pos=4,
                 ssm_global=True, epochs=40, patience=5),
            enc, dim, rand_enc,
            loss='ssm', k=8, lr=b['lr'], l2=1e-6, temp=b['temp'], neg_per_pos=4,
            ssm_global=True, epochs=40, patience=5)
        for ppp in (2, 4):
            run(f'12-bpr-ppp{ppp}-lr0.0002-k16',
                dict(loss='bpr', k=16, lr=0.0002, pairs_per_pos=ppp, epochs=60, patience=6),
                enc, dim, rand_enc,
                loss='bpr', k=16, lr=0.0002, l2=1e-6, pairs_per_pos=ppp,
                epochs=60, patience=6)

    elif a.stage == 'replicate':
        # cache/p12_finalists.json: [{lr,temp,neg_per_pos,k}, ...] -- the configs
        # within noise of the neg-stage best, all replicated on 5 seeds, pick by
        # valid mean (KB replication_rule). BPR is replicated alongside as the
        # standing reference.
        fin = json.load(open(os.path.join(ROOT, 'cache', 'p12_finalists.json')))
        for seed in range(5):
            sfx = f'-seed{seed}' if seed else '-rep-seed0'
            for c in fin:
                tag = f"lr{c['lr']:g}-t{c['temp']:g}-n{c['neg_per_pos']}-k{c['k']}"
                run(f'12-ssm-{tag}{sfx}',
                    dict(loss='ssm', epochs=45, patience=6, **c),
                    enc, dim, rand_enc, seed=seed,
                    loss='ssm', l2=1e-6, epochs=45, patience=6, **c)
            run(f'12-bpr-ref-k16{sfx}',
                dict(loss='bpr', k=16, lr=0.0002, epochs=60, patience=6),
                enc, dim, rand_enc, seed=seed,
                loss='bpr', k=16, lr=0.0002, l2=1e-6, epochs=60, patience=6)

    print('\n' + H.summarize([r for r in H.read_log() if r['phase'] == '12']))


if __name__ == '__main__':
    main()
