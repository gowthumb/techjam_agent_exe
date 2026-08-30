"""Phase 20: the single 27K attempt -- KuaiRand-27K, pointwise FM only.

Phase 5's facts pass (--bench 27k --stage facts, phase5_facts_27k.json) confirmed
27K is the 1K regime, not Pure's: 322,278,385 rows, 27,285 users (1.0x Pure's user
count -- same population, 224x more logged interactions each, i.e. full histories,
not a fresh sample), 32M-video catalog, only 17.3% of test videos seen in train
(vs 1K's 15.1%), label rate flat across splits (0.263 -> 0.257, no drift). This is
an even more extreme cold-start regime than 1K, where BPR/SSM/GBDT all lost to
plain pointwise FM.

Per scale_transfer.kuairand_27k.if_attempted and the plan's own Phase 5 directive
("skip the architecture ladder... a well-tuned plain FM that reliably finishes
inside the 6h ceiling"), this runs exactly ONE config: 1K's own confirmed winner,
transferred as-is, zero exploration, zero grid search. Given the 6h-per-benchmark
budget is dominated by the ~2.1h download + ~27min load + ~21min facts pass
already spent, there is no budget left for a BPR/SSM comparison that every prior
benchmark (Pure, 1K) and the KB's own directive already predict will lose.

CPU only, sparse=True (dim is far beyond dense-Adam feasibility at this vocab).

Run:  python experiments/p20_27k_run.py
"""
import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from explib import harness as H, fm
sys.path.insert(0, os.path.dirname(__file__))
from p5_scale_transfer import prepare, BASELINE_CFG

HYP = ("27K's facts pass confirms an even more extreme item-cold-start regime than "
       "1K (17.3% vs 15.1% test videos seen in train), where every within-user "
       "ranking loss (BPR, SSM) and every GBDT variant already lost to plain "
       "pointwise FM. This transfers 1K's confirmed winning config as-is -- zero "
       "exploration, per the plan's own directive to skip the architecture ladder "
       "at this scale and the KB's kuairand_27k.if_attempted note.")


def main():
    print('loading 27k (this is the ~27min step, per the facts-pass-scaled estimate) ...')
    t0 = time.time()
    logs, masks, enc, dim = prepare('27k')
    print(f"  {len(logs['user_id']):,} rows, dim={dim}, loaded in {time.time()-t0:.0f}s")

    eid = '20-27k-pointwise-k16-lr0.001-seed0'
    full = dict(model='fm', benchmark='27k', bs=8192, seed=0, device='cpu',
                sparse_updates=True, fields=['user_id', 'video_id', 'author_id',
                                             'tab', 'dur_bucket'], **BASELINE_CFG)
    with H.Experiment(eid, phase='20', axis='kuairand_27k', hypothesis=HYP,
                      config=full, tags=['27k', 'phase20'], baseline_ref='none') as ex:
        t0 = time.time()
        m, info = fm.train(enc, dim, loss=BASELINE_CFG['loss'], k=BASELINE_CFG['k'],
                           lr=BASELINE_CFG['lr'], l2=BASELINE_CFG['l2'],
                           epochs=BASELINE_CFG['epochs'], patience=BASELINE_CFG['patience'],
                           seed=0, evaluator=H.score, verbose=True, sparse=True)
        ex.record_train(**{k: v for k, v in info.items() if k != 'history'})
        ex.record_train(history=info['history'])
        ex.record_train(seconds_train=round(time.time() - t0, 1))
        for sp in ('valid', 'test'):
            X, y, u = enc[sp]
            ex.record_metrics(sp, H.score(u, y, m.predict(X)))
        print(f"  train wall-clock: {time.time()-t0:.0f}s")

    rows = [r for r in H.read_log() if r['phase'] == '20']
    print()
    print(H.summarize(rows))


if __name__ == '__main__':
    main()
