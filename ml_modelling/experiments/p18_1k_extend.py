"""Phase 18 (1K, stage A): the same axes as Pure, run on 1K's own baseline.

Phase 5 established Pure's *recipe* fails on 1K (BPR, -0.0152). Phase 10 found
that *tuning the baseline* fails too (noise, lr, k=32, (user,author) affinity --
4/4 negative on 3-seed replication). Neither of those tested whether Pure's
OTHER confirmed wins hold up on 1K's own baseline. This phase does that, for the
two candidates that need zero new infrastructure:

  ssm   Pure's confirmed BPR-peer (candidate_models.fm_ssm: lr=0.0003, temp=1.0,
        neg_per_pos=8, k=16, epochs=45, patience=6). It normalises per example
        exactly like BPR, so the KB's own tier12_note predicts it inverts on 1K
        for the same item-cold-start reason BPR did -- this is the test of that
        prediction, not an assumption of it. fm.py's _step_ssm already routes
        through apply_grad, which dispatches to the sparse path when sparse=True
        (confirmed by reading fm.py), so this needs no new code, exactly like
        Phase 5/10's use of pointwise/bpr on 1K.

  affinity  (user,tab) and (user,dur_bucket) causal affinity as a 6th field, same
        pattern as Phase 10's (user,author) test (features.encode_int_fields).
        (user,author) failed at 26.5% test coverage; tab and dur_bucket are much
        smaller-cardinality fields so their per-(user,X) coverage should be
        higher -- worth checking rather than assuming.

DISCIPLINE (per the plan): every candidate is 1 seed first. Only a candidate that
beats the 3-seed baseline mean (valid 0.6439, sd 0.0022) by more than one
baseline-seed SE (~0.0013) gets 3-seed replication. Nothing here is written to
the KB from a single run.

  unbiased  scores a named 1K config on 1K's OWN random-exposure log
        (log_random_4_22_to_5_08_1k.csv, confirmed present) via the new
        benchmarks.load_random_logs / features.apply_int_fields /
        unbiased.load_random_encoded(bench='1k') path. This is the check the
        Phase-10 entry in the KB explicitly asked for before trusting any
        valid-flat/test-up 1K result -- the same shape that turned out to be a
        bias artifact for embedding_noise on Pure.

CPU only, sparse=True throughout (1K's dim is ~2.9M).

Run:  python experiments/p18_1k_extend.py --stage ssm
      python experiments/p18_1k_extend.py --stage affinity
      python experiments/p18_1k_extend.py --stage unbiased --config ssm
"""
import os, sys, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from explib import (dataset as D, benchmarks as B, features as F, fm, harness as H,
                    history as HI, unbiased as U)
sys.path.insert(0, os.path.dirname(__file__))
from p5_scale_transfer import prepare

BASE = dict(loss='pointwise', k=16, lr=0.001, l2=1e-6, epochs=40, patience=4)
BASELINE_MEAN, BASELINE_SD, BASELINE_N = 0.6439, 0.0022, 3
GATE = BASELINE_SD / np.sqrt(BASELINE_N)   # ~0.0013: 1 baseline-seed SE


def run(exp_id, cfg, enc, dim, seed, hyp, axis, return_model=False,
        rand_enc=None, **train_kw):
    full = dict(model='fm', benchmark='1k', bs=8192, seed=seed, device='cpu',
                sparse_updates=True, fields=list(F.BASELINE_FIELDS), **cfg, **train_kw)
    with H.Experiment(exp_id, phase='18', axis=axis, hypothesis=hyp, config=full,
                      tags=['1k', 'phase18'], baseline_ref='none') as ex:
        m, info = fm.train(enc, dim, loss=cfg['loss'], k=cfg['k'], lr=cfg['lr'],
                           l2=cfg['l2'], epochs=cfg['epochs'], patience=cfg['patience'],
                           seed=seed, evaluator=H.score, verbose=False,
                           sparse=True, **train_kw)
        ex.record_train(**{k: v for k, v in info.items() if k != 'history'})
        ex.record_train(history=info['history'])
        for sp in ('valid', 'test'):
            X, y, u = enc[sp]
            ex.record_metrics(sp, H.score(u, y, m.predict(X)))
        if rand_enc is not None:
            for sp in ('rand_valid', 'rand_test'):
                X, y, u = rand_enc[sp]
                ex.record_metrics(sp, H.score(u, y, m.predict(X)))
        v = ex.metrics['valid']['primary']
    return (v, m) if return_model else v


def build_random_enc_1k(mapping, dur_edges):
    """Encode 1K's own random-exposure log with the STANDARD-log-fitted mapping.

    Mirrors unbiased.load_random_encoded, but for the int-fast-path encoder
    Phase 5/10/18 use on 1K (features.apply_int_fields) rather than the
    string-based Encoder class that function expects.
    """
    logs = B.load_random_logs('1k')
    vids, auths = B.load_video_authors('1k')
    order = np.argsort(vids)
    pos = np.clip(np.searchsorted(vids[order], logs['video_id']), 0, len(vids) - 1)
    hit = vids[order][pos] == logs['video_id']
    author = np.where(hit, auths[order][pos], -1)
    dur_bucket = np.searchsorted(dur_edges, logs['duration_ms']).astype(np.int64)
    cols = {'user_id': logs['user_id'], 'video_id': logs['video_id'],
            'author_id': author, 'tab': logs['tab'].astype(np.int64),
            'dur_bucket': dur_bucket}
    X = F.apply_int_fields(cols, mapping)
    y = (logs[D.LABEL] != 0).astype(np.float32)
    d = logs['date']
    out = {}
    for name, (lo, hi) in U.RAND_SPLITS.items():
        idx = (d >= lo) & (d <= hi)
        out[name] = (X[idx], y[idx], logs['user_id'][idx])
    return out


def gate_check(label, valid_score):
    d = valid_score - BASELINE_MEAN
    if d > GATE:
        print(f"  [{label}] {valid_score:.4f} ({d:+.4f}) CLEARS the replication gate "
              f"(>{GATE:.4f}) -- queue 3-seed replication")
        return True
    print(f"  [{label}] {valid_score:.4f} ({d:+.4f}) does not clear the gate "
          f"(need >{GATE:.4f}) -- treat as noise, do not replicate")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True,
                    choices=['ssm', 'affinity', 'replicate', 'unbiased'])
    ap.add_argument('--replicate-key', default=None,
                    help='for --stage replicate: ssm | user_tab | user_dur')
    ap.add_argument('--config', default='ssm', choices=['baseline', 'ssm'],
                    help='for --stage unbiased: which config to score on rand_valid/rand_test')
    ap.add_argument('--seeds', default='0')
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]

    if a.stage == 'unbiased':
        print('loading 1k with frozen-mapping encoder ...')
        logs, masks, enc, dim, cols, mapping, dur_edges = prepare('1k', return_mapping=True)
        print(f'  dim={dim}')
        print('encoding 1K random-exposure log with the SAME train-fitted mapping ...')
        rand_enc = build_random_enc_1k(mapping, dur_edges)
        for sp in ('rand_valid', 'rand_test'):
            X, y, u = rand_enc[sp]
            print(f'  {sp}: {len(y):,} rows, label rate {float(y.mean()):.4f}, '
                  f'{len(np.unique(u)):,} users')
        cfg = (dict(BASE, loss='ssm', lr=0.0003, k=16, epochs=45, patience=6,
                    neg_per_pos=8, temp=1.0) if a.config == 'ssm' else dict(BASE))
        train_kw = {k: cfg.pop(k) for k in ('neg_per_pos', 'temp') if k in cfg}
        for seed in seeds:
            eid = f"18-1k-unbiased-{a.config}-seed{seed}"
            hyp = (f"Score {a.config} on 1K's own random-exposure log -- the bias veto that "
                   f"retired embedding_noise on Pure, applied here before this config's "
                   f"valid/test numbers are trusted")
            run(eid, cfg, enc, dim, seed, hyp, 'onek_unbiased', rand_enc=rand_enc, **train_kw)
        print()
        print(H.summarize([r for r in H.read_log()
                           if r['phase'] == '18' and r['axis'] == 'onek_unbiased']))
        return

    print('loading 1k ...')
    logs, masks, enc, dim = prepare('1k')
    print(f'  dim={dim}')

    if a.stage == 'ssm':
        hyp = ("Pure's confirmed BPR-peer loss, at its confirmed Pure config; a within-user "
               "normalising loss should invert on 1K the same way BPR did (KB tier12_note), "
               "which this tests rather than assumes")
        cfg = dict(BASE, loss='ssm', lr=0.0003, k=16, epochs=45, patience=6)
        for seed in seeds:
            v = run(f'18-1k-ssm-lr0.0003-t1-n8-k16-seed{seed}', cfg, enc, dim, seed,
                    hyp, 'onek_loss', neg_per_pos=8, temp=1.0)
            if len(seeds) == 1:
                gate_check('ssm', v)

    elif a.stage == 'affinity':
        hyp = ("(user,author) affinity failed on 1K despite 10x Pure's coverage. tab and "
               "dur_bucket have far smaller cardinality than author, so per-(user,X) "
               "coverage should be much higher -- this checks that rather than assuming "
               "the (user,author) result generalises to every affinity field")
        for field, key_col in (('user_tab', 'tab'), ('user_dur', 'dur_bucket')):
            cols = {}
            if key_col == 'tab':
                key_arr = logs['tab'].astype(np.int64)
            else:
                edges = np.quantile(logs['duration_ms'][masks['train']],
                                    np.linspace(0, 1, 11)[1:-1])
                key_arr = np.searchsorted(edges, logs['duration_ms']).astype(np.int64)
            specs = [(f'aff_{field}', [logs['user_id'], key_arr])]
            extra, fitted, _ = HI.build_affinity_fields(logs, masks, specs, mode='causal')
            col = extra[f'aff_{field}']
            for sp, m in masks.items():
                print(f'  {field} {sp}: warm rows {float((col[m] != 0).mean()):.1%}')
            cols6 = {f'aff_{field}': col.astype(np.int64)}
            Xa, dima, _ = F.encode_int_fields(cols6, masks['train'])
            enc6 = {sp: (np.concatenate([enc[sp][0], Xa[masks[sp]] + dim], axis=1),
                        enc[sp][1], enc[sp][2]) for sp in enc}
            for seed in seeds:
                v = run(f'18-1k-pointwise-aff-{field}-seed{seed}', BASE, enc6,
                        dim + dima, seed, hyp, 'onek_affinity')
                if len(seeds) == 1:
                    gate_check(field, v)

    elif a.stage == 'replicate':
        if a.replicate_key == 'ssm':
            cfg = dict(BASE, loss='ssm', lr=0.0003, k=16, epochs=45, patience=6)
            for seed in seeds:
                run(f'18-1k-ssm-lr0.0003-t1-n8-k16-seed{seed}', cfg, enc, dim, seed,
                    'replication of the ssm gate-clearing candidate', 'onek_loss',
                    neg_per_pos=8, temp=1.0)
        else:
            print('affinity replication: re-run --stage affinity with --seeds; '
                  'the encoder rebuild is field-specific so it is not split out here')

    rows = [r for r in H.read_log() if r['phase'] == '18']
    print()
    print(H.summarize(rows))


if __name__ == '__main__':
    main()
