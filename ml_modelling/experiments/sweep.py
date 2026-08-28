"""Generic FM sweep runner.

One entry point for every FM-family experiment so the config, not the code, is
what changes between runs -- which is also what makes the log comparable.

  python experiments/sweep.py --phase 1A --axis loss_function \
      --loss bpr --lr 0.0001,0.0002,0.0005 --hypothesis "..."

  python experiments/sweep.py --phase 1D --axis capacity \
      --loss bpr --k 8,16,32,64,128 --lr 0.0005

Affinity (causal history) fields are opt-in via --affinity:
  --affinity user_author,user_tab,user_dur,video   --affinity-mode causal|loo
"""
import os, sys, argparse, itertools
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from explib import dataset as D, features as F, fm, harness as H, history as HI

AFFINITY_SPECS = {
    # name          -> key columns, resolved lazily against the loaded logs
    'user_author': ('user_id', 'author_id'),
    'user_tab':    ('user_id', 'tab'),
    'user_dur':    ('user_id', 'dur_bucket'),
    'user_music':  ('user_id', 'music_id'),
    'user_tag':    ('user_id', 'tag'),
    'video':       ('video_id',),
    'author':      ('author_id',),
}


def _floats(s):
    return [float(x) for x in s.split(',')] if s else []


def _ints(s):
    return [int(x) for x in s.split(',')] if s else []


def build_columns(logs, masks, needed):
    """Materialize the key columns the requested affinity features need."""
    cols = {'user_id': logs['user_id'], 'video_id': logs['video_id'],
            'tab': logs['tab'].astype(np.int64)}
    if any(c in needed for c in ('author_id', 'music_id', 'tag')):
        ids, cats, _ = D.load_video_features()
        pos = {int(v): i for i, v in enumerate(ids)}
        idx = np.array([pos.get(int(v), -1) for v in logs['video_id']])
        for name in ('author_id', 'music_id', 'tag'):
            if name in needed:
                j = D.VIDEO_CAT_COLS.index(name)
                raw = cats[:, j]
                # tag can be a comma-joined list; use the primary (first) tag
                vals = np.array([int(str(x).split(',')[0]) if str(x) not in ('', 'UNK', 'nan')
                                 else -1 for x in raw], dtype=np.int64)
                cols[name] = np.where(idx >= 0, vals[np.clip(idx, 0, None)], -1)
    if 'dur_bucket' in needed:
        edges = np.quantile(logs['duration_ms'][masks['train']], np.linspace(0, 1, 11)[1:-1])
        cols['dur_bucket'] = np.searchsorted(edges, logs['duration_ms']).astype(np.int64)
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True)
    ap.add_argument('--axis', required=True)
    ap.add_argument('--hypothesis', required=True)
    ap.add_argument('--loss', default='pointwise')
    ap.add_argument('--lr', default='0.001')
    ap.add_argument('--k', default='16')
    ap.add_argument('--l2', default='1e-6')
    ap.add_argument('--lam', default='1.0')
    ap.add_argument('--pairs-per-pos', default='1')
    ap.add_argument('--fields', default=None, help='comma list; default = baseline 5')
    ap.add_argument('--affinity', default=None, help=f'comma list from {list(AFFINITY_SPECS)}')
    ap.add_argument('--affinity-mode', default='causal', choices=['causal', 'loo'])
    ap.add_argument('--affinity-prior', type=float, default=20.0)
    ap.add_argument('--affinity-buckets', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--bs', type=int, default=8192)
    ap.add_argument('--seed', default='0')
    ap.add_argument('--tag-prefix', default='')
    ap.add_argument('--quiet', action='store_true')
    a = ap.parse_args()

    logs = D.load_logs()
    masks = D.split_slices(logs)

    fields = a.fields.split(',') if a.fields else list(F.BASELINE_FIELDS)
    extra = {}
    if a.affinity:
        names = a.affinity.split(',')
        needed = {c for n in names for c in AFFINITY_SPECS[n]}
        cols = build_columns(logs, masks, needed)
        specs = [(f'aff_{n}', [cols[c] for c in AFFINITY_SPECS[n]]) for n in names]
        print(f"building affinity features {names} (mode={a.affinity_mode}) ...")
        extra, _, _ = HI.build_affinity_fields(logs, masks, specs, mode=a.affinity_mode,
                                            prior=a.affinity_prior,
                                            n_buckets=a.affinity_buckets)
        fields += list(extra.keys())

    enc, dim, _ = F.encode_splits(logs, masks, fields, extra_cols=extra)
    print(f"fields={fields} dim={dim}\n")

    grid = list(itertools.product(_floats(a.lr), _ints(a.k), _floats(a.l2),
                                  _floats(a.lam), _ints(a.pairs_per_pos),
                                  _ints(a.seed)))
    for lr, k, l2, lam, ppp, seed in grid:
        # exp_id must be unique across the whole log, so any knob that is not at
        # its default goes into the name even when this sweep holds it fixed.
        parts = [a.phase, a.loss]
        if len(_floats(a.lr)) > 1 or lr != 0.001:
            parts.append(f'lr{lr:g}')
        if len(_ints(a.k)) > 1 or k != 16:
            parts.append(f'k{k}')
        if len(_floats(a.l2)) > 1 or l2 != 1e-6:
            parts.append(f'l2{l2:g}')
        if a.epochs != 40:
            parts.append(f'ep{a.epochs}')
        if len(_floats(a.lam)) > 1:
            parts.append(f'lam{lam:g}')
        if len(_ints(a.pairs_per_pos)) > 1:
            parts.append(f'ppp{ppp}')
        if a.affinity:
            parts.append(f'aff-{a.affinity.replace(",", "+")}-{a.affinity_mode}')
        if len(_ints(a.seed)) > 1 or int(seed) != 0:
            parts.append(f'seed{seed}')
        exp_id = (a.tag_prefix + '-' if a.tag_prefix else '') + '-'.join(parts)

        cfg = dict(model='fm', loss=a.loss, k=k, lr=lr, l2=l2, lam=lam,
                   pairs_per_pos=ppp, bs=a.bs, epochs=a.epochs,
                   patience=a.patience, seed=seed, fields=fields,
                   affinity=a.affinity, affinity_mode=a.affinity_mode,
                   affinity_prior=a.affinity_prior,
                   affinity_buckets=a.affinity_buckets)
        kw = {}
        if a.loss in ('bpr',):
            kw['pairs_per_pos'] = ppp
        if a.loss in ('hybrid',):
            kw['lam'] = lam
        with H.Experiment(exp_id, phase=a.phase, axis=a.axis,
                          hypothesis=a.hypothesis, config=cfg,
                          tags=[a.loss] + ([a.affinity] if a.affinity else [])) as ex:
            m, info = fm.train(enc, dim, loss=a.loss, k=k, lr=lr, l2=l2,
                               epochs=a.epochs, bs=a.bs, patience=a.patience,
                               seed=seed, evaluator=H.score,
                               verbose=not a.quiet, **kw)
            ex.record_train(**{key: v for key, v in info.items() if key != 'history'})
            ex.record_train(history=info['history'])
            for sp in ('valid', 'test'):
                X, y, u = enc[sp]
                ex.record_metrics(sp, H.score(u, y, m.predict(X)))
        print()

    print(H.summarize())


if __name__ == '__main__':
    main()
