"""Phase 1 / axis I: user behaviour sequences (DIN-style attention).

The organizers call this the single largest untouched direction: the kit's five
features are entirely flat, nothing encodes what the user did before the
impression. Measured coverage says the signal is there to use -- 99.7% of eval
rows carry a history, mean length 18.8 of a possible 20.

Controls, so the sequence's contribution is isolated rather than confounded with
"a neural net instead of an FM":
  --no-seq   same network, attention branch removed. Any gap between it and the
             full model is the sequence, not the architecture.
  the numpy FM control (1A-control-pointwise, valid 0.6022) remains the reference
  for whether a neural model is worth its cost at all.

Sequence kinds (see explib/sequence.py for the leakage contract):
  exposure -- last 20 videos shown to the user, label-free, safe everywhere
  positive -- last 20 videos the user long-viewed, built from TRAIN labels only

Run:  python experiments/p1_sequence.py --seq exposure
      python experiments/p1_sequence.py --no-seq        # architecture control
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
from explib import dataset as D, sequence as SQ, harness as H

PAD = 0


def dense(codes, pad_shift=1):
    """Map arbitrary ints to contiguous ids starting at pad_shift (0 = PAD)."""
    uniq, inv = np.unique(codes, return_inverse=True)
    return inv.astype(np.int64) + pad_shift, len(uniq) + pad_shift


class DIN(nn.Module):
    def __init__(self, n_user, n_video, n_author, n_tab, n_dur, d=16,
                 use_seq=True, att_hidden=48, mlp=(96, 48)):
        super().__init__()
        self.use_seq = use_seq
        self.e_user = nn.Embedding(n_user, d, padding_idx=PAD)
        self.e_video = nn.Embedding(n_video, d, padding_idx=PAD)
        self.e_author = nn.Embedding(n_author, d, padding_idx=PAD)
        self.e_tab = nn.Embedding(n_tab, d, padding_idx=PAD)
        self.e_dur = nn.Embedding(n_dur, d, padding_idx=PAD)
        for e in (self.e_user, self.e_video, self.e_author, self.e_tab, self.e_dur):
            nn.init.normal_(e.weight, 0, 0.01)
            with torch.no_grad():
                e.weight[PAD].zero_()
        item_d = 2 * d                                   # video + author
        # DIN local activation unit: [h, c, h*c, h-c] -> scalar
        self.att = nn.Sequential(nn.Linear(4 * item_d, att_hidden), nn.PReLU(),
                                 nn.Linear(att_hidden, 1))
        in_d = d + item_d + 2 * d + (item_d if use_seq else 0)
        layers, prev = [], in_d
        for h in mlp:
            layers += [nn.Linear(prev, h), nn.PReLU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, u, v, a, tab, dur, hv, ha, hmask):
        c = torch.cat([self.e_video(v), self.e_author(a)], -1)        # (B, 2d)
        parts = [self.e_user(u), c, self.e_tab(tab), self.e_dur(dur)]
        if self.use_seq:
            h = torch.cat([self.e_video(hv), self.e_author(ha)], -1)  # (B, L, 2d)
            cc = c.unsqueeze(1).expand_as(h)
            s = self.att(torch.cat([h, cc, h * cc, h - cc], -1)).squeeze(-1)  # (B,L)
            s = s.masked_fill(~hmask, float('-inf'))
            w = torch.softmax(s, dim=1)
            w = torch.nan_to_num(w, nan=0.0)          # rows with an empty history
            parts.append((w.unsqueeze(-1) * h).sum(1))
        return self.mlp(torch.cat(parts, -1)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', default='exposure', choices=['exposure', 'positive'])
    ap.add_argument('--no-seq', action='store_true', help='architecture control')
    ap.add_argument('--d', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.003)
    ap.add_argument('--bs', type=int, default=4096)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--l2', type=float, default=1e-6)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=min(16, os.cpu_count() or 8))
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    logs = D.load_logs()
    masks = D.split_slices(logs)

    ids, cats, _ = D.load_video_features()
    vpos = {int(v): i for i, v in enumerate(ids)}
    aj = D.VIDEO_CAT_COLS.index('author_id')
    avals = np.array([int(float(x)) for x in cats[:, aj].astype(str)], dtype=np.int64)
    vidx = np.array([vpos.get(int(v), -1) for v in logs['video_id']])
    author_raw = np.where(vidx >= 0, avals[np.clip(vidx, 0, None)], -1)

    edges = np.quantile(logs['duration_ms'][masks['train']], np.linspace(0, 1, 11)[1:-1])
    dur_raw = np.searchsorted(edges, logs['duration_ms']).astype(np.int64)

    u_c, n_user = dense(logs['user_id'])
    v_c, n_video = dense(logs['video_id'])
    a_c, n_author = dense(author_raw)
    t_c, n_tab = dense(logs['tab'].astype(np.int64))
    d_c, n_dur = dense(dur_raw)

    # video_id -> its dense code and its author's dense code, for sequence lookup
    v2code = np.zeros(int(logs['video_id'].max()) + 2, dtype=np.int64)
    v2acode = np.zeros_like(v2code)
    v2code[logs['video_id']] = v_c
    v2acode[logs['video_id']] = a_c

    cache = os.path.join(os.path.dirname(__file__), '..', 'cache', 'seq_L20.npz')
    y_bool = (logs['long_view'] != 0)
    if os.path.exists(cache):
        z = np.load(cache)
        Hs = z['H'] if a.seq == 'exposure' else z['Hp']
        hl = z['hl'] if a.seq == 'exposure' else z['hlp']
    else:
        vm = None if a.seq == 'exposure' else (y_bool & masks['train'])
        Hs, hl = SQ.build_sequences(logs['user_id'], logs['video_id'],
                                    logs['time_ms'], L=20, valid_mask=vm)
    L = Hs.shape[1]
    hv = v2code[Hs]
    ha = v2acode[Hs]
    hmask = (np.arange(L)[None, :] < hl[:, None])
    hv = np.where(hmask, hv, PAD); ha = np.where(hmask, ha, PAD)

    y = y_bool.astype(np.float32)
    use_seq = not a.no_seq

    def tens(m):
        return dict(u=torch.from_numpy(u_c[m]), v=torch.from_numpy(v_c[m]),
                    a=torch.from_numpy(a_c[m]), tab=torch.from_numpy(t_c[m]),
                    dur=torch.from_numpy(d_c[m]), hv=torch.from_numpy(hv[m]),
                    ha=torch.from_numpy(ha[m]),
                    hmask=torch.from_numpy(hmask[m]),
                    y=torch.from_numpy(y[m]))

    TR, VA, TE = tens(masks['train']), tens(masks['valid']), tens(masks['test'])
    uva, yva = logs['user_id'][masks['valid']], y[masks['valid']]
    ute, yte = logs['user_id'][masks['test']], y[masks['test']]

    model = DIN(n_user, n_video, n_author, n_tab, n_dur, d=a.d, use_seq=use_seq)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.l2)
    lossf = nn.BCEWithLogitsLoss()
    N = len(TR['y'])

    @torch.no_grad()
    def predict(T):
        model.eval()
        out = []
        for i in range(0, len(T['y']), 20000):
            sl = slice(i, i + 20000)
            out.append(model(T['u'][sl], T['v'][sl], T['a'][sl], T['tab'][sl],
                             T['dur'][sl], T['hv'][sl], T['ha'][sl],
                             T['hmask'][sl]).numpy())
        return np.concatenate(out)

    exp_id = (f"1I-din-{'noseq' if a.no_seq else a.seq}-d{a.d}-lr{a.lr:g}"
              + (f'-seed{a.seed}' if a.seed else ''))
    hyp = ('DIN attention over the user\'s recent behaviour adds information no flat '
           'feature encodes; 99.7% of eval rows carry a history so the signal is '
           'available where it is scored')
    if a.no_seq:
        hyp = ('ARCHITECTURE CONTROL: same network with the attention branch removed, '
               'to separate "sequences help" from "an MLP beats an FM"')
    cfg = dict(model='din', use_sequence=use_seq, seq_kind=(None if a.no_seq else a.seq),
               seq_len=L, d=a.d, lr=a.lr, bs=a.bs, l2=a.l2, epochs=a.epochs,
               patience=a.patience, seed=a.seed,
               n_params=sum(p.numel() for p in model.parameters()))
    print(f'{exp_id}: {cfg["n_params"]:,} params, {a.threads} threads, N={N:,}')

    with H.Experiment(exp_id, phase='1I', axis='behaviour_sequence',
                      hypothesis=hyp, config=cfg,
                      tags=['sequence', 'din', 'torch']) as ex:
        best, best_state, best_ep, bad, hist = -1.0, None, 0, 0, []
        for ep in range(1, a.epochs + 1):
            model.train()
            t0 = time.time()
            perm = torch.randperm(N)
            tot = 0.0
            for i in range(0, N, a.bs):
                b = perm[i:i + a.bs]
                opt.zero_grad()
                z = model(TR['u'][b], TR['v'][b], TR['a'][b], TR['tab'][b],
                          TR['dur'][b], TR['hv'][b], TR['ha'][b], TR['hmask'][b])
                loss = lossf(z, TR['y'][b])
                loss.backward()
                opt.step()
                tot += float(loss) * len(b)
            Lm = tot / N
            va = H.score(uva, yva, predict(VA))
            hist.append({'epoch': ep, 'loss': round(Lm, 4),
                         'valid_primary': round(va['primary'], 4),
                         'valid_GAUC': round(va['GAUC'], 4),
                         'valid_nDCG@5': round(va['nDCG@5'], 4),
                         'seconds': round(time.time() - t0, 1)})
            print(f"  epoch {ep:2d} | loss {Lm:.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} "
                  f"| {time.time()-t0:.0f}s")
            if va['primary'] > best + 1e-5:
                best, best_ep, bad = va['primary'], ep, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= a.patience:
                    print(f'  early stop at epoch {ep}')
                    break
        if best_state:
            model.load_state_dict(best_state)
        ex.record_train(epochs_run=len(hist), best_epoch=best_ep,
                        best_valid_primary=round(best, 5), history=hist)
        ex.record_metrics('valid', H.score(uva, yva, predict(VA)))
        ex.record_metrics('test', H.score(ute, yte, predict(TE)))

    print()
    print(H.summarize([r for r in H.read_log() if r['phase'] in ('1I', '1A')]))


if __name__ == '__main__':
    main()
