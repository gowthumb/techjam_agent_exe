"""Phase 7: the architecture ladder, run under the loss that actually wins.

The candidate shortlist (FFM, DeepFM, AutoInt) was never tried, and the one GBDT
pass ran under pointwise/lambdarank rather than a pairwise objective. Every
architecture here is therefore trained under BOTH pointwise and BPR, so the
comparison isolates the architecture instead of confounding it with the loss.

Architectures (all on the same five baseline fields, same flat embedding table):
  fm        second-order interactions only -- the numpy baseline, re-expressed in
            torch so the other arms differ ONLY in architecture, not in framework
  ffm       field-aware: a separate embedding per (field, interacting field)
  deepfm    FM second-order term + an MLP over the concatenated embeddings
  autoint   multi-head self-attention over the field embeddings

Sequence option (--seq mean): mean-pooled embedding of the user's last 20
long-viewed videos, built from TRAIN LABELS ONLY. This is the cell the DIN work
left empty -- the attention branch reversed, but plain pooling under BPR was never
tried.

CPU ONLY. torch is the +cpu build here; the script asserts no CUDA is used.

Run:  python experiments/p7_arch_bpr.py --arch fm,ffm,deepfm,autoint --loss bpr,pointwise
"""
import os, sys, time, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import torch.nn as nn
from explib import dataset as D, features as F, fm as npfm, harness as H, sequence as SQ

assert not torch.cuda.is_available() or os.environ.get('ALLOW_CUDA'), \
    'this workstream is CPU-only; refusing to run with CUDA visible'
DEVICE = torch.device('cpu')


class Net(nn.Module):
    def __init__(self, total_dim, n_fields, k=16, arch='fm', seq_dim=0,
                 mlp=(128, 64), heads=2):
        super().__init__()
        self.arch, self.F, self.k = arch, n_fields, k
        self.emb = nn.Embedding(total_dim, k)
        self.lin = nn.Embedding(total_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.emb.weight, 0, 0.01)
        nn.init.zeros_(self.lin.weight)
        self.seq_dim = seq_dim
        if arch == 'ffm':
            # one embedding per (field, interacting field) pair
            self.femb = nn.Embedding(total_dim, n_fields * k)
            nn.init.normal_(self.femb.weight, 0, 0.01)
        if arch == 'autoint':
            self.att = nn.MultiheadAttention(k, heads, batch_first=True)
            self.att_out = nn.Linear(n_fields * k, 1)
        if arch == 'deepfm':
            d = n_fields * k + seq_dim
            layers, prev = [], d
            for h in mlp:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            layers += [nn.Linear(prev, 1)]
            self.mlp = nn.Sequential(*layers)
        if seq_dim and arch != 'deepfm':
            self.seq_proj = nn.Linear(seq_dim, 1)

    def forward(self, X, seq=None):
        E = self.emb(X)                                   # (B,F,k)
        z = self.bias + self.lin(X).sum((1, 2))
        if self.arch == 'ffm':
            G = self.femb(X).view(X.shape[0], self.F, self.F, self.k)
            acc = 0.0
            for i in range(self.F):
                for j in range(i + 1, self.F):
                    acc = acc + (G[:, i, j, :] * G[:, j, i, :]).sum(-1)
            z = z + acc
        else:
            S = E.sum(1)
            z = z + 0.5 * ((S ** 2).sum(-1) - (E ** 2).sum((1, 2)))
        if self.arch == 'autoint':
            a, _ = self.att(E, E, E)
            z = z + self.att_out(a.flatten(1)).squeeze(-1)
        if self.arch == 'deepfm':
            flat = E.flatten(1)
            if seq is not None:
                flat = torch.cat([flat, seq], -1)
            z = z + self.mlp(flat).squeeze(-1)
        elif seq is not None:
            z = z + self.seq_proj(seq).squeeze(-1)
        return z


def make_pairs(groups, rng):
    P = np.concatenate([g[1] for g in groups])
    N = np.concatenate([g[2][rng.integers(0, len(g[2]), len(g[1]))] for g in groups])
    p = rng.permutation(len(P))
    return P[p], N[p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch', default='fm,ffm,deepfm,autoint')
    ap.add_argument('--loss', default='bpr,pointwise')
    ap.add_argument('--seq', default='none', choices=['none', 'mean'])
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=None,
                help='default: 0.001 pointwise / 0.0002 bpr (the validated values)')
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--patience', type=int, default=4)
    ap.add_argument('--bs', type=int, default=8192)
    ap.add_argument('--seeds', default='0')
    ap.add_argument('--threads', type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    logs = D.load_logs()
    masks = D.split_slices(logs)
    enc, dim, _ = F.encode_splits(logs, masks, F.BASELINE_FIELDS)
    nF = len(F.BASELINE_FIELDS)

    seq_t = {}
    seq_dim = 0
    if a.seq == 'mean':
        cache = os.path.join(os.path.dirname(__file__), '..', 'cache', 'seq_L20.npz')
        z = np.load(cache)
        Hs, hl = z['Hp'], z['hlp']            # train-label-only positive history
        v2slot = np.zeros(int(logs['video_id'].max()) + 2, dtype=np.int64)
        # map raw video ids onto the encoder's video_id field slots
        vid_field = F.BASELINE_FIELDS.index('video_id')
        Xall = np.concatenate([enc[s][0] for s in ('train', 'valid', 'test')])
        vall = np.concatenate([logs['video_id'][masks[s]] for s in ('train', 'valid', 'test')])
        v2slot[vall] = Xall[:, vid_field]
        seq_slots = v2slot[Hs]
        mask = (np.arange(Hs.shape[1])[None, :] < hl[:, None])
        seq_slots = np.where(mask, seq_slots, 0)
        seq_dim = a.k
        for sp, m in masks.items():
            seq_t[sp] = (torch.from_numpy(seq_slots[m]), torch.from_numpy(mask[m]))
        print(f'sequence: mean-pooled positives, dim={seq_dim}, '
              f'{(hl > 0).mean():.1%} of rows have history')

    T = {sp: (torch.from_numpy(enc[sp][0].astype(np.int64)),
              torch.from_numpy(enc[sp][1]),
              enc[sp][2]) for sp in ('train', 'valid', 'test')}

    for seed in [int(s) for s in a.seeds.split(',')]:
        for arch in a.arch.split(','):
            for loss in a.loss.split(','):
                # 0.0002 is the value 5 seeds validated for BPR. 0.002 sits in the danger
                # zone the KB documents, and running the grid there handicapped
                # every architecture under BPR.
                lr = a.lr if a.lr else (0.0002 if loss == 'bpr' else 0.001)
                run(arch, loss, lr, a, seed, T, seq_t, dim, nF, seq_dim, enc)

    print()
    print(H.summarize([r for r in H.read_log() if r['phase'] == '7']))


def run(arch, loss, lr, a, seed, T, seq_t, dim, nF, seq_dim, enc):
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, ytr, utr = T['train']
    Xva, yva, uva = T['valid']
    Xte, yte, ute = T['test']
    net = Net(dim, nF, k=a.k, arch=arch, seq_dim=seq_dim).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-6)
    rng = np.random.default_rng(seed)
    groups = npfm.user_groups(np.asarray(utr), ytr.numpy()) if loss == 'bpr' else None

    def emb_seq(sp, idx=None):
        if not seq_dim:
            return None
        s, m = seq_t[sp]
        if idx is not None:
            s, m = s[idx], m[idx]
        e = net.emb(s) * m.unsqueeze(-1)
        return e.sum(1) / m.sum(1, keepdim=True).clamp(min=1)

    @torch.no_grad()
    def predict(sp, X):
        net.eval(); out = []
        for i in range(0, len(X), 20000):
            sl = slice(i, i + 20000)
            out.append(net(X[sl], emb_seq(sp, torch.arange(i, min(i + 20000, len(X))))
                           if seq_dim else None).numpy())
        return np.concatenate(out)

    # lr goes in the id: a mis-tuned run must never be able to masquerade as a
    # tuned one in the log, which is exactly what happened on the first pass.
    eid = f"7-{arch}-{loss}-k{a.k}-lr{lr:g}" + (f'-seq{a.seq}' if seq_dim else '') + \
          (f'-seed{seed}' if seed else '')
    hyp = (f'{arch} under {loss}: the shortlist architectures were never tried, and the one '
           f'GBDT pass ran under a pointwise/lambdarank framing rather than the pairwise loss '
           f'that wins on this metric. Training both losses isolates architecture from loss.')
    cfg = dict(model=arch, loss=loss, k=a.k, lr=lr, bs=a.bs, epochs=a.epochs,
               patience=a.patience, seed=seed, seq=a.seq, device='cpu',
               n_params=sum(p.numel() for p in net.parameters()),
               fields=F.BASELINE_FIELDS)
    print(f"\n[{eid}] {cfg['n_params']:,} params, lr={lr}")

    with H.Experiment(eid, phase='7', axis='architecture_x_loss', hypothesis=hyp,
                      config=cfg, tags=['arch', arch, loss]) as ex:
        best, best_state, best_ep, bad, hist = -1.0, None, 0, 0, []
        N = len(ytr)
        for ep in range(1, a.epochs + 1):
            net.train(); t0 = time.time(); tot = 0.0; nb = 0
            if loss == 'bpr':
                P, Ng = make_pairs(groups, rng)
                for i in range(0, len(P), a.bs):
                    pi = torch.from_numpy(P[i:i + a.bs]); ni = torch.from_numpy(Ng[i:i + a.bs])
                    opt.zero_grad()
                    zp = net(Xtr[pi], emb_seq('train', pi) if seq_dim else None)
                    zn = net(Xtr[ni], emb_seq('train', ni) if seq_dim else None)
                    l = -torch.nn.functional.logsigmoid(zp - zn).mean()
                    l.backward(); opt.step(); tot += float(l.detach()); nb += 1
            else:
                perm = torch.randperm(N)
                for i in range(0, N, a.bs):
                    b = perm[i:i + a.bs]
                    opt.zero_grad()
                    z = net(Xtr[b], emb_seq('train', b) if seq_dim else None)
                    l = nn.functional.binary_cross_entropy_with_logits(z, ytr[b])
                    l.backward(); opt.step(); tot += float(l.detach()); nb += 1
            L = tot / max(nb, 1)
            va = H.score(uva, yva.numpy(), predict('valid', Xva))
            hist.append({'epoch': ep, 'loss': round(L, 4),
                         'valid_primary': round(va['primary'], 4),
                         'valid_GAUC': round(va['GAUC'], 4),
                         'valid_nDCG@5': round(va['nDCG@5'], 4),
                         'seconds': round(time.time() - t0, 1)})
            print(f"  ep {ep:2d} | loss {L:.4f} | GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.0f}s")
            if va['primary'] > best + 1e-5:
                best, best_ep, bad = va['primary'], ep, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= a.patience:
                    print(f'  early stop at epoch {ep}')
                    break
        if best_state:
            net.load_state_dict(best_state)
        ex.record_train(epochs_run=len(hist), best_epoch=best_ep,
                        best_valid_primary=round(best, 5), history=hist)
        ex.record_metrics('valid', H.score(uva, yva.numpy(), predict('valid', Xva)))
        ex.record_metrics('test', H.score(ute, yte.numpy(), predict('test', Xte)))


if __name__ == '__main__':
    main()
