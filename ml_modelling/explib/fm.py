"""Factorization Machine with pluggable loss.

The starter kit's FM optimizes pointwise logloss while the metric (GAUC / nDCG@5)
is a *within-user ranking* metric. The organizers flag this mismatch as the most
likely source of headroom. This module keeps the kit's exact model, optimizer and
init, and swaps only dL/dz -- so any score difference is attributable to the loss
and nothing else.

Losses:
  pointwise : sigmoid cross-entropy per row      (reproduces the official baseline)
  bpr       : within-user pairwise, -log sigmoid(z_pos - z_neg)
  listwise  : within-user softmax cross-entropy against the positive distribution
  hybrid    : pointwise + lam * listwise

Only the per-row gradient coefficient g = dL/dz changes; the FM's dz/dparams and
the Adam update are shared, byte-for-byte, across all four.
"""
import time
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
    """Identical math and init to kuairand-starter-kit/baseline.py::FM."""

    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]                                    # (B,F,k)
        S = E.sum(1)                                     # (B,k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def apply_grad(self, X, g, E, S, update_bias=True):
        """g: (B,) per-row dL/dz. Everything below is loss-agnostic."""
        g = g.astype(np.float32)
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, g[:, None])
        np.add.at(gV, X, g[:, None, None] * (S[:, None, :] - E))
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for P, G, M, Vv in ((self.V, gV, self.mV, self.vV), (self.W, gW, self.mW, self.vW)):
            M *= b1; M += (1 - b1) * G
            Vv *= b2; Vv += (1 - b2) * (G * G)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        if update_bias:
            self.b -= self.lr * g.sum()

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])

    def state(self):
        return (self.V.copy(), self.W.copy(), np.float32(self.b))

    def load(self, st):
        self.V, self.W, self.b = st[0].copy(), st[1].copy(), np.float32(st[2])


# ---------------------------------------------------------------- group utils
def user_groups(users, y, skip_degenerate=True):
    """Row indices per user, split into positives/negatives.

    skip_degenerate drops all-positive and all-negative users: their nDCG is
    constant (1 / 0) and GAUC excludes them, so they carry no ranking signal --
    exactly the exclusion evaluate.py applies.
    """
    order = np.argsort(users, kind='stable')
    su = users[order]
    bounds = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
    starts = np.r_[bounds, len(su)]
    groups = []
    for a, b in zip(starts[:-1], starts[1:]):
        rows = order[a:b]
        lab = y[rows]
        pos = rows[lab == 1]; neg = rows[lab == 0]
        if skip_degenerate and (len(pos) == 0 or len(neg) == 0):
            continue
        groups.append((rows, pos, neg))
    return groups


def seg_softmax(z, seg, nseg):
    mx = np.full(nseg, -np.inf, dtype=np.float64)
    np.maximum.at(mx, seg, z)
    e = np.exp(z - mx[seg])
    s = np.bincount(seg, weights=e, minlength=nseg)
    return (e / s[seg]).astype(np.float32)


# ---------------------------------------------------------------- loss steps
def _step_pointwise(m, X, y, rows, bs, rng):
    idx = rng.permutation(len(rows))
    losses = []
    for i in range(0, len(idx), bs):
        r = rows[idx[i:i + bs]]
        Xb, yb = X[r], y[r]
        z, E, S = m.logits(Xb)
        p = sigmoid(z)
        m.apply_grad(Xb, (p - yb) / len(yb), E, S)
        losses.append(float(-np.mean(yb * np.log(p + 1e-9) + (1 - yb) * np.log(1 - p + 1e-9))))
    return float(np.mean(losses))


def _step_bpr(m, X, y, groups, pairs_per_pos, bs, rng):
    pos_all, neg_all = [], []
    for _, pos, neg in groups:
        n = len(pos) * pairs_per_pos
        pos_all.append(np.repeat(pos, pairs_per_pos))
        neg_all.append(neg[rng.integers(0, len(neg), n)])
    P = np.concatenate(pos_all); N = np.concatenate(neg_all)
    perm = rng.permutation(len(P)); P, N = P[perm], N[perm]
    losses = []
    for i in range(0, len(P), bs):
        p_r, n_r = P[i:i + bs], N[i:i + bs]
        r = np.concatenate([p_r, n_r])
        Xb = X[r]
        z, E, S = m.logits(Xb)
        h = len(p_r)
        d = z[:h] - z[h:]
        s = sigmoid(-d)                       # dL/d(delta) magnitude
        g = np.empty(len(r), dtype=np.float32)
        g[:h] = -s / h
        g[h:] = s / h
        # the global bias cancels in a pairwise loss -- leave b untouched
        m.apply_grad(Xb, g, E, S, update_bias=False)
        losses.append(float(np.mean(-np.log(sigmoid(d) + 1e-9))))
    return float(np.mean(losses))


def _step_listwise(m, X, y, groups, users_per_batch, rng, scale=1.0):
    gidx = rng.permutation(len(groups))
    losses = []
    for i in range(0, len(gidx), users_per_batch):
        gs = [groups[j] for j in gidx[i:i + users_per_batch]]
        rows = np.concatenate([g[0] for g in gs])
        seg = np.repeat(np.arange(len(gs)), [len(g[0]) for g in gs])
        Xb, yb = X[rows], y[rows]
        z, E, S = m.logits(Xb)
        p = seg_softmax(z, seg, len(gs))
        npos = np.bincount(seg, weights=yb, minlength=len(gs))
        t = (yb / npos[seg]).astype(np.float32)     # target = uniform over positives
        m.apply_grad(Xb, scale * (p - t) / len(gs), E, S, update_bias=False)
        losses.append(float(-np.sum(t * np.log(p + 1e-9)) / len(gs)))
    return float(np.mean(losses))


def _step_hybrid(m, X, y, rows, groups, bs, users_per_batch, lam, rng):
    lp = _step_pointwise(m, X, y, rows, bs, rng)
    ll = _step_listwise(m, X, y, groups, users_per_batch, rng, scale=lam)
    return lp + lam * ll


# ---------------------------------------------------------------- training
def train(enc, dim, loss='pointwise', k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
          patience=4, seed=0, pairs_per_pos=1, users_per_batch=256, lam=1.0,
          skip_degenerate=True, evaluator=None, verbose=True):
    """Train and early-stop on valid primary. Returns (model, info)."""
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    m = FM(dim, k=k, lr=lr, l2=l2, seed=seed)
    rng = np.random.default_rng(seed)
    rows = np.arange(len(ytr))
    groups = (user_groups(utr, ytr, skip_degenerate)
              if loss in ('bpr', 'listwise', 'hybrid') else None)
    if groups is not None and verbose:
        print(f"  {len(groups)} training users with both classes "
              f"({sum(len(g[0]) for g in groups)} rows)")
    best, best_state, best_ep, bad = -1.0, None, 0, 0
    hist = []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        if loss == 'pointwise':
            L = _step_pointwise(m, Xtr, ytr, rows, bs, rng)
        elif loss == 'bpr':
            L = _step_bpr(m, Xtr, ytr, groups, pairs_per_pos, bs, rng)
        elif loss == 'listwise':
            L = _step_listwise(m, Xtr, ytr, groups, users_per_batch, rng)
        elif loss == 'hybrid':
            L = _step_hybrid(m, Xtr, ytr, rows, groups, bs, users_per_batch, lam, rng)
        else:
            raise ValueError(f'unknown loss {loss}')
        va = evaluator(uva, yva, m.predict(Xva))
        hist.append({'epoch': ep, 'loss': round(L, 4),
                     'valid_primary': round(va['primary'], 4),
                     'valid_GAUC': round(va['GAUC'], 4),
                     'valid_nDCG@5': round(va['nDCG@5'], 4),
                     'seconds': round(time.time() - t0, 1)})
        if verbose:
            print(f"  epoch {ep:2d} | loss {L:.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} "
                  f"| {time.time()-t0:.1f}s")
        if not np.isfinite(L):
            if verbose:
                print('  DIVERGED (non-finite loss) -- stopping')
            hist[-1]['diverged'] = True
            break
        if va['primary'] > best + 1e-5:
            best, best_ep, bad = va['primary'], ep, 0
            best_state = m.state()
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break
    if best_state is not None:
        m.load(best_state)
    info = {'epochs_run': len(hist), 'best_epoch': best_ep,
            'best_valid_primary': round(best, 5), 'history': hist,
            'train_users_used': (len(groups) if groups is not None else None),
            'diverged': bool(hist and hist[-1].get('diverged'))}
    return m, info
