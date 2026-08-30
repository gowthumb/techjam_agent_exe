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
  ssm       : sampled softmax / InfoNCE -- each positive vs neg_per_pos sampled
              negatives from the SAME user, softmax at temperature `temp`, target
              the positive. Distinct from `listwise`: that is a softmax over the
              whole impression list with a uniform-over-positives target, which is
              a poor fit when ~1/3 of the list is positive. ssm contrasts one
              positive against a few negatives, the standard implicit-feedback
              ranking loss. Phase 12 / KNOWLEDGE_BASE_PLAN.md Phase 1a.

Only the per-row gradient coefficient g = dL/dz changes; the FM's dz/dparams and
the Adam update are shared, byte-for-byte, across all of them.
"""
import time
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
    """Identical math and init to kuairand-starter-kit/baseline.py::FM."""

    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0, sparse=False,
                 emb_noise=0.0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.sparse = sparse       # opt-in; see apply_grad_sparse for the caveat
        # Gaussian noise added to embeddings during training only. Because the
        # noise is ADDITIVE, dE'/dV = I and the gradient keeps its exact form --
        # apply_grad needs no change, it just receives the noised E and S.
        self.emb_noise = emb_noise
        self._nrng = np.random.default_rng(seed + 9973)
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X, noise=False):
        E = self.V[X]                                    # (B,F,k)
        if noise and self.emb_noise > 0:
            E = E + self._nrng.normal(0, self.emb_noise, E.shape).astype(np.float32)
        S = E.sum(1)                                     # (B,k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def apply_grad(self, X, g, E, S, update_bias=True):
        """g: (B,) per-row dL/dz. Everything below is loss-agnostic."""
        if self.sparse:
            return self.apply_grad_sparse(X, g, E, S, update_bias)
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

    def apply_grad_sparse(self, X, g, E, S, update_bias=True):
        """Same gradient, but touching only the embedding rows in this batch.

        WHY THIS EXISTS: the dense path allocates and updates the whole (dim, k)
        table every batch, which is O(vocab) per step regardless of batch size. On
        KuaiRand-Pure vocab is 40K and that is free; on KuaiRand-1K it is 4.4M and
        it dominates everything.

        NOT NUMERICALLY IDENTICAL TO THE DENSE PATH. This is lazy/sparse Adam: rows
        absent from a batch get no L2 decay and no moment decay that step. That is
        what every production sparse optimizer does, and the KB already establishes
        L2 is not a live knob here, but it is an approximation -- so it is opt-in
        and every Pure result in the log uses the dense path.
        """
        g = g.astype(np.float32)
        F = X.shape[1]
        flat = X.ravel()
        uniq, inv = np.unique(flat, return_inverse=True)
        contrib = (g[:, None, None] * (S[:, None, :] - E)).reshape(-1, self.V.shape[1])
        gV = np.zeros((len(uniq), self.V.shape[1]), dtype=np.float32)
        np.add.at(gV, inv, contrib)
        gW = np.bincount(inv, weights=np.repeat(g, F),
                         minlength=len(uniq)).astype(np.float32)
        gV += self.l2 * self.V[uniq]
        gW += self.l2 * self.W[uniq]
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
        mV = b1 * self.mV[uniq] + (1 - b1) * gV
        vV = b2 * self.vV[uniq] + (1 - b2) * (gV * gV)
        self.mV[uniq], self.vV[uniq] = mV, vV
        self.V[uniq] -= self.lr * (mV / c1) / (np.sqrt(vV / c2) + eps)
        mW = b1 * self.mW[uniq] + (1 - b1) * gW
        vW = b2 * self.vW[uniq] + (1 - b2) * (gW * gW)
        self.mW[uniq], self.vW[uniq] = mW, vW
        self.W[uniq] -= self.lr * (mW / c1) / (np.sqrt(vW / c2) + eps)
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
def _step_pointwise(m, X, y, rows, bs, rng, w=None):
    idx = rng.permutation(len(rows))
    losses = []
    for i in range(0, len(idx), bs):
        r = rows[idx[i:i + bs]]
        Xb, yb = X[r], y[r]
        z, E, S = m.logits(Xb, noise=True)
        p = sigmoid(z)
        g = (p - yb)
        if w is not None:
            g = g * w[r]
        m.apply_grad(Xb, g / len(yb), E, S)
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
        z, E, S = m.logits(Xb, noise=True)
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
        z, E, S = m.logits(Xb, noise=True)
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


def _step_ssm(m, X, y, groups, neg_per_pos, temp, bs, rng, global_pool=None):
    """Sampled softmax / InfoNCE. One example = 1 positive + neg_per_pos negatives.

    Negatives are sampled per positive from the SAME user's negatives, so the
    contrast is within-user exactly as GAUC/nDCG@5 are. global_pool, if given, is
    an array of row indices to sample negatives from instead -- the control that
    isolates whether the within-user structure of the negatives matters.

    p_j = softmax(z_j / temp) over the (1 + neg_per_pos) logits of an example;
    L = -log p_0 ; dL/dz_j = (p_j - 1[j=0]) / temp.  The softmax is shift
    invariant so the global bias cancels -- b is left untouched, as in bpr.
    """
    pos_list, neg_list = [], []
    for _, pos, neg in groups:
        if len(neg) == 0:
            continue
        pos_list.append(pos)
        pool = global_pool if global_pool is not None else neg
        neg_list.append(pool[rng.integers(0, len(pool), (len(pos), neg_per_pos))])
    if not pos_list:
        return 0.0
    P = np.concatenate(pos_list)                          # (G,)
    N = np.concatenate(neg_list, axis=0)                  # (G, neg_per_pos)
    perm = rng.permutation(len(P))
    P, N = P[perm], N[perm]
    S = neg_per_pos + 1
    ex_per_batch = max(1, bs // S)
    losses = []
    for i in range(0, len(P), ex_per_batch):
        p_r = P[i:i + ex_per_batch]
        n_r = N[i:i + ex_per_batch]
        g = len(p_r)
        rows = np.empty(g * S, dtype=n_r.dtype)
        rows[0::S] = p_r
        for j in range(neg_per_pos):
            rows[j + 1::S] = n_r[:, j]
        Xb = X[rows]
        z, E, Sm = m.logits(Xb, noise=True)
        seg = np.repeat(np.arange(g), S)
        p = seg_softmax(z / temp, seg, g)
        t = np.zeros(g * S, dtype=np.float32)
        t[0::S] = 1.0
        m.apply_grad(Xb, ((p - t) / temp / g).astype(np.float32), E, Sm,
                     update_bias=False)
        losses.append(float(-np.mean(np.log(p[0::S] + 1e-9))))
    return float(np.mean(losses))


# ---------------------------------------------------------------- training
def train(enc, dim, loss='pointwise', k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192,
          patience=4, seed=0, pairs_per_pos=1, users_per_batch=256, lam=1.0,
          skip_degenerate=True, evaluator=None, verbose=True, row_weight=None,
          sparse=False, emb_noise=0.0, neg_per_pos=8, temp=1.0, ssm_global=False):
    """Train and early-stop on valid primary. Returns (model, info)."""
    Xtr, ytr, utr = enc['train']
    Xva, yva, uva = enc['valid']
    m = FM(dim, k=k, lr=lr, l2=l2, seed=seed, sparse=sparse,
           emb_noise=emb_noise)
    rng = np.random.default_rng(seed)
    rows = np.arange(len(ytr))
    groups = (user_groups(utr, ytr, skip_degenerate)
              if loss in ('bpr', 'listwise', 'hybrid', 'ssm') else None)
    if groups is not None and verbose:
        print(f"  {len(groups)} training users with both classes "
              f"({sum(len(g[0]) for g in groups)} rows)")
    best, best_state, best_ep, bad = -1.0, None, 0, 0
    hist = []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        if loss == 'pointwise':
            L = _step_pointwise(m, Xtr, ytr, rows, bs, rng, w=row_weight)
        elif loss == 'bpr':
            L = _step_bpr(m, Xtr, ytr, groups, pairs_per_pos, bs, rng)
        elif loss == 'listwise':
            L = _step_listwise(m, Xtr, ytr, groups, users_per_batch, rng)
        elif loss == 'hybrid':
            L = _step_hybrid(m, Xtr, ytr, rows, groups, bs, users_per_batch, lam, rng)
        elif loss == 'ssm':
            L = _step_ssm(m, Xtr, ytr, groups, neg_per_pos, temp, bs, rng,
                          global_pool=(rows if ssm_global else None))
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
