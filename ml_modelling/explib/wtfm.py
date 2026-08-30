"""Two-head shared-embedding FM: long_view (binary) + watch-time (regression).

GROUNDING. `KNOWLEDGE_BASE_PLAN.md` Phase 1c (multi-task / auxiliary signals);
starter kit README direction #4 (watch-time modelling / CWM -- "这是个有研究深度的方向").

WHY THIS IS NOT THE PHASE 1C MULTI-TASK RESULT. Phase 1C tried the 11 other
BINARY feedback signals as auxiliary heads and found they act as regularisers
only -- a random label at a matched sparsity scored the same as `is_follow`. The
watch-time target is different in kind: `long_view` is a DETERMINISTIC threshold
on watch time (`1` iff `play_time_ms >= min(duration_ms, 18000)`, verified to
match 97.9% of rows). The binary label is a coarsened view of a continuous signal
the log carries in full. A regression head on that continuous signal is trained
on strictly more information than the binary head, and gets gradient from every
row rather than a 0/1.

CENSORING (the CWM insight). A completed play (`play_time_ms >= duration_ms`)
right-censors the true watch desire: the video ended, we do not know how much
longer the user would have watched. For those rows the regression loss is
ONE-SIDED -- it penalises predicting LESS than the observed ratio, not more. The
two-sided-Huber variant is the control.

MODEL. Shared V (dim, k) and per-head linear W_h (dim) + interaction weights
A_h (k). Head 0 logit -> sigmoid -> BCE against long_view. Head 1 raw output ->
Huber (one-sided on censored rows) against the watch ratio. Ranking score is
configurable: head 0, head 1, or a rank-blend (chosen on valid, per the KB).

With w_wt = 0 and A_0 frozen at ones this is byte-for-byte the starter kit FM on
head 0 -- the single-task control lands on the baseline.
"""
import time
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class WTFM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0, learn_A=True):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros((dim, 2), dtype=np.float32)
        self.b = np.zeros(2, dtype=np.float32)
        self.A = np.ones((2, k), dtype=np.float32)
        self.lr, self.l2, self.learn_A = lr, l2, learn_A
        self.m = {n: np.zeros_like(p) for n, p in (('V', self.V), ('W', self.W), ('A', self.A))}
        self.v = {n: np.zeros_like(p) for n, p in (('V', self.V), ('W', self.W), ('A', self.A))}
        self.t = 0

    def forward(self, X):
        E = self.V[X]
        S = E.sum(1)
        Q = 0.5 * (S ** 2 - (E ** 2).sum(1))                 # (B,k)
        Z = self.b + self.W[X].sum(1) + Q @ self.A.T         # (B,2)
        return Z, E, S, Q

    def backward(self, X, G, E, S, Q):
        """G: (B,2) dL/dZ already summed with task weights."""
        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, X, G[:, None, :])
        dQ = G @ self.A                                      # (B,k)
        np.add.at(gV, X, dQ[:, None, :] * (S[:, None, :] - E))
        gA = G.T @ Q                                         # (2,k)
        gV += self.l2 * self.V
        gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        upd = [('V', self.V, gV), ('W', self.W, gW)]
        if self.learn_A:
            upd.append(('A', self.A, gA))
        for name, P, Gm in upd:
            M, Vv = self.m[name], self.v[name]
            M *= b1; M += (1 - b1) * Gm
            Vv *= b2; Vv += (1 - b2) * (Gm * Gm)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * G.sum(0)

    def predict(self, X, head=0, bs=200_000):
        out = []
        for i in range(0, len(X), bs):
            Z = self.forward(X[i:i + bs])[0]
            out.append(sigmoid(Z[:, 0]) if head == 0 else Z[:, head])
        return np.concatenate(out)

    def predict_both(self, X, bs=200_000):
        p0, p1 = [], []
        for i in range(0, len(X), bs):
            Z = self.forward(X[i:i + bs])[0]
            p0.append(sigmoid(Z[:, 0])); p1.append(Z[:, 1])
        return np.concatenate(p0), np.concatenate(p1)

    def state(self):
        return (self.V.copy(), self.W.copy(), self.b.copy(), self.A.copy())

    def load(self, st):
        self.V, self.W, self.b, self.A = [x.copy() for x in st]


def _huber_grad(r, delta, one_sided_mask):
    """Huber derivative clip(r, -delta, delta); zero the r>0 side on censored rows."""
    g = np.clip(r, -delta, delta)
    if one_sided_mask is not None:
        g = np.where(one_sided_mask & (r > 0), 0.0, g)
    return g.astype(np.float32)


def rank_blend(scores_list, users, weights):
    """Within-user percentile-rank average -- scale-free, so a probability and a
    raw regression output can be combined. weights sum-normalised."""
    users = np.asarray(users)
    order = np.lexsort((np.zeros(len(users)), users))
    su = users[order]
    starts = np.flatnonzero(np.r_[True, su[1:] != su[:-1]])
    bounds = np.r_[starts, len(su)]
    def pct(s):
        out = np.empty(len(s), np.float64)
        so = s[order]
        for a, b in zip(bounds[:-1], bounds[1:]):
            seg = so[a:b]; n = b - a
            rr = np.argsort(np.argsort(seg)).astype(np.float64)
            out[a:b] = rr / max(n - 1, 1)
        z = np.empty(len(s), np.float64); z[order] = out
        return z
    w = np.asarray(weights, float); w = w / w.sum()
    return sum(wi * pct(s) for wi, s in zip(w, scores_list))


def train(X_tr, y_lv, y_wt, censored, X_va, y_va, u_va, dim, w_wt=0.5,
          one_sided=True, huber_delta=0.5, k=16, lr=0.001, l2=1e-6, epochs=40,
          bs=8192, patience=4, seed=0, evaluator=None, verbose=True, learn_A=True,
          select_head='lv'):
    """Train both heads; early-stop on valid primary of the selected ranking score.

    select_head: 'lv' (head 0), 'wt' (head 1), or 'blend' (rank-average, weight
    swept coarsely at selection time -- reported, not tuned to test).
    """
    m = WTFM(dim, k=k, lr=lr, l2=l2, seed=seed, learn_A=learn_A)
    rng = np.random.default_rng(seed)
    best, best_state, best_ep, bad = -1.0, None, 0, 0
    hist = []
    y_lv = y_lv.astype(np.float32); y_wt = y_wt.astype(np.float32)
    cen = censored.astype(bool) if one_sided else None

    def valid_score():
        p0, p1 = m.predict_both(X_va)
        if select_head == 'lv':
            s = p0
        elif select_head == 'wt':
            s = p1
        else:
            s = rank_blend([p0, p1], u_va, [0.5, 0.5])
        return evaluator(u_va, y_va, s)

    for ep in range(1, epochs + 1):
        t0 = time.time()
        idx = rng.permutation(len(y_lv))
        losses = []
        for i in range(0, len(idx), bs):
            r = idx[i:i + bs]
            Xb = X_tr[r]
            Z, E, S, Q = m.forward(Xb)
            B = len(r)
            G = np.empty((B, 2), np.float32)
            p0 = sigmoid(Z[:, 0])
            G[:, 0] = (p0 - y_lv[r]) / B
            res = Z[:, 1] - y_wt[r]
            G[:, 1] = w_wt * _huber_grad(res, huber_delta,
                                         cen[r] if cen is not None else None) / B
            m.backward(Xb, G, E, S, Q)
            ll = -(y_lv[r] * np.log(p0 + 1e-9) + (1 - y_lv[r]) * np.log(1 - p0 + 1e-9))
            losses.append(float(ll.mean()))
        va = valid_score()
        hist.append({'epoch': ep, 'loss': round(float(np.mean(losses)), 4),
                     'valid_primary': round(va['primary'], 4),
                     'valid_GAUC': round(va['GAUC'], 4),
                     'valid_nDCG@5': round(va['nDCG@5'], 4),
                     'seconds': round(time.time() - t0, 1)})
        if verbose:
            print(f"  epoch {ep:2d} | loss {hist[-1]['loss']:.4f} | "
                  f"valid primary {va['primary']:.4f} | {hist[-1]['seconds']}s")
        if not np.isfinite(hist[-1]['loss']):
            hist[-1]['diverged'] = True
            break
        if va['primary'] > best + 1e-5:
            best, best_ep, bad = va['primary'], ep, 0
            best_state = m.state()
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        m.load(best_state)
    return m, {'epochs_run': len(hist), 'best_epoch': best_ep,
               'best_valid_primary': round(best, 5), 'history': hist,
               'diverged': bool(hist and hist[-1].get('diverged'))}
