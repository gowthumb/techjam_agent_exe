"""Multi-task FM with a shared embedding table (shared-bottom).

KuaiRand logs 12 feedback signals; only long_view is scored. The others are free
supervision -- if they share structure with long_view they regularize the shared
embeddings, and if they fight it they produce the "seesaw" the primer warns about.

Construction (chosen so the single-task case is not a new model):
    E = V[X]                     (B,F,k)   shared embeddings
    S = E.sum(1)                 (B,k)
    Q = 0.5*(S^2 - (E^2).sum(1)) (B,k)     per-dimension FM interaction
    Z = b_t + W_t[X].sum(1) + Q @ A_t      one logit per task

With T=1 and A fixed at ones, Z is *exactly* the starter kit's FM logit, so any
difference measured against the baseline is attributable to the auxiliary tasks
rather than to a changed model. A_t (per-task weights over the k interaction
dimensions) is what lets tasks diverge while sharing V.

Only task 0 (long_view) is ever predicted; auxiliaries exist to shape V.
"""
import time
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class MTFM:
    def __init__(self, dim, n_tasks, k=16, lr=0.001, l2=1e-6, seed=0, learn_A=True):
        rng = np.random.default_rng(seed)
        self.T = n_tasks
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros((dim, n_tasks), dtype=np.float32)
        self.b = np.zeros(n_tasks, dtype=np.float32)
        self.A = np.ones((n_tasks, k), dtype=np.float32)
        self.lr, self.l2, self.learn_A = lr, l2, learn_A
        self.m = {n: np.zeros_like(p) for n, p in
                  (('V', self.V), ('W', self.W), ('A', self.A))}
        self.v = {n: np.zeros_like(p) for n, p in
                  (('V', self.V), ('W', self.W), ('A', self.A))}
        self.t = 0

    def forward(self, X):
        E = self.V[X]
        S = E.sum(1)
        Q = 0.5 * (S ** 2 - (E ** 2).sum(1))            # (B,k)
        Z = self.b + self.W[X].sum(1) + Q @ self.A.T    # (B,T)
        return Z, E, S, Q

    def backward(self, X, G, E, S, Q):
        """G: (B,T) dL/dZ."""
        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        np.add.at(gW, X, G[:, None, :])
        dQ = G @ self.A                                  # (B,k)
        np.add.at(gV, X, dQ[:, None, :] * (S[:, None, :] - E))
        gA = G.T @ Q                                     # (T,k)
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

    def predict(self, X, task=0, bs=200_000):
        return np.concatenate([self.forward(X[i:i + bs])[0][:, task]
                               for i in range(0, len(X), bs)])

    def state(self):
        return (self.V.copy(), self.W.copy(), self.b.copy(), self.A.copy())

    def load(self, st):
        self.V, self.W, self.b, self.A = [x.copy() for x in st]


def train(X_tr, Y_tr, X_va, y_va, u_va, dim, task_weights, k=16, lr=0.001, l2=1e-6,
          epochs=40, bs=8192, patience=4, seed=0, evaluator=None, verbose=True,
          learn_A=True):
    """Y_tr: (N,T) float32, column 0 = long_view. task_weights: (T,) floats."""
    T = Y_tr.shape[1]
    w = np.asarray(task_weights, dtype=np.float32)
    assert len(w) == T and w[0] > 0, 'task 0 (long_view) must be weighted'
    m = MTFM(dim, T, k=k, lr=lr, l2=l2, seed=seed, learn_A=learn_A)
    rng = np.random.default_rng(seed)
    best, best_state, best_ep, bad = -1.0, None, 0, 0
    hist = []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        idx = rng.permutation(len(Y_tr))
        losses = []
        for i in range(0, len(idx), bs):
            r = idx[i:i + bs]
            Xb, Yb = X_tr[r], Y_tr[r]
            Z, E, S, Q = m.forward(Xb)
            P = sigmoid(Z)
            G = (w * (P - Yb) / len(r)).astype(np.float32)
            m.backward(Xb, G, E, S, Q)
            ll = -(Yb * np.log(P + 1e-9) + (1 - Yb) * np.log(1 - P + 1e-9))
            losses.append(float((ll.mean(0) * w).sum()))
        L = float(np.mean(losses))
        va = evaluator(u_va, y_va, m.predict(X_va))
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
            hist[-1]['diverged'] = True
            break
        if va['primary'] > best + 1e-5:
            best, best_ep, bad = va['primary'], ep, 0
            best_state = m.state()
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f'  early stop at epoch {ep}')
                break
    if best_state is not None:
        m.load(best_state)
    return m, {'epochs_run': len(hist), 'best_epoch': best_ep,
               'best_valid_primary': round(best, 5), 'history': hist,
               'diverged': bool(hist and hist[-1].get('diverged'))}
