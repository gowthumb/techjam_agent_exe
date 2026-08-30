"""ESMM-style multiplicative decomposition: P(long_view) = P(click) * P(lv | click).

GROUNDING. `KNOWLEDGE_BASE_PLAN.md` Phase 1c names an "ESMM-style multi-task
setup" explicitly; primer A.2 / A.3 describe the impression -> click -> deeper
engagement funnel and ESMM as the way to exploit it.

WHY THIS IS NOT THE PHASE 1C is_click RESULT. Phase 1C added `is_click` as a
CO-EQUAL 0.3-weighted auxiliary head on a shared table and found it HARMFUL -- the
dense signal competed with long_view for capacity (the "seesaw"). ESMM's
structure is different: the click head and the (implicit) conversion head are
composed MULTIPLICATIVELY, and the conversion head is supervised ONLY through the
long_view label, never directly. In this data a row with `is_click = 0` has
play_time ~ 0 and therefore long_view ~ 0 deterministically, so the funnel is
real and the factorisation is well posed.

MODEL. Shared V (dim, k); two logit heads z_ctr, z_cvr (each a linear term +
A_h-weighted FM interaction). p_ctr = sigma(z_ctr), p_cvr = sigma(z_cvr),
p_lv = p_ctr * p_cvr. Loss = BCE(p_lv, long_view) + w * BCE(p_ctr, is_click).
Ranking score is p_lv.

CONTROL (KB control_rule): `no_gate` -- the same two-head net scored by
sigma(z_cvr) alone (no multiplicative composition, click head still auxiliary).
Isolates the ESMM structure from "a second head".
"""
import time
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class ESMM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros((dim, 2), dtype=np.float32)          # 0 = ctr, 1 = cvr
        self.b = np.zeros(2, dtype=np.float32)
        self.A = np.ones((2, k), dtype=np.float32)
        self.lr, self.l2 = lr, l2
        self.m = {n: np.zeros_like(p) for n, p in (('V', self.V), ('W', self.W), ('A', self.A))}
        self.v = {n: np.zeros_like(p) for n, p in (('V', self.V), ('W', self.W), ('A', self.A))}
        self.t = 0

    def forward(self, X):
        E = self.V[X]
        S = E.sum(1)
        Q = 0.5 * (S ** 2 - (E ** 2).sum(1))
        Z = self.b + self.W[X].sum(1) + Q @ self.A.T           # (B,2)
        return Z, E, S, Q

    def backward(self, X, G, E, S, Q):
        gV = np.zeros_like(self.V); gW = np.zeros_like(self.W)
        np.add.at(gW, X, G[:, None, :])
        dQ = G @ self.A
        np.add.at(gV, X, dQ[:, None, :] * (S[:, None, :] - E))
        gA = G.T @ Q
        gV += self.l2 * self.V; gW += self.l2 * self.W
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for name, P, Gm in (('V', self.V, gV), ('W', self.W, gW), ('A', self.A, gA)):
            M, Vv = self.m[name], self.v[name]
            M *= b1; M += (1 - b1) * Gm
            Vv *= b2; Vv += (1 - b2) * (Gm * Gm)
            P -= self.lr * (M / (1 - b1 ** self.t)) / (np.sqrt(Vv / (1 - b2 ** self.t)) + eps)
        self.b -= self.lr * G.sum(0)

    def predict(self, X, mode='esmm', bs=200_000):
        out = []
        for i in range(0, len(X), bs):
            Z = self.forward(X[i:i + bs])[0]
            pc, pv = sigmoid(Z[:, 0]), sigmoid(Z[:, 1])
            out.append(pc * pv if mode == 'esmm' else pv)
        return np.concatenate(out)

    def state(self):
        return (self.V.copy(), self.W.copy(), self.b.copy(), self.A.copy())

    def load(self, st):
        self.V, self.W, self.b, self.A = [x.copy() for x in st]


def train(X_tr, y_lv, y_click, X_va, y_va, u_va, dim, w_click=1.0, mode='esmm',
          k=16, lr=0.001, l2=1e-6, epochs=40, bs=8192, patience=4, seed=0,
          evaluator=None, verbose=True):
    m = ESMM(dim, k=k, lr=lr, l2=l2, seed=seed)
    rng = np.random.default_rng(seed)
    y_lv = y_lv.astype(np.float32); y_click = y_click.astype(np.float32)
    best, best_state, best_ep, bad = -1.0, None, 0, 0
    hist = []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        idx = rng.permutation(len(y_lv))
        losses = []
        for i in range(0, len(idx), bs):
            r = idx[i:i + bs]
            Xb = X_tr[r]
            Z, E, S, Q = m.forward(Xb)
            pc, pv = sigmoid(Z[:, 0]), sigmoid(Z[:, 1])
            plv = np.clip(pc * pv, 1e-7, 1 - 1e-7)
            B = len(r)
            G = np.empty((B, 2), np.float32)
            # d BCE(plv, y_lv) / d z_ctr and / d z_cvr, via plv = pc*pv
            common = (plv - y_lv[r]) / (1.0 - plv)
            G[:, 0] = (common * (1.0 - pc) + w_click * (pc - y_click[r])) / B
            G[:, 1] = (common * (1.0 - pv)) / B
            m.backward(Xb, G, E, S, Q)
            losses.append(float(np.mean(-(y_lv[r] * np.log(plv) + (1 - y_lv[r]) * np.log(1 - plv)))))
        s_va = m.predict(X_va, mode=mode)
        va = evaluator(u_va, y_va, s_va)
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
