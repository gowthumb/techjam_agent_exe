"""KuaiRand-27K baseline -- sparse-Adam FM, transferred from 1K's confirmed winner.

Same contract, same math, same reasoning as baseline_1k.py (see that file's
docstring for the full rationale) -- only data_27k is swapped in for data_1k.
Two 27K-specific facts worth knowing before changing anything here
(knowledge_base/ONEK_RESULTS.md Phase 20, single-seed, not yet replicated):

  - load+encode alone measured ~2063s (~34min) the first time it runs uncached;
    agent/data_cache.py's npz cache avoids paying this more than once per
    FIELDS set.
  - the confirmed config peaked at epoch 1 and degraded every epoch after
    (0.6687 -> 0.6626 -> 0.6501 -> ... by epoch 5) -- 27K overfits faster than
    1K did. A hypothesis that increases epochs or lowers patience without a
    stated reason is fighting this measured shape, not building on it.

This file is the ONE config knowledge_base/ONEK_RESULTS.md Phase 20 actually
ran on 27K. Per that phase's own framing, treat any single run of this file as
one data point, not a tuned or confidence-intervaled result -- 27K's own seed
noise has never been measured (1K's was already 4x Pure's). Promote a candidate
here only after it has cleared the replication gate on 1K first (this is what
scripts/run_agent_scaled.py automates), not on the strength of a single 27K run.
"""
import argparse
import time

import numpy as np

from data_27k import FIELDS, encode, load
from evaluate import evaluate


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
    """Same math, optimizer, and init as baseline.py::FM, plus a sparse Adam step.

    See baseline_1k.py::FM for the full rationale -- identical implementation,
    duplicated rather than imported so a Coder-authored patch to this file stays
    self-contained and search/replace-able, exactly like baseline.py itself.
    """

    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]                                    # (B,F,k)
        S = E.sum(1)                                      # (B,k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def step(self, X, y):
        """One sparse-Adam step -- touches only the embedding rows in this batch."""
        B = len(y)
        z, E, S = self.logits(X)
        g = ((sigmoid(z) - y) / B).astype(np.float32)
        n_fields = X.shape[1]
        flat = X.ravel()
        uniq, inv = np.unique(flat, return_inverse=True)
        contrib = (g[:, None, None] * (S[:, None, :] - E)).reshape(-1, self.V.shape[1])
        gV = np.zeros((len(uniq), self.V.shape[1]), dtype=np.float32)
        np.add.at(gV, inv, contrib)
        gW = np.bincount(inv, weights=np.repeat(g, n_fields), minlength=len(uniq)).astype(np.float32)
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
        self.b -= self.lr * g.sum()
        return float(-np.mean(y * np.log(sigmoid(z) + 1e-9) + (1 - y) * np.log(1 - sigmoid(z) + 1e-9)))

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])


def run_fm(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0,
           verbose=True, return_predictions=False):
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]
    Xte, yte, ute = enc["test"]
    m = FM(dim, k=k, lr=lr, seed=seed)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr))
        t0 = time.time()
        losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]]) for i in range(0, len(idx), bs)]
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time() - t0:.1f}s")
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    test_scores = m.predict(Xte)
    result = {"valid": evaluate(uva, yva, m.predict(Xva)),
              "test": evaluate(ute, yte, test_scores)}
    if return_predictions:
        result["test_scores"] = test_scores
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print("loading KuaiRand-27K (this is the ~34min step the first time it runs uncached) ...")
    splits = load(None)
    print(f"fields={FIELDS}")
    res = run_fm(splits, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed)
    print(f"\n=== fm (27K, seed={a.seed}) ===")
    for sp in ("valid", "test"):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
