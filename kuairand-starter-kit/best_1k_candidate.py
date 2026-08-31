"""KuaiRand-1K -- winning candidate from agent run 0f6a4083fba54b798c7bb6f87c0a73a9.

Provenance (see runs/0f6a4083fba54b798c7bb6f87c0a73a9/{iterations.jsonl,
maximize_1k_report.json}, produced by scripts/maximize_1k.py):
  iteration 5 of 8 scored, status "accepted", stopping_reason "converged".
  3-seed REPLICATED (this is the one 1K result in this codebase's history that
  cleared that gate): valid primary 0.64565 / 0.64553 / 0.64767, mean 0.64628
  vs. baseline 0.64385 -- CONFIRMED under the (then) 0.0016 acceptance band.
  final test (seed 0): GAUC 0.6736 | nDCG@5 0.6062 | primary 0.6399
  (baseline test: GAUC ~0.6674 | nDCG@5 ~0.5936 | primary ~0.6305, Phase 5
  single-seed reference -- ONEK_RESULTS.md's 3-seed baseline is 0.6380±0.0021)

Mechanism: swaps sparse Adam for sparse Adagrad (lr=0.03) as the FM's
optimizer -- same forward pass, same loss, same init, same five fields. This
is the one config in ~13 tested axes across this codebase's entire 1K research
history (ONEK_RESULTS.md Phases 5/10/18/19, plus this run) that beat the
untuned pointwise-sparse-Adam baseline and survived 3-seed replication.

NOTE on the acceptance band: this candidate was accepted under the
Pure-inherited 0.0016 band (agent/executor.py::_ACCEPTANCE_BAND). That band
was briefly widened to 0.032 for 1K and then reset back to 0.0016 after a
follow-up run showed 0.032 was calibrated to a magnitude this benchmark has
never produced -- this candidate's own +0.0018 single-run delta (mean +0.0024
over 3 seeds) is itself smaller than that wider band would have allowed. It
remains the one genuinely-replicated 1K win in this codebase's history, and
under the current (reset) band it is exactly the kind of result the band is
calibrated to accept -- see ONEK_RESULTS.md's acceptance-band history for the
full account.
"""
import argparse
import time

import numpy as np

from checkpoint import save as save_checkpoint
from data_1k import FIELDS, encode, load
from evaluate import evaluate


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class FM:
    """Same math and init as baseline_1k.py::FM, with a selectable sparse
    optimizer -- Adagrad (this candidate's mechanism) or Adam (the baseline,
    kept for the paired-control comparison the accepted hypothesis specified).
    """

    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0,
                 optimizer="adagrad", eps=1e-10):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.optimizer = optimizer
        self.eps = eps
        # mV/mW: Adam first moment (unused by Adagrad).
        # vV/vW: Adam second moment OR Adagrad cumulative sum-of-squares.
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
        """One sparse optimizer step -- touches only the embedding rows in this batch."""
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
        if self.optimizer == "adagrad":
            # Sparse Adagrad: accumulate coordinate-wise squared grads only on
            # touched rows; frequent shared IDs shrink their step, rare IDs keep
            # larger effective steps. No moment/L2 decay on untouched rows.
            aV = self.vV[uniq] + gV * gV
            self.vV[uniq] = aV
            self.V[uniq] -= self.lr * gV / (np.sqrt(aV) + self.eps)
            aW = self.vW[uniq] + gW * gW
            self.vW[uniq] = aW
            self.W[uniq] -= self.lr * gW / (np.sqrt(aW) + self.eps)
            self.b -= self.lr * g.sum()
        else:
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


def run_fm(splits, k=16, lr=0.03, epochs=40, bs=8192, patience=4, seed=0,
           verbose=True, return_predictions=False, optimizer="adagrad", eps=1e-10,
           checkpoint_path=None):
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc["train"]
    Xva, yva, uva = enc["valid"]
    Xte, yte, ute = enc["test"]
    m = FM(dim, k=k, lr=lr, seed=seed, optimizer=optimizer, eps=eps)
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
    if checkpoint_path:
        save_checkpoint(checkpoint_path, m.V, m.W, m.b)
    test_scores = m.predict(Xte)
    result = {"valid": evaluate(uva, yva, m.predict(Xva)),
              "test": evaluate(ute, yte, test_scores)}
    if return_predictions:
        result["test_scores"] = test_scores
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optimizer", type=str, default="adagrad", choices=["adagrad", "adam"])
    ap.add_argument("--checkpoint", default=None, help="Path to save trained V/W/b weights (.npz) via checkpoint.save().")
    a = ap.parse_args()
    print("loading KuaiRand-1K ...")
    splits = load(None)
    print(f"fields={FIELDS}")
    res = run_fm(splits, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed,
                 optimizer=a.optimizer, checkpoint_path=a.checkpoint)
    if a.checkpoint:
        print(f"saved checkpoint to {a.checkpoint}")
    print(f"\n=== fm-sparse-adagrad (1K, seed={a.seed}) ===")
    for sp in ("valid", "test"):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
