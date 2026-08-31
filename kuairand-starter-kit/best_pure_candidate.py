"""KuaiRand-Pure — winning candidate from agent run 5cb9e936da024858815cd932df6ecfb7.

Provenance (see runs/5cb9e936da024858815cd932df6ecfb7/{iterations.jsonl,summary.json}):
  iteration 3 of 6, status "accepted", stopping_reason "converged".
  valid: GAUC 0.6699 | nDCG@5 0.5372 | primary 0.6035  (baseline: 0.6016)
  test:  GAUC 0.6635 | nDCG@5 0.5293 | primary 0.5964  (baseline: 0.5946, +0.0018)

Mechanism actually implemented (verified against the logged code_diff): plain
pointwise FM (same forward pass, optimizer, and init as baseline.py::FM),
reweighting the BCE loss/gradient by pos_weight = n_negative / n_positive on
the train split -- i.e. prevalence-balanced pointwise BCE. No sequence model,
no behavior history.

CAVEAT, worth reading before trusting this as a final submission: the
Planner's logged hypothesis for this iteration ("replicate the leakage-safe
DIN variant using positive-only behavior sequences, 3+ paired seeds, a
no-sequence control, and a rand_valid exposure-policy veto") does NOT match
what the Coder actually patched -- no DIN, no sequence, no 3-seed replication,
no rand_valid check ever ran. It happens to be the same prevalence-balancing
mechanism a *later* iteration (4) proposed correctly and by name, which was
then rejected for not clearing the acceptance band against this already-best
candidate. So: the mechanism is legitimate and plausible (class imbalance
reweighting is a standard, well-understood lever), but the specific validation
story attached to it in the run log is not real, and the gain (+0.0018 test,
+0.0019 valid) is a single seed clearing this codebase's own noise band
(~0.0016) by a thin margin -- exactly the shape of result
knowledge_base/knowledge_base_rationale.md and ONEK_RESULTS.md both warn has
evaporated on replication before. Treat this as "a plausible single-seed lead
that produces a legal, scoring submission," not as a KB-confirmed win, unless
it's replicated over a few more seeds first.

Original starter-kit header follows, unmodified:
---
KuaiRand-Pure baselines.
  --model pop   : item popularity (official baseline, no training)
  --model fm    : Factorization Machine (starting model)
  --model random: random scoring (lower bound, sanity check for the scorer)
numpy only. Usage: see README.md
"""
import argparse, collections, time
import numpy as np
from checkpoint import save as save_checkpoint
from data import load, encode, FIELDS
from evaluate import evaluate

def sigmoid(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# ---------------- item popularity (official baseline) ----------------
def run_pop(splits, prior=20.0):
    pos, imp = collections.Counter(), collections.Counter()
    for x in splits['train']:
        imp[x[2]] += 1; pos[x[2]] += x[6]
    gmean = sum(pos.values()) / sum(imp.values())
    score = lambda v: (pos[v] + prior * gmean) / (imp[v] + prior) if imp[v] else gmean
    out = {}
    for name in ('valid', 'test'):
        rws = splits[name]
        out[name] = evaluate([x[1] for x in rws], [x[6] for x in rws],
                             [score(x[2]) for x in rws])
    return out

def run_random(splits, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for name in ('valid', 'test'):
        rws = splits[name]
        out[name] = evaluate([x[1] for x in rws], [x[6] for x in rws],
                             rng.random(len(rws)))
    return out

# ---------------- Factorization Machine ----------------
class FM:
    def __init__(self, dim, k=16, lr=0.001, l2=1e-6, seed=0, pos_weight=1.0):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.lr, self.l2 = lr, l2
        self.pos_weight = np.float32(pos_weight)
        self.mV = np.zeros_like(self.V); self.vV = np.zeros_like(self.V)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.t = 0

    def logits(self, X):
        E = self.V[X]                                   # (B,F,k)
        S = E.sum(1)                                    # (B,k)
        inter = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
        return self.b + self.W[X].sum(1) + inter, E, S

    def step(self, X, y):
        B = len(y)
        z, E, S = self.logits(X)
        p = sigmoid(z)
        w = (self.pos_weight * y + (1.0 - y)).astype(np.float32)   # per-sample weight
        g = ((p * w - self.pos_weight * y) / B).astype(np.float32)  # (B,) weighted-BCE grad
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
        self.b -= self.lr * g.sum()
        return float(-np.mean(w * (y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))))

    def predict(self, X, bs=200_000):
        return np.concatenate([self.logits(X[i:i + bs])[0] for i in range(0, len(X), bs)])

def run_fm(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True, return_predictions=False, checkpoint_path=None):
    enc, dim = encode(splits)
    Xtr, ytr, _ = enc['train']; Xva, yva, uva = enc['valid']; Xte, yte, ute = enc['test']
    npos = float(np.sum(ytr)); nneg = float(len(ytr) - npos)
    pos_weight = (nneg / npos) if npos > 0 else 1.0
    m = FM(dim, k=k, lr=lr, seed=seed, pos_weight=pos_weight)
    rng = np.random.default_rng(seed)
    best, best_state, bad = -1, None, 0
    for ep in range(1, epochs + 1):
        idx = rng.permutation(len(ytr)); t0 = time.time()
        losses = [m.step(Xtr[idx[i:i + bs]], ytr[idx[i:i + bs]]) for i in range(0, len(idx), bs)]
        va = evaluate(uva, yva, m.predict(Xva))
        if verbose:
            print(f"  epoch {ep:2d} | loss {np.mean(losses):.4f} | valid GAUC {va['GAUC']:.4f} "
                  f"nDCG@5 {va['nDCG@5']:.4f} primary {va['primary']:.4f} | {time.time()-t0:.1f}s")
        if va['primary'] > best + 1e-5:
            best, bad = va['primary'], 0
            best_state = (m.V.copy(), m.W.copy(), np.float32(m.b))
        else:
            bad += 1
            if bad >= patience:
                if verbose: print(f"  early stop at epoch {ep}")
                break
    m.V, m.W, m.b = best_state
    if checkpoint_path:
        save_checkpoint(checkpoint_path, m.V, m.W, m.b)
    test_scores = m.predict(Xte)
    result = {'valid': evaluate(uva, yva, m.predict(Xva)),
              'test': evaluate(ute, yte, test_scores)}
    if return_predictions:
        result['test_scores'] = test_scores
    return result

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='./KuaiRand-Pure/data',
                    help='KuaiRand-Pure extracted data directory')
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--checkpoint', default=None, help='Path to save trained V/W/b weights (.npz) via checkpoint.save().')
    a = ap.parse_args()
    print(f"loading {a.data_dir} ...")
    splits = load(a.data_dir)
    print({k_: len(v) for k_, v in splits.items()}, f"fields={FIELDS}")
    res = run_fm(splits, k=a.k, lr=a.lr, epochs=a.epochs, seed=a.seed, checkpoint_path=a.checkpoint)
    if a.checkpoint:
        print(f"saved checkpoint to {a.checkpoint}")
    print(f"\n=== fm-pos_weight (seed={a.seed}) ===")
    for sp in ('valid', 'test'):
        r = res[sp]
        print(f"  {sp:5s}  GAUC {r['GAUC']:.4f} | nDCG@5 {r['nDCG@5']:.4f} | primary {r['primary']:.4f}")
