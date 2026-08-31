# Scale directives — condensed, for per-iteration Planner context

Lossy summary of `HARDWARE_AWARENESS.md` (hardware/ops rules) and
`ONEK_RESULTS.md` (the 1K/27K research trail), written to be injected into
every Planner call at a fraction of either doc's size. Read the full docs for
anything this compresses away — this file exists because doing that on every
iteration was ~15,000 tokens of fixed overhead for no benefit past the first
read.

## Hardware rules (non-negotiable on 1K/27K)
1. **Sparse Adam only.** Dense Adam is O(vocab) per batch — infeasible past
   ~500K encoder dim (1K's is ~2.9M, 27K's ~20.3M). Never propose dense-Adam
   code here; it isn't a modeling choice, the hardware rules it out.
2. **No wide per-row feature matrix** (a GBDT-style dense array, one float32
   column per feature). Calculated at ~22GB against a 23.7GB machine at 27K
   scale; never attempted. Stay inside the 5-field int-encoded representation.
3. **CPU only** — no GPU FM implementation exists in this codebase.

## 1K's confirmed baseline, and what has already failed against it
Baseline: pointwise loss, k=16, lr=1e-3, sparse Adam. valid 0.6439 ± 0.0022
(3 seeds), test 0.6380 ± 0.0021. 1K is item cold-start (85% of test videos
unseen in train) — no trained embedding to rank *from*, little within-user
signal to normalize against.

**Already tested, negative — do not re-propose without new evidence:**
- BPR (loss): −0.0151 valid — every within-user-normalizing loss inverts here
- SSM / sampled softmax: −0.0149 valid — same reason as BPR
- (user,author) / (user,tab) / (user,dur) affinity fields: all negative
  (−0.0081 / −0.0017 / −0.0077 valid)
- Embedding noise (σ 0.05/0.1/0.2): flat to slightly negative
- lr/k sweeps around the baseline: flat (k=16 stays optimal)
- xgboost `rank:pairwise`: −0.0037 valid
- xgboost `rank:ndcg`: looked +0.0024 on 1 seed, −0.0010 on 3-seed mean — noise

**Genuinely untested — prefer these over re-testing the above:**
- CatBoost YetiRank with the `cat_features` bug fixed (prior attempt had a
  real bug — ID columns quantized as continuous floats — and ran effectively
  single-threaded; killed before completion, never disproven). `catboost` is
  now installed in this environment (see requirements.txt) — a run that fails
  with `ModuleNotFoundError` is a stale assumption, not this benchmark's
  actual state; a genuine attempt should now train and score.
- Content/side-information features that don't require a trained ID embedding
  (duration, category/tag, author-level aggregates) — structurally motivated:
  an FM embedding has nothing to say about a video never seen in training;
  numeric content features are defined regardless

## Decision protocol
- A single-seed accept during search is a cheap screen, not a trusted result.
  Two prior single-seed leads evaporated on 3-seed replication (an lr sweep,
  an xgboost objective) — `scripts/maximize_1k.py` replicates any accept over
  3 seeds before reporting it as real.
- 27K is currently out of scope (incomplete local data archive); do not
  propose 27K-targeted work.
