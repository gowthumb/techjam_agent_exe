# ML Modelling: Architectures & Hypotheses

Comprehensive inventory of all architectures, loss functions, and hypotheses tested in the `ml_modelling` folder across 20+ experimental phases.

**Generated:** 2026-09-01

---

## Table of Contents

1. [Phase 1: Loss Functions & Foundations](#phase-1-loss-functions--foundations)
2. [Phase 7: Architecture Ladder](#phase-7-architecture-ladder)
3. [Phases 11-16: Advanced Models](#phases-11-16-advanced-models)
4. [Phases 18-20: Scale Transfer](#phases-18-20-scale-transfer)
5. [Supporting Library Modules](#supporting-library-modules)
6. [Benchmarks & Data Scales](#benchmarks--data-scales)
7. [Key Findings Summary](#key-findings-summary)

---

## Phase 1: Loss Functions & Foundations

### Phase 1A: Loss Function Alignment

**Model:** Factorization Machine (FM)  
**Benchmark:** KuaiRand-Pure  
**Status:** ✓ Confirmed  

**Hypothesis:**
The baseline FM optimizes pointwise logloss, but the metric (GAUC/nDCG@5) is a *within-user ranking* metric. A loss that only constrains within-user order should extract more signal from the same model, capacity, and features.

**Loss Functions Tested:**

| Loss | Description |
|------|-------------|
| **Pointwise** | Sigmoid cross-entropy per row (baseline) |
| **BPR** | Within-user pairwise: -log sigmoid(z_pos - z_neg) |
| **Listwise** | Within-user softmax CE (retired as dead) |
| **Hybrid** | Pointwise + λ * Listwise |

**Key Finding:** BPR consistently outperforms pointwise on Pure; trend reverses on cold-start regimes (1K, 27K).

---

### Phase 1B: Embedding Noise & Regularization

**Focus:** Regularization through additive noise

**Hypothesis:**
Gaussian noise added to embeddings during training regularizes without biasing the gradient (dE'/dV = I). The gradient keeps its exact form, so apply_grad needs no change.

**Implementation:**
- Noise magnitude: configurable (e.g., 0.1)
- Applied only during training
- Noise-free inference

---

### Phase 1C: Multi-Task Supervision

**Auxiliary Signals Tested:** 11 binary + continuous signals

**Hypothesis:**
Auxiliary signals sharing structure with `long_view` should regularize the shared embedding table. Tested if signals like `is_click`, `is_like`, `is_follow`, etc. help or hurt.

**Auxiliary Signals:**
- Binary: is_click, is_like, is_follow, is_comment, is_forward, is_hate, is_profile_enter
- Derived: play_complete, play_half (watch-time based)
- Controls: random_sparse, random_dense (random labels at matched sparsity)

**Key Control:**
Random labels at matched sparsity found that auxiliary heads act as regularizers only—no information transfer. The "seesaw" problem: dense auxiliary signals compete for shared capacity.

**Key Finding:** Auxiliary binary signals do NOT help. Pure's Phase 1C conclusion: auxiliary heads are regularizers, not transfer learning.

---

### Phase 1D: Capacity Sweep

**Parameter:** Embedding dimension k

**Values Tested:** 2, 4, 6, 8, 16, 32, 64, 128

**Key Finding:** Performance is **flat across the entire range**. Capacity (k) is NOT a live knob on Pure.

---

### Phase 1F: Gradient-Boosted Trees (GBDT)

**Algorithms:** LightGBM (lambdarank), XGBoost, CatBoost

**Hypothesis:**
Trees consume numeric features (affinity rates, durations, statistics) that categorical-only FM cannot represent. LightGBM's `lambdarank` objective optimizes within-group ranking directly.

**Feature Blocks:**

| Block | Description | Caveat |
|-------|-------------|--------|
| **base** | 5 baseline fields (categorical) + duration/hour (numeric) | — |
| **aff** | Causal affinity rates + evidence counts | Train-labels-only |
| **user** | User-side profile columns | Tested, found null |
| **vstat** | Video feature statistics (aggregates) | **Leakage caveat**: computed over whole dataset period |

**Key Finding:** GBDT never beats FM on Pure. Trees gain over embeddings only in cold-start regimes (1K, 27K), where numeric affinity rates are defined even for unseen items.

---

### Phase 1G: Temporal Drift

**Problem Statement:**
- 59% of training rows land in just 3 days (2022-04-10..12)
- Evaluation runs at ~15K rows/day over longer period
- Base rate falls from 0.3366 (train) to 0.3134 (valid/test)

**Hypothesis:**
Training is front-loaded on high-engagement traffic; downweighting stale rows should align the fitted distribution with the evaluated one.

**Interventions Tested:**

1. **Recency Weighting:** `exp(-(t_end - t_i)/τ)`
   - Half-life τ in {2, 4, 7, 14} days
   
2. **Recent Window:** Train only on last N days
   - Days N in {3, 5, 7, 10}

**Key Finding:** Recency weighting helps moderately; discarding old rows entirely helps more, suggesting volume matters more than recency for this regime.

---

### Phase 1I: User Behavior Sequences (DIN-style Attention)

**Framework:** PyTorch  
**Problem:** Kit's features are entirely flat; no encoding of what the user did before the impression

**Hypothesis:**
User's historical behavior (last ~20 videos shown) provides critical signal. Attention mechanism over exposure history allows the model to weight recent actions.

**Sequence Types:**

| Type | Description | Leakage |
|------|-------------|---------|
| **Exposure** | Last 20 videos shown (label-free) | None; safe on all splits |
| **Positive** | Last 20 long-viewed videos | Built from TRAIN labels only |

**Architecture:** DIN (Deep Interest Network)
- Embedding layers for user, video, author, tab, duration
- Attention mechanism over history, pooled with candidate
- MLP heads for ranking

**Status:** Confirmed to help on Pure; not tested at scale (1K/27K).

---

## Phase 7: Architecture Ladder

**Benchmark:** KuaiRand-Pure  
**Framework:** PyTorch (CPU only)  
**Constraint:** All architectures trained under BOTH pointwise and BPR to isolate architecture from loss choice

**Hypothesis:**
Architectures should be compared under the loss that actually wins (BPR), not confounded with loss selection. Every architecture re-evaluated under both losses.

### Architectures Tested

| Architecture | Description | Parameters |
|---|---|---|
| **FM** | Second-order interactions only | Baseline; same as numpy version |
| **FFM** | Field-aware FM | Separate embedding per (field, interacting field) pair |
| **DeepFM** | FM second-order + MLP | MLP over concatenated embeddings |
| **AutoInt** | Self-attention | Multi-head attention over field embeddings |

**Sequence Option:**
- Mean-pooled embedding of last 20 long-viewed videos (train-labels-only)
- Plain pooling (no attention)
- Tested to check if simpler pooling works with better loss

**Key Finding:** Simple FM competes with all fancier architectures. No architecture win justifies the added complexity.

---

## Phases 11-16: Advanced Models

### Phase 11: Unbiased Evaluation via Random Exposure Log

**Benchmark:** KuaiRand-Pure  

**Hypothesis:**
Valid split is exposure-biased (production recommender chose what to show). The KB documents "blind spots" where valid and test disagree (e.g., BPR: +0.0021 on valid but +0.0032 on test). Would an unbiased split (randomly-exposed videos) have caught these?

**Evaluation Splits:**
- `valid`: Exposure-biased (production choice)
- `test`: Holdout test period
- `rand_valid`: Randomly-exposed videos (unbiased eval)
- `rand_test`: Randomly-exposed test period (report-only)

**Key Configs Scored on All Four Splits:**
```
11-pointwise-k16
11-pointwise-k2
11-pointwise-k16-noise0.1
11-bpr-k6
11-bpr-k16
11-bpr-k16-noise0.1
11-listwise-lr0.001
```

**Key Finding:** Rank-correlation of rand_valid to test is strong; proposed as tie-breaker for decisions inside the 0.0016 noise band.

---

### Phase 12: Sampled Softmax (SSM / InfoNCE)

**Benchmark:** KuaiRand-Pure  

**Hypothesis:**
Listwise (softmax over whole impression list with uniform-over-positives target) is a poor fit when ~1/3 of impressions are positive. SSM contrasts ONE positive against a few sampled negatives from the SAME user—standard implicit-feedback ranking loss, smooth surrogate for top-k order.

**Parameters:**
- Negatives per positive: {4, 8, 16}
- Learning rate: swept (ranking losses need smaller lr than logloss)
- Temperature: softmax temperature parameter tuned
- Embedding dimension k: {8, 16}

**Stages:**
1. `--stage grid`: lr × temp at n=4, k=8
2. `--stage neg`: neg_per_pos in {8, 16}, lr/2, k=16 check
3. `--stage control`: ssm_global (negatives from all rows), bpr ppp in {2,4}
4. `--stage replicate`: best SSM + BPR reference, 5 seeds each, k=16

**Controls:**
- `ssm_global`: Negatives from all rows (not same user) → tests if within-user structure matters
- `bpr pairs_per_pos`: Existing BPR with more negatives → tests if edge is softmax, not just more negatives

**Key Finding:** SSM is a BPR-peer on Pure; Phase 18 predicted to invert on 1K cold-start.

---

### Phase 13: Watch-Time Regression (WTFM - Two-Head Shared-Embedding FM)

**Architecture Type:** Multi-task with specialized heads  

**Hypothesis:**
`long_view` is a deterministic threshold on watch_time: `1` iff `play_time_ms >= min(duration_ms, 18000)` (verified to match 97.9% of rows). A regression head on the continuous watch ratio, sharing the embedding table, trains on strictly more information than the binary head and gets gradient from every row.

**Model Structure:**
- **Shared:** Embeddings V (dim, k)
- **Head 0:** Binary long_view (BCE)
- **Head 1:** Continuous watch ratio (regression)
- **Censoring (CWM):** One-sided (Huber) loss for completed plays (right-censored)
  - For `play_time_ms >= duration_ms`: penalizes predicting LESS than observed ratio, not more

**Model Equation:**
```
E = V[X]                              (B,F,k)
S = E.sum(1)                          (B,k)
Q = 0.5*(S^2 - (E^2).sum(1))          (B,k)
Z_0 = b_0 + W_0[X].sum(1) + Q @ A_0   (B,)     long_view logit
Z_1 = b_1 + W_1[X].sum(1) + Q @ A_1   (B,)     watch ratio logit
```

**Loss:**
```
L = BCE(σ(Z_0), long_view) + w_wt * Huber_censored(Z_1, watch_ratio)
```

**Ranking Score:** Configurable (head 0, head 1, or rank-blend selected on valid)

**Controls:**
- Two-sided Huber: tests if censoring matters
- Random-continuous target: tests if head is just regularization
- w_wt = 0: single-task (must land on FM baseline)

**Stages:**
- `--stage main`: Sweep w_wt, compare heads
- `--stage control`: Run controls
- `--stage replicate`: Best config, 3 seeds

**Key Finding:** Watch-time regression shows promise but needs 1K validation to confirm transfer.

---

### Phase 14: Duration Regime & Video Freshness Features

**Benchmark:** KuaiRand-Pure (on BPR config)  

**Hypothesis:**
`long_view` bifurcates at duration 18 seconds. Video freshness varies within a user's impression list. A regime field and age bucket are interactions the baseline five fields cannot represent.

**New Features:**

| Feature | Description |
|---------|-------------|
| **Duration Regime** | Binary: ≤18s vs >18s. Represents discontinuity in prediction problem (complete-play vs watch≥18s) |
| **Video Age Bucket** | Days between upload and impression. Fresh videos behave differently (novelty, trending). Varies within user's list. |
| **Finer Duration Bucketing** | 20-way instead of 10-way grid |

**Mechanism:**
The FM will implicitly cross regime field with user_id/author_id, creating a new interaction the flat baseline cannot represent.

**Controls:**
- Duration regime with shuffled values: tests if gain is capacity/regularization only, not regime information

**Stages:**
- `--stage main`: Sweep feature combinations
- `--stage control`: Shuffled regime control
- `--stage replicate`: Best config, 3 seeds

**Key Finding:** Duration regime gains ~0.0003; modest but real. Video age less clear.

---

### Phase 15: ESMM Multiplicative Decomposition

**Architecture Type:** Two-head shared-embedding FM  

**Hypothesis:**
P(long_view) = P(click) × P(long_view|click). A row with `is_click = 0` has play_time ~ 0 and therefore `long_view ~ 0` deterministically, so the funnel is real and the factorization is well-posed. Click head is supervised directly; the conversion head only through the long_view label (ESMM's trick—avoids sample-selection bias).

**Why This Differs from Phase 1C:**
Phase 1C added `is_click` as a CO-EQUAL 0.3-weighted auxiliary head and found it HARMFUL (seesaw effect). ESMM structures heads MULTIPLICATIVELY and never supervises the conversion head directly—a different mechanism entirely.

**Model:**
```
E = V[X]                          (B,F,k)
S = E.sum(1)                      (B,k)
Q = 0.5*(S^2 - (E^2).sum(1))      (B,k)
Z_ctr = b_ctr + W_ctr[X] + Q @ A_ctr
Z_cvr = b_cvr + W_cvr[X] + Q @ A_cvr
P_ctr = σ(Z_ctr)
P_cvr = σ(Z_cvr)
P_lv = P_ctr × P_cvr
```

**Loss:**
```
L = BCE(P_lv, long_view) + w * BCE(P_ctr, is_click)
```

**Ranking Score:** P_lv (multiplicative product)

**Controls:**
- `no_gate`: Same two-head net scored by σ(Z_cvr) alone (no multiplicative composition)
  - Isolates the ESMM structure from "having a second head"

**Stages:**
- `--stage main`: Main sweep
- `--stage replicate`: Best config, 3 seeds

**Note:** Inherently pointwise on composed probability; honest comparison is ESMM vs pointwise FM baseline, separately vs BPR.

**Key Finding:** ESMM structure shows promise; not yet validated at scale.

---

### Phase 16: Diverse Ensemble (Objective-Based)

**Benchmark:** KuaiRand-Pure  

**Hypothesis:**
Ensemble across DIFFERENT training objectives (pointwise/BPR/SSM/watch-time/ESMM), not seeds. Phase 6 showed seed-averaging one FM family pays ~0 (members too correlated). Decorrelation lives across mechanisms.

**Members:**
- BPR (best performer)
- Pointwise (for diversity)
- SSM (alternative ranking loss)
- Watch-Time (WTFM, multi-head)
- ESMM (multiplicative funnel)

**Aggregation:**
- Within-user percentile ranks (scale-free, so models with different logit ranges can be averaged)
- Rank-average, not raw score average (critical: BPR never trains global bias)

**Selection & Veto:**
- Selected on `valid`
- **Vetoed** if drop ≥ 0.003 on `rand_valid` (Phase 11 bias overfitting check)

**Output:**
- Written to `submission_alt_{split}.csv` (repo root)
- Validated with official `submit.py`
- Alongside BPR submission, not as replacement

**Key Finding:** Ensemble across objectives beats seed-averaging single objective.

---

## Phases 18-20: Scale Transfer

### Phase 5: Scale Transfer (Pure → 1K → 27K)

**Benchmarks:** Pure → KuaiRand-1K → KuaiRand-27K  

**Hypothesis:**
Do Pure-derived KB priors (k, lr, iteration pacing) transfer to larger benchmarks? Over-parameterization finding likely to invert at scale with more data supporting capacity.

**Stages:**

| Stage | Cost | Purpose |
|-------|------|---------|
| `facts` | Cheap | Dataset shape, split dates, label rates, ID overlap |
| `headto2` | Moderate | Two configs (baseline + KB pick) × N seeds × 2 losses |
| `capacity` | Moderate | k sweep under both losses |
| `lr` | Moderate | Learning rate sweep for BPR |

**Transfer Target Configs:**

| Config | Pure Results | 1K Expected |
|--------|--------------|-------------|
| **Baseline** | Pointwise k=16 lr=0.001 | Transferred as-is? |
| **KB Pick** | BPR k=6 lr=0.0002 | Transferred as-is? |

**Key Finding on 1K:** Pure's recipe fails on 1K. BPR loses to pointwise on 1K cold-start (item cold-start: 85% of test videos unseen in train).

---

### Phase 10: 1K-Specific Optimization

**Benchmark:** KuaiRand-1K (~8x Pure)  

**Problem:** Phase 5 established Pure's recipe fails on 1K. Treat 1K as its own problem.

**Hypothesis:**
1K is an item cold-start regime (15.1% of test videos seen in train)
- best_epoch=2 at lr=0.001 → immediate overfitting → need regularization + smaller step
- k=16 beat k=4 by +0.005 and k=64 fell back → optimum interior, test k=32
- 73.9% of test have train-warm AUTHOR (vs 15.1% warm video) → lean on author via (user,author) affinity

**Hyperparameters Tested:**
- Embedding noise: 0.0, 0.05, 0.1 (regularization)
- Learning rates: below 0.001
- Embedding dimensions: k=32 (gap between 16 and 64)
- (user,author) affinity as 6th field

**Constraints:** Sparse Adam (vocab ~2.9M), CPU only, baseline_ref='none'

**Key Finding:** All four interventions negative on 3-seed replication. Underscores cold-start regime inversion.

---

### Phase 18: 1K Extended Validation (Revalidate Pure Winners)

**Benchmark:** KuaiRand-1K  

**Problem:** Pure's OTHER confirmed wins hold on 1K? Item-cold-start inverts within-user ranking losses.

**Discipline:** Every candidate 1 seed first. Only if beats 3-seed baseline mean (0.6439, sd 0.0022) by >1 baseline-seed SE (~0.0013) gets 3-seed replication.

**Stages:**

1. **`--stage ssm`**
   - SSM (Pure's confirmed BPR-peer)
   - Config: lr=0.0003, temp=1.0, neg_per_pos=8, k=16, epochs=45
   - Expected: Invert (per KB tier12_note prediction)

2. **`--stage affinity`**
   - (user,tab) and (user,dur_bucket) causal affinity
   - Smaller-cardinality fields than (user,author)
   - Higher test coverage expected

3. **`--stage unbiased`**
   - Score named config on 1K's OWN random-exposure log
   - Tests Phase 10 results on unbiased split (bias overfitting check)

**Constraints:** CPU only, sparse=True (dim ~2.9M)

**Status:** Predictions not yet validated (experimental queue).

---

### Phase 19: 1K GBDT with Pairwise Loss

**Benchmark:** KuaiRand-1K  
**Algorithms:** XGBoost (rank:pairwise, ndcg), CatBoost (CPU)  

**Hypothesis:**
GBDT never tested under genuinely pairwise objective on 1K. Trees' numeric affinity features (rates + evidence counts) should help cold-start where embeddings fail. But 1K's item-cold-start may reverse the ranking-loss advantage.

**Feature Scope (narrower than Pure's Phase 1F):**

| Block | Features | Notes |
|-------|----------|-------|
| **base** | user_id, video_id, author_id, tab, dur_bucket (cat) + log_duration_ms, hour (num) | Avoids 26M-object string array problem; tag/music_id OUT |
| **aff** | Causal affinity (user,tab), (user,dur), video, author, (user,author) | Explib/history.py is benchmark-agnostic |
| **user**, **vstat** | EXCLUDED | User features null under within-user ranking; vstat 3.4GB + leakage |

**Objectives Tested:**
- XGBoost: rank:pairwise (YetiRank), rank:ndcg (LambdaRank)
- CatBoost: custom_loss + eval_metric

**Status:** Experimental queue.

---

### Phase 20: KuaiRand-27K (Single Pointwise Run)

**Benchmark:** KuaiRand-27K (~224x Pure)  

**Data Characteristics:**
- Rows: 322,278,385
- Users: 27,285 (1.0x Pure's user count; same population, 224x interactions each)
- Videos: 32M catalog
- Test video coverage: 17.3% seen in train (vs 1K's 15.1%, Pure's 99.8%)
- Label rate: Flat across splits (0.263 → 0.257, no drift)

**Regime:** Even more extreme item-cold-start than 1K

**Hypothesis:**
Phase 5's facts pass confirmed 27K is a 1K regime, not Pure's. Per the plan's Phase 5 directive ("skip the architecture ladder... a well-tuned plain FM that reliably finishes inside the 6h ceiling"), this runs exactly ONE config: 1K's own confirmed winner, transferred as-is, **zero exploration, zero grid search**. BPR/SSM/GBDT all lost to plain pointwise FM on 1K, so no budget remains for comparison that prior data already predicts will lose.

**Config (Transferred from 1K):**
```python
loss='pointwise', k=16, lr=0.001, l2=1e-6, epochs=40, patience=4
```

**Constraints:**
- CPU only
- sparse=True (dim far beyond dense-Adam feasibility at 32M vocabulary)

**Budget Breakdown:**
- ~2.1h download
- ~27min load
- ~21min facts pass (already spent)
- No budget left for exploration

**Status:** Experimental queue.

---

## Supporting Library Modules

### explib/fm.py
**Factorization Machine with Pluggable Loss**

Keeps the starter kit's exact model, optimizer, and init; swaps only dL/dz.

**Available Losses:**

| Loss | Formula | Use Case |
|------|---------|----------|
| pointwise | sigmoid cross-entropy per row | baseline |
| bpr | within-user pairwise ranking | ranking, Pure regime |
| listwise | within-user softmax CE | ranking (retired) |
| hybrid | pointwise + λ·listwise | combined signal |
| ssm | sampled softmax / InfoNCE | implicit feedback ranking |

**Features:**
- Dense Adam (default) or sparse Adam (for large vocabularies)
- Embedding noise regularization (additive, gradient-preserving)
- Exact reproduction of official baseline with pointwise

**Sparse Path Caveat:**
Lazy/sparse Adam: rows absent from a batch get no L2 decay and no moment decay that step. Approximation (not numerically identical to dense path), so it's opt-in.

---

### explib/mtfm.py
**Multi-Task FM (Shared-Bottom Architecture)**

Shared embedding V with per-task weights over interaction dimensions.

**Construction:**
```
E = V[X]                     (B,F,k)   shared embeddings
S = E.sum(1)                 (B,k)
Q = 0.5*(S^2 - (E^2).sum(1)) (B,k)     per-dimension FM interaction
Z_t = b_t + W_t[X].sum(1) + Q @ A_t    (B,) one logit per task
```

**Single-Task Control:**
- T=1, A fixed at ones → Z is **exactly** the starter kit FM logit
- Any difference measured against baseline is attributable to auxiliary tasks

**Parameters:**
- n_tasks: number of auxiliary signals
- A_t (n_tasks, k): per-task weights over k interaction dimensions
- learn_A: whether to update A during training

**Key Property:** Only task 0 (long_view) is ever predicted; auxiliaries exist to shape V.

---

### explib/wtfm.py
**Two-Head Watch-Time FM (Shared-Embedding Multi-Task)**

Watch-time regression + long-view binary classification on shared embeddings.

**Model:**
```
E = V[X]                                (B,F,k)
S = E.sum(1)                            (B,k)
Q = 0.5*(S^2 - (E^2).sum(1))            (B,k)
Z_0 = b_0 + W_0[X].sum(1) + Q @ A_0     (B,) long_view logit → sigmoid → BCE
Z_1 = b_1 + W_1[X].sum(1) + Q @ A_1     (B,) watch ratio logit → Huber regression
```

**Loss:**
```
L = BCE(σ(Z_0), long_view) + w_wt * Huber_censored(Z_1, watch_ratio)
```

**Censoring (CWM Insight):**
For rows where play_time_ms >= duration_ms (completed plays), the true watch desire is right-censored. Loss is ONE-SIDED: penalizes predicting LESS, not more.

**Single-Task Control:**
- w_wt = 0, A_0 frozen at ones → byte-for-byte the starter kit FM on head 0

---

### explib/esmm.py
**ESMM-Style Multiplicative Decomposition**

P(long_view) = P(click) × P(long_view|click) funnel model.

**Model:**
```
E = V[X]                              (B,F,k)
S = E.sum(1)                          (B,k)
Q = 0.5*(S^2 - (E^2).sum(1))          (B,k)
Z = [b_0, b_1] + [W_0, W_1][X].sum(1) + Q @ [A_0, A_1]^T  (B,2)
Z_ctr = Z[:, 0]    P_ctr = σ(Z_ctr)
Z_cvr = Z[:, 1]    P_cvr = σ(Z_cvr)
P_lv = P_ctr × P_cvr  (multiplicative composition)
```

**Loss:**
```
L = BCE(P_lv, long_view) + w * BCE(P_ctr, is_click)
```

**Key Difference from Phase 1C:**
Phase 1C: is_click as CO-EQUAL 0.3-weighted auxiliary (harmful, seesaw).
ESMM: Multiplicative composition, conversion head supervised ONLY through long_view label.

**Funnel Justification:**
A row with is_click = 0 has play_time ~ 0 and therefore long_view ~ 0 deterministically. The funnel is real.

---

### explib/sequence.py
**Causal User Behavior Sequences (DIN, Attention)**

Last-L causal history per row, vectorized (no Python loop over millions of rows).

**Sequence Types:**

| Type | Description | Leakage |
|------|-------------|---------|
| **exposure** | Last L videos shown to user, strictly earlier in time | None; label-free, safe on all splits |
| **positive** | Last L long-viewed videos | Built from TRAIN-PERIOD LABELS ONLY |

**Construction:**
- Vectorized via lexsort + per-group lag gather
- Returns (H, hist_len): H is (N, L) int32 of item ids (most recent first, padded with 0); hist_len is (N,) count of real entries
- Positive sequence restricted to train period for eval rows (no label leakage)

**Parameters:**
- L: sequence length (default 20)
- valid_mask: if given, only rows where True may ENTER history (used for positive-sequence restriction)

---

### explib/history.py
**Causal Affinity Features**

Per-group rates built from training labels only.

**Affinity Types:**

| Affinity | Formula | Semantics |
|----------|---------|-----------|
| (user, video) | P(long_view \| user, video) | — |
| (user, author) | P(long_view \| user, author) | — |
| (user, tab) | P(long_view \| user, tab) | — |
| (user, dur_bucket) | P(long_view \| user, dur) | — |
| (video,) | P(long_view \| video) | Popularity |
| (author,) | P(long_view \| author) | Creator quality |

**Evidence Counts:**
For each group, also record how many rows contributed to the rate (support).

**Leakage Contract:**
- Rates built from **TRAIN labels only**
- No eval-period data enters the computation
- Safe to use on valid/test

**Key Finding on 1K:**
- (user, author): 26.5% test coverage (vs 2.6% on Pure where it was dead)
- (user, tab), (user, dur): higher coverage on 1K than Pure

---

## Benchmarks & Data Scales

| Benchmark | Rows | Users | Videos | Regime | Test Video Coverage |
|-----------|------|-------|--------|--------|---------------------|
| **Pure** | 1.4M | 27.3K | 40K | High coverage | 99.8% seen in train |
| **1K** | 9.5M (~8x) | 27.3K | 4.4M | Item cold-start | 15.1% seen in train |
| **27K** | 322M (~224x) | 27K | 32M | Extreme cold-start | 17.3% seen in train |

### Data Characteristics

**Pure:**
- Front-loaded training (59% in 3 days)
- High engagement period
- Near-complete ID coverage
- Baseline: pointwise FM 0.6016

**1K:**
- Uniform training distribution (~15K rows/day)
- Same users as Pure, broader video catalog
- Item cold-start regime (vendor catalog explosion)
- Baseline: pointwise FM **higher** than Pure due to flatter task

**27K:**
- Same population as Pure (27.3K users), 224x interactions each
- Full user histories available
- 32M-video catalog (massive)
- Extreme item cold-start (17.3% test coverage)

---

## Key Findings Summary

### Loss Functions
- ✓ **BPR wins on Pure** (organizer-flagged headroom)
- ✗ **BPR inverts on cold-start** (1K, 27K): pointwise FM beats ranking losses
- ✓ **SSM is BPR-peer on Pure**; predicted to invert on 1K
- ✗ **Listwise retired** (poor fit when ~1/3 positive)
- ✓ **Hybrid shows no clear benefit**

### Architecture
- ✓ **Capacity (k) is flat 2..128** (NOT a tuning knob)
- ✓ **Simple FM competes with FFM/DeepFM/AutoInt** (no architecture wins complexity cost)
- ✓ **Sequence/attention helps Pure**; not validated at scale
- ✓ **No clear winner among GBDT variants** (performance close to FM on Pure)

### Multi-Task Learning
- ✗ **Binary auxiliary signals act as regularizers only** (random label matches best real signal)
- ✓ **Watch-time regression different** (continuous signal + censored loss, not tested at scale)
- ✓ **ESMM multiplicative structure** differs from co-equal auxiliary; not yet validated
- ✗ **Dense auxiliary competes for capacity** (seesaw problem)

### Feature Engineering
- ✓ **Causal affinity rates strong on 1K** (26.5% (user,author) coverage vs 2.6% Pure)
- ✓ **Duration regime gains ~0.0003**; modest but real
- ✗ **User-side features contribute 0** (structural: within-user ranking metric)
- ⚠️ **Video-side aggregates carry leakage caveat** (computed over whole dataset period)

### Temporal
- ✓ **Drift exists** (base rate 0.3366 → 0.3134, front-loaded training)
- ✓ **Recency weighting helps moderately**
- ✓ **Discarding old rows helps more** (volume > recency)

### Regularization
- ✓ **Embedding noise regularizes** (helps Pure, esp. under BPR)
- ✓ **Early stopping + patience effective** (best_epoch << max_epochs)

### Scale Transfer
- ✗ **Pure KB priors fail on 1K** (regime inversion)
- ✗ **Architecture ladder skipped on 27K** (6h budget exhausted, prior data predicts pointwise FM)
- ✓ **Sparse Adam enables 1K/27K** (dense Adam infeasible at 2.9M/32M vocab)
- ✓ **1K, 27K are cold-start, not Pure** (15%, 17% test coverage; reverse ranking loss advantage)

### Ensemble
- ✓ **Rank-averaging across objectives beats seed-averaging** (decorrelation across mechanisms)
- ✗ **Seed-averaging single objective pays ~0** (high correlation)
- ✓ **Scale-free percentile ranks critical** (BPR doesn't train global bias)

### Unbiased Evaluation
- ✓ **Random-exposure log tracks test better on known blind spots** (valid's exposure bias a real issue)
- ✓ **Proposed as tie-breaker** inside 0.0016 noise band

---

## Experimental Hygiene

### Controls (KB control_rule)
Every major finding includes a control to isolate the mechanism:
- Shuffled features (tests regularization only)
- Random labels at matched sparsity (tests information transfer)
- Single-task baselines (verifies multi-task arithmetic)
- No_gate variants (isolates compositional structure)

### Scoring Rules (KB ground_rules)
1. **evaluate.py never reimplemented** – harness.score calls official scorer
2. **Decisions on valid** – test recorded for calibration only; rand_valid as overfitting veto
3. **Δ < 0.0016 not a result** – 2x official 0.0008 seed std
4. **Progress vs oracle** – reachable interval 0.4753 → 0.8645; baseline holds 30.7%

### Reproducibility
- **experiments.jsonl** – 300+ runs, one JSON record per run
- **Memoized configs** – identical (config, seed) pairs from log + cache
- **Cost accounting** – cache hits still consume iteration + real runtime
- **5-seed replication** – finalists run under 5 seeds before promotion to KB

---

## File Structure

```
ml_modelling/
├── README.md                    # Entry point
├── experiments.jsonl            # 300+ run log (JSON records)
├── experiments/
│   ├── p1_losses.py            # Phase 1A: loss alignment
│   ├── p1_multitask.py         # Phase 1C: multi-task aux signals
│   ├── p1_sequence.py          # Phase 1I: DIN attention
│   ├── p1_gbdt.py              # Phase 1F: GBDT
│   ├── p1_drift.py             # Phase 1G: temporal drift
│   ├── p7_arch_bpr.py          # Phase 7: architecture ladder
│   ├── p11_unbiased_eval.py    # Phase 11: random-exposure log
│   ├── p12_ssm_loss.py         # Phase 12: sampled softmax
│   ├── p13_watchtime.py        # Phase 13: watch-time regression
│   ├── p14_features.py         # Phase 14: duration + freshness
│   ├── p15_esmm.py             # Phase 15: ESMM decomposition
│   ├── p16_ensemble_final.py   # Phase 16: diverse ensemble
│   ├── p5_scale_transfer.py    # Phase 5: Pure→1K→27K transfer
│   ├── p10_1k_tune.py          # Phase 10: 1K tuning
│   ├── p18_1k_extend.py        # Phase 18: 1K extended validation
│   ├── p19_1k_gbdt.py          # Phase 19: 1K GBDT pairwise
│   ├── p20_27k_run.py          # Phase 20: 27K single run
│   ├── make_submission.py      # Train recommended config (BPR k=6)
│   ├── sweep.py                # General sweep runner
│   └── ...                      # 20+ experiment scripts
├── explib/
│   ├── dataset.py              # Log loader, row order verified
│   ├── features.py             # Configurable categorical encoding
│   ├── fm.py                   # FM with pluggable loss
│   ├── mtfm.py                 # Multi-task FM
│   ├── wtfm.py                 # Watch-time FM (two-head)
│   ├── esmm.py                 # ESMM decomposition
│   ├── sequence.py             # Causal sequences (exposure, positive)
│   ├── history.py              # Causal affinity rates
│   ├── harness.py              # Experiment logging, scoring
│   ├── unbiased.py             # Random-exposure eval split
│   ├── benchmarks.py           # 1K/27K dataset loading
│   ├── features.py             # Feature encoding (int fields)
│   └── ...
├── logs/                        # Raw stdout per run
├── tools/
│   ├── analyze.py              # Per-axis log view + hygiene
│   └── kb_check.py             # Validate KB claims vs log + scores
└── cache/                       # Parsed-log npz, predictions, state
```

---

## How to Use This Reference

1. **Find a Phase:** Locate by number (Phase 1A, Phase 7, etc.)
2. **Read Hypothesis:** Understand the research question
3. **Check Controls:** See how confounds were isolated
4. **Look at Key Finding:** What actually happened
5. **Cross-Reference:** Check supporting modules (fm.py, sequence.py, etc.)
6. **Validation Status:** Note whether result held at scale (Pure → 1K → 27K)

---

## Next Steps & Open Questions

1. **Phase 18 (1K validation):** Do Pure's SSM and affinity priors actually invert on 1K?
2. **Phase 19 (1K GBDT):** Does pairwise objective help GBDT on cold-start?
3. **Phase 20 (27K):** Will pointwise FM transferred from 1K match or fall?
4. **Watch-Time (Phase 13):** Does WTFM censored loss transfer to 1K/27K?
5. **ESMM (Phase 15):** Is multiplicative funnel structure valid at scale?
6. **Sequence on 1K:** Do DIN/attention help or hurt when 85% of videos are unseen?

---

**Generated:** 2026-09-01  
**Data Source:** ml_modelling/experiments.jsonl (300+ runs)  
**Official KB:** [knowledge_base/knowledge_base.yaml](../knowledge_base/knowledge_base.yaml)  
**Rationale:** [knowledge_base/knowledge_base_rationale.md](../knowledge_base/knowledge_base_rationale.md)
