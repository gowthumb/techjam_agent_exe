# Tier 1 + Tier 2 program — results & rationale

Companion to `knowledge_base_rationale.md`, covering the phases added after the
first KB was locked. Same discipline: one section per phase, hypothesis traced to
`KNOWLEDGE_BASE_PLAN.md`, what ran, the result on **all four splits**
(valid / test / rand_valid / rand_test), the control, the verdict against the
noise band, and the exact `knowledge_base.yaml` change it produced.

Every number here is in `../ml_modelling/experiments.jsonl` and re-checkable with
`python ml_modelling/tools/kb_check.py` (run from the repo root).

Run order and grounding:

| Phase | Executes (plan) | README dir. | Verdict |
|---|---|---|---|
| 11 Unbiased evaluation | Phase 0; Phase 2 §"Calibration reference" | #7 | ✅ method win — retired the noise lead, confirmed BPR at 4× the effect size |
| 12 Sampled-softmax loss | Phase 1a; KB prior "change the loss, not the capacity" | #1 | ✅ confirmed BPR-peer (+0.0004 test, ~2 se; 3–4× faster; ensemble member) |
| 13 Watch-time regression head | Phase 1c | #4 | ❌ negative — random-target control matches it exactly |
| 14 Duration-regime + freshness | Phase 1b | #4, #6 | ❌ regime neutral (shuffled control matches); video_age −0.0055 (drift) |
| 15 ESMM multiplicative decomp. | Phase 1c | #3 | ❌ neutral — beats pointwise, capped below BPR |
| 16 Diverse ensemble + alt submission | Phase 1d; extends Phase 6 | — | ~+0.0001–0.0003 over best single; `submission_alt_*.csv` |
| 17 KB distillation | Phase 2 | — | ✅ `knowledge_base.yaml` v3; `kb_check` + `analyze --check` pass |

**Bottom line.** Of the five Tier 1 levers, one (sampled softmax) is a confirmed
peer of the standing best and a useful ensemble member; the other four are
documented negatives, each closed with its own control. Plus Phase 11's
methodology win. This is the plan's own predicted outcome ("1–2 land; the rest
become documented negatives with controls").

| result | test primary | Δ vs official baseline (0.5946) |
|---|---|---|
| BPR k=6 (designated submission, unchanged) | 0.5980 | +0.0034 |
| **SSM lr 3e-4** (single model) | **0.5984** | **+0.0038** |
| **bpr + pointwise + ssm rank-ensemble** (`submission_alt`) | **0.5985** | **+0.0039** |

The designated submission stays **BPR**; the ensemble is `submission_alt_test.csv`
as the documented alternative, the choice left to the team.

---

## Phase 11 — unbiased evaluation via the randomly-exposed-video log

**Grounding.** `KNOWLEDGE_BASE_PLAN.md` Phase 0 ("before exploring anything, make
sure your numbers are trustworthy") and Phase 2's "Calibration reference"
section; starter kit README unexplored direction #7 ("log_random_*.csv 是随机曝光
日志，可作为额外的无偏验证集").

**Hypothesis.** Every verdict in this workstream is selected on `valid`, which is
a sample the production recommender chose to show — it carries exposure bias. The
KB already documents a "valid blind spot" (BPR is +0.0021 on valid but +0.0032
on test; embedding noise flat on valid but better on test). `log_random_4_22_to_
5_08_pure.csv` is 1.19M *randomly exposed* impressions over the valid+test
period. Split by the official date windows into `rand_valid` (20220422–28) and
`rand_test` (20220429–0508), encoded with the **frozen** train-fitted encoder, it
is a second, bias-free model-selection signal.

**What ran.** `explib/unbiased.py` (new) + `experiments/p11_unbiased_eval.py`. Six
configs whose valid-vs-test behaviour is already known from the log, 1–2 seeds
each, scored on all four splits:

| config (seed mean) | valid | test | rand_valid | rand_test |
|---|---|---|---|---|
| pointwise k16 | 0.6023 | 0.5948 | 0.3635 | 0.3687 |
| pointwise k2 | 0.6027 | 0.5965 | 0.3632 | 0.3686 |
| pointwise k16 + noise 0.1 | 0.6031 | 0.5959 | 0.3621 | 0.3678 |
| **bpr k6** | 0.6035 | **0.5978** | **0.3759** | **0.3882** |
| **bpr k16** | 0.6032 | **0.5979** | **0.3756** | **0.3882** |
| bpr k16 + noise 0.1 | 0.6039 | 0.5981 | 0.3692 | 0.3799 |
| listwise lr 0.001 | 0.5999 | 0.5924 | 0.3739 | 0.3866 |

Unbiased calibration rungs (label rate 0.081 vs 0.31 biased — the absolute
numbers are **not** comparable across the two): `rand_valid` random 0.3120 /
popularity 0.3605 / oracle 0.6854; `rand_test` random 0.3126 / popularity 0.3731
/ oracle 0.8138.

**Findings.**

1. **BPR's advantage is ~4× larger on the unbiased split.** `bpr − pointwise`:
   +0.0009 on biased valid, +0.0032 on biased test, **+0.0121 on rand_valid**.
   BPR's within-user ranking objective transfers to random exposure; pointwise
   partly fits the exposure policy. This is strong, independent confirmation of
   the one thing the KB is sure of.

2. **Embedding-noise regularisation is an artifact of biased traffic.** It gains
   marginally on biased valid/test (+0.0008 / +0.0012 pointwise) but **loses on
   the unbiased split** (−0.0014 pointwise, −0.0064 under BPR). The KB's
   `embedding_noise` "unclaimed lead" (σ=0.1 test +0.0006) is retired: the
   unbiased split says that lead is overfitting to exposure bias, not signal.

3. **`rand_valid` is a bias-overfitting *veto*, not a ranking-quality
   tiebreaker.** `spearman(valid, test) = +0.93`; `spearman(rand_valid, test) =
   +0.32`. The low correlation is driven by `listwise`, which ranks well under
   random exposure (0.374) but poorly under biased exposure (0.600 valid) — a
   distributional quirk, not a signal. `rand_valid` also carries ~0.002–0.003
   seed sd. So it is used to **veto** an intervention that gains on valid but
   loses on rand_valid, not to choose among interventions that agree.

**KB changes (applied in Phase 17).**
- `calibration`: add `random_exposure` block with the `rand_valid` / `rand_test`
  random / popularity / oracle rungs.
- `decision_protocol`: new `unbiased_veto` sub-section — every Tier 1/2
  intervention is also scored on `rand_valid`; one that gains on valid but drops
  ≥ 0.003 on `rand_valid` is treated as bias-overfitting and not shipped.
- `embedding_noise`: `under_bpr.unclaimed_lead` and `under_pointwise` verdicts
  downgraded — add the rand_valid evidence (−0.0014 / −0.0064).
- `candidate_models.fm_bpr`: add the `architecture_independent` sibling note that
  BPR's edge is +0.0121 on the unbiased split.

**Artifact.** `phase11_unbiased.json`. Reproduce:
`python experiments/p11_unbiased_eval.py`.

---

## Phase 12 — sampled-softmax (InfoNCE) loss — CONFIRMED BPR-PEER

**5-seed replication verdict.**

| config | valid | test | rand_valid |
|---|---|---|---|
| **SSM** (lr 3e-4, τ=1, n=8, k=16) | **0.6040 ± 0.0003** | **0.5984 ± 0.0004** | 0.3736 ± 0.0019 |
| BPR ref (lr 2e-4, k=16) | 0.6038 ± 0.0006 | 0.5980 ± 0.0002 | 0.3749 ± 0.0019 |
| SSM − BPR | +0.0002 (~0.7 se) | **+0.0004 (~2.1 se)** | −0.0013 (< 1 se) |

**SSM is a confirmed peer of BPR** — statistically indistinguishable on the
selection split, a ~2 se whisper ahead on test (+0.0004), tied on `rand_valid`.
It is **not** a clear improvement over BPR. What it adds:
1. **A second, independent objective that decisively beats pointwise** (+0.0017
   valid / +0.0036 test over the official baseline, vs BPR's +0.0022 / +0.0034) —
   strengthening the KB's core prior that the objective is the only live axis.
2. **3–4× faster convergence** — SSM peaks at epoch 3–7, BPR at ~21; even at n=8
   an SSM run is cheaper than a 60-epoch BPR run.
3. **A decorrelated ensemble member** (Phase 16): a different loss family than
   BPR, which is the diversity Phase 6 found was the only thing that moved an
   ensemble.

vs the official baseline: SSM **test 0.5984 = +0.0038**, marginally ahead of BPR's
+0.0034.



**Grounding.** `KNOWLEDGE_BASE_PLAN.md` Phase 1a (loss / optimizer); the KB's #1
prior ("change the loss, not the capacity"). README direction #1.

**Hypothesis.** The KB retired `listwise` as dead — but that `listwise` is a
softmax over the *whole impression list* with a uniform-over-positives target, a
poor fit when ~⅓ of the list is positive. **Sampled softmax** contrasts one
positive against a few sampled negatives (Wu et al., *On the Effectiveness of
Sampled Softmax Loss*, TOIS 2024 — beats BPR broadly, reduces overconfidence). It
is a different loss, and it is a smooth surrogate for top-k order. Added to
`explib/fm.py` as `loss='ssm'` — only `dL/dz` changes, per the module contract.

**What ran.** `p12_ssm_loss.py --stage grid` (lr × temp at n=4, k=8) then
`--stage neg` (k=16, n=8, tighter lr). Every run scored on all four splits.
Reference: BPR k16 lr 0.0002 (5-seed means) valid 0.6038 / test 0.5980 /
rand_valid 0.3756.

| config | valid | test | rand_valid | rand_test | peak ep |
|---|---|---|---|---|---|
| grid: ssm lr 5e-4 τ0.5 n4 k8 | 0.6036 | 0.5975 | 0.3725 | 0.3824 | 4 |
| grid: ssm lr 5e-4 τ1 n4 k8 | 0.6040 | 0.5980 | 0.3767 | 0.3892 | 4 |
| grid: ssm lr 5e-4 τ2 n4 k8 | 0.6043 | 0.5978 | 0.3749 | 0.3860 | 7 |
| grid: ssm lr 1e-3 τ* n4 k8 | 0.6032–39 | 0.5970–81 | ~0.373 | — | 2–4 |
| neg: ssm lr 3e-4 τ1 n8 k16 | 0.6036 | **0.5986** | **0.3759** | 0.3892 | 3 |
| neg: ssm lr 3e-4 τ2 n8 k16 | 0.6040 | **0.5987** | 0.3728 | 0.3837 | 6 |
| neg: ssm lr 5e-4 τ1 n8 k16 | 0.6034 | **0.5986** | 0.3721 | 0.3822 | 3 |

**Reading (pre-replication).**
- **Every SSM config beats pointwise** (0.6023 valid) by > the noise band and
  **lands on top of BPR**: valid 0.6034–0.6043 (BPR 0.6038), test **0.5986–0.5987
  vs BPR 0.5980**, rand_valid up to 0.3767 (BPR 0.3756).
- The consistent **+0.0006–0.0007 on test over BPR** across independent configs
  is the interesting signal — but every SSM run is single-seed so far, and the
  KB's `replication_rule` is explicit that anything within 0.0015 of the best
  must be 3–5-seeded before it is called. Replication is next.
- SSM converges fast (peak epoch 3–7) — cheaper per run than BPR's 60 epochs even
  at n=8.
- lr 5e-4 slightly overshoots (peak epoch 2–4); lr 3e-4 is the cleaner curve.
  temp 1 vs 2 within noise on valid, diverge on test/rand_valid — both go to
  replication.

**Controls (`--stage control`) — both confirm the edge is specifically
within-user sampled softmax, not "more negatives".**

| control | valid | test | rand_valid | reading |
|---|---|---|---|---|
| ssm, **global** negatives (all rows) | 0.6035 | 0.5962 | 0.3635 | −0.0016 test / −0.011 rand_valid vs within-user ssm — the within-user structure of the negatives is doing real work |
| bpr, pairs_per_pos = 2 | 0.6033 | 0.5975 | 0.3722 | more negatives makes BPR slightly *worse* |
| bpr, pairs_per_pos = 4 | 0.6034 | 0.5980 | 0.3710 | test back to ppp=1, rand_valid still below — not the mechanism |

So SSM's ~+0.0006 test edge over BPR is the **softmax over same-user negatives**,
not the negative count (BPR gains nothing from 2–4× negatives) and not global
contrast (that loses 0.0016 test / 0.011 rand_valid).

**Replication.** `--stage replicate`: the two lr-3e-4 finalists (τ1, τ2) + BPR
reference, 5 seeds each, select by valid mean. [pending]

**KB changes (Phase 17, if replication holds).**
- `validated_search_space`: new `ssm` block — `lr` 3e-4 (danger zone ≥ 1e-3:
  overshoots), `temp` [1, 2], `neg_per_pos` 8, `k` a cost knob as elsewhere.
- `candidate_models`: add `fm_ssm` (validated, result, cost — cheaper than BPR).
- `architecture_ladder`: SSM and BPR become co-equal stage-2 options; SSM
  preferred for the ensemble (a different objective → decorrelated member).
- `priors`: strengthen "change the loss, not the capacity" — a *second*
  objective now beats pointwise, capacity still does nothing.
- `dead_ends`: refine the `listwise` entry — "the uniform-target list softmax is
  dead; the *sampled* softmax is not — they are different losses".

**Artifact.** experiments.jsonl phase 12. Reproduce:
`python experiments/p12_ssm_loss.py --stage grid|neg|control|replicate`.

---

## Phase 13 — watch-time regression head (CWM-style censored loss) — NEGATIVE

**Grounding.** `KNOWLEDGE_BASE_PLAN.md` Phase 1c (multi-task / auxiliary
signals); README direction #4 (watch-time modelling / CWM, "研究深度").

**Hypothesis.** `long_view` is a deterministic threshold on watch time (`1` iff
`play_time_ms >= min(duration_ms, 18000)`, verified 97.9%). The binary label
coarsens a continuous signal the log carries in full. A regression head on the
watch ratio (`explib/wtfm.py`, new), sharing the embedding table, trains on
strictly more information and gets gradient from every row. Completed plays
right-censor the true watch desire, so the regression loss is one-sided.

**What ran.** `experiments/p13_watchtime.py --stage main`, watch-ratio target
`clip(play_time/duration, 0, 2)`, one-sided Huber (δ=0.5) on the 17% of rows that
are completed plays. Ranking score = the long_view head, the watch-time head, or
a rank-blend.

| config | valid | test | rand_valid | rand_test |
|---|---|---|---|---|
| pointwise baseline (ref) | 0.6023 | 0.5948 | 0.3635 | 0.3687 |
| wt head, w=0.3, rank by lv | 0.6012 | 0.5946 | 0.3617 | 0.3672 |
| wt head, w=0.5, rank by lv | 0.6014 | 0.5955 | 0.3616 | 0.3673 |
| wt head, w=1.0, rank by lv | 0.6022 | 0.5942 | 0.3594 | 0.3647 |
| wt head, w=0.5, rank by **wt** | 0.5868 | 0.5795 | 0.3583 | 0.3748 |
| wt head, w=0.5, rank by **blend** | 0.5915 | 0.5836 | 0.3536 | 0.3688 |

**Control** (`--stage control`):

| config | valid | test | rand_valid | rand_test |
|---|---|---|---|---|
| `w_wt = 0` single-task (== FM baseline) | 0.6014 | 0.5948 | 0.3663 | 0.3728 |
| `w_wt = 1` one-sided (censored) | 0.6022 | 0.5942 | 0.3594 | 0.3647 |
| `w_wt = 1` two-sided Huber | 0.6021 | 0.5943 | 0.3592 | 0.3645 |
| `w_wt = 1` **random continuous target** | 0.6015 | 0.5941 | 0.3598 | 0.3672 |

**Verdict: neutral on the selection split, negative on the unbiased one — and the
content does nothing.**
- **The random-continuous-target control matches the real watch-time head**
  (0.6015 vs 0.6022 valid, 0.3598 vs 0.3594 `rand_valid`). This is the decisive
  Phase 1C control repeated: a regression head on *pure noise* behaves the same as
  one on the actual watch ratio. The watch-time signal transfers nothing; the
  head is at most a weak regulariser that also drags the unbiased split.
- Against wtfm's *own* single-task baseline (0.6014 valid — it reproduces the FM
  baseline within noise), the watch-time head is **neutral on valid/test at every
  weight** — the same shape as the Phase 1C binary auxiliary heads, which were
  regularisers carrying no transfer.
- But it **consistently drops `rand_valid`** by 0.005–0.007 (0.3663 → 0.359 at
  w=1). Under the Phase 11 bias-veto that is a reject: the head is fitting
  something in the biased traffic that does not survive random exposure.
- **The censoring does nothing.** Two-sided Huber ≡ one-sided (0.6021 vs 0.6022,
  0.3592 vs 0.3594) — only 17% of rows are completed plays, too few for the
  one-sided correction to matter.
- **Ranking by the watch-time head directly is far worse** (0.5868 valid, below
  the popularity rung). The watch ratio is **duration-confounded** — short videos
  score a mechanically higher ratio — and `long_view`'s 18 s rule already
  accounts for that discontinuity. Predicting watch time and predicting
  `long_view` are not the same ranking problem.
- This *extends* Phase 1C: even a target that is mechanically the thing the label
  thresholds transfers nothing, and adds a bias-overfitting penalty on top.

**KB changes (Phase 17).**
- `dead_ends`: add "watch-time regression head (one-sided or two-sided Huber),
  any weight — at or below the pointwise baseline on valid, below it on
  rand_valid".
- `multi_task_signals`: extend the headline — the axis is unproductive for
  continuous targets too, not just the 11 binary ones.
- `feature_engineering_menu` / `diagnostics`: note "rank by predicted watch ratio
  scores below popularity — the ratio is duration-confounded and `long_view`
  already thresholds it".

**Artifact.** experiments.jsonl phase 13. Reproduce:
`python experiments/p13_watchtime.py --stage main`.

> **Logging note.** The phase-13 *main*-stage records carry `seconds = 0` — the
> training call sat outside the `H.Experiment` timing context in the first
> version of `p13_watchtime.py`. Fixed for the control/replicate stages and for
> `p15_esmm.py`. The metrics are unaffected; per-run wall-clock for the main
> stage is in `logs/p13_main.log` (~3 min/config).

---

## Phase 14 — duration-regime and video-freshness features

**Grounding.** `KNOWLEDGE_BASE_PLAN.md` Phase 1b (bucketing, explicit crosses,
per-field ablation); README directions #4 (duration) and #6 (time features). Run
on top of the confirmed BPR config, since the KB ranks feature work below the loss.

**Hypotheses.** `duration_regime` (0/1 at the 18000 ms `long_view` boundary) is an
interaction the flat 10-way `dur_bucket` cannot represent; `video_age_bucket`
(days from upload to impression — 100% coverage) varies within a user's list so it
can move the within-user order.

**What ran (`--stage main`).** On top of the BPR config. Reference: BPR k16
(5-seed) valid 0.6038 / test 0.5980 / rand_valid 0.3756.

| feature | valid | test | rand_valid | rand_test |
|---|---|---|---|---|
| + `duration_regime` | 0.6040 | 0.5984 | 0.3716 | 0.3806 |
| + `video_age_bucket` | **0.5983** | 0.5946 | 0.3727 | 0.3947 |
| + both | 0.5979 | 0.5945 | 0.3704 | 0.3919 |
| `dur_buckets = 20` (no new field) | 0.6036 | 0.5980 | 0.3745 | 0.3848 |

**Reading.**
- **`duration_regime` is a marginal positive** — +0.0002 valid / +0.0004 test
  over BPR, inside the single-run noise band; `rand_valid` 0.3716 (−0.004, not a
  veto). Needs 3-seed replication to separate from noise. If it holds it is small
  and cheap (one binary field).
- **`video_age_bucket` is a clear negative** (−0.0055 valid). The age distribution
  shifts hard between splits — train median 2 days, valid 14, test 23 — so the
  "old video" buckets are barely trained (few old videos exist in the train
  period) and the FM's embedding for them is noise at eval time. Same failure
  class as the KB's affinity-coverage rule: a feature whose eval-time regime is
  absent from train is a liability, not a signal. `video_age` is a *temporal
  drift* feature and Phase 1G already established drift is not exploitable the
  obvious way on this split.

`dur_buckets = 20` reproduces BPR exactly (0.6036 / 0.5980) — bucket count is a
cost knob like `k`, not a scoring knob. `regime + age` inherits `video_age`'s
drag.

**Control + replication verdict.**

| config | valid | test | rand_valid |
|---|---|---|---|
| + `duration_regime` (3 seeds) | 0.6038 ± 0.0003 | 0.5982 | 0.3730 |
| **shuffled-regime control** | 0.6040 | 0.5978 | 0.3677 |
| BPR baseline | 0.6038 | 0.5980 | 0.3749 |

**`duration_regime` is neutral.** Valid lands exactly on the BPR baseline, and the
**shuffled-regime control scores the same on valid/test** — so the tiny movement
is capacity/noise, not the regime information. (The real regime does help
`rand_valid` a little over shuffled, 0.3730 vs 0.3677, but both are below the BPR
baseline's 0.3749.) The whole P14 axis is negative: `duration_regime` neutral,
`video_age` negative (distribution shift), `dur_buckets` a cost knob.

**KB changes (Phase 17).**
- `feature_engineering_menu`: add `duration_regime` (neutral — shuffled control
  matches; priority skip) and `video_age_bucket` (−0.0055, skip, train/eval age
  distribution shift). Note `dur_buckets` is a cost knob, not a scoring knob.
- `dead_ends`: add `video_age_bucket` — "days from upload to impression; the age
  distribution shifts train median 2d → test median 23d, so the eval-time buckets
  are barely trained. Same failure class as the affinity-coverage rule." And
  `duration_regime` — "an explicit field at the 18s label boundary; shuffled
  control matches it. The FM's user×dur_bucket cross already carries it."

---

## Phase 15 — ESMM-style multiplicative decomposition

**Grounding.** `KNOWLEDGE_BASE_PLAN.md` Phase 1c (names "ESMM-style" explicitly);
primer A.2/A.3 (the click → engagement funnel).

**Hypothesis.** `is_click = 0` ⟹ `long_view = 0` deterministically (measured:
P(lv|no-click) = 0.003, P(lv|click) = 0.72), so `P(lv) = P(click)·P(lv|click)` is
well posed. `explib/esmm.py` (new): two heads, multiplicative, conversion head
supervised only through the `long_view` label. Distinct from the Phase 1C co-equal
`is_click` aux head that lost to the seesaw.

**What ran (`--stage main`).** Reference: pointwise 0.6023 / 0.5948 / 0.3635;
BPR 0.6038 / 0.5980 / 0.3756.

| config | valid | test | rand_valid | rand_test |
|---|---|---|---|---|
| esmm, w_click = 0.3, gated | 0.6025 | 0.5954 | 0.3618 | 0.3665 |
| esmm, w_click = 1.0, gated | 0.6016 | 0.5944 | 0.3589 | 0.3635 |
| **no-gate control** (`σ(z_cvr)` only) | 0.6013 | 0.5930 | 0.3644 | 0.3715 |

**Reading.** ESMM (w=0.3) beats **pointwise** by +0.0002 valid / +0.0006 test —
the multiplicative funnel does extract a little over a flat logloss, and the
**gate is what does it**: the no-gate control (two heads, no multiplication) is
−0.0012 valid / −0.0024 test *below* the gated model. But ESMM still loses
clearly to **BPR** (−0.0013 valid, −0.0026 test) and its `rand_valid` (0.3618) is
below pointwise's (0.3635). Same lesson as every calibration-oriented objective on
this metric: a well-calibrated `P(long_view)` is not a good within-user *ranking*
— within-user order never uses the global rate the calibration is spent on.
Higher click weight makes it worse (w=1 → 0.5944 test). ESMM is a documented
neutral: real over pointwise, capped below BPR.

**KB changes (Phase 17).**
- `multi_task_signals.esmm`: add — "+0.0006 test over pointwise (the funnel is
  real: P(lv|no-click) = 0.003), but −0.0026 vs BPR and −0.0017 `rand_valid`.
  Calibration objectives lose to ranking objectives on this metric."
- `dead_ends`: "ESMM / multiplicative P(click)·P(lv|click) — beats pointwise,
  loses to BPR, same as pointwise itself."

---

## Phase 16 — diverse ensemble + alternative submission

**Grounding.** `KNOWLEDGE_BASE_PLAN.md` Phase 1d; extends Phase 6. Phase 6 found
seed-averaging one FM family pays ~0 (members too correlated); this rank-averages
(within-user percentile) across *different objectives*.

**What ran.** `p16_ensemble_final.py` — retrains 5 members at seed 0 (BPR,
pointwise, SSM, watch-time head, ESMM), caches their predictions on all four
splits, rank-averages over combinations, selects on valid with the `rand_valid`
veto.

| combo | valid | test | rand_valid | rand_test |
|---|---|---|---|---|
| **bpr + pointwise + ssm** | **0.6042** | **0.5985** | 0.3730 | 0.3837 |
| bpr + ssm | 0.6034 | 0.5982 | 0.3753 | 0.3882 |
| ssm alone | 0.6036 | 0.5986 | 0.3759 | 0.3892 |
| bpr alone | 0.6031 | 0.5979 | 0.3742 | 0.3864 |
| all 5 (incl. wt + esmm) | 0.6040 | 0.5981 | 0.3685 | 0.3772 |

**Reading.** The **bpr + pointwise + ssm** rank-ensemble is the best valid number
(0.6042) and matches SSM on test (0.5985). It adds **~+0.0001–0.0003 over the best
single model** — real but small, exactly Phase 6's finding ("+0.0005 at best;
diversity is what little there is"). Adding the weak members (wt, esmm) *hurts*
(all-5 drops to 0.6040 valid, `rand_valid` 0.3685) — so the ensemble is
three-way, not five. `submission_alt_{valid,test}.csv` written and validated with
the official `submit.py --check` / `--score` (valid primary 0.6041).

**Submission.** The designated final submission stays **BPR k=6**
(`make_submission.py` / `submission_test.csv`, untouched). The ensemble is in
`submission_alt_test.csv` as the documented alternative — **test 0.5985 = +0.0039
over the official baseline** (BPR alone +0.0034). The choice is left to the team.

**KB changes (Phase 17).** New `kb_ensemble` section — the three-way combo, its
+0.0001–0.0003 delta over the best single, the "weak members hurt" finding, and
the `submission_alt` pointer.

---

## Phase 17 — knowledge-base distillation

`knowledge_base.yaml` bumped to `schema_version: 3`. Sections changed:
`meta` (v3_changes note), `calibration` (+`random_exposure` rungs),
`decision_protocol` (+`unbiased_veto`, control_rule now lists 4 reversals,
log_schema_rule count refreshed), `validated_search_space` (+`ssm` lr / temp /
neg_per_pos blocks, +`ssm` epochs), `priors` ("change the loss" strengthened —
two objectives now beat pointwise), `feature_engineering_menu`
(+`duration_regime`, `video_age_bucket`, `dur_buckets_count`), `multi_task_signals`
(headline extended, +`watch_time_regression` and `esmm` sub-blocks),
`candidate_models` (+`fm_ssm` recommended-for-ensemble, +`wtfm_watchtime`,
+`esmm`; `fm_bpr` note clarifying it stays the submission), `architecture_ladder`
(stage 2 now BPR-or-SSM, new `ensemble_bpr_ssm_pointwise` stage, `do_not_go_here`
expanded), `dead_ends` (+watch-time head, ESMM, duration_regime, video_age,
global/extra negatives), `embedding_noise` (**retired** — rand_valid is the
held-out split the v2 note asked for, and it says −0.0064).

Verification: `python tools/kb_check.py` → OK; `python tools/analyze.py --check` →
clean.
