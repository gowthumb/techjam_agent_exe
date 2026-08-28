# Knowledge Base — Rationale

Why every entry in `knowledge_base.yaml` exists, traced to the experiment that
produced it. The agent never reads this file; it is here so a claim in the YAML
can be checked rather than taken on trust.

Every number below is in `experiments.jsonl` and re-checkable with
`python tools/kb_check.py`.

---

## 0. Phase 0 — is the harness trustworthy?

Nothing downstream means anything if the numbers drift, so two things were pinned
before any exploration.

**The baseline reproduces.** `python3 baseline.py --model fm` gives valid primary
0.6015 / test 0.5953 against the official 0.6016 / 0.5946 — inside the published
0.0008 seed std.

**Row order is identical to the starter kit.** `explib/dataset.py` keeps all 19 log
columns and caches them, but `verify_row_order_matches_starter_kit()` asserts, split
by split, that its user/video/label arrays match `data.load()` element-for-element.
Submission `row_id` alignment is defined by that ordering, so this is checked rather
than assumed.

**Every experiment goes through the same control.** `explib/fm.py` swaps only
`dL/dz`; the FM forward pass, Adam update and initialization are the starter kit's,
unchanged. So `1A-control-pointwise` (valid 0.6022) landing on the baseline is
evidence that any later difference is the intervention and not the plumbing. The
multi-task model has the same property by construction: with one task and the
per-task interaction weights frozen at ones, `MTFM` is algebraically the kit's FM,
and `1C-control-singletask` reproduces 0.6022 / 0.5945 exactly.

---

## 1. Where the plan and the organizers' README disagreed, and who was right

`KNOWLEDGE_BASE_PLAN.md` Phase 1a proposed sweeping `k` over 8/16/32/64/128 and
tuning L2 and optimizers. The starter kit README already reports the organizers
measuring k = 8/16/32 → 0.5895/0.5902/0.5887 and getting nothing, and adding all 13
CWM feature fields → 0.5940 vs 0.5950, also nothing. It redirects to seven
unexplored directions, led by the loss function.

I followed the README's ordering and spent the bulk of the budget there. That was
the right call: the loss axis produced the only confirmed improvement, and the
capacity sweep (`1D-pointwise-k8..k128`, valid 0.6008–0.6022, flat within noise)
reproduced the organizers' negative result in about six minutes — enough to write a
danger zone from our own numbers without spending the exploration budget on it.

---

## 2. The loss function — the one confirmed win

**Why it should work.** The metric is a *within-user* ranking metric. GAUC asks
only whether a user's positives outrank that user's negatives, and nDCG@5 only about
order inside a user's list. A pointwise logloss spends capacity fitting the global
positive rate, which no within-user ordering depends on. A pairwise loss constrains
exactly what is scored and nothing else.

**What happened first.** At the baseline's `lr=0.001` the answer looked like a
shrug: BPR +0.0013, listwise −0.0017, hybrid −0.0018 — all inside the noise band.
Worse, every ranking-loss run peaked at epoch 1–4 and then decayed. My first sweep
went *up* in learning rate, which made it worse (BPR at lr=0.02: −0.0043).

**The diagnosis.** The decay signature is a step-size problem, not a loss problem.
The ranking losses normalize per pair or per group where pointwise normalizes per
row, so the effective step at a given `lr` is substantially larger. Sweeping *down*
(`1A-bpr-lr0.0001/0.0002/0.0005`) found the peak at lr = 0.0002.

**The verdict, after replication.** Single runs still read "neutral" against the
0.0016 band. Five seeds each (`1H-seedstudy-*`) settle it:

| config | valid | test |
|---|---|---|
| pointwise, k=16, lr=0.001 | 0.6018 ± 0.0005 | 0.5948 ± 0.0008 |
| BPR, k=16, lr=0.0002 | **0.6038 ± 0.0006** | **0.5980 ± 0.0002** |

Difference **+0.0021 valid (se 0.0004) and +0.0032 test (se 0.0004)** — separated at
two standard errors on both splits. BPR is also markedly more stable across seeds
(test sd 0.0002 vs 0.0008).

This is the single most important thing in the KB, and it was nearly discarded
twice: once by sweeping the learning rate in the wrong direction, and once by
applying a single-run noise band to what needed replication. Both mistakes are
written into `decision_protocol` as rules.

**Listwise is genuinely dead**, not mis-tuned — negative at every learning rate from
0.0001 to 0.02 (−0.0014 to −0.0149). The uniform-over-positives target is a poor fit
for lists where roughly a third of items are positive. The hybrid inherits the
problem.

---

## 3. Multi-task — a finding that reversed under its control

Nine auxiliary signals were tried as extra heads on a shared embedding table. Seven
came back positive, in a suspiciously tidy band of +0.0008 to +0.0011 valid.

Two things looked wrong. First, `is_hate` — 0.04% positives — helped as much as
`is_follow`. Second, my control had a confound: it froze the per-task interaction
weights `A` while every auxiliary run learned them, so part of the "gain" could have
been k extra parameters. Adding `1C-control-singletask-learnA` cleared the confound
(it scores 0.6014, *below* the frozen control, so learning `A` was not the cause).

The pattern that remained was monotone in the wrong variable: effect size rose as the
auxiliary got **sparser**, not as it got more related to `long_view`. That is what a
regularizer looks like, not what transfer looks like. So I ran the decisive control —
an auxiliary head trained on **pure random noise** at a matched positive rate:

| auxiliary | positive rate | Δ valid | Δ test |
|---|---|---|---|
| `is_follow` | 0.00101 | +0.0011 | +0.0017 |
| `is_forward` | 0.00100 | +0.0011 | +0.0011 |
| **`random_sparse`** | **0.00100** | **+0.0010** | **+0.0009** |
| `is_hate` | 0.00042 | +0.0011 | +0.0011 |
| `play_complete` | 0.17343 | +0.0004 | +0.0008 |
| **`random_dense`** | **0.46276** | **+0.0003** | **+0.0001** |
| `is_click` | 0.46345 | −0.0003 | −0.0013 |

A random label is indistinguishable from the real signals. The auxiliary content
does no work; only its sparsity matters. `is_click`, the one dense, strongly
correlated signal, is the only one that hurts — the textbook seesaw, competing for
shared capacity rather than regularizing it.

The KB therefore says **do not spend iterations selecting among auxiliary signals**.
Without the random-label control, this axis would have shipped as a ranked menu of
"helpful signals" and sent the agent down a path with nothing at the end of it.

---

## 4. Behaviour sequences — the other reversal

The organizers call this the biggest untouched direction, and the coverage supports
that: 99.7% of eval rows carry a history, mean length 18.8 of a possible 20. So it
got a real DIN — attention over the user's recent items, keyed on the candidate —
rather than a proxy.

The result only means something next to its control:

| model | valid | test | runtime |
|---|---|---|---|
| DIN, exposure sequence | 0.6022 | 0.5961 | 256s |
| **same network, attention branch removed** | **0.6034** | **0.5969** | **23s** |
| DIN, positive sequence (train labels only) | 0.6038 | 0.5978 | 216s |

The no-sequence control **beats** the exposure-sequence model at one eleventh of the
cost. Whatever lift the DIN showed over the FM is the MLP, not the behaviour
history — and the exposure sequence is actively harmful, which makes sense once you
notice the exposure list is largely a record of what the recommender chose to show,
not what the user wanted.

The one live thread: the *positive* sequence (what the user actually long-viewed,
built from train-period labels only) does edge the control, +0.0004 valid / +0.0009
test. That is inside the single-run band, so the KB records it as the one
unexhausted mechanism *pending 3-seed replication*, not as a recommendation.

---

## 5. History / affinity features — killed by coverage, not by concept

The organizers note that pure user-side features contribute exactly zero, which
follows from within-user ranking: anything constant inside a user cannot change the
order. Affinity features keyed on (user × item-attribute) dodge that, so they were
worth a try.

They were built carefully — counters from train labels only, eval rows reading
frozen train counters, train rows using causal or leave-one-out counts — and the
leakage contract is enforced by a probe that flips every eval label and asserts the
features do not move.

They still did nothing (+0.0001 to −0.0024). The reason is coverage, and it is
measurable before training:

| key | warm rows (train / valid / test) |
|---|---|
| `(user, tab)` | 94% / 93% / 90% |
| `(user, dur_bucket)` | 81% / 90% / 87% |
| `(video)` | 99% / 100% / 100% |
| `(user, author)` | 6% / 3% / **2.6%** |
| `(user, music)` | 7% / 4% / **2.9%** |

`(user, author)` is the one that sounds most promising and is 97% cold-start on
test — a constant in disguise. The dense keys fail for the opposite reason: the FM's
`user_id × video_id` and `user_id × author_id` crossings already learn what the
affinity feature is telling them.

Hence the KB rule: **measure eval warm coverage first, drop below ~70%**. It is a
one-line check that saves a 30–60s training run.

---

## 6. GBDTs — a fair trial, a clear loss

LightGBM was given feature blocks the FM structurally cannot use: numeric affinity
rates, evidence counts, log durations, user profile columns. Best result 0.6001
(binary objective), against a 0.6016 baseline. `lambdarank` — which optimizes the
ranking objective directly — did *worse* than plain binary here.

One diagnostic came out of it. With the affinity block, LightGBM early-stops at
**round 15**. The affinity columns dominate the gain ranking and then generalize
worse each round after, because causal target encoding gives train rows thinner
evidence than eval rows — a train/eval distribution shift the trees split hard on.
That is now a `diagnostics` entry: a GBDT stopping inside ~15 rounds is a symptom of
target-encoded features, not of good early convergence.

The `vstat` block deserves a note. `video_features_statistic_pure.csv` carries
aggregates computed over the whole dataset period, so they partially encode
test-period outcomes. I ran it separately and flagged it rather than folding it into
a headline. It scored 0.5992 — it did not help even with that advantage, which
settles the question without needing to adjudicate the ethics.

---

## 7. Temporal drift — real, but not exploitable the obvious way

Two measured facts said drift should matter:

- the `long_view` base rate falls from 0.3366 on train to 0.3133 / 0.3135 on
  valid / test;
- 59% of the 1.14M train rows fall in three days (2022-04-10..12), while the eval
  period runs at ~15K rows/day.

Meanwhile ID overlap is essentially total — 99.8% of test videos and 96.7% of test
users appear in train. So this is drift on known IDs, not cold start. (That fact
alone redirects effort away from a whole category of standard recommender machinery.)

Both interventions failed, and failed informatively:

| intervention | valid Δ |
|---|---|
| recency weighting, half-life 14d | +0.0009 |
| recency weighting, half-life 7d | +0.0009 |
| recency weighting, half-life 4d | −0.0011 |
| recency weighting, half-life 2d | −0.0086 |
| train on last 10 days only | −0.0025 |
| train on last 7 days only | −0.0091 |
| train on last 5 days only | −0.0164 |
| train on last 3 days only | −0.0274 |

Monotone in severity. Volume beats recency decisively on this split — the model
needs the old high-volume days to learn the ID embeddings at all, and the drift is
not large enough to pay for throwing them away. The KB records mild weighting as
harmless-to-marginal and hard windows as a dead end.

---

## 8. Capacity — where I nearly wrote the wrong entry myself

A `k=1` run had been included in the failure probes as a deliberate *underfit*
control, expected to land near the popularity rung. It came back at **0.6033
valid — one of the best single numbers in the whole log**, alongside BPR at k=8
scoring 0.6040. The obvious story wrote itself: the baseline is over-parameterised,
search `k` downward.

I drafted exactly that into the KB. Then I ran the replication the KB's own
`replication_rule` demands — three seeds per point:

| loss | k | valid (3 seeds) | test |
|---|---|---|---|
| pointwise | 1 | 0.6023 ± **0.0021** | 0.5962 ± 0.0026 |
| pointwise | 2 | 0.6028 ± 0.0005 | 0.5962 ± 0.0008 |
| pointwise | 4 | 0.6021 ± 0.0002 | 0.5956 ± 0.0003 |
| pointwise | 6 | 0.6020 ± 0.0006 | 0.5948 ± 0.0009 |
| pointwise | 16 | 0.6018 ± 0.0005 | 0.5948 ± 0.0008 (5 seeds) |
| BPR | 4 | 0.6035 ± 0.0004 | 0.5979 ± 0.0005 |
| BPR | 6 | 0.6037 ± 0.0004 | 0.5979 ± 0.0002 |
| BPR | 8 | 0.6037 ± 0.0004 | 0.5981 ± 0.0003 |
| BPR | 16 | 0.6038 ± 0.0006 | 0.5980 ± 0.0002 (5 seeds) |

The story does not survive. `k` is **flat from 1 to 16 under both losses**. The
0.6033 at k=1 was a lucky seed: k=1 carries four times the seed variance of k=2
(sd 0.0021 vs 0.0005), so it is precisely the configuration a single-run band
under-protects. Under BPR, k=4 through k=16 are indistinguishable at 0.6035–0.6038.

So the corrected entry says `k` is a **cost knob, not a scoring knob** — pick k=6
because it runs ~30% faster than k=16 for the same score, and never raise it in
response to a plateau. And a new prior was added: extreme configurations have
larger seed variance, so a fixed single-run band under-protects at the edges of the
search space.

L2 was swept at k=64 specifically, to give regularization its best chance of
rescuing an over-large model: 1e-7 → 0.6009, 1e-6 → 0.6008, 1e-5 → 0.6007,
1e-4 → 0.6002, 1e-3 → 0.5837. It never recovers the k=16 score. Regularization is
not the missing ingredient either.

This section exists because it is the clearest evidence for the KB's central
methodological claim. I made the exact mistake the KB warns about, on my own data,
while writing the warning.

---

## 9. Failure modes — and a correction to the plan's example

`KNOWLEDGE_BASE_PLAN.md` offers `loss = NaN → learning rate too high → halve it` as
its sample diagnostic. **That symptom does not occur in this codebase.** The FM uses
Adam, whose update is bounded by construction, so even a 1000× learning rate does not
produce NaN. At `lr=1.0` the loss sits finite at ~6.9 (against a healthy 0.48)
indefinitely, and valid primary parks at 0.547.

The real signature is: **loss magnitude an order of magnitude above healthy, valid
score below the popularity rung (0.5807), no recovery across epochs.** An agent
watching for NaN would miss it entirely and burn the full epoch budget. That
correction is now `diagnostics[0]`, with the reasoning attached so it is not
mistaken for a typo.

The probes also give the collapse floor: L2 = 1.0 drives valid to 0.5317, below
popularity — useful because "below the popularity rung" is a much better stop
condition than any absolute threshold.

---

## 10. What the KB deliberately does not say

- **No untested entries.** `candidate_models` marks `validated: true` only for
  models actually run here. XGBoost and CatBoost are installed but unrun, and say so.
  DeepCTR-Torch's model zoo is listed as *not recommended*, because those are
  capacity escalations and capacity is measurably not the bottleneck.
- **`scale_transfer` is honestly empty.** Phase 5 has not run. The section names
  which findings are expected to be scale-invariant (the diagnostics, the
  within-user-ranking consequences) and which need re-validation first — with the
  capacity finding flagged as the most likely to invert with 8×–230× more data.
- **The architecture ladder points downward.** Every capacity-increasing move in this
  log was flat or negative. Writing an escalation ladder that says "then try
  xDeepFM" would be repeating the instinct the evidence contradicts.

---

## 11. Phase 5 — the KB does not transfer to KuaiRand-1K, and that is the finding

The plan frames Phase 5 as a scale question: 1K is ~8x Pure, so re-validate `k`,
regularization and pacing. The facts pass says that framing is wrong before a
single model is trained:

| | Pure | 1K |
|---|---|---|
| rows | 1,436,609 | 11,713,045 (8.2x) |
| distinct users | 27,077 | **1,000** (0.04x) |
| distinct videos | 7,551 | **4,369,953** (579x) |
| rows per video | 190.3 | **2.7** |
| test videos seen in train | **99.9%** | **15.1%** |
| label rate train -> test | 0.3366 -> 0.3135 | 0.2635 -> 0.2588 (flat) |
| split shares | 79 / 9 / 12% | 43 / 22 / 35% |

Pure samples many users over a small catalog. 1K takes 1,000 users' *entire*
histories over the full catalog. So Pure is a **warm-ID problem with temporal
drift**, and 1K is an **item cold-start problem without drift**. That is a change
of regime, not of volume, and it is visible for the price of one parsing pass.

The head-to-head, three seeds each:

| config | valid | test |
|---|---|---|
| baseline pointwise k=16 lr=0.001 | **0.6439 ± 0.0022** | **0.6380 ± 0.0021** |
| KB pick: BPR k=6 lr=0.0002 | 0.6288 ± 0.0009 | 0.6230 ± 0.0079 |

**−0.0152 valid, −0.0151 test** — about seven standard errors. The KB's single
confirmed recommendation is actively harmful here.

I spent four extra runs separating "the loss does not transfer" from "its step size
does not", because the KB explicitly lists `lr` under `needs_revalidation` and that
distinction changes what gets written. BPR loses at **every** learning rate tried:
0.0002 → 0.6279, 0.001 → 0.6274, 0.02 → 0.6243, 0.005 → 0.6220, against 0.6438.
It is the loss.

The mechanism is consistent with the structure. BPR's Pure advantage was declining
to spend capacity on the global positive rate, since within-user ranking never uses
it. When 85% of test items have no trained embedding at all, there is little
within-user ordering signal left to exploit, and the pointwise model's calibration
carries more of the score.

Three further Pure claims inverted:

- **capacity.** Pure: flat 0.6018–0.6028 across k=1..16. 1K: peaked — k=4 0.6389,
  k=16 **0.6438**, k=64 0.6390. Capacity genuinely matters at 1K, so "k is a cost
  knob" is a Pure-only statement.
- **the noise band.** Pure valid sd 0.0005; 1K valid sd 0.0022, four times larger.
  Pure's 0.0016 band would be far too tight to make decisions with on 1K.
- **pacing.** Best epoch moves from 7 to 2 — 1K overfits much faster.

What did hold: the diagnostics (a property of Adam, not of the data), the
consequence that user-only features contribute zero (a property of within-user
ranking), and the replication rule — which mattered *more* at 1K, where seed noise
is four times larger.

So `scale_transfer` now says something falsifiable and useful: treat this file as a
method rather than a set of values, run the facts pass first, and let *rows per
video* and *share of eval items unseen in train* decide whether any of the numbers
apply at all.

Two engineering findings came out of the same work, both discovered by running
rather than reasoning. The dense Adam update is O(vocab × k) **per batch** — free at
Pure's 40K vocabulary, infeasible at 1K's 2.9M — so `fm.train(sparse=True)` exists,
verified inside the noise band on Pure before being trusted at scale. And the
full-column string-based loader peaks at 5.6GB on 1K against 3.1GB free; the
minimal chunked integer path peaks at 0.4GB.

**KuaiRand-27K was not attempted**, and the reason is a design gap rather than a
schedule one: ~7GB of parsed arrays on a 23.7GB machine, and ~7.7 hours for a single
40-epoch run at the measured per-row cost — more than one 6h benchmark budget for
*one config*, before any search. It needs out-of-core storage plus
subsample-search-then-final-fit. Since 27K is sampled like 1K, the 1K regime and not
the Pure one is what it should expect.

---

## 12. Phase 4 — the KB's value is efficiency, not ceiling

The plan says not to assume the KB helps. It does help, but not in the way the
phrase "does the KB improve things" suggests, and the distinction is worth being
precise about.

The ablation runs the same loop twice on the same iteration budget: one arm
proposing blind, one reading `knowledge_base.yaml` (dead-end filter, validated
ranges, replication rule, recommended opening config). The blind arm's space is
taken verbatim from `KNOWLEDGE_BASE_PLAN.md`'s own Phase 1a/1d proposals — k in
8..128, the four losses, the LR and L2 sweeps, affinity on/off — rather than
invented here, so it is not a straw man built to lose.

The circularity has to be stated plainly: the KB was derived from this dataset, so
it cannot honestly claim to *discover* faster than blind search. What it can claim
is transfer cost — how much a search pays to reach what the KB hands over on
iteration 1.

| metric (mean of 3 restarts) | blind | with KB |
|---|---|---|
| best valid reached | 0.6039 | **0.6042** |
| best valid, worst restart | 0.6032 | **0.6042** |
| best test reached | 0.5975 | **0.5983** |
| iterations to beat baseline | 3, 3, **14** | **1, 1, 1** |
| iterations to converge | 6, 6, 4 | **4, 4, 4** |
| trials at or below baseline | **12.67 / 15** | **0.33 / 15** |
| search wall-clock | 1938s | **1083s** |

Restart 0 alone suggested the blind arm reached a *higher* ceiling (0.6043 vs
0.6042), and I wrote that down before the other restarts finished. Across all three
that reading does not survive: blind averages 0.6039 with a worst case of 0.6032,
while the KB arm lands on 0.6042 every time. The KB's advantage on the ceiling is
small — about +0.0003, inside the noise band — but its advantage in *consistency*
is not, and neither is anything else in the table.

The two numbers that matter are the ones the plan asked for. Wasted trials go from
**12.67 to 0.33 out of 15**. And iterations-to-beat-baseline goes from a highly
variable 3, 3, 14 to a flat 1, 1, 1 — the blind arm's worst restart spent fourteen
iterations before it first cleared the baseline at all, which is the failure mode
the KB actually prevents. Search wall-clock falls 44%.

So the honest claim is not "the KB finds better models". Given 15 iterations,
random search over a sensible space usually finds a comparable config. The KB
removes the variance and the waste: it never has a bad restart, and it does not
spend two thirds of the budget below the line it started from.

For a 50-iteration / 6h budget that is the right kind of claim: the KB buys back
iterations for the axes that are still unexplored, rather than raising the score by
itself.

---

## 13. Honest summary of the headroom found

Against the official baseline (valid 0.6016 / test 0.5946), the best confirmed
configuration is BPR at k≈4–16, lr = 0.0002: **valid 0.6038, test 0.5980**.

In the calibration frame that is +0.0032 test on a reachable interval of 0.389
(random → oracle) — moving from 30.7% to 31.5% of it. It is real and replicated, and
it is small. The most useful thing this workstream can tell the agent is not a
technique but a posture: on this dataset the effect sizes are ~0.002, the seed noise
is ~0.0008, and **the discipline of replicating and controlling is worth more than
another architecture**. Two of the three axes that looked like wins here reversed
under their controls, and the one that survived did so only after replication.
