# ML Modeling Workstream — Exploration & Knowledge Base Build Plan

Scope: this covers **Workstream 2 (ML/Recommender Modeling)** from the project split. Goal is two-fold:
1. **Explore** the model/feature/architecture space on KuaiRand-Pure yourself, empirically.
2. **Distill** what you learn into a structured **Knowledge Base (KB)** — a directive the autonomous agent reads to decide what to try, in what order, and why, instead of searching blind.

The KB is the actual deliverable of this workstream. The exploration is how you earn the right to write it.

**Priority note:** KuaiRand-Pure is required and determines 100% of the primary score — it comes first, always. KuaiRand-1k and KuaiRand-27k are bonus benchmarks: same task, same metrics (GAUC/nDCG@5), but attempting them only adds bonus points on top of the primary score and never subtracts from it if skipped. Treat the bonus benchmarks as an **extension** of this plan (see Phase 5) that only starts once the Pure KB is solid — don't split exploration time across all three datasets from day one.

---

## Phase 0 — Setup & Sanity Check

Before exploring anything, make sure your numbers are trustworthy.

- [ ] Reproduce the official baseline exactly: `python3 baseline.py --model fm` → confirm validation GAUC 0.6674 / nDCG@5 0.5357 / primary 0.6016 (and hidden-test 0.6610 / 0.5282 / 0.5946 if you can check it once).
- [ ] Confirm your local `evaluate.py` matches the pinned conventions (zero-positive users → nDCG 0, GAUC excludes all-positive/all-negative users, weighted by positive count).
- [ ] Set up your **own experiment log** now, in the same shape the agent will eventually use (hypothesis / diff / metrics / notes). You are dogfooding the harness — this log becomes raw material for the KB, and later doubles as a sanity check that your findings match what the agent rediscovers on its own.
- [ ] Note the calibration rungs so every result you produce is judged against them, not against 1.0: random 0.4753 → popularity 0.5715 → baseline 0.5946 → ceiling 0.8645.

**Deliverable:** a working local loop that reproduces the baseline and logs experiments in a structured, greppable format (CSV/JSON, not just notebook cells).

---

## Phase 1 — Exploration (build the empirical search space)

Run these as **your own hypothesis-driven experiments**, not a blind sweep. For each, write down: what you expected, what happened, and why — this narrative is exactly what turns into KB entries later.

### 1a. Core FM hyperparameters
- Latent dimension `k` — sweep beyond the baseline's 16 (e.g. 8/16/32/64/128). Watch for the overfitting knee (train/val gap widening).
- Regularization (L2 on factor vectors) — find the point where it stops helping / starts hurting.
- Learning rate — baseline uses 0.001; check sensitivity.
- Optimizer — compare plain SGD (baseline) against FTRL and Adagrad, which are usually better-suited to sparse categorical features.

### 1b. Feature engineering
- Try bucketing/hashing for high-cardinality fields.
- Try explicit second-order crosses (user × category, etc.) beyond what FM does implicitly.
- Test each of the 5 categorical fields' individual contribution (ablate one at a time) to know which matter.

### 1c. Multi-task / auxiliary signals
- KuaiRand logs 12 feedback signals; only `long_view` is scored, but others (click, like, follow, play_time…) are fair game as auxiliary tasks.
- Try a simple shared-embedding multi-task setup (ESMM-style) and measure whether it improves `long_view` prediction over single-task FM.
- Note which auxiliary signals help vs. which just add noise or fight the primary task (the "seesaw" problem from the primer).

### 1d. Architecture ladder
- Don't limit exploration to FM → FFM → DeepFM. The hackathon allows any free-licensed open-source model or framework, so treat this as a genuine search across categories: FM-family (FFM, DeepFM, xDeepFM, DCN/DCN-V2, AutoInt, FiBiNET), behavior/sequence-aware (DIN, DIEN), and gradient-boosted trees (LightGBM, XGBoost, CatBoost) as a parallel, cheaper-to-train branch. See the **Candidate Model Shortlist** in Phase 2 for the full list with license and fit notes.
- For each option tried, record the cost (training time, iterations to converge, token/compute cost if LLM-assisted) against the accuracy gain — this feeds directly into the agent's decision of *when* it's worth escalating complexity given the 50-iteration / 6h budget.
- Don't assume "more complex = better" — GBDTs in particular are worth a real trial given the dataset size (Pure is 1.4M rows), since a strong GBDT result that trains fast helps the Feasibility & Practicality resource-consumption tiering as much as it helps the primary metric.

### 1e. Failure modes
- Deliberately try a couple of bad configs (too-high LR, `k` too large with no regularization, degenerate feature encoding) and record what the failure *looks like* (NaN loss, exploding embedding norms, val score cratering). This becomes the agent's diagnostic playbook.

**Deliverable:** a filled-out experiment log covering 1a–1e, each entry with hypothesis → result → takeaway.

---

## Phase 2 — Distill Findings into the Knowledge Base

This is where exploration becomes a **directive**, not just a log. The KB should answer, for the agent, four questions at any point in its loop: *what's worth trying next, what range should I search in, how do I know if something's going wrong, and when should I stop pushing on this axis.*

### KB sections to write

1. **Validated search space** — not "here's every possible value" but "here's the range that actually mattered," with the boundary values that caused problems, drawn straight from Phase 1a.
2. **Priors / rules of thumb** — conditional heuristics, e.g. *"if train/val gap exceeds X, increase regularization before increasing k"* or *"FTRL outperforms SGD on this feature set beyond N iterations."* These come from patterns you saw repeat across experiments.
3. **Feature engineering menu** — ranked list of transforms/crosses with their observed effect size, so the agent tries high-value ones first instead of randomly.
4. **Multi-task task menu** — which of the 12 signals helped, which didn't, and any interaction effects you noticed.
5. **Candidate model shortlist** — the licensed model options beyond the baseline, categorized by fit, so the agent has a menu to choose from instead of only ever escalating along a fixed FM→FFM→DeepFM path. See table below.
6. **Architecture decision ladder** — a decision rule: given iterations/time remaining and current score vs. baseline, when should the agent escalate to a more complex model (or switch category, e.g. to a GBDT) rather than keep tuning the current one.
7. **Diagnostic playbook** — symptom → likely cause → fix, built from Phase 1e (e.g. NaN loss → LR too high → halve it and retry).
8. **Calibration reference** — the fixed rungs (random/popularity/baseline/ceiling) so the agent's reflection step can contextualize any score it produces.
9. **Scale-transfer directives** — which Pure-derived findings are expected to hold as-is at larger scale vs. which need re-validation before the agent trusts them on KuaiRand-1k/27k (see Phase 5). Left empty until Phase 5 exploration happens; the KB schema should reserve space for it from the start so the agent-brain owner can build the read logic once.

### Candidate model shortlist

All licenses below were checked directly against source (LICENSE files / package registries), not assumed. Re-verify at submission time in case anything changes.

| Category | Model(s) | Library | License | Fit for this task |
|---|---|---|---|---|
| FM-family upgrade | FFM, DeepFM, xDeepFM, DCN/DCN-V2, AutoInt, FiBiNET, PNN, NFM, AFM | DeepCTR-Torch | Apache 2.0 | Direct next step from the baseline FM; DeepCTR-Torch ships all of these ready to run on tabular categorical data like KuaiRand's fields. |
| FM-family upgrade | Same categories, broader model zoo | FuxiCTR | Apache 2.0 | Alternative implementation, useful for cross-checking DeepCTR-Torch results or if a specific paper's exact config matters. |
| FM-family upgrade | 94 algorithms incl. FM/FFM/DeepFM/AutoInt etc. | RecBole | MIT (LICENSE file) | Large model zoo; note the repo's "About" page separately states academic-purpose-only wording alongside the MIT license — worth a quick check against event rules if that distinction matters. |
| Behavior/sequence-aware | DIN, DIEN | DeepCTR-Torch / RecBole | Apache 2.0 / MIT | Uses attention over a user's interaction history rather than flat features — a fit if you build sequences from the 12 feedback signals rather than treating them as static fields. |
| Multi-task | MMOE, PLE, ESMM, SharedBottom | DeepCTR-Torch | Apache 2.0 | Purpose-built for the multi-task exploration in 1c; PLE specifically targets the "seesaw" problem the primer flags. |
| Gradient-boosted trees | LightGBM | LightGBM | MIT | Fast, strong tabular baseline; cheap to iterate on within the compute budget; good candidate for the "low consumption" Feasibility tier. |
| Gradient-boosted trees | XGBoost | XGBoost | Apache 2.0 | Similar profile to LightGBM; worth comparing both since they sometimes trade places on sparse categorical data. |
| Gradient-boosted trees | CatBoost | CatBoost | Apache 2.0 | Handles categorical features natively without manual encoding — may reduce the feature-engineering burden from 1b. |
| Large-scale specialist (bonus benchmarks) | DLRM | TorchRec | BSD | Only relevant if attempting KuaiRand-27k (Phase 5) — designed for the sparse, large-scale regime that benchmark sits in. |

**Deliverable:** the table above (or its equivalent) copied into the `candidate_models` block of `knowledge_base.yaml`, trimmed down to whichever options you actually validated in Phase 1d — don't ship untested entries as if they were recommendations.



### Suggested KB format

Keep it as **two paired files**: a machine-readable directive the agent parses directly, plus a human-readable rationale doc for judges/teammates. Example skeleton for the machine-readable side:

```yaml
# knowledge_base.yaml
hyperparameters:
  k:
    validated_range: [16, 64]
    danger_zone: ">96 without added regularization: overfits within 8 iterations"
    default_start: 32
  regularization:
    validated_range: [0.001, 0.05]
    rule: "increase before increasing k if train/val gap > 0.01"
  optimizer:
    preferred: ftrl
    fallback: adagrad
    avoid: "plain sgd beyond iteration 15 — plateaus early"

feature_engineering:
  - name: user_x_category_cross
    effect: "+0.004 primary, cheap"
    priority: high
  - name: hashed_high_cardinality_bucketing
    effect: "marginal, adds preprocessing cost"
    priority: low

multi_task_signals:
  helpful: [click, play_time]
  neutral: [comment]
  harmful_or_noisy: [forward]

candidate_models:
  - name: lightgbm
    category: gbdt
    license: mit
    library: lightgbm
    validated: false  # set true once tried in Phase 1d
  - name: deepfm
    category: fm_family
    license: apache-2.0
    library: deepctr-torch
    validated: false
  - name: din
    category: sequence_aware
    license: apache-2.0
    library: deepctr-torch
    validated: false
  - name: ple
    category: multi_task
    license: apache-2.0
    library: deepctr-torch
    validated: false

architecture_ladder:
  - stage: fm
    escalate_if: "plateaued (ε=0.002/N=3) with >15 iterations budget remaining"
  - stage: ffm_or_gbdt
    escalate_if: "plateaued again with >10 iterations budget remaining — branch to GBDT (lightgbm) if feature engineering headroom looks exhausted, else ffm"
  - stage: deepfm_or_sequence_model
    escalate_if: "rare — only if iterations remaining > 20 and prior stages show headroom; prefer din/dien if behavior-sequence features were productive in 1b/1c, else deepfm"

diagnostics:
  - symptom: "loss = NaN"
    cause: "learning rate too high or unnormalized feature"
    fix: "halve LR, check feature scaling"
  - symptom: "val score flat from iteration 1"
    cause: "feature set likely uninformative or duplicated"
    fix: "re-check feature encoding pipeline"

calibration:
  random: 0.4753
  popularity: 0.5715
  official_baseline: 0.5946
  ceiling: 0.8645

scale_transfer:
  derived_from: kuairand_pure
  expected_invariant: ["optimizer_choice", "architecture_escalation_logic", "diagnostic_playbook"]
  needs_revalidation: ["k", "regularization", "iteration_budget_pacing"]
  status: "pending — fill in after Phase 5 exploration on 1k/27k"
```

The narrative rationale doc (markdown) should walk through *why* each entry exists — cite the specific experiment(s) from your Phase 1 log that justify it. This is what makes the KB defensible under Innovation & Problem Insight judging, not just a config dump.

**Deliverable:** `knowledge_base.yaml` + `knowledge_base_rationale.md`, both versioned in the repo.

---

## Phase 3 — Integration Contract with the Agent Brain

Coordinate with whoever owns the agent brain workstream on:

- **When the KB is read** — at proposal time (before each iteration), and whether the agent can also consult it mid-run during the reflection step.
- **Whether the agent can *write back*** — e.g. append newly-discovered rules to the KB as it runs, so the KB grows during the competition run itself, not just from your offline exploration.
- **Log schema alignment** — make sure your Phase 0 experiment log format matches what the harness owner is using, so your findings and the agent's run logs are directly comparable.

**Deliverable:** a short interface note (even a paragraph) confirming how/when the KB is consumed, agreed with the agent-brain owner.

---

## Phase 4 — Validate the KB Actually Helps

Don't assume the KB improves things — check it, the same way the challenge asks the agent to check its own progress.

- [ ] Run the agent loop **without** the KB (blind search) for a fixed iteration budget.
- [ ] Run it **with** the KB for the same budget.
- [ ] Compare: score reached, iterations to convergence, number of failed/wasted trials.

This ablation is also good evidence for the write-up — "the agent converged in N iterations with the KB vs. M without" is a concrete, judge-legible claim for both Innovation and Feasibility scoring.

**Deliverable:** a short before/after comparison you can drop straight into the results write-up.

---

## Phase 5 — Bonus Benchmark Strategy (KuaiRand-1k / KuaiRand-27k)

Only start this once Phase 4 is done and the Pure KB is stable — bonus points are worthless if they come at the cost of a shaky required submission.

### Why this needs its own phase, not just "run the same agent on a bigger file"

The scale jump is severe: Pure is 1.4M interactions (27K users × 7.6K items), 1k is 11.7M (~8x), and 27k is 322M (~230x Pure). The compute budget (50 iterations / 6h wall-clock) applies **per benchmark run**, so a single iteration on 27k can be dramatically more expensive than on Pure — the KB's pacing assumptions (e.g. "escalate architecture if >15 iterations remain") may not hold when each iteration eats far more of the 6h ceiling.

Scale also changes what's actually optimal, not just what's affordable:
- **Regularization** typically needs to *shrink* as data grows — the overfitting pressure that shaped your Pure-tuned values may not exist at 1k/27k scale.
- **`k` (latent dimension)** can often go *larger* without overfitting once there's more data to support it.
- **Feature sparsity** patterns shift — fields that were high-cardinality-and-noisy on Pure may behave differently with more observations per ID.

### Recommended approach

1. **Validate transfer on 1k first**, not 27k. 11.7M rows is a far more tractable place to test whether the Pure KB's priors hold before committing budget to the much heavier 27k run.
2. Run a handful of experiments applying the Pure KB's starting values as-is on 1k. Check which of the `needs_revalidation` entries (k, regularization, iteration pacing) actually needed adjustment, and by how much.
3. Update the `scale_transfer` block in the KB with real findings — turn `status: pending` into concrete rules (e.g. "regularization: halve Pure's value per order-of-magnitude increase in row count").
4. **Only attempt 27k if 1k results are positive and there's remaining time/compute budget.** Given the size, treat 27k as an efficiency problem as much as a modeling one — consider whether the agent should search on a subsample and only train the final candidate on the full 322M rows, and whether it's worth skipping the architecture ladder (FFM/DeepFM) entirely at this scale in favor of a well-tuned plain FM that reliably finishes inside the 6h ceiling.
5. Log 1k/27k experiments the same way as Phase 1 — these entries feed the `scale_transfer` KB section, not a separate document.

**Deliverable:** `scale_transfer` section of the KB filled in with real values (not placeholders), plus a short note on whether 27k was attempted and why.

---

## Rough Milestone Order

1. Baseline reproduced + experiment log format set up
2. Core hyperparameter sweep (1a) done, findings logged
3. Feature engineering + multi-task exploration (1b, 1c) done
4. Architecture ladder experiments (1d) + failure-mode notes (1e) done
5. First draft of `knowledge_base.yaml` + rationale doc
6. Integration check with agent-brain owner
7. With/without-KB ablation run
8. KB refined based on ablation results — **Pure KB locked/stable at this point**
9. *(Bonus, time permitting)* Transfer-validate KB priors on KuaiRand-1k, fill in `scale_transfer` section
10. *(Bonus, only if 9 succeeds and budget remains)* Attempt KuaiRand-27k with efficiency-adjusted strategy

Treat step 5 onward as iterative, not one-shot — the KB should keep absorbing new findings as long as you're still exploring, right up until the point the agent-brain owner needs a stable version to build against. Steps 9–10 are strictly additive: never let bonus-benchmark work delay or destabilize the Pure KB, since Pure is the only piece that determines the primary score.
