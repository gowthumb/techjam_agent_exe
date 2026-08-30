# Knowledge Base — KuaiRand-Pure

The distilled output of the ML-modelling workstream (Workstream 2 of
`KNOWLEDGE_BASE_PLAN.md`): a directive the autonomous ML-research agent reads to
decide **what to try next, in what range, how to tell it is going wrong, and when
to stop pushing an axis** — instead of searching blind.

The lab that produced it (code, the 300-run experiment log, tools) is in
[`../ml_modelling/`](../ml_modelling/).

---

## Files in this folder

| File | Audience | What it is |
|---|---|---|
| **`knowledge_base.yaml`** | **the agent** | The machine-readable directive. `yaml.safe_load` it; pin against `meta.schema_version` (currently **3**). This is the only file the agent parses. |
| `knowledge_base_rationale.md` | judges / humans | Why every KB entry exists (phases 0–10), each traced to the experiment that produced it. |
| `TIER12_RESULTS.md` | judges / humans | Same, for phases 11–17 (the Tier 1/2 improvement program). |
| `INTEGRATION_CONTRACT.md` | agent-brain owner | Proposed contract for *when in the loop* each KB section is read, log-schema alignment, and write-back. **Status: proposed, needs sign-off.** |

Verify at any time, from the repo root:

```bash
python ml_modelling/tools/kb_check.py     # every cited exp_id exists once; calibration matches the official file
python ml_modelling/tools/analyze.py --check   # experiment-log hygiene
```

---

## The task (pinned by the organizers — do not change)

Within-user ranking of each user's logged impressions. Label `long_view` (native
0/1 column; it is exactly `play_time_ms >= min(duration_ms, 18000)`). Metric
`mean(GAUC, nDCG@5)`. Scored on a hidden test set, once, on the converged result.
Scorer: `kuairand-starter-kit/evaluate.py` — **never reimplemented**.

Calibration (test primary): random **0.4753** · popularity **0.5715** · official
FM baseline **0.5946** · oracle ceiling **0.8645**. The baseline already holds
~31% of the reachable interval; judge progress against 0.8645, not 1.0.

---

## What the workstream did

~300 logged experiments across 22 axes, in two waves:

**Phases 0–10** (`knowledge_base_rationale.md`) — the original build. Reproduced
the baseline, then swept the loss function, capacity, L2, feature engineering,
multi-task auxiliary heads, GBDTs, behaviour sequences (DIN), temporal drift, and
the FM→FFM/DeepFM/AutoInt architecture ladder. Also Phase 4 (does the KB help?)
and Phase 5 (does it transfer to KuaiRand-1K? — no).

**Phases 11–17** (`TIER12_RESULTS.md`) — the Tier 1/2 program, five literature-
and data-grounded levers plus consolidation:

| Phase | Lever | Verdict |
|---|---|---|
| 11 | Unbiased evaluation via the randomly-exposed log | ✅ **method win** — a bias-overfitting veto; retired the embedding-noise lead; confirmed BPR at 4× the biased-split effect size |
| 12 | Sampled-softmax (InfoNCE) loss | ✅ **confirmed peer of BPR** — test 0.5984 ± 0.0004 vs 0.5980, 3–4× faster, good ensemble member |
| 13 | Watch-time regression head (CWM-style censored loss) | ❌ negative — a random-continuous-target control matches it exactly |
| 14 | Duration-regime + video-freshness features | ❌ regime neutral (shuffled control matches); `video_age` −0.0055 (train→test drift) |
| 15 | ESMM multiplicative `P(click)·P(lv\|click)` | ❌ neutral — beats pointwise, capped below BPR |
| 16 | Diverse rank-ensemble (`bpr+pointwise+ssm`) | ~+0.0001–0.0003 over the best single model |
| 17 | Distil into `knowledge_base.yaml` (→ v3) | ✅ |

**One lever landed as a peer, four are documented negatives — each closed with
its own control.** That is the plan's predicted outcome, and the discipline
(replicate, control, veto on the unbiased split) is the KB's central lesson:
effect sizes here are ~0.002, seed noise ~0.0008, and *two of the original three
"wins" reversed under their controls*.

---

## Headline recommendation

| model | test primary | Δ vs official baseline (0.5946) | notes |
|---|---|---|---|
| **BPR, k=6, lr=0.0002** | **0.5980** | **+0.0034** | the designated submission (`ml_modelling/experiments/make_submission.py`) — unchanged |
| SSM, lr=3e-4, τ=1, n=8 | 0.5984 | +0.0038 | confirmed BPR-peer; not a reason to switch the submission (tied on the selection split) |
| `bpr+pointwise+ssm` rank-ensemble | 0.5985 | +0.0039 | `submission_alt_test.csv`, the documented alternative |

The agent should **open on BPR (or SSM), tune `lr` within family, then rank-ensemble
BPR+SSM+pointwise** — and *not* escalate to more capacity, more features,
multi-task heads, watch-time modelling, or ESMM. `knowledge_base.yaml
architecture_ladder` is written as a **de-escalation** ladder for exactly this
reason: every capacity-increasing move in 300 experiments was flat or negative.

---

## How the agent consumes `knowledge_base.yaml`

Full contract in `INTEGRATION_CONTRACT.md`; in brief:

| Loop stage | KB sections read | Use |
|---|---|---|
| **Propose** (before each iteration) | `validated_search_space`, `priors`, `feature_engineering_menu`, `multi_task_signals`, `candidate_models`, `dead_ends` | pick the next config from a ranked menu; `dead_ends` is a hard filter — never propose those |
| **Execute** (on failure/anomaly) | `diagnostics` | symptom → cause → fix, so a blown run costs one iteration not three |
| **Reflect** (after each iteration) | `calibration`, `decision_protocol` | judge the score against the rungs and the **measured noise band**, not against 1.0; apply `decision_protocol.unbiased_veto` |
| **Escalate** (when plateaued) | `architecture_ladder`, `kb_ensemble` | decide whether to keep tuning, switch loss family, or ensemble — given iterations remaining |

`decision_protocol` is the section the reflection step **must not skip** — without
it the agent reads seed noise as progress. Key rules:
- **select on `valid`**; record `test`, `rand_valid`, `rand_test` alongside.
- a single-run `|delta| < 0.0016` is **not a result** (`harness.classify` applies this).
- anything within 0.0015 of the best gets **3+ seeds** before it is trusted or discarded.
- **`unbiased_veto`**: an intervention that gains on `valid` but drops ≥ 0.003 on
  `rand_valid` (the randomly-exposed log) is fitting the exposure policy — not shipped.
- run the **ablated control** before crediting any new mechanism.

---

## Scope warning

**Every value in `knowledge_base.yaml` is KuaiRand-Pure-only.** On KuaiRand-1K the
headline recommendation *loses* by −0.0152 (1K is an item-cold-start regime, not a
bigger Pure). The **method** travels — `decision_protocol`, the control rule, the
diagnostics, the unbiased veto — the numbers do not. See `scale_transfer` in the
YAML before applying this to any other benchmark; run the facts pass first
(`ml_modelling/experiments/p5_scale_transfer.py --stage facts`).

---

## Reproducing the Tier 1/2 phases

```bash
# from ml_modelling/
python experiments/p11_unbiased_eval.py
python experiments/p12_ssm_loss.py --stage grid   # then neg, control, replicate
python experiments/p13_watchtime.py --stage main  # then control
python experiments/p14_features.py --stage main   # then control, replicate
python experiments/p15_esmm.py --stage main
python experiments/p16_ensemble_final.py --members bpr,pointwise,ssm --write-submission
```
