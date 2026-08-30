# ML Modelling Workstream — the lab

Exploration behind the Knowledge Base for KuaiRand-Pure, per `KNOWLEDGE_BASE_PLAN.md`.

**The deliverable lives in [`../knowledge_base/`](../knowledge_base/)** —
`knowledge_base.yaml` (the directive the agent parses) plus its rationale docs and
the integration contract. Everything in *this* folder is the empirical work that
earned the right to write it: the code, the 300-run experiment log, the tools.

## Layout

| Path | What it is |
|---|---|
| `explib/dataset.py` | Cached loader keeping all 19 log columns. Row order verified identical to the starter kit's `data.load()`. |
| `explib/features.py` | Configurable categorical encoding. Reproduces the kit's 5 fields exactly; extra fields are a config change, not a fork. |
| `explib/fm.py` | The kit's FM with a pluggable loss (`pointwise` / `bpr` / `listwise` / `hybrid` / `ssm`). Only `dL/dz` differs between them. |
| `explib/history.py` | Causal / leave-one-out affinity features. Leakage contract documented in the module docstring. |
| `explib/harness.py` | Experiment logging + the pinned scorer + calibration rungs. |
| `explib/unbiased.py` | Randomly-exposed-video log as a bias-free eval split (Phase 11). `rand_valid` / `rand_test`, frozen encoder. |
| `explib/wtfm.py` | Two-head shared-embedding FM: `long_view` (binary) + watch ratio (one-sided/censored regression). Phase 13. |
| `explib/esmm.py` | Multiplicative decomposition `P(long_view) = P(click)·P(lv\|click)`. Phase 15. |
| `experiments/` | Runnable experiments. `sweep.py` is the general entry point; `p11`–`p16` are the Tier 1/2 program (`TIER12_RESULTS.md`). |
| `experiments.jsonl` | The experiment log — one JSON record per run. This is the raw material for the KB. |
| `logs/` | Raw stdout per run, for anything the structured record does not capture. |
| `cache/` | Parsed-log npz cache (gitignored, rebuilt on demand). |
| `tools/analyze.py` | Per-axis view of the log + hygiene checks (duplicate ids, failed runs). |
| `tools/kb_check.py` | Validates every KB claim against the log and the official scores. |

## Deliverables — in [`../knowledge_base/`](../knowledge_base/)

| File | What it is |
|---|---|
| `knowledge_base/knowledge_base.yaml` | **The deliverable.** The machine-readable directive the agent parses. |
| `knowledge_base/knowledge_base_rationale.md` | Why each entry exists (phases 0–10), traced to the experiment behind it. |
| `knowledge_base/TIER12_RESULTS.md` | Same, for the Tier 1/2 phases (11–17). |
| `knowledge_base/INTEGRATION_CONTRACT.md` | Proposed KB-to-agent interface (Phase 3); needs sign-off. |
| `knowledge_base/README.md` | Entry point: what the workstream did, how to consume the KB, how to verify it. |

Produced here in the lab:

| File | What it is |
|---|---|
| `experiments/make_submission.py` | Trains the recommended config (BPR k=6) and validates it with the official `submit.py`. Unchanged by Tier 1/2. |
| `submission_alt_*.csv` (repo root, gitignored) | The Phase 16 ensemble, written alongside the BPR submission — not a replacement. |

## Ground rules

1. **`evaluate.py` is never reimplemented.** `harness.score` calls the kit's scorer.
2. **Decisions are made on `valid`.** `test` is recorded for calibration only — it stands
   in for a hidden set we would not have. From Phase 11 on, `rand_valid` (the
   randomly-exposed log) is also recorded as a bias-overfitting *veto*: an
   intervention that gains on `valid` but drops ≥ 0.003 on `rand_valid` is not shipped.
3. **A delta below 0.0016 is not a result.** That is 2x the official 0.0008 seed std;
   `harness.classify` applies it so verdicts are not eyeballed.
4. **Progress is measured against the oracle**, not 1.0. Reachable interval is
   random 0.4753 -> oracle 0.8645 on test; the baseline already holds 30.7% of it.

## Running things

```bash
python explib/dataset.py                    # verify row order vs the starter kit
python experiments/p1_losses.py             # axis A: loss function
python experiments/sweep.py --help          # general sweep runner
python -c "import sys;sys.path.insert(0,'.');from explib import harness;print(harness.summarize())"
```
