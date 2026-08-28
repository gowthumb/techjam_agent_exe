# ML Modelling Workstream

Exploration and Knowledge Base build for KuaiRand-Pure, per `KNOWLEDGE_BASE_PLAN.md`.
The deliverable is `knowledge_base.yaml` + `knowledge_base_rationale.md`; everything
else here is the empirical work that earns the right to write them.

## Layout

| Path | What it is |
|---|---|
| `explib/dataset.py` | Cached loader keeping all 19 log columns. Row order verified identical to the starter kit's `data.load()`. |
| `explib/features.py` | Configurable categorical encoding. Reproduces the kit's 5 fields exactly; extra fields are a config change, not a fork. |
| `explib/fm.py` | The kit's FM with a pluggable loss (`pointwise` / `bpr` / `listwise` / `hybrid`). Only `dL/dz` differs between them. |
| `explib/history.py` | Causal / leave-one-out affinity features. Leakage contract documented in the module docstring. |
| `explib/harness.py` | Experiment logging + the pinned scorer + calibration rungs. |
| `experiments/` | Runnable experiments. `sweep.py` is the general entry point. |
| `experiments.jsonl` | The experiment log — one JSON record per run. This is the raw material for the KB. |
| `logs/` | Raw stdout per run, for anything the structured record does not capture. |
| `cache/` | Parsed-log npz cache (gitignored, rebuilt on demand). |

## Ground rules

1. **`evaluate.py` is never reimplemented.** `harness.score` calls the kit's scorer.
2. **Decisions are made on `valid`.** `test` is recorded for calibration only — it stands
   in for a hidden set we would not have.
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
