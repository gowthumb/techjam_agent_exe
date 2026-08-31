# KuaiRand Recommender — ML-modelling branch

Within-user ranking on KuaiRand: given a user's logged impressions, rank them so
the videos the user actually watched past the "long view" threshold sort to the
top. Label `long_view` (native 0/1 column, exactly `play_time_ms >=
min(duration_ms, 18000)`). Metric `mean(GAUC, nDCG@5)`, scored on a hidden test
set once, on the converged result. The original task brief, hackathon rules
(50 iterations / 6h wall-clock per benchmark), and the full phased plan this
branch executed are in [`KNOWLEDGE_BASE_PLAN.md`](KNOWLEDGE_BASE_PLAN.md).

## Repo map

| Path | What it is |
|---|---|
| [`kuairand-starter-kit/`](kuairand-starter-kit/) | The organizers' code (`baseline.py`, `evaluate.py`, `submit.py`, `data.py`) and datasets. Scorer is never reimplemented — every result in this branch is validated against it. |
| [`knowledge_base/`](knowledge_base/) | **The deliverable.** `knowledge_base.yaml` — a machine-readable directive an autonomous ML-research agent reads to decide what to try next, in what range, and when to stop — plus the rationale docs explaining every entry. |
| [`ml_modelling/`](ml_modelling/) | **The lab that produced it.** 24 experiment scripts, a ~300-run structured experiment log, and the tools that validate the KB against it. |

Start with [`knowledge_base/README.md`](knowledge_base/README.md) for the full
results narrative, or [`ml_modelling/README.md`](ml_modelling/README.md) for how
the lab is organized.

## Headline result

| config | test primary | Δ vs official baseline (0.5946) | |
|---|---|---|---|
| Official FM baseline | 0.5946 | — | organizers' baseline |
| **BPR, k=6, lr=0.0002** | **0.5980** | **+0.0034** | **the designated submission** (`ml_modelling/experiments/make_submission.py`, unchanged throughout) |
| `bpr + pointwise + ssm` rank-ensemble | 0.5985 | +0.0039 | documented alternative (`submission_alt_test.csv`), **not** the designated submission — the two are statistically tied on the selection split, and BPR is the simpler, single-model config |

Calibration: random 0.4753 → popularity 0.5715 → official baseline 0.5946 →
**BPR 0.5980** → oracle ceiling 0.8645. Progress is judged against the reachable
interval (baseline already holds ~31% of it going in), not against 1.0.

## Scope beyond the required benchmark

KuaiRand-Pure is the primary/required benchmark and the only one the designated
submission is scored on. Two bonus benchmarks were also attempted, full results
in [`knowledge_base/ONEK_RESULTS.md`](knowledge_base/ONEK_RESULTS.md):

- **KuaiRand-1K** — a fundamentally different regime (item cold-start, not
  Pure's warm-ID ranking). Four phases tested Pure's full toolkit against it —
  every candidate lost to 1K's own untuned baseline. Documented as a rigorous
  negative result, not a gap in the search.
- **KuaiRand-27K** — the largest bonus benchmark (322M rows, a 9.9GB download).
  1K's confirmed config was transferred as-is and ran clean, confirming the same
  cold-start regime at one more order of magnitude.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # or your usual env setup
pip install -r requirements.txt
```

Fetch KuaiRand-Pure's data (organizers' Zenodo mirror — see
[`kuairand-starter-kit/README.md`](kuairand-starter-kit/README.md) for the
exact `wget`/`tar` steps; 1K and 27K are optional, only needed to touch the
bonus-benchmark work).

## Reproduce / verify

```bash
python kuairand-starter-kit/baseline.py --model fm     # organizers' own baseline, ~40s CPU
python ml_modelling/experiments/make_submission.py     # trains the designated BPR submission, validates with submit.py
python ml_modelling/tools/kb_check.py                  # every KB claim checked against the experiment log + official scores
python ml_modelling/tools/analyze.py --check           # experiment-log hygiene (duplicate ids, failed runs, id readability)
```

Before running anything on a benchmark for the first time — especially 1K or
27K — read
[`ml_modelling/HARDWARE_AWARENESS.md`](ml_modelling/HARDWARE_AWARENESS.md):
what to check about the machine and the dataset's scale before deciding CPU vs
GPU, dense vs sparse optimization, or whether a feature matrix is safe to build
in memory.

## Data license

KuaiRand datasets are released by the organizers under CC BY-SA 4.0 — see
[kuairand.com](https://kuairand.com) for terms and the full dataset description.
