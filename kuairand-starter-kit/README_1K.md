# KuaiRand-1K — Autonomous FM Recommender Pipeline

Companion to the generic [README.md](README.md) and to
[README_PURE.md](README_PURE.md). This file documents the **KuaiRand-1K track
specifically** — a deliberately separate document because 1K is not "Pure but
bigger": it's a different sampling regime (item cold-start, 1,000 users' full
histories over a 4.37M-video catalog, vs. Pure's 27K users over a 7.5K-video
snapshot) that required a different architecture decision and produced a
different, and much more negative, headline finding. KuaiRand-27K was
originally in scope as a further scale-transfer confirmation but is currently
out of scope for this submission — see **Limitations** below.

## Project overview

Same task and metric as Pure (within-user ranking, `primary = mean(GAUC,
nDCG@5)`), same autonomous Planner/Coder/Debugger/Executor loop
(`agent/`) — but retargeted at KuaiRand-1K via a `bench` parameter threaded
through the whole pipeline, because 1K's scale rules out the plain baseline's
implementation choices:

- **Dense Adam is infeasible.** `data.py`'s string-keyed encoder and
  `baseline.py`'s dense-Adam FM update are `O(vocab)` per batch — free at
  Pure's ~40K vocab, but 1K's encoder dimension is ~2.9M. `data_1k.py` uses a
  vectorized int-fast-path encoder instead
  (`ml_modelling/explib/features.py`), and `baseline_1k.py` implements sparse
  Adam unconditionally, not as a tunable.
- **The confirmed result is negative, and that's the finding.** Four research
  phases and roughly a dozen distinct axes (BPR, sampled softmax, three
  causal-affinity fields, k/lr sweeps, embedding-noise regularization, two
  pairwise-GBDT objectives) were tested against 1K's own untuned pointwise
  baseline. **None beat it.** `knowledge_base/ONEK_RESULTS.md` is the full,
  re-checkable record of that search — not a failure of effort, but a genuine
  property of an 85%-cold-item test set: there's no trained embedding to rank
  *from* and little within-user signal to normalize against, which is exactly
  what every fancier lever in the knowledge base depends on.

Given that, the autonomous loop's job on this benchmark isn't "apply the same
tricks that worked on Pure" — the Planner is explicitly told not to
(`agent/coder.py`'s hard constraints on this benchmark, plus the full text of
`ONEK_RESULTS.md` injected into every planning prompt) — it's to either
confirm the baseline still stands, or find one of the few genuinely untested
directions the research already identified (see **Limitations**).

## Setup and installation

Same base setup as [README_PURE.md](README_PURE.md) — Python 3.9+,
`pip install -r requirements.txt` from `kuairand-starter-kit/`, and the same
`.env` LLM credentials, needed only for the autonomous loop, not for a plain
baseline run.

Download and extract KuaiRand-1K (Zenodo, no registration):

```bash
wget https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz
tar xzf KuaiRand-1K.tar.gz        # -> ./KuaiRand-1K/data/
```

No separate install for the 1K-specific data layer: `data_1k.py` imports
`ml_modelling/explib`, a sibling directory already part of this repository —
nothing outside `requirements.txt` is needed.

## Steps to reproduce results

**1. Sanity-check the confirmed baseline** (sparse Adam, k=16, lr=1e-3;
~1min load + ~30-45s/epoch on CPU):

```bash
python baseline_1k.py
```

Expect valid primary close to **0.6439** (± 0.0022 over 3 seeds, per
`ONEK_RESULTS.md` Phase 5) and test primary close to **0.6380** (± 0.0021).
Single-seed runs land inside this band, not exactly on it — that spread *is*
the finding (1K's seed noise is ~4x Pure's).

**2. Reproduce the manual research trail** that established the baseline is
already optimal (each traceable to a phase in `ONEK_RESULTS.md`). These
scripts are written to run with `ml_modelling/` as the working directory, not
`kuairand-starter-kit/`:

```bash
cd ../ml_modelling
python experiments/p5_scale_transfer.py --bench 1k --stage facts     # dataset shape, cheap
python experiments/p5_scale_transfer.py --bench 1k --stage headto2   # baseline vs. Pure's BPR recipe
python experiments/p10_1k_tune.py                                    # tuning 1K's own baseline
python experiments/p18_1k_extend.py --stage ssm                      # SSM loss, untested on 1K until here
python experiments/p19_1k_gbdt.py --models xgb_pairwise,xgb_ndcg     # pairwise GBDT objectives
cd ../kuairand-starter-kit
```

**3. Run the autonomous "maximize 1K" loop**:

```bash
python scripts/maximize_1k.py
```

This runs the full Planner/Coder/Debugger/Executor loop against 1K (up to 50
iterations / 6h — the entire budget, none of it held back for 27K), with the
Planner's prompt carrying `knowledge_base/SCALE_DIRECTIVES.md` — a ~3KB
condensed distillation of `ONEK_RESULTS.md` and `HARDWARE_AWARENESS.md`'s
operational directives, injected every iteration in place of the ~63KB an
earlier version of this pipeline injected in full (see `ONEK_RESULTS.md`'s
"Token-usage pass" for the measured before/after). If any candidate is
accepted, the script automatically re-runs that exact code over 3 seeds
(`--replication-seeds 0,1,2` by default) before reporting it as a genuine
result. The search-time acceptance band
(`agent/executor.py::_ACCEPTANCE_BAND["1k"]`) is 0.0016, the same as Pure's —
after briefly widening it to 0.032, a follow-up run showed that band was
calibrated to a magnitude 1K has never produced (best delta +0.0034 over 10
genuine attempts), while 1K's one confirmed win was itself only +0.0018; see
`ONEK_RESULTS.md`'s acceptance-band history for the full account. The real
defense against a false single-seed lead is the mandatory 3-seed replication
itself, not this screen — it already caught two false leads in this
codebase's own history (Phase 10's `lr_0.0005`, Phase 19's `xgb_ndcg`, both
documented in `ONEK_RESULTS.md`). Results land in
`runs/<run_id>/{iterations.jsonl, state.json, maximize_1k_report.json}`; the
last of those states plainly whether anything was confirmed or whether the
baseline stands.

## Limitations & what I'd improve given more time

- **Across ~12 tested axes, nothing beat the untuned baseline.** This is a
  real, well-controlled finding, not an unfinished search — but it does mean
  there is currently no confirmed way to *improve* 1K's score, only to
  confirm it holds. Given more time, the two directions
  `ONEK_RESULTS.md` explicitly calls out as genuinely untested (not just
  negative) are:
  - **CatBoost's YetiRank with the `cat_features` bug fixed.** Phase 19's
    attempt never completed a full run (it was running effectively
    single-threaded due to a real bug — categorical columns were being
    quantized as continuous floats) and was killed before producing a result,
    not disproven.
  - **Content/side-information features that don't require a trained ID
    embedding.** Every axis tried so far still routes through an FM embedding
    table, which by construction has nothing to say about a video the model
    never saw in training (85% of test videos). A hybrid model scoring cold
    items on content features alone has a real structural argument that
    nothing tested so far had.
- **KuaiRand-27K is currently out of scope.** It was meant to be a further
  scale-transfer confirmation of whatever won on 1K, but this environment's
  local `KuaiRand-27K.tar.gz` and extracted data directory are incomplete
  (missing the log files training needs; see
  `knowledge_base/HARDWARE_AWARENESS.md` rule 6) — a data problem, not a
  pipeline one. The bench-aware plumbing for it (`data_27k.py`,
  `baseline_27k.py`, `agent/runner.score_confirm`) is built and smoke-tested
  but unused; resuming it needs a correctly re-fetched archive, not new code.
- **The replication gate compares against a single fixed baseline number**,
  recomputed once at the start of each `maximize_1k.py` run rather than
  re-measured per candidate. If 1K's seed noise turns out to vary by config
  (plausible, given how much larger it already is than Pure's), a
  config-specific noise estimate would be a more honest gate than one shared
  band.
- **The Planner has no memory across separate `maximize_1k.py` invocations**
  beyond what's logged in `knowledge_base.yaml` and `ONEK_RESULTS.md` — a
  rejected hypothesis from one run isn't automatically prevented from being
  re-proposed, near-identically, in the next. A persistent "already tried on
  1K" ledger, distinct from the Pure-oriented `dead_ends` section, would
  close that gap.

## Team contributions

Same team as [README_PURE.md](README_PURE.md); contributions specific to this
track, reconstructed from commit history:

| Contributor | Primary contributions |
|---|---|
| **Harineesh Reddy** | Led the 1K-parity research program (Phases 5, 10, 18, 19 in `ONEK_RESULTS.md`) and the KuaiRand-27K scale-transfer extension (Phase 20) |
| **gowthumb** (Gautham) | 27K data acquisition and integration work |
| **YashR2005** | Autonomous agent orchestrator this track's loop builds on (`agent/`) |
| **AI-assisted (Claude, via Claude Code)** | This session's work: the bench-aware `data_1k.py`/`baseline_1k.py` pipeline, threading `bench` through the orchestrator, the Planner's `ONEK_RESULTS.md`/`HARDWARE_AWARENESS.md` context injection, `scripts/maximize_1k.py`, and this file |

If you're grading this as a team submission, please confirm/correct this table
against your own records — it's inferred, not authoritative.
