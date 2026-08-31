# KuaiRand-Pure — Autonomous FM Recommender Pipeline

Companion to the generic [README.md](README.md) (starter-kit setup, task
definition, official baseline ladder). This file documents the **KuaiRand-Pure
track specifically**: how to set it up, how to reproduce every result claimed
for it, and an honest account of what's still missing. See
[README_1K.md](README_1K.md) for the separate KuaiRand-1K track — the two
benchmarks turned out to need materially different architectures and are
documented separately on purpose (see that file for why).

## Project overview

The task is within-user ranking on KuaiRand-Pure: for each user, rank that
user's own logged impressions by predicted `long_view` probability. The metric
is `primary = mean(GAUC, nDCG@5)` (`evaluate.py`, pinned, not to be edited).

On top of the starter kit's plain Factorization Machine baseline, this project
adds an **autonomous research pipeline** (`agent/`): a Planner/Coder/Debugger/
Executor loop where an LLM proposes one hypothesis per iteration (grounded in a
curated `knowledge_base/knowledge_base.yaml`), a Coder LLM translates it into a
surgical patch against the current best model, an isolated subprocess Executor
scores it on validation only (test metrics never leak into the search), and a
deterministic acceptance rule (must clear the measured seed-noise band, not
just "look better") decides whether to keep it. The loop runs until it
converges, hits an iteration/wall-clock cap, or exhausts its retry budget.

Every hypothesis this loop or the manual research phases before it produced is
logged with its rationale, config, and result — `knowledge_base/
knowledge_base_rationale.md` and `knowledge_base/TIER12_RESULTS.md` are the
full, re-checkable trail (`python ml_modelling/tools/kb_check.py` verifies
every number against `ml_modelling/experiments.jsonl`).

## Setup and installation

Requires Python 3.9+ and, for the plain baseline, **only numpy** — the
autonomous agent loop additionally needs an OpenAI-compatible chat completions
endpoint.

```bash
cd kuairand-starter-kit
python -m venv .venv
.venv\Scripts\activate            # Windows; `source .venv/bin/activate` elsewhere
pip install -r requirements.txt
```

Download and extract the data (Zenodo, no registration needed):

```bash
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz      # -> ./KuaiRand-Pure/data/
```

**Only needed to run the autonomous agent loop** (not the plain baseline or
the manual `ml_modelling/experiments/` scripts): create a `.env` file at the
`kuairand-starter-kit/` root with an OpenAI-compatible endpoint's credentials:

```
LLM_API_KEY=...
LLM_BASE_URL=...
LLM_MODEL=...
# optional per-role overrides and knobs:
# PLANNER_MODEL=... / CODER_MODEL=... / DEBUGGER_MODEL=...
# PLANNER_TEMPERATURE=... / CODER_TEMPERATURE=... / DEBUGGER_TEMPERATURE=...
# LLM_REQUEST_TIMEOUT_S=120 / LLM_RATE_LIMIT_BACKOFF_S=300
```

## Steps to reproduce results

**1. Sanity-check the official baseline** (~40s, CPU, single core):

```bash
python baseline.py --model fm
```

Expect test `GAUC 0.6610 | nDCG@5 0.5282 | primary 0.5946` — this is the
number every result below is measured against.

**2. Reproduce the confirmed, documented improvement** (BPR loss, the
designated submission — `knowledge_base.yaml`'s `candidate_models.fm_bpr`,
k=6, lr=2e-4). `ml_modelling/experiments/*.py` scripts are written to run with
that directory as the working directory, and write submissions to the repo
root (one level above `kuairand-starter-kit/`) — from `kuairand-starter-kit/`:

```bash
cd ../ml_modelling
python experiments/make_submission.py --split test    # writes ../submission_test.csv
cd ../kuairand-starter-kit
python submit.py --check ../submission_test.csv
```

This should land at test primary **0.5980** (+0.0034 over baseline). The full
comparison — BPR vs. SSM (0.5984) vs. a 3-model rank ensemble (0.5985,
`submission_alt_test.csv`, the documented alternative) — is in
`knowledge_base/TIER12_RESULTS.md`'s summary table.

**3. Reproduce the manual research trail** that produced the knowledge base
(one script per phase, each traceable to a `KNOWLEDGE_BASE_PLAN.md` question;
run from `ml_modelling/`, its own scripts' documented working directory):

```bash
cd ../ml_modelling
python experiments/p1_losses.py           # loss-function sweep (phase 1a)
python experiments/p11_unbiased_eval.py   # unbiased eval via the random-exposure log (phase 11)
cd ..
python ml_modelling/tools/kb_check.py     # verify every KB number against the experiment log (run from the repo root)
cd kuairand-starter-kit
```

**4. Run the autonomous agent loop itself** (requires the `.env` credentials
above):

```bash
python scripts/run_agent.py --bench pure
```

Iterates up to 50 hypotheses / 6h wall-clock, logging every attempt to
`runs/<run_id>/iterations.jsonl` and the running state to
`runs/<run_id>/state.json`; a final `summary.json` reports the best validation
config's held-out test score. `--skip-final-test` stops without reading test
metrics if you only want to inspect the search itself.

## Limitations & what I'd improve given more time

- **The single-run trap is real and recurring.** Several leads that looked
  like genuine improvements on one seed (e.g. Pure's own `k=1` and
  `loss_x_capacity` results) evaporated under 3-seed replication. The
  acceptance rule now gates on a measured noise band, but the autonomous
  loop still screens on a single seed before that gate — a built-in 2-seed
  screen (cheap on Pure, ~40s/run) would catch more false leads before they
  ever reach a human or get logged as "accepted."
- **The unbiased-evaluation check (Phase 11) is not wired into the automated
  loop.** It was decisive in one case (retiring a lead that looked flat on
  `valid` but moved on `test`) but currently only runs as a manual script
  (`p11_unbiased_eval.py`). Folding `explib/unbiased.py` into the Executor's
  acceptance path would make that check automatic instead of opt-in.
- **The Planner's knowledge-base context is Pure-shaped by default** — it
  only pulls in the 1K/27K operational runbooks when explicitly targeting
  those benchmarks (`bench != "pure"`). A Planner that could reason across
  benchmarks in one run (rather than one `bench` per invocation) would let it
  notice cross-benchmark patterns on its own instead of relying on a human to
  have already written them down.
- **GBDT and multi-task axes were explored manually, never through the
  agent loop.** The Coder/Debugger prompts constrain patches to the FM
  architecture (`run_fm(splits, ...)`); a model-family-agnostic harness would
  let the autonomous loop try architectures beyond FM without a human writing
  a new one-off script for each.

## Team contributions

This was a team submission; contributions below are reconstructed from commit
history (`git log`), not self-reported, so treat the boundaries as approximate:

| Contributor | Primary contributions |
|---|---|
| **gowthumb** (Gautham) | Initial KuaiRand-Pure baseline integration; orchestrator ↔ knowledge-base wiring; branch/repo integration work |
| **Harineesh Reddy** | Manual research program: Tier 1/2 phases (11-17), knowledge-base packaging and rationale docs |
| **YashR2005** | Built and hardened the autonomous agent orchestrator itself (`agent/`: planner, coder, debugger, executor, retries, offline test coverage) |
| **AI-assisted (Claude, via Claude Code)** | Exploration harness and Phases 0-10 of the manual research program; knowledge-base drafting; this file |

If you're grading this as a team submission, please confirm/correct this table
against your own records — it's inferred, not authoritative.
