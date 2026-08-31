# Autonomous ML Research Agent for KuaiRand Recommender Systems

TikTok TechJam 2026 — Track 2: Autonomous ML Research Agent for Recommender Systems.

An LLM-driven agent (Planner → Coder → Executor → Debugger loop) that autonomously proposes, implements, tests, and iterates on modeling hypotheses for the KuaiRand benchmark family, with no manual intervention required to reach a converged, improved-over-baseline result.

## Approach

- **Planner**: proposes a modeling hypothesis each iteration, grounded in a knowledge base of prior results (replication-disciplined: confirmed wins, controls, and known non-transfers are surfaced first).
- **Coder**: implements the hypothesis as a Search/Replace diff against the current best candidate.
- **Executor**: deterministically applies the diff, trains, and scores on validation — accepts or rejects based on whether it beats the current best.
- **Debugger**: repairs failed patches (bounded retries) so a malformed diff doesn't crash the run.
- Convergence: validation primary score must not improve by more than ε=0.002 over the last N=3 iterations, and at least one accepted improvement must exist first. Hard caps: 50 iterations or 6h wall-clock, whichever comes first.
- One-time held-out test scoring only after convergence — the agent never accesses test data during iterative development (no-snooping discipline).

Model: Factorization Machine (baseline), extended via loss-function and architecture variants proposed by the agent across iterations.

## Results

### KuaiRand-Pure (required benchmark)

| Run | Best validation primary | Final test primary | Delta vs. baseline (0.5946) | Iterations | Manual interventions |
|---|---|---|---|---|---|
| **pure-real-run-1 (OFFICIAL SUBMISSION)** | 0.6041 | **0.5968** | **+0.0022** | 5 scored (+2 debugger-recovered) | 0 |
| ffm-attempt-1 (secondary) | 0.6037 | 0.5967 | +0.0021 | 5 scored | 0 |
| short-autonomous-loop (early, superseded) | 0.6016 | 0.5953 | +0.0007 | 2 | 0 |

Baseline (organizer-provided FM): test primary 0.5946 (GAUC 0.6610, nDCG@5 0.5282), std 0.0008 across 5 seeds. `pure-real-run-1`'s delta is about 2.76x the reported baseline test-seed standard deviation.

`pure-real-run-1` was selected as the official submission by validation score (0.6041, the higher of the two full runs) — never by peeking at test scores.

Additionally, a third independent run (`5cb9e936da024858815cd932df6ecfb7`, on the `orchestrator_with_kb_1k` branch — renamed from `orchestrator_with_kb_27k`, which no longer exists — a teammate's attempt) reached test primary 0.5964 (+0.0018), also below the official submission — its accepted iteration shows the same hypothesis/code mismatch pattern described below (claimed DIN, implemented prevalence-weighted BCE).

### KuaiRand-1K (bonus benchmark)

| Run | Best validation primary | Final test primary | Delta vs. baseline (0.6439 valid / 0.6405 test) | Iterations | Result |
|---|---|---|---|---|---|
| **`0f6a4083fba54b798c7bb6f87c0a73a9`** (independent run, `orchestrator_with_kb_1k` branch) | **0.6463** (3-seed mean 0.64628, seeds 0.64565/0.64553/0.64767) | 0.6399 (single seed, GAUC 0.6736 / nDCG@5 0.6062) | **+0.0024** valid (3-seed replicated), +0.0019 test (single-seed, not replicated) | 8 scored | **Confirmed improvement** |

**An independent run did find a real, replicated 1K improvement.** Run `0f6a4083fba54b798c7bb6f87c0a73a9` (8 scored iterations, 386,109 tokens, ~2256s wall-clock; on the `orchestrator_with_kb_1k` branch, not this one) accepted a hypothesis swapping the FM's sparse Adam optimizer for sparse Adagrad (lr=0.03) — same forward pass, loss, and init otherwise. Unlike the single-seed accepts elsewhere in this document, this one was taken through this project's own 3-seed replication protocol *before* being trusted: seeds 0/1/2 scored valid primary 0.64565 / 0.64553 / 0.64767 (mean 0.64628, +0.0024 over the 0.6439 baseline, comfortably clearing a noise band derived from the baseline's own measured seed sd). The 0.6399 test score is single-seed only (seed 0) and was never independently replicated on test — reported as such, not as a confirmed test-set delta, consistent with this document's no-snooping/one-time-test discipline elsewhere.

To inspect this run's full log, code diff, and generated submission CSV:
```bash
git fetch origin orchestrator_with_kb_1k
git checkout orchestrator_with_kb_1k
cd kuairand-starter-kit/runs/BEST_1K_RUN   # renamed from 0f6a4083fba54b798c7bb6f87c0a73a9
cat summary.json          # headline metrics
cat iterations.jsonl      # every iteration's hypothesis, diff, and score
```
The winning candidate itself is `kuairand-starter-kit/best_1k_candidate.py` on that branch, and `python scripts/make_1k_submission.py` regenerates its submission CSV deterministically (seed 0) without re-running the LLM loop.

KuaiRand-27K was not attempted as a full agent-driven submission; see `ml_modelling/` on the `ML-modelling` branch for a teammate's independent exploratory work at that scale (real measured hardware/timing data, a GBDT baseline attempt, and a 27K run).

### KuaiRand-27K (bonus benchmark)

Not attempted as a full agent-driven submission (Planner/Coder/Executor loop). A single manual scale-transfer measurement (`ml_modelling/experiments/p20_27k_run.py`, on the `orchestrator_with_kb_1k` branch — renamed from `orchestrator_with_kb_27k`, which no longer exists) confirms the pipeline runs correctly at full 27K scale: sparse pointwise FM, k=16, lr=0.001, seed=0, on the full 322,278,385-row dataset (71.1M validation rows / 26,729 users, 114.8M test rows / 27,249 users, 32M+ videos). Result: validation primary 0.6687 (GAUC 0.6911, nDCG@5 0.6463), test primary 0.6557 (GAUC 0.6852, nDCG@5 0.6261). No organizer baseline reference exists for 27K in this repo, so this is reported as a raw measurement, not a scored delta — it demonstrates infrastructure readiness rather than a benchmarked improvement.

## Resource Usage

| Benchmark | Total LLM tokens | Wall-clock | Iterations used | Compute |
|---|---|---|---|---|
| Pure (official run) | 109,787 | 22.75 min | 5 of 50 | CPU only |
| 1K (bonus run) | 109,307 | 78 min (4,675s) | 8 of 8 (self-capped) | CPU only |
| 1K (`0f6a4083...`, confirmed improvement) | 386,109 | ~37.6 min (2,256s) | 8 of 50 (converged) | CPU only |

No GPU was used at any stage — the FM baseline trains in ~40s/run on a single CPU core, and compute was never the binding constraint; iteration count and time were dominated by hypothesis generation and diff correctness, not training cost.

## Known Limitations

- **Hypothesis/code mismatch, observed systemically (3 independent instances)**: across `pure-real-run-1`, `ffm-attempt-1`, and a teammate's run on `orchestrator_with_kb_1k` (renamed from `orchestrator_with_kb_27k`), the accepted iteration's hypothesis text describes a different model each time (DIN positive-history sequence model, field-aware FM, DIN again) — but in every case the code that was actually implemented and scored is prevalence-weighted binary cross-entropy (`pos_weight`) on the plain FM. The scores are real and correctly computed; the reasoning trail attached to them does not match the code. See `runs/pure-real-run-1/CORRECTION_NOTE.md` for the full, unedited quote of the mismatch — the raw run logs (`iterations.jsonl`, `summary.json`) were deliberately left unaltered rather than retroactively corrected, to preserve an honest record of what the agent actually did. This is treated as a reproducible capability boundary of the current Coder stage, not an incidental bug.
- **Field-aware FM never implemented**: proposed as a hypothesis 3 separate times (Pure via `ffm-attempt-1`, 1K via `1k-real-run-1`), the Coder consistently produced only alias-level, functionally-empty diffs rather than the actual per-field-pair embedding architecture.
- **Triviality pattern**: across all runs, a nontrivial fraction of Coder diffs (5 of 8 in `1k-real-run-1`) were syntactically valid but semantically empty for complex hypotheses (ensembles, sequence models, attention). Partially mitigated with an explicit anti-triviality prompt instruction; not fully solved.
- **Scope leakage**: one 1K-scoped run drifted into attempting a full KuaiRand-27K rewrite mid-run and crashed (27K data was never downloaded in that context) — a real harness gap: no guard currently prevents an agent from proposing changes outside its assigned benchmark scope.
- **What we'd improve with more time**: (1) a benchmark-scope guard in the Coder/Executor contract, (2) stronger triviality detection (e.g. diffing the effective computation graph, not just the source diff), (3) extending the paired-seed statistical stopping policy to Pure as well, now that 1K's baseline variance turned out to be higher than assumed.

## Setup & Reproduction

```bash
git clone https://github.com/gowthumb/techjam_agent_exe.git
cd techjam_agent_exe/kuairand-starter-kit
pip install -r requirements.txt  # numpy-only, no torch/pandas/sklearn required
```

Reproduce the baseline (no API key needed):
```bash
python3 baseline.py --model fm
```

Reproduce the official submitted result deterministically, without re-running the LLM loop (no API key needed):
```bash
python3 scripts/finalize_submission.py --run-id pure-real-run-1
```

Re-run the full autonomous agent loop from scratch (requires an LLM API key, will not reproduce identical results since the Planner/Coder are non-deterministic):
```bash
python3 scripts/run_agent.py --dataset pure --run-id <your-run-name>
```

Run logs for every iteration (hypothesis, code diff, resulting metrics, errors/recoveries) are at `runs/<run-id>/iterations.jsonl`.

See also: `docs/HARDWARE_AWARENESS.md` for measured (not estimated) CPU/memory/timing guidance across Pure/1K/27K scales.

## Team Contributions

- **Yash** — `orchestrator`, `orchestrator_with_kb` branches: agent orchestrator (Planner/Coder/Executor/Debugger loop), knowledge base integration, Pure official submission (`pure-real-run-1`), KuaiRand-1K bonus adapter and both 1K runs, paired-seed statistical stopping policy.
- **Gautham** — `orchestrator_with_kb_gauthi`, `orchestrator_with_kb_1k` (renamed from `orchestrator_with_kb_27k`) branches (both descending from `orchestrator_with_kb`): additional Pure hypothesis exploration (L2 regularization tuning, DIN/SSM-labeled attempts), KuaiRand-27K infrastructure and scale measurement (`baseline_27k.py`, `data_27k.py`), and the confirmed, 3-seed-replicated KuaiRand-1K improvement (`0f6a4083fba54b798c7bb6f87c0a73a9`, sparse Adagrad) documented above.
- **Harineesh** — `ML-modelling` branch: core ML pipeline components (`explib`), KuaiRand-1K/27K scale-transfer research (Phases 10-20), hardware/performance measurement (`HARDWARE_AWARENESS.md`).
- **All three** — results writeup, README, Devpost submission.