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

Additionally, a third independent run (`5cb9e936da024858815cd932df6ecfb7`, on the `orchestrator_with_kb_27k` branch, a teammate's attempt) reached test primary 0.5964 (+0.0018), also below the official submission — its accepted iteration shows the same hypothesis/code mismatch pattern described below (claimed DIN, implemented prevalence-weighted BCE).

### KuaiRand-1K (bonus benchmark)

| Run | Best validation primary | Final test primary | Delta vs. baseline (0.6439 valid / 0.6405 test) | Iterations | Result |
|---|---|---|---|---|---|
| 1k-real-run-1 | 0.6439 | 0.6405 | ~0.0000 | 8 (iteration cap) | Null result |

No accepted improvement over the 1K baseline. Of 8 iterations, 5 were no-op diffs (alias-only changes reproducing the baseline score to 10 decimal places) and the 2 substantive changes attempted (lower learning rate, BPR loss) both scored worse than baseline — BPR in particular dropped validation by ~0.019, showing Pure's known BPR win does not transfer to 1K's cold-item regime.

A second attempt (`1k-paired-run-1`) used a purpose-built paired-seed statistical stopping policy (baseline characterized over 5 seeds: μ=0.6431, σ=0.0023; candidates required to beat baseline by a 2σ margin over 3 matched seeds before being accepted) to guard against noise-driven false positives, given 1K's baseline variance is notably higher than initially assumed. This run was cancelled before completion due to time constraints; the policy itself (`agent/one_k_policy.py`) is implemented and unit-tested (see `tests/test_one_k_policy.py`).

KuaiRand-27K was not attempted as a full agent-driven submission; see `ml_modelling/` on the `ML-modelling` branch for a teammate's independent exploratory work at that scale (real measured hardware/timing data, a GBDT baseline attempt, and a 27K run).

## Resource Usage

| Benchmark | Total LLM tokens | Wall-clock | Iterations used | Compute |
|---|---|---|---|---|
| Pure (official run) | 109,787 | 22.75 min | 5 of 50 | CPU only |
| 1K (bonus run) | 109,307 | 78 min (4,675s) | 8 of 8 (self-capped) | CPU only |

No GPU was used at any stage — the FM baseline trains in ~40s/run on a single CPU core, and compute was never the binding constraint; iteration count and time were dominated by hypothesis generation and diff correctness, not training cost.

## Known Limitations

- **Hypothesis/code mismatch, observed systemically (3 independent instances)**: across `pure-real-run-1`, `ffm-attempt-1`, and a teammate's run on `orchestrator_with_kb_27k`, the accepted iteration's hypothesis text describes a different model each time (DIN positive-history sequence model, field-aware FM, DIN again) — but in every case the code that was actually implemented and scored is prevalence-weighted binary cross-entropy (`pos_weight`) on the plain FM. The scores are real and correctly computed; the reasoning trail attached to them does not match the code. See `runs/pure-real-run-1/CORRECTION_NOTE.md` for the full, unedited quote of the mismatch — the raw run logs (`iterations.jsonl`, `summary.json`) were deliberately left unaltered rather than retroactively corrected, to preserve an honest record of what the agent actually did. This is treated as a reproducible capability boundary of the current Coder stage, not an incidental bug.
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

- **Gautham & Yash** — Track A: agent orchestrator (Planner/Coder/Executor/Debugger loop, convergence logic, knowledge base curation, Pure and 1K bonus runs).
- **Harineesh** — Track B: ML modelling components, KuaiRand-1K/27K scale-transfer exploration, hardware/performance measurement (see `ml_modelling/` on the `ML-modelling` branch).
- **Gautham, Yash & Harineesh** — Track C: results writeup, README, Devpost submission.