# Devpost Submission — Written Project Description

**Inspiration**

Track 2 challenged us to build something structurally different from a typical ML project: an agent that does the research itself — proposing hypotheses, writing the code, running the experiment, and deciding what to try next — rather than a human iterating with an ML toolkit. We were drawn to the honesty built into the task's own scoring: a negative or thin delta isn't a failure, it's just a lower score on one slice. That framing let us build something that reports what actually happened rather than what we wished had happened.

**What it does**

An autonomous ML research agent that runs a full Planner → Coder → Executor → Debugger loop against the KuaiRand-Pure recommender benchmark (and, as a bonus attempt, KuaiRand-1K). Each iteration: the Planner proposes a modeling hypothesis grounded in a curated knowledge base of prior findings; the Coder implements it as a code diff against the current best model; the Executor deterministically trains and validates the result, accepting or rejecting it; and the Debugger repairs malformed diffs so a bad patch never crashes the run. The loop runs unattended until it converges (validation score plateaus within ε=0.002 over 3 iterations) or hits a hard cap, then scores once, on held-out test data it never touched during development.

Our official Pure result: **+0.0022 delta over the organizer baseline** (test primary 0.5968 vs. baseline 0.5946), reached in 5 scored iterations, 22.75 minutes of wall-clock time, 109,787 LLM tokens, and **zero manual interventions**.

**How we built it**

Python, numpy-only (matching the starter kit — no torch/pandas/sklearn needed; the FM baseline trains in ~40 seconds on a single CPU core, so compute was never the bottleneck). LLM backend: Sakana AI's Fugu model for the agent's Planner/Coder/Debugger roles, with a parallel experiment using Llama-3.1-8B-Instruct. The knowledge base (`knowledge_base.yaml`) enforces a replication discipline — confirmed wins, controls, and known non-transfers are surfaced to the Planner first, trimmed to the decision-relevant subset to keep context focused. We split work three ways: agent orchestrator and harness reliability (Gautham, Yash), ML modelling components and scale-transfer exploration to KuaiRand-1K/27K (Harineesh), and this writeup (all three).

**Challenges we ran into**

The most persistent one wasn't a bug — it was a pattern: across three independent runs, on different branches, by different team members, the Coder repeatedly implemented the same prevalence-weighted binary cross-entropy loss regardless of what the Planner's hypothesis actually asked for (a DIN sequence model, an SSM, a field-aware FM interaction structure). The scores were real and correctly computed, but the reasoning trail attached to them wasn't what shipped. Rather than editing the run logs to look cleaner, we documented this openly (`CORRECTION_NOTE.md`) and treated it as a genuine finding about the current Coder's capability boundary. We also hit a real harness gap — a 1K-scoped run once drifted into rewriting our code for the 27K benchmark mid-run and crashed, since we had no guard against an agent proposing changes outside its assigned scope.

Our KuaiRand-1K bonus attempt came back a clean null result: no accepted improvement over baseline across 8 iterations, with 5 of them literal no-op diffs. Rather than treat that as wasted effort, we built a rigorous paired-seed statistical stopping policy in response — characterizing the 1K baseline's variance over 5 seeds (which turned out higher than we'd assumed) and requiring a 2σ margin over 3 matched seeds before accepting any candidate, specifically to avoid a noise-driven false positive.

**Accomplishments that we're proud of**

A fully autonomous, zero-manual-intervention run that beats the baseline by a reproducible margin roughly 2.75x the baseline's own measurement noise — and an honest paper trail showing exactly how it got there, including where the agent's stated reasoning and its actual code diverged. We'd rather submit a true account of a real system's real limitations than a run log edited to look better than what happened.

**What we learned**

Published calibration work (MLE-Bench, MLRC-Bench) suggests a positive, reproducible delta with zero intervention is closer to the exception than the norm for this class of agent — one benchmark found the best agent/framework combination reached only bronze-medal performance on 16.9% of tasks, and it's common for agents to score worse than baseline entirely. That context reframed how we read our own results: the interesting story isn't just the delta, it's the systemic Coder limitation we found and could name precisely, three times, across independent runs.

**What's next**

A scope guard preventing an agent from editing outside its assigned benchmark; stronger triviality detection that inspects the effective computation graph rather than just the text diff; and extending the paired-seed statistical stopping policy from 1K to Pure, now that we know baseline variance can run higher than assumed.

**Built With**: python, numpy, sakana-ai-fugu, llama-3.1-8b-instruct, kuairand-pure, kuairand-1k, git