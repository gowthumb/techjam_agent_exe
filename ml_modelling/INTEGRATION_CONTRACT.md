# KB ↔ Agent Brain Integration Contract (proposed)

**Status: proposed, not yet agreed.** This is the ML-modelling side of Phase 3 of
`KNOWLEDGE_BASE_PLAN.md`. It needs a yes/no from whoever owns the agent-brain
workstream; the open questions at the bottom are the ones I cannot decide alone.

## 1. What the agent reads, and when

| Point in the loop | Reads | Purpose |
|---|---|---|
| **Proposal** (before each iteration) | `validated_search_space`, `priors`, `feature_engineering_menu`, `multi_task_signals`, `candidate_models`, `dead_ends` | Choose the next config from a ranked menu instead of searching blind. `dead_ends` is a hard filter: anything listed there is not proposed at all. |
| **Execution** (on failure/anomaly) | `diagnostics` | Symptom → cause → fix, so a blown-up run costs one iteration, not three. |
| **Reflection** (after each iteration) | `calibration`, `decision_protocol` | Judge the score against the rungs and the noise band rather than against 1.0. |
| **Escalation** (when plateaued) | `architecture_ladder` | Decide whether to keep tuning or switch model family, given iterations remaining. |

`decision_protocol` is the section the reflection step must not skip. It carries
the measured noise band; without it the agent will read seed noise as progress and
"converge" on a config that is not actually better than the baseline.

## 2. Format and parsing

- `knowledge_base.yaml` — the machine-readable directive. Plain YAML, no anchors,
  no custom tags; `yaml.safe_load` is sufficient.
- `knowledge_base_rationale.md` — human-readable narrative for judges. **The agent
  never parses this.** It exists so a claim in the YAML can be traced to the
  experiment that produced it.
- `meta.schema_version` is bumped on any breaking shape change. Please pin against
  it and fail loudly rather than silently reading a missing key.

Every empirical entry carries an `evidence` block:

```yaml
evidence:
  exp_ids: [1J-bpr-lr0.0002-k8-ep60-seed0]   # keys into experiments.jsonl
  note: "what these runs actually showed"
```

`tools/kb_check.py` validates that every cited `exp_id` exists exactly once in the
log and that `calibration` matches the organizers' `baseline_scores.json`. It is
worth running in CI, or at least before any submission.

## 3. Log schema alignment

`ml_modelling/experiments.jsonl` is one JSON object per run. The agent's own run
log should use the same keys so offline findings and live runs are directly
comparable and can be concatenated:

```
exp_id, ts, phase, axis, hypothesis, config, metrics{valid,test},
train{epochs_run,best_epoch,history,...}, delta_vs_baseline, headroom_pct_valid,
outcome, takeaway, error, seconds, tags, env
```

- `outcome` ∈ `positive | neutral | negative | failed`, computed by
  `explib/harness.py::classify` from the measured noise band — **not** assigned by
  eye, and not by the agent's own judgment.
- `metrics` comes from the starter kit's `evaluate.py` via `harness.score`. Please
  call the same function rather than reimplementing the metric.
- Appends are taken under a lock (`harness.append_record`) so parallel runs do not
  interleave. If the agent writes to the same file, use that helper.

## 4. Write-back

I propose **append-only write-back to a separate file**, not in-place edits to
`knowledge_base.yaml`:

- the agent appends newly-discovered rules to `knowledge_base_learned.yaml`;
- at read time, the agent merges `knowledge_base.yaml` (curated, offline-validated)
  with `knowledge_base_learned.yaml` (in-run, unvalidated), curated winning on conflict.

Rationale: the offline KB's authority comes from every entry being backed by a
logged experiment. If a run can rewrite it mid-competition, that guarantee is gone
and `kb_check.py` can no longer certify it. Keeping them separate preserves both
the audit trail and the ability to learn during the run.

## 5. Open questions — need the agent-brain owner's call

1. **Does the reflection step get to consult the KB mid-iteration**, or only at
   proposal and reflection boundaries? Affects whether `diagnostics` must be
   loadable inside a training loop.
2. **Is the merge in §4 acceptable**, or does the agent need in-place write access?
   If in-place, I would want `kb_check.py` run after every write.
3. **Who owns the iteration budget accounting** that `architecture_ladder` keys off
   (`iterations_remaining`)? The ladder's escalation rules are written assuming the
   agent can tell the KB how many iterations are left.
4. **Selection split.** The KB tells the agent to select on `valid`. Phase 1 found
   that several interventions are flat on valid but consistently better on test,
   so valid-only selection has a measurable blind spot. If the agent has any
   additional held-out signal available, I would like to know before finalizing
   `decision_protocol`.
