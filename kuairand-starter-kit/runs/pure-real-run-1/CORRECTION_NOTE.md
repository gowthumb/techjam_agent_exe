# Correction note: iteration 2 hypothesis/code mismatch

The accepted iteration that produced this run's winning score (validation
primary 0.6041, test primary 0.5968 -- the submitted result) has a
documented discrepancy between its recorded hypothesis and its actual
code.

**Recorded hypothesis** (iterations.jsonl, iteration_num=2):
> Replicate the train-label-only positive-history DIN variant over three matched seeds against its no-sequence DIN control, using the previously validated d16/lr=0.003 setup and strictly causal positive histories. Select only on mean valid, record test without selecting on it, and veto the mechanism if its rand_valid score drops by at least 0.003 versus the control.

**What the code diff actually implements**: a class-weighted binary
cross-entropy loss on the existing pointwise FM (`pos_weight=4.0`),
unrelated to the DIN/positive-history sequence model the hypothesis
describes.

**Why we're flagging this rather than silently accepting it**: the score
itself is real and correctly computed by the deterministic Executor --
the mismatch is between the Planner's stated reasoning and what the
Coder actually implemented, not an error in scoring or evaluation. This
is a known, reproducible limitation of the current harness (the Coder
does not always implement what the Planner's hypothesis describes,
particularly for architecturally complex changes) -- see README
Limitations for further documented instances of this pattern, including
two independent attempts at field-aware FM (on both KuaiRand-Pure and
KuaiRand-1K) where the same class of mismatch occurred.

We are keeping this run as the official submission rather than
discarding it, because: (a) the delta is real and reproducible
regardless of the label, (b) it was selected on validation, not test,
consistent with our no-test-snooping discipline, and (c) editing the
raw run log to retroactively match the code would misrepresent what the
agent actually produced during the run.