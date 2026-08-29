# Research Hypotheses

This is a living document. Teammates may extend it with additional research and techniques; the Planner must read the current file contents rather than treating this seed as fixed forever.

## Already Tested, No Gain

The Planner must never propose these directions:

- Adding more feature fields: CWM's 13-field set tested at primary 0.5940 versus 0.5950 for five fields, flat to slightly worse.
- Bigger embeddings: k=8/16/32 tested at 0.5895/0.5902/0.5887, flat.
- Reason: the user_id x video_id crossing already captures most learnable signal. Pure user-side first-order terms contribute exactly zero to within-user ranking because any user-constant term cannot change order.

## Untested Directions, In Priority Order

1. Loss function: current pointwise logloss does not directly optimize ranking metrics. Try pairwise BPR or listwise softmax over a user's impressions.
2. Behavioral sequences: interaction history is unused despite hundreds to thousands of train interactions per user. DIN/SIM-style modeling is unexplored.
3. Multi-task: is_click, is_like, is_follow, is_comment, is_forward, and play_time_ms are unused and may support long_view as auxiliary losses.
4. Watch-time censored regression: CWM's research contribution; higher risk but research-depth opportunity.
5. Model architecture: DeepFM, DCN, or xDeepFM, deliberately last because capacity was not the bottleneck.
6. Temporal features and train-test drift.
7. Advanced: use log_random_4_22_to_5_08_pure.csv as an unbiased validation set.