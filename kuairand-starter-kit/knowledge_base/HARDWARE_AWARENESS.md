# Hardware awareness — deciding CPU vs GPU, sparse vs dense, before you run anything

This is an operational runbook, not a modeling document. `knowledge_base.yaml`
tells the agent *what config to try*; this tells it *what machine decisions to
make before trying it* — CPU or GPU, dense or sparse Adam, whether a feature
block is safe to build in memory — for KuaiRand-Pure / 1K / 27K, and for any
future benchmark in the same family at a scale nobody has measured yet.

Every number below is measured, not estimated, from Phases 5/10/18/19/20 (see
`../knowledge_base/ONEK_RESULTS.md`). Where a rule generalizes beyond the three
known benchmarks, it's stated as a formula with its measured error margin, not a
guess.

## The rule this whole document exists to enforce

**Measure the machine and the dataset before deciding how to run on them. Do not
infer either from what a library reports by default.** Two concrete traps hit
during this workstream:

1. `torch.cuda.is_available()` returned `False` on a machine that turned out to
   have a working NVIDIA GPU (RTX 4050, 6GB VRAM, CUDA 13.1 driver) — because the
   installed `torch` build was CPU-only (`2.8.0+cpu`). The check was correctly
   reporting "this framework build can't use a GPU," which is easy to misread as
   "this machine has no GPU." **Check the hardware directly** (`nvidia-smi`),
   separately from asking whether the currently-installed framework can use it.
2. Single-connection download throughput from the data host measured at
   ~1.3-1.7MB/s — nothing in any planning document predicts network speed, and a
   9.9GB archive at that rate is ~1h40m before any compute starts. **Measure a
   small range request before committing to a time budget for a new download**,
   don't assume.

Both traps look identical: a default answer that's technically true but leads to
the wrong operational decision. The fix is always the same — run a cheap, direct
check and act on that, not on what a framework or plan document implies.

## Agent pipeline integration

Every rule below is now enforced in code, not just read here by a human before
running a one-off script. `agent/planner.py` injects a condensed summary of
this file and `ONEK_RESULTS.md` (`knowledge_base/SCALE_DIRECTIVES.md`, ~3KB)
into the Planner's prompt whenever the target benchmark isn't Pure — not the
full ~47KB of both docs verbatim, which was the original approach and turned
out to be pure per-iteration token cost (see that file's own header, and
`ONEK_RESULTS.md`'s integration section, for the measured before/after);
`agent/coder.py`/`agent/debugger.py` add a hard constraint blocking dense-Adam
and wide-feature-matrix patches on those benchmarks (rules 1 and 2);
`agent/executor.py`'s `_BENCH_TIMEOUT_S` sets a per-benchmark timeout floor
(300s / 2700s / 14400s for Pure/1K/27K) so a candidate isn't killed mid-run by
a Pure-scale default; `data_1k.py`/`data_27k.py` are the sparse int-fast-path
data layer this document specifies, wired into `agent/data_cache.py` via a
`bench` parameter.

**27K is currently out of scope.** This machine's `KuaiRand-27K.tar.gz` and
extracted `KuaiRand-27K/data/` are incomplete -- see rule 6 above. The active
entry point is `scripts/maximize_1k.py`: it runs the full search budget against
1K alone (no budget held back for a 27K confirmation), then replicates any
winning candidate over 3 seeds before trusting it, exactly the discipline rule
4's time-budget math and `ONEK_RESULTS.md`'s own two false-lead incidents both
argue for. The 27K-facing plumbing (`data_27k.py`, `baseline_27k.py`,
`agent/runner.score_confirm`, the `"27k"` entries in `_BENCH_TIMEOUT_S`/
`_BENCH_DATA_DIR`) is left in place and still passes its own smoke tests --
resuming 27K later is a data problem (re-fetch the archive correctly), not a
code problem.

## Step 0 — hardware inventory (run once per machine, cheap, ~5s)

```bash
# CPU cores (affects nthread for FM's sparse Adam and xgboost's tree_method=hist)
python -c "import os; print(os.cpu_count())"

# RAM (affects whether a feature matrix can be built in memory at all)
# Windows:
powershell -c "Get-CimInstance Win32_ComputerSystem | Select @{N='RAM_GB';E={[math]::Round($_.TotalPhysicalMemory/1GB,1)}}"
# Linux:
free -g

# Disk free on the drive the dataset will land on
df -h .

# GPU — check the HARDWARE directly, not a framework's opinion of itself
nvidia-smi   # if this errors, there is genuinely no NVIDIA GPU / driver; trust that
```

**Then verify each compute framework's *actual, current* ability to use that
hardware** — a present GPU does not imply an installed, GPU-capable build:

```python
# torch: is the INSTALLED BUILD cuda-capable (not just "is there a GPU")
import torch
print(torch.__version__, torch.cuda.is_available())
# "+cpu" in the version string with a GPU present in nvidia-smi means: the
# hardware supports GPU work, this specific torch install does not. Fixing this
# means reinstalling torch against a matching CUDA build — real setup cost,
# budget it explicitly, don't assume it's free.

# xgboost: prove GPU training actually runs, don't infer from version alone
import numpy as np, xgboost as xgb
X = np.random.rand(1000, 5).astype(np.float32)
y = np.random.randint(0, 2, 1000)
bst = xgb.train({'tree_method': 'hist', 'device': 'cuda',
                  'objective': 'binary:logistic'}, xgb.DMatrix(X, label=y), num_boost_round=5)
print('xgboost GPU: OK')   # raises if it can't actually use the device
```

On the machine this workstream ran on: `xgboost` GPU training worked **zero
setup** with the installed package + driver. `torch` did not, and fixing it is a
real reinstall (new CUDA-matched wheel, ~2-3GB) plus rewriting `fm.py`'s
sparse-Adam path for GPU tensors — not currently done, not a quick win. Re-run
both checks on a new machine; do not carry these conclusions forward as facts
about hardware you haven't checked.

## Step 1 — dataset facts pass (run once per benchmark, cheap, ~1-20min depending on scale)

Before deciding anything about a benchmark — architecture, sparse vs dense,
memory budget — get its actual shape:

```bash
python experiments/p5_scale_transfer.py --bench <pure|1k|27k> --stage facts
```

This is deliberately cheap relative to a full run (21min for 27K's 322M rows,
vs. ~2h for the actual training run) and is what the rest of this document's
decision rules key off of. **Never skip straight to a full run on a benchmark
whose facts pass hasn't been read.**

## What's known, measured, per benchmark

| | Pure | 1K | 27K |
|---|---|---|---|
| rows | 1.4M | 11.7M | 322.3M |
| users | 27,077 | 1,000 | 27,285 |
| videos (vocab) | 7,551 | 4.37M | 32.0M |
| encoder total dim (5 baseline fields) | ~40K | 2,925,549 | 20,268,804 |
| test videos seen in train | 99.9% | 15.1% | 17.3% |
| regime | warm-ID, temporal drift | item cold-start, flat | item cold-start, flat (sharper) |
| dense Adam (`sparse=False`) | fine — this is what the kit ships | **infeasible**: O(vocab) per batch on 2.9M rows | **infeasible**, same reason, worse |
| `fm.train(sparse=...)` | either works; `sparse=False` is the kit default | **`sparse=True` required** | **`sparse=True` required** |
| measured full FM run wall-clock | ~40s (CPU, single core — starter kit README) | load 57-61s; ~27-44s/epoch | load 2063s; ~750-774s/epoch |
| peak process memory (FM, 5-field int path) | negligible | not separately profiled, small (~230MB matrix) | **~13.6GB measured** / 23.7GB machine |
| GPU used | no — unnecessary at this scale | no — no GPU-FM path exists; not needed anyway | no — same; xgboost GPU available if attempting GBDT |
| recommended device | **CPU** | **CPU** | **CPU** |

**The recommended device is CPU at every scale tested so far.** This is not
"GPU is unavailable" — a real GPU exists on this machine and xgboost can use it
zero-setup. It's that the model family which has actually won at every scale
(plain FM, `fm.py`) has no GPU implementation, and building one is real,
unbudgeted engineering (see `INTEGRATION_CONTRACT.md`-style cost/benefit: the
1K-parity plan explicitly scoped this out as "Phase C, conditional on evidence
that never materialized"). **Don't reach for GPU because it's there — reach for
it when a specific, already-decided-on model can actually use it, and check
that before, not after, budgeting time for a "GPU speedup."**

## Decision rules that generalize past these three benchmarks

If a fourth benchmark in this family shows up (a different KuaiRand release, or
a resample), don't re-derive these from scratch — apply the thresholds below to
its facts-pass output.

### 1. Dense vs sparse Adam (architecture-level decision)

Dense Adam updates the *entire* embedding table every batch — cost is
`O(vocab × k)` per batch, independent of batch size. This was free at Pure's
~40K vocab and became **infeasible** at 1K's 2.9M (the 8x-row jump isn't what
broke it; the ~70x-vocab jump is).

**Rule: if the facts pass reports encoder dim > ~500K, use `sparse=True`
unconditionally.** There's no benefit to trying dense first — 1K and 27K both
crossed this threshold by orders of magnitude, and `sparse=True` was already
verified to land inside the noise band vs. dense on Pure (pointwise +0.0007, bpr
−0.0000) before being trusted at scale, so there's no accuracy cost to defaulting
to it once vocab crosses this line.

### 2. Memory budget for building a feature matrix

Two different memory profiles exist in this codebase, and only one is safe at
27K-and-beyond scale:

- **FM's lean int-fast-path** (`features.encode_int_fields`, 5 baseline fields):
  `rows × 5 × 4 bytes`. At 27K's 322M rows that's **~6.4GB** — confirmed safe
  (measured peak 13.6GB including the raw log arrays and everything else in
  memory at once).
- **GBDT's wide feature matrix** (`build_matrix_*`, ~17 float32 columns
  including affinity rate/count pairs): `rows × 17 × 4 bytes`. At 27K scale
  that's **~22GB** — *before* adding the ~7GB of raw log arrays needed to build
  it, on a 23.7GB machine. **This was never run at full 27K scale for exactly
  this reason** (Phase 20's plan explicitly chose FM over GBDT here on this
  math, not on a modeling preference alone).

**Rule: compute `rows × n_float32_columns × 4 bytes` from the facts pass's row
count before building any feature matrix. If it exceeds ~50% of measured free
RAM, don't build it whole** — narrow the column set, subsample rows, or use an
out-of-core/streaming construction (xgboost's `QuantileDMatrix` from an
iterator, e.g.) instead of materializing the full array.

### 3. GPU is a training-time lever only, not a memory-ceiling fix

A GPU with 6GB VRAM does not change what has to fit in **host RAM** first —
GBDT's feature matrix is built by NumPy on the CPU before any GPU device ever
sees it. Adding a GPU to a config that's blocked by rule #2 doesn't unblock it;
narrowing the feature set does. GPU is worth reaching for only once the memory
math already works, and only for a model that already has a GPU code path
(currently: xgboost, not `fm.py`).

### 4. Time budget scaling (validated, with measured error)

Both of these scale close to linearly with row count — validated by predicting
27K's numbers from 1K's *before* running, then checking against what actually
happened:

- **Load+encode time**: `reference_time × (target_rows / reference_rows)`.
  1K → 27K prediction: 59s × 27.55 ≈ 1626s. Actual: 2063s. **~27% under-estimate**
  — budget a margin, don't take the linear projection as a ceiling.
- **Per-epoch training time** (sparse Adam, train-split rows only):
  1K → 27K prediction: ~740s/epoch. Actual: 750-774s/epoch. **~2-5% error** —
  this one's tight; trust it more than the load-time projection.

**Rule: before committing to a full run on an unmeasured scale, run the facts
pass, apply both formulas, and add ~30% margin to the load-time estimate
specifically** (it has the wider observed error, likely because vocab-size
effects — `np.unique` cost, cache locality on a much bigger embedding table —
don't scale purely linearly with row count the way per-epoch cost does).

### 5. Download and disk budget, for a benchmark whose data isn't local yet

1. **Measure bandwidth before estimating download time.** A small ranged GET
   (20-200MB) against the actual host gives a real number; 1.3-1.7MB/s was
   measured here and is not a number any planning document would have predicted.
2. **List the archive before extracting it** (`tar tzvf`, no extraction) —
   check for outlier-sized files. 27K's `video_features_statistic` turned out to
   be ~21.7GB across 3 parts, a feature block already excluded from every model
   in this codebase; extracting it would have needed disk space with no
   corresponding use. `tar --exclude='<pattern>'` skips it during extraction.
3. **Check free disk against the manifest's total *uncompressed* size**
   (`tar tzv` reports true, uncompressed sizes per member) before extracting,
   not against the compressed archive size — the two can differ 4x or more.
4. **Delete the compressed archive after a size-verified extraction**, not
   before. Verify byte-exact against the `Content-Length` header from a HEAD
   request, or against the sum of extracted member sizes.
5. **Make any download resumable and self-healing.** A multi-hour single-shot
   download with no resume logic risks losing all progress to one dropped
   connection or one tool-level timeout. `curl -C - --retry-all-errors` plus an
   outer poll loop that relaunches with `-C -` if the process dies before the
   target size is reached costs nothing and has already saved a run once.
6. **Verify the extracted directory has the files training actually needs, not
   just that extraction exited cleanly.** A truncated or partial archive can
   extract without error and still be missing `log_standard_*.csv`,
   `log_random_*.csv`, or `video_features_basic_*.csv` — the files the vocab
   depends on — while still containing the huge `video_features_statistic`
   parts rule 2 above says to exclude in the first place (seen directly on this
   machine's `KuaiRand-27K/data`: only `user_features_27k.csv` plus two
   `video_features_statistic` parts extracted, ~14.4GB, with every log file and
   `video_features_basic_27k.csv` absent). Before trusting a data directory,
   check for the three families of file `ml_modelling/explib/benchmarks.py`'s
   `resolve_files` globs for (`log_standard_*`, `log_random_*`,
   `video_features_basic_*`) — their absence fails fast and clearly
   (`FileNotFoundError`) the moment `data_1k.py`/`data_27k.py` try to load, but
   catching it before spending any budget on a run is cheaper than reading that
   traceback after the fact.

## Per-benchmark quick reference (copy-paste starting point)

```
Pure   -> CPU, sparse=False is fine (dense is what the kit ships and is fast),
          no facts-pass surprises expected, ~40s per run.
1K     -> CPU, sparse=True REQUIRED, run the facts pass anyway (cheap, and this
          is where "expect the pointwise baseline to be the right starting
          point" was first confirmed empirically, not assumed).
27K    -> CPU, sparse=True REQUIRED, run the facts pass FIRST (not optional —
          21min vs. a multi-hour full run that could be based on wrong
          assumptions), check disk/bandwidth BEFORE downloading, exclude
          video_features_statistic from extraction, budget ~30% margin over
          the linear load-time projection.
unknown benchmark, same family
       -> Step 0 (hardware inventory) once. Step 1 (facts pass) always, first.
          Apply rules 1-5 above to its own numbers. Do not assume it behaves
          like Pure, 1K, or 27K until its own facts pass says so.
```
