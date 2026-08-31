"""KuaiRand-27K data layer -- int-fast-path encoding via ml_modelling/explib.

Identical contract and implementation strategy to data_1k.py (see that file's
docstring); only BENCH differs. Kept as a separate module rather than a
parameterized one so a Coder-authored patch to the 27K candidate can never
accidentally retarget 1K (or vice versa) through a shared mutable BENCH value
imported by reference.

27K's encoder dim is ~20.3M and its raw logs are 322M rows -- see
knowledge_base/HARDWARE_AWARENESS.md's per-benchmark table before changing
anything here. In particular: do not add columns beyond the 5 baseline fields
without recomputing that table's memory math (rule 2), and never build a wide
per-row float32 feature matrix in this file or in a candidate that imports it
(GBDT-style matrices were calculated at ~22GB against a 23.7GB machine here and
deliberately never attempted at this scale).
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ML_MODELLING = os.path.abspath(os.path.join(_HERE, "..", "ml_modelling"))
if _ML_MODELLING not in sys.path:
    sys.path.insert(0, _ML_MODELLING)

from explib import benchmarks as B  # noqa: E402
from explib import dataset as D  # noqa: E402
from explib import features as F  # noqa: E402

BENCH = "27k"
FIELDS = list(F.BASELINE_FIELDS)


def _prepare():
    """-> (logs, masks, cols): the raw arrays encode() needs, per Phase 5's prepare()."""
    logs = B.load_logs(BENCH, minimal=True)
    masks = D.split_slices(logs)
    tr = masks["train"]

    vids, auths = B.load_video_authors(BENCH)
    order = np.argsort(vids)
    pos = np.searchsorted(vids[order], logs["video_id"])
    pos = np.clip(pos, 0, len(vids) - 1)
    hit = vids[order][pos] == logs["video_id"]
    author = np.where(hit, auths[order][pos], -1)

    dur_edges = np.quantile(logs["duration_ms"][tr], np.linspace(0, 1, 11)[1:-1])
    dur_bucket = np.searchsorted(dur_edges, logs["duration_ms"]).astype(np.int64)

    cols = {
        "user_id": logs["user_id"], "video_id": logs["video_id"],
        "author_id": author, "tab": logs["tab"].astype(np.int64),
        "dur_bucket": dur_bucket,
    }
    return logs, masks, cols


def load(data_dir: Optional[str] = None) -> Dict[str, Any]:
    """Return an opaque splits handle. See data_1k.load() -- same contract, same caveats."""
    return {"bench": BENCH}


def encode(splits: Dict[str, Any]):
    """Vectorized int-fast-path encode. -> (enc {split: (X, y, users)}, total_dim).

    On 27K this is the ~34min step (measured, ONEK_RESULTS.md Phase 20) the
    first time it runs uncached. agent/data_cache.py caches its output to npz
    so repeat iterations against unchanged FIELDS don't pay it again.
    """
    logs, masks, cols = _prepare()
    tr = masks["train"]
    X, dim, _unseen = F.encode_int_fields(cols, tr, order=FIELDS)
    y = (logs[D.LABEL] != 0).astype(np.float32)
    enc = {
        name: (X[m], y[m], logs["user_id"][m].astype(str).tolist())
        for name, m in masks.items()
    }
    return enc, dim
