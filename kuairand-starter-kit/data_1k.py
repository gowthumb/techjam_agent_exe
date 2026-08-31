"""KuaiRand-1K data layer -- int-fast-path encoding via ml_modelling/explib.

Mirrors data.py's contract (FIELDS, load(data_dir), encode(splits)) so
baseline_1k.py -- and any Coder-authored patch to it -- can import this exactly
like KuaiRand-Pure's baseline.py imports data. It does NOT mirror data.py's
implementation: data.py's Encoder round-trips every value through Python
strings, which is fine at Pure's ~40K vocabulary but is the exact thing
knowledge_base/HARDWARE_AWARENESS.md rule 1 rules out past ~500K encoder dim.
1K's is ~2.9M. This delegates to explib.features.encode_int_fields, which
works directly on integer numpy columns instead.

See knowledge_base/ONEK_RESULTS.md Phase 5's `prepare()` for the reference
implementation this mirrors -- kept in sync deliberately, not reimplemented
independently, so a fact confirmed there (e.g. per-field vocab/unseen rates)
is the same fact this module would compute.
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

BENCH = "1k"
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
    """Return an opaque splits handle.

    data_dir is accepted for interface parity with data.py's load(data_dir) but
    otherwise unused: ml_modelling.explib.benchmarks resolves KuaiRand-1K/data by
    benchmark name via its own KIT-relative path, not by an arbitrary directory.
    Row data itself is never held here -- it lives in explib's own npz log cache
    and is re-fetched (cheaply, from that cache) inside encode().
    """
    return {"bench": BENCH}


def encode(splits: Dict[str, Any]):
    """Vectorized int-fast-path encode. -> (enc {split: (X, y, users)}, total_dim).

    Note: agent/data_cache.py's executor-facing fast path bypasses this call by
    precomputing and monkeypatching around it (same trick it already uses for
    Pure). This implementation is the correct, always-available fallback for
    any other caller, including a Coder-authored candidate that calls it directly.
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
