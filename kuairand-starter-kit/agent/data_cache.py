"""Disk cache for the expensive deterministic feature-encoding step.

Pure's path (default, unchanged) round-trips through ``data.py``'s string-keyed
Encoder and a pickled raw-row cache. 1K/27K go through a different branch
entirely: their encoder dim (2.9M / 20.3M) is far past the ~500K threshold where
that string-keyed path is infeasible (knowledge_base/HARDWARE_AWARENESS.md rule
1), so ``bench in ("1k", "27k")`` delegates to ``data_1k``/``data_27k`` -- thin
wrappers around ``ml_modelling/explib``'s vectorized int-fast-path encoder --
and caches only the final encoded arrays, never a raw per-row Python list.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import data


EncodedSplit = Tuple[np.ndarray, np.ndarray, List[str]]
EncodedData = Dict[str, EncodedSplit]
_CACHE_FORMAT_VERSION = 1
_RAW_CACHE_FORMAT_VERSION = 1

# Benchmarks whose row/vocab scale requires the int-fast-path encoder instead of
# data.py's string-keyed one. See knowledge_base/HARDWARE_AWARENESS.md rule 1.
_SCALED_BENCHES = ("1k", "27k")
_SCALED_MODULES = {"1k": "data_1k", "27k": "data_27k"}


def cache_key(fields: Optional[List[str]] = None) -> str:
    """Return the stable cache key for a particular ordered feature set."""
    selected_fields = list(data.FIELDS if fields is None else fields)
    payload = json.dumps(
        {"version": _CACHE_FORMAT_VERSION, "fields": selected_fields},
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_splits(data_dir: Path | str, cache_dir: Optional[Path | str] = None) -> Dict[str, list]:
    """Load raw Pure splits from a cache keyed by source file metadata."""
    data_dir = Path(data_dir)
    cache_root = Path(cache_dir) if cache_dir is not None else data_dir / ".agent_cache"
    source_files = (
        "video_features_basic_pure.csv",
        "log_standard_4_08_to_4_21_pure.csv",
        "log_standard_4_22_to_5_08_pure.csv",
    )
    fingerprint = [
        (name, (data_dir / name).stat().st_size, (data_dir / name).stat().st_mtime_ns)
        for name in source_files
    ]
    cache_path = cache_root / "raw_splits.pkl"
    if cache_path.exists():
        try:
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
            if cached["version"] == _RAW_CACHE_FORMAT_VERSION and cached["fingerprint"] == fingerprint:
                return cached["splits"]
        except (OSError, KeyError, pickle.UnpicklingError):
            pass

    splits = data.load(str(data_dir))
    cache_root.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(".tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump(
            {"version": _RAW_CACHE_FORMAT_VERSION, "fingerprint": fingerprint, "splits": splits},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    os.replace(temporary_path, cache_path)
    return splits


def _load_and_encode_pure(
    data_dir: Path | str, cache_dir: Optional[Path | str]
) -> Tuple[Dict[str, list], EncodedData, int]:
    data_dir = Path(data_dir)
    cache_root = Path(cache_dir) if cache_dir is not None else data_dir / ".agent_cache"
    key = cache_key()
    cache_path = cache_root / ("encoded_" + key + ".npz")
    splits = load_splits(data_dir, cache_root)

    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as archive:
                cached_fields = json.loads(str(archive["fields"].item()))
                if cached_fields == list(data.FIELDS):
                    encoded = {
                        name: (
                            archive[name + "_X"].copy(),
                            archive[name + "_y"].copy(),
                            archive[name + "_users"].astype(str).tolist(),
                        )
                        for name in ("train", "valid", "test")
                    }
                    return splits, encoded, int(archive["field_dims"].item())
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    encoded, field_dims = data.encode(splits)
    _save_encoded_cache(cache_path, list(data.FIELDS), encoded, field_dims)
    return splits, encoded, field_dims


def _load_and_encode_scaled(
    data_dir: Path | str, cache_dir: Optional[Path | str], bench: str
) -> Tuple[Dict[str, Any], EncodedData, int]:
    data_dir = Path(data_dir)
    cache_root = Path(cache_dir) if cache_dir is not None else data_dir / ".agent_cache"
    module = importlib.import_module(_SCALED_MODULES[bench])
    fields = list(module.FIELDS)
    key = "scaled_" + bench + "_" + cache_key(fields)
    cache_path = cache_root / ("encoded_" + key + ".npz")
    splits = module.load(data_dir)

    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as archive:
                cached_fields = json.loads(str(archive["fields"].item()))
                if cached_fields == fields:
                    encoded = {
                        name: (
                            archive[name + "_X"].copy(),
                            archive[name + "_y"].copy(),
                            archive[name + "_users"].astype(str).tolist(),
                        )
                        for name in ("train", "valid", "test")
                    }
                    return splits, encoded, int(archive["field_dims"].item())
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

    encoded, field_dims = module.encode(splits)
    _save_encoded_cache(cache_path, fields, encoded, field_dims)
    return splits, encoded, field_dims


def _save_encoded_cache(cache_path: Path, fields: List[str], encoded: EncodedData, field_dims: int) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(".tmp.npz")
    arrays: Dict[str, Any] = {
        "fields": np.asarray(json.dumps(fields, ensure_ascii=True)),
        "field_dims": np.asarray(field_dims, dtype=np.int64),
    }
    for name, (features, labels, users) in encoded.items():
        arrays[name + "_X"] = features
        arrays[name + "_y"] = labels
        arrays[name + "_users"] = np.asarray(users, dtype=str)
    np.savez_compressed(temporary_path, **arrays)
    os.replace(temporary_path, cache_path)


def load_and_encode(
    data_dir: Path | str, cache_dir: Optional[Path | str] = None, bench: str = "pure"
) -> Tuple[Dict[str, Any], EncodedData, int]:
    """Load raw splits and retrieve their encoding from a FIELDS-keyed disk cache.

    ``bench`` selects the data layer: "pure" (default, unchanged) uses
    ``data.py``'s dense string-keyed encoder; "1k"/"27k" use ``data_1k``/
    ``data_27k``'s vectorized int-fast-path encoder, mandatory past Pure's scale
    per HARDWARE_AWARENESS.md rule 1. The returned ``splits`` is opaque for the
    scaled path (candidates never inspect it directly, only pass it back into
    the module's own ``encode``) exactly as it already is for Pure.
    """
    if bench == "pure":
        return _load_and_encode_pure(data_dir, cache_dir)
    if bench not in _SCALED_MODULES:
        raise ValueError("Unknown bench %r; expected one of: pure, %s" % (bench, ", ".join(_SCALED_MODULES)))
    return _load_and_encode_scaled(data_dir, cache_dir, bench)
