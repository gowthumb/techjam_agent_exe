"""Disk cache for the expensive deterministic feature-encoding step."""
from __future__ import annotations

import hashlib
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
    """Load raw splits from a cache keyed by source file metadata."""
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


def load_and_encode(
    data_dir: Path | str, cache_dir: Optional[Path | str] = None
) -> Tuple[Dict[str, list], EncodedData, int]:
    """Load raw splits and retrieve their encoding from a FIELDS-keyed disk cache."""
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
    cache_root.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(".tmp.npz")
    arrays: Dict[str, Any] = {
        "fields": np.asarray(json.dumps(list(data.FIELDS), ensure_ascii=True)),
        "field_dims": np.asarray(field_dims, dtype=np.int64),
    }
    for name, (features, labels, users) in encoded.items():
        arrays[name + "_X"] = features
        arrays[name + "_y"] = labels
        arrays[name + "_users"] = np.asarray(users, dtype=str)
    np.savez_compressed(temporary_path, **arrays)
    os.replace(temporary_path, cache_path)
    return splits, encoded, field_dims