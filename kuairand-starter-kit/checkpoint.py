"""Save/load trained FM weights -- numpy only, no framework lock-in.

Exists because none of run_fm's callers (agent/runner.py's isolated
subprocess, the plain CLI scripts) ever persisted a model's actual trained
state (V/W/b) anywhere -- only the code that produces it and the metrics it
scored. That's enough to reproduce a result deterministically (same code +
same seed = same model, verified in knowledge_base/ONEK_RESULTS.md's
"Verified, not just written" note), but "reproducible on demand" isn't the
same as "saved" when the point is to hand over a checkpoint. baseline.py /
baseline_1k.py / baseline_27k.py's run_fm all accept an optional
checkpoint_path=None kwarg that calls save() below when set -- backward
compatible (default None = no behavior change) and additive (the Executor's
isolated subprocess path, agent/runner.py, never passes it during search, so
the live acceptance loop is unaffected).
"""
from __future__ import annotations

import os
from typing import Tuple

import numpy as np


def save(path: str, V: np.ndarray, W: np.ndarray, b: float) -> None:
    """Save V/W/b to an .npz checkpoint, creating parent directories as needed."""
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.savez_compressed(path, V=V, W=W, b=np.float32(b))


def load(path: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """Return (V, W, b) from a checkpoint saved by save()."""
    with np.load(path) as archive:
        return archive["V"], archive["W"], float(archive["b"])
