"""Minimal KuaiRand-1K adapter built on the proven scale-transfer pipeline."""
from __future__ import annotations

import sys
from pathlib import Path


DATASET = "1k"
_ML_ROOT = Path.cwd().parent / "ml_modelling"
sys.path.insert(0, str(_ML_ROOT))
sys.path.insert(0, str(_ML_ROOT / "experiments"))

from explib import fm, harness
from p5_scale_transfer import prepare


def run_fm_1k(
    splits_or_bench="1k", k=16, lr=0.001, epochs=40, bs=8192, seed=0,
    sparse=True, return_predictions=False,
):
    """Train sparse pointwise FM on 1K and return baseline-compatible metrics."""
    if splits_or_bench not in (None, "1k"):
        raise ValueError("baseline_1k supports only the KuaiRand-1K benchmark")
    _, _, enc, dim = prepare("1k")
    model, _ = fm.train(
        enc, dim, loss="pointwise", k=k, lr=lr, epochs=epochs, bs=bs,
        patience=4, seed=seed, evaluator=harness.score, verbose=True, sparse=sparse,
    )
    result = {}
    for split in ("valid", "test"):
        features, labels, users = enc[split]
        scores = model.predict(features)
        result[split] = harness.score(users, labels, scores)
        if split == "test" and return_predictions:
            result["test_scores"] = scores
    return result


def run_fm(splits_or_bench="1k", **kwargs):
    """Runner-compatible alias; the argument is ignored unless it names 1K."""
    return run_fm_1k("1k", **kwargs)