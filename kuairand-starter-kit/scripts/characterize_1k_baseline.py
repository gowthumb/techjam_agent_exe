#!/usr/bin/env python3
"""Characterize the fixed KuaiRand-1K FM baseline across seeds 0 through 4."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline_1k import run_fm_1k


def main() -> int:
    scores = []
    for seed in range(5):
        result = run_fm_1k("1k", seed=seed)
        primary = float(result["valid"]["primary"])
        scores.append(primary)
        print("seed %d valid primary %.8f" % (seed, primary), flush=True)
    mean = sum(scores) / len(scores)
    std = math.sqrt(sum((score - mean) ** 2 for score in scores) / (len(scores) - 1))
    output = {
        "dataset": "1k",
        "seeds": list(range(5)),
        "valid_primary": scores,
        "mu_b": mean,
        "sigma_b": std,
    }
    destination = ROOT / "runs" / "1k-baseline-distribution.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote %s" % destination)
    print("mu_b=%.8f sigma_b=%.8f" % (mean, std))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())