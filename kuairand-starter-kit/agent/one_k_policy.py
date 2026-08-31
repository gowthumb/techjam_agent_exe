"""Paired-seed acceptance and stopping rules for KuaiRand-1K only."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence


PAIRED_SEEDS = (0, 1, 2)
PRACTICAL_DELTA = 0.001


@dataclass(frozen=True)
class PairedSeedResult:
    """Validation comparison between a candidate and matched baseline seeds."""

    deltas: tuple[float, ...]
    mean_delta: float
    delta_std: float
    lower_bound: float
    accepted: bool


def paired_seed_result(candidate_scores: Sequence[float], baseline_scores: Sequence[float]) -> PairedSeedResult:
    """Evaluate the fixed three-seed paired acceptance criterion."""
    if len(candidate_scores) != len(PAIRED_SEEDS) or len(baseline_scores) != len(PAIRED_SEEDS):
        raise ValueError("1K paired evaluation requires exactly seeds 0, 1, and 2.")
    deltas = tuple(candidate - baseline for candidate, baseline in zip(candidate_scores, baseline_scores))
    mean_delta = sum(deltas) / len(deltas)
    delta_std = math.sqrt(sum((delta - mean_delta) ** 2 for delta in deltas) / (len(deltas) - 1))
    lower_bound = mean_delta - 2 * delta_std / math.sqrt(len(deltas))
    accepted = mean_delta > 0 and lower_bound > 0 and mean_delta >= PRACTICAL_DELTA
    return PairedSeedResult(deltas, mean_delta, delta_std, lower_bound, accepted)


def validated_target_reached(incumbent_scores: Sequence[float], baseline_mean: float, baseline_std: float, paired: PairedSeedResult) -> bool:
    """Return whether a paired-validated incumbent clears the 2-sigma target."""
    if len(incumbent_scores) != len(PAIRED_SEEDS):
        raise ValueError("1K target evaluation requires exactly three incumbent seed scores.")
    return paired.accepted and sum(incumbent_scores) / len(incumbent_scores) >= baseline_mean + 2 * baseline_std


def plateau_reached(failed_distinct_mechanisms: int, has_remaining_kb_direction: bool) -> bool:
    """Stop only after six substantive failures and exhausted KB-supported directions."""
    return failed_distinct_mechanisms >= 6 and not has_remaining_kb_direction


def substantive_mechanism(diff: str, hypothesis: str) -> str | None:
    """Return a stable mechanism key, excluding forwarding-only alias edits."""
    replacement = "\n".join(part.split("=======", 1)[1].split(">>>>>>> REPLACE", 1)[0] for part in diff.split("<<<<<<< SEARCH") if "=======" in part and ">>>>>>> REPLACE" in part)
    meaningful = [line.strip() for line in replacement.splitlines() if line.strip() and not line.lstrip().startswith(("#", '"""', "'''"))]
    alias_only = meaningful and all(
        line.startswith(("def run_fm(", "return run_fm_1k(", "return run_fm("))
        or line in {"DATASET = \"1k\"", "DATASET = '1k'"}
        for line in meaningful
    )
    if not meaningful or alias_only:
        return None
    normalized = re.sub(r"[^a-z0-9]+", " ", hypothesis.lower()).strip()
    return normalized


def failed_mechanism_count(history: Sequence[dict]) -> int:
    """Count distinct substantive candidates rejected by the paired test."""
    failures = set()
    for entry in history:
        paired = (entry.get("metrics") or {}).get("paired_seed") or {}
        mechanism = substantive_mechanism(entry.get("code_diff", ""), entry.get("hypothesis", ""))
        if entry.get("status") == "rejected" and mechanism and not paired.get("accepted", False):
            failures.add(mechanism)
    return len(failures)