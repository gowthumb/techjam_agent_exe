"""Append-only audit logs for autonomous research runs."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from agent.state import RunState, _json_default


def _run_directory(state: RunState, runs_dir: Path | str = "runs") -> Path:
    directory = Path(runs_dir) / state.run_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def log_iteration(
    state: RunState, entry: Dict[str, Any], runs_dir: Path | str = "runs"
) -> None:
    """Append one completed experiment entry and persist the associated state."""
    required = {
        "iteration_num", "hypothesis", "rationale", "code_diff", "metrics", "status",
        "error_trace", "wall_time_s", "tokens_used",
    }
    missing = required.difference(entry)
    if missing:
        raise ValueError("Iteration entry is missing required keys: " + ", ".join(sorted(missing)))
    if entry["status"] not in {"accepted", "rejected", "error", "abandoned"}:
        raise ValueError("Iteration status must be accepted, rejected, error, or abandoned")

    directory = _run_directory(state, runs_dir)
    record = dict(entry)
    with (directory / "iterations.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True, default=_json_default) + "\n")
    state.experiment_history.append(record)
    state.total_wall_clock_s += float(record["wall_time_s"])
    state.total_tokens += int(record["tokens_used"])
    state.save(directory / "state.json")


def log_intervention(
    state: RunState, reason: str, runs_dir: Path | str = "runs"
) -> None:
    """Record human involvement conspicuously and persist the resulting state."""
    print("WARNING: manual intervention recorded: " + reason, file=sys.stderr)
    state.manual_interventions += 1
    directory = _run_directory(state, runs_dir)
    with (directory / "interventions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"reason": reason}, sort_keys=True, ensure_ascii=True) + "\n")
    state.save(directory / "state.json")