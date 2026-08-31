"""Persistent state for an autonomous recommender research run."""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _json_default(value: Any) -> Any:
    """Convert scalar values supplied by numerical libraries to JSON primitives."""
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError("Object of type %s is not JSON serializable" % type(value).__name__)


@dataclass
class RunState:
    """Serializable state shared by the research, execution, and logging stages."""

    current_code: str
    best_metrics: Optional[Dict[str, float]] = None
    experiment_history: List[Dict[str, Any]] = field(default_factory=list)
    retry_count: int = 0
    iteration_num: int = 0
    manual_interventions: int = 0
    total_wall_clock_s: float = 0.0
    total_tokens: int = 0
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    # Top-level module names agent/dependencies.py has already attempted to
    # pip install during this run -- caps auto-install to one attempt per
    # module per run, regardless of how many iterations re-hit the same
    # missing import (agent/executor.py checks this before calling install()).
    attempted_installs: List[str] = field(default_factory=list)

    @classmethod
    def from_baseline(cls, baseline_path: Optional[Path] = None, **kwargs: Any) -> "RunState":
        """Create a fresh run whose active code is the repository baseline."""
        if baseline_path is None:
            baseline_path = Path(__file__).resolve().parents[1] / "baseline.py"
        return cls(current_code=Path(baseline_path).read_text(encoding="utf-8"), **kwargs)

    def save(self, path: Path | str) -> None:
        """Atomically persist the complete state so interrupted runs can resume."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True, ensure_ascii=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)

    @classmethod
    def load(cls, path: Path | str) -> "RunState":
        """Restore state written by :meth:`save`."""
        with Path(path).open(encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(**payload)