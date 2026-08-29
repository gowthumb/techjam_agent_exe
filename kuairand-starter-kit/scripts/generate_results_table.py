#!/usr/bin/env python3
"""Generate the compact results-summary Markdown for one finalized run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = {"GAUC": 0.6610, "nDCG@5": 0.5282, "primary": 0.5946}


def _best_validation(run_dir: Path, fallback: dict) -> dict:
    entries = [json.loads(line) for line in (run_dir / "iterations.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    scored = [entry["metrics"]["valid"] for entry in entries if (entry.get("metrics") or {}).get("valid")]
    return max(scored, key=lambda metrics: metrics["primary"]) if scored else fallback


def generate_table(run_id: str, runs_dir: Path) -> str:
    run_dir = runs_dir / run_id
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    valid, test = _best_validation(run_dir, summary["best_validation_metrics"]), summary["final_test_metrics"]
    lines = ["| Metric | Official Baseline | Validation Best | Final Test | Test Delta vs Baseline |", "|---|---:|---:|---:|---:|"]
    lines.extend("| %s | %.4f | %.6f | %.6f | %+.6f |" % (name, BASELINE[name], valid[name], test[name], test[name] - BASELINE[name]) for name in BASELINE)
    lines.extend(["", "## Resource Usage", "", "- Total LLM tokens: %s" % summary["total_tokens"], "- Total agent wall-clock: %.2f seconds" % summary["total_wall_clock_s"], "- Iterations: %s / 50" % summary["total_iterations"], "- GPU-hours: 0 (CPU-only NumPy pipeline)"])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    output = generate_table(args.run_id, args.runs_dir)
    (args.runs_dir / args.run_id / "RESULTS.md").write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())