#!/usr/bin/env python3
"""Generate a Devpost-ready audit report from one autonomous run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASELINE = {"GAUC": 0.6610, "nDCG@5": 0.5282, "primary": 0.5946}


def _entries(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _metric_text(metrics: dict | None) -> str:
    if not metrics:
        return "No validation score produced."
    valid = metrics.get("valid", metrics)
    return " | ".join("%s: %.6f" % (name, valid[name]) for name in ("GAUC", "nDCG@5", "primary") if name in valid)


def _best_validation(entries: list[dict], fallback: dict | None) -> dict | None:
    scored = [entry["metrics"]["valid"] for entry in entries if (entry.get("metrics") or {}).get("valid")]
    return max(scored, key=lambda metrics: metrics["primary"]) if scored else fallback


def generate_report(run_id: str, runs_dir: Path) -> Path:
    run_dir = runs_dir / run_id
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    entries = _entries(run_dir / "iterations.jsonl")
    best_validation = _best_validation(entries, summary.get("best_validation_metrics", state.get("best_metrics")))
    models = next((entry["role_models"] for entry in reversed(entries) if entry.get("role_models")), None)
    lines = ["# Autonomous Run Report", "", "## Run Metadata", "", "- Run ID: `%s`" % run_id, "- Dataset: KuaiRand-Pure"]
    if models:
        lines.extend("- %s model: `%s`" % (role.title(), model) for role, model in sorted(models.items()))
    else:
        lines.append("- Role-model metadata: unavailable (this run predates per-entry model persistence).")
    lines.extend(["", "## Summary", ""])
    lines.extend([
        "- Stopping reason: %s" % summary.get("stopping_reason", "not finalized"),
        "- Total scored iterations: %s" % summary.get("total_iterations", state["iteration_num"]),
        "- Best validation: %s" % _metric_text(best_validation),
        "- Final test: %s" % _metric_text(summary.get("final_test_metrics")),
        "- Total LLM tokens: %s" % summary.get("total_tokens", state["total_tokens"]),
        "- Total wall-clock seconds: %.2f" % summary.get("total_wall_clock_s", state["total_wall_clock_s"]),
        "- Manual interventions: %s" % summary.get("manual_interventions", state["manual_interventions"]),
        "- Consecutive abandoned hypotheses at stop: %s" % summary.get("consecutive_abandoned", 0),
    ])
    if summary.get("final_test_metrics"):
        lines.extend(["", "### Final Test Delta vs Official Baseline", "", "| Metric | Official | Final Test | Delta |", "|---|---:|---:|---:|"])
        lines.extend("| %s | %.4f | %.6f | %+.6f |" % (name, BASELINE[name], summary["final_test_metrics"][name], summary["final_test_metrics"][name] - BASELINE[name]) for name in BASELINE)
    lines.extend(["", "## Iterations", ""])
    for index, entry in enumerate(entries, start=1):
        label = "abandoned, no iteration consumed" if entry["status"] == "abandoned" else str(entry["iteration_num"])
        lines.extend(["### Entry %d: %s" % (index, label), "", "- Status: **%s**" % entry["status"], "- Hypothesis: %s" % entry["hypothesis"], "- Rationale: %s" % entry["rationale"], "- Metrics: %s" % _metric_text(entry.get("metrics")), "- Wall time: %.2f seconds" % entry["wall_time_s"], "- Tokens: %s" % entry["tokens_used"], "", "```text", str(entry["code_diff"]), "```"])
        if entry.get("error_trace"):
            lines.extend(["", "Error trace:", "```text", entry["error_trace"], "```"])
        lines.append("")
    output = run_dir / "REPORT.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    print(generate_report(args.run_id, args.runs_dir).read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())