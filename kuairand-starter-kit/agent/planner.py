"""LLM-backed selection of the next research hypothesis."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from agent.coder import _validate_hypothesis
from agent.llm_client import call_llm, resolve_model
from agent.state import RunState


_ROOT = Path(__file__).resolve().parents[1]
_FENCE_PATTERN = re.compile(r"^\s*```(?:json)?\s*\n?|\n?\s*```\s*$")


@dataclass(frozen=True)
class PlannerResult:
    hypothesis: Dict[str, str]
    raw_response: str
    input_tokens: int
    output_tokens: int


def _clean_json(raw_response: str) -> str:
    return _FENCE_PATTERN.sub("", raw_response).strip()


def _score_delta(entry: Dict[str, Any]) -> str:
    metrics = (entry.get("metrics") or {}).get("valid") or {}
    primary = metrics.get("primary")
    return "primary=%s" % ("n/a" if primary is None else "%.6f" % primary)


def _history_context(history: list[Dict[str, Any]]) -> str:
    older, recent = history[:-5], history[-5:]
    lines = ["Earlier condensed history:"]
    for entry in older:
        lines.append("- %s | %s | %s" % (entry.get("hypothesis", "unknown"), entry.get("status", "unknown"), _score_delta(entry)))
    if not older:
        lines.append("- none")
    lines.append("Five most recent entries, full JSON:")
    lines.append(json.dumps(recent, ensure_ascii=True, default=str))
    return "\n".join(lines)


def _system_prompt(knowledge_base: str, state: RunState) -> str:
    return """You are the Planner in an autonomous recommender-system research pipeline. Select exactly one next hypothesis for the Coder; you do not write code.

Return only valid JSON, with exactly this shape:
{"description": "...", "rationale": "...", "target_module": "..."}
Do not use Markdown fences or any prose outside the JSON object.

Hard constraints:
- Do NOT propose anything matching the knowledge base's "Already Tested, No Gain" directions.
- Do NOT substantially duplicate a rejected historical hypothesis. Check the supplied history explicitly.
- The rationale must cite a numbered knowledge-base item, or clearly justify a direction beyond the knowledge base.

----- KNOWLEDGE BASE -----
""" + knowledge_base + "\n----- CURRENT BEST VALIDATION METRICS -----\n" + json.dumps(state.best_metrics, default=str) + "\n----- EXPERIMENT HISTORY -----\n" + _history_context(state.experiment_history)


def propose_hypothesis(
    state: RunState, knowledge_base_path: str = "knowledge_base/knowledge_base.yaml"
) -> PlannerResult:
    """Ask the Planner for one structured hypothesis using bounded run context."""
    path = Path(knowledge_base_path)
    if not path.is_absolute():
        path = _ROOT / path
    knowledge_base = path.read_text(encoding="utf-8")
    response = call_llm(
        _system_prompt(knowledge_base, state),
        "Choose the next hypothesis now.",
        model=resolve_model("PLANNER"),
        role="PLANNER",
    )
    try:
        hypothesis = json.loads(_clean_json(response.text))
    except json.JSONDecodeError as error:
        raise ValueError("Planner response was not valid JSON: %s" % error) from error
    _validate_hypothesis(hypothesis)
    return PlannerResult(hypothesis, response.text, response.input_tokens, response.output_tokens)