"""LLM-backed selection of the next research hypothesis."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from agent.coder import _validate_hypothesis
from agent.llm_client import call_llm, resolve_model
from agent.llm_client import resolve_temperature
from agent.state import RunState


_ROOT = Path(__file__).resolve().parents[1]
_FENCE_PATTERN = re.compile(r"^\s*```(?:json)?\s*\n?|\n?\s*```\s*$")
_PLANNER_SECTIONS = (
    "decision_protocol",
    "candidate_models",
    "dead_ends",
    "feature_engineering_menu",
    "priors",
)


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


def _confirmed_knowledge_context(knowledge_base: str) -> str:
    """Extract validated or recommended model entries before the full YAML context."""
    entries = re.findall(r"^\s*- name: ([^\n]+)(.*?)(?=^\s*- name:|^#|\Z)", knowledge_base, re.MULTILINE | re.DOTALL)
    confirmed = []
    for name, body in entries:
        if "validated: true" not in body and "recommended: true" not in body and "CONFIRMED" not in body:
            continue
        details = [line.strip() for line in body.splitlines() if line.strip().startswith(("validated:", "result:", "recommended:", "recommended_config:"))]
        confirmed.append("- %s%s" % (name, " | " + " | ".join(details) if details else ""))
    return "\n".join(confirmed) if confirmed else "- No explicitly confirmed/high-confidence entries found."


def _yaml_section(knowledge_base: str, section: str) -> str:
    """Return one top-level YAML section without requiring a YAML dependency."""
    match = re.search(r"^%s:\n(.*?)(?=^[A-Za-z_][A-Za-z0-9_]*:|\Z)" % re.escape(section), knowledge_base, re.MULTILINE | re.DOTALL)
    return section + ":\n" + match.group(1).rstrip() if match else ""


def _decision_rules_context(knowledge_base: str) -> str:
    decision_protocol = _yaml_section(knowledge_base, "decision_protocol")
    rules = []
    for rule_name in ("replication_rule", "control_rule", "unbiased_veto"):
        match = re.search(r"^  %s: (.*?)(?=^  [A-Za-z_][A-Za-z0-9_]*:|^#|^[A-Za-z_][A-Za-z0-9_]*:|\Z)" % rule_name, decision_protocol, re.MULTILINE | re.DOTALL)
        if match:
            rules.append("- %s: %s" % (rule_name, " ".join(match.group(1).split())))
    rules.append(
        "- attribution_invariant: Preserve the baseline forward-pass structure, Adam optimizer update, "
        "and initialization unless the hypothesis explicitly targets one of them; otherwise an accepted "
        "result cannot be attributed to the stated change."
    )
    return "\n".join(rules) if rules else "- No replication/control/veto rules found."


def _planner_knowledge_context(knowledge_base: str) -> str:
    """Keep only per-iteration planning sections from the curated YAML."""
    sections = [_yaml_section(knowledge_base, section) for section in _PLANNER_SECTIONS]
    return "\n\n".join(section for section in sections if section)


def _system_prompt(knowledge_base: str, state: RunState) -> str:
    return """You are the Planner in an autonomous recommender-system research pipeline. Select exactly one next hypothesis for the Coder; you do not write code.

Return only valid JSON, with exactly this shape:
{"description": "...", "rationale": "...", "target_module": "..."}
Do not use Markdown fences or any prose outside the JSON object.

Hard constraints:
- Do NOT propose anything matching the knowledge base's "Already Tested, No Gain" directions.
- Do NOT substantially duplicate a rejected historical hypothesis. Check the supplied history explicitly.
- The rationale must cite a numbered knowledge-base item, or clearly justify a direction beyond the knowledge base.

----- CONFIRMED / HIGH-CONFIDENCE KNOWLEDGE -----
""" + _confirmed_knowledge_context(knowledge_base) + """
----- DECISION PROTOCOL: REPLICATION / CONTROL / VETO -----
""" + _decision_rules_context(knowledge_base) + """
----- ITERATION-RELEVANT KNOWLEDGE BASE -----
""" + _planner_knowledge_context(knowledge_base) + "\n----- CURRENT BEST VALIDATION METRICS -----\n" + json.dumps(state.best_metrics, default=str) + "\n----- EXPERIMENT HISTORY -----\n" + _history_context(state.experiment_history)

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
        temperature=resolve_temperature("PLANNER"),
        role="PLANNER",
    )
    try:
        hypothesis = json.loads(_clean_json(response.text))
    except json.JSONDecodeError as error:
        raise ValueError("Planner response was not valid JSON: %s" % error) from error
    _validate_hypothesis(hypothesis)
    return PlannerResult(hypothesis, response.text, response.input_tokens, response.output_tokens)