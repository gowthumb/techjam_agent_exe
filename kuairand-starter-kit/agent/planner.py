"""LLM-backed selection of the next research hypothesis."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from agent.coder import _HYPOTHESIS_KEYS, _validate_hypothesis
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
    # scale_transfer, dataset_facts, meta, calibration, kb_ablation all stay
    # excluded: bulk per-benchmark evidence (full exp_id lists, raw tables)
    # with no per-iteration decision value. The 1K/27K directives that DO
    # matter every iteration live in SCALE_DIRECTIVES.md instead (see
    # _scaled_bench_context) -- a hand-condensed summary, not this section's
    # raw YAML, specifically because the raw section was ~15KB of mostly
    # evidence blocks for ~1KB of actual directive.
)
# Condensed, purpose-written summary of HARDWARE_AWARENESS.md + ONEK_RESULTS.md
# (~3KB) injected into the Planner's prompt when targeting a scaled benchmark.
# Not the full docs: those two together are ~47KB, and unlike the Coder (which
# needs the literal current code to patch), the Planner only ever needs the
# directives, not the narrative -- injecting the full text every iteration was
# pure token cost with no decision-quality benefit past the first read.
_SCALED_BENCH_DOCS = ("SCALE_DIRECTIVES.md",)


@dataclass(frozen=True)
class PlannerResult:
    hypothesis: Dict[str, str]
    raw_response: str
    input_tokens: int
    output_tokens: int


def _clean_json(raw_response: str) -> str:
    """Return the first balanced ``{...}`` object, tolerating fences or prose around it."""
    stripped = _FENCE_PATTERN.sub("", raw_response).strip()
    start = stripped.find("{")
    if start == -1:
        return stripped
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : index + 1]
    return stripped[start:]


def _normalize_hypothesis(parsed: Any) -> Dict[str, str]:
    """Coerce a parsed planner reply into exactly the three required string fields."""
    if isinstance(parsed, dict) and isinstance(parsed.get("hypothesis"), dict):
        parsed = parsed["hypothesis"]
    if not isinstance(parsed, dict):
        raise ValueError("Planner response was not a JSON object.")
    if _HYPOTHESIS_KEYS.issubset(parsed):
        parsed = {key: parsed[key] for key in _HYPOTHESIS_KEYS}
    _validate_hypothesis(parsed)
    return {key: str(parsed[key]).strip() for key in _HYPOTHESIS_KEYS}


def _score_delta(entry: Dict[str, Any]) -> str:
    metrics = (entry.get("metrics") or {}).get("valid") or {}
    primary = metrics.get("primary")
    return "primary=%s" % ("n/a" if primary is None else "%.6f" % primary)


# Fields from a logged iteration the Planner actually needs to avoid
# duplicating a rejected hypothesis and to reason about what's working.
# code_diff (the literal patch text) is deliberately dropped: it's the single
# largest field per entry and carries no information the hypothesis/rationale/
# status/metrics don't already summarize for planning purposes -- the Coder
# gets the current code directly, it doesn't need to be re-derived from history.
_HISTORY_ENTRY_KEEP_KEYS = ("iteration_num", "hypothesis", "rationale", "status", "metrics")
_MAX_ERROR_TRACE_CHARS = 300


def _trim_history_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    trimmed = {key: entry.get(key) for key in _HISTORY_ENTRY_KEEP_KEYS if key in entry}
    error_trace = entry.get("error_trace")
    if error_trace:
        trimmed["error_trace"] = (
            error_trace if len(error_trace) <= _MAX_ERROR_TRACE_CHARS
            else error_trace[:_MAX_ERROR_TRACE_CHARS] + "…"
        )
    return trimmed


def _history_context(history: list[Dict[str, Any]]) -> str:
    older, recent = history[:-5], history[-5:]
    lines = ["Earlier condensed history:"]
    for entry in older:
        lines.append("- %s | %s | %s" % (entry.get("hypothesis", "unknown"), entry.get("status", "unknown"), _score_delta(entry)))
    if not older:
        lines.append("- none")
    lines.append("Five most recent entries (code diffs omitted -- see hypothesis/rationale/status/metrics only):")
    lines.append(json.dumps([_trim_history_entry(entry) for entry in recent], ensure_ascii=True, default=str))
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


def _scaled_bench_context(bench: str) -> str:
    """Condensed operational directives a scaled-benchmark hypothesis must respect.

    Injects SCALE_DIRECTIVES.md (~3KB), not the full HARDWARE_AWARENESS.md /
    ONEK_RESULTS.md (~47KB combined) -- see that file's own header for why.
    """
    if bench == "pure":
        return ""
    parts = ["----- OPERATIONAL CONSTRAINTS: KuaiRand-%s (condensed; full record in "
             "knowledge_base/HARDWARE_AWARENESS.md and ONEK_RESULTS.md) -----" % bench.upper()]
    for name in _SCALED_BENCH_DOCS:
        doc_path = _ROOT / "knowledge_base" / name
        try:
            parts.append(doc_path.read_text(encoding="utf-8"))
        except OSError:
            continue
    return "\n\n".join(parts)


def _system_prompt(knowledge_base: str, state: RunState, bench: str = "pure") -> str:
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
""" + _planner_knowledge_context(knowledge_base) + "\n----- CURRENT BEST VALIDATION METRICS -----\n" + json.dumps(state.best_metrics, default=str) + "\n----- EXPERIMENT HISTORY -----\n" + _history_context(state.experiment_history) + _scaled_bench_context(bench)

def propose_hypothesis(
    state: RunState, knowledge_base_path: str = "knowledge_base/knowledge_base.yaml", bench: str = "pure"
) -> PlannerResult:
    """Ask the Planner for one structured hypothesis using bounded run context.

    ``bench`` (pure/1k/27k) gates whether SCALE_DIRECTIVES.md -- the condensed
    summary of HARDWARE_AWARENESS.md and ONEK_RESULTS.md -- is injected. See
    _scaled_bench_context.
    """
    path = Path(knowledge_base_path)
    if not path.is_absolute():
        path = _ROOT / path
    knowledge_base = path.read_text(encoding="utf-8")
    system_prompt = _system_prompt(knowledge_base, state, bench)
    model = resolve_model("PLANNER")
    temperature = resolve_temperature("PLANNER")

    last_error = "no response"
    input_tokens = output_tokens = 0
    raw_response = ""
    for attempt in range(3):
        if attempt == 0:
            user_prompt = "Choose the next hypothesis now."
        else:
            user_prompt = (
                'Your previous reply was rejected (%s). Return ONLY a single JSON object with '
                'exactly these three string keys and nothing else, no prose, no code fences: '
                '"description", "rationale", "target_module".' % last_error
            )
        response = call_llm(system_prompt, user_prompt, model=model, temperature=temperature, role="PLANNER")
        raw_response = response.text
        input_tokens += response.input_tokens
        output_tokens += response.output_tokens
        try:
            parsed = json.loads(_clean_json(response.text))
        except json.JSONDecodeError as error:
            last_error = "response was not valid JSON: %s" % error
            continue
        try:
            hypothesis = _normalize_hypothesis(parsed)
        except ValueError as error:
            last_error = str(error)
            continue
        return PlannerResult(hypothesis, raw_response, input_tokens, output_tokens)
    raise ValueError("Planner did not return a valid hypothesis after 3 attempts: %s" % last_error)