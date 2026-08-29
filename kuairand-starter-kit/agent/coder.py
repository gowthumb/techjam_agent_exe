"""LLM-backed translator from a fixed research hypothesis to a surgical patch."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict

from agent.llm_client import call_llm, resolve_model


_HYPOTHESIS_KEYS = {"description", "rationale", "target_module"}
_FENCE_PATTERN = re.compile(r"^\s*```(?:\w+)?\s*\n?|\n?\s*```\s*$")


@dataclass(frozen=True)
class CoderResult:
    diff: str
    raw_response: str
    input_tokens: int
    output_tokens: int


def _validate_hypothesis(hypothesis: Dict[str, Any]) -> None:
    if set(hypothesis) != _HYPOTHESIS_KEYS:
        raise ValueError("Hypothesis must contain exactly: description, rationale, target_module.")
    invalid = [key for key in _HYPOTHESIS_KEYS if not isinstance(hypothesis[key], str) or not hypothesis[key].strip()]
    if invalid:
        raise ValueError("Hypothesis values must be non-empty strings: " + ", ".join(sorted(invalid)))


def _clean_diff(raw_response: str) -> str:
    """Remove accidental Markdown fences while preserving exact patch contents."""
    return _FENCE_PATTERN.sub("", raw_response).strip()


def _system_prompt(current_code: str) -> str:
    return """You are the Coder in an autonomous recommender-system research pipeline.
Translate the supplied hypothesis into a surgical patch for the complete current Python module below.

Return ONLY one or more Search/Replace blocks. Do not include prose, Markdown fences, or explanations.
Use this exact format:
<<<<<<< SEARCH
(exact existing code, verbatim)
=======
(replacement code)
>>>>>>> REPLACE

Hard constraints:
- SEARCH text must be an exact verbatim substring of the current code.
- Never rewrite the whole file. Use the smallest sufficient SEARCH span that is unambiguous.
- Preserve the function signature run_fm(splits, ...) exactly; the Executor relies on it.
- run_fm must accept return_predictions: bool = False. When it is True, return a test_scores array or list aligned to the test split row order alongside valid and test metrics. Its default must remain False.
- Keep existing imports, including from data import load, encode, unless the hypothesis specifically requires changing them.

Current code follows:
----- CURRENT CODE -----
""" + current_code + "\n----- END CURRENT CODE -----"


def propose_patch(current_code: str, hypothesis: Dict[str, Any]) -> CoderResult:
    """Request a Search/Replace patch for one validated, planner-owned hypothesis."""
    _validate_hypothesis(hypothesis)
    user_prompt = "\n".join(
        (
            "Hypothesis description: " + hypothesis["description"],
            "Rationale: " + hypothesis["rationale"],
            "Target module: " + hypothesis["target_module"],
        )
    )
    response = call_llm(_system_prompt(current_code), user_prompt, model=resolve_model("CODER"), role="CODER")
    return CoderResult(
        diff=_clean_diff(response.text),
        raw_response=response.text,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )