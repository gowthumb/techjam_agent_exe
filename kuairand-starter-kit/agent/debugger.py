"""LLM-backed diagnosis and repair of a failed candidate patch."""
from __future__ import annotations

from agent.coder import CoderResult, _clean_diff
from agent.llm_client import call_llm, resolve_model


def _system_prompt(current_code: str, failed_diff: str, error_message: str) -> str:
    return """You are the Debugger in an autonomous recommender-system research pipeline.
Diagnose why the failed patch did not apply, had invalid syntax, or failed at runtime before proposing a correction. Do not guess blindly.
The correction must apply against the last known-good current code, not an intermediate broken version.

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
- Preserve the function signature run_fm(splits, ...) exactly.
- run_fm must accept return_predictions: bool = False. When it is True, return a test_scores array or list aligned to the test split row order alongside valid and test metrics. Its default must remain False.
- Keep existing imports unless the repair specifically requires changing them.

----- CURRENT CODE -----
""" + current_code + "\n----- FAILED DIFF -----\n" + failed_diff + "\n----- ERROR MESSAGE -----\n" + error_message


def fix_patch(current_code: str, failed_diff: str, error_message: str) -> CoderResult:
    """Request a patch correcting a particular pre-scoring failure."""
    response = call_llm(
        _system_prompt(current_code, failed_diff, error_message),
        "Return the corrected patch now.",
        model=resolve_model("DEBUGGER"),
        role="DEBUGGER",
    )
    return CoderResult(
        diff=_clean_diff(response.text),
        raw_response=response.text,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )