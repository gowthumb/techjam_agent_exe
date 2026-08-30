"""LLM-backed diagnosis and repair of a failed candidate patch."""
from __future__ import annotations

from agent.coder import CoderResult, _clean_diff
from agent.llm_client import call_llm, resolve_model
from agent.llm_client import resolve_temperature
from agent.patcher import extract_patch_blocks


def _system_prompt(current_code: str, failed_diff: str, error_message: str) -> str:
    return """You are the Debugger in an autonomous recommender-system research pipeline.
Diagnose why the failed patch did not apply, had invalid syntax, or failed at runtime before proposing a correction. Do not guess blindly.
The correction must apply against the last known-good current code, not an intermediate broken version.

Return ONLY one or more Search/Replace blocks. No explanation, no Markdown fences, and nothing outside the Search/Replace blocks.
Use this exact format:
<<<<<<< SEARCH
def run_fm(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True):
=======
def run_fm(splits, k=6, lr=0.0002, epochs=60, bs=8192, patience=4, seed=0, verbose=True):
>>>>>>> REPLACE

Hard constraints:
- SEARCH text must be an exact verbatim substring of the current code.
- Never rewrite the whole file. Use the smallest sufficient SEARCH span that is unambiguous.
- Preserve the function signature run_fm(splits, ...) exactly.
- run_fm must accept return_predictions: bool = False. When it is True, return a test_scores array or list aligned to the test split row order alongside valid and test metrics. Its default must remain False.
- Keep existing imports unless the repair specifically requires changing them.

Execution contract: the normal runner invokes run_fm(splits) in an isolated subprocess and receives only validation metrics. Finalization alone invokes run_fm(splits, return_predictions=True), which must return aligned test_scores. Do not alter evaluate.py, change the split names, remove valid/test metric keys, or add code that exposes test scores during normal iteration. Preserve data.load() row order: never reorder, filter, or resort rows in a way that breaks submission row_id-to-(user_id, video_id) alignment; ml_modelling.explib.dataset.verify_row_order_matches_starter_kit() verifies this invariant. Preserve the baseline forward-pass structure, Adam optimizer update, and initialization unless the hypothesis explicitly changes one of them, so an accepted result is attributable to the stated hypothesis.

----- CURRENT CODE -----
""" + current_code + "\n----- FAILED DIFF -----\n" + failed_diff + "\n----- ERROR MESSAGE -----\n" + error_message


def fix_patch(current_code: str, failed_diff: str, error_message: str) -> CoderResult:
    """Request a patch correcting a particular pre-scoring failure."""
    response = call_llm(
        _system_prompt(current_code, failed_diff, error_message),
        "Return the corrected patch now.",
        model=resolve_model("DEBUGGER"),
        temperature=resolve_temperature("DEBUGGER"),
        role="DEBUGGER",
    )
    return CoderResult(
        diff=extract_patch_blocks(_clean_diff(response.text)),
        raw_response=response.text,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )