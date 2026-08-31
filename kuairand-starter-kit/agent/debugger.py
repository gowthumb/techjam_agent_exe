"""LLM-backed diagnosis and repair of a failed candidate patch."""
from __future__ import annotations

from agent.coder import CoderResult, _SCALED_BENCH_CONSTRAINTS, _clean_diff
from agent.llm_client import call_llm, resolve_model
from agent.llm_client import resolve_temperature
from agent.patcher import extract_patch_blocks


def _system_prompt(current_code: str, failed_diff: str, error_message: str, bench: str = "pure") -> str:
    scaled_note = ("\n\n" + (_SCALED_BENCH_CONSTRAINTS % bench.upper())) if bench != "pure" else ""
    return """You are the Debugger in an autonomous recommender-system research pipeline.
Diagnose why the failed patch did not apply, had invalid syntax, failed at runtime, or -- a distinct
failure mode, check the error message for it explicitly -- scored bit-identical to the current best
because it never actually implemented the hypothesis's mechanism, before proposing a correction. Do
not guess blindly. The correction must apply against the last known-good current code, not an
intermediate broken version.

Return ONLY one or more Search/Replace blocks. No explanation, no Markdown fences, and nothing outside the Search/Replace blocks.
Use this exact format:
<<<<<<< SEARCH
def compute_score(x, y, weight=0.5):
    return weight * x + (1 - weight) * y
=======
def compute_score(x, y, weight=0.7):
    return weight * x + (1 - weight) * y
>>>>>>> REPLACE
This example demonstrates the required syntax only. Your patch must implement the specific mechanism described in the hypothesis below — do not reuse this example's function, parameters, or values unless the hypothesis independently calls for them.

Hard constraints:
- SEARCH text must be an exact verbatim substring of the current code.
- Never rewrite the whole file. Use the smallest sufficient SEARCH span that is unambiguous.
- Preserve the function signature run_fm(splits, ...) exactly.
- run_fm must accept return_predictions: bool = False. When it is True, return a test_scores array or list aligned to the test split row order alongside valid and test metrics. Its default must remain False.
- Keep existing imports (including the benchmark-specific data-loading import, e.g. from data_1k or data_27k import load, encode) unless the repair specifically requires changing them.
- The diff must implement the hypothesis's actual mechanism, not adjust existing parameters, reformat code, add unused constants, cast dtypes, or add helper functions that are not called from the active code path. A patch that does not change the model's actual computation is invalid regardless of its size or syntax validity.
- If the hypothesis requires a substantial change (a new loss function, new sampling logic, a new layer), write that logic and wire it into the function that's actually called -- don't leave it unused.
- The Executor calls run_fm(splits) with no extra keyword arguments beyond an optional seed override -- there is no sweep and no way to test more than one configuration in a single iteration. If the hypothesis names a parameter value (or several candidate values to try), pick ONE and bake it in as what actually executes: as the parameter's own default, or hardcoded. Never leave a new parameter's default at the value that reproduces the original computation exactly (a weight of 1.0, a probability or epsilon of 0.0, a dropout rate of 0, etc.) -- that is precisely what produced the no-op this repair exists to fix.

Execution contract: the normal runner invokes run_fm(splits) in an isolated subprocess and receives only validation metrics. Finalization alone invokes run_fm(splits, return_predictions=True), which must return aligned test_scores. Do not alter evaluate.py, change the split names, remove valid/test metric keys, or add code that exposes test scores during normal iteration. Preserve data.load() row order: never reorder, filter, or resort rows in a way that breaks submission row_id-to-(user_id, video_id) alignment; ml_modelling.explib.dataset.verify_row_order_matches_starter_kit() verifies this invariant. Preserve the baseline forward-pass structure, Adam optimizer update, and initialization unless the hypothesis explicitly changes one of them, so an accepted result is attributable to the stated hypothesis.""" + scaled_note + """

----- CURRENT CODE -----
""" + current_code + "\n----- FAILED DIFF -----\n" + failed_diff + "\n----- ERROR MESSAGE -----\n" + error_message


def fix_patch(current_code: str, failed_diff: str, error_message: str, bench: str = "pure") -> CoderResult:
    """Request a patch correcting a particular pre-scoring failure."""
    response = call_llm(
        _system_prompt(current_code, failed_diff, error_message, bench),
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