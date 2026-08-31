"""LLM-backed translator from a fixed research hypothesis to a surgical patch."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict

from agent.llm_client import call_llm, resolve_model
from agent.llm_client import resolve_temperature
from agent.patcher import extract_patch_blocks


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


# Extra hard constraints appended to the system prompt only when the target
# benchmark's scale actually requires them (knowledge_base/HARDWARE_AWARENESS.md).
# Keeping these out of the Pure prompt avoids constraining a benchmark they were
# never measured on.
_SCALED_BENCH_CONSTRAINTS = """
This candidate targets KuaiRand-%s, not Pure. Two additional hard constraints
apply, both measured in knowledge_base/HARDWARE_AWARENESS.md:
- Dense Adam (updating the whole embedding table every batch) is infeasible at
  this vocabulary size. Never remove or bypass the sparse-Adam update in FM.step;
  a hypothesis needing dense-only behavior on this benchmark is proposing
  something the measured hardware already rules out, not a modeling choice.
- Never build a wide per-row feature matrix (a GBDT-style dense float32 array
  with one column per feature, materialized for every row). That was calculated
  at ~22GB against a 23.7GB machine at 27K scale and deliberately never
  attempted. Stay inside the existing int-encoded-fields representation; add or
  remove *fields* in that encoding, don't add a second, wider matrix alongside it.
""".strip()


def _system_prompt(current_code: str, bench: str = "pure") -> str:
    scaled_note = ("\n\n" + (_SCALED_BENCH_CONSTRAINTS % bench.upper())) if bench != "pure" else ""
    return """You are the Coder in an autonomous recommender-system research pipeline.
Translate the supplied hypothesis into a surgical patch for the complete current Python module below.

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
- Preserve the function signature run_fm(splits, ...) exactly; the Executor relies on it.
- run_fm must accept return_predictions: bool = False. When it is True, return a test_scores array or list aligned to the test split row order alongside valid and test metrics. Its default must remain False.
- Keep the existing top-of-file data-loading imports (from data import load, encode on Pure; from data_1k or data_27k import load, encode on those benchmarks) unless the hypothesis specifically requires changing them.
- The diff must implement the hypothesis's actual mechanism, not adjust existing parameters, reformat code, add unused constants, cast dtypes, or add helper functions that are not called from the active code path.
- A patch that does not change the model's actual computation is invalid regardless of its size or syntax validity.
- If the hypothesis requires a substantial change (a new loss function, new sampling logic, a new layer), write that logic and wire it into the function that's actually called -- don't leave it unused.
- The Executor calls run_fm(splits) with no extra keyword arguments beyond an optional seed override -- there is no sweep and no way to test more than one configuration in a single iteration. If the hypothesis names a parameter value (or several candidate values to try), pick ONE and bake it in as what actually executes: as the parameter's own default, or hardcoded. Never add a new parameter or capability while leaving its default at the value that reproduces the original computation exactly (a weight of 1.0, a probability or epsilon of 0.0, a dropout rate of 0, etc.) -- that patch will score bit-identical to the current best and be flagged as a no-op, not evaluated as a negative result, wasting the iteration it consumed.

Execution contract: the normal runner invokes run_fm(splits) in an isolated subprocess and receives only validation metrics. Finalization alone invokes run_fm(splits, return_predictions=True), which must return aligned test_scores. Do not alter evaluate.py, change the split names, remove valid/test metric keys, or add code that exposes test scores during normal iteration. Preserve data.load() row order: never reorder, filter, or resort rows in a way that breaks submission row_id-to-(user_id, video_id) alignment; ml_modelling.explib.dataset.verify_row_order_matches_starter_kit() verifies this invariant. Preserve the baseline forward-pass structure, Adam optimizer update, and initialization unless the hypothesis explicitly changes one of them, so an accepted result is attributable to the stated hypothesis.""" + scaled_note + """

Current code follows:
----- CURRENT CODE -----
""" + current_code + "\n----- END CURRENT CODE -----"


def propose_patch(current_code: str, hypothesis: Dict[str, Any], bench: str = "pure") -> CoderResult:
    """Request a Search/Replace patch for one validated, planner-owned hypothesis."""
    _validate_hypothesis(hypothesis)
    user_prompt = "\n".join(
        (
            "Hypothesis description: " + hypothesis["description"],
            "Rationale: " + hypothesis["rationale"],
            "Target module: " + hypothesis["target_module"],
        )
    )
    response = call_llm(
        _system_prompt(current_code, bench), user_prompt, model=resolve_model("CODER"),
        temperature=resolve_temperature("CODER"), role="CODER",
    )
    return CoderResult(
        diff=extract_patch_blocks(_clean_diff(response.text)),
        raw_response=response.text,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )