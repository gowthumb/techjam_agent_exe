"""Search/replace patch application for candidate-model source code."""
from __future__ import annotations

import ast
import re


class PatchError(ValueError):
    """Raised when a candidate Search/Replace patch cannot be applied safely."""


class SearchMatchError(PatchError):
    """Raised when a SEARCH section does not identify exactly one source region."""


_BLOCK_PATTERN = re.compile(
    r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE", re.DOTALL
)


def _parse_blocks(diff: str) -> list[tuple[str, str]]:
    blocks = _BLOCK_PATTERN.findall(diff)
    if not blocks:
        raise PatchError(
            "No valid Search/Replace block found. Use <<<<<<< SEARCH, =======, and >>>>>>> REPLACE markers."
        )
    return blocks


def extract_patch_blocks(response: str) -> str:
    """Extract Search/Replace blocks from an LLM response that may contain prose."""
    blocks = _parse_blocks(response)
    return "\n\n".join(
        "<<<<<<< SEARCH\n%s\n=======\n%s\n>>>>>>> REPLACE" % (search, replacement)
        for search, replacement in blocks
    )


def apply_patch(current_code: str, diff: str) -> str:
    """Apply each exact, uniquely matching SEARCH block to candidate source."""
    updated_code = current_code
    for block_number, (search, replacement) in enumerate(_parse_blocks(diff), start=1):
        match_count = updated_code.count(search)
        if match_count != 1:
            raise SearchMatchError(
                "SEARCH block %d matched %d times; it must match exactly once. Searched text:\n%s"
                % (block_number, match_count, search)
            )
        updated_code = updated_code.replace(search, replacement, 1)
    return updated_code


def validate_syntax(code: str) -> None:
    """Reject invalid candidate source before executing it in a subprocess."""
    try:
        ast.parse(code)
    except SyntaxError as error:
        raise SyntaxError("Candidate code has invalid Python syntax: %s" % error) from error