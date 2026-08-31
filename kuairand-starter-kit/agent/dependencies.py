"""Best-effort auto-install for a candidate's missing third-party imports.

Motivated directly by a real incident: a Planner-proposed CatBoost hypothesis
burned 3 Coder/Debugger round-trips (real LLM tokens, real wall-clock) purely
because `catboost` wasn't installed -- not a code problem, an environment
problem no amount of patching could fix. `agent/executor.py::run_candidate`
now recognizes a `ModuleNotFoundError` specifically, attempts one `pip
install` for the missing package, and retries the SAME code once before
falling through to the normal error/Debugger-repair path. This never touches
the Coder/Debugger retry budget -- installing a dependency isn't a patch, so
it shouldn't cost one.

Safety, deliberately conservative given this installs and executes real
third-party code from PyPI on the strength of an LLM-generated import:
  - only ever triggers on a literal `ModuleNotFoundError: No module named
    'X'` parsed out of a real traceback, never on an arbitrary string a
    candidate could construct -- the module name comes from Python's own
    import machinery, not from candidate-controlled text;
  - the resolved package name is validated against a strict identifier
    pattern before ever reaching a subprocess argv list;
  - `pip install <pkg>` only, no custom index/extra-index URLs, no `-e`,
    no arbitrary pip arguments;
  - every attempt (success or failure) is recorded via
    agent.logging_utils.log_intervention, the same conspicuous
    audit trail already used for other autonomous, human-notable actions --
    an install is not something that should ever happen silently;
  - at most one install attempt per (module, RunState) -- state.attempted_installs
    tracks what's already been tried so a package that fails to fix the
    import (wrong pip name, install succeeds but the module still errors for
    an unrelated reason) doesn't get retried every subsequent iteration.
"""
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional


_MODULE_NOT_FOUND_RE = re.compile(r"ModuleNotFoundError: No module named '([A-Za-z_][A-Za-z0-9_.]*)'")
_SAFE_PACKAGE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]{0,213}$")  # PyPI's own name-length ceiling is 214

# import name -> actual PyPI distribution name, for the common cases where they
# differ. Not exhaustive by design: anything not listed here is assumed to
# share its import name and pip name, which covers the large majority of
# packages (including catboost, xgboost, pandas -- everything seen in this
# codebase's own dependency history so far).
_PIP_NAME_OVERRIDES = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "yaml": "pyyaml",
    "PIL": "pillow",
    "google": "google-api-python-client",
}

_INSTALL_TIMEOUT_S = 300.0


@dataclass(frozen=True)
class InstallResult:
    ok: bool
    message: str


def missing_module(error_trace: Optional[str]) -> Optional[str]:
    """Return the top-level missing module name if error_trace is a
    ModuleNotFoundError, else None. 'a.b.c' -> 'a' (pip installs top-level
    packages; a submodule import failure means the top-level one is missing).
    """
    if not error_trace:
        return None
    match = _MODULE_NOT_FOUND_RE.search(error_trace)
    if not match:
        return None
    return match.group(1).split(".")[0]


def install(module_name: str, timeout_s: float = _INSTALL_TIMEOUT_S) -> InstallResult:
    """Attempt ``pip install`` for the PyPI package that provides module_name.

    Never raises -- a failure to install is reported in the returned
    InstallResult, not as an exception, so a caller can always fall through
    to its existing error handling.
    """
    pip_name = _PIP_NAME_OVERRIDES.get(module_name, module_name)
    if not _SAFE_PACKAGE_NAME_RE.match(pip_name):
        return InstallResult(False, "refused: %r does not look like a safe package name" % pip_name)
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--no-input", pip_name],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return InstallResult(False, "pip install %s timed out after %.0fs" % (pip_name, timeout_s))
    except OSError as error:
        return InstallResult(False, "pip install %s could not start: %s" % (pip_name, error))
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip()[-1500:]
        return InstallResult(False, "pip install %s failed (exit %d): %s" % (pip_name, completed.returncode, tail))
    return InstallResult(True, "installed %s (for import %s)" % (pip_name, module_name))
