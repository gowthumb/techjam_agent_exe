"""Isolated candidate execution with a validation-only iterative result boundary."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


_ROOT = Path(__file__).resolve().parents[1]
_CHILD_PROGRAM = textwrap.dedent(
    """
    import contextlib
    import importlib.util
    import json
    import sys
    import traceback

    candidate_path, data_dir, cache_dir, result_kind, scores_path = sys.argv[1:]
    try:
        from agent.data_cache import load_and_encode
        spec = importlib.util.spec_from_file_location("candidate_model", candidate_path)
        candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)
        splits, encoded, field_dims = load_and_encode(data_dir, cache_dir)
        original_encode = getattr(candidate, "encode", None)
        candidate.encode = lambda requested_splits: (encoded, field_dims) if requested_splits is splits else original_encode(requested_splits)
        with contextlib.redirect_stdout(sys.stderr):
            results = candidate.run_fm(splits, return_predictions=True) if result_kind == "final" else candidate.run_fm(splits)
        if not isinstance(results, dict):
            raise TypeError("candidate.run_fm(splits) must return a dictionary")
        if result_kind == "validation":
            results.pop("test", None)
            results.pop("test_scores", None)
            valid_metrics = results.pop("valid")
            print(json.dumps({"status": "ok", "metrics": {"valid": valid_metrics}}, default=lambda value: value.item()))
        else:
            if "test_scores" not in results:
                raise ValueError("run_fm(..., return_predictions=True) must return test_scores")
            test_metrics = results["test"]
            import numpy as np
            np.save(scores_path, np.asarray(results["test_scores"], dtype=np.float64))
            print(json.dumps({"status": "ok", "metrics": {"test": test_metrics}}, default=lambda value: value.item()))
    except Exception:
        print(json.dumps({"status": "error", "error_trace": traceback.format_exc()}))
    """
)


def _run_subprocess(
    code: str, data_dir: Path | str, cache_dir: Optional[Path | str], timeout_s: float, result_kind: str
) -> Dict[str, Any]:
    cache_path = Path(cache_dir) if cache_dir is not None else _ROOT / ".cache"
    with tempfile.TemporaryDirectory(prefix="candidate-") as temporary_directory:
        candidate_path = Path(temporary_directory) / "candidate.py"
        scores_path = Path(temporary_directory) / "test_scores.npy"
        candidate_path.write_text(code, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-c", _CHILD_PROGRAM, str(candidate_path), str(data_dir), str(cache_path), result_kind, str(scores_path)],
                cwd=_ROOT,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout"}
        if result_kind == "final" and completed.returncode == 0:
            try:
                test_scores = np.load(scores_path)
            except (OSError, ValueError):
                test_scores = None

    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "status": "error",
            "error_trace": "Runner did not receive valid JSON from its child process.\n" + completed.stderr,
        }
    if completed.returncode != 0 and result.get("status") == "ok":
        return {"status": "error", "error_trace": completed.stderr}
    if result_kind == "final" and result.get("status") == "ok":
        if test_scores is None:
            return {"status": "error", "error_trace": "Final scorer did not produce readable test scores."}
        result["test_scores"] = test_scores
    return result


def run(
    code: str, data_dir: Path | str, cache_dir: Optional[Path | str] = None, timeout_s: float = 300
) -> Dict[str, Any]:
    """Execute a candidate and expose only validation metrics to the orchestrator."""
    return _run_subprocess(code, data_dir, cache_dir, timeout_s, "validation")


def score_final_on_test(
    code: str, data_dir: Path | str, cache_dir: Optional[Path | str] = None, timeout_s: float = 300
) -> Dict[str, Any]:
    """Return test metrics only after iterative optimization has stopped."""
    return _run_subprocess(code, data_dir, cache_dir, timeout_s, "final")