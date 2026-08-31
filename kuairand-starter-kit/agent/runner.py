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

    candidate_path, data_dir, cache_dir, result_kind, scores_path, bench, seed_arg = sys.argv[1:]
    try:
        from agent.data_cache import load_and_encode
        spec = importlib.util.spec_from_file_location("candidate_model", candidate_path)
        candidate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(candidate)
        splits, encoded, field_dims = load_and_encode(data_dir, cache_dir, bench=bench)
        original_encode = getattr(candidate, "encode", None)
        candidate.encode = lambda requested_splits: (encoded, field_dims) if requested_splits is splits else original_encode(requested_splits)
        run_kwargs = {} if seed_arg == "" else {"seed": int(seed_arg)}
        if result_kind in ("final", "confirm"):
            run_kwargs["return_predictions"] = True
        with contextlib.redirect_stdout(sys.stderr):
            results = candidate.run_fm(splits, **run_kwargs)
        if not isinstance(results, dict):
            raise TypeError("candidate.run_fm(splits) must return a dictionary")
        if result_kind == "validation":
            results.pop("test", None)
            results.pop("test_scores", None)
            valid_metrics = results.pop("valid")
            print(json.dumps({"status": "ok", "metrics": {"valid": valid_metrics}}, default=lambda value: value.item()))
        elif result_kind == "confirm":
            # One-shot valid+test read from a single training run, for a final
            # promotion confirmation only (e.g. the 27K run in
            # scripts/run_agent_scaled.py) -- never used inside the iterative
            # search loop, which stays validation-only by the "validation" branch
            # above so no candidate is ever selected on test-set performance.
            if "test_scores" not in results:
                raise ValueError("run_fm(..., return_predictions=True) must return test_scores")
            valid_metrics, test_metrics = results["valid"], results["test"]
            import numpy as np
            np.save(scores_path, np.asarray(results["test_scores"], dtype=np.float64))
            print(json.dumps({"status": "ok", "metrics": {"valid": valid_metrics, "test": test_metrics}}, default=lambda value: value.item()))
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
    code: str, data_dir: Path | str, cache_dir: Optional[Path | str], timeout_s: float, result_kind: str,
    bench: str = "pure", seed: Optional[int] = None,
) -> Dict[str, Any]:
    cache_path = Path(cache_dir) if cache_dir is not None else _ROOT / ".cache"
    with tempfile.TemporaryDirectory(prefix="candidate-") as temporary_directory:
        candidate_path = Path(temporary_directory) / "candidate.py"
        scores_path = Path(temporary_directory) / "test_scores.npy"
        candidate_path.write_text(code, encoding="utf-8")
        try:
            completed = subprocess.run(
                [
                    sys.executable, "-c", _CHILD_PROGRAM, str(candidate_path), str(data_dir), str(cache_path),
                    result_kind, str(scores_path), bench, "" if seed is None else str(seed),
                ],
                cwd=_ROOT,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout"}
        test_scores = None
        if result_kind in ("final", "confirm") and completed.returncode == 0:
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
    if result_kind in ("final", "confirm") and result.get("status") == "ok":
        if test_scores is None:
            return {"status": "error", "error_trace": "Final scorer did not produce readable test scores."}
        result["test_scores"] = test_scores
    return result


def run(
    code: str, data_dir: Path | str, cache_dir: Optional[Path | str] = None, timeout_s: float = 300,
    bench: str = "pure", seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute a candidate and expose only validation metrics to the orchestrator.

    ``seed`` is forwarded to the candidate's ``run_fm(splits, seed=...)`` when
    given (None keeps the candidate's own default -- almost always seed 0), so a
    caller can replicate the same candidate code over several seeds without
    regenerating a patch. See scripts/run_agent_scaled.py for how this is used
    to implement the codebase's 3-seed replication gate on a promising 1K result.
    """
    return _run_subprocess(code, data_dir, cache_dir, timeout_s, "validation", bench=bench, seed=seed)


def score_final_on_test(
    code: str, data_dir: Path | str, cache_dir: Optional[Path | str] = None, timeout_s: float = 300,
    bench: str = "pure", seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Return test metrics only after iterative optimization has stopped."""
    return _run_subprocess(code, data_dir, cache_dir, timeout_s, "final", bench=bench, seed=seed)


def score_confirm(
    code: str, data_dir: Path | str, cache_dir: Optional[Path | str] = None, timeout_s: float = 300,
    bench: str = "pure", seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Return BOTH valid and test metrics from a single training run.

    For a one-shot promotion confirmation only (a 1K-replicated candidate's
    single, explicitly non-replicated run on 27K -- see
    scripts/run_agent_scaled.py), where a benchmark this expensive can't afford
    two independent training passes just to keep valid/test artificially
    separated the way the iterative loop does. Never call this from inside a
    search loop: it exposes test metrics on every call, which run()/validation
    scoring deliberately never does.
    """
    return _run_subprocess(code, data_dir, cache_dir, timeout_s, "confirm", bench=bench, seed=seed)