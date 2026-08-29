#!/usr/bin/env python3
"""Generate and validate a submission CSV from a finalized run's active code."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.data_cache import load_splits
from agent.runner import score_final_on_test
from agent.state import RunState
from submit import write_submission


def _add_prediction_contract_for_legacy_run(code: str) -> str:
    """Upgrade older saved runs in memory; new Coder output already has this contract."""
    signature = "def run_fm(splits, k=16, lr=0.001, epochs=40, bs=8192, patience=4, seed=0, verbose=True):"
    legacy_return = """return {'valid': evaluate(uva, yva, m.predict(Xva)),
            'test':  evaluate(ute, yte, m.predict(Xte))}"""
    prediction_return = """test_scores = m.predict(Xte)
    result = {'valid': evaluate(uva, yva, m.predict(Xva)),
              'test': evaluate(ute, yte, test_scores)}
    if return_predictions:
        result['test_scores'] = test_scores
    return result"""
    if "return_predictions" in code:
        return code
    if signature not in code or legacy_return not in code:
        raise ValueError("Legacy run code cannot be upgraded to the final prediction contract automatically.")
    return code.replace(signature, signature[:-2] + ", return_predictions=False):", 1).replace(legacy_return, prediction_return, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "KuaiRand-Pure" / "data")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache")
    args = parser.parse_args()
    run_dir = args.runs_dir / args.run_id
    state = RunState.load(run_dir / "state.json")
    final_code = _add_prediction_contract_for_legacy_run(state.current_code)
    if final_code != state.current_code:
        print("Applied in-memory compatibility upgrade for legacy return_predictions contract.")
    result = score_final_on_test(final_code, args.data_dir, args.cache_dir)
    if result["status"] != "ok":
        raise RuntimeError(result.get("error_trace", "Final scorer timed out."))
    rows = load_splits(args.data_dir, args.cache_dir)["test"]
    scores = result["test_scores"]
    if len(scores) != len(rows):
        raise ValueError("Final scorer returned %d scores for %d test rows." % (len(scores), len(rows)))
    output = run_dir / "submission.csv"
    write_submission(output, rows, scores)
    subprocess.run([sys.executable, str(ROOT / "submit.py"), "--check", "--split", "test", "--data_dir", str(args.data_dir), str(output)], cwd=ROOT, check=True)
    metrics = result["metrics"]["test"]
    print("submission validated: %s" % output)
    print("final test: GAUC %.6f | nDCG@5 %.6f | primary %.6f" % (metrics["GAUC"], metrics["nDCG@5"], metrics["primary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())