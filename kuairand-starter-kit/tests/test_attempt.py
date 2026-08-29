import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.attempt import attempt_hypothesis
from agent.coder import CoderResult
from agent.executor import IterationResult
from agent.state import RunState


HYPOTHESIS = {"description": "Fix the candidate.", "rationale": "Exercise recovery.", "target_module": "model"}
BROKEN_DIFF = "<<<<<<< SEARCH\nvalue = 1\n=======\ndef broken(:\n>>>>>>> REPLACE"
FIXED_DIFF = "<<<<<<< SEARCH\nvalue = 1\n=======\nvalue = 2\n>>>>>>> REPLACE"


class AttemptSmokeTest(unittest.TestCase):
    def test_debugger_retries_after_pre_scoring_failure(self):
        state = RunState(current_code="value = 1\n")
        repair = CoderResult(FIXED_DIFF, FIXED_DIFF, 4, 2)
        with tempfile.TemporaryDirectory() as temporary_directory, patch(
            "agent.attempt.fix_patch", return_value=repair
        ) as debugger, patch(
            "agent.attempt.run_candidate",
            side_effect=[IterationResult("error", error_trace="SyntaxError: invalid"), IterationResult("rejected", metrics={"valid": {"primary": 0.5}})],
        ) as runner:
            result = attempt_hypothesis(
                state, HYPOTHESIS, max_retries=1, runs_dir=Path(temporary_directory), initial_diff=BROKEN_DIFF
            )
        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.retries_used, 1)
        self.assertEqual(len(result.attempted_diffs), 2)
        debugger.assert_called_once()
        self.assertEqual(runner.call_count, 2)


if __name__ == "__main__":
    unittest.main()