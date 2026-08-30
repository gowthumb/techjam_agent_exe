import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.attempt import attempt_hypothesis
from agent.coder import CoderResult
from agent.executor import IterationResult
from agent.patcher import PatchError
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

    def test_malformed_coder_and_debugger_responses_are_abandoned_without_crashing(self):
        state = RunState(current_code="value = 1\n")
        malformed = PatchError("No valid Search/Replace block found.")
        with tempfile.TemporaryDirectory() as temporary_directory, patch(
            "agent.attempt.propose_patch", side_effect=malformed
        ) as coder, patch(
            "agent.attempt.fix_patch", side_effect=malformed
        ) as debugger, patch("agent.attempt.run_candidate") as runner, patch(
            "agent.attempt.resolve_model", return_value="test-model"
        ):
            result = attempt_hypothesis(state, HYPOTHESIS, max_retries=3, runs_dir=Path(temporary_directory))
        self.assertEqual(result.status, "abandoned")
        self.assertEqual(result.retries_used, 3)
        self.assertEqual(state.iteration_num, 0)
        self.assertEqual(coder.call_count, 1)
        self.assertEqual(debugger.call_count, 3)
        runner.assert_not_called()
        self.assertEqual(len(result.attempted_diffs), 4)
        self.assertEqual(len(result.errors), 4)
        self.assertTrue(all("PatchError" in error for error in result.errors))
        self.assertEqual(state.experiment_history[-1]["status"], "abandoned")
        self.assertEqual(state.experiment_history[-1]["code_diff"], result.attempted_diffs)


if __name__ == "__main__":
    unittest.main()