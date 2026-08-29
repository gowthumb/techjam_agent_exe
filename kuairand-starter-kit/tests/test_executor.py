import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.executor import check_caps, check_convergence, run_candidate
from agent.state import RunState


DIFF = "<<<<<<< SEARCH\nvalue = 1\n=======\nvalue = 2\n>>>>>>> REPLACE"
VALID_METRICS = {"valid": {"GAUC": 0.7, "nDCG@5": 0.6, "primary": 0.65}}


class ExecutorSmokeTest(unittest.TestCase):
    def test_accepts_validation_improvement_without_test_metrics(self):
        state = RunState(current_code="value = 1\n", best_metrics={"GAUC": 0.6, "nDCG@5": 0.5, "primary": 0.55})
        with tempfile.TemporaryDirectory() as temporary_directory, patch(
            "agent.executor.runner.run", return_value={"status": "ok", "metrics": VALID_METRICS}
        ):
            result = run_candidate(state, DIFF, runs_dir=Path(temporary_directory), tokens_used=13)
        self.assertEqual(result.status, "accepted")
        self.assertEqual(state.iteration_num, 1)
        self.assertEqual(state.best_metrics["primary"], 0.65)
        self.assertEqual(state.total_tokens, 13)
        self.assertEqual(set(result.metrics), {"valid"})

    def test_patch_failure_does_not_execute_or_increment_iteration(self):
        state = RunState(current_code="value = 1\n")
        invalid_diff = "<<<<<<< SEARCH\nmissing\n=======\nreplacement\n>>>>>>> REPLACE"
        with tempfile.TemporaryDirectory() as temporary_directory, patch("agent.executor.runner.run") as run:
            result = run_candidate(state, invalid_diff, runs_dir=Path(temporary_directory))
        self.assertEqual(result.status, "error")
        self.assertEqual(state.iteration_num, 0)
        run.assert_not_called()

    def test_runtime_failure_does_not_increment_iteration(self):
        state = RunState(current_code="value = 1\n")
        with tempfile.TemporaryDirectory() as temporary_directory, patch(
            "agent.executor.runner.run", return_value={"status": "error", "error_trace": "RuntimeError: broken"}
        ):
            result = run_candidate(state, DIFF, runs_dir=Path(temporary_directory))
        self.assertEqual(result.status, "error")
        self.assertEqual(state.iteration_num, 0)
        self.assertEqual(state.retry_count, 1)

    def test_convergence_uses_best_primary_over_three_intervals(self):
        state = RunState(current_code="")
        state.experiment_history = [
            {"metrics": {"valid": {"primary": primary}}}
            for primary in (0.6000, 0.6010, 0.6015, 0.6019)
        ]
        self.assertTrue(check_convergence(state))

    def test_caps_cover_iteration_and_wall_clock_limits(self):
        self.assertTrue(check_caps(RunState(current_code="", iteration_num=50)))
        self.assertTrue(check_caps(RunState(current_code="", total_wall_clock_s=6 * 3600)))
        self.assertFalse(check_caps(RunState(current_code="", iteration_num=49, total_wall_clock_s=1.0)))


if __name__ == "__main__":
    unittest.main()