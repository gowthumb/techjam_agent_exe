import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.dependencies import InstallResult
from agent.executor import check_caps, check_convergence, run_candidate
from agent.state import RunState


DIFF = "<<<<<<< SEARCH\nvalue = 1\n=======\nvalue = 2\n>>>>>>> REPLACE"
VALID_METRICS = {"valid": {"GAUC": 0.7, "nDCG@5": 0.6, "primary": 0.65}}


class ExecutorSmokeTest(unittest.TestCase):
    def test_accepts_validation_improvement_without_test_metrics(self):
        state = RunState(current_code="value = 1\n", best_metrics={"GAUC": 0.6, "nDCG@5": 0.5, "primary": 0.55})
        with tempfile.TemporaryDirectory() as temporary_directory, patch(
            "agent.executor.runner.run", return_value={"status": "ok", "metrics": VALID_METRICS}
        ), patch("agent.executor.resolve_model", return_value="test-model"):
            result = run_candidate(state, DIFF, runs_dir=Path(temporary_directory), tokens_used=13)
        self.assertEqual(result.status, "accepted")
        self.assertEqual(state.iteration_num, 1)
        self.assertEqual(state.best_metrics["primary"], 0.65)
        self.assertEqual(state.total_tokens, 13)
        self.assertEqual(set(result.metrics), {"valid"})

    def test_patch_failure_does_not_execute_or_increment_iteration(self):
        state = RunState(current_code="value = 1\n")
        invalid_diff = "<<<<<<< SEARCH\nmissing\n=======\nreplacement\n>>>>>>> REPLACE"
        with tempfile.TemporaryDirectory() as temporary_directory, patch("agent.executor.runner.run") as run, patch(
            "agent.executor.resolve_model", return_value="test-model"
        ):
            result = run_candidate(state, invalid_diff, runs_dir=Path(temporary_directory))
        self.assertEqual(result.status, "error")
        self.assertEqual(state.iteration_num, 0)
        run.assert_not_called()

    def test_runtime_failure_does_not_increment_iteration(self):
        state = RunState(current_code="value = 1\n")
        with tempfile.TemporaryDirectory() as temporary_directory, patch(
            "agent.executor.runner.run", return_value={"status": "error", "error_trace": "RuntimeError: broken"}
        ), patch("agent.executor.resolve_model", return_value="test-model"):
            result = run_candidate(state, DIFF, runs_dir=Path(temporary_directory))
        self.assertEqual(result.status, "error")
        self.assertEqual(state.iteration_num, 0)
        self.assertEqual(state.retry_count, 1)

    def test_convergence_uses_best_primary_over_three_intervals(self):
        state = RunState(current_code="")
        state.experiment_history = [
            {"status": "accepted" if index == 0 else "rejected", "metrics": {"valid": {"primary": primary}}}
            for index, primary in enumerate((0.6000, 0.6010, 0.6015, 0.6019))
        ]
        self.assertTrue(check_convergence(state))

    def test_rejection_only_history_cannot_converge(self):
        state = RunState(current_code="")
        state.experiment_history = [
            {"status": "rejected", "metrics": {"valid": {"primary": 0.6013}}}
            for _ in range(4)
        ]
        self.assertFalse(check_convergence(state))

    def test_flat_rejections_before_first_acceptance_do_not_converge(self):
        state = RunState(current_code="")
        state.experiment_history = [
            {"status": "rejected", "metrics": {"valid": {"primary": 0.6015}}} for _ in range(6)
        ] + [{"status": "accepted", "metrics": {"valid": {"primary": 0.6032}}}]
        self.assertFalse(check_convergence(state))

    def test_three_flat_iterations_after_acceptance_converge(self):
        state = RunState(current_code="")
        state.experiment_history = [
            {"status": "rejected", "metrics": {"valid": {"primary": 0.6015}}} for _ in range(5)
        ] + [
            {"status": "accepted", "metrics": {"valid": {"primary": 0.6032}}},
            {"status": "rejected", "metrics": {"valid": {"primary": 0.6030}}},
            {"status": "rejected", "metrics": {"valid": {"primary": 0.6034}}},
            {"status": "rejected", "metrics": {"valid": {"primary": 0.6029}}},
        ]
        self.assertTrue(check_convergence(state))

    def test_abandoned_iterations_do_not_advance_the_window(self):
        state = RunState(current_code="")
        state.experiment_history = [
            {"status": "accepted", "metrics": {"valid": {"primary": 0.6032}}},
            {"status": "abandoned", "metrics": None},
            {"status": "error", "metrics": None},
            {"status": "rejected", "metrics": {"valid": {"primary": 0.6030}}},
        ]
        self.assertFalse(check_convergence(state))

    def test_second_acceptance_resets_the_window(self):
        state = RunState(current_code="")
        state.experiment_history = [
            {"status": "accepted", "metrics": {"valid": {"primary": 0.6032}}},
            {"status": "rejected", "metrics": {"valid": {"primary": 0.6031}}},
            {"status": "rejected", "metrics": {"valid": {"primary": 0.6030}}},
            {"status": "accepted", "metrics": {"valid": {"primary": 0.6060}}},
            {"status": "rejected", "metrics": {"valid": {"primary": 0.6058}}},
        ]
        self.assertFalse(check_convergence(state))

    def test_acceptance_requires_clearing_the_noise_band(self):
        state = RunState(current_code="value = 1\n", best_metrics={"GAUC": 0.66, "nDCG@5": 0.53, "primary": 0.6016})
        near_noise = {"valid": {"GAUC": 0.66, "nDCG@5": 0.53, "primary": 0.6026}}  # +0.0010, below band
        with tempfile.TemporaryDirectory() as temporary_directory, patch(
            "agent.executor.runner.run", return_value={"status": "ok", "metrics": near_noise}
        ), patch("agent.executor.resolve_model", return_value="test-model"):
            result = run_candidate(state, DIFF, runs_dir=Path(temporary_directory))
        self.assertEqual(result.status, "rejected")
        self.assertEqual(state.best_metrics["primary"], 0.6016)

    def test_acceptance_when_gain_exceeds_the_noise_band(self):
        state = RunState(current_code="value = 1\n", best_metrics={"GAUC": 0.66, "nDCG@5": 0.53, "primary": 0.6016})
        real_gain = {"valid": {"GAUC": 0.667, "nDCG@5": 0.536, "primary": 0.6038}}  # +0.0022, above band
        with tempfile.TemporaryDirectory() as temporary_directory, patch(
            "agent.executor.runner.run", return_value={"status": "ok", "metrics": real_gain}
        ), patch("agent.executor.resolve_model", return_value="test-model"):
            result = run_candidate(state, DIFF, runs_dir=Path(temporary_directory))
        self.assertEqual(result.status, "accepted")
        self.assertEqual(state.best_metrics["primary"], 0.6038)

    def test_no_op_patch_is_flagged_distinctly_not_silently_rejected(self):
        """A candidate whose valid metrics exactly match the current best did not
        change the model's computation (observed directly: a Coder diff that only
        added a no-op-default parameter). It must not be logged as an ordinary
        rejection -- that would make a wasted iteration look like real negative
        evidence."""
        best = {"GAUC": 0.66, "nDCG@5": 0.53, "primary": 0.6016}
        state = RunState(current_code="value = 1\n", best_metrics=dict(best))
        identical = {"valid": dict(best)}
        with tempfile.TemporaryDirectory() as temporary_directory, patch(
            "agent.executor.runner.run", return_value={"status": "ok", "metrics": identical}
        ), patch("agent.executor.resolve_model", return_value="test-model"):
            result = run_candidate(state, DIFF, runs_dir=Path(temporary_directory))
        self.assertEqual(result.status, "no_op")
        self.assertIsNotNone(result.error_trace)
        self.assertIn("bit-for-bit identical", result.error_trace)
        # Did not consume an iteration slot or the current best -- mirrors how a
        # pre-scoring error is handled, since nothing was actually tested.
        self.assertEqual(state.iteration_num, 0)
        self.assertEqual(state.retry_count, 1)
        self.assertEqual(state.best_metrics, best)

    def test_genuinely_different_worse_candidate_is_rejected_not_no_op(self):
        """Sanity check the no-op detector isn't over-triggering: a candidate that
        actually is different (and worse) must still be a normal rejection."""
        state = RunState(current_code="value = 1\n", best_metrics={"GAUC": 0.66, "nDCG@5": 0.53, "primary": 0.6016})
        worse = {"valid": {"GAUC": 0.64, "nDCG@5": 0.51, "primary": 0.5900}}
        with tempfile.TemporaryDirectory() as temporary_directory, patch(
            "agent.executor.runner.run", return_value={"status": "ok", "metrics": worse}
        ), patch("agent.executor.resolve_model", return_value="test-model"):
            result = run_candidate(state, DIFF, runs_dir=Path(temporary_directory))
        self.assertEqual(result.status, "rejected")
        self.assertEqual(state.iteration_num, 1)

    def test_missing_module_triggers_one_auto_install_and_a_retry(self):
        """A ModuleNotFoundError must trigger exactly one install attempt and one
        retry of the SAME code -- and if the retry then succeeds, the result is
        scored normally, not reported as an error."""
        state = RunState(current_code="value = 1\n")
        module_error = {"status": "error", "error_trace": "ModuleNotFoundError: No module named 'cowsay'"}
        with tempfile.TemporaryDirectory() as temporary_directory, patch(
            "agent.executor.runner.run", side_effect=[module_error, {"status": "ok", "metrics": VALID_METRICS}]
        ) as run, patch(
            "agent.executor.dependencies.install", return_value=InstallResult(True, "installed cowsay")
        ) as install, patch("agent.executor.resolve_model", return_value="test-model"):
            result = run_candidate(state, DIFF, runs_dir=Path(temporary_directory))
        install.assert_called_once_with("cowsay")
        self.assertEqual(run.call_count, 2)
        self.assertEqual(result.status, "accepted")
        self.assertEqual(state.attempted_installs, ["cowsay"])

    def test_missing_module_is_only_ever_attempted_once_per_run(self):
        """A second candidate hitting the SAME missing module must not trigger a
        second install attempt -- state.attempted_installs caps it."""
        state = RunState(current_code="value = 1\n", attempted_installs=["cowsay"])
        module_error = {"status": "error", "error_trace": "ModuleNotFoundError: No module named 'cowsay'"}
        with tempfile.TemporaryDirectory() as temporary_directory, patch(
            "agent.executor.runner.run", return_value=module_error
        ) as run, patch("agent.executor.dependencies.install") as install, patch(
            "agent.executor.resolve_model", return_value="test-model"
        ):
            result = run_candidate(state, DIFF, runs_dir=Path(temporary_directory))
        install.assert_not_called()
        self.assertEqual(run.call_count, 1)
        self.assertEqual(result.status, "error")

    def test_failed_install_falls_through_to_the_normal_error_path(self):
        state = RunState(current_code="value = 1\n")
        module_error = {"status": "error", "error_trace": "ModuleNotFoundError: No module named 'cowsay'"}
        with tempfile.TemporaryDirectory() as temporary_directory, patch(
            "agent.executor.runner.run", return_value=module_error
        ) as run, patch(
            "agent.executor.dependencies.install",
            return_value=InstallResult(False, "no matching distribution"),
        ), patch("agent.executor.resolve_model", return_value="test-model"):
            result = run_candidate(state, DIFF, runs_dir=Path(temporary_directory))
        # Install was attempted (and failed) but never re-ran the candidate.
        self.assertEqual(run.call_count, 1)
        self.assertEqual(result.status, "error")
        self.assertEqual(state.attempted_installs, ["cowsay"])

    def test_caps_cover_iteration_and_wall_clock_limits(self):
        self.assertTrue(check_caps(RunState(current_code="", iteration_num=50)))
        self.assertTrue(check_caps(RunState(current_code="", total_wall_clock_s=6 * 3600)))
        self.assertFalse(check_caps(RunState(current_code="", iteration_num=49, total_wall_clock_s=1.0)))


if __name__ == "__main__":
    unittest.main()