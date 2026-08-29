import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.attempt import AttemptResult
from agent.executor import IterationResult
from agent.planner import PlannerResult
from agent.state import RunState
from scripts.run_agent import run_loop


HYPOTHESIS = {"description": "Try a ranking loss.", "rationale": "Knowledge-base item 1.", "target_module": "loss_function"}


class RunAgentTest(unittest.TestCase):
    def test_iteration_cap_finalizes_once_and_persists_summary(self):
        state = RunState(current_code="baseline", best_metrics={"GAUC": 0.6, "nDCG@5": 0.5, "primary": 0.55}, run_id="loop-test")
        planner = PlannerResult(HYPOTHESIS, "{}", 10, 5)

        def scored_attempt(run_state, *args, **kwargs):
            run_state.iteration_num += 1
            run_state.total_wall_clock_s += 1.0
            return AttemptResult("rejected", IterationResult("rejected", {"valid": {"primary": 0.5}}), [], [], 0)

        with tempfile.TemporaryDirectory() as temporary_directory, patch(
            "scripts.run_agent.propose_hypothesis", return_value=planner
        ) as propose, patch(
            "scripts.run_agent.attempt_hypothesis", side_effect=scored_attempt
        ), patch(
            "scripts.run_agent.score_final_on_test", return_value={"status": "ok", "metrics": {"test": {"primary": 0.6}}}
        ) as final_score:
            root = Path(temporary_directory)
            summary = run_loop(state, max_iterations=2, max_wallclock_hours=6, data_dir=root, runs_dir=root, cache_dir=root)
            saved_summary = json.loads((root / state.run_id / "summary.json").read_text(encoding="utf-8"))
            saved_state = RunState.load(root / state.run_id / "state.json")

        self.assertEqual(summary["stopping_reason"], "iteration cap")
        self.assertEqual(summary["total_iterations"], 2)
        self.assertEqual(summary["total_tokens"], 30)
        self.assertEqual(saved_summary, summary)
        self.assertEqual(saved_state.iteration_num, 2)
        self.assertEqual(propose.call_count, 2)
        final_score.assert_called_once()

    def test_attempt_exception_persists_elapsed_wall_time(self):
        state = RunState(current_code="baseline", run_id="failure-test")
        planner = PlannerResult(HYPOTHESIS, "{}", 1, 1)
        with tempfile.TemporaryDirectory() as temporary_directory, patch(
            "scripts.run_agent.propose_hypothesis", return_value=planner
        ), patch("scripts.run_agent.attempt_hypothesis", side_effect=RuntimeError("coder failed")), patch(
            "scripts.run_agent.monotonic", side_effect=[10.0, 15.0]
        ):
            root = Path(temporary_directory)
            with self.assertRaisesRegex(RuntimeError, "coder failed"):
                run_loop(state, max_iterations=2, max_wallclock_hours=6, data_dir=root, runs_dir=root, cache_dir=root)
            saved_state = RunState.load(root / state.run_id / "state.json")

        self.assertEqual(state.total_tokens, 2)
        self.assertEqual(state.total_wall_clock_s, 5.0)
        self.assertEqual(saved_state.total_wall_clock_s, 5.0)


if __name__ == "__main__":
    unittest.main()