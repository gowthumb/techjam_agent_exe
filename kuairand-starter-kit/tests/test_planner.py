import unittest
from unittest.mock import patch

from agent.llm_client import LLMResponse
from agent.planner import _history_context, propose_hypothesis
from agent.state import RunState


class PlannerSmokeTest(unittest.TestCase):
    def test_parses_fenced_json_and_uses_planner_model(self):
        response = LLMResponse("```json\n{\"description\": \"Try BPR.\", \"rationale\": \"Priority item 1.\", \"target_module\": \"loss_function\"}\n```", 12, 4)
        with patch("agent.planner.call_llm", return_value=response) as call, patch(
            "agent.planner.resolve_model", return_value="planner-model"
        ):
            result = propose_hypothesis(RunState(current_code="", best_metrics={"primary": 0.6}))
        self.assertEqual(result.hypothesis["target_module"], "loss_function")
        self.assertEqual(call.call_args.kwargs["model"], "planner-model")

    def test_history_keeps_only_five_full_recent_entries(self):
        history = [{"hypothesis": "trial %d" % index, "status": "rejected", "metrics": {"valid": {"primary": 0.5}}} for index in range(7)]
        context = _history_context(history)
        self.assertIn("trial 0 | rejected | primary=0.500000", context)
        self.assertIn('"trial 6"', context)
        self.assertNotIn('"trial 0"', context)


if __name__ == "__main__":
    unittest.main()