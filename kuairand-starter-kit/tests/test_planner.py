import unittest
from unittest.mock import patch

from agent.llm_client import LLMResponse
from agent.planner import _history_context, _system_prompt, propose_hypothesis
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
        self.assertEqual(call.call_args.kwargs["temperature"], 0.6)

    def test_history_keeps_only_five_full_recent_entries(self):
        history = [{"hypothesis": "trial %d" % index, "status": "rejected", "metrics": {"valid": {"primary": 0.5}}} for index in range(7)]
        context = _history_context(history)
        self.assertIn("trial 0 | rejected | primary=0.500000", context)
        self.assertIn('"trial 6"', context)
        self.assertNotIn('"trial 0"', context)

    def test_prompt_surfaces_confirmed_models_before_full_knowledge_base(self):
        knowledge_base = """candidate_models:
    - name: fm_bpr
    validated: true
    result: confirmed ranking-loss improvement
    recommended: true
    - name: experimental_feature
    validated: false
# end
"""
        prompt = _system_prompt(knowledge_base, RunState(current_code=""))
        briefing = prompt.index("CONFIRMED / HIGH-CONFIDENCE")
        full = prompt.index("FULL KNOWLEDGE BASE")
        self.assertLess(briefing, full)
        self.assertIn("- fm_bpr", prompt[briefing:full])
        self.assertNotIn("experimental_feature", prompt[briefing:full])


if __name__ == "__main__":
    unittest.main()