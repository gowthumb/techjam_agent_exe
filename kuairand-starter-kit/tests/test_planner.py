import unittest
from unittest.mock import patch

from agent.llm_client import LLMResponse
from agent.planner import _decision_rules_context, _history_context, _planner_knowledge_context, _system_prompt, propose_hypothesis
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
        relevant = prompt.index("ITERATION-RELEVANT KNOWLEDGE BASE")
        self.assertLess(briefing, relevant)
        self.assertIn("- fm_bpr", prompt[briefing:relevant])
        self.assertNotIn("experimental_feature", prompt[briefing:relevant])

    def test_planner_context_keeps_only_iteration_relevant_sections(self):
        knowledge_base = """meta:
  schema_version: 3
calibration:
  official: 0.5946
decision_protocol:
  replication_rule: replicate near-best candidates
  control_rule: add a control
  unbiased_veto: reject biased wins
candidate_models:
  - name: fm_bpr
    validated: true
dead_ends:
  - dead end
feature_engineering_menu:
  menu: []
priors:
  - priority rule
dataset_facts:
  rows: 1
kb_ablation:
  result: ignored
scale_transfer:
  headline: do not apply this KB to 1K blind
"""
        context = _planner_knowledge_context(knowledge_base)
        self.assertIn("decision_protocol:", context)
        self.assertIn("candidate_models:", context)
        self.assertIn("dead_ends:", context)
        self.assertIn("feature_engineering_menu:", context)
        self.assertIn("priors:", context)
        self.assertIn("scale_transfer:", context)
        self.assertNotIn("meta:", context)
        self.assertNotIn("calibration:", context)
        self.assertNotIn("dataset_facts:", context)
        self.assertNotIn("kb_ablation:", context)
        rules = _decision_rules_context(knowledge_base)
        self.assertIn("replication_rule", rules)
        self.assertIn("control_rule", rules)
        self.assertIn("unbiased_veto", rules)
        self.assertIn("attribution_invariant", rules)
        self.assertIn("Adam optimizer update", rules)


if __name__ == "__main__":
    unittest.main()