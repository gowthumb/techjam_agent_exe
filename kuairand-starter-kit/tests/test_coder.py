import unittest
from pathlib import Path
from unittest.mock import patch

from agent.coder import propose_patch
from agent.llm_client import LLMResponse


class CoderSmokeTest(unittest.TestCase):
    def test_strips_markdown_fences_and_preserves_token_counts(self):
        raw_response = "```text\n<<<<<<< SEARCH\nvalue = 1\n=======\nvalue = 2\n>>>>>>> REPLACE\n```"
        hypothesis = {"description": "Change the value.", "rationale": "A focused test.", "target_module": "model"}
        with patch("agent.coder.call_llm", return_value=LLMResponse(raw_response, 10, 3)) as call, patch(
            "agent.coder.resolve_model", return_value="test-coder-model"
        ):
            result = propose_patch("value = 1\n", hypothesis)
        self.assertEqual(result.diff, "<<<<<<< SEARCH\nvalue = 1\n=======\nvalue = 2\n>>>>>>> REPLACE")
        self.assertEqual((result.input_tokens, result.output_tokens), (10, 3))
        self.assertIn("value = 1", call.call_args.args[0])
        self.assertEqual(call.call_args.kwargs["temperature"], 0.15)
        self.assertIn("nothing outside the Search/Replace blocks", call.call_args.args[0])
        self.assertIn("This example demonstrates the required syntax only.", call.call_args.args[0])
        self.assertIn("def compute_score(x, y, weight=0.5):", call.call_args.args[0])
        self.assertIn("def compute_score(x, y, weight=0.7):", call.call_args.args[0])
        self.assertIn("Execution contract:", call.call_args.args[0])
        self.assertIn("Do not alter evaluate.py", call.call_args.args[0])
        self.assertIn("Preserve data.load() row order", call.call_args.args[0])
        self.assertIn("verify_row_order_matches_starter_kit()", call.call_args.args[0])
        self.assertIn("Preserve the baseline forward-pass structure", call.call_args.args[0])
        self.assertIn("The diff must implement the hypothesis's actual mechanism", call.call_args.args[0])
        self.assertIn("A patch that does not change the model's actual computation is invalid", call.call_args.args[0])
        self.assertIn("write that logic and wire it into the function that's actually called", call.call_args.args[0])
        self.assertNotIn("candidate_models:", call.call_args.args[0])

    def test_syntax_example_uses_no_baseline_function_or_parameter_names(self):
        from agent.coder import _system_prompt

        baseline_code = (Path(__file__).resolve().parents[1] / "baseline.py").read_text(encoding="utf-8")
        example = _system_prompt("").split("Hard constraints:", 1)[0]
        self.assertNotIn("run_fm", example)
        self.assertNotIn("lr=", example)
        self.assertNotIn("epochs=", example)
        self.assertNotIn("compute_score", baseline_code)
        self.assertNotIn("weight=", baseline_code)

    def test_rejects_malformed_hypothesis(self):
        with self.assertRaisesRegex(ValueError, "exactly"):
            propose_patch("", {"description": "x"})


if __name__ == "__main__":
    unittest.main()