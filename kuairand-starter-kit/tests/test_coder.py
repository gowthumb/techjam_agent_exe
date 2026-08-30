import unittest
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

    def test_rejects_malformed_hypothesis(self):
        with self.assertRaisesRegex(ValueError, "exactly"):
            propose_patch("", {"description": "x"})


if __name__ == "__main__":
    unittest.main()