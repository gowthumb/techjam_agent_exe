import unittest

from agent.debugger import _system_prompt


class DebuggerPromptTest(unittest.TestCase):
    def test_prompt_preserves_row_order_and_baseline_mechanics(self):
        prompt = _system_prompt("def run_fm(splits): pass", "broken diff", "SyntaxError")
        self.assertIn("Preserve data.load() row order", prompt)
        self.assertIn("verify_row_order_matches_starter_kit()", prompt)
        self.assertIn("Preserve the baseline forward-pass structure", prompt)
        self.assertIn("Adam optimizer update", prompt)


if __name__ == "__main__":
    unittest.main()