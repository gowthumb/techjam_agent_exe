import unittest
from pathlib import Path

from agent.debugger import _system_prompt


class DebuggerExampleTest(unittest.TestCase):
    def test_prompt_uses_generic_example_and_anti_copying_instruction(self):
        prompt = _system_prompt("", "", "")
        baseline_code = (Path(__file__).resolve().parents[1] / "baseline.py").read_text(encoding="utf-8")
        example = prompt.split("Hard constraints:", 1)[0]
        self.assertIn("This example demonstrates the required syntax only.", example)
        self.assertIn("def compute_score(x, y, weight=0.5):", example)
        self.assertNotIn("run_fm", example)
        self.assertNotIn("compute_score", baseline_code)
        self.assertNotIn("weight=", baseline_code)
        self.assertIn("The diff must implement the hypothesis's actual mechanism", prompt)
        self.assertIn("A patch that does not change the model's actual computation is invalid", prompt)
        self.assertIn("write that logic and wire it into the function that's actually called", prompt)


if __name__ == "__main__":
    unittest.main()