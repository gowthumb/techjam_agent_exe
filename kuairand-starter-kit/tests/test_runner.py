import unittest
from pathlib import Path

from agent.runner import run, score_final_on_test


class RunnerSafetyTest(unittest.TestCase):
    def test_iteration_result_exposes_only_validation_metrics(self):
        code = "def run_fm(splits):\n    return {'valid': {'GAUC': 0.7, 'nDCG@5': 0.6, 'primary': 0.65}, 'test': {'GAUC': 0.99, 'nDCG@5': 0.99, 'primary': 0.99}}\n"
        root = Path(__file__).resolve().parents[1]
        result = run(code, root / "KuaiRand-Pure" / "data", root / ".cache", timeout_s=30)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(set(result["metrics"]), {"valid"})
        self.assertNotIn("test", result["metrics"])

    def test_final_scoring_returns_test_scores_only_at_final_boundary(self):
        code = "import numpy as np\ndef run_fm(splits, return_predictions=False):\n    result = {'valid': {'primary': 0.65}, 'test': {'GAUC': 0.7, 'nDCG@5': 0.6, 'primary': 0.65}}\n    if return_predictions:\n        result['test_scores'] = np.zeros(len(splits['test']))\n    return result\n"
        root = Path(__file__).resolve().parents[1]
        result = score_final_on_test(code, root / "KuaiRand-Pure" / "data", root / ".cache", timeout_s=30)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(set(result["metrics"]), {"test"})
        self.assertEqual(len(result["test_scores"]), 170588)


if __name__ == "__main__":
    unittest.main()