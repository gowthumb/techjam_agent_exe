import unittest
from pathlib import Path

from agent.runner import run, run_onek_paired, score_final_on_test


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

    def test_onek_candidate_bypasses_pure_data_cache(self):
        code = "DATASET = '1k'\ndef run_fm(bench, return_predictions=False):\n    assert bench == '1k'\n    return {'valid': {'GAUC': 0.7, 'nDCG@5': 0.6, 'primary': 0.65}, 'test': {'primary': 0.65}}\n"
        root = Path(__file__).resolve().parents[1]
        result = run(code, root / "missing-1k-data", root / ".cache", timeout_s=30)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(set(result["metrics"]), {"valid"})

    def test_onek_paired_runner_uses_matched_seeds_and_half_epochs(self):
        code = "DATASET = '1k'\ndef run_fm(bench, seed=0, epochs=40, **kwargs):\n    assert bench == '1k'\n    assert epochs == 20\n    return {'valid': {'GAUC': 0.7 + seed, 'nDCG@5': 0.6 + seed, 'primary': 0.65 + seed}}\n"
        root = Path(__file__).resolve().parents[1]
        result = run_onek_paired(code, root / "missing-1k-data", root / ".cache", timeout_s=30)
        self.assertEqual(result["status"], "ok")
        self.assertEqual([metrics["primary"] for metrics in result["per_seed_valid"]], [0.65, 1.65, 2.65])
        self.assertAlmostEqual(result["metrics"]["valid"]["primary"], 1.65)


if __name__ == "__main__":
    unittest.main()