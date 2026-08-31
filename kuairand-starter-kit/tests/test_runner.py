import tempfile
import unittest
from pathlib import Path

import numpy as np

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

    def test_checkpoint_path_saves_weights_when_candidate_supports_it(self):
        code = (
            "import numpy as np\n"
            "def run_fm(splits, return_predictions=False, checkpoint_path=None):\n"
            "    if checkpoint_path:\n"
            "        np.savez(checkpoint_path, V=np.ones((2, 2), dtype=np.float32))\n"
            "    result = {'valid': {'primary': 0.65}, 'test': {'GAUC': 0.7, 'nDCG@5': 0.6, 'primary': 0.65}}\n"
            "    if return_predictions:\n"
            "        result['test_scores'] = np.zeros(len(splits['test']))\n"
            "    return result\n"
        )
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "nested" / "checkpoint.npz"
            result = score_final_on_test(
                code, root / "KuaiRand-Pure" / "data", root / ".cache", timeout_s=30, checkpoint_path=checkpoint_path,
            )
            self.assertEqual(result["status"], "ok")
            self.assertTrue(result["checkpoint_saved"])
            self.assertTrue(checkpoint_path.exists())
            with np.load(checkpoint_path) as archive:
                self.assertTrue(np.array_equal(archive["V"], np.ones((2, 2), dtype=np.float32)))

    def test_checkpoint_path_is_best_effort_when_candidate_does_not_support_it(self):
        code = "import numpy as np\ndef run_fm(splits, return_predictions=False):\n    result = {'valid': {'primary': 0.65}, 'test': {'GAUC': 0.7, 'nDCG@5': 0.6, 'primary': 0.65}}\n    if return_predictions:\n        result['test_scores'] = np.zeros(len(splits['test']))\n    return result\n"
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "checkpoint.npz"
            result = score_final_on_test(
                code, root / "KuaiRand-Pure" / "data", root / ".cache", timeout_s=30, checkpoint_path=checkpoint_path,
            )
            self.assertEqual(result["status"], "ok")
            self.assertFalse(result["checkpoint_saved"])
            self.assertFalse(checkpoint_path.exists())


if __name__ == "__main__":
    unittest.main()