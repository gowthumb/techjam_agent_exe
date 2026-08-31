import unittest
from unittest.mock import patch

import numpy as np

import baseline_1k


class Baseline1KTest(unittest.TestCase):
    def test_adapter_returns_baseline_compatible_metrics_and_predictions(self):
        enc = {
            "train": (np.array([[0]], dtype=np.int32), np.array([1], dtype=np.float32), np.array([1])),
            "valid": (np.array([[0]], dtype=np.int32), np.array([1], dtype=np.float32), np.array([2])),
            "test": (np.array([[0]], dtype=np.int32), np.array([0], dtype=np.float32), np.array([3])),
        }

        class Model:
            def predict(self, features):
                return np.full(len(features), 0.5)

        with patch.object(baseline_1k, "prepare", return_value=(None, None, enc, 1)), patch.object(
            baseline_1k.fm, "train", return_value=(Model(), {})
        ), patch.object(baseline_1k.harness, "score", side_effect=[{"primary": 0.7}, {"primary": 0.6}]):
            result = baseline_1k.run_fm_1k(return_predictions=True)
        self.assertEqual(result["valid"]["primary"], 0.7)
        self.assertEqual(result["test"]["primary"], 0.6)
        self.assertEqual(result["test_scores"].tolist(), [0.5])