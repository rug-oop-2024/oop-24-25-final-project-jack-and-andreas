import unittest
from autoop.core.ml.metrics.recall import Recall
import numpy as np


class TestRecall(unittest.TestCase):
    """
    Tests for the Recall metric.
    """
    def test_recall(self):
        recall = Recall()
        y_true = np.array([0, 1, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        result = recall(y_true, y_pred)
        self.assertEqual(result, 0.75)
