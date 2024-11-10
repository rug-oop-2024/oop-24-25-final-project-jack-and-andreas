import unittest
import numpy as np
from autoop.core.ml.metrics.accuracy import Accuracy


class TestAccuracy(unittest.TestCase):
    def test_accuracy(self):
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4])
        accuracy = Accuracy()
        self.assertEqual(accuracy(y_true, y_pred), 1.0)

        y_pred = np.array([1, 1, 2, 3, 4])
        self.assertEqual(accuracy(y_true, y_pred), 0.8)
