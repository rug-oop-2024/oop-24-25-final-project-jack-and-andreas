import unittest
from autoop.core.ml.metrics.mean_absolute_error import MeanAbsoluteError
import numpy as np


class TestMeanAbsoluteError(unittest.TestCase):
    """
    Tests for the MeanAbsoluteError metric.
    """
    def test_mean_absolute_error(self):
        y_true = np.array([3, -0.5, 2, 7])
        y_pred = np.array([2.5, 0.0, 2, 8])
        metric = MeanAbsoluteError()
        self.assertEqual(metric(y_true, y_pred), 0.5)
