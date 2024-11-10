import unittest
import numpy as np
from autoop.core.ml.metrics.mean_squared_error import MeanSquaredError


class TestMeanSquaredError(unittest.TestCase):
    """
    Tests for the MeanSquaredError metric.
    """
    def test_mean_squared_error(self):
        y_true = np.array([3, -0.5, 2, 7])
        y_pred = np.array([2.5, 0.0, 2, 8])
        mse = MeanSquaredError()
        self.assertEqual(mse(y_true, y_pred), 0.375)
