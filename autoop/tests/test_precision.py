import unittest
import numpy as np
from autoop.core.ml.metrics.precision import Precision


class TestPrecision(unittest.TestCase):
    def test_precision(self):
        y_true = np.array([0, 1, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0, 1, 1])
        precision = Precision()

        self.assertEqual(precision(y_true, y_pred), 1.0)
