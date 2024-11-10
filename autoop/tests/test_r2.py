import unittest
from autoop.core.ml.metrics.r_2_score import R2Score
import numpy as np


class TestR2Score(unittest.TestCase):
    """
    Tests for the R2Score metric.
    """
    def test_r2_score(self):
        r2 = R2Score()
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        self.assertEqual(r2(y_true, y_pred), 1.0)
