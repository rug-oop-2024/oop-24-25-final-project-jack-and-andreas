from autoop.core.ml.metrics.metric import Metric
import numpy as np


class MeanSquaredError(Metric):
    """
    A class used to represent the Mean Squared Error metric.
    """
    def evaluate(
        self, ground_truth: np.ndarray, predictions: np.ndarray
    ) -> float:
        """Calculates mean squared error for regression tasks."""
        return np.mean((ground_truth - predictions) ** 2)
