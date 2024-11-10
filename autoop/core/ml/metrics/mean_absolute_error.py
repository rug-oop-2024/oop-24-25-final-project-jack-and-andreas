from autoop.core.ml.metrics.metric import Metric
import numpy as np


class MeanAbsoluteError(Metric):
    def evaluate(
        self, ground_truth: np.ndarray, predictions: np.ndarray
    ) -> float:
        """Calculates mean absolute error for regression tasks."""
        return np.mean(np.abs(ground_truth - predictions))
