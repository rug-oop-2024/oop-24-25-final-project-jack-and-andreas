from autoop.core.ml.metrics.metric import Metric
import numpy as np


class Accuracy(Metric):
    def evaluate(
        self, ground_truth: np.ndarray, predictions: np.ndarray
    ) -> float:
        """Calculate accuracy for classification tasks."""
        return np.mean(ground_truth == predictions)
