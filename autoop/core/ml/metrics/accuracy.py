from autoop.core.ml.metrics.metric import Metric
import numpy as np


class Accuracy(Metric):
    """
    A class used to calculate the accuracy of classification tasks.
    """
    def evaluate(
        self, ground_truth: np.ndarray, predictions: np.ndarray
    ) -> float:
        """Calculates accuracy for classification tasks."""
        return np.mean(ground_truth == predictions)
