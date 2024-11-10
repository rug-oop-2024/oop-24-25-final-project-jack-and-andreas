from autoop.core.ml.metrics.metric import Metric
import numpy as np


class Recall(Metric):
    """
    A class used to calculate the recall metric for classification tasks.
    """
    def evaluate(
        self, ground_truth: np.ndarray, predictions: np.ndarray
    ) -> float:
        """Calculates recall for classification tasks."""
        true_positives = np.sum((predictions == 1) & (ground_truth == 1))
        actual_positives = np.sum(ground_truth == 1)
        return (
            true_positives / actual_positives
            if actual_positives > 0
            else 0.0
        )
