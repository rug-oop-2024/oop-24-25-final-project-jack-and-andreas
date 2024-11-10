from autoop.core.ml.metrics.metric import Metric
import numpy as np


class Precision(Metric):
    """
    A precision metric for classification tasks
    """
    def evaluate(
        self, ground_truth: np.ndarray, predictions: np.ndarray
    ) -> float:
        """Calculates precision """
        true_positives = np.sum((predictions == 1) & (ground_truth == 1))
        predicted_positives = np.sum(predictions == 1)
        return (
            true_positives / predicted_positives
            if predicted_positives > 0
            else 0.0
        )
