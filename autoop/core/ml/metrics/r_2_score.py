from autoop.core.ml.metrics.metric import Metric
import numpy as np


class R2Score(Metric):
    def evaluate(
        self, ground_truth: np.ndarray, predictions: np.ndarray
    ) -> float:
        """Calculate R² score for regression tasks."""
        ss_total = np.sum((ground_truth - np.mean(ground_truth)) ** 2)
        ss_residual = np.sum((ground_truth - predictions) ** 2)
        return 1 - (ss_residual / ss_total)
