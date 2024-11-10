from abc import ABC, abstractmethod
import numpy as np


class Metric(ABC):
    """
    Base class for all metrics.
    """
    @abstractmethod
    def evaluate(self,
                 ground_truth: np.ndarray,
                 predictions: np.ndarray) -> float:
        """
        Evaluate the metric.
        """
        pass

    def __call__(self,
                 ground_truth: np.ndarray,
                 predictions: np.ndarray) -> float:
        """
        Evaluate the metric by comparing ground truth values with predictions.
        """
        return self.evaluate(ground_truth, predictions)
