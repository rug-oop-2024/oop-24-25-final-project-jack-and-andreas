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
        pass

    def __call__(self,
                 ground_truth: np.ndarray,
                 predictions: np.ndarray) -> float:
        return self.evaluate(ground_truth, predictions)
