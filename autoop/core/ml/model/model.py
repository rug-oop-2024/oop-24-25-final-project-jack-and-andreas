
from abc import abstractmethod, ABC
from autoop.core.ml.artifact import Artifact
from typing import Any, Dict, Literal
import pickle
import pandas as pd


class Model(ABC):
    """Base class for all models."""

    def __init__(self, type: Literal["classification", "regression"]):
        """ Initalize the model"""
        self.parameters: Dict[str, Any] = {}
        self.hyperparameters: Dict[str, Any] = {}
        self.type = type

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Any:
        """Make predictions."""
        pass

    def to_artifact(self, name: str) -> Artifact:
        """Convert the model into an Artifact."""
        data = pickle.dumps({
            "parameters": self.parameters,
            "hyperparameters": self.hyperparameters,
            "class": self.__class__.__name__
        })
        return Artifact(
            name=name,
            asset_path=name,
            version="1.0",
            data=data,
            metadata={},
            type="model",
            tags=[]
        )
