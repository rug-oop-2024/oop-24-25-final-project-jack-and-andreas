
from abc import abstractmethod, ABC
from autoop.core.ml.artifact import Artifact
from typing import Any, Dict, Literal
import pickle


class Model(ABC):
    """Base class for all models."""

    def __init__(self, type: Literal["classification", "regression"]):
        self.parameters: Dict[str, Any] = {}
        self.hyperparameters: Dict[str, Any] = {}
        self.type = type

    @abstractmethod
    def fit(self, X: Any, y: Any):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X: Any) -> Any:
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

    @staticmethod
    def from_artifact(artifact: Artifact) -> "Model":
        pass
