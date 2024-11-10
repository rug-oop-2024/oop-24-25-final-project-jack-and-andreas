
from autoop.core.ml.model.model import Model

from autoop.core.ml.model.classification import DecisionTree
from autoop.core.ml.model.classification import KNearestNeighbours
from autoop.core.ml.model.classification import SupportVectorMachine

from autoop.core.ml.model.regression import DecisionTreeRegressor
from autoop.core.ml.model.regression import MultipleLinearRegression
from autoop.core.ml.model.regression import SupportVectorRegressor

import pickle
from autoop.core.ml.artifact import Artifact

REGRESSION_MODELS = {
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "MultipleLinearRegression": MultipleLinearRegression,
    "SupportVectorRegressor": SupportVectorRegressor
}

CLASSIFICATION_MODELS = {
    "DecisionTree": DecisionTree,
    "KNearestNeighbours": KNearestNeighbours,
    "SupportVectorMachine": SupportVectorMachine
}


def get_model(model_name: str) -> Model:
    if model_name in REGRESSION_MODELS:
        return REGRESSION_MODELS[model_name]()
    elif model_name in CLASSIFICATION_MODELS:
        return CLASSIFICATION_MODELS[model_name]()
    else:
        raise ValueError(f"Model {model_name} not found")


def get_models() -> list[str]:
    return list(REGRESSION_MODELS.keys()) + list(CLASSIFICATION_MODELS.keys())


def get_regression_models() -> list[str]:
    return list(REGRESSION_MODELS.keys())


def get_classification_models() -> list[str]:
    return list(CLASSIFICATION_MODELS.keys())


def from_artifact(artifact: Artifact) -> "Model":
    """Load a model from an Artifact."""
    data = pickle.loads(artifact.data)
    model = get_model(data["class"])
    model.parameters = data["parameters"]
    model.hyperparameters = data["hyperparameters"]
    return model
