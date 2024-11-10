from typing import List
import pickle

from autoop.core.ml.artifact import Artifact
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.model import Model
from autoop.core.ml.feature import Feature
from autoop.core.ml.metrics import Metric
from autoop.functional.preprocessing import preprocess_features
import numpy as np


class Pipeline():
    """
    A class used to represent a Machine Learning Pipeline.
    """
    def __init__(self,

                 metrics: List[Metric],
                 dataset: Dataset,
                 model: Model,
                 input_features: List[Feature],
                 target_feature: Feature,
                 split: float = 0.8,
                 ) -> None:
        """
        Initializes the Pipeline with the given parameters.
        """
        self._dataset = dataset
        self._model = model
        self._input_features = input_features
        self._target_feature = target_feature
        self._metrics = metrics
        self._artifacts = {}
        self._split = split
        if (
            target_feature.type == "categorical"
            and model.type != "classification"
        ):
            raise ValueError(
                "Model type must be classification for "
                "categorical target feature"
            )
        if target_feature.type == "continuous" and model.type != "regression":
            raise ValueError(
                "Model type must be regression for continuous target feature"
            )

    def __str__(self) -> str:
        """
        Return a string representation of the Pipeline object.
        """
        return f"""
Pipeline(
    model={self._model.type},
    input_features={list(map(str, self._input_features))},
    target_feature={str(self._target_feature)},
    split={self._split},
    metrics={list(map(str, self._metrics))},
)
"""

    @property
    def model(self) -> Model:
        """
        Returns the model instance.
        """
        return self._model

    def artifacts(self, pipe_name: str, pipe_version: str) -> List[Artifact]:
        """
        Used to get the artifacts generated during the pipeline
        execution to be saved
        """
        artifacts = []
        for name, artifact in self._artifacts.items():
            artifact_type = artifact.get("type")
            if artifact_type in ["OneHotEncoder"]:
                data = artifact["encoder"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(
                    name=f"{name}_{pipe_name}_{pipe_version}",
                    data=data,
                    asset_path=(
                        f"{pipe_name}/{name}_{pipe_name}_{pipe_version}"
                    ),
                    version=pipe_version,
                    type="encoder"
                ))
            if artifact_type in ["StandardScaler"]:
                data = artifact["scaler"]
                data = pickle.dumps(data)
                artifacts.append(Artifact(
                    name=f"{name}_{pipe_name}_{pipe_version}",
                    data=data,
                    asset_path=(
                        f"{pipe_name}/{name}_{pipe_name}_{pipe_version}"
                    ),
                    version=pipe_version,
                    type="scaler"
                ))
        model_artifact = self._model.to_artifact(
            name=f"pipeline_model_{self._model.type}"
        )
        artifacts.append(model_artifact)

        pipeline_data = {
            "input_features": self._input_features,
            "target_feature": self._target_feature,
            "split": self._split,
            "metrics": self._metrics,
            "model": model_artifact.id,
        }

        artifacts.append(Artifact(
            name=f"pipeline_config_{pipe_name}_{pipe_version}",
            data=pickle.dumps(pipeline_data),
            type="pipeline_config",
            asset_path=f"{pipe_name}/pipeline_config",
            version=pipe_version
        ))
        return artifacts

    def _register_artifact(self, name: str, artifact: dict) -> None:
        """ Register an artifact generated during the pipeline execution """
        self._artifacts[name] = artifact

    def _preprocess_features(self) -> None:
        """ Preprocess the features """
        (target_feature_name, target_data, artifact) = preprocess_features(
            [self._target_feature], self._dataset
        )[0]
        self._register_artifact(target_feature_name, artifact)
        input_results = preprocess_features(
            self._input_features, self._dataset
        )
        for (feature_name, data, artifact) in input_results:
            self._register_artifact(feature_name, artifact)
        # Get the input vectors and output vector,
        # sort by feature name for consistency
        self._output_vector = target_data
        self._input_vectors = [
            data for (feature_name, data, artifact) in input_results
        ]

    def _split_data(self) -> None:
        """ Split the data into training and testing sets """
        split = self._split
        self._train_X = [
            vector[:int(split * len(vector))]
            for vector in self._input_vectors
        ]
        self._test_X = [
            vector[int(split * len(vector)):]
            for vector in self._input_vectors
        ]
        self._train_y = self._output_vector[
            :int(split * len(self._output_vector))
        ]
        self._test_y = self._output_vector[
            int(split * len(self._output_vector)):]

    def _compact_vectors(self, vectors: List[np.array]) -> np.array:
        """ Compact the input vectors into a single matrix """
        return np.concatenate(vectors, axis=1)

    def _train(self) -> None:
        """ Train the model """
        X = self._compact_vectors(self._train_X)
        Y = self._train_y
        self._model.fit(X, Y)

    def _evaluate(self) -> None:
        """ Evaluate the model """
        X = self._compact_vectors(self._test_X)
        Y = self._test_y
        self._metrics_results = []
        predictions = self._model.predict(X)
        for metric in self._metrics:
            result = metric.evaluate(predictions, Y)
            self._metrics_results.append((metric, result))
        self._predictions = predictions

    def execute(self) -> dict:
        """
        Execute the pipeline and return metrics for both training
        and testing sets.
        """
        self._preprocess_features()
        self._split_data()
        self._train()
        self._evaluate()

        return {
            "metrics": self._metrics_results,
            "predictions": self._predictions,
        }
