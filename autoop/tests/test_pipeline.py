from sklearn.datasets import fetch_openml
import unittest
import pandas as pd

from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model.regression import MultipleLinearRegression
from autoop.core.ml.metrics import MeanSquaredError


class TestPipeline(unittest.TestCase):
    """
    Test the Pipeline class.
    """
    def setUp(self) -> None:
        """
        Set up the test case.
        """
        data = fetch_openml(name="adult", version=1, parser="auto")
        df = pd.DataFrame(
            data.data,
            columns=data.feature_names,
        )
        self.dataset = Dataset.from_dataframe(
            name="adult",
            asset_path="adult.csv",
            data=df,
        )
        self.features = detect_feature_types(self.dataset)
        self.pipeline = Pipeline(
            dataset=self.dataset,
            model=MultipleLinearRegression(),
            input_features=list(
                filter(lambda x: x.name != "age", self.features)
            ),
            target_feature=Feature(name="age", type="numerical"),
            metrics=[MeanSquaredError()],
            split=0.8
        )
        self.ds_size = data.data.shape[0]

    def test_init(self):
        """
        Test the initialization of the Pipeline class.
        """
        self.assertIsInstance(self.pipeline, Pipeline)

    def test_preprocess_features(self):
        """
        Test the preprocess_features method.
        """
        self.pipeline._preprocess_features()
        self.assertEqual(len(self.pipeline._artifacts), len(self.features))

    def test_split_data(self):
        """
        Test the split_data method.
        """
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        train_size = int(0.8 * self.ds_size)
        self.assertEqual(self.pipeline._train_X[0].shape[0], train_size)
        self.assertEqual(
            self.pipeline._test_X[0].shape[0],
            self.ds_size - int(0.8 * self.ds_size)
        )

    def test_train(self):
        """
        Test the train method.
        """
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.pipeline._train()
        self.assertIsNotNone(self.pipeline._model.parameters)

    def test_evaluate(self):
        """
        Test the evaluate method.
        """
        self.pipeline._preprocess_features()
        self.pipeline._split_data()
        self.pipeline._train()
        self.pipeline._evaluate()
        self.assertIsNotNone(self.pipeline._predictions)
        self.assertIsNotNone(self.pipeline._metrics_results)
        self.assertEqual(len(self.pipeline._metrics_results), 1)
