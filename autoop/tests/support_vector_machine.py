import unittest
# import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
# from autoop.core.ml.model import Model
from autoop.core.ml.model.classification.support_vector_machine import (
    SupportVectorMachine,
)
from sklearn.svm import SVC


class TestSupportVectorMachine(unittest.TestCase):
    def setUp(self):
        self.X_train, self.y_train = make_classification(
            n_samples=100, n_features=10, random_state=42
        )
        self.X_test, self.y_test = make_classification(
            n_samples=20, n_features=10, random_state=43
        )
        self.svm_model = SupportVectorMachine()

    def test_initialization(self):
        self.assertEqual(self.svm_model.task_type, "classification")
        self.assertEqual(self.svm_model.hyperparameters["C"], 1.0)
        self.assertEqual(self.svm_model.hyperparameters["degree"], 3)
        self.assertIsInstance(self.svm_model.model, SVC)

    def test_fit(self):
        self.svm_model.fit(self.X_train, self.y_train)
        self.assertIn("support_vectors", self.svm_model.parameters)
        support_vectors = self.svm_model.parameters["support_vectors"]
        self.assertGreater(len(support_vectors), 0)

    def test_predict(self):
        self.svm_model.fit(self.X_train, self.y_train)
        predictions = self.svm_model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        accuracy = accuracy_score(self.y_test, predictions)
        self.assertGreaterEqual(accuracy, 0)


if __name__ == "__main__":
    unittest.main()
