import unittest
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from autoop.core.ml.model.classification.k_nearest_neighbours import (
    KNearestNeighbours,
)
from sklearn.neighbors import KNeighborsClassifier


class TestKNearestNeighbours(unittest.TestCase):
    def setUp(self):
        self.X_train, self.y_train = make_classification(
            n_samples=100, n_features=10, random_state=42
        )
        self.X_test, self.y_test = make_classification(
            n_samples=20, n_features=10, random_state=43
        )
        self.knn_model = KNearestNeighbours()

    def test_initialization(self):
        self.assertEqual(self.knn_model.type, "classification")
        self.assertEqual(self.knn_model.hyperparameters["n_neighbors"], 5)
        self.assertIsInstance(self.knn_model.model, KNeighborsClassifier)

    def test_fit(self):
        self.knn_model.fit(self.X_train, self.y_train)
        self.assertIn("knn", self.knn_model.parameters)
        self.assertEqual(
            len(self.knn_model.parameters["knn"]),
            len(self.knn_model.model.get_params())
        )

    def test_predict(self):
        self.knn_model.fit(self.X_train, self.y_train)
        predictions = self.knn_model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        accuracy = accuracy_score(self.y_test, predictions)
        self.assertGreaterEqual(accuracy, 0)


if __name__ == "__main__":
    unittest.main()
