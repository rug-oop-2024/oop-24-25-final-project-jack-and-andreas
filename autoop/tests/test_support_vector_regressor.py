import unittest
from sklearn.datasets import make_regression
from autoop.core.ml.model.regression.support_vector_regressor import (
    SupportVectorRegressor,
)
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR


class TestSupportVectorRegressor(unittest.TestCase):
    """
    Tests for the SupportVectorRegressor model.
    """
    def setUp(self):
        """
        Create the training and testing data.
        """
        self.X_train, self.y_train = make_regression(
            n_samples=100, n_features=10, random_state=42
        )
        self.X_test, self.y_test = make_regression(
            n_samples=20, n_features=10, random_state=43
        )
        self.regressor = SupportVectorRegressor()

    def test_initialization(self):
        """
        Test the initialization of the SupportVectorRegressor model.
        """
        self.assertEqual(self.regressor.type, "regression")
        self.assertEqual(self.regressor.hyperparameters["C"], 1.0)
        self.assertEqual(self.regressor.hyperparameters["degree"], 3)
        self.assertIsInstance(self.regressor.model, SVR)

    def test_fit(self):
        """
        Test the fit method of the SupportVectorRegressor model.
        """
        self.regressor.fit(self.X_train, self.y_train)
        self.assertIn("support_vectors", self.regressor.parameters)
        self.assertGreater(
            len(self.regressor.parameters["support_vectors"]), 0
        )

    def test_predict(self):
        """
        Test the predict method of the SupportVectorRegressor model.
        """
        self.regressor.fit(self.X_train, self.y_train)
        predictions = self.regressor.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        mse = mean_squared_error(self.y_test, predictions)
        self.assertGreaterEqual(mse, 0)


if __name__ == "__main__":
    unittest.main()
