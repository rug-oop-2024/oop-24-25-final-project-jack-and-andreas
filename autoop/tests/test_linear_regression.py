import unittest
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from autoop.core.ml.model.regression.linear_regression import (
    MultipleLinearRegression,
)
from sklearn.linear_model import LinearRegression


class TestMultipleLinearRegression(unittest.TestCase):
    def setUp(self):
        # Generate synthetic regression data for testing
        self.X_train, self.y_train = make_regression(
            n_samples=100, n_features=5, noise=0.1, random_state=42
        )
        self.X_test, self.y_test = make_regression(
            n_samples=20, n_features=5, noise=0.1, random_state=43
        )

        # Initialize the MultipleLinearRegression model
        self.model = MultipleLinearRegression()

    def test_initialization(self):
        # Test initialization
        self.assertEqual(self.model.type, "regression")
        self.assertEqual(self.model.hyperparameters["fit_intercept"], True)
        self.assertEqual(self.model.hyperparameters["normalize"], False)
        self.assertIsInstance(self.model.model, LinearRegression)

    def test_fit(self):
        # Test the fit method
        self.model.fit(self.X_train, self.y_train)
        self.assertIn("coefficients", self.model.parameters)
        self.assertEqual(
            len(self.model.parameters["coefficients"]), self.X_train.shape[1]
        )

    def test_predict(self):
        # Test the predict method
        self.model.fit(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)

        # Ensure predictions have the correct shape
        self.assertEqual(len(predictions), len(self.y_test))

        # Check mean squared error (basic validation)
        mse = mean_squared_error(self.y_test, predictions)
        self.assertGreaterEqual(mse, 0)


if __name__ == "__main__":
    unittest.main()
