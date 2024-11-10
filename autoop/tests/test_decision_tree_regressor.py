import unittest
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from autoop.core.ml.model.regression.decision_tree_regressor import (
    DecisionTreeRegressor,
)

"""
Class to test the DecisionTreeRegressor model
This class tests the initialization, and basic methods of the Decision Tree regressor class.
"""
class TestDecisionTreeRegressorModel(unittest.TestCase):
    """Setup"""
    def setUp(self):
        self.X_train, self.y_train = make_regression(
            n_samples=100, n_features=10, random_state=42
        )
        self.X_test, self.y_test = make_regression(
            n_samples=20, n_features=10, random_state=43
        )
        self.regressor = DecisionTreeRegressor()
    """ Check for succesful initialization """
    def test_initialization(self):
        self.assertEqual(self.regressor.task_type, "regression")
        self.assertEqual(self.regressor.hyperparameters["criterion"], "mse")
        self.assertEqual(self.regressor.hyperparameters["splitter"], "best")
        self.assertIsInstance(self.regressor.model, DecisionTreeRegressor)
    """ Test fit by checking if the tree gets built when fit is called """
    def test_fit(self):
        self.regressor.fit(self.X_train, self.y_train)
        self.assertIn("tree", self.regressor.parameters)
        self.assertGreater(len(self.regressor.parameters["tree"]), 0)
    """ Test predict by fitting and then trying to predict """
    def test_predict(self):
        self.regressor.fit(self.X_train, self.y_train)
        predictions = self.regressor.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        mse = mean_squared_error(self.y_test, predictions)
        self.assertGreaterEqual(mse, 0)


if __name__ == "__main__":
    unittest.main()
