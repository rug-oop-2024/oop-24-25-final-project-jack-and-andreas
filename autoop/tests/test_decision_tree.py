import unittest
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from autoop.core.ml.model.classification.decision_tree import DecisionTree
from autoop.core.ml.model.classification.decision_tree import (
    DecisionTreeClassifier
)

"""
Class to test the Decision Tree class 
This class contain tests for decision tree initialization 
as well as tests for the fit and predict methods
"""
class TestDecisionTree(unittest.TestCase):
    """Setup"""
    def setUp(self):
        self.X_train, self.y_train = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42
        )
        self.X_test, self.y_test = make_classification(
            n_samples=20,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=43
        )
        self.decision_tree = DecisionTree()
    """ Check for succesful initialization """
    def test_initialization(self):
        self.assertEqual(self.decision_tree.task_type, "classification")
        self.assertEqual(
            self.decision_tree.hyperparameters["criterion"], "gini"
        )
        self.assertEqual(
            self.decision_tree.hyperparameters["splitter"], "best"
        )
        self.assertIsInstance(self.decision_tree.model, DecisionTreeClassifier)
    """ Test fit by checking if the tree gets built when fit is called """
    def test_fit(self):
        self.decision_tree.fit(self.X_train, self.y_train)
        self.assertIn("tree", self.decision_tree.parameters)
        self.assertGreater(len(self.decision_tree.parameters["tree"]), 0)
    """ Test predict by fitting and then trying to predict """
    def test_predict(self):
        self.decision_tree.fit(self.X_train, self.y_train)
        predictions = self.decision_tree.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        accuracy = accuracy_score(self.y_test, predictions)
        self.assertGreaterEqual(accuracy, 0)


if __name__ == "__main__":
    unittest.main()
