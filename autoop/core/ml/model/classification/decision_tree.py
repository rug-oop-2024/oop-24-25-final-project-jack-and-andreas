from sklearn.tree import DecisionTreeClassifier
from autoop.core.ml.model import Model

from typing import Any


class DecisionTree(Model):
    def __init__(self, hyperparameters: dict[str, Any] = {}):
        super().__init__()
        self.hyperparameters["criterion"] = hyperparameters.get(
            "criterion", "gini"
        )
        self.hyperparameters["splitter"] = hyperparameters.get(
            "splitter", "best"
        )
        self.model = DecisionTreeClassifier()

    def fit(self, X, y):
        self.model.fit(X, y)
        self.parameters["tree"] = self.model.get_params()

    def predict(self, X):
        return self.model.predict(X)
