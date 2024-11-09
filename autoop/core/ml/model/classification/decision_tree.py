from sklearn.tree import DecisionTreeClassifier
from autoop.core.ml.model import Model


class DecisionTree(Model):
    def __init__(self):
        super().__init__("classification")
        self.hyperparameters["criterion"] = "gini"
        self.hyperparameters["splitter"] = "best"
        self.model = DecisionTreeClassifier()

    def fit(self, X, y):
        self.model.fit(X, y)
        self.parameters["tree"] = self.model.get_params()

    def predict(self, X):
        return self.model.predict(X)
