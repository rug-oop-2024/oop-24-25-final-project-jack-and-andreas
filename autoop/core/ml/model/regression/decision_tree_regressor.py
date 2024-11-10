from sklearn.tree import DecisionTreeRegressor as SKDecisionTreeRegressor
from autoop.core.ml.model import Model


class DecisionTreeRegressor(Model):
    def __init__(self):
        super().__init__("regression")
        self.hyperparameters["criterion"] = "mse"
        self.hyperparameters["splitter"] = "best"
        self.model = SKDecisionTreeRegressor()

    def fit(self, X, y):
        self.model = self.model.fit(X, y)
        self.parameters["tree"] = self.model.get_params()

    def predict(self, X):
        return self.model.predict(X)
