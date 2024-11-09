from sklearn.tree import DecisionTreeRegressor
from autoop.core.ml.model import Model


class DecisionTreeRegressorModel(Model):
    def __init__(self):
        super().__init__("regression")
        self.hyperparameters["criterion"] = "mse"
        self.hyperparameters["splitter"] = "best"
        self.model = DecisionTreeRegressor()

    def fit(self, X, y):
        self.model.fit(X, y)
        self.parameters["tree"] = self.model.get_params()

    def predict(self, X):
        return self.model.predict(X)
