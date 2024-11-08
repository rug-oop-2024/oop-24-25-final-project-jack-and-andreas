from sklearn.tree import DecisionTreeRegressor
from autoop.core.ml.model import Model


class DecisionTreeRegressorModel(Model):
    def __init__(self, hyperparameters={}):
        super().__init__()
        self.hyperparameters["criterion"] = hyperparameters.get(
            "criterion", "mse"
        )
        self.hyperparameters["splitter"] = hyperparameters.get(
            "splitter", "best"
        )
        self.model = DecisionTreeRegressor()

    def fit(self, X, y):
        self.model.fit(X, y)
        self.parameters["tree"] = self.model.get_params()

    def predict(self, X):
        return self.model.predict(X)
