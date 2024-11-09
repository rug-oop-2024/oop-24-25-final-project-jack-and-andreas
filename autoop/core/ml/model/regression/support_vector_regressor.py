from sklearn.svm import SVR
from autoop.core.ml.model import Model


class SupportVectorRegressor(Model):
    def __init__(self):
        super().__init__("regression")
        self.hyperparameters["C"] = 1.0
        self.hyperparameters["degree"] = 3
        self.model = SVR()

    def fit(self, X, y):
        self.model.fit(X, y)
        self.parameters["support_vectors"] = self.model.support_

    def predict(self, X):
        return self.model.predict(X)
