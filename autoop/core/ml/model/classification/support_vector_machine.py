from sklearn.svm import SVC
from autoop.core.ml.model import Model


class SupportVectorMachine(Model):
    def __init__(self, hyperparameters={}):
        super().__init__()
        self.hyperparameters["C"] = hyperparameters.get("C", 1.0)
        self.hyperparameters["degree"] = hyperparameters.get("degree", 3)
        self.model = SVC()

    def fit(self, X, y):
        self.model.fit(X, y)
        self.parameters["support_vectors"] = self.model.support_

    def predict(self, X):
        return self.model.predict(X)
