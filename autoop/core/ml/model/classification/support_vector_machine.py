from sklearn.svm import SVC
from autoop.core.ml.model import Model


class SupportVectorMachine(Model):
    def __init__(self):
        super().__init__("classification")
        self.hyperparameters["C"] = 1.0
        self.hyperparameters["degree"] = 3
        self.model = SVC()

    def fit(self, X, y):
        self.model.fit(X, y)
        self.parameters["support_vectors"] = self.model.support_

    def predict(self, X):
        return self.model.predict(X)
