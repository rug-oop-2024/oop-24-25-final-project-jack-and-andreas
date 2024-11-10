from sklearn.svm import SVR
from autoop.core.ml.model import Model


class SupportVectorRegressor(Model):
    """
    Support Vector Machine model for regression tasks.
    """
    def __init__(self):
        super().__init__("regression")
        self.hyperparameters["C"] = 1.0
        self.hyperparameters["degree"] = 3
        self.model = SVR()

    def fit(self, X, y):
        """
        Fit the model to the training data.
        """
        self.model.fit(X, y)
        self.parameters["support_vectors"] = self.model.support_

    def predict(self, X):
        """
        Make predictions on the test data.
        """
        return self.model.predict(X)
