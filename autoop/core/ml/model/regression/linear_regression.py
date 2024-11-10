from autoop.core.ml.model import Model
from sklearn.linear_model import LinearRegression as SKLinearRegression


class MultipleLinearRegression(Model):
    """
    Multiple Linear Regression model for regression tasks.
    """
    def __init__(self):
        super().__init__("regression")
        self.hyperparameters["fit_intercept"] = True
        self.hyperparameters["normalize"] = False
        self.model = SKLinearRegression()

    def fit(self, X, y):
        """
        Fit the model to the training data.
        """
        self.model.fit(X, y)
        self.parameters["coefficients"] = self.model.coef_

    def predict(self, X):
        """
        Make predictions on the test data.
        """
        return self.model.predict(X)
