from autoop.core.ml.model import Model
from sklearn.linear_model import LinearRegression as SKLinearRegression
import pandas as pd


class MultipleLinearRegression(Model):
    def __init__(self) -> None:
        """
        Multiple Linear Regression model for regression tasks.
        """
        super().__init__("regression")
        self.hyperparameters["fit_intercept"] = True
        self.hyperparameters["normalize"] = False
        self.model = SKLinearRegression()

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Fit the model to the training data.
        """
        self.model.fit(X, y)
        self.parameters["coefficients"] = self.model.coef_

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on the test data.
        """
        return self.model.predict(X)
