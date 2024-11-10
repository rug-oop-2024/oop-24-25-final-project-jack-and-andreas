from sklearn.tree import DecisionTreeRegressor as SKDecisionTreeRegressor
from autoop.core.ml.model import Model
import pandas as pd


class DecisionTreeRegressor(Model):
    """
    Decision Tree model for regression tasks.
    """
    def __init__(self) -> None:
        """
        Initalize the decision tree
        """
        super().__init__("regression")
        self.hyperparameters["criterion"] = "mse"
        self.hyperparameters["splitter"] = "best"
        self.model = SKDecisionTreeRegressor()

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Fit the model to the training data.
        """
        self.model = self.model.fit(X, y)
        self.parameters["tree"] = self.model.get_params()

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on the test data.
        """
        return self.model.predict(X)
