from sklearn.svm import SVC
from autoop.core.ml.model import Model
import pandas as pd


class SupportVectorMachine(Model):
    """
    Support Vector Machine model for classification tasks.
    """
    def __init__(self) -> None:
        """ Initalize the support vector machine """
        super().__init__("classification")
        self.hyperparameters["C"] = 1.0
        self.hyperparameters["degree"] = 3
        self.model = SVC()

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Fit the model to the training data.
        """
        self.model.fit(X, y)
        self.parameters["support_vectors"] = self.model.support_

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """ Predict with the model"""
        return self.model.predict(X)
