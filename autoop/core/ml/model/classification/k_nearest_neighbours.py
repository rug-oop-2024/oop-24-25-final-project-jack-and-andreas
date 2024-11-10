from autoop.core.ml.model import Model
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd


class KNearestNeighbours(Model):
    """
    K-Nearest Neighbours model for classification tasks.
    """
    def __init__(self) -> None:
        """ Initializes KNN with 5 neighbors"""
        super().__init__("classification")
        self.hyperparameters["n_neighbors"] = 5

        self.model = KNeighborsClassifier(
            n_neighbors=self.hyperparameters["n_neighbors"]
        )
        self.parameters = {}

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Fit the model to the training data.
        """
        self.model.fit(X, y)
        self.parameters["knn"] = self.model.get_params(True)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on the test data.
        """
        return self.model.predict(X)
