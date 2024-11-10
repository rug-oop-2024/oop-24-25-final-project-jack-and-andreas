from sklearn.tree import DecisionTreeClassifier
from autoop.core.ml.model import Model
import pandas as pd


class DecisionTree(Model):
    """
    Decision Tree model for classification tasks.
    """
    def __init__(self) -> None:
        """ Initalize the decision tree """
        super().__init__("classification")
        self.hyperparameters["criterion"] = "gini"
        self.hyperparameters["splitter"] = "best"
        self.model = DecisionTreeClassifier()

    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """
        Fit the model to the training data.
        """
        self.model.fit(X, y)
        self.parameters["tree"] = self.model.get_params()

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on the test data.
        """
        return self.model.predict(X)
