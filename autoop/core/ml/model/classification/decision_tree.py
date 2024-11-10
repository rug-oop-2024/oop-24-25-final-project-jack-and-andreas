from sklearn.tree import DecisionTreeClassifier
from autoop.core.ml.model import Model


class DecisionTree(Model):
    """
    Decision Tree model for classification tasks.
    """
    def __init__(self):
        super().__init__("classification")
        self.hyperparameters["criterion"] = "gini"
        self.hyperparameters["splitter"] = "best"
        self.model = DecisionTreeClassifier()

    def fit(self, X, y):
        """
        Fit the model to the training data.
        """
        self.model.fit(X, y)
        self.parameters["tree"] = self.model.get_params()

    def predict(self, X):
        """
        Make predictions on the test data.
        """
        return self.model.predict(X)
