from sklearn.tree import DecisionTreeRegressor as SKDecisionTreeRegressor
from autoop.core.ml.model import Model


class DecisionTreeRegressor(Model):
    """ 
    Decision Tree model for regression tasks.
    """
    def __init__(self):
        super().__init__("regression")
        self.hyperparameters["criterion"] = "mse"
        self.hyperparameters["splitter"] = "best"
        self.model = SKDecisionTreeRegressor()

    def fit(self, X, y):
        """
        Fit the model to the training data.
        """
        self.model = self.model.fit(X, y)
        self.parameters["tree"] = self.model.get_params()

    def predict(self, X):
        """
        Make predictions on the test data.
        """
        return self.model.predict(X)
