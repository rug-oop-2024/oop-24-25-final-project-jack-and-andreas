from autoop.core.ml.model import Model
from sklearn.neighbors import KNeighborsClassifier


class KNearestNeighbours(Model):
    def __init__(self):
        super().__init__("classification")
        self.hyperparameters["n_neighbors"] = 5

        self.model = KNeighborsClassifier(
            n_neighbors=self.hyperparameters["n_neighbors"]
        )
        self.parameters = {}

    def fit(self, X, y):
        self.model.fit(X, y)
        self.parameters["knn"] = self.model.get_params(True)

    def predict(self, X):
        return self.model.predict(X)
