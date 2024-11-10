
import unittest

from autoop.tests.test_database import TestDatabase  # no qa
from autoop.tests.test_storage import TestStorage  # no qa
from autoop.tests.test_features import TestFeatures  # no qa
from autoop.tests.test_pipeline import TestPipeline  # no qa

from autoop.tests.test_accuracy import TestAccuracy
from autoop.tests.test_recall import TestRecall
from autoop.tests.test_r2 import TestR2Score
from autoop.tests.test_precision import TestPrecision
from autoop.tests.test_mean_absolute_error import TestMeanAbsoluteError
from autoop.tests.test_mean_squared_error import TestMeanSquaredError

from autoop.tests.test_decision_tree import DecisionTree
from autoop.tests.test_k_nearest_neighbours import TestKNearestNeighbours
from autoop.tests.test_linear_regression import TestMultipleLinearRegression
from autoop.tests.test_support_vector_regressor import (
    TestSupportVectorRegressor
)
from autoop.tests.support_vector_machine import SupportVectorMachine
from autoop.tests.test_decision_tree_regressor import DecisionTreeRegressor


if __name__ == '__main__':
    unittest.main()
