
import unittest

from autoop.tests.test_database import TestDatabase  # noqa
from autoop.tests.test_storage import TestStorage  # noqa
from autoop.tests.test_features import TestFeatures  # noqa
from autoop.tests.test_pipeline import TestPipeline  # noqa

from autoop.tests.test_accuracy import TestAccuracy  # noqa
from autoop.tests.test_recall import TestRecall  # noqa
from autoop.tests.test_r2 import TestR2Score  # noqa
from autoop.tests.test_precision import TestPrecision  # noqa
from autoop.tests.test_mean_absolute_error import (  # noqa
    TestMeanAbsoluteError
)
from autoop.tests.test_mean_squared_error import TestMeanSquaredError  # noqa

from autoop.tests.test_decision_tree import DecisionTree  # noqa
from autoop.tests.test_k_nearest_neighbours import (  # noqa
    TestKNearestNeighbours
)
from autoop.tests.test_linear_regression import TestMultipleLinearRegression  # noqa
from autoop.tests.test_support_vector_regressor import (  # noqa
    TestSupportVectorRegressor
)
from autoop.tests.support_vector_machine import SupportVectorMachine  # noqa
from autoop.tests.test_decision_tree_regressor import DecisionTreeRegressor  # noqa


if __name__ == '__main__':
    unittest.main()
