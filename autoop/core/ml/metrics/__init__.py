from autoop.core.ml.metrics.accuracy import Accuracy  # noqa
from autoop.core.ml.metrics.mean_absolute_error import MeanAbsoluteError  # noqa
from autoop.core.ml.metrics.mean_squared_error import MeanSquaredError  # noqa
from autoop.core.ml.metrics.precision import Precision  # noqa
from autoop.core.ml.metrics.recall import Recall  # noqa
from autoop.core.ml.metrics.r_2_score import R2Score  # noqa

from autoop.core.ml.metrics.metric import Metric  # noqa

METRICS = {
    "accuracy": Accuracy(),
    "mean_squared_error": MeanSquaredError(),
    "precision": Precision(),
    "recall": Recall(),
    "mean_absolute_error": MeanAbsoluteError(),
    "r2_score": R2Score(),
}


def get_metric(name: str) -> Metric:
    """Get a metric by name."""
    return METRICS[name]


def get_metric_names() -> list[str]:
    """Get the names of all available metrics."""
    return list(METRICS.keys())