
from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """
    Detect the types of features in the dataset.
    Args: Dataset object containing the data.
    Returns: List of Feature objects with their detected types.
    Assumptions: The data is either categorical or numerical
    """
    features = []
    raw = dataset.read()

    for column in raw.columns:
        if raw[column].dtype == 'object':
            feature = Feature(name=column, type="categorical")
        else:
            feature = Feature(name=column, type="numerical")
        features.append(feature)

    return features
