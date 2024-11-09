
from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Detect the types of features in the dataset.

    Args:
        dataset: Dataset object containing the data.

    Returns:
        List[Feature]: List of Feature objects with their detected types.
    """
    features = []
    raw = dataset.read()

    for column in raw.columns:
        # Categorical features are usually of type 'object'
        if raw[column].dtype == 'object':
            feature = Feature(name=column, type="categorical")
        else:
            feature = Feature(name=column, type="numerical")
        features.append(feature)

    return features
