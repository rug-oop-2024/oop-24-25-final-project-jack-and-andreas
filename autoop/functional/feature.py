
from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
import pandas
from pandas.api.types import is_numeric_dtype

def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """Assumption: only categorical and numerical features and no NaN values.
    Args:
        dataset: Dataset
    Returns:
        List[Feature]: List of features with their types.
    """
    data_frame = pandas.read_csv(dataset)
    features = []

    for column in data_frame.columns:
        if data_frame[column].dtype.name == 'int64':
            features.append(Feature(name=column, feature_type='categorical'))
            print(f"{column} is numeric")
        elif data_frame[column].dtype.name == 'category':
            features.append(Feature(name=column, feature_type='numeric'))
            print(f"{column} is categorical")
    return features
