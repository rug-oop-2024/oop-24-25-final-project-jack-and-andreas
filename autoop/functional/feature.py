
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
    data_frame = pandas.read_csv('dataset.csv')

    for column in data_frame.columns:
        if data_frame[column].dtype.name == 'int64':
            print(f"{column} is numeric")
        elif data_frame[column].dtype.name == 'category':
            print(f"{column} is categorical")

    raise NotImplementedError("This should be implemented by you.")