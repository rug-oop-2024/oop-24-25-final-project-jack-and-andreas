
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np

from autoop.core.ml.dataset import Dataset

class Feature(BaseModel):
    """
    Feature class takes a name and type of a feature
    """
    name:  str = Field(..., description="The name of the feature")
    feature_type: Literal['numeric', 'categorical'] = Field(..., description="The type of feature")


    def __str__(self):
        return f"Feature(name={self.name}, type={self.feature_type})"
    