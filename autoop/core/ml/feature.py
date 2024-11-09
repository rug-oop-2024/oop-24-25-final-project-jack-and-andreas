
from pydantic import BaseModel, Field
from typing import Literal


class Feature(BaseModel):
    """
    Feature class takes a name and type of a feature
    """
    name:  str = Field(..., description="The name of the feature")
    type: Literal['numerical', 'categorical'] = Field(
        ...,
        description="Type of the feature, either 'categorical' or 'numerical'"
    )

    def __str__(self):
        return f"Feature(name={self.name}, type={self.type})"
