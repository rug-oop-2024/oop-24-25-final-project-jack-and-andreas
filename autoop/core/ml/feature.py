
from pydantic import BaseModel, Field
from typing import Literal


class Feature(BaseModel):
    """
    Feature class
    Args: name, type
    Returns: a string representation of the feature
    """
    name: str = Field(..., description="The name of the feature")
    type: Literal['numerical', 'categorical'] = Field(
        ...,
        description="Type of the feature, either 'categorical' or 'numerical'"
    )

    def __str__(self) -> str:
        """
        Return a string representation of the Feature object.
        """
        return f"Feature(name={self.name}, type={self.type})"
