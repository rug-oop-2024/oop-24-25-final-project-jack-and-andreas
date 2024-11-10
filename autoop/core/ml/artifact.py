from pydantic import BaseModel, Field
import base64


class Artifact(BaseModel):
    """
    A class to represent an ML artifact
    Args: asset_path, version, data, metadata, type, tags, name
    Returns: an artifact object
    """
    asset_path: str = Field(..., description="Path to the asset")
    version: str = Field(..., description="Version of the artifact")
    data: bytes = Field(..., description="Binary state data of the artifact")
    metadata: dict = Field(
        default_factory=dict,
        description="Metadata associated with the artifact"
    )
    type: str = Field(
        ...,
        description="Type of the artifact (e.g., model, dataset)"
    )
    tags: list = Field(
        default_factory=list,
        description="Tags for the artifact"
    )
    name: str = Field(
        ...,
        description="Name of the artifact"
    )

    @property
    def id(self) -> str:
        """Generate a unique ID for the artifact."""
        encoded_path = base64.b64encode(self.asset_path.encode()).decode()
        return f"{encoded_path}:{self.version}"

    def read(self) -> bytes:
        if self.data is None:
            self.data = open(self.asset_path, "rb").read()
        return self.data
