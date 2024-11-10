from autoop.core.ml.artifact import Artifact
import pandas as pd
import io


class Dataset(Artifact):
    """
    A class to represent an ML dataset
    Args: from Artifact; asset_path, version, data, metadata, type, tags, name
    Return: a dataset object
    """
    def __init__(self, *args, **kwargs):
        # If type not in kwargs, set it to "dataset"
        if "type" not in kwargs:
            kwargs["type"] = "dataset"

        super().__init__(*args, **kwargs)

    @staticmethod
    def from_dataframe(data: pd.DataFrame, name: str,
                       asset_path: str, version: str = "1.0.0"):
        """ Create a dataset from a pandas dataframe."""
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
        )

    def read(self) -> pd.DataFrame:
        """ Read data from a given path """
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """ Save data to a given path """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
