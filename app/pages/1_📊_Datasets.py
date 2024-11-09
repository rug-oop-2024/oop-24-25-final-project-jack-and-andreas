import streamlit as st

from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

# Get an instance of AutoML system
automl = AutoMLSystem.get_instance()

# Retrieve list of datasets from registry
datasets = automl.registry.list_with_cls(type="dataset", list_cls=Dataset)

# Page title
st.title("Datasets")

# Check if datasets exist
if datasets:
    for dataset in datasets:
        # Display dataset details
        st.write(f"### {dataset.name}")

        dataframe = dataset.read()

        # Show the first 10 rows
        st.write(dataframe.head(10))

        if st.button(f"Remove {dataset.name}"):
            automl.registry.delete(dataset.id)
            st.warning(f"Dataset {dataset.name} removed.")
else:
    st.write("No datasets available.")

dataset_file = st.file_uploader("Upload Dataset File")

if dataset_file:
    st.write(f"File uploaded: {dataset_file.name}")

    dataset = Dataset(
        asset_path="./datasets/"+dataset_file.name,
        name=dataset_file.name,
        version="1.0.0",
        data=dataset_file.read(),
    )

    automl.registry.register(dataset)
    st.success(f"Dataset {dataset_file.name} registered.")
