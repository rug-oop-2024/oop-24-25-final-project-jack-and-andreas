import streamlit as st
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset

automl = AutoMLSystem.get_instance()
datasets = automl.registry.list_with_cls(type="dataset", list_cls=Dataset)

st.title("Datasets")

# Display datasets if there are any
# Remove datasets if needed
if datasets:
    for dataset in datasets:
        # Display
        st.write(f"### {dataset.name}")
        dataframe = dataset.read()
        st.write(dataframe.head(10))
        # Removal
        if st.button(f"Remove {dataset.name}"):
            automl.registry.delete(dataset.id)
            st.warning(f"Dataset {dataset.name} removed.")
else:
    st.write("No datasets available.")


# Upload given dataset
# Then register it
dataset_file = st.file_uploader("Upload Dataset File")

if dataset_file:
    st.write(f"File uploaded: {dataset_file.name}")

    dataset = Dataset(
        asset_path="./datasets/" + dataset_file.name,
        name=dataset_file.name,
        version="1.0.0",
        data=dataset_file.read(),
    )
    # Register the dataset if one has been uploaded.
    automl.registry.register(dataset)
    st.success(f"Dataset {dataset_file.name} registered.")
