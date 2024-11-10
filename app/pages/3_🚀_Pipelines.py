import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
from autoop.core.ml.model import from_artifact
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.dataset import Dataset

import pickle

st.set_page_config(page_title="Pipelines", page_icon="🚀")


def write_helper_text(text: str) -> None:
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# 🚀 Pipelines")

automl = AutoMLSystem.get_instance()

new_file = st.file_uploader("Upload a new dataset to run this on")

st.write("### Data Split")
split = st.slider("Train-Test Split", 0.1, 0.9, 0.7, 0.1)

pipelines = automl.registry.list("pipeline_config")
pipeline_artifact = st.selectbox(
    "Select a pipeline",
    pipelines,
    format_func=lambda x: f"{x.name} - {x.version}"
)

if st.button("Load Pipeline"):
    """ Load datasets """
    if pipeline_artifact:
        st.write(
            f"### Dataset: {pipeline_artifact.name} "
            f"(version {pipeline_artifact.version})"
        )
        data_bytes = pipeline_artifact.read()

        pipeline_config = pickle.loads(data_bytes)
        # st.write(pipeline_config)

        st.write("## Pipeline Config")
        st.write("### Input Features")
        st.dataframe(pd.DataFrame(
            [(f.name, f.type) for f in pipeline_config["input_features"]]
        ))

        st.write("### Target Feature")
        target_feature = pipeline_config["target_feature"]
        st.write(target_feature.name, target_feature.type)

        st.write("### Model")

        model_artifact = automl.registry.get(pipeline_config["model"])
        model_bytes = model_artifact.read()
        model = pickle.loads(model_bytes)

        parsed_model = from_artifact(model_artifact)

        st.write(parsed_model.type)

        st.write("### Hyperparameters")
        st.write(parsed_model.hyperparameters)

        st.write("### Parameters")
        st.write(parsed_model.parameters)

        if new_file:
            dataset = Dataset(
                asset_path=new_file.name,
                version="1.0.0",
                data=new_file.read(),
                tags=[],
                name=new_file.name
            )

            pd = dataset.read()
            st.write(pd)

            pipeline = Pipeline(
                metrics=pipeline_config["metrics"],
                model=parsed_model,
                input_features=pipeline_config["input_features"],
                target_feature=pipeline_config["target_feature"],
                dataset=dataset,
                split=split
            )

            pipeline_results = pipeline.execute()

            st.write("## 📊 Results")
            metrics_values = pipeline_results["metrics"]
            metrics_values = [
                [type(x).__name__ for (x, _) in metrics_values],
                [y for (_, y) in metrics_values]
            ]

            st.dataframe(metrics_values, hide_index=True)

            st.write("## 📈 Predictions")
            st.write("Predictions on the test set.")

            predictions = pipeline_results["predictions"]
            st.dataframe(predictions, hide_index=True)
    else:
        st.write("No datasets selected.")

if st.button("Delete Pipeline"):
    if pipeline_artifact:
        automl.registry.delete(pipeline_artifact.id)
        st.write(f"Pipeline {pipeline_artifact.name} deleted.")
    else:
        st.write("No datasets selected.")
