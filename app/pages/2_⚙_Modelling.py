import streamlit as st
# import pandas as pd

from app.core.system import AutoMLSystem
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.dataset import Dataset

from autoop.core.ml.model import (
    get_classification_models,
    get_regression_models,
    get_model
)

from autoop.core.ml.metrics import (
    get_regression_metrics,
    get_classification_metrics,
    get_metric
)

from autoop.functional.feature import detect_feature_types


st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text(
    "In this section, you can design a machine learning pipeline to train a"
    "model on a dataset."
)

automl = AutoMLSystem.get_instance()
datasets = automl.registry.list_with_cls(type="dataset", list_cls=Dataset)

if not datasets:
    st.write("No datasets found.")
else:
    dataset = st.selectbox(
        "Select a dataset", datasets, format_func=lambda dataset: dataset.name
    )

    st.write(f"Dataset: {dataset.name}")

    if dataset is None:
        st.stop()

    features = detect_feature_types(dataset)

    # Input features
    st.write("### Input Features")
    input_features = st.multiselect(
        "Select features", features, format_func=lambda feature: feature.name
    )

    # Target feature
    st.write("### Target Feature")
    target_feature = st.selectbox(
        "Select target feature",
        features,
        format_func=lambda feature: feature.name
    )

    st.write("## 🧠 Model Selection")
    write_helper_text(
        "Select a model to train on the dataset. You can also select the"
        "hyperparameters for the model."
    )

    models = []
    if target_feature.type == "categorical":
        models = get_classification_models()
    elif target_feature.type == "numerical":
        models = get_regression_models()

    model = st.selectbox("Select a model", models)
    model_instance = get_model(model)

    st.write(f"Model: {model}")

    st.write("### Hyperparameters")
    hyperparameters = model_instance.hyperparameters

    for name, value in hyperparameters.items():
        if isinstance(value, (int, float)):
            hyperparameters[name] = st.number_input(name, value=value)
        elif isinstance(value, str):
            hyperparameters[name] = st.text_input(name, value=value)
        elif isinstance(value, bool):
            hyperparameters[name] = st.checkbox(name, value=value)
        else:
            hyperparameters[name] = st.text_input(name, value=str(value))

    st.write("## 🎯 Metrics")
    write_helper_text(
        "Select the evaluation metrics to use for training the model."
    )

    # metrics = get_metric_names()
    metrics = []
    if target_feature.type == "categorical":
        metrics = get_classification_metrics()
    elif target_feature.type == "numerical":
        metrics = get_regression_metrics()

    selected_metrics = st.multiselect("Select metrics", metrics)

    names = ", ".join(selected_metrics)
    st.write(f"Metrics: {names}")

    st.write("## 🚀 Training")
    write_helper_text(
        "Click the button below to start training the model on the dataset."
    )

    # Data split
    st.write("### Data Split")
    split = st.slider("Train-Test Split", 0.1, 0.9, 0.7, 0.1)

    pipeline = Pipeline(
        metrics=[get_metric(name) for name in selected_metrics],
        dataset=dataset,
        model=model_instance,
        input_features=input_features,
        target_feature=target_feature,
        split=split
    )

    if "trained" not in st.session_state:
        st.session_state["trained"] = False
    if st.button("Train"):
        st.session_state["trained"] = True
        pipeline_results = pipeline.execute()

        st.write("## 📊 Results")
        metrics_values = pipeline_results["metrics"]
        metrics_values = [
            [type(x).__name__ for (x, _) in metrics_values],
            [y for (_, y) in metrics_values]
        ]

        st.dataframe(metrics_values, hide_index=True)

        # Predictions
        st.write("## 📈 Predictions")
        st.write("Predictions on the test set.")

        predictions = pipeline_results["predictions"]
        st.dataframe(predictions, hide_index=True)

    if st.session_state["trained"]:
        st.write("## 📦 Save Pipeline")
        pipe_name = st.text_input("Enter pipeline name")
        pipe_version = st.text_input("Enter pipeline version")

        if st.button("Save"):
            artifacts = pipeline.artifacts

            for artifact in artifacts:
                artifact.name = f"{pipe_name}-{pipe_version}-{artifact.name}"
                artifact.asset_path = f"{pipe_name}/{artifact.asset_path}"
                artifact.version = pipe_version
                automl.registry.register(artifact)

            st.write("Pipeline saved successfully.")
