import streamlit as st
# import pandas as pd

from app.core.system import AutoMLSystem
# from autoop.core.ml.dataset import Dataset


st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text(
    "In this section, you can design a machine learning pipeline to train a"
    "model on a dataset."
)

automl = AutoMLSystem.get_instance()
datasets = automl.registry.list(type="dataset")

if not datasets:
    st.write("No datasets found.")
else:
    dataset = st.selectbox("Select a dataset", datasets)
    dataset = automl.registry.get(dataset)

    st.write(f"Dataset: {dataset.name}")

    st.write("## 🧠 Model Selection")
    write_helper_text(
        "Select a model to train on the dataset. You can also select the"
        "hyperparameters for the model."
    )

    models = automl.registry.list(type="model")
    model = st.selectbox("Select a model", models)
    model = automl.registry.get(model)

    st.write(f"Model: {model.name}")

    st.write("## 🎯 Metrics")
    write_helper_text(
        "Select the evaluation metrics to use for training the model."
    )

    metrics = automl.registry.list(type="metric")
    metric = st.selectbox("Select a metric", metrics)
    metric = automl.registry.get(metric)

    st.write(f"Metric: {metric.name}")

    st.write("## 🚀 Training")
    write_helper_text(
        "Click the button below to start training the model on the dataset."
    )

    if st.button("Train"):
        automl.train(dataset, model, metric)
        st.write("Training complete.")
        st.write("## 📊 Results")
        write_helper_text(
            "View the results of the training process."
        )

        results = automl.get_results()
        st.write(results)
