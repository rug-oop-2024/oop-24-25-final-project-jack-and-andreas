import streamlit as st
import pandas as pd
from app.core.system import AutoMLSystem
import io

st.set_page_config(page_title="Pipelines", page_icon="🚀")


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# 🚀 Pipelines")

automl = AutoMLSystem.get_instance()

pipelines = automl.registry.list("pipeline_config")
pipeline = st.selectbox("Select a pipeline", pipelines,
                        format_func=lambda x: f"{x.name} - {x.version}")

if st.button("Load Datasets"):
    """ Load datasets """
    if pipeline:
        st.write(f"### Dataset: {pipeline.name} (version {pipeline.version})")
        data_bytes = pipeline.read()
        dataframe = pd.read_csv(io.BytesIO(data_bytes), encoding='ISO-8859-1')
        st.dataframe(dataframe.head(10))
    else:
        st.write("No datasets selected.")
