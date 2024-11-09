import streamlit as st
from app.core.system import AutoMLSystem


st.set_page_config(page_title="Pipelines", page_icon="🚀")


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# 🚀 Pipelines")


automl = AutoMLSystem.get_instance()

pipelines = automl.registry.list("pipeline_config")
pipeline = st.selectbox("Select a pipeline", pipelines,
                        format_func=lambda x: f"{x.name} - {x.version}")
