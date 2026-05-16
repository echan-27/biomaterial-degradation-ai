"""Streamlit app for predicting biomaterial degradation."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.clean_data import get_category_options, load_training_data
from src.config import DATA_PATH, MODEL_PATH, SUMMARY_PATH
from src.predict import (
    load_model,
    predict_degradation_curve,
    predict_degradation_percentage,
    predict_mass_remaining,
)


DEFAULT_MATERIALS = ["Hemp", "Jute", "Sisal", "Viscose"]
DEFAULT_ENVIRONMENTS = ["Soil", "Compost", "Water"]


@st.cache_resource
def get_model():
    """Load the trained model once while the Streamlit app is running."""
    return load_model(MODEL_PATH)


@st.cache_data
def get_training_data() -> pd.DataFrame:
    """Load the spreadsheet for app defaults and dropdown options."""
    try:
        return load_training_data(DATA_PATH)
    except Exception:
        return pd.DataFrame()


def median_or_default(data: pd.DataFrame, column: str, default: float) -> float:
    """Use a dataset median when available; otherwise use a safe default."""
    if data.empty or column not in data:
        return default
    value = data[column].median()
    if pd.isna(value):
        return default
    return float(value)


def read_model_summary() -> str:
    """Return the saved model name if the training script wrote a summary file."""
    if not Path(SUMMARY_PATH).exists():
        return "Saved model"
    try:
        summary = pd.read_json(SUMMARY_PATH, typ="series")
        return str(summary.get("best_model", "Saved model"))
    except ValueError:
        return "Saved model"


st.set_page_config(
    page_title="Biomaterial Degradation Predictor",
    layout="wide",
)

st.title("Biomaterial Degradation Predictor")

try:
    model = get_model()
except Exception as error:
    st.error(str(error))
    st.info("Train the model first with `python src/train_model.py`.")
    st.stop()

training_data = get_training_data()
materials = (
    get_category_options(training_data, "Material_Type")
    if not training_data.empty
    else DEFAULT_MATERIALS
)
environments = (
    get_category_options(training_data, "Environment")
    if not training_data.empty
    else DEFAULT_ENVIRONMENTS
)

with st.sidebar:
    st.header("Inputs")

    material_type = st.selectbox("Material type", materials)
    environment = st.selectbox("Environment", environments)

    matching_material = (
        training_data[training_data["Material_Type"] == material_type]
        if not training_data.empty
        else pd.DataFrame()
    )
    matching_environment = (
        training_data[training_data["Environment"] == environment]
        if not training_data.empty
        else pd.DataFrame()
    )

    cellulose_default = median_or_default(
        matching_material,
        "Cellulose_Percentage",
        median_or_default(training_data, "Cellulose_Percentage", 60.0),
    )
    temperature_default = median_or_default(
        matching_environment,
        "Temperature_C",
        median_or_default(training_data, "Temperature_C", 25.0),
    )
    ph_default = median_or_default(
        matching_environment,
        "pH_Level",
        median_or_default(training_data, "pH_Level", 7.0),
    )
    day_default = int(round(median_or_default(training_data, "Days_Elapsed", 14.0)))
    ds_default = median_or_default(
        matching_material,
        "degree_substitution",
        median_or_default(training_data, "degree_substitution", 0.0),
    )

    cellulose_percentage = st.number_input(
        "Cellulose percentage",
        min_value=0.0,
        max_value=100.0,
        value=cellulose_default,
        step=1.0,
    )
    temperature_c = st.number_input(
        "Temperature (C)",
        min_value=-20.0,
        max_value=100.0,
        value=temperature_default,
        step=0.5,
    )
    ph_level = st.number_input(
        "pH level",
        min_value=0.0,
        max_value=14.0,
        value=ph_default,
        step=0.1,
    )
    days_elapsed = st.number_input(
        "Days elapsed",
        min_value=0,
        max_value=365,
        value=max(0, min(365, day_default)),
        step=1,
    )
    degree_substitution = st.number_input(
        "Degree of substitution",
        min_value=0.0,
        max_value=5.0,
        value=max(0.0, min(5.0, ds_default)),
        step=0.01,
        format="%.2f",
    )
    curve_end_day = st.slider(
        "Curve end day",
        min_value=1,
        max_value=365,
        value=max(30, int(days_elapsed)),
        step=1,
    )

mass_remaining = predict_mass_remaining(
    model=model,
    material_type=material_type,
    cellulose_percentage=cellulose_percentage,
    temperature_c=temperature_c,
    ph_level=ph_level,
    environment=environment,
    days_elapsed=days_elapsed,
    degree_substitution=degree_substitution,
)
degradation_percentage = predict_degradation_percentage(mass_remaining)

metric_columns = st.columns(3)
metric_columns[0].metric("Mass remaining", f"{mass_remaining:.1f}%")
metric_columns[1].metric("Degradation", f"{degradation_percentage:.1f}%")
metric_columns[2].metric("Model", read_model_summary())

days_for_curve = np.arange(0, curve_end_day + 1)
curve = predict_degradation_curve(
    model=model,
    material_type=material_type,
    cellulose_percentage=cellulose_percentage,
    temperature_c=temperature_c,
    ph_level=ph_level,
    environment=environment,
    degree_substitution=degree_substitution,
    days=days_for_curve,
)

st.subheader("Predicted degradation curve")
st.line_chart(
    curve,
    x="Days_Elapsed",
    y=["Mass_Remaining_Percentage", "Degradation_Percentage"],
)

with st.expander("Curve data"):
    st.dataframe(curve, width="stretch", hide_index=True)
