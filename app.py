"""Streamlit app for predicting biomaterial degradation."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.clean_data import get_category_options, load_training_data
from src.config import DATA_PATH, METRICS_PATH, MODEL_PATH, SUMMARY_PATH
from src.predict import (
    calculate_uncertainty_range,
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


def max_or_default(data: pd.DataFrame, column: str, default: float) -> float:
    """Use a dataset maximum when available; otherwise use a safe default."""
    if data.empty or column not in data:
        return default
    value = data[column].max()
    if pd.isna(value):
        return default
    return float(value)


def default_feature_values(
    data: pd.DataFrame,
    material: str,
    environment: str,
) -> dict[str, float]:
    """Choose reasonable numeric defaults for a material/environment pair."""
    matching_material = (
        data[data["Material_Type"] == material]
        if not data.empty and "Material_Type" in data
        else pd.DataFrame()
    )
    matching_environment = (
        data[data["Environment"] == environment]
        if not data.empty and "Environment" in data
        else pd.DataFrame()
    )

    return {
        "cellulose_percentage": median_or_default(
            matching_material,
            "Cellulose_Percentage",
            median_or_default(data, "Cellulose_Percentage", 60.0),
        ),
        "temperature_c": median_or_default(
            matching_environment,
            "Temperature_C",
            median_or_default(data, "Temperature_C", 25.0),
        ),
        "ph_level": median_or_default(
            matching_environment,
            "pH_Level",
            median_or_default(data, "pH_Level", 7.0),
        ),
        "degree_substitution": median_or_default(
            matching_material,
            "degree_substitution",
            median_or_default(data, "degree_substitution", 0.0),
        ),
    }


def read_model_summary() -> str:
    """Return the saved model name if the training script wrote a summary file."""
    if not Path(SUMMARY_PATH).exists():
        return "Saved model"
    try:
        summary = pd.read_json(SUMMARY_PATH, typ="series")
        return str(summary.get("best_model", "Saved model"))
    except ValueError:
        return "Saved model"


def read_model_metrics() -> pd.DataFrame:
    """Read the full model comparison table from model_metrics.csv."""
    if not Path(METRICS_PATH).exists():
        return pd.DataFrame()

    try:
        metrics_table = pd.read_csv(METRICS_PATH)
    except Exception:
        return pd.DataFrame()

    required_columns = ["Model", "MAE", "RMSE", "R2"]
    if metrics_table.empty or any(column not in metrics_table for column in required_columns):
        return pd.DataFrame()

    return metrics_table[required_columns].sort_values("RMSE").reset_index(drop=True)


def read_best_test_metrics(metrics_table: pd.DataFrame) -> dict[str, float] | None:
    """Read the best model's held-out test metrics from the comparison table."""
    if metrics_table.empty:
        return None

    required_columns = ["MAE", "RMSE", "R2"]
    if any(column not in metrics_table for column in required_columns):
        return None

    best_row = metrics_table.sort_values("RMSE").iloc[0]
    return {
        "MAE": float(best_row["MAE"]),
        "RMSE": float(best_row["RMSE"]),
        "R2": float(best_row["R2"]),
    }


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
    max_training_day = int(round(max_or_default(training_data, "Days_Elapsed", 365.0)))
    max_app_day = max(30, max_training_day)
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
        max_value=max_app_day,
        value=max(0, min(max_app_day, day_default)),
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
        max_value=max_app_day,
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

model_metrics = read_model_metrics()
test_metrics = read_best_test_metrics(model_metrics)
uncertainty = test_metrics["RMSE"] if test_metrics else None

st.subheader("Prediction")
if uncertainty is not None:
    lower_mass, upper_mass = calculate_uncertainty_range(mass_remaining, uncertainty)
    st.write(f"Predicted mass remaining: **{mass_remaining:.1f}% ± {uncertainty:.1f}%**")
    st.write(f"Estimated range: **{lower_mass:.1f}%–{upper_mass:.1f}%**")
else:
    st.write(f"Predicted mass remaining: **{mass_remaining:.1f}%**")

metric_columns = st.columns(3)
mass_metric_value = f"{mass_remaining:.1f}%"
if uncertainty is not None:
    mass_metric_value = f"{mass_remaining:.1f}% ± {uncertainty:.1f}%"
metric_columns[0].metric("Predicted mass remaining", mass_metric_value)
metric_columns[1].metric("Degradation", f"{degradation_percentage:.1f}%")
metric_columns[2].metric("Model", read_model_summary())

if test_metrics:
    st.subheader("Held-out test performance")
    performance_columns = st.columns(3)
    performance_columns[0].metric("MAE", f"{test_metrics['MAE']:.2f}")
    performance_columns[1].metric("RMSE", f"{test_metrics['RMSE']:.2f}")
    performance_columns[2].metric("R²", f"{test_metrics['R2']:.3f}")

if not model_metrics.empty:
    st.subheader("Model comparison")
    display_metrics = model_metrics.rename(columns={"R2": "R²"})
    st.dataframe(
        display_metrics,
        column_config={
            "Model": st.column_config.TextColumn("Model"),
            "MAE": st.column_config.NumberColumn("MAE", format="%.2f"),
            "RMSE": st.column_config.NumberColumn("RMSE", format="%.2f"),
            "R²": st.column_config.NumberColumn("R²", format="%.3f"),
        },
        width="stretch",
        hide_index=True,
    )

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
    uncertainty=uncertainty,
)

st.subheader("Predicted degradation curve")
st.line_chart(
    curve,
    x="Days_Elapsed",
    y=["Mass_Remaining_Percentage", "Degradation_Percentage"],
)

with st.expander("Curve data"):
    st.dataframe(curve, width="stretch", hide_index=True)

st.subheader("Compare degradation curves")

default_comparison_end_day = min(max_app_day, max(90, int(days_elapsed)))
comparison_end_day = st.slider(
    "Comparison period (days)",
    min_value=1,
    max_value=max_app_day,
    value=default_comparison_end_day,
    step=1,
)

material_b_index = 1 if len(materials) > 1 else 0
environment_b_index = 1 if len(environments) > 1 else 0

comparison_columns = st.columns(2)
with comparison_columns[0]:
    material_a = st.selectbox("Biomaterial A", materials, key="comparison_material_a")
    environment_a = st.selectbox(
        "Environment A",
        environments,
        key="comparison_environment_a",
    )

with comparison_columns[1]:
    material_b = st.selectbox(
        "Biomaterial B",
        materials,
        index=material_b_index,
        key="comparison_material_b",
    )
    environment_b = st.selectbox(
        "Environment B",
        environments,
        index=environment_b_index,
        key="comparison_environment_b",
    )

defaults_a = default_feature_values(training_data, material_a, environment_a)
defaults_b = default_feature_values(training_data, material_b, environment_b)

with st.expander("Comparison settings"):
    settings_columns = st.columns(2)
    with settings_columns[0]:
        st.markdown("**Curve A**")
        cellulose_a = st.number_input(
            "A cellulose percentage",
            min_value=0.0,
            max_value=100.0,
            value=defaults_a["cellulose_percentage"],
            step=1.0,
            key=f"cellulose_a_{material_a}_{environment_a}",
        )
        temperature_a = st.number_input(
            "A temperature (C)",
            min_value=-20.0,
            max_value=100.0,
            value=defaults_a["temperature_c"],
            step=0.5,
            key=f"temperature_a_{material_a}_{environment_a}",
        )
        ph_a = st.number_input(
            "A pH level",
            min_value=0.0,
            max_value=14.0,
            value=defaults_a["ph_level"],
            step=0.1,
            key=f"ph_a_{material_a}_{environment_a}",
        )
        degree_substitution_a = st.number_input(
            "A degree of substitution",
            min_value=0.0,
            max_value=5.0,
            value=max(0.0, min(5.0, defaults_a["degree_substitution"])),
            step=0.01,
            format="%.2f",
            key=f"degree_substitution_a_{material_a}_{environment_a}",
        )

    with settings_columns[1]:
        st.markdown("**Curve B**")
        cellulose_b = st.number_input(
            "B cellulose percentage",
            min_value=0.0,
            max_value=100.0,
            value=defaults_b["cellulose_percentage"],
            step=1.0,
            key=f"cellulose_b_{material_b}_{environment_b}",
        )
        temperature_b = st.number_input(
            "B temperature (C)",
            min_value=-20.0,
            max_value=100.0,
            value=defaults_b["temperature_c"],
            step=0.5,
            key=f"temperature_b_{material_b}_{environment_b}",
        )
        ph_b = st.number_input(
            "B pH level",
            min_value=0.0,
            max_value=14.0,
            value=defaults_b["ph_level"],
            step=0.1,
            key=f"ph_b_{material_b}_{environment_b}",
        )
        degree_substitution_b = st.number_input(
            "B degree of substitution",
            min_value=0.0,
            max_value=5.0,
            value=max(0.0, min(5.0, defaults_b["degree_substitution"])),
            step=0.01,
            format="%.2f",
            key=f"degree_substitution_b_{material_b}_{environment_b}",
        )

comparison_days = np.arange(0, comparison_end_day + 1)
curve_a = predict_degradation_curve(
    model=model,
    material_type=material_a,
    cellulose_percentage=cellulose_a,
    temperature_c=temperature_a,
    ph_level=ph_a,
    environment=environment_a,
    degree_substitution=degree_substitution_a,
    days=comparison_days,
    uncertainty=uncertainty,
)
curve_b = predict_degradation_curve(
    model=model,
    material_type=material_b,
    cellulose_percentage=cellulose_b,
    temperature_c=temperature_b,
    ph_level=ph_b,
    environment=environment_b,
    degree_substitution=degree_substitution_b,
    days=comparison_days,
    uncertainty=uncertainty,
)

label_a = f"A: {material_a} in {environment_a}"
label_b = f"B: {material_b} in {environment_b}"
comparison_curve = pd.DataFrame(
    {
        "Days_Elapsed": comparison_days,
        label_a: curve_a["Degradation_Percentage"],
        label_b: curve_b["Degradation_Percentage"],
    }
)

final_columns = st.columns(2)
final_columns[0].metric(
    "A final degradation",
    f"{curve_a['Degradation_Percentage'].iloc[-1]:.1f}%",
)
final_columns[1].metric(
    "B final degradation",
    f"{curve_b['Degradation_Percentage'].iloc[-1]:.1f}%",
)

st.line_chart(comparison_curve, x="Days_Elapsed", y=[label_a, label_b])

comparison_data = pd.DataFrame(
    {
        "Days_Elapsed": comparison_days,
        f"{label_a} Mass_Remaining_Percentage": curve_a[
            "Mass_Remaining_Percentage"
        ],
        f"{label_a} Degradation_Percentage": curve_a["Degradation_Percentage"],
        f"{label_b} Mass_Remaining_Percentage": curve_b[
            "Mass_Remaining_Percentage"
        ],
        f"{label_b} Degradation_Percentage": curve_b["Degradation_Percentage"],
    }
)

with st.expander("Comparison data"):
    st.dataframe(comparison_data, width="stretch", hide_index=True)
