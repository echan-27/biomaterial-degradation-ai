"""Streamlit app for predicting biomaterial degradation."""

import json
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
    predict_degradation_rate,
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


def min_or_default(data: pd.DataFrame, column: str, default: float) -> float:
    """Use a dataset minimum when available; otherwise use a safe default."""
    if data.empty or column not in data:
        return default
    value = data[column].min()
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


def make_key(*parts: object) -> str:
    """Create a stable Streamlit widget key from readable parts."""
    clean_parts = []
    for part in parts:
        text = str(part)
        clean_parts.append("".join(char if char.isalnum() else "_" for char in text))
    return "_".join(clean_parts)


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


def read_model_summary() -> dict:
    """Read training summary details saved by the training script."""
    if not Path(SUMMARY_PATH).exists():
        return {}
    try:
        return json.loads(Path(SUMMARY_PATH).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


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


def render_feature_inputs(
    prefix: str,
    defaults: dict[str, float],
) -> dict[str, float]:
    """Render compact numeric inputs for model features."""
    first_row = st.columns(2)
    second_row = st.columns(2)

    with first_row[0]:
        cellulose_percentage = st.number_input(
            "Cellulose (%)",
            min_value=0.0,
            max_value=100.0,
            value=defaults["cellulose_percentage"],
            step=1.0,
            key=make_key(prefix, "cellulose"),
        )
    with first_row[1]:
        temperature_c = st.number_input(
            "Temperature (C)",
            min_value=-20.0,
            max_value=100.0,
            value=defaults["temperature_c"],
            step=0.5,
            key=make_key(prefix, "temperature"),
        )
    with second_row[0]:
        ph_level = st.number_input(
            "pH",
            min_value=0.0,
            max_value=14.0,
            value=defaults["ph_level"],
            step=0.1,
            key=make_key(prefix, "ph"),
        )
    with second_row[1]:
        degree_substitution = st.number_input(
            "Degree of substitution",
            min_value=0.0,
            max_value=5.0,
            value=max(0.0, min(5.0, defaults["degree_substitution"])),
            step=0.01,
            format="%.2f",
            key=make_key(prefix, "degree_substitution"),
        )

    return {
        "cellulose_percentage": cellulose_percentage,
        "temperature_c": temperature_c,
        "ph_level": ph_level,
        "degree_substitution": degree_substitution,
    }


def format_metric_table(metrics_table: pd.DataFrame) -> pd.DataFrame:
    """Rename R2 for display without changing saved CSV columns."""
    return metrics_table.rename(columns={"R2": "R²"})


def show_metric_explanations() -> None:
    """Explain model metrics in beginner-friendly language."""
    st.markdown(
        """
        - **MAE**: Average error in the predicted degradation rate `k`. Lower is better.
        - **RMSE**: Similar to MAE, but it punishes larger `k` mistakes more. Lower is better.
        - **R²**: How much of the `k` pattern the model explains. 1.0 is perfect, 0.0 is close to guessing the average.
        """
    )


st.set_page_config(
    page_title="Biomaterial Degradation Predictor",
    layout="wide",
)

st.title("Biomaterial Degradation Predictor")
st.caption(
    "A small scikit-learn research app that predicts an exponential degradation rate."
)

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

model_summary = read_model_summary()
model_metrics = read_model_metrics()
test_metrics = read_best_test_metrics(model_metrics)
uncertainty_info = model_summary.get("uncertainty_estimate", {})
mass_uncertainty = uncertainty_info.get("value")

day_default = int(round(median_or_default(training_data, "Days_Elapsed", 14.0)))
max_training_day = int(round(max_or_default(training_data, "Days_Elapsed", 365.0)))
max_app_day = max(30, max_training_day)

predictor_tab, compare_tab, performance_tab, data_tab, about_tab = st.tabs(
    [
        "Predictor",
        "Compare Materials",
        "Model Performance",
        "Dataset Explorer",
        "About the Research",
    ]
)

with predictor_tab:
    st.subheader("Single prediction")

    input_columns = st.columns([1, 1, 1])
    with input_columns[0]:
        material_type = st.selectbox(
            "Material type",
            materials,
            key="predictor_material",
        )
    with input_columns[1]:
        environment = st.selectbox(
            "Environment",
            environments,
            key="predictor_environment",
        )
    with input_columns[2]:
        days_elapsed = st.number_input(
            "Days elapsed",
            min_value=0,
            max_value=max_app_day,
            value=max(0, min(max_app_day, day_default)),
            step=1,
            key="predictor_days_elapsed",
        )

    defaults = default_feature_values(training_data, material_type, environment)
    with st.expander("Adjust material and lab details"):
        predictor_features = render_feature_inputs(
            make_key("predictor", material_type, environment),
            defaults,
        )

    predicted_k = predict_degradation_rate(
        model=model,
        material_type=material_type,
        cellulose_percentage=predictor_features["cellulose_percentage"],
        temperature_c=predictor_features["temperature_c"],
        ph_level=predictor_features["ph_level"],
        environment=environment,
        degree_substitution=predictor_features["degree_substitution"],
    )
    mass_remaining = predict_mass_remaining(
        model=model,
        material_type=material_type,
        cellulose_percentage=predictor_features["cellulose_percentage"],
        temperature_c=predictor_features["temperature_c"],
        ph_level=predictor_features["ph_level"],
        environment=environment,
        days_elapsed=days_elapsed,
        degree_substitution=predictor_features["degree_substitution"],
    )
    degradation_percentage = predict_degradation_percentage(mass_remaining)

    if mass_uncertainty is not None:
        lower_mass, upper_mass = calculate_uncertainty_range(
            mass_remaining,
            mass_uncertainty,
        )
        mass_value = f"{mass_remaining:.1f}%"
        range_value = f"{lower_mass:.1f}%–{upper_mass:.1f}%"
    else:
        mass_value = f"{mass_remaining:.1f}%"
        range_value = "Not available"

    result_columns = st.columns(4)
    result_columns[0].metric("Predicted mass remaining", mass_value)
    result_columns[1].metric("Estimated range", range_value)
    result_columns[2].metric("Predicted degradation", f"{degradation_percentage:.1f}%")
    result_columns[3].metric("Predicted k", f"{predicted_k:.5f}/day")

    curve_end_day = st.slider(
        "Curve length (days)",
        min_value=1,
        max_value=max_app_day,
        value=min(max_app_day, max(30, int(days_elapsed))),
        step=1,
        key="predictor_curve_end_day",
    )
    days_for_curve = np.arange(0, curve_end_day + 1)
    curve = predict_degradation_curve(
        model=model,
        material_type=material_type,
        cellulose_percentage=predictor_features["cellulose_percentage"],
        temperature_c=predictor_features["temperature_c"],
        ph_level=predictor_features["ph_level"],
        environment=environment,
        degree_substitution=predictor_features["degree_substitution"],
        days=days_for_curve,
    )
    curve_display = curve.rename(
        columns={
            "Mass_Remaining_Percentage": "Mass remaining (%)",
            "Degradation_Percentage": "Degradation (%)",
        }
    )

    st.line_chart(
        curve_display,
        x="Days_Elapsed",
        y=["Mass remaining (%)", "Degradation (%)"],
    )

    with st.expander("Curve data"):
        st.dataframe(curve_display, width="stretch", hide_index=True)

with compare_tab:
    st.subheader("Compare two scenarios")

    comparison_end_day = st.slider(
        "Comparison period (days)",
        min_value=1,
        max_value=max_app_day,
        value=min(max_app_day, 90),
        step=1,
        key="comparison_period",
    )

    material_b_index = 1 if len(materials) > 1 else 0
    environment_b_index = 1 if len(environments) > 1 else 0

    comparison_columns = st.columns(2)
    with comparison_columns[0]:
        st.markdown("**Scenario A**")
        material_a = st.selectbox(
            "Material",
            materials,
            key="comparison_material_a",
        )
        environment_a = st.selectbox(
            "Environment",
            environments,
            key="comparison_environment_a",
        )

    with comparison_columns[1]:
        st.markdown("**Scenario B**")
        material_b = st.selectbox(
            "Material",
            materials,
            index=material_b_index,
            key="comparison_material_b",
        )
        environment_b = st.selectbox(
            "Environment",
            environments,
            index=environment_b_index,
            key="comparison_environment_b",
        )

    defaults_a = default_feature_values(training_data, material_a, environment_a)
    defaults_b = default_feature_values(training_data, material_b, environment_b)

    with st.expander("Adjust comparison details"):
        settings_columns = st.columns(2)
        with settings_columns[0]:
            st.markdown("**Scenario A details**")
            features_a = render_feature_inputs(
                make_key("compare_a", material_a, environment_a),
                defaults_a,
            )
        with settings_columns[1]:
            st.markdown("**Scenario B details**")
            features_b = render_feature_inputs(
                make_key("compare_b", material_b, environment_b),
                defaults_b,
            )

    comparison_days = np.arange(0, comparison_end_day + 1)
    curve_a = predict_degradation_curve(
        model=model,
        material_type=material_a,
        cellulose_percentage=features_a["cellulose_percentage"],
        temperature_c=features_a["temperature_c"],
        ph_level=features_a["ph_level"],
        environment=environment_a,
        degree_substitution=features_a["degree_substitution"],
        days=comparison_days,
    )
    curve_b = predict_degradation_curve(
        model=model,
        material_type=material_b,
        cellulose_percentage=features_b["cellulose_percentage"],
        temperature_c=features_b["temperature_c"],
        ph_level=features_b["ph_level"],
        environment=environment_b,
        degree_substitution=features_b["degree_substitution"],
        days=comparison_days,
    )

    label_a = f"A: {material_a} in {environment_a}"
    label_b = f"B: {material_b} in {environment_b}"
    comparison_curve = pd.DataFrame(
        {
            "Days elapsed": comparison_days,
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

    st.line_chart(comparison_curve, x="Days elapsed", y=[label_a, label_b])

    comparison_data = pd.DataFrame(
        {
            "Days elapsed": comparison_days,
            f"{label_a} mass remaining (%)": curve_a["Mass_Remaining_Percentage"],
            f"{label_a} degradation (%)": curve_a["Degradation_Percentage"],
            f"{label_b} mass remaining (%)": curve_b["Mass_Remaining_Percentage"],
            f"{label_b} degradation (%)": curve_b["Degradation_Percentage"],
        }
    )

    with st.expander("Comparison data"):
        st.dataframe(comparison_data, width="stretch", hide_index=True)

with performance_tab:
    st.subheader("Model performance")

    best_model_name = model_summary.get("best_model", "Saved model")
    train_rows = model_summary.get("train_rows", "N/A")
    test_rows = model_summary.get("test_rows", "N/A")

    summary_columns = st.columns(3)
    summary_columns[0].metric("Best model", best_model_name)
    summary_columns[1].metric("Training rows", train_rows)
    summary_columns[2].metric("Test rows", test_rows)

    if test_metrics:
        metric_columns = st.columns(3)
        metric_columns[0].metric("MAE", f"{test_metrics['MAE']:.5f}")
        metric_columns[1].metric("RMSE", f"{test_metrics['RMSE']:.5f}")
        metric_columns[2].metric("R²", f"{test_metrics['R2']:.3f}")

    show_metric_explanations()

    mass_metrics = model_summary.get("mass_prediction_test_metrics", {})
    if mass_metrics:
        st.markdown("**Mass remaining error on held-out test conditions**")
        mass_columns = st.columns(2)
        mass_columns[0].metric("Mass MAE", f"{mass_metrics['MAE']:.1f} points")
        mass_columns[1].metric("Mass RMSE", f"{mass_metrics['RMSE']:.1f} points")
        st.caption(
            "The Predictor tab uses Mass MAE as a simple typical-error range. "
            "This avoids the very wide 0%–100% ranges that came from propagating k uncertainty."
        )

    if not model_metrics.empty:
        st.markdown("**Model comparison on the held-out test set**")
        st.dataframe(
            format_metric_table(model_metrics),
            column_config={
                "Model": st.column_config.TextColumn("Model"),
                "MAE": st.column_config.NumberColumn("MAE", format="%.5f"),
                "RMSE": st.column_config.NumberColumn("RMSE", format="%.5f"),
                "R²": st.column_config.NumberColumn("R²", format="%.3f"),
            },
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("Run `python src/train_model.py` to generate model metrics.")

with data_tab:
    st.subheader("Dataset explorer")

    if training_data.empty:
        st.warning("No dataset could be loaded.")
    else:
        data_columns = st.columns(4)
        data_columns[0].metric("Rows", len(training_data))
        data_columns[1].metric("Materials", training_data["Material_Type"].nunique())
        data_columns[2].metric("Environments", training_data["Environment"].nunique())
        data_columns[3].metric(
            "Day range",
            f"{int(min_or_default(training_data, 'Days_Elapsed', 0))}–{int(max_app_day)}",
        )

        filter_columns = st.columns([1, 1, 1])
        with filter_columns[0]:
            selected_materials = st.multiselect(
                "Filter materials",
                materials,
                default=[],
                placeholder="All materials",
            )
        with filter_columns[1]:
            selected_environments = st.multiselect(
                "Filter environments",
                environments,
                default=[],
                placeholder="All environments",
            )
        with filter_columns[2]:
            min_day = int(min_or_default(training_data, "Days_Elapsed", 0))
            max_day = int(max_or_default(training_data, "Days_Elapsed", max_app_day))
            selected_day_range = st.slider(
                "Filter days",
                min_value=min_day,
                max_value=max_day,
                value=(min_day, max_day),
            )

        filtered_data = training_data.copy()
        if selected_materials:
            filtered_data = filtered_data[
                filtered_data["Material_Type"].isin(selected_materials)
            ]
        if selected_environments:
            filtered_data = filtered_data[
                filtered_data["Environment"].isin(selected_environments)
            ]
        filtered_data = filtered_data[
            filtered_data["Days_Elapsed"].between(
                selected_day_range[0],
                selected_day_range[1],
            )
        ]

        st.metric("Filtered rows", len(filtered_data))
        st.dataframe(filtered_data, width="stretch", hide_index=True, height=360)

with about_tab:
    st.subheader("About the research")

    st.markdown(
        """
        This project estimates how much mass a biomaterial may have remaining after
        exposure to a chosen environment for a chosen amount of time. The target
        shown to users is `Mass_Remaining_Percentage`.

        The machine learning model predicts a degradation rate constant called
        `k`. The app then uses the exponential decay equation
        `mass remaining = 100 * exp(-k * days)` to make predictions over time.
        Since `k` is kept non-negative, the predicted mass remaining cannot
        increase as days elapsed increases.
        """
    )

    st.markdown("**Features used by the model**")
    st.markdown(
        """
        - Material type
        - Cellulose percentage
        - Temperature
        - pH
        - Environment
        - Degree of substitution
        """
    )

    st.markdown("**Important limitations**")
    st.markdown(
        """
        - Predictions are only as reliable as the dataset used for training.
        - The estimated range is based on held-out test error for mass remaining.
          It is a rough typical-error range, not a formal confidence interval.
        - Real degradation can depend on factors not included here, such as
          humidity, microbial activity, sample thickness, and measurement error.
        - Use the predictions as a research guide, not as proof of laboratory results.
        """
    )
