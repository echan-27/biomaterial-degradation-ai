"""Prediction helpers for the Streamlit app and quick experiments."""

from collections.abc import Iterable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, MODEL_PATH


def load_model(model_path: Path = MODEL_PATH):
    """Load the saved scikit-learn Pipeline."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if model_path.stat().st_size == 0:
        raise ValueError(
            f"Model file is empty: {model_path}. Run `python src/train_model.py` first."
        )
    return joblib.load(model_path)


def make_input_dataframe(
    material_type: str,
    cellulose_percentage: float,
    temperature_c: float,
    ph_level: float,
    environment: str,
    days_elapsed: float,
    degree_substitution: float,
) -> pd.DataFrame:
    """Create a one-row table with the exact feature columns the model expects."""
    row = {
        "Material_Type": material_type,
        "Cellulose_Percentage": cellulose_percentage,
        "Temperature_C": temperature_c,
        "pH_Level": ph_level,
        "Environment": environment,
        "Days_Elapsed": days_elapsed,
        "degree_substitution": degree_substitution,
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def predict_mass_remaining(
    model,
    material_type: str,
    cellulose_percentage: float,
    temperature_c: float,
    ph_level: float,
    environment: str,
    days_elapsed: float,
    degree_substitution: float,
) -> float:
    """Predict mass remaining percentage and keep it in a realistic 0-100 range."""
    input_data = make_input_dataframe(
        material_type=material_type,
        cellulose_percentage=cellulose_percentage,
        temperature_c=temperature_c,
        ph_level=ph_level,
        environment=environment,
        days_elapsed=days_elapsed,
        degree_substitution=degree_substitution,
    )
    prediction = float(model.predict(input_data)[0])
    return float(np.clip(prediction, 0, 100))


def predict_degradation_percentage(mass_remaining_percentage: float) -> float:
    """Convert mass remaining into degradation percentage."""
    return float(np.clip(100 - mass_remaining_percentage, 0, 100))


def calculate_uncertainty_range(
    prediction: float,
    uncertainty: float,
) -> tuple[float, float]:
    """Create a 0-100% range around a prediction using the uncertainty value."""
    lower_bound = float(np.clip(prediction - uncertainty, 0, 100))
    upper_bound = float(np.clip(prediction + uncertainty, 0, 100))
    return lower_bound, upper_bound


def predict_degradation_curve(
    model,
    material_type: str,
    cellulose_percentage: float,
    temperature_c: float,
    ph_level: float,
    environment: str,
    degree_substitution: float,
    days: Iterable[float],
    uncertainty: float | None = None,
) -> pd.DataFrame:
    """Predict mass remaining and degradation for many time points."""
    days_array = np.array(list(days), dtype=float)
    if len(days_array) == 0:
        raise ValueError("At least one day value is needed to make a curve.")

    curve_inputs = pd.DataFrame(
        {
            "Material_Type": material_type,
            "Cellulose_Percentage": cellulose_percentage,
            "Temperature_C": temperature_c,
            "pH_Level": ph_level,
            "Environment": environment,
            "Days_Elapsed": days_array,
            "degree_substitution": degree_substitution,
        },
        columns=FEATURE_COLUMNS,
    )

    mass_remaining = np.clip(model.predict(curve_inputs), 0, 100)
    curve = pd.DataFrame(
        {
            "Days_Elapsed": days_array,
            "Mass_Remaining_Percentage": mass_remaining,
        }
    )
    curve["Degradation_Percentage"] = 100 - curve["Mass_Remaining_Percentage"]

    if uncertainty is not None:
        curve["Uncertainty"] = float(uncertainty)
        curve["Estimated_Lower_Mass_Remaining_Percentage"] = np.clip(
            curve["Mass_Remaining_Percentage"] - uncertainty,
            0,
            100,
        )
        curve["Estimated_Upper_Mass_Remaining_Percentage"] = np.clip(
            curve["Mass_Remaining_Percentage"] + uncertainty,
            0,
            100,
        )

    return curve
