"""Prediction helpers for the Streamlit app and quick experiments."""

from collections.abc import Iterable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_PATH, RATE_FEATURE_COLUMNS


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
    degree_substitution: float,
) -> pd.DataFrame:
    """Create a one-row table with the exact feature columns the model expects."""
    row = {
        "Material_Type": material_type,
        "Cellulose_Percentage": cellulose_percentage,
        "Temperature_C": temperature_c,
        "pH_Level": ph_level,
        "Environment": environment,
        "degree_substitution": degree_substitution,
    }
    return pd.DataFrame([row], columns=RATE_FEATURE_COLUMNS)


def mass_remaining_from_k(
    k_value: float | np.ndarray,
    days_elapsed: float | np.ndarray,
) -> float | np.ndarray:
    """Calculate mass remaining from the exponential decay equation.

    The equation is:

        mass remaining = 100 * exp(-k * days)

    Because k is clipped to be non-negative, mass remaining cannot increase as
    days elapsed increases.
    """
    non_negative_k = np.maximum(np.array(k_value, dtype=float), 0.0)
    days = np.maximum(np.array(days_elapsed, dtype=float), 0.0)
    mass_remaining = 100 * np.exp(-non_negative_k * days)
    mass_remaining = np.clip(mass_remaining, 0, 100)

    if np.isscalar(k_value) and np.isscalar(days_elapsed):
        return float(mass_remaining)
    return mass_remaining


def calculate_mass_range_from_k_uncertainty(
    k_value: float,
    k_uncertainty: float,
    days_elapsed: float,
) -> tuple[float, float]:
    """Create a mass remaining range from uncertainty in predicted k."""
    lower_k = max(k_value - k_uncertainty, 0.0)
    upper_k = max(k_value + k_uncertainty, 0.0)

    # A smaller k means slower degradation and more remaining mass.
    upper_mass = float(mass_remaining_from_k(lower_k, days_elapsed))
    lower_mass = float(mass_remaining_from_k(upper_k, days_elapsed))
    return lower_mass, upper_mass


def predict_degradation_rate(
    model,
    material_type: str,
    cellulose_percentage: float,
    temperature_c: float,
    ph_level: float,
    environment: str,
    degree_substitution: float,
) -> float:
    """Predict the exponential degradation rate constant k."""
    input_data = make_input_dataframe(
        material_type=material_type,
        cellulose_percentage=cellulose_percentage,
        temperature_c=temperature_c,
        ph_level=ph_level,
        environment=environment,
        degree_substitution=degree_substitution,
    )
    predicted_k = float(model.predict(input_data)[0])
    return max(predicted_k, 0.0)


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
    """Predict mass remaining using the fitted exponential decay model."""
    predicted_k = predict_degradation_rate(
        model=model,
        material_type=material_type,
        cellulose_percentage=cellulose_percentage,
        temperature_c=temperature_c,
        ph_level=ph_level,
        environment=environment,
        degree_substitution=degree_substitution,
    )
    return float(mass_remaining_from_k(predicted_k, days_elapsed))


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

    predicted_k = predict_degradation_rate(
        model=model,
        material_type=material_type,
        cellulose_percentage=cellulose_percentage,
        temperature_c=temperature_c,
        ph_level=ph_level,
        environment=environment,
        degree_substitution=degree_substitution,
    )
    mass_remaining = mass_remaining_from_k(predicted_k, days_array)
    curve = pd.DataFrame(
        {
            "Days_Elapsed": days_array,
            "Mass_Remaining_Percentage": mass_remaining,
            "Degradation_Rate_k": predicted_k,
        }
    )
    curve["Degradation_Percentage"] = 100 - curve["Mass_Remaining_Percentage"]

    if uncertainty is not None:
        lower_mass = mass_remaining_from_k(predicted_k + uncertainty, days_array)
        upper_mass = mass_remaining_from_k(
            max(predicted_k - uncertainty, 0.0),
            days_array,
        )
        curve["Rate_Uncertainty_k"] = float(uncertainty)
        curve["Estimated_Lower_Mass_Remaining_Percentage"] = lower_mass
        curve["Estimated_Upper_Mass_Remaining_Percentage"] = upper_mass

    return curve
