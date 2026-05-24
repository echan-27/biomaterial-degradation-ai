"""Train models that predict the exponential degradation rate constant k."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# This lets beginners run `python src/train_model.py` from the project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.clean_data import load_training_data
from src.config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    OUTLIER_REPORT_PATH,
    PROJECT_ROOT,
    RATE_FEATURE_COLUMNS,
    RATE_NUMERIC_FEATURES,
    RATE_TARGET_COLUMN,
    SUMMARY_PATH,
    TARGET_COLUMN,
    TEST_PREDICTIONS_PATH,
)
from src.predict import mass_remaining_from_k


RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_MASS_FRACTION = 0.001
OUTLIER_IQR_MULTIPLIER = 2.0


def fit_rate_constant_for_group(group: pd.DataFrame) -> dict:
    """Fit k for one material/environment condition.

    We use the exponential decay equation:

        mass remaining = 100 * exp(-k * days)

    Taking the natural log gives:

        log(mass remaining / 100) = -k * days

    This function fits the slope through the origin, then clips k so it cannot
    be negative. A non-negative k makes the predicted curve decrease or stay
    flat over time.
    """
    valid = group[group["Days_Elapsed"] > 0].copy()
    days = valid["Days_Elapsed"].to_numpy(dtype=float)
    mass_fraction = (valid[TARGET_COLUMN].to_numpy(dtype=float) / 100).clip(
        MIN_MASS_FRACTION,
        1.0,
    )

    denominator = np.sum(days**2)
    if denominator == 0:
        fitted_k = 0.0
    else:
        fitted_k = -np.sum(days * np.log(mass_fraction)) / denominator

    fitted_k = max(float(fitted_k), 0.0)
    fitted_mass = mass_remaining_from_k(fitted_k, days)
    fit_rmse = np.sqrt(mean_squared_error(valid[TARGET_COLUMN], fitted_mass))

    return {
        RATE_TARGET_COLUMN: fitted_k,
        "Curve_Fit_RMSE": fit_rmse,
        "Number_Of_Timepoints": int(len(valid)),
        "Evaluation_Day": float(valid["Days_Elapsed"].max()),
    }


def build_rate_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Convert raw mass remaining measurements into one k value per condition."""
    rows = []

    grouped = data.groupby(RATE_FEATURE_COLUMNS, dropna=False)
    for condition_values, group in grouped:
        if not isinstance(condition_values, tuple):
            condition_values = (condition_values,)

        rate_row = dict(zip(RATE_FEATURE_COLUMNS, condition_values))
        rate_row.update(fit_rate_constant_for_group(group))
        rows.append(rate_row)

    rate_data = pd.DataFrame(rows)
    return rate_data.dropna(subset=RATE_FEATURE_COLUMNS + [RATE_TARGET_COLUMN])


def remove_rate_outliers(rate_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Remove extreme fitted-k rows using a conservative 3x IQR upper fence."""
    q1 = float(rate_data[RATE_TARGET_COLUMN].quantile(0.25))
    q3 = float(rate_data[RATE_TARGET_COLUMN].quantile(0.75))
    iqr = q3 - q1
    upper_limit = q3 + OUTLIER_IQR_MULTIPLIER * iqr

    keep_mask = rate_data[RATE_TARGET_COLUMN] <= upper_limit
    filtered_data = rate_data[keep_mask].reset_index(drop=True)
    removed_data = rate_data[~keep_mask].sort_values(
        RATE_TARGET_COLUMN,
        ascending=False,
    )

    OUTLIER_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    removed_data.to_csv(OUTLIER_REPORT_PATH, index=False)

    outlier_summary = {
        "method": f"Remove fitted k values above Q3 + {OUTLIER_IQR_MULTIPLIER} * IQR.",
        "q1": q1,
        "q3": q3,
        "iqr": float(iqr),
        "upper_limit": float(upper_limit),
        "rows_before_filtering": int(len(rate_data)),
        "rows_after_filtering": int(len(filtered_data)),
        "rows_removed": int(len(removed_data)),
        "removed_outliers_path": str(OUTLIER_REPORT_PATH.relative_to(PROJECT_ROOT)),
    }
    return filtered_data, removed_data.reset_index(drop=True), outlier_summary


def build_preprocessor() -> ColumnTransformer:
    """Create the preprocessing steps for the k prediction model."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), RATE_NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_model(model_name: str) -> Pipeline:
    """Build a full preprocessing + regression Pipeline."""
    regressors = {
        "DummyRegressor": DummyRegressor(strategy="mean"),
        "Ridge Regression": Ridge(alpha=1.0),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=800,
            max_features=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "ExtraTreesRegressor": ExtraTreesRegressor(
            n_estimators=800,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            random_state=RANDOM_STATE,
        ),
    }

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", regressors[model_name]),
        ]
    )


def predict_non_negative_k(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Predict k and apply the physics-informed non-negative constraint."""
    return np.clip(model.predict(X), 0, None)


def evaluate_model(model: Pipeline, test_data: pd.DataFrame, y_test: pd.Series) -> dict:
    """Calculate user-facing mass metrics plus technical k metrics."""
    X_test = test_data[RATE_FEATURE_COLUMNS]
    predicted_k = predict_non_negative_k(model, X_test)
    k_rmse = np.sqrt(mean_squared_error(y_test, predicted_k))

    evaluation_days = test_data["Evaluation_Day"].to_numpy(dtype=float)
    actual_mass = mass_remaining_from_k(y_test.to_numpy(dtype=float), evaluation_days)
    predicted_mass = mass_remaining_from_k(predicted_k, evaluation_days)
    mass_rmse = np.sqrt(mean_squared_error(actual_mass, predicted_mass))

    return {
        "MAE": mean_absolute_error(actual_mass, predicted_mass),
        "RMSE": mass_rmse,
        "R2": r2_score(actual_mass, predicted_mass),
        "K_MAE": mean_absolute_error(y_test, predicted_k),
        "K_RMSE": k_rmse,
        "K_R2": r2_score(y_test, predicted_k),
    }


def train_and_compare_models(
    rate_data: pd.DataFrame,
) -> tuple[pd.DataFrame, str, Pipeline, pd.DataFrame, pd.Series, int, int]:
    """Train models on 80% of the fitted k values and evaluate on 20%."""
    train_data, test_data = train_test_split(
        rate_data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    X_train = train_data[RATE_FEATURE_COLUMNS]
    y_train = train_data[RATE_TARGET_COLUMN]
    X_test = test_data[RATE_FEATURE_COLUMNS]
    y_test = test_data[RATE_TARGET_COLUMN]

    results = []
    trained_models = {}
    model_names = [
        "DummyRegressor",
        "Ridge Regression",
        "RandomForestRegressor",
        "ExtraTreesRegressor",
        "GradientBoostingRegressor",
    ]

    for model_name in model_names:
        model = build_model(model_name)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, test_data, y_test)
        results.append({"Model": model_name, **metrics})
        trained_models[model_name] = model

    metrics_table = pd.DataFrame(results)
    metrics_table["MAE_Rank"] = metrics_table["MAE"].rank(method="min")
    metrics_table["RMSE_Rank"] = metrics_table["RMSE"].rank(method="min")
    metrics_table["R2_Rank"] = (-metrics_table["R2"]).rank(method="min")
    metrics_table["Balanced_Rank"] = (
        metrics_table["MAE_Rank"]
        + metrics_table["RMSE_Rank"]
        + metrics_table["R2_Rank"]
    )
    metrics_table = metrics_table.sort_values(
        ["Balanced_Rank", "RMSE"],
    ).reset_index(drop=True)
    best_model_name = str(metrics_table.loc[0, "Model"])
    best_model = trained_models[best_model_name]
    return (
        metrics_table,
        best_model_name,
        best_model,
        test_data.reset_index(drop=True),
        y_test,
        len(X_train),
        len(X_test),
    )


def save_best_model(best_model: Pipeline) -> None:
    """Save the best k prediction model trained only on the training set."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)


def save_test_predictions(
    best_model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    k_uncertainty: float,
) -> dict[str, float]:
    """Save actual vs predicted k and mass remaining for the test set."""
    X_test_features = X_test[RATE_FEATURE_COLUMNS]
    predicted_k = predict_non_negative_k(best_model, X_test_features)
    extra_columns = ["Curve_Fit_RMSE", "Number_Of_Timepoints", "Evaluation_Day"]
    prediction_table = X_test[RATE_FEATURE_COLUMNS + extra_columns].copy()
    prediction_table["Actual_Degradation_Rate_k"] = y_test.to_numpy()
    prediction_table["Predicted_Degradation_Rate_k"] = predicted_k

    # Use each row's maximum observed day as a common point to compare actual
    # and predicted mass remaining from the exponential curves.
    evaluation_days = prediction_table["Evaluation_Day"].to_numpy(dtype=float)
    actual_mass = mass_remaining_from_k(
        prediction_table["Actual_Degradation_Rate_k"].to_numpy(dtype=float),
        evaluation_days,
    )
    predicted_mass = mass_remaining_from_k(predicted_k, evaluation_days)

    prediction_table["Actual_Mass_Remaining_Percentage"] = actual_mass
    prediction_table["Predicted_Mass_Remaining_Percentage"] = predicted_mass
    prediction_table["Prediction_Error"] = (
        prediction_table["Predicted_Mass_Remaining_Percentage"]
        - prediction_table["Actual_Mass_Remaining_Percentage"]
    )
    absolute_errors = prediction_table["Prediction_Error"].abs()
    mass_mae = float(absolute_errors.mean())
    mass_rmse = float(np.sqrt(np.mean(prediction_table["Prediction_Error"] ** 2)))
    mass_median_absolute_error = float(absolute_errors.median())

    prediction_table["Mass_Uncertainty_Percentage_Points"] = mass_mae
    prediction_table["Estimated_Lower_Mass_Remaining_Percentage"] = np.clip(
        prediction_table["Predicted_Mass_Remaining_Percentage"] - mass_mae,
        0,
        100,
    )
    prediction_table["Estimated_Upper_Mass_Remaining_Percentage"] = np.clip(
        prediction_table["Predicted_Mass_Remaining_Percentage"] + mass_mae,
        0,
        100,
    )
    prediction_table["Rate_Uncertainty_k"] = k_uncertainty
    prediction_table.to_csv(TEST_PREDICTIONS_PATH, index=False)

    return {
        "MAE": mass_mae,
        "RMSE": mass_rmse,
        "MedianAbsoluteError": mass_median_absolute_error,
    }


def main() -> None:
    """Run the complete training workflow."""
    raw_data = load_training_data()
    rate_data = build_rate_dataset(raw_data)
    filtered_rate_data, removed_outliers, outlier_summary = remove_rate_outliers(
        rate_data,
    )
    (
        metrics_table,
        best_model_name,
        best_model,
        X_test,
        y_test,
        train_rows,
        test_rows,
    ) = train_and_compare_models(filtered_rate_data)
    save_best_model(best_model)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_table.to_csv(METRICS_PATH, index=False)
    best_metrics = metrics_table.iloc[0]
    k_uncertainty = float(best_metrics["K_RMSE"])
    mass_metrics = save_test_predictions(best_model, X_test, y_test, k_uncertainty)

    summary = {
        "best_model": best_model_name,
        "model_target": RATE_TARGET_COLUMN,
        "curve_equation": "Mass_Remaining_Percentage = 100 * exp(-k * Days_Elapsed)",
        "physics_constraint": "Predicted k is clipped to be non-negative, so mass remaining cannot increase over time.",
        "selection_rule": "Lowest balanced rank across mass MAE, mass RMSE, and mass R2 on the held-out test set",
        "raw_rows_used": int(len(raw_data)),
        "fitted_rate_rows_before_outlier_filtering": int(len(rate_data)),
        "fitted_rate_rows_used": int(len(filtered_rate_data)),
        "fitted_rate_outliers_removed": int(len(removed_outliers)),
        "outlier_filter": outlier_summary,
        "train_rows": int(train_rows),
        "test_rows": int(test_rows),
        "test_size": TEST_SIZE,
        "data_path": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "target_column": TARGET_COLUMN,
        "rate_target_column": RATE_TARGET_COLUMN,
        "feature_columns": RATE_FEATURE_COLUMNS,
        "best_model_test_metrics": {
            "MAE": float(best_metrics["MAE"]),
            "RMSE": float(best_metrics["RMSE"]),
            "R2": float(best_metrics["R2"]),
        },
        "rate_model_test_metrics": {
            "MAE": float(best_metrics["K_MAE"]),
            "RMSE": float(best_metrics["K_RMSE"]),
            "R2": float(best_metrics["K_R2"]),
        },
        "mass_prediction_test_metrics": mass_metrics,
        "uncertainty_estimate": {
            "method": "Use held-out test MAE for mass remaining as +/- percentage points.",
            "value": mass_metrics["MAE"],
            "unit": "percentage points",
        },
        "rate_uncertainty_estimate": {
            "method": "Use the best model's held-out test RMSE for k.",
            "value": k_uncertainty,
            "unit": "1/day",
        },
        "saved_model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "test_predictions_path": str(TEST_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Model comparison table for mass remaining (sorted by lowest RMSE):")
    print(metrics_table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print(f"\nBest model: {best_model_name}")
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
