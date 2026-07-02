"""Train models that predict the exponential degradation rate constant k.

The important rule in this script is that the hold-out test set is not used for
cleaning, tuning, or model selection. It is used once at the end to estimate
real-world performance.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# This lets beginners run `python src/train_model.py` from the project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.clean_data import load_training_data
from src.config import (
    CATEGORICAL_FEATURES,
    CV_RESULTS_PATH,
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
CV_FOLDS = 5
MIN_MASS_FRACTION = 0.001
OUTLIER_IQR_MULTIPLIER = 3.0
SCORING_FEATURE_COLUMNS = RATE_FEATURE_COLUMNS + ["Evaluation_Day"]
MODEL_NUMERIC_FEATURES = RATE_NUMERIC_FEATURES + ["Evaluation_Day"]


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
    if valid.empty:
        return {
            RATE_TARGET_COLUMN: 0.0,
            "Curve_Fit_RMSE": np.nan,
            "Number_Of_Timepoints": 0,
            "Evaluation_Day": 0.0,
        }

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
    usable_columns = RATE_FEATURE_COLUMNS + [RATE_TARGET_COLUMN, "Evaluation_Day"]
    rate_data = rate_data.dropna(subset=usable_columns)
    rate_data = rate_data[rate_data["Number_Of_Timepoints"] > 0]
    return rate_data.reset_index(drop=True)


def remove_training_rate_outliers(
    train_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Remove extreme fitted-k rows from the training set only.

    The IQR fence is calculated only from y_train. The test set is never used
    for the outlier calculation and is never filtered.
    """
    q1 = float(train_data[RATE_TARGET_COLUMN].quantile(0.25))
    q3 = float(train_data[RATE_TARGET_COLUMN].quantile(0.75))
    iqr = q3 - q1
    upper_limit = q3 + OUTLIER_IQR_MULTIPLIER * iqr

    keep_mask = train_data[RATE_TARGET_COLUMN] <= upper_limit
    filtered_data = train_data[keep_mask].reset_index(drop=True)
    removed_data = train_data[~keep_mask].sort_values(
        RATE_TARGET_COLUMN,
        ascending=False,
    )

    OUTLIER_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    removed_data.to_csv(OUTLIER_REPORT_PATH, index=False)

    outlier_summary = {
        "method": f"Remove training fitted-k values above Q3 + {OUTLIER_IQR_MULTIPLIER} * IQR.",
        "computed_on": "training set only",
        "test_set_policy": "The hold-out test set is untouched and unfiltered.",
        "q1": q1,
        "q3": q3,
        "iqr": float(iqr),
        "upper_limit": float(upper_limit),
        "training_rows_before_filtering": int(len(train_data)),
        "training_rows_after_filtering": int(len(filtered_data)),
        "training_rows_removed": int(len(removed_data)),
        "removed_outliers_path": str(OUTLIER_REPORT_PATH.relative_to(PROJECT_ROOT)),
    }
    return filtered_data, removed_data.reset_index(drop=True), outlier_summary


def build_preprocessor() -> ColumnTransformer:
    """Create the preprocessing steps for the k prediction model."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), MODEL_NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_pipeline(regressor) -> Pipeline:
    """Build a full preprocessing + regression Pipeline."""
    target_model = TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", target_model),
        ]
    )


def predict_non_negative_k(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Predict k and apply the physics-informed non-negative constraint."""
    return np.clip(model.predict(X), 0, None)


def mass_values_from_k_rows(
    estimator: Pipeline,
    X: pd.DataFrame,
    y_true_k: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert true and predicted k values into mass remaining values."""
    predicted_k = predict_non_negative_k(estimator, X)
    evaluation_days = X["Evaluation_Day"].to_numpy(dtype=float)
    actual_mass = mass_remaining_from_k(np.asarray(y_true_k, dtype=float), evaluation_days)
    predicted_mass = mass_remaining_from_k(predicted_k, evaluation_days)
    return actual_mass, predicted_mass


def negative_mass_mae_score(estimator: Pipeline, X: pd.DataFrame, y_true_k) -> float:
    """Cross-validation score for mass MAE. Higher is better because it is negative."""
    actual_mass, predicted_mass = mass_values_from_k_rows(estimator, X, y_true_k)
    return -float(mean_absolute_error(actual_mass, predicted_mass))


def negative_mass_rmse_score(estimator: Pipeline, X: pd.DataFrame, y_true_k) -> float:
    """Cross-validation score for mass RMSE. Higher is better because it is negative."""
    actual_mass, predicted_mass = mass_values_from_k_rows(estimator, X, y_true_k)
    rmse = np.sqrt(mean_squared_error(actual_mass, predicted_mass))
    return -float(rmse)


def mass_r2_score(estimator: Pipeline, X: pd.DataFrame, y_true_k) -> float:
    """Cross-validation R2 score for mass remaining."""
    actual_mass, predicted_mass = mass_values_from_k_rows(estimator, X, y_true_k)
    return float(r2_score(actual_mass, predicted_mass))


def evaluate_model(model: Pipeline, test_data: pd.DataFrame) -> dict:
    """Calculate user-facing mass metrics plus technical k metrics."""
    X_test = test_data[SCORING_FEATURE_COLUMNS]
    y_test = test_data[RATE_TARGET_COLUMN]
    predicted_k = predict_non_negative_k(model, X_test)
    k_rmse = np.sqrt(mean_squared_error(y_test, predicted_k))

    evaluation_days = test_data["Evaluation_Day"].to_numpy(dtype=float)
    actual_mass = mass_remaining_from_k(y_test.to_numpy(dtype=float), evaluation_days)
    predicted_mass = mass_remaining_from_k(predicted_k, evaluation_days)
    mass_rmse = np.sqrt(mean_squared_error(actual_mass, predicted_mass))

    return {
        "MAE": float(mean_absolute_error(actual_mass, predicted_mass)),
        "RMSE": float(mass_rmse),
        "R2": float(r2_score(actual_mass, predicted_mass)),
        "K_MAE": float(mean_absolute_error(y_test, predicted_k)),
        "K_RMSE": float(k_rmse),
        "K_R2": float(r2_score(y_test, predicted_k)),
    }


def build_model_searches(cv: KFold) -> dict[str, GridSearchCV | RandomizedSearchCV]:
    """Create cross-validation searches for every candidate model."""
    scoring = {
        "neg_mass_mae": negative_mass_mae_score,
        "neg_mass_rmse": negative_mass_rmse_score,
        "mass_r2": mass_r2_score,
    }

    dummy_model = build_pipeline(DummyRegressor(strategy="mean"))
    ridge_model = build_pipeline(Ridge())
    random_forest_model = build_pipeline(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    )
    gradient_boosting_model = build_pipeline(
        GradientBoostingRegressor(random_state=RANDOM_STATE)
    )

    return {
        "DummyRegressor": GridSearchCV(
            estimator=dummy_model,
            param_grid=[{}],
            scoring=scoring,
            refit="neg_mass_rmse",
            cv=cv,
            n_jobs=1,
        ),
        "Ridge Regression": GridSearchCV(
            estimator=ridge_model,
            param_grid={"regressor__regressor__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
            scoring=scoring,
            refit="neg_mass_rmse",
            cv=cv,
            n_jobs=1,
        ),
        "RandomForestRegressor": RandomizedSearchCV(
            estimator=random_forest_model,
            param_distributions={
                "regressor__regressor__n_estimators": [200, 400, 800],
                "regressor__regressor__max_depth": [None, 4, 8, 12, 20],
                "regressor__regressor__max_features": ["sqrt", 0.6, 0.8, 1.0],
            },
            n_iter=16,
            scoring=scoring,
            refit="neg_mass_rmse",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "GradientBoostingRegressor": RandomizedSearchCV(
            estimator=gradient_boosting_model,
            param_distributions={
                "regressor__regressor__n_estimators": [100, 200, 300, 500],
                "regressor__regressor__learning_rate": [0.02, 0.05, 0.1, 0.2],
                "regressor__regressor__max_depth": [2, 3, 4],
            },
            n_iter=16,
            scoring=scoring,
            refit="neg_mass_rmse",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def cv_row_from_search(model_name: str, search: GridSearchCV | RandomizedSearchCV) -> dict:
    """Summarize the best cross-validation result for one candidate model."""
    best_index = int(search.best_index_)
    cv_results = search.cv_results_
    best_params = {}
    for key, value in search.best_params_.items():
        clean_key = key.replace("regressor__regressor__", "")
        clean_key = clean_key.replace("regressor__", "")
        best_params[clean_key] = value

    return {
        "Model": model_name,
        "CV_MAE": float(-cv_results["mean_test_neg_mass_mae"][best_index]),
        "CV_RMSE": float(-cv_results["mean_test_neg_mass_rmse"][best_index]),
        "CV_R2": float(cv_results["mean_test_mass_r2"][best_index]),
        "Selection_Score": float(search.best_score_),
        "Best_Params": json.dumps(best_params, sort_keys=True),
    }


def train_and_compare_models(
    rate_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str, Pipeline, pd.DataFrame, dict, dict, int, int, int]:
    """Tune candidate models on training data and test only the selected model."""
    train_data, test_data = train_test_split(
        rate_data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    filtered_train_data, removed_outliers, outlier_summary = remove_training_rate_outliers(
        train_data,
    )
    if len(filtered_train_data) < CV_FOLDS:
        raise ValueError(
            f"Need at least {CV_FOLDS} training rows after outlier filtering; "
            f"only {len(filtered_train_data)} rows remain."
        )

    X_train = filtered_train_data[SCORING_FEATURE_COLUMNS]
    y_train = filtered_train_data[RATE_TARGET_COLUMN]
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    searches = build_model_searches(cv)
    cv_rows = []
    best_model_name = ""
    best_score = -np.inf
    best_model = None

    for model_name, search in searches.items():
        print(f"Running 5-fold CV search for {model_name}...")
        search.fit(X_train, y_train)
        cv_rows.append(cv_row_from_search(model_name, search))

        if float(search.best_score_) > best_score:
            best_score = float(search.best_score_)
            best_model_name = model_name
            best_model = search.best_estimator_

    if best_model is None:
        raise RuntimeError("No model was trained.")

    cv_results_table = pd.DataFrame(cv_rows).sort_values(
        ["CV_RMSE", "CV_MAE"],
    ).reset_index(drop=True)

    holdout_metrics = evaluate_model(best_model, test_data)
    selected_cv_row = cv_results_table[cv_results_table["Model"] == best_model_name].iloc[0]
    metrics_table = pd.DataFrame(
        [
            {
                "Model": best_model_name,
                **holdout_metrics,
                "CV_MAE": float(selected_cv_row["CV_MAE"]),
                "CV_RMSE": float(selected_cv_row["CV_RMSE"]),
                "CV_R2": float(selected_cv_row["CV_R2"]),
                "Best_Params": selected_cv_row["Best_Params"],
            }
        ]
    )

    return (
        metrics_table,
        cv_results_table,
        best_model_name,
        best_model,
        test_data,
        holdout_metrics,
        outlier_summary,
        int(len(train_data)),
        int(len(filtered_train_data)),
        int(len(removed_outliers)),
    )


def save_best_model(best_model: Pipeline) -> None:
    """Save the best k prediction model trained only on the filtered training set."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)


def save_test_predictions(
    best_model: Pipeline,
    test_data: pd.DataFrame,
    k_uncertainty: float,
) -> dict[str, float]:
    """Save actual vs predicted k and mass remaining for the untouched test set."""
    X_test = test_data[SCORING_FEATURE_COLUMNS]
    predicted_k = predict_non_negative_k(best_model, X_test)
    extra_columns = ["Curve_Fit_RMSE", "Number_Of_Timepoints", "Evaluation_Day"]
    prediction_table = test_data[RATE_FEATURE_COLUMNS + extra_columns].copy()
    prediction_table["Actual_Degradation_Rate_k"] = test_data[
        RATE_TARGET_COLUMN
    ].to_numpy(dtype=float)
    prediction_table["Predicted_Degradation_Rate_k"] = predicted_k

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

    prediction_table["Rate_Uncertainty_k"] = float(k_uncertainty)
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
    (
        metrics_table,
        cv_results_table,
        best_model_name,
        best_model,
        test_data,
        holdout_metrics,
        outlier_summary,
        train_rows_before_filtering,
        train_rows_after_filtering,
        outliers_removed,
    ) = train_and_compare_models(rate_data)
    save_best_model(best_model)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_table.to_csv(METRICS_PATH, index=False)
    cv_results_table.to_csv(CV_RESULTS_PATH, index=False)

    k_uncertainty = float(holdout_metrics["K_RMSE"])
    mass_metrics = save_test_predictions(best_model, test_data, k_uncertainty)

    summary = {
        "best_model": best_model_name,
        "model_target": RATE_TARGET_COLUMN,
        "curve_equation": "Mass_Remaining_Percentage = 100 * exp(-k * Days_Elapsed)",
        "physics_constraint": "Predicted k is clipped to be non-negative, so mass remaining cannot increase over time.",
        "selection_rule": "Highest 5-fold cross-validation score on the filtered training set only. The score is negative mass RMSE, so higher is better.",
        "raw_rows_used": int(len(raw_data)),
        "fitted_rate_rows_before_split": int(len(rate_data)),
        "train_rows_before_outlier_filtering": train_rows_before_filtering,
        "train_rows_after_outlier_filtering": train_rows_after_filtering,
        "train_rows": train_rows_after_filtering,
        "test_rows": int(len(test_data)),
        "untouched_test_rows": int(len(test_data)),
        "fitted_rate_outliers_removed": outliers_removed,
        "training_rate_outliers_removed": outliers_removed,
        "outlier_filter": outlier_summary,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "data_path": str(DATA_PATH.relative_to(PROJECT_ROOT)),
        "target_column": TARGET_COLUMN,
        "rate_target_column": RATE_TARGET_COLUMN,
        "feature_columns": RATE_FEATURE_COLUMNS,
        "best_model_test_metrics": {
            "MAE": float(holdout_metrics["MAE"]),
            "RMSE": float(holdout_metrics["RMSE"]),
            "R2": float(holdout_metrics["R2"]),
        },
        "rate_model_test_metrics": {
            "MAE": float(holdout_metrics["K_MAE"]),
            "RMSE": float(holdout_metrics["K_RMSE"]),
            "R2": float(holdout_metrics["K_R2"]),
        },
        "mass_prediction_test_metrics": mass_metrics,
        "uncertainty_estimate": {
            "method": "Use held-out test MAE for mass remaining as +/- percentage points.",
            "value": mass_metrics["MAE"],
            "unit": "percentage points",
        },
        "rate_uncertainty_estimate": {
            "method": "Use the selected model's untouched hold-out test RMSE for k.",
            "value": k_uncertainty,
            "unit": "1/day",
        },
        "saved_model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "model_metrics_path": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
        "cv_results_path": str(CV_RESULTS_PATH.relative_to(PROJECT_ROOT)),
        "test_predictions_path": str(TEST_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nCross-validation model comparison on training data:")
    print(
        cv_results_table.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print("\nFinal hold-out test metrics for selected model:")
    print(
        metrics_table.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print(f"\nBest model: {best_model_name}")
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
