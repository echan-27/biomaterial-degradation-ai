"""Train and compare regression models for biomaterial degradation."""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
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
    FEATURE_COLUMNS,
    METRICS_PATH,
    MODEL_PATH,
    NUMERIC_FEATURES,
    PROJECT_ROOT,
    SUMMARY_PATH,
    TARGET_COLUMN,
)


RANDOM_STATE = 42


def build_preprocessor() -> ColumnTransformer:
    """Create the preprocessing steps required by AGENTS.md."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
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
            n_estimators=300,
            random_state=RANDOM_STATE,
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


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Calculate the requested regression metrics."""
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": rmse,
        "R2": r2_score(y_test, predictions),
    }


def train_and_compare_models(data: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Train each model once and return a metrics table plus the best model name."""
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    results = []
    model_names = [
        "DummyRegressor",
        "Ridge Regression",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
    ]

    for model_name in model_names:
        model = build_model(model_name)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results.append({"Model": model_name, **metrics})

    metrics_table = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    best_model_name = str(metrics_table.loc[0, "Model"])
    return metrics_table, best_model_name


def save_best_model(data: pd.DataFrame, best_model_name: str) -> Pipeline:
    """Retrain the best model on all available data and save it to disk."""
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    best_model = build_model(best_model_name)
    best_model.fit(X, y)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    return best_model


def main() -> None:
    """Run the complete training workflow."""
    data = load_training_data()
    metrics_table, best_model_name = train_and_compare_models(data)
    save_best_model(data, best_model_name)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_table.to_csv(METRICS_PATH, index=False)

    summary = {
        "best_model": best_model_name,
        "selection_rule": "Lowest RMSE on the held-out test set",
        "rows_used": int(len(data)),
        "target_column": TARGET_COLUMN,
        "feature_columns": FEATURE_COLUMNS,
        "saved_model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Model comparison:")
    print(metrics_table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nBest model: {best_model_name}")
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
