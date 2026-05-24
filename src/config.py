"""Shared project settings.

Keeping these names in one file helps the training script, prediction helper,
and Streamlit app use the same columns in the same order.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "degradationdata.xlsx"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
METRICS_PATH = PROJECT_ROOT / "models" / "model_metrics.csv"
SUMMARY_PATH = PROJECT_ROOT / "models" / "model_summary.json"
TEST_PREDICTIONS_PATH = PROJECT_ROOT / "models" / "test_predictions.csv"
OUTLIER_REPORT_PATH = PROJECT_ROOT / "models" / "removed_rate_outliers.csv"

TARGET_COLUMN = "Mass_Remaining_Percentage"
RATE_TARGET_COLUMN = "Degradation_Rate_k"

CATEGORICAL_FEATURES = [
    "Material_Type",
    "Environment",
]

NUMERIC_FEATURES = [
    "Cellulose_Percentage",
    "Temperature_C",
    "pH_Level",
    "Days_Elapsed",
    "degree_substitution",
]

FEATURE_COLUMNS = [
    "Material_Type",
    "Cellulose_Percentage",
    "Temperature_C",
    "pH_Level",
    "Environment",
    "Days_Elapsed",
    "degree_substitution",
]

RATE_NUMERIC_FEATURES = [
    "Cellulose_Percentage",
    "Temperature_C",
    "pH_Level",
    "degree_substitution",
]

RATE_FEATURE_COLUMNS = [
    "Material_Type",
    "Cellulose_Percentage",
    "Temperature_C",
    "pH_Level",
    "Environment",
    "degree_substitution",
]
